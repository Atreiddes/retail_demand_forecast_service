"""Direct multi-horizon признаки для LightGBM без утечек и без рекурсии.

Признаки считаются на момент origin, горизонт h - это признак. Для целевой недели w и
горизонта h origin = w - h: лаги и rolling берутся относительно origin, признаки целевой
недели (цена, календарь, SNAP) известны заранее. Для каждого h строится свой кадр, потом
конкатенация. На прогнозе ряды вне ассортимента (available_days==0) маскируются в 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TCOL = "week_start_date"
HMAX = 8  # горизонт прогноза, недель
LAGS = [1, 2, 3, 4, 8, 13, 26, 52]   # относительно origin
ROLL = [4, 8, 13, 26]

CAT_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
NUM_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"rmean_{k}" for k in ROLL]
    + ["rstd_13", "rmax_13", "h"]
    + ["sell_price", "price_rel", "disc_pct", "is_promo", "weeks_since_promo"]
    + ["woy_sin", "woy_cos", "month", "is_xmas", "snap_days", "event_days", "age_weeks"]
)
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "units"


def _target_week_features(df):
    """Признаки целевой недели w (известны заранее на момент любого origin < w)."""
    g = df.groupby("id", observed=True)["sell_price"]
    med = g.shift(1).groupby(df["id"], observed=True).rolling(52, min_periods=8).median()
    med = med.reset_index(level=0, drop=True)
    df["price_rel"] = (df["sell_price"] / med).astype("float32")
    df["disc_pct"] = ((df["sell_price"] / med - 1.0) * 100).astype("float32")
    df["is_promo"] = (df["disc_pct"] <= -10).astype("int8")
    wk = df[TCOL].astype("int64").to_numpy() // (7 * 24 * 3600 * 10**9)
    promo_ord = pd.Series(np.where(df["is_promo"].to_numpy() == 1, wk, np.nan), index=df.index)
    last = promo_ord.groupby(df["id"], observed=True).ffill()
    df["weeks_since_promo"] = (pd.Series(wk, index=df.index) - last).clip(0, 52).fillna(52).astype("int16")
    woy = df[TCOL].dt.isocalendar().week.astype(int).clip(1, 53)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0).astype("float32")
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0).astype("float32")
    df["month"] = df[TCOL].dt.month.astype("int8")
    df["is_xmas"] = woy.isin([51, 52, 1]).astype("int8")
    # неделя первой продажи по ряду; map по словарю безопасен и для пустого словаря
    # (ряды без продаж получают NaT -> age_weeks = 0), в отличие от map по datetime-Series
    first_by_id = df.loc[df["units"] > 0].groupby("id", observed=True)[TCOL].min().to_dict()
    first = pd.to_datetime(df["id"].map(first_by_id))
    df["age_weeks"] = ((df[TCOL] - first).dt.days // 7).clip(lower=0).fillna(0).astype("int16")
    return df


def build_direct(full: pd.DataFrame) -> pd.DataFrame:
    """Кадр (id, week, h) с origin-относительными признаками. 4x строк (по h)."""
    df = full.sort_values(["id", TCOL]).reset_index(drop=True).copy()
    df = _target_week_features(df)
    u = df.groupby("id", observed=True)["units"]

    meta = ["id", TCOL, "units", "revenue", "available_days",
            "sell_price", "price_rel", "disc_pct", "is_promo", "weeks_since_promo",
            "woy_sin", "woy_cos", "month", "is_xmas", "snap_days", "event_days",
            "age_weeks"] + CAT_FEATURES
    frames = []
    for h in range(1, HMAX + 1):
        f = df[meta].copy()
        f["h"] = np.int8(h)
        for k in LAGS:
            f[f"lag_{k}"] = u.shift(h - 1 + k).astype("float32")  # units[w-h-(k-1)]
        sh = u.shift(h)                                            # units[w-h] = origin
        shg = sh.groupby(df["id"], observed=True)
        for m in ROLL:
            f[f"rmean_{m}"] = shg.rolling(m, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        f["rstd_13"] = shg.rolling(13, min_periods=2).std().reset_index(level=0, drop=True).astype("float32")
        f["rmax_13"] = shg.rolling(13, min_periods=1).max().reset_index(level=0, drop=True).astype("float32")
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    for c in CAT_FEATURES:
        out[c] = out[c].astype("category")
    return out


def select_test(frame, test_weeks):
    """Строки прогноза: для недели test_weeks[h-1] берём кадр h (его origin = неделя - h).
    origin неявно задан списком недель (test_weeks[h-1] - h одинаков для всех h)."""
    parts = []
    for h, tw in enumerate(test_weeks, 1):
        parts.append(frame[(frame[TCOL] == tw) & (frame["h"] == h)])
    return pd.concat(parts, ignore_index=True)


def train_slice(frame, origin, cap_weeks):
    lo = origin - pd.Timedelta(weeks=cap_weeks)
    return frame[(frame[TCOL] <= origin) & (frame[TCOL] > lo) & (frame["available_days"] > 0)]
