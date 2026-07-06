"""Общие утилиты hw5: конфиг, загрузка окна, базовые модели, калибровка LightGBM.

Единый источник для того, что иначе дублировалось бы в run_cv / cold_start / baselines_extra.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import features_direct as FD
import model_lgb as LGB

TCOL = "week_start_date"
HORIZON = 8
EPS = 1e-9  # защита от деления на ноль
DATA = (Path(__file__).resolve().parent.parent.parent
        / "hw4" / "data" / "processed" / "sales_weekly.parquet")
TRAIN_CAP = int(os.getenv("TRAIN_CAP_WEEKS", "104"))
WINDOW = int(os.getenv("WINDOW", "185"))


def load_window(window=WINDOW, data=DATA):
    """Полные недели M5, обрезанные до последних `window` недель.
    При USE_EVENTS=1 добавляет недельные event-признаки из календаря."""
    w = pd.read_parquet(data)
    full = w[w["n_days"] == 7].copy()
    weeks = np.array(sorted(full[TCOL].unique()))
    full = full[full[TCOL] > weeks[-window]].copy()
    if FD.USE_EVENTS:
        import events
        full = events.add_event_features(full)
    return full, np.array(sorted(full[TCOL].unique()))


def ma_level(train, k, value="units"):
    """Среднее `value` по последним k train-неделям на ряд. Без утечек."""
    wk = sorted(train[TCOL].unique())[-k:]
    return (train[train[TCOL].isin(wk)].groupby("id", observed=True)[value]
            .mean().rename("lvl").reset_index())


def moving_average(train, test, k=4):
    out = test[["id", TCOL]].merge(ma_level(train, k), on="id", how="left")
    out["pred"] = out["lvl"].fillna(0.0)
    return out[["id", TCOL, "pred"]]


def seasonal_naive(train, test):
    """Факт того же ряда 52 недели назад; fallback - MA-4."""
    hist = train[["id", TCOL, "units"]].rename(columns={TCOL: "ly", "units": "pred"})
    sn = test[["id", TCOL]].copy()
    sn["ly"] = sn[TCOL] - pd.Timedelta(days=364)
    sn = sn.merge(hist, on=["id", "ly"], how="left").merge(
        ma_level(train, 4).rename(columns={"lvl": "fb"}), on="id", how="left")
    sn["pred"] = sn["pred"].fillna(sn["fb"]).fillna(0.0)
    return sn[["id", TCOL, "pred"]]


def _woy(s):
    return s.dt.isocalendar().week.astype(int).clip(1, 53)


def seasonal_ma(train, test, k=8):
    """Структурный per-series бейзлайн: уровень (MA-k) × сезонный индекс ряда.

    Уровень - среднее последних k недель ряда. Индекс woy - средняя продажа ряда в эту
    неделю года, делённая на общее среднее ряда. Прогноз недели t = уровень × индекс[woy(t)].
    Без утечек: оба сомножителя из train. Индекс ограничен [0, 5] от шумных редких недель."""
    level = ma_level(train, k)
    tr = train[["id", TCOL, "units"]].copy()
    tr["woy"] = _woy(tr[TCOL])
    overall = tr.groupby("id", observed=True)["units"].mean()
    bywoy = tr.groupby(["id", "woy"], observed=True)["units"].mean()
    sidx = (bywoy / overall.reindex(bywoy.index.get_level_values("id")).to_numpy()
            ).rename("sidx").reset_index()
    out = test[["id", TCOL]].copy()
    out["woy"] = _woy(out[TCOL])
    out = out.merge(level, on="id", how="left").merge(sidx, on=["id", "woy"], how="left")
    out["sidx"] = out["sidx"].fillna(1.0).clip(0, 5)
    out["pred"] = (out["lvl"].fillna(0.0) * out["sidx"]).clip(lower=0)
    return out[["id", TCOL, "pred"]]


def _croston_rate(y, alpha=0.1, sba=True):
    """Прогнозная интенсивность Croston (SBA): сглаженный размер / сглаженный интервал.
    sba=True даёт поправку смещения Syntetos-Boylan (1 - alpha/2). Плоский прогноз."""
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return 0.0
    sizes = y[nz]
    intervals = np.diff(np.concatenate(([-1], nz)))  # интервалы между ненулевыми
    z, x = float(sizes[0]), float(intervals[0])
    for s, q in zip(sizes[1:], intervals[1:]):
        z += alpha * (s - z)
        x += alpha * (q - x)
    rate = z / x
    return rate * (1 - alpha / 2) if sba else rate


def croston(train, test, alpha=0.1, sba=True):
    """Croston/SBA per-series для прерывистого спроса. Плоский прогноз по горизонту.
    Без утечек: интенсивность считается только по train-истории ряда."""
    tr = train[["id", TCOL, "units"]].sort_values(["id", TCOL])
    rate = (tr.groupby("id", observed=True)["units"]
            .apply(lambda s: _croston_rate(s.to_numpy(), alpha, sba)).rename("pred"))
    out = test[["id", TCOL]].merge(rate.reset_index(), on="id", how="left")
    out["pred"] = out["pred"].fillna(0.0).clip(lower=0)
    return out[["id", TCOL, "pred"]]


def predict_masked(model, te_rows):
    """Прогноз LightGBM с маской в 0 для рядов вне ассортимента (available_days==0)."""
    p = LGB.predict(model, te_rows).merge(
        te_rows[["id", TCOL, "available_days"]], on=["id", TCOL])
    p["pred"] = np.where(p["available_days"] == 0, 0.0, p["pred"])
    return p[["id", TCOL, "pred"]]


def calibrate_and_predict(frame, fold, params=None, rounds=400, early_stop=False):
    """Калиброванный прогноз LightGBM на фолде.

    Обучение на <= cal_origin, фактор смещения sum(actual)/sum(pred) на cal-блоке (OOS),
    прогноз теста из origin × фактор, маска OOS. early_stop=True - n_estimators
    подбирается по early stopping на cal-блоке. Без утечек.
    Возвращает (preds[id,week,pred], factor, model).
    """
    tr = FD.train_slice(frame, fold["cal_origin"], TRAIN_CAP)
    cal = FD.select_test(frame, fold["cal_w"])
    model = LGB.train(tr, params, num_boost_round=rounds,
                      valid_df=cal if early_stop else None, early=50)
    factor = float(cal["units"].sum() / max(LGB.predict(model, cal)["pred"].sum(), EPS))
    p = predict_masked(model, FD.select_test(frame, fold["test_w"]))
    p["pred"] = p["pred"] * factor
    return p, factor, model
