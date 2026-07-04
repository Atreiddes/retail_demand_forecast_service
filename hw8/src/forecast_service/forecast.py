"""Применение предобученного артефакта: прогноз P10/P50/P90 по списку рядов.

Воркер только применяет модель, ничего не обучает. История берётся полными неделями
(n_days==7), будущие недели синтезируются от origin: цена переносится с origin, snap-дни
и события берутся из календаря, available_days как на origin. Категории приводятся к
сохранённому словарю, иначе на пачке коды разойдутся с обучением.
"""
from __future__ import annotations

import json
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import settings
from .ml.features import CAT_FEATURES, FEATURES, HMAX, TCOL, build_direct, select_test
from .ml.schemas import features_schema, forecast_output_schema, validate_sample

QUANTILES = (0.1, 0.9)
HIST_WEEKS = 78  # хватает на lag_52 + rolling_26

_ART: dict = {}
_CAL: dict = {}


def load_artifact():
    if _ART:
        return _ART
    d = settings.artifact_dir
    # собираем в локальный словарь и публикуем в кэш только после полной успешной загрузки:
    # иначе при отсутствии/порче любого файла в _ART осел бы полузаполненный словарь навсегда
    art = {}
    art["point"] = lgb.Booster(model_file=str(d / "model.txt"))
    art["q"] = {q: lgb.Booster(model_file=str(d / f"q{q}.txt")) for q in QUANTILES}
    art["cats"] = json.loads((d / "categories.json").read_text(encoding="utf-8"))
    art["factor"] = json.loads((d / "calibration.json").read_text())["factor"]
    card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
    art["model_version"] = card["model_version"]
    art["origin"] = pd.Timestamp(card["origin"])
    art["wrmsse"] = card.get("wrmsse_foods_backtest")
    feat = json.loads((d / "features.json").read_text(encoding="utf-8"))["feature_name"]
    if not (feat == art["point"].feature_name() == FEATURES):
        raise RuntimeError("признаки артефакта не совпадают с моделью")
    _ART.update(art)
    return _ART


def _weekly_calendar():
    """Недельная агрегация календаря: snap по штатам и события, по дате начала недели."""
    if _CAL:
        return _CAL["tbl"]
    c = pd.read_parquet(settings.calendar_path)
    g = c.groupby("wm_yr_wk", as_index=False).agg(
        week_start_date=("date", "min"),
        snap_CA=("snap_CA", "sum"), snap_TX=("snap_TX", "sum"), snap_WI=("snap_WI", "sum"),
        event_days=("has_event", "sum"))
    g["week_start_date"] = pd.to_datetime(g["week_start_date"])
    _CAL["tbl"] = g
    return g


def _future_rows(hist, origin, horizon):
    """H будущих полных недель на ряд: цена и available с origin, snap/события из календаря."""
    weeks = [origin + timedelta(weeks=h) for h in range(1, horizon + 1)]
    # последняя доступная неделя ряда (<= origin), а не строго origin: ряд без строки ровно
    # на origin (дыра в истории или origin не на границе недели) иначе молча выпал бы из прогноза
    base = hist.sort_values(TCOL).groupby("id", sort=False).tail(1)[
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "sell_price", "available_days"]]
    rows = base.merge(pd.DataFrame({TCOL: weeks}), how="cross")
    wk = _weekly_calendar()
    rows = rows.merge(wk, on=TCOL, how="left")
    st = rows["state_id"].astype(str).to_numpy()
    snap = np.where(st == "CA", rows["snap_CA"], np.where(st == "TX", rows["snap_TX"], rows["snap_WI"]))
    rows["snap_days"] = pd.Series(snap, index=rows.index).fillna(0).astype("int16")
    rows["event_days"] = rows["event_days"].fillna(0).astype("int16")
    rows["units"] = np.nan
    rows["revenue"] = 0.0
    rows["n_days"] = 7
    return rows.drop(columns=["snap_CA", "snap_TX", "snap_WI"])


def forecast_series(history: pd.DataFrame, origin, horizon=HMAX) -> pd.DataFrame:
    """history: строки SalesHistory нужных рядов с атрибутами ряда (колонка id = series_id).
    Возвращает [series_id, week_start_date, h, p10, p50, p90]."""
    art = load_artifact()
    origin = pd.Timestamp(origin)
    hist = history[(history["n_days"] == 7) & (history[TCOL] <= origin)].copy()
    panel = pd.concat([hist, _future_rows(hist, origin, horizon)], ignore_index=True)
    frame = build_direct(panel)
    weeks = [origin + timedelta(weeks=h) for h in range(1, horizon + 1)]
    te = select_test(frame, weeks).copy()
    for c in CAT_FEATURES:
        te[c] = pd.Categorical(te[c].astype(str), categories=art["cats"][c])
    validate_sample(te, features_schema)  # контроль признаков на входе модели

    oos = te["available_days"].to_numpy() == 0
    p50 = np.where(oos, 0.0, np.clip(art["point"].predict(te[FEATURES]), 0, None) * art["factor"])
    q10 = np.where(oos, 0.0, np.clip(art["q"][0.1].predict(te[FEATURES]), 0, None))
    q90 = np.where(oos, 0.0, np.clip(art["q"][0.9].predict(te[FEATURES]), 0, None))

    out = te[["id", TCOL, "h"]].rename(columns={"id": "series_id"}).copy()
    out["p10"] = np.minimum(q10, p50).astype("float32")
    out["p50"] = p50.astype("float32")
    out["p90"] = np.maximum(q90, p50).astype("float32")
    validate_sample(out, forecast_output_schema)  # контроль выхода: без NaN, >= 0, монотонность
    return out
