"""Декомпозиция прогноза LightGBM на базовый спрос, промо, сезонность и события.

Аддитивного разложения для дерева с мультипликативным Tweedie-линком нет, поэтому считаем
разницу drop-one: нейтрализуем группу признаков и берём разницу с полным прогнозом.
effect_g = full - pred(группа g нейтрализована); базовый спрос = full - sum(effect_g),
поэтому сумма компонент точно равна прогнозу.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features_direct import FEATURES, TCOL

# Нейтральные значения групп (отсутствие промо / средняя сезонность / нет событий).
NEUTRAL = {
    "promo": {"is_promo": 0, "disc_pct": 0.0, "price_rel": 1.0, "weeks_since_promo": 52},
    "seasonality": {"woy_sin": 0.0, "woy_cos": 0.0, "is_xmas": 0, "month": 6},
    "events": {"event_days": 0, "snap_days": 0},
}


def _raw_predict(model, df):
    # best_iteration=0 у модели, загруженной из файла -> None (все деревья)
    n = getattr(model, "best_iteration", None) or None
    return model.predict(df[FEATURES], num_iteration=n)


def decompose(model, te_rows, factor=1.0):
    """Разложить прогноз модели на te_rows; factor масштабирует все компоненты равномерно, аддитивность сохраняется."""
    full = _raw_predict(model, te_rows)
    out = te_rows[["id", TCOL]].copy()
    effects = {}
    for grp, neutral in NEUTRAL.items():
        cf = te_rows.copy()
        for col, val in neutral.items():
            cf[col] = val
        effects[grp] = full - _raw_predict(model, cf)
    baseline = full - sum(effects.values())
    out["full"] = (full * factor).astype("float32")
    out["baseline"] = (baseline * factor).astype("float32")
    for grp in NEUTRAL:
        out[grp] = (effects[grp] * factor).astype("float32")
    return out


def decompose_series(model, te_rows, item_id, store_id, factor=1.0):
    """Декомпозиция для одного ряда item x store по 4 неделям горизонта."""
    rows = te_rows[(te_rows["item_id"] == item_id) & (te_rows["store_id"] == store_id)]
    if rows.empty:
        raise ValueError(f"ряд {item_id} x {store_id} не найден в тестовом блоке")
    return decompose(model, rows, factor).sort_values(TCOL).reset_index(drop=True)
