"""Квантильный прогноз P10/P50/P90 поверх direct multi-horizon признаков.

Три LightGBM с pinball-loss (objective="quantile", alpha=q), та же матрица признаков и
веса по revenue, что у точечной модели. Монотонность P10 <= P50 <= P90 не гарантируется
тремя независимыми моделями, поэтому квантили в строке пересортировываем (rearrangement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features_direct as FD
import model_lgb as LGB
import util as U
from features_direct import FEATURES, TCOL

QUANTILES = (0.1, 0.5, 0.9)
TRAIN_CAP = U.TRAIN_CAP


def _train_one(train_df, alpha, rounds, valid_df=None):
    """Одна квантильная модель: обвязка model_lgb (веса по revenue, признаки) с objective=quantile."""
    return LGB.train(train_df, {"objective": "quantile", "alpha": alpha},
                     num_boost_round=rounds, valid_df=valid_df, early=50)


def forecast_quantiles(frame, fold, quantiles=QUANTILES, rounds=600, cap_weeks=TRAIN_CAP):
    """Квантильный прогноз на тестовом блоке фолда; без утечек, обучение на <= origin.

    Возвращает long DataFrame [id, week, q0.1, q0.5, q0.9] с монотонными по строке квантилями
    и нулём для рядов вне ассортимента (available_days==0).
    """
    tr = FD.train_slice(frame, fold["origin"], cap_weeks)
    te = FD.select_test(frame, fold["test_w"])
    mask = (te["available_days"] == 0).to_numpy()

    cols = {}
    for q in quantiles:
        model = _train_one(tr, q, rounds)
        pred = np.clip(model.predict(te[FEATURES], num_iteration=model.best_iteration), 0, None)
        pred = np.where(mask, 0.0, pred)
        cols[f"q{q}"] = pred.astype("float32")

    out = te[["id", TCOL]].copy()
    qmat = np.sort(np.column_stack([cols[f"q{q}"] for q in quantiles]), axis=1)  # rearrangement
    for i, q in enumerate(sorted(quantiles)):
        out[f"q{q}"] = qmat[:, i]
    return out


def pinball_loss(actual, pred, q):
    """Pinball (quantile) loss для квантиля q. Среднее по наблюдениям, меньше лучше."""
    d = actual - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def quantile_metrics(test_actual, qpred, quantiles=QUANTILES):
    """Сводка качества квантильного прогноза.

    pinball_{q} на квантиль и средний; покрытие cov_{q} = доля факта <= P_q (цель = q);
    interval_cov = доля факта в [P_lo, P_hi] (цель = hi-lo). test_actual: [id, week, units].
    """
    m = test_actual[["id", TCOL, "units"]].merge(qpred, on=["id", TCOL], validate="one_to_one")
    a = m["units"].to_numpy(dtype=float)
    res = {}
    pins = []
    for q in sorted(quantiles):
        p = m[f"q{q}"].to_numpy(dtype=float)
        res[f"pinball_{q}"] = round(pinball_loss(a, p, q), 4)
        res[f"cov_{q}"] = round(float((a <= p).mean()), 4)
        pins.append(res[f"pinball_{q}"])
    res["pinball_mean"] = round(float(np.mean(pins)), 4)
    lo, hi = min(quantiles), max(quantiles)
    inside = (a >= m[f"q{lo}"].to_numpy(dtype=float)) & (a <= m[f"q{hi}"].to_numpy(dtype=float))
    res["interval_cov"] = round(float(inside.mean()), 4)
    res["interval_target"] = round(hi - lo, 4)
    return res
