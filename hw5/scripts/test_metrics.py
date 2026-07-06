"""Тесты корректности метрик M5 (metrics.py): аналитический эталон, свойства, кросс-чек.

На контролируемом примере метрики совпадают с вычисленными вручную.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import metrics as M

TCOL = "week_start_date"


def _panel(units_train, units_test, pred_test, price=1.0):
    """Минимальная валидная панель из 1 ряда (item×store) с полной иерархией."""
    n = len(units_train) + len(units_test)
    weeks = pd.date_range("2014-01-06", periods=n, freq="W-MON")
    base = dict(id="FOODS_3_001_CA_1", item_id="FOODS_3_001", dept_id="FOODS_3",
                cat_id="FOODS", store_id="CA_1", state_id="CA")
    units = list(units_train) + list(units_test)
    df = pd.DataFrame({**{k: [v] * n for k, v in base.items()},
                       TCOL: weeks, "units": units})
    df["revenue"] = df["units"] * price
    tr = df.iloc[: len(units_train)].copy()
    te = df.iloc[len(units_train):].copy()
    pred = te[["id", TCOL]].copy()
    pred["pred"] = list(pred_test)
    return tr, te, pred


def test_hand_computed():
    """Аналитический эталон. train=[10,12]*3 -> diffs ±2, scale=mean(4)=4, |diff|=2.
    test actual=[10,12], pred=[11,11] -> errors ±1, mse=1, rmsse=sqrt(1/4)=0.5.
    MASE=MAE/mean|diff|=1/2=0.5. 1 ряд -> все 12 уровней совпадают -> WRMSSE=0.5, Bias=0."""
    tr, te, pred = _panel([10, 12, 10, 12, 10, 12], [10, 12], [11, 11])
    r = M.all_metrics(tr, te, pred)
    assert abs(r["WRMSSE"] - 0.5) < 1e-6, r
    assert abs(r["RMSSE_L12"] - 0.5) < 1e-6, r
    assert abs(r["MASE_L12"] - 0.5) < 1e-6, r
    assert abs(r["Bias"] - 0.0) < 1e-9, r
    # Scorer даёт то же
    s = M.Scorer(tr).score(te, pred)
    assert abs(s["WRMSSE"] - 0.5) < 1e-6 and abs(s["MASE_L12"] - 0.5) < 1e-6, s
    print("OK test_hand_computed: WRMSSE/RMSSE/MASE=0.5, Bias=0 (совпали с ручным расчётом)")


def test_perfect_forecast():
    """pred = actual -> все ошибки 0."""
    tr, te, pred = _panel([5, 7, 5, 7], [5, 7], [5, 7])
    r = M.all_metrics(tr, te, pred)
    assert r["WRMSSE"] == 0 and r["RMSSE_L12"] == 0 and r["MASE_L12"] == 0 and r["Bias"] == 0
    print("OK test_perfect_forecast: идеальный прогноз -> метрики 0")


def test_bias_sign():
    """pred = 2*actual -> Bias = +1.0; pred = 0 -> Bias = -1.0."""
    tr, te, _ = _panel([4, 6, 4, 6], [4, 6], [0, 0])
    over = te[["id", TCOL]].assign(pred=te["units"].to_numpy() * 2)
    assert abs(M.bias(te, over) - 1.0) < 1e-9
    zero = te[["id", TCOL]].assign(pred=0.0)
    assert abs(M.bias(te, zero) + 1.0) < 1e-9
    print("OK test_bias_sign: перепрогноз x2 -> +1.0, ноль -> -1.0")


def test_constant_series_no_div0():
    """Константный ряд (scale=0) исключается, без деления на ноль и без падения."""
    tr, te, pred = _panel([3, 3, 3, 3], [3, 3], [3, 4])
    r = M.all_metrics(tr, te, pred)  # scale=0 -> ряд выпадает, WRMSSE может быть nan, но не падение
    assert isinstance(r["WRMSSE"], float)
    print(f"OK test_constant_series_no_div0: не упало (WRMSSE={r['WRMSSE']})")


def test_weights_sum_to_one():
    """Веса уровня нормированы к сумме 1 (на 2 рядах с разным revenue)."""
    weeks = pd.date_range("2014-01-06", periods=8, freq="W-MON")
    rows = []
    for it, st, pr in [("FOODS_3_001", "CA_1", 5.0), ("FOODS_3_002", "CA_1", 1.0)]:
        for i, wk in enumerate(weeks):
            u = 10 + (i % 2)
            rows.append(dict(id=f"{it}_{st}", item_id=it, dept_id="FOODS_3", cat_id="FOODS",
                             store_id=st, state_id="CA", **{TCOL: wk}, units=u, revenue=u * pr))
    df = pd.DataFrame(rows)
    w = M._level_weights(df, ["item_id"])
    assert abs(w.sum() - 1.0) < 1e-9, w
    # ряд с ценой 5 должен весить больше ряда с ценой 1
    assert w.loc["FOODS_3_001"] > w.loc["FOODS_3_002"]
    print("OK test_weights_sum_to_one: веса по revenue нормированы к 1, дороже ряд весомее")


def _panel_multi(specs):
    """Панель из нескольких рядов. specs = [(item, store, units_all, pred_test, price)]."""
    n = len(specs[0][2])
    weeks = pd.date_range("2014-01-06", periods=n, freq="W-MON")
    n_test = len(specs[0][3])
    rows = []
    for it, st, units, _, price in specs:
        for i, wk in enumerate(weeks):
            rows.append(dict(id=f"{it}_{st}", item_id=it, dept_id="FOODS_3", cat_id="FOODS",
                             store_id=st, state_id="CA", **{TCOL: wk},
                             units=units[i], revenue=units[i] * price))
    df = pd.DataFrame(rows)
    tr = df[df[TCOL] < weeks[-n_test]].copy()
    te = df[df[TCOL] >= weeks[-n_test]].copy()
    preds = []
    for it, st, _, pred_test, _ in specs:
        sub = te[te["id"] == f"{it}_{st}"][["id", TCOL]].copy()
        sub["pred"] = list(pred_test)
        preds.append(sub)
    return tr, te, pd.concat(preds, ignore_index=True)


def test_rmsse_l12_consistency():
    """RMSSE_L12 одинаков в rmsse_l12(), all_metrics и Scorer на >1 ряде с разным весом и ошибкой."""
    tr, te, pred = _panel_multi([
        ("FOODS_3_001", "CA_1", [10, 12, 10, 12, 10, 12, 10, 12], [11, 11], 5.0),
        ("FOODS_3_002", "CA_1", [20, 24, 20, 24, 20, 24, 20, 24], [30, 30], 1.0),
    ])
    a = round(M.rmsse_l12(tr, te, pred), 4)   # округляем как остальные
    b = M.all_metrics(tr, te, pred)["RMSSE_L12"]
    c = M.Scorer(tr).score(te, pred)["RMSSE_L12"]
    assert a == b == c, (a, b, c)
    print(f"OK test_rmsse_l12_consistency: rmsse_l12={a} == all_metrics == Scorer (согласованы)")


def test_duplicate_keys_raise():
    """Дубликат (id, week) в прогнозе обязан вызвать ошибку, а не тихо исказить метрики."""
    tr, te, pred = _panel([10, 12, 10, 12], [10, 12], [11, 11])
    dup = pd.concat([pred, pred.iloc[[0]]], ignore_index=True)
    for fn in (lambda: M.Scorer(tr).score(te, dup),
               lambda: M.wrmsse(tr, te, dup),
               lambda: M.bias(te, dup)):
        raised = False
        try:
            fn()
        except Exception:
            raised = True
        assert raised, "метрика не упала на дубликате ключей (id, week)"
    print("OK test_duplicate_keys_raise: дубль (id, week) ловится, метрики не искажаются молча")


def test_naive_scale_within_series():
    """naive scale = средний квадрат недельных разностей внутри ряда, не через границы."""
    tr, _, _ = _panel([10, 12, 10, 12, 10, 12], [10, 12], [11, 11])
    lvl = M._agg_to_level(tr, ["id"], "units")
    scale = M._naive_scale(lvl)
    assert abs(list(scale.values())[0] - 4.0) < 1e-9   # diffs ±2 -> mean(4)=4
    print("OK test_naive_scale_within_series: scale=4 (ручной расчёт)")


if __name__ == "__main__":
    test_hand_computed()
    test_perfect_forecast()
    test_bias_sign()
    test_constant_series_no_div0()
    test_weights_sum_to_one()
    test_rmsse_l12_consistency()
    test_duplicate_keys_raise()
    test_naive_scale_within_series()
    print("\nвсе тесты метрик пройдены")
