"""Тесты корректности метрик M5 (перенос из hw5): аналитический эталон, свойства, кросс-чек."""
from __future__ import annotations

import pandas as pd

from forecast_service.ml import metrics as M

TCOL = "week_start_date"


def _panel(units_train, units_test, pred_test, price=1.0):
    """Минимальная валидная панель из 1 ряда (item x store) с полной иерархией."""
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
    tr, te, pred = _panel([10, 12, 10, 12, 10, 12], [10, 12], [11, 11])
    r = M.all_metrics(tr, te, pred)
    assert abs(r["WRMSSE"] - 0.5) < 1e-6, r
    assert abs(r["RMSSE_L12"] - 0.5) < 1e-6, r
    assert abs(r["MASE_L12"] - 0.5) < 1e-6, r
    assert abs(r["Bias"] - 0.0) < 1e-9, r
    s = M.Scorer(tr).score(te, pred)
    assert abs(s["WRMSSE"] - 0.5) < 1e-6 and abs(s["MASE_L12"] - 0.5) < 1e-6, s


def test_perfect_forecast():
    tr, te, pred = _panel([5, 7, 5, 7], [5, 7], [5, 7])
    r = M.all_metrics(tr, te, pred)
    assert r["WRMSSE"] == 0 and r["RMSSE_L12"] == 0 and r["MASE_L12"] == 0 and r["Bias"] == 0


def test_bias_sign():
    tr, te, _ = _panel([4, 6, 4, 6], [4, 6], [0, 0])
    over = te[["id", TCOL]].assign(pred=te["units"].to_numpy() * 2)
    assert abs(M.bias(te, over) - 1.0) < 1e-9
    zero = te[["id", TCOL]].assign(pred=0.0)
    assert abs(M.bias(te, zero) + 1.0) < 1e-9


def test_weights_sum_to_one():
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
    assert w.loc["FOODS_3_001"] > w.loc["FOODS_3_002"]


def test_duplicate_keys_raise():
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


def test_naive_scale_within_series():
    tr, _, _ = _panel([10, 12, 10, 12, 10, 12], [10, 12], [11, 11])
    lvl = M._agg_to_level(tr, ["id"], "units")
    scale = M._naive_scale(lvl)
    assert abs(list(scale.values())[0] - 4.0) < 1e-9
