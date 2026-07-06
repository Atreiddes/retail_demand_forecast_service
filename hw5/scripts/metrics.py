"""Метрики M5 для недельного прогноза: RMSSE, WRMSSE (12 уровней), MASE, Bias.

Совокупная метрика = WRMSSE: взвешенный RMSSE по 12 уровням иерархии M5,
веса = доля revenue последних 4 недель train. Определение из hw4/dataset_description.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TCOL = "week_start_date"

# 12 уровней агрегации M5: ключи группировки рядов.
LEVELS = [
    ("L1_total", []),
    ("L2_state", ["state_id"]),
    ("L3_store", ["store_id"]),
    ("L4_cat", ["cat_id"]),
    ("L5_dept", ["dept_id"]),
    ("L6_state_cat", ["state_id", "cat_id"]),
    ("L7_state_dept", ["state_id", "dept_id"]),
    ("L8_store_cat", ["store_id", "cat_id"]),
    ("L9_store_dept", ["store_id", "dept_id"]),
    ("L10_item", ["item_id"]),
    ("L11_item_state", ["item_id", "state_id"]),
    ("L12_item_store", ["id"]),
]


def _agg_to_level(df, keys, value):
    """Сумма value по уровню на каждую неделю. keys=[] -> один тотальный ряд."""
    if not keys:
        g = df.groupby(TCOL, observed=True)[value].sum().reset_index()
        g["__sid__"] = "TOTAL"
        return g.rename(columns={value: "v"})
    g = df.groupby(keys + [TCOL], observed=True)[value].sum().reset_index()
    g["__sid__"] = g[keys].astype(str).agg("|".join, axis=1)
    return g[["__sid__", TCOL, value]].rename(columns={value: "v"})


def _naive_scale(train_level):
    """Знаменатель RMSSE: средний квадрат недельной разности y_t - y_{t-1} по train-ряду."""
    g = train_level.sort_values(["__sid__", TCOL]).copy()
    g["d2"] = g.groupby("__sid__", observed=True)["v"].diff() ** 2
    s = g.groupby("__sid__", observed=True)["d2"].mean()  # mean пропускает первый NaN-diff
    s = s[s > 0]
    return s.to_dict()


def _level_weights(train, keys, last_weeks=4):
    """Веса рядов уровня = доля revenue последних last_weeks недель train, сумма = 1."""
    wk = sorted(train[TCOL].unique())[-last_weeks:]
    sub = train[train[TCOL].isin(wk)]
    lvl = _agg_to_level(sub, keys, "revenue")
    w = lvl.groupby("__sid__", observed=True)["v"].sum()
    tot = w.sum()
    return (w / tot) if tot > 0 else w


def _rmsse_per_series(test_level, scale):
    """RMSSE на ряд: sqrt(mean_h (a-p)^2 / scale). Только ряды с известным scale."""
    m = test_level.copy()
    m["se"] = (m["v_actual"].astype(float) - m["v_pred"].astype(float)) ** 2
    mse = m.groupby("__sid__", observed=True)["se"].mean()
    sc = pd.Series(scale)
    common = mse.index.intersection(sc.index)
    return np.sqrt(mse.loc[common] / sc.loc[common]).to_dict()


def wrmsse(train, test_actual, test_pred, return_levels=False):
    """WRMSSE по 12 уровням. test_pred = test_actual + столбец pred.

    train, test_actual: long с колонками иерархии, TCOL, units, revenue.
    test_pred: те же ряды-недели с колонкой pred.
    """
    te = test_actual.merge(
        test_pred[["id", TCOL, "pred"]], on=["id", TCOL], how="left", validate="one_to_one"
    )
    te["pred"] = te["pred"].fillna(0.0)

    level_scores = {}
    for name, keys in LEVELS:
        tr_lvl = _agg_to_level(train, keys, "units")
        scale = _naive_scale(tr_lvl)
        if not scale:
            continue
        act = _agg_to_level(te, keys, "units").rename(columns={"v": "v_actual"})
        prd = _agg_to_level(te.rename(columns={"units": "_u"}).assign(units=te["pred"]),
                            keys, "units").rename(columns={"v": "v_pred"})
        m = act.merge(prd, on=["__sid__", TCOL], how="left")
        m["v_pred"] = m["v_pred"].fillna(0.0)
        rmsse = pd.Series(_rmsse_per_series(m, scale))
        w = _level_weights(train, keys)
        common = rmsse.index.intersection(w.index)
        wsum = float(w.loc[common].sum())
        if wsum > 0:
            level_scores[name] = float((rmsse.loc[common] * w.loc[common]).sum() / wsum)
    overall = float(np.mean(list(level_scores.values()))) if level_scores else np.nan
    return (overall, level_scores) if return_levels else overall


def rmsse_l12(train, test_actual, test_pred):
    """Невзвешенный средний RMSSE на уровне item x store (для разбора фейлов)."""
    tr_lvl = _agg_to_level(train, ["id"], "units")
    scale = _naive_scale(tr_lvl)
    te = test_actual.merge(test_pred[["id", TCOL, "pred"]], on=["id", TCOL], how="left",
                           validate="one_to_one")
    te["pred"] = te["pred"].fillna(0.0)
    act = _agg_to_level(te, ["id"], "units").rename(columns={"v": "v_actual"})
    prd = _agg_to_level(te.assign(units=te["pred"]), ["id"], "units").rename(columns={"v": "v_pred"})
    m = act.merge(prd, on=["__sid__", TCOL], how="left")
    rmsse = _rmsse_per_series(m, scale)
    return float(np.mean(list(rmsse.values()))) if rmsse else np.nan


def mase_l12(train, test_actual, test_pred):
    """MASE на item x store: MAE / средний |y_t - y_{t-1}| по train."""
    tr = _agg_to_level(train, ["id"], "units").sort_values(["__sid__", TCOL]).copy()
    tr["ad"] = tr.groupby("__sid__", observed=True)["v"].diff().abs()
    scale = tr.groupby("__sid__", observed=True)["ad"].mean()
    scale = scale[scale > 0]
    te = test_actual.merge(test_pred[["id", TCOL, "pred"]], on=["id", TCOL], how="left",
                           validate="one_to_one")
    te["pred"] = te["pred"].fillna(0.0)
    te["ae"] = (te["units"].astype(float) - te["pred"].astype(float)).abs()
    mae = te.groupby("id", observed=True)["ae"].mean()
    common = mae.index.intersection(scale.index)
    return float((mae.loc[common] / scale.loc[common]).mean()) if len(common) else np.nan


def bias(test_actual, test_pred):
    """Систематический сдвиг прогноза: (sum pred - sum actual) / sum actual."""
    te = test_actual.merge(test_pred[["id", TCOL, "pred"]], on=["id", TCOL], how="left",
                           validate="one_to_one")
    te["pred"] = te["pred"].fillna(0.0)
    a = te["units"].sum()
    return float((te["pred"].sum() - a) / a) if a > 0 else np.nan


def all_metrics(train, test_actual, test_pred, name=""):
    """Сводка по модели: WRMSSE + RMSSE(L12) + MASE + Bias."""
    w, levels = wrmsse(train, test_actual, test_pred, return_levels=True)
    return {
        "model": name,
        "WRMSSE": round(w, 4),
        "RMSSE_L12": round(rmsse_l12(train, test_actual, test_pred), 4),
        "MASE_L12": round(mase_l12(train, test_actual, test_pred), 4),
        "Bias": round(bias(test_actual, test_pred), 4),
        "_levels": {k: round(v, 4) for k, v in levels.items()},
    }


class Scorer:
    """Кэш scale и весов по 12 уровням на один train (фолд). Скорит много моделей
    без пересчёта тяжёлой части. Для walk-forward CV: один Scorer на фолд."""

    def __init__(self, train):
        self.level_scale, self.level_w = {}, {}
        for name, keys in LEVELS:
            self.level_scale[name] = pd.Series(_naive_scale(_agg_to_level(train, keys, "units")))
            self.level_w[name] = _level_weights(train, keys)
        # MASE scale на item x store
        tr = _agg_to_level(train, ["id"], "units").sort_values(["__sid__", TCOL])
        ad = tr.groupby("__sid__", observed=True)["v"].diff().abs()
        self._mase = ad.groupby(tr["__sid__"].values).mean()
        self._mase = self._mase[self._mase > 0]

    def score(self, test_actual, test_pred, name=""):
        # validate="one_to_one" ловит дубли (id, week): иначе строки тихо размножились бы
        te = test_actual.merge(test_pred[["id", TCOL, "pred"]], on=["id", TCOL],
                               how="left", validate="one_to_one")
        te["pred"] = te["pred"].fillna(0.0)
        levels = {}            # взвешенный RMSSE на уровень (для WRMSSE)
        rmsse_l12_unw = np.nan  # невзвешенный средний RMSSE на item×store (как rmsse_l12)
        for nm, keys in LEVELS:
            scale = self.level_scale[nm]
            if scale.empty:
                continue
            act = _agg_to_level(te, keys, "units").rename(columns={"v": "v_actual"})
            prd = _agg_to_level(te.assign(units=te["pred"]), keys, "units").rename(
                columns={"v": "v_pred"})
            m = act.merge(prd, on=["__sid__", TCOL], how="left")
            m["v_pred"] = m["v_pred"].fillna(0.0)
            rmsse = pd.Series(_rmsse_per_series(m, scale.to_dict()))
            if nm == "L12_item_store":
                rmsse_l12_unw = float(rmsse.mean()) if len(rmsse) else np.nan
            w = self.level_w[nm]
            common = rmsse.index.intersection(w.index)
            wsum = float(w.loc[common].sum())
            if wsum > 0:
                levels[nm] = float((rmsse.loc[common] * w.loc[common]).sum() / wsum)
        overall = float(np.mean(list(levels.values()))) if levels else np.nan
        # MASE + Bias
        te["ae"] = (te["units"].astype(float) - te["pred"].astype(float)).abs()
        mae = te.groupby("id", observed=True)["ae"].mean()
        common = mae.index.intersection(self._mase.index)
        mase = float((mae.loc[common] / self._mase.loc[common]).mean()) if len(common) else np.nan
        a = te["units"].sum()
        bs = float((te["pred"].sum() - a) / a) if a > 0 else np.nan
        return {
            "model": name,
            "WRMSSE": round(overall, 4),
            "RMSSE_L12": round(rmsse_l12_unw, 4),   # невзвешенный, согласован с rmsse_l12()
            "MASE_L12": round(mase, 4),
            "Bias": round(bs, 4),
            "_levels": {k: round(v, 4) for k, v in levels.items()},
        }
