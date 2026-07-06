"""Согласование прогнозов по иерархии M5 (forecast reconciliation).

Прогноз строится на нижнем уровне item x store, суммы вверх задают остальные уровни
(bottom-up, когерентно по построению). Здесь добавляем пропорциональное (middle-out)
согласование: нижние прогнозы масштабируем к прогнозу более устойчивого агрегата группы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TCOL = "week_start_date"
EPS = 1e-9
ATTRS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def attach_keys(pred, full):
    """Добавить к прогнозу [id, week, pred] атрибуты иерархии (item/dept/cat/store/state)."""
    keys = full[["id"] + ATTRS].drop_duplicates("id")
    return pred.merge(keys, on="id", how="left", validate="many_to_one")


def aggregate(pred_keyed, keys, value="pred"):
    """Суммировать прогноз на уровень keys по каждой неделе. keys=[] -> один тотал."""
    if not keys:
        g = pred_keyed.groupby(TCOL, observed=True)[value].sum().reset_index()
        g["__grp__"] = "TOTAL"
        return g
    g = pred_keyed.groupby(keys + [TCOL], observed=True)[value].sum().reset_index()
    g["__grp__"] = g[keys].astype(str).agg("|".join, axis=1)
    return g


def reconciled_gap(pred_keyed, agg_forecast, group_keys):
    """Контроль: после согласования сумма нижнего уровня по группе равна прогнозу агрегата.
    Возвращает max|сумма - agg|, для непустых групп должно быть около нуля."""
    bu = aggregate(pred_keyed, group_keys).rename(columns={"pred": "bu"})
    m = bu.merge(agg_forecast, on=group_keys + [TCOL], how="inner")
    m = m[m["bu"] > EPS]  # пустые группы согласование не трогает
    return float((m["bu"] - m["agg"]).abs().max()) if len(m) else 0.0


def reconcile_proportional(pred_keyed, agg_forecast, group_keys):
    """Middle-out: масштабировать нижние прогнозы к прогнозу агрегата группы.

    Возвращает [id, week, pred] с сохранённой суммой по группе, равной agg. Если сумма
    группы нулевая, прогноз не трогаем (нечего перераспределять).
    """
    p = pred_keyed.copy()
    grp_sum = p.groupby(group_keys + [TCOL], observed=True)["pred"].transform("sum")
    p = p.merge(agg_forecast, on=group_keys + [TCOL], how="left", validate="many_to_one")
    factor = np.where(grp_sum.to_numpy() > EPS, p["agg"].to_numpy() / grp_sum.to_numpy(), 1.0)
    p["pred"] = p["pred"].to_numpy() * np.where(np.isfinite(factor), factor, 1.0)
    return p[["id", TCOL, "pred"]]


def aggregate_ma_forecast(train, test_weeks, group_keys, k=8):
    """Прогноз агрегата группы скользящим средним: сумма units по группе на неделю, среднее
    последних k недель train, повтор на тестовые недели. Устойчивая опора для middle-out."""
    g = train.groupby(group_keys + [TCOL], observed=True)["units"].sum().reset_index()
    last = sorted(g[TCOL].unique())[-k:]
    lvl = (g[g[TCOL].isin(last)].groupby(group_keys, observed=True)["units"].mean()
           .rename("agg").reset_index())
    rows = []
    for w in test_weeks:
        r = lvl.copy()
        r[TCOL] = w
        rows.append(r)
    return pd.concat(rows, ignore_index=True)
