"""Тест отсутствия утечки из будущего в признаках direct (перенос из hw5).

Признаки на момент origin не должны зависеть от того, что после origin. Портим будущие units и
проверяем, что признаки train-строк и тест-строк не меняются. Работает на FOODS-срезе сервиса.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_service.config import settings
from forecast_service.ml import features as FD

TCOL = "week_start_date"
FEAT_COLS = [c for c in FD.FEATURES if c not in FD.CAT_FEATURES]


def _subset():
    w = pd.read_parquet(settings.data_path)
    full = w[w["n_days"] == 7].copy()
    items = pd.Series(full["item_id"].astype("category").cat.categories).sample(40, random_state=0)
    full = full[full["item_id"].isin(items)].copy()
    for c in FD.CAT_FEATURES:
        full[c] = full[c].astype(str).astype("category")
    return full


def test_no_future_leak():
    full = _subset()
    weeks = np.array(sorted(full[TCOL].unique()))
    origin = weeks[-6]
    test_w = list(weeks[-5:-1])

    base = FD.build_direct(full).sort_values(["id", TCOL, "h"]).reset_index(drop=True)
    pert = full.copy()
    pert.loc[pert[TCOL] > origin, "units"] = 99999
    pdir = FD.build_direct(pert).sort_values(["id", TCOL, "h"]).reset_index(drop=True)

    m = base[TCOL] <= origin
    a = base.loc[m, FEAT_COLS].to_numpy(dtype=float)
    b = pdir.loc[m.values, FEAT_COLS].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True), "утечка: train-признаки изменились от будущего"

    te_keys = FD.select_test(base, test_w)[["id", TCOL, "h"]]
    te_a = te_keys.merge(base, on=["id", TCOL, "h"])[FEAT_COLS].to_numpy(float)
    te_b = te_keys.merge(pdir, on=["id", TCOL, "h"])[FEAT_COLS].to_numpy(float)
    assert np.allclose(te_a, te_b, equal_nan=True), "утечка: тест-признаки зависят от будущего"


def test_lag_uses_only_past():
    full = _subset()
    frame = FD.build_direct(full)
    one = frame[frame["id"] == frame["id"].iloc[0]].sort_values([TCOL, "h"])
    s = full[full["id"] == one["id"].iloc[0]].sort_values(TCOL).set_index(TCOL)["units"]
    row = one[one["h"] == 1].iloc[20]
    expected = s.shift(1).loc[row[TCOL]]
    assert np.isnan(row["lag_1"]) or abs(row["lag_1"] - expected) < 1e-6
