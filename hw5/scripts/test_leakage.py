"""Тест отсутствия утечки из будущего в признаках direct multi-horizon.

Признаки на момент origin не должны зависеть от того, что происходит после origin.
Портим будущие units и проверяем, что признаки train-строк (<= origin) и тест-строк не меняются.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_direct as FD

TCOL = "week_start_date"
DATA = ROOT.parent.parent / "hw4" / "data" / "processed" / "sales_weekly.parquet"
FEAT_COLS = [c for c in FD.FEATURES if c not in FD.CAT_FEATURES]  # числовые признаки


def _subset():
    w = pd.read_parquet(DATA)
    full = w[w["n_days"] == 7].copy()
    items = pd.Series(full["item_id"].cat.categories).sample(60, random_state=0)
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

    # портим только будущий таргет (units). Цена будущих недель в M5 известна заранее
    # и законно входит признаком, её не трогаем.
    pert = full.copy()
    fut = pert[TCOL] > origin
    pert.loc[fut, "units"] = 99999
    pdir = FD.build_direct(pert).sort_values(["id", TCOL, "h"]).reset_index(drop=True)

    # train-строки (target week <= origin): признаки обязаны совпасть бит-в-бит
    m = base[TCOL] <= origin
    a = base.loc[m, FEAT_COLS].to_numpy(dtype=float)
    b = pdir.loc[m.values, FEAT_COLS].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True), "УТЕЧКА: train-признаки изменились от будущего"

    # тест-строки из реального origin (target = origin+h): признаки тоже не должны зависеть
    # от недель после origin (используют units <= origin)
    te_keys = FD.select_test(base, test_w)[["id", TCOL, "h"]]
    te_a = te_keys.merge(base, on=["id", TCOL, "h"])[FEAT_COLS].to_numpy(float)
    te_b = te_keys.merge(pdir, on=["id", TCOL, "h"])[FEAT_COLS].to_numpy(float)
    # порча недель после origin не влияет на тест (origin+h использует <= origin):
    # меняется только таргет (units), но не входные признаки
    assert np.allclose(te_a, te_b, equal_nan=True), "УТЕЧКА: тест-признаки зависят от будущего"
    print("OK: train- и тест-признаки инвариантны к будущему (утечки нет)")


def test_lag_uses_only_past():
    """lag_k для горизонта h берёт units[w-h-(k-1)] <= w-h < w (строго прошлое)."""
    full = _subset()
    frame = FD.build_direct(full)
    one = frame[frame["id"] == frame["id"].iloc[0]].sort_values([TCOL, "h"])
    # для h=1 lag_1 = units на предыдущей неделе (origin = w-1)
    s = full[full["id"] == one["id"].iloc[0]].sort_values(TCOL).set_index(TCOL)["units"]
    row = one[one["h"] == 1].iloc[20]
    expected = s.shift(1).loc[row[TCOL]]
    assert np.isnan(row["lag_1"]) or abs(row["lag_1"] - expected) < 1e-6
    print("OK: lag_1 при h=1 равен units предыдущей недели (origin = w-1)")


if __name__ == "__main__":
    test_no_future_leak()
    test_lag_uses_only_past()
    print("все тесты на утечку пройдены")
