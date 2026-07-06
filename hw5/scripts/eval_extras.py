"""Доп. оценка на последнем фолде (с фактом): квантили, reconciliation, декомпозиция,
упущенные продажи. Модели заново не обучает: берёт готовые serve/forecast.parquet (p10/p50/p90)
и serve/model.txt от train_production.py на том же фолде make_folds[-1].
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import decompose as DEC
import features_direct as FD
import lost_sales as LS
import metrics as M
import quantiles as Q
import reconcile as REC
import run_cv as RC
import util as U
from util import EPS, TCOL

SERVE = ROOT.parent / "serve"


def _quantile_eval(forecast, test):
    """Покрытие и pinball служебных p10/p50/p90 против факта (то, что реально отдаём)."""
    q = forecast[["id", TCOL, "p10", "p50", "p90"]].rename(
        columns={"p10": "q0.1", "p50": "q0.5", "p90": "q0.9"})
    res = Q.quantile_metrics(test[["id", TCOL, "units"]], q)
    pd.Series(res).to_csv(ROOT.parent / "quantile_eval.csv")
    print("\n[квантили] покрытие и pinball (служебные p10/p50/p90 vs факт):")
    for k, v in res.items():
        print(f"  {k:16s} {v}")
    return res


def _reconcile_eval(forecast, train, test, scorer):
    """Bottom-up (p50) vs middle-out (масштаб к MA-прогнозу store x dept) по WRMSSE."""
    base = forecast[["id", TCOL, "p50"] + REC.ATTRS].rename(columns={"p50": "pred"})
    bu = scorer.score(test, base[["id", TCOL, "pred"]], name="bottom_up")["WRMSSE"]

    agg = REC.aggregate_ma_forecast(train, sorted(test[TCOL].unique()),
                                    ["store_id", "dept_id"], k=8)
    rec = REC.reconcile_proportional(base, agg, ["store_id", "dept_id"])
    gap = REC.reconciled_gap(REC.attach_keys(rec, train), agg, ["store_id", "dept_id"])
    mo = scorer.score(test, rec, name="middle_out")["WRMSSE"]

    out = pd.DataFrame([{"method": "bottom_up", "WRMSSE": bu},
                        {"method": "middle_out_storedept", "WRMSSE": mo}])
    out.to_csv(ROOT.parent / "reconcile_eval.csv", index=False)
    print(f"\n[reconciliation] bottom_up WRMSSE={bu}  middle_out WRMSSE={mo}  "
          f"(контроль когерентности, max|сумма-агрегат|={gap:.4f})")
    return out


def _decomp_example(frame, fold, train, test):
    """Декомпозиция прогноза топ-FOODS ряда: baseline + промо + сезонность + события."""
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(SERVE / "model.txt"))
    te = FD.select_test(frame, fold["test_w"])
    cal = FD.select_test(frame, fold["cal_w"])
    raw = np.clip(model.predict(cal[FD.FEATURES]), 0, None)
    factor = float(cal["units"].sum() / max(raw.sum(), EPS))

    top = (train[train["cat_id"] == "FOODS"].groupby(["item_id", "store_id"], observed=True)
           ["revenue"].sum().sort_values(ascending=False))
    for (item_id, store_id), _ in top.items():
        sub = te[(te["item_id"] == item_id) & (te["store_id"] == store_id)]
        if not sub.empty:
            break
    d = DEC.decompose_series(model, te, item_id, store_id, factor=factor)
    d.insert(0, "store_id", store_id)
    d.insert(0, "item_id", item_id)
    d.to_csv(ROOT.parent / "decomp_example.csv", index=False)
    print(f"\n[декомпозиция] {item_id} x {store_id} (factor {factor:.3f}):")
    print(d[["item_id", "store_id", TCOL, "full", "baseline",
             "promo", "seasonality", "events"]].to_string(index=False))
    return d


def main():
    t0 = time.perf_counter()
    if not (SERVE / "forecast.parquet").exists():
        sys.exit("нет serve/forecast.parquet: сначала запустите train_production.py")
    full, weeks = U.load_window(window=200)
    frame = FD.build_direct(full)
    fold = RC.make_folds(weeks)[-1]
    train = full[full[TCOL] <= fold["origin"]]
    test = full[full[TCOL].isin(fold["test_w"])].copy()
    forecast = pd.read_parquet(SERVE / "forecast.parquet")
    print(f"фолд origin={fold['origin'].date()}, тест "
          f"{fold['test_w'][0].date()}..{fold['test_w'][-1].date()}, "
          f"{forecast['id'].nunique():,} рядов", flush=True)

    scorer = M.Scorer(train)
    _quantile_eval(forecast, test)
    _reconcile_eval(forecast, train, test, scorer)
    _decomp_example(frame, fold, train, test)

    print("\n[упущенные продажи]")
    for k, v in LS.report(full).items():
        print(f"  {k:26s} {v}")
    print(f"\nготово за {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
