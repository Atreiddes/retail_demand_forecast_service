"""Простые бейзлайны по рядам на тех же фолдах плюс metriclog с приростом.

Считает seasonal_ma и croston/SBA рядом с seasonal_naive и moving_avg; последние два служат
сверкой и должны воспроизвести cv_summary. Сводит metriclog с приростом WRMSSE относительно
moving_avg.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import metriclog as ML
import metrics as M
import run_cv as RC
import util as U
from util import TCOL

NEW = {
    "seasonal_naive": U.seasonal_naive,
    "moving_avg": U.moving_average,
    "seasonal_ma": U.seasonal_ma,
    "croston_sba": U.croston,
}


def main():
    t0 = time.perf_counter()
    full, weeks = U.load_window(window=200)
    folds = RC.make_folds(weeks)
    print(f"данные: {len(full):,} строк, фолдов {len(folds)}", flush=True)

    rows = []
    for fi, fold in enumerate(folds, 1):
        train = full[full[TCOL] <= fold["origin"]]
        test = full[full[TCOL].isin(fold["test_w"])].copy()
        scorer = M.Scorer(train)
        for name, fn in NEW.items():
            res = scorer.score(test, fn(train, test), name=name)
            rows.append({k: v for k, v in res.items() if k != "_levels"} | {"fold": fi})
            print(f"  fold {fi} {name:14s} WRMSSE={res['WRMSSE']}", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("model").agg(
        WRMSSE_mean=("WRMSSE", "mean"), WRMSSE_std=("WRMSSE", "std"),
        RMSSE_L12=("RMSSE_L12", "mean"), MASE_L12=("MASE_L12", "mean"),
        Bias=("Bias", "mean")).round(4)
    df.to_csv(ROOT.parent / "baselines_extra_perfold.csv", index=False)

    # сверка: seasonal_naive / moving_avg должны совпасть с cv_summary
    cv = pd.read_csv(ROOT.parent / "cv_summary.csv", index_col=0)
    print("\n[сверка с cv_summary] (должны совпасть)")
    for m in ("seasonal_naive", "moving_avg"):
        new_v, old_v = agg.loc[m, "WRMSSE_mean"], cv.loc[m, "WRMSSE_mean"]
        ok = "OK" if abs(new_v - old_v) < 1e-3 else "РАСХОЖДЕНИЕ"
        print(f"  {m:14s} new={new_v} cv={old_v} {ok}")

    # metriclog: логируем тяжёлые модели из cv_summary и новые бейзлайны по рядам, рисуем лестницу
    ts = ML._now()
    ML.log_summary(cv[["WRMSSE_mean", "RMSSE_L12", "MASE_L12", "Bias"]], folds=len(folds),
                   note="cv_summary", ts=ts)
    ML.log_summary(agg.loc[["seasonal_ma", "croston_sba"]], folds=len(folds),
                   note="baselines_extra", ts=ts)
    ML.render()
    print("\n=== METRICLOG (эволюция) ===")
    print((ROOT.parent / "metriclog.md").read_text(encoding="utf-8"))
    print(f"готово за {time.perf_counter()-t0:.0f}s -> metriclog.md / metriclog.csv")


if __name__ == "__main__":
    main()
