"""WRMSSE по уровням иерархии на FOODS (один фолд бэктеста) для таблицы метрик в UI.

Переиспользует пайплайн hw5: модель до cal_origin, прогноз на последних реальных неделях, Scorer
по 12 уровням. Результат -> hw8/metrics/cv_summary_foods.csv (колонки level, value).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "hw5" / "scripts"))
import features_direct as FD
import metrics as M
import run_cv as RC
import util as U
from util import TCOL

DATA = ROOT / "data" / "foods_weekly.parquet"
OUT = ROOT / "metrics" / "cv_summary_foods.csv"


def main():
    full, weeks = U.load_window(window=200, data=DATA)
    frame = FD.build_direct(full)
    fold = RC.make_folds(weeks)[-1]
    pred, _, _ = U.calibrate_and_predict(frame, fold, None, 400)
    test = full[full[TCOL].isin(fold["test_w"])]
    res = M.Scorer(full[full[TCOL] <= fold["origin"]]).score(test, pred)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "value"])
        for level, value in res["_levels"].items():
            w.writerow([level, round(float(value), 4)])
    print(f"origin={pd.Timestamp(fold['origin']).date()} WRMSSE={res['WRMSSE']} -> {OUT}")
    for k, v in res["_levels"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
