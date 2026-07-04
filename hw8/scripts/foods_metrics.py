"""WRMSSE walk-forward по уровням на FOODS (горизонт из util.HORIZON) для метрик в UI.

Многофолдовый walk-forward: WRMSSE по 12 уровням и по 9 различимым (на срезе одной
категории L4/L6/L8 дублируют L1/L2/L3, их усреднение завышает качество). Результат ->
cv_summary_foods.csv (средние по уровням) и metrics_summary.json (mean/std по фолдам,
12 и 9 уровней, per-fold). Число фолдов - env N_FOLDS (по умолчанию 8).
Переиспользует пайплайн hw5. Запуск в его окружении.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "hw5" / "scripts"))
import features_direct as FD  # noqa: E402
import metrics as M  # noqa: E402
import run_cv as RC  # noqa: E402
import util as U  # noqa: E402
from util import TCOL  # noqa: E402

DATA = ROOT / "data" / "foods_weekly.parquet"
OUT = ROOT / "metrics" / "cv_summary_foods.csv"
SUMMARY = ROOT / "metrics" / "metrics_summary.json"
# на срезе одной категории FOODS уровни L4/L6/L8 дублируют L1/L2/L3, берём только различимые
UNIQUE = ["L1_total", "L2_state", "L3_store", "L5_dept", "L7_state_dept",
          "L9_store_dept", "L10_item", "L11_item_state", "L12_item_store"]


def main():
    full, weeks = U.load_window(window=200, data=DATA)
    frame = FD.build_direct(full)
    folds = RC.make_folds(weeks)
    print(f"фолдов={len(folds)} горизонт={U.HORIZON}", flush=True)

    rows = []
    for fi, fold in enumerate(folds, 1):
        pred, _, _ = U.calibrate_and_predict(frame, fold, {"num_threads": 4}, 400)
        test = full[full[TCOL].isin(fold["test_w"])]
        lv = M.Scorer(full[full[TCOL] <= fold["origin"]]).score(test, pred)["_levels"]
        wr12 = sum(lv.values()) / len(lv)
        wr9 = sum(lv[k] for k in UNIQUE) / len(UNIQUE)
        rows.append({"fold": fi, "wr12": wr12, "wr9": wr9, **lv})
        print(f"  фолд {fi} origin={pd.Timestamp(fold['origin']).date()} "
              f"WRMSSE(12)={wr12:.4f} WRMSSE(9)={wr9:.4f}", flush=True)

    df = pd.DataFrame(rows)
    level_names = [k for k in df.columns if k.startswith("L")]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "value"])
        for lv in level_names:
            w.writerow([lv, round(float(df[lv].mean()), 4)])

    summary = {
        "n_folds": len(folds),
        "horizon": int(U.HORIZON),
        "wrmsse12_mean": round(float(df["wr12"].mean()), 4),
        "wrmsse12_std": round(float(df["wr12"].std()), 4),
        "wrmsse9_mean": round(float(df["wr9"].mean()), 4),
        "wrmsse9_std": round(float(df["wr9"].std()), 4),
        "per_fold_wrmsse12": [round(float(x), 4) for x in df["wr12"]],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("итог:", summary, flush=True)


if __name__ == "__main__":
    main()
