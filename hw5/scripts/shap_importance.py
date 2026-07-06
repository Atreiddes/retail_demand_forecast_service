"""Интерпретация LightGBM: gain-важности + SHAP (нативный pred_contrib, без shap-зависимости).

Берёт прод-модель serve/model.txt (точечный LightGBM на последнем фолде, готовит
train_production.py), считает важность по gain и средний |SHAP| на выборке теста.
Артефакт -> hw5/importance.csv, визуализация в baseline.qmd.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_direct as FD
import run_cv as RC
import util as U

TCOL = "week_start_date"
MODEL = ROOT.parent / "serve" / "model.txt"


def main():
    import lightgbm as lgb

    t0 = time.perf_counter()
    if not MODEL.exists():
        sys.exit("нет serve/model.txt: сначала запустите serve/train_production.py")
    full, weeks = U.load_window(window=200)
    fold = RC.make_folds(weeks)[-1]
    frame = FD.build_direct(full)
    m = lgb.Booster(model_file=str(MODEL))
    print(f"модель загружена, фрейм построен за {time.perf_counter()-t0:.0f}s", flush=True)

    gain = m.feature_importance(importance_type="gain")
    feats = m.feature_name()

    # SHAP через нативный pred_contrib (LightGBM): последний столбец = base value
    te = FD.select_test(frame, fold["test_w"])
    Xs = te[FD.FEATURES].sample(min(30000, len(te)), random_state=0)
    contrib = m.predict(Xs, pred_contrib=True)
    mean_abs_shap = np.abs(contrib[:, :-1]).mean(axis=0)

    imp = pd.DataFrame({"feature": feats, "gain": gain, "mean_abs_shap": mean_abs_shap})
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(ROOT.parent / "importance.csv", index=False)
    print("\nтоп-15 признаков по среднему |SHAP|:")
    print(imp.head(15)[["feature", "gain_pct", "mean_abs_shap"]].to_string(index=False))
    print(f"\nготово за {time.perf_counter()-t0:.0f}s -> hw5/importance.csv")


if __name__ == "__main__":
    main()
