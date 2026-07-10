"""Тюнинг Optuna на NEW-признаках + замер прироста над дефолтом, старыми признаками и MA-4.

Протокол без утечки: самое раннее окно (folds[0]) отдаём под Optuna (валидация на блоке до
его origin), меряем на тех же 4 более поздних окнах, что и improve_eval (folds[1:]). Optuna
минимизирует WRMSSE на 15% рядов (как в run_cv). Для каждого эвал-окна считаем:
- OLD - базовые признаки, дефолтные параметры,
- NEW - новые признаки, дефолтные параметры,
- NEW+TUNE - новые признаки, параметры Optuna (2000 раундов, early stopping),
- MA-4.
Результат + лучшие параметры -> metrics/tune_eval.json.

Env: N_TRIALS (число проб Optuna), STEP=8, N_FOLDS=5 (1 на тюнинг + 4 на замер).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "stage5" / "scripts"))
# 1 окно на тюнинг + 4 на замер, непересекающиеся (шаг = горизонт); run_cv читает env на импорте
os.environ.setdefault("STEP", "8")
os.environ.setdefault("N_FOLDS", "5")
import features_direct as FD  # noqa: E402
import metrics as M  # noqa: E402
import model_lgb as LGB  # noqa: E402
import run_cv as RC  # noqa: E402
import util as U  # noqa: E402
from util import TCOL  # noqa: E402

DATA = ROOT / "data" / "foods_weekly.parquet"
OUT = ROOT / "metrics" / "tune_eval.json"
UNIQUE = ["L1_total", "L2_state", "L3_store", "L5_dept", "L7_state_dept",
          "L9_store_dept", "L10_item", "L11_item_state", "L12_item_store"]
NEWCOLS = ["nz_share_4", "nz_share_13", "item_rmean_4", "dept_rmean_4"]
OLD_FEATS = [f for f in FD.FEATURES if f not in NEWCOLS]


def wr(lv):
    return sum(lv.values()) / len(lv), sum(lv[k] for k in UNIQUE) / len(UNIQUE)


def fit_old(frame, fold):
    """OLD-признаки, дефолтные параметры, калибровка смещения, маска OOS."""
    tr = FD.train_slice(frame, fold["cal_origin"], U.TRAIN_CAP)
    cal = FD.select_test(frame, fold["cal_w"])
    model = LGB.train(tr, {"num_threads": 4}, num_boost_round=400, feats=OLD_FEATS)
    factor = float(cal["units"].sum() / max(LGB.predict(model, cal, feats=OLD_FEATS)["pred"].sum(), U.EPS))
    te = FD.select_test(frame, fold["test_w"])
    p = LGB.predict(model, te, feats=OLD_FEATS).merge(te[["id", TCOL, "available_days"]], on=["id", TCOL])
    p["pred"] = np.where(p["available_days"] == 0, 0.0, p["pred"] * factor)
    return p[["id", TCOL, "pred"]]


def main():
    full, weeks = U.load_window(window=200, data=DATA)
    frame = FD.build_direct(full)
    folds = RC.make_folds(weeks)
    tune_fold, eval_folds = folds[0], folds[1:]
    print(f"тюнинг-окно origin={pd.Timestamp(tune_fold['origin']).date()}; "
          f"эвал-окон={len(eval_folds)} шаг={RC.STEP} N_TRIALS={RC.N_TRIALS}", flush=True)

    best = RC.tune_params(frame, full, tune_fold["origin"])  # Optuna по WRMSSE, NEW = FEATURES по умолчанию
    print("лучшие параметры:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in best.items()}, ensure_ascii=False), flush=True)

    res = {"old": [], "new": [], "tune": [], "ma4": []}
    for fi, fold in enumerate(eval_folds, 1):
        train = full[full[TCOL] <= fold["origin"]]
        test = full[full[TCOL].isin(fold["test_w"])]
        scorer = M.Scorer(train)
        preds = {
            "old": fit_old(frame, fold),
            "new": U.calibrate_and_predict(frame, fold, None, 400)[0],
            "tune": U.calibrate_and_predict(frame, fold, best, 2000, early_stop=True)[0],
            "ma4": U.moving_average(train, test),
        }
        for name, pred in preds.items():
            lv = scorer.score(test, pred)["_levels"]
            w12, w9 = wr(lv)
            res[name].append({"wr12": w12, "wr9": w9, "L12": lv["L12_item_store"]})
        print(f"  окно {fi} origin={pd.Timestamp(fold['origin']).date()} "
              f"old={res['old'][-1]['wr12']:.4f} new={res['new'][-1]['wr12']:.4f} "
              f"tune={res['tune'][-1]['wr12']:.4f} ma4={res['ma4'][-1]['wr12']:.4f}", flush=True)

    out = {"n_eval_folds": len(eval_folds), "step": int(RC.STEP), "n_trials": int(RC.N_TRIALS),
           "new_cols": NEWCOLS, "best_params": {k: v for k, v in best.items()}}
    for name, rows in res.items():
        a12 = np.array([r["wr12"] for r in rows])
        a9 = np.array([r["wr9"] for r in rows])
        aL12 = np.array([r["L12"] for r in rows])
        out[name] = {"wr12_mean": round(float(a12.mean()), 4), "wr12_std": round(float(a12.std()), 4),
                     "wr9_mean": round(float(a9.mean()), 4), "wr9_std": round(float(a9.std()), 4),
                     "L12_mean": round(float(aL12.mean()), 4)}
    o, n, t, m = (out["old"]["wr12_mean"], out["new"]["wr12_mean"],
                  out["tune"]["wr12_mean"], out["ma4"]["wr12_mean"])
    out["new_vs_old_pct"] = round((o - n) / o * 100, 1)
    out["tune_vs_new_pct"] = round((n - t) / n * 100, 1)
    out["tune_vs_old_pct"] = round((o - t) / o * 100, 1)
    out["tune_vs_ma4_pct"] = round((m - t) / m * 100, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("итог:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
