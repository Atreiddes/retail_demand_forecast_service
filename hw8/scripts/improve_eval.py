"""Строгое сравнение признаков: базовая модель против модели с новыми признаками и MA-4.

Один протокол (walk-forward, шаг = горизонт, окна не пересекаются), один кадр признаков.
На каждом окне обучаются две LightGBM на идентичных строках:
- OLD - базовый набор признаков (как было),
- NEW - плюс прерывистость (nz_share) и заём силы (item_rmean_4, dept_rmean_4).
Плюс MA-4 как baseline. Считаем WRMSSE по 12 и по 9 различимым уровням, среднее/стд по окнам,
прирост NEW над OLD и над MA-4. Результат -> metrics/improve_eval.json.

Число окон - env N_FOLDS, шаг - env STEP (= горизонту для независимых окон).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "hw5" / "scripts"))
# непересекающиеся окна по умолчанию (шаг = горизонт): run_cv читает env на импорте,
# поэтому дефолты ставим до него, чтобы протокол совпал с докстрингом без ручного env
os.environ.setdefault("STEP", "8")
os.environ.setdefault("N_FOLDS", "4")
import features_direct as FD  # noqa: E402
import metrics as M  # noqa: E402
import model_lgb as LGB  # noqa: E402
import run_cv as RC  # noqa: E402
import util as U  # noqa: E402
from util import TCOL  # noqa: E402

DATA = ROOT / "data" / "foods_weekly.parquet"
OUT = ROOT / "metrics" / "improve_eval.json"
UNIQUE = ["L1_total", "L2_state", "L3_store", "L5_dept", "L7_state_dept",
          "L9_store_dept", "L10_item", "L11_item_state", "L12_item_store"]
NEWCOLS = ["nz_share_4", "nz_share_13", "item_rmean_4", "dept_rmean_4"]
OLD_FEATS = [f for f in FD.FEATURES if f not in NEWCOLS]
NEW_FEATS = list(FD.FEATURES)
PARAMS = {"num_threads": 4}
ROUNDS = 400


def wr(lv):
    return sum(lv.values()) / len(lv), sum(lv[k] for k in UNIQUE) / len(UNIQUE)


def fit_predict(frame, fold, feats):
    """Обучение на <= cal_origin, фактор смещения на cal-блоке, прогноз теста, маска OOS."""
    tr = FD.train_slice(frame, fold["cal_origin"], U.TRAIN_CAP)
    cal = FD.select_test(frame, fold["cal_w"])
    model = LGB.train(tr, PARAMS, num_boost_round=ROUNDS, feats=feats)
    factor = float(cal["units"].sum() / max(LGB.predict(model, cal, feats=feats)["pred"].sum(), U.EPS))
    te = FD.select_test(frame, fold["test_w"])
    p = LGB.predict(model, te, feats=feats).merge(te[["id", TCOL, "available_days"]], on=["id", TCOL])
    p["pred"] = np.where(p["available_days"] == 0, 0.0, p["pred"] * factor)
    return p[["id", TCOL, "pred"]]


def main():
    full, weeks = U.load_window(window=200, data=DATA)
    frame = FD.build_direct(full)
    folds = RC.make_folds(weeks)
    print(f"окон={len(folds)} горизонт={U.HORIZON} шаг={RC.STEP} "
          f"OLD={len(OLD_FEATS)} NEW={len(NEW_FEATS)} признаков", flush=True)

    res = {"old": [], "new": [], "ma4": []}
    for fi, fold in enumerate(folds, 1):
        train = full[full[TCOL] <= fold["origin"]]
        test = full[full[TCOL].isin(fold["test_w"])]
        scorer = M.Scorer(train)

        preds = {
            "old": fit_predict(frame, fold, OLD_FEATS),
            "new": fit_predict(frame, fold, NEW_FEATS),
            "ma4": U.moving_average(train, test),
        }
        for name, pred in preds.items():
            lv = scorer.score(test, pred)["_levels"]
            w12, w9 = wr(lv)
            res[name].append({"wr12": w12, "wr9": w9, "L12": lv["L12_item_store"]})
        print(f"  окно {fi} origin={pd.Timestamp(fold['origin']).date()} "
              f"old={res['old'][-1]['wr12']:.4f} new={res['new'][-1]['wr12']:.4f} "
              f"ma4={res['ma4'][-1]['wr12']:.4f}", flush=True)

    out = {"n_folds": len(folds), "horizon": int(U.HORIZON), "step": int(RC.STEP),
           "new_cols": NEWCOLS}
    for name, rows in res.items():
        a12 = np.array([r["wr12"] for r in rows]); a9 = np.array([r["wr9"] for r in rows])
        aL12 = np.array([r["L12"] for r in rows])
        out[name] = {"wr12_mean": round(float(a12.mean()), 4), "wr12_std": round(float(a12.std()), 4),
                     "wr9_mean": round(float(a9.mean()), 4), "wr9_std": round(float(a9.std()), 4),
                     "L12_mean": round(float(aL12.mean()), 4)}
    o, n, m = out["old"]["wr12_mean"], out["new"]["wr12_mean"], out["ma4"]["wr12_mean"]
    out["new_vs_old_pct"] = round((o - n) / o * 100, 1)
    out["new_vs_ma4_pct"] = round((m - n) / m * 100, 1)
    out["old_vs_ma4_pct"] = round((m - o) / m * 100, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("итог:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
