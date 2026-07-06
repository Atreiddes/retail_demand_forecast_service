"""Честная оценка на непересекающихся окнах: LightGBM против MA-4 и против гибрида.

Три прогноза на одном протоколе (walk-forward, шаг = горизонт, окна не пересекаются):
- модель LightGBM (как в сервисе),
- MA-4 (скользящее среднее 4 недель) - baseline,
- гибрид: прерывистые ряды (мало недель с продажами) отдаём на MA, остальное - модель.

Считает WRMSSE по 12 уровням и по 9 различимым, mean/std по окнам, отрыв модели и гибрида
над MA-4. Результат -> metrics/baseline_hybrid.json. Отвечает на вопросы критики К1/К2/К4/К6.
Число окон - env N_FOLDS, шаг - env STEP (ставим = горизонту для независимых окон).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "hw5" / "scripts"))
import features_direct as FD  # noqa: E402
import metrics as M  # noqa: E402
import run_cv as RC  # noqa: E402
import util as U  # noqa: E402
from util import TCOL  # noqa: E402

DATA = ROOT / "data" / "foods_weekly.parquet"
OUT = ROOT / "metrics" / "baseline_hybrid.json"
UNIQUE = ["L1_total", "L2_state", "L3_store", "L5_dept", "L7_state_dept",
          "L9_store_dept", "L10_item", "L11_item_state", "L12_item_store"]
INTERMITTENT_SHARE = 0.5  # прерывистый ряд: менее половины недель с продажами


def wr(lv):
    return sum(lv.values()) / len(lv), sum(lv[k] for k in UNIQUE) / len(UNIQUE)


def main():
    full, weeks = U.load_window(window=200, data=DATA)
    frame = FD.build_direct(full)
    folds = RC.make_folds(weeks)
    print(f"окон={len(folds)} горизонт={U.HORIZON} шаг={RC.STEP}", flush=True)

    res = {"model": [], "ma4": [], "hybrid": []}
    for fi, fold in enumerate(folds, 1):
        train = full[full[TCOL] <= fold["origin"]]
        test = full[full[TCOL].isin(fold["test_w"])]
        scorer = M.Scorer(train)

        model = U.calibrate_and_predict(frame, fold, {"num_threads": 4}, 400)[0]
        ma4 = U.moving_average(train, test)

        # прерывистые ряды по обучающему окну -> берём MA, остальное -> модель
        share = train.groupby("id")["units"].apply(lambda s: float((s > 0).mean()))
        interm = set(share[share < INTERMITTENT_SHARE].index)
        m = model.set_index(["id", TCOL])["pred"]
        a = ma4.set_index(["id", TCOL])["pred"]
        hybrid = m.copy()
        mask = hybrid.index.get_level_values("id").isin(interm)
        hybrid[mask] = a.reindex(hybrid.index[mask]).to_numpy()
        hybrid = hybrid.reset_index().rename(columns={0: "pred"})
        hybrid.columns = ["id", TCOL, "pred"]

        for name, pred in [("model", model), ("ma4", ma4), ("hybrid", hybrid)]:
            lv = scorer.score(test, pred)["_levels"]
            w12, w9 = wr(lv)
            res[name].append({"wr12": w12, "wr9": w9, "L12": lv["L12_item_store"]})
        print(f"  окно {fi} origin={pd.Timestamp(fold['origin']).date()} "
              f"model={res['model'][-1]['wr12']:.4f} ma4={res['ma4'][-1]['wr12']:.4f} "
              f"hybrid={res['hybrid'][-1]['wr12']:.4f} "
              f"(прерывистых рядов {len(interm)})", flush=True)

    out = {"n_folds": len(folds), "horizon": int(U.HORIZON), "step": int(RC.STEP),
           "intermittent_share": INTERMITTENT_SHARE}
    for name, rows in res.items():
        a12 = np.array([r["wr12"] for r in rows]); a9 = np.array([r["wr9"] for r in rows])
        aL12 = np.array([r["L12"] for r in rows])
        out[name] = {"wr12_mean": round(float(a12.mean()), 4), "wr12_std": round(float(a12.std()), 4),
                     "wr9_mean": round(float(a9.mean()), 4), "wr9_std": round(float(a9.std()), 4),
                     "L12_mean": round(float(aL12.mean()), 4)}
    m12, b12 = out["ma4"]["wr12_mean"], out["model"]["wr12_mean"]
    h12 = out["hybrid"]["wr12_mean"]
    out["model_vs_ma4_pct"] = round((m12 - b12) / m12 * 100, 1)
    out["hybrid_vs_ma4_pct"] = round((m12 - h12) / m12 * 100, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("итог:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
