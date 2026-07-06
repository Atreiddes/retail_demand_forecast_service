"""Cold-start: генерализация LightGBM на новые item_id.

5% случайных item_id держим вне обучения и сравниваем с моделью, обученной на всех,
на одних и тех же cold-рядах. Разрыв - цена незнакомого товара. Метрики на уровне ряда
(RMSSE_L12, MASE_L12) нечувствительны к агрегатному смещению, поэтому калибровку не применяем.
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
import metrics as M
import model_lgb as LGB
import schemas as S
import util as U
from util import HORIZON, TCOL, TRAIN_CAP

COLD_FRAC = 0.05


def main():
    t0 = time.perf_counter()
    full, weeks = U.load_window()
    origin = weeks[-HORIZON - 1]
    test_w = list(weeks[-HORIZON:])
    print(f"origin={origin.date()} test={test_w[0].date()}..{test_w[-1].date()}")

    items = pd.Series(sorted(full["item_id"].astype(str).unique()))
    cold = set(items.sample(frac=COLD_FRAC, random_state=7))
    print(f"cold item_id: {len(cold)} ({COLD_FRAC:.0%})")

    frame = FD.build_direct(full)
    is_cold = frame["item_id"].astype(str).isin(cold)
    tr_all = FD.train_slice(frame, origin, TRAIN_CAP)
    tr_cold_mask = tr_all["item_id"].astype(str).isin(cold)
    te_cold = FD.select_test(frame, test_w).pipe(lambda d: d[d["item_id"].astype(str).isin(cold)])

    print("обучаю LGB warm-only и all ...", flush=True)
    m_warm = LGB.train(tr_all[~tr_cold_mask], num_boost_round=400)
    m_all = LGB.train(tr_all, num_boost_round=400)
    p_warm = U.predict_masked(m_warm, te_cold)
    p_all = U.predict_masked(m_all, te_cold)

    train_cold = full[(full[TCOL] <= origin) & full["item_id"].astype(str).isin(cold)]
    test_cold = full[full[TCOL].isin(test_w) & full["item_id"].astype(str).isin(cold)]
    p_ma = U.moving_average(train_cold, test_cold)

    scorer = M.Scorer(train_cold)
    rows = []
    for name, p in [("seasonal/MA-4", p_ma),
                    ("LGB warm-only (не видел cold)", p_warm),
                    ("LGB all (видел cold, reference)", p_all)]:
        S.prediction_schema.validate(p)
        r = scorer.score(test_cold, p, name=name)
        rows.append({"model": name, "RMSSE_L12": r["RMSSE_L12"], "MASE_L12": r["MASE_L12"]})
        print(f"  {name:34s} RMSSE_L12={r['RMSSE_L12']} MASE_L12={r['MASE_L12']}", flush=True)

    pd.DataFrame(rows).set_index("model").to_csv(ROOT.parent / "cold_start.csv")
    print(f"\nготово за {time.perf_counter()-t0:.0f}s -> hw5/cold_start.csv")
    print("разрыв warm-only vs all = генерализация global-модели на новинки")


if __name__ == "__main__":
    main()
