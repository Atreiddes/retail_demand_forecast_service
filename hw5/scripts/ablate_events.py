"""Абляция event one-hot: вклад недельных признаков событий в WRMSSE.

USE_EVENTS читается features_direct при импорте, поэтому переключается через env и два процесса:
  USE_EVENTS=0 N_FOLDS=3 python ablate_events.py
  USE_EVENTS=1 N_FOLDS=3 python ablate_events.py
Сравниваем средний WRMSSE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_direct as FD
import metrics as M
import run_cv as RC
import util as U
from util import TCOL


def main():
    t0 = time.perf_counter()
    print(f"USE_EVENTS={int(FD.USE_EVENTS)}  (признаков: {len(FD.FEATURES)})", flush=True)
    full, weeks = U.load_window(window=200)
    frame = FD.build_direct(full)
    folds = RC.make_folds(weeks)
    ws = []
    for fi, fold in enumerate(folds, 1):
        train = full[full[TCOL] <= fold["origin"]]
        test = full[full[TCOL].isin(fold["test_w"])].copy()
        scorer = M.Scorer(train)
        pred, factor, _ = U.calibrate_and_predict(frame, fold, None, 400)
        w = scorer.score(test, pred, name="lgb")["WRMSSE"]
        ws.append(w)
        print(f"  fold {fi} origin={fold['origin'].date()} WRMSSE={w} (factor {factor:.3f})",
              flush=True)
    print(f"USE_EVENTS={int(FD.USE_EVENTS)} mean WRMSSE={np.mean(ws):.4f} "
          f"over {len(ws)} folds {[round(x,4) for x in ws]}")
    print(f"готово за {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
