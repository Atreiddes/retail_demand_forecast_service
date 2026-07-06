"""Walk-forward CV всех моделей на M5 weekly + калибровка смещения LightGBM.

Модели: seasonal naive, MA-4, структурный байес, LightGBM Tweedie (default),
LightGBM + Optuna и их ансамбль. Метрики WRMSSE / RMSSE / MASE / Bias усредняются по фолдам.
LightGBM: direct multi-horizon, калибровка смещения вне выборки и маска OOS
(общая логика в util.calibrate_and_predict). Усложнённая модель тюнится Optuna по WRMSSE
на срезе до первого тестового блока.

Конфиг через env: N_FOLDS(8) STEP(1) N_TRIALS(15) TRAIN_CAP_WEEKS(104) WINDOW(185).
Запуск интерпретатором с numpyro+jax+lightgbm (Python <= 3.12).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bayes_structural as bs
import features_direct as FD
import metriclog as ML
import metrics as M
import model_lgb as LGB
import schemas as S
import util as U
from util import HORIZON, TCOL, TRAIN_CAP

N_FOLDS = int(os.getenv("N_FOLDS", "8"))
STEP = int(os.getenv("STEP", "1"))
N_TRIALS = int(os.getenv("N_TRIALS", "15"))


def make_folds(weeks):
    """Walk-forward: origin, 4 тестовые недели, cal-блок (origin-3..origin), cal_origin=origin-4."""
    folds = []
    for k in range(N_FOLDS):
        e = len(weeks) - 1 - STEP * k
        o = e - HORIZON
        folds.append({
            "origin": weeks[o], "test_w": list(weeks[o + 1:e + 1]),
            "cal_origin": weeks[o - HORIZON], "cal_w": list(weeks[o - HORIZON + 1:o + 1]),
        })
    return list(reversed(folds))


def tune_params(frame, full, first_origin):
    """Optuna на срезе до первого тестового блока (без утечки), подвыборка 15% рядов."""
    cal_o = first_origin - pd.Timedelta(weeks=HORIZON)
    val_w = sorted(frame[frame[TCOL] <= first_origin][TCOL].unique())[-HORIZON:]
    tr = FD.train_slice(frame, cal_o, TRAIN_CAP)
    val = FD.select_test(frame, val_w)
    keep = set(pd.Series(tr["id"].unique()).sample(frac=0.15, random_state=1))
    scorer = M.Scorer(full[full[TCOL] <= first_origin])
    print(f"  tune: {(tr['id'].isin(keep)).sum():,} строк, {N_TRIALS} trials (по WRMSSE)", flush=True)
    best, study = LGB.tune(tr[tr["id"].isin(keep)], val, scorer, val,
                           n_trials=N_TRIALS, metric="WRMSSE")
    print(f"  tune best WRMSSE={study.best_value:.4f}", flush=True)
    return best


def _ensemble(p_a, p_b):
    """Среднее двух калиброванных прогнозов (Tweedie и тюнингованный)."""
    m = p_a.merge(p_b, on=["id", TCOL], suffixes=("_a", "_b"))
    m["pred"] = (m["pred_a"] + m["pred_b"]) / 2
    return m[["id", TCOL, "pred"]]


def _report_extras(rows, last):
    """Тест значимости (lgb_ensemble vs MA по фолдам) и разрез ошибки по cat×state."""
    df = pd.DataFrame(rows)
    piv = df.pivot(index="fold", columns="model", values="WRMSSE")
    if {"lgb_ensemble", "moving_avg"} <= set(piv.columns):
        d = piv["moving_avg"] - piv["lgb_ensemble"]   # > 0 -> lgb лучше
        print(f"\n[значимость] lgb_ensemble vs moving_avg: лучше в {(d > 0).sum()}/{len(d)} "
              f"фолдах, средняя Δ WRMSSE {d.mean():+.4f} ± {d.std():.4f}")
        print("  (фолды шаг 1 перекрываются -> оценка directional, не строго независимая)")
    test = last.get("_test")
    if test is not None:
        seg = []
        for name in ("moving_avg", "lgb_tweedie", "lgb_ensemble"):
            p = last.get(name)
            if p is None:
                continue
            te = test.merge(p, on=["id", TCOL])
            te["ae"] = (te["units"] - te["pred"]).abs()
            g = te.groupby(["cat_id", "state_id"], observed=True).agg(
                ae=("ae", "sum"), a=("units", "sum"))
            seg.append((g["ae"] / g["a"]).round(3).rename(name))
        out = pd.concat(seg, axis=1).reset_index()
        out.to_csv(ROOT.parent / "cv_segment_lastfold.csv", index=False)
        print("\n[разрез cat×state, последний фолд, WAPE]")
        print(out.to_string(index=False))


def main():
    t0 = time.perf_counter()
    full, weeks = U.load_window()
    S.validate_sample(full, S.weekly_input_schema)
    folds = make_folds(weeks)
    print(f"данные: {len(full):,} строк, {len(weeks)} недель")
    for f in folds:
        print(f"  origin={f['origin'].date()} test={f['test_w'][0].date()}..{f['test_w'][-1].date()}")

    print("\nстрою direct-признаки (один раз) ...", flush=True)
    tf = time.perf_counter()
    frame = FD.build_direct(full)
    S.validate_sample(frame, S.features_schema)
    print(f"  frame: {frame.shape} за {time.perf_counter()-tf:.0f}s", flush=True)

    print("\nтюнинг LightGBM по WRMSSE (усложнённая) ...", flush=True)
    best = tune_params(frame, full, folds[0]["origin"])

    rows, factors, last = [], [], {}
    for fi, fold in enumerate(folds, 1):
        print(f"\n=== fold {fi}/{len(folds)}  origin={fold['origin'].date()} ===", flush=True)
        train = full[full[TCOL] <= fold["origin"]]
        test = full[full[TCOL].isin(fold["test_w"])].copy()
        scorer = M.Scorer(train)
        tt = time.perf_counter()
        models = {
            "seasonal_naive": U.seasonal_naive(train, test),
            "moving_avg": U.moving_average(train, test),
            "bayes_structural": bs.forecast(train, test, group="dept_id", horizon=HORIZON),
        }
        p_def, f1, _ = U.calibrate_and_predict(frame, fold, None, 400)
        p_tun, f2, _ = U.calibrate_and_predict(frame, fold, best, 2000, early_stop=True)
        models["lgb_tweedie"], models["lgb_optuna"] = p_def, p_tun
        models["lgb_ensemble"] = _ensemble(p_def, p_tun)
        factors.append((fi, round(f1, 3), round(f2, 3)))
        print(f"  модели за {time.perf_counter()-tt:.0f}s (calib factor {f1:.3f}/{f2:.3f})", flush=True)
        for name, p in models.items():
            S.prediction_schema.validate(p)
            res = scorer.score(test, p, name=name)
            res["fold"] = fi
            rows.append({k: v for k, v in res.items() if k != "_levels"})
            if fi == len(folds):
                last[name] = p
                if name == "lgb_ensemble":
                    pd.Series(res["_levels"]).to_csv(ROOT.parent / "lgb_levels_lastfold.csv")
                    last["_test"] = test
            print(f"  {name:18s} WRMSSE={res['WRMSSE']}", flush=True)

    _report_extras(rows, last)
    df = pd.DataFrame(rows)
    agg = df.groupby("model").agg(
        WRMSSE_mean=("WRMSSE", "mean"), WRMSSE_std=("WRMSSE", "std"),
        RMSSE_L12=("RMSSE_L12", "mean"), MASE_L12=("MASE_L12", "mean"),
        Bias=("Bias", "mean")).round(4).sort_values("WRMSSE_mean")
    print("\n=== WALK-FORWARD CV: среднее по фолдам ===")
    print(agg.to_string())
    print("\ncalib factors (fold, lgb_tweedie, lgb_optuna):", factors)
    df.to_csv(ROOT.parent / "cv_per_fold.csv", index=False)
    agg.to_csv(ROOT.parent / "cv_summary.csv")
    ML.snapshot(agg, folds=len(folds), note="run_cv")  # автоведение metriclog
    print(f"\nготово за {time.perf_counter()-t0:.0f}s -> hw5/cv_summary.csv, metriclog.md")


if __name__ == "__main__":
    main()
