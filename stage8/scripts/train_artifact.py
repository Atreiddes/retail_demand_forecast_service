"""Собирает полный артефакт модели под фиксированный origin (последняя полная неделя).

Точечная LightGBM (Tweedie) + две квантильные модели (p10/p90) + калибровочный фактор
+ словарь категорий + список признаков + карточка модели. Воркер сервиса это только
применяет, ничего не обучает. Запуск из окружения с lightgbm/pandas (как у stage5):

    python scripts/train_artifact.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "stage5" / "scripts"))
import features_direct as FD
import model_lgb as LGB
import util as U
from util import TCOL, TRAIN_CAP

# каталог сборки переопределяется окружением: переобучение по расписанию собирает
# в artifacts_staging и промоутит в artifacts только после порога качества,
# чтобы воркеры не подхватили непроверенную модель
ART = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
DATA = ROOT / "data" / "foods_weekly.parquet"
QUANTILES = (0.1, 0.9)   # p50 = калиброванная точечная, q0.5 не нужен
HORIZON = 8


def _train(frame, origin, params, rounds):
    return LGB.train(FD.train_slice(frame, origin, TRAIN_CAP), params, num_boost_round=rounds)


def main():
    import lightgbm as lgb

    t0 = time.perf_counter()
    ART.mkdir(exist_ok=True)
    full, weeks = U.load_window(window=200, data=DATA)
    frame = FD.build_direct(full)
    origin = pd.Timestamp(weeks[-1])                 # последняя полная неделя
    cal_origin = pd.Timestamp(weeks[-1 - HORIZON])   # оценка фактора на OOS-блоке
    cal_w = [pd.Timestamp(w) for w in weeks[-HORIZON:]]
    print(f"origin={origin.date()}  рядов={full['id'].nunique():,}  строк frame={len(frame):,}", flush=True)

    def step(msg, fn):
        s = time.perf_counter()
        r = fn()
        print(f"  {msg} за {time.perf_counter() - s:.0f}s", flush=True)
        return r

    # производственные модели обучены на всей истории до origin
    point = step("точечная", lambda: _train(frame, origin, None, 400))
    point.save_model(str(ART / "model.txt"))
    for q in QUANTILES:
        step(f"квантиль {q}", lambda q=q: _train(frame, origin, {"objective": "quantile", "alpha": q}, 250)
             ).save_model(str(ART / f"q{q}.txt"))

    # калибровочный фактор: модель до cal_origin, оценка на cal-блоке (вне обучения)
    cal_model = step("калибровочная", lambda: _train(frame, cal_origin, None, 250))
    cal_pred = U.predict_masked(cal_model, FD.select_test(frame, cal_w))
    actual = full[full[TCOL].isin(cal_w)]
    factor = float(actual["units"].sum() / max(cal_pred["pred"].sum(), U.EPS))

    cats = {c: list(map(str, frame[c].cat.categories)) for c in FD.CAT_FEATURES}
    (ART / "categories.json").write_text(json.dumps(cats, ensure_ascii=False), encoding="utf-8")
    (ART / "features.json").write_text(json.dumps({
        "features": FD.FEATURES, "cat_features": FD.CAT_FEATURES,
        "use_events": FD.USE_EVENTS, "feature_name": point.feature_name(),
    }, ensure_ascii=False), encoding="utf-8")
    (ART / "calibration.json").write_text(json.dumps({"factor": factor, "origin": str(origin.date())}))
    (ART / "model_card.json").write_text(json.dumps({
        "model_version": f"foods-{origin.date()}",
        "origin": str(origin.date()),
        "horizon_weeks": HORIZON,
        "use_events": FD.USE_EVENTS,
        "lightgbm_version": lgb.__version__,
        "trained_at": str(date.today()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"factor={factor:.3f}  (WRMSSE по уровням считает scripts/foods_metrics.py)")
    print(f"готово за {time.perf_counter() - t0:.0f}s -> {ART}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Сборка артефакта модели в каталог")
    p.add_argument("--artifact-dir", default=os.environ.get("ARTIFACT_DIR"),
                   help="каталог сборки (по умолчанию ARTIFACT_DIR или ./artifacts)")
    args = p.parse_args()
    if args.artifact_dir:
        ART = Path(args.artifact_dir)
    main()
