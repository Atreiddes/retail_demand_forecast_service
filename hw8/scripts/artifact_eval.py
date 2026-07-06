"""Оценка собранного артефакта на последнем окне: WRMSSE именно того model.txt,
который деплоится (а не модели, обученной заново внутри бэктеста).

Прогноз строится тем же кодом, что и в сервисе (forecast_series), пачками рядов.
Результат -> metrics/artifact_eval.json; порог качества при переобучении читает его.
Запуск в окружении обучения (hw5): нужны pandas/lightgbm и скрипты метрик hw5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "hw5" / "scripts"))
import metrics as M  # noqa: E402

from forecast_service.forecast import forecast_series, load_artifact  # noqa: E402
from forecast_service.ml.features import HMAX, TCOL  # noqa: E402

DATA = ROOT / "data" / "foods_weekly.parquet"
OUT = ROOT / "metrics" / "artifact_eval.json"
CHUNK = 300
UNIQUE = ["L1_total", "L2_state", "L3_store", "L5_dept", "L7_state_dept",
          "L9_store_dept", "L10_item", "L11_item_state", "L12_item_store"]


def main():
    art = load_artifact()
    df = pd.read_parquet(DATA)
    df = df[df["n_days"] == 7]  # только полные недели: обрезанная хвостовая неделя иначе
    # сравнивается с полнонедельным прогнозом и завышает WRMSSE в разы (и валит гейт)
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[c] = df[c].astype(str)
    weeks = sorted(df[TCOL].unique())
    origin = weeks[-HMAX - 1]  # последнее окно: HMAX недель факта после origin
    print(f"артефакт {art['model_version']}, origin оценки {pd.Timestamp(origin).date()}", flush=True)

    ids = sorted(df["id"].unique())
    hist_all = df[df[TCOL] <= origin]
    # кросс-рядные агрегаты по всему срезу (как воркер берёт из БД), а не по пачке в 300 рядов
    full7 = hist_all[hist_all["n_days"] == 7]
    item_agg = full7.groupby(["item_id", TCOL])["units"].sum().rename("item_wk").reset_index()
    dept_agg = full7.groupby(["dept_id", TCOL])["units"].sum().rename("dept_wk").reset_index()
    preds = []
    for i in range(0, len(ids), CHUNK):
        chunk_ids = set(ids[i:i + CHUNK])
        hist = hist_all[hist_all["id"].isin(chunk_ids)]
        out = forecast_series(hist, origin, HMAX, item_agg, dept_agg)
        preds.append(out[["series_id", TCOL, "p50"]])
    pred = pd.concat(preds, ignore_index=True).rename(columns={"series_id": "id", "p50": "pred"})

    test = df[df[TCOL] > origin]
    lv = M.Scorer(hist_all).score(test, pred)["_levels"]
    res = {
        "model_version": art["model_version"],
        "eval_origin": str(pd.Timestamp(origin).date()),
        "horizon": int(HMAX),
        "wrmsse12": round(sum(lv.values()) / len(lv), 4),
        "wrmsse9": round(sum(lv[k] for k in UNIQUE) / len(UNIQUE), 4),
        "levels": {k: round(float(v), 4) for k, v in lv.items()},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("итог:", res["wrmsse12"], "/", res["wrmsse9"], "->", OUT, flush=True)


if __name__ == "__main__":
    main()
