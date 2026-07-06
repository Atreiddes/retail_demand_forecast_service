"""Проверка forecast.py без БД: прогноз на нескольких рядах FOODS прямо из parquet.

Имитирует то, что воркер получит из БД (история ряда с атрибутами), и проверяет монотонность
квантилей, отсутствие NaN и что недели прогноза идут вперёд от origin.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from forecast_service.forecast import forecast_series, load_artifact


def main():
    art = load_artifact()
    origin = art["origin"]
    df = pd.read_parquet(ROOT / "data" / "foods_weekly.parquet")
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[c] = df[c].astype(str)
    ids = sorted(df["id"].unique())[:50]
    hist = df[df["id"].isin(ids) & (df["week_start_date"] <= origin)]

    # кросс-рядные агрегаты по всему срезу (как воркер берёт из БД), не по 50 рядам пачки
    full = df[(df["n_days"] == 7) & (df["week_start_date"] <= origin)]
    item_agg = full.groupby(["item_id", "week_start_date"])["units"].sum().rename("item_wk").reset_index()
    dept_agg = full.groupby(["dept_id", "week_start_date"])["units"].sum().rename("dept_wk").reset_index()

    out = forecast_series(hist, origin, 4, item_agg, dept_agg)
    print(out.head(12).to_string(index=False))
    bad = ((out["p10"] > out["p50"]) | (out["p50"] > out["p90"])).sum()
    nan = out[["p10", "p50", "p90"]].isna().sum().sum()
    print(f"\nрядов={out['series_id'].nunique()} строк={len(out)} "
          f"нарушений монотонности={int(bad)} NaN={int(nan)}")
    print("origin:", origin.date(), "| недели прогноза:",
          [str(d) for d in sorted(out['week_start_date'].dt.date.unique())])
    print(f"модель: {art['model_version']}  WRMSSE(FOODS)={art.get('wrmsse')}")


if __name__ == "__main__":
    main()
