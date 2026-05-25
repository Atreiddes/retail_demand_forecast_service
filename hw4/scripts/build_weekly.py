"""Daily wide M5 -> weekly long parquet.

Вход (hw4/data/raw/):
  sales_train_evaluation.csv, calendar.csv, sell_prices.csv

Выход (hw4/data/processed/):
  sales_weekly.parquet, calendar.parquet, prices.parquet
"""
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import calendar_schema, prices_schema, sales_weekly_schema

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal = cal.sort_values("date").reset_index(drop=True)
    cal["d_num"] = pd.Series(range(1, len(cal) + 1), dtype="int16")
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int32")
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        cal[c] = cal[c].astype("int8")
    cal["has_event"] = (
        cal["event_name_1"].notna().astype("int8")
        + cal["event_name_2"].notna().astype("int8")
    )
    return cal


def load_prices() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "sell_prices.csv",
        dtype={"wm_yr_wk": "int32", "sell_price": "float32"},
    )


def load_sales_long(cal: pd.DataFrame) -> pd.DataFrame:
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    if "id" not in sales.columns:
        sales.insert(0, "id", sales["item_id"] + "_" + sales["store_id"] + "_evaluation")
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        sales[c] = sales[c].astype("category")

    long = sales.melt(
        id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="units",
    )
    long["units"] = long["units"].astype("int16")
    long["d_num"] = long["d"].str.removeprefix("d_").astype("int16")
    long = long.drop(columns=["d"])

    long = long.merge(
        cal[[
            "d_num", "date", "wm_yr_wk",
            "snap_CA", "snap_TX", "snap_WI",
            "has_event",
        ]],
        on="d_num", how="left",
    )

    state = long["state_id"].astype(str)
    snap = pd.Series(0, index=long.index, dtype="int8")
    snap[state == "CA"] = long.loc[state == "CA", "snap_CA"].to_numpy()
    snap[state == "TX"] = long.loc[state == "TX", "snap_TX"].to_numpy()
    snap[state == "WI"] = long.loc[state == "WI", "snap_WI"].to_numpy()
    long["snap"] = snap
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    return long


def join_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["revenue"] = (long["units"] * long["sell_price"]).astype("float32")
    long["available"] = long["sell_price"].notna().astype("int8")
    return long


def aggregate_weekly(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "wm_yr_wk"]
    weekly = long.groupby(keys, observed=True).agg(
        week_start_date=("date", "min"),
        units=("units", "sum"),
        revenue=("revenue", "sum"),
        sell_price=("sell_price", "mean"),
        snap_days=("snap", "sum"),
        available_days=("available", "sum"),
        event_days=("has_event", "sum"),
        n_days=("units", "size"),
    ).reset_index()

    weekly["units"] = weekly["units"].astype("int32")
    weekly["revenue"] = weekly["revenue"].astype("float32")
    weekly["sell_price"] = weekly["sell_price"].astype("float32")
    for c in ("snap_days", "available_days", "event_days", "n_days"):
        weekly[c] = weekly[c].astype("int8")
    for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        weekly[c] = weekly[c].astype("category")
    return weekly


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("[1/5] calendar + prices")
    cal = load_calendar()
    calendar_schema.validate(cal, lazy=True)
    prices = load_prices()
    prices_schema.validate(prices, lazy=True)

    print("[2/5] melt sales")
    long = load_sales_long(cal)
    print(f"      shape: {long.shape}")

    print("[3/5] join prices")
    long = join_prices(long, prices)

    print("[4/5] aggregate weekly")
    weekly = aggregate_weekly(long)
    print(f"      shape: {weekly.shape}")

    print("[5/5] validate output")
    sales_weekly_schema.validate(weekly, lazy=True)

    weekly.to_parquet(OUT / "sales_weekly.parquet", engine="pyarrow", compression="snappy")
    cal.to_parquet(OUT / "calendar.parquet", engine="pyarrow", compression="snappy")
    prices.to_parquet(OUT / "prices.parquet", engine="pyarrow", compression="snappy")

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")
    for f in ("sales_weekly.parquet", "calendar.parquet", "prices.parquet"):
        size_mb = (OUT / f).stat().st_size / 1024 / 1024
        print(f"  {f:30s} {size_mb:7.1f} MB")


if __name__ == "__main__":
    main()
