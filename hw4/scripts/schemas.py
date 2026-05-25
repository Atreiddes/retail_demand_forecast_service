"""Pandera-схемы для M5 pipeline.

Используется в build_weekly.py и в EDA.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
STATES = ["CA", "TX", "WI"]
STORES = [
    f"{s}_{i}" for s, n in [("CA", 4), ("TX", 3), ("WI", 3)] for i in range(1, n + 1)
]
DEPTS = [
    f"{c}_{i}" for c, n in [("FOODS", 3), ("HOBBIES", 2), ("HOUSEHOLD", 2)] for i in range(1, n + 1)
]
WM_YR_WK_MIN = 11101
WM_YR_WK_MAX = 11700


calendar_schema = DataFrameSchema(
    {
        "date": Column("datetime64[ns]"),
        "wm_yr_wk": Column("int32", Check.in_range(WM_YR_WK_MIN, WM_YR_WK_MAX)),
        "d_num": Column("int16", Check.in_range(1, 1969)),
        "snap_CA": Column("int8", Check.isin([0, 1])),
        "snap_TX": Column("int8", Check.isin([0, 1])),
        "snap_WI": Column("int8", Check.isin([0, 1])),
        "has_event": Column("int8", Check.in_range(0, 2)),
    },
    strict=False,
    coerce=True,
)

prices_schema = DataFrameSchema(
    {
        "store_id": Column(str, Check.isin(STORES)),
        "item_id": Column(str),
        "wm_yr_wk": Column("int32", Check.in_range(WM_YR_WK_MIN, WM_YR_WK_MAX)),
        "sell_price": Column("float32", Check.greater_than(0)),
    },
    strict=False,
)

sales_weekly_schema = DataFrameSchema(
    {
        "id": Column(str),
        "item_id": Column("category"),
        "dept_id": Column("category", Check.isin(DEPTS)),
        "cat_id": Column("category", Check.isin(CATEGORIES)),
        "store_id": Column("category", Check.isin(STORES)),
        "state_id": Column("category", Check.isin(STATES)),
        "wm_yr_wk": Column("int32", Check.in_range(WM_YR_WK_MIN, WM_YR_WK_MAX)),
        "week_start_date": Column("datetime64[ns]"),
        "units": Column("int32", Check.greater_than_or_equal_to(0)),
        "revenue": Column("float32", Check.greater_than_or_equal_to(0), nullable=True),
        "sell_price": Column("float32", Check.greater_than(0), nullable=True),
        "snap_days": Column("int8", Check.in_range(0, 7)),
        "available_days": Column("int8", Check.in_range(0, 7)),
        "event_days": Column("int8", Check.in_range(0, 14)),
        "n_days": Column("int8", Check.in_range(1, 7)),
    },
    strict=True,
    coerce=False,
)
