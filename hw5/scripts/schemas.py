"""Pandera-контракты данных для hw5: вход, фрейм признаков, прогнозы.

Валидируют пайплайн на границах (вход -> признаки -> прогноз) и падают при отклонении
схемы. На больших фреймах валидируем выборку (validate_sample) ради скорости.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
STATES = ["CA", "TX", "WI"]
TCOL = "week_start_date"

# Вход: weekly-панель M5 (то, что hw5 реально читает из hw4).
weekly_input_schema = DataFrameSchema(
    {
        "id": Column(str, nullable=False),
        "item_id": Column(nullable=False),
        "dept_id": Column(nullable=False),
        "cat_id": Column(nullable=False, checks=Check.isin(CATEGORIES)),
        "store_id": Column(nullable=False),
        "state_id": Column(nullable=False, checks=Check.isin(STATES)),
        TCOL: Column("datetime64[ns]"),
        "units": Column(checks=Check.greater_than_or_equal_to(0), coerce=True),
        "revenue": Column(nullable=True, coerce=True),
        "sell_price": Column(nullable=True, checks=Check.greater_than(0), coerce=True),
        "snap_days": Column(checks=Check.in_range(0, 7), coerce=True),
        "available_days": Column(checks=Check.in_range(0, 7), coerce=True),
        "event_days": Column(checks=Check.in_range(0, 14), coerce=True),
        "n_days": Column(checks=Check.in_range(1, 7), coerce=True),
    },
    strict=False,
    coerce=False,
)

# Фрейм direct multi-horizon признаков (выход features_direct.build_direct).
features_schema = DataFrameSchema(
    {
        "id": Column(str),
        TCOL: Column("datetime64[ns]"),
        "units": Column(checks=Check.greater_than_or_equal_to(0), coerce=True),
        "h": Column(checks=Check.in_range(1, 4), coerce=True),
        "is_promo": Column(checks=Check.isin([0, 1]), coerce=True),
        "is_xmas": Column(checks=Check.isin([0, 1]), coerce=True),
        "month": Column(checks=Check.in_range(1, 12), coerce=True),
        "available_days": Column(checks=Check.in_range(0, 7), coerce=True),
        "snap_days": Column(checks=Check.in_range(0, 7), coerce=True),
        "age_weeks": Column(checks=Check.greater_than_or_equal_to(0), coerce=True),
        "woy_sin": Column(checks=Check.in_range(-1.001, 1.001), coerce=True),
        "woy_cos": Column(checks=Check.in_range(-1.001, 1.001), coerce=True),
        # лаги/rolling nullable: на ранней истории ряда NaN (LightGBM их обрабатывает)
        "lag_1": Column(nullable=True, coerce=True),
        "rmean_4": Column(nullable=True, coerce=True),
    },
    strict=False,
    coerce=False,
)

# Прогноз модели.
prediction_schema = DataFrameSchema(
    {
        "id": Column(str, nullable=False),
        TCOL: Column("datetime64[ns]"),
        "pred": Column(
            float,
            nullable=False,
            checks=[Check.greater_than_or_equal_to(0), Check(lambda s: s.notna().all(),
                    error="прогноз содержит NaN")],
            coerce=True,
        ),
    },
    strict=False,
)


def validate_sample(df, schema, n=50_000, seed=42):
    """Валидация на выборке для скорости. Возвращает df без изменений."""
    sample = df.sample(min(n, len(df)), random_state=seed) if len(df) > n else df
    schema.validate(sample, lazy=True)
    return df
