"""Доменная модель: ряды, история продаж, прогоны, пачки, точки прогноза."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import CheckConstraint, Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

# статусы прогона и пачки
NEW, QUEUED, PROCESSING, COMPLETED, PARTIAL, FAILED = (
    "new", "queued", "processing", "completed", "partial", "failed")


class Series(SQLModel, table=True):
    __tablename__ = "series"

    id: str = Field(primary_key=True)        # item_id + "_" + store_id
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str


class SalesHistory(SQLModel, table=True):
    __tablename__ = "sales_history"
    __table_args__ = (
        UniqueConstraint("series_id", "week_start_date", name="uq_series_week"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    series_id: str = Field(foreign_key="series.id")
    week_start_date: date
    units: int
    revenue: float
    sell_price: Optional[float] = None
    snap_days: int
    event_days: int
    available_days: int
    n_days: int


class ForecastRun(SQLModel, table=True):
    __tablename__ = "forecast_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    origin: date
    horizon_weeks: int = 4
    mode: str = "forecast"                    # forecast | backtest
    filter_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = NEW
    n_series: int = 0
    n_chunks: int = 0
    model_version: str = ""
    finished_at: Optional[datetime] = None


class ForecastChunk(SQLModel, table=True):
    __tablename__ = "forecast_chunk"
    __table_args__ = (UniqueConstraint("run_id", "idx", name="uq_run_idx"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="forecast_run.id", index=True)
    idx: int
    n_series: int
    status: str = QUEUED
    attempt: int = 0
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class ForecastPoint(SQLModel, table=True):
    __tablename__ = "forecast_point"
    __table_args__ = (
        CheckConstraint("p10 <= p50 AND p50 <= p90", name="ck_monotone"),
        CheckConstraint("h >= 1", name="ck_h"),
    )

    run_id: int = Field(foreign_key="forecast_run.id", primary_key=True)
    series_id: str = Field(foreign_key="series.id", primary_key=True)
    week_start_date: date = Field(primary_key=True)
    h: int
    p10: float
    p50: float
    p90: float
