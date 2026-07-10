"""Наполнение PostgreSQL FOODS-панелью M5 через COPY. Идемпотентно: если пусто."""
from __future__ import annotations

import io
import logging

import pandas as pd
from sqlmodel import Session, select

from . import log, models
from .config import settings
from .db import create_db, engine
from .ml.schemas import validate_sample, weekly_input_schema

_log = logging.getLogger("forecast.seed")

SERIES_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
HIST_COLS = ["series_id", "week_start_date", "units", "revenue", "sell_price",
             "snap_days", "event_days", "available_days", "n_days"]


def _copy_all(tables):
    """Загрузка нескольких таблиц одной транзакцией: либо всё, либо ничего.
    tables: список (df, table, cols). Так половинчатое наполнение (ряды без истории) невозможно."""
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            for df, table, cols in tables:
                buf = io.StringIO()
                df[cols].to_csv(buf, index=False, header=False, na_rep="")
                buf.seek(0)
                with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')") as cp:
                    cp.write(buf.read())
        raw.commit()
    finally:
        raw.close()


def main():
    log.setup()
    create_db()
    with Session(engine) as s:
        if s.exec(select(models.Series).limit(1)).first():
            _log.info("база уже наполнена, пропускаю")
            return

    df = pd.read_parquet(settings.data_path)
    validate_sample(df, weekly_input_schema)
    series = df[SERIES_COLS].drop_duplicates("id")
    hist = df.rename(columns={"id": "series_id"})[HIST_COLS]
    _copy_all([(series, "series", SERIES_COLS), (hist, "sales_history", HIST_COLS)])
    _log.info("загружено: %s рядов, %s строк истории", f"{len(series):,}", f"{len(hist):,}")


if __name__ == "__main__":
    main()
