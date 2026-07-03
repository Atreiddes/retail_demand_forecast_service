"""Операции с БД: прогон и пачки, чтение истории, запись точек, финализация.

Тяжёлое (чтение истории, пачечный upsert точек) идёт через pandas/raw psycopg, мелкое - через
ORM-сессию. Финализация прогона - атомарный переход по статусам пачек, без гонок.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import func, text
from sqlmodel import Session, select

from . import models
from .config import settings
from .db import engine
from .forecast import HIST_WEEKS

UPSERT = """
INSERT INTO forecast_point (run_id, series_id, week_start_date, h, p10, p50, p90)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, series_id, week_start_date)
DO UPDATE SET h = EXCLUDED.h, p10 = EXCLUDED.p10, p50 = EXCLUDED.p50, p90 = EXCLUDED.p90
"""


def create_run(store_id, cat_id, horizon, origin, model_version):
    """Создаёт прогон и пачки, коммитит, возвращает (run_id, сообщения для очереди)."""
    with Session(engine) as s:
        q = select(models.Series.id).where(models.Series.cat_id == cat_id)
        if store_id:
            q = q.where(models.Series.store_id == store_id)
        ids = list(s.exec(q.order_by(models.Series.id)).all())
        if not ids:
            raise ValueError("в выбранном срезе нет рядов")

        run = models.ForecastRun(
            origin=origin, horizon_weeks=horizon, mode="forecast",
            filter_json={"store_id": store_id, "cat_id": cat_id},
            status=models.QUEUED, n_series=len(ids), model_version=model_version)
        s.add(run)
        s.flush()

        groups = [ids[i:i + settings.chunk_size] for i in range(0, len(ids), settings.chunk_size)]
        run.n_chunks = len(groups)
        msgs = []
        for idx, cids in enumerate(groups):
            c = models.ForecastChunk(run_id=run.id, idx=idx, n_series=len(cids))
            s.add(c)
            s.flush()
            msgs.append({"run_id": run.id, "chunk_id": c.id, "series_ids": cids,
                         "origin": str(origin), "horizon": horizon, "model_version": model_version})
        s.commit()
        return run.id, msgs


def read_history(series_ids, origin):
    """История нужных рядов за окно origin-HIST_WEEKS..origin, с атрибутами ряда (колонка id)."""
    lo = (pd.Timestamp(origin) - pd.Timedelta(weeks=HIST_WEEKS + 1)).date()
    sql = text("""
        SELECT s.id, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
               h.week_start_date, h.units, h.revenue, h.sell_price,
               h.snap_days, h.event_days, h.available_days, h.n_days
        FROM sales_history h JOIN series s ON s.id = h.series_id
        WHERE h.series_id = ANY(:ids) AND h.week_start_date <= :origin AND h.week_start_date > :lo
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={
            "ids": list(series_ids), "origin": pd.Timestamp(origin).date(), "lo": lo})
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


def start_chunk(chunk_id, worker_id):
    with Session(engine) as s:
        s.execute(text("UPDATE forecast_chunk SET status='processing', worker_id=:w, started_at=now() "
                       "WHERE id=:id AND status='queued'"), {"w": worker_id, "id": chunk_id})
        s.commit()


def complete_chunk(run_id, chunk_id, df, worker_id):
    """Пачечный upsert точек и перевод пачки в COMPLETED в одной транзакции, потом коммит."""
    recs = [(int(run_id), r.series_id, r.week_start_date.date(), int(r.h),
             float(r.p10), float(r.p50), float(r.p90)) for r in df.itertuples(index=False)]
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.executemany(UPSERT, recs)
            cur.execute("UPDATE forecast_chunk SET status='completed', worker_id=%s, finished_at=now() "
                        "WHERE id=%s AND status <> 'completed'", (worker_id, chunk_id))
        raw.commit()
    finally:
        raw.close()


def fail_chunk(chunk_id, error, worker_id):
    with Session(engine) as s:
        s.execute(text("UPDATE forecast_chunk SET status='failed', worker_id=:w, finished_at=now(), error=:e "
                       "WHERE id=:id AND status NOT IN ('completed','failed')"),
                  {"w": worker_id, "e": str(error)[:500], "id": chunk_id})
        s.commit()


def fail_run(run_id):
    """Пометить прогон и его незабранные пачки FAILED (например, если публикация в очередь не удалась),
    чтобы прогон стал терминальным, а не завис в QUEUED."""
    with Session(engine) as s:
        s.execute(text("UPDATE forecast_chunk SET status='failed', error='не опубликовано в очередь' "
                       "WHERE run_id=:id AND status='queued'"), {"id": run_id})
        s.execute(text("UPDATE forecast_run SET status='failed', finished_at=now() "
                       "WHERE id=:id AND status IN ('queued','processing')"), {"id": run_id})
        s.commit()


def finalize_run(run_id):
    """Если все пачки терминальны - атомарно перевести прогон в COMPLETED или PARTIAL."""
    with Session(engine) as s:
        run = s.get(models.ForecastRun, run_id)
        if not run or run.status in (models.COMPLETED, models.PARTIAL, models.FAILED):
            return run
        counts = dict(s.exec(
            select(models.ForecastChunk.status, func.count())
            .where(models.ForecastChunk.run_id == run_id)
            .group_by(models.ForecastChunk.status)).all())
        if counts.get(models.COMPLETED, 0) + counts.get(models.FAILED, 0) < run.n_chunks:
            return run
        final = models.PARTIAL if counts.get(models.FAILED, 0) else models.COMPLETED
        s.execute(text("UPDATE forecast_run SET status=:f, finished_at=now() "
                       "WHERE id=:id AND status IN ('queued','processing')"), {"f": final, "id": run_id})
        s.commit()
        return s.get(models.ForecastRun, run_id)


def run_progress(run_id):
    """Статус прогона и число готовых пачек (с ленивой финализацией)."""
    with Session(engine) as s:
        run = s.get(models.ForecastRun, run_id)
        if run is None:
            return None
        done = s.exec(select(func.count()).where(
            models.ForecastChunk.run_id == run_id,
            models.ForecastChunk.status.in_((models.COMPLETED, models.FAILED)))).one()
    if run.status in (models.QUEUED, models.PROCESSING) and done >= run.n_chunks > 0:
        run = finalize_run(run_id)
    return run, done


def reconcile():
    """Дочинить прогоны, у которых все пачки уже терминальны (вызывается при старте api)."""
    with Session(engine) as s:
        ids = s.exec(select(models.ForecastRun.id).where(
            models.ForecastRun.status.in_((models.QUEUED, models.PROCESSING)))).all()
    for rid in ids:
        finalize_run(rid)


def stores():
    with Session(engine) as s:
        return list(s.exec(select(models.Series.store_id).where(
            models.Series.cat_id == "FOODS").distinct().order_by(models.Series.store_id)).all())


def recent_runs(limit=15):
    with Session(engine) as s:
        runs = s.exec(select(models.ForecastRun).order_by(
            models.ForecastRun.id.desc()).limit(limit)).all()
        return [{"id": r.id, "store_id": r.filter_json.get("store_id") or "все",
                 "status": r.status, "created_at": str(r.created_at)[:19]} for r in runs]


def catalog(run_id):
    sql = """SELECT p.series_id, s.item_id, sum(p.p50) AS sum_p50, avg(p.p90 - p.p10) AS interval_width
             FROM forecast_point p JOIN series s ON s.id = p.series_id
             WHERE p.run_id = %(rid)s GROUP BY p.series_id, s.item_id ORDER BY sum_p50 DESC"""
    return pd.read_sql_query(sql, engine, params={"rid": run_id}).round(2)


def series_forecast(run_id, series_id):
    """Точки прогноза ряда и его недавняя история (полные недели) для графика."""
    pts = pd.read_sql_query(
        "SELECT week_start_date, h, p10, p50, p90 FROM forecast_point "
        "WHERE run_id = %(rid)s AND series_id = %(sid)s ORDER BY h",
        engine, params={"rid": run_id, "sid": series_id})
    hist = pd.read_sql_query(
        "SELECT week_start_date, units FROM sales_history "
        "WHERE series_id = %(sid)s AND n_days = 7 ORDER BY week_start_date DESC LIMIT 30",
        engine, params={"sid": series_id}).iloc[::-1]
    return pts, hist


def export_rows(run_id):
    return pd.read_sql_query(
        "SELECT p.series_id, s.item_id, s.store_id, p.week_start_date, p.h, p.p10, p.p50, p.p90 "
        "FROM forecast_point p JOIN series s ON s.id = p.series_id "
        "WHERE p.run_id = %(rid)s ORDER BY p.series_id, p.h",
        engine, params={"rid": run_id})
