"""Операции с БД: прогон и пачки, чтение истории, запись точек, финализация.

Тяжёлое (чтение истории, пачечный upsert точек) идёт через pandas/raw psycopg, мелкое - через
ORM-сессию. Финализация прогона - атомарный переход по статусам пачек, без гонок.
"""
from __future__ import annotations

import numpy as np
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
    # нормализация типа на границе: пайплайн и контракты pandera обучены на pyarrow-строках
    # из parquet, а из Postgres id приходит object - приводим здесь, контракт остаётся строгим
    df["id"] = df["id"].astype("string[pyarrow]")
    return df


# агрегат зависит только от origin и одинаков для всех пачек прогона - кэшируем в процессе,
# иначе каждый воркер гоняет два полных скана sales_history на каждую пачку
_AGG_CACHE: dict = {}


def read_item_dept_weekly(origin):
    """Недельные суммы спроса по товару (во всех магазинах) и по отделу за окно
    origin-HIST_WEEKS..origin. Воркеру нужны как кросс-рядные признаки: пачка - подмножество
    рядов, а сумма должна быть по всему срезу товара/отдела, как на обучении.
    Возвращает (item_df[item_id, week, item_wk], dept_df[dept_id, week, dept_wk])."""
    key = str(pd.Timestamp(origin).date())
    if key in _AGG_CACHE:
        return _AGG_CACHE[key]
    lo = (pd.Timestamp(origin) - pd.Timedelta(weeks=HIST_WEEKS + 1)).date()
    params = {"origin": key, "lo": lo}

    def agg(col, name):
        sql = text(f"""
            SELECT s.{col} AS {col}, h.week_start_date, SUM(h.units) AS {name}
            FROM sales_history h JOIN series s ON s.id = h.series_id
            WHERE h.n_days = 7 AND h.week_start_date <= :origin AND h.week_start_date > :lo
            GROUP BY s.{col}, h.week_start_date
        """)
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        df["week_start_date"] = pd.to_datetime(df["week_start_date"])
        df[name] = df[name].astype("float32")
        return df

    res = (agg("item_id", "item_wk"), agg("dept_id", "dept_wk"))
    _AGG_CACHE[key] = res
    if len(_AGG_CACHE) > 8:            # ограничение кэша: несколько последних origin
        _AGG_CACHE.pop(next(iter(_AGG_CACHE)))
    return res


def start_chunk(chunk_id, worker_id):
    with Session(engine) as s:
        s.execute(text("UPDATE forecast_chunk SET status='processing', worker_id=:w, started_at=now() "
                       "WHERE id=:id AND status='queued'"), {"w": worker_id, "id": chunk_id})
        s.commit()


def complete_chunk(run_id, chunk_id, df, worker_id):
    """Перевод пачки в COMPLETED и пачечный upsert точек одной транзакцией. Точки пишутся только
    если пачка была в работе (processing): пачку, помеченную failed (например при сбое публикации),
    повторно доставленное сообщение не воскрешает."""
    recs = [(int(run_id), r.series_id, r.week_start_date.date(), int(r.h),
             float(r.p10), float(r.p50), float(r.p90)) for r in df.itertuples(index=False)]
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("UPDATE forecast_chunk SET status='completed', worker_id=%s, finished_at=now() "
                        "WHERE id=%s AND status='processing'", (worker_id, chunk_id))
            if cur.rowcount:
                cur.executemany(UPSERT, recs)
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


def run_status_counts():
    """Число прогонов в каждом статусе, для гейджей мониторинга."""
    with Session(engine) as s:
        rows = s.exec(select(models.ForecastRun.status, func.count())
                      .group_by(models.ForecastRun.status)).all()
    return {status: n for status, n in rows}


def last_run_id(status):
    """id последнего прогона в заданном статусе, иначе None."""
    with Session(engine) as s:
        return s.exec(select(models.ForecastRun.id)
                      .where(models.ForecastRun.status == status)
                      .order_by(models.ForecastRun.id.desc()).limit(1)).first()


def last_matured_run_id():
    """id последнего завершённого прогона, у которого хотя бы часть недель уже имеет факт.
    Прогноз в будущее факта ещё не имеет, поэтому точность и разрезы считаем по нему, а не по
    просто последнему завершённому. None, если факта нет ни по одному прогону."""
    sql = text("""
        SELECT p.run_id
        FROM forecast_point p
        JOIN sales_history h ON h.series_id = p.series_id
             AND h.week_start_date = p.week_start_date AND h.n_days = 7
        JOIN forecast_run r ON r.id = p.run_id AND r.status = 'completed'
        GROUP BY p.run_id ORDER BY p.run_id DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).first()
    return row[0] if row else None


def accuracy_vs_actual(run_id):
    """Точность прогноза против пришедшего факта: WMAPE, смещение и покрытие интервала
    P10-P90 по тем неделям прогона, для которых уже есть полная неделя факта. None, если
    факта ещё нет (прогноз в будущее). Сравнивается точечный P50 с фактическими продажами."""
    sql = text("""
        SELECT p.p10, p.p50, p.p90, h.units
        FROM forecast_point p
        JOIN sales_history h ON h.series_id = p.series_id AND h.week_start_date = p.week_start_date
        WHERE p.run_id = :rid AND h.n_days = 7
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"rid": run_id})
    if df.empty:
        return None
    actual = df["units"].to_numpy(dtype=float)
    p50 = df["p50"].to_numpy(dtype=float)
    total = actual.sum()
    wmape = float(np.abs(p50 - actual).sum() / total) if total > 0 else None
    bias = float((p50 - actual).sum() / total) if total > 0 else None
    coverage = float(((df["p10"] <= df["units"]) & (df["units"] <= df["p90"])).mean())
    return {"n_points": int(len(df)), "wmape": wmape, "bias": bias, "coverage": coverage}


def _wmape_bias(g):
    """WMAPE, смещение и покрытие интервала по группе точек прогноз-факт. None при нулевом факте."""
    total = float(g["units"].sum())
    if total <= 0:
        return None
    p50 = g["p50"].to_numpy(dtype=float)
    act = g["units"].to_numpy(dtype=float)
    cov = float(((g["p10"] <= g["units"]) & (g["units"] <= g["p90"])).mean())
    return {"n": int(len(g)), "wmape": round(float(np.abs(p50 - act).sum() / total), 4),
            "bias": round(float((p50 - act).sum() / total), 4), "coverage": round(cov, 4)}


def accuracy_breakdowns(run_id):
    """Разрезы точности прогноз-факт для мониторинга деградации по прогону: по горизонту,
    по штату (плановый уровень), промо против базовых недель, сегментам движения и cold-start,
    плюс forecast value add против базы MA-4. None, если факта по прогону ещё нет.

    База MA-4, сегменты и cold-start считаются только по истории до origin (без утечки факта)."""
    with Session(engine) as s:
        run = s.get(models.ForecastRun, run_id)
        if run is None:
            return None
        origin = run.origin

    joined = text("""
        SELECT p.h, p.p10, p.p50, p.p90, h.units, p.series_id,
               (h.event_days > 0 OR h.snap_days > 0) AS promo, s.state_id
        FROM forecast_point p
        JOIN sales_history h ON h.series_id = p.series_id
             AND h.week_start_date = p.week_start_date AND h.n_days = 7
        JOIN series s ON s.id = p.series_id
        WHERE p.run_id = :rid
    """)
    hist_sql = text("""
        SELECT h.series_id, avg(h.units) AS mu, count(*) AS nweeks
        FROM sales_history h
        WHERE h.n_days = 7 AND h.week_start_date <= :origin
          AND h.series_id IN (SELECT DISTINCT series_id FROM forecast_point WHERE run_id = :rid)
        GROUP BY h.series_id
    """)
    ma_sql = text("""
        SELECT series_id, avg(units) AS ma4 FROM (
            SELECT h.series_id, h.units,
                   row_number() OVER (PARTITION BY h.series_id ORDER BY h.week_start_date DESC) AS rn
            FROM sales_history h
            WHERE h.n_days = 7 AND h.week_start_date <= :origin
              AND h.series_id IN (SELECT DISTINCT series_id FROM forecast_point WHERE run_id = :rid)
        ) t WHERE rn <= 4 GROUP BY series_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(joined, conn, params={"rid": run_id})
        if df.empty:
            return None
        hist = pd.read_sql_query(hist_sql, conn, params={"rid": run_id, "origin": origin})
        ma = pd.read_sql_query(ma_sql, conn, params={"rid": run_id, "origin": origin})

    # сегмент ряда по истории до origin: короткая история - cold-start, редкие продажи - прерывистый
    hist["segment"] = np.where(hist["nweeks"] < 26, "coldstart",
                               np.where(hist["mu"] < 1.0, "intermittent", "frequent"))
    df = df.merge(hist[["series_id", "segment"]], on="series_id", how="left")
    df = df.merge(ma, on="series_id", how="left")

    def by(col):
        return {str(k): v for k, v in ((k, _wmape_bias(g)) for k, g in df.groupby(col)) if v}

    promo = {"promo" if k else "base": v for k, v in
             ((k, _wmape_bias(g)) for k, g in df.groupby("promo")) if v}
    by_state = by("state_id")

    # forecast value add: WMAPE модели против плоской базы MA-4 (только история до origin)
    total = float(df["units"].sum())
    fva = None
    if total > 0:
        act = df["units"].to_numpy(dtype=float)
        model_wmape = float(np.abs(df["p50"].to_numpy(dtype=float) - act).sum() / total)
        ma4_wmape = float(np.abs(df["ma4"].fillna(0).to_numpy(dtype=float) - act).sum() / total)
        improvement = round((ma4_wmape - model_wmape) / ma4_wmape * 100, 1) if ma4_wmape > 0 else None
        fva = {"model_wmape": round(model_wmape, 4), "ma4_wmape": round(ma4_wmape, 4),
               "improvement_pct": improvement}

    planning_bias = max((abs(v["bias"]) for v in by_state.values()), default=None)
    return {
        "by_horizon": {str(int(k)): v for k, v in
                       ((k, _wmape_bias(g)) for k, g in df.groupby("h")) if v},
        "by_state": by_state,
        "promo": promo,
        "segments": by("segment"),
        "fva_ma4": fva,
        "planning_bias": round(planning_bias, 4) if planning_bias is not None else None,
    }


def data_freshness():
    """Свежесть факта: последняя полная неделя, глубина истории и полнота последней недели -
    доля недавно активных рядов, у которых уже есть факт за последнюю неделю. Низкая полнота -
    признак неполной или запоздавшей загрузки данных. None, если истории нет."""
    with engine.connect() as conn:
        latest = conn.execute(text(
            "SELECT max(week_start_date) FROM sales_history WHERE n_days = 7")).scalar()
        if latest is None:
            return None
        weeks = conn.execute(text(
            "SELECT count(DISTINCT week_start_date) FROM sales_history WHERE n_days = 7")).scalar()
        in_latest = conn.execute(text(
            "SELECT count(DISTINCT series_id) FROM sales_history "
            "WHERE n_days = 7 AND week_start_date = :w"), {"w": latest}).scalar()
        lo = (pd.Timestamp(latest) - pd.Timedelta(weeks=13)).date()
        active = conn.execute(text(
            "SELECT count(DISTINCT series_id) FROM sales_history "
            "WHERE n_days = 7 AND week_start_date > :lo"), {"lo": lo}).scalar()
    completeness = round(in_latest / active, 4) if active else None
    return {"latest_week": str(latest), "latest_week_ts": int(pd.Timestamp(latest).timestamp()),
            "history_weeks": int(weeks), "completeness": completeness}


def assortment_churn(window_weeks=13):
    """Дрейф ассортимента: новые и выбывшие ряды. Активный ряд - с продажами (units>0).
    Последнее окно window_weeks против предыдущего такого же. None, если истории нет."""
    with engine.connect() as conn:
        latest = conn.execute(text(
            "SELECT max(week_start_date) FROM sales_history WHERE n_days = 7")).scalar()
        if latest is None:
            return None
        hi = pd.Timestamp(latest)
        mid = (hi - pd.Timedelta(weeks=window_weeks)).date()
        lo = (hi - pd.Timedelta(weeks=2 * window_weeks)).date()
        q = text("SELECT DISTINCT series_id FROM sales_history WHERE n_days = 7 AND units > 0 "
                 "AND week_start_date > :a AND week_start_date <= :b")
        recent = {r[0] for r in conn.execute(q, {"a": mid, "b": str(latest)})}
        prior = {r[0] for r in conn.execute(q, {"a": lo, "b": mid})}
    return {"window_weeks": window_weeks, "new_series": len(recent - prior),
            "dead_series": len(prior - recent), "recent_active": len(recent)}


def revision_volatility(max_runs=12):
    """Стабильность прогноза: насколько расходится точечный P50 на одну и ту же неделю ряда
    между прогонами с разным origin. Среднее по (ряд, неделя) с >=2 origin: разброс P50 к
    среднему (коэффициент вариации). None, если пересечений origin нет."""
    sql = text("""
        SELECT p.series_id, p.week_start_date, p.p50, r.origin
        FROM forecast_point p
        JOIN forecast_run r ON r.id = p.run_id AND r.status IN ('completed', 'partial')
        WHERE p.run_id IN (SELECT id FROM forecast_run
                           WHERE status IN ('completed', 'partial') ORDER BY id DESC LIMIT :n)
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"n": max_runs})
    if df.empty:
        return None

    def cov(g):
        if g["origin"].nunique() < 2:
            return np.nan
        m = g["p50"].mean()
        return g["p50"].std(ddof=0) / m if m > 0 else np.nan

    vals = df.groupby(["series_id", "week_start_date"]).apply(cov).dropna()
    return round(float(vals.mean()), 4) if not vals.empty else None


def catalog(run_id):
    # ширина интервала суммируется за горизонт (как и P50), чтобы колонки были в одном масштабе
    sql = """SELECT p.series_id, s.item_id, s.dept_id, s.store_id, s.state_id,
                    sum(p.p50) AS sum_p50, sum(p.p90 - p.p10) AS interval_width
             FROM forecast_point p JOIN series s ON s.id = p.series_id
             WHERE p.run_id = %(rid)s
             GROUP BY p.series_id, s.item_id, s.dept_id, s.store_id, s.state_id
             ORDER BY sum_p50 DESC"""
    return pd.read_sql_query(sql, engine, params={"rid": run_id}).round(2)


def series_forecast(run_id, series_id):
    """Точки прогноза ряда и его история вокруг окна прогноза (полные недели) для графика:
    контекст до начала прогноза плюс факт на прогнозных неделях, если он уже есть."""
    pts = pd.read_sql_query(
        "SELECT week_start_date, h, p10, p50, p90 FROM forecast_point "
        "WHERE run_id = %(rid)s AND series_id = %(sid)s ORDER BY h",
        engine, params={"rid": run_id, "sid": series_id})
    hi = pts["week_start_date"].max() if len(pts) else None
    hist = pd.read_sql_query(
        "SELECT week_start_date, units FROM sales_history "
        "WHERE series_id = %(sid)s AND n_days = 7 AND (%(hi)s IS NULL OR week_start_date <= %(hi)s) "
        "ORDER BY week_start_date DESC LIMIT 40",
        engine, params={"sid": series_id, "hi": hi}).iloc[::-1]
    return pts, hist


def agg_forecast(run_id, state=None, store=None, dept=None):
    """Суммарный прогноз (p10/p50/p90 по неделям) по срезу иерархии штат/магазин/отдел плюс
    суммарная история того же среза (для графика и факта на прогнозных неделях)."""
    conds, params = ["p.run_id = %(rid)s"], {"rid": run_id}
    for col, val, key in [("s.state_id", state, "state"),
                          ("s.store_id", store, "store"),
                          ("s.dept_id", dept, "dept")]:
        if val:
            conds.append(f"{col} = %({key})s")
            params[key] = val
    where = " AND ".join(conds)
    pts = pd.read_sql_query(
        f"""SELECT p.week_start_date, sum(p.p10) AS p10, sum(p.p50) AS p50, sum(p.p90) AS p90
            FROM forecast_point p JOIN series s ON s.id = p.series_id
            WHERE {where} GROUP BY p.week_start_date ORDER BY p.week_start_date""",
        engine, params=params)
    # история агрегата нужна графику только как контекст и факт на окне теста:
    # ограничиваем её последним годом, чтобы не суммировать всю таблицу
    lo = (pd.Timestamp(pts["week_start_date"].min()) - pd.Timedelta(weeks=60)).date() \
        if len(pts) else None
    hist = pd.read_sql_query(
        f"""SELECT h.week_start_date, sum(h.units) AS units
            FROM sales_history h
            WHERE h.n_days = 7 AND h.week_start_date > %(lo)s AND h.series_id IN (
                SELECT p.series_id FROM forecast_point p JOIN series s ON s.id = p.series_id
                WHERE {where})
            GROUP BY h.week_start_date ORDER BY h.week_start_date""",
        engine, params={**params, "lo": lo}) if lo else pd.DataFrame(columns=["week_start_date", "units"])
    return pts, hist


DRIFT_WEEKS = 13  # окно оценки дрейфа; единственное место, где задаётся


def drift_windows(run_id, weeks=DRIFT_WEEKS):
    """Два окна признаков для оценки дрейфа: текущее (N недель до origin) и такое же
    год назад. Сравнение год-к-году снимает сезонность: сдвиг сезона не считается дрейфом."""
    with Session(engine) as s:
        run = s.get(models.ForecastRun, run_id)
        if run is None:
            return pd.DataFrame(), pd.DataFrame()
        origin = run.origin

    def window(hi):
        lo = (pd.Timestamp(hi) - pd.Timedelta(weeks=weeks)).date()
        sql = text("""
            SELECT h.units, h.sell_price
            FROM sales_history h
            WHERE h.n_days = 7 AND h.week_start_date <= :hi AND h.week_start_date > :lo
              AND h.series_id IN (SELECT DISTINCT series_id FROM forecast_point WHERE run_id = :rid)
        """)
        with engine.connect() as conn:
            return pd.read_sql_query(sql, conn, params={
                "hi": pd.Timestamp(hi).date(), "lo": lo, "rid": run_id})

    cur = window(origin)
    ref = window(pd.Timestamp(origin) - pd.Timedelta(weeks=52))
    return cur, ref


def fail_stale_chunks(minutes=15):
    """Пачки, зависшие в processing дольше таймаута (жёсткая смерть воркера), помечаются
    failed, затронутые прогоны финализируются. Вызывается фоновой задачей api."""
    with Session(engine) as s:
        rows = s.execute(text(
            "UPDATE forecast_chunk SET status='failed', finished_at=now(), "
            "error='таймаут обработки: воркер не завершил пачку' "
            "WHERE status='processing' AND started_at < now() - make_interval(mins => :m) "
            "RETURNING run_id"), {"m": minutes}).fetchall()
        s.commit()
    for (rid,) in set(rows):
        finalize_run(rid)
    return len(rows)


def export_rows(run_id):
    return pd.read_sql_query(
        "SELECT p.series_id, s.item_id, s.store_id, p.week_start_date, p.h, p.p10, p.p50, p.p90 "
        "FROM forecast_point p JOIN series s ON s.id = p.series_id "
        "WHERE p.run_id = %(rid)s ORDER BY p.series_id, p.h",
        engine, params={"rid": run_id})
