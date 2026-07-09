"""REST и серверный UI: создание прогона, статус, прогноз, метрики, выгрузка.

Постановка прогона асинхронная: POST кладёт пачки в очередь и сразу отдаёт run_id. Статус и
готовность считаются по пачкам в БД (ленивая финализация). При старте crud.reconcile дочиняет
прогоны, у которых все пачки уже терминальны. Доступ к БД целиком в crud, здесь только HTTP.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from contextlib import asynccontextmanager
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import crud, models, monitoring, mq, prom
from .config import settings
from .db import create_db
from .forecast import load_artifact
from .ml.features import HMAX

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


async def _reap_stale():
    """Фоновая задача: пачки, зависшие в processing после жёсткой смерти воркера,
    по таймауту помечаются failed, прогоны финализируются."""
    while True:
        await asyncio.sleep(60)
        try:
            n = await asyncio.to_thread(crud.fail_stale_chunks)
            if n:
                print(f"закрыто зависших пачек: {n}", flush=True)
        except Exception as e:
            print("проверка зависших пачек:", e, flush=True)


async def _refresh_metrics():
    """Фоновый сбор гейджей: статусы прогонов, качество модели, дрейф последнего
    завершённого прогона. Быстрые метрики (запросы, пачки) считаются в месте события."""
    while True:
        try:
            await asyncio.to_thread(_collect_metrics)
        except Exception as e:
            print("сбор метрик:", e, flush=True)
        await asyncio.sleep(15)


def _monitoring_report():
    """Отчёт гейта деградации: точность прогноз-факт и разрезы по последнему прогону с
    вызревшим фактом, дрейф по последнему завершённому. Общий для сбора метрик и эндпоинта."""
    matured = crud.last_matured_run_id()
    accuracy = crud.accuracy_vs_actual(matured) if matured else None
    breakdowns = crud.accuracy_breakdowns(matured) if matured else None
    completed = crud.last_run_id(models.COMPLETED)
    res = _drift_cached(completed) if completed else None
    drift = res["features"] if res else None
    health = {"freshness": crud.data_freshness(), "churn": crud.assortment_churn(),
              "revision_volatility": crud.revision_volatility()}
    return monitoring.gate(accuracy, drift, breakdowns, health)


def _collect_metrics():
    prom.set_runs_status(crud.run_status_counts())
    summ = settings.metrics_dir / "metrics_summary.json"
    if summ.exists():
        prom.set_quality(json.loads(summ.read_text(encoding="utf-8")))
    report = _monitoring_report()
    if report["drift"]:
        prom.set_drift(report["drift"])
    prom.set_accuracy(report["accuracy"])
    prom.set_breakdowns(report["breakdowns"])
    health = report["health"]
    prom.set_health(health["freshness"], health["churn"], health["revision_volatility"])
    prom.set_degraded(report)


@asynccontextmanager
async def lifespan(app):
    create_db()
    try:
        art = load_artifact()
        prom.set_model_version(art["model_version"])
    except Exception as e:
        print("артефакт не загружен:", e, flush=True)
    crud.reconcile()
    reaper = asyncio.create_task(_reap_stale())
    collector = asyncio.create_task(_refresh_metrics())
    yield
    reaper.cancel()
    collector.cancel()


app = FastAPI(title="Прогноз спроса", docs_url="/api/docs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.middleware("http")
async def _prometheus_mw(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", None) or "other"
    if path != "/metrics":  # сам эндпоинт метрик в статистику не берём
        prom.HTTP_REQUESTS.labels(
            method=request.method, path=path, status=str(response.status_code)).inc()
        prom.HTTP_LATENCY.labels(
            method=request.method, path=path).observe(time.perf_counter() - start)
    return response


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class RunRequest(BaseModel):
    store_id: str | None = None
    cat_id: str = "FOODS"
    horizon_weeks: int = Field(8, ge=1, le=HMAX)
    origin: str | None = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html",
                                      {"stores": crud.stores(), "runs": crud.recent_runs(100)})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: int):
    return templates.TemplateResponse(request, "run.html", {"run_id": run_id})


@app.get("/health")
def health():
    try:
        art = load_artifact()
    except Exception as e:
        return JSONResponse({"status": "degraded", "model_loaded": False, "error": str(e)}, status_code=503)
    # origin и hmax отдаём явно: интерфейс подставляет дату и окно теста из этих полей,
    # а не разбирает строку версии
    return {"status": "ok", "model_version": art["model_version"], "model_loaded": True,
            "origin": str(art["origin"].date()), "hmax": HMAX}


@app.post("/api/runs", status_code=202)
def create_run(req: RunRequest, x_api_key: str | None = Header(None, alias="X-API-Key")):
    if x_api_key != settings.api_key:
        raise HTTPException(401, "нет доступа: неверный ключ API")
    art = load_artifact()
    try:
        origin = date.fromisoformat(req.origin) if req.origin else art["origin"].date()
        run_id, msgs = crud.create_run(req.store_id or None, req.cat_id,
                                       req.horizon_weeks, origin, art["model_version"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        # мелкий прогон (один магазин) идёт с высоким приоритетом и не ждёт за крупным
        mq.publish(msgs, priority=5 if req.store_id else 1)
    except Exception as e:
        crud.fail_run(run_id)  # брокер недоступен: закрываем прогон, чтобы он не завис в очереди
        raise HTTPException(503, f"очередь недоступна: {e}")
    prom.RUNS_CREATED.labels(scope="store" if req.store_id else "all").inc()
    return {"run_id": run_id, "n_chunks": len(msgs)}


@app.get("/api/runs")
def list_runs():
    return crud.recent_runs(50)


@app.get("/api/runs/{run_id}")
def run_status(run_id: int):
    res = crud.run_progress(run_id)
    if res is None:
        raise HTTPException(404, "прогон не найден")
    run, done = res
    return {"status": run.status, "n_series": run.n_series, "n_chunks": run.n_chunks,
            "n_chunks_done": done, "model_version": run.model_version}


@app.get("/api/runs/{run_id}/catalog")
def catalog(run_id: int):
    return crud.catalog(run_id).to_dict("records")


def _chart_payload(pts, hist):
    """Единый формат ответа для графика: колонка week как строка, значения округлены."""
    def fmt(d):
        d = d.copy()
        d["week_start_date"] = pd.to_datetime(d["week_start_date"]).dt.strftime("%Y-%m-%d")
        return d.rename(columns={"week_start_date": "week"}).round(2).to_dict("records")
    return {"points": fmt(pts), "history": fmt(hist)}


@app.get("/api/runs/{run_id}/forecast")
def forecast(run_id: int, series_id: str = Query(...)):
    pts, hist = crud.series_forecast(run_id, series_id)
    if pts.empty:
        raise HTTPException(404, "нет прогноза по ряду")
    return {"series_id": series_id, **_chart_payload(pts, hist)}


@app.get("/api/runs/{run_id}/forecast_agg")
def forecast_agg(run_id: int, state: str | None = None,
                 store: str | None = None, dept: str | None = None):
    pts, hist = crud.agg_forecast(run_id, state, store, dept)
    if pts.empty:
        raise HTTPException(404, "нет прогноза по срезу")
    return _chart_payload(pts, hist)


@app.get("/api/runs/{run_id}/metrics")
def metrics(run_id: int):
    art = load_artifact()
    mdir = settings.metrics_dir
    rows = pd.read_csv(mdir / "cv_summary_foods.csv").to_dict("records") \
        if (mdir / "cv_summary_foods.csv").exists() else []
    summ = mdir / "metrics_summary.json"
    extra = json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else {}
    return {"model_version": art["model_version"], "rows": rows, **extra}


def _psi(ref_prop, cur_prop):
    """Population Stability Index: насколько текущее распределение отклонилось от эталона."""
    eps = 1e-6
    r = np.asarray(ref_prop, dtype=float) + eps
    c = np.asarray(cur_prop, dtype=float) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def _ks(ref, cur):
    """Двухвыборочная статистика Колмогорова-Смирнова: макс. расстояние между ЭФР выборок."""
    ref, cur = np.sort(ref), np.sort(cur)
    grid = np.concatenate([ref, cur])
    cdf_r = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_c = np.searchsorted(cur, grid, side="right") / cur.size
    return float(np.max(np.abs(cdf_r - cdf_c)))


@lru_cache(maxsize=256)
def _drift_cached(run_id: int):
    """PSI признаков: текущее окно против того же окна год назад по тем же рядам.
    Сравнение год-к-году снимает сезонность. Для завершённого прогона результат
    неизменен, поэтому кэшируется."""
    cur, ref = crud.drift_windows(run_id, weeks=crud.DRIFT_WEEKS)
    if cur.empty or ref.empty:
        return None
    out = []
    for feat in cur.columns:
        r = ref[feat].dropna().to_numpy(dtype=float)
        c = cur[feat].dropna().to_numpy(dtype=float)
        if r.size == 0 or c.size == 0:
            out.append({"feature": feat, "psi": None, "status": "no_data"})
            continue
        edges = np.unique(np.quantile(r, np.linspace(0, 1, 11)))
        if edges.size < 2:
            out.append({"feature": feat, "psi": None, "status": "no_data"})
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        ref_prop = np.histogram(r, bins=edges)[0] / r.size
        cur_prop = np.histogram(c, bins=edges)[0] / c.size
        p = _psi(ref_prop, cur_prop)
        status = "narrow" if p < 0.1 else ("mid" if p < 0.25 else "wide")
        out.append({"feature": feat, "psi": round(p, 3), "ks": round(_ks(r, c), 3), "status": status})
    return {"features": out, "window_weeks": crud.DRIFT_WEEKS}


@app.get("/api/runs/{run_id}/drift")
def drift(run_id: int):
    res = _drift_cached(run_id)
    if res is None:
        raise HTTPException(404, "нет данных для оценки дрейфа")
    return res


@app.get("/api/monitoring")
def monitoring_report():
    """Гейт деградации: точность прогноз-факт, дрейф и сводный флаг ok с предупреждениями.
    Этот же отчёт опрашивает DAG monitor_and_retrain для запуска переобучения."""
    return _monitoring_report()


@app.post("/api/alerts")
async def receive_alerts(request: Request):
    """Приёмник вебхука Alertmanager: логирует сработавшие алерты. Точка доставки уведомлений;
    в рабочей системе отсюда рассылают в почту или мессенджер."""
    payload = await request.json()
    alerts = payload.get("alerts", [])
    for a in alerts:
        name = a.get("labels", {}).get("alertname", "?")
        summary = a.get("annotations", {}).get("summary", "")
        print(f"[alert] {a.get('status', '?')} {name}: {summary}", flush=True)
    return {"received": len(alerts)}


@app.get("/api/runs/{run_id}/export.csv")
def export(run_id: int):
    buf = io.StringIO()
    crud.export_rows(run_id).to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=forecast_run_{run_id}.csv"})
