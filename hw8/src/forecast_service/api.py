"""REST и серверный UI: создание прогона, статус, прогноз, метрики, выгрузка.

Постановка прогона асинхронная: POST кладёт пачки в очередь и сразу отдаёт run_id. Статус и
готовность считаются по пачкам в БД (ленивая финализация). При старте crud.reconcile дочиняет
прогоны, у которых все пачки уже терминальны. Доступ к БД целиком в crud, здесь только HTTP.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import crud, mq
from .config import settings
from .db import create_db
from .forecast import load_artifact
from .ml.features import HMAX

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


@asynccontextmanager
async def lifespan(app):
    create_db()
    try:
        load_artifact()
    except Exception as e:
        print("артефакт не загружен:", e, flush=True)
    crud.reconcile()
    yield


app = FastAPI(title="Прогноз спроса", docs_url="/api/docs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


class RunRequest(BaseModel):
    store_id: str | None = None
    cat_id: str = "FOODS"
    horizon_weeks: int = Field(8, ge=1, le=HMAX)
    origin: str | None = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html",
                                      {"stores": crud.stores(), "runs": crud.recent_runs()})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: int):
    return templates.TemplateResponse(request, "run.html", {"run_id": run_id})


@app.get("/health")
def health():
    try:
        art = load_artifact()
    except Exception as e:
        return JSONResponse({"status": "degraded", "model_loaded": False, "error": str(e)}, status_code=503)
    return {"status": "ok", "model_version": art["model_version"], "model_loaded": True}


@app.post("/api/runs", status_code=202)
def create_run(req: RunRequest):
    art = load_artifact()
    try:
        origin = date.fromisoformat(req.origin) if req.origin else art["origin"].date()
        run_id, msgs = crud.create_run(req.store_id or None, req.cat_id,
                                       req.horizon_weeks, origin, art["model_version"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        mq.publish(msgs)
    except Exception as e:
        crud.fail_run(run_id)  # брокер недоступен: закрываем прогон, чтобы он не завис в очереди
        raise HTTPException(503, f"очередь недоступна: {e}")
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


@app.get("/api/runs/{run_id}/forecast")
def forecast(run_id: int, series_id: str = Query(...)):
    pts, hist = crud.series_forecast(run_id, series_id)
    if pts.empty:
        raise HTTPException(404, "нет прогноза по ряду")

    def fmt(d):
        return pd.to_datetime(d["week_start_date"]).dt.strftime("%Y-%m-%d")

    pts["week_start_date"] = fmt(pts)
    hist["week_start_date"] = fmt(hist)
    return {"series_id": series_id,
            "history": hist.rename(columns={"week_start_date": "week"}).to_dict("records"),
            "points": pts.rename(columns={"week_start_date": "week"}).round(2).to_dict("records")}


@app.get("/api/runs/{run_id}/metrics")
def metrics(run_id: int):
    art = load_artifact()
    csv = settings.metrics_dir / "cv_summary_foods.csv"
    rows, wrmsse = [], art.get("wrmsse")
    if csv.exists():
        df = pd.read_csv(csv)
        rows = df.to_dict("records")
        wrmsse = round(float(df["value"].mean()), 4)  # общий WRMSSE = среднее по 12 уровням
    return {"model_version": art["model_version"], "wrmsse": wrmsse, "rows": rows}


@app.get("/api/runs/{run_id}/export.csv")
def export(run_id: int):
    buf = io.StringIO()
    crud.export_rows(run_id).to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=forecast_run_{run_id}.csv"})
