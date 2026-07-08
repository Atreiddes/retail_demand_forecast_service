"""Метрики Prometheus: определения и наполнение из БД и файлов метрик.

Эндпоинт /metrics отдаёт api, воркер поднимает свой сервер метрик (serve). Prometheus
скрейпит оба, Grafana строит панели. Набор общий: каждый процесс регистрирует все метрики,
но наполняет только свою часть (лишние остаются нулевыми).
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# HTTP api: наполняет middleware в api.py. path - шаблон маршрута, а не сырой путь,
# иначе run_id в адресе разносит кардинальность
HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP-запросы к api", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "Время ответа api", ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10))

# Прогоны: постановка (api) и обработка пачек (воркер)
RUNS_CREATED = Counter("forecast_runs_created_total", "Поставлено прогонов", ["scope"])
CHUNKS_PROCESSED = Counter(
    "forecast_chunks_processed_total", "Обработано пачек воркером", ["result"])
CHUNK_DURATION = Histogram(
    "forecast_chunk_duration_seconds", "Время обработки пачки воркером",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300))
SERIES_FORECAST = Counter("forecast_series_total", "Рядов спрогнозировано")

# Состояние и качество: наполняет фоновый сбор в api
RUNS_STATUS = Gauge("forecast_runs_status", "Прогонов в каждом статусе", ["status"])
ACTIVE_RUNS = Gauge("forecast_active_runs", "Прогонов в незавершённом статусе")
QUALITY = Gauge("forecast_quality", "Метрики качества модели из metrics_summary.json", ["metric"])
DRIFT_PSI = Gauge(
    "forecast_data_drift_psi", "PSI дрейфа признаков (последний завершённый прогон)", ["feature"])
MODEL_INFO = Gauge("forecast_model_info", "Версия артефакта модели, значение 1", ["version"])

_ACTIVE = {"new", "queued", "processing"}


def set_runs_status(counts: dict) -> None:
    RUNS_STATUS.clear()
    active = 0
    for status, n in counts.items():
        RUNS_STATUS.labels(status=status).set(n)
        if status in _ACTIVE:
            active += n
    ACTIVE_RUNS.set(active)


def set_quality(summary: dict) -> None:
    for name, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            QUALITY.labels(metric=name).set(value)


def set_drift(features: list) -> None:
    DRIFT_PSI.clear()
    for f in features:
        if f.get("psi") is not None:
            DRIFT_PSI.labels(feature=f["feature"]).set(f["psi"])


def set_model_version(version: str) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=version).set(1)


def serve(port: int = 9100) -> None:
    """Сервер метрик воркера в фоновом потоке: воркер блокируется на очереди, метрики отдаёт рядом."""
    start_http_server(port)
