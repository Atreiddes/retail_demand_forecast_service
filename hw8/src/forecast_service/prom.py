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

# Прогноз против факта (по мере вызревания факта) и сводный флаг деградации
ACCURACY_WMAPE = Gauge("forecast_accuracy_wmape", "WMAPE прогноза против факта")
ACCURACY_BIAS = Gauge("forecast_accuracy_bias", "Смещение прогноза как доля объёма факта")
INTERVAL_COVERAGE = Gauge("forecast_interval_coverage", "Доля факта в интервале P10-P90")
DEGRADED = Gauge("forecast_degraded", "Модель деградировала по гейту (1) или нет (0)")

# Разрезы точности: по горизонту, по штату (плановый уровень), по сегментам (промо/база,
# движение), forecast value add против базы MA-4 и плановое смещение для гейта
WMAPE_BY_HORIZON = Gauge("forecast_wmape_by_horizon", "WMAPE по горизонту прогноза", ["h"])
WMAPE_SEGMENT = Gauge("forecast_wmape_segment", "WMAPE по сегменту", ["segment"])
BIAS_SEGMENT = Gauge("forecast_bias_segment", "Смещение по сегменту", ["segment"])
BIAS_BY_STATE = Gauge("forecast_bias_by_state", "Смещение по штату", ["state"])
FVA_MA4 = Gauge("forecast_fva_ma4_pct", "Forecast value add против базы MA-4, %")
PLANNING_BIAS = Gauge("forecast_planning_bias", "Максимум модуля смещения на плановом уровне (штат)")

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


def set_accuracy(accuracy: dict | None) -> None:
    # Безлейбловые гейджи выставляем всегда: при отсутствии факта - NaN, а не оставляем
    # дефолтный 0, иначе алерты по сырым гейджам зажигались бы на пустом состоянии
    acc = accuracy or {}
    ACCURACY_WMAPE.set(acc["wmape"] if acc.get("wmape") is not None else float("nan"))
    ACCURACY_BIAS.set(acc["bias"] if acc.get("bias") is not None else float("nan"))
    INTERVAL_COVERAGE.set(acc["coverage"] if acc.get("coverage") is not None else float("nan"))


def set_degraded(report: dict) -> None:
    DEGRADED.set(0 if report["ok"] else 1)


def set_breakdowns(bd: dict | None) -> None:
    # Лейбловые гейджи чистим каждый цикл, чтобы не копить устаревшие серии; безлейбловые
    # выставляем всегда (NaN при отсутствии данных, а не дефолтный 0)
    WMAPE_BY_HORIZON.clear()
    WMAPE_SEGMENT.clear()
    BIAS_SEGMENT.clear()
    BIAS_BY_STATE.clear()
    if not bd:
        FVA_MA4.set(float("nan"))
        PLANNING_BIAS.set(float("nan"))
        return
    for h, v in bd["by_horizon"].items():
        WMAPE_BY_HORIZON.labels(h=h).set(v["wmape"])
    for group in (bd["promo"], bd["segments"]):
        for seg, v in group.items():
            WMAPE_SEGMENT.labels(segment=seg).set(v["wmape"])
            BIAS_SEGMENT.labels(segment=seg).set(v["bias"])
    for state, v in bd["by_state"].items():
        BIAS_BY_STATE.labels(state=state).set(v["bias"])
    fva = bd.get("fva_ma4")
    FVA_MA4.set(fva["improvement_pct"] if fva and fva.get("improvement_pct") is not None else float("nan"))
    PLANNING_BIAS.set(bd["planning_bias"] if bd.get("planning_bias") is not None else float("nan"))


def serve(port: int = 9100) -> None:
    """Сервер метрик воркера в фоновом потоке: воркер блокируется на очереди, метрики отдаёт рядом."""
    start_http_server(port)
