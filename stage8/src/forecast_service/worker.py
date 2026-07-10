"""Воркер: берёт пачку из очереди, считает прогноз по своим рядам, пишет результат.

Модель грузится один раз при старте, ничего не обучается. При сбое пачка помечается FAILED и
не возвращается в очередь (детерминированный прогноз - повтор не лечит).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time

from . import crud, log, mq, prom
from .forecast import forecast_series, load_artifact

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
_log = logging.getLogger("forecast.worker")


def handle(body):
    msg = json.loads(body)
    start = time.perf_counter()
    try:
        crud.start_chunk(msg["chunk_id"], WORKER_ID)
        hist = crud.read_history(msg["series_ids"], msg["origin"])
        item_agg, dept_agg = crud.read_item_dept_weekly(msg["origin"])
        out = forecast_series(hist, msg["origin"], msg["horizon"], item_agg, dept_agg)
        crud.complete_chunk(msg["run_id"], msg["chunk_id"], out, WORKER_ID)
        crud.finalize_run(msg["run_id"])
        prom.CHUNK_DURATION.observe(time.perf_counter() - start)
        prom.CHUNKS_PROCESSED.labels(result="completed").inc()
        prom.SERIES_FORECAST.inc(out["series_id"].nunique())
        prom.FALLBACK.inc(out.attrs.get("n_fallback", 0))
        _log.info("chunk %s готов: %s рядов", msg["chunk_id"], out["series_id"].nunique())
    except Exception as e:
        prom.CHUNKS_PROCESSED.labels(result="failed").inc()
        crud.fail_chunk(msg["chunk_id"], e, WORKER_ID)
        crud.finalize_run(msg["run_id"])  # если это была последняя пачка, сразу закрыть прогон в PARTIAL
        _log.exception("chunk %s ошибка", msg["chunk_id"])
        raise


def main():
    log.setup()
    load_artifact()
    prom.serve()  # сервер метрик воркера для скрейпа Prometheus
    _log.info("воркер %s готов", WORKER_ID)
    mq.consume(handle)


if __name__ == "__main__":
    main()
