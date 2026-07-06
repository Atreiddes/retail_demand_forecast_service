"""Воркер: берёт пачку из очереди, считает прогноз по своим рядам, пишет результат.

Модель грузится один раз при старте, ничего не обучается. При сбое пачка помечается FAILED и
не возвращается в очередь (детерминированный прогноз - повтор не лечит).
"""
from __future__ import annotations

import json
import os
import socket
import traceback

from . import crud, mq
from .forecast import forecast_series, load_artifact

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


def handle(body):
    msg = json.loads(body)
    try:
        crud.start_chunk(msg["chunk_id"], WORKER_ID)
        hist = crud.read_history(msg["series_ids"], msg["origin"])
        item_agg, dept_agg = crud.read_item_dept_weekly(msg["origin"])
        out = forecast_series(hist, msg["origin"], msg["horizon"], item_agg, dept_agg)
        crud.complete_chunk(msg["run_id"], msg["chunk_id"], out, WORKER_ID)
        crud.finalize_run(msg["run_id"])
        print(f"chunk {msg['chunk_id']} готов: {out['series_id'].nunique()} рядов", flush=True)
    except Exception as e:
        crud.fail_chunk(msg["chunk_id"], e, WORKER_ID)
        crud.finalize_run(msg["run_id"])  # если это была последняя пачка, сразу закрыть прогон в PARTIAL
        print(f"chunk {msg['chunk_id']} ошибка: {e}\n{traceback.format_exc()}", flush=True)
        raise


def main():
    load_artifact()
    print(f"воркер {WORKER_ID} готов", flush=True)
    mq.consume(handle)


if __name__ == "__main__":
    main()
