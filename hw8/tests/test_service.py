"""Тесты критического пути: создание прогона -> обработка пачек -> финализация.

Воркер имитируется напрямую (read_history -> forecast_series -> complete_chunk), публикация в
очередь перехвачена фикстурой. Нужна поднятая БД (DB_URL) и собранный артефакт.
"""
import json

import pandas as pd

from forecast_service import crud, worker
from forecast_service.db import engine
from forecast_service.forecast import forecast_series


def _work(msgs):
    for m in msgs:
        crud.start_chunk(m["chunk_id"], "test")
        hist = crud.read_history(m["series_ids"], m["origin"])
        out = forecast_series(hist, m["origin"], m["horizon"])
        crud.complete_chunk(m["run_id"], m["chunk_id"], out, "test")


def _count(rid, extra=""):
    sql = f"SELECT count(*) c FROM forecast_point WHERE run_id=%(r)s {extra}"
    return int(pd.read_sql_query(sql, engine, params={"r": rid}).c[0])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"]


def test_empty_slice_returns_422(client, published):
    r = client.post("/api/runs", json={"store_id": "NOPE_9", "cat_id": "FOODS"})
    assert r.status_code == 422


def test_unknown_run_404(client):
    assert client.get("/api/runs/999999").status_code == 404


def test_create_process_finalize(client, published):
    r = client.post("/api/runs", json={"store_id": "TX_2", "cat_id": "FOODS"})
    assert r.status_code == 202
    rid = r.json()["run_id"]
    assert len(published) > 0
    _work(published)
    st = client.get(f"/api/runs/{rid}").json()
    assert st["status"] == "completed"
    assert st["n_chunks_done"] == st["n_chunks"]
    assert len(client.get(f"/api/runs/{rid}/catalog").json()) > 0


def test_idempotent_upsert(client, published):
    r = client.post("/api/runs", json={"store_id": "WI_2", "cat_id": "FOODS"})
    rid = r.json()["run_id"]
    m = published[0]
    crud.start_chunk(m["chunk_id"], "t")
    out = forecast_series(crud.read_history(m["series_ids"], m["origin"]), m["origin"], m["horizon"])
    crud.complete_chunk(m["run_id"], m["chunk_id"], out, "t")
    n1 = _count(rid)
    crud.complete_chunk(m["run_id"], m["chunk_id"], out, "t")  # повтор не должен задваивать
    assert _count(rid) == n1


def test_monotone_quantiles(client, published):
    r = client.post("/api/runs", json={"store_id": "TX_3", "cat_id": "FOODS"})
    rid = r.json()["run_id"]
    _work(published)
    assert _count(rid, "AND NOT (p10 <= p50 AND p50 <= p90)") == 0


def test_worker_handle_success(client, published):
    r = client.post("/api/runs", json={"store_id": "WI_3", "cat_id": "FOODS"})
    rid = r.json()["run_id"]
    for m in published:
        worker.handle(json.dumps(m))
    assert client.get(f"/api/runs/{rid}").json()["status"] == "completed"


def test_worker_handle_failure_marks_partial(client, published, monkeypatch):
    r = client.post("/api/runs", json={"store_id": "CA_4", "cat_id": "FOODS"})
    rid = r.json()["run_id"]

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "forecast_series", boom)
    for m in published:
        try:
            worker.handle(json.dumps(m))
        except RuntimeError:
            pass
    assert client.get(f"/api/runs/{rid}").json()["status"] == "partial"
