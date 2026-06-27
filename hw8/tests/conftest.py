"""Фикстуры тестов: схема БД, TestClient, перехват публикации в очередь."""
import pytest
from fastapi.testclient import TestClient

from forecast_service import mq
from forecast_service.db import create_db


@pytest.fixture(scope="session", autouse=True)
def schema():
    create_db()


@pytest.fixture
def published(monkeypatch):
    msgs = []
    monkeypatch.setattr(mq, "publish", lambda m: msgs.extend(m))
    return msgs


@pytest.fixture
def client():
    from forecast_service.api import app
    with TestClient(app) as c:
        yield c
