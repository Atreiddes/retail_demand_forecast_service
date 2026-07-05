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
    monkeypatch.setattr(mq, "publish", lambda m, priority=1: msgs.extend(m))
    return msgs


@pytest.fixture
def client():
    from forecast_service.api import app
    from forecast_service.config import settings
    # запуск прогона закрыт ключом: тестовый клиент ходит с ключом по умолчанию
    with TestClient(app, headers={"X-API-Key": settings.api_key}) as c:
        yield c
