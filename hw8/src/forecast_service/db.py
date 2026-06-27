"""Подключение к PostgreSQL и сессии."""
from sqlmodel import SQLModel, create_engine

from . import models  # noqa: F401  регистрирует таблицы в metadata
from .config import settings

engine = create_engine(settings.db_url, pool_pre_ping=True)


def create_db():
    SQLModel.metadata.create_all(engine)
