"""Подключение к PostgreSQL и сессии. Схема накатывается миграциями Alembic."""
from pathlib import Path

from sqlmodel import create_engine

from . import models  # noqa: F401  регистрирует таблицы в metadata
from .config import settings

ROOT = Path(__file__).resolve().parents[2]  # каталог сервиса с alembic.ini
engine = create_engine(settings.db_url, pool_pre_ping=True)


def create_db():
    """Накатывает схему до последней миграции Alembic (idempotent)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(cfg, "head")
