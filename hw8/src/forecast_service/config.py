"""Настройки сервиса из окружения и .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent  # hw8/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/forecast"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    data_path: Path = ROOT / "data" / "foods_weekly.parquet"
    calendar_path: Path = ROOT / "data" / "calendar.parquet"
    artifact_dir: Path = ROOT / "artifacts"
    metrics_dir: Path = ROOT / "metrics"

    chunk_size: int = 300


settings = Settings()
