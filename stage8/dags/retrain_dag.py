"""Airflow DAG периодического переобучения артефакта модели.

Приём новых недель -> проверка входа (pandera) -> переобучение -> бэктест -> quality gate ->
деплой. Если бэктест просел выше порога, новый артефакт не выкатывается (остаётся прежний).

Тяжёлые шаги (переобучение, бэктест) идут в отдельном контейнере через DockerOperator, а не
в окружении airflow: обучающий образ несёт lightgbm и пайплайн stage5, изолирован от зависимостей
airflow. Образ и хост-путь проекта задаются переменными TRAIN_IMAGE и HOST_PROJECT_DIR.
Это ops-компонент, в рантайм сервиса не входит.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pendulum
from docker.types import Mount

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.providers.docker.operators.docker import DockerOperator

PROJECT = Path(__file__).resolve().parent.parent            # каталог проекта внутри airflow
HOST_PROJECT = os.getenv("HOST_PROJECT_DIR", str(PROJECT))  # тот же каталог на хосте для sibling-контейнеров
TRAIN_IMAGE = os.getenv("TRAIN_IMAGE", "stage8-api")        # образ с обучающим пайплайном
NETWORK = os.getenv("COMPOSE_NETWORK", "stage8_default")
WRMSSE_GATE = 0.75  # порог: если бэктест хуже, артефакт не деплоим


def _train_step(task_id, command):
    """Шаг обучения в отдельном контейнере: изоляция тяжёлых зависимостей от окружения airflow."""
    return DockerOperator(
        task_id=task_id, image=TRAIN_IMAGE, command=command, working_dir="/app",
        environment={"ARTIFACT_DIR": "/app/artifacts_staging", "PYTHONPATH": "/app/src"},
        mounts=[Mount(source=HOST_PROJECT, target="/app", type="bind")],
        network_mode=NETWORK, docker_url="unix://var/run/docker.sock",
        auto_remove="success", mount_tmp_dir=False)


@dag(
    schedule="@weekly",  # данные недельные: приём раз в неделю, переобучение по расписанию
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["forecast", "retrain"],
)
def retrain_artifact():
    @task
    def validate_input() -> int:
        """Контракт входных данных: типы, границы, допустимые категории (pandera)."""
        import pandas as pd

        sys.path.insert(0, str(PROJECT / "src"))
        from forecast_service.config import settings
        from forecast_service.ml.schemas import validate_sample, weekly_input_schema

        df = pd.read_parquet(settings.data_path)
        validate_sample(df, weekly_input_schema)  # падает при нарушении контракта
        return int(len(df))

    rebuild = _train_step("rebuild", "python scripts/train_artifact.py")
    backtest = _train_step(
        "backtest", ["sh", "-c", "python scripts/foods_metrics.py && python scripts/artifact_eval.py"])

    @task
    def quality_gate() -> float:
        """Не выкатываем артефакт, если его оценка на последнем окне хуже порога."""
        res = json.loads((PROJECT / "metrics" / "artifact_eval.json").read_text(encoding="utf-8"))
        wrmsse = float(res["wrmsse12"])
        if wrmsse > WRMSSE_GATE:
            raise AirflowFailException(
                f"WRMSSE {wrmsse:.4f} хуже порога {WRMSSE_GATE}: артефакт не деплоим, нужен разбор")
        return wrmsse

    @task
    def deploy(wrmsse: float) -> None:
        """Промоушен: staging-артефакт копируется в рабочий ./artifacts, воркеры перечитывают
        модель по времени изменения файла - выкатка доезжает до рантайма без пересборки образа."""
        src, dst = PROJECT / "artifacts_staging", PROJECT / "artifacts"
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
        print(f"артефакт прошёл порог качества (WRMSSE={wrmsse:.4f}) и выложен в {dst}", flush=True)

    checked = validate_input()
    gate = quality_gate()
    checked >> rebuild >> backtest >> gate
    deploy(gate)


retrain_artifact()
