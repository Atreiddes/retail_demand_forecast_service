"""Airflow DAG периодического переобучения артефакта модели.

Продакшн-обёртка над офлайн-пайплайном: приём новых недель -> проверка входных данных
(pandera) -> переобучение артефакта -> бэктест -> quality gate -> деплой. Если бэктест
просел выше порога, новый артефакт не выкатывается (алерт, остаётся прежний). Это ловит
случай, когда переобучение ухудшает качество.

Это ops-компонент, в рантайм сервиса не входит (сервис только применяет готовый артефакт).
Airflow и apache-airflow ставятся отдельно в окружении оркестратора, в зависимости сервиса
не входят. Запуск в окружении обучения (lightgbm + пайплайн hw5).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

HW8 = Path(__file__).resolve().parent.parent
WRMSSE_GATE = 0.75  # порог: если бэктест хуже, артефакт не деплоим


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

        sys.path.insert(0, str(HW8 / "src"))
        from forecast_service.config import settings
        from forecast_service.ml.schemas import validate_sample, weekly_input_schema

        df = pd.read_parquet(settings.data_path)
        validate_sample(df, weekly_input_schema)  # падает при нарушении контракта
        return int(len(df))

    @task
    def rebuild() -> None:
        """Пересборка артефакта на свежих данных в staging-каталог: рабочий artifacts
        не трогаем, пока новая модель не пройдёт порог качества."""
        env = {**os.environ, "ARTIFACT_DIR": str(HW8 / "artifacts_staging")}
        subprocess.run([sys.executable, "scripts/train_artifact.py"], cwd=HW8, check=True, env=env)

    @task
    def backtest() -> None:
        """Walk-forward бэктест пайплайна (метрики для UI) и оценка именно staging-артефакта
        на последнем окне (metrics/artifact_eval.json) - порог качества проверяет тот файл
        модели, который поедет в развёртывание."""
        env = {**os.environ, "ARTIFACT_DIR": str(HW8 / "artifacts_staging")}
        subprocess.run([sys.executable, "scripts/foods_metrics.py"], cwd=HW8, check=True)
        subprocess.run([sys.executable, "scripts/artifact_eval.py"], cwd=HW8, check=True, env=env)

    @task
    def quality_gate() -> float:
        """Не выкатываем артефакт, если его оценка на последнем окне хуже порога."""
        import json

        res = json.loads((HW8 / "metrics" / "artifact_eval.json").read_text(encoding="utf-8"))
        wrmsse = float(res["wrmsse12"])
        if wrmsse > WRMSSE_GATE:
            raise AirflowFailException(
                f"WRMSSE {wrmsse:.4f} хуже порога {WRMSSE_GATE}: артефакт не деплоим, нужен разбор")
        return wrmsse

    @task
    def deploy(wrmsse: float) -> None:
        """Промоушен: staging-артефакт копируется в рабочий ./artifacts. Каталог
        примонтирован в контейнеры, воркеры перечитывают модель по времени изменения
        файла - выкатка доезжает до рантайма без пересборки образа."""
        import shutil

        src, dst = HW8 / "artifacts_staging", HW8 / "artifacts"
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
        print(f"артефакт прошёл порог качества (WRMSSE={wrmsse:.4f}) и выложен в {dst}", flush=True)

    checked = validate_input()
    rebuilt = rebuild()
    tested = backtest()
    gate = quality_gate()
    checked >> rebuilt >> tested >> gate
    deploy(gate)


retrain_artifact()
