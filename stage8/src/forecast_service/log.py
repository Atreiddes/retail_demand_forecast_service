"""Единая настройка логирования сервиса."""
import logging

FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup(level=logging.INFO):
    """Корневой логгер на WARNING (сторонние библиотеки молчат), наш - на INFO.
    Идемпотентно: если обработчики уже есть (например, от uvicorn), basicConfig их не трогает."""
    logging.basicConfig(level=logging.WARNING, format=FMT)
    logging.getLogger("forecast").setLevel(level)
