# Сборка зависимостей в отдельной стадии, на выходе только /app/.venv
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Сначала только зависимости: слой кэшируется, пока не менялись lock и pyproject
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Затем сам проект
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# Финальная стадия без uv и кэшей сборки
FROM python:3.12-slim

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

WORKDIR /app

# LightGBM требует libgomp, в slim его нет
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь
RUN useradd --create-home --uid 1000 app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/artifacts /app/artifacts
COPY --from=builder /app/data /app/data
COPY --from=builder /app/metrics /app/metrics
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

USER app

EXPOSE 8000

CMD ["uvicorn", "forecast_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
