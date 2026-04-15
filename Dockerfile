FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.7.1

RUN pip install "poetry==${POETRY_VERSION}"

# Copy dependency manifests first for better layer caching
COPY pyproject.toml poetry.lock* /app/

# Copy source packages referenced by pyproject.toml (src, v2, app)
# before poetry install, otherwise poetry's local path installs fail.
COPY src /app/src
COPY v2 /app/v2
COPY app /app/app

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Zeabur injects $PORT at runtime. Default to 8000 for local docker runs.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT}"]
