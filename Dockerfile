# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# System deps
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Poetry + project
RUN pip install --no-cache-dir poetry==$POETRY_VERSION
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-dev --no-root --only main

COPY quantdesk /app/quantdesk
COPY scripts /app/scripts

CMD ["uvicorn", "quantdesk.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"] 