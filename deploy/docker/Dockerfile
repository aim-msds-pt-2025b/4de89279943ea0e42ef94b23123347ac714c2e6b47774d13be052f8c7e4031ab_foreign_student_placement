# deploy/docker/Dockerfile

# ---- Stage 1: build a .venv with all deps from pyproject.toml ----
FROM python:3.12-slim AS builder
WORKDIR /app

# Copy lockfile, pyproject, README, and source so uv can build
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src

# Install pip and uv (skip hadolint pin/cache warnings)
# hadolint ignore=DL3013,DL3042
RUN pip install --upgrade pip \
 && pip install uv

# Build the editable .venv (runtime + dev)
RUN uv sync --dev

# ---- Stage 2: runtime with Airflow installed ----
FROM apache/airflow:2.9.3
WORKDIR /app

# Bring in the complete .venv so it has your deps
COPY --from=builder /app/.venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy your pipeline code (and any data/models if you want baked in)
COPY src/    ./src
COPY data/   ./data
COPY models/ ./models

# Default entrypoint will still be the airflow binary
# (webserver/scheduler commands come from docker-compose)
