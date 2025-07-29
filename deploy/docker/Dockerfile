# deploy/docker/Dockerfile

# ---- Stage 1: build a .venv with your requirements + uv ----
FROM python:3.12-slim AS builder
WORKDIR /app

# 1) Copy in your pinned-version requirements
COPY requirements.txt /requirements.txt

# 2) Install them with no cache
RUN pip install --no-cache-dir -r /requirements.txt

# 3) Install pip + uv (skip hadolint’s pin-and-cache rules here)
# hadolint ignore=DL3013,DL3042
RUN pip install --upgrade pip \
 && pip install uv

# 4) Copy project files for uv to resolve/transpile
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src

# 5) Create a .venv containing only your runtime deps
RUN uv sync

# ---- Stage 2: runtime only ----
FROM python:3.12-slim
WORKDIR /app

# 6) Bring in the complete .venv
COPY --from=builder /app/.venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 7) Copy code, data, and models
COPY src/    ./src
COPY data/   ./data
COPY models/ ./models

# 8) Default entrypoint
ENTRYPOINT ["python", "src/run_pipeline.py"]
