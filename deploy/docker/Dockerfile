# deploy/docker/Dockerfile

# ---- Stage 1: build a .venv with uv ----
FROM python:3.12-slim AS builder
WORKDIR /app

# 1) Install pip + uv
RUN pip install --upgrade pip \
 && pip install uv

# 2) Copy pyproject.toml, uv.lock, README.md so uv can resolve deps
COPY pyproject.toml uv.lock README.md ./

# 3) Also copy your source tree so uv can build if needed
COPY src/ ./src

# 4) Create .venv with just the runtime deps
RUN uv sync

# ---- Stage 2: runtime only ----
FROM python:3.12-slim
WORKDIR /app

# 5) Pull in the complete .venv
COPY --from=builder /app/.venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 6) Copy your code, data, and models folders
COPY src/    ./src
COPY data/   ./data
COPY models/ ./models

# 7) Run your pipeline script directly
ENTRYPOINT ["python", "src/run_pipeline.py"]
