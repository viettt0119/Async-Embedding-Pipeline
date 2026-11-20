# syntax=docker/dockerfile:1.7
# Reproducible image: dependencies come from uv.lock (--frozen) and the tiktoken
# encoding is baked in at build time, so the container needs no network except the API.

ARG PYTHON_VERSION=3.12

# ---------- builder: resolve & install locked dependencies ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependencies only (cached until pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) The project itself, installed non-editable into the venv.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# 3) Pre-fetch the cl100k_base BPE file so runtime is offline-safe.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---------- runtime: minimal, non-root ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

RUN groupadd --system --gid 1000 pipeline \
    && useradd --system --uid 1000 --gid pipeline --create-home pipeline

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
COPY configs ./configs
COPY data/generate_sample_csv.py ./data/generate_sample_csv.py

RUN mkdir -p data/raw data/processed data/checkpoints \
    && chown -R pipeline:pipeline /app/data

ENV PATH=/app/.venv/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json

USER pipeline

# Mount a host directory here so checkpoints survive container restarts (resume!).
VOLUME ["/app/data"]

# `docker stop` sends SIGTERM by default. Python running as PID 1 has no SIGTERM handler, so
# the kernel ignores it and Docker SIGKILLs after 10s. SIGINT maps to KeyboardInterrupt ->
# asyncio cancellation -> graceful "interrupted, progress checkpointed" shutdown (exit 130).
STOPSIGNAL SIGINT

ENTRYPOINT ["embed-pipeline"]
CMD ["run", "--config", "configs/pipeline_config.yaml"]
