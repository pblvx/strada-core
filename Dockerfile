# syntax=docker/dockerfile:1
# STRADA-CORE · production image (slim, CPU-only, ONNX inference — no PyTorch).
FROM python:3.12-slim

# uv from the official static binary (no pip, no build cruft).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Resolve/install runtime dependencies first for Docker layer caching.
# --no-install-project: we serve `src/` by path, not as an installed package.
# No --extra: torch/fastf1 (PINN training, data ingest) stay OUT of the runtime image.
COPY pyproject.toml ./
RUN uv sync --no-install-project --no-cache

# Application code, front-end and inference assets.
COPY src/ ./src/
COPY ui/ ./ui/
COPY data/models/ ./data/models/
COPY data/raw/silverstone_2024.parquet ./data/raw/silverstone_2024.parquet

EXPOSE 8000

# FastAPI + Uvicorn. ONNX Runtime serves the thermal surrogate on CPU.
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
