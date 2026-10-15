FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGE_HOST=0.0.0.0 \
    TRIAGE_PORT=8000

WORKDIR /app
# PyTorch is the largest layer and changes rarely; installing it before copying the
# source keeps code changes from re-downloading it.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu \
    && useradd --create-home --uid 10001 app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[inference]"

USER app
EXPOSE 8000
# Liveness only: /health does not load the model. Use /ready for readiness probes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['TRIAGE_PORT']}/health\", timeout=4)"]
CMD ["python", "-m", "triage_system"]
