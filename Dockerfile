FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGE_HOST=0.0.0.0 \
    TRIAGE_PORT=8000

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir ".[inference]"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn triage_system.api:app --host ${TRIAGE_HOST} --port ${TRIAGE_PORT}"]
