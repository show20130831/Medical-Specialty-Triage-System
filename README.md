# Medical Specialty Triage System

An educational NLP application that classifies an English medical description into one of eight medical specialties.

The project demonstrates an end-to-end ML application workflow: a fine-tuned PubMedBERT model, a FastAPI service, a browser interface, CPU inference, Docker packaging, automated tests, and GitHub-based development practices.

## Current Status

The local MVP is complete and runs with the private fine-tuned PubMedBERT export.

```text
Browser -> FastAPI -> PubMedBERT -> CPU inference -> Specialty + score
```

The public online deployment is not included yet. Model weights remain private and are excluded from Git.

## Features

- Classifies description-only input into eight specialties.
- Serves a same-origin browser interface at `/`.
- Exposes `/predict`, `/health`, and `/ready` endpoints.
- Loads the local model once per API process in CPU evaluation mode.
- Validates blank and oversized descriptions before inference.
- Logs request metadata without logging medical text.
- Provides Docker and Docker Compose configuration.

## Technology

- Python 3.10+
- FastAPI and Uvicorn
- PyTorch CPU
- Hugging Face Transformers
- PubMedBERT fine-tuned for description classification
- Pytest
- Docker
- GitHub Actions

## Quick Start

Create the environment and install the API and test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Install CPU inference dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[inference]"
```

Place the private model export at `models/pubmedbert_description/`.

Set the model path and start the API:

```powershell
$env:TRIAGE_MODEL_DIR = (Resolve-Path ".\models\pubmedbert_description").Path
$env:TRIAGE_HOST = "127.0.0.1"
$env:TRIAGE_PORT = "8000"
.\.venv\Scripts\python.exe -B -m uvicorn triage_system.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ in a browser.

## API Example

```powershell
$body = @{ description = "persistent headache and visual changes" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body $body
```

Example response:

```json
{
  "label": "Neurology",
  "score": 0.9051
}
```

The score is a model output, not a calibrated probability. Blank descriptions return `400`; descriptions over 10,000 characters return `422`.

## Docker

The container does not include model weights. Mount the private export read-only through Docker Compose:

```powershell
docker compose up --build
```

The browser interface is available at http://127.0.0.1:8000/. Stop the container with:

```powershell
docker compose down
```

## Testing

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

The test suite covers API behavior, input validation, model asset validation, offline CPU loading contracts, prediction logic, configuration, and privacy-safe logging. CI runs the test suite on Windows and Ubuntu with Python 3.10.

## Engineering Highlights

- Model assets are validated before loading, without reading weight bytes during metadata checks.
- Model loading is local-only with remote code disabled and safetensors enabled.
- `/health` remains independent from model availability; `/ready` reports model readiness.
- The model is cached after the first load and reused for subsequent requests.
- Request logs contain method, path, status, and duration, but never request bodies.
- Model weights, medical data, notebooks, and virtual environments are excluded from Git.
- Changes are developed through feature branches, tests, CI, and pull requests.

## Limitations

- Educational prototype, not a clinically validated system.
- English description input only.
- Specialty classification is not diagnosis, urgency assessment, or treatment advice.
- CPU inference is intended for demonstration and small-volume testing.
- Public deployment, authentication, rate limiting, and capacity testing are not implemented.

## Project Structure

```text
src/triage_system/
∫w^~)ﬁvÈ›y¯ßy€ßuÁ‚ùÁ@ api.py              # FastAPI routes and browser entry pointßuÁ‚ùÁ\∫w^~)ﬁvÈ›y¯ßy– config.py           # Environment-backed settingsßuÁ‚ùÁ\∫w^~)ﬁvÈ›y¯ßy– model_assets.py     # Local export validation
È›y¯ßy€ßuÁ‚ùÁ@∫w^~)ﬁt model_loader.py     # Offline CPU model loadingßuÁ‚ùÁ\∫w^~)ﬁvÈ›y¯ßy– predictor.py        # Description inferenceßuÁ‚ùÁ\∫w^~)ﬁvÈ›y¯ßy– logging_config.py   # Privacy-safe request logging
∫w^~)ﬁvÈ›y¯ßy€ßuÁ‚ùÁ@ web/index.html      # Browser client
tests/                  # Automated tests
Dockerfile              # Reproducible CPU image
compose.yaml            # Local container startup
```

## License

This project is intended for educational and portfolio use.
