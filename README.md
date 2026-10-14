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

- Classifies description-only input into eight specialties and returns the top three.
- Reports temperature-calibrated confidence and marks low-confidence inputs as
  `uncertain` using a threshold chosen on validation data.
- Serves a same-origin browser interface at `/`.
- Exposes `/predict`, `/health`, and `/ready` endpoints.
- Loads the local model once per API process in CPU evaluation mode.
- Rejects blank and oversized descriptions with `422`; returns `503` when the model is
  unavailable.
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
$body = @{ description = "Persistent knee pain and stiffness when walking." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body $body
```

Example response:

```json
{
  "label": "Orthopedics",
  "score": 0.9825,
  "uncertain": false,
  "top": [
    {"label": "Orthopedics", "score": 0.9825},
    {"label": "Neurology", "score": 0.0043},
    {"label": "General Medicine", "score": 0.0029}
  ]
}
```

Scores are softmax outputs after temperature scaling fitted on validation data
(`calibration.json` in the model bundle). When the top score falls below the
validated threshold, `uncertain` is `true` and the browser asks for manual review; on
the test split this withholds 7.5% of inputs and raises accuracy on the rest from
88.4% to 91.6%.

| Condition | Status |
| --- | --- |
| Prediction returned | `200` |
| Blank or over 10,000 characters | `422` |
| Model not configured or failed to load | `503` |

## Docker

The container does not include model weights. Mount the private export read-only through Docker Compose:

```powershell
docker compose up --build
```

The browser interface is available at http://127.0.0.1:8000/. Stop the container with:

```powershell
docker compose down
```

## Model Development

The classifier is trained by the `triage_training` package from the public Kaggle
MTSamples dataset: cleaning, taxonomy labeling, deduplication, leakage-free stratified
splits, data validation, a TF-IDF baseline, and PubMedBERT fine-tuning, all driven by
[configs/train.yaml](configs/train.yaml). See [docs/training.md](docs/training.md).

```bash
python -m triage_training prepare-data
python -m triage_training baseline
python -m triage_training finetune
```

### Results

Test split of 398 unique descriptions that never appear in training. Intervals are 95%
bootstrap intervals; full analysis in [reports/model_evaluation.md](reports/model_evaluation.md).

| Model | Test accuracy | Test macro F1 |
| --- | --- | --- |
| TF-IDF + logistic regression | 0.744 [0.701, 0.786] | 0.742 [0.691, 0.785] |
| PubMedBERT (seed 42, served) | 0.884 [0.854, 0.915] | 0.889 [0.857, 0.919] |
| PubMedBERT, 3 seeds | | 0.892 ± 0.003 (mean ± SD) |

- PubMedBERT beats the baseline by +0.148 macro F1 [+0.107, +0.192]; exact McNemar
  p < 0.001.
- The original row-level split put 365 identical descriptions in both train and test.
  On it, test rows seen in training scored 99% (memorised) and rows with conflicting
  labels 41%, so its 0.80 macro F1 mixed both effects. Unseen rows scored 81%.
- Rows labeled by keyword rules are easier (93%) than directly mapped rows (86%),
  because their labels derive from terms that also appear in the text.

## Testing

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

The test suite covers API behavior, input validation, model asset validation, offline CPU loading contracts, prediction logic, configuration, privacy-safe logging, and the training data pipeline (labeling, deduplication, split validation, baseline). Training tests use synthetic data, so CI needs neither the dataset nor a GPU. CI runs the test suite on Windows and Ubuntu with Python 3.10.

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
- Calibration and the `uncertain` flag hold for descriptions like the training data.
  Vague or out-of-scope text (for example "Follow-up visit.") can still receive a
  confident label; the model has no out-of-distribution detection.
- CPU inference is intended for demonstration and small-volume testing.
- Public deployment, authentication, rate limiting, and capacity testing are not implemented.

## Project Structure

```text
src/triage_system/
├── api.py              # FastAPI routes and browser entry point
├── config.py           # Environment-backed settings
├── model_assets.py     # Local export validation
├── model_loader.py     # Offline CPU model loading
├── predictor.py        # Description inference
├── logging_config.py   # Privacy-safe request logging
└── web/index.html      # Browser client
src/triage_training/
├── data.py             # Cleaning, labeling, deduplication, split validation
├── baseline.py         # TF-IDF + logistic regression
├── finetune.py         # PubMedBERT fine-tuning and export
└── __main__.py         # CLI: prepare-data, baseline, finetune
configs/                # Training config and specialty taxonomy
tests/                  # Automated tests
Dockerfile              # Reproducible CPU image
compose.yaml            # Local container startup
```

## License

This project is intended for educational and portfolio use.
