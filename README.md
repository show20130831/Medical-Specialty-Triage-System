# Medical Specialty Triage System

An educational NLP application that suggests which of eight medical specialties an
English clinical description belongs to, with calibrated confidence and an "uncertain"
answer when the model should defer to a person.

The project covers the full ML workflow: a reproducible data and training pipeline with
leakage-free splits, a statistically evaluated PubMedBERT model, a FastAPI service with a
browser interface, Docker packaging, automated tests, and CI.

![Demo: two confident predictions, then an ambiguous description marked uncertain](docs/images/demo.gif)

## Current Status

The application runs locally with the fine-tuned model on CPU.

```text
Browser -> FastAPI -> PubMedBERT (CPU) -> top-3 specialties + calibrated confidence
                                        -> "uncertain" below the validated threshold
```

The trained weights are kept in a private Hugging Face repository because the MTSamples
training data does not state a license, so no public online demo is hosted. Anyone can
reproduce the model from the public Kaggle dataset with the training pipeline in a few
minutes on one GPU; see [docs/training.md](docs/training.md).

| Confident prediction | Uncertain prediction |
| --- | --- |
| ![Orthopedics at 98% confidence](docs/images/prediction-confident.png) | ![Ambiguous chest and back pain marked uncertain](docs/images/prediction-uncertain.png) |

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

Place a model bundle at `models/pubmedbert_description/`: either train one with
[docs/training.md](docs/training.md) or, with access to the private Hugging Face repository,
run `hf download show20130831/medical-specialty-pubmedbert --local-dir models/pubmedbert_description`.

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
