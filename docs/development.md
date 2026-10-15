# Development Guide

Setup, API reference, testing, and repository layout. For the product overview, see the
[README](../README.md).

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[test,inference]"
```

Add `train` and `finetune` to the extras to work on the training pipeline
([training.md](training.md)).

### Model bundle

Place a bundle at `models/pubmedbert_description/`. Either train one with the pipeline
in [training.md](training.md) (needs the MTSamples dataset and one GPU for a few
minutes) or, with access to the private archive:

```powershell
hf download show20130831/medical-specialty-pubmedbert --local-dir models/pubmedbert_description
```

The weights are not public because the training data does not state a license.

### Run the service

```powershell
$env:TRIAGE_MODEL_DIR = (Resolve-Path ".\models\pubmedbert_description").Path
$env:TRIAGE_PRELOAD = "true"
.\.venv\Scripts\python.exe -B -m triage_system
```

Open http://127.0.0.1:8000/. Settings are validated at startup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRIAGE_MODEL_DIR` | — | Model bundle directory |
| `TRIAGE_HOST`, `TRIAGE_PORT` | `127.0.0.1`, `8000` | Bind address |
| `TRIAGE_PRELOAD` | `false` | Load and warm the model at startup |
| `TRIAGE_MONITORING_CONFIG` | built-in defaults | Drift thresholds (`configs/monitoring.yaml`) |

### Docker

```powershell
docker compose up --build                      # API only
docker compose --profile monitoring up --build # plus Prometheus, Alertmanager, Grafana
```

The image holds no weights; Compose mounts `./models` read-only. Ports bind to
localhost: API 8000, Grafana 3000, Prometheus 9090, Alertmanager 9093. Copy
`.env.example` to `.env` to set the Grafana admin password.

## API

| Route | Purpose |
| --- | --- |
| `POST /predict` | Classify one description |
| `GET /health` | Liveness; does not load the model |
| `GET /ready` | Model loaded and able to serve |
| `GET /version` | Served weights SHA-256, training commit, calibration |
| `GET /metrics` | Prometheus metrics; no text and no per-specialty values |
| `GET /` | Browser client |

```powershell
$body = @{ description = "Persistent knee pain and stiffness when walking." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```

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

Scores are temperature-calibrated softmax values. Below the validated threshold,
`uncertain` is `true` and the browser asks for manual review. Requests with
`X-Synthetic-Traffic: 1` (load tests, probes) are served but kept out of drift
monitoring.

| Condition | Status |
| --- | --- |
| Prediction returned | `200` |
| Blank or over 1,000 characters | `422` |
| Request body over 16 KB, or without `Content-Length` | `413` / `411` |
| Model not configured, missing, or failed to load | `503` |

## Testing

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

116 tests cover the API and its status codes, input limits, model asset and calibration
validation, loading, prediction and abstention, configuration, privacy-safe logging and
metrics, drift windows (concurrency, failure isolation, timeouts), the data pipeline,
statistics, reports, and the release gate. Training tests use synthetic data, so CI
needs neither the dataset nor a GPU.

CI (GitHub Actions) on every pull request:

| Job | Checks |
| --- | --- |
| API tests | Full suite on Ubuntu and Windows, Python 3.10 |
| Service smoke test | A tiny random BERT bundle served through every route |
| Docker image build | Non-root user, `/health`, `/ready` |
| Monitoring configuration | Alert rules and configs with promtool and amtool |
| Dependency audit | `pip-audit`, advisory only |

`main` is protected by a ruleset: pull requests only, required checks on an up-to-date
branch, no force pushes.

## Repository layout

```text
src/triage_system/      Serving: API, model loading, prediction, monitoring, web client
src/triage_training/    Data pipeline, baseline, fine-tuning, evaluation, release gate
configs/                Training, taxonomy, release criteria, monitoring thresholds
deploy/monitoring/      Prometheus, alert rules, Alertmanager, Grafana dashboard
deploy/loadtest/        k6 script and synthetic payloads
deploy/current.json     Deployment record (image and model archive revision)
reports/                Evaluation, release, split, load-test reports (aggregates only)
docs/                   Design, training, validation and operations, drills
scripts/                Drill traffic, drift window estimate, load test runner
tests/                  Automated tests and synthetic behavioural cases
```
