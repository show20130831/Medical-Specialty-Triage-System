# Medical Specialty Triage System

A web application in development for classifying English medical descriptions into eight specialties using a fine-tuned biomedical language model.

## Project Status

A minimal FastAPI service now exposes a browser interface, health endpoint, and single-description prediction endpoint. The local PubMedBERT export is loaded lazily on the first prediction request and remains CPU-only.

This repository is being built incrementally to practice software development through small features, tests, and pull requests.

## Local Setup

Requirements: Python 3.10 or newer with pip and venv support. Local verification used Windows and Python 3.10.9. GitHub Actions also passed installation, endpoint testing, and dependency checks on Ubuntu and Windows with Python 3.10; other Python versions have not been verified.

Run the following commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

If `python` is not available in your terminal, use the full path to your Python executable for the first command. No environment activation is required when using the explicit `.venv` executable path.

The editable installation uses the source files in `src/`. The `test` extra installs the test dependencies declared in `pyproject.toml`. Direct dependencies are pinned; transitive dependencies are not fully locked.

## Run the API

```powershell
.\.venv\Scripts\python.exe -B -m uvicorn triage_system.api:app --host 127.0.0.1 --port 8000
```

Keep the terminal open while using the service. Press `Ctrl+C` in that terminal to stop it. The service listens only on the local machine.

- Health endpoint: <http://127.0.0.1:8000/health>
- Readiness endpoint: <http://127.0.0.1:8000/ready>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Browser interface: <http://127.0.0.1:8000/>

`GET /health` returns HTTP `200` with:

```json
{"status": "ok"}
```

In `/docs`, expand `GET /health`, select **Try it out**, then **Execute**. Verify that the server response is HTTP `200` with the body above.

The health endpoint reports that the API is responding. It does not check model readiness and does not load model weights.

`GET /ready` loads the configured local model once and returns HTTP `200` with `{"status":"ready"}` when predictions can be served. If `TRIAGE_MODEL_DIR` is missing or the export cannot load, it returns HTTP `503` with an actionable error. This distinction is intended for local checks and future deployment probes.

Open the browser interface at `/` after setting `TRIAGE_MODEL_DIR`. Enter an English medical description and select **Classify description**. The page uses the same origin as the API, so no separate frontend server or CORS configuration is needed for this local prototype.

### Prediction Endpoint

Set the private model directory before starting the API:

```powershell
$env:TRIAGE_MODEL_DIR = (Resolve-Path ".\\models\\pubmedbert_description").Path
.\.venv\Scripts\python.exe -B -m uvicorn triage_system.api:app --host 127.0.0.1 --port 8000
```

Send one English medical description:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"description":"persistent headache and visual changes"}'
```

The response contains the predicted specialty and a softmax score. The score is a model output, not a calibrated probability or a diagnosis. Blank descriptions are rejected, and inputs are truncated to the model contract's 512-token limit.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

The endpoint test checks the HTTP status code and JSON response using `TestClient`. It does not require a running Uvicorn server, model weights, or medical data. `-B` disables Python bytecode writes; `-p no:cacheprovider` disables pytest's cache plugin.

### Health Endpoint Verification

The health endpoint implementation was verified on Windows with Python 3.10.9:

| Check | Result |
| --- | --- |
| Automated endpoint test | 1 passed, 2 warnings |
| Dependency compatibility (`pip check`) | No broken requirements found |
| Live HTTP request to `/health` | HTTP 200 with `{"status": "ok"}` |
| Live HTTP requests to `/docs` and `/openapi.json` | HTTP 200; documentation and health route confirmed |
| Manual browser verification | Health response and Swagger UI execution confirmed |

The two test warnings concern deprecated HTTPX usage in Starlette's test client and the `anyio.abc.BlockingPortal` alias. They did not cause test failures and have not been suppressed. Warning counts may change as transitive dependencies change. Remote CI verification passed on Ubuntu and Windows; see the recorded run below.

## Continuous Integration

The [API CI workflow](.github/workflows/ci.yml) is configured to run on pull requests targeting `main` and pushes to `main`. A push to a feature branch without an open pull request does not trigger this workflow.

Each run uses separate GitHub-hosted Ubuntu and Windows runners with Python 3.10. Each job checks out the code, sets up Python, installs the project with its test dependencies, runs pytest, and checks dependency compatibility with `pip check`. Model weights and medical data are not required.

The workflow grants read-only repository contents access and does not deploy or publish packages. A failed installation, test, or dependency check fails the job. Each job has a 15-minute timeout; failure in one operating system job does not cancel the other.

### View Results

After the workflow is pushed and a pull request is opened:

1. Open the pull request's **Checks** tab, or the repository's **Actions** tab and select **API CI**.
2. Check both jobs: `API tests (ubuntu-latest, Python 3.10)` and `API tests (windows-latest, Python 3.10)`.
3. Open a job and expand the relevant step to inspect its output when a check fails.

**Verification status:** Both Python 3.10 jobs passed installation, pytest, and `pip check` in the [first GitHub Actions run](https://github.com/show20130831/Medical-Specialty-Triage-System/actions/runs/35526840220) for commit `04bc25c`. Local YAML parsing and structure checks also passed. This records that specific run; use the latest PR checks to assess subsequent commits.

The `Protect main` ruleset requires both CI jobs and requires branches to be up to date before merging.

## Planned Scope

- Use existing fine-tuned PubMedBERT weights without retraining.
- Run single-description inference on a local CPU.
- Expose predictions through the FastAPI service.
- Provide a web interface for entering descriptions and viewing results.

## Model and Data

Model weights, medical datasets, and private training notebooks are not included in this repository. Use the local export setup below for the standalone loading check.

## Development Workflow

Each feature will follow an issue, feature branch, implementation, verification, and pull request review before merging into `main`.

## Intended Use

This is an educational prototype, not a clinically validated medical service. Planned predictions are specialty classifications, not diagnoses or urgency assessments.

## Local Model Loading

The standalone loader is independent of FastAPI startup and `/health`. The prediction endpoint reuses the same validated loader and caches one CPU model instance for the process.

Provide a trusted local export directory containing these non-empty files:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_config.json`

Use an existing export path or place your private export under `models/pubmedbert_description/`, which Git ignores. No model weights are distributed or downloaded by this project. Do not substitute unmodified base-model weights for the fine-tuned classifier.

The export must declare a single-label BERT classifier, the eight supported specialty labels with inverse ID mappings, `input_type: description`, and `max_length: 512`. This loader targets the current single-file export, not arbitrary Hugging Face models or sharded checkpoints. File checks cannot establish weight integrity or training provenance.

### Optional CPU Dependencies

Run from the project root in PowerShell, after creating `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[inference,test]"
.\.venv\Scripts\python.exe -m pip check
```

Install CPU PyTorch first so the optional dependency group reuses that distribution. These direct versions match the prior project's working CPU environment; the new project's local installation and real-model loading have now been verified. CI continues to install only `.[test]`.

### Manual Offline Loading Check

For an export placed under `models/pubmedbert_description/`, run from the repository root (or provide another local export path):

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -B -m triage_system.model_loader --model-dir ".\models\pubmedbert_description"
```

The loader uses local files only, disables custom remote code, selects safetensors, places the model on CPU, and switches to evaluation mode. It explicitly sets the tokenizer limit to 512; the export's placeholder tokenizer maximum is not used. Future prediction code must also enable truncation at that limit.

Expected success output:

```text
Loaded 8 labels on CPU in evaluation mode.
Maximum input length: 512 tokens. No prediction was run.
```

Metadata validation and tests with lightweight doubles do not prove real weights can load. The historical reload report belongs to an earlier environment; the new project's local export has loaded successfully on CPU. CI checks do not require private weights.

### Loader Verification Status

- Windows / Python 3.10.9: 30 tests passed, including the existing health test; the same two dependency deprecation warnings remain.
- `pip check` passed for the installed API/test environment. Inference dependencies have not been installed in this environment.
- The existing export passed metadata validation for the eight labels and 512-token limit without reading weight contents.
- New modules and tests passed Ruff checks using an existing local development tool; Ruff was not added as a project dependency.
- The actual local export loaded successfully on CPU in evaluation mode with eight labels and a 512-token limit. A prediction request is verified with test doubles; real prediction quality remains dependent on the private export.
