# Medical Specialty Triage System

A web application in development for classifying English medical descriptions into eight specialties using a fine-tuned biomedical language model.

## Project Status

A minimal FastAPI service with a health endpoint is implemented and has been verified locally. Model inference and the web interface are planned but are not implemented yet.

This repository is being built incrementally to practice software development through small features, tests, and pull requests.

## Local Setup

Requirements: Python 3.10 or newer with pip and venv support. Local verification used Windows and Python 3.10.9; other Python versions and operating systems have not been verified yet.

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
- Interactive API documentation: <http://127.0.0.1:8000/docs>

`GET /health` returns HTTP `200` with:

```json
{"status": "ok"}
```

In `/docs`, expand `GET /health`, select **Try it out**, then **Execute**. Verify that the server response is HTTP `200` with the body above.

The health endpoint reports that the API is responding. It does not check model readiness and does not load model weights.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

The endpoint test checks the HTTP status code and JSON response using `TestClient`. It does not require a running Uvicorn server, model weights, or medical data. `-B` disables Python bytecode writes; `-p no:cacheprovider` disables pytest's cache plugin.

### Local Verification

The current implementation was verified on Windows with Python 3.10.9:

| Check | Result |
| --- | --- |
| Automated endpoint test | 1 passed, 2 warnings |
| Dependency compatibility (`pip check`) | No broken requirements found |
| Live HTTP request to `/health` | HTTP 200 with `{"status": "ok"}` |
| Live HTTP requests to `/docs` and `/openapi.json` | HTTP 200; documentation and health route confirmed |
| Manual browser verification | Health response and Swagger UI execution confirmed |

The two test warnings concern deprecated HTTPX usage in Starlette's test client and the `anyio.abc.BlockingPortal` alias. They did not cause test failures and have not been suppressed. Warning counts may change as transitive dependencies change. CI verification has not been added yet.

## Planned Scope

- Use existing fine-tuned PubMedBERT weights without retraining.
- Run single-description inference on a local CPU.
- Expose predictions through the FastAPI service.
- Provide a web interface for entering descriptions and viewing results.

## Model and Data

Model weights, medical datasets, and private training notebooks are not included in this repository. Model setup instructions will be added with the inference feature.

## Development Workflow

Each feature will follow an issue, feature branch, implementation, verification, and pull request review before merging into `main`.

## Intended Use

This is an educational prototype, not a clinically validated medical service. Planned predictions are specialty classifications, not diagnoses or urgency assessments.
