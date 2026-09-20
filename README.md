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

The two test warnings concern deprecated HTTPX usage in Starlette's test client and the `anyio.abc.BlockingPortal` alias. They did not cause test failures and have not been suppressed. Warning counts may change as transitive dependencies change. Remote CI verification is pending; see the workflow details below.

## Continuous Integration

The [API CI workflow](.github/workflows/ci.yml) is configured to run on pull requests targeting `main` and pushes to `main`. A push to a feature branch without an open pull request does not trigger this workflow.

Each run uses separate GitHub-hosted Ubuntu and Windows runners with Python 3.10. Each job checks out the code, sets up Python, installs the project with its test dependencies, runs pytest, and checks dependency compatibility with `pip check`. Model weights and medical data are not required.

The workflow grants read-only repository contents access and does not deploy or publish packages. A failed installation, test, or dependency check fails the job. Each job has a 15-minute timeout; failure in one operating system job does not cancel the other.

### View Results

After the workflow is pushed and a pull request is opened:

1. Open the pull request's **Checks** tab, or the repository's **Actions** tab and select **API CI**.
2. Check both jobs: `API tests (ubuntu-latest, Python 3.10)` and `API tests (windows-latest, Python 3.10)`.
3. Open a job and expand the relevant step to inspect its output when a check fails.

**Verification status:** The workflow has passed local YAML parsing and structure checks only. It has not yet been executed on GitHub Actions; successful CI runs on both operating systems are still required to validate this setup.

This change does not configure required status checks in branch protection. Those checks will be selected separately after the workflow has run successfully.

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
