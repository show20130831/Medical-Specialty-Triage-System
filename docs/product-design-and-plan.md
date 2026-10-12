# Product Design and Delivery Plan

## 1. Product Definition

**Product name:** Medical Specialty Triage System

This is an educational NLP side project that classifies an English medical description into one of eight medical specialties. It demonstrates the ability to turn a fine-tuned language model into a usable software system with an API, browser interface, tests, container packaging, and an engineering workflow.

The system is a specialty classification prototype. It is not a diagnostic system, an urgency assessment tool, or a treatment recommendation service.

## 2. Product Goal

The product should allow a user to:

1. Enter one English medical description.
2. Submit the description through a browser.
3. Run the fine-tuned PubMedBERT model on CPU.
4. Receive a predicted specialty and model score.
5. Understand the system status and basic limitations.

The main portfolio goal is to demonstrate end-to-end software engineering rather than only model training.

## 3. Target Users

### Primary user

A technical reviewer, hiring manager, or software engineer who wants to inspect and run the project.

### Demonstration user

A developer who wants to test the local system with a synthetic or non-sensitive example.

### Out of scope

The system is not designed for direct clinical use, patient diagnosis, emergency triage, treatment selection, or storage of identifiable medical records.

## 4. Current User Flow

```text
User opens browser
    |
    v
Same-origin web page
    |
    v
POST /predict with one description
    |
    v
FastAPI validates the request
    |
    v
Tokenizer truncates input to 512 tokens
    |
    v
PubMedBERT runs one CPU inference
    |
    v
API returns specialty label and score
```

The model is loaded lazily on the first prediction or readiness request and then cached for reuse within the API process.

## 5. Supported Output Labels

The current model contract contains eight specialties:

- Cardiovascular & Pulmonary
- ENT & Ophthalmology
- Gastroenterology
- General Medicine
- Neurology
- OB-GYN
- Orthopedics
- Urology

The model score is a softmax output. It is not calibrated as a probability of clinical correctness.

## 6. System Architecture

### Application layers

| Layer | Responsibility |
| --- | --- |
| Browser client | Collects description input and displays the API result |
| FastAPI application | Provides HTTP routes, validation, errors, and readiness checks |
| Configuration | Reads model path, host, and port from environment variables |
| Model asset validator | Checks metadata and required local files before loading |
| Model loader | Loads tokenizer and classifier locally on CPU |
| Predictor | Tokenizes text, runs inference, and maps the result to a specialty |
| Logging middleware | Records request metadata without recording request bodies |
| Docker | Provides a reproducible CPU runtime |

### Runtime contract

- Input type: English `description`
- Maximum request length: 10,000 characters
- Maximum model sequence length: 512 tokens
- Inference device: CPU
- Model loading: local files only
- Custom remote code: disabled
- Weight format: safetensors
- Model mode: evaluation

## 7. API Contract

### `GET /`

Returns the browser client from the same origin as the API.

### `GET /health`

Checks that the HTTP service is responsive.

Expected response:

```json
{"status": "ok"}
```

This endpoint intentionally does not load the model.

### `GET /ready`

Checks whether the configured local model can be loaded and serve predictions.

Expected response:

```json
{"status": "ready"}
```

If the model path is missing or loading fails, the endpoint returns HTTP 503.

### `POST /predict`

Request:

```json
{
  "description": "persistent headache and visual changes"
}
```

Response:

```json
{
  "label": "Neurology",
  "score": 0.9051
}
```

Validation behavior:

| Condition | Response |
| --- | --- |
| Valid description | HTTP 200 |
| Blank description | HTTP 400 |
| More than 10,000 characters | HTTP 422 |
| Model unavailable | HTTP 400 for prediction, HTTP 503 for readiness |

## 8. Model Asset Design

The fine-tuned PubMedBERT export is kept outside Git. The expected local directory is:

```text
models/pubmedbert_description/
```

Required files include:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_config.json`

The application validates:

- BERT sequence classification metadata
- Description input configuration
- 512-token training limit
- Model position capacity
- Exactly eight labels
- Consistent `id2label` and `label2id` mappings

The repository does not publish the model weights or medical training data.

## 9. Privacy and Data Protection

The current prototype follows these rules:

- Request bodies are not written to application logs.
- Medical descriptions are not persisted by the API.
- Model weights and medical datasets are excluded from Git.
- The local Docker model mount is read-only.
- No authentication or access control is implemented yet.
- Users should use synthetic or non-identifiable examples during demonstrations.

A production deployment would still require access control, retention policy, encryption, monitoring, and a formal review of applicable privacy regulations.

## 10. Development Workflow

Each feature follows this workflow:

1. Define a small product or engineering goal.
2. Create or update an Issue.
3. Create a feature branch from `main`.
4. Implement the smallest useful change.
5. Add or update tests.
6. Run local checks.
7. Commit with an English conventional-style message.
8. Push the branch and open a pull request.
9. Wait for Ubuntu and Windows CI.
10. Merge with squash merge.
11. Synchronize the local `main` branch.

The protected `main` branch requires pull requests and CI checks.

## 11. Current Delivery Status

Completed:

- Fine-tuned PubMedBERT description classifier selected
- Local CPU model loading
- Model asset validation
- Prediction logic
- FastAPI service
- Browser interface
- Health and readiness endpoints
- Input length validation
- Privacy-safe request logging
- Environment-based configuration
- Docker and Docker Compose setup
- Windows and Ubuntu CI
- Protected main branch
- Portfolio-focused README
- Local Docker build and real model prediction verification

Not completed:

- Public online deployment
- Private cloud model distribution
- Authentication and authorization
- Rate limiting
- Production monitoring
- Capacity and load testing
- Clinical validation
- Automated model quality regression evaluation

## 12. Verification Evidence

The local project has been verified with:

- Python 3.10.9 on Windows
- CPU PyTorch and Transformers environment
- Automated test suite
- Dependency compatibility check
- Docker image build
- Docker container startup
- `/health` response
- `/ready` response
- Real CPU prediction through the container

The online deployment remains intentionally deferred because model storage, privacy, cost, and platform selection require a separate design decision.

## 13. Next Planned Work

### Priority 1: Version visibility

Add a `GET /version` endpoint that reports the service version, model contract version, and deployment commit without exposing secrets.

### Priority 2: Evaluation regression

Add a small versioned evaluation report that records accuracy, macro F1, weighted F1, and per-class metrics for the selected model.

### Priority 3: Deployment design

Choose a platform and private model distribution method. Document:

- Environment variables
- Model storage
- Health checks
- Startup time
- CPU capacity
- Log retention
- Rollback procedure
- Cost and platform limitations

### Priority 4: Production hardening

Consider authentication, rate limiting, request correlation IDs, structured log export, and monitoring only after the demonstration workflow is stable.

## 14. Product Success Criteria

The side project is successful when a reviewer can:

1. Understand the product purpose from the README in under one minute.
2. Install the documented dependencies.
3. Start the local service.
4. Open the browser interface.
5. Submit a description and receive a real model prediction.
6. Inspect the API contract and tests.
7. Trace a feature from branch to PR to merged commit.
8. See clear limitations and privacy boundaries.

