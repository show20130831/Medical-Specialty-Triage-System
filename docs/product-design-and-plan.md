# Product Design and Delivery Plan

**Last updated:** 2026-10-14
**Status:** Model development, evaluation, and calibrated serving are merged. The
validation and operations MVP (release gate, traceability, aggregate monitoring, drift
check, runbook) is complete and in review; see
[validation-and-operations.md](validation-and-operations.md).

## 1. Product Definition

**Product name:** Medical Specialty Triage System

An educational research and engineering prototype that suggests which of eight specialty groups a short English
clinical description belongs to. It returns the three most likely specialties with
calibrated confidence and answers "uncertain" when the model should defer to a person.

"Triage" here means **routing to a specialty**, not clinical urgency triage. The system is
not a diagnostic tool, an urgency assessment, or a treatment recommendation service, and
it has not been clinically validated.

## 2. Goals

### Portfolio goal

The project is a job-application portfolio piece for a data science internship in health
technology. It must show the whole ML lifecycle, not only model training:

| Target skill (from the internship description) | Evidence in this project |
| --- | --- |
| Data cleansing, validation, verification | Taxonomy labeling, deduplication, conflict resolution, split validation, data report (§6) |
| Training and retraining AI systems | Config-driven, device-agnostic training CLI with run provenance (§7) |
| Statistical analysis, hypothesis testing | Bootstrap intervals, paired bootstrap, exact McNemar test, seed variation (§8) |
| Integrating models into products | FastAPI service and browser client with asset validation (§9–§11) |
| Quality assurance before deployment | 81 automated tests, CI on Ubuntu and Windows, reload and parity checks (§14) |
| Data governance | Text-free artifacts, privacy-safe logging, license-driven hosting decision (§12–§13) |
| AI DevOps, monitoring | Release gate, `/version`, Prometheus `/metrics`, in-process drift windows with a validated size, runbook, CI with service smoke test and Docker build (§14, §17) |

### Product goal

A user enters one clinical description and receives a specialty suggestion that is
honest about its uncertainty.

## 3. Users and Use Case

| User | Need |
| --- | --- |
| Technical reviewer or hiring manager (primary) | Understand the design, results, and engineering quality from the repository in minutes |
| Developer | Reproduce the model and run the service locally |

The system is a demonstration prototype and has no supported operational users.

**Illustrative scenario (not a supported use):** a coordinator routing clinician-written
notes to specialty queues would want a first-pass suggestion and would need ambiguous
notes sent to a person. This scenario is used only to derive design requirements, such
as returning the top three specialties and abstaining when confidence is low.

**Out of scope:** direct clinical use, patient-facing symptom checking, diagnosis,
emergency triage, treatment selection, and storage of identifiable medical records.

**Input assumption:** the model is trained on clinician-written summaries (MTSamples
`description` field, median 115 characters, maximum 433). Lay symptom descriptions and
long notes are outside the training distribution.

## 4. User Flow

```text
User opens browser
  -> same-origin web page (synthetic example buttons available)
  -> POST /predict {"description": ...}
  -> body at most 16 KB (else 413); description non-blank, at most 1,000
     characters (else 422), so no input reaches the 512-token limit
  -> PubMedBERT CPU inference (model loaded once per process)
  -> temperature-scaled softmax (T = 0.95)
  -> top-3 specialties; uncertain = top score < 0.65
  -> browser shows ranked bars, or a manual-review message when uncertain
```

## 5. Output Labels and Taxonomy

MTSamples has 40 source specialties. They are mapped to eight groups by
`configs/taxonomy.yaml`, whose SHA-256 is recorded in every dataset report.

| Group | Source specialties (direct mapping) |
| --- | --- |
| Cardiovascular & Pulmonary | Cardiovascular / Pulmonary |
| ENT & Ophthalmology | ENT - Otolaryngology, Ophthalmology, Dentistry |
| Gastroenterology | Gastroenterology |
| General Medicine | Internal Medicine, General Medicine, Office Notes, Allergy / Immunology, Endocrinology, Hematology - Oncology, Nephrology, Rheumatology, Infectious Disease |
| Neurology | Neurology, Neurosurgery, Psychiatry / Psychology |
| OB-GYN | Obstetrics / Gynecology |
| Orthopedics | Orthopedic, Physical Medicine - Rehab, Pain Management |
| Urology | Urology |

- **Keyword rules:** rows from Surgery, Radiology, SOAP / Chart / Progress Notes, and
  Discharge Summary are assigned by summing weighted keyword matches in the `keywords`
  field (weight 2 for specific terms, 1 for shared anatomical terms). Rows with no match
  are dropped.
- **Excluded specialties (893 rows after cleaning; raw counts below):** Consult -
  History and Phy. (516), Emergency Room Reports (75), Pediatrics - Neonatal (70),
  Podiatry (47), Dermatology (29), Cosmetic / Plastic Surgery (27), and ten smaller ones. The model cannot output these specialties
  and will force such inputs into one of the eight groups.
- The groups are an engineering simplification for an eight-class problem, not a
  clinical referral standard. Some placements are debatable (Psychiatry under Neurology,
  Dentistry under ENT, Nephrology under General Medicine rather than Urology).

## 6. Data

### Source and license

MTSamples transcription samples (mtsamples.com) as published on Kaggle
(`mtsamples.csv`, 4,999 rows, Windows-1252 encoded). **The dataset does not state a
license.** Raw data, prepared splits, and anything containing medical text are excluded
from Git and Docker; only aggregate reports are committed.

### Pipeline (`python -m triage_training prepare-data`)

1. Drop rows without a specialty or description (4,999 → 4,993).
2. Assign groups by taxonomy (→ 3,687 labeled rows: 2,451 direct, 1,236 keyword).
3. Normalise descriptions (case, punctuation, whitespace) and remove duplicates
   (1,492 rows, 40% of labeled rows).
4. Resolve conflicting labels for the same description (409 descriptions): prefer labels
   from direct mappings; drop the 209 that remain tied.
5. Require at least 100 rows per group and exactly the eight expected groups.
6. Stratified split 64 / 16 / 20 with seed 42: **1,270 / 318 / 398 unique descriptions**.
7. Validate: no shared source row or description between splits, every split contains
   all eight labels, no blank text, label ids in range. Any failure stops the pipeline.
8. Write `data_report.json` (hashes, audit counts, class counts, lengths, overlap).

### Class distribution (train / val / test)

| Group | Train | Val | Test |
| --- | --- | --- | --- |
| Cardiovascular & Pulmonary | 225 | 56 | 70 |
| ENT & Ophthalmology | 129 | 32 | 40 |
| Gastroenterology | 128 | 32 | 40 |
| General Medicine | 246 | 62 | 77 |
| Neurology | 147 | 37 | 46 |
| OB-GYN | 93 | 24 | 30 |
| Orthopedics | 212 | 53 | 67 |
| Urology | 90 | 22 | 28 |

### Row-level split (comparison only)

Splitting rows instead of unique descriptions gives 2,359 / 590 / 738 with 365
normalised descriptions shared between train and test. `prepare-data --keep-duplicates`
builds this split only for the leakage study in §8, which justifies deduplicating before
splitting.

## 7. Model Development

### Models

| Model | Purpose | Settings |
| --- | --- | --- |
| TF-IDF + logistic regression | Baseline | Word 1–3-grams, 30,000 features, min_df 2, max_df 0.4, sublinear TF, C = 1.0, balanced class weights |
| PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, revision `e1354b7a`) | Served model | 8 epochs, learning rate 1e-5, batch 16, warmup 10%, weight decay 0.01, early stopping patience 3, max 512 tokens, best checkpoint by validation macro F1 |

Only the `description` field is used as input. Specialty and keywords are used only to
build labels.

### Training environment and provenance

- Lab Linux server, one RTX 4090, bf16; Python 3.13.12, torch 2.14.0, transformers
  5.16.1, CUDA 13.0. Precision is chosen automatically (bf16 / fp16 / fp32).
- Every run writes `run_manifest.json`: Git commit (all reported runs: `4d8e669`, clean
  tree), configuration, dataset and taxonomy SHA-256, package versions, device, and the
  pinned base-model revision.
- Every run stores per-row logits for validation and test, keyed by source row and
  without text, so evaluation can be recomputed without the GPU.
- The best checkpoint was the last epoch (epoch 8), so training may not have fully
  converged; more epochs were not explored.

### Training-serving parity

- The training environment (Python 3.13, GPU, bf16) differs from serving (Python 3.10,
  CPU, fp32).
- Each export passes an offline CPU reload check against the trained logits.
- The served model was re-scored on the full test set on CPU and compared with the GPU
  evaluation using the serving calibration: top label 99.7%, top-3 order 97.7%, and
  uncertain flag 99.7% agree; the largest probability difference is 0.015. CPU test
  macro F1 is 0.886 versus 0.889 from GPU evaluation.

## 8. Evaluation

Full report: `reports/model_evaluation.md`. Test split of 398 descriptions never seen in
training; 95% percentile bootstrap intervals with 2,000 resamples.

**Scope of these results.** All numbers are an internal test: the same source and the
same labeling rules. Three training seeds measure optimiser randomness; six data splits
measure dependence on which descriptions land in the test set. Neither shows how the
model performs on other data. No external validation (another source,
clinician-curated cases, or lay-language text) has been done.

| Model | Test accuracy | Test macro F1 | Weighted F1 | Val macro F1 |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.744 [0.701, 0.786] | 0.742 [0.691, 0.785] | 0.745 | 0.768 |
| PubMedBERT seed 42 (served) | 0.884 [0.854, 0.915] | 0.889 [0.857, 0.919] | 0.883 | 0.874 |
| PubMedBERT seed 1 | 0.892 [0.862, 0.920] | 0.895 [0.862, 0.924] | 0.892 | 0.874 |
| PubMedBERT seed 2 | 0.889 [0.857, 0.920] | 0.892 [0.859, 0.921] | 0.888 | 0.870 |

- **Seed variation:** macro F1 0.892 ± 0.003 (mean ± SD, 3 training seeds on one split).
- **Split variation:** across six data splits (seeds 1–5 and 42), PubMedBERT macro F1 is
  **0.876 ± 0.012** (0.858–0.889) and the baseline 0.761 ± 0.022; accuracy on directly
  mapped rows is 0.839 ± 0.016 versus 0.700 ± 0.031. The served split (seed 42) is
  PubMedBERT's best of the six, so 0.876 is the more representative estimate.
  PubMedBERT beats the baseline on every split (mean +0.115, smallest +0.071)
  (`reports/split_variation.md`).
- **Versus baseline:** +0.148 macro F1 [+0.107, +0.192], one-sided paired bootstrap
  p < 0.001; exact McNemar test 69 vs. 13 discordant rows, p < 0.001.
- **Per class (served model):** weakest are General Medicine (F1 0.795), Neurology
  (0.809), and Gastroenterology (0.829, recall 0.725). Most confusions go to or from
  General Medicine.
- **Label provenance:** accuracy is 92.9% on keyword-rule rows (n = 156) and 85.5% on
  directly mapped rows (n = 242). For 87.2% of keyword-rule test rows, the description
  itself contains a keyword of its assigned label; rows containing a specific (weight 2)
  keyword score 97.8%, the rest 86.4%. Part of the headline score therefore reflects the
  model reproducing the labeling rules. **85.5% on directly mapped rows is the more
  conservative estimate of classification ability.**
- **Near duplicates:** using character 3–5-gram TF-IDF cosine to the nearest training
  description, 4.0% of test rows score ≥ 0.8. Accuracy is 100% on those and 88.0% on the
  rest, so near duplicates do not explain the result.
- **Leakage study (row-level split):** the same baseline and PubMedBERT settings were
  trained on the row-level split to measure what deduplication changes.

  | Model | All test | Seen in train, consistent label | Unseen | Conflicting label |
  | --- | --- | --- | --- | --- |
  | TF-IDF baseline | 70.6% | 99.2% | 67.2% | 32.3% |
  | PubMedBERT | 79.3% | 98.8% | 81.4% | 41.3% |

  On the row-level split, PubMedBERT's 0.80 macro F1 mixed memorised rows with rows that
  have no single correct label. Deduplication removed both effects; the higher score on the new split comes
  mainly from removing conflicting labels, so it reflects descriptions with consistent
  labels and should not be read as performance on ambiguous cases.

### Calibration and abstention

- Temperature scaling fitted on validation NLL: T = 0.95 (model was already close to
  calibrated). Test ECE 0.062 → 0.053; NLL 0.376 → 0.375.
- Abstain threshold chosen on validation as the lowest confidence reaching 90% accuracy
  on answered inputs: **0.65**. On test it answers 92.5% [89.7%, 95.0%] of inputs with
  91.6% [88.6%, 94.4%] accuracy.
- **The threshold is not stable.** Re-choosing it on 1,000 bootstrap resamples of the
  318 validation rows gives a 95% range of 0.00–0.71; at the low end the model would not
  abstain at all. The release gate therefore checks the resulting test behaviour, not
  the threshold value.
- **Decision (2026-10-14): keep the 90% target and the 0.65 threshold.** Raising the
  target would narrow the threshold range, but the alternatives were inspected on test
  data, so choosing one now would bias the reported test figures. A new target needs a
  product rationale and goes through change control
  ([validation-and-operations.md §6](validation-and-operations.md#6-change-control-and-retraining)).
- **Calibration is uneven across confidence bands.** Test rows scored 0.65–0.80 are
  correct only 42.9% of the time (n = 28) and rows scored 0.80–0.90 79.3% (n = 29),
  while rows at 0.95 or above are 98.7% correct (n = 223). Overall ECE hides this.
  Temperature scaling barely changes anything (T = 0.95; Brier 0.174 → 0.175).
- **Abstention is uneven across specialties.** Among answered test rows, accuracy is
  77.1% for Gastroenterology and 81.8% for General Medicine, below the 90% target, but
  above 96% for Cardiovascular & Pulmonary, ENT & Ophthalmology, OB-GYN, Orthopedics,
  and Urology.
- Both values are stored in `calibration.json` in the model bundle and read by the API.
- Scores are relative probabilities over a closed set of eight groups, not the
  probability that a suggestion is clinically correct.
- **`uncertain` does not detect out-of-scope input.** On 745 unique descriptions from
  the 16 excluded specialties, 90.5% received a confident label (68.7% scored ≥ 0.9) and
  only 9.5% were marked uncertain, close to the 7.5% rate on in-scope test data
  (`reports/release_validation.md`, "Excluded specialties"). Some of
  these labels are clinically plausible (Podiatry → Orthopedics), but the model cannot
  say that an input belongs to none of the eight groups. Vague input ("Follow-up
  visit.") behaves the same way.

## 9. System Architecture

```text
┌─────────────────────── training (lab GPU server) ───────────────────────┐
│ mtsamples.csv ─▶ prepare-data ─▶ baseline / finetune ─▶ report ─▶ model  │
│                                                                 bundle  │
└──────────────────────────────────────────────────────────────────┬──────┘
                       copy bundle, verify SHA-256                  │
                 ┌──────────────────────────────────────────────────┤
                 ▼                                                  ▼
┌───────────────── serving (local CPU) ─────────────────┐  private Hugging Face
│ Browser ─▶ FastAPI ─▶ asset validation ─▶ PubMedBERT  │  model repository
│                        ─▶ calibrated top-3 + uncertain│  (archive and registry
└───────────────────────────────────────────────────────┘   copy, not a demo)
```

| Layer | Module | Responsibility |
| --- | --- | --- |
| Data pipeline | `triage_training.data`, `taxonomy` | Cleaning, labeling, deduplication, splits, validation |
| Training | `triage_training.baseline`, `finetune` | Baseline and fine-tuning, export, reload check |
| Evaluation | `triage_training.stats`, `report` | Statistics, calibration, leakage and near-duplicate analysis |
| Configuration | `configs/train.yaml`, `triage_system.config` | Strict training config; serving reads `TRIAGE_MODEL_DIR` |
| Model asset validator | `triage_system.model_assets` | Checks files, label mapping, 512-token contract, calibration values; never reads weights |
| Model loader | `triage_system.model_loader` | Local files only, remote code disabled, safetensors, CPU, eval mode |
| Predictor | `triage_system.predictor` | Tokenise, infer, temperature scaling, top-3, abstention |
| API | `triage_system.api` | Routes, request validation, status codes, typed responses |
| Logging middleware | `triage_system.logging_config` | Method, path, status, duration; never request bodies |
| Monitoring | `triage_system.monitoring` | In-memory aggregate counters and histograms; no per-request records |
| Release gate | `triage_training.release` | Offline criteria, behavioural cases, CPU latency, pass/fail report |
| Drift windows | `triage_system.monitoring` | Prometheus metrics; in-memory windows evaluated by a separate thread against the reference profile; no per-specialty output |
| Browser client | `triage_system/web/index.html` | Input, examples, ranked results, uncertain message |
| Packaging | `Dockerfile`, `compose.yaml` | CPU image without weights; model mounted read-only |

### Runtime contract

- Input: English `description`, 1–1,000 characters, non-blank; request body ≤ 16 KB
- Model sequence length: 512 tokens; accepted inputs are far shorter, so nothing is
  truncated (training maximum 433 characters)
- Inference: CPU, fp32, single process, model cached after first load
- Output: top label, calibrated score, `uncertain` flag, top three

The service starts with `python -m triage_system`, locally and in Docker, which validates
`TRIAGE_MODEL_DIR`, `TRIAGE_HOST`, and `TRIAGE_PORT` before starting uvicorn. Invalid
values stop startup with a clear message.

## 10. API Contract

| Route | Purpose | Responses |
| --- | --- | --- |
| `GET /` | Browser client (same origin) | 200 |
| `GET /health` | Process liveness; does not load the model | 200 `{"status": "ok"}` |
| `GET /ready` | Model can be loaded and serve | 200 `{"status": "ready"}`; 503 with reason |
| `POST /predict` | Classify one description | 200; 422 blank or over 1,000 characters; 413 body over 16 KB; 411 no `Content-Length`; 503 model unavailable; 500 prediction failure |
| `GET /version` | Service version, served weights SHA-256, training commit, calibration | 200; 503 model unavailable |
| `GET /metrics` | Prometheus text: requests, latency, model state and identity, drift result of the last completed window | 200 |
| `GET /docs` | OpenAPI documentation (FastAPI default) | 200 |

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

When `uncertain` is true, `label` still contains the highest-scoring specialty; clients
are expected to show it as a suggestion for manual review, as the browser client does.

## 11. Model Bundle and Registry

| File | Purpose |
| --- | --- |
| `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json` | Model and tokenizer |
| `training_config.json` | Serving contract (input type, 512 tokens, labels) and hyperparameters |
| `calibration.json` | Temperature and abstain threshold (optional; defaults to uncalibrated, no abstention) |
| `monitoring_reference.json` | Test-set prediction profile in the drift window buckets |
| `metrics.json`, `data_report.json`, `run_manifest.json`, `reload_check.json` | Evaluation, data audit, provenance, integrity |

- The bundle is produced on the training server and copied to the serving machine; the
  service loads it from a local directory (`TRIAGE_MODEL_DIR`) and never downloads at
  runtime.
- A copy with a model card is archived in the private Hugging Face model repository
  `show20130831/medical-specialty-pubmedbert`. It serves as the registry of record and
  the access path for anyone granted permission; it is not used for hosting a demo.
- The weight file SHA-256 (`a2f4ad5c…`) matches on the training server, the serving
  machine, and the archived copy.
- Weights are private because the training data has no stated license.
- `GET /version` reports the SHA-256 of the served weights, computed at load time. Each
  archive upload is a revision; rollback restores a previous revision
  ([runbook](validation-and-operations.md#5-response-runbook)). There are no signed
  release tags yet.

## 12. Privacy, Security, and Data Governance

Implemented:

- Request bodies are never logged or persisted; logs contain method, path, status, and
  duration only.
- Raw data, splits, and model weights are excluded from Git and the Docker build context.
- Committed reports and prediction files contain aggregate numbers or row ids, never text.
- The Docker Compose model mount is read-only; model loading uses local files only with
  remote code disabled.
- Demonstrations use synthetic descriptions only.
- Monitoring exports no per-specialty values: drift windows are held in memory, reduced
  to PSI values, an uncertain rate, and a status, then discarded. Prometheus keeps
  operational metrics and drift results for 15 days.
- Request bodies over 16 KB are rejected before parsing; the container runs as a
  non-root user with a health check.

Known gaps:

- No authentication, authorisation, or rate limiting.
- Validation error responses (422) echo the submitted input back to the caller, as
  FastAPI does by default. It is not logged, but a production service would strip it.
- No dependency vulnerability scanning.
- A production deployment would also need a retention policy, encryption in transit,
  audit logging, and a formal privacy and regulatory review.

## 13. Deployment

| Option | Status | Reason |
| --- | --- | --- |
| Local Python (uvicorn) | Done | Primary development and demo path |
| Docker / Docker Compose (CPU) | Done | Reproducible runtime; non-root user, health check, model mounted read-only |
| Public hosted demo | Not planned | The data license is unspecified, and the free Hugging Face Spaces tier no longer runs Docker apps (paid plan required) |
| Other free hosting | Not done | Free tiers typically offer 512 MB memory; PyTorch plus the 438 MB model needs an estimated 1.5–2 GB |
| Demo media in README | Done | GIF and screenshots captured from the served model |

Given access to the dataset, the training pipeline reproduces the model in a few minutes
on one GPU (`docs/training.md`). Users are responsible for complying with the data
source's terms; because no license is stated, neither the data nor the weights are
redistributed.

## 14. Development Workflow and Quality Assurance

- Work happens on feature branches and lands on `main` through pull requests (earlier
  ones squash-merged, recent ones merged with merge commits).
- CI (GitHub Actions) runs the full test suite on Ubuntu and Windows with Python 3.10 and
  checks dependency compatibility; serves a tiny randomly initialised BERT bundle through
  every route; and builds the Docker image, checking the non-root user, `/health`, and
  `/ready`. CI does not need the dataset, a GPU, or the private model.
- 112 tests cover API behaviour and status codes, input and body limits, model asset and
  calibration validation, offline loading contracts, prediction ranking and abstention,
  configuration and startup, privacy-safe logging and metrics, data pipeline invariants,
  baseline, statistics (checked against scikit-learn and known values), report
  generation, split variation, release gate, Prometheus metrics, and drift windows
  (concurrency, failure isolation, timeouts).
- Not yet in CI: the release gate against the real model (it needs the private bundle),
  linting.
- A repository ruleset protects `main`: changes must arrive through a pull request,
  both CI checks must pass on a branch that is up to date with `main`, and force pushes
  and branch deletion are blocked.

## 15. Delivery Status

Done (merged):

- Serving foundation: FastAPI, browser client, asset validation, CPU loading, logging,
  configuration, Docker, CI (#1–#15)
- Training pipeline: config, taxonomy, cleaning, deduplication, validation, splits,
  baseline, fine-tuning CLI with provenance (#17)
- Evaluation tooling: `evaluate-model`, `report`, statistics, calibration (#18)
- Evaluation report and README results (#19)
- Calibrated top-3 predictions, abstention, corrected status codes, UI update (#20)
- Demo GIF and screenshots, hosting explanation (#21)

Done outside Git:

- Retraining on the RTX 4090 (baseline, 3 seeds, row-level split comparison)
- Model bundle archived in a private Hugging Face model repository with a model card

In review:

- Product design document and results wording (#22)
- Validation and operations MVP: release gate, `/version`, metrics, drift check,
  runbook, input and body limits, startup validation, non-root Docker, CI jobs

Not done: see §16 and §17.

## 16. Known Limitations and Risks

| Risk | Impact | Current mitigation |
| --- | --- | --- |
| Labels are partly rule-generated and never clinician-reviewed | Metrics measure agreement with the labeling scheme, not clinical correctness | Label provenance reported separately |
| Keyword-rule labels often share terms with the input | Headline score partly reflects reproducing the rules | Direct-mapped accuracy (85.5%) reported as the conservative estimate |
| Taxonomy is a simplification; 16 specialties excluded | Inputs from excluded specialties are forced into one of eight groups | Documented; no mitigation in product |
| No out-of-distribution detection | 90.5% of excluded-specialty inputs get a confident label | Measured and documented; no mitigation in product |
| Training text is clinician-written summaries | Lay symptom descriptions may perform worse; not measured | Input assumption documented |
| Ambiguous descriptions removed from test | Reported 0.89 does not cover inherently ambiguous cases | Abstention; leakage study shows 41% on conflicting cases |
| Test set of 398; internal test only | Wide intervals (±0.03); the served split is the most favourable of six; no evidence of performance on other data | Bootstrap intervals, 3 training seeds, 6 data splits with the mean reported; external validation listed as not done |
| Abstain threshold chosen on 318 validation rows | Threshold value is unstable; mid-confidence answers are often wrong | Stability and per-band analysis reported; gate checks test behaviour |
| Row-level bootstrap assumes independent test rows | Intervals may be slightly narrow if templated descriptions cluster | Near-duplicate check (median similarity 0.39, 4.0% ≥ 0.8) suggests the effect is small |
| Training may not have converged (best epoch = last) | Possible headroom left | Not explored |
| Data license unspecified | Weights cannot be shared openly | Private repository; no public hosting |
| Monitoring has no ground-truth labels | Accuracy loss can go unnoticed; drift only signals change | Aggregate drift checks; out-of-scope traffic raised only WARN in the demonstration |

## 17. Roadmap

### Validation and operations (MVP done)

Described in [validation-and-operations.md](validation-and-operations.md): release gate
with criteria in configuration, behavioural cases and latency; `/version` and evaluation
linkage; Prometheus metrics, reference profile, and in-process drift windows;
runbook, rollback, and change control for retraining.

### Later

- Release gate in CI using a read-only token for the private model; publish images
- Scheduled drift checks with notifications; dashboards
- Reviewed label sample from real use to measure accuracy, not only drift
- Authentication, rate limiting, dependency scanning, load testing
- Evaluation on lay-language and out-of-scope inputs; out-of-distribution detection
- Clinician review of a label sample; taxonomy revision

## 18. Success Criteria

A reviewer can:

1. Understand the purpose, results, and limitations from the README in under two minutes.
2. See the demo working through the GIF without running anything.
3. Trace any reported number to a commit, configuration, and dataset hash.
4. Reproduce the data pipeline and baseline on a CPU and the model on one GPU.
5. Run the service locally with a model bundle and get calibrated top-3 predictions.
6. Inspect the API contract, tests, and CI results.

The model meets the current bar when test macro F1 is at least 0.85 (the lowest of six
splits is 0.858), the 95% paired
bootstrap interval for its macro F1 difference from the baseline lies above zero, every class has F1 of at least 0.75, and abstention
keeps answered accuracy at or above 90%. The served model meets all three.

## 19. Open Questions for Review

1. Is deduplicating before splitting, and resolving conflicts by preferring direct
   mappings, the right treatment, or should conflicting descriptions be kept as a
   separate "ambiguous" evaluation set?
2. Should excluded specialties (especially the 516 consultation notes) be handled, for
   example as an "Other" class, instead of being dropped?
3. The abstain threshold is unstable on 318 validation rows (95% resample range
   0.00–0.71). Should the target answered accuracy be raised (93% gives 0.64–0.83), or
   should the threshold be chosen with cross-validation or a larger validation split?
4. Is a 90% answered-accuracy target appropriate for a routing aid, and how should
   uncertain cases be handled in a real workflow?
5. Without ground-truth labels, which additional signal would detect out-of-scope
   traffic better than label and confidence drift?
