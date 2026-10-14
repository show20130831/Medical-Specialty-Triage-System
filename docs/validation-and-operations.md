# Validation and Operations

**Last updated:** 2026-10-14
**Parent document:** [Product Design and Delivery Plan](product-design-and-plan.md)

How a model bundle is validated before release, identified once deployed, monitored in
use, and changed or rolled back. The structure borrows ideas from Good Machine Learning
Practice (deployed models are monitored; testing reflects intended conditions) and from
predetermined change control plans (retraining conditions and acceptance criteria are
fixed in advance). It does not claim compliance with any regulation or standard.

## 1. Lifecycle

```text
 train ──▶ evaluate ──▶ ① release gate ──▶ ② deploy + identify ──▶ ③ monitor ──▶ ④ respond
             (report)     validate-model       /version, hashes      /metrics,        runbook,
                                                                     monitor-drift    rollback
   ▲                                                                                   │
   └──────────── retraining under §6 must pass the same release gate ◀─────────────────┘
```

## 2. Release Validation

`python -m triage_training validate-model --model-dir <bundle>` loads the bundle on CPU,
checks it against `configs/release_criteria.yaml`, writes
`reports/release_validation.md`, and exits non-zero on failure. The current bundle
passes; see [the release report](../reports/release_validation.md).

### Criteria

| Check | Requirement | Current | Why |
| --- | --- | --- | --- |
| Evaluation report belongs to this bundle | Same training run and commit | Yes | Prevents approving a bundle with another model's numbers |
| Test macro F1 | ≥ 0.85 | 0.889 | Margin below current performance |
| Macro F1 gain over baseline | 95% paired bootstrap CI lower bound > 0 | 0.107 | The model must beat a simple baseline, not just look higher |
| Lowest per-class F1 | ≥ 0.75 | 0.795 | A good average must not hide a failing specialty |
| Calibrated ECE | ≤ 0.08 | 0.053 | Scores shown to users must mean roughly what they say |
| Accuracy on answered inputs | ≥ 0.90 | 0.916 | The abstention threshold must deliver its promise |
| Share of inputs answered | ≥ 0.85 | 0.925 | Abstaining on everything is not a solution |
| Expected labels, 24 clear synthetic cases | 100% | 100% | Basic competence on unambiguous descriptions |
| Invariance to case, punctuation, spacing, 9 variants | 100% | 100% | Surface edits must not change the answer |
| Abstention on ambiguous descriptions, 5 cases | ≥ 80% | 80% | Ambiguity must surface as `uncertain` |
| Single-request CPU latency p95 | ≤ 500 ms | 92 ms | Interactive use on CPU |
| Out-of-scope inputs answered confidently | Reported only | 5 of 5 | Known gap; gating on it would block every bundle |

### Limits of the gate

- Offline metrics come from GPU evaluation. The served CPU model was checked separately
  (397 of 398 test predictions identical, macro F1 0.886), which also passes.
- Four of the five abstention cases were seen while exploring the model before the gate
  was written, so the 80% abstention rate is optimistic. The other behavioural cases were
  written before running the model.
- The behavioural set is small and synthetic. It catches regressions, not new failure
  modes.
- The gate runs where the private bundle is available. CI cannot run it; CI instead
  serves a tiny randomly initialised bundle to test the service contract (§7).

## 3. Identification and Traceability

| Question | Answered by |
| --- | --- |
| Which weights are serving? | `GET /version` → `weights_sha256`, computed when the model loads |
| Which code and data produced them? | `run_manifest.json`: training commit, configuration, dataset and taxonomy SHA-256 |
| Which evaluation approved them? | `model_run` in `reports/model_evaluation.json`, checked by the release gate |
| Which calibration is active? | `GET /version` → temperature and abstain threshold |
| Which model produced a metrics window? | `model_weights_sha256` in every `/metrics` snapshot |

Bundles are archived in the private Hugging Face model repository; each upload is a
revision that can be restored for rollback (§5).

## 4. Monitoring

### Design principles

- **Aggregate only.** The service keeps in-memory counters and fixed-bucket histograms.
  There are no per-request records, even without text, because label, score, length, and
  time together could still describe an individual request.
- **Short retention.** Counters reset when the process restarts. Operators save
  snapshots for drift checks; saved snapshots should be kept no longer than 90 days and
  only by people who operate the service.
- **Signals, not verdicts.** No ground-truth labels arrive after a prediction, so these
  checks show that inputs or outputs changed. They cannot show that accuracy dropped.
  Measuring accuracy in use needs reviewed labels (§8).

### Signals

| Signal | Source | Threshold (WARN / ALERT) | What it can indicate | Limitation |
| --- | --- | --- | --- | --- |
| Predicted label distribution | `/metrics` vs reference, PSI | 0.10 / 0.25 | Different case mix or out-of-scope input | A real change in case mix looks the same |
| Confidence distribution | PSI | 0.10 / 0.25 | Inputs less like training data | Confident errors stay invisible |
| Input length distribution | PSI | 0.10 / 0.25 | New document types or misuse | Weak signal for short inputs |
| Uncertain rate | rate | 15% / 25% (reference 7.5%) | Ambiguous or unfamiliar inputs | Out-of-scope input barely moves it (below) |
| Server error rate | share of 5xx on `/predict` | 1% / 5% | Service or model failure | — |
| Latency | `/metrics` histogram | Read manually; not checked by `monitor-drift` yet | Capacity problems | Histogram resolution only |

The reference (`monitoring_reference.json` in the bundle) is the calibrated test-set
prediction profile in the same buckets. Windows with fewer than 100 predictions are
reported as insufficient data. Thresholds live in `configs/monitoring.yaml`.

### Running a check

```bash
curl -s http://127.0.0.1:8000/metrics > snapshot.json
python -m triage_training monitor-drift --snapshot snapshot.json \
    --reference models/pubmedbert_description/monitoring_reference.json
```

Exit code 0 means OK or insufficient data, 1 WARN, 2 ALERT, so a scheduler can raise a
notification without parsing the report.

### Demonstration

`scripts/simulate_traffic.py` replays descriptions against a running service.

| Scenario | Label PSI | Confidence PSI | Uncertain rate | Status |
| --- | --- | --- | --- | --- |
| [Test split replay](../reports/drift_demo/in-distribution/drift_check.md) (398) | 0.000 | 0.000 | 7.3% | OK |
| [Excluded specialties](../reports/drift_demo/excluded-specialties/drift_check.md) (400) | 0.197 | 0.101 | 12.0% | WARN |

The replay reproduces the reference exactly, which checks the monitoring pipeline
itself. Traffic from specialties the model never learned raises only a warning: the
model assigns such input confidently to the existing groups, so the uncertain rate
barely moves. Thresholds were not tuned to make this demonstration alert. Out-of-scope
input is a documented gap that monitoring detects weakly and the model does not detect
at all.

## 5. Response Runbook

| Event | First action | Then |
| --- | --- | --- |
| `/health` fails | Restart the container; check logs | If repeated, roll back the image |
| `/ready` returns 503 | Read the reason; check `TRIAGE_MODEL_DIR`, files, and memory | Restore the bundle from the archive |
| Server error rate WARN or ALERT | Check logs for the failing step | Roll back image or bundle if errors began with a release |
| Drift WARN | Compare the window with recent changes in who uses the service | Review a sample of inputs with the data owner, if permitted |
| Drift ALERT | Consider pausing use for affected inputs | Decide between rollback, retraining (§6), or narrowing the documented scope |
| Latency over 500 ms p95 | Check CPU load and concurrent requests | Add capacity; the model runs one process on CPU |
| Release gate fails | Do not deploy | Fix the cause; never relax criteria to make a bundle pass |

### Rollback

1. Identify the previous bundle revision in the archive (or the previous local bundle).
2. `hf download show20130831/medical-specialty-pubmedbert --revision <rev> --local-dir <dir>`
3. Point `TRIAGE_MODEL_DIR` at it and restart the service.
4. Confirm `GET /version` reports the expected `weights_sha256`.
5. Re-run `validate-model` on the restored bundle and record the incident.

## 6. Change Control and Retraining

| Change | Allowed with | Required evidence |
| --- | --- | --- |
| Retrain with the same pipeline, configuration, and taxonomy on refreshed data | Release gate only | Data report, evaluation report, release report, all for the new run |
| New seed or hyperparameters within the training configuration | Release gate only | As above, plus the configuration diff |
| Change to calibration or abstention target | Gate plus a written rationale | Recomputed calibration on validation data |
| Change to taxonomy, labels, input type, or model architecture | Full re-review | Updated product design document and risk table, new evaluation |
| Relaxing any release criterion | Not allowed without full re-review | Rationale in the product design document |

Retraining triggers: a drift ALERT that persists for two consecutive windows, a
confirmed change in intended use, or a new labeled data source. A trigger starts an
investigation; it does not automatically retrain or deploy.

## 7. Continuous Integration

| Job | Checks | Needs the private model |
| --- | --- | --- |
| API tests (Ubuntu, Windows) | Unit and contract tests, data pipeline, statistics, gate and drift logic | No |
| Service smoke test | Builds a tiny random BERT bundle and serves it through every route | No |
| Docker image build | Builds the image, checks it runs as a non-root user, `/health` 200 and `/ready` 503 without a model | No |

Main is protected by a repository ruleset: pull requests only, required checks on an
up-to-date branch, no force pushes.

## 8. MVP Scope and Later Work

| Area | In the MVP | Later |
| --- | --- | --- |
| Release validation | Criteria file, behavioural set, latency, gate report | Gate in CI using a read-only token for the private bundle; larger behavioural set written blind |
| Identification | `/version`, weights hash, evaluation linkage | Signed release tags in the archive |
| Monitoring | Aggregate `/metrics`, reference profile, drift check, demonstration | Scheduled checks with notifications; dashboards |
| Accuracy in use | — | Reviewed label sample from real use to measure accuracy, not just drift |
| Out-of-scope input | Measured and documented | An out-of-distribution detector or an "other" class, validated on held-out specialties |
| Response | Runbook, rollback procedure, change control | Canary or shadow deployment before full rollout |
| Service hardening | Non-root image, health check, request size and length limits | Authentication, rate limiting, dependency scanning, load testing |
