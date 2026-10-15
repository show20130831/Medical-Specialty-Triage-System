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
                                                                     drift windows    rollback
   ▲                                                                                   │
   └──────────── retraining under §6 must pass the same release gate ◀─────────────────┘
```

## 2. Release Validation

`python -m triage_training validate-model --model-dir <bundle> [--out-of-taxonomy]` loads
the bundle on CPU, checks it against `configs/release_criteria.yaml`, writes
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
| Single-request CPU latency p95 | ≤ 500 ms | under 100 ms | Interactive use on CPU |
| Out-of-scope inputs answered confidently | Reported only | 5 of 5 | Known gap; gating on it would block every bundle |
| Excluded-specialty descriptions answered confidently (`--out-of-taxonomy`) | Reported only | 90.5% of 745 | Same gap measured on real text |

### Limits of the gate

- Offline metrics come from GPU evaluation. The served CPU model was checked separately
  (top label 99.7%, top-3 order 97.7%, uncertain flag 99.7% agreement; macro F1 0.886),
  which also passes.
- The abstention criteria check test behaviour at the chosen threshold, not the
  threshold itself, because the threshold is unstable on 318 validation rows (95%
  resample range 0.00–0.71; see the evaluation report).
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
| Which model is behind the metrics? | `triage_model_info` metric with the weights hash and training commit |

Bundles are archived in the private Hugging Face model repository; each upload is a
revision that can be restored for rollback (§5).

## 4. Monitoring

The design is detailed in the [operations plan](operations-plan.md); this section
summarises what the service does.

### Design principles

- **No per-specialty output.** The service never exports per-specialty counts or shares,
  even aggregated, because counts read at two moments can reveal the label of a single
  request. Model behaviour is exported only as the drift result of the last completed
  window: PSI values, the uncertain rate, and a status.
- **Windows in memory, then discarded.** Each prediction adds to an in-memory window of
  aggregate counts. When the window holds 400 predictions, a dedicated evaluator thread
  compares it with the bundle's reference and the counts are dropped. Monitoring can
  neither slow nor fail a prediction.
- **Honest states.** The drift status is "not evaluated" after every start, "insufficient
  data" if a window does not fill within 24 hours, and otherwise OK, WARN, or ALERT.
- **Signals, not verdicts.** No ground-truth labels arrive after a prediction, so these
  checks show that inputs or outputs changed. They cannot show that accuracy dropped.
  Measuring accuracy in use needs reviewed labels (§8).

### Signals

| Signal | Source | Threshold (WARN / ALERT) | What it can indicate | Limitation |
| --- | --- | --- | --- | --- |
| Predicted label distribution | Window PSI against the reference | 0.10 / 0.25 | Different case mix or out-of-scope input | A real change in case mix looks the same |
| Confidence distribution | Window PSI | 0.10 / 0.25 | Inputs less like training data | Confident errors stay invisible |
| Input length distribution | Window PSI | 0.10 / 0.25 | New document types or misuse | Weak signal for short inputs |
| Uncertain rate | Window rate | 15% / 25% (reference 7.5%) | Ambiguous or unfamiliar inputs | Out-of-scope input barely moves it |
| Server errors, latency, model state | Prometheus metrics at `/metrics` | Alert rules in the operations plan §7 | Service or capacity problems | — |

The reference (`monitoring_reference.json` in the bundle) is the calibrated test-set
prediction profile in the same buckets. Thresholds live in `configs/monitoring.yaml`.

### Window size

At 100 predictions, 64% of windows drawn from the validation split, which has no drift,
reached WARN, almost all from label PSI. A 400-prediction window brings this to 4.6%
([estimate](../reports/drift_window_estimate.md)). Detection is therefore slower and
low traffic more often ends in "insufficient data", a trade-off accepted to avoid
constant false alarms.

### Reading the current state

```bash
curl -s http://127.0.0.1:8000/metrics | grep '^triage_drift'
```

Drift drills replace the earlier manual demonstration; their records are in
`docs/operations/drills/`.

## 5. Response Runbook

Alert names refer to the rules in `deploy/monitoring/alerts.yml`.

| Event (alert) | First action | Then |
| --- | --- | --- |
| `/health` fails (`TriageApiDown`) | Restart the container; check logs | If repeated, roll back the image |
| `/ready` returns 503 (`TriageModelNotLoaded`) | Read the reason in the log; check `TRIAGE_MODEL_DIR`, files, and memory | Restore the bundle from the archive |
| Server error rate (`TriageServerErrorsWarning`, `TriageServerErrorsCritical`) | Check logs for the failing step | Roll back image or bundle if errors began with a release |
| Drift WARN (`TriageDriftWarning`) | Read which checks failed in the alert; compare with recent changes in who uses the service | Review a sample of inputs with the data owner, if permitted |
| Drift ALERT (`TriageDriftAlert`) | Consider pausing use for affected inputs | Decide between rollback, retraining (§6), or narrowing the documented scope |
| Latency over 500 ms p95 (`TriageHighLatency`) | Check CPU load and concurrent requests | Add capacity; the model runs one process on CPU |
| Drift evaluator stale (`TriageDriftEvaluatorStale`) | Check the log for evaluator errors; restart the API | Drift status returns to "not evaluated"; confirm the heartbeat moves again |
| Drift evaluation failed (`TriageDriftEvaluationFailed`) | Read the logged exception (it contains no window contents) | Fix the cause; dropped windows are lost and are not re-evaluated |
| Monitoring stack down (Prometheus, Alertmanager, or Grafana unreachable) | `docker compose --profile monitoring ps`; restart the failed service | Alerts may have been missed meanwhile; check the dashboard history |
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
| Monitoring | Prometheus `/metrics`, in-process drift windows with a validated size, reference profile | Dashboards, alert rules, and drills: [operations plan](operations-plan.md) |
| Accuracy in use | — | Reviewed label sample from real use to measure accuracy, not just drift |
| Out-of-scope input | Measured and documented | An out-of-distribution detector or an "other" class, validated on held-out specialties |
| Response | Runbook, rollback procedure, change control | Canary or shadow deployment before full rollout |
| Service hardening | Non-root image, health check, request size and length limits | Authentication, rate limiting, dependency scanning, load testing |
