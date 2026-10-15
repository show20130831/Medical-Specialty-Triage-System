# Operations Plan

**Last updated:** 2026-10-15
**Status:** Approved after three external reviews. M1–M5, M7, and M8 are implemented and
drilled (2026-10-15); M6, the release and rollback drill, waits for the candidate model.
See §13 for the status of each item. Until it is, [Validation and Operations](validation-and-operations.md)
and the README describe the system as it runs today (JSON `/metrics`, manual
`monitor-drift`); the implementing pull request updates them together.
**Related:** [Product Design and Delivery Plan](product-design-and-plan.md) ·
[Validation and Operations](validation-and-operations.md)

## 1. Purpose and Scope

[Validation and Operations](validation-and-operations.md) defines what to measure and
what to do: a release gate, `/version`, an aggregate `/metrics` endpoint, a drift check,
a runbook, and change control. All of it runs by hand. Nothing collects metrics over
time, nothing raises an alert, and the rollback procedure has never been performed.

This plan makes that design run and tests it through drills:

- continuous metric collection, a dashboard, and alert rules with a notification path;
- drift checks computed inside the service over completed windows;
- a release of a second model version and a rehearsed rollback;
- measured capacity under load;
- dependency scanning.

**Scope limits.** The system is a demonstration prototype with no real users and no
hosted deployment (product design document, §13). Everything runs on one machine with
Docker Compose. Traffic is synthetic descriptions written for this project, or replayed
descriptions from the public MTSamples dataset; no patient data is involved. The
objectives in §6 are drill targets, not commitments to users. A passing drill shows
that detection and response work as designed; it does not show that the system would
meet these targets under real use.

## 2. Current State and Gaps

| Area | Today | Gap this plan closes |
| --- | --- | --- |
| Metrics | `/metrics` returns cumulative JSON counters since process start, including counts per specialty | Not collected over time; no rates or percentiles; per-specialty counts readable by anyone who can reach the API |
| Dashboards | None | No view of traffic, errors, latency, or model behaviour |
| Alerting | Runbook lists events and actions | No rule fires; no notification path |
| Drift | `monitor-drift` run by hand on a saved snapshot | Not scheduled; cumulative counts let old traffic dilute new drift |
| Readiness | Model loads on the first `/ready` or `/predict` | The first user waits for loading; no signal that the model is loaded |
| Capacity | Single-request CPU latency only (p95 under 100 ms) | Throughput and latency under concurrent load unknown |
| Rollback | Documented, never performed; the model archive holds one model version | Unproven procedure; nothing to roll back to |
| Exposure | Compose publishes port 8000 on all interfaces | API reachable from the local network |
| Supply chain | Dependencies pinned; no scanning | Known vulnerabilities would go unnoticed |

## 3. Operating Model and Assumptions

- **Topology:** one host, Docker Compose, one API container running one Uvicorn process
  with model preloading enabled. In-process counters and windows are therefore complete.
  Several processes or replicas would need shared aggregation (§14).
- **Traffic:** synthetic descriptions from `tests/golden/behavioural_cases.yaml` for load
  tests; replayed MTSamples descriptions (test split, excluded specialties) for drift
  drills via `scripts/simulate_traffic.py`. Replayed text stays on the machine.
- **People:** one operator, who is also the developer. Notifications go to a local
  receiver whose log is the evidence of delivery.
- **Evidence:** configuration, alert rules, the dashboard definition, and drill records
  are committed, so results can be reproduced; screenshots illustrate them but are not
  the evidence.

## 4. Target Architecture

```text
                                ┌──────────────────────────────────────────────────┐
 browser / k6 load test ──────▶ │ api  (FastAPI, one process, model preloaded)     │
                                │   /predict /health /ready /version               │
                                │   /metrics  Prometheus text:                     │
                                │     operational counters and latency histogram   │
                                │     drift results of the last completed window   │
                                │   prediction path: increments window counters    │
                                │   evaluator thread: drift check, heartbeat       │
                                └───────────────────────┬──────────────────────────┘
                                                        │ scrape every 15 s
┌──────────────┐   alerts   ┌───────────────────────┐   │
│ Alertmanager │ ◀───────── │ Prometheus            │ ◀─┘
│              │            │ rules, 15-day storage │
└──────┬───────┘            └───────────┬───────────┘
       │ webhook                        │ query
       ▼                                ▼
┌──────────────┐            ┌───────────────────────┐
│ alert-sink   │            │ Grafana               │
│ logs alerts  │            │ provisioned dashboard │
└──────────────┘            └───────────────────────┘
```

- The monitoring services sit behind a Compose profile (`docker compose --profile
  monitoring up`); plain `docker compose up` still starts only the API.
- Every published port binds to `127.0.0.1`.
- There is no separate drift service. The API evaluates its own windows in a dedicated
  thread, isolated from the prediction path and the event loop (§8), so no interface
  ever exposes per-specialty counts (§5).

## 5. Telemetry Design and Privacy

Even without text, a time-stamped label count can describe a single request: if a
counter is read twice and one prediction happened in between, the difference reveals
its label. The design therefore removes every interface that exposes open-window or
cumulative per-specialty counts.

| Category | Contents | Exposed as | Granularity |
| --- | --- | --- | --- |
| **Operational** | Requests by route and status class, latency histogram, in-flight requests, model loaded flag, model identity, process metrics | `/metrics`, scraped by Prometheus | Per scrape (15 s) |
| **Model behaviour** | PSI for label, confidence, and length distributions; uncertain rate; drift status | `/metrics`, values of the **last completed window** only | Changes only when a window completes |

- **Breaking change:** the JSON `/metrics` endpoint is removed and replaced by
  Prometheus text at the same path. Its cumulative per-specialty counts were the main
  exposure. README, the validation document, and `scripts/simulate_traffic.py` are
  updated with it. The manual `monitor-drift` command is retired in favour of the
  in-service check, and the `reports/drift_demo/` reports it produced are replaced by
  the records of drills D0 and D2.
- **Windows:** the API adds each prediction to an open window held in memory. A full
  window is handed to a background evaluator, which computes and publishes the drift
  result and then discards the counts (§8). Open-window counts are never exposed.
- **No per-specialty output.** Even a share per specialty, combined with the window
  size, gives approximate counts, and anyone who knows the other predictions in a window
  can infer the remaining one. A small-cell threshold would reduce but not remove this,
  and alerting does not need per-specialty values, so none are published. The
  dashboard shows the label PSI instead.
- **In memory, per window:** counts per specialty, confidence buckets, length buckets,
  the number of uncertain predictions, and the window start time. No text and no
  per-prediction record. Counts are discarded once the window's result is computed.
- **Label values** in metrics are limited to route names, status classes, drift check
  names, drift states, and the model hash and training commit, so no metric can carry
  text or a specialty.
- **What operational metrics still reveal:** at low traffic, request counts per scrape
  show when a request arrived and whether it succeeded, but nothing about its content
  or result.
- **Retention:** Prometheus keeps 15 days; Grafana stores no metric data. Drift results
  saved by drills are kept 90 days, matching the validation document.
- **Access:** binding to localhost keeps the network out but not other users or
  processes on the same host. Removing sensitive counts, not access control, is the
  main protection. Authentication is in §14.
- **Never collected:** request text, client addresses, per-request records.
- **If real clinical text were ever processed,** a separate review would set minimum
  window sizes, suppression thresholds, access control, and retention before go-live.

## 6. Service Level Objectives

| Objective | Indicator (PromQL sketch) | Drill target |
| --- | --- | --- |
| Readiness after start | `triage_model_load_duration_seconds`: process start to model loaded | ≤ 60 s |
| Latency | `histogram_quantile(0.95, rate(triage_request_duration_seconds_bucket{route="/predict"}[5m]))` | ≤ 500 ms at the capacity found in §9 |
| Errors | 5xx share of `/predict` over 15 min | < 1% |

**Measured ([reports/load_test.md](../reports/load_test.md)):** on the development laptop
the latency target holds reliably for one concurrent client (p95 267–464 ms across
runs, no errors). Two clients met it in three of five runs (p95 439–863 ms), so they sit
on the threshold. Throughput saturated near 11–13 requests per second, and latency, not
errors, failed first. Results varied between runs on the same machine, so they describe
this laptop only. Model load including a warm-up inference took 27.9 s, within the
readiness target.

**Monitoring health (indicator, not an objective):** scrape success,
`avg_over_time(up{job="triage-api"}[5m])`. With a 15-second interval there are 20
scrapes in 5 minutes; at most one missed scrape (≥ 0.95) is treated as healthy. This
measures the monitoring path, not whether users can reach the service, so it is not
called availability. A user-facing availability objective is not meaningful on a single
host without real traffic and is left out.

## 7. Alerting

Each rule links to the runbook in
[validation-and-operations.md §5](validation-and-operations.md#5-response-runbook).
Detection times assume a 15-second scrape and evaluation interval.

| Alert | Condition | For | Severity | Expected detection |
| --- | --- | --- | --- | --- |
| `TriageApiDown` | `up == 0` | 1 min | critical | ≤ 2 min |
| `TriageModelNotLoaded` | `triage_model_loaded == 0` | 2 min | critical | ≤ 3 min |
| `TriageServerErrorsWarning` | 5xx share of `/predict` > 1% over 15 min, at least 100 requests | 5 min | warning | ≤ 21 min after errors begin |
| `TriageServerErrorsCritical` | 5xx share of `/predict` > 5% over 5 min, at least 20 requests | 5 min | critical | ≤ 11 min after errors begin |
| `TriageHighLatency` | p95 `/predict` latency > 500 ms over 5 min, at least 50 requests | 5 min | warning | ≤ 11 min after overload begins |
| `TriageDriftWarning` | current drift status = WARN | — | warning | ≤ 1 min after the window completes |
| `TriageDriftAlert` | current drift status = ALERT | — | critical | ≤ 1 min after the window completes |
| `TriageDriftEvaluatorStale` | drift heartbeat older than 3 heartbeat intervals | 2 min | warning | ≤ 3 intervals + 3 min |
| `TriageDriftEvaluationFailed` | increase in failed drift evaluations or dropped windows over 15 min | — | warning | ≤ 1 min after the failed run |

- Rate-based rules require minimum request counts, so a handful of requests at low
  traffic cannot produce an error-rate or p95 alert.
- Drift alerts carry the failing checks (label, confidence, or length PSI, or the
  uncertain rate) in their annotations.
- **Heartbeat versus data:** the heartbeat is updated only after the evaluator completes
  a run successfully, whether that run evaluated a window, found no window ready, or
  closed a window as insufficient data. A stopped task or a run that raises leaves the
  heartbeat stale; a run that raises also increments the failure counter. "Insufficient
  data" and "not evaluated" are valid states, shown on the dashboard, and never alert.
- Alertmanager groups alerts by severity and sends them by webhook to `alert-sink`,
  which writes each notification to its log.
- New runbook entries: drift evaluator stale, and the monitoring stack being down.

## 8. Drift Windows

**Prediction path (must stay fast and cannot fail because of monitoring).**

1. Once a prediction is computed, and before the response is sent, its label,
   confidence bucket, length bucket, and uncertain flag are added to the open window
   under a lock.
2. When the window reaches the minimum size, the same locked step hands the full window
   to the evaluator and opens an empty one. This is constant-time; each prediction lands
   in exactly one window.
3. Any error in this step is caught and counted, and never changes the prediction
   response.

**Evaluator (one dedicated thread, not a task on the event loop).**

The thread wakes as soon as a full window is handed over, and also on a fixed interval
(15 minutes; 1 minute in drills) for timeouts and the heartbeat. Running in its own
thread means the computation, however small, never holds up request handling. Handed-
over windows wait in a queue of at most 10; if it is full, the oldest is dropped and
counted, which raises `TriageDriftEvaluationFailed`.

1. Run the drift check on every handed-over window against `monitoring_reference.json`
   from the model bundle and publish the result. The check is plain Python (PSI and
   rates), so the API gains no numerical dependency; the logic moves from
   `triage_training.drift`, and its tests move with it. Thresholds come from
   `configs/monitoring.yaml`, mounted read-only, with matching defaults in code.
2. If the open window is older than 24 hours without reaching the minimum size, close it
   as "insufficient data". Low traffic therefore gives honest "insufficient data"
   results rather than unreliable alerts.
3. Update the heartbeat after a successful run. An exception is caught, logged without
   window contents, and counted; the affected window is dropped and counted.

**States.** The published drift status is one of: not evaluated, insufficient data, OK,
WARN, ALERT. After a start or restart it is "not evaluated" until the first window
completes, never a default OK; a restart loses both the open window and the last result,
which is not persisted. "Not evaluated" means no result yet, not "no problem". Only WARN
and ALERT raise drift alerts.

**Dashboard.** Current state and history are separate panels: a stat panel with the
current status and the time since the last completed evaluation, and a time series of
past results with restarts marked, so an earlier WARN is never read as the current
state.

**Single-process requirement.** Windows are complete only if one process sees every
prediction. The entry point runs one Uvicorn worker; several workers or replicas would
each hold partial windows and need shared aggregation or a separate drift service (§14).

**Choosing the minimum size (exploratory).** The reference profile comes from the 398
test-set predictions, so resampling the test set would understate false alarms. Instead,
windows are drawn with replacement from the 318 validation-set predictions, which come
from the same distribution but were not used to build the reference. For each candidate
size (100, 150, 200, …), 1,000 windows are checked, and the minimum size is the
smallest at which fewer than 5% of them reach WARN. With only 318 validation rows this
is an estimate, not a calibration, and is reported as such. The chosen size and the
measured rate go into `configs/monitoring.yaml` and this document.

**Result ([reports/drift_window_estimate.md](../reports/drift_window_estimate.md)).** At
100 predictions, 64% of normal windows reached WARN, almost all from label PSI: with
eight specialties, sampling noise alone pushes PSI past 0.10. The rate falls to 23% at
200, 8% at 300, and **4.6% at 400**, the smallest size under the 5% target, though only
just; 500 gives 1.0%. The minimum window is therefore 400 predictions. The earlier
manual drift demo looked clean only because it replayed the very data the reference was
built from. A larger window means slower detection and more "insufficient data" at low
traffic, which the 24-hour timeout already handles.

## 9. Capacity and Load Testing

- **Tool:** k6 from its official container under a `loadtest` Compose profile, so no
  new Python dependency is needed.
- **Payloads:** the 24 synthetic expected-label descriptions, and longer synthetic
  descriptions built by joining them up to the 1,000-character input limit. The joined
  texts are stress data, not realistic clinical notes. Because load depends on tokens,
  not characters, the token count of every payload is recorded (short 10–18 tokens,
  long 106–172 tokens).
- **Scenarios:** steady load at 1, 2, 4, and 8 concurrent virtual users for 3 minutes
  each.
- **Recorded:** throughput, p50, p95, and p99 latency, error rate, and CPU use per step.
  The highest step that meets the latency and error targets becomes the supported
  capacity in §6.
- **Expected limit:** one Uvicorn process running PyTorch on CPU. Higher concurrency
  queues requests, so latency, not errors, should fail first.
- **Evaluator interference:** one step is repeated with the minimum window lowered so
  the evaluator runs continually; `/predict` p95 must not rise noticeably compared with
  the same step at the normal window size.

## 10. Release and Rollback

A deployment is the pair of an **application image** and a **model bundle revision**.
They change independently, so both are recorded in a deployment record
(`deploy/current.json`: image tag, model archive revision, weights SHA-256, date).
`/version` and the `triage_model_info` metric must agree with the record after every
deployment.

**Candidate release.** The archive holds one model revision. The training-seed-1 model
from the existing three-seed experiment is the first candidate. It must pass the full
release gate on its own evaluation report, including calibration, abstention, and
per-class criteria, not only macro F1. If it fails, the training-seed-2 model is tried.
If neither passes, that is recorded as a finding and the drill covers only the gate
rejecting the candidate; an unqualified model is never released to complete a drill.

1. Download its bundle from the training server.
2. Generate its evaluation report into `reports/candidates/seed1/`, so the served
   model's report is not overwritten.
3. Run the release gate against the seed 42 report first: it must **fail** on the
   "evaluation report belongs to this bundle" check. Then run it against the candidate's
   own report: it must pass.
4. Upload the bundle as a new archive revision and deploy it; update the deployment
   record.

**Rollback.** Restore the previous archive revision with the procedure in the
validation document and restart the API.

Success criteria: the wrong report blocks the release; after rollback, `/version` and
`triage_model_info` report the original weights hash; the deployment record is updated;
and downtime, from stopping the old process to `triage_model_loaded == 1`, is at most
60 seconds.

## 11. Operational Drills

Each drill is recorded in `docs/operations/drills/` with: scenario, start time, detection
time, actions with times, outcome, and follow-ups. Records are blameless and include
what did not work.

| Drill | Injected condition | Expected detection | Success criteria |
| --- | --- | --- | --- |
| D0 Normal replay | Replay the test split, cycling through it until a 400-prediction window completes | None | Completed windows reproduce the reference (PSI near 0); no alert fires |
| D1 Model unavailable | Point `TRIAGE_MODEL_DIR` at an empty directory, restart, and send at least one request every 5 seconds | `TriageModelNotLoaded`; `TriageServerErrorsCritical` | `/health` stays up; first alert within 3 min, second within 12 min; drift status shows "not evaluated" after each restart; service restored with the runbook |
| D2 Out-of-scope traffic | Restart the API so windows hold only drill traffic, then replay excluded-specialty descriptions for at least one full window (745 available, so one window of 400) | `TriageDriftWarning` (the manual demo reached WARN) | Alert fires within 1 min of a window completing; dashboard shows the label PSI rising |
| D3 Overload | k6 above the supported capacity for 15 minutes | `TriageHighLatency` | Alert fires within 12 min and resolves after load stops |
| D4 Release and rollback | §10 | Gate result; `/version` change | All §10 criteria met |

D0 shows that the live pipeline reproduces offline results; the false-alarm estimate
comes from §8, not from D0. D2 is expected to reach WARN, not ALERT, because
out-of-scope input mostly lands in existing groups with high confidence. If no alert
fires, that is recorded as a finding, and thresholds are not tuned to make a drill pass.

Monitoring the monitoring stack itself (for example, an always-firing watchdog alert
whose absence is noticed) is in §14.

## 12. Security and Supply Chain

- `pip-audit` runs in CI on every pull request as an advisory job. It reports but does
  not block merging, because fixes often depend on upstream releases. Dependabot covers
  Python packages, GitHub Actions, and the Docker base image.
- All Compose ports bind to `127.0.0.1`. The Grafana admin password comes from a local
  `.env` file, which Git ignores; `.env.example` documents the variable.
- Monitoring images use pinned versions and mount only their configuration and data
  volumes.

## 13. MVP Scope

All items below are in the MVP. Estimates are rough and exclude integration and
debugging, so about 30% is added on top.

| Order | Item | Done when | Estimate |
| --- | --- | --- | --- |
| M1 | Prometheus `/metrics` (via `prometheus-client`, the only new dependency) replacing the JSON endpoint: operational metrics, `triage_model_loaded`, `triage_model_load_duration_seconds`, `triage_model_info`; model preload that logs a failure and keeps serving | Tests pass; no per-specialty counts in any response; failed preload leaves `/health` up and `triage_model_loaded` at 0 | 2 h |
| M2 | Compose `monitoring` profile: Prometheus, Alertmanager, alert-sink, Grafana with provisioned data source and dashboard; all ports on localhost | One command shows the dashboard with live data | 2 h |
| M3 | Alert rules from §7 with runbook links; new runbook entries | `promtool` validates the rules in CI | 1 h |
| M4 | Drift windows on the prediction path, background evaluator, states, heartbeat, failure counters; window-size estimate (§8) | Tests show: concurrent predictions each counted in exactly one window; an evaluator exception leaves `/predict` responses unchanged and increments the failure counter; "not evaluated" after start; 24-hour timeout; window size and false-alarm rate recorded | 3 h |
| M5 | Drills D0, D1, D2 with records | Expected alerts observed in Alertmanager and the alert-sink log; D0 raises none | 1.5 h |
| M6 | Candidate release and rollback, drill D4 | §10 success criteria met and recorded | 1.5 h |
| M7 | Load test, capacity in §6, drill D3 | Capacity, latencies, and token counts recorded | 1.5 h |
| M8 | `pip-audit` advisory job and Dependabot | Job runs on pull requests | 0.5 h |

Total: about 13 hours of estimates, roughly 17 hours with the integration allowance,
so two working days. Items are ordered so that the core path (M1–M3) is usable on its
own if later items slip.

**Status (2026-10-15).**

| Item | Status | Evidence |
| --- | --- | --- |
| M1 Prometheus metrics, preload | Done | PR #25 |
| M2 Monitoring stack | Done | PR #26; dashboard in `docs/images/grafana-drift-warning.png` |
| M3 Alert rules, runbook | Done | PR #26; promtool and amtool in CI |
| M4 Drift windows | Done | PR #25; window of 400 from `reports/drift_window_estimate.md` |
| M5 Drills D0–D2 | Done | `docs/operations/drills/`; D1 found a 500 that should have been 503, now fixed |
| M6 Release and rollback (D4) | Waiting | Needs the training-seed-1 bundle from the training server |
| M7 Load test, D3 | Done | `reports/load_test.md`; capacity one client on the development laptop |
| M8 Dependency audit | Done | Advisory CI job; first run reported jinja2, pytest, and accelerate advisories |

**Definition of done for the implementing pull requests.** Documents that describe the
removed JSON `/metrics` and `monitor-drift` are updated in the same pull request that
removes them: the README (API table, Validation and Operations section), the validation
document (§4 monitoring commands and §5 runbook), the product design document (§9
architecture, §10 API, §11 model bundle, §12 governance), and `docs/training.md`. Before
merging, a search for `monitor-drift` and the JSON snapshot command returns no remaining
instructions.

**Not in the MVP:** automated release pipeline, external notifications, authentication,
multi-process metrics, monitoring of the monitoring stack.

**Prerequisites from the owner:** Docker Desktop running (from M2); the training-seed-1
bundle downloaded from the training server (for M6).

## 14. Later Phases

**Phase 2: release pipeline**

- GitHub Actions release workflow, started by hand or by a version tag on `main`:
  download the bundle with a read-only token stored as a repository secret, run the
  release gate, build and scan the image, push it to GitHub Container Registry with
  labels for the Git commit and model weights hash, and write the deployment record.
- The secret is available only to this workflow on `main`, never to pull requests.

**Phase 3: production-shaped operations**

- Authentication on all endpoints; rate limiting.
- Shared aggregation for several processes or replicas.
- Notifications by email or chat; a watchdog alert to detect a broken alerting path.
- Reviewed label samples from real use, so accuracy can be measured, not only drift
  (needs its own privacy design).
- Canary or shadow deployment of a new model before full rollout.

## 15. Risks and Open Questions

| Risk | Mitigation |
| --- | --- |
| Drills on synthetic and replayed traffic may not reflect real use | Stated in every record; thresholds remain provisional |
| Window size estimated from 318 validation rows | Reported as an estimate; revisited when more normal traffic exists |
| Drift computed inside the API couples monitoring to serving | Prediction path only increments counters; evaluation runs in a dedicated thread whose failures are caught, counted, and tested not to affect responses |
| A restart loses the open window and the last result | Status returns to "not evaluated", never to a default OK; drill D1 checks this |
| More than one process would split windows | Entry point runs one worker; the requirement is stated in §8 |
| Removing the JSON `/metrics` endpoint breaks existing scripts and documents | Changed in the same pull request; the manual drift demo is replaced by drill D2 |
| The monitoring stack adds moving parts for one small service | Kept behind a Compose profile; default start-up unchanged |
| Screenshots can be staged | Configuration, rules, dashboard, and drill records are committed and reproducible |

Decisions from review:

1. A completed window wakes the evaluator immediately; the fixed interval only handles
   timeouts and the heartbeat.
2. The last drift result is not persisted across restarts; Prometheus keeps the history.
3. A "traffic expected but none received" alert is deferred until the service has
   expected traffic hours. Any future synthetic probe requests must be marked and
   excluded from drift windows.
