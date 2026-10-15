# Load Test

**Date:** 2026-10-15 · **Plan:** [operations plan §9](../docs/operations-plan.md#9-capacity-and-load-testing)

## Setup

- API container from `compose.yaml`, one Uvicorn process, PyTorch on CPU, on a Windows
  laptop running Docker Desktop (3.7 GiB container memory limit).
- k6 1.1.0 from the `loadtest` Compose profile (`deploy/loadtest/load.js`), driven by
  `scripts/run_load_test.py`; each step holds a fixed number of virtual users for
  3 minutes with no pause between requests. Requests carry `X-Synthetic-Traffic: 1`, so
  they are served normally but kept out of the drift windows.
- Payloads (`deploy/loadtest/payloads.json`) are synthetic: 24 short descriptions
  (10–18 tokens) and 8 long ones joined from them (617–980 characters, 106–172 tokens),
  drawn at random, so about a quarter of requests are long. The long texts are stress
  data, not realistic clinical notes.

## Capacity

Second sweep, measured without CPU sampling and after the warm-up fix (below):

| Virtual users | Throughput | p50 | p95 | p99 | Max | Errors | p95 ≤ 500 ms and errors < 1% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.2 req/s | 57 ms | 267 ms | 322 ms | 1.2 s | 0% | yes |
| 2 | 11.3 req/s | 110 ms | 439 ms | 571 ms | 0.9 s | 0% | yes |
| 4 | 11.6 req/s | 244 ms | 736 ms | 842 ms | 1.0 s | 0% | no |
| 8 | 12.6 req/s | 494 ms | 1.29 s | 1.47 s | 3.4 s | 0% | no |

Throughput saturates near 11–13 requests per second; beyond that, more concurrency only
queues requests. Latency, not errors, failed first, as the plan predicted. No request
errored in any run.

## Run-to-run variation

The same 2-user, 3-minute load was measured five times on this laptop:

| Run | Conditions | Throughput | p95 |
| --- | --- | --- | --- |
| First sweep | `docker stats` sampling every 5 s, before the warm-up fix | 6.1 req/s | 863 ms |
| First D3 attempt | No sampling | 11.2 req/s | 455 ms |
| A/B test | Sampling | 9.6 req/s | 531 ms |
| A/B test | No sampling | 10.3 req/s | 476 ms |
| Second sweep | No sampling | 11.3 req/s | 439 ms |

CPU sampling explains about 10% (the A/B pair), not the first sweep, which was slower
at every load level (1 user: 5.8 requests per second, p95 464 ms). The likely causes are
the laptop's state at that time (other workloads, Docker VM warm-up), but this was not
established.

**Supported capacity: one concurrent client** (p95 267–464 ms across runs). Two clients
met the target in three of five runs and sit on the threshold. These numbers describe
this laptop only; dedicated hardware would need its own measurement.

## Drift evaluator interference

With the minimum window lowered to 20 predictions (`configs/monitoring.loadtest.yaml`),
the evaluator completed 51 windows during a 1-user step, before synthetic traffic was
excluded from drift, without failures or dropped windows.

| Step (first sweep conditions) | Windows evaluated | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| 1 user, normal window (400) | 2 | 72 ms | 464 ms | 1.09 s |
| 1 user, window of 20 | 51 | 82 ms | 443 ms | 862 ms |

The p95 did not rise, so evaluating drift in a separate thread does not hold up
predictions at this load.

## Findings

- **Slow first inference.** The first request after the model loaded took 6.4 s, the
  second 1.0 s, then about 0.3 s. The service now runs one synthetic warm-up inference
  before reporting the model as loaded; afterwards the first request took 0.63 s and
  loading, including warm-up, took 27.9 s. The warm-up is not counted in drift windows.
- **Load-test traffic polluted drift.** During the first overload drill, the synthetic
  payloads, whose label mix differs from the test split, raised `TriageDriftAlert`. The
  plan had decided that synthetic traffic must be excluded, but the load test did not
  mark it. Requests with `X-Synthetic-Traffic: 1` are now excluded and counted in
  `triage_synthetic_requests_total`; after 3,573 marked requests the drift status stayed
  "not evaluated". Any client can set the header in this prototype; production would
  accept it only from internal callers.
- **Unexplained stalls.** In the first sweep, two requests took 29.7 s (8 users) and
  33.3 s (the first request of a step after a restart). The warm-up explains part of the
  second; the full length is not explained. No stall above 3.4 s occurred after the
  warm-up fix.

## Follow-ups (not in the MVP)

- Limit PyTorch intra-op threads and run several workers, or batch requests, to raise
  throughput; this needs the shared drift aggregation described in the operations plan.
- Measure on dedicated hardware, repeating each step several times.
