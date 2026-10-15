# Drill D0: Normal Replay

**Date:** 2026-10-15 · **Operator:** project owner · **Result:** passed

## Scenario

Replay the test split, cycling through its 398 descriptions until one 400-prediction
window completes, against the monitoring stack (`docker compose --profile monitoring`,
drill configuration with a 1-minute evaluator interval). Expected: the window reproduces
the reference and no alert fires.

## Timeline (UTC)

| Time | Event |
| --- | --- |
| before 05:25:53 | API restarted with the drill configuration; model loaded; drift status `not_evaluated` |
| 05:25:53 | Replay started |
| 05:26:51 | 400 predictions sent; window evaluated: status `ok` |
| 05:26:51 – 05:29:51 | Alerts watched for 180 s with `scripts/drill_watch.py`: none pending or firing |

## Observed

| Signal | Value | Reference |
| --- | --- | --- |
| Label PSI | 0.0002 | 0 |
| Confidence PSI | 0.0001 | 0 |
| Length PSI | 0.00003 | 0 |
| Uncertain rate | 7.5% | 7.5% |

## Outcome

The live service reproduces the offline reference, so the in-process evaluator, bucket
definitions, and reference file agree. This checks the pipeline only; the false-alarm
rate comes from the validation-window estimate (`reports/drift_window_estimate.md`),
not from this drill, because the replayed data is the data the reference was built from.

## Follow-ups

None.
