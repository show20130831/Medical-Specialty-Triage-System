# Drill D2: Out-of-Scope Traffic

**Date:** 2026-10-15 · **Operator:** project owner · **Result:** passed

## Scenario

Restart the API so the open window holds only drill traffic, then replay 400 unique
MTSamples descriptions from the 16 specialties outside the taxonomy. Expected:
`TriageDriftWarning` within 1 minute of the window completing, delivered to alert-sink.

## Timeline (UTC)

| Time | Event |
| --- | --- |
| before 05:31 | API restarted; after the model loaded, drift status was `not_evaluated` (the earlier `ok` stayed in history only) |
| 05:31:45 | 400 predictions sent; window evaluated: status `warn` |
| 05:32:04 | Prometheus: `TriageDriftWarning` firing (no `for` delay; 19 s after the window) |
| 05:32:19 | alert-sink logged the notification from Alertmanager (34 s after the window) |

## Observed

| Signal | Value | Threshold (WARN) |
| --- | --- | --- |
| Label PSI | 0.197 | 0.10 |
| Confidence PSI | 0.101 | 0.10 |
| Length PSI | 0.024 | 0.10 |
| Uncertain rate | 12.0% | 15% |

Alert description as delivered: "label PSI 0.197; confidence PSI 0.101; length PSI
0.024; uncertain rate 12%. A signal of change, not proof of lower accuracy."

## Outcome

Detection and delivery worked end to end within the 1-minute target. The result also
confirms a documented limitation: traffic the model was never trained for reached only
WARN, and the uncertain rate stayed below its own threshold, because the model assigns
such input to existing groups with high confidence. Thresholds were not changed.

## Follow-ups

- The first dashboard screenshot showed every drift state ever seen in the panel range
  as "current". Fixed: stat panels use instant queries; the uncertain-rate gauge starts
  as NaN so a restart does not plot a false 0%; process metrics were added so restarts
  are marked.
