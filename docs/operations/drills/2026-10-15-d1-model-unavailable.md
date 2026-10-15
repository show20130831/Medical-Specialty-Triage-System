# Drill D1: Model Unavailable

**Date:** 2026-10-15 · **Operator:** project owner · **Result:** passed, one defect found
and fixed

## Scenario

Start the API with `TRIAGE_MODEL_NAME=missing_bundle`, so `TRIAGE_MODEL_DIR` points at a
directory that does not exist, and send one `/predict` request every 5 seconds.
Expected: `/health` stays up, `TriageModelNotLoaded` fires within 3 minutes and
`TriageServerErrorsCritical` within 12 minutes, both delivered to alert-sink, and
`/predict` answers 503.

## Timeline (UTC)

| Time | Event |
| --- | --- |
| ≈ 05:43:20 | Container started with the missing bundle; preload failed; `/health` 200 |
| 05:43:21 | `TriageModelNotLoaded` pending |
| 05:45:21 | `TriageModelNotLoaded` firing (≈ 2.0 min after start; target ≤ 3) |
| 05:45:36 | alert-sink: `TriageModelNotLoaded` delivered |
| 05:45:36 | `TriageServerErrorsCritical` pending, once 20 requests were inside the 5-minute window |
| 05:50:36 | `TriageServerErrorsCritical` firing (≈ 7.3 min after start; target ≤ 12) |
| 05:51:36 | alert-sink: `TriageServerErrorsCritical` delivered in the next group interval of the critical group |

Timestamps come from the Prometheus alerts API and the alert-sink log;
`scripts/drill_watch.py` started before the image rebuild, so its elapsed times are
offset by the build.

## Observed

- `/health` returned 200 throughout.
- **Defect:** `/predict` and `/ready` returned **500**, not the documented 503. A missing
  directory raises `ModelAssetError` during validation, and the API only translated
  `ModelLoadError`. Unit tests had covered an unset `TRIAGE_MODEL_DIR` (which raises
  `ModelLoadError`) but not a path to a missing directory.
- `TriageServerErrorsWarning` went pending and then resolved: its 15-minute window first
  counted the successful requests of drill D2 (enough to pass the 100-request minimum);
  once those aged out, the drill's 5-second traffic alone was below the minimum. This
  is the minimum-count rule working as designed.
- Severity grouping in Alertmanager delayed the second critical notification to the
  next one-minute group interval, as configured.

## Fix

The API now translates `ModelAssetError` into `ModelLoadError`, so a missing or invalid
bundle answers 503 on `/ready` and `/predict` and the preload logs it without a traceback.
A regression test points `TRIAGE_MODEL_DIR` at a missing directory; it fails without
the fix and passes with it.

## Follow-ups

- Verified after rebuilding with the fix: with the missing bundle, `/health`
  200, `/ready` 503, `/predict` 503, and the log shows one line, "Model preload failed:
  Model directory does not exist"; after switching back, `/predict` 200.
- The Dockerfile now installs PyTorch before copying the source, because each code
  change during this drill re-downloaded it (about 8 minutes per build).
