# Drill D4: Release and Rollback

**Date:** 2026-10-15 · **Operator:** project owner · **Result:** no candidate qualified for
release; restore from the archive rehearsed twice; three findings

## Scenario

Plan §10: qualify a second model version through the full release gate, deploy it, then
roll back to the served model from the archive and measure downtime (target ≤ 60 s,
from stopping the old process to `triage_model_loaded == 1`). An unqualified model is
never released to complete a drill.

## Part 1: Release gate on the candidates

Both candidates come from the existing three-seed experiment (same data and split,
different training seed). Each got its own evaluation report and calibration in
`reports/candidates/`.

| Check | Served (seed 42) | Candidate seed 1 | Candidate seed 2 |
| --- | --- | --- | --- |
| Gate with the served model's report | — | **FAIL**: "evaluation report belongs to this bundle" | — |
| Test macro F1 (≥ 0.85) | 0.889 | 0.895 | 0.892 |
| Calibrated ECE (≤ 0.08) | 0.053 | 0.022 | 0.041 |
| Temperature / abstain threshold | 0.95 / 0.65 | 0.75 / 0.50 | 0.84 / 0.57 |
| Expected labels, 24 clear cases (100%) | 24 | **23** | 24 |
| Abstention, 5 ambiguous cases (≥ 80%) | 4 | **2** | **3** |
| Decision | PASS | **FAIL** | **FAIL** |

- Seed 1 is better than the served model on every offline metric, yet labelled
  "Ureteroscopy and laser lithotripsy of a left ureteral stone" as Gastroenterology and,
  with its lower abstain threshold, answered three of the ambiguous cases confidently.
  The behavioural checks caught what the scores did not.
- Seed 2 failed only abstention, answering "Abdominal pain and urinary frequency"
  (Gastroenterology, 0.60) and "Headache and abdominal pain with nausea"
  (General Medicine, 0.97).
- **The abstention set is biased toward the served model.** Four of its five cases were
  chosen after observing seed 42's behaviour (noted in the validation document), so a
  different model is less likely to pass. Both rejections stand, because criteria are
  never relaxed to make a bundle pass; the follow-up is a blind-written abstention set
  that no model has seen.

No candidate was released, so the version switch could not be rehearsed with a
different model.

## Part 2: Restore the served model from the archive

The rollback procedure was rehearsed as a restore: stage the served model from the
private archive into a fresh directory, switch the API to it, and verify its identity.

| Step | First restore (revision `6a4d76b`) | Second restore (revision `27488fb`) |
| --- | --- | --- |
| Stage download (before switching, not downtime) | 76 s | 73 s |
| Downtime: stop to model loaded | **68.1 s** (over target) | **56.6 s** |
| Model load including warm-up | 54.7 s | 41.7 s |
| `/version` weights | `a2f4ad5c6d97…` | `a2f4ad5c6d97…` |
| Drift monitoring | **"not evaluated" forever** | OK after 400 predictions, label PSI 0.0002 |

## Findings

1. **The archive was incomplete.** It had been uploaded before drift monitoring existed,
   so it lacked `monitoring_reference.json`; a model restored from it served correctly
   but could never evaluate drift, and only a log warning showed it. Fixed by uploading
   the file as revision `27488fb`; the second restore confirmed drift works, and every
   file of the local bundle matches that revision.
2. **Downtime sits on the 60-second target.** A stop-then-start switch always includes
   the full model load (22–55 s on this laptop), so the two restores fell on either side
   of the target. The fix is a different deployment pattern (start and warm the new
   instance before switching traffic, so downtime approaches zero), not a relaxed
   target. Recorded as a follow-up.
3. **Two candidates rejected** by the release gate, with the abstention set's bias noted
   above.

## Follow-ups

- Write a blind abstention set before evaluating the next candidate.
- Blue-green or warm-standby switching to remove the load time from downtime.
- When a candidate passes, repeat the drill with an actual version switch.
- `deploy/current.json` records the deployed image and archive revision.
