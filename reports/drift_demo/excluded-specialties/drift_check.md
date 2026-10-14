# Drift Check

Overall status: **WARN**  
Window start: 2026-10-14T08:46:01+00:00; predictions: 400  
Model weights: `a2f4ad5c6d97`

Signals compare production outputs and inputs with the test-set reference. Without ground-truth labels they show that something changed, not that accuracy dropped.

| Check | Metric | Value | Reference | Status |
| --- | --- | --- | --- | --- |
| Predicted label distribution | PSI | 0.197 | 0.000 | WARN |
| Confidence distribution | PSI | 0.101 | 0.000 | WARN |
| Input length distribution | PSI | 0.024 | 0.000 | OK |
| Uncertain rate | rate | 0.120 | 0.075 | OK |
| Server error rate | rate | 0.000 | 0.000 | OK |

Label distribution chi-square p-value: 3.4e-14 (informational; with many predictions tiny shifts become significant, so status uses PSI).
