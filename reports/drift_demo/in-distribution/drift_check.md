# Drift Check

Overall status: **OK**  
Window start: 2026-10-14T08:44:51+00:00; predictions: 398  
Model weights: `a2f4ad5c6d97`

Signals compare production outputs and inputs with the test-set reference. Without ground-truth labels they show that something changed, not that accuracy dropped.

| Check | Metric | Value | Reference | Status |
| --- | --- | --- | --- | --- |
| Predicted label distribution | PSI | 0.000 | 0.000 | OK |
| Confidence distribution | PSI | 0.000 | 0.000 | OK |
| Input length distribution | PSI | 0.000 | 0.000 | OK |
| Uncertain rate | rate | 0.073 | 0.075 | OK |
| Server error rate | rate | 0.000 | 0.000 | OK |

Label distribution chi-square p-value: 1 (informational; with many predictions tiny shifts become significant, so status uses PSI).
