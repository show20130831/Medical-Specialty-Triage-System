# Drift Window Size Estimate

1,000 windows per size drawn with replacement from 318 validation predictions (not used to build the test-set reference). Any WARN or ALERT on these windows is a false alarm. Exploratory: 318 rows cannot calibrate the rate precisely.

| Window size | False WARN or ALERT | False ALERT |
| --- | --- | --- |
| 100 | 64.2% | 4.7% |
| 200 | 22.9% | 0.0% |
| 300 | 8.4% | 0.0% |
| 400 | 4.6% | 0.0% |
| 500 | 1.0% | 0.0% |
| 600 | 0.5% | 0.0% |

Target: fewer than 5% false WARN or ALERT. Smallest size meeting it: **400**.
