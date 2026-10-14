# Variation Across Data Splits

Each split seed reshuffles which unique descriptions fall into train, validation, and test; training settings are unchanged. This measures dependence on the split, not performance on other data: all splits share one source and one labeling process.

| Split seed | Model | Test n | Accuracy | Macro F1 | Direct-mapped accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | baseline | 398 | 0.776 | 0.787 | 0.734 |
| 1 | pubmedbert | 398 | 0.852 | 0.858 | 0.816 |
| 2 | baseline | 398 | 0.724 | 0.738 | 0.657 |
| 2 | pubmedbert | 398 | 0.864 | 0.875 | 0.824 |
| 3 | baseline | 398 | 0.746 | 0.760 | 0.690 |
| 3 | pubmedbert | 398 | 0.867 | 0.877 | 0.841 |
| 4 | baseline | 398 | 0.771 | 0.788 | 0.734 |
| 4 | pubmedbert | 398 | 0.869 | 0.869 | 0.843 |
| 5 | baseline | 398 | 0.749 | 0.753 | 0.708 |
| 5 | pubmedbert | 398 | 0.877 | 0.887 | 0.854 |
| 42 | baseline | 398 | 0.744 | 0.742 | 0.678 |
| 42 | pubmedbert | 398 | 0.884 | 0.889 | 0.855 |

| Model | Splits | Macro F1 mean ± SD (min–max) | Direct-mapped accuracy mean ± SD (min–max) |
| --- | --- | --- | --- |
| baseline | 6 | 0.761 ± 0.022 (0.738–0.788) | 0.700 ± 0.031 (0.657–0.734) |
| pubmedbert | 6 | 0.876 ± 0.012 (0.858–0.889) | 0.839 ± 0.016 (0.816–0.855) |

PubMedBERT minus baseline macro F1 across 6 splits: mean +0.115, smallest +0.071.
