# Revised evaluation results

Run `16b8293accdaf07c`; schema 2; Python 3.11.11.

Development n=1069; held-out test n=268.
Exact development cutoffs: q1=6250.4349999999995, q2=12730.99959999999.

| Model | Accuracy | Balanced accuracy | Macro F1 | High recall |
|---|---:|---:|---:|---:|
| k-NN | 85.82% | 86.04% | 0.8585 | 75.27% |
| LDA | 83.58% | 83.99% | 0.8350 | 64.52% |
| QDA | 82.84% | 83.23% | 0.8280 | 64.52% |

Selected: `{'model__n_neighbors': 21, 'model__p': 2, 'model__weights': 'distance'}`. Development selection macro F1: 0.8375 (fold SD 0.0320); fold High recall spans 56.9% to 76.1%.

k-NN misses 23 High records: 5 go to Low and 18 to Medium. High-cost non-smoker recall is 33.3% on n=33. This is a material failure, despite higher aggregate accuracy in this split.

LDA: subgroup n=33, recall=0.0; QDA: subgroup n=33, recall=0.0. The profile omits diagnosis, treatment and utilization information. These are plausible missing predictors, not demonstrated causal explanations. Small differences on one split do not establish statistical superiority.

## Sensitivity to k (best development selection macro F1 at each k)

| k | Best mean macro F1 |
|---|---:|
| 5 | 0.8130 |
| 11 | 0.8233 |
| 21 | 0.8375 |
| 31 | 0.8342 |

Selection scores are optimistic estimates after tuning, not nested-CV estimates. The held-out split was created during this refactor of an already-explored dataset, not external validation.

QDA development covariance diagnostics and any captured warnings are in results.json. No classical hypothesis test validates k-NN.
