# Baselines — pooled cross-validation results

Pooled = every fold's test predictions concatenated, metrics computed once (headline number).

## Regression
| model | mae | rmse | r2 | skill |
| --- | --- | --- | --- | --- |
| mlp_numpy | 11.2111 | 15.4209 | 0.7234 | -0.2887 |
| persistence | 8.6993 | 14.6959 | 0.7488 | 0.0000 |
| random_forest | 6.7889 | 10.6127 | 0.8690 | 0.2196 |
| ridge | 7.6798 | 11.1045 | 0.8566 | 0.1172 |

## Classification
| model | route | accuracy | macro_f1 | recall_0 | recall_1 | recall_2 | recall_3 | recall_4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | direct | 0.6763 | 0.5088 | 0.6769 | 0.8215 | 0.6267 | 0.4140 | 0.1200 |
| mlp_numpy | regress_then_band | 0.6701 | 0.5042 | 0.2000 | 0.7060 | 0.4521 | 0.9204 | 0.2000 |
| persistence | regress_then_band | 0.7888 | 0.6780 | 0.7231 | 0.8832 | 0.6199 | 0.7611 | 0.4000 |
| random_forest | direct | 0.7942 | 0.5776 | 0.5846 | 0.8766 | 0.5890 | 0.8917 | 0.0000 |
| random_forest | regress_then_band | 0.7915 | 0.5980 | 0.4769 | 0.9081 | 0.4829 | 0.9172 | 0.0800 |
| ridge | regress_then_band | 0.7963 | 0.6573 | 0.7385 | 0.8911 | 0.4452 | 0.9459 | 0.2800 |

## Confusion matrix — best classifier by macro-F1: persistence (regress_then_band)
| true \ pred | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | 47 | 18 | 0 | 0 | 0 |
| 1 | 17 | 673 | 62 | 9 | 1 |
| 2 | 0 | 61 | 181 | 50 | 0 |
| 3 | 0 | 6 | 55 | 239 | 14 |
| 4 | 0 | 0 | 0 | 15 | 10 |
