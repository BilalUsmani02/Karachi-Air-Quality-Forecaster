# Classical model tuning — pooled cross-validation results (Phase 5)

Pooled = every fold's test predictions concatenated, metrics computed once. `params` is the winning configuration from a staged search (coarse scan per hyperparameter, refine around the best), not a full grid.

Phase 3 pooled baselines (`results/baselines_summary.md`): MAE -- RF 6.79, Ridge 7.68, persistence 8.70; band macro-F1 -- persistence 0.678, Ridge->band 0.657, RF->band 0.598, RF direct 0.578 (Very Unhealthy recall 0), logistic 0.509.

Total tuning runtime: 3085.0s (51.4 min).

## Regressors (ranked by pooled MAE)
| family | params | mae | rmse | r2 | skill | band_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| hist_gb | {'learning_rate': 0.1, 'max_leaf_nodes': 31, 'max_iter': 100, 'l2_regularization': 0.1} | 6.3031 | 9.8052 | 0.8882 | 0.2755 | 0.6449 |
| random_forest | {'n_estimators': 500, 'max_depth': 15, 'min_samples_leaf': 5, 'max_features': 0.5} | 6.5401 | 10.0870 | 0.8817 | 0.2482 | 0.6087 |
| ridge | {'alpha': 150.0} | 7.2174 | 10.5399 | 0.8708 | 0.1704 | 0.5822 |

## Classifiers (ranked by pooled macro-F1)
| family | params | accuracy | macro_f1 | recall_0 | recall_1 | recall_2 | recall_3 | recall_4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gb_clf | {'learning_rate': 0.2, 'max_leaf_nodes': 31, 'max_iter': 300, 'l2_regularization': 0.0} | 0.8196 | 0.6645 | 0.6154 | 0.9357 | 0.5479 | 0.8822 | 0.2000 |
| random_forest_clf | {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced', 'min_samples_leaf': 5, 'max_features': 0.5} | 0.7833 | 0.6514 | 0.7846 | 0.8294 | 0.6815 | 0.8057 | 0.2800 |
| logistic_regression | {'C': 0.15, 'class_weight': 'balanced'} | 0.7311 | 0.5890 | 0.7692 | 0.7835 | 0.6336 | 0.7293 | 0.2000 |
