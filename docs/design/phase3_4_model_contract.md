# Phase 3–4 model contract

How models, preprocessing and evaluation fit together, so the baselines (Phase 3) and the
NumPy / PyTorch networks (Phase 4) can be built in parallel and compared on equal terms.

```
features.parquet ──► dataset.py (X, y) ──► Preprocessor (fit on fold train only)
                                                    │
                     any model with fit(X, y) / predict(X)
                                                    │
                     run_cv.py over split.project_splits folds ──► metrics.py ──► results/
```

## 1. Model interface

Every model — scikit-learn, NumPy, PyTorch — is an object with:

- `fit(X: np.ndarray, y: np.ndarray) -> self` — `X` is float64 `(n, d)`, already imputed and
  scaled; `y` is float64 `(n,)` for regression or int `(n,)` for classification.
- `predict(X: np.ndarray) -> np.ndarray` — `(n,)`.
- `get_params()` / `set_params(**kw)` (sklearn-style) so a tuner can clone and vary it.
- Deterministic given its `random_state` / `seed` parameter.

scikit-learn estimators satisfy this already. The NumPy and PyTorch networks must too.

## 2. Inputs — `src/models/dataset.py` (Phase 3)

- `FEATURE_COLUMNS`: every column of the contract-§2 daily table except the keys (`sensor_id`,
  `date`), the targets (`y_pm25`, `y_hours`, `y_band`) and `pm25_mean_24h` (identical to
  `pm25_prev_day_mean` by construction).
- `make_xy(df, target)` → `(X_frame, y)` with `target` in `{"y_pm25", "y_band"}`; for
  `y_band` the Hazardous → Very Unhealthy merge (`src/classify/bands.merge_bands`) is applied,
  so classes are `0..4`.
- `Preprocessor`: median imputation + standard scaling, **fitted on the training rows of each
  fold only**, returns float64 NumPy arrays. One-hot sensor columns are not scaled.

## 3. Evaluation — `src/evaluate/` (Phase 3)

- Folds: `src/data/split.project_splits(df)` → 8 rolling-origin folds + final test.
  **The final test (from 2026-09-01) is never loaded in Phases 3–4.**
- Regression metrics: MAE, RMSE, R², and **skill vs persistence** = 1 − MAE_model / MAE_persistence.
- Classification metrics: accuracy, **macro-F1**, per-class precision/recall, confusion matrix
  over labels `0..4` (always all five, even if a fold lacks a class).
- Classification is evaluated two ways: **direct** (classifier predicts the band) and
  **regress-then-band** (regressor's PM2.5 → `bands_for` → merge).
- Results: per-fold rows **and** pooled over all folds (concatenate every fold's test
  predictions, then compute metrics once) — pooled is the headline number.

## 4. Networks — `src/models/numpy_nn/`, `src/models/torch_nn/` (Phase 4)

- Regression MLP on the same `X` (`d` inputs → hidden layers → 1 output).
- The NumPy version uses NumPy only (no autograd, no deep-learning library).
- The PyTorch twin has the identical architecture and can **load the NumPy model's initial
  weights**, so the two can be shown to train identically.
- Early stopping holds out the **last** `validation_fraction` of the training rows (time order),
  never a random sample.
