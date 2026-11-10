# NumPy network vs PyTorch twin (real data)

Tuned network (`final.MLP_PARAMS`), 3585 training rows (2025-06-20 to 2026-08-31), seed 42. Both start from the same weights and see the same mini-batches.

| Quantity | Value |
|---|---|
| Prediction difference, max (µg/m³) | 1.4e-14 |
| Prediction difference, mean (µg/m³) | 8.4e-16 |
| Prediction difference on training rows, max (µg/m³) | 2.8e-14 |
| Final weight difference, max, `dense1.W` | 6.9e-17 |
| Final weight difference, max, `dense1.b` | 8.3e-17 |
| Final weight difference, max, `dense2.W` | 1.1e-16 |
| Final weight difference, max, `dense2.b` | 2.8e-17 |
| Per-epoch training loss difference, max | 1.1e-16 |
| Per-epoch validation loss difference, max | 1.4e-17 |
| Per-step (mini-batch) loss difference, max over 3552 steps | 8.9e-16 |
| Epochs run (NumPy / PyTorch) | 37 / 37 |
| Best epoch (NumPy / PyTorch) | 17 / 17 |
| Training time, NumPy (s) | 2.7 |
| Training time, PyTorch (s) | 14.1 |

Predictions compared on 442 final-test rows (2026-09-01 to 2026-11-08). Only the two models' outputs are compared; the targets are not read, so this is not a final-test score.

Like-for-like: both networks run inside the same MLPVariant wrapper (only regressor_cls differs); no_calendar: doy_sin, doy_cos, month dropped from both inputs; residual: both learn y - (a + b * pm25_prev_day_mean), (a, b) fitted once on the training rows; float64 on CPU; same seed -> same initial weights (checked equal) and same mini-batch order.

Python 3.12.13, NumPy 2.5.4, PyTorch 2.14.1+cpu (10 threads). Regenerate with `uv run python -m src.models.torch_nn`.
