# NumPy neural network (from scratch)

A multi-layer perceptron for next-day PM2.5, written with NumPy only — no autograd, no
deep-learning library. A PyTorch twin (`src/models/torch_nn/`) proves it is correct.

**Full derivation** (L-layer forward/backward, MSE and Huber, L2, SGD and Adam, initialisation,
early stopping, verification results, learning curve): [`docs/report/numpy_network.md`](../../../docs/report/numpy_network.md).

## Files

| File | What it does |
|---|---|
| `layers.py` | `Dense`, `ReLU`, `Tanh`, `LeakyReLU`, `Dropout` (inverted). Each has `forward` (caches what backward needs) and `backward` (one chain-rule step). |
| `losses.py` | `MSELoss`, `HuberLoss(delta)` — value and gradient, averaged over the batch like PyTorch's `reduction="mean"`. |
| `init.py` | He (ReLU) and Xavier/Glorot (tanh, output layer) initial weights from a seeded generator. |
| `optimizers.py` | `SGD` and `Adam` (bias-corrected, PyTorch defaults); L2 as weight decay on weights only. |
| `network.py` | `Sequential` container, `build_mlp(weights, ...)`, and the user-facing `MLPRegressorNP`. |
| `estimator_base.py` | The training loop shared with the PyTorch twin: input checks, last-rows validation split, target scaling, shuffled mini-batches, early stopping, `get_params`/`set_params`. |
| `gradcheck.py` | Numerical gradient check (central differences, float64). `python -m src.models.numpy_nn.gradcheck` writes `results/numpy_vs_torch/gradcheck.csv`. |

## Usage

```python
model = MLPRegressorNP(hidden_sizes=(64, 32), activation="relu", loss="huber",
                       optimizer="adam", learning_rate=1e-3, l2=1e-4, random_state=42)
model.fit(X_train, y_train)          # rows in time order; the last 15 % are validation
pm25 = model.predict(X_test)         # (n,), µg/m³
model.history_["val_loss"], model.n_epochs_, model.best_epoch_
```

## The maths for one Dense + ReLU + MSE step

Batch of `n` rows, `d` inputs, `h` hidden units, one output.

**Forward**

1. `Z = X W1 + b1` — `X (n, d)` times `W1 (d, h)` gives `Z (n, h)`; `b1 (h,)` is added to every row.
2. `A = ReLU(Z) = max(0, Z)` — `(n, h)`.
3. `P = A W2 + b2` — `(n, 1)`, the prediction.
4. `L = mean((P − y)²)` — one number.

**Backward** — walk the same steps in reverse, multiplying local slopes (the chain rule):

1. `dL/dP = 2 (P − y) / n` — `(n, 1)`. How much the loss changes if each prediction moves.
2. Output layer: `dL/dW2 = Aᵀ (dL/dP)` → `(h, 1)`; `dL/db2 = sum of dL/dP over rows` → `(1,)`;
   `dL/dA = (dL/dP) W2ᵀ` → `(n, h)`, the error passed back.
3. ReLU: `dL/dZ = dL/dA` where `Z > 0`, and `0` where `Z ≤ 0` (a switched-off unit passes no error).
4. First layer: `dL/dW1 = Xᵀ (dL/dZ)` → `(d, h)`; `dL/db1 = sum of dL/dZ over rows` → `(h,)`.

Every gradient has the same shape as the thing it belongs to. The optimiser then moves each
weight a little against its gradient (Adam scales that step per weight).

**L2** adds `(l2 / 2) Σ W²` to the loss, i.e. `l2 · W` to each weight gradient (not biases).
**Target scaling**: the network learns `(y − mean) / std` of the training targets and
`predict` converts back, so losses and `huber_delta` are in standardised units.

## How we know it is correct

1. **Gradient check** (`gradcheck.py`). A derivative is a slope, so nudge each weight up and
   down by `1e-6` and measure the change in loss: `(L(w+ε) − L(w−ε)) / 2ε`. This uses only the
   forward pass. Comparing it with the backward-pass gradient for every weight gives relative
   errors of about **1e-10** for Dense, ReLU, Tanh, LeakyReLU, MSE, Huber and a full
   2-hidden-layer network with L2 (test threshold 1e-6; a deliberately broken tanh scores ~0.1).
2. **PyTorch twin** (`torch_nn/model.py`). Same architecture, same starting weights, same
   mini-batches in the same order — but PyTorch's own layers, autograd and optimisers. After 20
   training steps (SGD and Adam, MSE and Huber, with and without L2) every per-step loss and every
   final weight agrees to about **1e-16**, i.e. floating-point rounding. Two independent
   implementations of the same maths agreeing that closely means the NumPy one is right.
   On the real data with the tuned settings (`python -m src.models.torch_nn` →
   `results/numpy_vs_torch/`), predictions agree to ~1e-14 µg/m³ and weights to ~1e-16.
