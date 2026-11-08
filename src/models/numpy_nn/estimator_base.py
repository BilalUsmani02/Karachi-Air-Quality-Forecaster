"""The training loop shared by the NumPy network and its PyTorch twin.

Everything here is *bookkeeping* — checking inputs, holding out validation rows, scaling the
target, shuffling mini-batches, early stopping. The maths (forward pass, gradients, optimiser
step) lives in the subclasses: ``MLPRegressorNP`` does it by hand, ``MLPRegressorTorch`` lets
PyTorch do it. Because both run this same loop with the same seed, they see the same initial
weights and the same mini-batches in the same order, so any difference in their results can
only come from the maths — which is what makes the equivalence test meaningful.

Conventions (identical for both models):

- **Validation split**: with ``early_stopping=True`` the *last* ``validation_fraction`` of the
  rows (time order) is held out — never a random sample, which would leak future days.
- **Target scaling** (``scale_y=True``): ``y`` is standardised with the mean / std of the
  training rows, the network learns ``(y − mean) / std``, and ``predict`` maps back. PM2.5
  spans roughly 5–250 µg/m³; centred, unit-size targets let the default learning rate work.
  Losses in ``history_`` — and ``huber_delta`` — are therefore in standardised units.
- **Losses recorded**: data loss only (no L2 term). ``history_["train_loss"]`` is the
  batch-size-weighted mean of that epoch's mini-batch losses (taken before each update);
  ``history_["val_loss"]`` is measured after the epoch in prediction mode (no dropout);
  ``history_["step_loss"]`` lists every mini-batch loss.
- **Randomness**: one ``random_state`` is split (``np.random.SeedSequence.spawn``) into
  independent streams for weight init, batch shuffling and dropout.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from src.config import SEED
from src.models.numpy_nn.init import init_mlp_weights

ACTIVATION_NAMES = ("relu", "tanh", "leaky_relu")


class ParamsMixin:
    """sklearn-style ``get_params`` / ``set_params``, read off the constructor signature.

    Works with ``sklearn.base.clone``, so a tuner can copy a model with new settings.
    """

    @classmethod
    def _param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        return [name for name in signature.parameters if name != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> "ParamsMixin":
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


def epoch_batches(n_rows: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle ``0..n_rows-1`` and cut it into mini-batches (the last one may be smaller)."""
    order = rng.permutation(n_rows)
    return [order[i : i + batch_size] for i in range(0, n_rows, batch_size)]


def check_X(X: Any, n_features: int | None = None) -> np.ndarray:
    """Return ``X`` as a new float64 ``(n, d)`` array, or raise ``ValueError``."""
    X = np.array(X, dtype=np.float64)  # always a copy: training never touches caller data
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2-D array (n, d), got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values; impute before fitting")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, the model was fitted on {n_features}")
    return X


def check_y(y: Any, n_rows: int) -> np.ndarray:
    """Return ``y`` as a new float64 ``(n,)`` array (``(n, 1)`` is accepted), or raise."""
    y = np.array(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or y.shape[0] != n_rows:
        raise ValueError(f"y must have shape ({n_rows},), got {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    return y


class BaseMLPRegressor(ParamsMixin):
    """Shared constructor, ``fit`` loop and ``predict`` for both MLP regressors.

    Subclasses implement five small hooks: ``_build(weights, dropout_rng)``,
    ``_train_step(Xb, yb) -> loss``, ``_loss(X, y) -> loss``, ``_forward(X) -> (n,)`` and
    ``get_weights() / _set_weights(weights)`` (weights always in NumPy layout:
    ``[W1 (d_in, h1), b1 (h1,), W2, b2, ...]``).
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        loss: str = "mse",
        huber_delta: float = 1.0,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 200,
        l2: float = 1e-4,
        dropout: float = 0.0,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        patience: int = 20,
        scale_y: bool = True,
        random_state: int | None = SEED,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.loss = loss
        self.huber_delta = huber_delta
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.l2 = l2
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.scale_y = scale_y
        self.random_state = random_state

    # --- hooks for subclasses -------------------------------------------------------------

    def _build(self, weights: list[np.ndarray], dropout_rng: np.random.Generator) -> None:
        raise NotImplementedError

    def _train_step(self, Xb: np.ndarray, yb: np.ndarray) -> float:
        raise NotImplementedError

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def _forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_weights(self) -> list[np.ndarray]:
        raise NotImplementedError

    def _set_weights(self, weights: list[np.ndarray]) -> None:
        raise NotImplementedError

    # --- shared logic ---------------------------------------------------------------------

    def _check_hyperparams(self) -> None:
        if self.activation not in ACTIVATION_NAMES:
            raise ValueError(f"activation must be one of {ACTIVATION_NAMES}, got {self.activation!r}")
        if self.loss not in ("mse", "huber"):
            raise ValueError(f"loss must be 'mse' or 'huber', got {self.loss!r}")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {self.optimizer!r}")
        if any(int(h) < 1 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be >= 1")
        if self.learning_rate <= 0 or self.l2 < 0 or self.huber_delta <= 0:
            raise ValueError("learning_rate and huber_delta must be > 0, l2 must be >= 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.early_stopping and not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction}")

    def _layer_sizes(self, n_features: int) -> list[int]:
        return [n_features, *(int(h) for h in self.hidden_sizes), 1]

    def _check_weights(self, weights: list[np.ndarray], n_features: int) -> list[np.ndarray]:
        sizes = self._layer_sizes(n_features)
        expected = [s for a, b in zip(sizes[:-1], sizes[1:]) for s in ((a, b), (b,))]
        got = [np.shape(w) for w in weights]
        if got != expected:
            raise ValueError(f"initial_weights shapes {got} do not match the architecture {expected}")
        return [np.array(w, dtype=np.float64) for w in weights]

    def _split(self, n_rows: int) -> int:
        """Number of training rows; the rest (the last rows) are the validation set."""
        if not self.early_stopping:
            return n_rows
        n_val = max(1, int(round(n_rows * self.validation_fraction)))
        if n_rows - n_val < 1:
            raise ValueError(f"need at least 2 rows to hold out a validation set, got {n_rows}")
        return n_rows - n_val

    def fit(self, X: Any, y: Any, initial_weights: list[np.ndarray] | None = None):
        """Train on ``X (n, d)``, ``y (n,)``; rows must be in time order.

        ``initial_weights`` (optional, NumPy layout) replaces the random initialisation — used
        to start the PyTorch twin from exactly the NumPy model's ``initial_weights_``.
        """
        self._check_hyperparams()
        X = check_X(X)
        y = check_y(y, X.shape[0])
        init_rng, shuffle_rng, dropout_rng = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.random_state).spawn(3)
        )

        n_train = self._split(X.shape[0])
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        self.y_mean_, self.y_std_ = 0.0, 1.0
        if self.scale_y:
            self.y_mean_ = float(y_train.mean())
            self.y_std_ = float(y_train.std()) or 1.0  # a constant target would divide by 0
        y_train = (y_train - self.y_mean_) / self.y_std_
        y_val = (y_val - self.y_mean_) / self.y_std_

        if initial_weights is None:
            weights = init_mlp_weights(self._layer_sizes(X.shape[1]), self.activation, init_rng)
        else:
            weights = self._check_weights(initial_weights, X.shape[1])
        self.n_features_in_ = X.shape[1]
        self.initial_weights_ = [w.copy() for w in weights]
        self._build(weights, dropout_rng)

        self.history_ = {"train_loss": [], "val_loss": [], "step_loss": []}
        best_val, best_weights, best_epoch, since_best = np.inf, None, 0, 0
        for epoch in range(1, self.max_epochs + 1):
            batch_losses, batch_rows = [], []
            for idx in epoch_batches(n_train, self.batch_size, shuffle_rng):
                batch_losses.append(self._train_step(X_train[idx], y_train[idx]))
                batch_rows.append(len(idx))
            self.history_["step_loss"] += batch_losses
            self.history_["train_loss"].append(float(np.average(batch_losses, weights=batch_rows)))
            self.n_epochs_ = epoch
            if not self.early_stopping:
                continue
            val_loss = self._loss(X_val, y_val)
            self.history_["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val, best_weights, best_epoch, since_best = val_loss, self.get_weights(), epoch, 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        self.best_epoch_ = best_epoch if self.early_stopping else self.n_epochs_
        if best_weights is not None:
            self._set_weights(best_weights)  # keep the weights from the best validation epoch
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicted target ``(n,)`` in the original units (µg/m³ for PM2.5)."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit() first")
        X = check_X(X, self.n_features_in_)
        return self._forward(X) * self.y_std_ + self.y_mean_
