"""Figures for the NumPy-vs-PyTorch report: overlaid loss curves and the tuned network's learning curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRAIN_COLOR = "#2a78d6"  # categorical slot 1 (blue)
VAL_COLOR = "#eb6834"  # categorical slot 2 (orange)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ZERO_FLOOR = 1e-18  # an exactly-zero difference is drawn here so the log axis can show it


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _epochs(values: list[float]) -> np.ndarray:
    return np.arange(1, len(values) + 1)


def loss_curves(numpy_history: dict, torch_history: dict, diff: dict, path: Path) -> Path:
    """Left: per-epoch train / validation loss of both models overlaid. Right: |difference|, log scale."""
    fig, (ax, ax_diff) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for key, color, label in (("train_loss", TRAIN_COLOR, "training"), ("val_loss", VAL_COLOR, "validation")):
        np_curve, t_curve = numpy_history[key], torch_history[key]
        if not np_curve:
            continue
        ax.plot(_epochs(np_curve), np_curve, color=color, linewidth=3.5, alpha=0.45, label=f"NumPy, {label}")
        ax.plot(_epochs(t_curve), t_curve, color=color, linewidth=1.2, linestyle="--", label=f"PyTorch, {label}")
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("loss (standardised units)", color=INK)
    ax.set_title("Loss per epoch: NumPy (wide) and PyTorch (dashed) overlap", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    top = 1e-9
    for key, color, label in (("train_per_epoch", TRAIN_COLOR, "training"), ("val_per_epoch", VAL_COLOR, "validation")):
        values = np.asarray(diff[key], dtype=np.float64)
        if values.size:
            top = max(top, 10 * float(values.max()))
            ax_diff.plot(_epochs(values), np.maximum(values, ZERO_FLOOR), color=color, linewidth=1.5,
                         marker="o", markersize=3, label=label)
    ax_diff.set_yscale("log")
    ax_diff.set_ylim(ZERO_FLOOR / 3, top)
    ax_diff.set_xlabel("epoch", color=INK)
    ax_diff.set_ylabel("|NumPy loss − PyTorch loss|", color=INK)
    ax_diff.set_title(f"Absolute difference (exact 0 drawn at {ZERO_FLOOR:g})", color=INK, fontsize=11, loc="left")
    ax_diff.legend(frameon=False, fontsize=9)
    _style(ax_diff)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def learning_curve(history: dict, best_epoch: int, path: Path, title: str) -> Path:
    """Training and validation loss per epoch with the early-stopping epoch marked."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    train, val = history["train_loss"], history["val_loss"]
    ax.plot(_epochs(train), train, color=TRAIN_COLOR, linewidth=2, label="training (mean of mini-batch losses)")
    if val:
        ax.plot(_epochs(val), val, color=VAL_COLOR, linewidth=2, label="validation (last 15 % of rows, after the epoch)")
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=":")
        ax.plot([best_epoch], [val[best_epoch - 1]], "o", color=VAL_COLOR, markersize=8,
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"best epoch {best_epoch}: weights kept", (best_epoch, val[best_epoch - 1]),
                    xytext=(8, 14), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("loss (standardised units)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
