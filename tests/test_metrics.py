import math

import numpy as np
import pytest

from src.evaluate.metrics import classification_metrics, regression_metrics


# --- regression_metrics ---------------------------------------------------------


def test_regression_metrics_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    y_persistence = np.array([10.0, 10.0, 20.0, 20.0])  # same as previous day

    m = regression_metrics(y_true, y_pred, y_persistence)

    mae = (2 + 2 + 3 + 4) / 4
    mae_persistence = (0 + 10 + 10 + 20) / 4
    assert math.isclose(m["mae"], mae)
    assert math.isclose(m["skill"], 1 - mae / mae_persistence)

    rmse = math.sqrt(((2) ** 2 + (-2) ** 2 + (3) ** 2 + (-4) ** 2) / 4)
    assert math.isclose(m["rmse"], rmse, rel_tol=1e-9)


def test_regression_metrics_r2_perfect_fit_is_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y_true, y_true, y_true + 5)
    assert math.isclose(m["r2"], 1.0, abs_tol=1e-9)


def test_regression_metrics_skill_positive_when_model_beats_persistence():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])  # perfect
    y_persistence = np.array([0.0, 0.0, 0.0])  # terrible
    m = regression_metrics(y_true, y_pred, y_persistence)
    assert m["skill"] > 0


def test_regression_metrics_skill_negative_when_model_worse_than_persistence():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([0.0, 0.0, 0.0])  # terrible: off by 10/20/30
    y_persistence = np.array([9.0, 19.0, 31.0])  # close: off by 1 each time
    m = regression_metrics(y_true, y_pred, y_persistence)
    assert m["skill"] < 0


def test_regression_metrics_skill_is_nan_when_persistence_is_perfect():
    """mae_persistence == 0 would divide by zero; skill is reported as NaN instead."""
    y_true = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y_true, np.array([8.0, 19.0, 33.0]), y_persistence=y_true)
    assert math.isnan(m["skill"])


def test_regression_metrics_skill_zero_when_model_equals_persistence():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([8.0, 22.0, 33.0])
    m = regression_metrics(y_true, y_pred, y_pred)
    assert math.isclose(m["skill"], 0.0, abs_tol=1e-9)


# --- classification_metrics ------------------------------------------------------


def test_classification_metrics_hand_computed_with_missing_class():
    # Labels 0..4; class 3 never appears in y_true or y_pred (missing class).
    y_true = [0, 0, 1, 1, 2, 4]
    y_pred = [0, 1, 1, 1, 2, 2]

    m = classification_metrics(y_true, y_pred, labels=range(5))

    assert math.isclose(m["accuracy"], 4 / 6)
    assert m["per_class"][3]["support"] == 0
    assert m["per_class"][3]["precision"] == 0.0
    assert m["per_class"][3]["recall"] == 0.0
    # class 4: true has one sample (index 5), predicted as 2 -> recall 0, support 1
    assert m["per_class"][4]["support"] == 1
    assert m["per_class"][4]["recall"] == 0.0
    # class 0: 2 true, 1 correctly predicted -> recall 0.5; predicted-as-0 count 1, both correct -> precision 1.0
    assert math.isclose(m["per_class"][0]["recall"], 0.5)
    assert math.isclose(m["per_class"][0]["precision"], 1.0)


def test_classification_metrics_no_warnings_on_missing_class(recwarn):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    classification_metrics(y_true, y_pred, labels=range(5))
    assert len(recwarn) == 0


def test_classification_metrics_confusion_matrix_shape_and_values():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 1, 2]
    m = classification_metrics(y_true, y_pred, labels=range(5))
    cm = m["confusion_matrix"]
    assert len(cm) == 5
    assert all(len(row) == 5 for row in cm)
    assert cm[0][0] == 1
    assert cm[1][1] == 1
    assert cm[2][1] == 1
    assert cm[2][2] == 1


def test_classification_metrics_perfect_predictions():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    m = classification_metrics(y_true, y_pred, labels=range(5))
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
