import json

import numpy as np
import pandas as pd
import pytest

from src.config import SEED
from src.evaluate import tune, tune_mlp
from src.models.numpy_nn.network import MLPRegressorNP
from src.evaluate.tune_mlp import (
    CALENDAR_COLUMNS,
    PERSISTENCE_INDEX,
    TRANSFORMED_COLUMNS,
    MLPVariant,
)
from src.models.dataset import Preprocessor, make_xy
from tests.test_run_cv import _synthetic_frame

FAST = {"hidden_sizes": (4,), "max_epochs": 3, "patience": 1, "batch_size": 64}


@pytest.fixture(scope="module")
def frame():
    return _synthetic_frame(sensor_ids=("S1", "S2"))


@pytest.fixture(scope="module")
def folds(frame):
    return tune.prepare_folds(frame)


def test_transformed_columns_match_preprocessor_output(frame):
    X_df, _ = make_xy(frame, "y_pm25")
    X = Preprocessor().fit_transform(X_df)
    assert X.shape[1] == len(TRANSFORMED_COLUMNS)
    # The persistence column survives scaling as an exact linear map of the raw one.
    raw = X_df["pm25_prev_day_mean"].to_numpy()
    assert np.corrcoef(raw, X[:, PERSISTENCE_INDEX])[0, 1] == pytest.approx(1.0)


def test_no_calendar_drops_exactly_the_calendar_columns(folds):
    f = folds[0]
    model = MLPVariant(feature_set="no_calendar", **FAST).fit(f.X_train, f.y_train)
    kept = {TRANSFORMED_COLUMNS[i] for i in model.columns_}
    assert kept == set(TRANSFORMED_COLUMNS) - set(CALENDAR_COLUMNS)
    assert model.model_.n_features_in_ == len(TRANSFORMED_COLUMNS) - len(CALENDAR_COLUMNS)


def test_level_target_is_the_plain_network(folds):
    f = folds[0]
    wrapped = MLPVariant(**FAST).fit(f.X_train, f.y_train).predict(f.X_test)
    plain = MLPRegressorNP(random_state=SEED, **FAST).fit(f.X_train, f.y_train).predict(f.X_test)
    np.testing.assert_allclose(wrapped, plain)


def test_residual_anchor_recovers_a_linear_persistence_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(TRANSFORMED_COLUMNS)))
    y = 40 + 15 * X[:, PERSISTENCE_INDEX] + rng.normal(0, 0.1, 300)
    model = MLPVariant(target="residual", **FAST).fit(X, y)
    np.testing.assert_allclose(model.anchor_coef_, [40, 15], atol=0.05)
    # Prediction = anchor + network output on the residual.
    expected = model._anchor(X) + model.model_.predict(X[:, model.columns_])
    np.testing.assert_allclose(model.predict(X), expected)
    assert np.mean(np.abs(model.predict(X) - y)) < 1.0


def test_wrapper_validates_inputs():
    with pytest.raises(ValueError, match="feature_set"):
        MLPVariant(feature_set="nope")
    with pytest.raises(ValueError, match="target"):
        MLPVariant(target="nope")
    with pytest.raises(RuntimeError, match="not fitted"):
        MLPVariant().predict(np.zeros((2, len(TRANSFORMED_COLUMNS))))
    with pytest.raises(ValueError, match="preprocessed features"):
        MLPVariant(**FAST).fit(np.zeros((10, 3)), np.zeros(10))


def test_staged_search_runs_all_three_stages(folds):
    base = {**tune_mlp.QUICK_BASE, **FAST}
    coarse = {"target": ["level", "residual"]}
    sweep = {"l2": [{"l2": 1e-2}], "dropout": [{"dropout": 0.1}], "noop": [{"l2": base["l2"]}]}
    trials = tune_mlp.staged_search(folds, base, coarse, sweep, verbose=False)

    assert trials["mae"].is_monotonic_increasing
    assert set(trials["stage"]) <= {"coarse", "sweep", "combine"}
    assert (trials["stage"] == "coarse").sum() == 2
    assert (trials["stage"] == "sweep").sum() == 2  # the no-op override is skipped
    assert {"band_macro_f1", "rmse", "r2", "skill"} <= set(trials.columns)


def test_diagnose_reports_both_feature_sets_per_fold(folds):
    diag = tune_mlp.diagnose(folds, {**tune_mlp.QUICK_BASE, **FAST})
    assert len(diag) == 2 * len(folds)
    assert set(diag["feature_set"]) == {"all", "no_calendar"}
    assert (diag["train_mae"] >= 0).all() and diag["val_mae"].notna().all()


def test_cli_quick_writes_the_result_files(frame, tmp_path):
    features = tmp_path / "features.parquet"
    frame.to_parquet(features, index=False)
    out = tmp_path / "tuning"

    tune_mlp.main(["--features", str(features), "--results-dir", str(out), "--quick"])

    trials = pd.read_csv(out / "mlp_numpy_trials.csv")
    best = json.loads((out / "mlp_numpy_best.json").read_text(encoding="utf-8"))
    preds = pd.read_parquet(out / "mlp_numpy_cv_predictions.parquet")
    assert (out / "mlp_numpy_diagnosis.csv").exists()

    assert best["quick"] is True
    assert best["pooled"]["mae"] == pytest.approx(trials["mae"].min())
    assert {"mae", "rmse", "r2", "skill", "band_macro_f1"} <= set(best["pooled"])
    assert len(best["per_fold_mae"]) == 8
    assert {"fold", "sensor_id", "date", "y_true", "y_pred", "persistence", "band_pred"} <= set(preds.columns)
    pooled_mae = np.mean(np.abs(preds["y_true"] - preds["y_pred"]))
    assert pooled_mae == pytest.approx(best["pooled"]["mae"])
