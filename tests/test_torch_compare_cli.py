"""`python -m src.models.torch_nn`: the real-data NumPy-vs-PyTorch report, on a tiny synthetic table."""

import json

import numpy as np
import pandas as pd
import pytest

from src.classify.bands import bands_for
from src.data.features import COLUMN_ORDER
from src.data.sensors import SENSORS
from src.models.numpy_nn.network import MLPRegressorNP
from src.models.numpy_nn.variant import CALENDAR_COLUMNS, TRANSFORMED_COLUMNS, MLPVariant
from src.models.torch_nn import report
from src.models.torch_nn.compare import equivalence, layer_names
from src.models.torch_nn.model import MLPRegressorTorch

TINY = dict(
    feature_set="no_calendar", target="residual", hidden_sizes=(4,), activation="relu", loss="mse",
    learning_rate=1e-2, batch_size=16, l2=0.1, dropout=0.0, early_stopping=True,
    validation_fraction=0.15, patience=20, max_epochs=3,
)


def _frame(start="2026-07-01", end="2026-09-10", sensors=("S1", "S3"), seed=0) -> pd.DataFrame:
    """Feature-table-shaped rows straddling FINAL_TEST_START (2026-09-01)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([{"sensor_id": s, "date": d} for d in pd.date_range(start, end) for s in sensors])
    n = len(df)
    pm25 = np.clip(40 + 20 * np.sin(np.arange(n) / 9) + rng.normal(0, 5, n), 0, None)
    df["y_pm25"], df["y_hours"], df["y_band"] = pm25, 20, bands_for(pd.Series(pm25)).astype(int)
    for col in COLUMN_ORDER:
        if col in df.columns or col.startswith("sensor_"):
            continue
        if col == "pm25_prev_day_mean":
            df[col] = df.groupby("sensor_id")["y_pm25"].shift(1).fillna(pm25.mean())
        else:
            df[col] = rng.normal(50, 10, n)
    for sid in (s.id for s in SENSORS):
        df[f"sensor_{sid}"] = (df["sensor_id"] == sid).astype(int)
    return df[list(COLUMN_ORDER)].sample(frac=1.0, random_state=1)  # shuffled: run() must re-sort


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("numpy_vs_torch")
    return report.run(_frame(), out, params=TINY), out


def test_run_writes_every_output(summary):
    _, out = summary
    for name in ("summary.json", "summary.md", "loss_curves.png", "learning_curve.png"):
        assert (out / name).stat().st_size > 0
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 42 and saved["params"]["hidden_sizes"] == [4]
    assert {"numpy", "torch", "torch_threads"} <= set(saved["versions"])
    assert "not a score on the final test" in saved["data"]["prediction_rows"]


def test_twins_match_on_real_shaped_data(summary):
    s, _ = summary
    assert s["prediction_diff"]["n_rows"] == 2 * 10  # 1-10 Sep, two sensors
    assert s["prediction_diff"]["max_abs"] < 1e-9
    assert max(s["weight_max_abs_diff"].values()) < 1e-12
    assert set(s["initial_weight_max_abs_diff"].values()) == {0.0}
    assert s["loss_diff"]["max_step"] < 1e-12
    assert len(s["loss_diff"]["train_per_epoch"]) == s["epochs"]["numpy"] == s["epochs"]["torch"] == 3
    assert s["data"]["train_period"] == ["2026-07-01", "2026-08-31"]
    assert s["data"]["n_inputs_after_wrapper"] == len(TRANSFORMED_COLUMNS) - len(CALENDAR_COLUMNS)


def test_summary_markdown_has_a_table(summary):
    _, out = summary
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "| Prediction difference, max (µg/m³) |" in text
    assert "`dense2.W`" in text and "not a final-test score" in text


def test_torch_variant_wraps_the_twin():
    assert MLPVariant.regressor_cls is MLPRegressorNP
    assert report.MLPVariantTorch.regressor_cls is MLPRegressorTorch


def test_run_needs_rows_after_final_test_start(tmp_path):
    with pytest.raises(ValueError, match="no rows on or after"):
        report.run(_frame(end="2026-08-31"), tmp_path, params=TINY)


def test_main_reads_a_features_file(tmp_path, capsys, monkeypatch):
    path = tmp_path / "features.parquet"
    _frame().to_parquet(path)
    monkeypatch.setattr(report, "MLP_PARAMS", TINY)
    report.main(["--features", str(path), "--out-dir", str(tmp_path / "out")])
    assert "NumPy network vs PyTorch twin" in capsys.readouterr().out
    assert (tmp_path / "out" / "summary.json").exists()


def test_layer_names_follow_build_mlp():
    assert layer_names([np.zeros((2, 3)), np.zeros(3), np.zeros((3, 1)), np.zeros(1)]) == [
        "dense1.W", "dense1.b", "dense2.W", "dense2.b",
    ]


def test_equivalence_refuses_histories_of_different_length():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 3)), rng.normal(size=40)
    a = MLPRegressorNP(hidden_sizes=(4,), max_epochs=2).fit(X, y)
    b = MLPRegressorTorch(hidden_sizes=(4,), max_epochs=3).fit(X, y)
    with pytest.raises(ValueError, match="differ in length"):
        equivalence(a, b, a.predict(X), b.predict(X))
