"""`python -m src.models.numpy_nn.gradcheck`: per-parameter relative errors for MSE and Huber."""

import csv

from src.models.numpy_nn import gradcheck
from src.models.numpy_nn.layers import ACTIVATIONS


def test_run_gradcheck_covers_every_parameter_activation_and_loss():
    rows = gradcheck.run_gradcheck()
    assert len(rows) == 2 * len(ACTIVATIONS) * 6  # 3 Dense layers x (W, b)
    assert {r["loss"] for r in rows} == {"mse", "huber"}
    assert max(r["relative_error"] for r in rows) < 1e-6


def test_gradcheck_catches_a_wrong_gradient(monkeypatch):
    monkeypatch.setattr(gradcheck.HuberLoss, "backward", lambda self: self._residual / self._residual.size)
    rows = [r for r in gradcheck.run_gradcheck() if r["loss"] == "huber"]
    assert max(r["relative_error"] for r in rows) > 1e-3  # unclipped residual != Huber gradient


def test_main_prints_and_saves_csv(tmp_path, capsys):
    out = tmp_path / "gradcheck.csv"
    gradcheck.main(["--out", str(out)])
    printed = capsys.readouterr().out
    assert "worst relative error" in printed and "dense1.W" in printed
    with out.open(encoding="utf-8") as fh:
        saved = list(csv.DictReader(fh))
    assert len(saved) == 36 and saved[0].keys() == set(gradcheck.CSV_FIELDS)
    assert all(float(r["relative_error"]) < 1e-6 for r in saved)
