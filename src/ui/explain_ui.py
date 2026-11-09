"""Thin, pure wrappers around `src.explain.*` for the "Why this forecast?" panel (spec §9, item 6).

Kept separate from the actual SHAP machinery so `app/views/forecast.py` can wrap the slow
calls (`build_explainer`, `shap_reasons_for_sensor`) in `st.cache_resource`/`st.cache_data`
without any Streamlit import leaking into this module — it stays importable and testable with
a fake explainer, no real model or `shap` run needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.explain.reasons import top_reasons
from src.explain.shap_explain import build_explainer, explain_row
from src.models.dataset import FEATURE_COLUMNS


def feature_medians(background_raw: pd.DataFrame) -> dict[str, float]:
    """Training-data medians for every raw feature, for the "low"/"high" wording in reasons."""
    return background_raw[list(FEATURE_COLUMNS)].median().to_dict()


def build_sensor_explainer(
    bundle: dict[str, Any],
    background_raw: pd.DataFrame,
    seed: int,
    background_size: int,
) -> shap.Explainer:
    """A SHAP explainer over `bundle`'s model, backgrounded on `background_raw` (unscaled).

    `background_raw` is scaled with `bundle["preprocessor"]` here, so callers (the app) only
    ever need to load the raw background table once (`load_background`).
    """
    background_scaled = bundle["preprocessor"].transform(background_raw[list(FEATURE_COLUMNS)])
    return build_explainer(bundle, background_scaled, seed=seed, background_size=background_size)


def shap_reasons_for_row(
    explainer: shap.Explainer,
    bundle: dict[str, Any],
    row_raw: pd.Series,
    medians: dict[str, float],
    k: int = 3,
) -> list[str]:
    """The top-`k` plain-English SHAP reasons for one raw feature row (spec §9, item 6)."""
    row_raw = row_raw[list(FEATURE_COLUMNS)]
    row_scaled = bundle["preprocessor"].transform(row_raw.to_frame().T)
    explanation = explain_row(explainer, np.asarray(row_scaled, dtype=np.float64), row_raw)
    return top_reasons(explanation, k=k, feature_medians=medians)
