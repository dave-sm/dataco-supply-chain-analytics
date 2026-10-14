"""
inference.py
------------
Loads the deployed LightGBM pipeline and exposes a single predict()
function for the Streamlit app.

The function is the only coupling point between the app and the model.
Everything the app needs to render inputs is read from the feature
schema; everything the model expects is derived here.

Design principle: no logic about feature semantics lives in app.py.
If the model changes, only this module and the artifact directory change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path(__file__).parent / "model-artifact"

MODEL_PATH       = ARTIFACT_DIR / "deploy_model.joblib"
SCHEMA_PATH      = ARTIFACT_DIR / "deploy_feature_schema.json"
METRICS_PATH     = ARTIFACT_DIR / "deploy_metrics.json"
IMPORTANCES_PATH = ARTIFACT_DIR / "deploy_importances.parquet"

# Percentile anchors - pairs of (percentile_in_test_set, cutoff_value).
# Read as: "P% of test-set predictions fall below this cutoff."
# Extracted from the deployed model's test-set predictions.
_PERCENTILE_ANCHORS = {
    0.50: 0.411281,   # median - half of predictions are below this
    0.70: 0.715983,   # top 30% cutoff
    0.80: 0.802623,   # top 20% cutoff
    0.90: 0.998100,   # top 10% cutoff
    0.95: 0.998416,   # top 5% cutoff
    0.99: 0.998796,   # top 1% cutoff
}
_BASE_RATE = 0.5488
_OBSERVED_MAX = 0.9995
_OBSERVED_MIN = 0.0632


# ---------------------------------------------------------------------------
# Lazy loaders - cached so the app pays the load cost once
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _load_model():
    if "model" not in _cache:
        _cache["model"] = joblib.load(MODEL_PATH)
    return _cache["model"]


def _load_schema() -> dict:
    if "schema" not in _cache:
        with open(SCHEMA_PATH) as f:
            _cache["schema"] = json.load(f)
    return _cache["schema"]


def _load_metrics() -> dict:
    if "metrics" not in _cache:
        with open(METRICS_PATH) as f:
            _cache["metrics"] = json.load(f)
    return _cache["metrics"]


def _load_importances() -> pd.DataFrame:
    if "importances" not in _cache:
        _cache["importances"] = pd.read_parquet(IMPORTANCES_PATH)
    return _cache["importances"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_schema() -> dict:
    """Return the feature schema - what the app uses to render inputs."""
    return _load_schema()


def get_metrics() -> dict:
    """Return the metrics dictionary - what the About tab uses to display performance."""
    return _load_metrics()


def get_importances() -> pd.DataFrame:
    """Return global feature importances for the About tab."""
    return _load_importances()


def rank_percentile(probability: float) -> float:
    """
    Approximate the percentile rank of a predicted probability against the
    test-set distribution. Higher return value = higher risk.

    Returns a value in [0, 1] meaning: "this prediction is riskier than
    X% of the test set". Interpolated across the saved anchors.
    """
    p = float(probability)

    # Anchors sorted ascending by cutoff value
    anchors = sorted(_PERCENTILE_ANCHORS.items(), key=lambda kv: kv[1])
    # → [(0.50, 0.411281), (0.70, 0.715983), ..., (0.99, 0.998796)]

    # Below the lowest anchor: below the median
    if p <= anchors[0][1]:
        return 0.50

    # Above the highest anchor: top 1%
    if p >= anchors[-1][1]:
        return 1.00

    # Interpolate between adjacent anchors
    for (lo_pct, lo_val), (hi_pct, hi_val) in zip(anchors[:-1], anchors[1:]):
        if lo_val <= p <= hi_val:
            frac = (p - lo_val) / (hi_val - lo_val)
            return round(lo_pct + frac * (hi_pct - lo_pct), 3)

    return 0.50  # unreachable safety net


def classify(probability: float, percentile_threshold: float = 0.70) -> str:
    """
    Binary classification against a percentile-based cutoff.
    Default corresponds to the top-30% operating point, which the notebook
    recommended as the balance between precision and recall.

    threshold 0.90 → top 10% (perfect precision, low recall)
    threshold 0.80 → top 20%
    threshold 0.70 → top 30% (recommended default)
    threshold 0.50 → top 50% (broad coverage)
    """
    anchors = _PERCENTILE_ANCHORS
    pct_to_cutoff = {1.0 - pct: val for pct, val in anchors.items()}
    cutoff = pct_to_cutoff.get(round(percentile_threshold, 2), 0.7160)
    return "Late risk - flag for intervention" if probability >= cutoff else "On-time — standard handling"


def predict(input_dict: dict) -> dict:
    """
    Score a single order.

    Parameters
    ----------
    input_dict : dict
        Must contain every field listed in the schema's categorical and numeric
        sections, plus 'product_image' (which the app auto-populates).

    Returns
    -------
    dict with keys:
        probability      - P(late) as a float in [0, 1].
        percentile_rank  - where this prediction sits in the test distribution.
        classification   - readable label.
        base_rate        - the model's overall late rate (0.5488).
        above_base_rate  - True if probability > base rate.
    """
    schema = _load_schema()
    model = _load_model()

    expected_cols = list(schema["categorical"].keys()) + list(schema["numeric"].keys())

    # Validate: every expected column must be present in the input.
    missing = [c for c in expected_cols if c not in input_dict]
    if missing:
        raise KeyError(f"predict() is missing required features: {missing}")

    # Build a single-row DataFrame with only the expected columns, in order.
    row = {c: input_dict[c] for c in expected_cols}
    X = pd.DataFrame([row])

    # Predict
    proba = float(model.predict_proba(X)[0, 1])

    return {
        "probability": proba,
        "percentile_rank": rank_percentile(proba),
        "classification": classify(proba),
        "base_rate": _BASE_RATE,
        "above_base_rate": proba > _BASE_RATE,
    }