"""SHAP-based explanation generator for the CodePulse bug-risk predictions.

Uses TreeExplainer to compute per-file feature contributions and formats them
as human-readable driver strings suitable for display in the results dashboard.
"""
from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Human-readable format strings for every feature in the canonical feature list.
# Each value is a Python str.format-compatible template with a single {value}
# placeholder. Features that do not need the numeric value (boolean-like) may
# omit it.
_DRIVER_TEMPLATES: Final[dict[str, str]] = {
    "lines_changed_30d":        "high recent churn ({value:.0f} lines in 30d)",
    "lines_changed_60d":        "elevated 60d churn ({value:.0f} lines)",
    "lines_changed_90d":        "elevated 90d churn ({value:.0f} lines)",
    "commit_count_30d":         "frequent recent changes ({value:.0f} commits in 30d)",
    "commit_count_60d":         "frequent changes in 60d ({value:.0f} commits)",
    "commit_count_90d":         "frequent changes in 90d ({value:.0f} commits)",
    "churn_spike_ratio":        "churn spike ratio: {value:.1f}x",
    "total_commits":            "high total commit count ({value:.0f})",
    "total_lines_changed":      "large lifetime churn ({value:.0f} lines total)",
    "file_age_days":            "file age: {value:.0f} days",
    "days_since_last_change":   "last changed {value:.0f} days ago",
    "coupling_score_mean":      "tightly coupled to other files (mean score: {value:.2f})",
    "coupling_score_max":       "max coupling score: {value:.2f}",
    "coupled_high_churn":       "coupled to high-churn files",
    "unique_coupling_partners": "co-changes with {value:.0f} unique files",
    "unique_authors_total":     "touched by {value:.0f} authors total",
    "unique_authors_30d":       "active authors recently: {value:.0f}",
    "bus_factor_score":         "single author dominates (bus factor: {value:.2f})",
    "author_entropy":           "author entropy: {value:.2f}",
    "cyclomatic_complexity":    "cyclomatic complexity: {value:.1f}",
    "complexity_rank":          "complexity grade: {value:.0f}/6",
    "is_high_complexity":       "high code complexity (grade C or above)",
}

_TOP_N_DRIVERS: Final[int] = 3


def _format_driver(feature: str, value: float) -> str:
    """Format a single SHAP driver into a human-readable string."""
    template = _DRIVER_TEMPLATES.get(feature)
    if template is None:
        return f"{feature}: {value:.2f}"
    try:
        return template.format(value=value)
    except (KeyError, ValueError):
        return f"{feature}: {value:.2f}"


def explain_predictions(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: list[str],
    file_paths_test: list[str],
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Generate SHAP explanations for every file in the test set.

    Uses shap.TreeExplainer for efficient, exact Shapley values on the XGBoost
    model. For each file the top-3 features by absolute SHAP magnitude are
    formatted as human-readable driver strings. A global importance summary
    (mean absolute SHAP per feature) is also returned.

    Args:
        model: Fitted XGBClassifier from trainer.train_model.
        X_test: Feature matrix for the test split (rows = files, indexed by
            file_path).
        feature_names: Ordered list of feature column names matching X_test.
        file_paths_test: Ordered list of file paths matching X_test row order,
            as returned by merger.build_feature_matrix.

    Returns:
        drivers_map: Dict mapping file_path → list of top-3 driver strings,
            e.g. {"src/core/app.py": ["high recent churn (412 lines in 30d)",
            "churn spike ratio: 3.2x", "single author dominates (bus factor: 0.91)"]}.
        global_importance: Dict mapping feature_name → mean absolute SHAP value
            across the test set, ordered by importance descending.
    """
    if len(X_test) == 0:
        logger.warning("[explainer] Empty X_test — returning empty explanations.")
        return {}, {}

    try:
        tree_explainer = shap.TreeExplainer(model)
        shap_values: np.ndarray = tree_explainer.shap_values(X_test)
    except Exception as exc:
        logger.warning("[explainer] SHAP computation failed: %s", exc)
        fallback_drivers = {fp: [] for fp in file_paths_test}
        return fallback_drivers, {}

    # shap_values may be a list (binary classification) or a 2-D array.
    if isinstance(shap_values, list):
        # Take the positive-class SHAP values
        shap_matrix: np.ndarray = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values

    if shap_matrix.ndim != 2 or shap_matrix.shape[1] != len(feature_names):
        logger.warning(
            "[explainer] Unexpected SHAP shape %s for %d features — returning empty.",
            shap_matrix.shape,
            len(feature_names),
        )
        return {fp: [] for fp in file_paths_test}, {}

    # Per-file top-3 drivers
    drivers_map: dict[str, list[str]] = {}
    feature_array = np.array(feature_names)

    for i, fp in enumerate(file_paths_test):
        row_shap = shap_matrix[i]
        row_values = X_test.iloc[i].values

        top_indices = np.argsort(np.abs(row_shap))[::-1][:_TOP_N_DRIVERS]
        drivers: list[str] = []
        for idx in top_indices:
            feat = feature_array[idx]
            val = float(row_values[idx])
            drivers.append(_format_driver(feat, val))
        drivers_map[fp] = drivers

    # Global feature importance: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    global_importance: dict[str, float] = {
        feature_names[i]: float(mean_abs_shap[i]) for i in sorted_indices
    }

    logger.info(
        "[explainer] Top global features: %s",
        list(global_importance.keys())[:5],
    )

    return drivers_map, global_importance
