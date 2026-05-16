"""Model evaluation utilities for the CodePulse bug-risk prediction pipeline.

Computes quantitative metrics and generates a Plotly precision-recall curve
serialised as JSON, suitable for serving via the FastAPI results endpoint.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probas: np.ndarray,
) -> tuple[dict[str, float], str]:
    """Compute evaluation metrics and build a Plotly precision-recall curve.

    Args:
        model: Fitted XGBClassifier (used only for type consistency; metrics
            are computed from probas directly).
        X_test: Feature matrix for the test split.
        y_test: Binary int64 ground-truth labels for the test split.
        probas: 1-D array of positive-class probabilities from model.predict_proba.

    Returns:
        metrics: Dict with keys auc_roc, avg_precision, threshold_05_precision,
            threshold_05_recall, threshold_05_f1 — all float, rounded to 4 dp.
        pr_curve_json: Plotly Figure serialised via fig.to_json(). Returns an
            empty JSON object string if evaluation cannot be run (e.g. only one
            class in y_test).
    """
    if len(y_test) == 0 or y_test.nunique() < 2:
        logger.warning(
            "[evaluator] Skipping evaluation — test set has fewer than 2 classes."
        )
        empty_metrics: dict[str, float] = {
            "auc_roc": 0.0,
            "avg_precision": 0.0,
            "threshold_05_precision": 0.0,
            "threshold_05_recall": 0.0,
            "threshold_05_f1": 0.0,
        }
        return empty_metrics, "{}"

    auc_roc = float(roc_auc_score(y_test, probas))
    avg_precision = float(average_precision_score(y_test, probas))

    preds_05 = (probas >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds_05, average="binary", zero_division=0
    )

    metrics: dict[str, float] = {
        "auc_roc": round(auc_roc, 4),
        "avg_precision": round(avg_precision, 4),
        "threshold_05_precision": round(float(prec), 4),
        "threshold_05_recall": round(float(rec), 4),
        "threshold_05_f1": round(float(f1), 4),
    }

    print(
        f"[evaluator] AUC-ROC:        {metrics['auc_roc']:.4f}\n"
        f"[evaluator] Avg Precision:  {metrics['avg_precision']:.4f}\n"
        f"[evaluator] Precision@0.5:  {metrics['threshold_05_precision']:.4f}\n"
        f"[evaluator] Recall@0.5:     {metrics['threshold_05_recall']:.4f}\n"
        f"[evaluator] F1@0.5:         {metrics['threshold_05_f1']:.4f}"
    )

    pr_curve_json = _build_pr_curve_json(y_test, probas, avg_precision)
    return metrics, pr_curve_json


def _build_pr_curve_json(
    y_test: pd.Series,
    probas: np.ndarray,
    avg_precision: float,
) -> str:
    """Build a dark-themed Plotly precision-recall curve and return it as JSON.

    Args:
        y_test: Binary ground-truth labels.
        probas: Predicted positive-class probabilities.
        avg_precision: Pre-computed average precision (shown in subtitle).

    Returns:
        JSON string from fig.to_json().
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probas)
    baseline = float(y_test.mean())

    fig = go.Figure()

    # Precision-recall curve
    fig.add_trace(go.Scatter(
        x=recall_vals,
        y=precision_vals,
        mode="lines",
        name=f"PR Curve (AP={avg_precision:.3f})",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    ))

    # No-skill baseline
    fig.add_hline(
        y=baseline,
        line=dict(color="#8b949e", width=1, dash="dash"),
        annotation_text=f"Baseline ({baseline:.3f})",
        annotation_position="right",
        annotation_font_color="#8b949e",
    )

    fig.update_layout(
        title=dict(
            text="Precision-Recall Curve",
            font=dict(family="Inter", size=14, color="#e6edf3"),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(
            title="Recall",
            title_font=dict(family="Inter", size=12, color="#8b949e"),
            tickfont=dict(family="Inter", size=11, color="#8b949e"),
            gridcolor="#21262d",
            linecolor="#30363d",
            range=[0, 1],
        ),
        yaxis=dict(
            title="Precision",
            title_font=dict(family="Inter", size=12, color="#8b949e"),
            tickfont=dict(family="Inter", size=11, color="#8b949e"),
            gridcolor="#21262d",
            linecolor="#30363d",
            range=[0, 1],
        ),
        legend=dict(
            font=dict(family="Inter", size=11, color="#e6edf3"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=40, t=50, b=50),
        font=dict(family="Inter", color="#e6edf3"),
    )

    return fig.to_json()
