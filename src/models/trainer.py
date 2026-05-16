"""XGBoost model trainer for the CodePulse bug-risk prediction pipeline.

Trains a binary classifier on the merged feature matrix produced by merger.py.
Uses a strict temporal train/test split — no random shuffle is ever applied.
Class imbalance is handled via scale_pos_weight derived from the training set.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[XGBClassifier, np.ndarray]:
    """Train an XGBoost classifier and evaluate it on the temporal test split.

    Computes scale_pos_weight from the training set to handle class imbalance,
    trains with early stopping evaluated on the test set, then prints AUC-ROC,
    average precision, and a classification report at threshold 0.5.

    Args:
        X_train: Feature matrix for the training split, indexed by file_path.
        y_train: Binary int64 labels for the training split.
        X_test:  Feature matrix for the test split, indexed by file_path.
        y_test:  Binary int64 labels for the test split.

    Returns:
        model: The fitted XGBClassifier (best iteration selected via early
            stopping on AUC).
        probas: 1-D numpy array of predicted positive-class probabilities for
            X_test, in row order matching X_test.index.
    """
    if len(X_train) == 0:
        logger.warning("[trainer] Empty training set — returning untrained model.")
        model = XGBClassifier(random_state=42)
        probas = np.full(len(X_test), 0.5)
        return model, probas

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)

    # Guard against all-one-class edge case
    if pos_count == 0 or neg_count == 0:
        logger.warning(
            "[trainer] Training set has only one class (pos=%d, neg=%d). "
            "Skipping training — returning constant 0.5 predictions.",
            pos_count,
            neg_count,
        )
        model = XGBClassifier(random_state=42)
        probas = np.full(len(X_test), 0.5)
        return model, probas

    scale_pos_weight = max(neg_count / max(pos_count, 1), 1.0)
    logger.info(
        "[trainer] scale_pos_weight=%.2f (neg=%d, pos=%d)",
        scale_pos_weight, neg_count, pos_count,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    probas: np.ndarray = model.predict_proba(X_test)[:, 1]

    if len(y_test) > 0 and y_test.nunique() > 1:
        auc = roc_auc_score(y_test, probas)
        avg_prec = average_precision_score(y_test, probas)
        preds_05 = (probas >= 0.5).astype(int)
        report = classification_report(y_test, preds_05, zero_division=0)

        print(
            f"[trainer] AUC-ROC:        {auc:.4f}\n"
            f"[trainer] Avg Precision:  {avg_prec:.4f}\n"
            f"[trainer] Classification Report (threshold=0.5):\n{report}"
        )
    else:
        logger.warning(
            "[trainer] Test set has fewer than 2 classes — skipping metric computation."
        )

    return model, probas
