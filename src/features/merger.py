"""Feature matrix builder and temporal train/test splitter for CodePulse.

Merges all per-file feature DataFrames into one unified matrix and constructs
binary bug-risk labels using a strict temporal split. Training data never sees
future information — the cutoff is 6 months before the most recent commit.
"""
from __future__ import annotations

import warnings
from typing import Final

import pandas as pd

_FEATURE_COLUMNS: Final[list[str]] = [
    # Churn features (src/features/churn.py)
    "lines_changed_30d",
    "lines_changed_60d",
    "lines_changed_90d",
    "commit_count_30d",
    "commit_count_60d",
    "commit_count_90d",
    "churn_spike_ratio",
    "total_commits",
    "total_lines_changed",
    "file_age_days",
    "days_since_last_change",
    # Coupling features (src/features/coupling.py)
    "coupling_score_mean",
    "coupling_score_max",
    "coupled_high_churn",
    "unique_coupling_partners",
    # Authorship features (src/features/authorship.py)
    "unique_authors_total",
    "unique_authors_30d",
    "bus_factor_score",
    "author_entropy",
    # Complexity features (src/features/complexity.py)
    "cyclomatic_complexity",
    "complexity_rank",
    "is_high_complexity",
]


def build_feature_matrix(
    df: pd.DataFrame,
    churn_df: pd.DataFrame,
    coupling_df: pd.DataFrame,
    auth_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    """Merge features and construct a temporally-split train/test dataset.

    Combines all four per-file feature DataFrames into a single feature matrix,
    builds binary bug-risk labels from the labeled commit history, and splits
    into training and test sets using a strict temporal cutoff. No random split
    is ever used.

    Temporal logic:
        ref_date         = df["commit_date"].max()  (never datetime.now())
        cutoff           = ref_date - 6 months
        label_window_end = cutoff + 30 days

        train_files : files with at least one commit before cutoff
        test_files  : files with at least one commit on or after cutoff
        label = 1 iff file appears in an is_bugfix=True commit in
                [cutoff, label_window_end)

    Args:
        df: Full labeled commit DataFrame from mine_repo + label_bugfix_commits.
            Required columns: commit_hash, commit_date, file_path, is_bugfix.
            commit_date must be timezone-aware UTC.
        churn_df: Per-file churn features, indexed by file_path.
        coupling_df: Per-file coupling features, indexed by file_path.
        auth_df: Per-file authorship features, indexed by file_path.
        complexity_df: Per-file complexity features, indexed by file_path.

    Returns:
        X_train: Feature matrix for training files (DataFrame, rows = files,
            cols = feature columns, index = file_path).
        X_test: Feature matrix for test files (same structure as X_train).
        y_train: Binary int64 labels for training files (pd.Series).
        y_test: Binary int64 labels for test files (pd.Series).
        feature_names: Ordered list of feature column names used in X_train /
            X_test. Matches list(X_train.columns).
        file_paths_test: Ordered list of file paths in X_test, matching row
            order. Used to map model predictions back to file names.
    """
    # Reference date anchored to the dataset — never wall-clock time.
    ref_date: pd.Timestamp = df["commit_date"].max()
    cutoff = ref_date - pd.DateOffset(months=6)
    label_window_end = cutoff + pd.Timedelta(days=30)

    # Temporal file membership
    train_file_set = set(df.loc[df["commit_date"] < cutoff, "file_path"])
    test_file_set = set(df.loc[df["commit_date"] >= cutoff, "file_path"])

    # Label: file was touched in a bug-fix commit inside the label window
    bugfix_mask = (
        (df["is_bugfix"] == True)  # noqa: E712
        & (df["commit_date"] >= cutoff)
        & (df["commit_date"] < label_window_end)
    )
    labeled_files: set[str] = set(df.loc[bugfix_mask, "file_path"])

    # Merge all feature DataFrames along the shared file_path index
    features = pd.concat(
        [churn_df, coupling_df, auth_df, complexity_df], axis=1
    )
    features = features.fillna(0)

    # Guard against duplicate index entries (shouldn't occur with well-formed inputs)
    features = features[~features.index.duplicated(keep="first")]

    # complexity_rank 0 is not a valid radon grade; clamp to 1 (grade A) for
    # files that were absent from complexity_df (non-Python or unparseable).
    if "complexity_rank" in features.columns:
        features["complexity_rank"] = features["complexity_rank"].clip(lower=1)

    # Enforce canonical column order; gracefully skip columns not yet computed
    available_columns = [c for c in _FEATURE_COLUMNS if c in features.columns]
    features = features[available_columns]

    # Build labels for every file present in the feature matrix
    all_files = features.index
    y_all = pd.Series(
        [int(fp in labeled_files) for fp in all_files],
        index=all_files,
        dtype="int64",
        name="label",
    )

    train_idx = [fp for fp in all_files if fp in train_file_set]
    test_idx = [fp for fp in all_files if fp in test_file_set]

    X_train = features.loc[train_idx]
    X_test = features.loc[test_idx]
    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]

    feature_names: list[str] = list(features.columns)
    file_paths_test: list[str] = list(X_test.index)

    train_pos_rate = float(y_train.mean()) if len(y_train) > 0 else 0.0
    test_pos_rate = float(y_test.mean()) if len(y_test) > 0 else 0.0

    print(
        f"[merger] ref_date={ref_date.date()}, cutoff={cutoff.date()}, "
        f"label_window=[{cutoff.date()}, {label_window_end.date()})\n"
        f"[merger] train={len(X_train)} files, test={len(X_test)} files\n"
        f"[merger] train_pos_rate={train_pos_rate:.3f}, "
        f"test_pos_rate={test_pos_rate:.3f}"
    )

    if len(y_test) > 0 and y_test.sum() == 0:
        warnings.warn(
            "[merger] No positive labels in test set — model evaluation will be unreliable.",
            stacklevel=2,
        )

    return X_train, X_test, y_train, y_test, feature_names, file_paths_test
