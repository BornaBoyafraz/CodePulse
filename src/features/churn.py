"""Churn-based risk features for per-file code risk prediction.

Computes rolling churn, change frequency, churn spike detection, and
lifetime activity metrics for every file in a labeled commit DataFrame.
All windows are anchored to the most recent commit in the dataset — not
to today — so results are reproducible and free of future data leakage.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({
    "commit_hash",
    "commit_date",
    "file_path",
    "lines_added",
    "lines_deleted",
})

_WINDOWS_DAYS: Final[tuple[int, ...]] = (30, 60, 90)

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
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
)

_INT_COLUMNS: Final[tuple[str, ...]] = (
    "lines_changed_30d",
    "lines_changed_60d",
    "lines_changed_90d",
    "commit_count_30d",
    "commit_count_60d",
    "commit_count_90d",
    "total_commits",
    "total_lines_changed",
    "file_age_days",
    "days_since_last_change",
)


def _validate_input(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}"
        )


def _aggregate_window(
    df: pd.DataFrame,
    days: int,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return per-file lines_changed and commit_count for a trailing window.

    Args:
        df:       Commit DataFrame with a pre-computed 'lines_changed' column.
        days:     Window length in calendar days (30, 60, or 90).
        ref_date: Anchor date — the most recent commit_date in the full dataset.

    Returns:
        DataFrame indexed by file_path with columns 'lines_changed_{days}d'
        and 'commit_count_{days}d'. Files absent from the window are not
        included; the caller fills zeros via an outer join.
    """
    cutoff = ref_date - pd.Timedelta(days=days)
    window = df.loc[df["commit_date"] >= cutoff]
    grouped = window.groupby("file_path")
    return pd.DataFrame({
        f"lines_changed_{days}d": grouped["lines_changed"].sum(),
        f"commit_count_{days}d": grouped["commit_hash"].nunique(),
    })


def compute_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute churn-based risk features per file from a labeled commit DataFrame.

    The reference date is fixed at the maximum commit_date in the dataset,
    ensuring reproducibility across repeated calls on the same cached mining
    output and preventing future data leakage into the feature set.

    Args:
        df: Labeled commit DataFrame produced by mine_repo and
            label_bugfix_commits. Must contain at minimum the columns:
            commit_hash, commit_date, file_path, lines_added, lines_deleted.
            The commit_date column must be timezone-aware (UTC).

    Returns:
        DataFrame indexed by file_path with one row per unique file and
        the following columns (in order):

        Rolling window churn:
            lines_changed_30d, lines_changed_60d, lines_changed_90d

        Change frequency:
            commit_count_30d, commit_count_60d, commit_count_90d

        Spike detection:
            churn_spike_ratio — ratio of lines_changed_30d to
            (lines_changed_90d / 3). Defaults to 1.0 when lines_changed_90d
            is zero (no anomaly assumed for completely inactive files).

        Lifetime activity:
            total_commits, total_lines_changed, file_age_days,
            days_since_last_change

        All values are numeric. Files with no activity in a window receive 0,
        not NaN.

    Raises:
        ValueError: If any required column is absent from df.
    """
    _validate_input(df)

    if df.empty:
        return pd.DataFrame(
            columns=list(_OUTPUT_COLUMNS),
            dtype="float64",
        ).rename_axis("file_path")

    df = df.copy()
    df["lines_changed"] = df["lines_added"] + df["lines_deleted"]

    ref_date: pd.Timestamp = df["commit_date"].max()

    # Rolling window aggregations (30, 60, 90 days)
    window_frames = [_aggregate_window(df, d, ref_date) for d in _WINDOWS_DAYS]
    features = pd.concat(window_frames, axis=1)

    # Lifetime aggregations — timedelta math is done here, before the join,
    # to prevent fillna(0) from corrupting datetime columns.
    grouped = df.groupby("file_path")
    lifetime = pd.DataFrame({
        "total_commits": grouped["commit_hash"].nunique(),
        "total_lines_changed": grouped["lines_changed"].sum(),
        "file_age_days": (ref_date - grouped["commit_date"].min()).dt.days,
        "days_since_last_change": (ref_date - grouped["commit_date"].max()).dt.days,
    })

    features = features.join(lifetime, how="outer")
    features = features.fillna(0)
    features[list(_INT_COLUMNS)] = features[list(_INT_COLUMNS)].astype("int64")

    # Churn spike: 30-day churn relative to the expected 30-day slice of
    # the 90-day baseline. Default 1.0 (neutral) when 90-day activity is zero.
    features["churn_spike_ratio"] = np.where(
        features["lines_changed_90d"] > 0,
        features["lines_changed_30d"] / (features["lines_changed_90d"] / 3.0),
        1.0,
    )

    return features[list(_OUTPUT_COLUMNS)]
