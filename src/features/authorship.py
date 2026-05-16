"""Authorship-based risk features for per-file code risk prediction.

Files touched by many authors tend to accumulate inconsistencies. Files dominated
by a single author are a bus factor risk — knowledge concentrates in one person.
All rolling windows are anchored to the most recent commit in the dataset, never
to datetime.now(), ensuring reproducibility across repeated runs on cached data.
"""
from __future__ import annotations

from typing import Final
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({
    "commit_hash",
    "commit_date",
    "author_name",
    "file_path",
})

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "unique_authors_total",
    "unique_authors_30d",
    "bus_factor_score",
    "author_entropy",
)

_INT_COLUMNS: Final[tuple[str, ...]] = (
    "unique_authors_total",
    "unique_authors_30d",
)


def _validate_input(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}"
        )


def compute_authorship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute authorship-based risk features per file.

    Bus factor and author diversity are strong predictors of future instability.
    A file owned by one person becomes high-risk when that person is unavailable
    or changes their patterns. Shannon entropy captures the full distribution of
    authorship — not just the top contributor.

    Args:
        df: Labeled commit DataFrame produced by mine_repo and
            label_bugfix_commits. Must contain: commit_hash, commit_date,
            author_name, file_path. commit_date must be timezone-aware (UTC).

    Returns:
        DataFrame indexed by file_path with columns:
            unique_authors_total: Number of distinct authors who ever committed
                to this file, counted by unique commit hashes per author.
            unique_authors_30d: Distinct authors in the 30-day window ending at
                ref_date = df["commit_date"].max().
            bus_factor_score: Fraction of all commits to this file made by the
                single most active author. Range 0–1; 1.0 means one person did
                everything.
            author_entropy: Shannon entropy (base-2) of the author commit
                distribution. Higher values indicate more distributed authorship.
                0.0 for single-author files.
    """
    _validate_input(df)

    if df.empty:
        return pd.DataFrame(
            columns=list(_OUTPUT_COLUMNS), dtype="float64"
        ).rename_axis("file_path")

    ref_date: pd.Timestamp = df["commit_date"].max()
    window_30d_start = ref_date - pd.Timedelta(days=30)

    # Deduplicate to one row per (file_path, commit_hash) — avoids inflating
    # counts for commits that touch the same file multiple times.
    df_unique = df[["commit_hash", "commit_date", "author_name", "file_path"]].drop_duplicates(
        subset=["commit_hash", "file_path"]
    )

    grouped = df_unique.groupby("file_path")

    # unique_authors_total — distinct author names across all commits to the file
    total_authors = grouped["author_name"].nunique().rename("unique_authors_total")

    # unique_authors_30d — distinct authors in the trailing 30-day window
    df_30d = df_unique[df_unique["commit_date"] >= window_30d_start]
    authors_30d = (
        df_30d.groupby("file_path")["author_name"].nunique().rename("unique_authors_30d")
    )

    # bus_factor_score and author_entropy require per-author commit counts
    bus_factors: dict[str, float] = {}
    entropies: dict[str, float] = {}

    for file_path, group in grouped:
        author_counts = group.groupby("author_name")["commit_hash"].nunique()
        total = int(author_counts.sum())
        if total == 0:
            bus_factors[file_path] = 0.0
            entropies[file_path] = 0.0
            continue

        bus_factors[file_path] = float(author_counts.max()) / total

        probs = author_counts / total
        # Small epsilon prevents log2(0); max(0) guards against floating-point -0.0
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        entropies[file_path] = max(0.0, entropy)

    bus_factor_series = pd.Series(bus_factors, name="bus_factor_score")
    entropy_series = pd.Series(entropies, name="author_entropy")

    features = pd.concat(
        [total_authors, authors_30d, bus_factor_series, entropy_series], axis=1
    )
    features.index.name = "file_path"
    features = features.fillna(0)
    features[list(_INT_COLUMNS)] = features[list(_INT_COLUMNS)].astype("int64")

    return features[list(_OUTPUT_COLUMNS)]
