"""File co-change coupling features for per-file code risk prediction.

Computes how frequently files change together in the same commits. Tightly coupled
files share risk — if one breaks, the other often follows. All logic is anchored
to the commit dataset itself; no wall-clock time is ever used.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Final
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({
    "commit_hash",
    "file_path",
})

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "coupling_score_mean",
    "coupling_score_max",
    "coupled_high_churn",
    "unique_coupling_partners",
)

_MAX_FILES_PER_COMMIT: Final[int] = 30


def _validate_input(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}"
        )


def compute_coupling_features(
    df: pd.DataFrame,
    churn_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute file coupling features from a labeled commit DataFrame.

    Two files are 'coupled' if they frequently appear together in the same
    commits. Tightly coupled files share risk — if one is bug-prone, its
    partners inherit elevated risk. Commits touching more than 30 files are
    excluded as mass refactors that produce noise rather than meaningful signal.

    Args:
        df: Labeled commit DataFrame produced by mine_repo and
            label_bugfix_commits. Must contain at minimum: commit_hash,
            file_path.
        churn_df: Per-file churn features indexed by file_path, as produced
            by compute_churn_features. Must contain total_lines_changed for
            the coupled_high_churn calculation.

    Returns:
        DataFrame indexed by file_path with columns:
            coupling_score_mean: Average co-change frequency with the top-5
                coupling partners, normalized by total unique commits in the
                dataset. If fewer than 5 partners exist, averages what is
                available.
            coupling_score_max: Maximum co-change count with any single other
                file, same normalization. 0.0 for files with no coupling.
            coupled_high_churn: 1 if any of the file's top-3 coupling
                partners is in the top 25% by total_lines_changed from
                churn_df, else 0.
            unique_coupling_partners: Count of unique files this file has
                co-changed with at least once.
    """
    _validate_input(df)

    if df.empty:
        return pd.DataFrame(
            columns=list(_OUTPUT_COLUMNS), dtype="float64"
        ).rename_axis("file_path")

    total_commits = max(df["commit_hash"].nunique(), 1)

    # Build per-commit file lists, then filter mass-refactor commits.
    commit_files = df.groupby("commit_hash")["file_path"].apply(list)
    total_commit_groups = len(commit_files)
    commit_files = commit_files[commit_files.apply(len) <= _MAX_FILES_PER_COMMIT]
    skipped = total_commit_groups - len(commit_files)
    if skipped:
        logger.info(
            "[coupling] Skipped %d mass-refactor commits (>%d files touched)",
            skipped, _MAX_FILES_PER_COMMIT,
        )

    # Count co-changes for every ordered pair (f1 < f2 lexicographically).
    co_change_counts: dict[tuple[str, str], int] = defaultdict(int)
    for files in commit_files:
        for f1, f2 in combinations(sorted(set(files)), 2):
            co_change_counts[(f1, f2)] += 1

    # Build per-file partner lookup: file -> {partner: raw_count}.
    file_partners: dict[str, dict[str, int]] = defaultdict(dict)
    for (f1, f2), count in co_change_counts.items():
        file_partners[f1][f2] = count
        file_partners[f2][f1] = count

    # High-churn threshold: top 25% of files by lifetime lines changed.
    high_churn_files: set[str] = set()
    if "total_lines_changed" in churn_df.columns and not churn_df.empty:
        threshold = churn_df["total_lines_changed"].quantile(0.75)
        high_churn_files = set(
            churn_df[churn_df["total_lines_changed"] >= threshold].index
        )

    all_files = df["file_path"].unique()
    rows: list[dict] = []

    for fp in all_files:
        partners = file_partners.get(fp, {})
        if not partners:
            rows.append({
                "file_path": fp,
                "coupling_score_mean": 0.0,
                "coupling_score_max": 0.0,
                "coupled_high_churn": 0,
                "unique_coupling_partners": 0,
            })
            continue

        sorted_partners = sorted(partners.items(), key=lambda x: x[1], reverse=True)

        top5_counts = [cnt for _, cnt in sorted_partners[:5]]
        score_mean = (sum(top5_counts) / len(top5_counts)) / total_commits

        score_max = sorted_partners[0][1] / total_commits

        top3_partners = [p for p, _ in sorted_partners[:3]]
        coupled_hc = int(any(p in high_churn_files for p in top3_partners))

        rows.append({
            "file_path": fp,
            "coupling_score_mean": score_mean,
            "coupling_score_max": score_max,
            "coupled_high_churn": coupled_hc,
            "unique_coupling_partners": len(partners),
        })

    result = pd.DataFrame(rows).set_index("file_path")
    result = result.fillna(0)
    return result[list(_OUTPUT_COLUMNS)]
