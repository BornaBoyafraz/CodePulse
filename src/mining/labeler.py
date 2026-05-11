"""Bug-fix commit identifier via commit message regex matching."""
from __future__ import annotations

import pandas as pd

_BUGFIX_REGEX = (
    r"\b(fix(?:e[ds])?|bug|error|defect|patch|issue|crash|"
    r"fault|repair|resolve[ds]?|hotfix)\b"
)


def label_bugfix_commits(df: pd.DataFrame) -> pd.DataFrame:
    """Add an is_bugfix column based on keyword matching in commit messages.

    Performs a case-insensitive, word-boundary-anchored regex search on the
    message column. Returns a new DataFrame; the input is never mutated.

    Args:
        df: DataFrame produced by mine_repo, must contain a 'message' column.

    Returns:
        Copy of df with a new boolean column 'is_bugfix'.
    """
    result = df.copy()
    result["is_bugfix"] = result["message"].str.contains(
        _BUGFIX_REGEX, case=False, regex=True, na=False
    )
    return result


def summary(df: pd.DataFrame) -> None:
    """Print bug-fix commit statistics and top-10 most bug-prone files.

    Expects df to already have the is_bugfix column (call label_bugfix_commits
    first).

    Args:
        df: Labeled DataFrame from label_bugfix_commits.
    """
    commits = df.drop_duplicates("commit_hash")
    total = len(commits)
    n_bugfix = int(commits["is_bugfix"].sum())
    pct = n_bugfix / total * 100 if total > 0 else 0.0

    print(f"Total commits   : {total:,}")
    print(f"Bug-fix commits : {n_bugfix:,}  ({pct:.1f}%)")

    top_files = (
        df[df["is_bugfix"]]
        .groupby("file_path")["commit_hash"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 files by bug-fix commit count:")
    for rank, (path, count) in enumerate(top_files.items(), start=1):
        print(f"  {rank:2}. {count:4d}  {path}")
