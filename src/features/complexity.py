"""Cyclomatic complexity features for Python source files.

Uses radon to measure code complexity per file. Non-Python files receive zero
values. Files that cannot be parsed are logged as warnings and also receive zero
values so the feature matrix remains complete.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final
import logging

import pandas as pd
from radon.complexity import cc_rank, cc_visit

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "cyclomatic_complexity",
    "complexity_rank",
    "is_high_complexity",
)

_RANK_MAP: Final[dict[str, int]] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
}

_HIGH_COMPLEXITY_THRESHOLD: Final[float] = 10.0


def _zero_row(file_path: str) -> dict:
    return {
        "file_path": file_path,
        "cyclomatic_complexity": 0.0,
        "complexity_rank": 1,
        "is_high_complexity": 0,
    }


def compute_complexity_features(
    file_paths: list[str],
    clone_path: str | Path,
) -> pd.DataFrame:
    """Compute cyclomatic complexity features for a list of repository files.

    Only Python (.py) files are analysed. All other file types and any files
    that fail to parse receive zero-valued rows so downstream code can safely
    concatenate without NaN gaps.

    Args:
        file_paths: Relative paths from the repository root, e.g.
            ["src/core/app.py", "docs/conf.py"]. Duplicates are silently
            deduplicated; non-Python paths are accepted and receive zeros.
        clone_path: Local filesystem path to the cloned repository directory.
            Each file is read from Path(clone_path) / file_path.

    Returns:
        DataFrame indexed by file_path with columns:
            cyclomatic_complexity: Average cyclomatic complexity across all
                functions in the file as reported by radon. 0.0 for files
                with no functions, non-Python files, or parse failures.
            complexity_rank: Integer 1 (grade A) through 6 (grade F) computed
                via radon.complexity.cc_rank on the average complexity.
            is_high_complexity: 1 if cyclomatic_complexity > 10, else 0.
    """
    if not file_paths:
        return pd.DataFrame(
            columns=list(_OUTPUT_COLUMNS), dtype="float64"
        ).rename_axis("file_path")

    clone_path = Path(clone_path)
    seen: set[str] = set()
    rows: list[dict] = []

    for file_path in file_paths:
        if file_path in seen:
            continue
        seen.add(file_path)

        if not file_path.endswith(".py"):
            rows.append(_zero_row(file_path))
            continue

        full_path = clone_path / file_path
        if not full_path.exists():
            logger.warning("File not on disk, skipping complexity: %s", file_path)
            rows.append(_zero_row(file_path))
            continue

        try:
            source = full_path.read_text(encoding="utf-8", errors="ignore")
            results = cc_visit(source)
            avg_cc = (
                sum(r.complexity for r in results) / len(results)
                if results
                else 0.0
            )
            rank_letter = cc_rank(avg_cc)
            rows.append({
                "file_path": file_path,
                "cyclomatic_complexity": float(avg_cc),
                "complexity_rank": _RANK_MAP.get(rank_letter, 1),
                "is_high_complexity": int(avg_cc > _HIGH_COMPLEXITY_THRESHOLD),
            })
        except Exception as exc:
            logger.warning("Failed to parse %s for complexity: %s", file_path, exc)
            rows.append(_zero_row(file_path))

    result = pd.DataFrame(rows).set_index("file_path")
    result["cyclomatic_complexity"] = result["cyclomatic_complexity"].astype("float64")
    result["complexity_rank"] = result["complexity_rank"].astype("int64")
    result["is_high_complexity"] = result["is_high_complexity"].astype("int64")
    return result[list(_OUTPUT_COLUMNS)]
