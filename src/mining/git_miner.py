"""PyDriller-based commit history extractor for any public GitHub repository."""
from __future__ import annotations

import logging
from datetime import timezone
from pathlib import Path

import joblib
import pandas as pd
from pydriller import Repository
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp",
    ".cs", ".kt", ".swift",
})


def _repo_slug(repo_url: str) -> tuple[str, str]:
    """Return (owner, name) parsed from a GitHub URL."""
    parts = repo_url.rstrip("/").removesuffix(".git").split("/")
    return parts[-2], parts[-1]


def mine_repo(
    repo_url: str,
    clone_dir: str | Path | None = None,
    cache_dir: Path = Path("data/cache"),
) -> pd.DataFrame:
    """Extract the full commit history of a public GitHub repository.

    Each row represents one (commit, modified_file) pair. Merge commits are
    skipped. File paths are rename-resolved to their post-rename canonical form.

    Results are persisted to <cache_dir>/<owner>_<name>.joblib on first run
    and loaded from cache on subsequent calls.

    Args:
        repo_url:  Public GitHub URL, e.g. "https://github.com/pallets/flask".
        clone_dir: Optional local path to clone into. Uses a temp dir if None.
        cache_dir: Directory for joblib cache files.

    Returns:
        DataFrame with columns: commit_hash, commit_date, author_name, message,
        file_path, lines_added, lines_deleted, is_code_file.
    """
    owner, name = _repo_slug(repo_url)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{owner}_{name}.joblib"

    if cache_path.exists():
        df: pd.DataFrame = joblib.load(cache_path)
        logger.info("Cache hit — loaded %d rows from %s", len(df), cache_path)
        return df

    pydriller_kwargs: dict = {}
    if clone_dir is not None:
        pydriller_kwargs["clone_repo_to"] = str(clone_dir)

    rows: list[dict] = []

    for commit in tqdm(
        Repository(repo_url, **pydriller_kwargs).traverse_commits(),
        desc=f"mining {owner}/{name}",
        unit="commit",
    ):
        if commit.merge:
            continue
        try:
            author_date = commit.author_date
            if author_date.tzinfo is None:
                author_date = author_date.replace(tzinfo=timezone.utc)

            for mf in commit.modified_files:
                file_path = mf.new_path if mf.new_path is not None else mf.old_path
                if file_path is None:
                    continue
                ext = Path(file_path).suffix.lower()
                rows.append({
                    "commit_hash": commit.hash,
                    "commit_date": author_date,
                    "author_name": commit.author.name,
                    "message": commit.msg,
                    "file_path": file_path,
                    "lines_added": mf.added_lines,
                    "lines_deleted": mf.deleted_lines,
                    "is_code_file": ext in _CODE_EXTENSIONS,
                })
        except Exception as exc:
            logger.warning("Skipping commit %s: %s", commit.hash[:7], exc)

    if not rows:
        logger.warning("No commits extracted from %s", repo_url)
        return pd.DataFrame(columns=[
            "commit_hash", "commit_date", "author_name", "message",
            "file_path", "lines_added", "lines_deleted", "is_code_file",
        ])

    df = pd.DataFrame(rows)
    df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True)
    df = df.sort_values("commit_date").reset_index(drop=True)

    joblib.dump(df, cache_path, compress=3)
    logger.info("Mined %d rows → cached at %s", len(df), cache_path)
    return df
