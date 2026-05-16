"""POST /analyze — accept a GitHub repo URL, start the ML pipeline, return a job_id.

The HTTP response is returned immediately with a job_id. The full pipeline runs
in a background daemon thread and updates the job store on completion or error.
The frontend polls GET /results/{job_id} until status == "complete".
"""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.store import create_job, update_job
from src.features.authorship import compute_authorship_features
from src.features.complexity import compute_complexity_features
from src.features.coupling import compute_coupling_features
from src.features.churn import compute_churn_features
from src.features.merger import build_feature_matrix
from src.mining.git_miner import mine_repo
from src.mining.labeler import label_bugfix_commits
from src.models.explainer import explain_predictions
from src.models.trainer import train_model

logger = logging.getLogger(__name__)

router = APIRouter()

_CACHE_DIR = Path("data/cache")


def _repo_slug(repo_url: str) -> str:
    """Return a filesystem-safe 'owner_name' slug from a GitHub URL."""
    parts = repo_url.rstrip("/").removesuffix(".git").split("/")
    owner = parts[-2] if len(parts) >= 2 else "unknown"
    name = parts[-1] if parts else "repo"
    return f"{owner}_{name}"


def _ensure_clone(repo_url: str, clone_dir: Path) -> None:
    """Clone the repo into clone_dir if it does not already exist.

    Uses --depth 1 because complexity.py only needs current file content;
    the full commit history is already in the joblib cache from mine_repo.

    Args:
        repo_url: Public GitHub repository URL.
        clone_dir: Target directory for the clone.
    """
    if clone_dir.exists():
        return
    logger.info("Cloning %s → %s (depth=1)", repo_url, clone_dir)
    result = subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", repo_url, str(clone_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed for {repo_url}: {result.stderr.strip()}"
        )


def _run_pipeline(job_id: str, repo_url: str) -> None:
    """Execute the full CodePulse ML pipeline and update the job store.

    Steps:
        1. Clone the repo (shallow) for on-disk file access.
        2. Mine the full commit history via PyDriller (or load from cache).
        3. Label bug-fix commits via regex.
        4. Compute churn, coupling, authorship, complexity features.
        5. Merge features and construct temporal train/test split with labels.
        6. Train XGBoost model.
        7. Generate SHAP explanations.
        8. Write ranked FileRisk list to the job store.

    Args:
        job_id: Store key to update throughout the pipeline.
        repo_url: GitHub repository URL to analyse.
    """
    try:
        update_job(job_id, "running")

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        slug = _repo_slug(repo_url)
        clone_dir = _CACHE_DIR / f"{slug}_clone"

        # Shallow clone for complexity analysis (idempotent)
        _ensure_clone(repo_url, clone_dir)

        # Step 1–2: Mine and label
        commits_df = mine_repo(repo_url, clone_dir=clone_dir, cache_dir=_CACHE_DIR)
        labeled_df = label_bugfix_commits(commits_df)

        # Step 3: Feature engineering
        churn_df = compute_churn_features(labeled_df)
        coupling_df = compute_coupling_features(labeled_df, churn_df)
        auth_df = compute_authorship_features(labeled_df)

        all_file_paths = labeled_df["file_path"].unique().tolist()
        complexity_df = compute_complexity_features(all_file_paths, clone_dir)

        # Step 4: Merge + temporal split
        X_train, X_test, y_train, y_test, feature_names, file_paths_test = (
            build_feature_matrix(labeled_df, churn_df, coupling_df, auth_df, complexity_df)
        )

        # Step 5: Train
        model, probas = train_model(X_train, y_train, X_test, y_test)

        # Step 6: SHAP explanations
        drivers, _ = explain_predictions(model, X_test, feature_names, file_paths_test)

        # Step 7: Build ranked FileRisk list
        # === MOCK DATA REMOVED — real risk scores from ML pipeline replace it ===
        files = sorted(
            [
                {
                    "file_path": fp,
                    "risk_score": round(float(prob) * 100, 1),
                    "top_drivers": drivers.get(fp, []),
                }
                for fp, prob in zip(file_paths_test, probas)
            ],
            key=lambda f: f["risk_score"],
            reverse=True,
        )

        update_job(job_id, "complete", files=files)
        logger.info("Pipeline complete for job %s — %d files ranked.", job_id, len(files))

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        update_job(job_id, f"error: {exc}", error=str(exc))


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Accept a repo URL, create a job, and start the pipeline in the background.

    Returns immediately with a job_id. The client should poll
    GET /results/{job_id} every few seconds until status == "complete".

    Args:
        request: AnalyzeRequest containing the GitHub repo URL.

    Returns:
        AnalyzeResponse with the job_id and a confirmation message.
    """
    job_id = create_job(request.repo_url)
    logger.info("Received analyze request for %s — job_id=%s", request.repo_url, job_id)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, request.repo_url),
        daemon=True,
        name=f"pipeline-{job_id[:8]}",
    )
    thread.start()

    return AnalyzeResponse(
        job_id=job_id,
        message="Analysis started. Poll /results/{job_id} for progress.",
    )
