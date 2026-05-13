"""POST /analyze — accept a GitHub repo URL and return a job_id.

Mock implementation stores pre-built risk data immediately so the frontend
can render results on the first poll. The real ML pipeline replaces the mock
block below once the feature + model modules are wired end-to-end.
"""
from __future__ import annotations

import logging
import uuid
from typing import Final

from fastapi import APIRouter

from api.schemas import AnalyzeRequest, AnalyzeResponse, FileRisk, ResultsResponse
from api.store import job_store

logger = logging.getLogger(__name__)

router = APIRouter()

_MOCK_FILES: Final[tuple[dict, ...]] = (
    {
        "file_path": "src/core/app.py",
        "risk_score": 91.4,
        "top_drivers": ["extreme churn rate", "5 bug-fix commits", "single author (bus factor 1)"],
    },
    {
        "file_path": "src/db/migrations.py",
        "risk_score": 78.2,
        "top_drivers": ["spike in change frequency", "high coupling score", "3 bug-fix commits"],
    },
    {
        "file_path": "tests/test_models.py",
        "risk_score": 71.6,
        "top_drivers": ["frequent rewrites", "co-changes with high-risk files", "low author diversity"],
    },
    {
        "file_path": "src/api/endpoints.py",
        "risk_score": 63.8,
        "top_drivers": ["moderate churn", "complexity growth", "2 bug-fix commits"],
    },
    {
        "file_path": "src/auth/middleware.py",
        "risk_score": 57.1,
        "top_drivers": ["recent activity spike", "high cyclomatic complexity", "co-change coupling"],
    },
    {
        "file_path": "src/services/processor.py",
        "risk_score": 44.9,
        "top_drivers": ["irregular commit cadence", "moderate coupling", "1 bug-fix commit"],
    },
    {
        "file_path": "src/utils/validators.py",
        "risk_score": 29.3,
        "top_drivers": ["occasional churn", "stable authorship", "low complexity"],
    },
    {
        "file_path": "src/config/settings.py",
        "risk_score": 11.7,
        "top_drivers": ["rare changes", "single concern", "no bug-fix history"],
    },
)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Accept a repo URL, store mock risk results, and return a job_id."""
    job_id = str(uuid.uuid4())
    logger.info("Received analyze request for %s — job_id=%s", request.repo_url, job_id)

    # === REAL ML PIPELINE WIRES IN HERE ===
    # Replace the mock block below with:
    #   1. git_miner.mine_repo(request.repo_url) → commits_df
    #   2. labeler.label_bugfixes(commits_df) → labeled_df
    #   3. churn.compute_churn_features(labeled_df) → churn_df
    #   4. coupling.compute_coupling(labeled_df) → coupling_df
    #   5. authorship.compute_authorship(labeled_df) → auth_df
    #   6. complexity.compute_complexity(labeled_df) → complexity_df
    #   7. trainer.train(churn_df, coupling_df, auth_df, complexity_df) → model
    #   8. explainer.explain(model, ...) → shap_values
    #   9. Build ResultsResponse from real risk scores + SHAP top drivers
    # =========================================

    files = [FileRisk(**f) for f in _MOCK_FILES]
    result = ResultsResponse(
        job_id=job_id,
        status="complete",
        repo_url=request.repo_url,
        files=files,
    )
    job_store[job_id] = result
    return AnalyzeResponse(job_id=job_id, message="Analysis complete.")
