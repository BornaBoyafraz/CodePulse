"""GET /results/{job_id} — return stored risk results for a job."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import FileRisk, ResultsResponse
from api.store import get_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/results/{job_id}", response_model=ResultsResponse)
def get_results(job_id: str) -> ResultsResponse:
    """Return the current state of a job, including risk results when complete.

    The frontend polls this endpoint every 2 seconds. While the pipeline is
    running, status will be "pending" or "running" and files will be empty.
    On completion, status becomes "complete" and files contains the ranked
    risk list. On failure, status starts with "error".

    Args:
        job_id: UUID hex string returned by POST /analyze.

    Returns:
        ResultsResponse with job_id, status, repo_url, and files list.

    Raises:
        HTTPException 404: If job_id is not found in the store.
    """
    job = get_job(job_id)
    if job is None:
        logger.warning("Results requested for unknown job_id=%s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    files = [FileRisk(**f) for f in job.get("files", [])]

    return ResultsResponse(
        job_id=job_id,
        status=job["status"],
        repo_url=job["repo_url"],
        files=files,
    )
