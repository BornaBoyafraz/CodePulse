"""GET /results/{job_id} — return stored risk results for a job."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from api.schemas import FileDetailResponse, FileRisk, MetricsResponse, ResultsResponse
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


@router.get("/results/{job_id}/metrics", response_model=MetricsResponse)
def get_metrics(job_id: str) -> MetricsResponse:
    """Return evaluation metrics and PR curve JSON for a completed job.

    Returns zero values gracefully if the job is still running or has no
    metrics stored (e.g. test set had only one class).

    Args:
        job_id: UUID hex string returned by POST /analyze.

    Returns:
        MetricsResponse with auc_roc, avg_precision, and pr_curve_json.

    Raises:
        HTTPException 404: If job_id is not found in the store.
    """
    job = get_job(job_id)
    if job is None:
        logger.warning("Metrics requested for unknown job_id=%s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    m = job.get("metrics", {})
    return MetricsResponse(
        job_id=job_id,
        auc_roc=float(m.get("auc_roc", 0.0)),
        avg_precision=float(m.get("avg_precision", 0.0)),
        pr_curve_json=m.get("pr_curve_json", "{}"),
    )


@router.get("/results/{job_id}/file", response_model=FileDetailResponse)
def get_file_detail(
    job_id: str,
    path: str = Query(..., description="URL-encoded file path to look up"),
) -> FileDetailResponse:
    """Return extended details for a single file from a completed job.

    Args:
        job_id: UUID hex string returned by POST /analyze.
        path: The exact file_path string to look up (URL-encoded by the client).

    Returns:
        FileDetailResponse with risk score, all SHAP drivers, and feature stats.

    Raises:
        HTTPException 404: If job_id or file_path is not found.
    """
    job = get_job(job_id)
    if job is None:
        logger.warning("File detail requested for unknown job_id=%s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    extended_files: list[dict] = job.get("extended_files", [])
    matched = next((f for f in extended_files if f["file_path"] == path), None)

    if matched is None:
        logger.warning("File '%s' not found in job %s", path, job_id)
        raise HTTPException(status_code=404, detail=f"File '{path}' not found in job results.")

    return FileDetailResponse(job_id=job_id, **matched)
