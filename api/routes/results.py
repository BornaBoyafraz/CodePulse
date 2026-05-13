"""GET /results/{job_id} — return stored risk results for a completed job."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import ResultsResponse
from api.store import job_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/results/{job_id}", response_model=ResultsResponse)
def get_results(job_id: str) -> ResultsResponse:
    """Return the ResultsResponse for job_id, or 404 if not found."""
    result = job_store.get(job_id)
    if result is None:
        logger.warning("Results requested for unknown job_id=%s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return result
