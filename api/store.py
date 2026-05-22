"""Shared in-memory job store for the CodePulse API.

Provides a module-level dict and three helper functions so that the analyze and
results routes can share state without importing each other.

Job lifecycle: pending → running → complete | error
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


jobs: dict[str, dict[str, Any]] = {}


def create_job(repo_url: str) -> str:
    """Create a new job entry and return its job_id.

    Args:
        repo_url: The GitHub repository URL submitted by the user.

    Returns:
        A unique job_id string (hex UUID4).
    """
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "pending",
        "repo_url": repo_url,
        "files": [],
        "extended_files": [],
        "metrics": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    return job_id


def update_job(
    job_id: str,
    status: str,
    *,
    files: list[dict] | None = None,
    extended_files: list[dict] | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> None:
    """Update the status (and optionally the files, metrics, or error) of an existing job.

    Silently does nothing if job_id is not found — callers should not raise on
    a missing job during pipeline teardown.

    Args:
        job_id: The job to update.
        status: New status string — one of "pending", "running", "complete",
            or an error string starting with "error".
        files: If provided, replaces the job's file list (summary, for results endpoint).
        extended_files: If provided, replaces the job's extended file list (detail view data).
        metrics: If provided, replaces the job's metrics dict (auc_roc, avg_precision, pr_curve_json).
        error: If provided, stored under the "error" key.
    """
    if job_id not in jobs:
        return
    jobs[job_id]["status"] = status
    if files is not None:
        jobs[job_id]["files"] = files
    if extended_files is not None:
        jobs[job_id]["extended_files"] = extended_files
    if metrics is not None:
        jobs[job_id]["metrics"] = metrics
    if error is not None:
        jobs[job_id]["error"] = error


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return the job dict for job_id, or None if not found.

    Args:
        job_id: The job to look up.

    Returns:
        The job dict or None.
    """
    return jobs.get(job_id)
