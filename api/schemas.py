"""Pydantic request/response models for the CodePulse API."""
from __future__ import annotations

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    repo_url: str


class AnalyzeResponse(BaseModel):
    job_id: str
    message: str


class FileRisk(BaseModel):
    file_path: str
    risk_score: float
    top_drivers: list[str]


class ResultsResponse(BaseModel):
    job_id: str
    status: str
    repo_url: str
    files: list[FileRisk]
