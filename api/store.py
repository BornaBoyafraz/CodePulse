"""Shared in-memory job store.

Keyed by job_id (uuid4 string). Both routes import from here to avoid
coupling the analyze and results modules to each other.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.schemas import ResultsResponse

job_store: dict[str, ResultsResponse] = {}
