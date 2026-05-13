"""CodePulse FastAPI application entry point.

Run from the project root:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import analyze, results

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="CodePulse",
    description="AI-powered predictive code risk system.",
    version="0.3.0",
)

# Allow all origins during development — tighten before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(results.router)

# Static mount last — explicit API routes above take precedence.
# html=True makes index.html the fallback for any unmatched path (SPA mode).
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
