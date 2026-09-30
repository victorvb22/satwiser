"""Satwiser read-only API.

All heavy computation happens offline in the pipeline; this service only serves the
exported aggregates, from local files (development) or Postgres (hosted).

    uvicorn satwiser_api.main:app --reload --app-dir backend
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from satwiser_api import schemas
from satwiser_api.config import Settings, load_settings
from satwiser_api.repository import KINDS, DatabaseRepository, LocalRepository, Repository

COPERNICUS = "Contains modified Copernicus Sentinel data 2014–2026"


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_repository() -> Repository:
    settings = get_settings()
    if settings.backend == "database":
        if not settings.database_url:
            raise RuntimeError("SATWISER_DATA_BACKEND=database requires DATABASE_URL")
        return DatabaseRepository(settings.database_url)
    return LocalRepository(settings.app_data)


Repo = Annotated[Repository, Depends(get_repository)]

app = FastAPI(title="Satwiser API", version="0.1.0",
              description="Read-only access to Sentinel-1 manoeuvre detection results. "
                          + COPERNICUS + ".")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().cors_origins),
                   allow_methods=["GET"], allow_headers=["*"])


def _not_found(what: str):
    raise HTTPException(status_code=404, detail=f"{what} not found")


@app.get("/api/health", response_model=schemas.Health)
def health(repo: Repo) -> dict:
    """Liveness probe; also touches the database (keeps a free Supabase project awake)."""
    repo.check()
    return {"status": "ok", "backend": get_settings().backend, "attribution": COPERNICUS}


@app.get("/api/satellites", response_model=list[schemas.Satellite])
def satellites(repo: Repo) -> list[dict]:
    return repo.satellites()


@app.get("/api/satellites/{satellite}", response_model=schemas.Satellite)
def satellite(satellite: str, repo: Repo) -> dict:
    for sat in repo.satellites():
        if sat["id"] == satellite:
            return sat
    _not_found("Satellite")


@app.get("/api/satellites/{satellite}/series", response_model=schemas.Series)
def series(satellite: str, repo: Repo,
           year: Annotated[int, Query(ge=2014, le=2100)]) -> dict:
    """Per-revolution mean semi-major axis (ground-track signature removed) for a year."""
    payload = repo.series(satellite, year)
    if payload is None:
        _not_found("Series")
    return payload


@app.get("/api/satellites/{satellite}/overview", response_model=schemas.Overview)
def overview(satellite: str, repo: Repo) -> dict:
    """Daily means over the whole mission (navigation mini-view)."""
    payload = repo.overview(satellite)
    if payload is None:
        _not_found("Overview")
    return payload


@app.get("/api/satellites/{satellite}/events", response_model=list[schemas.EventSummary])
def events(satellite: str, repo: Repo,
           year: Annotated[int | None, Query(ge=2014, le=2100)] = None,
           kind: Annotated[str | None, Query(pattern="^(" + "|".join(KINDS) + ")$")] = None
           ) -> list[dict]:
    """Detected, missed and false-alarm events (markers of the mission view)."""
    return repo.events(satellite, year, kind)


@app.get("/api/events/{event_id}", response_model=schemas.Event)
def event(event_id: str, repo: Repo) -> dict:
    payload = repo.event(event_id)
    if payload is None:
        _not_found("Event")
    return payload


@app.get("/api/robustness", response_model=schemas.Robustness)
def robustness(repo: Repo) -> dict:
    """Precomputed detection probability over noise × Δv × sampling × correlation."""
    payload = repo.robustness()
    if payload is None:
        _not_found("Robustness grid")
    return payload


@app.get("/api/metrics", response_model=schemas.Metrics)
def metrics(repo: Repo) -> dict:
    """Headline evaluation numbers of the main detector (from the pipeline reports)."""
    payload = repo.metrics()
    if payload is None:
        _not_found("Metrics")
    return payload


@app.get("/api/lab/{event_id}", response_model=schemas.LabWindow)
def lab_window(event_id: str, repo: Repo) -> dict:
    """Ten-day window of inertial states around an event, recomputed in the browser."""
    payload = repo.lab_window(event_id)
    if payload is None:
        _not_found("Lab window")
    return payload
