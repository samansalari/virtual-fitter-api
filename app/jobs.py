from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from .schemas import RenderJobResponse


_JOBS: Dict[str, RenderJobResponse] = {}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def save_job(job: RenderJobResponse) -> RenderJobResponse:
    _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> RenderJobResponse | None:
    return _JOBS.get(job_id)


def update_job(job_id: str, **changes) -> RenderJobResponse:
    job = _JOBS[job_id]
    updated = job.model_copy(update={**changes, "updated_at": utc_now()})
    _JOBS[job_id] = updated
    return updated
