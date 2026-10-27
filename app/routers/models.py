from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import GenerateImageRequest, GenerateVideoRequest, ModelInfo
from app.services.runner import jobs, registry
from app.services.stats import SECTIONS, Section, server_stats

router = APIRouter(prefix="/v1")


@router.get("/models", response_model=list[ModelInfo], tags=["models"])
def list_models():
    return list(registry.models.values())


def _submit(request: BaseModel, task: str) -> dict:
    # Model problems (unknown/disabled/wrong task) fail here, before queueing.
    adapter = registry.resolve(request.model, task)
    # dict(), not model_dump(): model_dump re-encodes Base64Bytes back to base64.
    job_id = jobs.submit(adapter, {k: v for k, v in dict(request).items() if k != "model"})
    return jobs.get(job_id)


@router.post("/generate/image", status_code=202, tags=["generation"])
def generate_image(request: GenerateImageRequest):
    return _submit(request, "image")


@router.post("/generate/video", status_code=202, tags=["generation"])
def generate_video(request: GenerateVideoRequest):
    return _submit(request, "video")


@router.get("/jobs", tags=["generation"])
def list_jobs(status: Literal["queued", "running", "completed", "failed"] | None = None):
    """All jobs, newest first."""
    return jobs.list(status)


@router.get("/stats", tags=["operations"])
def get_stats():
    """Every section at once."""
    return server_stats()


@router.get("/stats/{section}", tags=["operations"])
def get_stats_section(section: Section):
    """One section only, e.g. /v1/stats/gpu for a cheap poll."""
    return SECTIONS[section]()


@router.get("/jobs/{job_id}", tags=["generation"])
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    return job
