from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


JobStatus = Literal["pending", "segmenting", "placing", "rendering", "complete", "failed"]
PlacementQuality = Literal["high", "medium", "low"]


class RenderWarning(BaseModel):
    code: str
    message: str


class RenderError(BaseModel):
    code: str
    status: JobStatus
    message: str
    recoverable: bool = False


class RenderResult(BaseModel):
    image_url: str
    confidence: float
    detected_vehicle: Optional[str] = None
    detected_angle: Optional[str] = None
    placement_quality: PlacementQuality
    render_mode: Optional[str] = None
    render_model: Optional[str] = None
    processing_time_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    segmentation_mask_url: Optional[str] = None
    vehicle_bbox: Optional[list[int]] = None


class RenderJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    message: str = ""
    result: Optional[RenderResult] = None
    warnings: list[RenderWarning] = Field(default_factory=list)
    error: Optional[RenderError] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
