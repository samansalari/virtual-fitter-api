from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..config import get_settings
from ..jobs import get_job, update_job
from ..schemas import RenderError, RenderResult, RenderWarning
from .placement import PlacementConfig, calculate_placement
from .renderer import RenderMode, get_rendering_service
from .segmentation import SegmentationResult, segment_vehicle
from .shopify_catalog import ProductRenderAssets, fetch_product_render_assets
from .validation import IncompatiblePlacementAngle, RenderTimeout, VirtualFitterError, get_user_guidance

logger = logging.getLogger(__name__)


@dataclass
class RenderJob:
    job_id: str
    status: str
    progress: int
    result_url: Optional[str]
    error: Optional[str]
    metadata: dict = field(default_factory=dict)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.array(image.convert("RGB"))


def _job_dir(job_id: str) -> Path:
    settings = get_settings()
    path = settings.storage_dir / "media" / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _media_url(job_id: str, filename: str) -> str:
    settings = get_settings()
    return f"{settings.media_base_url}/jobs/{job_id}/{filename}"


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _save_jpg(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path, format="JPEG", quality=92)


def _placement_quality_score(label: str) -> float:
    return {"high": 0.95, "medium": 0.8, "low": 0.62}.get(label, 0.75)


def _combined_confidence(segmentation: SegmentationResult, placement_quality: str) -> float:
    return round((segmentation.confidence * 0.7) + (_placement_quality_score(placement_quality) * 0.3), 3)


def _resolve_render_mode(request_overrides: dict[str, Optional[str]]) -> RenderMode:
    requested_mode = (request_overrides.get("render_mode") or get_settings().render_mode or "overlay").strip().lower()
    try:
        return RenderMode(requested_mode)
    except ValueError:
        return RenderMode.OVERLAY


def generate_placement_mask(vehicle_mask: np.ndarray, placement, placement_zone: str) -> np.ndarray:
    mask = np.zeros_like(vehicle_mask, dtype=np.uint8)
    if placement.perspective_warp is not None:
        cv2.fillConvexPoly(mask, placement.perspective_warp.astype(np.int32), 255)
    else:
        x1, y1, x2, y2 = placement.target_bbox
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255

    mask = cv2.bitwise_and(mask, vehicle_mask.astype(np.uint8) * 255)
    if placement_zone == "full_body":
        mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, vehicle_mask.astype(np.uint8) * 255)
    if placement.feather_radius > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, placement.feather_radius / 2))
    return mask


def _validate_compatibility(segmentation: SegmentationResult, assets: ProductRenderAssets) -> list[RenderWarning]:
    warnings: list[RenderWarning] = []
    if not assets.compatible_models:
        return warnings

    compatible_blob = " ".join(assets.compatible_models).lower()
    detected_vehicle = (segmentation.detected_vehicle_type or "").lower()
    style_map = {
        "sedan": ("sedan", "saloon", "f30", "g20", "a4", "c class"),
        "suv": ("suv", "x5", "x3", "q5", "glc", "tiguan", "range rover"),
        "coupe": ("coupe", "f32", "m4", "c coupe"),
        "hatchback": ("hatch", "golf", "focus", "a class", "1 series"),
    }

    explicitly_supported_styles = {style for style, tokens in style_map.items() if any(token in compatible_blob for token in tokens)}
    if detected_vehicle and explicitly_supported_styles and detected_vehicle not in explicitly_supported_styles:
        raise IncompatiblePlacementAngle("The detected vehicle body style doesn't match this product's compatibility setup.")

    if not explicitly_supported_styles:
        warnings.append(
            RenderWarning(
                code="VF_W03",
                message="Compatibility was checked with coarse vehicle heuristics only. Exact make-model matching is a later phase.",
            )
        )
    return warnings


async def process_render_job(
    job_id: str,
    image_bytes: bytes,
    product_id: str,
    variant_id: str,
    shop_domain: str,
    request_overrides: Optional[dict[str, Optional[str]]] = None,
) -> RenderJob:
    started_at = time.perf_counter()
    request_overrides = request_overrides or {}
    warnings: list[RenderWarning] = []
    selected_render_mode = _resolve_render_mode(request_overrides)
    job_dir = _job_dir(job_id)
    input_path = job_dir / "input.jpg"
    input_path.write_bytes(image_bytes)

    try:
        logger.info(
            "[%s] Pipeline start: product_id=%s variant_id=%s shop_domain=%s render_mode=%s",
            job_id,
            product_id,
            variant_id,
            shop_domain,
            selected_render_mode.value,
        )
        update_job(job_id, status="segmenting", progress=12, message="Loading product assets and segmenting the vehicle.")
        assets = await fetch_product_render_assets(shop_domain, product_id, variant_id, overrides=request_overrides)
        logger.info(
            "[%s] Assets loaded: product_type=%s placement_zone=%s overlay_url=%s",
            job_id,
            assets.product_type,
            assets.placement_zone,
            assets.overlay_url,
        )

        segmentation = await segment_vehicle(image_bytes)
        logger.info(
            "[%s] Segmentation complete: source=%s confidence=%.3f angle=%s vehicle=%s",
            job_id,
            segmentation.source,
            segmentation.confidence,
            segmentation.detected_angle,
            segmentation.detected_vehicle_type,
        )
        if segmentation.confidence < 0.82:
            warnings.append(
                RenderWarning(
                    code="VF_W01",
                    message="For best results, use a clearer side or rear photo with the full car visible.",
                )
            )

        warnings.extend(_validate_compatibility(segmentation, assets))

        mask_path = job_dir / "vehicle-mask.png"
        _save_png(mask_path, (segmentation.vehicle_mask * 255).astype(np.uint8))

        update_job(
            job_id,
            status="placing",
            progress=46,
            message="Calculating placement for the selected product.",
            warnings=warnings,
            metadata={
                "segmentation_source": segmentation.source,
                "segmentation_confidence": segmentation.confidence,
                "detected_angle": segmentation.detected_angle,
                "detected_vehicle": segmentation.detected_vehicle_type,
            },
        )

        base_image = _decode_image(image_bytes)
        placement = calculate_placement(
            segmentation=segmentation,
            config=PlacementConfig(
                product_type=assets.product_type,
                placement_zone=assets.placement_zone,
                anchors=assets.anchors,
                overlay_url=assets.overlay_url,
                mask_url=assets.mask_url,
                compatible_models=assets.compatible_models,
                render_prompt=assets.render_prompt,
            ),
            image_dimensions=(base_image.shape[1], base_image.shape[0]),
        )
        logger.info(
            "[%s] Placement complete: quality=%s bbox=%s",
            job_id,
            placement.placement_quality,
            placement.target_bbox,
        )

        if placement.placement_quality != "high":
            warnings.append(
                RenderWarning(
                    code="VF_W02",
                    message="Placement quality is lower for this angle. A cleaner matching angle photo should improve the fit.",
                )
            )

        update_job(job_id, status="rendering", progress=74, message="Rendering your product preview.", warnings=warnings)

        placement_mask = generate_placement_mask(segmentation.vehicle_mask, placement, assets.placement_zone)
        placement_mask_path = job_dir / "placement-mask.png"
        _save_png(placement_mask_path, placement_mask.astype(np.uint8))

        rendering_service = get_rendering_service()
        render_execution = await rendering_service.render(
            car_image=base_image,
            car_image_bytes=image_bytes,
            product_image_url=assets.overlay_url,
            mask=placement_mask,
            placement=placement,
            product_type=assets.product_type,
            placement_zone=assets.placement_zone,
            prompt_hint=assets.render_prompt,
            mode=selected_render_mode,
        )
        logger.info(
            "[%s] Render complete: mode=%s model=%s result_url=%s bytes=%s",
            job_id,
            render_execution.mode.value,
            render_execution.model,
            bool(render_execution.result_url),
            bool(render_execution.result_bytes),
        )

        result_path = job_dir / "result.jpg"
        if render_execution.result_bytes:
            result_path.write_bytes(render_execution.result_bytes)
            result_url = _media_url(job_id, result_path.name)
        elif render_execution.result_url:
            result_url = render_execution.result_url
        else:
            raise RenderTimeout("The renderer completed without returning an image.")

        for warning_message in render_execution.warnings:
            warnings.append(RenderWarning(code="VF_W04", message=warning_message))

        result = RenderResult(
            image_url=result_url,
            confidence=round((_combined_confidence(segmentation, placement.placement_quality) * 0.55) + (render_execution.confidence * 0.45), 3),
            detected_vehicle=segmentation.detected_vehicle_type,
            detected_angle=segmentation.detected_angle,
            placement_quality=placement.placement_quality,  # type: ignore[arg-type]
            render_mode=render_execution.mode.value,
            render_model=render_execution.model,
            processing_time_ms=render_execution.processing_time_ms,
            cost_usd=render_execution.cost_usd,
            segmentation_mask_url=_media_url(job_id, mask_path.name),
            vehicle_bbox=list(segmentation.vehicle_bbox),
        )

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        update_job(
            job_id,
            status="complete",
            progress=100,
            message="Preview ready.",
            result=result,
            warnings=warnings,
            metadata={
                "timings_ms": {"total": elapsed_ms},
                "segmentation_source": segmentation.source,
                "segmentation_confidence": segmentation.confidence,
                "placement_zone": assets.placement_zone,
                "placement_quality": placement.placement_quality,
                "product_type": assets.product_type,
                "render_mode": render_execution.mode.value,
                "render_model": render_execution.model,
                "render_cost_usd": render_execution.cost_usd,
                "render_time_ms": render_execution.processing_time_ms,
            },
        )

        return RenderJob(
            job_id=job_id,
            status="complete",
            progress=100,
            result_url=result_url,
            error=None,
            metadata={"elapsed_ms": elapsed_ms},
        )
    except VirtualFitterError as exc:
        logger.error("[%s] Virtual fitter error: %s %s", job_id, exc.code, exc.message)
        job = get_job(job_id)
        progress = min((job.progress if job else 0) + 4, 96)
        guidance = get_user_guidance(exc.code, detected_angle=exc.detected_angle, product_zone=exc.product_zone)
        update_job(
            job_id,
            status="failed",
            progress=progress,
            message=exc.message,
            error=RenderError(code=exc.code, status="failed", message=exc.message, recoverable=exc.recoverable),
            warnings=warnings,
            metadata={
                **(job.metadata if job else {}),
                "guidance": exc.guidance,
                "user_guidance": guidance,
                "detected": {
                    "angle": exc.detected_angle,
                    "confidence": exc.confidence,
                },
            },
        )
        return RenderJob(job_id=job_id, status="failed", progress=progress, result_url=None, error=exc.message, metadata={"code": exc.code})
    except TimeoutError as exc:
        logger.exception("[%s] Timeout error", job_id)
        timeout_error = RenderTimeout()  # pragma: no cover - defensive contract
        update_job(
            job_id,
            status="failed",
            progress=96,
            message=timeout_error.message,
            error=RenderError(code=timeout_error.code, status="failed", message=timeout_error.message, recoverable=True),
            warnings=warnings,
        )
        return RenderJob(job_id=job_id, status="failed", progress=96, result_url=None, error=str(exc), metadata={"code": timeout_error.code})
    except Exception as exc:  # pragma: no cover - integration safety
        logger.exception("[%s] Unexpected pipeline error", job_id)
        update_job(
            job_id,
            status="failed",
            progress=96,
            message="The render pipeline failed unexpectedly.",
            error=RenderError(code="VF_999", status="failed", message="The render pipeline failed unexpectedly.", recoverable=False),
            warnings=warnings,
        )
        return RenderJob(job_id=job_id, status="failed", progress=96, result_url=None, error=str(exc), metadata={"code": "VF_999"})
