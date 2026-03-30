from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import get_settings
from ..jobs import get_job, update_job
from ..schemas import RenderError, RenderResult, RenderWarning
from .content_moderation import RejectionReason, validate_uploaded_image
from .placement import PlacementConfig, calculate_placement
from .renderer import RenderMode, get_rendering_service
from .segmentation import SegmentationResult, segment_vehicle
from .shopify_catalog import ProductRenderAssets, fetch_product_render_assets
from .validation import (
    HumanFaceDetected,
    ImageTooSmall,
    IncompatiblePlacementAngle,
    InvalidUploadFormat,
    OverlayAssetNotFound,
    RenderTimeout,
    SensitiveContentDetected,
    VirtualFitterError,
    get_user_guidance,
)

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


def _bytes_to_data_url(payload: bytes, media_type: str) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _detect_media_type(payload: bytes) -> str:
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _image_array_to_data_url(array: np.ndarray, *, format: str = "JPEG", quality: int = 92) -> str:
    image = Image.fromarray(array.astype(np.uint8))
    buffer = io.BytesIO()
    save_kwargs: dict[str, object] = {"format": format}
    if format.upper() in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
    image.save(buffer, **save_kwargs)
    media_type = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return _bytes_to_data_url(buffer.getvalue(), media_type)


def _placement_quality_score(label: str) -> float:
    return {"high": 0.95, "medium": 0.8, "low": 0.62}.get(label, 0.75)


def _combined_confidence(segmentation: SegmentationResult, placement_quality: str) -> float:
    return round((segmentation.confidence * 0.7) + (_placement_quality_score(placement_quality) * 0.3), 3)


def _format_vehicle_style(style: Optional[str]) -> Optional[str]:
    if not style:
        return None
    labels = {
        "suv": "SUV",
        "sedan": "Sedan",
        "coupe": "Coupe",
        "hatchback": "Hatchback",
        "wagon": "Wagon",
        "truck": "Truck",
        "convertible": "Convertible",
    }
    return labels.get(style.lower(), style.replace("_", " ").title())


def _infer_vehicle_display_name(
    product_title: Optional[str],
    variant_title: Optional[str],
    fallback_style: Optional[str],
) -> Optional[str]:
    context = " ".join(part for part in [product_title, variant_title] if part).strip()
    if not context:
        return _format_vehicle_style(fallback_style)

    lowered = context.lower()
    brand = None
    brand_patterns = (
        ("BMW", ("bmw",)),
        ("Mercedes", ("mercedes", "benz", "amg")),
        ("Audi", ("audi",)),
        ("Volkswagen", ("volkswagen", "vw", "golf", "tiguan")),
        ("Porsche", ("porsche",)),
        ("Toyota", ("toyota",)),
        ("Ford", ("ford", "mustang", "focus", "fiesta")),
    )
    for brand_name, tokens in brand_patterns:
        if any(token in lowered for token in tokens):
            brand = brand_name
            break

    model = None
    model_patterns = (
        ("BMW", (
            ("8 Series", ("8 series",)),
            ("7 Series", ("7 series",)),
            ("6 Series", ("6 series",)),
            ("5 Series", ("5 series", "g30", "f10", "g60")),
            ("4 Series", ("4 series", "g22", "f32", "f33", "f36")),
            ("3 Series", ("3 series", "g20", "g21", "f30", "f31", "e90", "e91", "e92", "e93")),
            ("2 Series", ("2 series", "f22", "g42")),
            ("1 Series", ("1 series", "f20", "f21")),
            ("M5", ("m5",)),
            ("M4", ("m4",)),
            ("M3", ("m3",)),
            ("X7", ("x7", "g07")),
            ("X6", ("x6", "g06", "f16")),
            ("X5", ("x5", "g05", "f15")),
            ("X4", ("x4", "g02", "f26")),
            ("X3", ("x3", "g01", "f25")),
            ("X2", ("x2", "f39", "u10")),
            ("X1", ("x1", "f48", "u11")),
        )),
        ("Mercedes", (
            ("C-Class", ("c class", "c-class", "w205", "w206")),
            ("E-Class", ("e class", "e-class", "w213", "w214")),
            ("A-Class", ("a class", "a-class", "w177")),
            ("CLA", ("cla",)),
            ("GLC", ("glc", "x253", "x254")),
            ("GLE", ("gle", "w167")),
        )),
        ("Audi", (
            ("A5", ("a5",)),
            ("A4", ("a4",)),
            ("A3", ("a3",)),
            ("Q7", ("q7",)),
            ("Q5", ("q5",)),
            ("Q3", ("q3",)),
        )),
        ("Volkswagen", (
            ("Golf", ("golf",)),
            ("Tiguan", ("tiguan",)),
            ("Polo", ("polo",)),
        )),
        ("Porsche", (
            ("911", ("911", "992", "991")),
            ("Cayenne", ("cayenne",)),
            ("Macan", ("macan",)),
            ("Panamera", ("panamera",)),
        )),
        ("Ford", (
            ("Mustang", ("mustang",)),
            ("Focus", ("focus",)),
            ("Fiesta", ("fiesta",)),
            ("Ranger", ("ranger",)),
        )),
        ("Toyota", (
            ("Supra", ("supra",)),
            ("GR Yaris", ("gr yaris",)),
            ("Corolla", ("corolla",)),
            ("Hilux", ("hilux",)),
        )),
    )

    for expected_brand, candidates in model_patterns:
        if brand and expected_brand != brand:
            continue
        for candidate_name, tokens in candidates:
            if any(token in lowered for token in tokens):
                model = candidate_name
                if not brand:
                    brand = expected_brand
                break
        if model:
            break

    if brand and model:
        return f"{brand} {model}"
    if brand:
        style = _format_vehicle_style(fallback_style)
        return f"{brand} {style}".strip() if style else brand
    return _format_vehicle_style(fallback_style)


def _resolve_render_mode(request_overrides: dict[str, Optional[str]]) -> RenderMode:
    requested_mode = (request_overrides.get("render_mode") or get_settings().render_mode or "overlay").strip().lower()
    try:
        return RenderMode(requested_mode)
    except ValueError:
        return RenderMode.OVERLAY


def _raise_for_moderation_result(moderation_result) -> None:
    if moderation_result.is_valid:
        return

    message = moderation_result.rejection_message or "The uploaded image could not be used."
    if moderation_result.rejection_reason == RejectionReason.NSFW:
        raise SensitiveContentDetected(message)
    if moderation_result.rejection_reason == RejectionReason.HUMAN_FACE:
        raise HumanFaceDetected(message)
    if moderation_result.rejection_reason == RejectionReason.LOW_QUALITY:
        raise ImageTooSmall(message)
    if moderation_result.rejection_reason == RejectionReason.INVALID_FORMAT:
        raise InvalidUploadFormat(message)
    raise VirtualFitterError(code="VF_001", message=message, recoverable=True, http_status=422)


def _create_detection_only_image(image_bytes: bytes, segmentation: SegmentationResult) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = segmentation.vehicle_bbox
    accent = (24, 180, 104)
    draw.rectangle((x1, y1, x2, y2), outline=accent, width=4)
    label = f"Vehicle detected: {segmentation.detected_angle.replace('_', ' ')}"
    label_x = x1
    label_y = max(8, y1 - 28)
    label_width = (len(label) * 8) + 12
    draw.rectangle((label_x, label_y, label_x + label_width, label_y + 24), fill=accent)
    draw.text((label_x + 6, label_y + 4), label, fill=(255, 255, 255))
    return np.array(canvas)


def _create_detection_only_result(
    job_id: str,
    image_bytes: bytes,
    segmentation: SegmentationResult,
    detected_vehicle_label: Optional[str] = None,
) -> RenderResult:
    detection_image = _create_detection_only_image(image_bytes, segmentation)
    result_url = _image_array_to_data_url(detection_image, format="JPEG")
    return RenderResult(
        image_url=result_url,
        result_url=result_url,
        confidence=round(segmentation.confidence, 3),
        detected_vehicle=detected_vehicle_label or segmentation.detected_vehicle_type,
        detected_angle=segmentation.detected_angle,
        placement_quality="low",
        render_mode="detection_only",
        render_model="vehicle_detection",
        processing_time_ms=0,
        cost_usd=0.0,
        segmentation_mask_url=None,
        vehicle_bbox=list(segmentation.vehicle_bbox),
    )


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
        update_job(job_id, status="segmenting", progress=8, message="Validating the uploaded image.")
        moderation_result = await validate_uploaded_image(image_bytes)
        _raise_for_moderation_result(moderation_result)

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
        detected_vehicle_label = _infer_vehicle_display_name(
            assets.product_title,
            assets.variant_title,
            segmentation.detected_vehicle_type,
        )
        logger.info(
            "[%s] Vehicle display label resolved to %s from product_title=%s variant_title=%s",
            job_id,
            detected_vehicle_label,
            assets.product_title,
            assets.variant_title,
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
                "detected_vehicle": detected_vehicle_label,
                "detected_vehicle_style": segmentation.detected_vehicle_type,
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
            "[%s] Placement complete: zone=%s quality=%s bbox=%s",
            job_id,
            assets.placement_zone,
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

        if not assets.overlay_url:
            warnings.append(
                RenderWarning(
                    code="VF_W01",
                    message="Product image was not available, so we're showing vehicle detection only.",
                )
            )
            result = _create_detection_only_result(job_id, image_bytes, segmentation, detected_vehicle_label)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            update_job(
                job_id,
                status="complete",
                progress=100,
                message="Vehicle detected.",
                result=result,
                warnings=warnings,
                metadata={
                    "timings_ms": {"total": elapsed_ms},
                    "segmentation_source": segmentation.source,
                    "segmentation_confidence": segmentation.confidence,
                    "placement_zone": assets.placement_zone,
                    "placement_quality": placement.placement_quality,
                    "product_type": assets.product_type,
                    "render_mode": "detection_only",
                    "render_model": "vehicle_detection",
                },
            )
            return RenderJob(
                job_id=job_id,
                status="complete",
                progress=100,
                result_url=result.image_url,
                error=None,
                metadata={"elapsed_ms": elapsed_ms, "mode": "detection_only"},
            )

        placement_mask = generate_placement_mask(segmentation.vehicle_mask, placement, assets.placement_zone)
        placement_mask_path = job_dir / "placement-mask.png"
        _save_png(placement_mask_path, placement_mask.astype(np.uint8))

        rendering_service = get_rendering_service()
        try:
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
        except OverlayAssetNotFound:
            warnings.append(
                RenderWarning(
                    code="VF_W05",
                    message="We couldn't download the product image, so we're showing vehicle detection only.",
                )
            )
            result = _create_detection_only_result(job_id, image_bytes, segmentation, detected_vehicle_label)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            update_job(
                job_id,
                status="complete",
                progress=100,
                message="Vehicle detected.",
                result=result,
                warnings=warnings,
                metadata={
                    "timings_ms": {"total": elapsed_ms},
                    "segmentation_source": segmentation.source,
                    "segmentation_confidence": segmentation.confidence,
                    "placement_zone": assets.placement_zone,
                    "placement_quality": placement.placement_quality,
                    "product_type": assets.product_type,
                    "render_mode": "detection_only",
                    "render_model": "vehicle_detection",
                },
            )
            return RenderJob(
                job_id=job_id,
                status="complete",
                progress=100,
                result_url=result.image_url,
                error=None,
                metadata={"elapsed_ms": elapsed_ms, "mode": "detection_only"},
            )
        logger.info(
            "[%s] Render complete: mode=%s model=%s result_url=%s bytes=%s",
            job_id,
            render_execution.mode.value,
            render_execution.model,
            bool(render_execution.result_url),
            bool(render_execution.result_bytes),
        )

        segmentation_mask_url = _image_array_to_data_url((segmentation.vehicle_mask * 255).astype(np.uint8), format="PNG")
        if render_execution.result_bytes:
            media_type = _detect_media_type(render_execution.result_bytes)
            result_url = _bytes_to_data_url(render_execution.result_bytes, media_type)
        elif render_execution.result_url:
            result_url = render_execution.result_url
        else:
            raise RenderTimeout("The renderer completed without returning an image.")
        logger.info(
            "[%s] Result URL created: %s chars from bytes=%s",
            job_id,
            len(result_url),
            len(render_execution.result_bytes or b""),
        )

        for warning_message in render_execution.warnings:
            warnings.append(RenderWarning(code="VF_W04", message=warning_message))

        result = RenderResult(
            image_url=result_url,
            result_url=result_url,
            confidence=round((_combined_confidence(segmentation, placement.placement_quality) * 0.55) + (render_execution.confidence * 0.45), 3),
            detected_vehicle=detected_vehicle_label or segmentation.detected_vehicle_type,
            detected_angle=segmentation.detected_angle,
            placement_quality=placement.placement_quality,  # type: ignore[arg-type]
            render_mode=render_execution.mode.value,
            render_model=render_execution.model,
            processing_time_ms=render_execution.processing_time_ms,
            cost_usd=render_execution.cost_usd,
            segmentation_mask_url=segmentation_mask_url,
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
                "detected_vehicle": detected_vehicle_label,
                "detected_vehicle_style": segmentation.detected_vehicle_type,
                "placement_zone": assets.placement_zone,
                "placement_quality": placement.placement_quality,
                "product_type": assets.product_type,
                "render_mode": render_execution.mode.value,
                "render_model": render_execution.model,
                "render_cost_usd": render_execution.cost_usd,
                "render_time_ms": render_execution.processing_time_ms,
            },
        )
        logger.info(
            "[%s] Final result_url length=%s placement_zone=%s render_model=%s",
            job_id,
            len(result.result_url or result.image_url or ""),
            assets.placement_zone,
            render_execution.model,
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
