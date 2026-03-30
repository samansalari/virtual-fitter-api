from __future__ import annotations

import logging
import secrets

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .jobs import get_job, save_job, utc_now
from .schemas import RenderJobResponse
from .services.pipeline import process_render_job
from .services.renderer import RenderMode, close_rendering_service, get_rendering_service
from .services.segmentation import close_segmentation_service
from .services.validation import VirtualFitterError, get_user_guidance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Virtual Fitter Render Service",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("Configured CORS origins for Virtual Fitter: %s", settings.cors_origins)

media_dir = settings.storage_dir / "media"
media_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_dir), name="media")


def verify_token(request: Request) -> None:
    if not settings.api_token:
        return

    auth_header = request.headers.get("Authorization", "")
    expected = f"Bearer {settings.api_token}"
    if auth_header != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def build_error_response(exc: VirtualFitterError) -> JSONResponse:
    guidance = get_user_guidance(
        error_code=exc.code,
        detected_angle=exc.detected_angle,
        product_zone=exc.product_zone,
    )
    return JSONResponse(
        status_code=exc.http_status or 422,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "technical_message": exc.message,
            },
            "guidance": guidance,
            "detected": {
                "angle": exc.detected_angle,
                "confidence": exc.confidence,
            },
        },
    )

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "mode": settings.render_mode}


@app.get("/v1/health")
def versioned_health_check() -> dict[str, object]:
    return {
        "status": "healthy",
        "version": app.version,
        "replicate_configured": bool(settings.replicate_api_token),
        "default_render_mode": settings.render_mode,
    }


@app.get("/v1/render-modes")
def get_render_modes() -> dict[str, list[dict[str, object]]]:
    rendering_service = get_rendering_service()
    ai_available = bool(settings.replicate_api_token and settings.enable_ai_rendering)
    return {
        "modes": [
            {
                "id": "overlay",
                "name": "Basic Preview",
                "description": "Fast overlay preview so shoppers can confirm fit and angle.",
                "cost_usd": 0.0,
                "quality": "basic",
                "available": True,
            },
            {
                "id": "ai_basic",
                "name": "AI Enhanced",
                "description": "FLUX inpainting with softer lighting and better blending.",
                "cost_usd": rendering_service.estimate_cost(RenderMode.AI_BASIC) if ai_available else 0.008,
                "quality": "good",
                "available": ai_available,
            },
            {
                "id": "ai_premium",
                "name": "Photorealistic",
                "description": "Reference-conditioned FLUX render that stays closer to your actual product image.",
                "cost_usd": rendering_service.estimate_cost(RenderMode.AI_PREMIUM) if ai_available else 0.015,
                "quality": "premium",
                "available": ai_available,
            },
        ]
    }


@app.post("/v1/render-jobs", status_code=202, response_model=RenderJobResponse)
async def create_render_job(
    request: Request,
    background_tasks: BackgroundTasks,
    photo: UploadFile = File(...),
    product_handle: str = Form(...),
    variant_id: str = Form(...),
    shop_domain: str | None = Form(default=None),
    product_id: str | None = Form(default=None),
    product_title: str | None = Form(default=None),
    variant_title: str | None = Form(default=None),
    placement_hint: str | None = Form(default=None),
    overlay_asset_url: str | None = Form(default=None),
    mask_asset_url: str | None = Form(default=None),
    featured_image_url: str | None = Form(default=None),
    render_mode: str | None = Form(default=None),
) -> RenderJobResponse:
    verify_token(request)
    logger.info(
        "Create render job request received: product_id=%s variant_id=%s handle=%s shop_domain=%s render_mode=%s",
        product_id,
        variant_id,
        product_handle,
        shop_domain,
        render_mode or settings.render_mode,
    )

    content_type = photo.content_type or ""
    if not content_type.startswith("image/"):
        logger.warning("Rejected render upload with invalid content_type=%s", content_type)
        raise HTTPException(status_code=400, detail="Photo must be an image file.")

    file_bytes = await photo.read()
    logger.info("Uploaded photo accepted: filename=%s bytes=%s content_type=%s", photo.filename, len(file_bytes), content_type)
    if len(file_bytes) > 12 * 1024 * 1024:
        logger.warning("Rejected oversized render upload: bytes=%s", len(file_bytes))
        raise HTTPException(status_code=413, detail="Photo is too large. Keep uploads under 12MB.")

    job_id = secrets.token_hex(8)
    job = RenderJobResponse(
        job_id=job_id,
        status="pending",
        progress=5,
        message="Upload received. Preparing the render job.",
        created_at=utc_now(),
        updated_at=utc_now(),
        metadata={
            "product_id": product_id,
            "variant_id": variant_id,
            "product_handle": product_handle,
            "product_title": product_title,
            "variant_title": variant_title,
            "shop_domain": shop_domain,
            "render_mode": render_mode or settings.render_mode,
            "overlay_asset_url_provided": bool(overlay_asset_url),
            "mask_asset_url_provided": bool(mask_asset_url),
            "featured_image_url_provided": bool(featured_image_url),
        },
    )
    save_job(job)
    logger.info("Queued render job %s", job_id)

    background_tasks.add_task(
        process_render_job,
        job_id,
        file_bytes,
        product_id or "",
        variant_id,
        shop_domain or "",
        {
            "product_handle": product_handle,
            "product_title": product_title,
            "variant_title": variant_title,
            "placement_hint": placement_hint,
            "overlay_asset_url": overlay_asset_url,
            "mask_asset_url": mask_asset_url,
            "featured_image_url": featured_image_url,
            "render_mode": render_mode or settings.render_mode,
        },
    )
    return job


@app.get("/v1/render-jobs/{job_id}", response_model=RenderJobResponse)
def get_render_job(job_id: str, request: Request) -> RenderJobResponse:
    verify_token(request)
    logger.info("Checking render job status: job_id=%s", job_id)

    job = get_job(job_id)
    if not job:
        logger.warning("Render job not found: job_id=%s", job_id)
        raise HTTPException(status_code=404, detail="Render job not found.")

    return job


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP exception: status=%s detail=%s", exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(VirtualFitterError)
async def virtual_fitter_error_handler(_: Request, exc: VirtualFitterError) -> JSONResponse:
    logger.error("Virtual fitter error: code=%s message=%s", exc.code, exc.message)
    return build_error_response(exc)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await close_segmentation_service()
    await close_rendering_service()
