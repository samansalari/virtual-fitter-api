from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image

from ..config import get_settings
from .ai_providers.replicate_provider import FLUXConfig, RenderError, RenderRequest, ReplicateProvider
from .placement import PlacementResult
from .renderer_overlay import render_overlay, smart_overlay_composite

logger = logging.getLogger(__name__)


class RenderMode(str, Enum):
    AI_PREMIUM = "ai_premium"
    AI_BASIC = "ai_basic"
    OVERLAY = "overlay"


@dataclass
class RenderExecution:
    mode: RenderMode
    result_url: Optional[str] = None
    result_bytes: Optional[bytes] = None
    confidence: float = 0.0
    processing_time_ms: int = 0
    cost_usd: float = 0.0
    model: str = ""
    warnings: list[str] = field(default_factory=list)


class RenderingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._replicate: Optional[ReplicateProvider] = None

        if self.settings.enable_ai_rendering and self.settings.replicate_api_token:
            self._replicate = ReplicateProvider(
                self.settings.replicate_api_token,
                primary_model=self.settings.segmentation_sam2_model,
                fallback_models=[self.settings.segmentation_sam2_fallback_model],
                flux_ip_adapter_model=self.settings.flux_ip_adapter_model,
                flux_inpaint_model=self.settings.flux_inpaint_model,
                timeout_seconds=self.settings.render_ai_timeout,
            )
            logger.info("AI rendering enabled via Replicate.")

    async def render(
        self,
        *,
        car_image: np.ndarray,
        car_image_bytes: bytes,
        product_image_url: str,
        mask: np.ndarray,
        placement: PlacementResult,
        product_type: str,
        placement_zone: str,
        prompt_hint: str = "",
        mode: RenderMode = RenderMode.OVERLAY,
    ) -> RenderExecution:
        requested_mode = mode
        if requested_mode in {RenderMode.AI_PREMIUM, RenderMode.AI_BASIC} and not self._replicate:
            logger.warning("Requested %s without Replicate token. Falling back to overlay.", requested_mode.value)
            requested_mode = RenderMode.OVERLAY

        if requested_mode == RenderMode.AI_PREMIUM:
            return await self._render_ai_premium(
                car_image=car_image,
                car_image_bytes=car_image_bytes,
                product_image_url=product_image_url,
                mask=mask,
                placement=placement,
                product_type=product_type,
                placement_zone=placement_zone,
                prompt_hint=prompt_hint,
            )
        if requested_mode == RenderMode.AI_BASIC:
            return await self._render_ai_basic(
                car_image=car_image,
                car_image_bytes=car_image_bytes,
                product_image_url=product_image_url,
                mask=mask,
                placement=placement,
                product_type=product_type,
                placement_zone=placement_zone,
                prompt_hint=prompt_hint,
            )
        return await self._render_overlay(
            car_image=car_image,
            product_image_url=product_image_url,
            placement=placement,
            vehicle_mask=mask,
        )

    async def _render_ai_premium(
        self,
        *,
        car_image: np.ndarray,
        car_image_bytes: bytes,
        product_image_url: str,
        mask: np.ndarray,
        placement: PlacementResult,
        product_type: str,
        placement_zone: str,
        prompt_hint: str,
    ) -> RenderExecution:
        if self._should_prefer_exact_overlay(product_type):
            logger.info("Using exact smart overlay for premium render of product_type=%s", product_type)
            exact_overlay = await self._render_smart_overlay(
                car_image=car_image,
                product_image_url=product_image_url,
                placement=placement,
                vehicle_mask=mask,
            )
            exact_overlay.mode = RenderMode.AI_PREMIUM
            exact_overlay.model = "smart_overlay_exact"
            exact_overlay.confidence = 0.94
            return exact_overlay

        if not self._replicate:
            fallback = await self._render_smart_overlay(
                car_image=car_image,
                product_image_url=product_image_url,
                placement=placement,
                vehicle_mask=mask,
            )
            fallback.mode = RenderMode.AI_PREMIUM
            fallback.model = "smart_overlay_exact"
            fallback.confidence = 0.9
            return fallback

        try:
            request = RenderRequest(
                car_image_bytes=car_image_bytes,
                product_image_url=product_image_url,
                mask_bytes=self._mask_to_bytes(mask),
                product_type=product_type,
                placement_zone=placement_zone,
                prompt_hint=prompt_hint,
                config=FLUXConfig(
                    ip_adapter_scale=self.settings.flux_ip_adapter_scale,
                    guidance_scale=self.settings.flux_guidance_scale,
                    num_inference_steps=self.settings.flux_num_inference_steps,
                    strength=self.settings.flux_strength,
                ),
            )
            result = await self._replicate.render_with_flux_ip_adapter(request)
            return RenderExecution(
                mode=RenderMode.AI_PREMIUM,
                result_url=result.image_url,
                result_bytes=result.image_bytes,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                cost_usd=result.cost_usd,
                model=result.model,
            )
        except RenderError as exc:
            logger.error("AI premium render failed: %s", exc)
            if self._replicate:
                try:
                    fallback_result = await self._replicate.render_with_flux_inpaint(request)
                    fallback_execution = RenderExecution(
                        mode=RenderMode.AI_BASIC,
                        result_url=fallback_result.image_url,
                        result_bytes=fallback_result.image_bytes,
                        confidence=fallback_result.confidence,
                        processing_time_ms=fallback_result.processing_time_ms,
                        cost_usd=fallback_result.cost_usd,
                        model=fallback_result.model,
                    )
                    fallback_execution.warnings.append(
                        f"Photorealistic render was unavailable, so we used the AI enhanced renderer instead: {exc}"
                    )
                    return fallback_execution
                except RenderError as fallback_exc:
                    logger.error("AI basic fallback render failed after premium failure: %s", fallback_exc)

            fallback = await self._render_smart_overlay(
                car_image=car_image,
                product_image_url=product_image_url,
                placement=placement,
                vehicle_mask=mask,
            )
            fallback.mode = RenderMode.AI_PREMIUM
            fallback.model = "smart_overlay_exact"
            fallback.confidence = 0.9
            fallback.warnings.append(
                f"Photorealistic AI render was unavailable, so we used the exact product compositor instead: {exc}"
            )
            return fallback

    async def _render_ai_basic(
        self,
        *,
        car_image: np.ndarray,
        car_image_bytes: bytes,
        product_image_url: str,
        mask: np.ndarray,
        placement: PlacementResult,
        product_type: str,
        placement_zone: str,
        prompt_hint: str,
    ) -> RenderExecution:
        if not self._replicate:
            return await self._render_overlay(
                car_image=car_image,
                product_image_url=product_image_url,
                placement=placement,
                vehicle_mask=mask,
            )

        try:
            request = RenderRequest(
                car_image_bytes=car_image_bytes,
                product_image_url=product_image_url,
                mask_bytes=self._mask_to_bytes(mask),
                product_type=product_type,
                placement_zone=placement_zone,
                prompt_hint=prompt_hint,
                config=FLUXConfig(
                    guidance_scale=self.settings.flux_guidance_scale,
                    num_inference_steps=max(18, self.settings.flux_num_inference_steps - 3),
                    strength=self.settings.flux_strength,
                ),
            )
            result = await self._replicate.render_with_flux_inpaint(request)
            return RenderExecution(
                mode=RenderMode.AI_BASIC,
                result_url=result.image_url,
                result_bytes=result.image_bytes,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                cost_usd=result.cost_usd,
                model=result.model,
            )
        except RenderError as exc:
            logger.error("AI basic render failed: %s", exc)
            fallback = await self._render_overlay(
                car_image=car_image,
                product_image_url=product_image_url,
                placement=placement,
                vehicle_mask=mask,
            )
            fallback.warnings.append(f"AI enhanced render was unavailable, so we generated a standard preview instead: {exc}")
            return fallback

    async def _render_overlay(
        self,
        *,
        car_image: np.ndarray,
        product_image_url: str,
        placement: PlacementResult,
        vehicle_mask: np.ndarray,
    ) -> RenderExecution:
        start_time = time.perf_counter()
        result_image = await render_overlay(
            base_image=car_image,
            overlay_url=product_image_url,
            placement=placement,
            vehicle_mask=vehicle_mask,
        )
        return RenderExecution(
            mode=RenderMode.OVERLAY,
            result_bytes=self._image_to_bytes(result_image),
            confidence=0.72,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            cost_usd=0.0,
            model="opencv_overlay",
        )

    async def _render_smart_overlay(
        self,
        *,
        car_image: np.ndarray,
        product_image_url: str,
        placement: PlacementResult,
        vehicle_mask: np.ndarray,
    ) -> RenderExecution:
        start_time = time.perf_counter()
        result_image = await smart_overlay_composite(
            base_image=car_image,
            overlay_url=product_image_url,
            placement=placement,
            vehicle_mask=vehicle_mask,
        )
        return RenderExecution(
            mode=RenderMode.OVERLAY,
            result_bytes=self._image_to_bytes(result_image, quality=95),
            confidence=0.88,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            cost_usd=0.0,
            model="smart_overlay",
        )

    @staticmethod
    def _should_prefer_exact_overlay(product_type: str) -> bool:
        return product_type in {"badge", "mirror_cap", "wheel"}

    @staticmethod
    def _mask_to_bytes(mask: np.ndarray) -> bytes:
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        image = Image.fromarray(mask_uint8, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _image_to_bytes(image: np.ndarray, *, quality: int = 92) -> bytes:
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    def estimate_cost(self, mode: RenderMode) -> float:
        if not self._replicate:
            return 0.0
        return self._replicate.estimate_cost(mode.value)

    async def close(self) -> None:
        if self._replicate:
            await self._replicate.close()


_rendering_service: Optional[RenderingService] = None


def get_rendering_service() -> RenderingService:
    global _rendering_service
    if _rendering_service is None:
        _rendering_service = RenderingService()
    return _rendering_service


async def close_rendering_service() -> None:
    global _rendering_service
    if _rendering_service is not None:
        await _rendering_service.close()
        _rendering_service = None
