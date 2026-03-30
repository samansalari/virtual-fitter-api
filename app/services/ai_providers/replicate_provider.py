"""
Replicate API integration for SAM2 segmentation and FLUX rendering.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SAM2Config:
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 1
    min_mask_region_area: int = 100


@dataclass
class SAM2Result:
    mask: np.ndarray
    confidence: float
    bbox: tuple[int, int, int, int]
    model: str = "sam2"
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class FLUXConfig:
    ip_adapter_scale: float = 0.75
    guidance_scale: float = 7.5
    num_inference_steps: int = 28
    strength: float = 0.85
    output_format: str = "webp"
    output_quality: int = 90


@dataclass
class RenderRequest:
    car_image_bytes: bytes
    product_image_url: str
    mask_bytes: bytes
    product_type: str
    placement_zone: str
    prompt_hint: str = ""
    config: FLUXConfig = field(default_factory=FLUXConfig)


@dataclass
class RenderResult:
    image_url: str
    image_bytes: Optional[bytes] = None
    confidence: float = 0.9
    model: str = "flux"
    processing_time_ms: int = 0
    cost_usd: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


class SAM2Error(Exception):
    """SAM2 segmentation error."""


class RenderError(Exception):
    """AI rendering error."""


class ReplicateProvider:
    """
    Replicate API provider for AI models.

    Uses the official Replicate HTTP API to create predictions on hosted models.
    """

    COSTS = {
        "sam2": 0.0015,
        "flux_ip_adapter": 0.015,
        "flux_inpaint": 0.008,
    }

    def __init__(
        self,
        api_token: str,
        *,
        primary_model: str = "meta/sam-2-base",
        fallback_models: Optional[list[str]] = None,
        flux_ip_adapter_model: str = "lucataco/flux-dev-ip-adapter",
        flux_inpaint_model: str = "black-forest-labs/flux-fill-dev",
        timeout_seconds: int = 120,
    ) -> None:
        self.api_token = api_token
        self.base_url = "https://api.replicate.com/v1"
        self.primary_model = primary_model.replace(":latest", "")
        self.fallback_models = [model.replace(":latest", "") for model in (fallback_models or ["meta/sam-2-large"])]
        self.flux_ip_adapter_model = flux_ip_adapter_model.replace(":latest", "")
        self.flux_inpaint_model = flux_inpaint_model.replace(":latest", "")
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self.timeout_seconds)),
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def segment_with_sam2(self, image_bytes: bytes, config: Optional[SAM2Config] = None) -> SAM2Result:
        config = config or SAM2Config()
        last_error: Optional[Exception] = None

        for model in [self.primary_model, *self.fallback_models]:
            try:
                return await self._segment_with_model(model, image_bytes, config)
            except SAM2Error as exc:
                last_error = exc
                logger.warning("SAM2 model %s failed: %s", model, exc)

        raise SAM2Error(str(last_error or "SAM2 prediction failed"))

    async def render_with_flux_ip_adapter(self, request: RenderRequest) -> RenderResult:
        start_time = time.perf_counter()
        prompt = self._build_render_prompt(
            request.product_type,
            request.placement_zone,
            prompt_hint=request.prompt_hint,
            detailed=False,
        )
        prediction_input = {
            "image": self._bytes_to_data_uri(request.car_image_bytes),
            "mask": self._bytes_to_data_uri(request.mask_bytes, "image/png"),
            "ip_adapter_image": request.product_image_url,
            "ip_adapter_scale": request.config.ip_adapter_scale,
            "prompt": prompt,
            "guidance_scale": request.config.guidance_scale,
            "num_inference_steps": request.config.num_inference_steps,
            "strength": request.config.strength,
            "output_format": request.config.output_format,
            "output_quality": request.config.output_quality,
        }
        try:
            prediction = await self._create_prediction(self.flux_ip_adapter_model, prediction_input)
        except RenderError:
            prediction = await self._create_prediction(
                self.flux_ip_adapter_model,
                {
                    "image": prediction_input["image"],
                    "mask": prediction_input["mask"],
                    "ip_adapter_image": prediction_input["ip_adapter_image"],
                    "prompt": prediction_input["prompt"],
                },
            )
        result_url = self._extract_output_url(prediction.get("output"))
        if not result_url:
            raise RenderError("Replicate did not return an image URL for the premium render.")

        return RenderResult(
            image_url=result_url,
            confidence=0.95,
            model=self.flux_ip_adapter_model,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            cost_usd=self.COSTS["flux_ip_adapter"],
            debug={
                "prompt": prompt,
                "ip_adapter_scale": request.config.ip_adapter_scale,
                "product_url": request.product_image_url,
            },
        )

    async def render_with_flux_inpaint(self, request: RenderRequest) -> RenderResult:
        start_time = time.perf_counter()
        prompt = self._build_render_prompt(
            request.product_type,
            request.placement_zone,
            prompt_hint=request.prompt_hint,
            detailed=True,
        )
        prediction_input = {
            "image": self._bytes_to_data_uri(request.car_image_bytes),
            "mask": self._bytes_to_data_uri(request.mask_bytes, "image/png"),
            "prompt": prompt,
            "guidance_scale": request.config.guidance_scale,
            "num_inference_steps": max(18, request.config.num_inference_steps - 3),
            "strength": request.config.strength,
            "output_format": request.config.output_format,
            "output_quality": request.config.output_quality,
        }
        try:
            prediction = await self._create_prediction(self.flux_inpaint_model, prediction_input)
        except RenderError:
            prediction = await self._create_prediction(
                self.flux_inpaint_model,
                {
                    "image": prediction_input["image"],
                    "mask": prediction_input["mask"],
                    "prompt": prediction_input["prompt"],
                },
            )
        result_url = self._extract_output_url(prediction.get("output"))
        if not result_url:
            raise RenderError("Replicate did not return an image URL for the AI enhanced render.")

        return RenderResult(
            image_url=result_url,
            confidence=0.86,
            model=self.flux_inpaint_model,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            cost_usd=self.COSTS["flux_inpaint"],
            debug={"prompt": prompt},
        )

    async def _segment_with_model(self, model_name: str, image_bytes: bytes, config: SAM2Config) -> SAM2Result:
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
            image_uri = f"data:{media_type};base64,{image_b64}"

            prediction_input = {
                "image": image_uri,
                "points_per_side": config.points_per_side,
                "pred_iou_thresh": config.pred_iou_thresh,
                "stability_score_thresh": config.stability_score_thresh,
                "crop_n_layers": config.crop_n_layers,
                "min_mask_region_area": config.min_mask_region_area,
            }

            prediction = await self._create_prediction(model_name, prediction_input)
            return await self._process_sam2_output(prediction, image_bytes, model_name)
        except RenderError as exc:
            raise SAM2Error(str(exc)) from exc

    async def _create_prediction(self, model_name: str, prediction_input: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        owner, name = self._split_model_name(model_name)
        try:
            response = await client.post(
                f"{self.base_url}/models/{owner}/{name}/predictions",
                json={"input": prediction_input},
            )
            response.raise_for_status()
            prediction = response.json()
            return await self._wait_for_prediction(prediction)
        except httpx.HTTPStatusError as exc:
            logger.error("Replicate prediction error for %s: %s", model_name, exc.response.text)
            raise RenderError(f"Replicate API error for {model_name}: {exc.response.status_code}") from exc
        except RenderError:
            raise
        except Exception as exc:  # pragma: no cover - network safety
            raise RenderError(f"Replicate request failed for {model_name}: {exc}") from exc

    async def _wait_for_prediction(self, prediction: dict[str, Any], poll_interval: float = 1.0) -> dict[str, Any]:
        client = await self._get_client()
        deadline = asyncio.get_running_loop().time() + float(self.timeout_seconds)
        current = prediction

        while True:
            status = current.get("status")
            if status == "succeeded":
                return current
            if status == "failed":
                error_message = current.get("error", "Unknown error")
                raise RenderError(f"Prediction failed: {error_message}")
            if status == "canceled":
                raise RenderError("Prediction was canceled")

            if asyncio.get_running_loop().time() >= deadline:
                raise RenderError(f"Prediction timed out after {self.timeout_seconds}s")

            get_url = current.get("urls", {}).get("get")
            if not get_url:
                raise RenderError("Prediction polling URL missing from Replicate response")

            await asyncio.sleep(poll_interval)
            response = await client.get(get_url)
            response.raise_for_status()
            current = response.json()

    async def _process_sam2_output(self, prediction: dict[str, Any], original_image_bytes: bytes, model_name: str) -> SAM2Result:
        output = prediction.get("output", {})
        original_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        img_width, img_height = original_image.size
        img_area = float(img_width * img_height)

        masks_data = self._coerce_masks_data(output)
        if not masks_data:
            raise SAM2Error("No masks returned from SAM2")

        client = await self._get_client()
        best_mask: Optional[np.ndarray] = None
        best_score = 0.0
        best_bbox = (0, 0, img_width, img_height)

        for index, mask_info in enumerate(masks_data):
            try:
                mask_url, provider_score = self._parse_mask_info(mask_info)
                if not mask_url:
                    continue

                mask_response = await client.get(mask_url)
                mask_response.raise_for_status()
                mask_array = self._decode_mask(mask_response.content, (img_width, img_height))
                mask_score, bbox = self._score_mask(mask_array, provider_score, img_area)

                logger.debug(
                    "SAM2 mask %s: provider_score=%.2f candidate_score=%.2f bbox=%s",
                    index,
                    provider_score,
                    mask_score,
                    bbox,
                )

                if mask_score > best_score:
                    best_score = mask_score
                    best_mask = mask_array
                    best_bbox = bbox
            except Exception as exc:  # pragma: no cover - defensive integration
                logger.warning("Failed to process SAM2 mask %s: %s", index, exc)

        if best_mask is None:
            raise SAM2Error("Could not process any masks from SAM2")

        return SAM2Result(
            mask=best_mask,
            confidence=min(best_score, 0.99),
            bbox=best_bbox,
            model=model_name,
            debug={
                "prediction_id": prediction.get("id"),
                "total_masks": len(masks_data),
                "best_score": best_score,
            },
        )

    @staticmethod
    def _build_render_prompt(product_type: str, placement_zone: str, *, prompt_hint: str = "", detailed: bool = False) -> str:
        base_prompts = {
            "spoiler": "carbon fiber rear spoiler wing, professionally installed",
            "diffuser": "carbon fiber rear diffuser with aggressive styling",
            "side_skirt": "carbon fiber side skirts, flush mounted",
            "body_kit": "full carbon fiber aero body kit",
            "mirror_cap": "carbon fiber mirror caps, OEM fit",
            "front_lip": "carbon fiber front lip splitter",
            "hood": "carbon fiber hood with vents",
            "trunk_lip": "carbon fiber trunk lip spoiler",
            "wheel": "aftermarket alloy wheel upgrade",
            "badge": "automotive exterior badge",
        }
        detailed_prompts = {
            "spoiler": "gloss carbon fiber weave pattern, bolt-on design, subtle performance styling",
            "diffuser": "premium diffuser fins, OEM-style fitment, realistic rear bumper integration",
            "side_skirt": "smooth flowing side sill extension, subtle aggressive stance, realistic body contour match",
        }

        base = base_prompts.get(product_type, "premium automotive accessory")
        if detailed:
            extra = detailed_prompts.get(product_type, "clean integration, realistic reflections and shadows")
            base = f"{base}, {extra}"

        if prompt_hint:
            base = f"{base}, matching catalog guidance: {prompt_hint}"

        return (
            f"{base}, installed on the vehicle {placement_zone}, photorealistic automotive photo, "
            "natural lighting, realistic reflections, accurate perspective, factory-quality fitment"
        )

    @staticmethod
    def _bytes_to_data_uri(data: bytes, media_type: Optional[str] = None) -> str:
        if media_type is None:
            media_type = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{media_type};base64,{b64}"

    @staticmethod
    def _extract_output_url(output: Any) -> Optional[str]:
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    candidate = item.get("image") or item.get("url") or item.get("output")
                    if isinstance(candidate, str):
                        return candidate
        if isinstance(output, dict):
            for key in ("image", "url", "output"):
                candidate = output.get(key)
                if isinstance(candidate, str):
                    return candidate
                if isinstance(candidate, list):
                    nested = ReplicateProvider._extract_output_url(candidate)
                    if nested:
                        return nested
        return None

    @staticmethod
    def _split_model_name(model_name: str) -> tuple[str, str]:
        cleaned = model_name.replace(":latest", "")
        if "/" not in cleaned:
            raise SAM2Error(f"Invalid Replicate model name: {model_name}")
        owner, name = cleaned.split("/", 1)
        return owner, name

    @staticmethod
    def _coerce_masks_data(output: Any) -> list[Any]:
        if isinstance(output, list):
            return output
        if isinstance(output, dict):
            for key in ("masks", "mask", "mask_url", "output"):
                value = output.get(key)
                if isinstance(value, list):
                    return value
                if value:
                    return [value]
        if isinstance(output, str):
            return [output]
        return []

    @staticmethod
    def _parse_mask_info(mask_info: Any) -> tuple[Optional[str], float]:
        if isinstance(mask_info, str):
            return mask_info, 0.9
        if isinstance(mask_info, dict):
            return (
                mask_info.get("mask_url") or mask_info.get("url") or mask_info.get("mask"),
                float(mask_info.get("iou_score") or mask_info.get("predicted_iou") or mask_info.get("score") or 0.9),
            )
        return None, 0.0

    @staticmethod
    def _decode_mask(mask_bytes: bytes, image_size: tuple[int, int]) -> np.ndarray:
        mask_image = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_image)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        if mask_array.max() > 1:
            mask_array = (mask_array > 127).astype(np.uint8)
        else:
            mask_array = mask_array.astype(np.uint8)

        if mask_array.shape != (image_size[1], image_size[0]):
            resized = Image.fromarray(mask_array * 255).resize(image_size, Image.NEAREST)
            mask_array = (np.array(resized) > 127).astype(np.uint8)
        return mask_array

    @staticmethod
    def _score_mask(mask_array: np.ndarray, provider_score: float, image_area: float) -> tuple[float, tuple[int, int, int, int]]:
        y_coords, x_coords = np.where(mask_array > 0)
        if x_coords.size == 0 or y_coords.size == 0:
            return 0.0, (0, 0, mask_array.shape[1], mask_array.shape[0])

        x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
        y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
        bbox = (x1, y1, x2, y2)
        bbox_width = max(1, x2 - x1)
        bbox_height = max(1, y2 - y1)
        area_ratio = float(np.sum(mask_array) / image_area)
        aspect_ratio = bbox_width / bbox_height
        centroid_x = float(np.mean(x_coords) / mask_array.shape[1])

        score = provider_score * 0.45
        if 0.15 < area_ratio < 0.70:
            score += 0.28
        elif 0.10 < area_ratio < 0.80:
            score += 0.14

        if abs(centroid_x - 0.5) < 0.2:
            score += 0.17
        elif abs(centroid_x - 0.5) < 0.3:
            score += 0.08

        if 1.3 < aspect_ratio < 3.5:
            score += 0.10

        return float(min(score, 0.99)), bbox

    def estimate_cost(self, mode: str = "sam2") -> float:
        if mode in {"ai_premium", "flux_ip_adapter"}:
            return self.COSTS["flux_ip_adapter"]
        if mode in {"ai_basic", "flux_inpaint"}:
            return self.COSTS["flux_inpaint"]
        return self.COSTS["sam2"]
