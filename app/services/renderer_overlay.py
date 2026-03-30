from __future__ import annotations

import io
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from PIL import Image

from ..config import get_settings
from .placement import PlacementResult
from .validation import OverlayAssetNotFound


async def _download_bytes(url: str) -> bytes:
    direct_path = Path(url)
    if direct_path.exists():
        return direct_path.read_bytes()

    if os.path.isabs(url):
        path = Path(url)
        if not path.exists():
            raise OverlayAssetNotFound()
        return path.read_bytes()

    parsed = urlparse(url)
    if parsed.scheme in {"", "file"} or (len(parsed.scheme) == 1 and parsed.scheme.isalpha()):
        path = Path(parsed.path if parsed.scheme == "file" else url)
        if not path.exists():
            raise OverlayAssetNotFound()
        return path.read_bytes()

    timeout = get_settings().request_timeout_seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        if response.status_code >= 400:
            raise OverlayAssetNotFound()
        return response.content


async def _load_rgba(url: str) -> np.ndarray:
    data = await _download_bytes(url)
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGBA"))


def _normalized_to_source(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [1.0 / max(width - 1, 1), 0.0, 0.0],
            [0.0, 1.0 / max(height - 1, 1), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _apply_blend_mode(base_region: np.ndarray, overlay_region: np.ndarray, alpha: np.ndarray, blend_mode: str) -> np.ndarray:
    base = base_region.astype(np.float32) / 255.0
    overlay = overlay_region.astype(np.float32) / 255.0
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]

    if blend_mode == "multiply":
        blended = base * overlay
    elif blend_mode == "overlay":
        blended = np.where(base <= 0.5, 2.0 * base * overlay, 1.0 - 2.0 * (1.0 - base) * (1.0 - overlay))
    else:
        blended = overlay

    composed = (blended * alpha) + (base * (1.0 - alpha))
    return np.clip(composed * 255.0, 0, 255).astype(np.uint8)


def _color_match(overlay_rgb: np.ndarray, base_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    mask = alpha > 0.01
    if np.count_nonzero(mask) < 20:
        return overlay_rgb

    overlay_hsv = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    base_hsv = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    overlay_v = overlay_hsv[..., 2][mask]
    base_v = base_hsv[..., 2][mask]
    if overlay_v.size == 0 or base_v.size == 0:
        return overlay_rgb

    brightness_ratio = float(np.mean(base_v) / max(np.mean(overlay_v), 1.0))
    overlay_hsv[..., 2] = np.clip(overlay_hsv[..., 2] * brightness_ratio, 0, 255)
    matched = cv2.cvtColor(overlay_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return matched


async def render_overlay(
    base_image: np.ndarray,
    overlay_url: str,
    placement: PlacementResult,
    vehicle_mask: np.ndarray,
) -> np.ndarray:
    """
    Simple but effective overlay rendering:
    1. Download overlay asset
    2. Apply perspective transform
    3. Feather edges using vehicle mask
    4. Composite with appropriate blend mode
    5. Color match to base image lighting
    """
    overlay_rgba = await _load_rgba(overlay_url)
    if placement.mirrored:
        overlay_rgba = np.ascontiguousarray(np.fliplr(overlay_rgba))

    overlay_rgb = overlay_rgba[..., :3]
    overlay_alpha = overlay_rgba[..., 3].astype(np.float32) / 255.0

    output_height, output_width = base_image.shape[:2]
    source_height, source_width = overlay_alpha.shape[:2]

    if placement.perspective_warp is not None:
        src_quad = np.array(
            [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]],
            dtype=np.float32,
        )
        warp_matrix = cv2.getPerspectiveTransform(src_quad, placement.perspective_warp.astype(np.float32))
    else:
        warp_matrix = placement.transform_matrix @ _normalized_to_source(source_width, source_height)

    warped_rgb = cv2.warpPerspective(
        overlay_rgb,
        warp_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    warped_alpha = cv2.warpPerspective(
        overlay_alpha,
        warp_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    kernel_size = max(3, placement.feather_radius | 1)
    alpha_mask = cv2.GaussianBlur(warped_alpha, (kernel_size, kernel_size), placement.feather_radius / 2)
    if vehicle_mask.ndim == 3:
        vehicle_mask = vehicle_mask[..., 0]

    mask_clip = cv2.GaussianBlur(vehicle_mask.astype(np.float32), (kernel_size, kernel_size), placement.feather_radius / 2)
    alpha_mask *= np.clip(mask_clip, 0.25, 1.0)

    color_matched_overlay = _color_match(warped_rgb, base_image, alpha_mask)
    composed = _apply_blend_mode(base_image, color_matched_overlay, alpha_mask, placement.blend_mode)
    return composed
