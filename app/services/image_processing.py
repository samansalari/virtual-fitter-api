from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def remove_white_background(image_bytes: bytes, threshold: int = 240) -> bytes:
    """
    Remove light/white backgrounds from a product image and return PNG bytes.

    This is intentionally conservative: it preserves existing transparency,
    strips near-white studio backgrounds, and softens the alpha edge slightly
    so badges and similar parts blend more naturally in overlay mode.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error("Failed to decode product image for background removal")
        return image_bytes

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        logger.warning("Unsupported channel count for background removal: %s", image.shape[2])
        return image_bytes

    bgr = image[:, :, :3]
    existing_alpha = image[:, :, 3].astype(np.float32) / 255.0

    # Treat bright, low-saturation regions as background. This catches plain
    # white studio backgrounds while keeping darker badge details intact.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    saturation_channel = hsv[:, :, 1]

    white_mask = np.all(bgr > threshold, axis=2)
    light_mask = value_channel > threshold
    low_saturation_mask = saturation_channel < 28
    background_mask = white_mask | (light_mask & low_saturation_mask)

    alpha = np.where(background_mask, 0.0, 1.0).astype(np.float32)
    if np.any(existing_alpha < 0.99):
        alpha = np.minimum(alpha, existing_alpha)

    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    image[:, :, 3] = (alpha * 255.0).astype(np.uint8)

    total_pixels = float(alpha.size)
    kept_pixels = float(np.count_nonzero(alpha > 0.5))
    removed_ratio = 1.0 - (kept_pixels / max(total_pixels, 1.0))
    logger.info(
        "Background removal complete: removed=%.1f%% kept=%.1f%%",
        removed_ratio * 100.0,
        (1.0 - removed_ratio) * 100.0,
    )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        logger.error("Failed to encode product image after background removal")
        return image_bytes
    return encoded.tobytes()
