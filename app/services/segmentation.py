from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..config import get_settings
from .ai_providers import ReplicateProvider
from .ai_providers.replicate_provider import SAM2Config, SAM2Error
from .validation import InvalidImage, LowConfidenceResult, NoVehicleDetected

logger = logging.getLogger(__name__)


CONFIDENCE_THRESHOLDS = {
    "sam2": 0.70,
    "opencv": 0.45,
}


@dataclass
class SegmentationDebug:
    raw_contours_count: int
    largest_contour_area: float
    image_total_area: float
    contour_area_ratio: float
    bounding_box: tuple[int, int, int, int]
    bbox_aspect_ratio: float
    bbox_fill_ratio: float
    centroid_x_ratio: float
    centroid_y_ratio: float
    hull_area: float
    solidity: float
    angle_decision_reason: str
    symmetry_score: float = 0.0
    method: str = "opencv_combined"
    circularity: float = 0.0
    area_score: float = 0.0
    center_score: float = 0.0
    global_symmetry: float = 0.0
    upper_half_symmetry: float = 0.0
    lower_half_symmetry: float = 0.0
    upper_half_mass: float = 0.0
    lower_half_mass: float = 0.0
    left_half_mass: float = 0.0
    right_half_mass: float = 0.0
    wheel_score_left: float = 0.0
    wheel_score_right: float = 0.0
    wheel_symmetry: float = 0.0
    horizontal_edge_ratio: float = 0.0
    vertical_edge_ratio: float = 0.0
    angle_confidence: float = 0.0
    lower_red_ratio: float = 0.0
    upper_brightness: float = 0.0
    lower_brightness: float = 0.0
    brightness_gradient: float = 0.0
    dark_center_ratio: float = 0.0
    segmentation_model: str = "opencv"


@dataclass
class AngleFeatures:
    bbox_aspect_ratio: float
    bbox_fill_ratio: float
    centroid_x_ratio: float
    centroid_y_ratio: float
    global_symmetry: float
    upper_half_symmetry: float
    lower_half_symmetry: float
    upper_half_mass: float
    lower_half_mass: float
    left_half_mass: float
    right_half_mass: float
    solidity: float
    extent: float
    lower_left_circle_score: float
    lower_right_circle_score: float
    wheel_symmetry: float
    horizontal_edge_ratio: float
    vertical_edge_ratio: float
    row_width_mean_ratio: float
    row_width_std_ratio: float
    wheel_band_ratio: float
    lower_red_ratio: float
    upper_brightness: float
    lower_brightness: float
    brightness_gradient: float
    dark_center_ratio: float


@dataclass
class SegmentationResult:
    vehicle_mask: np.ndarray
    vehicle_bbox: tuple[int, int, int, int]
    confidence: float
    detected_angle: str
    detected_vehicle_type: Optional[str]
    view_side: Optional[str] = None
    source: str = "heuristic_cv"
    metadata: dict[str, object] = field(default_factory=dict)
    debug: Optional[SegmentationDebug] = None


def _decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            return np.array(image)
    except Exception as exc:
        raise InvalidImage() from exc


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise NoVehicleDetected()
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _estimate_vehicle_type(width: int, height: int, image_height: int) -> str:
    ratio = width / max(height, 1)
    vertical_coverage = height / max(image_height, 1)

    if vertical_coverage > 0.55 and ratio < 1.45:
        return "suv"
    if ratio > 1.95:
        return "coupe"
    if ratio > 1.65:
        return "sedan"
    if vertical_coverage < 0.42:
        return "hatchback"
    return "sedan"


def _normalize_binary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def _calc_symmetry(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0

    region = _normalize_binary(region)
    _, region_width = region.shape[:2]
    if region_width < 2:
        return 0.0

    left = region[:, : region_width // 2]
    right = region[:, region_width // 2 :]
    right_flipped = cv2.flip(right, 1)

    min_width = min(left.shape[1], right_flipped.shape[1])
    if min_width <= 0:
        return 0.0

    left = left[:, :min_width]
    right_flipped = right_flipped[:, :min_width]
    diff = np.abs(left.astype(np.float32) - right_flipped.astype(np.float32))
    total = float(np.sum(left) + np.sum(right_flipped)) + 1e-6
    return float(max(0.0, min(1.0, 1.0 - (np.sum(diff) / total))))


def _detect_wheel_score(region_mask: np.ndarray, gray_region: np.ndarray) -> float:
    if region_mask.size == 0 or gray_region.size == 0:
        return 0.0

    region_mask = (_normalize_binary(region_mask) * 255).astype(np.uint8)
    gray_region = cv2.GaussianBlur(gray_region, (5, 5), 0)
    masked_gray = cv2.bitwise_and(gray_region, gray_region, mask=region_mask)

    max_radius = max(8, min(region_mask.shape[:2]) // 4)
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, region_mask.shape[1] // 4),
        param1=60,
        param2=18,
        minRadius=8,
        maxRadius=max_radius,
    )
    if circles is not None:
        return float(min(1.0, 0.4 + (0.2 * len(circles[0]))))

    bottom_band = region_mask[int(region_mask.shape[0] * 0.45) :, :]
    if bottom_band.size == 0:
        return 0.0

    contours, _ = cv2.findContours(bottom_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity > 0.22:
            score = max(score, min(0.7, circularity + 0.1))
    return float(score)


def extract_color_features(image_rgb: np.ndarray, bbox_mask: np.ndarray, gray_bbox: np.ndarray) -> dict[str, float]:
    """Extract color cues to separate front from rear views."""
    if bbox_mask.size == 0:
        return {
            "lower_red_ratio": 0.0,
            "upper_brightness": 0.5,
            "lower_brightness": 0.5,
            "brightness_gradient": 0.0,
            "dark_center_ratio": 0.0,
        }

    height, width = bbox_mask.shape[:2]
    third_height = max(1, height // 3)
    hsv_bbox = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    upper_mask = bbox_mask[:third_height, :]
    lower_mask = bbox_mask[height - third_height :, :]

    lower_hsv = hsv_bbox[height - third_height :, :]
    red_mask1 = cv2.inRange(lower_hsv, (0, 100, 90), (12, 255, 255))
    red_mask2 = cv2.inRange(lower_hsv, (168, 100, 90), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = cv2.bitwise_and(red_mask, (_normalize_binary(lower_mask) * 255).astype(np.uint8))

    lower_pixels = int(np.sum(lower_mask > 0))
    red_pixels = int(np.sum(red_mask > 0))
    lower_red_ratio = float(red_pixels / (lower_pixels + 1e-6))

    upper_gray = gray_bbox[:third_height, :]
    lower_gray = gray_bbox[height - third_height :, :]
    upper_values = upper_gray[upper_mask > 0]
    lower_values = lower_gray[lower_mask > 0]

    upper_brightness = float(np.mean(upper_values) / 255.0) if upper_values.size else 0.5
    lower_brightness = float(np.mean(lower_values) / 255.0) if lower_values.size else 0.5
    brightness_gradient = float(upper_brightness - lower_brightness)

    center_x1 = width // 3
    center_x2 = max(center_x1 + 1, 2 * width // 3)
    center_y1 = max(0, third_height // 2)
    center_y2 = min(height, third_height * 2)
    center_region = gray_bbox[center_y1:center_y2, center_x1:center_x2]
    center_mask = bbox_mask[center_y1:center_y2, center_x1:center_x2]
    center_values = center_region[center_mask > 0]
    if center_values.size:
        dark_center_ratio = float(np.sum(center_values < 55) / center_values.size)
    else:
        dark_center_ratio = 0.0

    return {
        "lower_red_ratio": lower_red_ratio,
        "upper_brightness": upper_brightness,
        "lower_brightness": lower_brightness,
        "brightness_gradient": brightness_gradient,
        "dark_center_ratio": dark_center_ratio,
    }


def extract_angle_features(contour: np.ndarray, mask: np.ndarray, image_rgb: np.ndarray) -> AngleFeatures:
    mask = _normalize_binary(mask)
    height, width = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_aspect = bw / max(bh, 1)
    bbox_area = float(max(bw * bh, 1))
    contour_area = float(cv2.contourArea(contour))
    bbox_fill = contour_area / bbox_area

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        centroid_x = width / 2
        centroid_y = height / 2

    bbox_mask = mask[y : y + bh, x : x + bw]
    total_mass = float(np.sum(bbox_mask)) + 1e-6
    upper_mask = bbox_mask[: max(1, bh // 2), :]
    lower_mask = bbox_mask[max(1, bh // 2) :, :]
    left_mask = bbox_mask[:, : max(1, bw // 2)]
    right_mask = bbox_mask[:, max(1, bw // 2) :]

    upper_mass = float(np.sum(upper_mask) / total_mass)
    lower_mass = float(np.sum(lower_mask) / total_mass)
    left_mass = float(np.sum(left_mask) / total_mass)
    right_mass = float(np.sum(right_mask) / total_mass)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    extent = contour_area / float(height * width)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_bbox = gray[y : y + bh, x : x + bw]
    image_bbox = image_rgb[y : y + bh, x : x + bw]
    half_height = max(1, bh // 2)
    half_width = max(1, bw // 2)

    lower_left_mask = bbox_mask[half_height:, :half_width]
    lower_right_mask = bbox_mask[half_height:, half_width:]
    lower_left_gray = gray_bbox[half_height:, :half_width]
    lower_right_gray = gray_bbox[half_height:, half_width:]
    lower_left_score = _detect_wheel_score(lower_left_mask, lower_left_gray)
    lower_right_score = _detect_wheel_score(lower_right_mask, lower_right_gray)
    wheel_symmetry = 1.0 - abs(lower_left_score - lower_right_score) if (lower_left_score + lower_right_score) > 0 else 0.5

    sobel_x = cv2.Sobel(gray_bbox, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_bbox, cv2.CV_64F, 0, 1, ksize=3)
    h_edges = float(np.sum(np.abs(sobel_y) * bbox_mask))
    v_edges = float(np.sum(np.abs(sobel_x) * bbox_mask))
    total_edges = h_edges + v_edges + 1e-6

    row_widths = np.sum(bbox_mask > 0, axis=1).astype(np.float32)
    row_width_mean_ratio = float(np.mean(row_widths) / max(bw, 1))
    row_width_std_ratio = float(np.std(row_widths) / max(bw, 1))

    wheel_band = bbox_mask[int(bh * 0.72) :, :]
    wheel_band_ratio = float(np.sum(wheel_band) / total_mass) if wheel_band.size else 0.0
    color_features = extract_color_features(image_bbox, bbox_mask, gray_bbox)

    return AngleFeatures(
        bbox_aspect_ratio=float(bbox_aspect),
        bbox_fill_ratio=float(bbox_fill),
        centroid_x_ratio=float(centroid_x / max(width, 1)),
        centroid_y_ratio=float(centroid_y / max(height, 1)),
        global_symmetry=_calc_symmetry(bbox_mask),
        upper_half_symmetry=_calc_symmetry(upper_mask),
        lower_half_symmetry=_calc_symmetry(lower_mask),
        upper_half_mass=upper_mass,
        lower_half_mass=lower_mass,
        left_half_mass=left_mass,
        right_half_mass=right_mass,
        solidity=float(solidity),
        extent=float(extent),
        lower_left_circle_score=float(lower_left_score),
        lower_right_circle_score=float(lower_right_score),
        wheel_symmetry=float(wheel_symmetry),
        horizontal_edge_ratio=float(h_edges / total_edges),
        vertical_edge_ratio=float(v_edges / total_edges),
        row_width_mean_ratio=row_width_mean_ratio,
        row_width_std_ratio=row_width_std_ratio,
        wheel_band_ratio=wheel_band_ratio,
        lower_red_ratio=color_features["lower_red_ratio"],
        upper_brightness=color_features["upper_brightness"],
        lower_brightness=color_features["lower_brightness"],
        brightness_gradient=color_features["brightness_gradient"],
        dark_center_ratio=color_features["dark_center_ratio"],
    )


def _infer_view_side(features: AngleFeatures) -> Optional[str]:
    if abs(features.left_half_mass - features.right_half_mass) < 0.02:
        return None
    return "left" if features.right_half_mass > features.left_half_mass else "right"


def classify_angle(features: AngleFeatures) -> tuple[str, float, str]:
    scores = {
        "rear": 0.0,
        "front": 0.0,
        "side": 0.0,
        "three_quarter": 0.0,
    }
    reasons: list[str] = []
    wheel_diff = abs(features.lower_left_circle_score - features.lower_right_circle_score)
    wheel_max = max(features.lower_left_circle_score, features.lower_right_circle_score)
    wheel_min = min(features.lower_left_circle_score, features.lower_right_circle_score)
    mass_imbalance = abs(features.left_half_mass - features.right_half_mass)
    symmetry_gap = abs(features.lower_half_symmetry - features.upper_half_symmetry)

    if features.lower_red_ratio > 0.035 and features.dark_center_ratio < 0.60:
        scores["rear"] += 0.22
        reasons.append(f"taillights={features.lower_red_ratio:.3f}")
    elif (
        features.lower_red_ratio > 0.003
        and 1.85 <= features.bbox_aspect_ratio <= 2.35
        and features.dark_center_ratio < 0.35
    ):
        scores["rear"] += 0.10
        scores["three_quarter"] += 0.18
        reasons.append(f"rear_cue={features.lower_red_ratio:.4f}")
    elif features.brightness_gradient < -0.05:
        scores["rear"] += 0.12
        reasons.append(f"lower_bright={features.brightness_gradient:.2f}")

    if (
        features.lower_red_ratio > 0.003
        and features.global_symmetry < 0.62
        and 1.90 <= features.bbox_aspect_ratio <= 2.30
        and wheel_max > 0.8
        and 0.3 < wheel_min < 0.8
        and wheel_diff > 0.2
    ):
        scores["three_quarter"] += 0.34
        scores["side"] -= 0.10
        reasons.append(f"three_quarter_wheels={wheel_max:.1f}/{wheel_min:.1f}")

    if (
        features.dark_center_ratio > 0.92
        and features.bbox_fill_ratio > 0.78
        and features.global_symmetry > 0.95
        and features.bbox_aspect_ratio < 1.80
        and features.centroid_x_ratio < 0.46
    ):
        scores["front"] += 1.05
        reasons.append(f"front_signature={features.dark_center_ratio:.2f}")
    elif (
        features.dark_center_ratio > 0.55
        and features.brightness_gradient > 0.12
        and features.lower_red_ratio < 0.03
        and features.bbox_fill_ratio > 0.68
    ):
        scores["front"] += 0.18
        reasons.append(f"front_cues={features.dark_center_ratio:.2f}")

    if abs(features.centroid_x_ratio - 0.5) < 0.10:
        scores["rear"] += 0.14
        reasons.append(f"centered={features.centroid_x_ratio:.2f}")
    elif abs(features.centroid_x_ratio - 0.5) < 0.18:
        scores["three_quarter"] += 0.12
        reasons.append(f"mild_offset={features.centroid_x_ratio:.2f}")
    else:
        scores["side"] += 0.08

    if features.global_symmetry > 0.82:
        scores["rear"] += 0.12
        reasons.append(f"global_sym={features.global_symmetry:.2f}")
    elif features.global_symmetry < 0.68:
        scores["three_quarter"] += 0.12

    if features.bbox_aspect_ratio < 1.82:
        scores["rear"] += 0.22
        reasons.append(f"compact_aspect={features.bbox_aspect_ratio:.2f}")
    elif features.bbox_aspect_ratio < 2.25:
        scores["rear"] += 0.06
        scores["three_quarter"] += 0.18
        reasons.append(f"mid_aspect={features.bbox_aspect_ratio:.2f}")
    else:
        scores["side"] += 0.14
        scores["three_quarter"] += 0.26
        reasons.append(f"wide_aspect={features.bbox_aspect_ratio:.2f}")

    if features.bbox_fill_ratio > 0.68:
        scores["rear"] += 0.16
        scores["three_quarter"] += 0.14
        reasons.append(f"filled={features.bbox_fill_ratio:.2f}")
    elif features.bbox_fill_ratio < 0.60:
        scores["side"] += 0.22
        reasons.append(f"lean_fill={features.bbox_fill_ratio:.2f}")

    if features.row_width_mean_ratio > 0.72 and features.row_width_std_ratio < 0.17 and features.bbox_fill_ratio < 0.64:
        scores["side"] += 0.22
        reasons.append(f"profile_rows={features.row_width_mean_ratio:.2f}")
    elif features.row_width_mean_ratio < 0.62:
        scores["rear"] += 0.12

    if features.lower_half_mass > 0.54:
        scores["rear"] += 0.10
        reasons.append(f"lower_heavy={features.lower_half_mass:.2f}")

    if 0.04 < mass_imbalance < 0.22:
        scores["three_quarter"] += 0.16
        reasons.append(f"mass_imbalance={mass_imbalance:.2f}")
    elif mass_imbalance >= 0.22:
        scores["side"] += 0.12

    if features.vertical_edge_ratio > 0.56:
        scores["side"] += 0.08
        reasons.append(f"v_edges={features.vertical_edge_ratio:.2f}")
    if (
        features.horizontal_edge_ratio > 0.60
        and features.bbox_fill_ratio > 0.78
        and features.global_symmetry > 0.96
        and features.bbox_aspect_ratio < 1.80
        and features.centroid_x_ratio < 0.46
    ):
        scores["front"] += 0.20
        reasons.append(f"front_edges={features.horizontal_edge_ratio:.2f}")

    if features.lower_half_symmetry > 0.84 and features.upper_half_symmetry < 0.78 and features.bbox_fill_ratio < 0.66:
        scores["side"] += 0.10
    if features.upper_half_symmetry > 0.82 and features.lower_half_symmetry > 0.82:
        scores["rear"] += 0.10

    if 2.20 <= features.bbox_aspect_ratio <= 2.50 and features.global_symmetry > 0.80 and features.bbox_fill_ratio > 0.69:
        scores["rear"] += 0.16
        reasons.append("wide_but_filled")

    if 2.28 <= features.bbox_aspect_ratio <= 2.65 and features.bbox_fill_ratio > 0.67 and symmetry_gap > 0.03:
        scores["three_quarter"] += 0.18
        reasons.append(f"sym_gap={symmetry_gap:.2f}")

    if 2.25 <= features.bbox_aspect_ratio <= 2.65 and 0.47 <= features.lower_half_mass <= 0.53 and features.global_symmetry > 0.88:
        scores["three_quarter"] += 0.22
        reasons.append("balanced_three_quarter")

    if features.bbox_aspect_ratio > 2.05 and features.bbox_fill_ratio < 0.58:
        scores["side"] += 0.34
        reasons.append("wide_and_lean")

    if 1.82 <= features.bbox_aspect_ratio <= 2.20 and features.bbox_fill_ratio < 0.58:
        scores["side"] += 0.28
        reasons.append("mid_profile")

    if features.bbox_aspect_ratio > 2.0 and features.bbox_fill_ratio < 0.56 and features.upper_half_mass > 0.50:
        scores["side"] += 0.18
        reasons.append("profile_mass")

    total_score = sum(scores.values()) + 1e-6
    normalized = {angle: value / total_score for angle, value in scores.items()}
    best_angle = max(normalized, key=normalized.get)
    sorted_scores = sorted(normalized.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    confidence = float(max(0.5, min(0.98, 0.58 + (margin * 0.9))))
    reason = f"{best_angle}({normalized[best_angle]:.2f}): " + ", ".join(reasons[:6])
    return best_angle, confidence, reason


def detect_angle(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    mask: np.ndarray,
    image_rgb: np.ndarray,
) -> tuple[str, str, Optional[str], dict[str, float]]:
    """Classify vehicle angle from weighted visual features."""
    features = extract_angle_features(contour, mask, image_rgb)
    angle_candidate, angle_confidence, reason = classify_angle(features)
    view_side = _infer_view_side(features)
    return angle_candidate, reason, view_side, {
        "bbox_aspect_ratio": features.bbox_aspect_ratio,
        "bbox_fill_ratio": features.bbox_fill_ratio,
        "centroid_x_ratio": features.centroid_x_ratio,
        "centroid_y_ratio": features.centroid_y_ratio,
        "solidity": features.solidity,
        "symmetry_score": features.global_symmetry,
        "global_symmetry": features.global_symmetry,
        "upper_half_symmetry": features.upper_half_symmetry,
        "lower_half_symmetry": features.lower_half_symmetry,
        "upper_half_mass": features.upper_half_mass,
        "lower_half_mass": features.lower_half_mass,
        "left_half_mass": features.left_half_mass,
        "right_half_mass": features.right_half_mass,
        "wheel_score_left": features.lower_left_circle_score,
        "wheel_score_right": features.lower_right_circle_score,
        "wheel_symmetry": features.wheel_symmetry,
        "horizontal_edge_ratio": features.horizontal_edge_ratio,
        "vertical_edge_ratio": features.vertical_edge_ratio,
        "angle_confidence": angle_confidence,
        "lower_red_ratio": features.lower_red_ratio,
        "upper_brightness": features.upper_brightness,
        "lower_brightness": features.lower_brightness,
        "brightness_gradient": features.brightness_gradient,
        "dark_center_ratio": features.dark_center_ratio,
    }


def _score_candidate(contour: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[float, dict[str, float]]:
    height, width = image_shape[:2]
    contour_area = cv2.contourArea(contour)
    if contour_area <= 0:
        return 0.0, {"contour_area": 0.0}

    x, y, bw, bh = cv2.boundingRect(contour)
    image_area = float(height * width)
    area_ratio = contour_area / image_area
    bbox_area = max(float(bw * bh), 1.0)
    bbox_fill_ratio = contour_area / bbox_area
    centroid_x = x + bw / 2
    centroid_y = y + bh / 2
    centroid_x_ratio = centroid_x / max(width, 1)
    centroid_y_ratio = centroid_y / max(height, 1)
    bbox_aspect = bw / max(bh, 1)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0.0

    center_score = 1.0 - min(abs(centroid_x_ratio - 0.5) / 0.5, 1.0)
    lower_bias_score = 1.0 - min(abs(centroid_y_ratio - 0.62) / 0.62, 1.0)

    if 0.15 <= area_ratio <= 0.65:
        area_score = 1.0
    elif 0.08 <= area_ratio <= 0.8:
        area_score = 0.65
    else:
        area_score = 0.15

    fill_score = min(max((bbox_fill_ratio - 0.18) / 0.58, 0.0), 1.0)
    solidity_score = min(max((solidity - 0.45) / 0.45, 0.0), 1.0)
    edge_touch_count = int(x <= width * 0.02) + int(y <= height * 0.02) + int((x + bw) >= width * 0.98) + int((y + bh) >= height * 0.98)
    edge_penalty = 0.25 if edge_touch_count >= 3 else (0.12 if edge_touch_count == 2 else 0.0)

    score = (
        0.30 * area_score
        + 0.22 * solidity_score
        + 0.18 * fill_score
        + 0.18 * center_score
        + 0.12 * lower_bias_score
        - edge_penalty
    )

    return max(0.0, min(score, 0.98)), {
        "contour_area": contour_area,
        "area_ratio": area_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "centroid_x_ratio": centroid_x_ratio,
        "centroid_y_ratio": centroid_y_ratio,
        "bbox_aspect_ratio": bbox_aspect,
        "solidity": solidity,
        "hull_area": hull_area,
        "circularity": circularity,
        "area_score": area_score,
        "center_score": center_score,
        "edge_touch_count": float(edge_touch_count),
    }


def opencv_segment_vehicle(image_rgb: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
    """
    Improved OpenCV vehicle segmentation.
    Returns: (mask, confidence, best_contour, debug_info)
    """
    height, width = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    lower_saturation = hsv[:, :, 1] > 24
    not_too_bright = hsv[:, :, 2] < 248
    not_too_dark = hsv[:, :, 2] > 18
    color_mask = (lower_saturation & not_too_bright & not_too_dark).astype(np.uint8) * 255

    rect_margin = 0.08
    rect = (
        int(width * rect_margin),
        int(height * rect_margin),
        int(width * (1 - 2 * rect_margin)),
        int(height * (1 - 2 * rect_margin)),
    )

    mask_grabcut = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.setRNGSeed(42)
        cv2.grabCut(image_bgr, mask_grabcut, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        grabcut_mask = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype(np.uint8) * 255
    except Exception:
        grabcut_mask = np.ones((height, width), np.uint8) * 255

    combined = cv2.bitwise_or(color_mask, edges)
    combined = cv2.bitwise_and(combined, grabcut_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(grabcut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((height, width), dtype=np.uint8), 0.0, np.empty((0, 1, 2), dtype=np.int32), {
            "raw_contours_count": 0.0,
            "method": "opencv_combined",
        }

    image_area = float(height * width)
    valid_contours: list[tuple[np.ndarray, float, dict[str, float]]] = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / image_area
        if 0.05 < area_ratio < 0.90:
            score, metrics = _score_candidate(contour, image_rgb.shape)
            valid_contours.append((contour, score, metrics))

    if not valid_contours:
        for contour in contours:
            score, metrics = _score_candidate(contour, image_rgb.shape)
            valid_contours.append((contour, score, metrics))

    valid_contours.sort(key=lambda item: item[1], reverse=True)
    best_contour, best_score, best_metrics = valid_contours[0]

    final_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(final_mask, [best_contour], -1, 1, -1)

    confidence = (
        0.28
        + (0.22 * min(best_metrics["area_score"], 1.0))
        + (0.20 * min(best_metrics["solidity"], 1.0))
        + (0.18 * min(best_metrics["bbox_fill_ratio"], 1.0))
        + (0.12 * min(best_metrics["center_score"], 1.0))
    )
    if best_metrics["edge_touch_count"] >= 3:
        confidence -= 0.18
    elif best_metrics["edge_touch_count"] == 2:
        confidence -= 0.08
    confidence = float(max(0.0, min(confidence, 0.95)))

    debug_info = {
        "method": "opencv_combined",
        "raw_contours_count": float(len(contours)),
        "valid_contours": float(len(valid_contours)),
        **best_metrics,
        "largest_contour_area": float(max(cv2.contourArea(contour) for contour in contours)),
        "image_total_area": image_area,
    }
    return final_mask, confidence, best_contour, debug_info


def _build_debug(
    contour: np.ndarray,
    mask: np.ndarray,
    image_shape: tuple[int, int, int],
    debug_info: dict[str, float],
    angle_reason: str,
) -> SegmentationDebug:
    x, y, bw, bh = cv2.boundingRect(contour)
    return SegmentationDebug(
        raw_contours_count=int(debug_info.get("raw_contours_count", 0)),
        largest_contour_area=float(debug_info.get("largest_contour_area", 0.0)),
        image_total_area=float(debug_info.get("image_total_area", image_shape[0] * image_shape[1])),
        contour_area_ratio=float(debug_info.get("area_ratio", 0.0)),
        bounding_box=(x, y, bw, bh),
        bbox_aspect_ratio=float(debug_info.get("bbox_aspect_ratio", bw / max(bh, 1))),
        bbox_fill_ratio=float(debug_info.get("bbox_fill_ratio", 0.0)),
        centroid_x_ratio=float(debug_info.get("centroid_x_ratio", 0.5)),
        centroid_y_ratio=float(debug_info.get("centroid_y_ratio", 0.5)),
        hull_area=float(debug_info.get("hull_area", 0.0)),
        solidity=float(debug_info.get("solidity", 0.0)),
        angle_decision_reason=angle_reason,
        symmetry_score=float(debug_info.get("symmetry_score", 0.0)),
        method=str(debug_info.get("method", "opencv_combined")),
        circularity=float(debug_info.get("circularity", 0.0)),
        area_score=float(debug_info.get("area_score", 0.0)),
        center_score=float(debug_info.get("center_score", 0.0)),
        global_symmetry=float(debug_info.get("global_symmetry", debug_info.get("symmetry_score", 0.0))),
        upper_half_symmetry=float(debug_info.get("upper_half_symmetry", 0.0)),
        lower_half_symmetry=float(debug_info.get("lower_half_symmetry", 0.0)),
        upper_half_mass=float(debug_info.get("upper_half_mass", 0.0)),
        lower_half_mass=float(debug_info.get("lower_half_mass", 0.0)),
        left_half_mass=float(debug_info.get("left_half_mass", 0.0)),
        right_half_mass=float(debug_info.get("right_half_mass", 0.0)),
        wheel_score_left=float(debug_info.get("wheel_score_left", 0.0)),
        wheel_score_right=float(debug_info.get("wheel_score_right", 0.0)),
        wheel_symmetry=float(debug_info.get("wheel_symmetry", 0.0)),
        horizontal_edge_ratio=float(debug_info.get("horizontal_edge_ratio", 0.0)),
        vertical_edge_ratio=float(debug_info.get("vertical_edge_ratio", 0.0)),
        angle_confidence=float(debug_info.get("angle_confidence", 0.0)),
        lower_red_ratio=float(debug_info.get("lower_red_ratio", 0.0)),
        upper_brightness=float(debug_info.get("upper_brightness", 0.0)),
        lower_brightness=float(debug_info.get("lower_brightness", 0.0)),
        brightness_gradient=float(debug_info.get("brightness_gradient", 0.0)),
        dark_center_ratio=float(debug_info.get("dark_center_ratio", 0.0)),
        segmentation_model=str(debug_info.get("segmentation_model", debug_info.get("method", "opencv"))),
    )


def _segment_with_heuristics(image_rgb: np.ndarray) -> SegmentationResult:
    mask, confidence, contour, debug_info = opencv_segment_vehicle(image_rgb)
    if contour.size == 0 or np.count_nonzero(mask) == 0:
        raise NoVehicleDetected()

    bbox = _bbox_from_mask(mask)
    image_height, image_width = image_rgb.shape[:2]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    if bbox_width < image_width * 0.12 or bbox_height < image_height * 0.10:
        raise NoVehicleDetected("We found something vehicle-like, but the car is too small in frame.")

    detected_angle, angle_reason, view_side, angle_metrics = detect_angle(contour, image_rgb.shape, mask, image_rgb)
    debug_info.update(angle_metrics)
    debug = _build_debug(contour, mask, image_rgb.shape, debug_info, angle_reason)

    return SegmentationResult(
        vehicle_mask=_normalize_binary(mask),
        vehicle_bbox=bbox,
        confidence=confidence,
        detected_angle=detected_angle,
        detected_vehicle_type=_estimate_vehicle_type(bbox_width, bbox_height, image_height),
        view_side=view_side,
        source="heuristic_cv",
        metadata={
            "image_width": image_width,
            "image_height": image_height,
            "candidate_count": int(debug_info.get("valid_contours", 1)),
        },
        debug=debug,
    )


def _build_result_from_mask(
    *,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    confidence: float,
    image_rgb: np.ndarray,
    model: str,
    provider_debug: Optional[dict[str, object]] = None,
) -> SegmentationResult:
    normalized_mask = _normalize_binary(mask)
    contour_candidates, _ = cv2.findContours(normalized_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contour_candidates:
        raise NoVehicleDetected("No vehicle contour found in the segmentation mask.")

    contour = max(contour_candidates, key=cv2.contourArea)
    image_area = float(image_rgb.shape[0] * image_rgb.shape[1])
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    detected_angle, angle_reason, view_side, angle_metrics = detect_angle(contour, image_rgb.shape, normalized_mask, image_rgb)

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    contour_area = float(cv2.contourArea(contour))
    debug_info = {
        "method": model,
        "segmentation_model": model,
        "raw_contours_count": float(len(contour_candidates)),
        "largest_contour_area": contour_area,
        "image_total_area": image_area,
        "area_ratio": contour_area / max(image_area, 1.0),
        "bbox_fill_ratio": contour_area / max(float(bbox_width * bbox_height), 1.0),
        "hull_area": hull_area,
        "circularity": 0.0,
        "area_score": min(max(contour_area / max(image_area * 0.2, 1.0), 0.0), 1.0),
        "center_score": 1.0,
        **angle_metrics,
    }
    if provider_debug:
        for key, value in provider_debug.items():
            if isinstance(value, (int, float)):
                debug_info[key] = float(value)
            elif key == "method":
                debug_info[key] = str(value)

    debug = _build_debug(contour, normalized_mask, image_rgb.shape, debug_info, f"[{model.upper()}] {angle_reason}")
    return SegmentationResult(
        vehicle_mask=normalized_mask,
        vehicle_bbox=bbox,
        confidence=confidence,
        detected_angle=detected_angle,
        detected_vehicle_type=_estimate_vehicle_type(bbox_width, bbox_height, image_rgb.shape[0]),
        view_side=view_side,
        source=model,
        metadata={"image_width": image_rgb.shape[1], "image_height": image_rgb.shape[0]},
        debug=debug,
    )


def _method_threshold(method: str) -> float:
    if method == "opencv":
        return float(get_settings().low_confidence_threshold or CONFIDENCE_THRESHOLDS["opencv"])
    return CONFIDENCE_THRESHOLDS.get(method, get_settings().segmentation_confidence_threshold)


class SegmentationService:
    """
    Vehicle segmentation with multiple backends.

    Priority:
    1. SAM2 via Replicate when configured.
    2. OpenCV heuristics as the default fallback.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self._replicate: Optional[ReplicateProvider] = None

        if settings.enable_sam2 and settings.replicate_api_token:
            self._replicate = ReplicateProvider(
                settings.replicate_api_token,
                primary_model=settings.segmentation_sam2_model,
                fallback_models=[settings.segmentation_sam2_fallback_model] if settings.segmentation_sam2_fallback_model else [],
                timeout_seconds=settings.segmentation_sam2_timeout,
            )
            logger.info("SAM2 segmentation enabled via Replicate.")
        else:
            logger.info("SAM2 not available - using OpenCV fallback only.")

    async def segment_vehicle(self, image_bytes: bytes, prefer_sam2: bool = True) -> SegmentationResult:
        image_rgb = _decode_image(image_bytes)
        sam2_result: Optional[SegmentationResult] = None

        if prefer_sam2 and self.settings.segmentation_prefer_sam2 and self._replicate:
            try:
                sam2_result = await self._segment_with_sam2_provider(image_bytes, image_rgb)
                logger.info("SAM2 segmentation: confidence=%.2f", sam2_result.confidence)
                if sam2_result.confidence >= _method_threshold("sam2"):
                    return sam2_result
                logger.info("SAM2 confidence %.2f is below threshold. Trying OpenCV fallback.", sam2_result.confidence)
            except SAM2Error as exc:
                logger.warning("SAM2 failed, falling back to OpenCV: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive integration
                logger.exception("Unexpected SAM2 error: %s", exc)

        if not self.settings.enable_opencv_fallback:
            if sam2_result:
                return sam2_result
            raise LowConfidenceResult("Segmentation fallback is disabled and SAM2 is unavailable.")

        heuristic_result = _segment_with_heuristics(image_rgb)
        logger.info("OpenCV segmentation: confidence=%.2f", heuristic_result.confidence)

        if sam2_result and sam2_result.confidence > heuristic_result.confidence:
            return sam2_result

        if heuristic_result.confidence < _method_threshold("opencv"):
            raise LowConfidenceResult(
                f"Low confidence segmentation ({heuristic_result.confidence:.2f}) below OpenCV threshold {_method_threshold('opencv'):.2f}.",
                confidence=heuristic_result.confidence,
            )
        return heuristic_result

    async def segment_vehicle_opencv_only(self, image_bytes: bytes) -> SegmentationResult:
        image_rgb = _decode_image(image_bytes)
        heuristic_result = _segment_with_heuristics(image_rgb)
        if heuristic_result.confidence < _method_threshold("opencv"):
            raise LowConfidenceResult(
                f"Low confidence segmentation ({heuristic_result.confidence:.2f}) below OpenCV threshold {_method_threshold('opencv'):.2f}.",
                confidence=heuristic_result.confidence,
            )
        return heuristic_result

    async def _segment_with_sam2_provider(self, image_bytes: bytes, image_rgb: np.ndarray) -> SegmentationResult:
        if not self._replicate:
            raise SAM2Error("Replicate provider is not configured")

        sam2_result = await self._replicate.segment_with_sam2(
            image_bytes,
            SAM2Config(
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                min_mask_region_area=100,
            ),
        )
        return _build_result_from_mask(
            mask=sam2_result.mask,
            bbox=sam2_result.bbox,
            confidence=sam2_result.confidence,
            image_rgb=image_rgb,
            model="sam2",
            provider_debug=sam2_result.debug,
        )

    async def close(self) -> None:
        if self._replicate:
            await self._replicate.close()


_segmentation_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service


async def close_segmentation_service() -> None:
    global _segmentation_service
    if _segmentation_service is not None:
        await _segmentation_service.close()
        _segmentation_service = None


async def segment_vehicle_opencv_only(image_bytes: bytes) -> SegmentationResult:
    return await get_segmentation_service().segment_vehicle_opencv_only(image_bytes)


async def segment_vehicle(image_bytes: bytes) -> SegmentationResult:
    """
    Segment a vehicle using SAM2 via Replicate when available, otherwise fall back to OpenCV.
    """
    return await get_segmentation_service().segment_vehicle(image_bytes)
