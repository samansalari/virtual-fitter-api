from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .segmentation import SegmentationResult
from .validation import IncompatiblePlacementAngle


@dataclass
class PlacementConfig:
    product_type: str
    placement_zone: str
    anchors: dict
    overlay_url: str
    mask_url: Optional[str] = None
    compatible_models: list[str] = field(default_factory=list)
    render_prompt: Optional[str] = None


@dataclass
class PlacementResult:
    transform_matrix: np.ndarray
    target_bbox: tuple[int, int, int, int]
    blend_mode: str
    feather_radius: int
    perspective_warp: Optional[np.ndarray]
    mirrored: bool = False
    placement_quality: str = "medium"
    metadata: dict[str, object] = field(default_factory=dict)


ZONE_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "front": {"center": (0.5, 0.36), "size": (0.46, 0.18)},
    "rear": {"center": (0.5, 0.82), "size": (0.54, 0.20)},
    "side_left": {"center": (0.24, 0.62), "size": (0.30, 0.18)},
    "side_right": {"center": (0.76, 0.62), "size": (0.30, 0.18)},
    "hood": {"center": (0.5, 0.38), "size": (0.35, 0.22)},
    "roof": {"center": (0.5, 0.26), "size": (0.38, 0.15)},
    "full_body": {"center": (0.5, 0.54), "size": (0.92, 0.88)},
}


BLEND_MODES: dict[str, str] = {
    "badge": "overlay",
    "mirror_cap": "overlay",
    "wheel": "multiply",
}


ANGLE_COMPATIBILITY: dict[tuple[str, str], float] = {
    ("rear", "rear"): 1.0,
    ("three_quarter", "rear"): 1.0,
    ("side", "rear"): 0.0,
    ("front", "rear"): 0.0,
    ("rear", "front"): 0.0,
    ("three_quarter", "front"): 0.6,
    ("front", "front"): 1.0,
    ("side", "front"): 0.35,
    ("side", "side_left"): 1.0,
    ("side", "side_right"): 1.0,
    ("three_quarter", "side_left"): 0.8,
    ("three_quarter", "side_right"): 0.8,
    ("rear", "side_left"): 0.0,
    ("rear", "side_right"): 0.0,
    ("front", "side_left"): 0.0,
    ("front", "side_right"): 0.0,
    ("side", "full_body"): 0.95,
    ("three_quarter", "full_body"): 0.8,
    ("rear", "full_body"): 0.0,
    ("front", "full_body"): 0.0,
    ("rear", "hood"): 0.0,
    ("front", "hood"): 1.0,
    ("three_quarter", "hood"): 0.55,
    ("rear", "roof"): 0.35,
    ("front", "roof"): 0.45,
    ("side", "roof"): 0.75,
    ("three_quarter", "roof"): 0.85,
}


def _normalized_to_matrix(dest_quad: np.ndarray) -> np.ndarray:
    src_norm = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_norm, dest_quad.astype(np.float32))


def _target_rect(vehicle_bbox: tuple[int, int, int, int], zone: str, anchors: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = vehicle_bbox
    vehicle_width = max(x2 - x1, 1)
    vehicle_height = max(y2 - y1, 1)
    zone_preset = ZONE_PRESETS.get(zone, ZONE_PRESETS["rear"])

    center_x_ratio, center_y_ratio = zone_preset["center"]
    size_w_ratio, size_h_ratio = zone_preset["size"]

    x_offset = float(anchors.get("x_offset", 0.0))
    y_offset = float(anchors.get("y_offset", 0.0))
    scale_factor = float(anchors.get("scale_factor", 1.0))

    center_x = x1 + (vehicle_width * center_x_ratio) + (vehicle_width * (x_offset - 0.5) * 0.18)
    center_y = y1 + (vehicle_height * center_y_ratio) + (vehicle_height * (y_offset - 0.5) * 0.18)
    target_width = vehicle_width * size_w_ratio * max(scale_factor, 0.15)
    target_height = vehicle_height * size_h_ratio * max(scale_factor, 0.15)

    rect_x1 = center_x - (target_width / 2)
    rect_y1 = center_y - (target_height / 2)
    rect_x2 = center_x + (target_width / 2)
    rect_y2 = center_y + (target_height / 2)
    return rect_x1, rect_y1, rect_x2, rect_y2


def _perspective_quad(
    rect: tuple[float, float, float, float],
    angle: str,
    zone: str,
    view_side: Optional[str],
) -> np.ndarray:
    x1, y1, x2, y2 = rect
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    if angle == "three_quarter":
        skew = width * 0.16
        if view_side == "left":
            return np.array(
                [[x1 + skew, y1], [x2, y1 + height * 0.06], [x2 - skew, y2], [x1, y2 - height * 0.04]],
                dtype=np.float32,
            )
        return np.array(
            [[x1, y1 + height * 0.06], [x2 - skew, y1], [x2, y2 - height * 0.04], [x1 + skew, y2]],
            dtype=np.float32,
        )

    if angle == "rear" and zone == "rear":
        inset = width * 0.04
        return np.array(
            [[x1 + inset, y1], [x2 - inset, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )

    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def check_angle_compatibility(detected_angle: str, product_zone: str) -> tuple[bool, float, str]:
    score = ANGLE_COMPATIBILITY.get((detected_angle, product_zone), 0.0)
    if score >= 0.8:
        return True, score, "Good match"
    if score >= 0.5:
        return True, score, f"Partial match: {detected_angle} view can support {product_zone} placement."
    if detected_angle == "front":
        return False, score, f"Front view cannot display {product_zone} products. Please upload a rear or side photo."
    return False, score, f"Incompatible: {detected_angle} angle cannot show {product_zone} products."


def _validate_zone_angle(segmentation: SegmentationResult, config: PlacementConfig) -> tuple[bool, bool, float, str]:
    zone = config.placement_zone
    angle = segmentation.detected_angle
    mirrored = False
    perspective = False

    compatible, compatibility_score, compatibility_message = check_angle_compatibility(angle, zone)

    if zone == "rear":
        if compatible:
            return mirrored, angle == "three_quarter", compatibility_score, compatibility_message
        raise IncompatiblePlacementAngle(
            "Rear products need a rear or rear three-quarter vehicle photo.",
            detected_angle=angle,
            product_zone=zone,
            confidence=segmentation.confidence,
        )

    if zone in {"side_left", "side_right"}:
        if not compatible:
            raise IncompatiblePlacementAngle(
                "Side products need a side or three-quarter vehicle photo.",
                detected_angle=angle,
                product_zone=zone,
                confidence=segmentation.confidence,
            )

        if angle == "three_quarter":
            if segmentation.view_side and segmentation.view_side not in zone:
                mirrored = bool(config.anchors.get("allow_mirror", True))
                if not mirrored:
                    raise IncompatiblePlacementAngle(
                        "The visible side of the car does not match this part placement.",
                        detected_angle=angle,
                        product_zone=zone,
                        confidence=segmentation.confidence,
                    )
            return mirrored, True, compatibility_score, compatibility_message

        if angle == "side":
            if segmentation.view_side and segmentation.view_side not in zone:
                mirrored = bool(config.anchors.get("allow_mirror", True))
                if not mirrored:
                    raise IncompatiblePlacementAngle(
                        "The visible side of the car does not match this part placement.",
                        detected_angle=angle,
                        product_zone=zone,
                        confidence=segmentation.confidence,
                    )
            return mirrored, False, compatibility_score, compatibility_message

    if zone == "full_body":
        if not compatible:
            raise IncompatiblePlacementAngle(
                "Body kits work best with side or three-quarter photos.",
                detected_angle=angle,
                product_zone=zone,
                confidence=segmentation.confidence,
            )
        return mirrored, angle == "three_quarter", compatibility_score, compatibility_message

    if zone in {"front", "hood", "roof"} and not compatible:
        raise IncompatiblePlacementAngle(
            "This product needs a front or side-oriented photo.",
            detected_angle=angle,
            product_zone=zone,
            confidence=segmentation.confidence,
        )

    if angle == "three_quarter":
        perspective = True

    return mirrored, perspective, compatibility_score, compatibility_message


def calculate_placement(
    segmentation: SegmentationResult,
    config: PlacementConfig,
    image_dimensions: tuple[int, int],
) -> PlacementResult:
    """
    Calculate where and how to place product overlay based on:
    - Detected vehicle angle
    - Product placement zone
    - Anchor offsets from metafields
    """
    image_width, image_height = image_dimensions
    mirrored, needs_perspective, compatibility_score, compatibility_message = _validate_zone_angle(segmentation, config)
    rect = _target_rect(segmentation.vehicle_bbox, config.placement_zone, config.anchors)

    rect = (
        max(0.0, rect[0]),
        max(0.0, rect[1]),
        min(float(image_width - 1), rect[2]),
        min(float(image_height - 1), rect[3]),
    )
    dest_quad = _perspective_quad(rect, segmentation.detected_angle if needs_perspective else "rear", config.placement_zone, segmentation.view_side)
    if not needs_perspective:
        dest_quad = _perspective_quad(rect, "rear", config.placement_zone, segmentation.view_side)

    x_coords = dest_quad[:, 0]
    y_coords = dest_quad[:, 1]
    target_bbox = (
        int(np.floor(x_coords.min())),
        int(np.floor(y_coords.min())),
        int(np.ceil(x_coords.max())),
        int(np.ceil(y_coords.max())),
    )

    placement_quality = "high"
    if needs_perspective or segmentation.confidence < 0.82 or compatibility_score < 0.85:
        placement_quality = "medium"
    if segmentation.confidence < 0.65 or compatibility_score < 0.65:
        placement_quality = "low"

    return PlacementResult(
        transform_matrix=_normalized_to_matrix(dest_quad),
        target_bbox=target_bbox,
        blend_mode=BLEND_MODES.get(config.product_type, "normal"),
        feather_radius=max(6, int((target_bbox[2] - target_bbox[0]) * 0.035)),
        perspective_warp=dest_quad if needs_perspective else None,
        mirrored=mirrored,
        placement_quality=placement_quality,
        metadata={
            "zone": config.placement_zone,
            "product_type": config.product_type,
            "view_side": segmentation.view_side,
            "compatibility_score": compatibility_score,
            "compatibility_message": compatibility_message,
        },
    )
