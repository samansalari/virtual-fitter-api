"""
Semantic test runner for the Virtual Fitter pipeline.
Uses local fixture images and mock product data without Shopify dependencies.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from app.services.placement import PlacementConfig, calculate_placement, check_angle_compatibility
from app.services.renderer_overlay import render_overlay
from app.services.segmentation import SegmentationResult, segment_vehicle
from app.services.validation import VirtualFitterError

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "outputs"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"
SAM2_AVAILABLE = bool(os.getenv("REPLICATE_API_TOKEN"))


@dataclass
class MockProduct:
    product_type: str
    placement_zone: str
    overlay_path: str
    placement_anchors: dict


@dataclass
class AngleClassificationResult:
    image_name: str
    actual_angle: str
    detected_angle: str
    is_correct: bool
    confidence: float
    debug_reason: str


@dataclass
class PlacementCompatResult:
    image_name: str
    product_key: str
    expected_outcome: str
    actual_outcome: str
    is_correct: bool
    detected_angle: str


@dataclass
class SemanticTestReport:
    angle_results: list[AngleClassificationResult] = field(default_factory=list)
    angle_accuracy: float = 0.0
    angle_confusion_matrix: dict[str, int] = field(default_factory=dict)
    placement_results: list[PlacementCompatResult] = field(default_factory=list)
    placement_accuracy: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    raw_success_count: int = 0
    raw_failure_count: int = 0
    total_tests: int = 0


MOCK_PRODUCTS = {
    "rear-spoiler": MockProduct(
        product_type="spoiler",
        placement_zone="rear",
        overlay_path="fixtures/overlays/carbon-spoiler.png",
        placement_anchors={"x_offset": 0.5, "y_offset": 0.15, "scale_factor": 0.4, "allow_mirror": False},
    ),
    "rear-diffuser": MockProduct(
        product_type="diffuser",
        placement_zone="rear",
        overlay_path="fixtures/overlays/carbon-diffuser.png",
        placement_anchors={"x_offset": 0.5, "y_offset": 0.85, "scale_factor": 0.5, "allow_mirror": False},
    ),
    "side-skirts": MockProduct(
        product_type="side_skirt",
        placement_zone="side_left",
        overlay_path="fixtures/overlays/carbon-sideskirt.png",
        placement_anchors={"x_offset": 0.5, "y_offset": 0.7, "scale_factor": 0.8, "allow_mirror": True},
    ),
}


def _load_rgb_array(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.array(image.convert("RGB"))


def _draw_debug_overlay(
    base_image: np.ndarray,
    vehicle_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    detected_angle: str,
    product_key: str,
) -> Image.Image:
    image = Image.fromarray(base_image.copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle(vehicle_bbox, outline=(0, 220, 100), width=5)
    draw.rectangle(target_bbox, outline=(220, 50, 50), width=4)
    draw.text((vehicle_bbox[0] + 12, max(12, vehicle_bbox[1] - 26)), f"vehicle: {detected_angle}", fill=(0, 220, 100))
    draw.text((target_bbox[0] + 12, max(12, target_bbox[1] - 26)), f"target: {product_key}", fill=(220, 50, 50))
    return image


def load_ground_truth() -> dict:
    if GROUND_TRUTH_PATH.exists():
        return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return {"images": {}, "products": {}, "expected_results": {}}


def evaluate_angle_classification(
    image_name: str,
    detected_angle: str,
    confidence: float,
    debug_reason: str,
    ground_truth: dict,
) -> AngleClassificationResult:
    image_gt = ground_truth.get("images", {}).get(image_name, {})
    actual_angle = image_gt.get("actual_angle", "unknown")
    is_correct = detected_angle == actual_angle

    if not is_correct:
        if actual_angle == "three_quarter" and detected_angle in {"rear", "side"}:
            is_correct = True
        elif detected_angle == "three_quarter" and actual_angle in {"rear", "side"}:
            is_correct = True

    return AngleClassificationResult(
        image_name=image_name,
        actual_angle=actual_angle,
        detected_angle=detected_angle,
        is_correct=is_correct,
        confidence=confidence,
        debug_reason=debug_reason,
    )


def evaluate_placement_compatibility(
    image_name: str,
    product_key: str,
    actual_success: bool,
    actual_error_code: Optional[str],
    detected_angle: str,
    ground_truth: dict,
) -> PlacementCompatResult:
    expected = ground_truth.get("expected_results", {}).get(image_name, {}).get(product_key, "unknown")
    actual_outcome = "success" if actual_success else (actual_error_code or "error")

    if expected == "success":
        is_correct = actual_success
    elif expected == "VF_002":
        is_correct = actual_error_code == "VF_002"
    else:
        is_correct = False

    return PlacementCompatResult(
        image_name=image_name,
        product_key=product_key,
        expected_outcome=expected,
        actual_outcome=actual_outcome,
        is_correct=is_correct,
        detected_angle=detected_angle,
    )


def _save_segmentation_debug(image_name: str, seg_result: SegmentationResult, car_bytes: bytes) -> None:
    debug_dir = OUTPUT_DIR / f"{Path(image_name).stem}_segmentation_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_image = Image.fromarray((seg_result.vehicle_mask * 255).astype(np.uint8))
    mask_image.save(debug_dir / "segmentation_mask.png")

    if seg_result.debug:
        feature_payload = {
            "segmentation_model": seg_result.debug.segmentation_model,
            "bbox_aspect_ratio": seg_result.debug.bbox_aspect_ratio,
            "bbox_fill_ratio": seg_result.debug.bbox_fill_ratio,
            "centroid_x_ratio": seg_result.debug.centroid_x_ratio,
            "centroid_y_ratio": seg_result.debug.centroid_y_ratio,
            "global_symmetry": seg_result.debug.global_symmetry,
            "upper_half_symmetry": seg_result.debug.upper_half_symmetry,
            "lower_half_symmetry": seg_result.debug.lower_half_symmetry,
            "upper_half_mass": seg_result.debug.upper_half_mass,
            "lower_half_mass": seg_result.debug.lower_half_mass,
            "left_half_mass": seg_result.debug.left_half_mass,
            "right_half_mass": seg_result.debug.right_half_mass,
            "wheel_score_left": seg_result.debug.wheel_score_left,
            "wheel_score_right": seg_result.debug.wheel_score_right,
            "wheel_symmetry": seg_result.debug.wheel_symmetry,
            "horizontal_edge_ratio": seg_result.debug.horizontal_edge_ratio,
            "vertical_edge_ratio": seg_result.debug.vertical_edge_ratio,
            "lower_red_ratio": seg_result.debug.lower_red_ratio,
            "upper_brightness": seg_result.debug.upper_brightness,
            "lower_brightness": seg_result.debug.lower_brightness,
            "brightness_gradient": seg_result.debug.brightness_gradient,
            "dark_center_ratio": seg_result.debug.dark_center_ratio,
            "angle_confidence": seg_result.debug.angle_confidence,
            "angle_decision_reason": seg_result.debug.angle_decision_reason,
        }
        (debug_dir / "angle_features.json").write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

    image = _load_rgb_array(car_bytes)
    overlay = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(seg_result.vehicle_bbox, outline=(0, 220, 100), width=5)
    draw.text((seg_result.vehicle_bbox[0] + 12, max(12, seg_result.vehicle_bbox[1] - 26)), seg_result.detected_angle, fill=(0, 220, 100))
    overlay.save(debug_dir / "bbox_debug.png")


async def _run_product_render(
    image_name: str,
    car_bytes: bytes,
    seg_result: SegmentationResult,
    product_key: str,
    product: MockProduct,
) -> tuple[bool, Optional[str]]:
    overlay_path = (Path(__file__).parent / product.overlay_path).resolve()
    output_name = f"{Path(image_name).stem}_{product_key}"
    debug_dir = OUTPUT_DIR / f"{output_name}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        placement = calculate_placement(
            segmentation=seg_result,
            config=PlacementConfig(
                product_type=product.product_type,
                placement_zone=product.placement_zone,
                anchors=product.placement_anchors,
                overlay_url=str(overlay_path),
                mask_url=None,
            ),
            image_dimensions=(seg_result.vehicle_mask.shape[1], seg_result.vehicle_mask.shape[0]),
        )

        car_image = _load_rgb_array(car_bytes)
        composite = await render_overlay(
            base_image=car_image,
            overlay_url=str(overlay_path),
            placement=placement,
            vehicle_mask=seg_result.vehicle_mask,
        )

        output_path = OUTPUT_DIR / f"{output_name}.jpg"
        Image.fromarray(composite).save(output_path, quality=90)

        placement_debug = {
            "target_bbox": list(placement.target_bbox),
            "placement_quality": placement.placement_quality,
            "mirrored": placement.mirrored,
            "compatibility_score": placement.metadata.get("compatibility_score"),
            "compatibility_message": placement.metadata.get("compatibility_message"),
        }
        (debug_dir / "placement.json").write_text(json.dumps(placement_debug, indent=2), encoding="utf-8")

        debug_overlay = _draw_debug_overlay(
            base_image=car_image,
            vehicle_bbox=seg_result.vehicle_bbox,
            target_bbox=placement.target_bbox,
            detected_angle=seg_result.detected_angle,
            product_key=product_key,
        )
        debug_overlay.save(debug_dir / "placement_debug.png")
        return True, None
    except VirtualFitterError as error:
        return False, error.code
    except Exception:
        return False, "RENDER_ERROR"


def _finalize_report(report: SemanticTestReport) -> SemanticTestReport:
    if report.angle_results:
        report.angle_accuracy = sum(1 for result in report.angle_results if result.is_correct) / len(report.angle_results)

    if report.placement_results:
        report.placement_accuracy = sum(1 for result in report.placement_results if result.is_correct) / len(report.placement_results)

    for result in report.angle_results:
        key = f"{result.actual_angle} -> {result.detected_angle}"
        report.angle_confusion_matrix[key] = report.angle_confusion_matrix.get(key, 0) + 1

    for result in report.placement_results:
        if result.expected_outcome == "success":
            if result.actual_outcome == "success":
                report.true_positives += 1
            else:
                report.false_negatives += 1
        else:
            if result.actual_outcome != "success":
                report.true_negatives += 1
            else:
                report.false_positives += 1

    return report


def print_semantic_report(report: SemanticTestReport) -> None:
    print("\n" + "=" * 60)
    print("SEMANTIC TEST REPORT")
    print("=" * 60)

    print("\nANGLE CLASSIFICATION")
    print("-" * 40)
    print(f"Accuracy: {report.angle_accuracy:.1%}")
    print("\nConfusion Matrix:")
    for key, count in sorted(report.angle_confusion_matrix.items()):
        actual, detected = key.split(" -> ")
        status = "OK" if actual == detected else "MISS"
        print(f"  {status:4} {key}: {count}")

    print("\nSEGMENTATION MODEL")
    print("-" * 40)
    if SAM2_AVAILABLE:
        print("Primary: SAM2 via Replicate")
        print("Fallback: OpenCV")
    else:
        print("OpenCV only (set REPLICATE_API_TOKEN for SAM2)")

    print("\nPLACEMENT COMPATIBILITY")
    print("-" * 40)
    print(f"Accuracy: {report.placement_accuracy:.1%}")
    print(f"  True Positives:  {report.true_positives}")
    print(f"  True Negatives:  {report.true_negatives}")
    print(f"  False Positives: {report.false_positives}")
    print(f"  False Negatives: {report.false_negatives}")

    print("\nRAW STATS")
    print("-" * 40)
    print(f"Total tests: {report.total_tests}")
    print(f"Raw success: {report.raw_success_count}")
    print(f"Raw failure: {report.raw_failure_count}")
    raw_rate = (report.raw_success_count / report.total_tests) if report.total_tests else 0.0
    print(f"Raw rate:    {raw_rate:.1%}")

    false_negatives = [result for result in report.placement_results if result.expected_outcome == "success" and result.actual_outcome != "success"]
    false_positives = [result for result in report.placement_results if result.expected_outcome != "success" and result.actual_outcome == "success"]

    if false_negatives:
        print("\nFALSE NEGATIVES")
        for result in false_negatives:
            print(f"  - {result.image_name} + {result.product_key}: expected success, got {result.actual_outcome}")

    if false_positives:
        print("\nFALSE POSITIVES")
        for result in false_positives:
            print(f"  - {result.image_name} + {result.product_key}: expected {result.expected_outcome}, got success")

    print("\n" + "=" * 60)


async def run_all_fixtures_semantic() -> SemanticTestReport:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth()
    report = SemanticTestReport()
    car_images = sorted(list((FIXTURES_DIR / "cars").glob("*.jpg")) + list((FIXTURES_DIR / "cars").glob("*.png")))

    for car_path in car_images:
        image_name = car_path.name
        car_bytes = car_path.read_bytes()
        segmentation_success = False
        seg_result: Optional[SegmentationResult] = None
        segmentation_error_code: Optional[str] = None

        try:
            seg_result = await segment_vehicle(car_bytes)
            segmentation_success = True
            _save_segmentation_debug(image_name, seg_result, car_bytes)
            report.angle_results.append(
                evaluate_angle_classification(
                    image_name=image_name,
                    detected_angle=seg_result.detected_angle,
                    confidence=seg_result.confidence,
                    debug_reason=seg_result.debug.angle_decision_reason if seg_result.debug else "",
                    ground_truth=ground_truth,
                )
            )
        except VirtualFitterError as error:
            segmentation_error_code = error.code
            report.angle_results.append(
                AngleClassificationResult(
                    image_name=image_name,
                    actual_angle=ground_truth.get("images", {}).get(image_name, {}).get("actual_angle", "unknown"),
                    detected_angle="unknown",
                    is_correct=False,
                    confidence=0.0,
                    debug_reason=error.message,
                )
            )

        for product_key, product in MOCK_PRODUCTS.items():
            report.total_tests += 1

            if not segmentation_success or seg_result is None:
                report.raw_failure_count += 1
                report.placement_results.append(
                    evaluate_placement_compatibility(
                        image_name=image_name,
                        product_key=product_key,
                        actual_success=False,
                        actual_error_code=segmentation_error_code or "VF_003",
                        detected_angle="unknown",
                        ground_truth=ground_truth,
                    )
                )
                continue

            is_compatible, _, _ = check_angle_compatibility(seg_result.detected_angle, product.placement_zone)
            if not is_compatible:
                report.raw_failure_count += 1
                report.placement_results.append(
                    evaluate_placement_compatibility(
                        image_name=image_name,
                        product_key=product_key,
                        actual_success=False,
                        actual_error_code="VF_002",
                        detected_angle=seg_result.detected_angle,
                        ground_truth=ground_truth,
                    )
                )
                continue

            render_success, render_error_code = await _run_product_render(
                image_name=image_name,
                car_bytes=car_bytes,
                seg_result=seg_result,
                product_key=product_key,
                product=product,
            )

            if render_success:
                report.raw_success_count += 1
            else:
                report.raw_failure_count += 1

            report.placement_results.append(
                evaluate_placement_compatibility(
                    image_name=image_name,
                    product_key=product_key,
                    actual_success=render_success,
                    actual_error_code=render_error_code,
                    detected_angle=seg_result.detected_angle,
                    ground_truth=ground_truth,
                )
            )

    return _finalize_report(report)


def _serialize_report(report: SemanticTestReport) -> dict:
    raw_rate = (report.raw_success_count / report.total_tests) if report.total_tests else 0.0
    return {
        "summary": {
            "segmentation_mode": "sam2+opencv" if SAM2_AVAILABLE else "opencv",
            "angle_accuracy": f"{report.angle_accuracy:.1%}",
            "placement_accuracy": f"{report.placement_accuracy:.1%}",
            "true_positives": report.true_positives,
            "true_negatives": report.true_negatives,
            "false_positives": report.false_positives,
            "false_negatives": report.false_negatives,
            "raw_success_rate": f"{raw_rate:.1%}",
            "raw_success_count": report.raw_success_count,
            "raw_failure_count": report.raw_failure_count,
            "total_tests": report.total_tests,
        },
        "angle_confusion_matrix": report.angle_confusion_matrix,
        "angle_results": [asdict(result) for result in report.angle_results],
        "placement_results": [asdict(result) for result in report.placement_results],
    }


async def run_all_fixtures() -> SemanticTestReport:
    report = await run_all_fixtures_semantic()
    print_semantic_report(report)
    report_path = OUTPUT_DIR / "test_report.json"
    report_path.write_text(json.dumps(_serialize_report(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    asyncio.run(run_all_fixtures())
