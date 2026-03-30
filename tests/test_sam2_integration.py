"""
Integration tests for SAM2 segmentation through Replicate.

These tests require REPLICATE_API_TOKEN. Without that token they exit cleanly.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ai_providers.replicate_provider import ReplicateProvider, SAM2Config, SAM2Error
from app.services.segmentation import SegmentationService, segment_vehicle_opencv_only


async def test_sam2_single_image() -> bool:
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("REPLICATE_API_TOKEN not set - skipping SAM2 test")
        return False

    fixtures_dir = Path(__file__).parent / "fixtures" / "cars"
    test_images = sorted(fixtures_dir.glob("*.jpg"))
    if not test_images:
        print("No test images found")
        return False

    test_image = test_images[0]
    print(f"Testing SAM2 with: {test_image.name}")
    image_bytes = test_image.read_bytes()

    provider = ReplicateProvider(api_token)
    try:
        result = await provider.segment_with_sam2(image_bytes, SAM2Config(points_per_side=32))
        print("SAM2 success")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Bbox: {result.bbox}")
        print(f"  Mask shape: {result.mask.shape}")
        print(f"  Debug: {result.debug}")

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        mask_path = output_dir / f"{test_image.stem}_sam2_mask.png"
        Image.fromarray((result.mask * 255).astype("uint8")).save(mask_path)
        print(f"  Saved mask to: {mask_path}")
        return True
    except SAM2Error as exc:
        print(f"SAM2 error: {exc}")
        return False
    finally:
        await provider.close()


async def test_sam2_vs_opencv() -> None:
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("REPLICATE_API_TOKEN not set - skipping comparison")
        return

    fixtures_dir = Path(__file__).parent / "fixtures" / "cars"
    print("\n" + "=" * 60)
    print("SAM2 vs OpenCV Comparison")
    print("=" * 60)

    service = SegmentationService()
    try:
        for img_path in sorted(fixtures_dir.glob("*.jpg")):
            image_bytes = img_path.read_bytes()
            print(f"\n{img_path.name}")
            try:
                sam2_result = await service.segment_vehicle(image_bytes, prefer_sam2=True)
                opencv_result = await segment_vehicle_opencv_only(image_bytes)

                sam2_model = sam2_result.debug.segmentation_model if sam2_result.debug else sam2_result.source
                opencv_model = opencv_result.debug.segmentation_model if opencv_result.debug else opencv_result.source

                print(f"  {sam2_model}: conf={sam2_result.confidence:.2f}, angle={sam2_result.detected_angle}")
                print(f"  {opencv_model}: conf={opencv_result.confidence:.2f}, angle={opencv_result.detected_angle}")
                print("  Angles match" if sam2_result.detected_angle == opencv_result.detected_angle else "  Angle mismatch")
            except Exception as exc:  # pragma: no cover - integration safety
                print(f"  Error: {exc}")
    finally:
        await service.close()


if __name__ == "__main__":
    print("SAM2 Integration Tests")
    print("=" * 60)
    success = asyncio.run(test_sam2_single_image())
    if success:
        asyncio.run(test_sam2_vs_opencv())
    print("\n" + "=" * 60)
    print("Tests complete")
