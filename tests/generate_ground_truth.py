"""
Interactive helper to generate or update fixture ground truth labels.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_PATH = FIXTURES_DIR / "ground_truth.json"

VALID_ANGLES = ["rear", "front", "side", "three_quarter", "unknown"]
VALID_VEHICLE_TYPES = ["sedan", "hatchback", "suv", "coupe", "truck", "unknown"]


def _default_products() -> dict:
    return {
        "rear-spoiler": {"placement_zone": "rear", "compatible_angles": ["rear", "three_quarter"]},
        "rear-diffuser": {"placement_zone": "rear", "compatible_angles": ["rear", "three_quarter"]},
        "side-skirts": {"placement_zone": "side_left", "compatible_angles": ["side", "three_quarter"]},
    }


def generate_ground_truth() -> None:
    if OUTPUT_PATH.exists():
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    else:
        data = {"images": {}, "products": {}, "expected_results": {}}

    car_images = sorted(list((FIXTURES_DIR / "cars").glob("*.jpg")) + list((FIXTURES_DIR / "cars").glob("*.png")))

    print("=" * 50)
    print("GROUND TRUTH LABELING")
    print("=" * 50)

    for img_path in car_images:
        img_name = img_path.name

        if img_name in data.get("images", {}):
            existing = data["images"][img_name]
            print(f"\n{img_name}: already labeled as '{existing.get('actual_angle', 'unknown')}'")
            update = input("  Update? (y/N): ").strip().lower()
            if update != "y":
                continue

        print(f"\n{img_name}")
        print(f"  Angles: {', '.join(VALID_ANGLES)}")

        while True:
            angle = input("  Actual angle: ").strip().lower()
            if angle in VALID_ANGLES:
                break
            print(f"  Invalid. Choose from: {', '.join(VALID_ANGLES)}")

        vehicle_type = input(f"  Vehicle type ({', '.join(VALID_VEHICLE_TYPES)}): ").strip().lower()
        if vehicle_type not in VALID_VEHICLE_TYPES:
            vehicle_type = "unknown"

        notes = input("  Notes (optional): ").strip()
        data.setdefault("images", {})[img_name] = {
            "actual_angle": angle,
            "vehicle_type": vehicle_type,
            "notes": notes,
        }

    products = _default_products()
    data["products"] = products
    data["expected_results"] = {}

    for img_name, img_data in data.get("images", {}).items():
        actual_angle = img_data.get("actual_angle", "unknown")
        data["expected_results"][img_name] = {}
        for product_key, product_data in products.items():
            if actual_angle in product_data["compatible_angles"]:
                data["expected_results"][img_name][product_key] = "success"
            else:
                data["expected_results"][img_name][product_key] = "VF_002"

    OUTPUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  {len(data.get('images', {}))} images labeled")
    print(f"  {len(data.get('expected_results', {}))} expected result groups")


if __name__ == "__main__":
    generate_ground_truth()
