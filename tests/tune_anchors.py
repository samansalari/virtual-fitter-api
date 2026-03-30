"""
Interactive anchor tuning tool.
Opens result image and lets you adjust placement parameters.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tests.test_harness import MOCK_PRODUCTS, OUTPUT_DIR, run_test

ANCHOR_STATE_PATH = OUTPUT_DIR / "anchor_overrides.json"


def load_anchor_state() -> dict[str, dict]:
    if ANCHOR_STATE_PATH.exists():
        return json.loads(ANCHOR_STATE_PATH.read_text(encoding="utf-8"))
    return {key: value.placement_anchors.copy() for key, value in MOCK_PRODUCTS.items()}


def save_anchor_state(anchor_state: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANCHOR_STATE_PATH.write_text(json.dumps(anchor_state, indent=2), encoding="utf-8")


def create_tuning_report(test_results_path: str):
    """
    Analyze test results and suggest anchor adjustments.
    """
    with open(test_results_path, encoding="utf-8") as file:
        payload = json.load(file)
    results = payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    suggestions = []

    for result in results:
        if not result["success"]:
            continue

        seg = result["stages"]["segmentation"]
        placement = result["stages"]["placement"]

        product = result["product"]
        detected_angle = seg["detected_angle"]

        if detected_angle == "three_quarter" and "rear" in product:
            suggestions.append(
                {
                    "image": result["car_image"],
                    "product": product,
                    "issue": "Three-quarter angle detected but product prefers a direct rear angle.",
                    "suggestion": "Consider perspective warp or reduce scale_factor by 0.1.",
                }
            )

        if seg["confidence"] < 0.7:
            suggestions.append(
                {
                    "image": result["car_image"],
                    "product": product,
                    "issue": f"Low segmentation confidence ({seg['confidence']:.2f})",
                    "suggestion": "Image may need manual review or a better capture angle.",
                }
            )

        if placement["placement_quality"] == "low":
            suggestions.append(
                {
                    "image": result["car_image"],
                    "product": product,
                    "issue": "Placement quality came back low.",
                    "suggestion": "Try reducing scale_factor slightly or using a cleaner matching angle.",
                }
            )

    return suggestions


def interactive_tuner():
    """
    CLI tool to manually adjust anchors and re-render.
    """
    anchor_state = load_anchor_state()

    print("Anchor Tuning Mode")
    print("Commands: adjust <product> x_offset|y_offset|scale_factor <delta>")
    print("          render <car_image_name> <product>")
    print("          show <product>")
    print("          save")
    print("          quit")

    while True:
        raw = input("> ").strip()
        if not raw:
            continue

        cmd = raw.split()
        action = cmd[0].lower()

        if action == "quit":
            break

        if action == "show" and len(cmd) == 2:
            product = cmd[1]
            print(json.dumps(anchor_state.get(product, {}), indent=2))
            continue

        if action == "adjust" and len(cmd) == 4:
            product, key, delta_raw = cmd[1], cmd[2], cmd[3]
            if product not in anchor_state:
                print(f"Unknown product: {product}")
                continue
            if key not in anchor_state[product]:
                print(f"Unknown anchor key: {key}")
                continue
            try:
                delta = float(delta_raw)
            except ValueError:
                print("Delta must be numeric.")
                continue
            anchor_state[product][key] = round(float(anchor_state[product][key]) + delta, 4)
            MOCK_PRODUCTS[product].placement_anchors = anchor_state[product].copy()
            print(f"{product}.{key} = {anchor_state[product][key]}")
            continue

        if action == "render" and len(cmd) == 3:
            car_image_name, product = cmd[1], cmd[2]
            car_path = Path(__file__).parent / "fixtures" / "cars" / car_image_name
            if not car_path.exists():
                print(f"Fixture not found: {car_path}")
                continue
            if product not in MOCK_PRODUCTS:
                print(f"Unknown product: {product}")
                continue

            MOCK_PRODUCTS[product].placement_anchors = anchor_state[product].copy()
            output_name = f"tuned_{car_path.stem}_{product}"
            result = asyncio.run(run_test(str(car_path), product, output_name))
            print(json.dumps(result, indent=2))
            continue

        if action == "save":
            save_anchor_state(anchor_state)
            print(f"Saved overrides to {ANCHOR_STATE_PATH}")
            continue

        print("Unknown command.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        suggestions = create_tuning_report(str(OUTPUT_DIR / "test_report.json"))
        for suggestion in suggestions:
            print(f"\n{suggestion['image']} + {suggestion['product']}:")
            print(f"  Issue: {suggestion['issue']}")
            print(f"  Suggestion: {suggestion['suggestion']}")
    else:
        interactive_tuner()
