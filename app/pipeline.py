from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_input_image(file_bytes: bytes, destination: Path) -> None:
    ensure_parent(destination)
    destination.write_bytes(file_bytes)


def build_demo_render(
    source_path: Path,
    destination: Path,
    product_title: str,
    placement_hint: str,
) -> None:
    ensure_parent(destination)

    with Image.open(source_path).convert("RGBA") as image:
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        band_height = max(78, int(height * 0.14))
        draw.rounded_rectangle(
            (
                int(width * 0.05),
                height - band_height - int(height * 0.05),
                int(width * 0.95),
                height - int(height * 0.05),
            ),
            radius=28,
            fill=(17, 17, 17, 210),
        )

        badge_width = max(180, int(width * 0.28))
        badge_height = max(44, int(height * 0.08))
        badge_left = int(width * 0.08)
        badge_top = int(height * 0.08)

        draw.rounded_rectangle(
            (
                badge_left,
                badge_top,
                badge_left + badge_width,
                badge_top + badge_height,
            ),
            radius=24,
            fill=(200, 16, 46, 235),
        )
        draw.text((badge_left + 18, badge_top + 12), "DEMO RENDER", fill=(255, 255, 255, 255))
        draw.text(
            (int(width * 0.08), height - band_height - int(height * 0.02)),
            product_title[:80],
            fill=(255, 255, 255, 255),
        )
        draw.text(
            (int(width * 0.08), height - int(height * 0.085)),
            f"Pipeline placeholder for {placement_hint}",
            fill=(222, 222, 222, 255),
        )

        composed = Image.alpha_composite(image, overlay).convert("RGB")
        composed.save(destination, format="JPEG", quality=92)
