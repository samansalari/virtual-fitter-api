"""
Download sample car images for testing.
Uses royalty-free image URLs and falls back to generated fixtures.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CARS_DIR = FIXTURES_DIR / "cars"
OVERLAYS_DIR = FIXTURES_DIR / "overlays"

SAMPLE_URLS = [
    ("bmw_f30_rear", "https://images.unsplash.com/photo-1555215695-3004980ad54e?w=1400&fit=max&auto=format"),
    ("bmw_f30_threequarter", "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=1400&fit=max&auto=format"),
    ("bmw_f30_side", "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=1400&fit=max&auto=format"),
    ("mercedes_w177_rear", "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=1400&fit=max&auto=format"),
]


def _draw_car(draw: ImageDraw.ImageDraw, angle: str, body_color: tuple[int, int, int], canvas_width: int, canvas_height: int) -> None:
    if angle == "side":
        body = [(180, 420), (320, 330), (760, 330), (890, 400), (980, 400), (1040, 470), (210, 470)]
        windows = [(360, 345), (430, 290), (700, 290), (770, 345)]
        wheels = [(300, 450, 390, 540), (760, 450, 850, 540)]
    elif angle == "rear":
        body = [(300, 250), (380, 200), (660, 200), (740, 250), (790, 490), (250, 490)]
        windows = [(390, 235), (460, 205), (580, 205), (650, 235)]
        wheels = [(320, 430, 410, 520), (630, 430, 720, 520)]
    elif angle == "front":
        body = [(310, 250), (390, 190), (650, 190), (730, 250), (780, 500), (260, 500)]
        windows = [(410, 225), (470, 200), (570, 200), (630, 225)]
        wheels = [(320, 430, 410, 520), (630, 430, 720, 520)]
    else:
        body = [(220, 380), (360, 260), (690, 230), (870, 300), (950, 420), (1000, 500), (260, 500)]
        windows = [(420, 285), (520, 235), (700, 240), (790, 300)]
        wheels = [(340, 450, 440, 550), (760, 435, 860, 535)]

    draw.polygon(body, fill=body_color)
    draw.polygon(windows, fill=(195, 220, 235))
    for wheel in wheels:
        draw.ellipse(wheel, fill=(40, 40, 42))
        inset = 16
        draw.ellipse((wheel[0] + inset, wheel[1] + inset, wheel[2] - inset, wheel[3] - inset), fill=(118, 118, 120))

    if angle == "rear":
        draw.rectangle((365, 388, 645, 410), fill=(28, 28, 28))
        draw.rectangle((290, 300, 350, 340), fill=(200, 40, 40))
        draw.rectangle((690, 300, 750, 340), fill=(200, 40, 40))
    elif angle == "front":
        draw.rectangle((370, 398, 610, 420), fill=(28, 28, 28))
        draw.rectangle((300, 300, 360, 340), fill=(255, 240, 180))
        draw.rectangle((660, 300, 720, 340), fill=(255, 240, 180))


def _create_car_fixture(path: Path, angle: str, title: str, body_color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (1200, 700), (230, 233, 236))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 490, 1200, 700), fill=(82, 84, 88))
    draw.rectangle((0, 470, 1200, 490), fill=(235, 240, 246))
    draw.rectangle((0, 0, 1200, 470), fill=(216, 229, 238))

    _draw_car(draw, angle, body_color, 1200, 700)
    draw.text((42, 38), title, fill=(24, 24, 24))
    image.save(path, quality=92)


def _ensure_placeholder_cars() -> None:
    CARS_DIR.mkdir(parents=True, exist_ok=True)
    placeholders = {
        "bmw_f30_rear.jpg": ("rear", "BMW F30 Rear", (35, 35, 37)),
        "bmw_f30_threequarter.jpg": ("threequarter", "BMW F30 Three Quarter", (36, 38, 42)),
        "bmw_f30_side.jpg": ("side", "BMW F30 Side", (40, 40, 44)),
        "mercedes_w177_rear.jpg": ("rear", "Mercedes W177 Rear", (210, 210, 214)),
        "generic_sedan_rear.jpg": ("rear", "Generic Sedan Rear", (78, 100, 128)),
        "problematic_angle.jpg": ("front", "Problematic Front Angle", (95, 35, 35)),
    }

    for filename, (angle, title, body_color) in placeholders.items():
        path = CARS_DIR / filename
        if not path.exists():
            _create_car_fixture(path, angle, title, body_color)


def _create_spoiler_overlay() -> None:
    image = Image.new("RGBA", (420, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 82, 385, 112), radius=14, fill=(24, 24, 26, 235))
    draw.polygon([(68, 82), (116, 30), (138, 30), (110, 82)], fill=(30, 30, 32, 230))
    draw.polygon([(352, 82), (304, 30), (282, 30), (310, 82)], fill=(30, 30, 32, 230))
    image.save(OVERLAYS_DIR / "carbon-spoiler.png")


def _create_diffuser_overlay() -> None:
    image = Image.new("RGBA", (640, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 34, 620, 118), radius=28, fill=(22, 22, 24, 235))
    for x in range(90, 550, 92):
        draw.polygon([(x, 34), (x + 22, 34), (x + 40, 124), (x - 18, 124)], fill=(48, 48, 52, 225))
    image.save(OVERLAYS_DIR / "carbon-diffuser.png")


def _create_sideskirt_overlay() -> None:
    image = Image.new("RGBA", (860, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.polygon([(40, 120), (140, 92), (760, 84), (830, 114), (804, 146), (110, 154)], fill=(28, 28, 30, 235))
    draw.polygon([(150, 92), (248, 70), (745, 64), (790, 86), (742, 98), (182, 104)], fill=(48, 48, 52, 210))
    image.save(OVERLAYS_DIR / "carbon-sideskirt.png")


def create_placeholder_overlays() -> None:
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    _create_spoiler_overlay()
    _create_diffuser_overlay()
    _create_sideskirt_overlay()


async def download_fixtures() -> None:
    CARS_DIR.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(15.0, connect=8.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for name, url in SAMPLE_URLS:
            output_path = CARS_DIR / f"{name}.jpg"
            if output_path.exists():
                print(f"Skipping {name} (already exists)")
                continue

            try:
                print(f"Downloading {name}...")
                response = await client.get(url)
                response.raise_for_status()
                output_path.write_bytes(response.content)
                print(f"  Saved to {output_path}")
            except Exception as error:
                print(f"  Download failed for {name}: {error}")

    _ensure_placeholder_cars()
    create_placeholder_overlays()


if __name__ == "__main__":
    asyncio.run(download_fixtures())
