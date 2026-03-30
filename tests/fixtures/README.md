# Virtual Fitter Test Fixtures

## Adding Car Images

Place customer-representative car photos in `cars/`.

Naming convention:

- `{make}_{model}_{angle}.{ext}`

Angles to test:

- `rear` for spoilers and diffusers
- `threequarter` for common customer photos
- `side` for side skirts and wheels
- `front` for edge-case validation

Image requirements:

- 1000px or larger on the longest edge
- JPG or PNG
- Vehicle fills at least half the frame when possible
- Clear daylight images work best

## Adding Product Overlays

Place transparent PNG product images in `overlays/`.

Overlay rules:

- PNG with alpha transparency
- tightly cropped around the product
- matched to expected placement angle
- 500px to 1000px on the longest edge is enough for fixture testing

## Running Tests

```bash
cd render-service
python -m tests.download_fixtures
python -m tests.test_harness
python -m tests.tune_anchors analyze
```

Results are written to `tests/outputs/`.
