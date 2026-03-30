# Virtual Fitter Tests

This folder contains the standalone integration harness for the Phase 2 overlay pipeline.

## Commands

```bash
python -m tests.download_fixtures
python -m tests.test_harness
python -m tests.tune_anchors analyze
```

## What gets generated

- `tests/fixtures/cars/` contains downloaded or synthetic car fixtures
- `tests/fixtures/overlays/` contains transparent overlay assets
- `tests/outputs/` contains rendered results, debug masks, and `test_report.json`

## Why this exists

The test harness lets us tune:

- segmentation confidence thresholds
- angle detection edge cases
- product anchor presets
- overlay scaling and placement

without needing live Shopify Admin credentials.
