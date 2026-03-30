#!/bin/bash
set -e

echo "=== Virtual Fitter Pipeline Tests ==="

mkdir -p tests/outputs

echo "Checking fixtures..."
python -m tests.download_fixtures

echo "Running pipeline tests..."
python -m tests.test_harness

echo "Analyzing results..."
python -m tests.tune_anchors analyze

echo "=== Tests complete ==="
echo "Review outputs in tests/outputs/"
