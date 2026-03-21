#!/bin/bash
set -e

FEATURE_MATRIX="data/cache/feature_matrix.parquet"
MODEL_DIR="data/models/saved"

echo "=== F5 Predictor Entrypoint ==="

# If no feature matrix or no trained model exists, run full pipeline first
if [ ! -f "$FEATURE_MATRIX" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "No data/models found. Running initial pipeline (2021-2025)..."
    python main.py pipeline --start-season 2021 --end-season 2025
    echo "Initial pipeline complete."
else
    echo "Data and models found. Skipping initial pipeline."
fi

echo "Starting daily scheduler..."
python scheduler.py
