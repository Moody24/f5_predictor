#!/bin/bash
set -e

FEATURE_MATRIX="storage/cache/feature_matrix.parquet"
MODEL_DIR="storage/models/saved"

echo "=== F5 Predictor Entrypoint ==="

# Validate feature matrix: expect ~1 row per game (12k-15k). If bloated (>20k), rebuild.
MATRIX_ROWS=0
if [ -f "$FEATURE_MATRIX" ]; then
    MATRIX_ROWS=$(python -c "import pandas as pd; print(len(pd.read_parquet('$FEATURE_MATRIX', columns=['game_pk'])))")
fi

# If no feature matrix, no trained model, or matrix is corrupted (>20k rows), run full pipeline
if [ ! -f "$FEATURE_MATRIX" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ] || [ "$MATRIX_ROWS" -gt 20000 ]; then
    if [ "$MATRIX_ROWS" -gt 20000 ]; then
        echo "Feature matrix corrupted ($MATRIX_ROWS rows). Rebuilding..."
        rm -f "$FEATURE_MATRIX"
    else
        echo "No data/models found. Running initial pipeline (2021-2025)..."
    fi
    python main.py pipeline --start-season 2021 --end-season 2025
    echo "Initial pipeline complete."
else
    echo "Data and models found ($MATRIX_ROWS rows). Skipping initial pipeline."
fi

echo "Starting daily scheduler..."
python scheduler.py
