#!/bin/bash
# Do NOT use set -e here — we want the scheduler loop to survive individual run failures.

FEATURE_MATRIX="storage/cache/feature_matrix.parquet"
MODEL_DIR="storage/models/saved"
# Bump this version when a code change requires a full pipeline rebuild
PIPELINE_VERSION="v6"
PIPELINE_VERSION_FILE="storage/cache/.pipeline_version"

echo "=== F5 Predictor Entrypoint ==="

# Start Telegram bot immediately so /status and /predict work even during pipeline build
python -c "
from notifications.telegram_bot import start_bot_thread
import time
start_bot_thread()
print('Telegram bot polling started')
# Keep this process alive — bot thread is daemon so we park here until pipeline script takes over
import signal, sys
signal.pause()
" &
BOT_PID=$!

# Read stored pipeline version first — if it doesn't match, delete matrix before
# trying to read it (avoids crashing on a corrupt matrix from a prior bad run).
STORED_VERSION=""
if [ -f "$PIPELINE_VERSION_FILE" ]; then
    STORED_VERSION=$(cat "$PIPELINE_VERSION_FILE")
fi

if [ "$STORED_VERSION" != "$PIPELINE_VERSION" ] && [ -f "$FEATURE_MATRIX" ]; then
    echo "Pipeline version changed ($STORED_VERSION -> $PIPELINE_VERSION). Removing stale matrix."
    rm -f "$FEATURE_MATRIX"
fi

# Validate feature matrix by file size (>1MB = valid).
# Avoids pyarrow/column-name issues that silently return 0 rows.
MATRIX_SIZE=0
if [ -f "$FEATURE_MATRIX" ]; then
    MATRIX_SIZE=$(stat -c%s "$FEATURE_MATRIX" 2>/dev/null || stat -f%z "$FEATURE_MATRIX" 2>/dev/null || echo 0)
fi

NEED_REBUILD=false
REBUILD_REASON=""

if [ ! -f "$FEATURE_MATRIX" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    NEED_REBUILD=true
    REBUILD_REASON="No data/models found"
elif [ "$MATRIX_SIZE" -lt 1048576 ]; then
    NEED_REBUILD=true
    REBUILD_REASON="Feature matrix too small (${MATRIX_SIZE} bytes — likely corrupt)"
    rm -f "$FEATURE_MATRIX"
fi

if [ "$NEED_REBUILD" = true ]; then
    echo "$REBUILD_REASON. Running full pipeline (2021-2025)..."
    python main.py pipeline --start-season 2021 --end-season 2025
    PIPELINE_EXIT=$?
    if [ $PIPELINE_EXIT -ne 0 ]; then
        echo "WARNING: Initial pipeline exited with code $PIPELINE_EXIT — continuing to scheduler anyway."
    else
        echo "$PIPELINE_VERSION" > "$PIPELINE_VERSION_FILE"
        echo "Initial pipeline complete."
    fi
else
    echo "Data and models found (${MATRIX_SIZE} bytes, $STORED_VERSION). Skipping initial pipeline."
fi

# Kill the standalone bot process — bot_runner.py starts its own thread
kill $BOT_PID 2>/dev/null

echo "Starting bot runner (scheduler + Telegram bot)..."
python bot_runner.py
