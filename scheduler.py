"""
F5 Predictor Daily Scheduler
------------------------------
Orchestrates the daily prediction pipeline:
  1. Check yesterday's accuracy
  2. Incremental data fetch (new games since last run)
  3. Conditional retrain (if enough new games accumulated)
  4. Generate today's predictions
  5. Send notifications (Phase 10)

Usage:
  python scheduler.py                    # Run full daily pipeline
  python scheduler.py --skip-retrain     # Skip retraining, just predict
  python scheduler.py --force-retrain    # Force retrain even if not due

Cron setup (10 AM ET daily):
  0 10 * * * cd /path/to/f5_predictor && python scheduler.py >> data/logs/scheduler.log 2>&1
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import DATA_DIR, PREDICTIONS_DIR
from data.fetchers.mlb_stats import MLBStatsFetcher
from data.fetchers.weather import WeatherFetcher
from data.fetchers.umpire import UmpireFetcher
from data.feature_engineering import FeatureEngineer
from models.combined_predictor import CombinedF5Predictor
from evaluation.accuracy_tracker import check_yesterday_accuracy

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"scheduler_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scheduler")

# ── Configuration ───────────────────────────────────────────────────
RETRAIN_THRESHOLD = 50  # New games before retraining


def get_last_fetch_date() -> str:
    """Determine the last date we have data for."""
    matrix_path = DATA_DIR / "feature_matrix.parquet"
    if not matrix_path.exists():
        return "2021-03-01"  # start of earliest season
    df = pd.read_parquet(matrix_path, columns=["date"])
    return str(df["date"].max())[:10]


def fetch_incremental():
    """Fetch only new games since last run."""
    last_date = get_last_fetch_date()
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Incremental fetch: {last_date} → {today}")

    mlb = MLBStatsFetcher()
    new_games = mlb.get_schedule(last_date, today)
    final = new_games[new_games["status"] == "Final"]
    logger.info(f"Found {len(final)} new completed games")

    if final.empty:
        return 0

    # Fetch stats for new pitchers/teams
    pitcher_ids = set()
    for col in ["away_starter_id", "home_starter_id"]:
        pitcher_ids.update(final[col].dropna().astype(int).unique())

    pitcher_stats = {}
    for pid in pitcher_ids:
        try:
            stats = mlb.get_pitcher_f5_stats(pid)
            if stats:
                pitcher_stats[pid] = stats
        except Exception:
            continue

    team_ids = set()
    for col in ["away_team_id", "home_team_id"]:
        team_ids.update(final[col].dropna().astype(int).unique())

    team_stats = {}
    for tid in team_ids:
        try:
            team_stats[tid] = mlb.get_team_stats(tid)
        except Exception:
            continue

    # Weather for new games
    weather = WeatherFetcher()
    weather_data = weather.get_batch_weather(final)

    # Build features for new games
    fe = FeatureEngineer()
    new_features = fe.build_game_features(
        final, pitcher_stats, {}, team_stats,
        weather_data=weather_data,
    )

    # Append to existing matrix
    matrix_path = DATA_DIR / "feature_matrix.parquet"
    if matrix_path.exists():
        existing = pd.read_parquet(matrix_path)
        # Avoid duplicates
        existing_pks = set(existing["game_pk"].values)
        new_features = new_features[~new_features["game_pk"].isin(existing_pks)]
        combined = pd.concat([existing, new_features], ignore_index=True)
    else:
        combined = new_features

    combined.to_parquet(matrix_path, index=False)
    n_new = len(new_features)
    logger.info(f"Added {n_new} new games. Total: {len(combined)}")
    return n_new


def should_retrain(force: bool = False) -> bool:
    """Check if we should retrain the model."""
    if force:
        return True

    matrix_path = DATA_DIR / "feature_matrix.parquet"
    if not matrix_path.exists():
        return False

    from config.settings import get_latest_model_dir, list_model_versions
    versions = list_model_versions()
    if not versions:
        return True  # No model exists

    # Count games since last train
    latest = versions[0]
    train_date = latest.name[:10]  # YYYY-MM-DD from directory name

    df = pd.read_parquet(matrix_path, columns=["date"])
    new_since = df[df["date"].astype(str) > train_date]
    n_new = len(new_since)

    logger.info(f"Games since last train ({train_date}): {n_new}")
    return n_new >= RETRAIN_THRESHOLD


def run_predict():
    """Generate predictions for today's games."""
    from main import cmd_predict
    import argparse
    args = argparse.Namespace()
    cmd_predict(args)


def main():
    parser = argparse.ArgumentParser(description="F5 Predictor Daily Scheduler")
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"DAILY SCHEDULER RUN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # Step 1: Check yesterday's accuracy
    if not args.skip_accuracy:
        logger.info("Step 1: Checking yesterday's accuracy...")
        mlb = MLBStatsFetcher()
        accuracy = check_yesterday_accuracy(mlb)
        if accuracy:
            logger.info(f"Yesterday: ML {accuracy.get('ml_accuracy', '?')}% accurate")

    # Step 2: Incremental fetch
    logger.info("Step 2: Fetching new game data...")
    n_new = fetch_incremental()

    # Step 3: Conditional retrain
    if not args.skip_retrain:
        if should_retrain(force=args.force_retrain):
            logger.info("Step 3: Retraining model...")
            from main import cmd_train
            import argparse as _ap
            train_args = _ap.Namespace()
            cmd_train(train_args)
        else:
            logger.info("Step 3: Skipping retrain (not enough new data)")

    # Step 4: Predict today's games
    logger.info("Step 4: Generating predictions...")
    run_predict()

    # Step 5: Notifications (Phase 10 — placeholder)
    logger.info("Step 5: Notifications...")
    try:
        from notifications.whatsapp import send_daily_predictions
        send_daily_predictions()
    except ImportError:
        logger.info("  Notification module not installed — skipping")

    logger.info("=" * 60)
    logger.info("SCHEDULER COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
