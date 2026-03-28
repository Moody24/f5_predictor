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

    # Recompute rolling and travel features on the full combined matrix so
    # the newly appended rows get correct rolling averages and rest-day counts.
    derived = fe.derived_column_names()
    combined = combined.drop(columns=[c for c in derived if c in combined.columns])
    combined = fe.add_rolling_features(combined)
    combined = fe.add_travel_features(combined)

    tmp_path = matrix_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.rename(matrix_path)
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
    new_since = df[pd.to_datetime(df["date"]) > pd.to_datetime(train_date)]
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

    run_start = datetime.now()
    metrics = {}  # collects observability data for the run summary

    logger.info("=" * 60)
    logger.info(f"DAILY SCHEDULER RUN — {run_start.strftime('%Y-%m-%d %H:%M')} UTC")
    logger.info("=" * 60)

    # Step 1: Check yesterday's accuracy (+ CLV via paid Odds API historical endpoint)
    if not args.skip_accuracy:
        logger.info("Step 1: Checking yesterday's accuracy...")
        t0 = datetime.now()
        mlb = MLBStatsFetcher()
        from data.fetchers.odds_api import OddsApiFetcher
        odds_fetcher = OddsApiFetcher()
        accuracy = check_yesterday_accuracy(mlb, odds_fetcher=odds_fetcher if odds_fetcher.api_key else None)
        if accuracy:
            metrics["yesterday_ml_accuracy"] = accuracy.get("ml_accuracy")
            metrics["yesterday_total_error"] = accuracy.get("avg_total_error")
            metrics["yesterday_edge_accuracy"] = accuracy.get("edge_bet_accuracy")
            clv_line = ""
            if "avg_clv" in accuracy:
                metrics["yesterday_avg_clv"] = accuracy.get("avg_clv")
                clv_line = f" | Avg CLV: {accuracy['avg_clv']:+.2f}% ({accuracy.get('positive_clv_pct', '?')}% positive)"
            logger.info(
                f"  ML accuracy: {accuracy.get('ml_accuracy', '?')}% | "
                f"Avg total error: {accuracy.get('avg_total_error', '?')} runs | "
                f"Edge accuracy: {accuracy.get('edge_bet_accuracy', 'N/A')}%"
                + clv_line
            )
        else:
            logger.info("  No predictions from yesterday to evaluate")
        logger.info(f"  Done in {(datetime.now() - t0).seconds}s")

    # Step 2: Incremental fetch
    logger.info("Step 2: Fetching new game data...")
    t0 = datetime.now()
    n_new = fetch_incremental()
    metrics["new_games_fetched"] = n_new
    logger.info(f"  {n_new} new games fetched in {(datetime.now() - t0).seconds}s")

    # Step 3: Conditional retrain
    retrained = False
    if not args.skip_retrain:
        if should_retrain(force=args.force_retrain):
            logger.info("Step 3: Retraining model...")
            t0 = datetime.now()
            from main import cmd_train
            import argparse as _ap
            cmd_train(_ap.Namespace())
            retrained = True
            metrics["retrain_duration_s"] = (datetime.now() - t0).seconds
            logger.info(f"  Retrain complete in {metrics['retrain_duration_s']}s")
        else:
            logger.info("Step 3: Skipping retrain (not enough new data)")
    metrics["retrained"] = retrained

    # Step 4: Predict today's games
    logger.info("Step 4: Generating predictions...")
    t0 = datetime.now()
    run_predict()
    metrics["predict_duration_s"] = (datetime.now() - t0).seconds

    # Count today's predictions and edges
    try:
        from utils import predictions_path
        import json
        pred_path = predictions_path()
        if pred_path.exists():
            with open(pred_path) as f:
                preds = json.load(f)
            metrics["games_predicted"] = preds.get("n_games", 0)
            metrics["edges_found"] = sum(len(g.get("edges", [])) for g in preds.get("games", []))
            logger.info(
                f"  {metrics['games_predicted']} games predicted, "
                f"{metrics['edges_found']} edges found in {metrics['predict_duration_s']}s"
            )
    except Exception:
        pass

    # Step 5: Notifications
    logger.info("Step 5: Notifications...")
    try:
        from notifications.telegram import send_daily_predictions
        send_daily_predictions()
        metrics["notification_sent"] = True
        logger.info("  Telegram notification sent")
    except ImportError:
        logger.info("  Notification module not installed — skipping")
        metrics["notification_sent"] = False
    except Exception as e:
        logger.warning(f"  Notification failed (non-fatal): {e}")
        metrics["notification_sent"] = False

    # Run summary
    total_s = (datetime.now() - run_start).seconds
    metrics["total_duration_s"] = total_s
    logger.info("=" * 60)
    logger.info(f"SCHEDULER COMPLETE in {total_s}s")
    logger.info(f"  Games: {metrics.get('games_predicted', '?')} | "
                f"Edges: {metrics.get('edges_found', '?')} | "
                f"Retrained: {retrained} | "
                f"Notification: {metrics.get('notification_sent', False)}")
    if metrics.get("yesterday_ml_accuracy"):
        logger.info(f"  Yesterday accuracy: {metrics['yesterday_ml_accuracy']}% ML")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
