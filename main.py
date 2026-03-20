"""
F5 Predictor — Main Entry Point
==================================
CLI for the First 5 Innings prediction system.

Usage:
    python main.py fetch     — Fetch historical data
    python main.py train     — Train models on historical data
    python main.py predict   — Predict today's games
    python main.py backtest  — Run walk-forward backtest
    python main.py pipeline  — Full pipeline (fetch → train → predict)
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from config.settings import CURRENT_SEASON, DATA_DIR, MODEL_DIR
from data.fetchers.mlb_stats import MLBStatsFetcher
from data.fetchers.statcast import StatcastFetcher, PYBASEBALL_AVAILABLE
from data.fetchers.odds_api import OddsApiFetcher
from data.feature_engineering import FeatureEngineer
from models.zinb_model import ZINBModel
from models.xgboost_model import XGBoostF5Model
from models.combined_predictor import CombinedF5Predictor
from evaluation.backtester import F5Backtester

# ── Logging Setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("f5_predictor")


def cmd_fetch(args):
    """Fetch and cache historical game data."""
    logger.info("=" * 60)
    logger.info("FETCHING HISTORICAL DATA")
    logger.info("=" * 60)

    mlb = MLBStatsFetcher()
    seasons = list(range(args.start_season, args.end_season + 1))

    # ── Game Schedule Data ─────────────────────────────────────────────
    logger.info(f"Fetching seasons: {seasons}")
    games = mlb.fetch_multi_season(seasons)
    logger.info(f"Total games fetched: {len(games)}")

    # ── Filter to completed games with F5 data ─────────────────────────
    valid = games[games["total_f5_runs"].notna()].copy()
    logger.info(f"Games with F5 data: {len(valid)}")

    # ── Pitcher Stats ──────────────────────────────────────────────────
    pitcher_ids = set()
    for col in ["away_starter_id", "home_starter_id"]:
        pitcher_ids.update(valid[col].dropna().astype(int).unique())

    logger.info(f"Fetching stats for {len(pitcher_ids)} unique pitchers...")
    pitcher_stats = {}
    for i, pid in enumerate(pitcher_ids):
        if i % 50 == 0:
            logger.info(f"  Pitcher {i}/{len(pitcher_ids)}...")
        try:
            stats = mlb.get_pitcher_f5_stats(pid, seasons[-1])
            if stats:
                pitcher_stats[pid] = stats
        except Exception as e:
            logger.debug(f"  Failed for pitcher {pid}: {e}")

    logger.info(f"Pitcher stats collected: {len(pitcher_stats)}")

    # ── Team Stats ─────────────────────────────────────────────────────
    team_ids = set()
    for col in ["away_team_id", "home_team_id"]:
        team_ids.update(valid[col].dropna().astype(int).unique())

    logger.info(f"Fetching stats for {len(team_ids)} teams...")
    team_stats = {}
    for tid in team_ids:
        try:
            team_stats[tid] = mlb.get_team_stats(tid, seasons[-1])
        except Exception as e:
            logger.debug(f"  Failed for team {tid}: {e}")

    # ── Statcast Profiles (optional) ───────────────────────────────────
    statcast_profiles = {}
    if PYBASEBALL_AVAILABLE and args.include_statcast:
        logger.info("Fetching Statcast profiles (this may take a while)...")
        sc = StatcastFetcher()
        season_start = f"{seasons[-1]}-04-01"
        season_end = f"{seasons[-1]}-10-01"
        for i, pid in enumerate(list(pitcher_ids)[:args.max_pitchers]):
            if i % 20 == 0:
                logger.info(f"  Statcast pitcher {i}...")
            try:
                profile = sc.get_pitcher_f5_profile(pid, season_start, season_end)
                if profile:
                    statcast_profiles[pid] = profile
            except Exception:
                continue

    # ── Feature Engineering ────────────────────────────────────────────
    logger.info("Engineering features...")
    fe = FeatureEngineer()
    feature_df = fe.build_game_features(
        valid, pitcher_stats, statcast_profiles, team_stats
    )

    # ── Save ───────────────────────────────────────────────────────────
    out_path = DATA_DIR / "feature_matrix.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info(f"Feature matrix saved: {out_path}")
    logger.info(f"Shape: {feature_df.shape}")
    logger.info(f"Columns: {len(feature_df.columns)}")

    return feature_df


def cmd_train(args):
    """Train models on feature matrix."""
    logger.info("=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────
    data_path = DATA_DIR / "feature_matrix.parquet"
    if not data_path.exists():
        logger.error("Feature matrix not found. Run 'python main.py fetch' first.")
        return

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} games")

    # ── Prepare Features & Targets ─────────────────────────────────────
    fe = FeatureEngineer()
    target_cols = ["away_f5_runs", "home_f5_runs", "total_f5_runs", "home_f5_win"]

    # Drop rows with missing targets
    df = df.dropna(subset=target_cols)
    df["f5_diff"] = df["home_f5_runs"] - df["away_f5_runs"]
    logger.info(f"Valid training rows: {len(df)}")

    # Feature columns (everything that's not a target or ID)
    exclude = {"game_pk", "date", "venue_name", "season"} | set(target_cols) | {"f5_push", "f5_diff"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
    logger.info(f"Feature columns: {len(feature_cols)}")

    # ── Time-Based Split ───────────────────────────────────────────────
    df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]
    logger.info(f"Train: {len(train)} games | Val: {len(val)} games")

    # ── Impute Missing Features ────────────────────────────────────────
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    for col in feature_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_val[col] = X_val[col].fillna(median)

    # ── Train Combined Model ───────────────────────────────────────────
    predictor = CombinedF5Predictor()

    y_train = train[["away_f5_runs", "home_f5_runs", "total_f5_runs", "home_f5_win"]].copy()
    y_val = val[["away_f5_runs", "home_f5_runs", "total_f5_runs", "home_f5_win"]].copy()

    predictor.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ───────────────────────────────────────────────────────
    logger.info("\n── XGBoost Evaluation ──")
    xgb_eval = predictor.xgb.evaluate(
        X_val,
        y_val["home_f5_win"],
        y_val["total_f5_runs"],
        y_val["home_f5_runs"] - y_val["away_f5_runs"],
    )
    for model_name, metrics in xgb_eval.items():
        logger.info(f"  {model_name}: {metrics}")

    logger.info("\n── Feature Importance (Top 15) ──")
    importance = predictor.xgb.get_feature_importance(top_n=15)
    if not importance.empty:
        logger.info(f"\n{importance.to_string()}")

    # ── Save ───────────────────────────────────────────────────────────
    predictor.save("f5_combined")
    logger.info(f"Models saved to {MODEL_DIR}")


def cmd_predict(args):
    """Predict today's games."""
    logger.info("=" * 60)
    logger.info("PREDICTING TODAY'S GAMES")
    logger.info("=" * 60)

    # ── Load Model ─────────────────────────────────────────────────────
    predictor = CombinedF5Predictor()
    try:
        predictor.load("f5_combined")
    except FileNotFoundError:
        logger.error("No trained model found. Run 'python main.py train' first.")
        return

    # ── Fetch Today's Schedule ─────────────────────────────────────────
    mlb = MLBStatsFetcher()
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    schedule = mlb.get_schedule(today, tomorrow)
    upcoming = schedule[schedule["status"] != "Final"]
    logger.info(f"Found {len(upcoming)} upcoming games")

    if upcoming.empty:
        logger.info("No upcoming games found.")
        return

    # ── Fetch Current Odds ─────────────────────────────────────────────
    odds = OddsApiFetcher()
    try:
        current_odds = odds.get_current_odds()
        current_odds = odds.add_implied_probs(current_odds)
        logger.info(f"Fetched odds for {len(current_odds)} markets")
    except Exception as e:
        logger.warning(f"Could not fetch odds: {e}")
        current_odds = pd.DataFrame()

    # ── Build Features & Predict ───────────────────────────────────────
    fe = FeatureEngineer()

    # Fetch pitcher/team stats for today's games
    pitcher_stats = {}
    team_stats = {}
    statcast_profiles = {}

    for _, game in upcoming.iterrows():
        for pid_col in ["away_starter_id", "home_starter_id"]:
            pid = game.get(pid_col)
            if pid and pid not in pitcher_stats:
                try:
                    pitcher_stats[int(pid)] = mlb.get_pitcher_f5_stats(int(pid))
                except Exception:
                    pass

        for tid_col in ["away_team_id", "home_team_id"]:
            tid = game.get(tid_col)
            if tid and tid not in team_stats:
                try:
                    team_stats[int(tid)] = mlb.get_team_stats(int(tid))
                except Exception:
                    pass

    features = fe.build_game_features(
        upcoming, pitcher_stats, statcast_profiles, team_stats
    )

    # ── Generate Predictions ───────────────────────────────────────────
    for idx, (_, game) in enumerate(upcoming.iterrows()):
        game_info = {
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "date": game["date"],
            "venue": game["venue_name"],
            "away_starter": game["away_starter_name"],
            "home_starter": game["home_starter_name"],
        }

        X_game = features.iloc[[idx]]

        # Drop non-numeric columns for prediction
        numeric_cols = [c for c in X_game.columns if X_game[c].dtype in [np.float64, np.int64, float, int]]
        X_game = X_game[numeric_cols]

        try:
            prediction = predictor.predict_game(X_game, game_info)

            # Find edges vs market
            edges = []
            if not current_odds.empty:
                market = {}  # TODO: match game to odds by team names
                edges = predictor.find_all_edges(prediction, market)

            card = predictor.generate_game_card(prediction, edges)
            print(card)
            print()

        except Exception as e:
            logger.warning(f"Prediction failed for {game['away_team']} @ {game['home_team']}: {e}")


def cmd_backtest(args):
    """Run walk-forward backtest."""
    logger.info("=" * 60)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 60)

    data_path = DATA_DIR / "feature_matrix.parquet"
    if not data_path.exists():
        logger.error("Feature matrix not found. Run 'python main.py fetch' first.")
        return

    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["away_f5_runs", "home_f5_runs", "total_f5_runs", "home_f5_win"])
    df["f5_diff"] = df["home_f5_runs"] - df["away_f5_runs"]
    df = df.sort_values("date")

    exclude = {"game_pk", "date", "venue_name", "season",
               "away_f5_runs", "home_f5_runs", "total_f5_runs",
               "home_f5_win", "f5_push", "f5_diff"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    backtester = F5Backtester(
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
    )

    target_mapping = {
        "away_f5_runs": "away_f5_runs",
        "home_f5_runs": "home_f5_runs",
        "total_f5_runs": "total_f5_runs",
        "home_f5_win": "home_f5_win",
    }

    results = backtester.run(
        full_data=df,
        feature_cols=feature_cols,
        target_cols=target_mapping,
        predictor_class=CombinedF5Predictor,
        train_start_idx=args.min_train_size,
    )

    # ── Print Results ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    for key, value in results["summary"].items():
        logger.info(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="F5 Inning Predictor")
    subparsers = parser.add_subparsers(dest="command")

    # ── Fetch ──────────────────────────────────────────────────────────
    p_fetch = subparsers.add_parser("fetch", help="Fetch historical data")
    p_fetch.add_argument("--start-season", type=int, default=2022)
    p_fetch.add_argument("--end-season", type=int, default=2024)
    p_fetch.add_argument("--include-statcast", action="store_true")
    p_fetch.add_argument("--max-pitchers", type=int, default=100,
                         help="Max pitchers for Statcast (rate limited)")

    # ── Train ──────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train models")

    # ── Predict ────────────────────────────────────────────────────────
    p_predict = subparsers.add_parser("predict", help="Predict today's games")

    # ── Backtest ───────────────────────────────────────────────────────
    p_backtest = subparsers.add_parser("backtest", help="Run backtest")
    p_backtest.add_argument("--bankroll", type=float, default=1000.0)
    p_backtest.add_argument("--min-edge", type=float, default=3.0)
    p_backtest.add_argument("--kelly-fraction", type=float, default=0.5)
    p_backtest.add_argument("--min-train-size", type=int, default=500)

    # ── Pipeline ───────────────────────────────────────────────────────
    p_pipeline = subparsers.add_parser("pipeline", help="Full pipeline")
    p_pipeline.add_argument("--start-season", type=int, default=2022)
    p_pipeline.add_argument("--end-season", type=int, default=2024)
    p_pipeline.add_argument("--include-statcast", action="store_true")
    p_pipeline.add_argument("--max-pitchers", type=int, default=100)

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "pipeline":
        cmd_fetch(args)
        cmd_train(args)
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
