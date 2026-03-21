"""
Accuracy Tracker
-----------------
Compares previous day's predictions to actual outcomes.
Tracks running metrics: hit rate, ROI, calibration drift.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

from config.settings import PREDICTIONS_DIR, DATA_DIR

logger = logging.getLogger(__name__)

ACCURACY_DIR = DATA_DIR / "accuracy"
ACCURACY_DIR.mkdir(parents=True, exist_ok=True)


def check_yesterday_accuracy(mlb_fetcher) -> dict:
    """
    Compare yesterday's predictions to actual outcomes.

    Args:
        mlb_fetcher: MLBStatsFetcher instance to get actual results

    Returns:
        Dict with accuracy metrics, or empty dict if no predictions found
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    pred_path = PREDICTIONS_DIR / f"{yesterday}.json"

    if not pred_path.exists():
        logger.info(f"No predictions found for {yesterday}")
        return {}

    with open(pred_path) as f:
        predictions = json.load(f)

    # Fetch actual results
    actual_games = mlb_fetcher.get_schedule(yesterday, yesterday)
    final_games = actual_games[actual_games["status"] == "Final"]

    if final_games.empty:
        logger.info(f"No final games found for {yesterday}")
        return {}

    results = []
    for pred_game in predictions.get("games", []):
        info = pred_game["game_info"]
        home = info["home_team"]
        away = info["away_team"]

        # Match to actual game
        match = final_games[
            (final_games["home_team"] == home) & (final_games["away_team"] == away)
        ]
        if match.empty:
            continue

        actual = match.iloc[0]
        actual_home_runs = actual.get("home_f5_runs")
        actual_away_runs = actual.get("away_f5_runs")
        if actual_home_runs is None or actual_away_runs is None:
            continue

        actual_total = actual_home_runs + actual_away_runs
        actual_home_win = 1 if actual_home_runs > actual_away_runs else 0

        ml = pred_game["moneyline"]
        pred_home_win = 1 if ml["home_prob"] > 0.5 else 0

        results.append({
            "date": yesterday,
            "home_team": home,
            "away_team": away,
            "pred_home_prob": ml["home_prob"],
            "pred_home_win": pred_home_win,
            "actual_home_win": actual_home_win,
            "ml_correct": pred_home_win == actual_home_win,
            "pred_total": pred_game.get("total", {}).get("predicted", 0),
            "actual_total": actual_total,
            "total_error": abs(pred_game.get("total", {}).get("predicted", 0) - actual_total),
            "edges": pred_game.get("edges", []),
        })

    if not results:
        return {}

    df = pd.DataFrame(results)

    summary = {
        "date": yesterday,
        "games_tracked": len(df),
        "ml_accuracy": round(df["ml_correct"].mean() * 100, 1),
        "avg_total_error": round(df["total_error"].mean(), 2),
        "edges_flagged": sum(len(r["edges"]) for r in results),
    }

    # Track edge bet results
    edge_results = []
    for r in results:
        for edge in r.get("edges", []):
            market = edge.get("market", "")
            if "Moneyline" in market:
                side = edge["side"]
                won = (side == "Home" and r["actual_home_win"] == 1) or \
                      (side == "Away" and r["actual_home_win"] == 0)
                edge_results.append({"won": won, "edge_pct": edge["edge_pct"]})

    if edge_results:
        edge_df = pd.DataFrame(edge_results)
        summary["edge_bet_accuracy"] = round(edge_df["won"].mean() * 100, 1)
        summary["avg_edge_on_bets"] = round(edge_df["edge_pct"].mean(), 1)

    # Append to running log
    log_path = ACCURACY_DIR / "daily_accuracy.json"
    running_log = []
    if log_path.exists():
        with open(log_path) as f:
            running_log = json.load(f)

    running_log.append(summary)
    with open(log_path, "w") as f:
        json.dump(running_log, f, indent=2)

    logger.info(f"Accuracy for {yesterday}: ML {summary['ml_accuracy']}%, "
                f"Avg total error: {summary['avg_total_error']}")

    return summary
