"""
Accuracy Tracker
-----------------
Compares previous day's predictions to actual outcomes.
Tracks running metrics: hit rate, ROI, calibration drift, and Closing Line Value (CLV).

CLV measures whether our edge predictions aligned with where sharp money moved the line.
Positive CLV = we had the right side before the market agreed. Requires paid Odds API tier.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

from config.settings import PREDICTIONS_DIR, DATA_DIR, MLB_TO_ODDS_TEAM_MAP

logger = logging.getLogger(__name__)

ACCURACY_DIR = DATA_DIR / "accuracy"
ACCURACY_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_closing_odds(odds_fetcher, game_time_iso: str, home_team: str, away_team: str) -> dict:
    """
    Fetch closing-line implied probabilities for a game using the historical odds endpoint.
    Requires paid Odds API tier. Returns {} if unavailable.

    Args:
        game_time_iso: ISO 8601 string of game's scheduled start time
        home_team / away_team: MLB team names (will map to Odds API names)
    """
    try:
        df = odds_fetcher.get_historical_odds(game_time_iso)
        if df.empty:
            return {}

        odds_home = MLB_TO_ODDS_TEAM_MAP.get(home_team, home_team)
        odds_away = MLB_TO_ODDS_TEAM_MAP.get(away_team, away_team)

        game_rows = df[
            (df["home_team"] == odds_home) & (df["away_team"] == odds_away)
            & (df["market"] == "moneyline")
        ]
        if game_rows.empty:
            return {}

        from data.fetchers.odds_api import OddsApiFetcher
        result = {}
        if "home_ml" in game_rows.columns:
            home_ml = game_rows["home_ml"].median()
            away_ml = game_rows["away_ml"].median()
            result["closing_home_implied"] = OddsApiFetcher.american_to_implied_prob(home_ml)
            result["closing_away_implied"] = OddsApiFetcher.american_to_implied_prob(away_ml)
        return result
    except Exception as e:
        logger.debug(f"Could not fetch closing odds for {away_team} @ {home_team}: {e}")
        return {}


def check_yesterday_accuracy(mlb_fetcher, odds_fetcher=None) -> dict:
    """
    Compare yesterday's predictions to actual outcomes and compute CLV.

    Args:
        mlb_fetcher: MLBStatsFetcher instance to get actual results
        odds_fetcher: OddsApiFetcher instance for closing-line CLV (paid tier).
                      Pass None to skip CLV computation.

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

        # ── CLV: compare our model prob to closing implied prob ─────────
        home_clv = None
        closing_home_implied = None
        if odds_fetcher is not None:
            game_time = pred_game.get("game_info", {}).get("commence_time")
            if not game_time:
                logger.warning(
                    f"CLV skipped for {away} @ {home}: 'commence_time' missing from game_info"
                )
            else:
                closing = _fetch_closing_odds(odds_fetcher, game_time, home, away)
                if closing:
                    closing_home_implied = closing.get("closing_home_implied")
                    opening_home_implied = pred_game.get("opening_market", {}).get("home_ml_implied")
                    if closing_home_implied and opening_home_implied:
                        # Raw CLV: positive = market moved toward home side
                        raw_clv = (closing_home_implied - opening_home_implied) * 100
                        # Flip sign when our pick was away — CLV should be
                        # directional relative to the side our model bet, not
                        # always relative to home.
                        model_picked_home = ml["home_prob"] > 0.5
                        home_clv = round(raw_clv if model_picked_home else -raw_clv, 2)

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
            "closing_home_implied": closing_home_implied,
            "home_clv": home_clv,
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

    # CLV summary (only if closing odds were fetched)
    clv_vals = df["home_clv"].dropna()
    if not clv_vals.empty:
        summary["avg_clv"] = round(clv_vals.mean(), 2)
        summary["positive_clv_pct"] = round((clv_vals > 0).mean() * 100, 1)
        summary["clv_games_tracked"] = len(clv_vals)
        logger.info(
            f"CLV: avg={summary['avg_clv']:+.2f}%, "
            f"positive={summary['positive_clv_pct']}% of {len(clv_vals)} games"
        )

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
