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

import json as _json
import pandas as pd
import numpy as np


class _NumpyEncoder(_json.JSONEncoder):
    """Serialize numpy scalars as native Python types instead of strings."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from config.settings import CURRENT_SEASON, DATA_DIR, MODEL_DIR, MLB_TO_ODDS_TEAM_MAP, F5_RATIO, get_f5_ratio
from data.fetchers.mlb_stats import MLBStatsFetcher
from data.fetchers.statcast import StatcastFetcher, PYBASEBALL_AVAILABLE
from data.fetchers.odds_api import OddsApiFetcher
from data.fetchers.weather import WeatherFetcher
from data.fetchers.umpire import UmpireFetcher
from data.fetchers.lineups import LineupFetcher
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

    # ── Pitcher Stats (prior-season to avoid temporal leakage) ─────────
    # For a game in season S, we fetch stats from season S-1 so models
    # never see full-season data that includes future games.
    pitcher_season_pairs = set()
    for _, game in valid.iterrows():
        gs = int(game.get("season", seasons[-1]))
        for col in ["away_starter_id", "home_starter_id"]:
            pid = game.get(col)
            if pd.notna(pid) and pid:
                pitcher_season_pairs.add((int(pid), gs - 1))

    logger.info(f"Fetching stats for {len(pitcher_season_pairs)} pitcher-season pairs...")
    pitcher_stats = {}
    for i, (pid, prior_season) in enumerate(sorted(pitcher_season_pairs)):
        if i % 50 == 0:
            logger.info(f"  Pitcher {i}/{len(pitcher_season_pairs)}...")
        try:
            stats = mlb.get_pitcher_f5_stats(pid, prior_season)
            if stats:
                pitcher_stats[(pid, prior_season + 1)] = stats  # key by game_season
        except Exception as e:
            logger.debug(f"  Failed for pitcher {pid} prior_season {prior_season}: {e}")

    logger.info(f"Pitcher stats collected: {len(pitcher_stats)}")

    # ── Team Stats (prior-season to avoid temporal leakage) ────────────
    team_season_pairs = set()
    for _, game in valid.iterrows():
        gs = int(game.get("season", seasons[-1]))
        for col in ["away_team_id", "home_team_id"]:
            tid = game.get(col)
            if pd.notna(tid) and tid:
                team_season_pairs.add((int(tid), gs - 1))

    logger.info(f"Fetching stats for {len(team_season_pairs)} team-season pairs...")
    team_stats = {}
    for tid, prior_season in sorted(team_season_pairs):
        try:
            stats = mlb.get_team_stats(tid, prior_season)
            if stats:
                team_stats[(tid, prior_season + 1)] = stats  # key by game_season
        except Exception as e:
            logger.debug(f"  Failed for team {tid} prior_season {prior_season}: {e}")

    # ── Statcast Profiles (optional) ───────────────────────────────────
    statcast_profiles = {}
    if PYBASEBALL_AVAILABLE and args.include_statcast:
        logger.info("Fetching Statcast profiles (this may take a while)...")
        sc = StatcastFetcher()
        season_start = f"{seasons[-1]}-04-01"
        season_end = f"{seasons[-1]}-10-01"
        pitcher_ids = {
            int(pid)
            for _, game in valid.iterrows()
            for col in ["away_starter_id", "home_starter_id"]
            if pd.notna(pid := game.get(col)) and pid
        }
        for i, pid in enumerate(list(pitcher_ids)[:args.max_pitchers]):
            if i % 20 == 0:
                logger.info(f"  Statcast pitcher {i}...")
            try:
                profile = sc.get_pitcher_f5_profile(pid, season_start, season_end)
                if profile:
                    statcast_profiles[pid] = profile
            except Exception as e:
                logger.debug(f"Statcast profile fetch failed for pitcher {pid}: {e}")
                continue
        if statcast_profiles:
            sc.save_profile_cache(statcast_profiles, seasons[-1])
            logger.info(f"Statcast profiles saved: {len(statcast_profiles)}")

    # ── Weather Data ────────────────────────────────────────────────────
    logger.info("Fetching weather data...")
    weather_fetcher = WeatherFetcher()
    weather_data = weather_fetcher.get_batch_weather(valid)

    # ── Umpire Data ───────────────────────────────────────────────────
    logger.info("Fetching umpire data...")
    ump_fetcher = UmpireFetcher()
    ump_assignments = ump_fetcher.build_umpire_assignments(valid)
    ump_tendencies = ump_fetcher.build_umpire_tendencies(ump_assignments, valid)
    umpire_data = {}
    for _, row in ump_assignments.iterrows():
        gpk = row.get("game_pk")
        uid = row.get("umpire_id")
        if uid in ump_tendencies:
            umpire_data[gpk] = ump_tendencies[uid]

    # ── Lineup Data ───────────────────────────────────────────────────
    lineup_features = pd.DataFrame()
    if args.include_lineups:
        logger.info("Fetching lineup data...")
        lineup_fetcher = LineupFetcher()
        lineup_frames = []
        for season in seasons:
            lf = lineup_fetcher.build_batch_lineup_features(valid, season)
            if not lf.empty:
                lineup_frames.append(lf)
        if lineup_frames:
            lineup_features = pd.concat(lineup_frames, ignore_index=True)

    # ── Feature Engineering ────────────────────────────────────────────
    logger.info("Engineering features...")
    fe = FeatureEngineer()
    feature_df = fe.build_game_features(
        valid, pitcher_stats, statcast_profiles, team_stats,
        weather_data=weather_data,
        umpire_data=umpire_data,
        lineup_features=lineup_features,
    )

    # ── Rolling Features (per-team, backward-looking) ─────────────────
    logger.info("Adding rolling features...")
    feature_df = fe.add_rolling_features(feature_df)

    # ── Travel / Fatigue Features ─────────────────────────────────────
    logger.info("Adding travel/fatigue features...")
    feature_df = fe.add_travel_features(feature_df)

    # ── Save (atomic write — avoids corrupt file if container is killed mid-write) ──
    out_path = DATA_DIR / "feature_matrix.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        logger.warning(f"Removing stale tmp file from previous crashed run: {tmp_path}")
        tmp_path.unlink()
    try:
        feature_df.to_parquet(tmp_path, index=False)
        tmp_path.rename(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
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
    # Exclude team/venue IDs — encoding team identity directly causes overfitting
    exclude = {
        "game_pk", "date", "venue_name", "venue_id", "season",
        "away_team_id", "home_team_id",
    } | set(target_cols) | {"f5_push", "f5_diff"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    logger.info(f"Feature columns: {len(feature_cols)}")

    # ── Time-Based Split ───────────────────────────────────────────────
    df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    # Exclude tied F5 games from the moneyline classifier — ties are coded
    # as home_f5_win=0 which would be confused with away wins.
    # ZINB and total/diff models still see all games.
    train_no_push = train[train["f5_diff"] != 0]
    val_no_push = val[val["f5_diff"] != 0]
    push_pct = (df["f5_diff"] == 0).mean() * 100
    logger.info(
        f"Train: {len(train)} games | Val: {len(val)} games "
        f"| Push rate: {push_pct:.1f}% (excluded from ML classifier)"
    )

    # ── Impute Missing Features ────────────────────────────────────────
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    for col in feature_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_val[col] = X_val[col].fillna(median)

    X_train_no_push = train_no_push[feature_cols].copy()
    X_val_no_push = val_no_push[feature_cols].copy()
    for col in feature_cols:
        median = X_train[col].median()  # use same median from full train set
        X_train_no_push[col] = X_train_no_push[col].fillna(median)
        X_val_no_push[col] = X_val_no_push[col].fillna(median)

    # ── Train Combined Model ───────────────────────────────────────────
    predictor = CombinedF5Predictor()

    y_train = train[["away_f5_runs", "home_f5_runs", "total_f5_runs", "home_f5_win"]].copy()
    y_val = val[["away_f5_runs", "home_f5_runs", "total_f5_runs", "home_f5_win"]].copy()

    predictor.fit(
        X_train, y_train, X_val, y_val,
        X_ml_train=X_train_no_push,
        y_ml_train=train_no_push["home_f5_win"],
        X_ml_val=X_val_no_push,
        y_ml_val=val_no_push["home_f5_win"],
    )

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

    # ── Build diagnostics before saving ────────────────────────────────
    diagnostics = {
        "trained_at": datetime.now().isoformat(),
        "n_train": len(train),
        "n_val": len(val),
        "feature_count": len(feature_cols),
        "zinb_weight": predictor.zinb_weight,
        "xgb_weight": predictor.xgb_weight,
        "calibrator_fitted": predictor.calibrator is not None,
    }

    # Brier score on validation set
    brier_probs = []
    brier_actuals = []
    try:
        for i in range(min(len(val), 300)):
            x_row = X_val.iloc[[i]]
            pred_row = predictor.xgb.predict(x_row)
            brier_probs.append(float(pred_row["home_win_prob"][0]))
            brier_actuals.append(float(val["home_f5_win"].iloc[i]))
        if brier_probs:
            brier = float(np.mean(
                [(p - a) ** 2 for p, a in zip(brier_probs, brier_actuals)]
            ))
            diagnostics["brier_score_xgb"] = round(brier, 4)
    except Exception as e:
        logger.debug(f"Brier score failed: {e}")

    # Calibration curve (10 bins) using XGBoost probabilities
    try:
        if brier_probs and brier_actuals:
            bins = np.linspace(0, 1, 11)
            cal_curve = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = [(lo <= p < hi) for p in brier_probs]
                if any(mask):
                    mean_pred = float(np.mean([p for p, m in zip(brier_probs, mask) if m]))
                    mean_actual = float(np.mean([a for a, m in zip(brier_actuals, mask) if m]))
                    cal_curve.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                                      "mean_pred": round(mean_pred, 3), "mean_actual": round(mean_actual, 3)})
            diagnostics["calibration_curve"] = cal_curve
    except Exception as e:
        logger.debug(f"Calibration curve failed: {e}")

    # Top-10 feature importances
    try:
        fi = predictor.xgb.get_feature_importance(top_n=10)
        if not fi.empty:
            diagnostics["top10_feature_importance"] = fi.to_dict(orient="records")
    except Exception as e:
        logger.debug(f"Feature importance failed: {e}")

    # ── Save (versioned) ────────────────────────────────────────────────
    version_dir = predictor.save("f5_combined", diagnostics=diagnostics)
    logger.info(f"Models saved to {version_dir}")


def _match_odds_to_game(
    mlb_home: str, mlb_away: str, odds_df: pd.DataFrame,
    f5_ratio: float = F5_RATIO,
) -> dict:
    """
    Match a game's odds from the Odds API DataFrame by team name.
    Returns a market dict with consensus (mean) implied probabilities
    across bookmakers, suitable for find_all_edges().
    """
    odds_home = MLB_TO_ODDS_TEAM_MAP.get(mlb_home, mlb_home)
    odds_away = MLB_TO_ODDS_TEAM_MAP.get(mlb_away, mlb_away)

    game_odds = odds_df[
        (odds_df["home_team"] == odds_home) & (odds_df["away_team"] == odds_away)
    ]
    if game_odds.empty:
        return {}

    market = {}

    # Moneyline consensus
    ml_rows = game_odds[game_odds["market"] == "moneyline"]
    if not ml_rows.empty and "home_implied" in ml_rows.columns:
        market["home_ml_implied"] = ml_rows["home_implied"].mean()
        market["away_ml_implied"] = ml_rows["away_implied"].mean()
        if "home_ml" in ml_rows.columns:
            market["home_ml_american"] = ml_rows["home_ml"].median()
            market["away_ml_american"] = ml_rows["away_ml"].median()

    # Totals consensus (apply F5 ratio to convert full-game to F5)
    total_rows = game_odds[game_odds["market"] == "total"]
    if not total_rows.empty and "total_line" in total_rows.columns:
        full_game_line = total_rows["total_line"].median()
        market["total_line"] = round(full_game_line * f5_ratio * 2) / 2  # round to nearest 0.5
        if "over_implied" in total_rows.columns:
            market["over_implied"] = total_rows["over_implied"].mean()
            market["under_implied"] = total_rows["under_implied"].mean()
        if "over_price" in total_rows.columns:
            market["over_american"] = total_rows["over_price"].median()
            market["under_american"] = total_rows["under_price"].median()

    # Spread consensus (apply F5 ratio)
    spread_rows = game_odds[game_odds["market"] == "spread"]
    if not spread_rows.empty and "home_spread" in spread_rows.columns:
        full_spread = spread_rows["home_spread"].median()
        market["home_spread"] = round(full_spread * f5_ratio * 2) / 2
        if "home_spread_implied" in spread_rows.columns:
            market["home_spread_implied"] = spread_rows["home_spread_implied"].mean()
        if "home_spread_price" in spread_rows.columns:
            market["home_spread_american"] = spread_rows["home_spread_price"].median()

    return market


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
    in_two_days = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

    schedule = mlb.get_schedule(today, in_two_days)
    if schedule.empty or "status" not in schedule.columns:
        logger.info("No regular season games found. Regular season may not have started yet.")
        return

    # Filter to non-Final games, then take only the earliest game date found.
    # This ensures we always predict the next upcoming slate — whether that's
    # today (morning run) or tomorrow (late-evening run after today's games finish).
    upcoming_all = schedule[schedule["status"] != "Final"]
    if upcoming_all.empty:
        logger.info("No upcoming games in the next 2 days.")
        return
    next_game_date = str(upcoming_all["date"].min())[:10]
    upcoming = upcoming_all[upcoming_all["date"].astype(str).str[:10] == next_game_date]
    logger.info(f"Predicting {len(upcoming)} games for {next_game_date}")

    if upcoming.empty:
        logger.info("No upcoming games today.")
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

    # Load Statcast profiles from training-time cache (avoids live API calls)
    statcast_profiles = {}
    if PYBASEBALL_AVAILABLE:
        try:
            sc = StatcastFetcher()
            statcast_profiles = sc.load_profile_cache(CURRENT_SEASON)
            if not statcast_profiles:
                statcast_profiles = sc.load_profile_cache(CURRENT_SEASON - 1)
        except Exception as e:
            logger.warning(f"Could not load Statcast profile cache: {e}")

    # Fetch pitcher/team stats for today's games
    # For pitchers with < 10 qualified starts this season, blend current stats
    # with prior-season stats so rookies/returning pitchers get a fair baseline.
    pitcher_stats = {}
    team_stats = {}

    for _, game in upcoming.iterrows():
        for pid_col in ["away_starter_id", "home_starter_id"]:
            pid = game.get(pid_col)
            if not pd.notna(pid) or not pid:
                continue
            pid = int(pid)
            if pid in pitcher_stats:
                continue
            try:
                current = mlb.get_pitcher_f5_stats(pid, CURRENT_SEASON)
                qualified_starts = (current or {}).get("qualified_starts", 0)
                if qualified_starts < 10:
                    prior = mlb.get_pitcher_f5_stats(pid, CURRENT_SEASON - 1)
                    if current and prior and qualified_starts > 0:
                        weight = qualified_starts / 10.0
                        blend_keys = [
                            "avg_ip", "avg_runs_per_start", "avg_er_per_start",
                            "avg_hits_per_start", "avg_walks_per_start",
                            "avg_k_per_start", "avg_pitches", "k_bb_ratio",
                            "last5_avg_runs", "last10_avg_runs",
                        ]
                        blended = dict(current)
                        for key in blend_keys:
                            c_val = current.get(key)
                            p_val = prior.get(key)
                            if c_val is not None and p_val is not None:
                                blended[key] = round(weight * float(c_val) + (1 - weight) * float(p_val), 3)
                        pitcher_stats[pid] = blended
                        logger.debug(f"Pitcher {pid}: blended {qualified_starts} current + prior ({weight:.1f}/{1-weight:.1f})")
                    elif prior:
                        pitcher_stats[pid] = prior
                    elif current:
                        pitcher_stats[pid] = current
                else:
                    pitcher_stats[pid] = current
            except Exception as e:
                logger.warning(f"Failed to fetch pitcher stats for ID {pid}: {e}")

        for tid_col in ["away_team_id", "home_team_id"]:
            tid = game.get(tid_col)
            if tid and tid not in team_stats:
                try:
                    team_stats[int(tid)] = mlb.get_team_stats(int(tid))
                except Exception as e:
                    logger.warning(f"Failed to fetch team stats for ID {tid}: {e}")

    # ── Lineup + Recent Batter Form ────────────────────────────────────
    # Fetch today's confirmed lineups and each starter's 14-day rolling wOBA.
    # Cached per player per day so the second daily run is instant.
    # Fails gracefully: falls back to league-average defaults if API is down.
    lineup_features_today = pd.DataFrame()
    try:
        from data.fetchers.lineups import LineupFetcher
        lf = LineupFetcher()
        batter_stats_df = lf.get_batter_season_stats(CURRENT_SEASON)
        if batter_stats_df.empty:
            batter_stats_df = lf.get_batter_season_stats(CURRENT_SEASON - 1)

        lineup_rows = []
        for _, game in upcoming.iterrows():
            gpk = int(game["game_pk"])
            # For today's pre-game, try probable lineup from boxscore (won't exist yet),
            # fall back to empty — build_lineup_features uses league-average defaults.
            lineup = lf.get_game_lineup(gpk)

            # Collect all batter IDs in this game to batch recent-form fetches
            all_batters = lineup.get("away", []) + lineup.get("home", [])
            recent_form = {}
            for batter in all_batters:
                pid = batter.get("player_id")
                if pid:
                    recent_form[pid] = lf.get_batter_recent_form(pid, CURRENT_SEASON)

            feats = lf.build_lineup_features(
                gpk, lineup, batter_stats_df,
                recent_form=recent_form,
            )
            feats["game_pk"] = gpk
            lineup_rows.append(feats)

        if lineup_rows:
            lineup_features_today = pd.DataFrame(lineup_rows)
            logger.info(
                f"Lineup features built for {len(lineup_rows)} games "
                f"(recent form: {sum(1 for r in lineup_rows if r.get('away_lineup_recent_woba', 0.320) != 0.320)} games with live data)"
            )
    except Exception as e:
        logger.warning(f"Lineup fetch failed (non-fatal): {e} — using league-average defaults")

    today_base = fe.build_game_features(
        upcoming, pitcher_stats, statcast_profiles, team_stats,
        lineup_features=lineup_features_today if not lineup_features_today.empty else None,
    )

    # Rolling and travel features require historical context.
    # Load the feature matrix, drop its derived columns, append today's base
    # features, recompute rolling+travel on the combined set, then extract
    # only today's rows. shift(1) in add_rolling_features ensures today's
    # (not-yet-played) game does not contribute to its own rolling stats.
    hist_path = DATA_DIR / "feature_matrix.parquet"
    if hist_path.exists():
        hist_df = pd.read_parquet(hist_path)
        derived = fe.derived_column_names()
        hist_df = hist_df.drop(columns=[c for c in derived if c in hist_df.columns])
        combined = pd.concat([hist_df, today_base], ignore_index=True)
        combined = fe.add_rolling_features(combined)
        combined = fe.add_travel_features(combined)
        today_pks = set(today_base["game_pk"].values)
        features = combined[combined["game_pk"].isin(today_pks)].set_index("game_pk")
    else:
        features = today_base.set_index("game_pk")

    # ── Generate Predictions ───────────────────────────────────────────
    all_predictions = []

    for _, game in upcoming.iterrows():
        game_pk = game["game_pk"]

        if game_pk not in features.index:
            logger.warning(
                f"No features for game_pk {game_pk} "
                f"({game['away_team']} @ {game['home_team']}) — skipping"
            )
            continue

        game_info = {
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "date": game["date"],
            "venue": game["venue_name"],
            "away_starter": game.get("away_starter_name", "TBD"),
            "home_starter": game.get("home_starter_name", "TBD"),
            # ISO timestamp for closing-line CLV lookup (paid Odds API tier)
            "commence_time": str(game.get("game_datetime", game["date"])),
        }

        X_game = features.loc[[game_pk]]

        # Drop non-numeric and ID columns — same exclusions as training
        _pred_exclude = {"game_pk", "date", "venue_name", "venue_id", "season",
                         "away_team_id", "home_team_id",
                         "away_f5_runs", "home_f5_runs", "total_f5_runs",
                         "home_f5_win", "f5_push", "f5_diff"}
        numeric_cols = [
            c for c in X_game.columns
            if c not in _pred_exclude and pd.api.types.is_numeric_dtype(X_game[c])
        ]
        X_game = X_game[numeric_cols]

        try:
            prediction = predictor.predict_game(X_game, game_info)

            # Find edges vs market — use per-game dynamic F5 ratio
            edges = []
            market = {}
            if not current_odds.empty:
                starter_runs = [
                    float(pitcher_stats[int(pid)]["avg_runs_per_start"])
                    for col in ["away_starter_id", "home_starter_id"]
                    if (pid := game.get(col))
                    and int(pid) in pitcher_stats
                    and pitcher_stats[int(pid)].get("avg_runs_per_start") is not None
                ]
                game_f5_ratio = get_f5_ratio(
                    sum(starter_runs) / len(starter_runs) if starter_runs else None
                )
                market = _match_odds_to_game(
                    game["home_team"], game["away_team"], current_odds, f5_ratio=game_f5_ratio
                )
                if market:
                    edges = predictor.find_all_edges(prediction, market)

            card = predictor.generate_game_card(prediction, edges)
            print(card)
            print()

            # Collect for JSON output
            all_predictions.append({
                "game_info": game_info,
                "moneyline": prediction["moneyline"],
                "total": {
                    k: v for k, v in prediction["total"].items()
                    if k != "over_under_probs"  # not JSON serializable as-is
                },
                "run_line": prediction["run_line"],
                "edges": edges,
                # Opening market implied probs — used for CLV computation next day
                "opening_market": {
                    "home_ml_implied": market.get("home_ml_implied"),
                    "away_ml_implied": market.get("away_ml_implied"),
                    "total_line": market.get("total_line"),
                    "over_implied": market.get("over_implied"),
                } if market else {},
            })

        except Exception as e:
            logger.warning(f"Prediction failed for {game['away_team']} @ {game['home_team']}: {e}")

    # ── Save Predictions as JSON ─────────────────────────────────────
    if all_predictions:
        from config.settings import PREDICTIONS_DIR, get_latest_model_dir
        import json
        # Use the actual game date (not run date) so accuracy tracker can match
        # predictions to results even when the pipeline runs late in the evening
        # and tomorrow's games are fetched instead of today's.
        pred_date = str(upcoming["date"].iloc[0])[:10] if not upcoming.empty else today
        output = {
            "date": pred_date,
            "model_version": str(get_latest_model_dir().name),
            "n_games": len(all_predictions),
            "games": all_predictions,
        }
        json_path = PREDICTIONS_DIR / f"{pred_date}.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2, cls=_NumpyEncoder)
        logger.info(f"Predictions saved to {json_path}")


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
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

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
    p_fetch.add_argument("--start-season", type=int, default=2021)
    p_fetch.add_argument("--end-season", type=int, default=2025)
    p_fetch.add_argument("--include-statcast", action="store_true")
    p_fetch.add_argument("--include-lineups", action="store_true",
                         help="Fetch per-game lineups (slow, ~175k lookups)")
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
    p_pipeline.add_argument("--start-season", type=int, default=2021)
    p_pipeline.add_argument("--end-season", type=int, default=2025)
    p_pipeline.add_argument("--include-statcast", action="store_true")
    p_pipeline.add_argument("--include-lineups", action="store_true")
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
