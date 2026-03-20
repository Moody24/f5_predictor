"""
Feature Engineering Pipeline
-----------------------------
Transforms raw MLB/Statcast/Odds data into the feature matrix
used by both the XGBoost and ZINB models.

Feature categories:
  1. Starter quality (F5-specific ERA, WHIP, K-BB%, pitch efficiency)
  2. Batted ball / Statcast (xwOBA, barrel%, hard-hit%, whiff%)
  3. Matchup (handedness splits, pitcher vs. team history)
  4. Team offense (wRC+, OPS, ISO, K%, BB%)
  5. Contextual (park factor, weather proxy, rest days, home/away)
  6. Rolling form (last 5/10 game trends)
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

from config.settings import PARK_FACTORS, DEFAULT_PARK_FACTOR, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Builds the F5 feature matrix from multiple data sources."""

    # ── Column Groups ──────────────────────────────────────────────────

    PITCHER_FEATURES = [
        "starter_era_season", "starter_whip_season", "starter_kbb_ratio",
        "starter_avg_ip", "starter_pct_5ip_plus",
        "starter_last5_runs", "starter_last10_runs",
        "starter_avg_pitches", "starter_avg_runs_per_start",
    ]

    STATCAST_FEATURES = [
        "starter_whiff_rate", "starter_csw_pct",
        "starter_avg_ev_against", "starter_barrel_rate",
        "starter_hard_hit_rate", "starter_xwoba_against",
        "starter_avg_fb_velo", "starter_n_pitch_types",
    ]

    HANDEDNESS_FEATURES = [
        "starter_vs_r_xwoba", "starter_vs_l_xwoba",
        "starter_vs_r_whiff", "starter_vs_l_whiff",
        "lineup_pct_lhb",  # % left-handed batters in lineup
    ]

    TEAM_OFFENSE_FEATURES = [
        "team_ops", "team_woba", "team_iso",
        "team_k_pct", "team_bb_pct",
        "team_runs_per_game", "team_wrc_plus",
    ]

    CONTEXTUAL_FEATURES = [
        "park_factor", "is_home", "rest_days",
        "day_night", "dome",
    ]

    ROLLING_FEATURES = [
        "team_f5_runs_last5", "team_f5_runs_last10",
        "team_f5_runs_allowed_last5", "team_f5_runs_allowed_last10",
    ]

    def __init__(self):
        self.feature_cols = (
            self.PITCHER_FEATURES
            + self.STATCAST_FEATURES
            + self.HANDEDNESS_FEATURES
            + self.TEAM_OFFENSE_FEATURES
            + self.CONTEXTUAL_FEATURES
            + self.ROLLING_FEATURES
        )

    # ── Main Pipeline ──────────────────────────────────────────────────

    def build_game_features(
        self,
        games_df: pd.DataFrame,
        pitcher_stats: dict,
        statcast_profiles: dict,
        team_stats: dict,
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for a set of games.

        Args:
            games_df: Game schedule with game_pk, teams, starters
            pitcher_stats: {pitcher_id: stats_dict} from MLB Stats API
            statcast_profiles: {pitcher_id: profile_dict} from Statcast
            team_stats: {team_id: stats_dict} from MLB Stats API

        Returns:
            DataFrame with one row per game, features for both sides.
        """
        features = []

        for _, game in games_df.iterrows():
            row = {"game_pk": game["game_pk"], "date": game["date"]}

            # ── Away Side Features ─────────────────────────────────────
            away_pitcher = self._build_pitcher_features(
                game.get("home_starter_id"),  # home starter faces away batters
                pitcher_stats,
                statcast_profiles,
                prefix="away_facing_",
            )

            away_offense = self._build_offense_features(
                game.get("away_team_id"),
                team_stats,
                prefix="away_",
            )

            # ── Home Side Features ─────────────────────────────────────
            home_pitcher = self._build_pitcher_features(
                game.get("away_starter_id"),  # away starter faces home batters
                pitcher_stats,
                statcast_profiles,
                prefix="home_facing_",
            )

            home_offense = self._build_offense_features(
                game.get("home_team_id"),
                team_stats,
                prefix="home_",
            )

            # ── Context Features ───────────────────────────────────────
            context = self._build_context_features(game)

            # ── Combine ────────────────────────────────────────────────
            row.update(away_pitcher)
            row.update(away_offense)
            row.update(home_pitcher)
            row.update(home_offense)
            row.update(context)

            # ── Targets (if available) ─────────────────────────────────
            if game.get("away_f5_runs") is not None:
                row["away_f5_runs"] = game["away_f5_runs"]
                row["home_f5_runs"] = game["home_f5_runs"]
                row["total_f5_runs"] = game["total_f5_runs"]
                row["home_f5_win"] = int(game["home_f5_runs"] > game["away_f5_runs"])
                row["f5_push"] = int(game["home_f5_runs"] == game["away_f5_runs"])

            features.append(row)

        df = pd.DataFrame(features)
        logger.info(f"Built features for {len(df)} games with {len(df.columns)} columns")
        return df

    # ── Pitcher Feature Builder ────────────────────────────────────────

    def _build_pitcher_features(
        self,
        pitcher_id: Optional[int],
        pitcher_stats: dict,
        statcast_profiles: dict,
        prefix: str,
    ) -> dict:
        """Build features for the pitcher a team is facing."""
        features = {}

        if pitcher_id is None:
            # Unknown starter — use league average defaults
            return self._default_pitcher_features(prefix)

        # MLB Stats API features
        pstats = pitcher_stats.get(pitcher_id, {})
        features[f"{prefix}starter_era_season"] = float(pstats.get("era", 4.50))
        features[f"{prefix}starter_whip_season"] = float(pstats.get("whip", 1.30))
        features[f"{prefix}starter_kbb_ratio"] = pstats.get("k_bb_ratio", 2.5)
        features[f"{prefix}starter_avg_ip"] = pstats.get("avg_ip", 5.0)
        features[f"{prefix}starter_pct_5ip_plus"] = pstats.get("pct_5ip_plus", 60.0)
        features[f"{prefix}starter_avg_pitches"] = pstats.get("avg_pitches", 85.0)
        features[f"{prefix}starter_avg_runs_per_start"] = pstats.get("avg_runs_per_start", 3.5)
        features[f"{prefix}starter_last5_runs"] = pstats.get("last5_avg_runs")
        features[f"{prefix}starter_last10_runs"] = pstats.get("last10_avg_runs")

        # Statcast features
        sprofile = statcast_profiles.get(pitcher_id, {})
        features[f"{prefix}starter_whiff_rate"] = sprofile.get("whiff_rate", 24.0)
        features[f"{prefix}starter_csw_pct"] = sprofile.get("csw_pct", 29.0)
        features[f"{prefix}starter_avg_ev_against"] = sprofile.get("avg_exit_velo_against", 88.5)
        features[f"{prefix}starter_barrel_rate"] = sprofile.get("barrel_rate_against", 7.0)
        features[f"{prefix}starter_hard_hit_rate"] = sprofile.get("hard_hit_rate_against", 35.0)
        features[f"{prefix}starter_xwoba_against"] = sprofile.get("xwOBA_against", 0.320)
        features[f"{prefix}starter_avg_fb_velo"] = sprofile.get("avg_fastball_velo", 93.0)
        features[f"{prefix}starter_n_pitch_types"] = sprofile.get("n_pitch_types", 4)

        # Handedness splits
        features[f"{prefix}starter_vs_r_xwoba"] = sprofile.get("vs_R_xwOBA", 0.320)
        features[f"{prefix}starter_vs_l_xwoba"] = sprofile.get("vs_L_xwOBA", 0.320)
        features[f"{prefix}starter_vs_r_whiff"] = sprofile.get("vs_R_whiff_rate", 24.0)
        features[f"{prefix}starter_vs_l_whiff"] = sprofile.get("vs_L_whiff_rate", 24.0)

        return features

    def _default_pitcher_features(self, prefix: str) -> dict:
        """League-average defaults when starter is unknown."""
        return {
            f"{prefix}starter_era_season": 4.50,
            f"{prefix}starter_whip_season": 1.30,
            f"{prefix}starter_kbb_ratio": 2.5,
            f"{prefix}starter_avg_ip": 5.0,
            f"{prefix}starter_pct_5ip_plus": 60.0,
            f"{prefix}starter_avg_pitches": 85.0,
            f"{prefix}starter_avg_runs_per_start": 3.5,
            f"{prefix}starter_last5_runs": 3.5,
            f"{prefix}starter_last10_runs": 3.5,
            f"{prefix}starter_whiff_rate": 24.0,
            f"{prefix}starter_csw_pct": 29.0,
            f"{prefix}starter_avg_ev_against": 88.5,
            f"{prefix}starter_barrel_rate": 7.0,
            f"{prefix}starter_hard_hit_rate": 35.0,
            f"{prefix}starter_xwoba_against": 0.320,
            f"{prefix}starter_avg_fb_velo": 93.0,
            f"{prefix}starter_n_pitch_types": 4,
            f"{prefix}starter_vs_r_xwoba": 0.320,
            f"{prefix}starter_vs_l_xwoba": 0.320,
            f"{prefix}starter_vs_r_whiff": 24.0,
            f"{prefix}starter_vs_l_whiff": 24.0,
        }

    # ── Team Offense Feature Builder ───────────────────────────────────

    def _build_offense_features(
        self,
        team_id: Optional[int],
        team_stats: dict,
        prefix: str,
    ) -> dict:
        """Build team batting features."""
        features = {}
        tstats = team_stats.get(team_id, {}).get("hitting", {})

        features[f"{prefix}team_ops"] = float(tstats.get("ops", ".720"))
        features[f"{prefix}team_runs_per_game"] = self._calc_rpg(tstats)
        features[f"{prefix}team_k_pct"] = self._calc_rate(
            tstats.get("strikeOuts", 0), tstats.get("plateAppearances", 1)
        )
        features[f"{prefix}team_bb_pct"] = self._calc_rate(
            tstats.get("baseOnBalls", 0), tstats.get("plateAppearances", 1)
        )
        features[f"{prefix}team_iso"] = (
            float(tstats.get("slg", ".400")) - float(tstats.get("avg", ".250"))
        )

        return features

    # ── Context Feature Builder ────────────────────────────────────────

    def _build_context_features(self, game: pd.Series) -> dict:
        """Build park factor, home/away, and scheduling features."""
        venue = game.get("venue_name", "")
        return {
            "park_factor": PARK_FACTORS.get(venue, DEFAULT_PARK_FACTOR),
            "venue_name": venue,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _calc_rpg(stats: dict) -> float:
        """Calculate runs per game from season totals."""
        runs = stats.get("runs", 0)
        games = stats.get("gamesPlayed", 1)
        return round(runs / max(games, 1), 2)

    @staticmethod
    def _calc_rate(numerator: int, denominator: int) -> float:
        """Calculate a rate stat as percentage."""
        return round(numerator / max(denominator, 1) * 100, 1)

    # ── Rolling Features ───────────────────────────────────────────────

    def add_rolling_features(
        self, df: pd.DataFrame, window_sizes: list = None
    ) -> pd.DataFrame:
        """
        Add rolling averages of F5 runs scored/allowed.
        Must be applied per-team with games sorted by date.
        """
        if window_sizes is None:
            window_sizes = ROLLING_WINDOWS

        df = df.sort_values("date").copy()

        for side in ["away", "home"]:
            for window in window_sizes:
                # Runs scored
                col = f"{side}_f5_runs"
                if col in df.columns:
                    df[f"{side}_f5_runs_roll{window}"] = (
                        df[col].rolling(window, min_periods=1).mean()
                    )

        return df

    # ── Feature Matrix for Model ───────────────────────────────────────

    def get_feature_columns(self) -> list[str]:
        """Get list of all feature column names (both sides)."""
        cols = []
        for prefix in ["away_facing_", "home_facing_", "away_", "home_"]:
            for feat_list in [
                self.PITCHER_FEATURES,
                self.STATCAST_FEATURES,
                self.HANDEDNESS_FEATURES,
                self.TEAM_OFFENSE_FEATURES,
            ]:
                cols.extend([f"{prefix}{f}" for f in feat_list])
        cols.extend(["park_factor"])
        return cols

    def prepare_model_input(
        self, df: pd.DataFrame, target: str = None
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Extract feature matrix X and optional target y.
        Handles missing values with median imputation.
        """
        feature_cols = [c for c in df.columns if c not in [
            "game_pk", "date", "venue_name",
            "away_f5_runs", "home_f5_runs", "total_f5_runs",
            "home_f5_win", "f5_push",
        ]]

        X = df[feature_cols].copy()

        # Impute missing with column medians
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64, float, int]:
                X[col] = X[col].fillna(X[col].median())

        y = df[target] if target and target in df.columns else None
        return X, y
