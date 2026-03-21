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
        "park_factor",
    ]

    WEATHER_FEATURES = [
        "temperature_f", "wind_speed_mph", "wind_direction_deg",
        "humidity_pct", "is_dome",
    ]

    UMPIRE_FEATURES = [
        "ump_rpg_factor", "ump_experience",
    ]

    TRAVEL_FEATURES = [
        "travel_distance_miles", "games_in_last_7d",
        "rest_days", "is_back_to_back",
    ]

    BULLPEN_FEATURES = [
        "team_bullpen_era", "bullpen_innings_last_3d", "starter_hook_rate",
    ]

    LINEUP_FEATURES = [
        "lineup_avg_woba", "lineup_avg_ops", "lineup_total_iso", "lineup_platoon_pct",
    ]

    ROLLING_FEATURES = [
        "team_f5_runs_last5", "team_f5_runs_last10", "team_f5_runs_last20",
        "team_f5_allowed_last5", "team_f5_allowed_last10", "team_f5_allowed_last20",
    ]

    def __init__(self):
        self.feature_cols = (
            self.PITCHER_FEATURES
            + self.STATCAST_FEATURES
            + self.HANDEDNESS_FEATURES
            + self.TEAM_OFFENSE_FEATURES
            + self.CONTEXTUAL_FEATURES
            + self.WEATHER_FEATURES
            + self.UMPIRE_FEATURES
            + self.TRAVEL_FEATURES
            + self.BULLPEN_FEATURES
            + self.LINEUP_FEATURES
            + self.ROLLING_FEATURES
        )

    # ── Main Pipeline ──────────────────────────────────────────────────

    def build_game_features(
        self,
        games_df: pd.DataFrame,
        pitcher_stats: dict,
        statcast_profiles: dict,
        team_stats: dict,
        weather_data: pd.DataFrame = None,
        umpire_data: dict = None,
        bullpen_stats: dict = None,
        lineup_features: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for a set of games.

        Args:
            games_df: Game schedule with game_pk, teams, starters
            pitcher_stats: {pitcher_id: stats_dict} from MLB Stats API
            statcast_profiles: {pitcher_id: profile_dict} from Statcast
            team_stats: {team_id: stats_dict} from MLB Stats API
            weather_data: DataFrame with game_pk + weather columns (optional)
            umpire_data: {game_pk: {ump_rpg_factor, ump_experience}} (optional)
            bullpen_stats: {team_id: {bullpen_era, ...}} (optional)
            lineup_features: DataFrame with game_pk + lineup columns (optional)

        Returns:
            DataFrame with one row per game, features for both sides.
        """
        features = []

        # Index optional DataFrames by game_pk for fast lookup
        weather_lookup = {}
        if weather_data is not None and not weather_data.empty:
            weather_lookup = dict(zip(weather_data["game_pk"], weather_data.to_dict("records")))

        lineup_lookup = {}
        if lineup_features is not None and not lineup_features.empty:
            lineup_lookup = dict(zip(lineup_features["game_pk"], lineup_features.to_dict("records")))

        for _, game in games_df.iterrows():
            row = {
                "game_pk": game["game_pk"],
                "date": game["date"],
                "away_team_id": game.get("away_team_id"),
                "home_team_id": game.get("home_team_id"),
            }

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

            # ── Weather Features ──────────────────────────────────────
            gpk = game["game_pk"]
            weather = weather_lookup.get(gpk, {})
            weather_feats = {
                "temperature_f": weather.get("temperature_f", 72.0),
                "wind_speed_mph": weather.get("wind_speed_mph", 5.0),
                "wind_direction_deg": weather.get("wind_direction_deg", 0.0),
                "humidity_pct": weather.get("humidity_pct", 50.0),
                "is_dome": weather.get("is_dome", 0),
            }

            # ── Umpire Features ───────────────────────────────────────
            ump = (umpire_data or {}).get(gpk, {})
            ump_feats = {
                "ump_rpg_factor": ump.get("ump_rpg_factor", 1.0),
                "ump_experience": ump.get("ump_experience", 100),
            }

            # ── Bullpen Features (per side) ───────────────────────────
            for side, tid_col in [("away", "away_team_id"), ("home", "home_team_id")]:
                tid = game.get(tid_col)
                bp = (bullpen_stats or {}).get(tid, {})
                row[f"{side}_team_bullpen_era"] = bp.get("bullpen_era", 4.50)
                row[f"{side}_bullpen_innings_last_3d"] = bp.get("bullpen_innings_last_3d", 6.0)
                row[f"{side}_starter_hook_rate"] = bp.get("starter_hook_rate", 20.0)

            # ── Lineup Features ───────────────────────────────────────
            lineup = lineup_lookup.get(gpk, {})
            for side in ["away", "home"]:
                row[f"{side}_lineup_avg_woba"] = lineup.get(f"{side}_lineup_avg_woba", 0.320)
                row[f"{side}_lineup_avg_ops"] = lineup.get(f"{side}_lineup_avg_ops", 0.720)
                row[f"{side}_lineup_total_iso"] = lineup.get(f"{side}_lineup_total_iso", 1.350)
                row[f"{side}_lineup_platoon_pct"] = lineup.get(f"{side}_lineup_platoon_pct", 50.0)

            # ── Combine ────────────────────────────────────────────────
            row.update(away_pitcher)
            row.update(away_offense)
            row.update(home_pitcher)
            row.update(home_offense)
            row.update(context)
            row.update(weather_feats)
            row.update(ump_feats)

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

    # ── Travel / Fatigue Features ────────────────────────────────────

    def add_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add travel distance, rest days, and schedule density per team.
        Uses numpy arrays for output to avoid df.at[] overhead and memory blowup.
        """
        from data.fetchers.weather import VENUE_COORDS
        from collections import defaultdict, deque

        df = df.sort_values("date").reset_index(drop=True).copy()
        dates = pd.to_datetime(df["date"])
        n = len(df)

        for side, tid_col in [("away", "away_team_id"), ("home", "home_team_id")]:
            rest_days = np.full(n, 3.0)
            is_back_to_back = np.zeros(n, dtype=int)
            travel_distance = np.zeros(n)
            games_7d = np.full(n, 3.0)

            # team_id -> (last_date, last_venue)
            team_last: dict = {}
            # team_id -> deque of recent dates pruned to 7-day window (O(1) count)
            team_recent: dict = defaultdict(deque)

            for i in range(n):
                tid = df.at[i, tid_col]
                if pd.isna(tid):
                    continue
                tid = int(tid)
                gdate = dates.iloc[i]
                venue = df.at[i, "venue_name"]

                if tid in team_last:
                    last_date, last_venue = team_last[tid]
                    rest = (gdate - last_date).days
                    rest_days[i] = rest
                    is_back_to_back[i] = 1 if rest <= 1 else 0
                    p = VENUE_COORDS.get(last_venue)
                    c = VENUE_COORDS.get(venue)
                    if p and c:
                        travel_distance[i] = round(self._haversine(p[0], p[1], c[0], c[1]), 0)

                # Prune dates outside 7-day window from the left, then count — O(1) amortized
                dq = team_recent[tid]
                while dq and (gdate - dq[0]).days > 7:
                    dq.popleft()
                games_7d[i] = len(dq)
                dq.append(gdate)
                team_last[tid] = (gdate, venue)

            df[f"{side}_rest_days"] = rest_days
            df[f"{side}_is_back_to_back"] = is_back_to_back
            df[f"{side}_travel_distance_miles"] = travel_distance
            df[f"{side}_games_in_last_7d"] = games_7d

        return df

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in miles between two coordinates."""
        import math
        R = 3959  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    # ── Rolling Features ───────────────────────────────────────────────

    def add_rolling_features(
        self, df: pd.DataFrame, window_sizes: list = None
    ) -> pd.DataFrame:
        """
        Add rolling averages of F5 runs scored/allowed per team.

        Reshapes data so each team's games are tracked chronologically
        regardless of home/away, then merges rolling stats back.
        Uses shift(1) to avoid leaking current game into its own features.
        """
        if window_sizes is None:
            window_sizes = ROLLING_WINDOWS

        if "away_f5_runs" not in df.columns or "home_f5_runs" not in df.columns:
            logger.warning("F5 run columns missing — skipping rolling features")
            return df

        df = df.sort_values("date").copy()

        # Build per-team game log: each row = one team's appearance in a game
        away_games = df[["game_pk", "date", "away_team_id", "away_f5_runs", "home_f5_runs"]].rename(
            columns={"away_team_id": "team_id", "away_f5_runs": "runs_scored", "home_f5_runs": "runs_allowed"}
        )
        home_games = df[["game_pk", "date", "home_team_id", "home_f5_runs", "away_f5_runs"]].rename(
            columns={"home_team_id": "team_id", "home_f5_runs": "runs_scored", "away_f5_runs": "runs_allowed"}
        )
        team_log = pd.concat([away_games, home_games], ignore_index=True)
        team_log = team_log.sort_values(["team_id", "date"])

        # Compute rolling stats per team (shifted to exclude current game)
        for window in window_sizes:
            team_log[f"roll{window}_scored"] = (
                team_log.groupby("team_id")["runs_scored"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            team_log[f"roll{window}_allowed"] = (
                team_log.groupby("team_id")["runs_allowed"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

        # Split back into away/home lookups and merge onto original df
        away_roll = team_log[team_log["game_pk"].isin(df["game_pk"])].copy()
        away_lookup = away_roll.merge(
            df[["game_pk", "away_team_id"]], on="game_pk"
        )
        away_lookup = away_lookup[away_lookup["team_id"] == away_lookup["away_team_id"]]

        home_lookup = away_roll.merge(
            df[["game_pk", "home_team_id"]], on="game_pk"
        )
        home_lookup = home_lookup[home_lookup["team_id"] == home_lookup["home_team_id"]]

        # Deduplicate before indexing to prevent cartesian product on merge
        away_lookup = away_lookup.drop_duplicates(subset="game_pk")
        home_lookup = home_lookup.drop_duplicates(subset="game_pk")

        for window in window_sizes:
            # Away team rolling features
            away_map = away_lookup.set_index("game_pk")[[f"roll{window}_scored", f"roll{window}_allowed"]]
            away_map.columns = [f"away_team_f5_runs_last{window}", f"away_team_f5_allowed_last{window}"]
            df = df.merge(away_map, left_on="game_pk", right_index=True, how="left")

            # Home team rolling features
            home_map = home_lookup.set_index("game_pk")[[f"roll{window}_scored", f"roll{window}_allowed"]]
            home_map.columns = [f"home_team_f5_runs_last{window}", f"home_team_f5_allowed_last{window}"]
            df = df.merge(home_map, left_on="game_pk", right_index=True, how="left")

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
