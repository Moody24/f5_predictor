"""
Lineup Data Fetcher
--------------------
Fetches confirmed lineups and individual batter stats to compute
lineup-level features (avg wOBA, platoon advantage, etc.).

Data sources:
  - MLB Stats API boxscore endpoint (historical lineups)
  - pybaseball batting_stats() (season-level batter stats, bulk)
"""
import requests
import pandas as pd
import numpy as np
from typing import Optional
import logging
import time

from config.settings import MLB_STATS_BASE, DATA_DIR

logger = logging.getLogger(__name__)

try:
    from pybaseball import batting_stats
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False


class LineupFetcher:
    """Fetches lineup data and computes lineup-level features."""

    def __init__(self):
        self.session = requests.Session()
        self.cache_dir = DATA_DIR / "lineups"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._batter_cache = {}  # {season: DataFrame}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{MLB_STATS_BASE}/{endpoint}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        time.sleep(0.25)
        return resp.json()

    def get_game_lineup(self, game_pk: int) -> dict:
        """
        Get batting lineups for a completed game from boxscore.

        Returns:
            {"away": [{"player_id": int, "name": str, "bat_side": str}, ...],
             "home": [...]}
        """
        try:
            data = self._get(f"game/{game_pk}/boxscore")
            result = {"away": [], "home": []}

            for side, team_key in [("away", "away"), ("home", "home")]:
                team_data = data.get("teams", {}).get(team_key, {})
                batting_order = team_data.get("battingOrder", [])
                players = team_data.get("players", {})

                for pid in batting_order:
                    player_key = f"ID{pid}"
                    player = players.get(player_key, {})
                    person = player.get("person", {})
                    result[side].append({
                        "player_id": pid,
                        "name": person.get("fullName", "Unknown"),
                        "bat_side": person.get("batSide", {}).get("code", "R"),
                    })

            return result
        except Exception as e:
            logger.debug(f"Could not fetch lineup for game {game_pk}: {e}")
            return {"away": [], "home": []}

    def get_batter_season_stats(self, season: int) -> pd.DataFrame:
        """
        Get season batting stats for all qualified batters.
        Uses pybaseball for bulk fetch, caches per season.
        """
        if season in self._batter_cache:
            return self._batter_cache[season]

        cache_path = self.cache_dir / f"batter_stats_{season}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._batter_cache[season] = df
            return df

        if not PYBASEBALL_AVAILABLE:
            logger.warning("pybaseball not available — cannot fetch batter stats")
            return pd.DataFrame()

        try:
            logger.info(f"Fetching batter stats for {season} (bulk)...")
            df = batting_stats(season, qual=50)  # min 50 PA
            if df is not None and not df.empty:
                df.to_parquet(cache_path, index=False)
                self._batter_cache[season] = df
                logger.info(f"Cached {len(df)} batter stat lines for {season}")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch batter stats for {season}: {e}")

        return pd.DataFrame()

    def build_lineup_features(
        self, game_pk: int, lineup: dict, batter_stats_df: pd.DataFrame,
        opposing_pitcher_hand: str = "R",
    ) -> dict:
        """
        Compute lineup-level features from individual batter stats.

        Args:
            game_pk: Game identifier
            lineup: {"away": [...], "home": [...]} from get_game_lineup
            batter_stats_df: Season batter stats DataFrame
            opposing_pitcher_hand: "R" or "L"

        Returns:
            Dict with away_ and home_ prefixed lineup features
        """
        features = {}
        for side in ["away", "home"]:
            batters = lineup.get(side, [])
            if not batters or batter_stats_df.empty:
                features.update(self._default_lineup_features(side))
                continue

            # Match batters to their stats
            woba_vals = []
            ops_vals = []
            iso_vals = []
            platoon_advantage_count = 0

            for batter in batters[:9]:  # top 9 in order
                bat_side = batter.get("bat_side", "R")
                name = batter.get("name", "")

                # Try to match by name (pybaseball uses "Name" column)
                match = batter_stats_df[
                    batter_stats_df["Name"].str.contains(name.split()[-1], case=False, na=False)
                ]
                if match.empty:
                    woba_vals.append(0.320)
                    ops_vals.append(0.720)
                    iso_vals.append(0.150)
                else:
                    row = match.iloc[0]
                    woba_vals.append(float(row.get("wOBA", 0.320)))
                    ops_vals.append(float(row.get("OPS", 0.720)))
                    iso_vals.append(float(row.get("ISO", 0.150)))

                # Platoon advantage: L batter vs R pitcher or R vs L
                if (bat_side == "L" and opposing_pitcher_hand == "R") or \
                   (bat_side == "R" and opposing_pitcher_hand == "L"):
                    platoon_advantage_count += 1

            n = max(len(woba_vals), 1)
            features[f"{side}_lineup_avg_woba"] = round(np.mean(woba_vals), 3)
            features[f"{side}_lineup_avg_ops"] = round(np.mean(ops_vals), 3)
            features[f"{side}_lineup_total_iso"] = round(sum(iso_vals), 3)
            features[f"{side}_lineup_platoon_pct"] = round(platoon_advantage_count / 9 * 100, 1)

        return features

    @staticmethod
    def _default_lineup_features(side: str) -> dict:
        """League average lineup features when lineup data unavailable."""
        return {
            f"{side}_lineup_avg_woba": 0.320,
            f"{side}_lineup_avg_ops": 0.720,
            f"{side}_lineup_total_iso": 1.350,  # ~0.150 * 9
            f"{side}_lineup_platoon_pct": 50.0,
        }

    def build_batch_lineup_features(
        self, games_df: pd.DataFrame, season: int
    ) -> pd.DataFrame:
        """
        Build lineup features for all games in a season.
        Caches results.
        """
        cache_path = self.cache_dir / f"lineup_features_{season}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached lineup features for {season}")
            return pd.read_parquet(cache_path)

        batter_stats = self.get_batter_season_stats(season)
        if batter_stats.empty:
            logger.warning(f"No batter stats for {season} — using defaults")
            return pd.DataFrame()

        results = []
        season_games = games_df[games_df["date"].astype(str).str.startswith(str(season))]

        for i, (_, game) in enumerate(season_games.iterrows()):
            gpk = game["game_pk"]
            lineup = self.get_game_lineup(gpk)
            features = self.build_lineup_features(gpk, lineup, batter_stats)
            features["game_pk"] = gpk
            results.append(features)

            if (i + 1) % 100 == 0:
                logger.info(f"  Lineup features: {i+1}/{len(season_games)} games...")

        if results:
            df = pd.DataFrame(results)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached lineup features for {len(df)} games in {season}")
            return df

        return pd.DataFrame()
