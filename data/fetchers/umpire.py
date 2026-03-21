"""
Umpire Data Fetcher
--------------------
Fetches home plate umpire assignments and builds umpire tendency profiles
from the MLB Stats API (free, no key required).

Umpire zone tendencies affect:
  - K rate: generous zone → more Ks → fewer runs
  - BB rate: tight zone → more walks → more baserunners
  - Overall runs per game
"""
import requests
import pandas as pd
import numpy as np
from typing import Optional
import logging
import time

from config.settings import MLB_STATS_BASE, DATA_DIR

logger = logging.getLogger(__name__)


class UmpireFetcher:
    """Fetches umpire data and builds tendency profiles."""

    def __init__(self):
        self.session = requests.Session()
        self.cache_dir = DATA_DIR / "umpires"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make MLB Stats API request."""
        url = f"{MLB_STATS_BASE}/{endpoint}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        time.sleep(0.25)  # rate limit
        return resp.json()

    def get_game_umpires(self, game_pk: int) -> dict:
        """
        Get umpire assignments for a specific game.

        Returns dict with HP umpire info or empty dict if unavailable.
        """
        try:
            data = self._get(f"game/{game_pk}/boxscore")
            officials = data.get("officials", [])
            for official in officials:
                if official.get("officialType") == "Home Plate":
                    person = official.get("official", {})
                    return {
                        "umpire_id": person.get("id"),
                        "umpire_name": person.get("fullName", "Unknown"),
                    }
        except Exception as e:
            logger.debug(f"Could not fetch umpires for game {game_pk}: {e}")
        return {}

    def build_umpire_assignments(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch HP umpire for all games. Caches results.

        Args:
            games_df: Must have 'game_pk' column

        Returns:
            DataFrame with game_pk, umpire_id, umpire_name
        """
        cache_path = self.cache_dir / "umpire_assignments.parquet"

        # Load existing cache
        cached = {}
        if cache_path.exists():
            cached_df = pd.read_parquet(cache_path)
            cached = set(cached_df["game_pk"].values)
            existing = cached_df.to_dict("records")
        else:
            existing = []

        results = list(existing)
        fetch_count = 0

        for _, game in games_df.iterrows():
            gpk = game["game_pk"]
            if gpk in cached:
                continue

            ump = self.get_game_umpires(gpk)
            if ump:
                ump["game_pk"] = gpk
                results.append(ump)

            fetch_count += 1
            if fetch_count % 100 == 0:
                logger.info(f"  Fetched umpires for {fetch_count} games...")

        ump_df = pd.DataFrame(results)

        if fetch_count > 0 and not ump_df.empty:
            ump_df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(ump_df)} umpire assignments")

        return ump_df

    def build_umpire_tendencies(
        self, ump_assignments: pd.DataFrame, games_df: pd.DataFrame
    ) -> dict:
        """
        Build umpire tendency profiles from historical game outcomes.

        For each umpire, computes:
          - avg total F5 runs in their games vs league avg
          - games behind plate (experience)

        Args:
            ump_assignments: DataFrame with game_pk, umpire_id
            games_df: Full game data with game_pk, total_f5_runs, etc.

        Returns:
            Dict {umpire_id: {rpg_factor, experience, ...}}
        """
        # Merge umpire assignments with game outcomes
        merged = ump_assignments.merge(
            games_df[["game_pk", "total_f5_runs"]].dropna(),
            on="game_pk",
            how="inner",
        )

        if merged.empty:
            return {}

        league_avg_f5 = merged["total_f5_runs"].mean()

        tendencies = {}
        for ump_id, group in merged.groupby("umpire_id"):
            n_games = len(group)
            if n_games < 10:  # need minimum sample
                continue

            avg_runs = group["total_f5_runs"].mean()

            tendencies[ump_id] = {
                "ump_rpg_factor": round(avg_runs / max(league_avg_f5, 0.1), 3),
                "ump_experience": n_games,
                "ump_avg_f5_runs": round(avg_runs, 2),
                "ump_name": group["umpire_name"].iloc[0] if "umpire_name" in group.columns else "Unknown",
            }

        logger.info(f"Built tendencies for {len(tendencies)} umpires (min 10 games)")
        return tendencies

    def get_umpire_features(
        self, game_pk: int, ump_assignments: pd.DataFrame, tendencies: dict
    ) -> dict:
        """
        Get umpire features for a specific game.

        Returns dict with umpire feature values or defaults.
        """
        defaults = {
            "ump_rpg_factor": 1.0,
            "ump_experience": 100,
        }

        if ump_assignments.empty:
            return defaults

        row = ump_assignments[ump_assignments["game_pk"] == game_pk]
        if row.empty:
            return defaults

        ump_id = row.iloc[0].get("umpire_id")
        if ump_id is None or ump_id not in tendencies:
            return defaults

        t = tendencies[ump_id]
        return {
            "ump_rpg_factor": t["ump_rpg_factor"],
            "ump_experience": t["ump_experience"],
        }
