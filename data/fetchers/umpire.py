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
        Build umpire tendency profiles using leave-one-out (LOO) estimates.

        For each historical game, the umpire's tendency is computed from all
        OTHER games they called — the current game's outcome is excluded so it
        cannot leak into its own feature. A full-history entry (keyed by
        ump_id) is also stored for use at inference time (today's games have
        no outcome to exclude).

        Returns:
            Dict with two key types:
              game_pk (int)      -> tendency for that specific historical game (LOO)
              "ump_{ump_id}" (str) -> tendency for inference (full history)
        """
        merged = ump_assignments.merge(
            games_df[["game_pk", "total_f5_runs"]].dropna(),
            on="game_pk",
            how="inner",
        )

        if merged.empty:
            return {}

        league_avg_f5 = merged["total_f5_runs"].mean()

        # Per-umpire aggregates for LOO computation
        ump_agg = merged.groupby("umpire_id")["total_f5_runs"].agg(
            total_runs="sum", n_games="count"
        )

        tendencies: dict = {}

        for _, row in merged.iterrows():
            ump_id = row["umpire_id"]
            if ump_id not in ump_agg.index:
                continue
            agg = ump_agg.loc[ump_id]
            n = int(agg["n_games"])
            if n < 11:  # need at least 10 prior games after LOO
                continue
            loo_avg = (agg["total_runs"] - row["total_f5_runs"]) / (n - 1)
            tendencies[int(row["game_pk"])] = {
                "ump_rpg_factor": round(loo_avg / max(league_avg_f5, 0.1), 3),
                "ump_experience": n - 1,
            }

        # Full-history entries keyed by ump_id string (used at inference)
        for ump_id, agg in ump_agg.iterrows():
            n = int(agg["n_games"])
            if n < 11:  # match LOO threshold: need at least 10 prior games after LOO
                continue
            full_avg = agg["total_runs"] / n
            tendencies[f"ump_{ump_id}"] = {
                "ump_rpg_factor": round(full_avg / max(league_avg_f5, 0.1), 3),
                "ump_experience": n,
            }

        n_games = sum(1 for k in tendencies if isinstance(k, int))
        n_umps = sum(1 for k in tendencies if isinstance(k, str))
        logger.info(f"Built tendencies for {n_umps} umpires (min 11 games), {n_games} LOO game entries")
        return tendencies

    def get_umpire_features(
        self, game_pk: int, ump_assignments: pd.DataFrame, tendencies: dict
    ) -> dict:
        """
        Get umpire features for a specific game.

        For historical games (training): uses the LOO entry keyed by game_pk.
        For inference (today's games): falls back to the full-history entry
        keyed by ump_id.

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

        # Prefer LOO entry for historical games (no target leakage)
        if game_pk in tendencies:
            t = tendencies[game_pk]
            return {
                "ump_rpg_factor": t["ump_rpg_factor"],
                "ump_experience": t["ump_experience"],
            }

        # Inference path: use full-history entry keyed by ump_id
        ump_key = f"ump_{ump_id}"
        if ump_key in tendencies:
            t = tendencies[ump_key]
            return {
                "ump_rpg_factor": t["ump_rpg_factor"],
                "ump_experience": t["ump_experience"],
            }

        return defaults
