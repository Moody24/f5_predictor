"""
Statcast / Baseball Savant Fetcher
-----------------------------------
Advanced pitch-level and batted-ball data via pybaseball.
Key F5 features: stuff+, pitch mix, barrel rates, xStats.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# pybaseball imports (lazy to handle import errors gracefully)
try:
    from pybaseball import (
        statcast,
        statcast_pitcher,
        statcast_batter,
        pitching_stats,
        batting_stats,
        playerid_lookup,
    )
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    logger.warning("pybaseball not installed. Statcast features disabled.")


class StatcastFetcher:
    """Fetches Statcast-level data for advanced pitcher/batter metrics."""

    def __init__(self):
        if not PYBASEBALL_AVAILABLE:
            raise ImportError("Install pybaseball: pip install pybaseball")
        self.cache_dir = DATA_DIR / "statcast"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Pitcher-Level Advanced Metrics ─────────────────────────────────

    def get_pitcher_statcast(
        self,
        pitcher_id: int,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Raw pitch-level Statcast data for a pitcher.
        Returns every pitch thrown in the date range.
        """
        cache_key = f"pitcher_{pitcher_id}_{start_date}_{end_date}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df = statcast_pitcher(start_date, end_date, pitcher_id)
        if not df.empty:
            df.to_parquet(cache_path, index=False)
        return df

    def get_pitcher_f5_profile(
        self,
        pitcher_id: int,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Derive F5-specific Statcast features for a pitcher.

        Returns:
            dict with pitch mix, velocity, movement, whiff rates,
            barrel rates against, and expected stats through ~75 pitches
            (approximate F5 pitch count).
        """
        df = self.get_pitcher_statcast(pitcher_id, start_date, end_date)
        if df.empty:
            return {}

        profile = {}

        # ── Pitch Mix ──────────────────────────────────────────────────
        if "pitch_type" in df.columns:
            mix = df["pitch_type"].value_counts(normalize=True)
            profile["pitch_mix"] = mix.to_dict()
            profile["n_pitch_types"] = len(mix)
            profile["primary_pitch"] = mix.index[0] if len(mix) > 0 else None
            profile["primary_pitch_pct"] = round(mix.iloc[0] * 100, 1) if len(mix) > 0 else 0

        # ── Velocity ───────────────────────────────────────────────────
        if "release_speed" in df.columns:
            fb_types = ["FF", "SI", "FC"]
            fastballs = df[df["pitch_type"].isin(fb_types)]
            profile["avg_fastball_velo"] = round(fastballs["release_speed"].mean(), 1) if not fastballs.empty else None
            profile["max_fastball_velo"] = round(fastballs["release_speed"].max(), 1) if not fastballs.empty else None

        # ── Whiff Rate ─────────────────────────────────────────────────
        if "description" in df.columns:
            swings = df[df["description"].isin([
                "swinging_strike", "swinging_strike_blocked",
                "foul", "foul_tip", "hit_into_play",
                "hit_into_play_no_out", "hit_into_play_score",
            ])]
            whiffs = df[df["description"].isin([
                "swinging_strike", "swinging_strike_blocked"
            ])]
            profile["whiff_rate"] = round(
                len(whiffs) / max(len(swings), 1) * 100, 1
            )

        # ── Called Strike + Whiff (CSW%) ───────────────────────────────
        if "description" in df.columns:
            csw_events = ["called_strike", "swinging_strike", "swinging_strike_blocked"]
            csw = df[df["description"].isin(csw_events)]
            profile["csw_pct"] = round(len(csw) / max(len(df), 1) * 100, 1)

        # ── Batted Ball Quality ────────────────────────────────────────
        batted = df[df["launch_speed"].notna()].copy()
        if not batted.empty:
            profile["avg_exit_velo_against"] = round(batted["launch_speed"].mean(), 1)
            profile["avg_launch_angle_against"] = round(batted["launch_angle"].mean(), 1)

            # Barrel rate (EV >= 98 + optimal LA)
            barrels = batted[
                (batted["launch_speed"] >= 98)
                & (batted["launch_angle"].between(26, 30))
            ]
            profile["barrel_rate_against"] = round(
                len(barrels) / max(len(batted), 1) * 100, 1
            )

            # Hard hit rate (>= 95 mph)
            hard_hit = batted[batted["launch_speed"] >= 95]
            profile["hard_hit_rate_against"] = round(
                len(hard_hit) / max(len(batted), 1) * 100, 1
            )

        # ── Expected Stats ─────────────────────────────────────────────
        for col in ["estimated_ba_using_speedangle", "estimated_woba_using_speedangle"]:
            if col in df.columns and df[col].notna().any():
                clean_name = "xBA_against" if "ba" in col else "xwOBA_against"
                profile[clean_name] = round(df[col].mean(), 3)

        # ── First Time Through Order (FTTO) ────────────────────────────
        # Approximate: first ~27 batters faced = ~first time through lineup x3
        # For F5, we care about first ~15 batters (1st + partial 2nd TTO)
        if "at_bat_number" in df.columns:
            grouped = df.groupby("game_pk")
            ftto_runs = []
            for game_pk, game_df in grouped:
                first_15 = game_df[game_df["at_bat_number"] <= 15]
                runs_scored = first_15[
                    first_15["events"].isin(["home_run", "single", "double", "triple"])
                    if "events" in first_15.columns else []
                ]
                # Simplified — real impl would track base states
            profile["ftto_sample_size"] = len(grouped)

        # ── Pitch Count Efficiency ─────────────────────────────────────
        if "game_pk" in df.columns:
            pitches_per_game = df.groupby("game_pk").size()
            profile["avg_pitches_per_game"] = round(pitches_per_game.mean(), 1)
            # Estimate if pitcher typically reaches 5 IP
            profile["pct_games_75plus_pitches"] = round(
                (pitches_per_game >= 75).mean() * 100, 1
            )

        return profile

    # ── Team-Level Batting Statcast ────────────────────────────────────

    def get_team_batting_statcast(
        self, season: int, min_pa: int = 50
    ) -> pd.DataFrame:
        """
        Get team-level batting stats from FanGraphs via pybaseball.
        Includes wRC+, wOBA, ISO, K%, BB%, etc.
        """
        cache_path = self.cache_dir / f"team_batting_{season}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df = batting_stats(season, qual=min_pa)
        if not df.empty:
            df.to_parquet(cache_path, index=False)
        return df

    def get_team_pitching_statcast(
        self, season: int, min_ip: int = 10
    ) -> pd.DataFrame:
        """
        Get pitcher-level stats from FanGraphs via pybaseball.
        Filter for starters, aggregate to team level.
        """
        cache_path = self.cache_dir / f"team_pitching_{season}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df = pitching_stats(season, qual=min_ip)
        if not df.empty:
            df.to_parquet(cache_path, index=False)
        return df

    # ── Profile Cache ──────────────────────────────────────────────────

    def save_profile_cache(self, profiles: dict, season: int):
        """Persist pitcher profile dict (pid → profile) as a parquet cache."""
        if not profiles:
            return
        rows = [{"pitcher_id": int(pid), **profile} for pid, profile in profiles.items()]
        cache_path = self.cache_dir / f"profiles_{season}.parquet"
        pd.DataFrame(rows).to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(profiles)} Statcast profiles to {cache_path}")

    def load_profile_cache(self, season: int) -> dict:
        """Load pitcher profile dict from parquet cache. Returns {} if not found."""
        cache_path = self.cache_dir / f"profiles_{season}.parquet"
        if not cache_path.exists():
            return {}
        df = pd.read_parquet(cache_path)
        profiles = {
            int(row["pitcher_id"]): {k: v for k, v in row.items() if k != "pitcher_id"}
            for _, row in df.iterrows()
        }
        logger.info(f"Loaded {len(profiles)} Statcast profiles from {cache_path}")
        return profiles

    # ── Matchup-Level Features ─────────────────────────────────────────

    def get_handedness_splits(
        self,
        pitcher_id: int,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Pitcher's performance split by batter handedness.
        Critical for F5 — lineup construction heavily favors platoon advantages.
        """
        df = self.get_pitcher_statcast(pitcher_id, start_date, end_date)
        if df.empty or "stand" not in df.columns:
            return {}

        splits = {}
        for hand in ["R", "L"]:
            subset = df[df["stand"] == hand]
            if subset.empty:
                continue
            batted = subset[subset["launch_speed"].notna()]
            splits[f"vs_{hand}_whiff_rate"] = round(
                len(subset[subset["description"].str.contains("swinging_strike", na=False)])
                / max(len(subset[subset["description"].notna()]), 1) * 100,
                1,
            )
            splits[f"vs_{hand}_avg_ev"] = (
                round(batted["launch_speed"].mean(), 1) if not batted.empty else None
            )
            if "estimated_woba_using_speedangle" in subset.columns:
                splits[f"vs_{hand}_xwOBA"] = round(
                    subset["estimated_woba_using_speedangle"].mean(), 3
                )
            splits[f"vs_{hand}_pa"] = len(subset["at_bat_number"].unique()) if "at_bat_number" in subset.columns else 0

        return splits
