"""
MLB Stats API Fetcher
---------------------
Free, no-key-required access to game schedules, box scores,
pitcher game logs, and team stats from statsapi.mlb.com.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time
import logging

from config.settings import MLB_STATS_BASE, CURRENT_SEASON, DATA_DIR

logger = logging.getLogger(__name__)


class MLBStatsFetcher:
    """Fetches data from the official MLB Stats API."""

    def __init__(self):
        self.base = MLB_STATS_BASE
        self.session = requests.Session()
        self.cache_dir = DATA_DIR / "mlb_stats"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Rate-limited GET request."""
        url = f"{self.base}/{endpoint}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        time.sleep(0.25)  # respect rate limits
        return resp.json()

    # ── Schedule & Game Data ───────────────────────────────────────────

    def get_schedule(
        self, start_date: str, end_date: str, season: int = CURRENT_SEASON
    ) -> pd.DataFrame:
        """
        Fetch game schedule between dates.
        Dates in 'YYYY-MM-DD' format.
        Returns DataFrame with game_pk, date, teams, venue, status.
        """
        data = self._get(
            "schedule",
            {
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
                "hydrate": "team,venue,probablePitcher,linescore",
            },
        )
        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                away = game.get("teams", {}).get("away", {})
                home = game.get("teams", {}).get("home", {})
                linescore = game.get("linescore", {})

                # Extract F5 score from linescore innings
                innings = linescore.get("innings", [])
                away_f5 = sum(
                    inn.get("away", {}).get("runs", 0) for inn in innings[:5]
                )
                home_f5 = sum(
                    inn.get("home", {}).get("runs", 0) for inn in innings[:5]
                )

                rows.append(
                    {
                        "game_pk": game["gamePk"],
                        "date": date_entry["date"],
                        "status": game.get("status", {}).get("detailedState", ""),
                        "venue_name": game.get("venue", {}).get("name", ""),
                        "venue_id": game.get("venue", {}).get("id"),
                        "away_team": away.get("team", {}).get("name", ""),
                        "away_team_id": away.get("team", {}).get("id"),
                        "home_team": home.get("team", {}).get("name", ""),
                        "home_team_id": home.get("team", {}).get("id"),
                        "away_starter_id": away.get("probablePitcher", {}).get("id"),
                        "away_starter_name": away.get("probablePitcher", {}).get(
                            "fullName", ""
                        ),
                        "home_starter_id": home.get("probablePitcher", {}).get("id"),
                        "home_starter_name": home.get("probablePitcher", {}).get(
                            "fullName", ""
                        ),
                        "away_f5_runs": away_f5 if innings else None,
                        "home_f5_runs": home_f5 if innings else None,
                        "total_f5_runs": (away_f5 + home_f5) if innings else None,
                        "total_innings_played": len(innings),
                    }
                )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} games from {start_date} to {end_date}")
        return df

    # ── Pitcher Stats ──────────────────────────────────────────────────

    def get_pitcher_season_stats(self, player_id: int, season: int = CURRENT_SEASON) -> dict:
        """Get a pitcher's season-level stats."""
        data = self._get(
            f"people/{player_id}/stats",
            {"stats": "season", "season": season, "group": "pitching"},
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        return splits[0].get("stat", {})

    def get_pitcher_game_log(
        self, player_id: int, season: int = CURRENT_SEASON
    ) -> pd.DataFrame:
        """Get pitcher's game-by-game log for a season."""
        data = self._get(
            f"people/{player_id}/stats",
            {"stats": "gameLog", "season": season, "group": "pitching"},
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        rows = []
        for split in splits:
            stat = split.get("stat", {})
            game = split.get("game", {})
            opponent = split.get("opponent", {})
            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "date": split.get("date", ""),
                    "opponent": opponent.get("name", ""),
                    "opponent_id": opponent.get("id"),
                    "innings_pitched": float(stat.get("inningsPitched", "0")),
                    "hits": stat.get("hits", 0),
                    "runs": stat.get("runs", 0),
                    "earned_runs": stat.get("earnedRuns", 0),
                    "walks": stat.get("baseOnBalls", 0),
                    "strikeouts": stat.get("strikeOuts", 0),
                    "home_runs": stat.get("homeRuns", 0),
                    "pitches_thrown": stat.get("numberOfPitches", 0),
                    "strikes": stat.get("strikes", 0),
                    "era": float(stat.get("era", "0")),
                    "whip": float(stat.get("whip", "0")),
                }
            )
        return pd.DataFrame(rows)

    def get_pitcher_f5_stats(
        self, player_id: int, season: int = CURRENT_SEASON
    ) -> dict:
        """
        Derive F5-specific stats from game log.
        Returns rolling averages of runs allowed through 5 IP.

        Falls back to prior seasons if the requested season has no data
        (covers retired pitchers, injury years, and bullpen conversions).
        """
        # Try requested season first, then fall back through prior seasons
        seasons_to_try = [season] + [s for s in range(season - 1, 2020, -1)]
        log = pd.DataFrame()
        used_season = season
        for s in seasons_to_try:
            log = self.get_pitcher_game_log(player_id, s)
            if not log.empty:
                used_season = s
                break

        if log.empty:
            return {}

        if used_season != season:
            logger.debug(f"Pitcher {player_id}: no {season} data, using {used_season}")

        # Filter games where pitcher threw at least 5 innings
        qualified = log[log["innings_pitched"] >= 5.0].copy()
        if qualified.empty:
            return {"games": 0, "qualified_starts": 0}

        return {
            "games": len(log),
            "qualified_starts": len(qualified),
            "avg_ip": round(qualified["innings_pitched"].mean(), 2),
            "avg_runs_per_start": round(qualified["runs"].mean(), 2),
            "avg_er_per_start": round(qualified["earned_runs"].mean(), 2),
            "avg_hits_per_start": round(qualified["hits"].mean(), 2),
            "avg_walks_per_start": round(qualified["walks"].mean(), 2),
            "avg_k_per_start": round(qualified["strikeouts"].mean(), 2),
            "avg_pitches": round(qualified["pitches_thrown"].mean(), 1),
            "k_bb_ratio": round(
                qualified["strikeouts"].sum()
                / max(qualified["walks"].sum(), 1),
                2,
            ),
            "pct_5ip_plus": round(len(qualified) / len(log) * 100, 1),
            # Last N rolling
            "last5_avg_runs": round(
                qualified.tail(5)["runs"].mean(), 2
            ) if len(qualified) >= 5 else None,
            "last10_avg_runs": round(
                qualified.tail(10)["runs"].mean(), 2
            ) if len(qualified) >= 10 else None,
        }

    # ── Team Stats ─────────────────────────────────────────────────────

    def get_team_stats(
        self, team_id: int, season: int = CURRENT_SEASON
    ) -> dict:
        """Get team-level batting and pitching stats."""
        data = self._get(
            f"teams/{team_id}/stats",
            {"stats": "season", "season": season, "group": "hitting,pitching"},
        )
        result = {}
        for stat_group in data.get("stats", []):
            group_name = stat_group.get("group", {}).get("displayName", "")
            splits = stat_group.get("splits", [])
            if splits:
                result[group_name.lower()] = splits[0].get("stat", {})
        return result

    def get_team_roster(self, team_id: int, season: int = CURRENT_SEASON) -> pd.DataFrame:
        """Get active roster for a team."""
        data = self._get(f"teams/{team_id}/roster", {"season": season})
        rows = []
        for player in data.get("roster", []):
            person = player.get("person", {})
            rows.append(
                {
                    "player_id": person.get("id"),
                    "name": person.get("fullName", ""),
                    "position": player.get("position", {}).get("abbreviation", ""),
                    "status": player.get("status", {}).get("code", ""),
                }
            )
        return pd.DataFrame(rows)

    # ── Bulk Historical Data ───────────────────────────────────────────

    def fetch_season_games(
        self, season: int = CURRENT_SEASON, game_type: str = "R"
    ) -> pd.DataFrame:
        """
        Fetch all games for a full season. 
        game_type: 'R' = regular season, 'P' = postseason
        """
        # MLB regular season roughly April 1 - October 1
        start = f"{season}-03-20"
        end = f"{season}-10-05"
        df = self.get_schedule(start, end, season)
        # Filter to Final games only for training data
        final = df[df["status"] == "Final"].copy()
        logger.info(
            f"Season {season}: {len(final)} completed games out of {len(df)} total"
        )
        return final

    def fetch_multi_season(
        self, seasons: list[int] = None
    ) -> pd.DataFrame:
        """Fetch multiple seasons and concatenate."""
        if seasons is None:
            seasons = [2021, 2022, 2023, 2024]

        cache_path = self.cache_dir / f"games_{'_'.join(map(str, seasons))}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        frames = []
        for season in seasons:
            logger.info(f"Fetching season {season}...")
            df = self.fetch_season_games(season)
            df["season"] = season
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(combined)} games to {cache_path}")
        return combined
