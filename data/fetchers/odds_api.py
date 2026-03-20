"""
Odds API Fetcher
-----------------
Fetches current and historical F5 (first 5 innings) lines
from The Odds API (https://the-odds-api.com).
"""
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
import logging

from config.settings import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, ODDS_BOOKMAKERS

logger = logging.getLogger(__name__)


class OddsApiFetcher:
    """Fetches betting lines from The Odds API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            logger.warning("No ODDS_API_KEY set. Odds fetching disabled.")
        self.base = ODDS_API_BASE
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with key."""
        if not self.api_key:
            raise ValueError("ODDS_API_KEY required. Set in .env file.")
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{self.base}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()

        # Log remaining requests
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API requests remaining: {remaining}")
        return resp.json()

    # ── Current Lines ──────────────────────────────────────────────────

    def get_current_odds(self, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
        """
        Get current odds for all upcoming MLB games.
        
        Markets:
            h2h     = moneyline
            spreads = run line
            totals  = over/under
        """
        data = self._get(
            f"sports/{ODDS_SPORT}/odds",
            {
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
                "bookmakers": ",".join(ODDS_BOOKMAKERS),
            },
        )
        return self._parse_odds(data)

    def get_event_odds(
        self, event_id: str, markets: str = "h2h,spreads,totals"
    ) -> pd.DataFrame:
        """Get odds for a specific game event."""
        data = self._get(
            f"sports/{ODDS_SPORT}/events/{event_id}/odds",
            {
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
                "bookmakers": ",".join(ODDS_BOOKMAKERS),
            },
        )
        return self._parse_odds([data] if isinstance(data, dict) else data)

    # ── Historical Odds ────────────────────────────────────────────────

    def get_historical_odds(
        self, date: str, markets: str = "h2h,spreads,totals"
    ) -> pd.DataFrame:
        """
        Get historical odds snapshot for a given date.
        Date format: 'YYYY-MM-DDTHH:MM:SSZ' (ISO 8601).
        Note: Historical endpoint requires paid tier.
        """
        data = self._get(
            f"sports/{ODDS_SPORT}/odds-history",
            {
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
                "date": date,
            },
        )
        odds_data = data.get("data", [])
        return self._parse_odds(odds_data)

    # ── Parsing ────────────────────────────────────────────────────────

    def _parse_odds(self, events: list) -> pd.DataFrame:
        """Parse raw odds API response into clean DataFrame."""
        rows = []
        for event in events:
            base = {
                "event_id": event.get("id", ""),
                "commence_time": event.get("commence_time", ""),
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
            }

            for bookmaker in event.get("bookmakers", []):
                book_name = bookmaker.get("key", "")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")

                    if market_key == "h2h":
                        rows.append(self._parse_h2h(base, book_name, market))
                    elif market_key == "spreads":
                        rows.append(self._parse_spread(base, book_name, market))
                    elif market_key == "totals":
                        rows.append(self._parse_total(base, book_name, market))

        df = pd.DataFrame(rows)
        if not df.empty:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
        return df

    def _parse_h2h(self, base: dict, book: str, market: dict) -> dict:
        """Parse moneyline market."""
        row = {**base, "bookmaker": book, "market": "moneyline"}
        for outcome in market.get("outcomes", []):
            if outcome["name"] == base["home_team"]:
                row["home_ml"] = outcome.get("price")
            elif outcome["name"] == base["away_team"]:
                row["away_ml"] = outcome.get("price")
        return row

    def _parse_spread(self, base: dict, book: str, market: dict) -> dict:
        """Parse run line / spread market."""
        row = {**base, "bookmaker": book, "market": "spread"}
        for outcome in market.get("outcomes", []):
            if outcome["name"] == base["home_team"]:
                row["home_spread"] = outcome.get("point")
                row["home_spread_price"] = outcome.get("price")
            elif outcome["name"] == base["away_team"]:
                row["away_spread"] = outcome.get("point")
                row["away_spread_price"] = outcome.get("price")
        return row

    def _parse_total(self, base: dict, book: str, market: dict) -> dict:
        """Parse over/under total market."""
        row = {**base, "bookmaker": book, "market": "total"}
        for outcome in market.get("outcomes", []):
            if outcome["name"] == "Over":
                row["total_line"] = outcome.get("point")
                row["over_price"] = outcome.get("price")
            elif outcome["name"] == "Under":
                row["under_price"] = outcome.get("price")
        return row

    # ── F5-Specific Line Parsing ───────────────────────────────────────

    def get_f5_lines(self) -> pd.DataFrame:
        """
        Fetch F5-specific lines if available.
        The Odds API may have alternate markets for F5.
        Falls back to full-game lines with F5 adjustment factors.
        """
        try:
            # Try F5-specific markets (alternate_totals, alternate_spreads)
            data = self._get(
                f"sports/{ODDS_SPORT}/odds",
                {
                    "regions": "us",
                    "markets": "alternate_totals,alternate_spreads",
                    "oddsFormat": "american",
                    "bookmakers": ",".join(ODDS_BOOKMAKERS),
                },
            )
            df = self._parse_odds(data)
            if not df.empty:
                return df
        except Exception as e:
            logger.info(f"F5 alternate markets not available: {e}")

        # Fallback: use full game lines and apply F5 ratio
        logger.info("Using full-game lines with F5 adjustment (ratio ~0.556)")
        full = self.get_current_odds()
        return self._apply_f5_adjustment(full)

    def _apply_f5_adjustment(self, full_game_df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate F5 lines from full-game lines.
        Historical F5/FG ratio is approximately 5/9 ≈ 0.556 for totals.
        """
        F5_RATIO = 5 / 9

        df = full_game_df.copy()
        if "total_line" in df.columns:
            df["f5_total_est"] = (df["total_line"] * F5_RATIO).round(1)
        if "home_spread" in df.columns:
            df["f5_spread_est"] = (df["home_spread"] * F5_RATIO).round(1)

        return df

    # ── Implied Probability Conversion ─────────────────────────────────

    @staticmethod
    def american_to_implied_prob(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds is None:
            return None
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def implied_prob_to_american(prob: float) -> float:
        """Convert implied probability to American odds."""
        if prob is None or prob <= 0 or prob >= 1:
            return None
        if prob >= 0.5:
            return round(-prob / (1 - prob) * 100)
        else:
            return round((1 - prob) / prob * 100)

    def add_implied_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add implied probability columns to odds DataFrame."""
        df = df.copy()
        for col in ["home_ml", "away_ml", "over_price", "under_price",
                     "home_spread_price", "away_spread_price"]:
            if col in df.columns:
                prob_col = col.replace("_ml", "_implied").replace("_price", "_implied")
                df[prob_col] = df[col].apply(self.american_to_implied_prob)
        return df
