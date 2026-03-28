"""
Weather Data Fetcher
---------------------
Fetches game-time weather from Open-Meteo (free, no API key required).

Historical: archive-api.open-meteo.com
Forecast: api.open-meteo.com

Weather features significantly impact run scoring:
  - Temperature: every 10°F above 70 → ~0.5 more runs/game
  - Wind: blowing out → more HRs; blowing in → fewer
  - Humidity: high humidity → ball carries less (common myth is reversed)
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import logging
import time

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


# Venue coordinates and timezone for all 30 MLB parks
VENUE_COORDS = {
    "Chase Field": (33.4455, -112.0667, "America/Phoenix"),
    "Truist Park": (33.8907, -84.4677, "America/New_York"),
    "Oriole Park": (39.2839, -76.6217, "America/New_York"),
    "Fenway Park": (42.3467, -71.0972, "America/New_York"),
    "Wrigley Field": (41.9484, -87.6553, "America/Chicago"),
    "Guaranteed Rate Field": (41.8299, -87.6338, "America/Chicago"),
    "Great American Ball Park": (39.0975, -84.5069, "America/New_York"),
    "Progressive Field": (41.4962, -81.6852, "America/New_York"),
    "Coors Field": (39.7561, -104.9942, "America/Denver"),
    "Comerica Park": (42.3390, -83.0485, "America/New_York"),
    "Minute Maid Park": (29.7572, -95.3555, "America/Chicago"),
    "Kauffman Stadium": (39.0517, -94.4803, "America/Chicago"),
    "Angel Stadium": (33.8003, -117.8827, "America/Los_Angeles"),
    "Dodger Stadium": (34.0739, -118.2400, "America/Los_Angeles"),
    "loanDepot Park": (25.7781, -80.2196, "America/New_York"),
    "American Family Field": (43.0280, -87.9712, "America/Chicago"),
    "Target Field": (44.9817, -93.2776, "America/Chicago"),
    "Citi Field": (40.7571, -73.8458, "America/New_York"),
    "Yankee Stadium": (40.8296, -73.9262, "America/New_York"),
    "Oakland Coliseum": (37.7516, -122.2005, "America/Los_Angeles"),
    "Citizens Bank Park": (39.9061, -75.1665, "America/New_York"),
    "PNC Park": (40.4469, -80.0057, "America/New_York"),
    "Petco Park": (32.7076, -117.1570, "America/Los_Angeles"),
    "Oracle Park": (37.7786, -122.3893, "America/Los_Angeles"),
    "T-Mobile Park": (47.5914, -122.3325, "America/Los_Angeles"),
    "Busch Stadium": (38.6226, -90.1928, "America/Chicago"),
    "Tropicana Field": (27.7682, -82.6534, "America/New_York"),
    "Globe Life Field": (32.7512, -97.0832, "America/Chicago"),
    "Rogers Centre": (43.6414, -79.3894, "America/New_York"),
    "Nationals Park": (38.8730, -77.0074, "America/New_York"),
}

# Stadiums with fixed or retractable roofs (weather = neutral when closed)
DOME_STADIUMS = {
    "Tropicana Field",       # fixed dome
    "Globe Life Field",      # retractable
    "Rogers Centre",         # retractable
    "loanDepot Park",        # retractable
    "Chase Field",           # retractable
    "Minute Maid Park",      # retractable
    "T-Mobile Park",         # retractable
    "American Family Field", # retractable
}

# Neutral weather values for dome stadiums
DOME_WEATHER = {
    "temperature_f": 72.0,
    "wind_speed_mph": 0.0,
    "wind_direction_deg": 0.0,
    "humidity_pct": 50.0,
    "is_dome": 1,
}


class WeatherFetcher:
    """Fetches game-time weather from Open-Meteo API."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.session = requests.Session()
        self.cache_dir = DATA_DIR / "weather"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_game_weather(
        self, venue_name: str, date: str, hour: int = 19
    ) -> dict:
        """
        Get weather for a specific game.

        Args:
            venue_name: MLB venue name (must be in VENUE_COORDS)
            date: Game date 'YYYY-MM-DD'
            hour: Local hour of game start (default 19 = 7pm)

        Returns:
            Dict with temperature_f, wind_speed_mph, wind_direction_deg,
            humidity_pct, is_dome
        """
        if venue_name in DOME_STADIUMS:
            return DOME_WEATHER.copy()

        coords = VENUE_COORDS.get(venue_name)
        if not coords:
            logger.warning(
                f"No coordinates for venue '{venue_name}' — using neutral weather. "
                f"Add to VENUE_COORDS in config/settings.py if this is a real MLB park."
            )
            return self._default_weather()

        lat, lon, tz = coords

        # Determine if this is historical or forecast
        today = datetime.now().strftime("%Y-%m-%d")
        is_forecast = date >= today

        try:
            url = self.FORECAST_URL if is_forecast else self.ARCHIVE_URL
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
                "timezone": tz,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
            }
            if is_forecast:
                params["start_date"] = date
                params["end_date"] = date
            else:
                params["start_date"] = date
                params["end_date"] = date

            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            # Find the closest hour to game time
            target_hour = f"{date}T{hour:02d}:00"
            idx = None
            for i, t in enumerate(times):
                if t == target_hour:
                    idx = i
                    break
            if idx is None and times:
                idx = min(range(len(times)), key=lambda i: abs(
                    int(times[i].split("T")[1].split(":")[0]) - hour
                ))

            if idx is not None:
                temp = hourly.get("temperature_2m", [None])[idx]
                humidity = hourly.get("relative_humidity_2m", [None])[idx]
                wind_speed = hourly.get("wind_speed_10m", [None])[idx]
                wind_dir = hourly.get("wind_direction_10m", [None])[idx]

                return {
                    "temperature_f": temp if temp is not None else 72.0,
                    "wind_speed_mph": wind_speed if wind_speed is not None else 5.0,
                    "wind_direction_deg": wind_dir if wind_dir is not None else 0.0,
                    "humidity_pct": humidity if humidity is not None else 50.0,
                    "is_dome": 0,
                }

        except Exception as e:
            logger.debug(f"Weather fetch failed for {venue_name} on {date}: {e}")

        return self._default_weather()

    def get_batch_weather(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch weather for all games in a DataFrame.
        Caches results to avoid redundant API calls.

        Args:
            games_df: Must have 'game_pk', 'date', 'venue_name' columns

        Returns:
            DataFrame with weather columns indexed by game_pk
        """
        cache_path = self.cache_dir / "game_weather.parquet"

        # Load existing cache
        cached = {}
        if cache_path.exists():
            cached_df = pd.read_parquet(cache_path)
            cached = dict(zip(cached_df["game_pk"], cached_df.to_dict("records")))
            logger.info(f"Loaded {len(cached)} cached weather records")

        results = []
        fetch_count = 0

        for _, game in games_df.iterrows():
            gpk = game["game_pk"]
            if gpk in cached:
                results.append(cached[gpk])
                continue

            weather = self.get_game_weather(
                game.get("venue_name", ""),
                str(game["date"])[:10],
            )
            weather["game_pk"] = gpk
            results.append(weather)
            fetch_count += 1

            # Rate limit: Open-Meteo allows ~10k req/day but be polite
            if fetch_count % 100 == 0:
                logger.info(f"  Fetched weather for {fetch_count} games...")
                time.sleep(1)

        weather_df = pd.DataFrame(results)

        # Update cache
        if fetch_count > 0:
            weather_df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(weather_df)} weather records")

        return weather_df

    @staticmethod
    def _default_weather() -> dict:
        """Neutral weather defaults when data unavailable."""
        return {
            "temperature_f": 72.0,
            "wind_speed_mph": 5.0,
            "wind_direction_deg": 0.0,
            "humidity_pct": 50.0,
            "is_dome": 0,
        }
