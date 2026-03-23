"""
Shared fixtures for the F5 Predictor test suite.
All fixtures use synthetic data — no network calls, no API keys required.
"""
import pandas as pd
import pytest


@pytest.fixture
def minimal_games():
    """Two completed games with all required fields."""
    return pd.DataFrame([
        {
            "game_pk": 1001,
            "date": "2025-04-15",
            "season": 2025,
            "away_team_id": 111,
            "home_team_id": 147,
            "away_starter_id": 1001,
            "home_starter_id": 2001,
            "venue_name": "Yankee Stadium",
            "status": "Final",
            "away_f5_runs": 2.0,
            "home_f5_runs": 3.0,
            "total_f5_runs": 5.0,
        },
        {
            "game_pk": 1002,
            "date": "2025-04-16",
            "season": 2025,
            "away_team_id": 111,
            "home_team_id": 147,
            "away_starter_id": 1002,
            "home_starter_id": 2002,
            "venue_name": "Yankee Stadium",
            "status": "Final",
            "away_f5_runs": 1.0,
            "home_f5_runs": 1.0,
            "total_f5_runs": 2.0,
        },
    ])


@pytest.fixture
def minimal_pitcher_stats():
    """Minimal pitcher stats dict keyed by pitcher_id."""
    return {
        1001: {
            "era": 3.20,
            "whip": 1.10,
            "k_bb_ratio": 3.5,
            "avg_ip": 5.8,
            "pct_5ip_plus": 72.0,
            "avg_pitches": 92.0,
            "avg_runs_per_start": 2.5,
            "last5_avg_runs": 2.2,
            "last10_avg_runs": 2.6,
            "qualified_starts": 15,
        },
        2001: {
            "era": 4.80,
            "whip": 1.45,
            "k_bb_ratio": 2.1,
            "avg_ip": 4.8,
            "pct_5ip_plus": 55.0,
            "avg_pitches": 82.0,
            "avg_runs_per_start": 3.8,
            "last5_avg_runs": 4.1,
            "last10_avg_runs": 3.9,
            "qualified_starts": 10,
        },
    }


@pytest.fixture
def minimal_team_stats():
    """Minimal team stats dict keyed by team_id."""
    return {
        111: {"hitting": {
            "ops": ".740", "slg": ".430", "avg": ".255",
            "runs": 120, "gamesPlayed": 30,
            "strikeOuts": 250, "baseOnBalls": 90, "plateAppearances": 1100,
        }},
        147: {"hitting": {
            "ops": ".780", "slg": ".450", "avg": ".265",
            "runs": 140, "gamesPlayed": 30,
            "strikeOuts": 240, "baseOnBalls": 100, "plateAppearances": 1150,
        }},
    }
