"""
Tests for data/feature_engineering.py.
All tests use synthetic DataFrames — no network calls.
"""
import pandas as pd
import numpy as np
import pytest

from data.feature_engineering import FeatureEngineer


# ── Haversine ─────────────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        d = FeatureEngineer._haversine(40.7, -74.0, 40.7, -74.0)
        assert d == pytest.approx(0.0, abs=0.1)

    def test_nyc_to_la_approx_2445_miles(self):
        # NYC (40.7128, -74.0060) → LA (34.0522, -118.2437)
        d = FeatureEngineer._haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert 2400 < d < 2500

    def test_is_symmetric(self):
        d1 = FeatureEngineer._haversine(40.0, -75.0, 34.0, -118.0)
        d2 = FeatureEngineer._haversine(34.0, -118.0, 40.0, -75.0)
        assert d1 == pytest.approx(d2, rel=1e-6)


# ── Stat Helpers ──────────────────────────────────────────────────────────────

class TestCalcHelpers:
    def test_calc_rpg_normal(self):
        assert FeatureEngineer._calc_rpg({"runs": 120, "gamesPlayed": 30}) == pytest.approx(4.0)

    def test_calc_rpg_zero_games_no_divison_error(self):
        result = FeatureEngineer._calc_rpg({"runs": 50, "gamesPlayed": 0})
        assert result == pytest.approx(50.0)  # max(0, 1) = 1

    def test_calc_rpg_missing_keys_returns_zero(self):
        assert FeatureEngineer._calc_rpg({}) == pytest.approx(0.0)

    def test_calc_rate_normal(self):
        assert FeatureEngineer._calc_rate(25, 100) == pytest.approx(25.0)

    def test_calc_rate_zero_denom_no_division_error(self):
        result = FeatureEngineer._calc_rate(5, 0)
        assert result == pytest.approx(500.0)  # 5/max(0,1)*100


# ── Pitcher Feature Builder ───────────────────────────────────────────────────

class TestPitcherFeatures:
    def test_default_features_use_league_averages_not_zero(self):
        fe = FeatureEngineer()
        result = fe._default_pitcher_features("away_facing_")
        # ERA of 0 would look like an ace to the model — must use 4.50
        assert result["away_facing_starter_era_season"] == pytest.approx(4.50)
        # last5/last10 of 0 would look like an elite pitcher — must use 3.5
        assert result["away_facing_starter_last5_runs"] == pytest.approx(3.5)
        assert result["away_facing_starter_last10_runs"] == pytest.approx(3.5)

    def test_unknown_pitcher_id_uses_defaults(self):
        fe = FeatureEngineer()
        result = fe._build_pitcher_features(None, {}, {}, prefix="away_facing_")
        assert result["away_facing_starter_era_season"] == pytest.approx(4.50)

    def test_known_pitcher_uses_provided_stats(self, minimal_pitcher_stats):
        fe = FeatureEngineer()
        result = fe._build_pitcher_features(1001, minimal_pitcher_stats, {}, prefix="away_facing_")
        assert result["away_facing_starter_era_season"] == pytest.approx(3.20)
        assert result["away_facing_starter_avg_ip"] == pytest.approx(5.8)
        assert result["away_facing_starter_kbb_ratio"] == pytest.approx(3.5)

    def test_missing_last5_defaults_to_league_avg_not_zero(self):
        """Regression: None last5/last10 previously became 0 → ace signal."""
        fe = FeatureEngineer()
        sparse = {999: {"era": 4.00, "whip": 1.25}}  # no last5/last10 keys
        result = fe._build_pitcher_features(999, sparse, {}, prefix="away_facing_")
        assert result["away_facing_starter_last5_runs"] == pytest.approx(3.5)
        assert result["away_facing_starter_last10_runs"] == pytest.approx(3.5)

    def test_statcast_defaults_when_no_profile(self, minimal_pitcher_stats):
        fe = FeatureEngineer()
        result = fe._build_pitcher_features(1001, minimal_pitcher_stats, {}, prefix="away_facing_")
        # No statcast profile → league-average defaults
        assert result["away_facing_starter_whiff_rate"] == pytest.approx(24.0)
        assert result["away_facing_starter_xwoba_against"] == pytest.approx(0.320)

    def test_prior_season_key_lookup(self, minimal_pitcher_stats):
        """Training-time stats are keyed by (pid, game_season) to prevent leakage."""
        fe = FeatureEngineer()
        keyed_stats = {(1001, 2025): minimal_pitcher_stats[1001]}
        result = fe._build_pitcher_features(1001, keyed_stats, {}, prefix="away_facing_", game_season=2025)
        assert result["away_facing_starter_era_season"] == pytest.approx(3.20)


# ── Offense Feature Builder ───────────────────────────────────────────────────

class TestOffenseFeatures:
    def test_known_team_uses_provided_stats(self, minimal_team_stats):
        fe = FeatureEngineer()
        result = fe._build_offense_features(147, minimal_team_stats, prefix="home_")
        assert result["home_team_ops"] == pytest.approx(0.780)
        assert result["home_team_runs_per_game"] == pytest.approx(140 / 30, rel=1e-3)

    def test_unknown_team_uses_defaults_not_zero(self):
        fe = FeatureEngineer()
        result = fe._build_offense_features(9999, {}, prefix="home_")
        # .720 OPS default — not 0
        assert result["home_team_ops"] == pytest.approx(0.720)

    def test_iso_calculated_as_slg_minus_avg(self, minimal_team_stats):
        fe = FeatureEngineer()
        result = fe._build_offense_features(147, minimal_team_stats, prefix="home_")
        assert result["home_team_iso"] == pytest.approx(0.450 - 0.265, rel=1e-3)


# ── Context Features ──────────────────────────────────────────────────────────

class TestContextFeatures:
    def test_known_venue_park_factor(self, minimal_games):
        fe = FeatureEngineer()
        row = minimal_games.iloc[0]
        result = fe._build_context_features(row)
        assert result["park_factor"] == pytest.approx(1.04)  # Yankee Stadium

    def test_unknown_venue_default_park_factor(self):
        fe = FeatureEngineer()
        row = pd.Series({"venue_name": "Generic Ballpark"})
        result = fe._build_context_features(row)
        assert result["park_factor"] == pytest.approx(1.0)


# ── build_game_features ───────────────────────────────────────────────────────

class TestBuildGameFeatures:
    def test_row_count_matches_input(self, minimal_games, minimal_pitcher_stats, minimal_team_stats):
        fe = FeatureEngineer()
        df = fe.build_game_features(minimal_games, minimal_pitcher_stats, {}, minimal_team_stats)
        assert len(df) == len(minimal_games)

    def test_target_columns_present_when_f5_data_available(self, minimal_games):
        fe = FeatureEngineer()
        df = fe.build_game_features(minimal_games, {}, {}, {})
        assert "away_f5_runs" in df.columns
        assert "home_f5_runs" in df.columns
        assert "total_f5_runs" in df.columns
        assert "home_f5_win" in df.columns
        assert "f5_push" in df.columns

    def test_home_f5_win_flag_correct(self, minimal_games):
        fe = FeatureEngineer()
        df = fe.build_game_features(minimal_games, {}, {}, {})
        # game_pk=1001: home 3 > away 2 → win
        assert df.loc[df["game_pk"] == 1001, "home_f5_win"].iloc[0] == 1
        # game_pk=1002: tie (1-1) → no win
        assert df.loc[df["game_pk"] == 1002, "home_f5_win"].iloc[0] == 0

    def test_f5_push_flag_correct(self, minimal_games):
        fe = FeatureEngineer()
        df = fe.build_game_features(minimal_games, {}, {}, {})
        assert df.loc[df["game_pk"] == 1001, "f5_push"].iloc[0] == 0
        assert df.loc[df["game_pk"] == 1002, "f5_push"].iloc[0] == 1

    def test_weather_defaults_when_no_data_provided(self, minimal_games):
        fe = FeatureEngineer()
        df = fe.build_game_features(minimal_games, {}, {}, {})
        assert df["temperature_f"].iloc[0] == pytest.approx(72.0)
        assert df["wind_speed_mph"].iloc[0] == pytest.approx(5.0)
        assert df["humidity_pct"].iloc[0] == pytest.approx(50.0)

    def test_weather_data_overrides_defaults(self, minimal_games):
        fe = FeatureEngineer()
        weather = pd.DataFrame([{
            "game_pk": 1001,
            "temperature_f": 95.0, "wind_speed_mph": 20.0,
            "wind_direction_deg": 270.0, "humidity_pct": 80.0, "is_dome": 0,
        }])
        df = fe.build_game_features(minimal_games, {}, {}, {}, weather_data=weather)
        row = df.loc[df["game_pk"] == 1001].iloc[0]
        assert row["temperature_f"] == pytest.approx(95.0)
        assert row["wind_speed_mph"] == pytest.approx(20.0)

    def test_umpire_defaults_when_no_data(self, minimal_games):
        fe = FeatureEngineer()
        df = fe.build_game_features(minimal_games, {}, {}, {})
        assert df["ump_rpg_factor"].iloc[0] == pytest.approx(1.0)
        assert df["ump_experience"].iloc[0] == 100

    def test_umpire_data_used_when_provided(self, minimal_games):
        fe = FeatureEngineer()
        ump = {1001: {"ump_rpg_factor": 1.15, "ump_experience": 850}}
        df = fe.build_game_features(minimal_games, {}, {}, {}, umpire_data=ump)
        row = df.loc[df["game_pk"] == 1001].iloc[0]
        assert row["ump_rpg_factor"] == pytest.approx(1.15)
        assert row["ump_experience"] == 850


# ── Derived Column Names ──────────────────────────────────────────────────────

class TestDerivedColumnNames:
    def test_total_count(self):
        fe = FeatureEngineer()
        cols = fe.derived_column_names()
        # 2 sides × (3 windows × 2 + 4 travel) = 2 × 10 = 20
        assert len(cols) == 20

    def test_no_duplicates(self):
        fe = FeatureEngineer()
        cols = fe.derived_column_names()
        assert len(cols) == len(set(cols))

    def test_all_sides_present(self):
        fe = FeatureEngineer()
        cols = fe.derived_column_names()
        away_cols = [c for c in cols if c.startswith("away_")]
        home_cols = [c for c in cols if c.startswith("home_")]
        assert len(away_cols) == len(home_cols) == 10


# ── Rolling Features ──────────────────────────────────────────────────────────

class TestRollingFeatures:
    def test_skips_when_f5_columns_missing(self):
        fe = FeatureEngineer()
        games = pd.DataFrame([{
            "game_pk": 1, "date": "2025-04-01",
            "away_team_id": 111, "home_team_id": 147,
        }])
        result = fe.add_rolling_features(games)
        assert "away_team_f5_runs_last5" not in result.columns

    def test_min_periods_requires_3_games(self):
        """Only 2 games → min_periods=3 not met → all NaN."""
        fe = FeatureEngineer()
        games = pd.DataFrame([
            {"game_pk": 1, "date": "2025-04-01", "away_team_id": 111, "home_team_id": 147,
             "away_f5_runs": 3.0, "home_f5_runs": 2.0},
            {"game_pk": 2, "date": "2025-04-02", "away_team_id": 111, "home_team_id": 147,
             "away_f5_runs": 1.0, "home_f5_runs": 4.0},
        ])
        result = fe.add_rolling_features(games, window_sizes=[5])
        assert result["away_team_f5_runs_last5"].isna().all()

    def test_no_data_leakage_shift_applied(self):
        """Rolling features use shift(1) — current game is NOT in its own average."""
        fe = FeatureEngineer()
        games = pd.DataFrame([
            {"game_pk": i, "date": f"2025-04-{i:02d}", "away_team_id": 111, "home_team_id": 999,
             "away_f5_runs": float(i), "home_f5_runs": 1.0}
            for i in range(1, 12)  # 11 games, enough for window=5 with min_periods=3
        ])
        result = fe.add_rolling_features(games, window_sizes=[5])
        last_row = result.iloc[-1]
        # The last game's runs should NOT be in its own rolling average
        last_game_runs = float(games.iloc[-1]["away_f5_runs"])
        avg = last_row["away_team_f5_runs_last5"]
        if pd.notna(avg):
            assert avg != pytest.approx(last_game_runs)
