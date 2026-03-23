"""
Tests for evaluation/accuracy_tracker.py.

Uses tmp_path + monkeypatch to avoid touching the real storage directory.
All MLB API and Odds API calls are mocked.
"""
import json
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock


# ── _fetch_closing_odds ────────────────────────────────────────────────────────

class TestFetchClosingOdds:
    def test_returns_empty_on_exception(self):
        import evaluation.accuracy_tracker as tracker
        mock_fetcher = MagicMock()
        mock_fetcher.get_historical_odds.side_effect = Exception("API unavailable")
        result = tracker._fetch_closing_odds(mock_fetcher, "2025-04-15T18:00:00Z", "Yankees", "Red Sox")
        assert result == {}

    def test_returns_empty_when_no_matching_game(self):
        import evaluation.accuracy_tracker as tracker
        mock_fetcher = MagicMock()
        mock_fetcher.get_historical_odds.return_value = pd.DataFrame()
        result = tracker._fetch_closing_odds(mock_fetcher, "2025-04-15T18:00:00Z", "Yankees", "Red Sox")
        assert result == {}


# ── check_yesterday_accuracy ──────────────────────────────────────────────────

class TestCheckYesterdayAccuracy:
    @pytest.fixture(autouse=True)
    def patch_dirs(self, tmp_path, monkeypatch):
        """Redirect PREDICTIONS_DIR and ACCURACY_DIR to tmp_path for all tests."""
        import evaluation.accuracy_tracker as tracker
        self.pred_dir = tmp_path / "predictions"
        self.acc_dir = tmp_path / "accuracy"
        self.pred_dir.mkdir()
        self.acc_dir.mkdir()
        monkeypatch.setattr(tracker, "PREDICTIONS_DIR", self.pred_dir)
        monkeypatch.setattr(tracker, "ACCURACY_DIR", self.acc_dir)

    def _yesterday(self):
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    def test_returns_empty_when_no_prediction_file(self):
        import evaluation.accuracy_tracker as tracker
        mock_mlb = MagicMock()
        result = tracker.check_yesterday_accuracy(mock_mlb)
        assert result == {}
        mock_mlb.get_schedule.assert_not_called()

    def test_returns_empty_when_no_final_games(self):
        import evaluation.accuracy_tracker as tracker
        yesterday = self._yesterday()
        (self.pred_dir / f"{yesterday}.json").write_text(json.dumps({"games": []}))

        mock_mlb = MagicMock()
        mock_mlb.get_schedule.return_value = pd.DataFrame(columns=["status", "home_team", "away_team"])
        result = tracker.check_yesterday_accuracy(mock_mlb)
        assert result == {}

    def test_correct_ml_accuracy_both_right(self):
        import evaluation.accuracy_tracker as tracker
        yesterday = self._yesterday()

        preds = {"games": [
            {
                "game_info": {
                    "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                    "commence_time": None,
                },
                "moneyline": {"home_prob": 0.62, "away_prob": 0.38},
                "total": {"predicted": 5.0},
                "edges": [],
            },
            {
                "game_info": {
                    "home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants",
                    "commence_time": None,
                },
                "moneyline": {"home_prob": 0.38, "away_prob": 0.62},
                "total": {"predicted": 4.5},
                "edges": [],
            },
        ]}
        (self.pred_dir / f"{yesterday}.json").write_text(json.dumps(preds))

        # Yankees won (home win=1), Giants won (home win=0 for Dodgers)
        actual = pd.DataFrame([
            {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
             "status": "Final", "home_f5_runs": 3, "away_f5_runs": 1},
            {"home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants",
             "status": "Final", "home_f5_runs": 1, "away_f5_runs": 2},
        ])
        mock_mlb = MagicMock()
        mock_mlb.get_schedule.return_value = actual

        result = tracker.check_yesterday_accuracy(mock_mlb)
        assert result["games_tracked"] == 2
        assert result["ml_accuracy"] == pytest.approx(100.0)

    def test_correct_ml_accuracy_both_wrong(self):
        import evaluation.accuracy_tracker as tracker
        yesterday = self._yesterday()

        preds = {"games": [
            {
                "game_info": {
                    "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                    "commence_time": None,
                },
                "moneyline": {"home_prob": 0.62, "away_prob": 0.38},  # predicted home win
                "total": {"predicted": 5.0},
                "edges": [],
            },
        ]}
        (self.pred_dir / f"{yesterday}.json").write_text(json.dumps(preds))

        # Away team won — prediction was wrong
        actual = pd.DataFrame([
            {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
             "status": "Final", "home_f5_runs": 1, "away_f5_runs": 3},
        ])
        mock_mlb = MagicMock()
        mock_mlb.get_schedule.return_value = actual

        result = tracker.check_yesterday_accuracy(mock_mlb)
        assert result["ml_accuracy"] == pytest.approx(0.0)

    def test_total_error_calculation(self):
        import evaluation.accuracy_tracker as tracker
        yesterday = self._yesterday()

        preds = {"games": [
            {
                "game_info": {
                    "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                    "commence_time": None,
                },
                "moneyline": {"home_prob": 0.60, "away_prob": 0.40},
                "total": {"predicted": 5.0},  # predicted 5
                "edges": [],
            },
        ]}
        (self.pred_dir / f"{yesterday}.json").write_text(json.dumps(preds))

        actual = pd.DataFrame([
            {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
             "status": "Final", "home_f5_runs": 2, "away_f5_runs": 1},  # actual total = 3
        ])
        mock_mlb = MagicMock()
        mock_mlb.get_schedule.return_value = actual

        result = tracker.check_yesterday_accuracy(mock_mlb)
        assert result["avg_total_error"] == pytest.approx(2.0)  # |5 - 3| = 2

    def test_accuracy_log_written_to_file(self):
        import evaluation.accuracy_tracker as tracker
        yesterday = self._yesterday()

        preds = {"games": [
            {
                "game_info": {
                    "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                    "commence_time": None,
                },
                "moneyline": {"home_prob": 0.60, "away_prob": 0.40},
                "total": {"predicted": 4.0},
                "edges": [],
            },
        ]}
        (self.pred_dir / f"{yesterday}.json").write_text(json.dumps(preds))

        actual = pd.DataFrame([
            {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
             "status": "Final", "home_f5_runs": 3, "away_f5_runs": 1},
        ])
        mock_mlb = MagicMock()
        mock_mlb.get_schedule.return_value = actual

        tracker.check_yesterday_accuracy(mock_mlb)

        log_path = self.acc_dir / "daily_accuracy.json"
        assert log_path.exists()
        with open(log_path) as f:
            log = json.load(f)
        assert len(log) == 1
        assert log[0]["date"] == yesterday

    def test_edge_bet_accuracy_tracked(self):
        import evaluation.accuracy_tracker as tracker
        yesterday = self._yesterday()

        preds = {"games": [
            {
                "game_info": {
                    "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                    "commence_time": None,
                },
                "moneyline": {"home_prob": 0.62, "away_prob": 0.38},
                "total": {"predicted": 5.0},
                "edges": [{"market": "Moneyline", "side": "Home", "edge_pct": 5.2}],
            },
        ]}
        (self.pred_dir / f"{yesterday}.json").write_text(json.dumps(preds))

        actual = pd.DataFrame([
            {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
             "status": "Final", "home_f5_runs": 3, "away_f5_runs": 1},
        ])
        mock_mlb = MagicMock()
        mock_mlb.get_schedule.return_value = actual

        result = tracker.check_yesterday_accuracy(mock_mlb)
        assert "edge_bet_accuracy" in result
        assert result["edge_bet_accuracy"] == pytest.approx(100.0)  # edge bet won
