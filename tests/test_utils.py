"""
Tests for utils/__init__.py — pure helper functions.
"""
from datetime import datetime
from utils import today_str, yesterday_str, predictions_path


def test_today_str_is_valid_date():
    result = today_str()
    datetime.strptime(result, "%Y-%m-%d")  # raises ValueError if format is wrong


def test_yesterday_str_is_one_day_before_today():
    today = datetime.strptime(today_str(), "%Y-%m-%d")
    yesterday = datetime.strptime(yesterday_str(), "%Y-%m-%d")
    assert (today - yesterday).days == 1


def test_predictions_path_default_uses_today():
    path = predictions_path()
    assert path.name == f"{today_str()}.json"


def test_predictions_path_custom_date():
    path = predictions_path("2025-04-15")
    assert path.name == "2025-04-15.json"


def test_predictions_path_in_predictions_dir():
    path = predictions_path("2025-04-15")
    assert "predictions" in str(path)
