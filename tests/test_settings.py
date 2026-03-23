"""
Tests for config/settings.py — pure functions only, no side effects.
"""
import pytest
from config.settings import get_f5_ratio, F5_RATIO, F5_RATIO_TIERS, PARK_FACTORS


def test_f5_ratio_no_args_returns_historical_average():
    assert get_f5_ratio() == F5_RATIO  # 5/9 ≈ 0.556


def test_f5_ratio_ace_tier():
    assert get_f5_ratio(2.0) == F5_RATIO_TIERS["ace"]  # 0.52


def test_f5_ratio_mid_tier():
    assert get_f5_ratio(3.0) == F5_RATIO_TIERS["mid"]  # 0.556


def test_f5_ratio_back_end_tier():
    assert get_f5_ratio(4.0) == F5_RATIO_TIERS["back_end"]  # 0.60


def test_f5_ratio_ace_mid_boundary_is_mid():
    # 2.5 is not < 2.5, so falls into mid tier
    assert get_f5_ratio(2.5) == F5_RATIO_TIERS["mid"]


def test_f5_ratio_mid_back_boundary_is_back_end():
    # 3.5 is not < 3.5, so falls into back_end tier
    assert get_f5_ratio(3.5) == F5_RATIO_TIERS["back_end"]


def test_f5_ratio_none_returns_historical_average():
    assert get_f5_ratio(None) == F5_RATIO


def test_park_factors_coors_is_highest_run_environment():
    assert PARK_FACTORS["Coors Field"] == max(PARK_FACTORS.values())


def test_park_factors_all_positive():
    assert all(v > 0 for v in PARK_FACTORS.values())


def test_park_factors_coors_above_1():
    assert PARK_FACTORS["Coors Field"] > 1.0


def test_park_factors_petco_below_1():
    assert PARK_FACTORS["Petco Park"] < 1.0


def test_f5_ratio_tiers_ordered():
    # Ace starters suppress runs the most (lowest ratio)
    assert F5_RATIO_TIERS["ace"] < F5_RATIO_TIERS["mid"] < F5_RATIO_TIERS["back_end"]
