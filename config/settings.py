"""
F5 Predictor Configuration
--------------------------
Central settings for all modules. API keys loaded from .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cache"
MODEL_DIR = BASE_DIR / "models" / "saved"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
# MLB Stats API is free / no key required
# Statcast via pybaseball is free / no key required

# ── MLB Stats API ──────────────────────────────────────────────────────
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
CURRENT_SEASON = 2025

# ── Feature Engineering ────────────────────────────────────────────────
# Rolling windows for pitcher/team stats
ROLLING_WINDOWS = [5, 10, 20]  # games
PITCHER_MIN_INNINGS = 15.0      # minimum IP to include in training

# ── Park Factors (2024 estimates — update yearly) ──────────────────────
# Runs factor relative to 1.0 (league avg)
PARK_FACTORS = {
    "Coors Field": 1.28,
    "Fenway Park": 1.08,
    "Globe Life Field": 1.06,
    "Great American Ball Park": 1.10,
    "Citizens Bank Park": 1.05,
    "Wrigley Field": 1.03,
    "Yankee Stadium": 1.04,
    "Guaranteed Rate Field": 1.02,
    "Chase Field": 1.01,
    "Angel Stadium": 0.98,
    "Truist Park": 0.98,
    "Busch Stadium": 0.97,
    "Dodger Stadium": 0.96,
    "T-Mobile Park": 0.95,
    "Oracle Park": 0.93,
    "Petco Park": 0.92,
    "Oakland Coliseum": 0.94,
    "Tropicana Field": 0.96,
    "Kauffman Stadium": 0.97,
    "Comerica Park": 0.95,
    "Progressive Field": 0.99,
    "Target Field": 1.00,
    "Minute Maid Park": 1.02,
    "PNC Park": 0.96,
    "Nationals Park": 1.00,
    "Citi Field": 0.95,
    "Rogers Centre": 1.01,
    "American Family Field": 1.03,
    "loanDepot Park": 0.96,
    "Oriole Park": 1.01,
}

# Default park factor for unknown venues
DEFAULT_PARK_FACTOR = 1.0

# ── Model Hyperparameters ──────────────────────────────────────────────
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_REGRESSOR_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# ── ZINB Settings ──────────────────────────────────────────────────────
ZINB_MAX_ITER = 1000
ZINB_DISP_FLAG = True  # parameterize dispersion with covariates

# ── Simulation ─────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000  # Monte Carlo sims for probability estimation

# ── Betting Thresholds ─────────────────────────────────────────────────
MIN_EDGE_PCT = 3.0       # minimum edge % to flag a bet
MIN_KELLY_FRACTION = 0.5 # fraction of full Kelly to use
BANKROLL = 1000.0        # default bankroll for Kelly sizing

# ── Odds API ───────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "baseball_mlb"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"
ODDS_BOOKMAKERS = ["fanduel", "draftkings", "betmgm", "caesars"]
