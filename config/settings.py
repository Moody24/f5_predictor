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
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MAX_MODEL_VERSIONS = 5


def get_latest_model_dir() -> Path:
    """Get the most recent versioned model directory."""
    versions = sorted(MODEL_DIR.glob("20*"), reverse=True)
    if versions:
        return versions[0]
    return MODEL_DIR  # fallback to flat directory for backward compat


def create_model_version_dir() -> Path:
    """Create a new timestamped model directory and prune old versions."""
    from datetime import datetime as _dt
    version_dir = MODEL_DIR / _dt.now().strftime("%Y-%m-%d_%H%M%S")
    version_dir.mkdir(parents=True, exist_ok=True)
    # Prune old versions
    versions = sorted(MODEL_DIR.glob("20*"), reverse=True)
    for old in versions[MAX_MODEL_VERSIONS:]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)
    return version_dir


def list_model_versions() -> list[Path]:
    """List all saved model versions, newest first."""
    return sorted(MODEL_DIR.glob("20*"), reverse=True)

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
F5_RATIO = 5 / 9  # Historical ratio of F5 runs to full-game runs

# ── Team Name Mapping (Odds API → MLB Stats API) ─────────────────────
# The Odds API and MLB Stats API use slightly different team name formats.
ODDS_TO_MLB_TEAM_MAP = {
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Oakland Athletics": "Oakland Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",
}
# Reverse map for MLB → Odds API lookups
MLB_TO_ODDS_TEAM_MAP = {v: k for k, v in ODDS_TO_MLB_TEAM_MAP.items()}
