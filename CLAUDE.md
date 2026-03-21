# F5 Predictor

MLB First-5-Innings betting predictor. Combines a Zero-Inflated Negative Binomial (ZINB) model with an XGBoost ensemble to predict F5 outcomes: moneyline, totals, and run lines.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env

# Full pipeline: fetch 4 seasons → build features → train → predict
python main.py pipeline --start-season 2021 --end-season 2024

# Individual commands
python main.py fetch --start-season 2021 --end-season 2024
python main.py train
python main.py predict
python main.py backtest

# Daily automated pipeline (after initial setup)
python scheduler.py
```

## Architecture

```
f5_predictor/
├── main.py                          # CLI: fetch, train, predict, backtest, pipeline
├── scheduler.py                     # Daily cron orchestrator
├── config/
│   └── settings.py                  # All paths, constants, team maps, model versioning
├── data/
│   ├── feature_engineering.py       # ~72 features across 8 categories
│   └── fetchers/
│       ├── mlb_stats.py             # MLB Stats API (schedules, pitchers, teams)
│       ├── statcast.py              # Pitch-level data via pybaseball
│       ├── odds_api.py              # The Odds API (betting lines)
│       ├── weather.py               # Open-Meteo (temperature, wind, humidity)
│       ├── umpire.py                # HP umpire tendencies from MLB API
│       └── lineups.py               # Confirmed lineups + batter stats
├── models/
│   ├── zinb_model.py                # ZINB distribution model (run probabilities)
│   ├── xgboost_model.py             # XGBoost ensemble (classifier + 2 regressors)
│   ├── combined_predictor.py        # Weighted ensemble (55% ZINB / 45% XGBoost)
│   └── saved/                       # Timestamped model versions (last 5 kept)
├── evaluation/
│   ├── backtester.py                # Walk-forward backtest with Kelly sizing
│   └── accuracy_tracker.py          # Daily prediction vs outcome tracking
└── notifications/
    ├── claude_analyzer.py           # Claude Haiku analysis of daily predictions
    └── whatsapp.py                  # Twilio WhatsApp delivery
```

## Models

**ZINB** — Models F5 run counts as a zero-inflated negative binomial distribution. Outputs full probability distribution over possible scores (0-15 runs per team). Uses Monte Carlo simulation (10k draws) to derive win/loss/draw probabilities.

**XGBoost** — Three sub-models: binary classifier (home win), total runs regressor, run differential regressor. Features include pitcher stats, team offense, park factors, weather, umpire tendencies, travel fatigue, bullpen workload, and lineup strength.

**Combined** — `P_final = 0.55 * P_zinb + 0.45 * P_xgb`. Edge detection compares model probability to market implied probability (minimum 3% edge). Bet sizing uses half-Kelly criterion with 5% bankroll cap.

## Feature Categories (~72 features)

| Category | Count | Source |
|----------|-------|--------|
| Pitcher | ~20 | MLB Stats API, Statcast |
| Team offense | ~10 | MLB Stats API |
| Park factors | ~5 | Statcast |
| Rolling (last 20 games) | ~8 | Computed from game history |
| Weather | 5 | Open-Meteo |
| Umpire | 2 | MLB Stats API |
| Travel/Fatigue | 4 | Computed from schedule + venue coords |
| Bullpen | 3 | MLB Stats API |
| Lineup | 4 | MLB Stats API + pybaseball |

## Data Sources & API Keys

| Source | Key Required | Config |
|--------|-------------|--------|
| MLB Stats API | No | Free, no auth |
| Statcast/pybaseball | No | Free, no auth |
| Open-Meteo | No | Free, no auth |
| The Odds API | `ODDS_API_KEY` | 500 req/mo free tier |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Optional, ~$0.01/day |
| Twilio (WhatsApp) | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Optional |

## Data Storage

- `data/cache/` — All fetched raw data (MLB stats, Statcast, weather, umpires, lineups)
- `data/cache/feature_matrix.parquet` — Final training matrix
- `data/predictions/YYYY-MM-DD.json` — Daily prediction output
- `data/accuracy/daily_accuracy.json` — Running accuracy log
- `models/saved/YYYY-MM-DD_HHMMSS/` — Versioned model files (ZINB + XGBoost + config)

## Scheduler

`scheduler.py` runs the daily pipeline: accuracy check → incremental fetch → conditional retrain (every 50 new games) → predict → notify.

```bash
# Cron setup (10 AM ET daily)
0 10 * * * cd /path/to/f5_predictor && python scheduler.py >> data/logs/scheduler.log 2>&1
```

## Key Design Decisions

- **Skip 2020 season** — 60-game COVID season with fundamentally different conditions
- **Dome stadiums** get neutral weather values (no wind/temperature effects)
- **Umpire tendencies** require minimum 10 games sample
- **Lineup fallback** — uses team averages if lineups not yet posted
- **Model versioning** — keeps last 5 versions, auto-prunes older ones
- **Backtester baseline** — uses 0.525 home-win rate (historical F5 average) instead of 0.50
