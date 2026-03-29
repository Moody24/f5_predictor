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
├── bot_runner.py                    # Railway entry point: Telegram bot + scheduler loop
├── scheduler.py                     # Daily pipeline orchestrator
├── config/
│   └── settings.py                  # All paths, constants, team maps, model versioning
├── data/
│   ├── feature_engineering.py       # ~74 features across 8 categories
│   └── fetchers/
│       ├── mlb_stats.py             # MLB Stats API (schedules, pitchers, teams)
│       ├── statcast.py              # Pitch-level data via pybaseball
│       ├── odds_api.py              # The Odds API (betting lines)
│       ├── weather.py               # Open-Meteo (temperature, wind, humidity)
│       ├── umpire.py                # HP umpire tendencies from MLB API
│       └── lineups.py               # Confirmed lineups + batter stats + 14-day rolling wOBA
├── models/
│   ├── zinb_model.py                # ZINB distribution model (run probabilities)
│   ├── xgboost_model.py             # XGBoost ensemble (classifier + 2 regressors)
│   ├── combined_predictor.py        # Weighted ensemble (55% ZINB / 45% XGBoost)
│   └── saved/                       # Timestamped model versions (last 5 kept)
├── evaluation/
│   ├── backtester.py                # Walk-forward backtest with calendar-day windows
│   └── accuracy_tracker.py          # Daily prediction vs outcome tracking (7-day backfill)
└── notifications/
    ├── claude_analyzer.py           # Claude Haiku analysis of daily predictions
    ├── telegram.py                  # Telegram message sender
    └── telegram_bot.py              # Interactive Telegram bot (polling)
                                     # (whatsapp.py removed — was never wired up)
```

## Models

**ZINB** — Models F5 run counts as a zero-inflated negative binomial distribution. Outputs full probability distribution over possible scores (0-15 runs per team). Uses Monte Carlo simulation (10k draws) to derive win/loss/draw probabilities.

**XGBoost** — Three sub-models: binary classifier (home win), total runs regressor, run differential regressor. Features include pitcher stats, team offense, park factors, weather, umpire tendencies, travel fatigue, bullpen workload, and lineup strength.

**Combined** — `P_final = 0.55 * P_zinb + 0.45 * P_xgb`. Edge detection compares model probability to market implied probability (minimum 3% edge). Bet sizing uses half-Kelly criterion with 5% bankroll cap.

## Feature Categories (~74 features)

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
| Lineup | 6 | MLB Stats API + pybaseball (includes 14-day rolling wOBA) |

Lineup features: `lineup_avg_woba`, `lineup_avg_ops`, `lineup_total_iso`, `lineup_platoon_pct`, `lineup_recent_woba` (14-day), `lineup_hot_pct` (% of starters running hot vs season baseline).

## Data Sources & API Keys

| Source | Key Required | Config |
|--------|-------------|--------|
| MLB Stats API | No | Free, no auth |
| Statcast/pybaseball | No | Free, no auth |
| Open-Meteo | No | Free, no auth |
| The Odds API | `ODDS_API_KEY` | Starter tier ($20/mo) — historical odds endpoint enabled for CLV |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Optional, ~$0.01/day |

## Data Storage

- `data/cache/` — All fetched raw data (MLB stats, Statcast, weather, umpires, lineups)
- `data/cache/lineups/recent_form/` — Per-batter 14-day wOBA cache (refreshed daily)
- `data/cache/feature_matrix.parquet` — Final training matrix
- `data/predictions/YYYY-MM-DD.json` — Daily prediction output (written atomically)
- `data/accuracy/daily_accuracy.json` — Running accuracy log (7-day backfill on each run)
- `data/bot_state.json` — Persisted Telegram bot state (survives container restarts)
- `models/saved/YYYY-MM-DD_HHMMSS/` — Versioned model files (ZINB + XGBoost + config + imputation medians)

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/predict` | Full today's predictions sent immediately |
| `/edges` | Edge bets only, with countdown timers and locked-game separation |
| `/record` | W-L record, ROI, and per-confidence breakdown (STRONG / MODERATE / LEAN) |
| `/status` | Last run time, next scheduled run, games and edges count |
| `/ask <question>` | Ask Claude about today's games — maintains 6-message conversation history, resets after 30 min idle |
| `/help` | List all commands |

## Scheduler

`scheduler.py` runs the daily pipeline: accuracy check → incremental fetch → conditional retrain (every 50 new games) → predict → notify. Hosted on Railway via `bot_runner.py`.

```bash
# Cron setup (10 AM ET daily)
0 10 * * * cd /path/to/f5_predictor && python scheduler.py >> data/logs/scheduler.log 2>&1
```

## Key Design Decisions

- **Skip 2020 season** — 60-game COVID season with fundamentally different conditions
- **Dome stadiums** get neutral weather values (no wind/temperature effects)
- **Umpire tendencies** require minimum 11 games sample (LOO and inference thresholds unified)
- **Lineup fallback** — uses league-average defaults if lineups not yet posted
- **Imputation medians** — saved alongside each model version; inference applies same fills as training
- **Atomic file writes** — predictions JSON and bot state use tmp→rename to prevent partial reads
- **Model versioning** — keeps last 5 versions, auto-prunes older ones
- **Kelly criterion** — single implementation in `config/settings.kelly_criterion()`; `KELLY_FRACTION = 0.5` (half-Kelly); backtester and combined predictor both import from there
- **Pitcher defaults** — defined once in `_default_pitcher_features()`; `_build_pitcher_features()` starts from that dict and overrides only present fields
- **Backtester baseline** — uses 0.525 home-win rate (historical F5 average) instead of 0.50
- **Backtester windows** — calendar-day based (not row-count) so doubleheaders don't distort window sizes
- **Accuracy backfill** — checks last 7 days on each run, auto-fills gaps from downtime
- **CLV tracking** — requires paid Odds API tier; measures whether model edge aligned with sharp money movement
