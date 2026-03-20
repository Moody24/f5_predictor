# F5 Predictor 🎯⚾

**First 5 Innings MLB Prediction System** — XGBoost + Zero-Inflated Negative Binomial ensemble for moneyline, over/under, and run line markets.

## Why F5?

First 5 innings is the sharpest MLB betting market because:
- **Starter-dominated**: eliminates bullpen variance (the biggest noise source in full-game MLB)
- **Fewer variables**: no manager decisions on pinch hitters, double switches, or bullpen sequencing
- **Market inefficiency**: books price F5 as ~5/9 of the full game, but the ratio varies heavily by starter matchup

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  MLB Stats API ──→ Game logs, pitcher stats, team stats     │
│  Statcast ───────→ Pitch-level: whiff%, barrel%, xwOBA      │
│  Odds API ───────→ Market lines & implied probabilities      │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────▼───────┐
       │   FEATURE      │  ~50 features per game:
       │   ENGINEERING   │  Starter quality, Statcast, matchup splits,
       │                │  team offense, park factors, rolling form
       └───────┬───────┘
               │
       ┌───────▼───────────────────────────────────┐
       │              MODEL LAYER                    │
       │                                             │
       │  ┌──────────┐        ┌──────────────────┐  │
       │  │  XGBoost  │        │      ZINB         │  │
       │  │           │        │                    │  │
       │  │ • ML clf  │        │ P(away_runs = k)  │  │
       │  │ • Total   │        │ P(home_runs = k)  │  │
       │  │   reg     │        │                    │  │
       │  │ • Diff    │        │ Zero-inflation:    │  │
       │  │   reg     │        │   shutout prob     │  │
       │  │           │        │ NB count:          │  │
       │  │ Edge      │        │   run distribution │  │
       │  │ detection │        │                    │  │
       │  └─────┬─────┘        └────────┬──────────┘  │
       │        │                       │              │
       │        └───────┬───────────────┘              │
       │                │                              │
       │        ┌───────▼───────┐                      │
       │        │   ENSEMBLE     │                      │
       │        │                │                      │
       │        │ w_zinb * ZINB  │                      │
       │        │ + w_xgb * XGB │                      │
       │        │                │                      │
       │        │ Weights tuned  │                      │
       │        │ on val set     │                      │
       │        └───────┬───────┘                      │
       └────────────────┼──────────────────────────────┘
                        │
       ┌────────────────▼──────────────────────────────┐
       │              OUTPUT LAYER                      │
       │                                                │
       │  Moneyline:  P(home win F5), P(away win F5)   │
       │  Over/Under: P(over k) for k = 2.5 ... 7.5    │
       │  Run Line:   P(home covers -1.5, -0.5, etc)   │
       │  Edges:      Model prob vs market implied      │
       │  Sizing:     Half-Kelly bet sizing             │
       └────────────────────────────────────────────────┘
```

## Why ZINB?

The **Zero-Inflated Negative Binomial** is the right distribution for F5 runs because:

| Property | Why it matters for F5 |
|---|---|
| **Zero inflation** | ~18% of teams score exactly 0 in F5. Regular NB underestimates this. The ZINB's logit component models the probability of a complete shutout. |
| **Overdispersion** | Variance of F5 runs > mean. Multi-run innings (rallies) create fat right tails that Poisson can't capture. NB handles this via the dispersion parameter α. |
| **Covariate-driven** | Both the zero-inflation probability AND the count mean are functions of features. A team facing deGrom in Oracle Park has different P(0 runs) than one facing a rookie in Coors. |
| **Full PMF** | Unlike regression (which gives a point estimate), ZINB gives P(runs=k) for every k, enabling Monte Carlo simulation of exact game outcomes. |

### ZINB Formula

```
P(Y = 0) = π + (1 - π) · NB(0; μ, α)
P(Y = k) = (1 - π) · NB(k; μ, α)    for k ≥ 1

where:
  π = logistic(X_inflate · γ)     ← zero-inflation probability
  μ = exp(X · β)                   ← NB mean (expected runs)
  α = dispersion parameter          ← captures overdispersion
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo> && cd f5_predictor
pip install -r requirements.txt
cp .env.example .env   # add your ODDS_API_KEY

# 2. Fetch 3 seasons of data (~20 min with rate limiting)
python main.py fetch --start-season 2022 --end-season 2024

# 3. Train models
python main.py train

# 4. Predict today's games
python main.py predict

# 5. Backtest performance
python main.py backtest --bankroll 1000 --min-edge 3.0

# Or run the full pipeline
python main.py pipeline --start-season 2022 --end-season 2024
```

## Feature Categories (~50 features)

### Pitcher Quality (F5-specific)
- ERA, WHIP, K-BB% (season + rolling 5/10 game)
- % of starts reaching 5 IP (critical for F5 bet validity)
- Average pitch count per start
- Runs allowed per start through F5

### Statcast / Batted Ball
- Whiff rate, CSW% (called strike + whiff)
- Average exit velocity & barrel rate against
- Hard-hit rate against
- xwOBA against (expected weighted on-base average)
- Fastball velocity, pitch type count

### Matchup / Handedness
- xwOBA vs RHB / LHB splits
- Whiff rate vs RHB / LHB splits
- Lineup handedness composition

### Team Offense
- OPS, wOBA, ISO (isolated power)
- K%, BB%
- Runs per game

### Contextual
- Park factor (Coors 1.28 → Petco 0.92)
- Rest days

### Rolling Form
- F5 runs scored/allowed last 5 and 10 games

## Project Structure

```
f5_predictor/
├── config/
│   └── settings.py          # Central configuration
├── data/
│   ├── fetchers/
│   │   ├── mlb_stats.py     # MLB Stats API (free)
│   │   ├── statcast.py      # Baseball Savant via pybaseball
│   │   └── odds_api.py      # The Odds API
│   └── feature_engineering.py
├── models/
│   ├── zinb_model.py        # Zero-Inflated Negative Binomial
│   ├── xgboost_model.py     # XGBoost classifier + regressors
│   └── combined_predictor.py # Ensemble + edge detection
├── evaluation/
│   └── backtester.py        # Walk-forward backtesting
├── main.py                  # CLI entry point
├── requirements.txt
└── .env.example
```

## Edge Detection & Bet Sizing

The system identifies edges by comparing model probabilities to market-implied probabilities:

```
Edge% = (Model_Prob - Market_Implied_Prob) × 100

If Edge% ≥ 3%: Flag as LEAN
If Edge% ≥ 6%: Flag as MODERATE
If Edge% ≥ 10%: Flag as STRONG
```

Bet sizing uses **half-Kelly criterion** for conservative bankroll management:

```
Kelly% = (b·p - q) / b    where b = decimal_odds - 1
Bet Size = Bankroll × Kelly% × 0.5
Max Bet = 5% of bankroll (hard cap)
```

## API Keys

| Service | Cost | Required? |
|---|---|---|
| MLB Stats API | Free | Yes |
| Statcast (pybaseball) | Free | Optional (enhances accuracy) |
| The Odds API | Free tier: 500 req/mo | Required for live edge detection |

## Next Steps / Roadmap

- [ ] Add weather data (wind speed/direction at game time)
- [ ] Lineup-aware features (actual batting order, not just team averages)
- [ ] Umpire tendencies (zone size affects K rate → run scoring)
- [ ] Live model updating with in-season data
- [ ] Flask/FastAPI web dashboard
- [ ] Integration with existing sports_betting_tracker
