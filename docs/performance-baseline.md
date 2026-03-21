# F5 Predictor — Performance Baseline

## Initial Pipeline (2021–2025, 12,863 games)

Measured on Railway (us-east4) during first boot, 2026-03-21.

| Stage | Duration | Notes |
|-------|----------|-------|
| Games fetch (5 seasons) | ~37 sec | MLB Stats API, 0.25s rate limit |
| Pitcher stats (928 pitchers) | ~10 min | MLB Stats API, not cached to disk |
| Team stats (40 teams) | ~11 sec | MLB Stats API |
| Weather fetch (12,863 games) | ~20 min | Open-Meteo, cached after first run |
| Umpire fetch (12,863 games) | ~90 min | MLB Stats API, 1 call/game, cached after first run |
| Feature engineering | ~5 sec | Vectorized pandas |
| Travel features | <10 sec | numpy array approach after O(n²) fix |
| Rolling features | ~2 sec | pandas groupby+shift |
| Model training (ZINB + XGBoost) | ~15 min | CPU-bound |
| Prediction (today's games) | ~30 sec | Inference only |
| **Total (first boot)** | **~2.5 hours** | Mostly umpire + weather fetches |
| **Total (subsequent boots)** | **~25 min** | Pitcher/team stats + training only |
| **Daily scheduler run** | **~3 min** | Incremental only, skip retrain |

---

## Performance Fix: O(n²) → O(n) Travel Features

**Date:** 2026-03-21
**File:** `data/feature_engineering.py` — `add_travel_features()`

### Problem
`games_7d` computation scanned the entire historical date list for each row:
```python
# O(n²) — scans all prior dates for every game
games_7d[i] = sum(1 for d in team_dates[tid] if (gdate - d).days <= 7)
```
With 12,863 rows × ~400 games per team history = **~5M+ comparisons**.

Also used `df.at[idx, col]` for writes inside a loop — pandas index lookup overhead on every cell.

### Root Cause
Row-by-row DataFrame mutation with `df.at[]` and linear list scanning — both O(n) per row.

### Fix Applied
- Replaced `team_dates` list with a `deque` that prunes entries older than 7 days
- Count becomes `len(dq)` — O(1) per row, O(1) amortized pruning
- Output stored in pre-allocated numpy arrays, assigned once per column at end

```python
# O(1) amortized — deque prunes old dates, count is len()
dq = team_recent[tid]
while dq and (gdate - dq[0]).days > 7:
    dq.popleft()
games_7d[i] = len(dq)
dq.append(gdate)
```

### Result

| Metric | Before | After |
|--------|--------|-------|
| Algorithm | O(n²) | O(n) amortized |
| Estimated ops (12,863 games) | ~5M comparisons | ~90k deque ops |
| Observed behavior | Froze / OOM killed | Completes in <10 sec |
| Memory | OOM on Railway | Minimal (deque max ~7 entries/team) |

---

## Caching Strategy

| Data | Cached | Cache Key | Invalidation |
|------|--------|-----------|-------------|
| Game schedule | ✅ parquet | Season list e.g. `games_2021_2022_2023_2024_2025.parquet` | New season range |
| Weather | ✅ parquet | `game_weather.parquet` (incremental) | Never (historical) |
| Umpires | ✅ parquet | `umpire_assignments.parquet` (incremental) | Never (historical) |
| Pitcher stats | ❌ in-memory | N/A | Every boot |
| Team stats | ❌ in-memory | N/A | Every boot |
| Feature matrix | ✅ parquet | `feature_matrix.parquet` | On retrain |
| Models | ✅ directory | `YYYY-MM-DD_HHMMSS/` (last 5 kept) | Auto-pruned |

**Next optimization:** Cache pitcher and team stats to disk to save ~11 min per boot.
