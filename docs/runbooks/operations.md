# F5 Predictor — Operations Runbook

**Stack:** Python 3.11 + XGBoost/ZINB + Railway + Telegram Bot API
**Last verified:** 2026-03-21
**Source configs:** `railway.toml`, `config/settings.py`, `entrypoint.sh`
**Est. daily run time:** ~2 min (after initial pipeline)

---

## Pre-Season Setup (One-Time)

Run once before the season starts. All data cached to persistent volume after.

### Step 1 — Verify Railway deployment is live (2 min)
```bash
railway status
railway logs --lines 20
```
✅ Expected: Service `f5_predictor` in project `vigilant-eagerness`, recent logs showing scheduler output.

### Step 2 — Verify all env vars are set (1 min)
```bash
railway variables
```
✅ Expected: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ODDS_API_KEY`, `ANTHROPIC_API_KEY`, `STORAGE_DIR` all present.

### Step 3 — Verify persistent volume is mounted (1 min)
```bash
railway volume list
```
✅ Expected: `f5_predictor-volume` mounted at `/app/storage`, storage used > 0MB.

---

## Daily Operations

The scheduler runs automatically. This section covers manual intervention only.

### Trigger a manual prediction run
```bash
railway run python scheduler.py --skip-retrain
```
✅ Expected: Telegram message received within 2 minutes.

### Force retrain the model
```bash
railway run python scheduler.py --force-retrain
```
✅ Expected: Logs show `Retraining model...`, completes in ~15 min. Telegram message follows.

### Check yesterday's accuracy
```bash
railway run python -c "
from data.fetchers.mlb_stats import MLBStatsFetcher
from evaluation.accuracy_tracker import check_yesterday_accuracy
print(check_yesterday_accuracy(MLBStatsFetcher()))
"
```
✅ Expected: Dict with `ml_accuracy`, `avg_total_error`, `edge_bet_accuracy`.

---

## Incident Response

### Symptom: No Telegram message received by 11 AM ET

```bash
# Step 1 — Check if scheduler ran
railway logs --lines 50 | grep "SCHEDULER"

# Step 2 — Check for errors
railway logs --lines 100 | grep -i "error\|traceback\|killed"

# Step 3 — Check if predictions file was created
railway run ls storage/predictions/
```

**Decision tree:**
- `Killed` in logs → OOM — Railway may need more memory. Check resource usage in dashboard.
- `ModuleNotFoundError` → Volume wiped the data package. Check volume mount path is `/app/storage` not `/app/data`.
- `No games today` in logs → Off-day, no games scheduled. Normal.
- `Traceback` in logs → Code error. See error and fix → redeploy via `git push`.

---

### Symptom: Pipeline crashed mid-fetch (first boot only)

All fetchers use caching — safe to redeploy. Cached data on volume persists.

```bash
# Verify cached data survived
railway run ls storage/cache/
```
✅ Expected: `mlb_stats/`, `weather/`, `umpires/` directories present.

Redeploy triggers automatic resume from cache:
```bash
git commit --allow-empty -m "Trigger redeploy" && git push
```

---

### Symptom: Telegram bot not responding

```bash
# Test bot token is valid
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe"
```
✅ Expected: `{"ok":true,"result":{"username":"Baseball_F5_bot",...}}`

If token invalid → regenerate via `@BotFather` → `/token` → update Railway:
```bash
railway variables --set "TELEGRAM_BOT_TOKEN=<new_token>"
```

---

### Symptom: Model predictions seem wrong / off

```bash
# Check accuracy log
railway run python -c "
import json
with open('storage/cache/accuracy/daily_accuracy.json') as f:
    log = json.load(f)
for entry in log[-7:]:
    print(entry)
"
```

If ML accuracy drops below 50% for 3+ consecutive days → force retrain with latest data:
```bash
railway run python main.py pipeline --start-season 2021 --end-season 2026
```

---

## Deployment Runbook

### Standard deploy (code change)
```bash
git add .
git commit -m "description of change"
git push  # Railway auto-deploys from main
```
✅ Expected: Railway dashboard shows new deployment within 30 seconds. Build completes in ~3 min.

### Rollback a bad deploy
In Railway dashboard → Deployments → click previous deployment → **Redeploy**.

Or via CLI:
```bash
git revert HEAD && git push
```

### Update environment variables
```bash
railway variables --set "KEY=value"
# Railway auto-restarts the service after env var changes
```

---

## Staleness Check

| Config File | Affects |
|-------------|---------|
| `config/settings.py` | All paths, model params, park factors — review yearly |
| `railway.toml` | Build/deploy config |
| `entrypoint.sh` | Boot logic |
| `requirements.txt` | Dependencies |

Check last modified:
```bash
git log -1 --format="%ci %s" -- config/settings.py railway.toml entrypoint.sh
```

**Park factors** in `config/settings.py` should be updated each April with new season estimates.

---

## Escalation

| Issue | Action |
|-------|--------|
| No predictions for 1 day | Check logs, trigger manual run |
| No predictions for 2+ days | Check Railway service health, redeploy |
| Model accuracy < 45% for 1 week | Force retrain with latest season data |
| Railway service down | Check railway.app status page |
