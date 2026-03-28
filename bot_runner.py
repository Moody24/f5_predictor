"""
Bot Runner — entry point for Railway
--------------------------------------
Starts the Telegram polling bot in a background thread,
then runs the scheduler loop every 22 hours.
"""
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import DATA_DIR

LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / f"bot_runner_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bot_runner")

RUN_HOUR_UTC = 13  # 9 AM ET / 1 PM UTC — before first MLB games start


def _seconds_until_next_run() -> float:
    """Seconds until the next 13:00 UTC."""
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


def main():
    # Start Telegram bot in background
    from notifications.telegram_bot import start_bot_thread, update_state
    start_bot_thread()
    logger.info("Telegram bot thread started")

    import scheduler as sched_module
    import json

    while True:
        sleep_secs = _seconds_until_next_run()
        next_run_dt = datetime.now(timezone.utc) + timedelta(seconds=sleep_secs)
        logger.info(f"Next scheduler run at {next_run_dt.strftime('%Y-%m-%d %H:%M UTC')} ({sleep_secs/3600:.1f}h away)")

        # Update bot state with next run time before sleeping
        try:
            update_state(
                last_run=datetime.now(timezone.utc),
                next_run=next_run_dt.replace(tzinfo=None),
                games=0,
                edges=0,
            )
        except Exception:
            pass

        time.sleep(sleep_secs)

        run_start = datetime.now(timezone.utc)
        try:
            sched_module.main()
        except Exception as e:
            logger.error(f"Scheduler run failed: {e}")

        # Update bot state after run
        try:
            from utils import predictions_path
            pred_path = predictions_path()
            games, edges = 0, 0
            if pred_path.exists():
                with open(pred_path) as f:
                    preds = json.load(f)
                games = preds.get("n_games", 0)
                edges = sum(len(g.get("edges", [])) for g in preds.get("games", []))
            update_state(
                last_run=run_start.replace(tzinfo=None),
                next_run=(run_start + timedelta(days=1)).replace(
                    hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0, tzinfo=None
                ),
                games=games,
                edges=edges,
            )
        except Exception as e:
            logger.warning(f"Could not update bot state: {e}")


if __name__ == "__main__":
    main()
