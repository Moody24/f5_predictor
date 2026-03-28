"""
Bot Runner — entry point for Railway
--------------------------------------
Starts the Telegram polling bot in a background thread,
then runs the scheduler loop every 22 hours.
"""
import logging
import sys
import time
from datetime import datetime, timedelta
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

SLEEP_BETWEEN_RUNS = 22 * 3600  # 22 hours


def main():
    # Start Telegram bot in background
    from notifications.telegram_bot import start_bot_thread, update_state
    start_bot_thread()
    logger.info("Telegram bot thread started")

    import scheduler as sched_module
    import json

    while True:
        run_start = datetime.utcnow()
        next_run = run_start + timedelta(seconds=SLEEP_BETWEEN_RUNS)

        try:
            sched_module.main()
        except Exception as e:
            logger.error(f"Scheduler run failed: {e}")

        # Update bot state so /status reflects current info
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
                last_run=datetime.utcnow(),
                next_run=next_run,
                games=games,
                edges=edges,
            )
        except Exception as e:
            logger.warning(f"Could not update bot state: {e}")

        logger.info(f"Sleeping {SLEEP_BETWEEN_RUNS // 3600}h until next run...")
        time.sleep(SLEEP_BETWEEN_RUNS)


if __name__ == "__main__":
    main()
