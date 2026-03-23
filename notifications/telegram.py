"""
Telegram Notification Sender
------------------------------
Sends daily F5 predictions via Telegram Bot API.

Requires in .env:
  TELEGRAM_BOT_TOKEN   (from @BotFather)
  TELEGRAM_CHAT_ID     (your chat ID)
"""
import os
import logging
import requests

from notifications.claude_analyzer import analyze_today

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram(message: str) -> bool:
    """
    Send a message via Telegram Bot API.
    Splits messages over 4096 chars (Telegram limit).

    Returns True on success, False on failure.
    """
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.warning("Telegram credentials not configured — skipping notification")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Telegram max message length is 4096 chars
    chunks = [message[i:i+4096] for i in range(0, len(message), 4096)]

    success = True
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
            }, timeout=10)
            resp.raise_for_status()
            logger.info("Telegram message sent successfully")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            success = False

    return success


def _load_yesterday_accuracy() -> dict:
    """Load the most recent entry from the daily accuracy log."""
    import json
    from config.settings import DATA_DIR
    log_path = DATA_DIR / "accuracy" / "daily_accuracy.json"
    if not log_path.exists():
        return {}
    try:
        with open(log_path) as f:
            log = json.load(f)
        if log:
            return log[-1]  # most recent entry
    except Exception as e:
        logger.warning(f"Could not load accuracy log: {e}")
    return {}


def send_daily_predictions():
    """
    Main entry point: analyze today's predictions and send via Telegram.
    Called by scheduler.py.
    """
    from utils import predictions_path
    pred_path = predictions_path()

    if not pred_path.exists():
        logger.info("No predictions to send")
        return

    accuracy_context = _load_yesterday_accuracy()
    analysis = analyze_today(accuracy_context=accuracy_context or None)

    success = send_telegram(analysis)
    if success:
        logger.info("Daily predictions sent to Telegram")
    else:
        print("\n" + "=" * 50)
        print("DAILY F5 ANALYSIS")
        print("=" * 50)
        print(analysis)
        print("=" * 50)
