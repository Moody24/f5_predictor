"""
WhatsApp Notification Sender
------------------------------
Sends daily F5 predictions via Twilio WhatsApp API.

Requires in .env:
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_WHATSAPP_FROM  (e.g., whatsapp:+14155238886)
  WHATSAPP_TO           (e.g., whatsapp:+1XXXXXXXXXX)
"""
import os
import json
import logging
from datetime import datetime

from config.settings import PREDICTIONS_DIR
from notifications.claude_analyzer import analyze_today

logger = logging.getLogger(__name__)

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
WHATSAPP_TO = os.getenv("WHATSAPP_TO", "")


def send_whatsapp(message: str) -> bool:
    """
    Send a WhatsApp message via Twilio.

    Returns True on success, False on failure.
    """
    if not all([TWILIO_SID, TWILIO_TOKEN, WHATSAPP_TO]):
        logger.warning("Twilio credentials not configured — skipping WhatsApp")
        return False

    try:
        from twilio.rest import Client
    except ImportError:
        logger.warning("twilio package not installed. Run: pip install twilio")
        return False

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)

        # WhatsApp has a 1600 char limit per message
        if len(message) > 1500:
            message = message[:1497] + "..."

        msg = client.messages.create(
            from_=TWILIO_FROM,
            body=message,
            to=WHATSAPP_TO,
        )
        logger.info(f"WhatsApp sent. SID: {msg.sid}")
        return True

    except Exception as e:
        logger.error(f"WhatsApp send failed: {e}")
        return False


def send_daily_predictions():
    """
    Main entry point: analyze today's predictions and send via WhatsApp.
    Called by scheduler.py.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    pred_path = PREDICTIONS_DIR / f"{today}.json"

    if not pred_path.exists():
        logger.info("No predictions to send")
        return

    # Get Claude analysis (or fallback summary)
    analysis = analyze_today()

    # Send
    success = send_whatsapp(analysis)
    if success:
        logger.info("Daily predictions sent to WhatsApp")
    else:
        # Print to console as fallback
        print("\n" + "=" * 50)
        print("DAILY F5 ANALYSIS")
        print("=" * 50)
        print(analysis)
        print("=" * 50)
