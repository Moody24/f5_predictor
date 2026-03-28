"""
Interactive Telegram Bot (polling)
------------------------------------
Runs in a background daemon thread alongside the scheduler.
Polls Telegram for incoming commands and responds.

Commands:
  /predict  — send today's full predictions
  /edges    — show only edge bets (or "no edges today")
  /status   — last run time, next run, games count
  /ask <q>  — ask Claude anything about today's games
  /help     — list commands
"""
import json
import logging
import os
import threading
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_TIMEOUT = 30  # long-poll seconds — reduces idle API calls


def _api(method: str, **kwargs) -> dict:
    """Call a Telegram Bot API method."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    resp = requests.post(url, json=kwargs, timeout=POLL_TIMEOUT + 5)
    resp.raise_for_status()
    return resp.json()


def _send(text: str, parse_mode: str = "Markdown") -> None:
    """Send a message to the configured chat."""
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)]
    for chunk in chunks:
        try:
            _api("sendMessage", chat_id=TELEGRAM_CHAT_ID, text=chunk, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")


# ── State shared with the scheduler loop ────────────────────────────────────

_state = {
    "last_run": None,       # datetime of last scheduler run
    "next_run": None,       # datetime of next scheduler run
    "games_today": 0,
    "edges_today": 0,
}
_state_lock = threading.Lock()


def update_state(last_run: datetime, next_run: datetime, games: int, edges: int) -> None:
    """Called by the scheduler after each run to keep bot state current."""
    with _state_lock:
        _state["last_run"] = last_run
        _state["next_run"] = next_run
        _state["games_today"] = games
        _state["edges_today"] = edges


def _get_state() -> dict:
    """Thread-safe snapshot of current bot state."""
    with _state_lock:
        return dict(_state)


# ── Command handlers ─────────────────────────────────────────────────────────

def _handle_predict() -> None:
    from notifications.telegram import send_daily_predictions
    send_daily_predictions()


def _handle_edges() -> None:
    from utils import predictions_path
    pred_path = predictions_path()
    if not pred_path.exists():
        _send("No predictions for today yet. Try again after the scheduler runs.")
        return

    with open(pred_path) as f:
        preds = json.load(f)

    all_edges = []
    for game in preds.get("games", []):
        info = game["game_info"]
        matchup = f"{info['away_team']} @ {info['home_team']}"
        for e in game.get("edges", []):
            all_edges.append((matchup, e))

    if not all_edges:
        _send("No actionable edges found today. 0 bets flagged.")
        return

    lines = [f"*F5 Edges — {preds.get('date', 'Today')}*", ""]
    for matchup, e in sorted(all_edges, key=lambda x: x[1].get("edge_pct", 0), reverse=True):
        kelly = e.get("kelly_half", 0) * 100
        lines.append(
            f"[{e['confidence']}] *{matchup}*\n"
            f"  {e['market']} → {e['side']} | Edge: +{e['edge_pct']:.1f}% | Kelly: {kelly:.1f}%"
        )

    _send("\n".join(lines))


def _handle_status() -> None:
    s = _get_state()
    last = s["last_run"]
    nxt = s["next_run"]
    lines = [
        "*F5 Predictor — Status*",
        f"Last run: {last.strftime('%Y-%m-%d %H:%M UTC') if last else 'unknown'}",
        f"Next run: {nxt.strftime('%Y-%m-%d %H:%M UTC') if nxt else 'unknown'}",
        f"Games today: {s['games_today']}",
        f"Edges today: {s['edges_today']}",
    ]
    _send("\n".join(lines))


def _handle_ask(question: str) -> None:
    from utils import predictions_path
    pred_path = predictions_path()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        _send("ANTHROPIC_API_KEY not set — can't answer questions.")
        return

    try:
        import anthropic
    except ImportError:
        _send("anthropic package not installed.")
        return

    context = "No predictions available for today yet."
    if pred_path.exists():
        with open(pred_path) as f:
            preds = json.load(f)
        games = preds.get("games", [])
        summary_lines = []
        for g in games:
            info = g["game_info"]
            ml = g["moneyline"]
            total = g.get("total", {})
            edges = g.get("edges", [])
            edge_str = ", ".join(
                f"{e['market']} {e['side']} +{e['edge_pct']:.1f}%"
                for e in edges
            ) or "no edges"
            summary_lines.append(
                f"{info['away_team']} @ {info['home_team']} "
                f"({info.get('away_starter','?')} vs {info.get('home_starter','?')}): "
                f"home {float(ml['home_prob'])*100:.0f}%, total {total.get('predicted','?')} runs, {edge_str}"
            )
        context = f"Date: {preds.get('date')}\n" + "\n".join(summary_lines)

    prompt = (
        f"You are an MLB F5 betting analyst. Answer the user's question using today's predictions.\n\n"
        f"Today's predictions:\n{context}\n\n"
        f"User question: {question}\n\n"
        f"Answer concisely in 2-4 sentences."
    )

    client = anthropic.Anthropic(api_key=anthropic_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    _send(msg.content[0].text)


def _handle_help() -> None:
    _send(
        "*F5 Predictor Bot*\n\n"
        "/predict — full today's predictions\n"
        "/edges — edge bets only\n"
        "/status — last/next run times\n"
        "/ask <question> — ask Claude about today's games\n"
        "/help — show this message"
    )


# ── Polling loop ─────────────────────────────────────────────────────────────

def _poll_loop() -> None:
    """Long-poll Telegram for updates and dispatch commands. Runs forever."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing — bot polling disabled")
        return

    logger.info("Telegram bot polling started")
    offset = 0

    while True:
        try:
            data = _api("getUpdates", offset=offset, timeout=POLL_TIMEOUT)
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue

                # Only respond to the configured chat
                if str(msg.get("chat", {}).get("id", "")) != str(TELEGRAM_CHAT_ID):
                    continue

                text = (msg.get("text") or "").strip()
                if not text.startswith("/"):
                    continue

                parts = text.split(None, 1)
                command = parts[0].lower().split("@")[0]  # strip @botname suffix
                arg = parts[1] if len(parts) > 1 else ""

                logger.info(f"Bot command received: {command} {arg!r}")

                if command == "/predict":
                    _handle_predict()
                elif command == "/edges":
                    _handle_edges()
                elif command == "/status":
                    _handle_status()
                elif command == "/ask":
                    if arg:
                        _handle_ask(arg)
                    else:
                        _send("Usage: /ask <your question>")
                elif command == "/help":
                    _handle_help()
                else:
                    _send(f"Unknown command: {command}\nType /help for available commands.")

        except requests.exceptions.Timeout:
            pass  # expected from long-polling, just loop again
        except Exception as e:
            logger.error(f"Bot poll error: {e}")
            time.sleep(5)  # back off briefly on unexpected errors


def start_bot_thread() -> threading.Thread:
    """Start the polling loop in a background daemon thread."""
    t = threading.Thread(target=_poll_loop, name="telegram-bot", daemon=True)
    t.start()
    return t
