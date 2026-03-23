"""
Claude AI Prediction Analyzer
-------------------------------
Sends daily predictions to Claude API for natural language analysis.
Uses Haiku for cost efficiency (~$0.01/day).

Requires: ANTHROPIC_API_KEY in .env
"""
import json
import os
import logging
from datetime import datetime
from pathlib import Path

from config.settings import PREDICTIONS_DIR

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def analyze_predictions(predictions_json: dict, accuracy_context: dict = None) -> str:
    """
    Send predictions to Claude for natural language analysis.

    Args:
        predictions_json: Full predictions dict from cmd_predict output
        accuracy_context: Optional dict with yesterday's accuracy metrics
                          (ml_accuracy, avg_total_error, avg_clv, edge_bet_accuracy)

    Returns:
        Natural language analysis string
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — skipping Claude analysis")
        return _fallback_summary(predictions_json, accuracy_context)

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed — using fallback summary")
        return _fallback_summary(predictions_json, accuracy_context)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Accuracy context block ─────────────────────────────────────────
    accuracy_section = ""
    if accuracy_context:
        ml_acc = accuracy_context.get("ml_accuracy", "?")
        total_err = accuracy_context.get("avg_total_error", "?")
        edge_acc = accuracy_context.get("edge_bet_accuracy")
        clv = accuracy_context.get("avg_clv")

        acc_parts = [f"ML hit rate: {ml_acc}%", f"avg total error: {total_err} runs"]
        if edge_acc is not None:
            acc_parts.append(f"edge bet accuracy: {edge_acc}%")
        if clv is not None:
            acc_parts.append(f"avg CLV: {clv:+.2f}%")
        accuracy_section = f"Yesterday's model performance: {' | '.join(acc_parts)}\n\n"

    # ── Build per-game summary with divergence signal ──────────────────
    games_summary = []
    for game in predictions_json.get("games", []):
        info = game["game_info"]
        ml = game["moneyline"]
        total = game.get("total", {})
        edges = game.get("edges", [])

        # Model divergence: if ZINB and XGBoost disagree by >8% on home win prob,
        # the game is uncertain — flag it so Claude can warn users.
        zinb_home = ml.get("zinb_home", ml["home_prob"])
        xgb_home = ml.get("xgb_home", ml["home_prob"])
        divergence = abs(zinb_home - xgb_home)
        divergence_flag = " [MODELS SPLIT]" if divergence > 0.08 else ""

        entry = (
            f"{info['away_team']} @ {info['home_team']} "
            f"({info.get('away_starter', '?')} vs {info.get('home_starter', '?')}){divergence_flag}\n"
            f"  ML: Home {ml['home_prob']*100:.0f}% "
            f"[ZINB {zinb_home*100:.0f}% / XGB {xgb_home*100:.0f}%] | "
            f"Away {ml['away_prob']*100:.0f}%\n"
            f"  Total: {total.get('predicted', '?')} runs\n"
        )
        if edges:
            for e in edges:
                entry += (
                    f"  EDGE: [{e['confidence']}] {e['market']} {e['side']} "
                    f"+{e['edge_pct']:.1f}% (half-Kelly {e.get('kelly_half', 0)*100:.1f}%)\n"
                )
        games_summary.append(entry)

    prompt = f"""You are an expert MLB F5 (first 5 innings) betting analyst. Analyze today's predictions and give a sharp daily briefing.

Date: {predictions_json.get('date', 'today')}
Games: {predictions_json.get('n_games', 0)}

{accuracy_section}Predictions (MODELS SPLIT = ZINB and XGBoost disagree by >8%, treat as uncertain):
{chr(10).join(games_summary)}

Provide:
1. Top 3 plays — prioritize games where ZINB and XGBoost AGREE (no [MODELS SPLIT] flag) and edge confidence is STRONG or MODERATE. 1-2 sentences of reasoning each.
2. Games to avoid — specifically call out any [MODELS SPLIT] games or thin edges.
3. Market read — pitcher-heavy day? run environment? notable matchups?
4. Best 2-leg parlay — pick legs from DIFFERENT games with cross-model agreement. Show combined probability estimate.

Keep it under 450 words. Be direct and specific — this goes straight to a bettor's phone."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def analyze_today(accuracy_context: dict = None) -> str:
    """Load today's predictions and analyze them."""
    from utils import predictions_path
    pred_path = predictions_path()

    if not pred_path.exists():
        return "No predictions found for today."

    with open(pred_path) as f:
        predictions = json.load(f)

    return analyze_predictions(predictions, accuracy_context=accuracy_context)


def _fallback_summary(predictions_json: dict) -> str:
    """Generate a simple summary without Claude API."""
    games = predictions_json.get("games", [])
    if not games:
        return "No games to analyze."

    # Find strongest edges
    all_edges = []
    for game in games:
        for edge in game.get("edges", []):
            edge["game"] = f"{game['game_info']['away_team']} @ {game['game_info']['home_team']}"
            all_edges.append(edge)

    all_edges.sort(key=lambda x: x.get("edge_pct", 0), reverse=True)

    lines = [
        f"F5 Predictions — {predictions_json.get('date', 'Today')}",
        f"Games analyzed: {len(games)}",
        f"Edges found: {len(all_edges)}",
        "",
    ]

    if all_edges:
        lines.append("Top Edges:")
        for i, e in enumerate(all_edges[:5]):
            lines.append(
                f"  {i+1}. {e['game']} — {e['market']} {e['side']} "
                f"({e['confidence']}, +{e['edge_pct']:.1f}%)"
            )

        # Suggest 2-leg parlay from top 2 edges in different games
        parlay_legs = []
        seen_games = set()
        for e in all_edges:
            if e["game"] not in seen_games:
                parlay_legs.append(e)
                seen_games.add(e["game"])
            if len(parlay_legs) == 2:
                break

        if len(parlay_legs) == 2:
            lines.append("")
            lines.append("Suggested 2-Leg Parlay:")
            for leg in parlay_legs:
                lines.append(f"  - {leg['game']} — {leg['market']} {leg['side']}")
        else:
            lines.append("  (Not enough edges from different games for a parlay)")
    else:
        lines.append("No actionable edges found today.")

    return "\n".join(lines)
