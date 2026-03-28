"""
Backtesting Framework
----------------------
Walk-forward backtesting for the F5 predictor.

Uses expanding window training:
  - Train on all data up to date T
  - Predict on date T+1 ... T+7
  - Roll forward and retrain periodically
  - Track simulated P&L with realistic vig assumptions
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from config.settings import MIN_EDGE_PCT, MIN_KELLY_FRACTION, BANKROLL

logger = logging.getLogger(__name__)


class F5Backtester:
    """Walk-forward backtesting with simulated betting P&L."""

    def __init__(
        self,
        initial_bankroll: float = BANKROLL,
        min_edge: float = MIN_EDGE_PCT,
        kelly_fraction: float = MIN_KELLY_FRACTION,
        retrain_every_n_days: int = 14,
    ):
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.retrain_days = retrain_every_n_days
        self.bet_log = []
        self.daily_bankroll = []

    def run(
        self,
        full_data: pd.DataFrame,
        feature_cols: list[str],
        target_cols: dict,
        predictor_class,
        train_start_idx: int = 500,
        test_window: int = 7,
    ) -> dict:
        """
        Execute walk-forward backtest.

        Args:
            full_data: Complete dataset sorted by date
            feature_cols: List of feature column names
            target_cols: Dict mapping target names to column names
                         e.g. {"away_f5_runs": "away_f5_runs", ...}
            predictor_class: CombinedF5Predictor class
            train_start_idx: Minimum training set size
            test_window: Days per test window

        Returns:
            Dict with backtest results, metrics, and bet log
        """
        dates = full_data["date"].unique()
        dates.sort()

        current_model = None
        last_train_date = None

        for i in range(train_start_idx, len(full_data), test_window):
            window_end = min(i + test_window, len(full_data))
            train = full_data.iloc[:i]
            test = full_data.iloc[i:window_end]

            if test.empty:
                break

            current_date = test["date"].iloc[0]

            # ── Retrain Check ──────────────────────────────────────────
            should_retrain = (
                current_model is None
                or last_train_date is None
                or (pd.to_datetime(current_date) - pd.to_datetime(last_train_date)).days
                >= self.retrain_days
            )

            if should_retrain:
                logger.info(f"Retraining model at {current_date} with {len(train)} games")
                current_model = predictor_class()
                X_train = train[feature_cols]
                y_train = train[list(target_cols.values())]
                y_train.columns = list(target_cols.keys())

                try:
                    current_model.fit(X_train, y_train)
                    last_train_date = current_date
                except Exception as e:
                    logger.warning(f"Training failed at {current_date}: {e}")
                    continue

            # ── Predict Test Window ────────────────────────────────────
            for j in range(len(test)):
                game = test.iloc[j]
                X_game = test[feature_cols].iloc[[j]]

                try:
                    prediction = current_model.predict_game(X_game)
                except Exception as e:
                    continue

                # ── Simulate Bets ──────────────────────────────────────
                self._simulate_bet(game, prediction)

            # ── Track Daily Bankroll ───────────────────────────────────
            self.daily_bankroll.append({
                "date": current_date,
                "bankroll": self.bankroll,
                "cumulative_bets": len(self.bet_log),
            })

        return self._compile_results()

    # Historical F5 home-win rate used as naive market baseline
    BASELINE_HOME_WIN_RATE = 0.525

    def _simulate_bet(self, game: pd.Series, prediction: dict):
        """
        Simulate bets on a single game based on model edge.
        Uses historical F5 home-win rate as market baseline for moneyline,
        and a naive 50/50 for over/under (no line advantage assumed).
        """
        ml = prediction["moneyline"]
        total = prediction["total"]

        # ── Moneyline Bet ──────────────────────────────────────────────
        market_home_implied = self.BASELINE_HOME_WIN_RATE
        home_edge = (ml["home_prob"] - market_home_implied) * 100

        if abs(home_edge) >= self.min_edge:
            bet_side = "Home" if home_edge > 0 else "Away"
            model_prob = ml["home_prob"] if home_edge > 0 else ml["away_prob"]
            market_prob = market_home_implied if home_edge > 0 else (1 - market_home_implied)

            if self.bankroll <= 0:
                return
            kelly = self._kelly(model_prob, market_prob)
            bet_size = self.bankroll * kelly * self.kelly_fraction
            bet_size = min(bet_size, self.bankroll * 0.05)

            actual_home_win = game.get("home_f5_win", 0)
            won = (bet_side == "Home" and actual_home_win == 1) or \
                  (bet_side == "Away" and actual_home_win == 0)

            payout = bet_size * 0.909 if won else -bet_size
            self.bankroll += payout

            self.bet_log.append({
                "date": game.get("date"),
                "game_pk": game.get("game_pk"),
                "market": "F5 ML",
                "side": bet_side,
                "model_prob": round(model_prob, 3),
                "edge_pct": round(abs(home_edge), 1),
                "bet_size": round(bet_size, 2),
                "won": won,
                "payout": round(payout, 2),
                "bankroll_after": round(self.bankroll, 2),
            })

        # ── Over/Under Bet ──────────────────────────────────────────────
        if "over_under_probs" in total:
            actual_total = game.get("total_f5_runs")
            predicted_total = total.get("predicted", 4.5)
            # Use predicted total rounded to nearest 0.5 as the "line"
            line = round(predicted_total * 2) / 2
            ou_probs = total["over_under_probs"]
            if line in ou_probs and actual_total is not None:
                over_prob = ou_probs[line]["over"] / 100
                under_prob = ou_probs[line]["under"] / 100
                # Baseline: 50/50 on any given line
                over_edge = (over_prob - 0.50) * 100
                if abs(over_edge) >= self.min_edge:
                    bet_side = "Over" if over_edge > 0 else "Under"
                    model_prob = over_prob if over_edge > 0 else under_prob
                    kelly = self._kelly(model_prob, 0.50)
                    bet_size = self.bankroll * kelly * self.kelly_fraction
                    bet_size = min(bet_size, self.bankroll * 0.05)

                    won = (bet_side == "Over" and actual_total > line) or \
                          (bet_side == "Under" and actual_total < line)
                    # Push (exact line) = no action
                    if actual_total == line:
                        return

                    payout = bet_size * 0.909 if won else -bet_size
                    self.bankroll += payout

                    self.bet_log.append({
                        "date": game.get("date"),
                        "game_pk": game.get("game_pk"),
                        "market": f"F5 O/U {line}",
                        "side": bet_side,
                        "model_prob": round(model_prob, 3),
                        "edge_pct": round(abs(over_edge), 1),
                        "bet_size": round(bet_size, 2),
                        "won": won,
                        "payout": round(payout, 2),
                        "bankroll_after": round(self.bankroll, 2),
                    })

    @staticmethod
    def _kelly(model_prob: float, market_implied: float) -> float:
        """Full Kelly criterion."""
        if market_implied <= 0 or market_implied >= 1:
            return 0.0
        b = (1 / market_implied) - 1
        p = model_prob
        q = 1 - p
        return max((b * p - q) / b, 0)

    def _compile_results(self) -> dict:
        """Compile backtest results into summary."""
        if not self.bet_log:
            return {"error": "No bets placed during backtest."}

        log_df = pd.DataFrame(self.bet_log)

        total_bets = len(log_df)
        wins = log_df["won"].sum()
        total_wagered = log_df["bet_size"].sum()
        total_pnl = log_df["payout"].sum()

        return {
            "summary": {
                "total_bets": total_bets,
                "wins": int(wins),
                "losses": total_bets - int(wins),
                "win_rate": round(wins / total_bets * 100, 1),
                "total_wagered": round(total_wagered, 2),
                "total_pnl": round(total_pnl, 2),
                "roi_pct": round(total_pnl / total_wagered * 100, 2) if total_wagered > 0 else 0,
                "starting_bankroll": self.initial_bankroll,
                "ending_bankroll": round(self.bankroll, 2),
                "bankroll_growth": round(
                    (self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 1
                ),
                "avg_edge": round(log_df["edge_pct"].mean(), 1),
                "avg_bet_size": round(log_df["bet_size"].mean(), 2),
                "max_drawdown": self._max_drawdown(log_df),
                "sharpe_ratio": self._sharpe(log_df),
            },
            "by_market": log_df.groupby("market").agg(
                bets=("won", "count"),
                wins=("won", "sum"),
                pnl=("payout", "sum"),
            ).to_dict("index") if "market" in log_df.columns else {},
            "bet_log": log_df,
            "daily_bankroll": pd.DataFrame(self.daily_bankroll),
        }

    def _max_drawdown(self, log_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from peak."""
        bankrolls = log_df["bankroll_after"].values
        peak = bankrolls[0]
        max_dd = 0
        for b in bankrolls:
            peak = max(peak, b)
            dd = (peak - b) / peak * 100
            max_dd = max(max_dd, dd)
        return round(max_dd, 1)

    def _sharpe(self, log_df: pd.DataFrame) -> float:
        """Annualized Sharpe ratio of daily returns."""
        returns = log_df["payout"] / log_df["bet_size"]
        if returns.std() == 0:
            return 0.0
        return round(returns.mean() / returns.std() * np.sqrt(252), 2)
