"""
Combined F5 Predictor
-----------------------
Ensembles XGBoost and ZINB to produce final predictions
across all three markets (ML, O/U, Run Line).

Ensemble Strategy:
  - ZINB provides the probability distributions (P(runs=k))
  - XGBoost provides calibrated edge detection and feature weighting
  - Final probabilities are a weighted blend:
      P_final = w_zinb * P_zinb + w_xgb * P_xgb
  - Weights are optimized on validation set via log-loss minimization
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Optional
import logging
import json

from models.zinb_model import ZINBModel
from models.xgboost_model import XGBoostF5Model
from data.fetchers.odds_api import OddsApiFetcher
from config.settings import (
    N_SIMULATIONS, MIN_EDGE_PCT, MIN_KELLY_FRACTION,
    create_model_version_dir, get_latest_model_dir,
)

logger = logging.getLogger(__name__)


class CombinedF5Predictor:
    """
    Ensemble predictor combining ZINB distributions with XGBoost edges.

    Workflow:
      1. ZINB produces P(away_runs=k), P(home_runs=k) distributions
      2. Monte Carlo simulation derives raw market probabilities
      3. XGBoost produces calibrated P(home_win), E[total], E[diff]
      4. Ensemble blends both signals
      5. Compare vs market lines for edge identification
      6. Kelly criterion for bet sizing
    """

    def __init__(
        self,
        zinb_weight: float = 0.55,
        xgb_weight: float = 0.45,
    ):
        self.zinb = ZINBModel()
        self.xgb = XGBoostF5Model()
        self.odds = OddsApiFetcher()
        self.zinb_weight = zinb_weight
        self.xgb_weight = xgb_weight

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        away_feature_cols: list[str] = None,
        home_feature_cols: list[str] = None,
        X_ml_train: pd.DataFrame = None,
        y_ml_train: pd.Series = None,
        X_ml_val: pd.DataFrame = None,
        y_ml_val: pd.Series = None,
    ):
        """
        Train both models.

        Args:
            X_train: Full feature matrix
            y_train: DataFrame with columns:
                     [away_f5_runs, home_f5_runs, total_f5_runs,
                      home_f5_win, f5_diff]
            away_feature_cols: Features for away team ZINB
            home_feature_cols: Features for home team ZINB
        """
        # ── Identify feature subsets for ZINB ──────────────────────────
        if away_feature_cols is None:
            away_feature_cols = [c for c in X_train.columns
                                 if c.startswith(("away_facing_", "away_team_", "park_"))]
        if home_feature_cols is None:
            home_feature_cols = [c for c in X_train.columns
                                 if c.startswith(("home_facing_", "home_team_", "park_"))]

        # ── Train ZINB ─────────────────────────────────────────────────
        X_away_zinb = X_train[away_feature_cols]
        X_home_zinb = X_train[home_feature_cols]

        self.zinb.fit(
            X_away=X_away_zinb,
            y_away=y_train["away_f5_runs"],
            X_home=X_home_zinb,
            y_home=y_train["home_f5_runs"],
        )

        # ── Train XGBoost ──────────────────────────────────────────────
        y_diff = y_train["home_f5_runs"] - y_train["away_f5_runs"]

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, {
                "ml": y_val["home_f5_win"],
                "total": y_val["total_f5_runs"],
                "diff": y_val["home_f5_runs"] - y_val["away_f5_runs"],
            })

        self.xgb.fit(
            X=X_train,
            y_ml=y_ml_train if y_ml_train is not None else y_train["home_f5_win"],
            y_total=y_train["total_f5_runs"],
            y_diff=y_diff,
            eval_set=eval_set,
            X_ml=X_ml_train,
            y_ml_val=y_ml_val,
        )

        # ── Optimize Ensemble Weights ──────────────────────────────────
        if X_val is not None:
            self._optimize_weights(X_val, y_val, away_feature_cols, home_feature_cols)

        self._away_feature_cols = away_feature_cols
        self._home_feature_cols = home_feature_cols
        logger.info(
            f"Combined model trained. Weights: ZINB={self.zinb_weight:.2f}, "
            f"XGB={self.xgb_weight:.2f}"
        )

    def _optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        away_cols: list,
        home_cols: list,
    ):
        """Find optimal ensemble weight on validation set."""
        def loss(w_zinb):
            w_xgb = 1 - w_zinb
            total_loss = 0
            count = 0
            for i in range(min(len(X_val), 200)):  # sample for speed
                try:
                    zinb_sim = self.zinb.simulate_game(
                        X_val[away_cols].iloc[[i]],
                        X_val[home_cols].iloc[[i]],
                        n_sims=1000,
                    )
                    xgb_pred = self.xgb.predict(X_val.iloc[[i]])

                    # Blend moneyline probabilities
                    zinb_home = zinb_sim["moneyline"]["home_win_pct"] / 100
                    xgb_home = xgb_pred["home_win_prob"][0]
                    blended = w_zinb * zinb_home + w_xgb * xgb_home

                    actual = y_val["home_f5_win"].iloc[i]
                    # Log loss
                    blended = np.clip(blended, 0.01, 0.99)
                    total_loss -= (
                        actual * np.log(blended)
                        + (1 - actual) * np.log(1 - blended)
                    )
                    count += 1
                except Exception:
                    continue

            return total_loss / max(count, 1)

        result = minimize_scalar(loss, bounds=(0.3, 0.8), method="bounded")
        self.zinb_weight = round(result.x, 2)
        self.xgb_weight = round(1 - result.x, 2)
        logger.info(f"Optimized weights: ZINB={self.zinb_weight}, XGB={self.xgb_weight}")

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_game(
        self,
        X: pd.DataFrame,
        game_info: dict = None,
    ) -> dict:
        """
        Full prediction for a single game across all markets.

        Returns comprehensive prediction with:
            - Moneyline probabilities and edge
            - Over/under probabilities for each total line
            - Run line probabilities
            - ZINB distribution details
            - Bet recommendations
        """
        # ── ZINB Simulation ────────────────────────────────────────────
        X_away = X[[c for c in X.columns if c in self._away_feature_cols]]
        X_home = X[[c for c in X.columns if c in self._home_feature_cols]]
        zinb_sim = self.zinb.simulate_game(X_away, X_home)

        # ── XGBoost Predictions ────────────────────────────────────────
        xgb_pred = self.xgb.predict(X)

        # ── Ensemble ───────────────────────────────────────────────────
        prediction = self._ensemble_predictions(zinb_sim, xgb_pred)

        # ── Add Game Info ──────────────────────────────────────────────
        if game_info:
            prediction["game_info"] = game_info

        return prediction

    def _ensemble_predictions(self, zinb_sim: dict, xgb_pred: dict) -> dict:
        """Blend ZINB and XGBoost predictions."""
        w_z = self.zinb_weight
        w_x = self.xgb_weight

        # ── Moneyline ─────────────────────────────────────────────────
        zinb_home_pct = zinb_sim["moneyline"]["home_win_pct"] / 100
        xgb_home_prob = xgb_pred["home_win_prob"][0]
        ensemble_home = w_z * zinb_home_pct + w_x * xgb_home_prob
        ensemble_away = 1 - ensemble_home

        # ── Total ──────────────────────────────────────────────────────
        zinb_total = zinb_sim["total_mean"]
        xgb_total = xgb_pred["predicted_total"][0]
        ensemble_total = w_z * zinb_total + w_x * xgb_total

        # ── Per-side runs ──────────────────────────────────────────────
        ensemble_away_runs = w_z * zinb_sim["away_mean"] + w_x * xgb_pred["est_away_runs"][0]
        ensemble_home_runs = w_z * zinb_sim["home_mean"] + w_x * xgb_pred["est_home_runs"][0]

        return {
            "moneyline": {
                "home_prob": round(ensemble_home, 3),
                "away_prob": round(ensemble_away, 3),
                "zinb_home": round(zinb_home_pct, 3),
                "xgb_home": round(xgb_home_prob, 3),
            },
            "total": {
                "predicted": round(ensemble_total, 2),
                "zinb_total": round(zinb_total, 2),
                "xgb_total": round(xgb_total, 2),
                "over_under_probs": zinb_sim["totals"],
            },
            "run_line": {
                "predicted_diff": round(ensemble_home_runs - ensemble_away_runs, 2),
                "home_runs": round(ensemble_home_runs, 2),
                "away_runs": round(ensemble_away_runs, 2),
                "spread_probs": zinb_sim["run_lines"],
            },
            "distributions": {
                "away_pmf": zinb_sim["distributions"]["away_probs"].tolist(),
                "home_pmf": zinb_sim["distributions"]["home_probs"].tolist(),
            },
            "metadata": {
                "n_simulations": zinb_sim["n_simulations"],
                "zinb_weight": self.zinb_weight,
                "xgb_weight": self.xgb_weight,
            },
        }

    # ── Edge Detection & Sizing ────────────────────────────────────────

    def find_all_edges(
        self,
        predictions: dict,
        market_odds: dict,
        min_edge: float = MIN_EDGE_PCT,
    ) -> list[dict]:
        """
        Compare model predictions to market odds across all three markets.
        Returns list of actionable edges with Kelly sizing.
        """
        edges = []

        # ── Moneyline Edge ─────────────────────────────────────────────
        if "home_ml_implied" in market_odds:
            home_edge = (predictions["moneyline"]["home_prob"] - market_odds["home_ml_implied"]) * 100
            if home_edge >= min_edge:
                edges.append(self._build_edge(
                    market="F5 Moneyline",
                    side="Home",
                    model_prob=predictions["moneyline"]["home_prob"],
                    market_prob=market_odds["home_ml_implied"],
                    market_odds=market_odds.get("home_ml_american"),
                ))
            elif -home_edge >= min_edge:
                edges.append(self._build_edge(
                    market="F5 Moneyline",
                    side="Away",
                    model_prob=predictions["moneyline"]["away_prob"],
                    market_prob=market_odds.get("away_ml_implied", 1 - market_odds["home_ml_implied"]),
                    market_odds=market_odds.get("away_ml_american"),
                ))

        # ── Over/Under Edge ────────────────────────────────────────────
        if "total_line" in market_odds:
            line = market_odds["total_line"]
            ou_probs = predictions["total"]["over_under_probs"]
            # Find closest line in our simulation
            closest_line = min(ou_probs.keys(), key=lambda x: abs(x - line))
            sim_over = ou_probs[closest_line]["over"] / 100
            sim_under = ou_probs[closest_line]["under"] / 100

            if "over_implied" in market_odds:
                over_edge = (sim_over - market_odds["over_implied"]) * 100
                if over_edge >= min_edge:
                    edges.append(self._build_edge(
                        market=f"F5 Over {line}",
                        side="Over",
                        model_prob=sim_over,
                        market_prob=market_odds["over_implied"],
                        market_odds=market_odds.get("over_american"),
                    ))
                elif -over_edge >= min_edge:
                    edges.append(self._build_edge(
                        market=f"F5 Under {line}",
                        side="Under",
                        model_prob=sim_under,
                        market_prob=market_odds.get("under_implied"),
                        market_odds=market_odds.get("under_american"),
                    ))

        # ── Run Line Edge ──────────────────────────────────────────────
        if "home_spread" in market_odds:
            spread = market_odds["home_spread"]
            spread_probs = predictions["run_line"]["spread_probs"]
            closest_spread = min(spread_probs.keys(), key=lambda x: abs(x - spread))
            home_cover = spread_probs[closest_spread] / 100

            if "home_spread_implied" in market_odds:
                rl_edge = (home_cover - market_odds["home_spread_implied"]) * 100
                if rl_edge >= min_edge:
                    edges.append(self._build_edge(
                        market=f"F5 Run Line {spread:+.1f}",
                        side="Home",
                        model_prob=home_cover,
                        market_prob=market_odds["home_spread_implied"],
                        market_odds=market_odds.get("home_spread_american"),
                    ))

        return edges

    def _build_edge(
        self,
        market: str,
        side: str,
        model_prob: float,
        market_prob: float,
        market_odds: float = None,
    ) -> dict:
        """Build standardized edge dict with Kelly sizing."""
        edge_pct = (model_prob - market_prob) * 100
        kelly = self._kelly(model_prob, market_prob)

        return {
            "market": market,
            "side": side,
            "model_prob": round(model_prob, 3),
            "market_implied": round(market_prob, 3),
            "edge_pct": round(edge_pct, 1),
            "market_odds_american": market_odds,
            "kelly_full": round(kelly, 3),
            "kelly_half": round(kelly * 0.5, 3),
            "kelly_quarter": round(kelly * 0.25, 3),
            "confidence": self._edge_confidence(edge_pct),
        }

    @staticmethod
    def _kelly(model_prob: float, market_implied: float) -> float:
        """Full Kelly criterion."""
        if market_implied <= 0 or market_implied >= 1:
            return 0.0
        b = (1 / market_implied) - 1
        p = model_prob
        q = 1 - p
        return max((b * p - q) / b, 0)

    @staticmethod
    def _edge_confidence(edge_pct: float) -> str:
        """Categorize edge strength."""
        if edge_pct >= 10:
            return "STRONG"
        elif edge_pct >= 6:
            return "MODERATE"
        elif edge_pct >= 3:
            return "LEAN"
        else:
            return "NO EDGE"

    # ── Summary Output ─────────────────────────────────────────────────

    def generate_game_card(self, prediction: dict, edges: list) -> str:
        """
        Generate human-readable game card for a prediction.
        """
        info = prediction.get("game_info", {})
        ml = prediction["moneyline"]
        total = prediction["total"]
        rl = prediction["run_line"]

        lines = [
            "=" * 60,
            f"F5 PREDICTION: {info.get('away_team', 'Away')} @ {info.get('home_team', 'Home')}",
            f"Date: {info.get('date', 'TBD')} | Venue: {info.get('venue', 'TBD')}",
            f"Starters: {info.get('away_starter', '?')} vs {info.get('home_starter', '?')}",
            "=" * 60,
            "",
            "── MONEYLINE ──",
            f"  Home Win: {ml['home_prob']*100:.1f}%  (ZINB: {ml['zinb_home']*100:.1f}% | XGB: {ml['xgb_home']*100:.1f}%)",
            f"  Away Win: {ml['away_prob']*100:.1f}%",
            "",
            "── TOTAL ──",
            f"  Predicted Total: {total['predicted']:.1f} runs",
            f"  (ZINB: {total['zinb_total']:.1f} | XGB: {total['xgb_total']:.1f})",
            "",
            "── RUN LINE ──",
            f"  Predicted Diff: Home {rl['predicted_diff']:+.1f}",
            f"  Est. Score: Away {rl['away_runs']:.1f} - Home {rl['home_runs']:.1f}",
            "",
        ]

        if edges:
            lines.append("── EDGES FOUND ──")
            for edge in edges:
                lines.append(
                    f"  [{edge['confidence']}] {edge['market']} {edge['side']}: "
                    f"Model {edge['model_prob']*100:.1f}% vs Market {edge['market_implied']*100:.1f}% "
                    f"(Edge: {edge['edge_pct']:+.1f}%) | "
                    f"Half-Kelly: {edge['kelly_half']*100:.1f}%"
                )
        else:
            lines.append("── NO ACTIONABLE EDGES ──")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, name: str = "combined_f5", version_dir=None):
        """Save both models and ensemble config to a versioned directory."""
        if version_dir is None:
            version_dir = create_model_version_dir()

        self.zinb.save(f"{name}_zinb", save_dir=version_dir)
        self.xgb.save(f"{name}_xgb", save_dir=version_dir)

        config = {
            "zinb_weight": self.zinb_weight,
            "xgb_weight": self.xgb_weight,
            "away_feature_cols": self._away_feature_cols,
            "home_feature_cols": self._home_feature_cols,
        }
        config_path = version_dir / f"{name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Combined model saved to {version_dir}")
        return version_dir

    def load(self, name: str = "combined_f5", version_dir=None):
        """Load both models and ensemble config from a versioned directory."""
        if version_dir is None:
            version_dir = get_latest_model_dir()

        self.zinb.load(f"{name}_zinb", load_dir=version_dir)
        self.xgb.load(f"{name}_xgb", load_dir=version_dir)

        config_path = version_dir / f"{name}_config.json"
        with open(config_path) as f:
            config = json.load(f)
        self.zinb_weight = config["zinb_weight"]
        self.xgb_weight = config["xgb_weight"]
        self._away_feature_cols = config["away_feature_cols"]
        self._home_feature_cols = config["home_feature_cols"]
        logger.info(f"Combined model loaded from {version_dir}")
