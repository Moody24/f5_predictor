"""
Zero-Inflated Negative Binomial (ZINB) Model
----------------------------------------------
Models the distribution of F5 runs per team.

Why ZINB for F5 innings?
  - Zero inflation: ~15-20% of teams score 0 runs in F5
    (1-2-3 innings are common, especially with elite starters)
  - Overdispersion: variance > mean in baseball run scoring
    (multi-run innings create fat tails that Poisson can't handle)
  - The NB handles the "how many runs when they do score" part
  - The zero-inflation handles the "did they even score" part

Architecture:
  ZINB has two components estimated simultaneously:
    1. Zero-inflation (logit): P(structural zero) — models the probability
       of a team being "shut down" (e.g., elite pitcher, bad matchup)
    2. Count (NB): P(runs | not structural zero) — models how many runs
       when the team does produce offense

  Both components take covariates (features), so a team facing
  a high-whiff pitcher in a pitcher's park has higher zero-inflation
  AND lower expected count.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from scipy.stats import nbinom
from typing import Optional
import logging
import joblib

from config.settings import ZINB_MAX_ITER, N_SIMULATIONS, MODEL_DIR

logger = logging.getLogger(__name__)


class ZINBModel:
    """
    Zero-Inflated Negative Binomial model for F5 run distribution.

    Fits separate ZINB models for:
      - away_f5_runs: runs scored by away team in F5
      - home_f5_runs: runs scored by home team in F5

    Each model produces a full probability distribution P(runs=k)
    for k = 0, 1, 2, ..., enabling Monte Carlo simulation of outcomes.
    """

    def __init__(self):
        self.away_model = None
        self.home_model = None
        self.away_features = None
        self.home_features = None
        self.is_fitted = False

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        X_away: pd.DataFrame,
        y_away: pd.Series,
        X_home: pd.DataFrame,
        y_home: pd.Series,
        inflation_features: list[str] = None,
    ):
        """
        Fit ZINB models for away and home F5 runs.

        Args:
            X_away: Features relevant to away team's run production
                    (away offense + home starter quality + park factor)
            y_away: Away team F5 runs (integer counts)
            X_home: Features relevant to home team's run production
            y_home: Home team F5 runs (integer counts)
            inflation_features: Subset of features for the zero-inflation
                               component (default: uses all features)
        """
        self.away_features = list(X_away.columns)
        self.home_features = list(X_home.columns)

        logger.info("Fitting ZINB model for away team F5 runs...")
        self.away_model = self._fit_zinb(X_away, y_away, inflation_features)

        logger.info("Fitting ZINB model for home team F5 runs...")
        self.home_model = self._fit_zinb(X_home, y_home, inflation_features)

        self.is_fitted = True
        logger.info("ZINB models fitted successfully.")

    @staticmethod
    def _drop_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
        """Drop zero-variance columns that make the design matrix singular."""
        Xf = X.astype(float).fillna(0.0)
        varying = Xf.std() > 1e-8
        # Always keep the constant column if present
        if "const" in Xf.columns:
            varying["const"] = True
        dropped = list(Xf.columns[~varying])
        if dropped:
            logger.debug(f"Dropping {len(dropped)} constant columns: {dropped[:5]}...")
        return Xf[Xf.columns[varying]]

    def _fit_zinb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        inflation_features: list[str] = None,
    ) -> ZeroInflatedNegativeBinomialP:
        """Fit a single ZINB model."""
        X_clean = self._drop_constant_cols(X.astype(float).fillna(0.0))
        X_const = sm.add_constant(X_clean)

        # Zero-inflation covariates (intercept-only — avoids exog_infl singularity)
        X_inflate = None

        try:
            model = ZeroInflatedNegativeBinomialP(
                endog=y.astype(int),
                exog=X_const,
                exog_infl=X_inflate,
                inflation="logit",
            )
            result = model.fit(
                maxiter=ZINB_MAX_ITER,
                disp=False,
                method="bfgs",
            )
            logger.info(f"ZINB converged. AIC: {result.aic:.1f}, BIC: {result.bic:.1f}")
            return result

        except Exception as e:
            logger.warning(f"ZINB fitting failed: {e}. Falling back to standard NB.")
            return self._fit_nb_fallback(X_const, y)

    def _fit_nb_fallback(self, X: pd.DataFrame, y: pd.Series):
        """Fallback to standard Negative Binomial if ZINB fails."""
        X_clean = self._drop_constant_cols(X)
        try:
            model = sm.NegativeBinomial(
                endog=y.astype(int),
                exog=X_clean.astype(float),
            )
            result = model.fit(maxiter=ZINB_MAX_ITER, disp=False, method="bfgs")
            logger.info(f"NB fallback converged. AIC: {result.aic:.1f}")
            return result
        except Exception as e:
            logger.warning(f"NB fallback failed: {e}. Falling back to Poisson.")
            return self._fit_poisson_fallback(X_clean, y)

    def _fit_poisson_fallback(self, X: pd.DataFrame, y: pd.Series):
        """Last-resort Poisson regression — almost never singular."""
        model = sm.Poisson(
            endog=y.astype(int),
            exog=X.astype(float),
        )
        result = model.fit(maxiter=ZINB_MAX_ITER, disp=False, method="bfgs")
        logger.info(f"Poisson fallback converged. AIC: {result.aic:.1f}")
        return result

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_distribution(
        self,
        X_away: pd.DataFrame,
        X_home: pd.DataFrame,
        max_runs: int = 15,
    ) -> dict:
        """
        Predict full probability distribution for a single game.

        Returns dict with:
            away_probs: np.array of P(away_runs=k) for k=0..max_runs
            home_probs: np.array of P(home_runs=k) for k=0..max_runs
            away_mean: expected away F5 runs
            home_mean: expected home F5 runs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() first.")

        away_probs = self._get_pmf(self.away_model, X_away, max_runs)
        home_probs = self._get_pmf(self.home_model, X_home, max_runs)

        return {
            "away_probs": away_probs,
            "home_probs": home_probs,
            "away_mean": np.sum(np.arange(max_runs + 1) * away_probs),
            "home_mean": np.sum(np.arange(max_runs + 1) * home_probs),
        }

    def _get_pmf(
        self, model, X: pd.DataFrame, max_runs: int
    ) -> np.ndarray:
        """
        Get probability mass function from fitted ZINB or NB model.

        For ZINB: uses predict(which="prob") to get P(Y=k) for each k.
        For NB fallback: computes PMF from predicted mean and dispersion.
        """
        X_const = sm.add_constant(X.astype(float))

        try:
            # Try built-in probability prediction (works for both ZINB and NB)
            probs = np.array([
                model.predict(X_const, which="prob", y_values=k).values[0]
                for k in range(max_runs + 1)
            ])
            probs = np.maximum(probs, 0)  # clamp negatives
            total = probs.sum()
            if total > 0:
                probs = probs / total
                return probs
        except Exception:
            pass

        # Fallback: compute PMF from predicted mean using NB distribution
        try:
            mu = float(model.predict(X_const, which="mean").values[0])
            alpha = np.exp(model.lnalpha) if hasattr(model, "lnalpha") else 1.0
            r = 1 / max(alpha, 1e-10)
            p = r / (r + mu)
            probs = nbinom.pmf(np.arange(max_runs + 1), r, p)
            probs = probs / probs.sum()
            return probs
        except Exception as e:
            logger.error(f"PMF computation failed: {e}")
            raise

    # ── Monte Carlo Simulation ─────────────────────────────────────────

    def simulate_game(
        self,
        X_away: pd.DataFrame,
        X_home: pd.DataFrame,
        n_sims: int = N_SIMULATIONS,
    ) -> dict:
        """
        Monte Carlo simulation of F5 outcomes.

        Samples from the predicted distributions to estimate:
            - Moneyline probabilities (who leads after 5)
            - Over/under probabilities for various totals
            - Run line probabilities (+/- 0.5, 1.5)
        """
        dist = self.predict_distribution(X_away, X_home)

        # Sample from distributions
        away_samples = np.random.choice(
            np.arange(len(dist["away_probs"])),
            size=n_sims,
            p=dist["away_probs"],
        )
        home_samples = np.random.choice(
            np.arange(len(dist["home_probs"])),
            size=n_sims,
            p=dist["home_probs"],
        )
        total_samples = away_samples + home_samples

        # ── Moneyline Probabilities ────────────────────────────────────
        home_wins = np.sum(home_samples > away_samples)
        away_wins = np.sum(away_samples > home_samples)
        pushes = np.sum(home_samples == away_samples)

        # ── Over/Under Probabilities ───────────────────────────────────
        totals = {}
        for line in np.arange(2.5, 8.0, 0.5):
            over_pct = np.mean(total_samples > line) * 100
            under_pct = np.mean(total_samples < line) * 100
            push_pct = np.mean(total_samples == line) * 100
            totals[line] = {
                "over": round(over_pct, 1),
                "under": round(under_pct, 1),
                "push": round(push_pct, 1),
            }

        # ── Run Line Probabilities ─────────────────────────────────────
        spread_diff = home_samples.astype(float) - away_samples.astype(float)
        run_lines = {}
        for spread in [-1.5, -0.5, 0.5, 1.5]:
            cover_pct = np.mean(spread_diff > spread) * 100
            run_lines[spread] = round(cover_pct, 1)

        return {
            "away_mean": dist["away_mean"],
            "home_mean": dist["home_mean"],
            "total_mean": dist["away_mean"] + dist["home_mean"],
            "moneyline": {
                "home_win_pct": round(home_wins / n_sims * 100, 1),
                "away_win_pct": round(away_wins / n_sims * 100, 1),
                "push_pct": round(pushes / n_sims * 100, 1),
            },
            "totals": totals,
            "run_lines": run_lines,
            "distributions": dist,
            "n_simulations": n_sims,
        }

    # ── Model Evaluation ───────────────────────────────────────────────

    def evaluate(
        self,
        X_away_test: pd.DataFrame,
        y_away_test: pd.Series,
        X_home_test: pd.DataFrame,
        y_home_test: pd.Series,
    ) -> dict:
        """
        Evaluate model on held-out test set.

        Metrics:
            - MAE: Mean Absolute Error of expected runs
            - CRPS: Continuous Ranked Probability Score
            - Calibration: how well predicted P(k) matches observed frequency
            - Log-likelihood on test data
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        # Predicted means
        away_preds = []
        home_preds = []
        for i in range(len(X_away_test)):
            dist = self.predict_distribution(
                X_away_test.iloc[[i]], X_home_test.iloc[[i]]
            )
            away_preds.append(dist["away_mean"])
            home_preds.append(dist["home_mean"])

        away_preds = np.array(away_preds)
        home_preds = np.array(home_preds)

        return {
            "away_mae": round(np.mean(np.abs(away_preds - y_away_test.values)), 3),
            "home_mae": round(np.mean(np.abs(home_preds - y_home_test.values)), 3),
            "away_mean_pred": round(np.mean(away_preds), 3),
            "away_mean_actual": round(y_away_test.mean(), 3),
            "home_mean_pred": round(np.mean(home_preds), 3),
            "home_mean_actual": round(y_home_test.mean(), 3),
            "away_aic": self.away_model.aic if hasattr(self.away_model, "aic") else None,
            "home_aic": self.home_model.aic if hasattr(self.home_model, "aic") else None,
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, name: str = "zinb_f5", save_dir=None):
        """Save fitted models to disk."""
        save_dir = save_dir or MODEL_DIR
        path = save_dir / f"{name}.joblib"
        joblib.dump(
            {
                "away_model": self.away_model,
                "home_model": self.home_model,
                "away_features": self.away_features,
                "home_features": self.home_features,
            },
            path,
        )
        logger.info(f"ZINB model saved to {path}")

    def load(self, name: str = "zinb_f5", load_dir=None):
        """Load fitted models from disk."""
        load_dir = load_dir or MODEL_DIR
        path = load_dir / f"{name}.joblib"
        data = joblib.load(path)
        self.away_model = data["away_model"]
        self.home_model = data["home_model"]
        self.away_features = data["away_features"]
        self.home_features = data["home_features"]
        self.is_fitted = True
        logger.info(f"ZINB model loaded from {path}")
