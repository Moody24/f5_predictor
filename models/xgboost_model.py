"""
XGBoost Model for F5 Predictions
----------------------------------
Handles classification (moneyline winner) and regression (total runs)
with feature importance analysis and edge detection.

Architecture:
  1. ML Classifier: P(home wins F5) — binary classification
  2. Total Regressor: Predicted total F5 runs — regression
  3. Spread Regressor: Predicted F5 run differential — regression

XGBoost complements ZINB by:
  - Capturing non-linear feature interactions ZINB misses
  - Providing robust feature importance rankings
  - Handling missing data natively
  - Serving as an ensemble member alongside ZINB probabilities
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error,
)
from sklearn.calibration import calibration_curve
from typing import Optional
import logging
import joblib

from config.settings import XGBOOST_PARAMS, XGBOOST_REGRESSOR_PARAMS, MODEL_DIR

logger = logging.getLogger(__name__)


class XGBoostF5Model:
    """
    XGBoost ensemble for F5 predictions.

    Three sub-models:
      1. ml_classifier  — P(home F5 win)
      2. total_regressor — Expected total F5 runs
      3. diff_regressor  — Expected home - away F5 runs
    """

    def __init__(self):
        self.ml_classifier = None
        self.total_regressor = None
        self.diff_regressor = None
        self.feature_names = None
        self.is_fitted = False

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y_ml: pd.Series,
        y_total: pd.Series,
        y_diff: pd.Series,
        eval_set: tuple = None,
        X_ml: pd.DataFrame = None,
        X_ml_val: pd.DataFrame = None,
        y_ml_val: pd.Series = None,
    ):
        """
        Fit all three XGBoost sub-models.

        Args:
            X: Feature matrix (used for total and diff regressors)
            y_ml: Binary target (1 = home wins F5)
            y_total: Total F5 runs
            y_diff: Home F5 runs - Away F5 runs
            eval_set: Optional (X_val, y_dict) for early stopping on regressors
            X_ml: Optional feature matrix for ML classifier (push-excluded rows).
            X_ml_val: Matching validation features for ML classifier eval.
            y_ml_val: Matching validation target for ML classifier eval.
        """
        self.feature_names = list(X.columns)

        # Unpack eval_set once so all three models can use it
        X_val, y_val = None, None
        if eval_set:
            X_val, y_val = eval_set

        # ── Moneyline Classifier ───────────────────────────────────────
        logger.info("Training F5 moneyline classifier...")
        self.ml_classifier = xgb.XGBClassifier(**XGBOOST_PARAMS)

        X_fit = X_ml if X_ml is not None else X
        fit_params = {"verbose": False}
        if eval_set and X_ml_val is not None and y_ml_val is not None:
            # Push-excluded val set — early stopping uses this eval set
            fit_params["eval_set"] = [(X_ml_val, y_ml_val)]
        elif eval_set:
            fit_params["eval_set"] = [(X_val, y_val["ml"])]

        self.ml_classifier.fit(X_fit, y_ml, **fit_params)

        # ── Total Runs Regressor ───────────────────────────────────────
        logger.info("Training F5 total runs regressor...")
        self.total_regressor = xgb.XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)

        fit_params_reg = {"verbose": False}
        if eval_set:
            fit_params_reg["eval_set"] = [(X_val, y_val["total"])]

        self.total_regressor.fit(X, y_total, **fit_params_reg)

        # ── Run Differential Regressor ─────────────────────────────────
        logger.info("Training F5 run differential regressor...")
        self.diff_regressor = xgb.XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)

        fit_params_diff = {"verbose": False}
        if eval_set:
            fit_params_diff["eval_set"] = [(X_val, y_val["diff"])]

        self.diff_regressor.fit(X, y_diff, **fit_params_diff)

        self.is_fitted = True
        logger.info("All XGBoost models trained successfully.")

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Generate predictions from all three models.

        Returns:
            dict with:
                home_win_prob: P(home wins F5)
                predicted_total: expected total F5 runs
                predicted_diff: expected home-away run differential
                away_win_prob: 1 - home_win_prob (adjusted)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() first.")

        home_prob = self.ml_classifier.predict_proba(X)[:, 1]
        total = self.total_regressor.predict(X)
        diff = self.diff_regressor.predict(X)

        return {
            "home_win_prob": home_prob,
            "away_win_prob": 1 - home_prob,
            "predicted_total": total,
            "predicted_diff": diff,
            # Derived: estimated runs per side
            "est_home_runs": (total + diff) / 2,
            "est_away_runs": (total - diff) / 2,
        }

    # ── Cross-Validation ───────────────────────────────────────────────

    def cross_validate(
        self,
        X: pd.DataFrame,
        y_ml: pd.Series,
        y_total: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """
        Time-series cross-validation (no future leakage).
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        ml_scores = cross_val_score(
            xgb.XGBClassifier(**XGBOOST_PARAMS),
            X, y_ml, cv=tscv, scoring="neg_log_loss",
        )
        total_scores = cross_val_score(
            xgb.XGBRegressor(**XGBOOST_REGRESSOR_PARAMS),
            X, y_total, cv=tscv, scoring="neg_mean_absolute_error",
        )

        return {
            "ml_logloss": {
                "mean": round(-ml_scores.mean(), 4),
                "std": round(ml_scores.std(), 4),
                "folds": [-round(s, 4) for s in ml_scores],
            },
            "total_mae": {
                "mean": round(-total_scores.mean(), 3),
                "std": round(total_scores.std(), 3),
                "folds": [-round(s, 3) for s in total_scores],
            },
        }

    # ── Feature Importance ─────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from all models.
        Uses gain-based importance (most reliable for XGBoost).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        importances = []
        for name, model in [
            ("moneyline", self.ml_classifier),
            ("total", self.total_regressor),
            ("diff", self.diff_regressor),
        ]:
            imp = model.get_booster().get_score(importance_type="gain")
            for feat, score in imp.items():
                importances.append({
                    "model": name,
                    "feature": feat,
                    "importance": score,
                })

        df = pd.DataFrame(importances)
        if df.empty:
            return df

        # Pivot to wide format
        pivot = df.pivot_table(
            index="feature", columns="model", values="importance", fill_value=0
        )
        pivot["avg_importance"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg_importance", ascending=False)

        return pivot.head(top_n)

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_ml_test: pd.Series,
        y_total_test: pd.Series,
        y_diff_test: pd.Series,
    ) -> dict:
        """Evaluate all models on test set."""
        preds = self.predict(X_test)

        # Moneyline metrics
        ml_pred_class = (preds["home_win_prob"] >= 0.5).astype(int)

        # Calibration
        prob_true, prob_pred = calibration_curve(
            y_ml_test, preds["home_win_prob"], n_bins=10, strategy="uniform"
        )

        return {
            "moneyline": {
                "accuracy": round(accuracy_score(y_ml_test, ml_pred_class), 3),
                "log_loss": round(log_loss(y_ml_test, preds["home_win_prob"]), 4),
                "brier_score": round(brier_score_loss(y_ml_test, preds["home_win_prob"]), 4),
                "calibration_bins": {
                    "predicted": prob_pred.round(3).tolist(),
                    "actual": prob_true.round(3).tolist(),
                },
            },
            "total": {
                "mae": round(mean_absolute_error(y_total_test, preds["predicted_total"]), 3),
                "rmse": round(np.sqrt(mean_squared_error(y_total_test, preds["predicted_total"])), 3),
                "mean_pred": round(preds["predicted_total"].mean(), 3),
                "mean_actual": round(y_total_test.mean(), 3),
            },
            "differential": {
                "mae": round(mean_absolute_error(y_diff_test, preds["predicted_diff"]), 3),
                "rmse": round(np.sqrt(mean_squared_error(y_diff_test, preds["predicted_diff"])), 3),
            },
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, name: str = "xgb_f5", save_dir=None):
        """Save all models to disk."""
        save_dir = save_dir or MODEL_DIR
        path = save_dir / f"{name}.joblib"
        joblib.dump(
            {
                "ml_classifier": self.ml_classifier,
                "total_regressor": self.total_regressor,
                "diff_regressor": self.diff_regressor,
                "feature_names": self.feature_names,
            },
            path,
        )
        logger.info(f"XGBoost models saved to {path}")

    def load(self, name: str = "xgb_f5", load_dir=None):
        """Load all models from disk."""
        load_dir = load_dir or MODEL_DIR
        path = load_dir / f"{name}.joblib"
        data = joblib.load(path)
        self.ml_classifier = data["ml_classifier"]
        self.total_regressor = data["total_regressor"]
        self.diff_regressor = data["diff_regressor"]
        self.feature_names = data["feature_names"]
        self.is_fitted = True
        logger.info(f"XGBoost models loaded from {path}")
