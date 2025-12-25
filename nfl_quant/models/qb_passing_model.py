"""
QB Passing Yards Model

A specialized model for predicting QB passing yards using:
1. XGBoost regressor for expected passing yards
2. Variance model for uncertainty estimation
3. P(UNDER) derived from the predicted distribution

This approach is more principled than direct classification for
high-variance continuous outcomes like QB passing yards.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy.stats import norm

import xgboost as xgb

from configs.qb_model_config import (
    QB_FEATURES,
    QB_MODEL_PARAMS,
    MIN_PREDICTED_STD,
    MAX_PREDICTED_STD,
    QB_SPREAD_FILTER_ENABLED,
    QB_MAX_SPREAD_ABS,
)
from nfl_quant.config_paths import MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class QBPrediction:
    """Prediction result for a QB passing yards prop."""
    expected_yards: float          # Point estimate of passing yards
    estimated_std: float           # Estimated standard deviation
    p_under: float                 # P(actual < line)
    edge_yards: float              # line - expected_yards (positive = under edge)
    confidence: float              # How confident the model is (0-1)


class QBPassingModel:
    """
    Dedicated model for QB passing yards predictions.

    Architecture:
    1. Regressor: XGBoost predicting expected passing yards
    2. Variance model: XGBoost predicting abs(residual) from regressor
    3. P(UNDER) = norm.cdf(line, mean=expected_yards, std=estimated_std)

    This is more appropriate for QB passing yards because:
    - The market has 8.8x variance ratio (actuals much more variable than lines)
    - Direct classification ignores the magnitude of edge
    - Regression + variance allows principled probability estimation
    """

    def __init__(self):
        """Initialize the QB passing model."""
        self.regressor: Optional[xgb.XGBRegressor] = None
        self.variance_model: Optional[xgb.XGBRegressor] = None
        self.feature_cols: List[str] = []
        self.is_fitted: bool = False
        self.version: str = "QB_V1"

        # Training metadata
        self.train_n_samples: int = 0
        self.train_mae: float = 0.0
        self.train_rmse: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y_yards: np.ndarray,
        feature_cols: List[str] = None,
    ) -> 'QBPassingModel':
        """
        Fit the QB passing model.

        Args:
            X: Feature DataFrame
            y_yards: Actual passing yards
            feature_cols: Feature column names (uses QB_FEATURES if None)

        Returns:
            self (fitted model)
        """
        self.feature_cols = feature_cols or [c for c in QB_FEATURES if c in X.columns]

        if not self.feature_cols:
            raise ValueError("No valid feature columns found")

        # Prepare training data
        X_train = X[self.feature_cols].copy()

        # Fill missing values with 0 (XGBoost handles this well)
        X_train = X_train.fillna(0)

        logger.info(f"Training QB model on {len(X_train)} samples with {len(self.feature_cols)} features")

        # Step 1: Fit regressor
        self.regressor = xgb.XGBRegressor(
            n_estimators=QB_MODEL_PARAMS.regressor_n_estimators,
            max_depth=QB_MODEL_PARAMS.regressor_max_depth,
            learning_rate=QB_MODEL_PARAMS.regressor_learning_rate,
            subsample=QB_MODEL_PARAMS.regressor_subsample,
            colsample_bytree=QB_MODEL_PARAMS.regressor_colsample_bytree,
            min_child_weight=QB_MODEL_PARAMS.regressor_min_child_weight,
            random_state=QB_MODEL_PARAMS.random_state,
            verbosity=QB_MODEL_PARAMS.verbosity,
        )
        self.regressor.fit(X_train, y_yards)

        # Calculate residuals
        preds = self.regressor.predict(X_train)
        residuals = np.abs(y_yards - preds)

        # Step 2: Fit variance model on abs(residuals)
        self.variance_model = xgb.XGBRegressor(
            n_estimators=QB_MODEL_PARAMS.variance_n_estimators,
            max_depth=QB_MODEL_PARAMS.variance_max_depth,
            learning_rate=QB_MODEL_PARAMS.variance_learning_rate,
            subsample=QB_MODEL_PARAMS.variance_subsample,
            random_state=QB_MODEL_PARAMS.random_state,
            verbosity=QB_MODEL_PARAMS.verbosity,
        )
        self.variance_model.fit(X_train, residuals)

        # Calculate training metrics
        self.train_n_samples = len(y_yards)
        self.train_mae = np.mean(residuals)
        self.train_rmse = np.sqrt(np.mean((y_yards - preds) ** 2))

        self.is_fitted = True

        logger.info(f"QB model fitted: MAE={self.train_mae:.1f}, RMSE={self.train_rmse:.1f}")

        return self

    def predict_distribution(
        self,
        X: pd.DataFrame,
        lines: np.ndarray,
    ) -> List[QBPrediction]:
        """
        Predict passing yards distribution for each row.

        Args:
            X: Feature DataFrame
            lines: Betting lines for each row

        Returns:
            List of QBPrediction objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        X_pred = X[self.feature_cols].copy()
        X_pred = X_pred.fillna(0)

        # Get predictions
        expected_yards = self.regressor.predict(X_pred)
        raw_std = self.variance_model.predict(X_pred)

        # Clip std to reasonable bounds
        estimated_std = np.clip(raw_std, MIN_PREDICTED_STD, MAX_PREDICTED_STD)

        # Calculate P(UNDER) using normal CDF
        p_under = norm.cdf(lines, loc=expected_yards, scale=estimated_std)

        # Calculate edge and confidence
        edge_yards = lines - expected_yards

        # Confidence based on how different p_under is from 0.5
        confidence = 2 * np.abs(p_under - 0.5)

        predictions = []
        for i in range(len(X)):
            predictions.append(QBPrediction(
                expected_yards=float(expected_yards[i]),
                estimated_std=float(estimated_std[i]),
                p_under=float(p_under[i]),
                edge_yards=float(edge_yards[i]),
                confidence=float(confidence[i]),
            ))

        return predictions

    def predict_p_under(
        self,
        X: pd.DataFrame,
        lines: np.ndarray,
    ) -> np.ndarray:
        """
        Get P(UNDER) predictions for compatibility with classifier interface.

        Args:
            X: Feature DataFrame
            lines: Betting lines

        Returns:
            Array of P(UNDER) values
        """
        predictions = self.predict_distribution(X, lines)
        return np.array([p.p_under for p in predictions])

    def predict_with_edge(
        self,
        X: pd.DataFrame,
        lines: np.ndarray,
        vegas_spread: np.ndarray = None,
        min_edge_yards: float = 10.0,
        min_confidence: float = 0.55,
        apply_spread_filter: bool = None,
    ) -> pd.DataFrame:
        """
        Get predictions with edge filtering.

        Args:
            X: Feature DataFrame
            lines: Betting lines
            vegas_spread: Vegas spread values (negative = favorite)
            min_edge_yards: Minimum yards edge to recommend bet
            min_confidence: Minimum P(UNDER) or P(OVER) to recommend
            apply_spread_filter: Override global spread filter setting

        Returns:
            DataFrame with predictions and recommendations
        """
        predictions = self.predict_distribution(X, lines)

        # Determine whether to apply spread filter
        use_spread_filter = apply_spread_filter if apply_spread_filter is not None else QB_SPREAD_FILTER_ENABLED

        results = []
        for i, pred in enumerate(predictions):
            # Determine pick
            if pred.p_under > min_confidence:
                pick = 'UNDER'
                pick_confidence = pred.p_under
            elif pred.p_under < (1 - min_confidence):
                pick = 'OVER'
                pick_confidence = 1 - pred.p_under
            else:
                pick = 'NO_BET'
                pick_confidence = 0.5

            # Check edge threshold
            has_edge = abs(pred.edge_yards) >= min_edge_yards

            # Check spread filter (only bet on close games)
            spread_val = vegas_spread[i] if vegas_spread is not None else None
            passes_spread_filter = True
            if use_spread_filter and spread_val is not None and not np.isnan(spread_val):
                passes_spread_filter = abs(spread_val) <= QB_MAX_SPREAD_ABS

            results.append({
                'expected_yards': pred.expected_yards,
                'estimated_std': pred.estimated_std,
                'p_under': pred.p_under,
                'edge_yards': pred.edge_yards,
                'line': float(lines[i]),
                'pick': pick,
                'pick_confidence': pick_confidence,
                'has_edge': has_edge,
                'passes_spread_filter': passes_spread_filter,
                'recommend_bet': pick != 'NO_BET' and has_edge and passes_spread_filter,
            })

        return pd.DataFrame(results)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the regressor."""
        if not self.is_fitted:
            return {}

        importance = dict(zip(
            self.feature_cols,
            self.regressor.feature_importances_
        ))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path = None) -> Path:
        """
        Save model to disk.

        Args:
            path: Save path (default: data/models/qb_passing_model.joblib)

        Returns:
            Path where model was saved
        """
        if path is None:
            path = MODELS_DIR / 'qb_passing_model.joblib'

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_dict = {
            'version': self.version,
            'regressor': self.regressor,
            'variance_model': self.variance_model,
            'feature_cols': self.feature_cols,
            'train_n_samples': self.train_n_samples,
            'train_mae': self.train_mae,
            'train_rmse': self.train_rmse,
        }

        joblib.dump(model_dict, path)
        logger.info(f"QB model saved to {path}")

        return path

    @classmethod
    def load(cls, path: Path = None) -> 'QBPassingModel':
        """
        Load model from disk.

        Args:
            path: Load path (default: data/models/qb_passing_model.joblib)

        Returns:
            Loaded QBPassingModel
        """
        if path is None:
            path = MODELS_DIR / 'qb_passing_model.joblib'

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"QB model not found at {path}")

        model_dict = joblib.load(path)

        model = cls()
        model.version = model_dict.get('version', 'QB_V1')
        model.regressor = model_dict['regressor']
        model.variance_model = model_dict['variance_model']
        model.feature_cols = model_dict['feature_cols']
        model.train_n_samples = model_dict.get('train_n_samples', 0)
        model.train_mae = model_dict.get('train_mae', 0.0)
        model.train_rmse = model_dict.get('train_rmse', 0.0)
        model.is_fitted = True

        logger.info(f"QB model loaded from {path}: version={model.version}, features={len(model.feature_cols)}")

        return model


def load_qb_model(path: Path = None) -> Optional[QBPassingModel]:
    """
    Load QB model if it exists.

    Args:
        path: Model path (default: data/models/qb_passing_model.joblib)

    Returns:
        QBPassingModel or None if not found
    """
    try:
        return QBPassingModel.load(path)
    except FileNotFoundError:
        logger.warning("QB model not found")
        return None
