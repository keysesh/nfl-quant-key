"""
NFL QUANT - Baseline Regression Models
======================================

Tier 3 Blueprint requirement: Traditional regression baselines for comparison.

Provides simple logistic/linear regression models to:
1. Validate that XGBoost is actually providing lift over simple models
2. Serve as explainable baseline for feature importance analysis
3. Provide quick predictions when complex models fail

Usage:
    from nfl_quant.models.baseline_regression import (
        BaselineLogistic,
        BaselineLinear,
        compare_to_xgboost,
    )

    # Train baseline
    model = BaselineLogistic()
    model.fit(X_train, y_train)

    # Compare to XGBoost
    comparison = compare_to_xgboost(baseline_model, xgb_model, X_test, y_test)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Results of comparing baseline to XGBoost."""
    baseline_accuracy: float
    baseline_log_loss: float
    baseline_brier: float
    xgb_accuracy: float
    xgb_log_loss: float
    xgb_brier: float
    xgb_lift_accuracy: float
    xgb_lift_log_loss: float
    xgb_lift_brier: float
    features_used: int
    n_samples: int


class BaselineLogistic:
    """
    Simple logistic regression baseline for P(UNDER) prediction.

    Uses L2 regularization (Ridge) to prevent overfitting.
    Provides probability calibration similar to XGBoost output.

    Attributes:
        model: sklearn Pipeline with scaler and logistic regression
        feature_names: List of feature names used in training
        classes_: Class labels (0, 1)
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
    ):
        """
        Initialize baseline logistic regression.

        Args:
            C: Inverse regularization strength (lower = more regularization)
            max_iter: Maximum iterations for convergence
            class_weight: 'balanced' weights classes inversely proportional to frequency
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                solver='lbfgs',
                random_state=42,
            ))
        ])
        self.feature_names: List[str] = []
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> 'BaselineLogistic':
        """
        Fit the logistic regression model.

        Args:
            X: Feature matrix
            y: Binary target (0 = OVER hit, 1 = UNDER hit)

        Returns:
            self for chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        self.model.fit(X, y)
        logger.info(f"Trained baseline logistic on {len(y)} samples, {X.shape[1]} features")

        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict probability of UNDER.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with [P(OVER), P(UNDER)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict_proba(X)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Probability threshold for UNDER class

        Returns:
            Array of 0s and 1s
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on coefficient magnitudes.

        For logistic regression, larger absolute coefficients = more important.

        Returns:
            Dictionary mapping feature name to importance
        """
        if not self.feature_names:
            return {}

        coefs = self.model.named_steps['classifier'].coef_[0]
        importance = np.abs(coefs) / np.sum(np.abs(coefs))

        return dict(zip(self.feature_names, importance))

    def get_coefficients(self) -> Dict[str, float]:
        """
        Get raw coefficients (useful for interpretation).

        Positive coefficient = feature increases P(UNDER)
        Negative coefficient = feature decreases P(UNDER)

        Returns:
            Dictionary mapping feature name to coefficient
        """
        if not self.feature_names:
            return {}

        coefs = self.model.named_steps['classifier'].coef_[0]
        return dict(zip(self.feature_names, coefs))

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary with accuracy, log_loss, brier, auc
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.nan_to_num(X, nan=0.0)

        probs = self.predict_proba(X)
        preds = self.predict(X)

        return {
            'accuracy': accuracy_score(y, preds),
            'log_loss': log_loss(y, probs),
            'brier': brier_score_loss(y, probs[:, 1]),
            'auc': roc_auc_score(y, probs[:, 1]),
            'n_samples': len(y),
        }


class BaselineLinear:
    """
    Simple linear regression baseline for continuous outcomes.

    Useful for predicting actual stat values (yards, receptions).
    Uses Ridge regression (L2) for regularization.

    Attributes:
        model: sklearn Pipeline with scaler and Ridge regression
        feature_names: List of feature names used in training
    """

    def __init__(
        self,
        alpha: float = 1.0,
    ):
        """
        Initialize baseline linear regression.

        Args:
            alpha: Regularization strength (higher = more regularization)
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha, random_state=42))
        ])
        self.feature_names: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> 'BaselineLinear':
        """
        Fit the linear regression model.

        Args:
            X: Feature matrix
            y: Continuous target (actual stat value)

        Returns:
            self for chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        self.model.fit(X, y)
        logger.info(f"Trained baseline linear on {len(y)} samples, {X.shape[1]} features")

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict stat values.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on coefficient magnitudes."""
        if not self.feature_names:
            return {}

        coefs = self.model.named_steps['regressor'].coef_
        importance = np.abs(coefs) / np.sum(np.abs(coefs))

        return dict(zip(self.feature_names, importance))

    def get_coefficients(self) -> Dict[str, float]:
        """Get raw coefficients."""
        if not self.feature_names:
            return {}

        coefs = self.model.named_steps['regressor'].coef_
        return dict(zip(self.feature_names, coefs))

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dictionary with rmse, mae, r2
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        preds = self.predict(X)

        return {
            'rmse': np.sqrt(mean_squared_error(y, preds)),
            'mae': mean_absolute_error(y, preds),
            'r2': r2_score(y, preds),
            'n_samples': len(y),
        }


def compare_to_xgboost(
    baseline: BaselineLogistic,
    xgb_model,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> ModelComparison:
    """
    Compare baseline logistic regression to XGBoost model.

    Args:
        baseline: Trained BaselineLogistic model
        xgb_model: Trained XGBoost classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        ModelComparison with metrics and lift
    """
    if isinstance(X_test, pd.DataFrame):
        X_test_arr = X_test.values
    else:
        X_test_arr = X_test

    if isinstance(y_test, pd.Series):
        y_test_arr = y_test.values
    else:
        y_test_arr = y_test

    X_test_arr = np.nan_to_num(X_test_arr, nan=0.0)

    # Baseline predictions
    baseline_probs = baseline.predict_proba(X_test_arr)
    baseline_preds = baseline.predict(X_test_arr)

    # XGBoost predictions
    xgb_probs = xgb_model.predict_proba(X_test_arr)
    xgb_preds = xgb_model.predict(X_test_arr)

    # Calculate metrics
    baseline_acc = accuracy_score(y_test_arr, baseline_preds)
    baseline_ll = log_loss(y_test_arr, baseline_probs)
    baseline_brier = brier_score_loss(y_test_arr, baseline_probs[:, 1])

    xgb_acc = accuracy_score(y_test_arr, xgb_preds)
    xgb_ll = log_loss(y_test_arr, xgb_probs)
    xgb_brier = brier_score_loss(y_test_arr, xgb_probs[:, 1])

    return ModelComparison(
        baseline_accuracy=baseline_acc,
        baseline_log_loss=baseline_ll,
        baseline_brier=baseline_brier,
        xgb_accuracy=xgb_acc,
        xgb_log_loss=xgb_ll,
        xgb_brier=xgb_brier,
        xgb_lift_accuracy=(xgb_acc - baseline_acc) / baseline_acc * 100,
        xgb_lift_log_loss=(baseline_ll - xgb_ll) / baseline_ll * 100,
        xgb_lift_brier=(baseline_brier - xgb_brier) / baseline_brier * 100,
        features_used=X_test_arr.shape[1],
        n_samples=len(y_test_arr),
    )


def train_baseline_for_market(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    market: str,
) -> BaselineLogistic:
    """
    Train a baseline model for a specific market.

    Args:
        X_train: Training features
        y_train: Binary target (1 = UNDER hit)
        market: Market name for logging

    Returns:
        Trained BaselineLogistic model
    """
    model = BaselineLogistic(C=0.1, class_weight='balanced')
    model.fit(X_train, y_train)

    # Log top features
    importance = model.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"[{market}] Top features: {top_features}")

    return model
