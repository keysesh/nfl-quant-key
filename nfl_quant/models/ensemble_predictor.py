#!/usr/bin/env python3
"""
Ensemble Predictor - Combines multiple models for improved accuracy.

Implements:
1. Stacking ensemble (meta-learner combines base model predictions)
2. Weighted averaging (weights based on recent validation performance)
3. Variance reduction through model diversity

Expected improvements:
- 10-15% reduction in MAE through ensemble averaging
- More robust predictions (less sensitive to any single model's errors)
- Better calibration through meta-learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor that combines usage and efficiency predictions.

    Architecture:
    - Base models: UsagePredictor, EfficiencyPredictor
    - Meta-learner: Ridge regression (simple, robust, prevents overfitting)
    - Fallback: Weighted average (if meta-learner fails or not fitted)
    """

    def __init__(
        self,
        base_models: Optional[List] = None,
        ensemble_method: str = 'stacking',  # 'stacking' or 'weighted_avg'
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble predictor.

        Args:
            base_models: List of base predictor objects
            ensemble_method: 'stacking' (meta-learner) or 'weighted_avg'
            weights: Model weights for weighted averaging (default: equal weights)
        """
        self.base_models = base_models or []
        self.ensemble_method = ensemble_method
        self.meta_learners = {}  # One meta-learner per stat
        self.is_fitted = False

        # Default weights (equal)
        if weights is None:
            n_models = len(self.base_models) if self.base_models else 2
            self.weights = {f'model_{i}': 1.0 / n_models for i in range(n_models)}
        else:
            self.weights = weights

    def fit(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, pd.Series],
        base_predictions: Optional[Dict[str, pd.DataFrame]] = None
    ):
        """
        Fit meta-learners on base model predictions.

        Args:
            X: Feature matrix (may not be used if base_predictions provided)
            y_dict: Dict of stat_name -> ground truth values
            base_predictions: Pre-computed predictions from base models
                Format: {stat_name: DataFrame with columns [model_0, model_1, ...]}
        """
        if self.ensemble_method != 'stacking':
            logger.info("Weighted averaging mode - no meta-learner training needed")
            self.is_fitted = True
            return

        if base_predictions is None:
            raise ValueError("base_predictions required for stacking ensemble training")

        logger.info("Training meta-learners for stacking ensemble...")

        for stat_name, y_true in y_dict.items():
            if stat_name not in base_predictions:
                logger.warning(f"No base predictions for {stat_name}, skipping")
                continue

            # Get base model predictions as features for meta-learner
            X_meta = base_predictions[stat_name]

            # Align indices
            common_idx = X_meta.index.intersection(y_true.index)
            X_meta = X_meta.loc[common_idx]
            y_aligned = y_true.loc[common_idx]

            if len(X_meta) < 10:
                logger.warning(f"Insufficient data for {stat_name} meta-learner ({len(X_meta)} samples)")
                continue

            # Train meta-learner (Ridge for simplicity and regularization)
            meta_model = Ridge(alpha=1.0, fit_intercept=True)
            meta_model.fit(X_meta, y_aligned)

            self.meta_learners[stat_name] = meta_model

            # Log weights learned
            weights = meta_model.coef_
            logger.info(
                f"  {stat_name}: learned weights = {weights} "
                f"(intercept={meta_model.intercept_:.3f})"
            )

        self.is_fitted = True
        logger.info(f"Trained {len(self.meta_learners)} meta-learners")

    def predict(
        self,
        X: pd.DataFrame = None,
        base_predictions: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix (may not be used if base_predictions provided)
            base_predictions: Pre-computed predictions from base models
                Format: {stat_name: DataFrame with columns [model_0, model_1, ...]}

        Returns:
            Dict of stat_name -> predictions
        """
        if base_predictions is None:
            raise ValueError("base_predictions required for ensemble prediction")

        predictions = {}

        for stat_name, X_meta in base_predictions.items():
            if self.ensemble_method == 'stacking' and stat_name in self.meta_learners:
                # Use meta-learner
                meta_model = self.meta_learners[stat_name]
                pred = meta_model.predict(X_meta)
                predictions[stat_name] = pd.Series(pred, index=X_meta.index)
                logger.debug(f"  {stat_name}: stacking prediction (meta-learner)")

            else:
                # Fallback to weighted average
                # Assume X_meta columns are [model_0, model_1, ...]
                weights_list = []
                for col in X_meta.columns:
                    weights_list.append(self.weights.get(col, 1.0 / len(X_meta.columns)))

                # Normalize weights
                total_weight = sum(weights_list)
                weights_list = [w / total_weight for w in weights_list]

                # Weighted average
                pred = (X_meta * weights_list).sum(axis=1)
                predictions[stat_name] = pred
                logger.debug(f"  {stat_name}: weighted average ({weights_list})")

        return predictions

    def save(self, filepath: str):
        """Save ensemble to disk."""
        save_data = {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'meta_learners': self.meta_learners,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(save_data, filepath)
        logger.info(f"Saved ensemble to {filepath}")

    def load(self, filepath: str):
        """Load ensemble from disk."""
        save_data = joblib.load(filepath)
        self.ensemble_method = save_data['ensemble_method']
        self.weights = save_data['weights']
        self.meta_learners = save_data['meta_learners']
        self.is_fitted = save_data['is_fitted']
        logger.info(f"Loaded ensemble from {filepath}")


def create_simple_ensemble(
    usage_predictor,
    efficiency_predictor,
    weights: Optional[Dict[str, float]] = None
) -> EnsemblePredictor:
    """
    Create a simple weighted average ensemble.

    Args:
        usage_predictor: UsagePredictor instance
        efficiency_predictor: EfficiencyPredictor instance
        weights: Optional weights (default: equal)

    Returns:
        EnsemblePredictor configured for weighted averaging
    """
    if weights is None:
        # Equal weights by default
        weights = {'usage': 0.5, 'efficiency': 0.5}

    ensemble = EnsemblePredictor(
        base_models=[usage_predictor, efficiency_predictor],
        ensemble_method='weighted_avg',
        weights=weights
    )
    ensemble.is_fitted = True  # No training needed for weighted avg

    return ensemble


def train_stacking_ensemble(
    usage_predictor,
    efficiency_predictor,
    training_data: pd.DataFrame,
    targets: Dict[str, pd.Series]
) -> EnsemblePredictor:
    """
    Train a stacking ensemble with meta-learners.

    Args:
        usage_predictor: Fitted UsagePredictor
        efficiency_predictor: Fitted EfficiencyPredictor
        training_data: Training features
        targets: Dict of stat_name -> ground truth values

    Returns:
        Fitted EnsemblePredictor
    """
    ensemble = EnsemblePredictor(
        base_models=[usage_predictor, efficiency_predictor],
        ensemble_method='stacking'
    )

    # Generate base model predictions
    logger.info("Generating base model predictions for ensemble training...")

    base_predictions = {}

    # Usage predictor predictions
    if hasattr(usage_predictor, 'predict'):
        try:
            usage_preds = usage_predictor.predict(training_data)
            for stat_name, pred in usage_preds.items():
                if stat_name not in base_predictions:
                    base_predictions[stat_name] = pd.DataFrame()
                base_predictions[stat_name]['model_0_usage'] = pred
        except Exception as e:
            logger.warning(f"Failed to get usage predictions: {e}")

    # Efficiency predictor predictions
    if hasattr(efficiency_predictor, 'predict'):
        try:
            eff_preds = efficiency_predictor.predict(training_data)
            for stat_name, pred in eff_preds.items():
                if stat_name not in base_predictions:
                    base_predictions[stat_name] = pd.DataFrame()
                base_predictions[stat_name]['model_1_efficiency'] = pred
        except Exception as e:
            logger.warning(f"Failed to get efficiency predictions: {e}")

    # Train meta-learners
    ensemble.fit(training_data, targets, base_predictions)

    return ensemble


def compute_ensemble_weights_from_cv(
    cv_results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute ensemble weights from cross-validation performance.

    Args:
        cv_results: Dict of model_name -> {metric_name -> score}
            Example: {'usage': {'mae': 1.5}, 'efficiency': {'mae': 1.8}}

    Returns:
        Dict of model_name -> weight (inverse of error)
    """
    # Use inverse MAE as weights (lower MAE = higher weight)
    weights = {}
    total_inv_mae = 0.0

    for model_name, metrics in cv_results.items():
        if 'mae' in metrics:
            inv_mae = 1.0 / (metrics['mae'] + 0.01)  # Add epsilon to avoid div by zero
            weights[model_name] = inv_mae
            total_inv_mae += inv_mae
        else:
            weights[model_name] = 1.0

    # Normalize to sum to 1
    if total_inv_mae > 0:
        weights = {k: v / total_inv_mae for k, v in weights.items()}
    else:
        # Equal weights if no MAE available
        n_models = len(cv_results)
        weights = {k: 1.0 / n_models for k in cv_results.keys()}

    return weights
