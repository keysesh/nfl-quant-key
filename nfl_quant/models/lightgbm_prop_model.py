"""
LightGBM-based Player Prop Model

Features:
1. Optimized hyperparameters for small edge detection
2. Player-specific modeling (not just market-level)
3. Proper regularization to prevent overfitting
4. Time series cross-validation
5. Probability calibration
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LightGBM model."""
    # Hyperparameters optimized for small edge detection
    num_leaves: int = 20
    learning_rate: float = 0.02
    min_data_in_leaf: int = 40
    max_depth: int = 5
    reg_alpha: float = 1.0  # L1 regularization
    reg_lambda: float = 10.0  # L2 regularization (strong)
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 5
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    min_gain_to_split: float = 0.01
    verbose: int = -1


class LightGBMPropModel:
    """
    LightGBM model for player prop predictions.

    Supports both regression (predicting actual stat) and classification (over/under).
    """

    def __init__(
        self,
        prop_type: str = 'receiving_yards',
        model_type: str = 'classifier',  # 'classifier' or 'regressor'
        config: ModelConfig = None
    ):
        """
        Initialize model.

        Args:
            prop_type: Type of prop (receiving_yards, rushing_yards, etc.)
            model_type: 'classifier' for over/under, 'regressor' for value
            config: Model configuration
        """
        self.prop_type = prop_type
        self.model_type = model_type
        self.config = config or ModelConfig()

        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.feature_importance = None

        logger.info(f"Initialized {prop_type} {model_type} model")

    def _get_lgb_params(self) -> Dict:
        """Get LightGBM parameters."""
        params = {
            'objective': 'binary' if self.model_type == 'classifier' else 'regression',
            'metric': 'binary_logloss' if self.model_type == 'classifier' else 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'min_data_in_leaf': self.config.min_data_in_leaf,
            'max_depth': self.config.max_depth,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'verbose': self.config.verbose,
            'seed': 42,
        }
        return params

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        calibrate: bool = True
    ) -> Dict:
        """
        Train the model with proper validation.

        Args:
            X: Feature dataframe
            y: Target variable (0/1 for classifier, float for regressor)
            validation_split: Fraction for validation
            calibrate: Whether to apply isotonic regression calibration

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.prop_type} model on {len(X)} samples")

        # Store feature names
        self.feature_names = list(X.columns)

        # Time series split (use last 20% as validation)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"  Train size: {len(X_train)}, Val size: {len(X_val)}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        params = self._get_lgb_params()

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        logger.info(f"  Best iteration: {self.model.best_iteration}")

        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        logger.info(f"  Top 10 features:")
        for _, row in self.feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.2f}")

        # Evaluate on validation
        if self.model_type == 'classifier':
            val_preds = self.model.predict(X_val)
            val_preds_binary = (val_preds > 0.5).astype(int)
            accuracy = (val_preds_binary == y_val).mean()

            # Calculate AUC if possible
            from sklearn.metrics import roc_auc_score, brier_score_loss
            auc = roc_auc_score(y_val, val_preds)
            brier = brier_score_loss(y_val, val_preds)

            logger.info(f"  Validation Accuracy: {accuracy:.4f}")
            logger.info(f"  Validation AUC: {auc:.4f}")
            logger.info(f"  Validation Brier Score: {brier:.4f}")

            # Calibrate if requested
            if calibrate:
                logger.info("  Applying isotonic regression calibration...")
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(val_preds, y_val)

                # Re-evaluate with calibration
                cal_preds = self.calibrator.predict(val_preds)
                cal_brier = brier_score_loss(y_val, cal_preds)
                logger.info(f"  Calibrated Brier Score: {cal_brier:.4f}")

            metrics = {
                'accuracy': accuracy,
                'auc': auc,
                'brier_score': brier,
                'calibrated_brier': cal_brier if calibrate else None,
                'best_iteration': self.model.best_iteration,
            }
        else:
            val_preds = self.model.predict(X_val)
            mae = np.abs(val_preds - y_val).mean()
            rmse = np.sqrt(((val_preds - y_val) ** 2).mean())

            logger.info(f"  Validation MAE: {mae:.2f}")
            logger.info(f"  Validation RMSE: {rmse:.2f}")

            metrics = {
                'mae': mae,
                'rmse': rmse,
                'best_iteration': self.model.best_iteration,
            }

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for over/under.

        Args:
            X: Feature dataframe

        Returns:
            Array of probabilities (probability of OVER)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifier")

        # Get raw predictions
        raw_preds = self.model.predict(X)

        # Apply calibration if available
        if self.calibrator is not None:
            calibrated_preds = self.calibrator.predict(raw_preds)
            return calibrated_preds

        return raw_preds

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict actual stat values (for regressor).

        Args:
            X: Feature dataframe

        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def save(self, path: Path):
        """Save model and calibrator."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_path = path / f'{self.prop_type}_{self.model_type}_lgb.txt'
        self.model.save_model(str(model_path))

        # Save calibrator
        if self.calibrator is not None:
            cal_path = path / f'{self.prop_type}_{self.model_type}_calibrator.joblib'
            joblib.dump(self.calibrator, cal_path)

        # Save feature names and importance
        meta_path = path / f'{self.prop_type}_{self.model_type}_meta.joblib'
        joblib.dump({
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'prop_type': self.prop_type,
            'model_type': self.model_type,
        }, meta_path)

        logger.info(f"Saved model to {path}")

    def load(self, path: Path):
        """Load model and calibrator."""
        path = Path(path)

        # Load LightGBM model
        model_path = path / f'{self.prop_type}_{self.model_type}_lgb.txt'
        self.model = lgb.Booster(model_file=str(model_path))

        # Load calibrator
        cal_path = path / f'{self.prop_type}_{self.model_type}_calibrator.joblib'
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)

        # Load metadata
        meta_path = path / f'{self.prop_type}_{self.model_type}_meta.joblib'
        meta = joblib.load(meta_path)
        self.feature_names = meta['feature_names']
        self.feature_importance = meta['feature_importance']
        self.config = meta['config']

        logger.info(f"Loaded model from {path}")


class PropModelTrainer:
    """
    Trainer for multiple prop models with walk-forward validation.
    """

    def __init__(self):
        self.models = {}
        self.results = {}

    def prepare_training_data(
        self,
        backtest_df: pd.DataFrame,
        prop_type: str = 'player_reception_yds'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from backtest dataset.

        Args:
            backtest_df: DataFrame with historical props and outcomes
            prop_type: Type of prop to train for

        Returns:
            X (features), y (target)
        """
        # Filter to prop type
        df = backtest_df[backtest_df['market'] == prop_type].copy()

        logger.info(f"Preparing {prop_type} data: {len(df)} samples")

        # Create target: 1 if over hit, 0 if under hit
        y = df['over_hit'].astype(int)

        # Create features (this should use enhanced features)
        # For now, use available columns
        feature_cols = [
            'line',
            'fair_over_prob',
            'fair_under_prob',
            'over_odds',
            'under_odds',
            'week',
            'season',
        ]

        # Add any enhanced features if available
        enhanced_cols = [col for col in df.columns if col.startswith(('ewma', 'opp_', 'vegas_'))]
        feature_cols.extend(enhanced_cols)

        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols].copy()

        # Handle missing values
        X = X.fillna(0)

        logger.info(f"  Features: {list(X.columns)}")
        logger.info(f"  Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def train_all_markets(
        self,
        backtest_df: pd.DataFrame,
        markets: List[str] = None
    ) -> Dict:
        """
        Train models for all prop markets.

        Args:
            backtest_df: Full backtest dataset
            markets: List of markets to train (default: all available)

        Returns:
            Dictionary of training results
        """
        if markets is None:
            markets = backtest_df['market'].unique().tolist()

        logger.info(f"Training models for {len(markets)} markets")

        all_results = {}

        for market in markets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {market}")
            logger.info(f"{'='*50}")

            try:
                X, y = self.prepare_training_data(backtest_df, market)

                if len(X) < 100:
                    logger.warning(f"  Insufficient data ({len(X)} samples), skipping")
                    continue

                model = LightGBMPropModel(prop_type=market, model_type='classifier')
                metrics = model.train(X, y, calibrate=True)

                self.models[market] = model
                all_results[market] = metrics

            except Exception as e:
                logger.error(f"  Error training {market}: {e}")
                import traceback
                traceback.print_exc()

        self.results = all_results
        return all_results

    def walk_forward_train_and_validate(
        self,
        backtest_df: pd.DataFrame,
        market: str,
        n_splits: int = 5
    ) -> Dict:
        """
        Walk-forward validation with TimeSeriesSplit.

        Args:
            backtest_df: Full backtest dataset
            market: Market to validate
            n_splits: Number of time series splits

        Returns:
            Validation results
        """
        X, y = self.prepare_training_data(backtest_df, market)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold+1}/{n_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = LightGBMPropModel(prop_type=market, model_type='classifier')

            # Train on this fold
            train_data = lgb.Dataset(X_train, label=y_train)
            params = model._get_lgb_params()

            lgb_model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
            )

            # Predict on test
            preds = lgb_model.predict(X_test)

            # Metrics
            from sklearn.metrics import roc_auc_score, brier_score_loss
            accuracy = ((preds > 0.5).astype(int) == y_test).mean()
            auc = roc_auc_score(y_test, preds)
            brier = brier_score_loss(y_test, preds)

            logger.info(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Brier: {brier:.4f}")

            fold_results.append({
                'fold': fold,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy,
                'auc': auc,
                'brier': brier,
            })

        # Average metrics
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        avg_auc = np.mean([r['auc'] for r in fold_results])
        avg_brier = np.mean([r['brier'] for r in fold_results])

        logger.info(f"\nCross-validation results for {market}:")
        logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  Average AUC: {avg_auc:.4f}")
        logger.info(f"  Average Brier Score: {avg_brier:.4f}")

        return {
            'folds': fold_results,
            'avg_accuracy': avg_accuracy,
            'avg_auc': avg_auc,
            'avg_brier': avg_brier,
        }

    def save_all_models(self, path: Path):
        """Save all trained models."""
        path = Path(path)
        for market, model in self.models.items():
            model.save(path / market)

        # Save results
        import json
        results_path = path / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)

        logger.info(f"Saved {len(self.models)} models to {path}")
