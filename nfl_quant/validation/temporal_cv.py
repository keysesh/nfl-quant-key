"""
Temporal Cross-Validation for NFL Prediction Models

Provides walk-forward validation methods to prevent data leakage
and ensure models generalize to future weeks.

Key Principles:
- Never train on future data
- Respect temporal ordering of NFL weeks
- Use expanding or rolling window strategies
- Properly handle early-season low-data scenarios
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Iterator, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_id: int
    train_weeks: List[int]
    test_week: int
    train_size: int
    test_size: int


class TemporalCrossValidator:
    """
    Walk-forward cross-validation for time series data.

    Two strategies:
    1. Expanding window: Train on all data up to test week
    2. Rolling window: Train on fixed window (e.g., last 10 weeks)

    Example:
        >>> cv = TemporalCrossValidator(strategy='expanding', min_train_weeks=4)
        >>> for fold in cv.split(df, week_col='week'):
        >>>     train_data = df[df['week'].isin(fold.train_weeks)]
        >>>     test_data = df[df['week'] == fold.test_week]
        >>>     # Train and evaluate model
    """

    def __init__(
        self,
        strategy: str = 'expanding',
        min_train_weeks: int = 4,
        rolling_window: int = 10,
        test_from_week: int = 5,
        test_to_week: Optional[int] = None
    ):
        """
        Initialize temporal cross-validator.

        Args:
            strategy: 'expanding' or 'rolling'
            min_train_weeks: Minimum weeks needed for training
            rolling_window: Window size for rolling strategy
            test_from_week: First week to test on (default: 5)
            test_to_week: Last week to test on (None = all weeks)
        """
        if strategy not in ['expanding', 'rolling']:
            raise ValueError("strategy must be 'expanding' or 'rolling'")

        self.strategy = strategy
        self.min_train_weeks = min_train_weeks
        self.rolling_window = rolling_window
        self.test_from_week = test_from_week
        self.test_to_week = test_to_week

    def split(
        self,
        df: pd.DataFrame,
        week_col: str = 'week'
    ) -> Iterator[CVFold]:
        """
        Generate train/test splits in temporal order.

        Args:
            df: DataFrame with temporal data
            week_col: Column name containing week numbers

        Yields:
            CVFold objects with train/test week information
        """
        weeks = sorted(df[week_col].unique())

        # Determine test weeks
        test_weeks = [
            w for w in weeks
            if w >= self.test_from_week
            and (self.test_to_week is None or w <= self.test_to_week)
        ]

        fold_id = 0
        for test_week in test_weeks:
            # Get training weeks based on strategy
            if self.strategy == 'expanding':
                # All weeks before test week
                train_weeks = [w for w in weeks if w < test_week]
            else:  # rolling
                # Last N weeks before test week
                start_week = max(
                    min(weeks),
                    test_week - self.rolling_window
                )
                train_weeks = [
                    w for w in weeks
                    if start_week <= w < test_week
                ]

            # Check minimum training size
            if len(train_weeks) < self.min_train_weeks:
                logger.debug(
                    f"Skipping week {test_week}: "
                    f"only {len(train_weeks)} training weeks available"
                )
                continue

            # Count samples
            train_size = len(df[df[week_col].isin(train_weeks)])
            test_size = len(df[df[week_col] == test_week])

            if test_size == 0:
                continue

            fold = CVFold(
                fold_id=fold_id,
                train_weeks=train_weeks,
                test_week=test_week,
                train_size=train_size,
                test_size=test_size
            )

            fold_id += 1
            yield fold

    def get_n_splits(self, df: pd.DataFrame, week_col: str = 'week') -> int:
        """Get number of splits."""
        return sum(1 for _ in self.split(df, week_col))


class TemporalHyperparameterTuner:
    """
    Tune hyperparameters using temporal cross-validation.

    Example:
        >>> tuner = TemporalHyperparameterTuner(
        >>>     cv=TemporalCrossValidator(),
        >>>     metric='accuracy'
        >>> )
        >>> best_params = tuner.tune(
        >>>     df, X_cols, y_col,
        >>>     param_grid={'max_depth': [3, 5, 7]}
        >>> )
    """

    def __init__(
        self,
        cv: TemporalCrossValidator,
        metric: str = 'mae'
    ):
        """
        Initialize tuner.

        Args:
            cv: TemporalCrossValidator instance
            metric: Metric to optimize ('mae', 'rmse', 'accuracy', 'auc')
        """
        self.cv = cv
        self.metric = metric

    def evaluate_params(
        self,
        df: pd.DataFrame,
        X_cols: List[str],
        y_col: str,
        params: Dict,
        model_class,
        week_col: str = 'week'
    ) -> Tuple[float, List[float]]:
        """
        Evaluate parameters using cross-validation.

        Args:
            df: Training data
            X_cols: Feature columns
            y_col: Target column
            params: Model parameters to test
            model_class: Model class to instantiate
            week_col: Week column name

        Returns:
            Tuple of (mean_score, fold_scores)
        """
        fold_scores = []

        for fold in self.cv.split(df, week_col):
            # Get train/test data
            train_data = df[df[week_col].isin(fold.train_weeks)]
            test_data = df[df[week_col] == fold.test_week]

            # Train model
            model = model_class(**params)
            X_train = train_data[X_cols].fillna(0)
            y_train = train_data[y_col]
            model.fit(X_train, y_train)

            # Evaluate
            X_test = test_data[X_cols].fillna(0)
            y_test = test_data[y_col]
            y_pred = model.predict(X_test)

            # Calculate score
            score = self._calculate_score(y_test, y_pred)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        return mean_score, fold_scores

    def _calculate_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate evaluation score."""
        if self.metric == 'mae':
            return -np.mean(np.abs(y_true - y_pred))  # Negative for minimization
        elif self.metric == 'rmse':
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.metric == 'accuracy':
            return np.mean(y_true == np.round(y_pred))
        elif self.metric == 'auc':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


def evaluate_model_temporal(
    model,
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str,
    week_col: str = 'week',
    strategy: str = 'expanding',
    min_train_weeks: int = 4
) -> pd.DataFrame:
    """
    Evaluate model using temporal cross-validation.

    Args:
        model: Fitted model or model class
        df: Data with temporal structure
        X_cols: Feature columns
        y_col: Target column
        week_col: Week column name
        strategy: 'expanding' or 'rolling'
        min_train_weeks: Minimum training weeks

    Returns:
        DataFrame with per-week evaluation results
    """
    cv = TemporalCrossValidator(
        strategy=strategy,
        min_train_weeks=min_train_weeks
    )

    results = []

    for fold in cv.split(df, week_col):
        # Get data
        train_data = df[df[week_col].isin(fold.train_weeks)]
        test_data = df[df[week_col] == fold.test_week]

        # Train
        X_train = train_data[X_cols].fillna(0)
        y_train = train_data[y_col]
        model.fit(X_train, y_train)

        # Predict
        X_test = test_data[X_cols].fillna(0)
        y_test = test_data[y_col]
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        # Calculate R² (coefficient of determination)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results.append({
            'test_week': fold.test_week,
            'train_weeks': len(fold.train_weeks),
            'train_size': fold.train_size,
            'test_size': fold.test_size,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL CROSS-VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Total folds: {len(results_df)}")
    logger.info(f"Mean MAE: {results_df['mae'].mean():.3f}")
    logger.info(f"Mean RMSE: {results_df['rmse'].mean():.3f}")
    logger.info(f"Mean R²: {results_df['r2'].mean():.3f}")
    logger.info(f"MAE std: {results_df['mae'].std():.3f}")
    logger.info("="*60)

    return results_df


def get_optimal_window_size(
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str,
    model_class,
    window_sizes: List[int] = [4, 6, 8, 10, 12],
    week_col: str = 'week'
) -> int:
    """
    Find optimal rolling window size using cross-validation.

    Args:
        df: Training data
        X_cols: Feature columns
        y_col: Target column
        model_class: Model class to test
        window_sizes: Window sizes to test
        week_col: Week column name

    Returns:
        Optimal window size
    """
    results = []

    for window_size in window_sizes:
        cv = TemporalCrossValidator(
            strategy='rolling',
            rolling_window=window_size,
            min_train_weeks=min(4, window_size)
        )

        fold_errors = []
        for fold in cv.split(df, week_col):
            train_data = df[df[week_col].isin(fold.train_weeks)]
            test_data = df[df[week_col] == fold.test_week]

            model = model_class()
            X_train = train_data[X_cols].fillna(0)
            y_train = train_data[y_col]
            model.fit(X_train, y_train)

            X_test = test_data[X_cols].fillna(0)
            y_test = test_data[y_col]
            y_pred = model.predict(X_test)

            mae = np.mean(np.abs(y_test - y_pred))
            fold_errors.append(mae)

        mean_error = np.mean(fold_errors)
        results.append({
            'window_size': window_size,
            'mean_mae': mean_error,
            'n_folds': len(fold_errors)
        })

        logger.info(
            f"Window {window_size:2d}: MAE = {mean_error:.3f} "
            f"({len(fold_errors)} folds)"
        )

    results_df = pd.DataFrame(results)
    optimal_window = results_df.loc[results_df['mean_mae'].idxmin(), 'window_size']

    logger.info(f"\nOptimal window size: {optimal_window}")
    return int(optimal_window)
