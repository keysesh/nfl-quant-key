"""
Usage predictor - predicts player opportunities (snaps, targets, carries).

Uses XGBoost to predict:
- Snaps (all positions)
- Targets (QB, WR, TE, RB)
- Carries (QB, RB)

Based on:
- Historical player usage (trailing 4 weeks)
- Game context (projected score, game script, pace)
- Opponent strength
- Position and team
"""

import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from nfl_quant.features.feature_defaults import safe_fillna

logger = logging.getLogger(__name__)


class UsagePredictor:
    """Predict player usage (snaps, targets, carries) using XGBoost."""

    def __init__(self):
        """Initialize predictor with empty models."""
        self.models = {
            'snaps': None,
            'targets': None,
            'carries': None,
        }
        self.is_fitted = False
        self.feature_cols = []

    def prepare_training_data(self, pbp_df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
        """
        Prepare training data from play-by-play.

        Args:
            pbp_df: nflfastR play-by-play data
            min_games: Minimum games played to include player

        Returns:
            DataFrame with player-game rows and features
        """
        logger.info("Preparing usage training data...")

        # Get player-game aggregations
        player_games = []

        for (player_id, week), group in pbp_df.groupby(['receiver_player_id', 'week']):
            if pd.isna(player_id):
                continue

            # Count opportunities
            snaps = len(group)  # Approximation: plays involved
            targets = group['complete_pass'].count()

            player_games.append({
                'player_id': player_id,
                'week': week,
                'snaps': snaps,
                'targets': targets,
                'carries': 0,  # Will be filled for RBs
            })

        # Similar for rushers
        for (player_id, week), group in pbp_df[pbp_df['play_type'] == 'run'].groupby(['rusher_player_id', 'week']):
            if pd.isna(player_id):
                continue

            carries = len(group)

            # Check if player already in list (RBs catch passes too)
            existing = [pg for pg in player_games if pg['player_id'] == player_id and pg['week'] == week]
            if existing:
                existing[0]['carries'] = carries
                existing[0]['snaps'] += carries  # Add rushing snaps
            else:
                player_games.append({
                    'player_id': player_id,
                    'week': week,
                    'snaps': carries,
                    'targets': 0,
                    'carries': carries,
                })

        df = pd.DataFrame(player_games)

        # Calculate trailing averages (4-week rolling)
        df = df.sort_values(['player_id', 'week'])

        # Use EWMA for trailing averages (exponential weighting)
        # EWMA gives more weight to recent games: Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
        df['trailing_snaps'] = df.groupby('player_id')['snaps'].transform(
            lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
        )
        df['trailing_targets'] = df.groupby('player_id')['targets'].transform(
            lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
        )
        df['trailing_carries'] = df.groupby('player_id')['carries'].transform(
            lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
        )

        # Drop first game (no trailing data)
        df = df[df['week'] > 1].copy()

        # Filter to players with minimum games
        player_counts = df.groupby('player_id').size()
        valid_players = player_counts[player_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]

        logger.info(f"Prepared {len(df)} player-game samples from {len(valid_players)} players")

        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Extract features and targets from prepared data.

        Args:
            df: Prepared player-game DataFrame

        Returns:
            Tuple of (features, snaps_target, targets_target, carries_target)
        """
        feature_cols = [
            'trailing_snaps',
            'trailing_targets',
            'trailing_carries',
            'week',  # Week number (some seasonality)
        ]

        # Additional features if available
        if 'team_total' in df.columns:
            feature_cols.append('team_total')
        if 'game_script' in df.columns:
            feature_cols.append('game_script')
        if 'pace' in df.columns:
            feature_cols.append('pace')

        X = df[feature_cols].copy()

        # Fill NaN with semantic defaults
        X = safe_fillna(X)

        y_snaps = df['snaps']
        y_targets = df['targets']
        y_carries = df['carries']

        self.feature_cols = feature_cols

        return X, y_snaps, y_targets, y_carries

    def fit(self, X: pd.DataFrame, y_snaps: pd.Series, y_targets: pd.Series, y_carries: pd.Series):
        """
        Fit XGBoost models for snaps, targets, and carries.

        Args:
            X: Feature matrix
            y_snaps: Target snaps
            y_targets: Target targets
            y_carries: Target carries
        """
        logger.info("Fitting usage predictor models...")

        # XGBoost parameters (similar to nflfastR EP model)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }

        # Fit snaps model
        self.models['snaps'] = xgb.XGBRegressor(**params)
        self.models['snaps'].fit(X, y_snaps)
        logger.info(f"✓ Snaps model fitted (MAE: {mean_absolute_error(y_snaps, self.models['snaps'].predict(X)):.2f})")

        # Fit targets model (only for players with targets)
        target_mask = y_targets > 0
        if target_mask.sum() > 0:
            self.models['targets'] = xgb.XGBRegressor(**params)
            self.models['targets'].fit(X[target_mask], y_targets[target_mask])
            logger.info(f"✓ Targets model fitted (MAE: {mean_absolute_error(y_targets[target_mask], self.models['targets'].predict(X[target_mask])):.2f})")

        # Fit carries model (only for players with carries)
        carry_mask = y_carries > 0
        if carry_mask.sum() > 0:
            self.models['carries'] = xgb.XGBRegressor(**params)
            self.models['carries'].fit(X[carry_mask], y_carries[carry_mask])
            logger.info(f"✓ Carries model fitted (MAE: {mean_absolute_error(y_carries[carry_mask], self.models['carries'].predict(X[carry_mask])):.2f})")

        self.is_fitted = True

    def predict(self, features: pd.DataFrame, position: str = None) -> Dict[str, np.ndarray]:
        """
        Predict usage for new player-games.

        Args:
            features: Feature matrix (must have same columns as training)
            position: Player position ('QB', 'RB', 'WR', 'TE') - required for v4 models

        Returns:
            Dictionary with 'snaps', 'targets', 'carries' predictions
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        # Determine if this is a position-specific model (v4)
        is_position_specific = False
        if isinstance(self.models, dict) and self.models:
            first_key = list(self.models.keys())[0]
            if first_key in ['QB', 'RB', 'WR', 'TE', 'QB_rush']:
                is_position_specific = True

        # Handle position-specific feature columns (v4 models)
        if isinstance(self.feature_cols, dict):
            # Position-specific feature columns
            if position is None:
                raise ValueError("Position required when model has position-specific feature columns")

            # Handle TE -> WR mapping (TE uses WR models)
            feature_position = position if position in self.feature_cols else ('WR' if position == 'TE' else None)
            if feature_position is None or feature_position not in self.feature_cols:
                raise ValueError(f"No feature columns found for position {position} (tried {position} and WR)")

            feature_cols_to_use = self.feature_cols[feature_position]
        else:
            # Legacy: flat list of feature columns
            feature_cols_to_use = self.feature_cols

        # Ensure features match training with semantic defaults
        features_subset = safe_fillna(features[feature_cols_to_use])

        predictions = {}

        # Handle position-specific models (v4 models)
        if is_position_specific:
            # Models are position-specific
            if position is None:
                raise ValueError("Position required when model has position-specific models")

            # Handle TE -> WR mapping (TE uses WR models)
            model_position = position if position in self.models else ('WR' if position == 'TE' else None)
            if model_position is None or model_position not in self.models:
                raise ValueError(f"No models found for position {position} (tried {position} and WR)")

            models = self.models[model_position]
        else:
            # Legacy: models are flat dict keyed by model type
            models = self.models

        if models.get('snaps') is not None:
            predictions['snaps'] = models['snaps'].predict(features_subset)
        else:
            predictions['snaps'] = np.zeros(len(features_subset))

        # Handle targets prediction
        # For WR/TE/RB positions, the model key is 'attempts' (which represents targets)
        # Training script sets rb_games['attempts'] = rb_games['targets'] (line 133)
        # For other positions, use 'targets' directly
        if models.get('targets') is not None:
            predictions['targets'] = models['targets'].predict(features_subset)
        elif models.get('attempts') is not None and position in ['WR', 'TE', 'RB']:
            # Map 'attempts' to 'targets' for WR/TE/RB (all trained on receiving targets)
            predictions['targets'] = models['attempts'].predict(features_subset)
        else:
            predictions['targets'] = np.zeros(len(features_subset))

        if models.get('carries') is not None:
            predictions['carries'] = models['carries'].predict(features_subset)
        else:
            predictions['carries'] = np.zeros(len(features_subset))

        # Ensure non-negative
        for key in predictions:
            predictions[key] = np.maximum(predictions[key], 0)

        return predictions

    def save(self, filepath: str):
        """Save fitted models to disk."""
        if not self.is_fitted:
            raise ValueError("Models not fitted")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'models': self.models,
            'feature_cols': self.feature_cols,
        }

        joblib.dump(save_data, filepath)
        logger.info(f"✅ Saved usage predictor to {filepath}")

    def load(self, filepath: str):
        """Load fitted models from disk."""
        save_data = joblib.load(filepath)

        self.models = save_data['models']
        self.feature_cols = save_data['feature_cols']
        self.is_fitted = True

        logger.info(f"✅ Loaded usage predictor from {filepath}")


def train_usage_predictor(pbp_df: pd.DataFrame, save_path: Optional[str] = None) -> UsagePredictor:
    """
    Convenience function to train usage predictor from PBP data.

    Args:
        pbp_df: Play-by-play DataFrame
        save_path: Optional path to save fitted model

    Returns:
        Fitted UsagePredictor
    """
    predictor = UsagePredictor()

    # Prepare data
    training_data = predictor.prepare_training_data(pbp_df)

    # Extract features
    X, y_snaps, y_targets, y_carries = predictor.extract_features(training_data)

    # Train-test split
    X_train, X_test, y_snaps_train, y_snaps_test, y_targets_train, y_targets_test, y_carries_train, y_carries_test = train_test_split(
        X, y_snaps, y_targets, y_carries, test_size=0.2, random_state=42
    )

    # Fit
    predictor.fit(X_train, y_snaps_train, y_targets_train, y_carries_train)

    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("USAGE PREDICTOR VALIDATION")
    logger.info("="*60)

    test_preds = predictor.predict(X_test)

    for metric_name, y_test in [('Snaps', y_snaps_test), ('Targets', y_targets_test), ('Carries', y_carries_test)]:
        y_pred = test_preds[metric_name.lower()]
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"{metric_name}: MAE={mae:.2f}, R²={r2:.3f}")

    # Save if requested
    if save_path:
        predictor.save(save_path)

    return predictor
