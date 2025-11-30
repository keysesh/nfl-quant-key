"""
Efficiency predictor - predicts player efficiency metrics (yards per opportunity, TD rates).

Uses XGBoost to predict:
- Yards per target (WR, TE, RB)
- Yards per carry (RB, QB)
- Completion percentage (QB)
- Yards per completion (QB)
- TD rate per opportunity (all positions)

Based on:
- Historical player efficiency (trailing 4 weeks, regressed)
- Opponent defense strength vs position
- Game context (game script, pace)
- Weather conditions (if available)
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

logger = logging.getLogger(__name__)


class EfficiencyPredictor:
    """Predict player efficiency metrics using XGBoost."""

    def __init__(self):
        """Initialize predictor with empty models."""
        self.models = {
            'yards_per_target': None,
            'yards_per_carry': None,
            'completion_pct': None,
            'yards_per_completion': None,
            'td_rate_pass': None,
            'td_rate_rush': None,
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
            DataFrame with player-game rows and efficiency metrics
        """
        logger.info("Preparing efficiency training data...")

        # Passing efficiency (QB)
        pass_plays = pbp_df[pbp_df['play_type'] == 'pass'].copy()

        qb_efficiency = []
        for (player_id, week), group in pass_plays.groupby(['passer_player_id', 'week']):
            if pd.isna(player_id):
                continue

            attempts = len(group)
            if attempts < 5:  # Minimum attempts threshold
                continue

            completions = group['complete_pass'].sum()
            passing_yards = group['passing_yards'].sum()
            pass_tds = group['pass_touchdown'].sum()

            comp_pct = completions / attempts if attempts > 0 else 0
            yards_per_completion = passing_yards / completions if completions > 0 else 0
            td_rate = pass_tds / attempts if attempts > 0 else 0

            qb_efficiency.append({
                'player_id': player_id,
                'week': week,
                'position': 'QB',
                'attempts': attempts,
                'completions': completions,
                'comp_pct': comp_pct,
                'yards_per_completion': yards_per_completion,
                'td_rate_pass': td_rate,
            })

        # Receiving efficiency (WR, TE, RB)
        rec_efficiency = []
        for (player_id, week), group in pass_plays.groupby(['receiver_player_id', 'week']):
            if pd.isna(player_id):
                continue

            targets = len(group)
            if targets < 3:  # Minimum targets threshold
                continue

            receptions = group['complete_pass'].sum()
            rec_yards = group['receiving_yards'].sum()
            rec_tds = group['pass_touchdown'].sum()

            yards_per_target = rec_yards / targets if targets > 0 else 0
            td_rate = rec_tds / targets if targets > 0 else 0

            rec_efficiency.append({
                'player_id': player_id,
                'week': week,
                'position': 'WR',  # Will be updated with actual position
                'targets': targets,
                'yards_per_target': yards_per_target,
                'td_rate_pass': td_rate,
            })

        # Rushing efficiency (RB, QB)
        rush_plays = pbp_df[pbp_df['play_type'] == 'run'].copy()

        rush_efficiency = []
        for (player_id, week), group in rush_plays.groupby(['rusher_player_id', 'week']):
            if pd.isna(player_id):
                continue

            carries = len(group)
            if carries < 3:  # Minimum carries threshold
                continue

            rush_yards = group['rushing_yards'].sum()
            rush_tds = group['rush_touchdown'].sum()

            yards_per_carry = rush_yards / carries if carries > 0 else 0
            td_rate = rush_tds / carries if carries > 0 else 0

            rush_efficiency.append({
                'player_id': player_id,
                'week': week,
                'position': 'RB',  # Will be updated with actual position
                'carries': carries,
                'yards_per_carry': yards_per_carry,
                'td_rate_rush': td_rate,
            })

        # Combine all efficiency data
        qb_df = pd.DataFrame(qb_efficiency)
        rec_df = pd.DataFrame(rec_efficiency)
        rush_df = pd.DataFrame(rush_efficiency)

        # Merge receiving and rushing for RBs
        all_efficiency = []
        if len(qb_df) > 0:
            all_efficiency.append(qb_df)
        if len(rec_df) > 0:
            all_efficiency.append(rec_df)
        if len(rush_df) > 0:
            all_efficiency.append(rush_df)

        if len(all_efficiency) == 0:
            raise ValueError("No efficiency data available")

        df = pd.concat(all_efficiency, ignore_index=True)

        # Calculate trailing averages (4-week rolling)
        df = df.sort_values(['player_id', 'week'])

        # Trailing efficiency metrics - Use EWMA for exponential weighting
        # EWMA gives more weight to recent games: Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
        for col in ['comp_pct', 'yards_per_completion', 'yards_per_target', 'yards_per_carry', 'td_rate_pass', 'td_rate_rush']:
            if col in df.columns:
                df[f'trailing_{col}'] = df.groupby('player_id')[col].transform(
                    lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
                )

        # Drop first game (no trailing data)
        df = df[df['week'] > 1].copy()

        # Filter to players with minimum games
        player_counts = df.groupby('player_id').size()
        valid_players = player_counts[player_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]

        logger.info(f"Prepared {len(df)} player-game efficiency samples from {len(valid_players)} players")

        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Extract features and targets from prepared data.

        Args:
            df: Prepared player-game DataFrame

        Returns:
            Tuple of (features, targets_dict)
        """
        feature_cols = ['week']  # Week number for seasonality

        # Add trailing efficiency metrics
        for col in df.columns:
            if col.startswith('trailing_'):
                feature_cols.append(col)

        # Additional features if available
        if 'team_total' in df.columns:
            feature_cols.append('team_total')
        if 'game_script' in df.columns:
            feature_cols.append('game_script')
        if 'pace' in df.columns:
            feature_cols.append('pace')
        if 'opponent_def_epa' in df.columns:
            feature_cols.append('opponent_def_epa')

        X = df[feature_cols].copy()

        # Fill NaN with 0
        X = X.fillna(0)

        # Extract targets
        targets = {}
        if 'yards_per_target' in df.columns:
            targets['yards_per_target'] = df['yards_per_target']
        if 'yards_per_carry' in df.columns:
            targets['yards_per_carry'] = df['yards_per_carry']
        if 'comp_pct' in df.columns:
            targets['completion_pct'] = df['comp_pct']
        if 'yards_per_completion' in df.columns:
            targets['yards_per_completion'] = df['yards_per_completion']
        if 'td_rate_pass' in df.columns:
            targets['td_rate_pass'] = df['td_rate_pass']
        if 'td_rate_rush' in df.columns:
            targets['td_rate_rush'] = df['td_rate_rush']

        self.feature_cols = feature_cols

        return X, targets

    def fit(self, X: pd.DataFrame, targets: Dict[str, pd.Series]):
        """
        Fit XGBoost models for efficiency metrics.

        Args:
            X: Feature matrix
            targets: Dictionary of target series
        """
        logger.info("Fitting efficiency predictor models...")

        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,  # Slightly shallower for efficiency (less complex)
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }

        # Fit each efficiency model
        for metric_name, y_target in targets.items():
            if y_target is None or len(y_target) == 0:
                continue

            # Filter out NaN targets
            valid_mask = y_target.notna()
            if valid_mask.sum() == 0:
                continue

            X_valid = X[valid_mask]
            y_valid = y_target[valid_mask]

            # Fit model
            model = xgb.XGBRegressor(**params)
            model.fit(X_valid, y_valid)

            # Store model
            self.models[metric_name] = model

            # Calculate MAE
            y_pred = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)

            logger.info(f"✓ {metric_name} model fitted (MAE: {mae:.3f}, R²: {r2:.3f})")

        self.is_fitted = True

    def predict(self, features: pd.DataFrame, position: str = None, player_name: str = None) -> Dict[str, np.ndarray]:
        """
        Predict efficiency metrics for new player-games.

        Args:
            features: Feature matrix (must have same columns as training)
            position: Player position ('QB', 'RB', 'WR', 'TE') - required for v2_defense models
            player_name: Player name for debug logging (optional)

        Returns:
            Dictionary with efficiency predictions
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        # 🔍 DEBUG LOGGING: Log input features for first player
        if player_name and len(features) > 0:
            first_row = features.iloc[0]
            logger.info(f"🔍 EFFICIENCY DEBUG [{player_name}] Input Features:")

            # Log key features with interpretation
            for col in features.columns:
                value = first_row[col]
                if 'trailing_yards_per_carry' in col:
                    logger.info(f"   📊 {col}: {value:.3f} (input historical YPC)")
                elif 'trailing_yards_per_target' in col:
                    logger.info(f"   📊 {col}: {value:.3f} (input historical Y/Tgt)")
                elif 'opp_rush_def_epa' in col:
                    logger.info(f"   🛡️  {col}: {value:+.4f} (POSITIVE = WEAK defense)")
                elif 'opp_pass_def_epa' in col:
                    logger.info(f"   🛡️  {col}: {value:+.4f} (POSITIVE = WEAK defense)")
                elif col in ['week', 'team_pace', 'opp_rush_def_rank', 'opp_pass_def_rank']:
                    logger.info(f"   ℹ️  {col}: {value}")

        # Handle position-specific feature columns (v2_defense models)
        if isinstance(self.feature_cols, dict):
            # Position-specific feature columns
            if position is None:
                raise ValueError("Position required when model has position-specific feature columns")

            # Handle TE -> WR mapping and QB_rush -> QB mapping
            if position == 'TE':
                feature_position = 'WR'
            elif position == 'QB':
                # Try QB first, fall back to QB_rush if QB not available
                feature_position = 'QB' if 'QB' in self.feature_cols else ('QB_rush' if 'QB_rush' in self.feature_cols else None)
            else:
                feature_position = position if position in self.feature_cols else None

            if feature_position is None or feature_position not in self.feature_cols:
                raise ValueError(f"No feature columns found for position {position} (tried {position}, {'WR' if position == 'TE' else 'QB' if position == 'QB' else 'N/A'})")

            feature_cols_to_use = self.feature_cols[feature_position]
        else:
            # Legacy: flat list of feature columns
            feature_cols_to_use = self.feature_cols

        # Ensure features match training
        features_subset = features[feature_cols_to_use].fillna(0)

        predictions = {}

        # Filter models to only use ones for this position
        position_prefix = position if position != 'TE' else 'WR'  # TE uses WR models

        # Special handling for QB - separate passing and rushing models
        # QB passing models use QB features, QB rushing models use QB_rush features
        if position == 'QB':
            # Only use passing models (comp_pct, yards_per_completion, td_rate_pass)
            relevant_models = {
                metric_name: model
                for metric_name, model in self.models.items()
                if metric_name in ['QB_comp_pct', 'QB_yards_per_completion', 'QB_td_rate_pass']
            }
        elif position == 'QB_rush':
            # Only use rushing models (yards_per_carry, td_rate_rush)
            relevant_models = {
                metric_name: model
                for metric_name, model in self.models.items()
                if metric_name in ['QB_yards_per_carry', 'QB_td_rate_rush']
            }
        else:
            relevant_models = {
                metric_name: model
                for metric_name, model in self.models.items()
                if metric_name.startswith(position_prefix + '_')
            }

        # If no position-specific models found, use all models (legacy)
        if not relevant_models:
            relevant_models = self.models

        for metric_name, model in relevant_models.items():
            if model is not None:
                predictions[metric_name] = model.predict(features_subset)
            else:
                predictions[metric_name] = np.zeros(len(features_subset))

        # Map position-specific metric names to generic names
        # e.g., 'QB_comp_pct' -> 'completion_pct'
        generic_predictions = {}
        for metric_name, values in predictions.items():
            # Remove position prefix (e.g., 'QB_comp_pct' -> 'comp_pct')
            generic_name = metric_name.replace(position_prefix + '_', '')
            # Map to standard names
            if generic_name == 'comp_pct':
                generic_predictions['completion_pct'] = values
            elif generic_name == 'yards_per_completion':
                generic_predictions['yards_per_completion'] = values
            elif generic_name == 'td_rate_pass':
                generic_predictions['td_rate_pass'] = values
            elif generic_name == 'yards_per_carry':
                generic_predictions['yards_per_carry'] = values
            elif generic_name == 'td_rate_rush':
                generic_predictions['td_rate_rush'] = values
            elif generic_name == 'yards_per_target':
                generic_predictions['yards_per_target'] = values
            else:
                generic_predictions[generic_name] = values

        predictions = generic_predictions

        # 🔍 DEBUG LOGGING: Log predictions BEFORE clipping
        if player_name and len(features) > 0:
            for metric, values in predictions.items():
                if len(values) > 0:
                    pred_value = values[0]

                    # Get input trailing value for comparison
                    if metric == 'yards_per_carry' and 'trailing_yards_per_carry' in first_row:
                        input_val = first_row['trailing_yards_per_carry']
                        adjustment = pred_value - input_val
                        logger.info(f"   🎯 PREDICTED {metric}: {pred_value:.3f} (input: {input_val:.3f}, adjustment: {adjustment:+.3f})")

                        # 🚨 FLAG ISSUE if prediction is significantly below input
                        if adjustment < -1.0:
                            logger.warning(f"   🚨 WARNING: Large NEGATIVE adjustment ({adjustment:.3f}) for {metric}!")
                            logger.warning(f"      Model predicting {pred_value:.3f} despite input {input_val:.3f}")

                    elif metric == 'yards_per_target' and 'trailing_yards_per_target' in first_row:
                        input_val = first_row['trailing_yards_per_target']
                        adjustment = pred_value - input_val
                        logger.info(f"   🎯 PREDICTED {metric}: {pred_value:.3f} (input: {input_val:.3f}, adjustment: {adjustment:+.3f})")

                        if adjustment < -1.5:
                            logger.warning(f"   🚨 WARNING: Large NEGATIVE adjustment ({adjustment:.3f}) for {metric}!")
                    else:
                        logger.info(f"   🎯 PREDICTED {metric}: {pred_value:.3f}")

        # Ensure valid ranges
        # Completion percentage: 0-1
        if 'completion_pct' in predictions:
            predictions['completion_pct'] = np.clip(predictions['completion_pct'], 0, 1)

        # TD rates: 0-1
        for key in ['td_rate_pass', 'td_rate_rush']:
            if key in predictions:
                predictions[key] = np.clip(predictions[key], 0, 1)

        # Yards per opportunity: non-negative
        for key in ['yards_per_target', 'yards_per_carry', 'yards_per_completion']:
            if key in predictions:
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
        logger.info(f"✅ Saved efficiency predictor to {filepath}")

    def load(self, filepath: str):
        """Load fitted models from disk."""
        save_data = joblib.load(filepath)

        self.models = save_data['models']
        self.feature_cols = save_data['feature_cols']
        self.is_fitted = True

        logger.info(f"✅ Loaded efficiency predictor from {filepath}")


def train_efficiency_predictor(pbp_df: pd.DataFrame, save_path: Optional[str] = None) -> EfficiencyPredictor:
    """
    Convenience function to train efficiency predictor from PBP data.

    Args:
        pbp_df: Play-by-play DataFrame
        save_path: Optional path to save fitted model

    Returns:
        Fitted EfficiencyPredictor
    """
    predictor = EfficiencyPredictor()

    # Prepare data
    training_data = predictor.prepare_training_data(pbp_df)

    # Extract features
    X, targets = predictor.extract_features(training_data)

    # Train-test split (simple index-based split)
    test_size = int(len(X) * 0.2)
    train_indices = list(range(len(X) - test_size))
    test_indices = list(range(len(X) - test_size, len(X)))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    # Split targets
    targets_train = {}
    targets_test = {}

    for metric_name, y_series in targets.items():
        targets_train[metric_name] = y_series.iloc[train_indices]
        targets_test[metric_name] = y_series.iloc[test_indices]

    # Fit
    predictor.fit(X_train, targets_train)

    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("EFFICIENCY PREDICTOR VALIDATION")
    logger.info("="*60)

    test_preds = predictor.predict(X_test)

    for metric_name, y_test in targets_test.items():
        if metric_name not in test_preds:
            continue

        # Filter out NaN values for validation
        valid_mask = y_test.notna()
        if valid_mask.sum() == 0:
            continue

        y_test_valid = y_test[valid_mask]
        y_pred_valid = test_preds[metric_name][valid_mask]

        mae = mean_absolute_error(y_test_valid, y_pred_valid)
        r2 = r2_score(y_test_valid, y_pred_valid)
        logger.info(f"{metric_name}: MAE={mae:.3f}, R²={r2:.3f}")

    # Save if requested
    if save_path:
        predictor.save(save_path)

    return predictor
