"""
Player Bias Edge Implementation

The Player Bias edge captures persistent player tendencies.
Some players consistently go over or under their lines due to:
- Vegas systematically over/undervaluing them
- Usage patterns not fully captured in lines
- Consistency/variance profiles

Key characteristics:
- Uses player-specific features (11 features)
- Lower confidence threshold (~55-57%)
- Higher volume than LVT
- Target: 55-60% hit rate
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base_edge import BaseEdge
from .edge_calibrator import EdgeCalibrator
from configs.edge_config import (
    PLAYER_BIAS_FEATURES,
    PLAYER_BIAS_THRESHOLDS,
    PLAYER_BIAS_MODEL_PARAMS,
    get_player_bias_threshold,
    EDGE_EWMA_SPAN,
)


class PlayerBiasEdge(BaseEdge):
    """
    Player Bias Edge Implementation.

    This edge triggers when:
    1. Player has sufficient betting history (min_bets)
    2. Player shows strong directional bias (under_rate > min_rate or < 1-min_rate)
    3. Model P(UNDER) > confidence threshold

    Uses 19 features: 13 core (player tendencies) + 6 V23/V28 (context).
    """

    def __init__(self):
        """Initialize Player Bias edge with configured features and thresholds."""
        super().__init__(
            name="player_bias",
            features=PLAYER_BIAS_FEATURES,
            thresholds=PLAYER_BIAS_THRESHOLDS,
        )
        self.scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.calibrator: EdgeCalibrator = EdgeCalibrator()

    def extract_features(
        self,
        df: pd.DataFrame,
        market: str,
        historical_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract player-specific features.

        Extracts 19 features for Player Bias edge:
        Core (13):
        - player_under_rate, player_bias (core signals)
        - Alignment features (LVT_x_player_*)
        - Usage features (target_share, snap_share, etc.)
        - Context (pos_rank, is_starter, market_bias_strength)
        - Season stats (current_season_under_rate, season_games_played)

        V23/V28 (6 - Dec 2025):
        - opp_pass_yds_def_vs_avg, opp_rush_yds_def_vs_avg, opp_def_epa
        - rest_days, injury_status_encoded, has_injury_designation

        Args:
            df: DataFrame with player betting data
            market: Market being processed
            historical_data: Optional historical data for computing player stats

        Returns:
            DataFrame with 19 player bias features
        """
        result = pd.DataFrame(index=df.index)

        # 1. player_under_rate - Core signal (% times player goes under)
        if 'player_under_rate' in df.columns:
            result['player_under_rate'] = df['player_under_rate']
        else:
            result['player_under_rate'] = 0.50  # Neutral default

        # 2. player_bias - Average (actual - line)
        if 'player_bias' in df.columns:
            result['player_bias'] = df['player_bias']
        else:
            result['player_bias'] = 0.0

        # 3. LVT_x_player_tendency - Does LVT align with player tendency?
        lvt = df.get('line_vs_trailing', 0)
        player_tendency = result['player_under_rate'] - 0.5  # Center around 0
        if 'LVT_x_player_tendency' in df.columns:
            result['LVT_x_player_tendency'] = df['LVT_x_player_tendency']
        else:
            result['LVT_x_player_tendency'] = lvt * player_tendency

        # 4. LVT_x_player_bias - Does LVT align with bias direction?
        if 'LVT_x_player_bias' in df.columns:
            result['LVT_x_player_bias'] = df['LVT_x_player_bias']
        else:
            result['LVT_x_player_bias'] = lvt * result['player_bias']

        # 5. player_market_aligned - Player tendency vs market regime
        market_rate = df.get('market_under_rate', 0.50)
        if 'player_market_aligned' in df.columns:
            result['player_market_aligned'] = df['player_market_aligned']
        else:
            # 1 if both favor same direction, -1 if opposite
            result['player_market_aligned'] = np.where(
                (result['player_under_rate'] > 0.5) == (market_rate > 0.5),
                1.0,
                -1.0
            )

        # 6. target_share - Receiving opportunity
        result['target_share'] = df.get('target_share', 0.15)

        # 7. snap_share - Playing time
        result['snap_share'] = df.get('snap_share', 0.70)

        # 8. trailing_catch_rate - Efficiency
        result['trailing_catch_rate'] = df.get('trailing_catch_rate', 0.65)

        # 9. pos_rank - Depth chart position (1=WR1, 2=WR2, etc.)
        result['pos_rank'] = df.get('pos_rank', 2)

        # 10. is_starter - Binary starter flag
        if 'is_starter' in df.columns:
            result['is_starter'] = df['is_starter']
        else:
            result['is_starter'] = (result['pos_rank'] == 1).astype(int)

        # 11. market_bias_strength - How strong is market regime?
        if 'market_bias_strength' in df.columns:
            result['market_bias_strength'] = df['market_bias_strength']
        else:
            result['market_bias_strength'] = abs(market_rate - 0.5) * 2

        # 12. current_season_under_rate - Current season bias
        result['current_season_under_rate'] = df.get('current_season_under_rate', 0.50)

        # 13. season_games_played - Sample size for current season
        result['season_games_played'] = df.get('season_games_played', 0)

        # =====================================================================
        # V23/V28 Features (Dec 2025 Enhancement)
        # =====================================================================

        # 14. opp_pass_yds_def_vs_avg - Pass defense z-score
        result['opp_pass_yds_def_vs_avg'] = df.get('opp_pass_yds_def_vs_avg', 0.0)
        if isinstance(result['opp_pass_yds_def_vs_avg'], pd.Series):
            result['opp_pass_yds_def_vs_avg'] = result['opp_pass_yds_def_vs_avg'].fillna(0.0)

        # 15. opp_rush_yds_def_vs_avg - Rush defense z-score
        result['opp_rush_yds_def_vs_avg'] = df.get('opp_rush_yds_def_vs_avg', 0.0)
        if isinstance(result['opp_rush_yds_def_vs_avg'], pd.Series):
            result['opp_rush_yds_def_vs_avg'] = result['opp_rush_yds_def_vs_avg'].fillna(0.0)

        # 16. opp_def_epa - Overall defense EPA
        result['opp_def_epa'] = df.get('opp_def_epa', 0.0)
        if isinstance(result['opp_def_epa'], pd.Series):
            result['opp_def_epa'] = result['opp_def_epa'].fillna(0.0)

        # 17. rest_days - Days since last game
        result['rest_days'] = df.get('rest_days', 7.0)
        if isinstance(result['rest_days'], pd.Series):
            result['rest_days'] = result['rest_days'].fillna(7.0)

        # 18. injury_status_encoded - 0=None, 1=Quest, 2=Doubt, 3=Out
        result['injury_status_encoded'] = df.get('injury_status_encoded', 0)
        if isinstance(result['injury_status_encoded'], pd.Series):
            result['injury_status_encoded'] = result['injury_status_encoded'].fillna(0).astype(int)

        # 19. has_injury_designation - Binary flag
        if 'has_injury_designation' in df.columns:
            result['has_injury_designation'] = df['has_injury_designation']
        else:
            result['has_injury_designation'] = (result['injury_status_encoded'] > 0).astype(int)

        return result[PLAYER_BIAS_FEATURES]

    def should_trigger(
        self,
        row: pd.Series,
        market: str,
    ) -> bool:
        """
        Check if Player Bias edge triggers.

        Triggers when:
        1. Player has sufficient betting history (min_bets)
        2. Player has strong directional bias (under_rate > min_rate or < 1-min_rate)

        Args:
            row: Series with player bias features
            market: Market being evaluated

        Returns:
            True if player bias edge conditions are met
        """
        threshold = get_player_bias_threshold(market)

        # Check player has enough history
        bet_count = row.get('player_bet_count', 0)
        if bet_count < threshold.min_bets:
            return False

        # Check player has strong bias
        under_rate = row.get('player_under_rate', 0.5)

        # Strong under bias OR strong over bias
        has_strong_bias = (
            under_rate >= threshold.min_rate or  # Strong UNDER tendency
            under_rate <= (1 - threshold.min_rate)  # Strong OVER tendency
        )

        return has_strong_bias

    def get_direction(
        self,
        row: pd.Series,
        market: str,
    ) -> str:
        """
        Get recommended direction based on player's historical tendency.

        For Player Bias edge, direction is determined by the player's
        actual under_rate, NOT the model's confidence. This ensures we
        bet in the direction of the player's observed tendency.

        Args:
            row: Series with features for a single bet
            market: Market being evaluated

        Returns:
            "UNDER" if player tends to go under, else "OVER"
        """
        threshold = get_player_bias_threshold(market)
        under_rate = row.get('player_under_rate', 0.5)

        # If player has strong UNDER tendency (>= min_rate), bet UNDER
        # If player has strong OVER tendency (<= 1 - min_rate), bet OVER
        if under_rate >= threshold.min_rate:
            return "UNDER"
        elif under_rate <= (1 - threshold.min_rate):
            return "OVER"
        else:
            # Shouldn't reach here if should_trigger was called first
            return "UNDER" if under_rate > 0.5 else "OVER"

    def get_confidence(
        self,
        row: pd.Series,
        market: str,
    ) -> float:
        """
        Get calibrated P(UNDER) confidence.

        Args:
            row: Series with player bias features
            market: Market being evaluated

        Returns:
            Calibrated P(UNDER) from trained model, or player_under_rate if no model
        """
        if market not in self.models:
            # Fallback: use player_under_rate directly
            return row.get('player_under_rate', 0.5)

        # Use trained model
        model = self.models[market]
        features = row[PLAYER_BIAS_FEATURES].values.reshape(1, -1)

        # Apply preprocessing
        if market in self.imputers:
            features = self.imputers[market].transform(features)
        if market in self.scalers:
            features = self.scalers[market].transform(features)

        probs = model.predict_proba(features)
        raw_prob = float(probs[0, 1])  # P(UNDER)

        # NOTE: Calibration disabled - IsotonicRegression was squashing outputs
        # to constant values regardless of input features.
        # Raw probabilities provide better differentiation.
        return raw_prob

    def train(
        self,
        train_data: pd.DataFrame,
        market: str,
        validation_weeks: int = 10,
    ) -> Dict[str, Any]:
        """
        Train Player Bias edge model for a market.

        Key differences from unified model:
        1. Filter to players with sufficient betting history
        2. Use 11 player-specific features
        3. Focus on players with strong directional bias

        Args:
            train_data: Historical odds/actuals with player stats
            market: Market to train for
            validation_weeks: Weeks for walk-forward validation

        Returns:
            Training metrics
        """
        threshold = get_player_bias_threshold(market)

        # Compute player betting history if not present
        train_data = self._compute_player_history(train_data.copy(), market)

        # Extract features
        features_df = self.extract_features(train_data, market)
        for col in features_df.columns:
            train_data[col] = features_df[col].values

        # Filter to players with sufficient history
        if 'player_bet_count' in train_data.columns:
            has_history = train_data['player_bet_count'] >= threshold.min_bets
        else:
            # Assume all rows have history if we can't check
            has_history = pd.Series(True, index=train_data.index)

        # Filter to players with strong bias
        has_strong_bias = (
            (train_data['player_under_rate'] >= threshold.min_rate) |
            (train_data['player_under_rate'] <= (1 - threshold.min_rate))
        )

        filtered_data = train_data[has_history & has_strong_bias].copy()

        if len(filtered_data) < 50:
            raise ValueError(
                f"Insufficient samples for {market}: "
                f"{len(filtered_data)} (need 50+)"
            )

        # Prepare features and target
        X = filtered_data[PLAYER_BIAS_FEATURES]
        y = filtered_data['under_hit']

        # Create preprocessing pipeline
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        self.imputers[market] = imputer
        self.scalers[market] = scaler

        # Split data: 80% train, 20% calibration using TEMPORAL split (no leakage)
        # FIXED: Use time-based split instead of random split to prevent future data leak
        if len(X_scaled) >= 100:
            # Ensure we have global_week for temporal ordering
            if 'global_week' not in filtered_data.columns:
                filtered_data['global_week'] = (
                    (filtered_data['season'] - 2023) * 18 + filtered_data['week']
                )

            # Sort by time and split temporally (first 80% train, last 20% calibration)
            sorted_indices = filtered_data.sort_values('global_week').index
            n_train = int(len(sorted_indices) * 0.8)

            train_indices = sorted_indices[:n_train]
            calib_indices = sorted_indices[n_train:]

            # Get position indices for X_scaled array
            train_pos = [filtered_data.index.get_loc(idx) for idx in train_indices]
            calib_pos = [filtered_data.index.get_loc(idx) for idx in calib_indices]

            X_train = X_scaled[train_pos]
            X_calib = X_scaled[calib_pos]
            y_train = y.iloc[train_pos].values
            y_calib = y.iloc[calib_pos].values
        else:
            # Small sample: use all for training, skip calibration
            X_train, X_calib, y_train, y_calib = X_scaled, None, y.values, None

        # Train model on TRAIN SPLIT ONLY
        model = xgb.XGBClassifier(**PLAYER_BIAS_MODEL_PARAMS)
        model.fit(X_train, y_train)

        self.models[market] = model

        # Calculate metrics on TRAINING SET ONLY (not full data)
        train_preds = model.predict_proba(X_train)[:, 1]
        train_accuracy = ((train_preds > 0.5) == y_train).mean()

        # Calculate TEST accuracy on held-out calibration set (honest estimate)
        test_accuracy = None
        if X_calib is not None and len(X_calib) >= 20:
            test_preds = model.predict_proba(X_calib)[:, 1]
            test_accuracy = ((test_preds > 0.5) == y_calib).mean()
            print(f"  {market}: Train={train_accuracy:.1%} | Test={test_accuracy:.1%} (gap={train_accuracy - test_accuracy:.1%})")
        else:
            print(f"  {market}: Train={train_accuracy:.1%} | Test=N/A (insufficient holdout)")

        # Fit calibrator on HELD-OUT CALIBRATION SET (NOT training data)
        try:
            if X_calib is not None and len(X_calib) >= 50:
                calib_preds = model.predict_proba(X_calib)[:, 1]
                y_calib_arr = y_calib.values if hasattr(y_calib, 'values') else y_calib
                calib_metrics = self.calibrator.fit(market, y_calib_arr, calib_preds)
                print(f"  Calibration: ECE improved by {calib_metrics['ece_improvement']:.1f}% (on held-out set)")
            else:
                print(f"  Calibration: Skipped (insufficient samples for held-out set)")
        except Exception as e:
            print(f"  Warning: Calibration failed: {e}")

        metrics = {
            'market': market,
            'n_samples': len(filtered_data),
            'n_total': len(train_data),
            'filter_rate': len(filtered_data) / len(train_data),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,  # Honest OOS estimate
            'train_test_gap': (train_accuracy - test_accuracy) if test_accuracy else None,
            'feature_importance': dict(zip(PLAYER_BIAS_FEATURES, model.feature_importances_)),
        }

        self.metrics[market] = metrics
        self.trained_date = datetime.now()
        self.version = "player_bias_edge_v4"  # v4: Fixed temporal validation (no random split leakage)

        return metrics

    def _compute_player_history(
        self,
        df: pd.DataFrame,
        market: str,
    ) -> pd.DataFrame:
        """
        Compute player-specific betting history features.

        Adds:
        - player_under_rate: Rolling % of unders
        - player_bias: Rolling average (actual - line)
        - player_bet_count: Number of historical bets

        Args:
            df: DataFrame with player betting data
            market: Market being processed

        Returns:
            DataFrame with player history features added
        """
        if 'player_norm' not in df.columns and 'player' in df.columns:
            # Normalize player names
            df['player_norm'] = df['player'].str.lower().str.strip()

        if 'player_norm' not in df.columns:
            return df

        # Sort by player and time
        df = df.sort_values(['player_norm', 'season', 'week'])

        # Calculate rolling under rate (shifted to prevent leakage)
        df['player_under_rate'] = (
            df.groupby('player_norm')['under_hit']
            .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
        )

        # Calculate rolling bias
        if 'actual' in df.columns and 'line' in df.columns:
            df['_diff'] = df['actual'] - df['line']
            df['player_bias'] = (
                df.groupby('player_norm')['_diff']
                .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
            )
            df.drop(columns=['_diff'], inplace=True)
        else:
            df['player_bias'] = 0.0

        # Count bets per player
        df['player_bet_count'] = (
            df.groupby('player_norm').cumcount()
        )

        return df

    @classmethod
    def load(cls, path: Path = None) -> 'PlayerBiasEdge':
        """
        Load Player Bias edge from disk.

        Args:
            path: Path to saved model. Defaults to standard location.

        Returns:
            Loaded PlayerBiasEdge instance
        """
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'player_bias_edge_model.joblib'

        import joblib
        bundle = joblib.load(path)

        edge = cls()
        edge.models = bundle.get('models', {})
        edge.metrics = bundle.get('metrics', {})
        edge.scalers = bundle.get('scalers', {})
        edge.imputers = bundle.get('imputers', {})
        edge.version = bundle.get('version')
        edge.trained_date = (
            datetime.fromisoformat(bundle['trained_date'])
            if bundle.get('trained_date')
            else None
        )

        # Load calibrator if present
        if 'calibrator' in bundle:
            edge.calibrator = EdgeCalibrator()
            edge.calibrator.calibrators = bundle['calibrator'].get('calibrators', {})
            edge.calibrator.calibration_metrics = bundle['calibrator'].get('metrics', {})

        return edge

    def save(self, path: Path = None) -> None:
        """
        Save Player Bias edge to disk.

        Args:
            path: Path to save file. Defaults to standard location.
        """
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'player_bias_edge_model.joblib'

        import joblib
        bundle = {
            'name': self.name,
            'features': self.features,
            'thresholds': {k: v.__dict__ for k, v in self.thresholds.items()},
            'models': self.models,
            'metrics': self.metrics,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'version': self.version,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
            'calibrator': {
                'calibrators': self.calibrator.calibrators,
                'metrics': self.calibrator.calibration_metrics,
            },
        }
        joblib.dump(bundle, path)
        print(f"Player Bias edge saved to: {path}")
