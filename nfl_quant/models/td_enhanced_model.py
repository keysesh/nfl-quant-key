"""
Enhanced TD Model - XGBoost-based Anytime TD Prediction

Uses XGBoost with red zone features for profitable TD predictions.
Validated via walk-forward: RB @ 60% confidence = 58.2% WR, +6.6% ROI

Key features:
- trailing_tds: 26.4% importance (most important)
- trailing_target_share: 11.4% importance
- opp_rush_tds_allowed: 6.3% importance (defense matters)
- Position-specific handling (RBs are most profitable)

Usage:
    model = TDEnhancedModel()
    model.train(train_data)
    prob = model.predict(features)  # P(score at least 1 TD)
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import train_test_split
import joblib
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import DATA_DIR, MODELS_DIR
from nfl_quant.utils.player_names import normalize_player_name

logger = logging.getLogger(__name__)

# Features validated in walk-forward backtest
TD_ENHANCED_FEATURES = [
    # Trailing stats (26.4% + 11.4% + 10.6% + 10.2% importance)
    'trailing_tds',
    'trailing_targets',
    'trailing_carries',
    'trailing_target_share',
    'trailing_rec_yds',
    'trailing_rush_yds',

    # Red zone features (from PBP)
    'rz_targets_per_game',
    'rz_carries_per_game',
    'gl_carries_per_game',
    'rz_td_rate',

    # Opponent defense (6.3% + 4.7% importance)
    'opp_tds_allowed_per_game',
    'opp_pass_tds_allowed',
    'opp_rush_tds_allowed',
    'opp_rz_td_rate',

    # Team efficiency
    'team_rz_td_rate',

    # Position encoding
    'is_rb',
    'is_wr',
    'is_te',
]

# Confidence thresholds from walk-forward validation
TD_CONFIDENCE_THRESHOLDS = {
    'RB': 0.55,   # RB @ 55%+ is profitable
    'WR': 0.65,   # WR needs higher confidence
    'TE': 0.65,   # TE needs higher confidence
    'default': 0.60,
}


class TDEnhancedModel:
    """
    Enhanced TD Model using XGBoost with red zone and opponent features.

    Walk-forward validated:
    - RB @ 60%: 58.2% WR, +6.6% ROI
    - All positions @ 60%: 55.8% WR, +2.2% ROI
    """

    def __init__(self):
        self.model = None
        self.calibrator = None
        self.features = TD_ENHANCED_FEATURES.copy()
        self.trained_date = None
        self.validation_metrics = {}

    def train(
        self,
        train_data: pd.DataFrame,
        calibration_split: float = 0.2,
    ) -> Dict:
        """
        Train the enhanced TD model.

        Args:
            train_data: DataFrame with features and 'scored_td' target
            calibration_split: Fraction for calibration (default 0.2)

        Returns:
            Training metrics dict
        """
        logger.info("Training Enhanced TD Model...")

        # Filter to available features
        available_features = [f for f in self.features if f in train_data.columns]
        self.features = available_features

        logger.info(f"  Features: {len(available_features)}")

        # Prepare X, y
        X = train_data[available_features].fillna(0)
        y = train_data['scored_td']

        # Drop NaN targets
        valid = ~y.isna()
        X = X[valid]
        y = y[valid]

        logger.info(f"  Training samples: {len(X)}")
        logger.info(f"  TD rate: {y.mean()*100:.1f}%")

        # Train/calibration split
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=calibration_split, random_state=42
        )

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        self.model.fit(X_train, y_train)

        # Calibrate
        calib_probs = self.model.predict_proba(X_calib)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(calib_probs, y_calib)

        self.trained_date = datetime.now().isoformat()

        # Compute validation metrics
        train_probs = self.predict_proba(X_train)
        calib_probs_calibrated = self.predict_proba(X_calib)

        metrics = {
            'n_samples': len(X),
            'n_features': len(available_features),
            'td_rate': float(y.mean()),
            'train_auc': self._compute_auc(y_train, train_probs),
            'calib_auc': self._compute_auc(y_calib, calib_probs_calibrated),
            'feature_importance': self.get_feature_importance(),
        }

        self.validation_metrics = metrics

        logger.info(f"  Train AUC: {metrics['train_auc']:.3f}")
        logger.info(f"  Calib AUC: {metrics['calib_auc']:.3f}")

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict P(score at least 1 TD).

        Args:
            X: DataFrame with features

        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure correct features
        X_features = X[self.features].fillna(0)

        # Raw predictions
        raw_probs = self.model.predict_proba(X_features)[:, 1]

        # Calibrate
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.predict(raw_probs)
        else:
            calibrated_probs = raw_probs

        return calibrated_probs

    def predict(
        self,
        df: pd.DataFrame,
        position_col: str = 'position',
    ) -> pd.DataFrame:
        """
        Predict TD probabilities with recommendations.

        Args:
            df: DataFrame with features and position
            position_col: Name of position column

        Returns:
            DataFrame with predictions and recommendations
        """
        result = df.copy()

        # Get probabilities
        result['p_score_td'] = self.predict_proba(df)

        # Get position-specific thresholds
        def get_threshold(pos):
            return TD_CONFIDENCE_THRESHOLDS.get(pos, TD_CONFIDENCE_THRESHOLDS['default'])

        result['td_threshold'] = result[position_col].apply(get_threshold)

        # Recommendation: bet if probability >= threshold
        result['td_bet'] = result['p_score_td'] >= result['td_threshold']

        # Edge vs implied market probability (assume ~40% for anytime TD)
        result['td_market_implied'] = 0.40
        result['td_edge'] = result['p_score_td'] - result['td_market_implied']

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            return {}

        importance = dict(zip(self.features, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _compute_auc(self, y_true, y_pred) -> float:
        """Compute AUC score."""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.0

    def save(self, path: Optional[Path] = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / 'td_enhanced_model.joblib'

        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'features': self.features,
            'trained_date': self.trained_date,
            'validation_metrics': self.validation_metrics,
            'version': 'enhanced_v1',
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved TD Enhanced Model to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'TDEnhancedModel':
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / 'td_enhanced_model.joblib'

        if not path.exists():
            raise FileNotFoundError(f"TD Enhanced Model not found at {path}")

        model_data = joblib.load(path)

        instance = cls()
        instance.model = model_data['model']
        instance.calibrator = model_data['calibrator']
        instance.features = model_data['features']
        instance.trained_date = model_data.get('trained_date')
        instance.validation_metrics = model_data.get('validation_metrics', {})

        return instance

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None


def compute_td_features_from_pbp(
    pbp: pd.DataFrame,
    max_week: int,
    season: int = 2024,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute TD-specific features from PBP data.

    Returns:
        Tuple of (player_rz_features, opponent_defense, team_efficiency)
    """
    # Filter to historical data
    pbp = pbp[
        ((pbp['season'] < season) |
         ((pbp['season'] == season) & (pbp['week'] < max_week)))
    ].copy()

    if len(pbp) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Normalize names
    pbp['receiver_norm'] = pbp['receiver_player_name'].apply(
        lambda x: normalize_player_name(x) if pd.notna(x) else None
    )
    pbp['rusher_norm'] = pbp['rusher_player_name'].apply(
        lambda x: normalize_player_name(x) if pd.notna(x) else None
    )

    # Red zone plays
    rz = pbp[pbp['yardline_100'] <= 20]
    gl = pbp[pbp['yardline_100'] <= 5]

    # Player games
    player_games = pbp.groupby('receiver_norm')['game_id'].nunique().reset_index()
    player_games.columns = ['player_norm', 'games']

    # RZ targets
    rz_targets = rz[rz['play_type'] == 'pass'].groupby('receiver_norm').agg({
        'play_id': 'count',
        'pass_touchdown': 'sum'
    }).reset_index()
    rz_targets.columns = ['player_norm', 'rz_targets', 'rz_pass_tds']

    # RZ carries
    rz_carries = rz[rz['play_type'] == 'run'].groupby('rusher_norm').agg({
        'play_id': 'count',
        'rush_touchdown': 'sum'
    }).reset_index()
    rz_carries.columns = ['player_norm', 'rz_carries', 'rz_rush_tds']

    # GL carries
    gl_carries = gl[gl['play_type'] == 'run'].groupby('rusher_norm').size().reset_index(name='gl_carries')
    gl_carries.columns = ['player_norm', 'gl_carries']

    # Merge player features
    player_features = player_games.merge(rz_targets, on='player_norm', how='left')
    player_features = player_features.merge(rz_carries, on='player_norm', how='left')
    player_features = player_features.merge(gl_carries, on='player_norm', how='left')

    for col in ['rz_targets', 'rz_pass_tds', 'rz_carries', 'rz_rush_tds', 'gl_carries']:
        player_features[col] = player_features[col].fillna(0)

    player_features['rz_targets_per_game'] = player_features['rz_targets'] / player_features['games']
    player_features['rz_carries_per_game'] = player_features['rz_carries'] / player_features['games']
    player_features['gl_carries_per_game'] = player_features['gl_carries'] / player_features['games']
    player_features['rz_opportunities'] = player_features['rz_targets'] + player_features['rz_carries']
    player_features['rz_tds'] = player_features['rz_pass_tds'] + player_features['rz_rush_tds']
    player_features['rz_td_rate'] = np.where(
        player_features['rz_opportunities'] > 0,
        player_features['rz_tds'] / player_features['rz_opportunities'],
        0.0
    )

    # Opponent defense
    def_games = pbp.groupby('defteam')['game_id'].nunique().reset_index()
    def_games.columns = ['team', 'games']

    def_tds = pbp.groupby('defteam').agg({
        'touchdown': 'sum',
        'pass_touchdown': 'sum',
        'rush_touchdown': 'sum'
    }).reset_index()
    def_tds.columns = ['team', 'tds_allowed', 'pass_tds_allowed', 'rush_tds_allowed']

    rz_def = rz.groupby('defteam').agg({
        'play_id': 'count',
        'touchdown': 'sum'
    }).reset_index()
    rz_def.columns = ['team', 'rz_plays_faced', 'rz_tds_allowed']

    opponent_defense = def_games.merge(def_tds, on='team')
    opponent_defense = opponent_defense.merge(rz_def, on='team', how='left')

    opponent_defense['opp_tds_allowed_per_game'] = opponent_defense['tds_allowed'] / opponent_defense['games']
    opponent_defense['opp_pass_tds_allowed'] = opponent_defense['pass_tds_allowed'] / opponent_defense['games']
    opponent_defense['opp_rush_tds_allowed'] = opponent_defense['rush_tds_allowed'] / opponent_defense['games']
    opponent_defense['opp_rz_td_rate'] = np.where(
        opponent_defense['rz_plays_faced'] > 0,
        opponent_defense['rz_tds_allowed'] / opponent_defense['rz_plays_faced'],
        0.13
    )

    # Team efficiency
    team_rz = rz.groupby('posteam').agg({
        'play_id': 'count',
        'touchdown': 'sum'
    }).reset_index()
    team_rz.columns = ['team', 'rz_plays', 'rz_tds']
    team_rz['team_rz_td_rate'] = team_rz['rz_tds'] / team_rz['rz_plays']

    return (
        player_features[['player_norm', 'rz_targets_per_game', 'rz_carries_per_game',
                         'gl_carries_per_game', 'rz_td_rate']],
        opponent_defense[['team', 'opp_tds_allowed_per_game', 'opp_pass_tds_allowed',
                          'opp_rush_tds_allowed', 'opp_rz_td_rate']],
        team_rz[['team', 'team_rz_td_rate']]
    )
