"""
Direction-Specific Edge

Uses separate UNDER and OVER models to validate direction picks.

Based on analysis showing:
- OVER picks were overconfident (model said 70%, reality 37%)
- Different markets have different edge by direction
- UNDER works broadly, OVER only works for rushing markets
"""
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import MODELS_DIR
from nfl_quant.utils.player_names import normalize_player_name


# Minimum confidence to recommend a direction
# Based on training results AND holdout validation:
# - UNDER models: work well across markets
# - OVER models: VERY conservative - holdout shows 35.8% win rate on OVER
DIRECTION_CONFIDENCE_THRESHOLDS = {
    'UNDER': {
        'player_receptions': 0.55,       # 57.8% holdout win rate ✓
        'player_rush_yds': 0.55,         # 53.4% holdout win rate ✓
        'player_reception_yds': 0.55,    # 51.6% holdout win rate (marginal)
        'player_rush_attempts': 0.90,    # 45.0% holdout - disable
        'player_pass_attempts': 0.55,    # No UNDER bets in holdout
        'player_pass_completions': 0.55, # 56.6% holdout win rate ✓
    },
    'OVER': {
        # OVER bets are 35.8% overall - essentially disable all OVER
        'player_receptions': 0.90,       # Disable
        'player_rush_yds': 0.90,         # Disable (0% in holdout)
        'player_reception_yds': 0.90,    # Disable
        'player_rush_attempts': 0.90,    # Disable
        'player_pass_attempts': 0.90,    # Disable (37.8% in holdout)
        'player_pass_completions': 0.90, # Disable
    },
}


class DirectionEdge:
    """
    Validates direction picks using direction-specific models.

    Uses separate UNDER and OVER models that were trained on different
    signal regimes to provide more accurate confidence estimates.
    """

    def __init__(self):
        """Load direction-specific models."""
        self.under_models = {}
        self.over_models = {}
        self.loaded = False

        self._load_models()

    def _load_models(self):
        """Load direction models from disk."""
        model_path = MODELS_DIR / 'direction_edge_models.joblib'

        if not model_path.exists():
            print(f"Warning: Direction models not found at {model_path}")
            return

        try:
            bundle = joblib.load(model_path)
            self.under_models = bundle.get('under_models', {})
            self.over_models = bundle.get('over_models', {})
            self.version = bundle.get('version', 'unknown')
            self.loaded = True
            print(f"Loaded direction models v{self.version}")
            print(f"  UNDER models: {list(self.under_models.keys())}")
            print(f"  OVER models: {list(self.over_models.keys())}")
        except Exception as e:
            print(f"Error loading direction models: {e}")

    def compute_features(self, row: pd.Series, trailing_avg: float = None) -> Dict[str, float]:
        """
        Compute features needed for direction models from a single row.

        Args:
            row: Series with bet data (needs 'line', optionally trailing stats)
            trailing_avg: Pre-computed trailing average if available

        Returns:
            Dict of feature values
        """
        line = row.get('line', 0)

        # Use provided trailing_avg or try to get from row
        if trailing_avg is None:
            trailing_avg = row.get('trailing_avg', row.get('line_vs_trailing', 0) + line)

        lvt = line - trailing_avg
        lvt_pct = (lvt / trailing_avg * 100) if trailing_avg > 0 else 0

        return {
            'line_vs_trailing': lvt,
            'line_vs_trailing_pct': lvt_pct,
            'line_level': line,
            'trailing_cv': row.get('trailing_cv', 0.3),
            'games_played': row.get('games_played', 6),
            'market_under_rate': row.get('market_under_rate', 0.5),
            'vegas_spread': row.get('vegas_spread', 0),
            'implied_team_total': row.get('implied_team_total', 24),
            'snap_share': row.get('snap_share', 0.7),
        }

    def get_direction_confidence(
        self,
        row: pd.Series,
        market: str,
        direction: str,
        trailing_avg: float = None,
    ) -> float:
        """
        Get confidence for a specific direction from the direction-specific model.

        Args:
            row: Series with bet features
            market: Market being evaluated
            direction: 'UNDER' or 'OVER'
            trailing_avg: Pre-computed trailing average if available

        Returns:
            Probability that this direction is correct (0-1)
        """
        if not self.loaded:
            return 0.5  # No model - return neutral

        # Select appropriate model bundle
        if direction == 'UNDER':
            model_bundle = self.under_models.get(market)
        else:
            model_bundle = self.over_models.get(market)

        if not model_bundle:
            return 0.5  # No model for this market/direction

        # Compute features
        features = self.compute_features(row, trailing_avg)

        # Get feature order from model
        feature_names = model_bundle['features']
        X = np.array([[features.get(f, 0) for f in feature_names]])

        # Apply preprocessing
        X = model_bundle['imputer'].transform(X)
        X = model_bundle['scaler'].transform(X)

        # Get probability
        prob = model_bundle['model'].predict_proba(X)[0, 1]

        return float(prob)

    def should_allow_direction(
        self,
        row: pd.Series,
        market: str,
        direction: str,
        trailing_avg: float = None,
    ) -> tuple[bool, float, str]:
        """
        Check if a direction should be allowed based on direction-specific model.

        Args:
            row: Series with bet features
            market: Market being evaluated
            direction: 'UNDER' or 'OVER'
            trailing_avg: Pre-computed trailing average if available

        Returns:
            Tuple of (allowed, confidence, reason)
        """
        if not self.loaded:
            # Fall back to old behavior - allow UNDER, be cautious on OVER
            if direction == 'UNDER':
                return (True, 0.55, "Direction model not loaded, allowing UNDER")
            else:
                return (False, 0.5, "Direction model not loaded, blocking OVER")

        # Get confidence from direction-specific model
        confidence = self.get_direction_confidence(row, market, direction, trailing_avg)

        # Get threshold for this market/direction
        thresholds = DIRECTION_CONFIDENCE_THRESHOLDS.get(direction, {})
        threshold = thresholds.get(market, 0.55)

        if confidence >= threshold:
            return (
                True,
                confidence,
                f"Direction model approves {direction}: {confidence:.1%} >= {threshold:.1%}"
            )
        else:
            return (
                False,
                confidence,
                f"Direction model rejects {direction}: {confidence:.1%} < {threshold:.1%}"
            )

    def get_recommended_direction(
        self,
        row: pd.Series,
        market: str,
        trailing_avg: float = None,
    ) -> tuple[Optional[str], float, str]:
        """
        Get the recommended direction based on which has higher confidence.

        Args:
            row: Series with bet features
            market: Market being evaluated
            trailing_avg: Pre-computed trailing average if available

        Returns:
            Tuple of (direction, confidence, reason)
        """
        under_conf = self.get_direction_confidence(row, market, 'UNDER', trailing_avg)
        over_conf = self.get_direction_confidence(row, market, 'OVER', trailing_avg)

        # Check thresholds
        under_threshold = DIRECTION_CONFIDENCE_THRESHOLDS['UNDER'].get(market, 0.55)
        over_threshold = DIRECTION_CONFIDENCE_THRESHOLDS['OVER'].get(market, 0.55)

        under_passes = under_conf >= under_threshold
        over_passes = over_conf >= over_threshold

        if under_passes and over_passes:
            # Both pass - pick higher confidence
            if under_conf >= over_conf:
                return ('UNDER', under_conf, f"Both pass, UNDER higher: {under_conf:.1%} vs {over_conf:.1%}")
            else:
                return ('OVER', over_conf, f"Both pass, OVER higher: {over_conf:.1%} vs {under_conf:.1%}")
        elif under_passes:
            return ('UNDER', under_conf, f"Only UNDER passes: {under_conf:.1%}")
        elif over_passes:
            return ('OVER', over_conf, f"Only OVER passes: {over_conf:.1%}")
        else:
            return (None, max(under_conf, over_conf), f"Neither passes: UNDER={under_conf:.1%}, OVER={over_conf:.1%}")


# Singleton instance
_direction_edge = None

def get_direction_edge() -> DirectionEdge:
    """Get or create singleton DirectionEdge instance."""
    global _direction_edge
    if _direction_edge is None:
        _direction_edge = DirectionEdge()
    return _direction_edge
