"""
Wrapper for the fixed usage predictor to maintain compatibility with PlayerSimulator.

The new predictor has a different structure (position-specific models),
so this wrapper provides the same interface as the old predictor.
"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class UsagePredictorWrapper:
    """
    Wrapper that makes the fixed predictor compatible with PlayerSimulator.

    New predictor structure:
    {
        'models': {
            'QB': {'snaps': model, 'attempts': model, 'carries': model},
            'RB': {...},
            'WR': {...}
        },
        'feature_cols': [...],
        ...
    }

    Old interface expected by PlayerSimulator:
    predictor.predict(features_df) -> {'snaps': [value], 'targets': [value], 'carries': [value]}
    """

    def __init__(self, predictor_path: Path):
        """Load the fixed predictor."""
        self.predictor = joblib.load(predictor_path)
        self.models = self.predictor['models']
        self.feature_cols = self.predictor['feature_cols']

    def predict(self, features_df: pd.DataFrame, position: str = 'WR') -> Dict[str, list]:
        """
        Predict usage for a player.

        Args:
            features_df: DataFrame with features (must include 'trailing_snaps', 'trailing_attempts', 'trailing_carries', 'week')
            position: Player position (QB, RB, WR, TE)

        Returns:
            Dictionary with 'snaps', 'targets', 'carries' predictions
        """
        # Map TE to WR (they're similar)
        if position == 'TE':
            position = 'WR'

        if position not in self.models:
            raise ValueError(f"No models for position: {position}")

        # Ensure features are in correct order
        X = features_df[self.feature_cols]

        # Get position-specific models
        models = self.models[position]

        # Predict
        snaps_pred = models['snaps'].predict(X)[0]
        attempts_pred = models['attempts'].predict(X)[0]
        carries_pred = models['carries'].predict(X)[0]

        # Return in old format
        # Note: 'attempts' maps to 'targets' for compatibility
        return {
            'snaps': [max(snaps_pred, 1.0)],  # Ensure positive
            'targets': [max(attempts_pred, 0.0)],  # 'targets' is pass_attempts for QB, targets for WR/RB
            'carries': [max(carries_pred, 0.0)],
        }


def load_fixed_predictor() -> UsagePredictorWrapper:
    """
    Load the fixed usage predictor.

    Returns:
        UsagePredictorWrapper instance
    """
    predictor_path = Path('data/models/usage_predictor_fixed.joblib')

    if not predictor_path.exists():
        # Fall back to regular predictor
        predictor_path = Path('data/models/usage_predictor.joblib')
        logger.warning(f"Fixed predictor not found, using old predictor: {predictor_path}")

    logger.info(f"Loading usage predictor from {predictor_path}")

    return UsagePredictorWrapper(predictor_path)
