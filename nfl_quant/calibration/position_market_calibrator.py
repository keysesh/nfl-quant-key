"""
Position and Market-Specific Calibrators

Fixes the overconfidence problem by training separate calibrators for each:
- Position (QB, RB, WR, TE)
- Market type (passing_yards, rushing_yards, receiving_yards, receptions, TDs)

Also implements mean bias correction to fix the systematic underestimation
of RB rushing yards identified in ROI diagnostics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class PositionMarketCalibrator:
    """
    Position and market-specific probability calibrator.

    Fixes two major issues:
    1. Overconfidence (model says 70% but wins 41%)
    2. Mean bias (model systematically underestimates RB rushing yards)
    """

    def __init__(self):
        """Initialize empty calibrators for each position-market combo."""
        self.calibrators = {}  # Key: (position, market) -> IsotonicRegression
        self.mean_bias_corrections = {}  # Key: (position, market) -> float
        self.edge_thresholds = {}  # Key: (position, market) -> float
        self.is_fitted = False

    def fit(
        self,
        predictions_df,
        outcomes_df=None
    ) -> 'PositionMarketCalibrator':
        """
        Fit calibrators for each position-market combination.

        Args:
            predictions_df: DataFrame with columns:
                - player_name
                - position (QB, RB, WR, TE)
                - market (player_pass_yds, player_rush_yds, etc.)
                - model_prob (raw model probability)
                - actual_outcome (1 = over hit, 0 = under hit)
                - model_prediction (predicted value)
                - actual_value (actual stat)
            outcomes_df: Optional separate outcomes DataFrame

        Returns:
            self
        """
        import pandas as pd

        if outcomes_df is not None:
            # Merge predictions with outcomes
            df = predictions_df.merge(outcomes_df, on=['player_name', 'week', 'season'])
        else:
            df = predictions_df.copy()

        # Get unique position-market combinations
        if 'position' not in df.columns:
            # Infer position from market
            df['position'] = df['market'].apply(self._infer_position_from_market)

        position_markets = df.groupby(['position', 'market']).size().reset_index()[['position', 'market']]

        logger.info("=" * 70)
        logger.info("FITTING POSITION-MARKET SPECIFIC CALIBRATORS")
        logger.info("=" * 70)

        for _, row in position_markets.iterrows():
            pos = row['position']
            market = row['market']

            subset = df[(df['position'] == pos) & (df['market'] == market)]

            if len(subset) < 10:
                logger.warning(f"Skipping {pos}/{market}: only {len(subset)} samples (need 10+)")
                continue

            # Extract probabilities and outcomes
            raw_probs = subset['model_prob'].values
            actual_outcomes = subset['actual_outcome'].values

            # Fit isotonic calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            calibrator.fit(raw_probs, actual_outcomes)
            self.calibrators[(pos, market)] = calibrator

            # Calculate calibration quality
            brier_before = brier_score_loss(actual_outcomes, raw_probs)
            calibrated = calibrator.transform(raw_probs)
            brier_after = brier_score_loss(actual_outcomes, calibrated)

            # Calculate mean bias correction
            if 'model_prediction' in subset.columns and 'actual_value' in subset.columns:
                mean_model = subset['model_prediction'].mean()
                mean_actual = subset['actual_value'].mean()
                bias = mean_actual - mean_model
                self.mean_bias_corrections[(pos, market)] = bias
            else:
                self.mean_bias_corrections[(pos, market)] = 0.0

            # Calculate optimal edge threshold based on ROI
            optimal_threshold = self._calculate_optimal_threshold(subset)
            self.edge_thresholds[(pos, market)] = optimal_threshold

            logger.info(f"\n{pos} - {market}:")
            logger.info(f"  Samples: {len(subset)}")
            logger.info(f"  Win Rate: {actual_outcomes.mean():.1%}")
            logger.info(f"  Brier: {brier_before:.4f} → {brier_after:.4f}")
            logger.info(f"  Mean Bias: {self.mean_bias_corrections[(pos, market)]:.2f}")
            logger.info(f"  Optimal Edge Threshold: {optimal_threshold:.1%}")

            # Check calibration quality by probability bin
            self._log_calibration_quality(pos, market, raw_probs, actual_outcomes, calibrated)

        self.is_fitted = True
        logger.info("\n" + "=" * 70)
        logger.info(f"Fitted {len(self.calibrators)} position-market calibrators")
        logger.info("=" * 70)

        return self

    def _infer_position_from_market(self, market: str) -> str:
        """Infer position from market type."""
        if 'pass' in market.lower():
            return 'QB'
        elif 'rush' in market.lower():
            return 'RB'
        elif 'rec' in market.lower() or 'reception' in market.lower():
            return 'WR'  # Default, could be TE
        else:
            return 'UNK'

    def _calculate_optimal_threshold(self, subset) -> float:
        """
        Calculate optimal edge threshold that maximizes ROI.

        Based on ROI diagnostics:
        - RB rushing needs 20%+ edge
        - QB passing works at 5%+ edge
        """
        if 'edge_pct' not in subset.columns or 'actual_outcome' not in subset.columns:
            return 0.05  # Default 5%

        # Test different thresholds
        thresholds = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
        best_roi = -float('inf')
        best_threshold = 0.05

        for thresh in thresholds:
            filtered = subset[subset['edge_pct'] >= thresh]
            if len(filtered) < 5:
                continue

            # Calculate ROI (simplified: win rate vs implied break-even)
            win_rate = filtered['actual_outcome'].mean()
            # Assuming -110 odds, need 52.4% to break even
            break_even = 0.524
            roi = (win_rate - break_even) / break_even

            if roi > best_roi:
                best_roi = roi
                best_threshold = thresh

        return best_threshold

    def _log_calibration_quality(
        self,
        position: str,
        market: str,
        raw_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        calibrated_probs: np.ndarray
    ):
        """Log calibration quality by probability bin."""
        bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        logger.debug(f"  Calibration by bin for {position}/{market}:")
        for i in range(len(bins) - 1):
            mask = (raw_probs >= bins[i]) & (raw_probs < bins[i+1])
            if mask.sum() > 0:
                model_avg = raw_probs[mask].mean()
                actual_rate = actual_outcomes[mask].mean()
                calib_avg = calibrated_probs[mask].mean()
                diff = actual_rate - model_avg
                logger.debug(f"    {bins[i]:.0%}-{bins[i+1]:.0%}: Model={model_avg:.1%}, "
                           f"Actual={actual_rate:.1%}, Calibrated={calib_avg:.1%} "
                           f"(diff={diff:+.1%})")

    def transform(
        self,
        raw_prob: float,
        position: str,
        market: str,
        apply_shrinkage: bool = True
    ) -> float:
        """
        Apply position-market specific calibration.

        Args:
            raw_prob: Raw model probability
            position: Player position
            market: Market type
            apply_shrinkage: Whether to apply high-prob shrinkage

        Returns:
            Calibrated probability
        """
        key = (position, market)

        if key not in self.calibrators:
            # Fall back to generic position calibrator
            generic_key = self._find_generic_calibrator(position, market)
            if generic_key:
                key = generic_key
            else:
                logger.debug(f"No calibrator for {position}/{market}, using raw prob with shrinkage")
                # Apply conservative shrinkage
                return 0.5 + (raw_prob - 0.5) * 0.6

        calibrator = self.calibrators[key]

        # Use numpy interpolation instead of sklearn transform (for loaded calibrators)
        if hasattr(calibrator, 'X_thresholds_') and hasattr(calibrator, 'y_thresholds_'):
            calibrated = np.interp(
                raw_prob,
                calibrator.X_thresholds_,
                calibrator.y_thresholds_
            )
        else:
            # Fallback to sklearn transform if fitted normally
            try:
                calibrated = calibrator.transform([raw_prob])[0]
            except Exception:
                calibrated = 0.5 + (raw_prob - 0.5) * 0.6

        # Apply additional shrinkage for very high probabilities
        if apply_shrinkage and raw_prob > 0.70:
            # More aggressive shrinkage for high probs (overconfidence fix)
            shrinkage_factor = 0.3  # Pull toward 50%
            calibrated = 0.5 + (calibrated - 0.5) * shrinkage_factor

        return np.clip(calibrated, 0.01, 0.99)

    def _find_generic_calibrator(self, position: str, market: str) -> Optional[Tuple[str, str]]:
        """Find a generic calibrator for position or market."""
        # Try same position, any market
        for key in self.calibrators:
            if key[0] == position:
                return key

        # Try same market, any position
        for key in self.calibrators:
            if key[1] == market:
                return key

        return None

    def get_mean_bias_correction(self, position: str, market: str) -> float:
        """
        Get mean bias correction for position-market.

        This fixes the systematic underestimation (e.g., RB rushing yards).

        Args:
            position: Player position
            market: Market type

        Returns:
            Correction to add to model prediction (can be negative)
        """
        key = (position, market)
        return self.mean_bias_corrections.get(key, 0.0)

    def get_edge_threshold(self, position: str, market: str) -> float:
        """
        Get optimal edge threshold for position-market.

        Args:
            position: Player position
            market: Market type

        Returns:
            Minimum edge % to bet (e.g., 0.20 for 20%)
        """
        key = (position, market)

        # Default thresholds based on ROI diagnostics
        defaults = {
            ('QB', 'player_pass_yds'): 0.05,
            ('QB', 'player_pass_tds'): 0.10,
            ('RB', 'player_rush_yds'): 0.20,  # High threshold due to overconfidence
            ('RB', 'player_receptions'): 0.15,
            ('WR', 'player_reception_yds'): 0.10,
            ('WR', 'player_receptions'): 0.10,
            ('TE', 'player_reception_yds'): 0.10,
            ('TE', 'player_receptions'): 0.10,
        }

        return self.edge_thresholds.get(key, defaults.get(key, 0.10))

    def save(self, filepath: Path):
        """Save calibrators to JSON file."""
        filepath = Path(filepath)

        data = {
            'calibrators': {},
            'mean_bias_corrections': {f"{k[0]}_{k[1]}": v for k, v in self.mean_bias_corrections.items()},
            'edge_thresholds': {f"{k[0]}_{k[1]}": v for k, v in self.edge_thresholds.items()},
        }

        # Save calibrator curves
        for key, cal in self.calibrators.items():
            cal_key = f"{key[0]}_{key[1]}"
            data['calibrators'][cal_key] = {
                'X_thresholds': cal.X_thresholds_.tolist(),
                'y_thresholds': cal.y_thresholds_.tolist(),
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved position-market calibrators to {filepath}")

    def load(self, filepath: Path):
        """Load calibrators from JSON file."""
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct calibrators
        self.calibrators = {}
        for cal_key, cal_data in data['calibrators'].items():
            parts = cal_key.split('_')
            pos = parts[0]
            market = '_'.join(parts[1:])

            calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            calibrator.X_thresholds_ = np.array(cal_data['X_thresholds'])
            calibrator.y_thresholds_ = np.array(cal_data['y_thresholds'])
            calibrator.X_min_ = calibrator.X_thresholds_[0]
            calibrator.X_max_ = calibrator.X_thresholds_[-1]
            calibrator.f_ = None  # Will use interpolation

            self.calibrators[(pos, market)] = calibrator

        # Load bias corrections
        self.mean_bias_corrections = {}
        for key, val in data['mean_bias_corrections'].items():
            parts = key.split('_')
            pos = parts[0]
            market = '_'.join(parts[1:])
            self.mean_bias_corrections[(pos, market)] = val

        # Load edge thresholds
        self.edge_thresholds = {}
        for key, val in data['edge_thresholds'].items():
            parts = key.split('_')
            pos = parts[0]
            market = '_'.join(parts[1:])
            self.edge_thresholds[(pos, market)] = val

        self.is_fitted = True
        logger.info(f"Loaded {len(self.calibrators)} position-market calibrators from {filepath}")

    def get_confidence_tier(
        self,
        calibrated_prob: float,
        edge_pct: float,
        position: str,
        market: str
    ) -> str:
        """
        Assign confidence tier based on calibrated probability and edge.

        Args:
            calibrated_prob: Calibrated probability
            edge_pct: Edge percentage
            position: Player position
            market: Market type

        Returns:
            Confidence tier: 'High', 'Medium', or 'Low'
        """
        threshold = self.get_edge_threshold(position, market)

        # High confidence: significantly exceeds threshold AND good probability
        if edge_pct >= threshold * 1.5 and calibrated_prob >= 0.55:
            return 'High'
        # Medium confidence: meets threshold
        elif edge_pct >= threshold and calibrated_prob >= 0.52:
            return 'Medium'
        # Low confidence: below threshold
        else:
            return 'Low'


# Convenience function for quick calibration
def calibrate_probability(
    raw_prob: float,
    position: str,
    market: str,
    calibrator_path: Path = None
) -> float:
    """
    Quick function to calibrate a probability.

    Args:
        raw_prob: Raw model probability
        position: Player position (QB, RB, WR, TE)
        market: Market type (player_rush_yds, etc.)
        calibrator_path: Path to calibrator JSON

    Returns:
        Calibrated probability
    """
    if calibrator_path is None:
        calibrator_path = Path('configs/position_market_calibrator.json')

    cal = PositionMarketCalibrator()
    if calibrator_path.exists():
        cal.load(calibrator_path)
        return cal.transform(raw_prob, position, market)
    else:
        # Apply generic shrinkage
        return 0.5 + (raw_prob - 0.5) * 0.6
