"""
Isotonic regression calibration - industry standard.
Replaces manual piecewise function with sklearn isotonic regression.

Based on professional betting model standards.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class NFLProbabilityCalibrator:
    """
    Calibrate model probabilities using isotonic regression.

    Isotonic regression is the industry standard for probability calibration
    used by professional betting models and FiveThirtyEight.

    Usage:
        # Training phase (on backtest data)
        calibrator = NFLProbabilityCalibrator()
        calibrator.fit(raw_probabilities, actual_outcomes)
        calibrator.save('configs/calibrator.json')

        # Prediction phase (on new games)
        calibrator = NFLProbabilityCalibrator()
        calibrator.load('configs/calibrator.json')
        calibrated_prob = calibrator.transform(raw_prob)
    """

    def __init__(self, high_prob_threshold: float = 0.70, high_prob_shrinkage: float = 0.3):
        """
        Initialize the calibrator with hybrid isotonic regression.

        Args:
            high_prob_threshold: Probability above which to apply additional shrinkage (default 0.70)
            high_prob_shrinkage: Shrinkage factor for high probabilities (default 0.3)
                                 Lower values = more aggressive shrinkage toward 50%
        """
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.is_fitted = False
        self.high_prob_threshold = high_prob_threshold
        self.high_prob_shrinkage = high_prob_shrinkage

    def fit(self, raw_probabilities: np.ndarray, actual_outcomes: np.ndarray) -> 'NFLProbabilityCalibrator':
        """
        Fit calibration curve on historical backtest data.

        Args:
            raw_probabilities: Array of raw model probabilities (0-1)
            actual_outcomes: Array of actual binary outcomes (0 or 1)

        Returns:
            self (for method chaining)
        """
        # Convert to numpy arrays if needed
        raw_probabilities = np.asarray(raw_probabilities)
        actual_outcomes = np.asarray(actual_outcomes)

        # Validate inputs
        if len(raw_probabilities) != len(actual_outcomes):
            raise ValueError("Probabilities and outcomes must have same length")

        if len(raw_probabilities) < 10:
            logger.warning(f"Only {len(raw_probabilities)} samples for calibration. Need at least 30 for reliable calibration.")

        # Fit isotonic regression
        self.calibrator.fit(raw_probabilities, actual_outcomes)
        self.is_fitted = True

        # Calculate improvement
        brier_before = brier_score_loss(actual_outcomes, raw_probabilities)
        calibrated = self.transform(raw_probabilities)
        brier_after = brier_score_loss(actual_outcomes, calibrated)
        improvement = brier_before - brier_after

        logger.info("=" * 60)
        logger.info("ISOTONIC CALIBRATION FIT")
        logger.info("=" * 60)
        logger.info(f"Samples: {len(raw_probabilities)}")
        logger.info(f"Brier Score: {brier_before:.4f} → {brier_after:.4f}")
        logger.info(f"Improvement: {improvement:+.4f}")
        if improvement > 0:
            logger.info("✅ Calibration improved probability quality")
        else:
            logger.warning("⚠️  Calibration did not improve Brier score")
        logger.info("=" * 60)

        return self

    def transform(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """
        Apply hybrid calibration to new predictions.

        For probabilities below high_prob_threshold: Use isotonic regression
        For probabilities >= high_prob_threshold: Apply aggressive linear shrinkage
        For out-of-range probabilities: Apply conservative shrinkage

        Args:
            raw_probabilities: Array or single value of raw model probabilities

        Returns:
            calibrated_probabilities: Calibrated probabilities (same shape as input)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform. Call fit() first.")

        # Handle both scalar and array inputs
        raw_probabilities = np.asarray(raw_probabilities)
        was_scalar = raw_probabilities.ndim == 0

        if was_scalar:
            raw_probabilities = raw_probabilities.reshape(1)

        # Get trained range
        min_trained = float(self.calibrator.X_thresholds_[0])
        max_trained = float(self.calibrator.X_thresholds_[-1])

        # Apply isotonic calibration first
        calibrated = np.interp(
            raw_probabilities,
            self.calibrator.X_thresholds_,
            self.calibrator.y_thresholds_
        )

        # Handle out-of-range probabilities with conservative shrinkage
        # If raw prob is outside trained range, don't trust the extrapolation
        out_of_range_high = raw_probabilities > max_trained
        out_of_range_low = raw_probabilities < min_trained

        if out_of_range_high.any():
            # For probabilities above trained range, apply conservative shrinkage
            # Pull toward 50% more aggressively since we're extrapolating
            out_probs = raw_probabilities[out_of_range_high]
            # Use 40% shrinkage for out-of-range (more conservative than normal)
            shrunk = 0.5 + (out_probs - 0.5) * 0.4
            calibrated[out_of_range_high] = shrunk
            logger.debug(f"Out-of-range high probs shrunk: {out_probs} -> {shrunk}")

        if out_of_range_low.any():
            # For very low probabilities below trained range
            out_probs = raw_probabilities[out_of_range_low]
            shrunk = 0.5 + (out_probs - 0.5) * 0.4
            calibrated[out_of_range_low] = shrunk

        # CRITICAL: Constrain maximum deviation from raw probability
        # Prevent overfitting - calibrated prob shouldn't be more than 0.35 away from raw
        # This prevents cases like raw=0.47 -> cal=0.99 due to small sample overfitting
        max_deviation = 0.35
        deviation = calibrated - raw_probabilities
        excessive_positive = deviation > max_deviation
        excessive_negative = deviation < -max_deviation

        if excessive_positive.any():
            # Calibrated is too much higher than raw - cap it
            calibrated[excessive_positive] = raw_probabilities[excessive_positive] + max_deviation

        if excessive_negative.any():
            # Calibrated is too much lower than raw - cap it
            calibrated[excessive_negative] = raw_probabilities[excessive_negative] - max_deviation

        # PRESERVE DIRECTION: If raw < 0.5, calibrated should also be < 0.5
        # This ensures model direction is respected - ONLY for extreme cases
        # Allow some flexibility for borderline cases (0.45-0.55 range)
        wrong_direction_over = (raw_probabilities < 0.45) & (calibrated > 0.55)
        wrong_direction_under = (raw_probabilities > 0.55) & (calibrated < 0.45)

        if wrong_direction_over.any():
            # Model clearly says UNDER but calibrated clearly says OVER - cap at 0.52
            calibrated[wrong_direction_over] = np.minimum(calibrated[wrong_direction_over], 0.52)

        if wrong_direction_under.any():
            # Model clearly says OVER but calibrated clearly says UNDER - floor at 0.48
            calibrated[wrong_direction_under] = np.maximum(calibrated[wrong_direction_under], 0.48)

        # Apply additional shrinkage for high-confidence predictions
        # This prevents overconfidence in the 70%+ range
        high_mask = (raw_probabilities >= self.high_prob_threshold) & ~out_of_range_high

        if high_mask.any():
            # For high probabilities, apply linear shrinkage toward 50%
            # This is more aggressive than isotonic regression alone
            high_probs = raw_probabilities[high_mask]
            shrunk_probs = 0.5 + (high_probs - 0.5) * self.high_prob_shrinkage

            # Use the MORE CONSERVATIVE of isotonic or shrinkage
            # (i.e., the one closer to 50%)
            isotonic_high = calibrated[high_mask]
            calibrated[high_mask] = np.where(
                np.abs(shrunk_probs - 0.5) < np.abs(isotonic_high - 0.5),
                shrunk_probs,
                isotonic_high
            )

        # Ensure output is in [y_min, y_max]
        calibrated = np.clip(calibrated, self.calibrator.y_min, self.calibrator.y_max)

        return float(calibrated[0]) if was_scalar else calibrated

    def transform_bidirectional(self, raw_prob_over: float) -> dict:
        """
        Apply calibration and return both Over and Under calibrated probabilities.

        Args:
            raw_prob_over: Raw model probability for Over

        Returns:
            Dictionary with:
                - prob_over: Calibrated probability for Over
                - prob_under: Calibrated probability for Under (1 - prob_over)
                - confidence: Confidence score based on distribution width
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform. Call fit() first.")

        # Calibrate the Over probability
        calibrated_over = self.transform(raw_prob_over)

        # Under probability is complement
        calibrated_under = 1.0 - calibrated_over

        # Calculate confidence based on how far from 50/50
        # Higher confidence when probability is further from 0.5
        distance_from_even = abs(calibrated_over - 0.5)
        confidence = 0.5 + (distance_from_even * 1.0)  # Scale to 0.5-1.0 range

        return {
            'prob_over': float(calibrated_over),
            'prob_under': float(calibrated_under),
            'confidence': float(np.clip(confidence, 0.5, 1.0)),
            'edge_magnitude': float(distance_from_even)
        }

    def save(self, filepath: str = 'configs/calibrator.json') -> None:
        """
        Save fitted calibrator parameters to JSON.

        Args:
            filepath: Path to save calibrator parameters
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before saving")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save isotonic regression parameters + hybrid calibration settings
        params = {
            'X_thresholds': self.calibrator.X_thresholds_.tolist(),
            'y_thresholds': self.calibrator.y_thresholds_.tolist(),
            'X_min': float(self.calibrator.X_min_),
            'X_max': float(self.calibrator.X_max_),
            'y_min': float(self.calibrator.y_min),
            'y_max': float(self.calibrator.y_max),
            'high_prob_threshold': float(self.high_prob_threshold),
            'high_prob_shrinkage': float(self.high_prob_shrinkage),
        }

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

        logger.info(f"✅ Calibrator saved to {filepath}")
        logger.info(f"   Hybrid mode: threshold={self.high_prob_threshold:.2f}, shrinkage={self.high_prob_shrinkage:.2f}")

    def load(self, filepath: str = 'configs/calibrator.json') -> 'NFLProbabilityCalibrator':
        """
        Load fitted calibrator parameters from JSON.

        Args:
            filepath: Path to load calibrator parameters from

        Returns:
            self (for method chaining)
        """
        with open(filepath) as f:
            params = json.load(f)

        # Handle multiple formats:
        # 1. New format: calibration_points as [[x1, y1], [x2, y2], ...]
        # 2. Old format: X_thresholds and y_thresholds as separate arrays
        # 3. Alternative format: x and y as separate arrays

        if 'calibration_points' in params:
            # New format from retrain_calibrators_from_backtest.py
            cal_points = np.array(params['calibration_points'])
            x_vals = cal_points[:, 0]
            y_vals = cal_points[:, 1]
        else:
            # Old formats
            x_key = 'X_thresholds' if 'X_thresholds' in params else 'x'
            y_key = 'y_thresholds' if 'y_thresholds' in params else 'y'
            x_vals = np.array(params[x_key])
            y_vals = np.array(params[y_key])

        x_min_key = 'X_min' if 'X_min' in params else ('X_min_' if 'X_min_' in params else None)
        x_max_key = 'X_max' if 'X_max' in params else ('X_max_' if 'X_max_' in params else None)

        # Reconstruct isotonic regression from saved parameters
        self.calibrator.X_thresholds_ = x_vals
        self.calibrator.y_thresholds_ = y_vals

        # Set min/max if available, otherwise infer from data
        if x_min_key and x_min_key in params:
            self.calibrator.X_min_ = params[x_min_key]
        else:
            self.calibrator.X_min_ = float(x_vals[0])

        if x_max_key and x_max_key in params:
            self.calibrator.X_max_ = params[x_max_key]
        else:
            self.calibrator.X_max_ = float(x_vals[-1])

        self.calibrator.y_min = params.get('y_min', 0.0)
        self.calibrator.y_max = params.get('y_max', 1.0)

        # Load hybrid calibration parameters (use defaults if not present for backward compatibility)
        self.high_prob_threshold = params.get('high_prob_threshold', 0.70)
        self.high_prob_shrinkage = params.get('high_prob_shrinkage', 0.3)

        self.is_fitted = True

        logger.info(f"✅ Calibrator loaded from {filepath}")
        logger.info(f"   {len(self.calibrator.X_thresholds_)} calibration points")
        logger.info(f"   Range: [{self.calibrator.X_min_:.3f}, {self.calibrator.X_max_:.3f}]")
        if 'high_prob_threshold' in params:
            logger.info(f"   Hybrid mode: threshold={self.high_prob_threshold:.2f}, shrinkage={self.high_prob_shrinkage:.2f}")
        return self


def calibrate_probability_simple(raw_prob: float) -> float:
    """
    Simple fallback calibration when isotonic calibrator is not available.

    This is a conservative linear calibration that pulls probabilities
    toward 50% to avoid overconfidence.

    Args:
        raw_prob: Raw model probability (0-1)

    Returns:
        Calibrated probability (0-1)
    """
    # Pull toward 50% by 40% (conservative)
    return 0.5 + (raw_prob - 0.5) * 0.6
