"""
Platt Scaling Probability Calibrator
====================================

Long-term calibration fix using Platt scaling (sigmoid/logistic calibration).

Advantages over Isotonic Regression:
- Handles full probability range [0, 1] without clipping
- Better generalization on unseen data (parametric vs non-parametric)
- No minimum sample requirements
- Smooth, monotonic transformations

Based on Platt (1999): "Probabilistic Outputs for Support Vector Machines"

Root Cause (from backtest analysis):
- Models predict 67.2% average win rate but actual is 51.5%
- Isotonic regression clips probabilities <51.9% to 0.0 (broken)
- Need calibration that handles full probability spectrum

Solution:
Fit logistic regression: calibrated_prob = 1 / (1 + exp(A * raw_prob + B))

Usage:
    from nfl_quant.calibration import PlattCalibrator

    # Train calibrator on historical outcomes
    calibrator = PlattCalibrator()
    calibrator.fit(raw_probabilities, actual_outcomes)

    # Apply to new predictions
    calibrated_prob = calibrator.calibrate(raw_prob=0.70)  # e.g., 0.70 → 0.62
"""

import numpy as np
import pickle
from typing import Optional
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PlattCalibrator:
    """Calibrate probabilities using Platt scaling (sigmoid calibration)."""

    def __init__(self, min_prob: float = 0.01, max_prob: float = 0.99):
        """
        Initialize Platt calibrator.

        Args:
            min_prob: Minimum allowed probability (floor to prevent 0)
            max_prob: Maximum allowed probability (ceiling to prevent 1)
        """
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.calibrator = LogisticRegression(
            C=1.0,  # Regularization (higher = less regularization)
            solver='lbfgs',
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        market: Optional[str] = None,
        side: Optional[str] = None
    ):
        """
        Train Platt calibrator on historical data.

        Args:
            raw_probs: Array of raw model probabilities (0-1)
            outcomes: Array of actual outcomes (0 or 1, where 1 = bet won)
            market: Optional market type (for market-specific calibration)
            side: Optional bet side (for side-specific calibration)
        """
        # Validate inputs
        if len(raw_probs) != len(outcomes):
            raise ValueError(f"Length mismatch: {len(raw_probs)} probs vs {len(outcomes)} outcomes")

        if len(raw_probs) < 10:
            raise ValueError(f"Need at least 10 samples for Platt scaling, got {len(raw_probs)}")

        # Convert to numpy arrays
        raw_probs = np.asarray(raw_probs).reshape(-1, 1)
        outcomes = np.asarray(outcomes)

        # Clip probabilities to valid range (avoid log(0) issues)
        raw_probs_clipped = np.clip(raw_probs, 0.001, 0.999)

        # Standardize raw probabilities for better numerical stability
        raw_probs_scaled = self.scaler.fit_transform(raw_probs_clipped)

        # Fit logistic regression
        self.calibrator.fit(raw_probs_scaled, outcomes)

        self.is_fitted = True

    def calibrate(
        self,
        raw_prob: float,
        market: Optional[str] = None,
        side: Optional[str] = None
    ) -> float:
        """
        Calibrate a raw probability using Platt scaling.

        Args:
            raw_prob: Raw probability from model (0-1)
            market: Not used (for compatibility with other calibrators)
            side: Not used (for compatibility with other calibrators)

        Returns:
            Calibrated probability (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        # Clip to valid range
        raw_prob_clipped = np.clip(raw_prob, 0.001, 0.999)

        # Reshape for sklearn
        raw_prob_array = np.array([[raw_prob_clipped]])

        # Scale using fitted scaler
        raw_prob_scaled = self.scaler.transform(raw_prob_array)

        # Get calibrated probability from logistic regression
        calibrated_prob = self.calibrator.predict_proba(raw_prob_scaled)[0, 1]

        # Apply floor/ceiling
        calibrated_prob = np.clip(calibrated_prob, self.min_prob, self.max_prob)

        return float(calibrated_prob)

    def calibrate_batch(
        self,
        raw_probs: np.ndarray,
        markets: Optional[np.ndarray] = None,
        sides: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calibrate multiple probabilities at once.

        Args:
            raw_probs: Array of raw probabilities
            markets: Not used (for compatibility)
            sides: Not used (for compatibility)

        Returns:
            Array of calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        # Clip to valid range
        raw_probs_clipped = np.clip(raw_probs, 0.001, 0.999).reshape(-1, 1)

        # Scale
        raw_probs_scaled = self.scaler.transform(raw_probs_clipped)

        # Get calibrated probabilities
        calibrated = self.calibrator.predict_proba(raw_probs_scaled)[:, 1]

        # Apply floor/ceiling
        calibrated = np.clip(calibrated, self.min_prob, self.max_prob)

        return calibrated

    def save(self, filepath: Path):
        """Save calibrator to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'calibrator': self.calibrator,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'min_prob': self.min_prob,
                'max_prob': self.max_prob
            }, f)

    @classmethod
    def load(cls, filepath: Path) -> 'PlattCalibrator':
        """Load calibrator from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls(min_prob=data['min_prob'], max_prob=data['max_prob'])
        calibrator.calibrator = data['calibrator']
        calibrator.scaler = data['scaler']
        calibrator.is_fitted = data['is_fitted']

        return calibrator


def create_platt_calibrator(
    raw_probs: Optional[np.ndarray] = None,
    outcomes: Optional[np.ndarray] = None,
    calibration_file: Optional[Path] = None
) -> Optional[PlattCalibrator]:
    """
    Create a Platt calibrator from training data or load from file.

    Args:
        raw_probs: Historical raw probabilities (required if training)
        outcomes: Historical outcomes (required if training)
        calibration_file: Path to saved calibrator (for loading)

    Returns:
        PlattCalibrator instance or None if neither training data nor file provided
    """
    if calibration_file and calibration_file.exists():
        # Load from file
        return PlattCalibrator.load(calibration_file)

    elif raw_probs is not None and outcomes is not None:
        # Train new calibrator
        calibrator = PlattCalibrator()
        calibrator.fit(raw_probs, outcomes)
        return calibrator

    else:
        # No data or file provided
        return None


# Example usage
if __name__ == "__main__":
    # Simulate model overconfidence (like our backtest results)
    # Model predicts 67.2% on average, but actual is 51.5%
    np.random.seed(42)

    # Generate synthetic training data
    n_samples = 1000

    # Raw probabilities (model predictions) - skewed high (overconfident)
    raw_probs = np.random.beta(a=4, b=2, size=n_samples)  # Mean ~0.67

    # Actual outcomes - closer to 50/50 (model is overconfident)
    # Use raw_probs but shrink them towards 0.5 for "true" probabilities
    true_probs = 0.5 + 0.3 * (raw_probs - 0.5)  # Shrink by 70%
    outcomes = np.random.binomial(n=1, p=true_probs)

    print("Platt Scaling Calibration Test")
    print("=" * 60)
    print(f"Training samples: {n_samples}")
    print(f"Raw probs mean: {raw_probs.mean():.1%}")
    print(f"Actual outcomes mean: {outcomes.mean():.1%}")
    print(f"Calibration error (before): {abs(raw_probs.mean() - outcomes.mean()):.1%}")
    print()

    # Train Platt calibrator
    calibrator = create_platt_calibrator(raw_probs, outcomes)

    # Test calibration
    test_probs = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    print("Calibration Examples:")
    print("=" * 60)
    for raw_prob in test_probs:
        calibrated = calibrator.calibrate(raw_prob)
        deflation = (raw_prob - calibrated) * 100
        print(f"{raw_prob:.0%} → {calibrated:.1%} (deflation: {deflation:+.1f} points)")

    print("\nExpected Impact:")
    print("  - No bet filtering (handles all probabilities 0-1)")
    print("  - Better calibration than shrinkage (data-driven, not fixed)")
    print("  - Smooth, monotonic transformations")
    print("  - Superior to broken isotonic regression")
