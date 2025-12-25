"""
Edge Calibrator - Per-Market Probability Calibration

This module provides probability calibration for edge models.
Raw XGBoost probabilities often don't match actual hit rates.
Calibration ensures P(UNDER)=60% means ~60% of those bets hit.

Uses Isotonic Regression by default (non-parametric, monotonic).
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


class EdgeCalibrator:
    """
    Per-market probability calibration for edge models.

    Calibrates raw model probabilities to match actual hit rates.
    Uses Isotonic Regression which is non-parametric and preserves
    probability ordering.

    Example:
        calibrator = EdgeCalibrator()
        calibrator.fit('player_receptions', y_true, y_prob)
        calibrated_prob = calibrator.calibrate('player_receptions', 0.65)
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ('isotonic' or 'platt')
        """
        self.method = method
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.calibration_metrics: Dict[str, Dict] = {}

    def fit(
        self,
        market: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """
        Fit calibrator for a specific market.

        Args:
            market: Market name (e.g., 'player_receptions')
            y_true: Actual outcomes (0 or 1 for under hit)
            y_prob: Raw model probabilities P(UNDER)
            n_bins: Number of bins for calibration curve metrics

        Returns:
            Dict with calibration metrics
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if len(y_true) < 50:
            raise ValueError(
                f"Insufficient samples for calibration: {len(y_true)} (need 50+)"
            )

        # Fit calibrator
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(
                out_of_bounds='clip',
                increasing=True,
            )
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        calibrator.fit(y_prob, y_true)
        self.calibrators[market] = calibrator

        # Calculate calibration metrics
        calibrated_prob = calibrator.predict(y_prob)

        # Compute calibration curve
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true, y_prob, n_bins)
        ece_calibrated = self._compute_ece(y_true, calibrated_prob, n_bins)

        metrics = {
            'market': market,
            'n_samples': len(y_true),
            'ece_raw': ece,
            'ece_calibrated': ece_calibrated,
            'ece_improvement': (ece - ece_calibrated) / ece * 100 if ece > 0 else 0,
            'mean_raw_prob': float(np.mean(y_prob)),
            'mean_calibrated_prob': float(np.mean(calibrated_prob)),
            'actual_positive_rate': float(np.mean(y_true)),
        }

        self.calibration_metrics[market] = metrics
        return metrics

    def calibrate(
        self,
        market: str,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            market: Market name
            y_prob: Raw probabilities (single value or array)

        Returns:
            Calibrated probabilities
        """
        if market not in self.calibrators:
            # No calibrator available, return raw probabilities
            return y_prob

        y_prob = np.asarray(y_prob)
        was_scalar = y_prob.ndim == 0

        if was_scalar:
            y_prob = y_prob.reshape(1)

        calibrated = self.calibrators[market].predict(y_prob)

        if was_scalar:
            return float(calibrated[0])

        return calibrated

    def _compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures the average difference between predicted probability
        and actual frequency, weighted by bin size.

        Args:
            y_true: Actual outcomes
            y_prob: Predicted probabilities
            n_bins: Number of bins

        Returns:
            ECE value (lower is better, 0 is perfect calibration)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            bin_size = mask.sum()

            if bin_size > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)

        return float(ece)

    def get_calibration_report(self, market: str) -> str:
        """
        Get human-readable calibration report for a market.

        Args:
            market: Market name

        Returns:
            Formatted report string
        """
        if market not in self.calibration_metrics:
            return f"No calibration data for {market}"

        m = self.calibration_metrics[market]

        return f"""
Calibration Report: {market}
{'=' * 40}
Samples: {m['n_samples']}
Actual Positive Rate: {m['actual_positive_rate']:.1%}

Raw Probabilities:
  Mean: {m['mean_raw_prob']:.1%}
  ECE:  {m['ece_raw']:.4f}

Calibrated Probabilities:
  Mean: {m['mean_calibrated_prob']:.1%}
  ECE:  {m['ece_calibrated']:.4f}

ECE Improvement: {m['ece_improvement']:.1f}%
"""

    def save(self, path: Path) -> None:
        """
        Save calibrator to disk.

        Args:
            path: Path to save file (.joblib)
        """
        bundle = {
            'method': self.method,
            'calibrators': self.calibrators,
            'calibration_metrics': self.calibration_metrics,
        }
        joblib.dump(bundle, path)
        print(f"EdgeCalibrator saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> 'EdgeCalibrator':
        """
        Load calibrator from disk.

        Args:
            path: Path to saved file (.joblib)

        Returns:
            Loaded EdgeCalibrator instance
        """
        bundle = joblib.load(path)

        calibrator = cls(method=bundle.get('method', 'isotonic'))
        calibrator.calibrators = bundle.get('calibrators', {})
        calibrator.calibration_metrics = bundle.get('calibration_metrics', {})

        return calibrator

    def has_calibrator(self, market: str) -> bool:
        """Check if calibrator exists for a market."""
        return market in self.calibrators

    def get_markets(self) -> list:
        """Get list of calibrated markets."""
        return list(self.calibrators.keys())

    def __repr__(self) -> str:
        return (
            f"EdgeCalibrator(method='{self.method}', "
            f"markets={list(self.calibrators.keys())})"
        )
