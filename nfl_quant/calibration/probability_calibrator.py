"""
Probability Calibration Module
===============================

Applies trained isotonic calibrators to model probabilities to improve accuracy.
Addresses systematic biases (OVER/UNDER, overconfidence) identified in backtesting.

Usage:
    from nfl_quant.calibration import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator(calibration_dir='data/calibration')
    calibrated_prob = calibrator.calibrate(
        raw_prob=0.75,
        market='player_receptions',
        side='OVER'
    )
"""

import pickle
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Calibrates model probabilities using trained isotonic regressors."""

    def __init__(self, calibration_dir: Path, strategy: str = 'side'):
        """
        Initialize calibrator.

        Args:
            calibration_dir: Directory containing calibrator pickle files
            strategy: Calibration strategy to use:
                - 'overall': Single calibrator for all bets
                - 'market': Separate calibrator per market
                - 'side': Separate calibrator for OVER vs UNDER (recommended)
        """
        self.calibration_dir = Path(calibration_dir)
        self.strategy = strategy
        self.calibrators = {}
        self._load_calibrators()

    def _load_calibrators(self):
        """Load calibrators from disk."""
        if self.strategy == 'overall':
            overall_path = self.calibration_dir / 'calibrator_overall.pkl'
            if overall_path.exists():
                with open(overall_path, 'rb') as f:
                    self.calibrators['overall'] = pickle.load(f)
            else:
                raise FileNotFoundError(f"Overall calibrator not found: {overall_path}")

        elif self.strategy == 'market':
            # Load all market-specific calibrators
            for cal_file in self.calibration_dir.glob('calibrator_player*.pkl'):
                market_name = cal_file.stem.replace('calibrator_', '')
                with open(cal_file, 'rb') as f:
                    self.calibrators[market_name] = pickle.load(f)

            if len(self.calibrators) == 0:
                raise FileNotFoundError(f"No market calibrators found in {self.calibration_dir}")

        elif self.strategy == 'side':
            # Load OVER/UNDER calibrators
            over_path = self.calibration_dir / 'calibrator_side_over.pkl'
            under_path = self.calibration_dir / 'calibrator_side_under.pkl'

            if over_path.exists() and under_path.exists():
                with open(over_path, 'rb') as f:
                    self.calibrators['OVER'] = pickle.load(f)
                with open(under_path, 'rb') as f:
                    self.calibrators['UNDER'] = pickle.load(f)
            else:
                raise FileNotFoundError(f"Side calibrators not found in {self.calibration_dir}")

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'overall', 'market', or 'side'")

    def calibrate(
        self,
        raw_prob: float,
        market: Optional[str] = None,
        side: Optional[str] = None
    ) -> float:
        """
        Calibrate a raw model probability.

        Args:
            raw_prob: Raw probability from model (0-1)
            market: Market type (e.g., 'player_receptions') - required for 'market' strategy
            side: Bet side ('OVER' or 'UNDER') - required for 'side' strategy

        Returns:
            Calibrated probability (0-1)
        """
        # Clip to valid probability range
        raw_prob = np.clip(raw_prob, 0.0, 1.0)

        if self.strategy == 'overall':
            calibrator = self.calibrators['overall']
            return float(calibrator.predict([raw_prob])[0])

        elif self.strategy == 'market':
            if market is None:
                raise ValueError("Market must be specified for 'market' strategy")

            # Normalize market name
            market_key = market.lower().replace('_', '').replace(' ', '')

            # Try to find matching calibrator
            calibrator = None
            for key, cal in self.calibrators.items():
                if key in market_key or market_key in key:
                    calibrator = cal
                    break

            if calibrator is None:
                # Fallback to overall if available
                overall_path = self.calibration_dir / 'calibrator_overall.pkl'
                if overall_path.exists():
                    if 'overall' not in self.calibrators:
                        with open(overall_path, 'rb') as f:
                            self.calibrators['overall'] = pickle.load(f)
                    calibrator = self.calibrators['overall']
                else:
                    # No calibration available, return raw probability
                    return raw_prob

            return float(calibrator.predict([raw_prob])[0])

        elif self.strategy == 'side':
            if side is None:
                raise ValueError("Side must be specified for 'side' strategy")

            side = side.upper()
            if side not in self.calibrators:
                # Fallback to overall if available
                overall_path = self.calibration_dir / 'calibrator_overall.pkl'
                if overall_path.exists():
                    if 'overall' not in self.calibrators:
                        with open(overall_path, 'rb') as f:
                            self.calibrators['overall'] = pickle.load(f)
                    calibrator = self.calibrators['overall']
                else:
                    return raw_prob
            else:
                calibrator = self.calibrators[side]

            return float(calibrator.predict([raw_prob])[0])

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
            markets: Array of market types (for 'market' strategy)
            sides: Array of bet sides (for 'side' strategy)

        Returns:
            Array of calibrated probabilities
        """
        calibrated = np.zeros_like(raw_probs)

        for i, raw_prob in enumerate(raw_probs):
            market = markets[i] if markets is not None else None
            side = sides[i] if sides is not None else None
            calibrated[i] = self.calibrate(raw_prob, market, side)

        return calibrated


def create_calibrator(
    calibration_dir: str = 'data/calibration',
    strategy: str = 'side'
) -> Optional[ProbabilityCalibrator]:
    """
    Create a calibrator instance if calibration files exist.

    Args:
        calibration_dir: Directory containing calibrator files
        strategy: Calibration strategy ('overall', 'market', or 'side')

    Returns:
        ProbabilityCalibrator instance or None if files don't exist
    """
    cal_dir = Path(calibration_dir)

    if not cal_dir.exists():
        return None

    try:
        return ProbabilityCalibrator(cal_dir, strategy)
    except FileNotFoundError:
        return None


# Example usage
if __name__ == "__main__":
    # Create calibrator
    calibrator = create_calibrator(strategy='side')

    if calibrator:
        # Test calibration
        raw_prob_over = 0.75  # Model thinks 75% chance of OVER
        raw_prob_under = 0.65  # Model thinks 65% chance of UNDER

        calibrated_over = calibrator.calibrate(raw_prob_over, side='OVER')
        calibrated_under = calibrator.calibrate(raw_prob_under, side='UNDER')

        print(f"OVER: {raw_prob_over:.1%} → {calibrated_over:.1%}")
        print(f"UNDER: {raw_prob_under:.1%} → {calibrated_under:.1%}")
    else:
        print("No calibrators found - run train_calibration.py first")
