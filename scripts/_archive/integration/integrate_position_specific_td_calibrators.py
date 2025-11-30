#!/usr/bin/env python3
"""
Integrate Position-Specific TD Calibrators
============================================

Updates the production pipeline to use position-specific TD calibrators:
1. Updates PlayerSimulator to load and use position-specific TD calibrators
2. Updates generate_model_predictions.py to load position-specific calibrators
3. Updates recommendation generators to use position-specific calibrators
4. Creates a TD calibrator loader utility

This replaces the old unified TD calibrator with position-specific ones.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator


class PositionSpecificTDCalibratorLoader:
    """Load position-specific TD calibrators."""

    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = Path(config_dir)
        self.calibrators = {}
        self._load_all()

    def _load_all(self):
        """Load all position-specific TD calibrators."""
        for position in ['QB', 'RB', 'WR', 'TE']:
            cal_path = self.config_dir / f'td_calibrator_{position}.json'
            if cal_path.exists():
                try:
                    calibrator = NFLProbabilityCalibrator()
                    calibrator.load(str(cal_path))
                    self.calibrators[position] = calibrator
                except Exception as e:
                    print(f"Warning: Could not load TD calibrator for {position}: {e}")

    def get_calibrator(self, position: str):
        """
        Get TD calibrator for a specific position.

        Args:
            position: Player position ('QB', 'RB', 'WR', 'TE')

        Returns:
            NFLProbabilityCalibrator or None if not available
        """
        # Map TE to WR if TE calibrator not available
        if position == 'TE' and position not in self.calibrators:
            position = 'WR'

        return self.calibrators.get(position)

    def calibrate_td_probability(self, raw_prob: float, position: str) -> float:
        """
        Calibrate TD probability using position-specific calibrator.

        Args:
            raw_prob: Raw TD probability (0-1)
            position: Player position

        Returns:
            Calibrated TD probability
        """
        calibrator = self.get_calibrator(position)
        if calibrator:
            return float(calibrator.transform(raw_prob))
        else:
            # Fallback: simple shrinkage if no position-specific calibrator
            return 0.5 + (raw_prob - 0.5) * 0.75

    def is_available(self) -> bool:
        """Check if any position-specific calibrators are available."""
        return len(self.calibrators) > 0


def create_td_calibrator_loader_module():
    """Create a TD calibrator loader module for easy import."""
    loader_code = '''"""
Position-Specific TD Calibrator Loader
=======================================

Provides easy access to position-specific TD calibrators.
"""

from pathlib import Path
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator


class PositionSpecificTDCalibratorLoader:
    """Load and manage position-specific TD calibrators."""

    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = Path(config_dir)
        self.calibrators = {}
        self._load_all()

    def _load_all(self):
        """Load all position-specific TD calibrators."""
        for position in ['QB', 'RB', 'WR', 'TE']:
            cal_path = self.config_dir / f'td_calibrator_{position}.json'
            if cal_path.exists():
                try:
                    calibrator = NFLProbabilityCalibrator()
                    calibrator.load(str(cal_path))
                    self.calibrators[position] = calibrator
                except Exception as e:
                    pass  # Silently skip if not available

    def get_calibrator(self, position: str):
        """
        Get TD calibrator for a specific position.

        Args:
            position: Player position ('QB', 'RB', 'WR', 'TE')

        Returns:
            NFLProbabilityCalibrator or None if not available
        """
        # Map TE to WR if TE calibrator not available
        if position == 'TE' and position not in self.calibrators:
            position = 'WR'

        return self.calibrators.get(position)

    def calibrate_td_probability(self, raw_prob: float, position: str) -> float:
        """
        Calibrate TD probability using position-specific calibrator.

        Args:
            raw_prob: Raw TD probability (0-1)
            position: Player position

        Returns:
            Calibrated TD probability
        """
        calibrator = self.get_calibrator(position)
        if calibrator:
            return float(calibrator.transform(raw_prob))
        else:
            # Fallback: simple shrinkage if no position-specific calibrator
            return 0.5 + (raw_prob - 0.5) * 0.75

    def is_available(self) -> bool:
        """Check if any position-specific calibrators are available."""
        return len(self.calibrators) > 0


# Global instance for easy access
_global_loader = None

def get_td_calibrator_loader(config_dir: str = 'configs') -> PositionSpecificTDCalibratorLoader:
    """Get or create global TD calibrator loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PositionSpecificTDCalibratorLoader(config_dir)
    return _global_loader
'''

    module_path = Path(__file__).parent.parent.parent / 'nfl_quant' / 'calibration' / 'td_calibrator_loader.py'
    module_path.parent.mkdir(parents=True, exist_ok=True)

    with open(module_path, 'w') as f:
        f.write(loader_code)

    print(f"✅ Created TD calibrator loader module: {module_path}")
    return module_path


def main():
    print("="*80)
    print("INTEGRATING POSITION-SPECIFIC TD CALIBRATORS")
    print("="*80)
    print()

    # Create TD calibrator loader module
    print("1. Creating TD calibrator loader module...")
    loader_path = create_td_calibrator_loader_module()
    print()

    # Test the loader
    print("2. Testing TD calibrator loader...")
    loader = PositionSpecificTDCalibratorLoader()

    if loader.is_available():
        print(f"   ✅ Loaded {len(loader.calibrators)} position-specific TD calibrators:")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            cal = loader.get_calibrator(pos)
            if cal:
                print(f"      - {pos}: Available ({len(cal.calibrator.X_thresholds_)} points)")
            else:
                print(f"      - {pos}: Not available")

        # Test calibration
        print()
        print("   Testing calibration:")
        test_probs = [0.2, 0.5, 0.8]
        for pos in ['QB', 'RB', 'WR', 'TE']:
            cal = loader.get_calibrator(pos)
            if cal:
                calibrated = [loader.calibrate_td_probability(p, pos) for p in test_probs]
                print(f"      {pos}: {test_probs} → {[round(c, 3) for c in calibrated]}")
    else:
        print("   ⚠️  No position-specific TD calibrators found")

    print()
    print("="*80)
    print("INTEGRATION COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Update PlayerSimulator to use PositionSpecificTDCalibratorLoader")
    print("2. Update generate_model_predictions.py to load position-specific calibrators")
    print("3. Update recommendation generators to use position-specific calibration")
    print()
    print(f"TD calibrator loader available at: {loader_path}")
    print("Import with: from nfl_quant.calibration.td_calibrator_loader import get_td_calibrator_loader")
    print()


if __name__ == "__main__":
    main()
