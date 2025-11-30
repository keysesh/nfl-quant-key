"""
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
