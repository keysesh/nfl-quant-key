"""
Platt Calibrator Loader Utility
================================

REPLACEMENT for isotonic calibrator_loader.py.

Uses Platt scaling instead of isotonic regression to fix:
- Probability clipping issues (isotonic clips <51.9% to 0.0)
- Better generalization on full probability range [0, 1]
- No minimum sample requirements
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from .platt_calibrator import PlattCalibrator

logger = logging.getLogger(__name__)

# Cache loaded calibrators
_calibrator_cache: Dict[str, PlattCalibrator] = {}


def load_platt_calibrator_for_market(
    market: str,
    config_dir: str = 'data/calibration',
    use_cache: bool = True
) -> Optional[PlattCalibrator]:
    """
    Load market-specific Platt calibrator or return None (use shrinkage fallback).

    Priority:
    1. Market-specific: data/calibration/platt_{market}.pkl
    2. Unified fallback: data/calibration/platt_unified.pkl
    3. Return None (use 30% shrinkage in recommendations)

    Args:
        market: Market type (e.g., 'player_reception_yds', 'player_receptions')
        config_dir: Directory containing calibrator pickle files
        use_cache: Whether to use cached calibrators (default True)

    Returns:
        PlattCalibrator or None (if no calibrator available)
    """

    # Check cache first
    cache_key = f"{config_dir}/{market}"
    if use_cache and cache_key in _calibrator_cache:
        logger.debug(f"Using cached Platt calibrator for {market}")
        return _calibrator_cache[cache_key]

    # Try market-specific calibrator first
    market_calibrator_path = Path(config_dir) / f"platt_{market}.pkl"

    if market_calibrator_path.exists():
        try:
            import pickle
            with open(market_calibrator_path, 'rb') as f:
                calibrator = pickle.load(f)

            if not isinstance(calibrator, PlattCalibrator):
                raise ValueError(f"Invalid calibrator type: {type(calibrator)}")

            logger.info(f"✅ Loaded Platt calibrator for {market}")

            # Cache it
            if use_cache:
                _calibrator_cache[cache_key] = calibrator

            return calibrator

        except Exception as e:
            logger.warning(f"Failed to load Platt calibrator for {market}: {e}")
            logger.info("Trying unified fallback...")

    # Fall back to unified calibrator
    unified_calibrator_path = Path(config_dir) / "platt_unified.pkl"

    if unified_calibrator_path.exists():
        try:
            import pickle
            with open(unified_calibrator_path, 'rb') as f:
                calibrator = pickle.load(f)

            if not isinstance(calibrator, PlattCalibrator):
                raise ValueError(f"Invalid calibrator type: {type(calibrator)}")

            logger.info(f"⚠️  Using unified Platt calibrator for {market}")

            # Cache it
            if use_cache:
                _calibrator_cache[cache_key] = calibrator

            return calibrator

        except Exception as e:
            logger.warning(f"Failed to load unified Platt calibrator: {e}")

    # No calibrator available - fall back to shrinkage
    logger.info(f"ℹ️  No Platt calibrator for {market}, will use 30% shrinkage fallback")
    return None


def clear_calibrator_cache():
    """Clear calibrator cache (useful for testing/reloading)."""
    global _calibrator_cache
    _calibrator_cache = {}
    logger.info("Cleared calibrator cache")
