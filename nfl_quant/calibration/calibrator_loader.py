"""
Calibrator Loader Utility
==========================

Handles loading the appropriate calibrator based on market type.
Supports market-specific calibrators with fallback to unified calibrator.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from .isotonic_calibrator import NFLProbabilityCalibrator

logger = logging.getLogger(__name__)

# Cache loaded calibrators to avoid repeated file I/O
_calibrator_cache: Dict[str, NFLProbabilityCalibrator] = {}


def load_calibrator_for_market(
    market: str,
    config_dir: str = 'configs',
    use_cache: bool = True
) -> NFLProbabilityCalibrator:
    """
    Load market-specific calibrator or fall back to unified calibrator.

    Priority:
    1. Market-specific: configs/calibrator_{market}.json
    2. Unified fallback: configs/calibrator.json

    Args:
        market: Market type (e.g., 'player_reception_yds', 'player_receptions')
        config_dir: Directory containing calibrator config files
        use_cache: Whether to use cached calibrators (default True)

    Returns:
        Loaded NFLProbabilityCalibrator with hybrid parameters

    Raises:
        FileNotFoundError: If neither market-specific nor unified calibrator exists
        ValueError: If calibrator file is invalid or missing hybrid parameters
    """

    # Check cache first
    cache_key = f"{config_dir}/{market}"
    if use_cache and cache_key in _calibrator_cache:
        logger.debug(f"Using cached calibrator for {market}")
        return _calibrator_cache[cache_key]

    # Try market-specific calibrator first
    market_calibrator_path = Path(config_dir) / f"calibrator_{market}.json"

    if market_calibrator_path.exists():
        try:
            calibrator = NFLProbabilityCalibrator()
            calibrator.load(str(market_calibrator_path))

            # Validate hybrid parameters
            if not hasattr(calibrator, 'high_prob_threshold'):
                logger.warning(f"Market calibrator {market} missing hybrid parameters, adding defaults")
                calibrator.high_prob_threshold = 0.70
                calibrator.high_prob_shrinkage = 0.3

            logger.info(f"✅ Loaded market-specific calibrator for {market}")
            logger.info(f"   Threshold: {calibrator.high_prob_threshold:.2f}, Shrinkage: {calibrator.high_prob_shrinkage:.2f}")

            # Cache it
            if use_cache:
                _calibrator_cache[cache_key] = calibrator

            return calibrator

        except Exception as e:
            logger.warning(f"Failed to load market-specific calibrator for {market}: {e}")
            logger.info("Falling back to unified calibrator...")

    # Fall back to unified calibrator
    unified_calibrator_path = Path(config_dir) / "calibrator.json"

    if not unified_calibrator_path.exists():
        raise FileNotFoundError(
            f"Neither market-specific ({market_calibrator_path}) "
            f"nor unified ({unified_calibrator_path}) calibrator found"
        )

    try:
        calibrator = NFLProbabilityCalibrator()
        calibrator.load(str(unified_calibrator_path))

        # Validate hybrid parameters
        if not hasattr(calibrator, 'high_prob_threshold'):
            logger.warning("Unified calibrator missing hybrid parameters, adding defaults")
            calibrator.high_prob_threshold = 0.70
            calibrator.high_prob_shrinkage = 0.3

        logger.info(f"⚠️  Using unified calibrator for {market} (market-specific not available)")
        logger.info(f"   Threshold: {calibrator.high_prob_threshold:.2f}, Shrinkage: {calibrator.high_prob_shrinkage:.2f}")

        # Cache it
        if use_cache:
            _calibrator_cache[cache_key] = calibrator

        return calibrator

    except Exception as e:
        raise ValueError(f"Failed to load unified calibrator: {e}")


def load_game_line_calibrator(config_dir: str = 'configs') -> NFLProbabilityCalibrator:
    """
    Load the game line calibrator (for spreads, totals, moneylines).

    Args:
        config_dir: Directory containing calibrator config files

    Returns:
        Loaded game line calibrator
    """

    calibrator_path = Path(config_dir) / "game_line_calibrator.json"

    if not calibrator_path.exists():
        raise FileNotFoundError(f"Game line calibrator not found: {calibrator_path}")

    calibrator = NFLProbabilityCalibrator()
    calibrator.load(str(calibrator_path))

    logger.info("✅ Loaded game line calibrator")
    logger.info(f"   Threshold: {calibrator.high_prob_threshold:.2f}, Shrinkage: {calibrator.high_prob_shrinkage:.2f}")

    return calibrator


def clear_calibrator_cache():
    """Clear the calibrator cache. Useful for testing or reloading updated calibrators."""
    global _calibrator_cache
    _calibrator_cache.clear()
    logger.info("Calibrator cache cleared")


def get_available_market_calibrators(config_dir: str = 'configs') -> list:
    """
    Get list of markets that have specific calibrators.

    Args:
        config_dir: Directory containing calibrator config files

    Returns:
        List of market names with specific calibrators
    """

    config_path = Path(config_dir)
    calibrator_files = list(config_path.glob("calibrator_player_*.json"))

    markets = []
    for file_path in calibrator_files:
        # Extract market name from filename
        # E.g., "calibrator_player_reception_yds.json" -> "player_reception_yds"
        market_name = file_path.stem.replace("calibrator_", "")
        markets.append(market_name)

    return sorted(markets)


def validate_all_calibrators(config_dir: str = 'configs') -> Dict[str, bool]:
    """
    Validate all calibrator files can be loaded successfully.

    Args:
        config_dir: Directory containing calibrator config files

    Returns:
        Dictionary mapping calibrator names to validation status
    """

    results = {}

    # Check unified calibrator
    try:
        calibrator = NFLProbabilityCalibrator()
        calibrator.load(f"{config_dir}/calibrator.json")
        assert calibrator.is_fitted
        assert hasattr(calibrator, 'high_prob_threshold')
        results['unified'] = True
        logger.info("✅ Unified calibrator validation passed")
    except Exception as e:
        results['unified'] = False
        logger.error(f"❌ Unified calibrator validation failed: {e}")

    # Check game line calibrator
    try:
        calibrator = NFLProbabilityCalibrator()
        calibrator.load(f"{config_dir}/game_line_calibrator.json")
        assert calibrator.is_fitted
        assert hasattr(calibrator, 'high_prob_threshold')
        results['game_line'] = True
        logger.info("✅ Game line calibrator validation passed")
    except Exception as e:
        results['game_line'] = False
        logger.error(f"❌ Game line calibrator validation failed: {e}")

    # Check market-specific calibrators
    markets = get_available_market_calibrators(config_dir)
    for market in markets:
        try:
            calibrator = load_calibrator_for_market(market, config_dir, use_cache=False)
            assert calibrator.is_fitted
            assert hasattr(calibrator, 'high_prob_threshold')
            results[market] = True
            logger.info(f"✅ {market} calibrator validation passed")
        except Exception as e:
            results[market] = False
            logger.error(f"❌ {market} calibrator validation failed: {e}")

    return results
