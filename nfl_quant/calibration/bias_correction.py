"""
Bias Correction Module for NFL QUANT

Applies market-specific bias corrections to model predictions based on
historical backtest analysis. This addresses systematic over-prediction
observed across all markets.

Usage:
    from nfl_quant.calibration.bias_correction import apply_bias_correction, get_correction_factor

    # Apply correction to a single value
    corrected_value = apply_bias_correction(predicted_value, market="player_reception_yds")

    # Get the correction factor for manual application
    factor = get_correction_factor(market="player_reception_yds")
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# Cache for loaded corrections
_corrections_cache: Optional[Dict] = None


def load_bias_corrections() -> Dict:
    """
    Load bias correction factors from config file.

    Returns:
        Dict with correction factors and market mappings
    """
    global _corrections_cache

    if _corrections_cache is not None:
        return _corrections_cache

    config_path = Path(__file__).parent.parent.parent / "configs" / "market_bias_corrections.json"

    if not config_path.exists():
        logger.warning(f"Bias correction config not found at {config_path}, using defaults")
        _corrections_cache = get_default_corrections()
        return _corrections_cache

    try:
        with open(config_path, 'r') as f:
            _corrections_cache = json.load(f)
        logger.info(f"Loaded bias corrections from {config_path}")
        return _corrections_cache
    except Exception as e:
        logger.error(f"Error loading bias corrections: {e}")
        _corrections_cache = get_default_corrections()
        return _corrections_cache


def get_default_corrections() -> Dict:
    """Return default bias corrections if config file is missing."""
    return {
        "corrections": {
            "passing_yards": {"factor": 0.866},
            "receiving_yards": {"factor": 0.863},
            "receptions": {"factor": 0.858},
            "rushing_yards": {"factor": 0.905},
            "rushing_attempts": {"factor": 0.95},
            "passing_tds": {"factor": 0.90},
            "passing_completions": {"factor": 0.87},
            "passing_attempts": {"factor": 0.92},
            "targets": {"factor": 0.87},
            "receiving_tds": {"factor": 0.85},
            "rushing_tds": {"factor": 0.88}
        },
        "market_mapping": {
            "player_pass_yds": "passing_yards",
            "player_reception_yds": "receiving_yards",
            "player_receptions": "receptions",
            "player_rush_yds": "rushing_yards",
            "player_rush_attempts": "rushing_attempts",
            "player_pass_tds": "passing_tds",
            "player_pass_completions": "passing_completions",
            "player_pass_attempts": "passing_attempts",
            "player_rush_reception_yds": "combined_yards",
            "player_anytime_td": "anytime_td",
            "player_1st_td": "first_td",
            "player_targets": "targets"
        }
    }


def get_correction_factor(market: str) -> float:
    """
    Get the bias correction factor for a specific market.

    Args:
        market: Market name (e.g., 'player_reception_yds' or 'receiving_yards')

    Returns:
        Correction factor (multiply prediction by this value)
    """
    config = load_bias_corrections()
    corrections = config.get("corrections", {})
    market_mapping = config.get("market_mapping", {})

    # Try direct lookup first
    if market in corrections:
        return corrections[market].get("factor", 1.0)

    # Try mapping from DraftKings market name
    mapped_market = market_mapping.get(market)
    if mapped_market and mapped_market in corrections:
        return corrections[mapped_market].get("factor", 1.0)

    # Handle combined yards specially
    if market in ["player_rush_reception_yds", "combined_yards"]:
        rush_factor = corrections.get("rushing_yards", {}).get("factor", 0.905)
        rec_factor = corrections.get("receiving_yards", {}).get("factor", 0.863)
        # Use weighted average
        return (rush_factor + rec_factor) / 2

    # Default: no correction
    logger.debug(f"No bias correction found for market '{market}', using factor=1.0")
    return 1.0


def get_additive_bias(market: str) -> float:
    """
    Get the ADDITIVE bias correction for a specific market.

    Formula: corrected = prediction - additive_bias

    Args:
        market: Market name (e.g., 'player_reception_yds' or 'rushing_yards')

    Returns:
        Additive bias value (subtract from prediction)
    """
    config = load_bias_corrections()
    corrections = config.get("corrections", {})
    market_mapping = config.get("market_mapping", {})

    # Try direct lookup first
    if market in corrections:
        return corrections[market].get("additive_bias", 0.0)

    # Try mapping from DraftKings market name
    mapped_market = market_mapping.get(market)
    if mapped_market and mapped_market in corrections:
        return corrections[mapped_market].get("additive_bias", 0.0)

    # Handle combined yards specially
    if market in ["player_rush_reception_yds", "combined_yards"]:
        rush_bias = corrections.get("rushing_yards", {}).get("additive_bias", 0.0)
        rec_bias = corrections.get("receiving_yards", {}).get("additive_bias", 0.0)
        # Use weighted average
        return (rush_bias + rec_bias) / 2

    # Default: no correction
    return 0.0


def apply_bias_correction(
    value: Union[float, int],
    market: str,
    apply_to_std: bool = False
) -> float:
    """
    Apply bias correction to a predicted value.

    Uses ADDITIVE bias correction (P0 fix, 2025-12-07):
        corrected = prediction - additive_bias

    Falls back to multiplicative factor if no additive bias is set.

    Args:
        value: The predicted value to correct
        market: Market name (e.g., 'player_reception_yds')
        apply_to_std: If True, also scale the std deviation (default False)

    Returns:
        Corrected value
    """
    if value is None or value <= 0:
        return value

    # First apply additive correction (P0 fix)
    additive_bias = get_additive_bias(market)
    corrected = value - additive_bias

    # Then apply multiplicative factor (legacy, usually 1.0)
    factor = get_correction_factor(market)
    corrected = corrected * factor

    # Ensure non-negative predictions
    corrected = max(0.0, corrected)

    return corrected


def apply_corrections_to_predictions(predictions: Dict[str, float], market: str) -> Dict[str, float]:
    """
    Apply bias corrections to a dictionary of predictions.

    Uses ADDITIVE bias correction (P0 fix, 2025-12-07):
        corrected = prediction - additive_bias

    Args:
        predictions: Dict with keys like 'mean', 'std', 'median', etc.
        market: Market name for correction lookup

    Returns:
        Corrected predictions dict
    """
    additive_bias = get_additive_bias(market)
    factor = get_correction_factor(market)

    corrected = predictions.copy()

    # Apply additive correction then multiplicative factor to mean values
    if 'mean' in corrected:
        corrected['mean'] = max(0.0, (corrected['mean'] - additive_bias) * factor)
    if 'median' in corrected:
        corrected['median'] = max(0.0, (corrected['median'] - additive_bias) * factor)

    # Apply to percentiles
    for key in ['p10', 'p25', 'p50', 'p75', 'p90']:
        if key in corrected:
            corrected[key] = max(0.0, (corrected[key] - additive_bias) * factor)

    # Scale std proportionally to maintain coefficient of variation
    # Note: additive bias doesn't change std, only multiplicative factor does
    if 'std' in corrected and corrected.get('mean', 0) > 0:
        corrected['std'] = corrected['std'] * factor

    return corrected


def get_stat_to_market_mapping() -> Dict[str, str]:
    """
    Get mapping from internal stat names to market names.

    Returns:
        Dict mapping stat names (rushing_yards) to markets (player_rush_yds)
    """
    return {
        "passing_yards": "player_pass_yds",
        "receiving_yards": "player_reception_yds",
        "receptions": "player_receptions",
        "rushing_yards": "player_rush_yds",
        "rushing_attempts": "player_rush_attempts",
        "passing_tds": "player_pass_tds",
        "passing_completions": "player_pass_completions",
        "passing_attempts": "player_pass_attempts",
        "targets": "player_targets",
        "receiving_tds": "player_anytime_td",
        "rushing_tds": "player_anytime_td"
    }


def clear_cache():
    """Clear the corrections cache (useful for testing)."""
    global _corrections_cache
    _corrections_cache = None
