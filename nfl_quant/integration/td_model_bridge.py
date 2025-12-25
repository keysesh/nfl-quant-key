"""
TD Model Bridge

Provides P(UNDER) predictions for TD props using:
- Poisson regression for player_pass_tds (count data: 0, 1, 2, 3+)
- The existing RedZoneTDModel infrastructure as fallback

Usage:
    from nfl_quant.integration.td_model_bridge import get_td_prediction

    result = get_td_prediction(
        market='player_pass_tds',
        line=1.5,
        trailing_pass_tds=1.8,
        trailing_td_rate=0.05
    )
    # Returns: {'p_under': 0.55, 'predicted_lambda': 1.3, ...}
"""

import logging
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import joblib
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Singleton for model cache
_td_model_data: Optional[Dict] = None
_model_loaded: bool = False


def _load_td_model() -> Dict:
    """Load the TD Poisson model."""
    global _td_model_data, _model_loaded

    if _model_loaded:
        return _td_model_data

    model_path = Path('data/models/td_poisson_model.joblib')

    if not model_path.exists():
        logger.warning(f"TD model not found at {model_path}")
        _model_loaded = True
        _td_model_data = None
        return None

    try:
        _td_model_data = joblib.load(model_path)
        logger.info(f"Loaded TD model version: {_td_model_data.get('version', 'unknown')}")
        _model_loaded = True
    except Exception as e:
        logger.error(f"Failed to load TD model: {e}")
        _model_loaded = True
        _td_model_data = None

    return _td_model_data


def get_td_prediction(
    market: str,
    line: float,
    trailing_pass_tds: float = 1.5,
    trailing_td_rate: float = 0.05,
    **kwargs
) -> Dict:
    """
    Get P(UNDER) prediction for a TD prop.

    Args:
        market: TD market type (player_pass_tds, player_anytime_td)
        line: Betting line (e.g., 1.5 for over/under 1.5 TDs)
        trailing_pass_tds: EWMA of player's recent passing TDs
        trailing_td_rate: TD per attempt rate

    Returns:
        Dict with:
            - p_under: Probability actual < line
            - p_over: Probability actual >= line
            - predicted_lambda: Expected TD count (Poisson)
            - source: 'poisson_model' or 'fallback'
    """
    model_data = _load_td_model()

    if model_data is None or market not in model_data.get('models', {}):
        # Fallback: simple heuristic
        return _fallback_prediction(market, line, trailing_pass_tds)

    model = model_data['models'][market]
    features = model_data['features'].get(market, [])
    model_type = model_data.get('model_types', {}).get(market, 'poisson')

    # Build feature vector
    feature_values = {
        'trailing_pass_tds': trailing_pass_tds,
        'trailing_td_rate': trailing_td_rate,
        'line': line,
    }

    X = [[feature_values.get(f, 0) for f in features]]

    if model_type == 'poisson':
        # Predict expected TD count (lambda)
        predicted_lambda = model.predict(X)[0]

        # P(UNDER line) = P(X < line) = P(X <= floor(line - 0.5))
        # For line=1.5: P(X <= 1) = P(0) + P(1)
        p_under = scipy_stats.poisson.cdf(int(line - 0.5), predicted_lambda)

        return {
            'p_under': float(p_under),
            'p_over': float(1 - p_under),
            'predicted_lambda': float(predicted_lambda),
            'direction': 'UNDER' if p_under > 0.5 else 'OVER',
            'edge': abs(p_under - 0.5) * 2,  # 0-1 scale
            'source': 'poisson_model',
            'market': market,
            'line': line,
        }

    elif model_type == 'logistic':
        # For anytime TD, predict P(scores at least 1)
        p_over = model.predict_proba(X)[0, 1]
        p_under = 1 - p_over

        return {
            'p_under': float(p_under),
            'p_over': float(p_over),
            'predicted_lambda': None,  # Not applicable for binary
            'direction': 'UNDER' if p_under > 0.5 else 'OVER',
            'edge': abs(p_under - 0.5) * 2,
            'source': 'logistic_model',
            'market': market,
            'line': line,
        }

    return _fallback_prediction(market, line, trailing_pass_tds)


def _fallback_prediction(
    market: str,
    line: float,
    trailing_tds: float
) -> Dict:
    """
    Fallback prediction when model is not available.

    Uses simple Poisson with trailing_tds as lambda.
    """
    # Use trailing TDs as lambda estimate
    predicted_lambda = max(trailing_tds, 0.5)

    # P(UNDER line)
    p_under = scipy_stats.poisson.cdf(int(line - 0.5), predicted_lambda)

    return {
        'p_under': float(p_under),
        'p_over': float(1 - p_under),
        'predicted_lambda': float(predicted_lambda),
        'direction': 'UNDER' if p_under > 0.5 else 'OVER',
        'edge': abs(p_under - 0.5) * 2,
        'source': 'fallback',
        'market': market,
        'line': line,
    }


# TD Markets that use this model
TD_MARKETS = ['player_pass_tds', 'player_anytime_td']


def is_td_market(market: str) -> bool:
    """Check if a market is a TD market."""
    return market in TD_MARKETS or 'td' in market.lower()
