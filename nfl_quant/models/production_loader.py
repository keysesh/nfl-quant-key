"""
Production Model Loader

Load trained models for live prediction.
"""

import joblib
import lightgbm as lgb
from pathlib import Path
import logging

from nfl_quant.features.feature_defaults import safe_fillna, FEATURE_DEFAULTS
from nfl_quant.validation.input_validation import validate_and_log

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_production_model(name: str):
    """Load a production model by name."""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'production' / name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model {name} not found at {model_dir}")

    model = lgb.Booster(model_file=str(model_dir / 'model.txt'))
    calibrator = joblib.load(model_dir / 'calibrator.joblib')
    config = joblib.load(model_dir / 'config.joblib')

    return model, calibrator, config


def get_deployable_models():
    """Get list of all deployable models."""
    import json
    config_path = PROJECT_ROOT / 'data' / 'models' / 'production' / 'deployment_config.json'

    with open(config_path) as f:
        config = json.load(f)

    return {
        'high': config['high_confidence'],
        'medium': config['medium_confidence'],
        'low': config['low_confidence'],
    }


def predict_prop(model_name: str, features: dict, over_odds: int = -110, under_odds: int = -110) -> dict:
    """
    Make prediction for a single prop with Kelly criterion bet sizing.

    Args:
        model_name: Name of production model
        features: Dictionary of features
        over_odds: American odds for over (default -110)
        under_odds: American odds for under (default -110)

    Returns:
        Dictionary with prediction and bet sizing information
    """
    model, calibrator, config = load_production_model(model_name)

    # Prepare features
    import pandas as pd
    X = pd.DataFrame([features])[config['feature_names']]

    # Apply semantic defaults instead of blanket 0-fill
    X = safe_fillna(X, FEATURE_DEFAULTS)

    # Validate before prediction
    if not validate_and_log(X, context="production prediction"):
        logger.warning("Proceeding with prediction despite validation issues")

    X = X.astype(float)

    # Predict
    raw_prob = model.predict(X)[0]
    calibrated_prob = calibrator.predict([raw_prob])[0]

    threshold = config['optimal_threshold']

    recommendation = None
    bet_odds = None
    edge = 0.0

    if calibrated_prob > threshold:
        recommendation = 'OVER'
        bet_odds = over_odds
        # Calculate edge vs fair probability (vig-removed)
        over_implied = 100 / (over_odds + 100) if over_odds > 0 else abs(over_odds) / (abs(over_odds) + 100)
        under_implied = 100 / (under_odds + 100) if under_odds > 0 else abs(under_odds) / (abs(under_odds) + 100)
        fair_over = over_implied / (over_implied + under_implied)
        edge = calibrated_prob - fair_over
    elif calibrated_prob < (1 - threshold):
        recommendation = 'UNDER'
        bet_odds = under_odds
        over_implied = 100 / (over_odds + 100) if over_odds > 0 else abs(over_odds) / (abs(over_odds) + 100)
        under_implied = 100 / (under_odds + 100) if under_odds > 0 else abs(under_odds) / (abs(under_odds) + 100)
        fair_under = under_implied / (over_implied + under_implied)
        edge = (1 - calibrated_prob) - fair_under

    # Calculate Kelly criterion bet sizing
    bet_sizing = {'should_bet': False, 'units': 0, 'bet_amount': 0}
    if recommendation and bet_odds:
        try:
            from nfl_quant.betting.bet_sizing import calculate_bet_size
            market_type = 'player_' + model_name.split('_')[0] + '_' + model_name.split('_')[1] if '_' in model_name else 'player_receptions'
            bet_sizing = calculate_bet_size(
                win_prob=calibrated_prob if recommendation == 'OVER' else (1 - calibrated_prob),
                odds=bet_odds,
                edge=edge,
                market=market_type
            )
        except ImportError:
            # Fallback if bet_sizing module not available
            bet_sizing = {'should_bet': edge > 0.03, 'units': 1 if edge > 0.03 else 0, 'bet_amount': 100 if edge > 0.03 else 0}

    return {
        'model': model_name,
        'probability': float(calibrated_prob),
        'threshold': threshold,
        'recommendation': recommendation,
        'expected_roi': config['expected_roi'],
        'confidence': 'HIGH' if config['is_significant'] and config['ci_lower'] > 0 else 'MEDIUM',
        'edge': edge,
        'bet_sizing': bet_sizing,
    }
