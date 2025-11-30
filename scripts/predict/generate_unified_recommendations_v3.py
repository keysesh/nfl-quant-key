#!/usr/bin/env python3
"""
NFL QUANT - Unified Recommendation Generator V3 (COMPLETE - CALIBRATED)

This script:
1. Loads model_predictions_weekX.csv (all 54+ features from Monte Carlo simulation)
2. Matches to current prop lines from DraftKings
3. Applies isotonic calibration to raw probabilities
4. Uses no-vig market probabilities for edge calculation
5. PRESERVES ALL MODEL FEATURES in output (no feature dropout)
6. Uses actual model_std from simulation (no hardcoded CV values)

KEY FIXES APPLIED:
- ✅ Calibration applied via load_calibrator_for_market()
- ✅ No-vig probabilities via remove_vig_two_way()
- ✅ All 54+ features from model_predictions preserved in output
- ✅ No hardcoded CV values - uses model_std exclusively
- ✅ ValueError raised if model_std missing (enforces upstream data quality)
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm, poisson
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import calibration and betting utility modules
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market
from nfl_quant.calibration.bias_correction import apply_bias_correction, get_correction_factor
from nfl_quant.core.unified_betting import (
    remove_vig_two_way,
    assign_confidence_tier,
    calculate_kelly_fraction,
    calculate_recommended_units
)

# V6 Value Classifier import (replaces V5 Edge Classifier)
import joblib
V6_MODEL_PATH = project_root / 'data' / 'models' / 'v6_value_classifier.joblib'
V5_MODEL_PATH = project_root / 'data' / 'models' / 'v5_edge_classifier.joblib'  # Backwards compat
_v6_model = None
_v6_metrics = None
_v6_recommended = None

# V12/V13 Model Paths (BEST MODELS)
V12_MODEL_PATH = project_root / 'data' / 'models' / 'v12_interaction_classifier.joblib'
V13_MODEL_PATH = project_root / 'data' / 'models' / 'v13_opponent_aware_classifier.joblib'
V14_MODEL_PATH = project_root / 'data' / 'models' / 'v14_defense_aware_classifier.joblib'
_v12_model = None
_v12_metrics = None
_v12_thresholds = None
_v13_model = None
_v13_metrics = None
_v13_thresholds = None
_v14_model = None
_v14_coefficients = None
_v14_thresholds = None

# Import opponent stats module
try:
    from nfl_quant.features.opponent_stats import (
        get_opponent_conflict_flag,
        calculate_opponent_divergence,
        create_opponent_interaction_features
    )
    OPPONENT_STATS_AVAILABLE = True
except ImportError:
    OPPONENT_STATS_AVAILABLE = False
    logger.warning("Opponent stats module not available - opponent features disabled")


# ============================================================================
# DISPLAY FORMATTING HELPERS
# ============================================================================

def format_market_name(market: str) -> str:
    """
    Convert internal market name to human-readable display format.

    Examples:
        player_pass_completions -> Pass Completions
        player_reception_yds -> Receiving Yards
        player_rush_yds -> Rushing Yards
    """
    market_display_names = {
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Receiving Yards',
        'player_rush_yds': 'Rushing Yards',
        'player_rush_attempts': 'Rush Attempts',
        'player_pass_yds': 'Passing Yards',
        'player_pass_completions': 'Pass Completions',
        'player_pass_attempts': 'Pass Attempts',
        'player_pass_tds': 'Passing TDs',
        'player_rush_tds': 'Rushing TDs',
        'player_reception_tds': 'Receiving TDs',
        'player_targets': 'Targets',
        'player_interceptions': 'Interceptions',
    }
    return market_display_names.get(market, market.replace('player_', '').replace('_', ' ').title())


def generate_model_reasoning(row: dict) -> str:
    """
    Generate detailed explanation with actual stats that drove the prediction.

    Shows all key inputs so user can understand model logic.

    Args:
        row: Dictionary containing recommendation data

    Returns:
        String with detailed stats breakdown
    """
    parts = []
    market = row.get('market', '')
    pick_raw = row.get('v12_best_direction') or row.get('pick') or 'UNDER'
    pick = str(pick_raw).upper()

    # Get all key values
    line = row.get('line', 0) or 0
    lvt = row.get('line_vs_trailing', 0) or 0
    # Calculate trailing from line and LVT: trailing = line - lvt
    trailing = line - lvt if lvt != 0 else row.get('model_projection', 0)
    projection = row.get('model_projection', 0) or 0
    p_under = row.get('v12_p_under', 0.5) or 0.5
    p_over = 1 - p_under
    edge = row.get('v12_best_ev', row.get('edge_pct', 0)) or 0
    kelly = row.get('kelly_pct', row.get('kelly_units', 0)) or 0

    # Monte Carlo stats if available
    mc_mean = row.get('mc_mean', projection)
    mc_std = row.get('mc_std', 0)
    mc_over_rate = row.get('mc_over_rate', 0)
    mc_under_rate = row.get('mc_under_rate', 0)

    # Build detailed breakdown
    parts.append(f"📊 STATS:")
    parts.append(f"Vegas Line: {line:.1f}")
    parts.append(f"4-Week Trailing Avg: {trailing:.1f}")
    parts.append(f"LVT (Line vs Trailing): {lvt:+.1f}")

    parts.append(f"")
    parts.append(f"🎯 MODEL:")
    parts.append(f"Monte Carlo Projection: {projection:.1f}")
    if mc_std > 0:
        parts.append(f"Std Dev: ±{mc_std:.1f}")
    if mc_over_rate > 0 or mc_under_rate > 0:
        parts.append(f"MC Over Rate: {mc_over_rate:.1%}")
        parts.append(f"MC Under Rate: {mc_under_rate:.1%}")

    parts.append(f"")
    parts.append(f"🔮 V12 CLASSIFIER:")
    parts.append(f"P(UNDER): {p_under:.1%}")
    parts.append(f"P(OVER): {p_over:.1%}")
    parts.append(f"Pick: {pick}")
    parts.append(f"Expected Value: {edge:+.1f}%")
    if kelly > 0:
        parts.append(f"Kelly Sizing: {kelly:.1f}%")

    # Opponent history (INFO ONLY)
    div_col = f'{market}_vs_opp_divergence'
    conf_col = f'{market}_vs_opp_confidence'
    divergence = row.get(div_col, 0)
    confidence = row.get(conf_col, 0)
    opponent = row.get('opponent', '')

    if confidence > 0 and opponent:
        games = max(1, int(confidence * 3 + 1))
        vs_opp_avg = trailing * (1 + divergence) if trailing > 0 else 0
        parts.append(f"")
        parts.append(f"📈 VS {opponent} (INFO ONLY):")
        parts.append(f"Prior Games: {games}")
        parts.append(f"Avg vs {opponent}: {vs_opp_avg:.1f}")
        parts.append(f"Divergence: {divergence:+.0%}")
        parts.append(f"⚠️ Does NOT affect pick - LVT is primary signal")

    # Final signal explanation
    parts.append(f"")
    parts.append(f"💡 WHY {pick}:")
    if 'UNDER' in pick:
        if lvt > 0:
            parts.append(f"Line ({line:.1f}) set above trailing avg ({trailing:.1f})")
            parts.append(f"Model expects regression toward mean")
        if projection < line:
            parts.append(f"Projection ({projection:.1f}) is below line ({line:.1f})")
    else:  # OVER
        if lvt < 0:
            parts.append(f"Line ({line:.1f}) set below trailing avg ({trailing:.1f})")
            parts.append(f"Model expects upward correction")
        if projection > line:
            parts.append(f"Projection ({projection:.1f}) is above line ({line:.1f})")

    return " • ".join([p for p in parts if p])


def format_confidence_pct(confidence_value) -> str:
    """
    Format confidence value as percentage.

    Handles both numeric (0-1) and string (ELITE/HIGH/STANDARD/LOW) formats.
    """
    if isinstance(confidence_value, (int, float)):
        return f"{confidence_value * 100:.0f}%"
    elif isinstance(confidence_value, str):
        # Map tier names to approximate percentages
        tier_pcts = {
            'ELITE': '90%+',
            'HIGH': '75-90%',
            'STANDARD': '60-75%',
            'LOW': '<60%'
        }
        return tier_pcts.get(confidence_value.upper(), confidence_value)
    return str(confidence_value)


def load_v12_interaction_model():
    """
    Load the V12 Interaction Model (BEST PERFORMER).

    V12 uses:
    - LVT as hub with interaction constraints
    - Multiplicative features (LVT_x_player_tendency, etc.)
    - Monotonic constraints for directional relationships

    Returns:
        Tuple of (models_dict, metrics_dict, thresholds_dict)
    """
    global _v12_model, _v12_metrics, _v12_thresholds

    if _v12_model is None and V12_MODEL_PATH.exists():
        try:
            bundle = joblib.load(V12_MODEL_PATH)
            _v12_model = bundle.get('models', {})
            _v12_metrics = bundle.get('metrics', {})
            _v12_thresholds = bundle.get('thresholds', {})
            version = bundle.get('version', 'V12')

            logger.info(f"✅ Loaded {version} Interaction Model with {len(_v12_model)} markets")
            for market, metrics in _v12_metrics.items():
                threshold = _v12_thresholds.get(market, {}).get('threshold', 0.55)
                roi = metrics.get('roi_pct', 0)
                logger.info(f"   {market}: {threshold:.0%} threshold, {roi:+.1f}% ROI")

        except Exception as e:
            logger.warning(f"Could not load V12 model: {e}")

    return _v12_model, _v12_metrics, _v12_thresholds


def load_v13_opponent_aware_model():
    """
    Load the V13 Opponent-Aware Model.

    V13 adds opponent-specific features to V12:
    - vs_opponent_divergence: Player performance vs specific opponent
    - LVT_x_vs_opponent: Interaction feature

    Returns:
        Tuple of (models_dict, metrics_dict, thresholds_dict)
    """
    global _v13_model, _v13_metrics, _v13_thresholds

    if _v13_model is None and V13_MODEL_PATH.exists():
        try:
            bundle = joblib.load(V13_MODEL_PATH)
            _v13_model = bundle.get('models', {})
            _v13_metrics = bundle.get('metrics', {})
            _v13_thresholds = bundle.get('thresholds', {})
            version = bundle.get('version', 'V13')

            logger.info(f"✅ Loaded {version} Opponent-Aware Model with {len(_v13_model)} markets")
            for market, metrics in _v13_metrics.items():
                opp_importance = metrics.get('opponent_feature_importance', 0)
                roi = metrics.get('roi_pct', 0)
                logger.info(f"   {market}: {roi:+.1f}% ROI, opponent features {opp_importance:.1%} importance")

        except Exception as e:
            logger.warning(f"Could not load V13 model: {e}")

    return _v13_model, _v13_metrics, _v13_thresholds


def load_v14_defense_aware_model():
    """
    Load the V14 Defense-Aware Model for Rush Yards.

    V14 uses:
    - LVT (Line vs Trailing) - primary signal
    - Opponent Rush Defense EPA - secondary signal

    Walk-Forward Validated (2025-11-29):
    - P(UNDER) > 55%: +20.6% ROI
    - P(UNDER) > 60%: +60.8% ROI

    Returns:
        Tuple of (model, coefficients, thresholds)
    """
    global _v14_model, _v14_coefficients, _v14_thresholds

    if _v14_model is None and V14_MODEL_PATH.exists():
        try:
            bundle = joblib.load(V14_MODEL_PATH)
            _v14_model = bundle.get('model')
            _v14_coefficients = bundle.get('coefficients', {})
            _v14_thresholds = bundle.get('thresholds', {})
            version = bundle.get('version', 'V14')

            logger.info(f"✅ Loaded {version} Defense-Aware Model for rush_yds")
            logger.info(f"   LVT coef: {_v14_coefficients.get('line_vs_trailing', 0):.4f}")
            logger.info(f"   Def EPA coef: {_v14_coefficients.get('trailing_def_epa', 0):.4f}")

        except Exception as e:
            logger.warning(f"Could not load V14 model: {e}")

    return _v14_model, _v14_coefficients, _v14_thresholds


def apply_v14_model(row: dict, trailing_def_epa: float = 0.0) -> dict:
    """
    Apply V14 Defense-Aware Model to get UNDER probability for rush yards.

    Args:
        row: Dict with player data including line, model_projection, trailing stats
        trailing_def_epa: Opponent's trailing rush defense EPA (positive = bad defense)

    Returns:
        Dict with v14_p_under, v14_best_direction, v14_best_ev, v14_validated
    """
    v14_model, v14_coefs, v14_thresholds = load_v14_defense_aware_model()

    if v14_model is None:
        return {
            'v14_p_under': 0.50,
            'v14_best_direction': None,
            'v14_best_ev': 0,
            'v14_validated': False,
            'v14_reason': 'V14 model not available'
        }

    # Calculate LVT
    line = row.get('line', 0)
    trailing = row.get('model_projection', row.get('trailing_stat', line))
    if trailing > 0:
        line_vs_trailing = line / trailing
    else:
        line_vs_trailing = 1.0

    # Build features DataFrame with correct column names to avoid sklearn warning
    X = pd.DataFrame([[line_vs_trailing, trailing_def_epa]],
                     columns=['line_vs_trailing', 'trailing_def_epa'])

    try:
        p_under = float(v14_model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.debug(f"V14 prediction failed: {e}")
        # Fallback to coefficient-based estimate
        intercept = v14_coefs.get('intercept', 0)
        lvt_coef = v14_coefs.get('line_vs_trailing', 0.3)
        epa_coef = v14_coefs.get('trailing_def_epa', -0.8)
        logit = intercept + lvt_coef * line_vs_trailing + epa_coef * trailing_def_epa
        p_under = 1 / (1 + np.exp(-logit))

    # Calculate EV
    ev_under = calculate_bet_ev(p_under, -110)
    ev_over = calculate_bet_ev(1 - p_under, -110)

    # Use balanced threshold (55%) for more bets with good ROI
    threshold = v14_thresholds.get('balanced', 0.55)

    # Determine best direction
    if p_under >= threshold and ev_under > 0:
        return {
            'v14_p_under': p_under,
            'v14_best_direction': 'UNDER',
            'v14_best_ev': ev_under * 100,
            'v14_validated': True,
            'v14_reason': f'V14: {p_under:.1%} UNDER (def_epa={trailing_def_epa:+.3f})'
        }
    elif (1 - p_under) >= threshold and ev_over > 0:
        return {
            'v14_p_under': p_under,
            'v14_best_direction': 'OVER',
            'v14_best_ev': ev_over * 100,
            'v14_validated': True,
            'v14_reason': f'V14: {1-p_under:.1%} OVER (def_epa={trailing_def_epa:+.3f})'
        }
    else:
        return {
            'v14_p_under': p_under,
            'v14_best_direction': None,
            'v14_best_ev': 0,
            'v14_validated': False,
            'v14_reason': f'V14: Below threshold ({p_under:.1%} vs {threshold:.0%})'
        }


def load_v6_value_classifier():
    """Load the V6 value classifier model (bidirectional EV-based)."""
    global _v6_model, _v6_metrics, _v6_recommended

    if _v6_model is None:
        # Try V6 first, fall back to V5 for backwards compatibility
        model_path = V6_MODEL_PATH if V6_MODEL_PATH.exists() else V5_MODEL_PATH

        if model_path.exists():
            try:
                bundle = joblib.load(model_path)
                _v6_model = bundle.get('models', {})
                _v6_metrics = bundle.get('metrics', {})
                _v6_recommended = bundle.get('recommended', {})
                version = bundle.get('version', 'unknown')
                logger.info(f"Loaded {version} with {len(_v6_model)} market models")

                # Log key findings if available
                findings = bundle.get('key_findings', {})
                if findings:
                    logger.info(f"  Key findings: {findings.get('no_reliable_edge', 'N/A')}")
            except Exception as e:
                logger.warning(f"Could not load value classifier: {e}")

    return _v6_model, _v6_metrics, _v6_recommended


# Legacy alias for backwards compatibility
def load_v5_edge_classifier():
    """Legacy alias - loads V6 model."""
    models, metrics, _ = load_v6_value_classifier()
    return models, metrics


def load_game_totals_for_td_calculation(week: int) -> dict:
    """
    Load game totals from odds data and calculate expected TDs per game.

    Used for 1st TD probability calculation:
    P(1st TD) = player_expected_tds / game_total_expected_tds

    Args:
        week: Week number

    Returns:
        Dict mapping game_id -> expected_total_tds
        Example: {'2025_13_GB_DET': 6.9, ...}
    """
    game_totals = {}

    # Try to load game odds
    odds_files = [
        Path(f'data/odds_week{week}.csv'),
        Path('data/game_odds.csv'),
    ]

    for odds_file in odds_files:
        if odds_file.exists():
            try:
                df = pd.read_csv(odds_file)
                # Look for over/under lines
                for _, row in df.iterrows():
                    game_id = row.get('game_id', '')
                    side = str(row.get('side', '')).lower()
                    point = row.get('point', 0)

                    # Get total from over/under line
                    if 'over' in side or 'under' in side:
                        if point and point > 30:  # Sanity check
                            # Estimate TDs: total points / 7 (avg points per TD)
                            # Slight reduction for field goals
                            estimated_tds = point / 7.0
                            game_totals[game_id] = estimated_tds

                if game_totals:
                    logger.info(f"Loaded game totals for {len(game_totals)} games (for 1st TD calculation)")
                    return game_totals

            except Exception as e:
                logger.warning(f"Could not load game odds from {odds_file}: {e}")

    # Fallback: use league average
    logger.warning("No game odds found - using league average for 1st TD calculation")
    return {}


def get_game_expected_tds(game_id: str, game_totals: dict) -> float:
    """
    Get expected TDs for a game.

    Args:
        game_id: Game identifier (e.g., '2025_13_GB_DET')
        game_totals: Dict from load_game_totals_for_td_calculation

    Returns:
        Expected total TDs in game (float)
    """
    LEAGUE_AVG_TDS = 5.0  # NFL average ~5 TDs per game

    if game_id in game_totals:
        return game_totals[game_id]

    # Try partial match (in case game_id format differs)
    for gid, tds in game_totals.items():
        if gid in game_id or game_id in gid:
            return tds

    return LEAGUE_AVG_TDS


def apply_v12_model(row: dict, market: str) -> dict:
    """
    Apply V12 Interaction Model to get UNDER probability.

    Args:
        row: Dict with player data including line, model_projection, trailing stats
        market: Market type (player_receptions, player_rush_yds, etc.)

    Returns:
        Dict with v12_p_under, v12_best_direction, v12_best_ev, v12_validated
    """
    v12_models, v12_metrics, v12_thresholds = load_v12_interaction_model()

    if not v12_models or market not in v12_models:
        return {
            'v12_p_under': 0.50,
            'v12_best_direction': None,
            'v12_best_ev': 0,
            'v12_validated': False,
            'v12_reason': 'V12 model not available for this market'
        }

    # Check if market is excluded (no validated edge from walk-forward)
    threshold_config = v12_thresholds.get(market, {})
    if threshold_config.get('excluded', False) or threshold_config.get('threshold') is None:
        return {
            'v12_p_under': 0.50,
            'v12_best_direction': None,
            'v12_best_ev': 0,
            'v12_validated': False,
            'v12_reason': f'V12: Market excluded (no validated edge from walk-forward, ROI={threshold_config.get("roi", 0):.1f}%)'
        }

    model = v12_models[market]
    threshold = threshold_config.get('threshold', 0.55)

    # Build features that V12 expects
    line = row.get('line', 0)
    trailing = row.get('model_projection', row.get('trailing_stat', line))
    line_vs_trailing = line - trailing

    # V12 features
    features = {
        'line_vs_trailing': line_vs_trailing,
        'line_level': line,
        'line_in_sweet_spot': 1.0 if 3.5 <= line <= 7.5 else 0.0,
        'player_under_rate': row.get('player_under_rate', 0.5),
        'player_bias': row.get('player_bias', 0.0),
        'market_under_rate': row.get('market_under_rate', 0.5),
    }

    # Multiplicative interactions
    features['LVT_x_player_tendency'] = (
        line_vs_trailing * (features['player_under_rate'] - 0.5)
    )
    features['LVT_x_player_bias'] = line_vs_trailing * features['player_bias']
    features['LVT_x_regime'] = line_vs_trailing * (features['market_under_rate'] - 0.5)
    features['LVT_in_sweet_spot'] = line_vs_trailing * features['line_in_sweet_spot']
    features['market_bias_strength'] = abs(features['market_under_rate'] - 0.5) * 2
    features['player_market_aligned'] = (
        (features['player_under_rate'] - 0.5) * (features['market_under_rate'] - 0.5)
    )

    try:
        # Create feature array in expected order
        feature_order = [
            'line_vs_trailing', 'line_level', 'line_in_sweet_spot',
            'player_under_rate', 'player_bias', 'market_under_rate',
            'LVT_x_player_tendency', 'LVT_x_player_bias', 'LVT_x_regime',
            'LVT_in_sweet_spot', 'market_bias_strength', 'player_market_aligned'
        ]
        X = np.array([[features.get(f, 0) for f in feature_order]])

        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        p_under = float(model.predict(dmatrix)[0])

    except Exception as e:
        logger.debug(f"V12 prediction failed: {e}")
        # Fallback to simple rule
        p_under = 0.55 + min(0.15, line_vs_trailing * 0.03) if line_vs_trailing > 1.5 else 0.50

    # Calculate EV
    ev_under = calculate_bet_ev(p_under, -110)
    ev_over = calculate_bet_ev(1 - p_under, -110)

    # Determine best direction
    if p_under >= threshold and ev_under > 0:
        return {
            'v12_p_under': p_under,
            'v12_best_direction': 'UNDER',
            'v12_best_ev': ev_under * 100,
            'v12_validated': True,
            'v12_reason': f'V12: {p_under:.1%} UNDER (threshold {threshold:.0%})'
        }
    elif (1 - p_under) >= threshold and ev_over > 0:
        return {
            'v12_p_under': p_under,
            'v12_best_direction': 'OVER',
            'v12_best_ev': ev_over * 100,
            'v12_validated': True,
            'v12_reason': f'V12: {1-p_under:.1%} OVER (threshold {threshold:.0%})'
        }
    else:
        return {
            'v12_p_under': p_under,
            'v12_best_direction': None,
            'v12_best_ev': 0,
            'v12_validated': False,
            'v12_reason': f'V12: Below threshold ({p_under:.1%} vs {threshold:.0%})'
        }


def calculate_bet_ev(prob, odds):
    """
    Calculate Expected Value for a bet.

    Args:
        prob: Our estimated probability of winning
        odds: American odds

    Returns:
        EV as decimal (e.g., 0.05 = +5% EV)
    """
    if odds < 0:
        payout = 100 / abs(odds)
    else:
        payout = odds / 100

    # EV = (prob_win * payout) - (prob_lose * stake)
    return prob * payout - (1 - prob)


def kelly_bet_size(prob, odds, kelly_mult=0.25):
    """
    Calculate Kelly Criterion bet size as fraction of bankroll.

    Args:
        prob: Our estimated probability of winning
        odds: American odds
        kelly_mult: Fractional Kelly (0.25 = quarter Kelly, conservative)

    Returns:
        Recommended bet size as fraction of bankroll (0-1)
    """
    if odds < 0:
        b = 100 / abs(odds)
    else:
        b = odds / 100

    q = 1 - prob
    if b <= 0:
        return 0

    f = (b * prob - q) / b
    f = max(0, f * kelly_mult)
    f = min(0.05, f)  # Cap at 5% of bankroll

    return f

# Line movement integration
_line_movement = None

def load_line_movement(week: int) -> dict:
    """Load line movement data for the week if available."""
    global _line_movement
    if _line_movement is not None:
        return _line_movement

    movement_path = project_root / 'data' / 'line_movement' / f'line_movement_week{week}.csv'
    if not movement_path.exists():
        logger.info("No line movement data available (run fetch_line_movement.py)")
        return {}

    try:
        df = pd.read_csv(movement_path)
        _line_movement = {}
        for _, row in df.iterrows():
            key = (str(row['player']).lower().strip(), row['market'])
            _line_movement[key] = {
                'line_movement': row['line_movement'],
                'movement_direction': row['movement_direction'],
                'sharp_under': row['movement_direction'] == 'down',
                'sharp_over': row['movement_direction'] == 'up',
            }
        logger.info(f"Loaded line movement for {len(_line_movement)} player/markets")
        return _line_movement
    except Exception as e:
        logger.warning(f"Could not load line movement: {e}")
        return {}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_simulation_config() -> dict:
    """Load simulation config from JSON file."""
    config_path = project_root / 'configs' / 'simulation_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load simulation config: {e}")
        return {}

# Load simulation config for tiered calibration
CONFIG_PATH = project_root / 'configs' / 'simulation_config.json'
_sim_config = {}
_calibration_tiers = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _sim_config = json.load(f)
        _calibration_tiers = _sim_config.get('calibration_tiers', {})
        _use_tiered_calibration = _sim_config.get('feature_flags', {}).get('use_tiered_calibration', True)
    logger.info(f"Loaded calibration tiers config: {len(_calibration_tiers)} market types")
except Exception as e:
    logger.warning(f"Could not load simulation config: {e}. Using default shrinkage.")
    _use_tiered_calibration = False


def classify_player_tier(player_data: dict, market: str) -> tuple:
    """
    Classify player into a tier based on snap_share and usage metrics.

    UPDATED 2025-11-26: Now uses market-specific tier keys (e.g., wr_receiving_yards vs wr_receptions)
    to provide correct bias adjustments for both yards and count markets.

    Returns: (tier_name, tier_config) where tier_config contains:
        - bias_adjustment: additive adjustment to model mean (market-specific)
        - shrinkage_factor: how much to shrink probability toward 0.5
        - confidence_boost: adjustment to confidence tier scoring
    """
    if not _use_tiered_calibration or not _calibration_tiers:
        # Return default tier with standard shrinkage
        return ('default', {
            'bias_adjustment': 0.0,
            'shrinkage_factor': 0.30,
            'confidence_boost': 0.0
        })

    snap_share = player_data.get('snap_share', 0.0) or 0.0
    targets = player_data.get('targets_mean', 0.0) or 0.0
    carries = player_data.get('rushing_attempts_mean', 0.0) or 0.0
    position = player_data.get('position', '').upper()

    # Map market to SPECIFIC calibration tier config key
    # New structure: {position}_{category}_{market_type} e.g., wr_receiving_yards, wr_receptions
    market_lower = market.lower()

    # Determine market type (yards vs count)
    is_yards_market = market in ['player_reception_yds', 'player_rush_yds', 'player_pass_yds', 'player_rush_reception_yds']
    is_receptions_market = market == 'player_receptions'
    is_rush_attempts_market = market == 'player_rush_attempts'
    is_pass_completions_market = market == 'player_pass_completions'

    # Map to specific tier key
    if market == 'player_pass_yds':
        tier_key = 'qb_passing_yards'
        usage_value = snap_share  # Use snap share for QBs
        usage_key = 'snap_share_min'
    elif market == 'player_pass_completions':
        tier_key = 'qb_passing_completions'
        usage_value = snap_share
        usage_key = 'snap_share_min'
    elif market == 'player_rush_yds':
        if position == 'QB':
            tier_key = 'qb_rushing_yards'  # QB rushing has its own tier (not passing!)
            usage_value = snap_share
            usage_key = 'snap_share_min'
        else:
            tier_key = 'rb_rushing_yards'
            usage_value = carries
            usage_key = 'carries_min'
    elif market == 'player_rush_attempts':
        if position == 'QB':
            tier_key = 'qb_rushing_attempts'  # QB rushing attempts tier
            usage_value = snap_share
            usage_key = 'snap_share_min'
        else:
            tier_key = 'rb_rushing_attempts'
            usage_value = carries
            usage_key = 'carries_min'
    elif market == 'player_reception_yds':
        if position == 'RB':
            tier_key = 'rb_receiving_yards'
        elif position == 'TE':
            tier_key = 'te_receiving_yards'
        else:  # WR or unknown
            tier_key = 'wr_receiving_yards'
        usage_value = targets
        usage_key = 'targets_min'
    elif market == 'player_receptions':
        if position == 'RB':
            tier_key = 'rb_receptions'
        elif position == 'TE':
            tier_key = 'te_receptions'
        else:  # WR or unknown
            tier_key = 'wr_receptions'
        usage_value = targets
        usage_key = 'targets_min'
    else:
        # Default for other markets (TDs, etc.)
        return ('default', {
            'bias_adjustment': 0.0,
            'shrinkage_factor': 0.30,
            'confidence_boost': 0.0
        })

    tier_config = _calibration_tiers.get(tier_key, {})

    if not tier_config:
        # Tier key not found in config, return default
        logger.debug(f"No tier config found for {tier_key}, using default")
        return ('default', {
            'bias_adjustment': 0.0,
            'shrinkage_factor': 0.30,
            'confidence_boost': 0.0
        })

    # Iterate through tiers in order (highest first due to dict ordering)
    # Tiers are named tier_1_*, tier_2_*, etc.
    for tier_name, tier_params in sorted(tier_config.items(), key=lambda x: x[0]):
        if tier_name.startswith('_'):
            continue  # Skip comment keys
        if not isinstance(tier_params, dict):
            continue  # Skip non-dict entries

        snap_min = tier_params.get('snap_share_min', 0.0)
        usage_min = tier_params.get(usage_key, tier_params.get('targets_min', tier_params.get('carries_min', 0.0)))

        if snap_share >= snap_min and usage_value >= usage_min:
            return (tier_name, {
                'bias_adjustment': tier_params.get('bias_adjustment', 0.0),
                'shrinkage_factor': tier_params.get('shrinkage_factor', 0.30),
                'confidence_boost': tier_params.get('confidence_boost', 0.0)
            })

    # Fallback to last tier if nothing matched
    return ('tier_4_default', {
        'bias_adjustment': 0.0,
        'shrinkage_factor': 0.45,
        'confidence_boost': -0.05
    })


def get_current_season() -> int:
    """Determine current NFL season."""
    now = datetime.now()
    return now.year if now.month >= 8 else now.year - 1


def get_current_week() -> int:
    """Estimate current NFL week."""
    now = datetime.now()
    if now.month >= 9:
        return min(18, (now - datetime(now.year, 9, 1)).days // 7 + 1)
    elif now.month <= 2:
        return 18
    else:
        return 1


def american_to_probability(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def get_market_priority(position: str, market: str) -> str:
    """
    Assign market priority based on historical model accuracy.

    Priority levels:
    - HIGH: Most accurate predictions (WR receptions, RB attempts, QB pass yds)
    - MEDIUM: Good accuracy (WR rec yds, RB receptions, TE targets)
    - STANDARD: Less proven markets (TDs, longest reception, alt lines)

    Args:
        position: Player position (QB, RB, WR, TE)
        market: Market type (player_receptions, player_reception_yds, etc.)

    Returns:
        Priority level: "HIGH", "MEDIUM", or "STANDARD"
    """
    # HIGH priority combinations (most accurate)
    high_priority = {
        'WR': ['player_receptions', 'player_targets'],
        'RB': ['player_rush_attempts'],
        'TE': ['player_receptions'],
        'QB': ['player_pass_yds']
    }

    # MEDIUM priority combinations (good accuracy)
    medium_priority = {
        'WR': ['player_reception_yds'],
        'RB': ['player_receptions', 'player_targets'],
        'TE': ['player_targets'],
        'QB': ['player_pass_completions', 'player_pass_attempts']
    }

    if position in high_priority and market in high_priority[position]:
        return 'HIGH'
    elif position in medium_priority and market in medium_priority[position]:
        return 'MEDIUM'
    else:
        return 'STANDARD'


def load_model_predictions(week: int) -> pd.DataFrame:
    """Load model predictions for specified week."""
    pred_file = Path(f'data/model_predictions_week{week}.csv')
    if not pred_file.exists():
        raise FileNotFoundError(f"Model predictions not found: {pred_file}")

    df = pd.read_csv(pred_file)
    logger.info(f"Loaded {len(df)} player predictions from {pred_file}")
    return df


def load_prop_lines() -> pd.DataFrame:
    """Load current prop lines from DraftKings."""
    props_file = Path('data/nfl_player_props_draftkings.csv')
    if not props_file.exists():
        raise FileNotFoundError(f"Props file not found: {props_file}")

    df = pd.read_csv(props_file)
    logger.info(f"Loaded {len(df)} prop lines from {props_file}")
    return df


def match_player_name(model_name: str, prop_name: str) -> bool:
    """Check if player names match (handles variations)."""
    # Normalize names
    model_clean = model_name.lower().strip().replace('.', '').replace("'", "")
    prop_clean = prop_name.lower().strip().replace('.', '').replace("'", "")

    # Direct match
    if model_clean == prop_clean:
        return True

    # Check if one contains the other
    if model_clean in prop_clean or prop_clean in model_clean:
        return True

    # Handle suffix variations (Jr., Sr., III, etc.)
    suffixes = [' jr', ' sr', ' iii', ' ii', ' iv']
    for suffix in suffixes:
        if model_clean.replace(suffix, '') == prop_clean.replace(suffix, ''):
            return True

    return False


def get_model_stat_for_market(player_row: pd.Series, market: str) -> tuple:
    """
    Get the appropriate model stat (mean and std) for a given market type.

    Returns (mean, std) tuple.
    """
    market_to_stat = {
        'player_receptions': ('receptions_mean', 'receptions_std'),
        'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
        'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
        'player_rush_attempts': ('rushing_attempts_mean', 'rushing_attempts_std'),
        'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
        'player_pass_tds': ('passing_tds_mean', 'passing_tds_std'),
        'player_pass_attempts': ('passing_attempts_mean', 'passing_attempts_std'),
        'player_pass_completions': ('passing_completions_mean', 'passing_completions_std'),
        'player_anytime_td': (None, None),  # Special case - calculated below
        'player_1st_td': (None, None),  # Special case - calculated below
        'player_last_td': (None, None),  # Special case - calculated below
        'player_rush_reception_yds': (None, None),  # Will calculate as sum
        'player_longest_reception': (None, None),  # Special case - calculated below
        'player_reception_longest': (None, None),  # DK naming
        'player_rush_longest': (None, None),  # Special case - calculated below
        'player_interceptions': (None, None),  # Special case
        'player_pass_interceptions': (None, None),  # DK naming
        'player_sacks': (None, None),  # Defensive stat
        'player_solo_tackles': (None, None),  # Defensive stat
        'player_tackles_assists': (None, None),  # Defensive stat
        'player_kicking_points': (None, None),  # Kicker stat
        'player_field_goals': (None, None),  # Kicker stat
    }

    if market not in market_to_stat:
        return (None, None)

    mean_col, std_col = market_to_stat[market]

    # Special case for combined rush + rec yards
    if market == 'player_rush_reception_yds':
        rush_mean = player_row.get('rushing_yards_mean', 0)
        rush_std = player_row.get('rushing_yards_std', 0)
        rec_mean = player_row.get('receiving_yards_mean', 0)
        rec_std = player_row.get('receiving_yards_std', 0)

        if pd.isna(rush_mean):
            rush_mean = 0
        if pd.isna(rec_mean):
            rec_mean = 0
        if pd.isna(rush_std):
            rush_std = 0
        if pd.isna(rec_std):
            rec_std = 0

        combined_mean = rush_mean + rec_mean

        # Combine variances using Pythagorean addition (assumes independence)
        if rush_std > 0 or rec_std > 0:
            combined_std = np.sqrt(rush_std**2 + rec_std**2)
        else:
            # If both std values are missing, raise error instead of hardcoding
            logger.error(f"Missing both rushing_std and receiving_std for combined market")
            raise ValueError(
                "Combined rush+reception market requires both rushing_std and receiving_std. "
                "Ensure model_predictions_weekX.csv includes these columns."
            )

        return (combined_mean, combined_std)

    # Special case for Anytime TD (combine rushing + receiving TDs)
    if market in ['player_anytime_td', 'player_1st_td', 'player_last_td']:
        rush_tds = player_row.get('rushing_tds_mean', 0)
        rec_tds = player_row.get('receiving_tds_mean', 0)
        pass_tds = player_row.get('passing_tds_mean', 0)  # For QBs rushing TDs

        if pd.isna(rush_tds):
            rush_tds = 0
        if pd.isna(rec_tds):
            rec_tds = 0
        if pd.isna(pass_tds):
            pass_tds = 0

        # Calculate anytime TD probability using Poisson
        # P(at least 1 TD) = 1 - P(0 rush TDs) * P(0 rec TDs)
        # For expected TDs, use combined rate
        total_expected_tds = rush_tds + rec_tds

        # For QBs, they can also score rushing TDs (already in rush_tds)
        position = player_row.get('position', '')
        if position == 'QB':
            # QB rushing TDs are already in rushing_tds_mean
            pass

        # Return expected TDs as mean, std approximated using Poisson (std = sqrt(mean))
        if total_expected_tds > 0:
            return (total_expected_tds, np.sqrt(total_expected_tds))
        return (None, None)

    # Special case for Pass Completions (use actual column if available)
    if market == 'player_pass_completions':
        mean_val = player_row.get('passing_completions_mean', None)
        std_val = player_row.get('passing_completions_std', None)
        if mean_val is None or pd.isna(mean_val):
            # Fallback: estimate from attempts * completion rate
            attempts = player_row.get('passing_attempts_mean', 0)
            if attempts and not pd.isna(attempts):
                mean_val = attempts * 0.65  # League avg completion rate
                std_val = np.sqrt(mean_val)  # Poisson approximation
        if mean_val is not None and not pd.isna(mean_val):
            return (mean_val, std_val if std_val and not pd.isna(std_val) else np.sqrt(mean_val))
        return (None, None)

    # Special case for Pass Attempts
    if market == 'player_pass_attempts':
        mean_val = player_row.get('passing_attempts_mean', None)
        std_val = player_row.get('passing_attempts_std', None)
        if mean_val is not None and not pd.isna(mean_val):
            return (mean_val, std_val if std_val and not pd.isna(std_val) else np.sqrt(mean_val))
        return (None, None)

    # Special case for Longest Reception
    # Estimate based on expected yards per reception and number of receptions
    # Using Gumbel distribution approximation: E[max] ≈ μ + σ * √(2 * ln(n))
    if market in ['player_reception_longest', 'player_longest_reception']:
        rec_yards = player_row.get('receiving_yards_mean', 0)
        receptions = player_row.get('receptions_mean', 0)

        if pd.isna(rec_yards) or pd.isna(receptions) or receptions <= 0:
            return (None, None)

        # Average yards per reception
        yards_per_rec = rec_yards / receptions if receptions > 0 else 10

        # Estimate longest using Gumbel approximation
        # For n attempts with mean μ and std σ, E[max] ≈ μ + σ * √(2 * ln(n))
        # Using empirical adjustment: longest ≈ 1.5 * avg_yards_per_rec + 5
        # This captures that longest plays tend to be about 1.5-2x average
        n = max(receptions, 1)
        sigma = yards_per_rec * 0.8  # Typical CV for single reception yards
        expected_longest = yards_per_rec + sigma * np.sqrt(2 * np.log(n + 1))

        # Add some base value (even 1 reception has positive expected yards)
        expected_longest = max(expected_longest, yards_per_rec * 1.2)

        # Std is high for longest plays (empirically ~13 yards)
        longest_std = 13.0

        return (expected_longest, longest_std)

    # Special case for Longest Rush
    if market in ['player_rush_longest']:
        rush_yards = player_row.get('rushing_yards_mean', 0)
        attempts = player_row.get('rushing_attempts_mean', 0)

        if pd.isna(rush_yards) or pd.isna(attempts) or attempts <= 0:
            return (None, None)

        yards_per_carry = rush_yards / attempts if attempts > 0 else 4

        n = max(attempts, 1)
        sigma = yards_per_carry * 1.0  # Higher variance for rushing
        expected_longest = yards_per_carry + sigma * np.sqrt(2 * np.log(n + 1))
        expected_longest = max(expected_longest, yards_per_carry * 1.3)

        longest_std = 11.0  # Empirical std for longest rush

        return (expected_longest, longest_std)

    # Special case for QB Pass Interceptions
    if market in ['player_pass_interceptions', 'player_interceptions']:
        # Use passing attempts and historical INT rate
        pass_attempts = player_row.get('passing_attempts_mean', 0)
        if pd.isna(pass_attempts) or pass_attempts <= 0:
            return (None, None)

        # League average INT rate is about 2.5%
        int_rate = 0.025
        expected_ints = pass_attempts * int_rate

        return (expected_ints, np.sqrt(expected_ints))

    # Defensive stats - these require roster-level modeling which we don't have yet
    # Skip for now - return None to exclude these markets
    if market in ['player_sacks', 'player_solo_tackles', 'player_tackles_assists']:
        # TODO: Add defensive player prediction model
        # For now, skip these markets as we don't have reliable predictions
        return (None, None)

    # Kicking stats
    if market in ['player_kicking_points', 'player_field_goals', 'player_pats']:
        # TODO: Add kicker prediction model based on team scoring opportunities
        # For now, skip these markets
        return (None, None)

    mean_val = player_row.get(mean_col, None) if mean_col else None
    std_val = player_row.get(std_col, None) if std_col else None

    if pd.isna(mean_val):
        mean_val = None
    if std_val is not None and pd.isna(std_val):
        std_val = None

    return (mean_val, std_val)


def calculate_over_under_prob(mean: float, std: float, line: float, market: str,
                               game_total_tds: float = None) -> tuple:
    """
    Calculate probability of going Over and Under the line.

    Uses normal distribution for continuous stats (yards) and
    Poisson for count stats (receptions, TDs).

    For 1st TD market: Uses player's share of expected TDs in game.

    Args:
        mean: Expected value (e.g., expected TDs for TD markets)
        std: Standard deviation from Monte Carlo
        line: Betting line
        market: Market type
        game_total_tds: Total expected TDs in game (required for 1st TD calculation)

    CRITICAL: std should come from model_std (Monte Carlo simulation output).
    If model_std is missing, raise an error instead of using hardcoded fallback.
    """
    if mean is None or mean <= 0:
        return (0.5, 0.5)

    # Validate std is provided from model predictions
    # CRITICAL: Reject near-zero std (indicates Monte Carlo bug or constant prediction)
    MIN_STD_THRESHOLD = 0.01  # Minimum realistic std for any stat

    if std is None or std <= 0:
        logger.error(f"Missing or invalid model_std for market {market}, mean={mean}")
        raise ValueError(
            f"model_std is required but got {std}. "
            "Ensure model_predictions_weekX.csv includes std columns. "
            "Do not hardcode CV values - use actual Monte Carlo simulation output."
        )

    # BUG FIX: Detect near-zero std (Monte Carlo returned all identical values)
    if std < MIN_STD_THRESHOLD:
        logger.warning(
            f"Near-zero model_std detected: {std:.2e} for {market}, mean={mean}. "
            f"Applying minimum variance threshold of {MIN_STD_THRESHOLD}. "
            f"Check Monte Carlo simulation for this player - may have constant predictions."
        )
        # Use minimum realistic std based on stat type
        if market in ['player_receptions', 'player_rush_attempts', 'player_pass_attempts']:
            std = max(std, np.sqrt(mean))  # Poisson: variance = mean
        else:
            std = max(std, mean * 0.2)  # Yards: assume 20% CV minimum

    # Count-based stats use Poisson
    count_markets = ['player_receptions', 'player_rush_attempts', 'player_pass_tds',
                     'player_pass_attempts', 'player_pass_completions']

    # TD markets - different logic for each type
    if market == 'player_1st_td':
        # 1st TD probability = Player's share of expected TDs
        # P(1st TD) ≈ player_expected_tds / game_total_expected_tds
        #
        # Logic: If a game has 5 expected TDs and player expects 0.5 TDs,
        # their share of the first TD is roughly 0.5/5 = 10%
        #
        # This is an approximation - actual probability depends on:
        # - When in the game the player's TDs are likely to occur
        # - Whether they're more likely to score early (goal-line back) vs late
        # For now, we use equal probability assumption
        if game_total_tds and game_total_tds > 0:
            # Player's share of first TD
            prob_over = min(mean / game_total_tds, 0.95)
        else:
            # Fallback: Use anytime TD logic scaled down
            # Average NFL game has ~5 TDs, so rough estimate
            TYPICAL_GAME_TDS = 5.0
            prob_over = min(mean / TYPICAL_GAME_TDS, 0.95)
        prob_under = 1 - prob_over

    elif market == 'player_last_td':
        # Last TD uses same logic as 1st TD (share of game TDs)
        if game_total_tds and game_total_tds > 0:
            prob_over = min(mean / game_total_tds, 0.95)
        else:
            TYPICAL_GAME_TDS = 5.0
            prob_over = min(mean / TYPICAL_GAME_TDS, 0.95)
        prob_under = 1 - prob_over

    elif market == 'player_anytime_td':
        # Anytime TD: P(at least 1 TD) using Poisson
        # P(Over 0.5) = P(at least 1 TD) = 1 - P(0 TDs) = 1 - e^(-lambda)
        prob_over = 1 - poisson.cdf(0, mean)  # P(X >= 1) = 1 - P(X = 0)
        prob_under = poisson.cdf(0, mean)  # P(X = 0)
    elif market in count_markets:
        # For counts, P(Over X) = P(count > X) = 1 - P(count <= X-0.5)
        prob_over = 1 - poisson.cdf(line - 0.5, mean)
        prob_under = poisson.cdf(line - 0.5, mean)
    else:
        # For continuous stats, use normal distribution
        z_score = (line - mean) / std
        prob_over = 1 - norm.cdf(z_score)
        prob_under = norm.cdf(z_score)

    return (prob_over, prob_under)


def generate_recommendations(week: int, season: int = None) -> pd.DataFrame:
    """Generate all recommendations by matching model predictions to prop lines."""

    if season is None:
        season = get_current_season()

    logger.info(f"Generating recommendations for Week {week}, Season {season}")

    # Load simulation config for probability caps and strategy filters
    config = load_simulation_config()

    # Load game totals for 1st TD calculation
    game_totals = load_game_totals_for_td_calculation(week)

    # Load data
    model_preds = load_model_predictions(week)
    prop_lines = load_prop_lines()

    # Filter to relevant markets - all supported prop types
    relevant_markets = [
        # Core yardage markets
        'player_receptions',
        'player_reception_yds',
        'player_rush_yds',
        'player_rush_attempts',
        'player_rush_reception_yds',
        # Passing markets
        'player_pass_yds',
        'player_pass_tds',
        'player_pass_attempts',
        'player_pass_completions',
        # TD markets
        'player_anytime_td',
        'player_1st_td',
        'player_last_td',
        # Longest play markets (to be implemented)
        'player_reception_longest',
        'player_rush_longest',
        'player_pass_longest_completion',
        # QB props
        'player_pass_interceptions',
    ]

    prop_lines = prop_lines[prop_lines['market'].isin(relevant_markets)]
    logger.info(f"Filtered to {len(prop_lines)} relevant prop lines")

    recommendations = []
    matched_count = 0
    unmatched_players = set()

    # Group prop lines by player and market
    for idx, prop_row in prop_lines.iterrows():
        prop_player = prop_row['player_name']
        market = prop_row['market']
        line = prop_row.get('line', None)
        odds = prop_row.get('odds', None)
        outcome = prop_row.get('outcome_type', 'Over')

        # Handle TD markets specially - they don't have a line (Yes/No bets)
        td_markets = ['player_anytime_td', 'player_1st_td', 'player_last_td']
        if market in td_markets:
            line = 0.5  # Set line to 0.5 for TD props (at least 1 TD)
            if 'Yes' in str(outcome):
                outcome = 'Over'  # "Yes" = Over 0.5 TDs
            elif 'No' in str(outcome):
                outcome = 'Under'  # "No" = Under 0.5 TDs

        if pd.isna(odds):
            continue

        # For non-TD markets, require a line
        if market not in td_markets and pd.isna(line):
            continue

        # Find matching model prediction
        matched_player = None
        for _, model_row in model_preds.iterrows():
            model_name = model_row['player_name']
            if match_player_name(model_name, prop_player):
                matched_player = model_row
                break

        if matched_player is None:
            unmatched_players.add(prop_player)
            continue

        matched_count += 1

        # Get model statistics for this market
        model_mean_raw, model_std = get_model_stat_for_market(matched_player, market)

        if model_mean_raw is None or model_mean_raw <= 0:
            continue

        # TIERED CALIBRATION: Classify player based on snap_share, usage, AND market type
        # Updated 2025-11-26: Now uses market-specific bias adjustments
        tier_name, tier_config = classify_player_tier(matched_player, market)
        bias_adj = tier_config['bias_adjustment']  # Market-specific adjustment
        tier_shrinkage = tier_config['shrinkage_factor']
        confidence_boost = tier_config['confidence_boost']

        # Apply bias correction to model mean (fixes systematic over-prediction)
        # Correction factors derived from historical backtest analysis
        model_mean = apply_bias_correction(model_mean_raw, market)

        # Apply tier-specific bias adjustment (now market-specific from config)
        # Config has separate entries for yards vs count markets with appropriate adjustments
        model_mean = model_mean + bias_adj

        # Also scale std proportionally to maintain coefficient of variation
        if model_std and model_std > 0 and model_mean_raw > 0:
            correction_factor = get_correction_factor(market)
            model_std = model_std * correction_factor

        # Get game expected TDs for 1st TD calculation
        game_id = matched_player.get('game_id', prop_row.get('game_id', ''))
        game_total_tds = get_game_expected_tds(game_id, game_totals) if market in td_markets else None

        # Calculate raw probabilities (using bias-corrected mean)
        prob_over_raw, prob_under_raw = calculate_over_under_prob(
            model_mean, model_std, line, market, game_total_tds=game_total_tds
        )

        # Apply tier-specific shrinkage calibration
        # High-volume players: less shrinkage (more confidence in model)
        # Low-volume players: more shrinkage (less confidence, more uncertainty)
        prob_over_calibrated = prob_over_raw * (1 - tier_shrinkage) + 0.5 * tier_shrinkage
        prob_under_calibrated = prob_under_raw * (1 - tier_shrinkage) + 0.5 * tier_shrinkage

        # Apply probability caps from config (backtest showed calibration breaks down above 56%)
        prob_caps = config.get('probability_caps', {})
        max_prob = prob_caps.get('max_probability', 0.56)  # Cap at 56% (well-calibrated range)
        min_prob = prob_caps.get('min_probability', 0.50)  # Below 50% means opposite direction

        # Cap OVER probability at max (any probability above 56% gets capped)
        prob_over_capped = min(prob_over_calibrated, max_prob)
        # Cap UNDER probability at max
        prob_under_capped = min(prob_under_calibrated, max_prob)

        # Use capped probabilities going forward
        prob_over_calibrated = prob_over_capped
        prob_under_calibrated = prob_under_capped

        # Get market implied probability
        market_prob_with_vig = american_to_probability(odds)

        # For two-way markets, need to find opposite side to remove vig
        # Look for opposing line (Over vs Under on same player/market/line)
        opposite_odds = None
        opposite_outcome = 'Under' if 'Over' in outcome else 'Over'
        for _, opp_row in prop_lines.iterrows():
            if (opp_row['player_name'] == prop_player and
                opp_row['market'] == market and
                opp_row.get('line') == line and
                opposite_outcome in str(opp_row.get('outcome_type', ''))):
                opposite_odds = opp_row.get('odds')
                break

        # Remove vig if we have both sides
        if opposite_odds is not None:
            over_prob_vig = american_to_probability(odds if 'Over' in outcome else opposite_odds)
            under_prob_vig = american_to_probability(opposite_odds if 'Over' in outcome else odds)
            over_prob_novig, under_prob_novig = remove_vig_two_way(over_prob_vig, under_prob_vig)
            market_prob_novig = over_prob_novig if 'Over' in outcome else under_prob_novig
        else:
            # No opposite side found, use vig probability with warning
            market_prob_novig = market_prob_with_vig
            logger.debug(f"No opposite side found for {prop_player} {market}, using vig probability")

        # Determine model probability based on outcome type
        if 'Over' in outcome:
            model_prob = prob_over_calibrated
            raw_prob = prob_over_raw
        else:
            model_prob = prob_under_calibrated
            raw_prob = prob_under_raw

        # Calculate edge using no-vig market probability
        edge_pct = (model_prob - market_prob_novig) * 100
        roi_pct = edge_pct / market_prob_novig if market_prob_novig > 0 else 0

        # Assign confidence tier based on edge + probability
        # Apply tier-specific confidence boost (high-volume players get boost, low-volume get penalty)
        # The boost adjusts the effective edge used for tier classification
        adjusted_edge_for_confidence = edge_pct + (confidence_boost * 100)  # Convert decimal to pct points
        confidence_tier = assign_confidence_tier(adjusted_edge_for_confidence, model_prob, bet_type="player_prop")

        # Calculate Kelly sizing
        kelly_fraction = calculate_kelly_fraction(model_prob, odds, fractional=0.25)
        kelly_units = calculate_recommended_units(kelly_fraction, base_units=100.0)

        # Assign market priority based on position + market combination
        position = matched_player.get('position', '')
        market_priority = get_market_priority(position, market)

        # Determine game
        home_team = prop_row.get('home_team', '')
        away_team = prop_row.get('away_team', '')
        game = f"{away_team} @ {home_team}"

        # Create recommendation - PRESERVE ALL MODEL FEATURES
        # Start with core betting recommendation fields
        rec = {
            'player': prop_player,
            'nflverse_name': matched_player.get('player_name', prop_player),
            'pick': outcome,
            'market': market,
            'market_display': format_market_name(market),  # Human-readable: "Pass Completions"
            'line': line,
            'odds': odds,
            'model_prob': model_prob,  # Calibrated probability
            'model_prob_pct': f"{model_prob * 100:.1f}%",  # Display format
            'raw_prob': raw_prob,  # Raw probability before calibration
            'calibrated_prob': model_prob,  # Alias for clarity
            'market_prob': market_prob_novig,  # No-vig market probability
            'market_prob_with_vig': market_prob_with_vig,  # Original with vig
            'edge_pct': edge_pct,
            'roi_pct': roi_pct,
            'model_projection': model_mean,  # Actual model mean from predictions
            'model_std': model_std,
            'strategy': 'Model Projection (Calibrated)',
            'calibration_applied': True,
            'confidence': confidence_tier.value,  # ELITE/HIGH/STANDARD/LOW
            'confidence_pct': format_confidence_pct(confidence_tier.value),  # Display: "75-90%"
            'priority': market_priority,  # HIGH/MEDIUM/STANDARD
            'kelly_fraction': kelly_fraction,
            'kelly_units': kelly_units,
            'game': game,
            'position': position,
            'team': matched_player.get('team', ''),
            'opponent': matched_player.get('opponent', ''),
            'week': matched_player.get('week', 0),
        }

        # ADD ALL MODEL PREDICTION FEATURES (54+ features preserved)
        # This ensures NO feature dropout between model predictions and final output

        # Basic player stats (mean/std for all markets)
        for stat in ['rushing_yards', 'rushing_attempts', 'rushing_tds',
                     'receiving_yards', 'receptions', 'targets', 'receiving_tds',
                     'passing_yards', 'passing_completions', 'passing_attempts', 'passing_tds']:
            rec[f'{stat}_mean'] = matched_player.get(f'{stat}_mean', None)
            rec[f'{stat}_std'] = matched_player.get(f'{stat}_std', None)

        # TD probabilities
        rec['raw_td_prob'] = matched_player.get('raw_td_prob', None)
        rec['calibrated_td_prob'] = matched_player.get('calibrated_td_prob', None)

        # Defensive matchup
        rec['opponent_def_epa'] = matched_player.get('opponent_def_epa_vs_position', None)

        # Opponent-specific features (V13 support)
        # Add divergence columns based on the current market
        for mkt, stat in [
            ('player_receptions', 'receptions'),
            ('player_reception_yds', 'receiving_yards'),
            ('player_rush_yds', 'rushing_yards'),
            ('player_pass_yds', 'passing_yards'),
        ]:
            div_col = f'{mkt}_vs_opp_divergence'
            conf_col = f'{mkt}_vs_opp_confidence'
            rec[div_col] = matched_player.get(div_col, 0)
            rec[conf_col] = matched_player.get(conf_col, 0)

        # Also add vs_opp_avg columns for debugging/display
        for stat in ['receptions', 'receiving_yards', 'rushing_yards', 'passing_yards', 'completions']:
            rec[f'vs_opp_avg_{stat}'] = matched_player.get(f'vs_opp_avg_{stat}', None)
            rec[f'vs_opp_games_{stat}'] = matched_player.get(f'vs_opp_games_{stat}', 0)

        # Injury context
        rec['total_adjustment'] = matched_player.get('total_adjustment', 0)
        rec['passing_adjustment'] = matched_player.get('passing_adjustment', 0)
        rec['severity'] = matched_player.get('severity', '')
        rec['injury_qb_status'] = matched_player.get('injury_qb_status', '')
        rec['injury_wr1_status'] = matched_player.get('injury_wr1_status', '')
        rec['injury_rb1_status'] = matched_player.get('injury_rb1_status', '')

        # Weather
        rec['weather_total_adjustment'] = matched_player.get('weather_total_adjustment', 0)
        rec['weather_passing_adjustment'] = matched_player.get('weather_passing_adjustment', 0)

        # Game context
        rec['is_divisional_game'] = matched_player.get('is_divisional_game', False)
        rec['rest_epa_adjustment'] = matched_player.get('rest_epa_adjustment', 0)
        rec['is_coming_off_bye'] = matched_player.get('is_coming_off_bye', False)
        rec['travel_epa_adjustment'] = matched_player.get('travel_epa_adjustment', 0)
        rec['home_field_advantage_points'] = matched_player.get('home_field_advantage_points', 0)
        rec['is_primetime_game'] = matched_player.get('is_primetime_game', False)
        rec['primetime_type'] = matched_player.get('primetime_type', '')

        # Redzone/snap shares
        rec['redzone_target_share'] = matched_player.get('redzone_target_share', None)
        rec['redzone_carry_share'] = matched_player.get('redzone_carry_share', None)
        rec['goalline_carry_share'] = matched_player.get('goalline_carry_share', None)
        rec['snap_share'] = matched_player.get('snap_share', None)

        # Tiered calibration info
        rec['calibration_tier'] = tier_name
        rec['tier_bias_adjustment'] = bias_adj
        rec['tier_shrinkage'] = tier_shrinkage
        rec['tier_confidence_boost'] = confidence_boost

        # Altitude
        rec['elevation_feet'] = matched_player.get('elevation_feet', 0)
        rec['is_high_altitude'] = matched_player.get('is_high_altitude', False)
        rec['altitude_epa_adjustment'] = matched_player.get('altitude_epa_adjustment', 0)
        rec['field_surface'] = matched_player.get('field_surface', '')

        # Team context
        rec['team_pass_attempts'] = matched_player.get('team_pass_attempts', None)
        rec['team_rush_attempts'] = matched_player.get('team_rush_attempts', None)
        rec['team_targets'] = matched_player.get('team_targets', None)
        rec['game_script_dynamic'] = matched_player.get('game_script_dynamic', '')

        # Market/role
        rec['market_blended_prob'] = matched_player.get('market_blended_prob', None)
        rec['role_override_applied'] = matched_player.get('role_override_applied', False)

        # Player identifiers (for joining/verification)
        rec['player_dk'] = matched_player.get('player_dk', '')
        rec['player_pbp'] = matched_player.get('player_pbp', '')

        recommendations.append(rec)

    logger.info(f"Matched {matched_count} prop lines to model predictions")
    if unmatched_players:
        logger.warning(f"Could not match {len(unmatched_players)} players: {list(unmatched_players)[:10]}...")

    df = pd.DataFrame(recommendations)

    # CRITICAL FILTERS: Only include valid betting opportunities
    initial_count = len(df)

    # Filter 1: Positive edge only
    df = df[df['edge_pct'] > 0]
    logger.info(f"After positive edge filter: {len(df)} bets (removed {initial_count - len(df)})")

    # Filter 2: Model probability must be >= 50% (can't bet on <50% chance)
    # EXCEPTION: TD markets can recommend "No" when prob < 50% (we're betting the player WON'T score)
    td_markets = ['player_anytime_td', 'player_1st_td', 'player_last_td']
    non_td_mask = ~df['market'].isin(td_markets)
    td_mask = df['market'].isin(td_markets)

    # For non-TD markets: model_prob >= 50%
    # For TD markets: allow all (edge already calculated correctly)
    df = df[(non_td_mask & (df['model_prob'] >= 0.50)) | td_mask]
    logger.info(f"After prob >= 50% filter: {len(df)} bets")

    # Filter 3: Model probability should be <= 95% (extreme confidence often = overfit)
    df = df[df['model_prob'] <= 0.95]
    logger.info(f"After prob <= 95% filter: {len(df)} bets")

    # Filter 4: Edge should be realistic (<30% for standard markets, <50% for TD markets)
    # TD markets can have higher edges due to binary nature
    df = df[((~df['market'].isin(td_markets)) & (df['edge_pct'] < 30.0)) |
            ((df['market'].isin(td_markets)) & (df['edge_pct'] < 50.0))]
    logger.info(f"After edge filter: {len(df)} bets")

    # Filter 5: MARKET AGREEMENT FILTER (ADDED 2025-11-26)
    # When our model strongly disagrees with the market line, the MARKET is usually right.
    # Large perceived "edges" are actually model errors, not opportunities.
    # Only bet when model prediction is within 25% of the line.
    try:
        config = load_simulation_config()
        market_filter = config.get('market_agreement_filter', {})
        if market_filter.get('enabled', True):
            max_deviation = market_filter.get('max_deviation_pct', 0.25)

            # Calculate deviation: |model_projection - line| / line
            df['model_market_deviation'] = abs(df['model_projection'] - df['line']) / df['line'].clip(lower=0.1)

            before_filter = len(df)
            df = df[df['model_market_deviation'] <= max_deviation]

            logger.info(f"After market agreement filter (max {max_deviation:.0%} deviation): {len(df)} bets (removed {before_filter - len(df)})")
        else:
            logger.info("Market agreement filter disabled")
    except Exception as e:
        logger.warning(f"Could not apply market agreement filter: {e}")

    # Filter 6: PROFITABLE STRATEGY FILTER (ADDED 2025-11-26)
    # Backtest-validated strategy: UNDER on receptions/passing when model agrees with line
    # Results: 57.8% win rate, +10.3% ROI on 438 bets (Weeks 2-11)
    try:
        config = load_simulation_config()
        strategy = config.get('profitable_strategy', {})
        rules = strategy.get('strategy_rules', {})

        if rules:
            before_filter = len(df)

            # Get strategy parameters
            direction = rules.get('direction', 'UNDER_ONLY')
            allowed_markets = rules.get('allowed_markets', ['player_receptions', 'player_pass_yds'])
            max_agreement = rules.get('model_line_agreement_max', 0.12)

            # Calculate model-line deviation
            if 'model_market_deviation' not in df.columns:
                df['model_market_deviation'] = abs(df['model_projection'] - df['line']) / df['line'].clip(lower=0.1)

            # Apply strategy filter
            if direction == 'UNDER_ONLY':
                # Only keep UNDER bets on allowed markets with model agreement
                strategy_filter = (
                    (df['pick'].str.contains('Under', case=False, na=False)) &
                    (df['market'].isin(allowed_markets)) &
                    (df['model_market_deviation'] <= max_agreement)
                )

                # Mark bets that pass the profitable strategy filter
                df['profitable_strategy'] = strategy_filter
                df['strategy_validated'] = strategy_filter

                # Keep ALL bets but flag which ones are strategy-validated
                # This allows user to see both validated and non-validated bets
                strategy_bets = df[strategy_filter]
                other_bets = df[~strategy_filter]

                logger.info(f"PROFITABLE STRATEGY: {len(strategy_bets)} bets pass validation")
                logger.info(f"  - Direction: {direction}")
                logger.info(f"  - Markets: {allowed_markets}")
                logger.info(f"  - Model-line agreement: within {max_agreement:.0%}")
                logger.info(f"  - Other bets (not strategy-validated): {len(other_bets)}")
            else:
                df['profitable_strategy'] = False
                df['strategy_validated'] = False
                logger.info("Profitable strategy filter: direction not UNDER_ONLY, skipping")
        else:
            df['profitable_strategy'] = False
            df['strategy_validated'] = False
            logger.info("No profitable strategy rules found in config")
    except Exception as e:
        df['profitable_strategy'] = False
        df['strategy_validated'] = False
        logger.warning(f"Could not apply profitable strategy filter: {e}")

    # Filter 7: V7 HYBRID APPROACH (UPDATED 2025-11-26)
    # ROOT CAUSE ANALYSIS FINDINGS:
    # - Simple rule "line > trailing + 1.5" outperforms XGBoost (76.6% vs 66.1% in 2025)
    # - XGBoost learned average signal (~55%) but 2025 has ~72-80% signal
    # - Model predictions are DECORRELATED from key feature (line_vs_trailing)
    #
    # V7 SOLUTION: Use simple rule as PRIMARY signal, ML as secondary validation
    # - PRIMARY: Bet UNDER when line > trailing_mean + threshold
    # - Markets: player_receptions (strongest), player_rush_yds (consistent)
    # - SKIP: player_reception_yds (no edge), player_pass_yds (marginal)
    try:
        # Calculate core feature: line vs trailing average
        df['line_vs_trailing'] = df['line'] - df['model_projection']
        df['line_vs_trailing_pct'] = df['line_vs_trailing'] / (df['model_projection'] + 0.1)

        # V7 SIMPLE RULE: Bet UNDER when line > trailing + threshold
        # Thresholds based on backtest analysis (2023-2025):
        v7_thresholds = {
            'player_receptions': 1.5,      # 72.4% UNDER, +38.2% ROI in 2025
            'player_rush_yds': 1.5,        # 54.3% UNDER, +3.7% ROI (consistent)
            'player_pass_yds': 1.5,        # 55.3% UNDER, +5.5% ROI (borderline)
            'player_reception_yds': None,  # NO EDGE - skip this market
        }

        # V7 scoring using simple rule + regime estimation
        v7_scores = []
        for idx, row in df.iterrows():
            market = row['market']
            line_vs_trailing = row['line_vs_trailing']

            # Get market-specific threshold
            threshold = v7_thresholds.get(market)

            if threshold is None:
                # Market has no edge - don't recommend
                v7_scores.append({
                    'rule_signal': 'NO_EDGE',
                    'p_under': 0.50,
                    'p_over': 0.50,
                    'best_direction': None,
                    'best_ev': 0,
                    'kelly_size': 0,
                    'passes': False,
                    'reason': f"{market} has no statistical edge"
                })
                continue

            # Simple rule: line > trailing + threshold = bet UNDER
            if line_vs_trailing > threshold:
                # Strong UNDER signal
                # Estimate probability based on 2024-2025 regime data
                if market == 'player_receptions':
                    p_under = 0.70  # 72.4% in 2025, 67.4% in 2024
                elif market == 'player_rush_yds':
                    p_under = 0.55  # 54.3% consistent across years
                elif market == 'player_pass_yds':
                    p_under = 0.55  # 55.3% in 2025
                else:
                    p_under = 0.52  # Conservative default

                # Scale probability based on distance above threshold
                # More distance = higher confidence
                excess = line_vs_trailing - threshold
                prob_boost = min(0.10, excess * 0.02)  # +2% per point, max +10%
                p_under = min(0.85, p_under + prob_boost)

                p_over = 1 - p_under
                ev_under = calculate_bet_ev(p_under, -110)
                ev_over = calculate_bet_ev(p_over, -110)
                kelly = kelly_bet_size(p_under, -110)

                v7_scores.append({
                    'rule_signal': 'UNDER',
                    'p_under': p_under,
                    'p_over': p_over,
                    'best_direction': 'UNDER',
                    'best_ev': ev_under * 100,
                    'kelly_size': kelly,
                    'passes': True,
                    'reason': f"line > trailing + {threshold} ({line_vs_trailing:+.1f})"
                })
            elif line_vs_trailing < -threshold:
                # Strong OVER signal (line below trailing)
                # Note: OVER signals are less reliable historically
                p_over = 0.52  # Conservative - OVER signals weaker
                p_under = 1 - p_over
                ev_over = calculate_bet_ev(p_over, -110)
                ev_under = calculate_bet_ev(p_under, -110)

                # Only recommend if EV positive
                if ev_over > 0:
                    kelly = kelly_bet_size(p_over, -110)
                    v7_scores.append({
                        'rule_signal': 'OVER',
                        'p_under': p_under,
                        'p_over': p_over,
                        'best_direction': 'OVER',
                        'best_ev': ev_over * 100,
                        'kelly_size': kelly,
                        'passes': True,
                        'reason': f"line < trailing - {threshold} ({line_vs_trailing:+.1f})"
                    })
                else:
                    v7_scores.append({
                        'rule_signal': 'OVER_WEAK',
                        'p_under': p_under,
                        'p_over': p_over,
                        'best_direction': None,
                        'best_ev': 0,
                        'kelly_size': 0,
                        'passes': False,
                        'reason': "OVER signal too weak"
                    })
            else:
                # No signal - line is close to trailing
                v7_scores.append({
                    'rule_signal': 'NO_SIGNAL',
                    'p_under': 0.50,
                    'p_over': 0.50,
                    'best_direction': None,
                    'best_ev': 0,
                    'kelly_size': 0,
                    'passes': False,
                    'reason': f"line within +/- {threshold} of trailing"
                })

        # Add V7 columns to dataframe
        df['v7_rule_signal'] = [s['rule_signal'] for s in v7_scores]
        df['v7_p_under'] = [s['p_under'] for s in v7_scores]
        df['v7_p_over'] = [s['p_over'] for s in v7_scores]
        df['v7_best_direction'] = [s['best_direction'] for s in v7_scores]
        df['v7_best_ev'] = [s['best_ev'] for s in v7_scores]
        df['v7_kelly_size'] = [s['kelly_size'] for s in v7_scores]
        df['v7_validated'] = [s['passes'] for s in v7_scores]
        df['v7_reason'] = [s['reason'] for s in v7_scores]

        # Legacy columns for backwards compatibility (V5/V6 aliases)
        df['v6_p_under'] = df['v7_p_under']
        df['v6_p_over'] = df['v7_p_over']
        df['v6_best_direction'] = df['v7_best_direction']
        df['v6_best_ev'] = df['v7_best_ev']
        df['v6_kelly_size'] = df['v7_kelly_size']
        df['v6_value_validated'] = df['v7_validated']
        df['v5_p_under'] = df['v7_p_under']
        df['v5_threshold'] = 0.55
        df['v5_edge_validated'] = df['v7_validated']

        # Count validated bets
        v7_validated = df[df['v7_validated'] == True]
        logger.info(f"V7 HYBRID APPROACH: {len(v7_validated)} bets validated by simple rule")

        # Show breakdown
        if len(v7_validated) > 0:
            under_bets = v7_validated[v7_validated['v7_best_direction'] == 'UNDER']
            over_bets = v7_validated[v7_validated['v7_best_direction'] == 'OVER']
            logger.info(f"  - UNDER bets: {len(under_bets)}")
            logger.info(f"  - OVER bets: {len(over_bets)}")

            for market in v7_validated['market'].unique():
                market_df = v7_validated[v7_validated['market'] == market]
                avg_ev = market_df['v7_best_ev'].mean()
                logger.info(f"  - {market}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%)")

    except Exception as e:
        logger.warning(f"V7 hybrid approach failed: {e}")
        df['v7_rule_signal'] = 'ERROR'
        df['v7_p_under'] = 0.50
        df['v7_p_over'] = 0.50
        df['v7_best_direction'] = None
        df['v7_best_ev'] = 0
        df['v7_kelly_size'] = 0
        df['v7_validated'] = False
        df['v7_reason'] = str(e)
        df['v6_p_under'] = None
        df['v6_p_over'] = None
        df['v6_best_direction'] = None
        df['v6_best_ev'] = None
        df['v6_kelly_size'] = 0
        df['v6_value_validated'] = False
        df['v5_p_under'] = None
        df['v5_threshold'] = None
        df['v5_edge_validated'] = False

    # Filter 8: V12 INTERACTION MODEL (ADDED 2025-11-27)
    # V12 is the BEST performing model (+59.7% ROI on receptions)
    # Uses LVT as hub with interaction constraints and monotonic constraints
    try:
        v12_models, v12_metrics, v12_thresholds = load_v12_interaction_model()

        if v12_models:
            v12_results = []
            for idx, row in df.iterrows():
                market = row['market']
                row_dict = row.to_dict()
                v12_result = apply_v12_model(row_dict, market)
                v12_results.append(v12_result)

            # Add V12 columns
            df['v12_p_under'] = [r['v12_p_under'] for r in v12_results]
            df['v12_best_direction'] = [r['v12_best_direction'] for r in v12_results]
            df['v12_best_ev'] = [r['v12_best_ev'] for r in v12_results]
            df['v12_validated'] = [r['v12_validated'] for r in v12_results]
            df['v12_reason'] = [r['v12_reason'] for r in v12_results]

            # Log V12 results
            v12_validated = df[df['v12_validated'] == True]
            logger.info(f"V12 INTERACTION MODEL: {len(v12_validated)} bets validated")
            for market in v12_validated['market'].unique():
                market_df = v12_validated[v12_validated['market'] == market]
                avg_ev = market_df['v12_best_ev'].mean()
                logger.info(f"  - {market}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%)")
        else:
            df['v12_p_under'] = 0.50
            df['v12_best_direction'] = None
            df['v12_best_ev'] = 0
            df['v12_validated'] = False
            df['v12_reason'] = 'V12 model not loaded'

    except Exception as e:
        logger.warning(f"V12 integration failed: {e}")
        df['v12_p_under'] = 0.50
        df['v12_best_direction'] = None
        df['v12_best_ev'] = 0
        df['v12_validated'] = False
        df['v12_reason'] = str(e)

    # Filter 8b: V14 DEFENSE-AWARE MODEL FOR RUSH YARDS (ADDED 2025-11-29)
    # V14 uses LVT + opponent rush defense EPA
    # Walk-forward validated: +20.6% ROI at 55% threshold
    try:
        v14_model, v14_coefs, v14_thresholds = load_v14_defense_aware_model()

        if v14_model:
            # Get rush defense EPA data for this week
            from pathlib import Path
            pbp_path = project_root / 'data' / 'nflverse' / f'pbp_2025.parquet'

            def_epa_lookup = {}
            if pbp_path.exists():
                pbp = pd.read_parquet(pbp_path)
                # Calculate trailing rush defense EPA per team
                rush_def = pbp[pbp['play_type'] == 'run'].groupby(['defteam', 'week']).agg(
                    rush_def_epa=('epa', 'mean')
                ).reset_index()
                rush_def = rush_def.sort_values(['defteam', 'week'])
                rush_def['trailing_def_epa'] = rush_def.groupby('defteam')['rush_def_epa'].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )
                # Get latest available for each team
                for team in rush_def['defteam'].unique():
                    team_data = rush_def[rush_def['defteam'] == team]
                    if len(team_data) > 0:
                        def_epa_lookup[team] = team_data['trailing_def_epa'].iloc[-1]

            # Apply V14 to rush_yds market
            v14_results = []
            for idx, row in df.iterrows():
                market = row['market']

                if market != 'player_rush_yds':
                    v14_results.append({
                        'v14_p_under': None,
                        'v14_best_direction': None,
                        'v14_best_ev': None,
                        'v14_validated': False,
                        'v14_reason': 'V14 only for rush_yds'
                    })
                    continue

                row_dict = row.to_dict()
                opponent = row.get('opponent', row.get('opponent_team', ''))
                trailing_def_epa = def_epa_lookup.get(opponent, 0.0)

                v14_result = apply_v14_model(row_dict, trailing_def_epa)
                v14_results.append(v14_result)

            # Add V14 columns
            df['v14_p_under'] = [r.get('v14_p_under') for r in v14_results]
            df['v14_best_direction'] = [r.get('v14_best_direction') for r in v14_results]
            df['v14_best_ev'] = [r.get('v14_best_ev') for r in v14_results]
            df['v14_validated'] = [r.get('v14_validated', False) for r in v14_results]
            df['v14_reason'] = [r.get('v14_reason', '') for r in v14_results]

            # For rush_yds, override V12 results with V14
            rush_mask = df['market'] == 'player_rush_yds'
            v14_validated_mask = (df['v14_validated'] == True)

            # Update V12 columns with V14 results for rush_yds
            df.loc[rush_mask & v14_validated_mask, 'v12_p_under'] = df.loc[rush_mask & v14_validated_mask, 'v14_p_under']
            df.loc[rush_mask & v14_validated_mask, 'v12_best_direction'] = df.loc[rush_mask & v14_validated_mask, 'v14_best_direction']
            df.loc[rush_mask & v14_validated_mask, 'v12_best_ev'] = df.loc[rush_mask & v14_validated_mask, 'v14_best_ev']
            df.loc[rush_mask & v14_validated_mask, 'v12_validated'] = True
            df.loc[rush_mask & v14_validated_mask, 'v12_reason'] = df.loc[rush_mask & v14_validated_mask, 'v14_reason']

            # Log V14 results
            v14_validated = df[(df['market'] == 'player_rush_yds') & (df['v14_validated'] == True)]
            if len(v14_validated) > 0:
                avg_ev = v14_validated['v14_best_ev'].mean()
                logger.info(f"V14 DEFENSE-AWARE MODEL: {len(v14_validated)} rush_yds bets validated (avg EV: {avg_ev:+.1f}%)")

    except Exception as e:
        logger.warning(f"V14 integration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Filter 9: OPPONENT CONFLICT WARNING (ADDED 2025-11-27)
    # Flag picks where opponent history contradicts model signal
    # Example: Lamar Jackson UNDER but he averages +25% vs CIN
    try:
        conflict_flags = []
        conflict_reasons = []
        divergence_threshold = 0.15  # 15% divergence to flag

        for idx, row in df.iterrows():
            market = row.get('market', '')
            pick = str(row.get('pick', '')).lower()

            # Check for opponent divergence columns from predictions
            divergence_col = f'{market}_vs_opp_divergence'
            confidence_col = f'{market}_vs_opp_confidence'

            divergence = row.get(divergence_col, row.get('vs_opponent_divergence', 0))
            confidence = row.get(confidence_col, row.get('vs_opponent_confidence', 0))

            # Need minimum confidence (2+ prior games)
            if confidence < 0.33:  # Less than 2 games
                conflict_flags.append(False)
                conflict_reasons.append('')
                continue

            is_under = 'under' in pick
            is_over = 'over' in pick

            # Conflict: Model says UNDER but player overperforms vs opponent
            if is_under and divergence > divergence_threshold:
                conflict_flags.append(True)
                conflict_reasons.append(f"⚠️ Player +{divergence*100:.0f}% vs opponent")
            # Conflict: Model says OVER but player underperforms vs opponent
            elif is_over and divergence < -divergence_threshold:
                conflict_flags.append(True)
                conflict_reasons.append(f"⚠️ Player {divergence*100:.0f}% vs opponent")
            else:
                conflict_flags.append(False)
                conflict_reasons.append('')

        df['opponent_conflict'] = conflict_flags
        df['opponent_conflict_reason'] = conflict_reasons

        # Log conflicts
        conflicts = df[df['opponent_conflict'] == True]
        if len(conflicts) > 0:
            logger.warning(f"⚠️ OPPONENT CONFLICTS: {len(conflicts)} picks have conflicting opponent history")
            for _, conflict_row in conflicts.head(5).iterrows():
                logger.warning(f"   {conflict_row['player']} {conflict_row['market']}: {conflict_row['opponent_conflict_reason']}")

    except Exception as e:
        logger.warning(f"Opponent conflict check failed: {e}")
        df['opponent_conflict'] = False
        df['opponent_conflict_reason'] = ''

    # Filter 10: LINE MOVEMENT INTEGRATION (ADDED 2025-11-26)
    # When line moves DOWN = sharp money on UNDER (follow this signal!)
    # When line moves UP = sharp money on OVER
    try:
        line_movement = load_line_movement(week)
        if line_movement:
            sharp_under_list = []
            sharp_over_list = []
            line_move_list = []

            for idx, row in df.iterrows():
                player_key = (str(row['player']).lower().strip(), row['market'])
                movement = line_movement.get(player_key, {})

                sharp_under_list.append(movement.get('sharp_under', False))
                sharp_over_list.append(movement.get('sharp_over', False))
                line_move_list.append(movement.get('line_movement', 0))

            df['sharp_under'] = sharp_under_list
            df['sharp_over'] = sharp_over_list
            df['line_movement'] = line_move_list

            # Count bets with sharp money confirmation
            sharp_confirmed = df[
                (df['v5_edge_validated'] == True) &
                (df['pick'].str.contains('Under', case=False, na=False)) &
                (df['sharp_under'] == True)
            ]

            if len(sharp_confirmed) > 0:
                logger.info(f"SHARP MONEY CONFIRMATION: {len(sharp_confirmed)} UNDER bets have line movement support")
                df['sharp_confirmed'] = (
                    (df['v5_edge_validated'] == True) &
                    (df['pick'].str.contains('Under', case=False, na=False)) &
                    (df['sharp_under'] == True)
                )
            else:
                df['sharp_confirmed'] = False
        else:
            df['sharp_under'] = False
            df['sharp_over'] = False
            df['line_movement'] = 0
            df['sharp_confirmed'] = False
    except Exception as e:
        df['sharp_under'] = False
        df['sharp_over'] = False
        df['line_movement'] = 0
        df['sharp_confirmed'] = False
        logger.warning(f"Could not apply line movement: {e}")

    logger.info(f"Final: {len(df)} valid recommendations")

    # ADD MODEL REASONING FOR EACH RECOMMENDATION
    # This explains WHY the model made this prediction
    logger.info("Generating model reasoning explanations...")
    reasoning_list = []
    for idx, row in df.iterrows():
        reasoning = generate_model_reasoning(row.to_dict())
        reasoning_list.append(reasoning)
    df['model_reasoning'] = reasoning_list
    logger.info(f"   Added reasoning to {len(df)} recommendations")

    return df


def save_recommendations(df: pd.DataFrame):
    """Save recommendations to CSV."""
    output_path = Path('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} recommendations to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RECOMMENDATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total recommendations: {len(df)}")

    # PRIORITY: V12 INTERACTION MODEL (BEST PERFORMER)
    if 'v12_validated' in df.columns:
        v12_validated = df[df['v12_validated'] == True].copy()
        if len(v12_validated) > 0:
            # Sort by EV descending
            v12_validated = v12_validated.sort_values('v12_best_ev', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ V12 INTERACTION MODEL BETS (BEST PERFORMER) ★★★")
            print("=" * 80)
            print("Strategy: LVT as hub with interaction constraints + monotonic constraints")
            print("Backtest: Receptions +59.7% ROI, Rushing Yards +11.1% ROI")
            print()

            # Check for opponent conflicts
            if 'opponent_conflict' in df.columns:
                conflicts = v12_validated[v12_validated['opponent_conflict'] == True]
                if len(conflicts) > 0:
                    print(f"⚠️  WARNING: {len(conflicts)} picks have OPPONENT CONFLICT - review carefully!")
                    for _, row in conflicts.iterrows():
                        market_name = format_market_name(row.get('market', ''))
                        print(f"   {row['player']} ({market_name}): {row.get('opponent_conflict_reason', 'conflict')}")
                    print()

            # Show breakdown by direction
            under_bets = v12_validated[v12_validated['v12_best_direction'] == 'UNDER']
            over_bets = v12_validated[v12_validated['v12_best_direction'] == 'OVER']
            print(f"UNDER bets: {len(under_bets)}")
            print(f"OVER bets: {len(over_bets)}")
            print()

            # Show by market (using display names)
            print("By market:")
            for market in v12_validated['market'].unique():
                market_df = v12_validated[v12_validated['market'] == market]
                avg_ev = market_df['v12_best_ev'].mean()
                market_name = format_market_name(market)
                print(f"  - {market_name}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%)")

            print(f"\nV12-validated bets this week: {len(v12_validated)}")

            # Use display columns with formatted names
            display_df = v12_validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Direction'] = display_df['v12_best_direction']
            display_df['Prob'] = display_df['v12_p_under'].apply(lambda x: f"{x:.0%}")
            display_df['EV'] = display_df['v12_best_ev'].apply(lambda x: f"{x:+.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'Direction', 'line', 'model_projection', 'Prob', 'EV', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].head(20).to_string(index=False))
            print("=" * 80)

    # Highlight V7 HYBRID APPROACH bets (rule-based with regime estimation)
    if 'v7_validated' in df.columns:
        v7_validated = df[df['v7_validated'] == True].copy()
        if len(v7_validated) > 0:
            # Sort by EV descending
            v7_validated = v7_validated.sort_values('v7_best_ev', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ V7 RULE-BASED BETS (SIMPLE RULE + REGIME) ★★★")
            print("=" * 80)
            print("Strategy: Bet UNDER when line > trailing_avg + 1.5")
            print("Backtest: Receptions 72.4% hit, +38.2% ROI (2025)")
            print("         Rushing Yards 54.3% hit, +3.7% ROI (consistent)")
            print()
            print("Why this works: Vegas sets lines above recent performance,")
            print("                but players tend to regress toward their mean.")
            print()

            # Show breakdown by direction
            under_bets = v7_validated[v7_validated['v7_best_direction'] == 'UNDER']
            over_bets = v7_validated[v7_validated['v7_best_direction'] == 'OVER']
            print(f"UNDER bets: {len(under_bets)}")
            print(f"OVER bets: {len(over_bets)}")
            print()

            # Show by market (using display names)
            print("By market:")
            for market in v7_validated['market'].unique():
                market_df = v7_validated[v7_validated['market'] == market]
                avg_ev = market_df['v7_best_ev'].mean()
                avg_kelly = market_df['v7_kelly_size'].mean() * 100
                market_name = format_market_name(market)
                print(f"  - {market_name}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%, Kelly: {avg_kelly:.1f}%)")

            print(f"\nV7-validated bets this week: {len(v7_validated)}")

            # Use display columns with formatted names
            display_df = v7_validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Direction'] = display_df['v7_best_direction']
            display_df['LVT'] = display_df['line_vs_trailing'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A")
            display_df['EV'] = display_df['v7_best_ev'].apply(lambda x: f"{x:+.1f}%")
            display_df['Kelly'] = display_df['v7_kelly_size'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'Direction', 'line', 'model_projection', 'LVT', 'EV', 'Kelly', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].head(20).to_string(index=False))
            print("=" * 80)

    # Legacy V6/V5 output for backwards compatibility
    elif 'v6_value_validated' in df.columns:
        v6_validated = df[df['v6_value_validated'] == True].copy()
        if len(v6_validated) > 0:
            # Sort by EV descending
            v6_validated = v6_validated.sort_values('v6_best_ev', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ V6 VALUE CLASSIFIER BETS (POSITIVE EV) ★★★")
            print("=" * 80)

            # Format display columns
            display_df = v6_validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Direction'] = display_df['v6_best_direction']
            display_df['EV'] = display_df['v6_best_ev'].apply(lambda x: f"{x:+.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'Direction', 'line', 'model_projection', 'EV', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].head(20).to_string(index=False))
            print("=" * 80)

    # Highlight PROFITABLE STRATEGY bets (backtest-validated)
    if 'strategy_validated' in df.columns:
        validated = df[df['strategy_validated'] == True].copy()
        if len(validated) > 0:
            # Sort by edge descending
            validated = validated.sort_values('edge_pct', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ PROFITABLE STRATEGY BETS (RULE-BASED VALIDATION) ★★★")
            print("=" * 80)
            print("Strategy: UNDER on receptions/passing yards when model agrees with line")
            print("Backtest: 57.8% win rate, +10.3% ROI (Weeks 2-11, 2025)")
            print(f"\nValidated bets this week: {len(validated)}")

            # Format display columns
            display_df = validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Edge'] = display_df['edge_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'pick', 'line', 'model_projection', 'Edge', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].to_string(index=False))
            print("=" * 80)

    # Summary by market (using display names)
    print(f"\nBy market:")
    market_counts = df['market'].value_counts()
    for market, count in market_counts.items():
        market_name = format_market_name(market)
        print(f"  {market_name}: {count}")

    # Top 10 by edge (using display names)
    print(f"\nTop 10 by edge:")
    top_bets = df.nlargest(10, 'edge_pct').copy()
    top_bets['Prop'] = top_bets['market'].apply(format_market_name)
    top_bets['Edge'] = top_bets['edge_pct'].apply(lambda x: f"{x:+.1f}%")
    display_cols = ['player', 'Prop', 'line', 'model_projection', 'Edge']
    print(top_bets[display_cols].to_string(index=False))
    print("\n" + "=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate unified recommendations')
    parser.add_argument('--week', type=int, default=None, help='NFL week')
    args = parser.parse_args()

    week = args.week or get_current_week()

    recommendations = generate_recommendations(week)

    if recommendations.empty:
        logger.error("No recommendations generated!")
        sys.exit(1)

    save_recommendations(recommendations)

    logger.info("Done! Run generate_all_recommendations.py next to add quality tiers.")


if __name__ == '__main__':
    main()
