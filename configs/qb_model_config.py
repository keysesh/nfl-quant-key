"""
QB Model Configuration - Separate config for QB passing yards model

This model uses regression + variance estimation instead of direct classification,
which is more appropriate for the high-variance QB passing yards market.
"""

from typing import Dict, List
from dataclasses import dataclass


# =============================================================================
# QB-SPECIFIC FEATURES
# =============================================================================
# These replace the receiver-centric features in the main model

QB_FEATURES: List[str] = [
    # -------------------------------------------------------------------------
    # Core QB Trailing Stats (from weekly_stats)
    # -------------------------------------------------------------------------
    'qb_trailing_pass_yds',        # EWMA of passing yards
    'qb_trailing_attempts',        # EWMA of pass attempts
    'qb_trailing_completion_pct',  # EWMA of completion percentage
    'qb_passing_epa_trailing',     # EWMA of passing EPA
    'qb_passing_cpoe_trailing',    # EWMA of CPOE (if available)
    'sacks_suffered_trailing',     # EWMA of times sacked

    # -------------------------------------------------------------------------
    # NGS Passing Features (from ngs_passing.parquet)
    # -------------------------------------------------------------------------
    'avg_time_to_throw',           # Seconds to release
    'completion_pct_above_exp',    # CPOE from NGS
    'aggressiveness',              # % of throws into tight windows
    'avg_intended_air_yards',      # Average downfield depth

    # -------------------------------------------------------------------------
    # QB Starter Detection (from qb_starter.py)
    # -------------------------------------------------------------------------
    'qb_is_starter',               # 1 if confirmed starter, 0 otherwise
    'qb_starter_confidence',       # Confidence in starter classification (0-1)

    # -------------------------------------------------------------------------
    # Game Context Features
    # -------------------------------------------------------------------------
    'vegas_total',                 # Expected total points
    'vegas_spread',                # Point spread (negative = favorite)
    'implied_team_total',          # Team implied total

    # -------------------------------------------------------------------------
    # Game Script Features (derived from spread)
    # -------------------------------------------------------------------------
    'expected_game_script_multiplier',  # How game script affects attempts
    'implied_pass_attempts',            # Expected pass attempts given game script
    'game_script_volatility',           # Uncertainty in game script

    # -------------------------------------------------------------------------
    # Opponent Defense Features
    # -------------------------------------------------------------------------
    'opp_pass_defense_epa',        # Opponent's pass defense EPA
    'opp_pass_yds_def_vs_avg',     # Opponent pass yards allowed vs league avg

    # -------------------------------------------------------------------------
    # Line-Based Features (same as main model)
    # -------------------------------------------------------------------------
    'line_vs_trailing',            # Line - trailing stat
    'line_level',                  # Raw line value
    'line_in_sweet_spot',          # Gaussian decay from sweet spot
    'player_under_rate',           # Historical under hit rate for player
    'player_bias',                 # Player's tendency vs line
]


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

@dataclass
class QBModelParams:
    """Hyperparameters for QB passing yards model."""

    # Regressor (predicts expected passing yards)
    regressor_n_estimators: int = 200
    regressor_max_depth: int = 4
    regressor_learning_rate: float = 0.08
    regressor_subsample: float = 0.8
    regressor_colsample_bytree: float = 0.7
    regressor_min_child_weight: int = 5

    # Variance model (predicts abs(residual))
    variance_n_estimators: int = 100
    variance_max_depth: int = 3
    variance_learning_rate: float = 0.1
    variance_subsample: float = 0.8

    # Common
    random_state: int = 42
    verbosity: int = 0


QB_MODEL_PARAMS = QBModelParams()


# =============================================================================
# TRAINING FILTERS
# =============================================================================

# Minimum starter confidence to include in training
MIN_STARTER_CONFIDENCE: float = 0.6

# Minimum pass attempts to include (filters out injury games)
MIN_PASS_ATTEMPTS: int = 15

# EWMA span for QB trailing stats
QB_EWMA_SPAN: int = 6


# =============================================================================
# GAME SCRIPT ADJUSTMENTS
# =============================================================================
# Based on analysis of pass rate by score differential

GAME_SCRIPT_MULTIPLIERS: Dict[str, float] = {
    'big_underdog': 1.15,      # Spread < -7: +15% attempts
    'slight_underdog': 1.05,   # Spread -7 to -3: +5% attempts
    'neutral': 1.00,           # Spread -3 to +3: baseline
    'slight_favorite': 0.95,   # Spread +3 to +7: -5% attempts
    'big_favorite': 0.85,      # Spread > +7: -15% attempts
}

GAME_SCRIPT_VOLATILITY: Dict[str, float] = {
    'big_underdog': 0.25,      # High volatility (could abandon pass)
    'slight_underdog': 0.15,   # Moderate volatility
    'neutral': 0.10,           # Low volatility
    'slight_favorite': 0.15,   # Moderate volatility
    'big_favorite': 0.25,      # High volatility (could be blowout)
}


# =============================================================================
# SWEET SPOT CONFIGURATION
# =============================================================================

QB_SWEET_SPOT_CENTER: float = 225.0   # Typical starter passing yards
QB_SWEET_SPOT_WIDTH: float = 40.0     # Width of sweet spot


# =============================================================================
# VARIANCE BOUNDS
# =============================================================================
# Clip variance predictions to realistic range

MIN_PREDICTED_STD: float = 40.0    # Minimum std (very predictable game)
MAX_PREDICTED_STD: float = 120.0   # Maximum std (chaotic game)


# =============================================================================
# BASELINE PASS ATTEMPTS
# =============================================================================
# Used to estimate implied_pass_attempts

BASELINE_PASS_ATTEMPTS: float = 32.0  # League average pass attempts per game


# =============================================================================
# SPREAD FILTER (CRITICAL FOR PROFITABILITY)
# =============================================================================
# Only bet QB props in close games where game script variance is manageable
#
# Backtest Results (Weeks 12-14, 2025):
#   Close games (|spread| <= 3): 64.6% hit rate, +23.3% ROI
#   Moderate (3 < |spread| <= 7): 31.7% hit rate, -39.5% ROI
#   Big spread (|spread| > 7): 47.6% hit rate, -9.5% ROI
#
# Hypothesis: Close games have stable game scripts, while lopsided games
# cause extreme pass volume variance that models cannot predict.

QB_SPREAD_FILTER_ENABLED: bool = True
QB_MAX_SPREAD_ABS: float = 3.0  # Only bet when |spread| <= this value
