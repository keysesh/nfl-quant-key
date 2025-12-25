"""
Central Model Configuration - Single Source of Truth for NFL QUANT

This module provides a centralized location for all model-related configuration.
ALL version numbers and feature lists are defined here ONLY.

Usage:
    from configs.model_config import (
        MODEL_VERSION,
        FEATURES,
        get_model_path,
    )

To upgrade to a new version:
    1. Change MODEL_VERSION
    2. Add new features to FEATURES list
    That's it. No other files need version updates.
"""
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


# =============================================================================
# VERSION CONFIGURATION - THE ONLY PLACE VERSION IS DEFINED
# =============================================================================
MODEL_VERSION = "29"  # V29: Added vegas_agreement + lvt_direction features (Week 16 analysis)

# Derived version strings (DO NOT hardcode elsewhere)
MODEL_VERSION_FULL = f"V{MODEL_VERSION}"  # "V19"
MODEL_VERSION_LOWER = f"v{MODEL_VERSION}"  # "v19"


# =============================================================================
# FEATURE LIST - THE ONLY PLACE FEATURES ARE DEFINED
# =============================================================================
# Add new features to the end of this list. That's all you need for a new version.

FEATURES = [
    # -------------------------------------------------------------------------
    # Core V12 Features (12 features)
    # -------------------------------------------------------------------------
    'line_vs_trailing',
    'line_level',
    'line_in_sweet_spot',
    'player_under_rate',
    'player_bias',
    'market_under_rate',
    'LVT_x_player_tendency',
    'LVT_x_player_bias',
    'LVT_x_regime',
    'LVT_in_sweet_spot',
    'market_bias_strength',
    'player_market_aligned',

    # -------------------------------------------------------------------------
    # V17 Features (8 features) - Interactions + Skill
    # -------------------------------------------------------------------------
    'lvt_x_defense',
    'lvt_x_rest',
    'avg_separation',
    'avg_cushion',
    'trailing_catch_rate',
    'snap_share',
    'target_share',
    'opp_wr1_receptions_allowed',

    # -------------------------------------------------------------------------
    # V18 Features (7 features) - Game Context + Skill
    # -------------------------------------------------------------------------
    'game_pace',
    'vegas_total',
    'vegas_spread',
    'implied_team_total',
    'adot',
    'pressure_rate',
    'opp_pressure_rate',

    # -------------------------------------------------------------------------
    # V19 Features (4 features) - Rush/Receiving Improvements
    # -------------------------------------------------------------------------
    'oline_health_score',
    'box_count_expected',
    'slot_snap_pct',
    'target_share_trailing',

    # -------------------------------------------------------------------------
    # V23 Features (4 features) - Opponent Context (ACTUAL data, not hardcoded)
    # NaN values are NOT filled with defaults - XGBoost handles missing natively
    # NOTE: Trailing features (opp_*_allowed_trailing) removed - 100% NaN in training data
    #       These can be added back when historical opponent data is properly joined
    # -------------------------------------------------------------------------
    'opp_pass_yds_def_vs_avg',  # Fixed: was opp_pass_def_vs_avg (missing 'yds')
    'opp_rush_yds_def_vs_avg',  # Fixed: was opp_rush_def_vs_avg (missing 'yds')
    'opp_def_epa',
    'has_opponent_context',  # Flag: 1 if opponent data available, 0 if missing

    # -------------------------------------------------------------------------
    # V24 Features (11 features) - Position-Specific Defensive Matchup Analysis
    # Analyzes how defenses perform against WR1s, WR2s, slot receivers, etc.
    # Uses depth_charts.parquet (pos_rank), ngs_receiving (avg_cushion), participation (coverage type)
    # -------------------------------------------------------------------------
    'pos_rank',  # Player's depth chart rank (1, 2, 3) from depth_charts.parquet
    'is_starter',  # Binary: 1 if pos_rank == 1
    'is_slot_receiver',  # Binary: from NGS avg_cushion > 5.5 and aDOT < 8.0
    'slot_alignment_pct',  # Percentage of snaps in slot alignment
    'opp_position_yards_allowed_trailing',  # EWMA yards opponent allows to this position role
    'opp_position_volume_allowed_trailing',  # EWMA receptions/carries opponent allows to this role
    'opp_man_coverage_rate_trailing',  # Opponent's man coverage % (EWMA from participation.parquet)
    'slot_funnel_score',  # How much defense allows to slot vs outside (higher = more slot vulnerable)
    'man_coverage_adjustment',  # Multiplier based on coverage type (0.93-1.07)
    'position_role_x_opp_yards',  # Interaction: pos_rank * opp yards allowed
    'has_position_context',  # Flag: 1 if position data available, 0 if missing

    # -------------------------------------------------------------------------
    # V25 Features REMOVED (8 features) - Team Health Synergy
    # REMOVED in V26: All 8 features had 0% importance across all markets
    # Original features: team_synergy_multiplier, oline_health_score_v25,
    # wr_corps_health, has_synergy_bonus, cascade_efficiency_boost,
    # wr_coverage_reduction, returning_player_count, has_synergy_context
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # V28 Features (6 features) - Elo & Situational (Tier 3-4 Blueprint)
    # Implements NFL Edge Betting Blueprint requirements:
    # - Tier 3: Elo ratings for team power rankings
    # - Tier 4: Rest days, home field advantage adjustments
    # -------------------------------------------------------------------------
    'elo_rating_home',                  # Home team Elo rating (1350-1700 range)
    'elo_rating_away',                  # Away team Elo rating
    'elo_diff',                         # Home Elo - Away Elo (with HFA)
    'ybc_proxy',                        # Yards Before Contact proxy (air_yards for receivers)
    'rest_days',                        # Days since last game (4-14 range)
    'hfa_adjustment',                   # Team-specific home field advantage factor

    # -------------------------------------------------------------------------
    # V28.1 Features (3 features) - Player Injury Data (Tier 4 Blueprint)
    # Uses NFLverse injury data for backtesting compatibility
    # -------------------------------------------------------------------------
    'injury_status_encoded',            # 0=None/Probable, 1=Questionable, 2=Doubtful, 3=Out
    'practice_status_encoded',          # 0=Full, 1=Limited, 2=DNP
    'has_injury_designation',           # Binary flag if player has any injury status

    # -------------------------------------------------------------------------
    # V29 Features (2 features) - Vegas Agreement Signal
    # Analysis showed: WITH Vegas = 67% win rate, AGAINST = 27%
    # Let ML learn when to fight Vegas instead of hard filter
    # -------------------------------------------------------------------------
    'vegas_agreement',                  # 1 if betting WITH Vegas direction, 0 if AGAINST
    'lvt_direction',                    # Sign of LVT: +1 if line > trailing, -1 if line < trailing
]

# Current feature count for validation
FEATURE_COUNT = len(FEATURES)  # 57 (V29: added vegas_agreement, lvt_direction)

# Alias for current features (use FEATURES directly when possible)
CURRENT_FEATURE_COLS = FEATURES


# =============================================================================
# MARKET-SPECIFIC FEATURE SELECTION (V26)
# =============================================================================
# Features to EXCLUDE for each market type. These features had 0% importance
# in holdout analysis for the specific market (e.g., receiving features don't
# help predict rushing yards).

# Features only relevant to RECEIVING markets (receptions, reception_yds)
_RECEIVING_ONLY_FEATURES = {
    'avg_separation',           # NGS receiving metric
    'avg_cushion',              # NGS receiving metric
    'trailing_catch_rate',      # Receiving efficiency
    'target_share',             # Receiving volume
    'target_share_trailing',    # 4-week rolling target share
    'opp_wr1_receptions_allowed',  # WR1 coverage metric
    'adot',                     # Average depth of target
    'is_slot_receiver',         # Slot alignment
    'slot_alignment_pct',       # Slot snap percentage
    'slot_funnel_score',        # Slot vulnerability
}

# Features only relevant to RUSHING markets (rush_yds, rush_attempts)
_RUSHING_ONLY_FEATURES = {
    'oline_health_score',       # O-line health (rush blocking)
    'box_count_expected',       # Defenders in the box
}

# Features only relevant to PASSING markets (pass_yds, pass_attempts)
_PASSING_ONLY_FEATURES = {
    'pressure_rate',            # QB pressure
    'opp_pressure_rate',        # Opponent pressure generated
}

# Define which features to EXCLUDE for each market
MARKET_FEATURE_EXCLUSIONS: dict = {
    # Receiving markets - exclude rushing features
    'player_receptions': _RUSHING_ONLY_FEATURES,
    'player_reception_yds': _RUSHING_ONLY_FEATURES,

    # Rushing markets - exclude receiving features
    'player_rush_yds': _RECEIVING_ONLY_FEATURES,
    'player_rush_attempts': _RECEIVING_ONLY_FEATURES,

    # Passing markets - exclude receiving and rushing specific features
    'player_pass_yds': _RECEIVING_ONLY_FEATURES | _RUSHING_ONLY_FEATURES,
    'player_pass_completions': _RECEIVING_ONLY_FEATURES | _RUSHING_ONLY_FEATURES,
    'player_pass_attempts': _RECEIVING_ONLY_FEATURES | _RUSHING_ONLY_FEATURES,
}


def get_market_features(market: str) -> list:
    """
    Get the feature list for a specific market, excluding irrelevant features.

    Args:
        market: Market name (e.g., 'player_rush_yds')

    Returns:
        List of features to use for this market
    """
    exclusions = MARKET_FEATURE_EXCLUSIONS.get(market, set())
    return [f for f in FEATURES if f not in exclusions]


def get_market_feature_count(market: str) -> int:
    """Get the number of features for a specific market."""
    return len(get_market_features(market))


# =============================================================================
# BACKWARD COMPATIBILITY - Legacy aliases (will deprecate)
# =============================================================================
# These exist only for backward compatibility with old code.
# New code should use FEATURES directly.

V12_FEATURE_COLS = FEATURES[:12]
V17_FEATURE_COLS = FEATURES[:20]
V18_FEATURE_COLS = FEATURES[:27]
V19_FEATURE_COLS = FEATURES  # Current


# =============================================================================
# SWEET SPOT PARAMETERS - Gaussian Decay Configuration
# =============================================================================
@dataclass(frozen=True)
class SweetSpotConfig:
    """Configuration for smooth sweet spot calculation using Gaussian decay."""
    center: float
    width: float

    def calculate(self, line: float) -> float:
        """Calculate smooth sweet spot value using Gaussian decay."""
        return float(np.exp(-((line - self.center) ** 2) / (2 * self.width ** 2)))


SWEET_SPOT_PARAMS: Dict[str, SweetSpotConfig] = {
    # Core volume markets
    'player_receptions': SweetSpotConfig(center=5.5, width=2.5),  # V27: Recalibrated for actual sweet spot (5.5-6.5 rec)
    'player_rush_yds': SweetSpotConfig(center=55.0, width=25.0),
    'player_reception_yds': SweetSpotConfig(center=55.0, width=25.0),
    'player_pass_yds': SweetSpotConfig(center=250.0, width=50.0),
    # V21 Expansion markets
    'player_rush_attempts': SweetSpotConfig(center=15.0, width=5.0),
    'player_pass_completions': SweetSpotConfig(center=20.0, width=5.0),
    'player_pass_attempts': SweetSpotConfig(center=32.0, width=6.0),
    'player_pass_tds': SweetSpotConfig(center=1.5, width=0.5),  # TD markets are binary-ish
}

DEFAULT_SWEET_SPOT = SweetSpotConfig(center=5.0, width=2.5)


# =============================================================================
# TRAILING STAT DEFLATION - Regression to Mean Adjustment
# =============================================================================
# Trailing stats (EWMA) systematically overshoot actuals due to:
# 1. Players getting props are typically top performers on hot streaks
# 2. Regression to mean is not applied
# 3. EWMA span=4 overweights recent performances
#
# These factors deflate trailing stats toward realistic expectations.
# Based on Week 12-14 2025 holdout analysis:
#   - player_reception_yds: +75% overshoot → 0.83 deflation
#   - player_rush_yds: +59% overshoot → 0.95 deflation
#   - player_receptions: +36% overshoot → 0.92 deflation
#   - player_pass_yds: +49% overshoot → 1.02 (slight under)
TRAILING_DEFLATION_FACTORS: Dict[str, float] = {
    'player_receptions': 0.92,
    'player_reception_yds': 0.83,
    'player_rush_yds': 0.95,
    'player_pass_yds': 1.02,
    # Conservative defaults for other markets
    'player_rush_attempts': 0.95,
    'player_pass_completions': 0.95,
    'player_pass_attempts': 0.95,
    'player_pass_tds': 1.0,  # TD props are different
}

DEFAULT_TRAILING_DEFLATION = 0.90  # Overall median deflation

# EWMA span for trailing stats (increased from 4 to 6 for stability)
EWMA_SPAN = 6


def smooth_sweet_spot(line: float, market: str = None, center: float = None, width: float = None) -> float:
    """Calculate smooth sweet spot using Gaussian decay."""
    if market and market in SWEET_SPOT_PARAMS:
        config = SWEET_SPOT_PARAMS[market]
        center = config.center
        width = config.width
    elif center is None or width is None:
        config = DEFAULT_SWEET_SPOT
        center = center if center is not None else config.center
        width = width if width is not None else config.width

    return float(np.exp(-((line - center) ** 2) / (2 * width ** 2)))


# =============================================================================
# FEATURE FLAGS - Toggle Features On/Off
# =============================================================================
@dataclass
class FeatureFlags:
    """Feature flags for toggling experimental features."""
    use_lvt_x_defense: bool = True
    use_lvt_x_rest: bool = True
    use_smooth_sweet_spot: bool = True
    use_weather_interactions: bool = False
    use_injury_interactions: bool = False


FEATURE_FLAGS = FeatureFlags()


# =============================================================================
# MARKET CONFIGURATION
# =============================================================================
# All markets the system supports (for data fetching)
SUPPORTED_MARKETS = [
    'player_receptions',
    'player_rush_yds',
    'player_reception_yds',
    'player_pass_yds',
    # V21 Expansion - Additional markets
    'player_rush_attempts',
    'player_pass_completions',
    'player_pass_attempts',
    'player_pass_tds',
    # TD Props - YES only bets (OVER_ONLY constraint)
    'player_anytime_td',
    'player_1st_td',
]

# =============================================================================
# MARKET-SPECIFIC FILTERS (V27) - Game Context Filters
# =============================================================================
# These filters exclude low-value bets based on game context
# E.g., receptions bets in blowouts have high variance

@dataclass
class MarketFilter:
    """Filter configuration for a specific market."""
    max_spread: Optional[float] = None    # Skip games with |spread| > this
    min_snap_share: Optional[float] = None  # Only established players
    exclude_positions: Optional[List[str]] = None  # Positions to exclude


MARKET_FILTERS: Dict[str, MarketFilter] = {
    # Receptions: TEs have no edge (50% win rate), skip lopsided games
    'player_receptions': MarketFilter(
        max_spread=7.0,           # V27: Skip blowouts where game script kills targets
        min_snap_share=0.40,      # V27: Only established players
        exclude_positions=['TE'],  # V27: TEs have 50% edge (no value)
    ),
    # Pass yds: Only bet close games (existing logic)
    'player_pass_yds': MarketFilter(
        max_spread=3.0,           # Existing: close games only
    ),
}


def get_market_filter(market: str) -> Optional[MarketFilter]:
    """Get filter configuration for a market."""
    return MARKET_FILTERS.get(market)


# Markets for XGBoost classifier training (volume-based only, no TD props)
# TD props are binary and need different modeling (Poisson/logistic)
CLASSIFIER_MARKETS = [
    'player_receptions',
    'player_rush_yds',
    'player_reception_yds',
    'player_rush_attempts',
    'player_pass_attempts',      # Re-enabled Dec 2025: Adding full edge support
    'player_pass_completions',   # Re-enabled Dec 2025: Adding full edge support
    # player_pass_yds excluded: -15.8% ROI in holdout, failing both directions (Dec 14 2025)
    # player_pass_tds excluded: -18.2% ROI, binary distribution wrong for XGBoost
]

# =============================================================================
# MARKET DIRECTION CONSTRAINTS - Market-Specific (Dec 20, 2025 Backtest)
# =============================================================================
# Direction constraints are now MARKET-SPECIFIC based on historical data:
#
# UNDER markets (line typically set too high):
#   - player_receptions: 58% UNDER rate
#   - player_reception_yds: 55% UNDER rate
#   - player_pass_attempts: 58% UNDER rate
#   - player_pass_completions: 55% UNDER rate
#   - player_rush_attempts: 53% UNDER rate
#
# OVER markets (line typically set too low):
#   - player_rush_yds: 87% OVER rate (lines consistently under-estimate)
#
# Valid values: 'UNDER_ONLY', 'OVER_ONLY', 'BOTH'
# FIXED Dec 2025: Changed to BOTH for markets where constraint was suppressing wins
# Previous UNDER_ONLY on pass_attempts, pass_completions, rush_attempts was wrong
MARKET_DIRECTION_CONSTRAINTS: Dict[str, str] = {
    'player_receptions': 'UNDER_ONLY',       # Validated: 56.8% UNDER in true OOS
    'player_reception_yds': 'UNDER_ONLY',    # Validated: 52.5% UNDER in true OOS
    'player_rush_yds': 'BOTH',               # Changed: Let model decide direction
    'player_rush_attempts': 'BOTH',          # FIXED: Was UNDER_ONLY (wrong)
    'player_pass_attempts': 'BOTH',          # FIXED: Was UNDER_ONLY (wrong)
    'player_pass_completions': 'BOTH',       # FIXED: Was UNDER_ONLY (wrong)
    # TD props - can only bet YES (player scores), never NO
    'player_anytime_td': 'OVER_ONLY',        # Anytime TD = YES only
    'player_1st_td': 'OVER_ONLY',            # First TD = YES only
}

# Markets for Monte Carlo simulation (all markets)
SIMULATOR_MARKETS = SUPPORTED_MARKETS

MARKET_TO_STAT = {
    'player_receptions': 'receptions',
    'player_reception_yds': 'receiving_yards',
    'player_receiving_yards': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_rushing_yards': 'rushing_yards',
    'player_pass_yds': 'passing_yards',
    'player_passing_yards': 'passing_yards',
    'player_rush_att': 'carries',
    'player_rush_attempts': 'carries',
    'player_pass_attempts': 'attempts',
    'player_pass_tds': 'passing_tds',
    'player_pass_completions': 'completions',
    # TD Props (anytime = rushing_tds + receiving_tds)
    'player_anytime_td': 'total_tds',  # Combined TDs
    'player_1st_td': 'total_tds',
}

MARKET_TO_PREDICTION_COLS = {
    'player_receptions': ('receptions_mean', 'receptions_std'),
    'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
    'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
    'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
    # V21 Expansion - Additional markets
    'player_rush_attempts': ('rushing_attempts_mean', 'rushing_attempts_std'),
    'player_pass_completions': ('passing_completions_mean', 'passing_completions_std'),
    'player_pass_attempts': ('passing_attempts_mean', 'passing_attempts_std'),
    'player_pass_tds': ('passing_tds_mean', 'passing_tds_std'),
    # TD Props - use Poisson model (mean = expected TDs)
    'player_anytime_td': ('total_tds_mean', 'total_tds_std'),
    'player_1st_td': ('total_tds_mean', 'total_tds_std'),
}


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Import from centralized config_paths for consistency
from nfl_quant.config_paths import PROJECT_ROOT, MODELS_DIR


def get_model_path(version: str = None, suffix: str = '') -> Path:
    """Get path to model file for a specific version."""
    v = version or MODEL_VERSION_LOWER
    return MODELS_DIR / f'{v}_interaction_classifier{suffix}.joblib'


def get_active_model_path() -> Path:
    """Get path to the active (production) model."""
    return MODELS_DIR / 'active_model.joblib'


def get_versioned_model_path() -> Path:
    """Get path to the current version's model file."""
    return MODELS_DIR / f'{MODEL_VERSION_LOWER}_interaction_classifier.joblib'


# =============================================================================
# MODEL TRAINING PARAMETERS
# =============================================================================
@dataclass
class ModelParams:
    """XGBoost model parameters."""
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    verbosity: int = 0


MODEL_PARAMS = ModelParams()


# =============================================================================
# VALIDATION HELPERS
# =============================================================================
def validate_features(feature_list: List[str]) -> Dict[str, List[str]]:
    """Check which features are valid and which are missing."""
    valid = [f for f in feature_list if f in FEATURES]
    invalid = [f for f in feature_list if f not in FEATURES]
    return {'valid': valid, 'invalid': invalid}


def get_monotonic_constraints(feature_cols: List[str]) -> str:
    """Get monotonic constraints string for XGBoost."""
    constraint_map = {
        'line_vs_trailing': 1,
        'player_under_rate': 1,
        'market_under_rate': 1,
        'player_bias': -1,
        'lvt_x_defense': -1,
        'lvt_x_rest': 0,
    }

    constraints = []
    for feat in feature_cols:
        constraints.append(constraint_map.get(feat, 0))

    return '(' + ','.join(map(str, constraints)) + ')'


# =============================================================================
# SIGNAL-TO-NOISE RATIO (SNR) CONFIGURATION
# =============================================================================
# Key insight: 0.5 unit edge is 25% of std for receptions, but only 1.5% for yards
# This determines which markets are mathematically exploitable

@dataclass
class MarketSNRConfig:
    """Signal-to-Noise configuration for a market."""
    typical_std: float          # Standard deviation of outcomes
    line_increment: float       # Smallest line movement (usually 0.5)
    snr_pct: float              # line_increment / typical_std * 100
    tier: str                   # HIGH, MEDIUM, LOW
    min_edge_pct: float         # Minimum edge as % of std to bet
    confidence_threshold: float # Min model confidence to bet
    min_line_deviation: float   # Min |line - trailing| to consider
    enabled: bool = True        # Whether this market is recommended for betting
    reason_disabled: str = ""   # Why market is disabled (if enabled=False)


MARKET_SNR_CONFIG: Dict[str, MarketSNRConfig] = {
    # =========================================================================
    # HIGH SNR - Primary focus, discrete outcomes with low variance
    # =========================================================================
    'player_receptions': MarketSNRConfig(
        typical_std=2.4,
        line_increment=0.5,
        snr_pct=20.8,
        tier='HIGH',
        min_edge_pct=5.0,        # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5), was 0.58
        min_line_deviation=0.0,
    ),

    'player_rush_attempts': MarketSNRConfig(
        typical_std=4.0,
        line_increment=0.5,
        snr_pct=12.5,
        tier='HIGH',
        min_edge_pct=5.0,   # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5)
        min_line_deviation=0.0,
    ),

    'player_pass_completions': MarketSNRConfig(
        typical_std=5.0,
        line_increment=0.5,
        snr_pct=10.0,
        tier='HIGH',
        min_edge_pct=5.0,   # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5), was 0.60
        min_line_deviation=0.0,  # V30: Removed - not validated
        enabled=True,
    ),

    'player_pass_attempts': MarketSNRConfig(
        typical_std=6.0,
        line_increment=0.5,
        snr_pct=8.3,
        tier='HIGH',
        min_edge_pct=5.0,   # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5), was 0.70
        min_line_deviation=0.0,  # V30: Removed - not validated
        enabled=True,
    ),

    # =========================================================================
    # MEDIUM SNR - Yards markets with higher variance
    # V30: Holdout analysis shows confidence thresholds don't predict wins
    # Key signal is LVT direction (WITH Vegas = 58.7% vs AGAINST = 51.2%)
    # =========================================================================
    'player_rush_yds': MarketSNRConfig(
        typical_std=35.5,
        line_increment=0.5,
        snr_pct=1.4,
        tier='HIGH',
        min_edge_pct=5.0,   # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5), was 0.54
        min_line_deviation=0.0,
    ),

    'player_reception_yds': MarketSNRConfig(
        typical_std=34.2,
        line_increment=0.5,
        snr_pct=1.5,
        tier='HIGH',  # V30: Upgraded - holdout shows 51.6% but confidence filter was limiting volume
        min_edge_pct=5.0,   # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5), was 0.55
        min_line_deviation=0.0,
    ),

    'player_pass_yds': MarketSNRConfig(
        typical_std=82.8,
        line_increment=0.5,
        snr_pct=0.6,
        tier='MEDIUM',
        min_edge_pct=5.0,   # V30: Lowered - holdout shows confidence doesn't predict wins
        confidence_threshold=0.50,  # V30: Match holdout validation (clf>0.5), was 0.60
        min_line_deviation=0.0,  # V30: Removed - not validated
        enabled=True,
    ),

    # =========================================================================
    # BINARY MARKETS - TD props (special handling, high vig)
    # These are binary (Yes/No) so use probability thresholds only
    # =========================================================================
    'player_anytime_td': MarketSNRConfig(
        typical_std=0.5,         # Binary - 0 or 1
        line_increment=0.5,
        snr_pct=100.0,           # N/A for binary
        tier='MEDIUM',
        min_edge_pct=5.0,        # Lowered from 10% - TD lines are efficient
        confidence_threshold=0.65,
        min_line_deviation=0.0,  # No line deviation for binary
    ),

    'player_1st_td': MarketSNRConfig(
        typical_std=0.5,
        line_increment=0.5,
        snr_pct=100.0,
        tier='LOW',              # Very hard to predict
        min_edge_pct=15.0,
        confidence_threshold=0.70,
        min_line_deviation=0.0,
    ),

    'player_pass_tds': MarketSNRConfig(
        typical_std=1.0,
        line_increment=0.5,
        snr_pct=50.0,
        tier='MEDIUM',
        min_edge_pct=15.0,
        confidence_threshold=0.65,
        min_line_deviation=0.3,
        enabled=False,
        reason_disabled="Same starter/backup issue as other QB passing markets",
    ),

    'player_rush_tds': MarketSNRConfig(
        typical_std=0.6,
        line_increment=0.5,
        snr_pct=83.0,
        tier='LOW',              # Very volatile
        min_edge_pct=20.0,
        confidence_threshold=0.70,
        min_line_deviation=0.0,
    ),

    # =========================================================================
    # INTERCEPTIONS - Very noisy, rarely bet
    # =========================================================================
    'player_interceptions': MarketSNRConfig(
        typical_std=0.8,
        line_increment=0.5,
        snr_pct=62.5,
        tier='LOW',
        min_edge_pct=25.0,
        confidence_threshold=0.72,
        min_line_deviation=0.3,
    ),

    # Alias for pass interceptions (DraftKings naming)
    'player_pass_interceptions': MarketSNRConfig(
        typical_std=0.8,
        line_increment=0.5,
        snr_pct=62.5,
        tier='LOW',
        min_edge_pct=25.0,
        confidence_threshold=0.72,
        min_line_deviation=0.3,
    ),

    # =========================================================================
    # TD SCORER MARKETS - Binary, high variance
    # =========================================================================
    'player_last_td': MarketSNRConfig(
        typical_std=0.5,
        line_increment=0.5,
        snr_pct=100.0,
        tier='LOW',              # Even harder than 1st TD
        min_edge_pct=20.0,
        confidence_threshold=0.75,
        min_line_deviation=0.0,
    ),

    # =========================================================================
    # LONGEST PLAY MARKETS - Extremely high variance
    # Single big play determines outcome, nearly impossible to predict
    # =========================================================================
    'player_reception_longest': MarketSNRConfig(
        typical_std=15.0,        # Huge variance on longest play
        line_increment=0.5,
        snr_pct=3.3,
        tier='LOW',
        min_edge_pct=30.0,       # Need massive edge
        confidence_threshold=0.80,
        min_line_deviation=10.0,
    ),

    'player_rush_longest': MarketSNRConfig(
        typical_std=12.0,
        line_increment=0.5,
        snr_pct=4.2,
        tier='LOW',
        min_edge_pct=30.0,
        confidence_threshold=0.80,
        min_line_deviation=8.0,
    ),

    'player_pass_longest_completion': MarketSNRConfig(
        typical_std=18.0,
        line_increment=0.5,
        snr_pct=2.8,
        tier='LOW',
        min_edge_pct=30.0,
        confidence_threshold=0.80,
        min_line_deviation=12.0,
    ),

    # =========================================================================
    # COMBINED STATS - Sum of rush + receiving yards
    # =========================================================================
    'player_rush_reception_yds': MarketSNRConfig(
        typical_std=45.0,        # Combined variance
        line_increment=0.5,
        snr_pct=1.1,
        tier='LOW',
        min_edge_pct=25.0,
        confidence_threshold=0.72,
        min_line_deviation=10.0,
    ),

    # =========================================================================
    # KICKER MARKETS - Decent predictability
    # =========================================================================
    'player_field_goals': MarketSNRConfig(
        typical_std=0.9,
        line_increment=0.5,
        snr_pct=55.6,
        tier='MEDIUM',
        min_edge_pct=15.0,
        confidence_threshold=0.65,
        min_line_deviation=0.3,
    ),

    'player_kicking_points': MarketSNRConfig(
        typical_std=3.5,
        line_increment=0.5,
        snr_pct=14.3,
        tier='MEDIUM',
        min_edge_pct=15.0,
        confidence_threshold=0.65,
        min_line_deviation=1.0,
    ),

    'player_pats': MarketSNRConfig(
        typical_std=1.0,
        line_increment=0.5,
        snr_pct=50.0,
        tier='MEDIUM',
        min_edge_pct=15.0,
        confidence_threshold=0.65,
        min_line_deviation=0.3,
    ),

    # =========================================================================
    # DEFENSIVE MARKETS - Very noisy
    # =========================================================================
    'player_sacks': MarketSNRConfig(
        typical_std=0.8,
        line_increment=0.5,
        snr_pct=62.5,
        tier='LOW',
        min_edge_pct=25.0,
        confidence_threshold=0.72,
        min_line_deviation=0.3,
    ),

    'player_tackles_assists': MarketSNRConfig(
        typical_std=3.5,
        line_increment=0.5,
        snr_pct=14.3,
        tier='LOW',
        min_edge_pct=25.0,
        confidence_threshold=0.72,
        min_line_deviation=1.5,
    ),

    'player_solo_tackles': MarketSNRConfig(
        typical_std=2.5,
        line_increment=0.5,
        snr_pct=20.0,
        tier='LOW',
        min_edge_pct=25.0,
        confidence_threshold=0.72,
        min_line_deviation=1.0,
    ),
}


def get_market_snr_config(market: str) -> Optional[MarketSNRConfig]:
    """Get SNR configuration for a market."""
    return MARKET_SNR_CONFIG.get(market)


def get_market_tier(market: str) -> str:
    """Get SNR tier for a market (HIGH, MEDIUM, LOW)."""
    config = MARKET_SNR_CONFIG.get(market)
    return config.tier if config else 'LOW'


def get_confidence_threshold(market: str) -> float:
    """Get minimum confidence threshold for a market."""
    config = MARKET_SNR_CONFIG.get(market)
    return config.confidence_threshold if config else 0.70


def is_market_enabled(market: str) -> bool:
    """
    Check if a market is enabled for betting recommendations.

    Markets may be disabled if:
    - Model has no predictive power (correlation ~0)
    - High variance makes it unexploitable
    - Structural issues (e.g., can't distinguish starters from backups)
    """
    config = MARKET_SNR_CONFIG.get(market)
    if config is None:
        return True  # Unknown markets default to enabled
    return config.enabled


def get_market_disabled_reason(market: str) -> str:
    """Get the reason why a market is disabled (empty string if enabled)."""
    config = MARKET_SNR_CONFIG.get(market)
    if config is None:
        return ""
    if config.enabled:
        return ""
    return config.reason_disabled


def get_enabled_markets() -> List[str]:
    """Get list of all enabled markets."""
    return [
        market for market, config in MARKET_SNR_CONFIG.items()
        if config.enabled
    ]


def get_disabled_markets() -> Dict[str, str]:
    """Get dict of disabled markets and their reasons."""
    return {
        market: config.reason_disabled
        for market, config in MARKET_SNR_CONFIG.items()
        if not config.enabled
    }


def get_min_edge(market: str) -> float:
    """Get minimum edge (in stat units) required to bet a market."""
    config = MARKET_SNR_CONFIG.get(market)
    if config:
        return config.typical_std * config.min_edge_pct / 100.0
    return 5.0  # Default for unknown markets


def should_bet_market(
    market: str,
    model_confidence: float,
    line_vs_trailing: float,
) -> tuple:
    """
    Quick check if a bet passes SNR thresholds.

    Returns:
        (should_bet: bool, reason: str)
    """
    config = MARKET_SNR_CONFIG.get(market)
    if not config:
        return False, f"Unknown market: {market}"

    # Check confidence threshold
    if model_confidence < config.confidence_threshold:
        return False, f"Confidence {model_confidence:.1%} < {config.confidence_threshold:.1%}"

    # Check line deviation
    if abs(line_vs_trailing) < config.min_line_deviation:
        return False, f"Line deviation {abs(line_vs_trailing):.1f} < {config.min_line_deviation:.1f}"

    return True, f"PASS: {config.tier} tier, conf={model_confidence:.1%}"
