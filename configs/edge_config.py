"""
Edge Configuration - Separate Edge Pipelines

This module defines feature sets and thresholds for the two independent edge strategies:
1. LVT Edge: Statistical reversion (line vs trailing)
2. Player Bias Edge: Player-specific tendencies

These edges are INDEPENDENT and should NOT be mixed in a single model.
See: scripts/_archive/analysis/validate_two_edge_hypothesis.py for validation.
"""
from typing import Dict, List
from dataclasses import dataclass


# =============================================================================
# LVT EDGE CONFIGURATION
# =============================================================================
# The LVT (Line vs Trailing) edge captures statistical reversion.
# When lines diverge significantly from trailing performance, regression occurs.
# Target: 65-70% hit rate at low volume

LVT_FEATURES: List[str] = [
    # Primary reversion signal
    'line_vs_trailing',      # Core signal: (line - trailing) / trailing * 100
    'line_level',            # Raw line value (context)
    'line_in_sweet_spot',    # Gaussian decay based on optimal line ranges

    # Reversion amplifiers
    'LVT_in_sweet_spot',     # Interaction: LVT * sweet_spot
    'market_under_rate',     # Market regime (is this a reversion-friendly market?)

    # Minimal context
    'vegas_spread',          # Game script context
    'implied_team_total',    # Scoring environment

    # V23 Opponent Defense Context (Dec 2025)
    'opp_def_epa',           # Opponent defense EPA - strong D prevents reversion
    'has_opponent_context',  # Binary flag for data availability

    # V28 Situational Features (Dec 2025)
    'rest_days',             # Days since last game
    'elo_diff',              # Home Elo - Away Elo (team quality differential)
    'lvt_x_defense',         # Interaction: LVT signal * defense quality
]

# LVT feature count for validation
LVT_FEATURE_COUNT = len(LVT_FEATURES)  # 12 (was 7)


@dataclass(frozen=True)
class LVTThreshold:
    """Threshold configuration for LVT edge per market."""
    confidence: float      # Min P(UNDER) to trigger
    min_lvt: float         # Min |line_vs_trailing| to consider
    min_samples: int = 10  # Min training samples required


LVT_THRESHOLDS: Dict[str, LVTThreshold] = {
    # Updated thresholds for trained model (Dec 2025)
    # Trained model outputs ~52-62% confidences, not 70%+
    'player_receptions': LVTThreshold(
        confidence=0.55,   # Lowered: trained model outputs ~52-62%
        min_lvt=1.5,  # Line > trailing by 1.5 receptions
    ),
    'player_rush_yds': LVTThreshold(
        confidence=0.55,   # Lowered: trained model outputs ~52-62%
        min_lvt=8.0,  # Line > trailing by 8 yards
    ),
    'player_reception_yds': LVTThreshold(
        confidence=0.55,   # Lowered: trained model outputs ~52-62%
        min_lvt=15.0,      # Require larger deviation
    ),
    'player_rush_attempts': LVTThreshold(
        confidence=0.55,   # Lowered: trained model outputs ~52-62%
        min_lvt=2.0,  # Line > trailing by 2 attempts
    ),
    'player_pass_attempts': LVTThreshold(
        confidence=0.55,   # Lowered: trained model outputs ~52-62%
        min_lvt=2.0,  # Line > trailing by 2 attempts
    ),
    'player_pass_completions': LVTThreshold(
        confidence=0.55,   # Lowered: trained model outputs ~52-62%
        min_lvt=1.0,  # Line > trailing by 1 completion
    ),
}


# XGBoost parameters for LVT edge (conservative to prevent overfitting)
LVT_MODEL_PARAMS = {
    'n_estimators': 50,      # Fewer trees
    'max_depth': 3,          # Shallower trees
    'learning_rate': 0.05,   # Slower learning
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
}


# =============================================================================
# PLAYER BIAS EDGE CONFIGURATION
# =============================================================================
# The Player Bias edge captures persistent player tendencies.
# Some players consistently go over or under their lines.
# Target: 55-60% hit rate at higher volume

PLAYER_BIAS_FEATURES: List[str] = [
    # Player historical tendency (core signals)
    'player_under_rate',     # % of times player goes under (trailing N games)
    'player_bias',           # Average (actual - line) over history

    # Alignment with current bet
    'LVT_x_player_tendency', # Does LVT align with player tendency?
    'LVT_x_player_bias',     # Does LVT align with player bias direction?
    'player_market_aligned', # Player tendency vs market regime

    # Player usage/opportunity
    'target_share',          # Receiving opportunity (for receptions/rec yards)
    'snap_share',            # Playing time indicator
    'trailing_catch_rate',   # Efficiency metric

    # Player context
    'pos_rank',              # Depth chart position (1=WR1, 2=WR2, etc.)
    'is_starter',            # Binary starter flag

    # Regime context
    'market_bias_strength',  # How strong is market regime?

    # Current season bias
    'current_season_under_rate',  # % under this season only
    'season_games_played',        # Sample size for current season

    # V23 Opponent Defense Context (Dec 2025)
    'opp_pass_yds_def_vs_avg',  # Pass defense z-score (for pass/rec markets)
    'opp_rush_yds_def_vs_avg',  # Rush defense z-score (for rush markets)
    'opp_def_epa',              # Overall defense EPA

    # V28/V28.1 Situational Features (Dec 2025)
    'rest_days',                # Days since last game
    'injury_status_encoded',    # 0=None, 1=Quest, 2=Doubt, 3=Out
    'has_injury_designation',   # Binary flag for injury data availability
]

# Player bias feature count for validation
PLAYER_BIAS_FEATURE_COUNT = len(PLAYER_BIAS_FEATURES)  # 19 (was 13)


@dataclass(frozen=True)
class PlayerBiasThreshold:
    """Threshold configuration for Player Bias edge per market."""
    confidence: float      # Min P(UNDER) to trigger
    min_bets: int          # Min historical bets required for player
    min_rate: float        # Min player_under_rate to consider (or 1 - min_rate for OVER)


PLAYER_BIAS_THRESHOLDS: Dict[str, PlayerBiasThreshold] = {
    # Optimized thresholds from walk-forward validation (Max ROI strategy)
    'player_receptions': PlayerBiasThreshold(
        confidence=0.70,    # Optimized: +13.4% ROI at 70%
        min_bets=15,        # More history = more reliable
        min_rate=0.65,
    ),
    'player_rush_yds': PlayerBiasThreshold(
        confidence=0.70,    # Optimized: +46.2% ROI at 70%
        min_bets=10,
        min_rate=0.65,
    ),
    'player_reception_yds': PlayerBiasThreshold(
        confidence=0.70,    # Optimized: +44.3% ROI at 70%
        min_bets=15,
        min_rate=0.65,
    ),
    'player_rush_attempts': PlayerBiasThreshold(
        confidence=0.70,    # Optimized: +48.1% ROI at 70%
        min_bets=10,
        min_rate=0.65,
    ),
    'player_pass_attempts': PlayerBiasThreshold(
        confidence=0.70,    # Optimized: +16.7% ROI at top 10% (Dec 2025 backtest)
        min_bets=10,
        min_rate=0.60,
    ),
    'player_pass_completions': PlayerBiasThreshold(
        confidence=0.60,    # Optimized: +6.7% ROI at top 40% (Dec 2025 backtest)
        min_bets=10,
        min_rate=0.60,
    ),
}


# XGBoost parameters for Player Bias edge (standard params)
PLAYER_BIAS_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
}


# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

# Markets supported by edge system
EDGE_MARKETS: List[str] = [
    'player_receptions',
    'player_rush_yds',
    'player_reception_yds',
    'player_rush_attempts',
    'player_pass_attempts',  # 58.3% UNDER (uses 50% threshold - exception)
    'player_pass_completions',  # NEW - testing (55.4% UNDER historical)
    # player_pass_yds REMOVED - -14.1% ROI in walk-forward
]

# EWMA span for trailing calculations
EDGE_EWMA_SPAN = 6

# Minimum samples for training (per market)
MIN_TRAINING_SAMPLES = 100


# =============================================================================
# TD POISSON EDGE CONFIGURATION
# =============================================================================
# TD props use Poisson regression (not XGBoost) because TDs are count data.
# Distribution: 34% zero, 29% one, 23% two (Poisson-shaped)
# Target: 55-60% hit rate with proper probability thresholds

TD_POISSON_MARKETS: List[str] = [
    'player_pass_tds',
    'player_rush_tds',
    'player_rec_tds',
]


@dataclass(frozen=True)
class TDPoissonThreshold:
    """Threshold configuration for TD Poisson edge per market."""
    min_confidence: float   # Min P(OVER) or P(UNDER) to trigger bet
    min_samples: int = 50   # Min training samples required


TD_POISSON_THRESHOLDS: Dict[str, TDPoissonThreshold] = {
    'player_pass_tds': TDPoissonThreshold(
        min_confidence=0.58,  # QB pass TDs - moderate confidence
    ),
    'player_rush_tds': TDPoissonThreshold(
        min_confidence=0.60,  # Rush TDs - higher (more volatile)
    ),
    'player_rec_tds': TDPoissonThreshold(
        min_confidence=0.60,  # Rec TDs - higher (more volatile)
    ),
}


# =============================================================================
# ATTD ENSEMBLE CONFIGURATION
# =============================================================================

@dataclass
class ATTDThreshold:
    """Threshold configuration for ATTD ensemble edge."""
    min_confidence: float = 0.55  # Minimum probability to recommend
    min_games: int = 4            # Minimum games for player sample (skip low-data players)
    min_edge: float = 0.05        # Minimum edge vs implied odds


ATTD_THRESHOLDS: Dict[str, ATTDThreshold] = {
    'QB': ATTDThreshold(
        min_confidence=0.60,  # Higher bar for QBs (lower TD rate)
        min_games=4,
        min_edge=0.08,
    ),
    'RB': ATTDThreshold(
        min_confidence=0.55,
        min_games=4,
        min_edge=0.05,
    ),
    'WR': ATTDThreshold(
        min_confidence=0.55,
        min_games=4,
        min_edge=0.05,
    ),
    'TE': ATTDThreshold(
        min_confidence=0.55,
        min_games=4,
        min_edge=0.05,
    ),
}


def get_attd_threshold(position: str) -> ATTDThreshold:
    """Get ATTD threshold for a position."""
    if position not in ATTD_THRESHOLDS:
        return ATTDThreshold()  # Default
    return ATTD_THRESHOLDS[position]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_td_poisson_threshold(market: str) -> TDPoissonThreshold:
    """Get TD Poisson threshold for a market."""
    if market not in TD_POISSON_THRESHOLDS:
        raise ValueError(f"Market {market} not supported for TD Poisson edge")
    return TD_POISSON_THRESHOLDS[market]


def get_lvt_threshold(market: str) -> LVTThreshold:
    """Get LVT threshold for a market."""
    if market not in LVT_THRESHOLDS:
        raise ValueError(f"Market {market} not supported for LVT edge")
    return LVT_THRESHOLDS[market]


def get_player_bias_threshold(market: str) -> PlayerBiasThreshold:
    """Get Player Bias threshold for a market."""
    if market not in PLAYER_BIAS_THRESHOLDS:
        raise ValueError(f"Market {market} not supported for Player Bias edge")
    return PLAYER_BIAS_THRESHOLDS[market]


def is_market_supported(market: str) -> bool:
    """Check if a market is supported by the edge system."""
    return market in EDGE_MARKETS


# =============================================================================
# DATA QUALITY REQUIREMENTS (NO DATA = NO BET)
# =============================================================================
# These define the MINIMUM data required to make a bet.
# If any required field is missing/NaN, we skip the bet entirely.

# Map market to required trailing stat column
MARKET_TRAILING_REQUIREMENTS: Dict[str, str] = {
    'player_receptions': 'trailing_receptions',
    'player_rush_yds': 'trailing_rushing_yards',
    'player_reception_yds': 'trailing_receiving_yards',
    'player_rush_attempts': 'trailing_carries',
    'player_pass_attempts': 'trailing_pass_attempts',
    'player_pass_completions': 'trailing_completions',
}

# Minimum games required for player history (used for player_under_rate)
MIN_PLAYER_GAMES = 4

# Required fields for LVT edge (must have real data, not defaults)
LVT_REQUIRED_FIELDS: List[str] = [
    # Trailing stat is checked per-market via MARKET_TRAILING_REQUIREMENTS
    'line',  # Need the betting line
]

# Required fields for Player Bias edge
PLAYER_BIAS_REQUIRED_FIELDS: List[str] = [
    'player_under_rate',  # Must have player history
    'player_bet_count',   # Must meet min games threshold
]


@dataclass
class DataQualityResult:
    """Result of data quality check."""
    has_required_data: bool
    missing_fields: List[str]
    reason: str


def check_data_quality(
    row: 'pd.Series',
    market: str,
    edge_type: str = 'both',
) -> DataQualityResult:
    """
    Check if a row has sufficient data quality for betting.

    Args:
        row: Series with bet features
        market: Market being evaluated
        edge_type: 'lvt', 'player_bias', or 'both'

    Returns:
        DataQualityResult with pass/fail and reason
    """
    import pandas as pd
    missing = []

    # Check trailing stat for LVT edge
    if edge_type in ['lvt', 'both']:
        trailing_col = MARKET_TRAILING_REQUIREMENTS.get(market)
        if trailing_col:
            value = row.get(trailing_col)
            if pd.isna(value):
                missing.append(f"{trailing_col} (no trailing stats)")

        # Check other LVT required fields
        for field in LVT_REQUIRED_FIELDS:
            value = row.get(field)
            if pd.isna(value):
                missing.append(field)

    # Check player history for Player Bias edge
    if edge_type in ['player_bias', 'both']:
        # Check player_under_rate exists
        under_rate = row.get('player_under_rate')
        if pd.isna(under_rate):
            missing.append('player_under_rate (no player history)')

        # Check minimum games
        bet_count = row.get('player_bet_count', 0)
        if pd.isna(bet_count) or bet_count < MIN_PLAYER_GAMES:
            missing.append(f'player_bet_count (need {MIN_PLAYER_GAMES}+ games, have {int(bet_count) if not pd.isna(bet_count) else 0})')

    if missing:
        return DataQualityResult(
            has_required_data=False,
            missing_fields=missing,
            reason=f"Missing required data: {', '.join(missing)}"
        )

    return DataQualityResult(
        has_required_data=True,
        missing_fields=[],
        reason="Data quality OK"
    )
