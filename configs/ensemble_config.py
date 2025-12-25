"""
Ensemble Configuration - Betting Rules and Unit Sizing

This module defines how the LVT and Player Bias edges are combined
for final betting decisions.

Ensemble Rules:
- BOTH AGREE: Highest conviction (2 units)
- LVT ONLY: High conviction (1.5 units)
- PLAYER BIAS ONLY: Moderate conviction (1 unit)
- CONFLICT: No bet (edges disagree on direction)
- NEITHER: No bet (no edge detected)
"""
from typing import Dict
from dataclasses import dataclass
from enum import Enum


class EdgeSource(Enum):
    """Source of betting signal."""
    BOTH = "BOTH"              # Both edges agree
    LVT_ONLY = "LVT_ONLY"      # Only LVT edge triggers
    PLAYER_BIAS_ONLY = "PLAYER_BIAS_ONLY"  # Only Player Bias triggers
    CONFLICT = "CONFLICT"      # Edges disagree on direction
    NEITHER = "NEITHER"        # No edge triggers
    NO_DATA = "NO_DATA"        # Insufficient data to bet (skip)


@dataclass(frozen=True)
class MarketEnsembleConfig:
    """Ensemble configuration per market."""
    # Weighting for confidence calculation
    lvt_weight: float          # Weight for LVT edge in combined confidence
    player_bias_weight: float  # Weight for Player Bias edge

    # Unit sizing by edge source
    both_agree_units: float    # Units when both edges agree
    lvt_only_units: float      # Units when only LVT triggers
    player_bias_only_units: float  # Units when only Player Bias triggers

    # Risk management
    max_daily_bets: int        # Maximum bets per day for this market
    max_exposure_units: float  # Maximum total units exposed per day


MARKET_ENSEMBLE_CONFIG: Dict[str, MarketEnsembleConfig] = {
    'player_receptions': MarketEnsembleConfig(
        lvt_weight=0.6,
        player_bias_weight=0.4,
        both_agree_units=2.0,
        lvt_only_units=1.5,
        player_bias_only_units=1.0,
        max_daily_bets=10,
        max_exposure_units=15.0,
    ),

    'player_rush_yds': MarketEnsembleConfig(
        lvt_weight=0.7,          # LVT stronger for rushing
        player_bias_weight=0.3,
        both_agree_units=2.0,
        lvt_only_units=1.5,
        player_bias_only_units=0.75,  # Lower confidence in player bias for yards
        max_daily_bets=8,
        max_exposure_units=12.0,
    ),

    'player_reception_yds': MarketEnsembleConfig(
        lvt_weight=0.5,          # More balanced
        player_bias_weight=0.5,
        both_agree_units=1.5,    # More conservative on yards
        lvt_only_units=1.0,
        player_bias_only_units=0.75,
        max_daily_bets=8,
        max_exposure_units=10.0,
    ),

    'player_rush_attempts': MarketEnsembleConfig(
        lvt_weight=0.55,
        player_bias_weight=0.45,
        both_agree_units=2.0,
        lvt_only_units=1.5,
        player_bias_only_units=1.0,
        max_daily_bets=6,
        max_exposure_units=10.0,
    ),

    'player_pass_attempts': MarketEnsembleConfig(
        lvt_weight=0.5,          # Balanced - both edges contribute equally
        player_bias_weight=0.5,
        both_agree_units=1.5,    # Conservative - newly enabled market
        lvt_only_units=1.0,
        player_bias_only_units=0.75,
        max_daily_bets=8,
        max_exposure_units=10.0,
    ),

    'player_pass_completions': MarketEnsembleConfig(
        lvt_weight=0.5,          # Balanced - both edges contribute equally
        player_bias_weight=0.5,
        both_agree_units=1.5,    # Conservative - newly enabled market
        lvt_only_units=1.0,
        player_bias_only_units=0.75,
        max_daily_bets=8,
        max_exposure_units=10.0,
    ),
}


# =============================================================================
# GLOBAL ENSEMBLE SETTINGS
# =============================================================================

@dataclass(frozen=True)
class GlobalEnsembleSettings:
    """Global settings for the ensemble system."""
    # Unit sizing
    base_unit_size: float = 5.0  # Dollar value per unit

    # Minimum confidence to trigger
    min_lvt_confidence: float = 0.60
    min_player_bias_confidence: float = 0.52

    # Conflict handling
    conflict_buffer: float = 0.05  # Ignore conflict if one edge barely triggers

    # Direction constraints - now market-specific (see MARKET_DIRECTION_CONSTRAINTS in model_config.py)
    enforce_under_only: bool = False  # Disabled - use MARKET_DIRECTION_CONSTRAINTS instead

    # Daily limits
    max_total_daily_bets: int = 25
    max_total_daily_units: float = 40.0


GLOBAL_SETTINGS = GlobalEnsembleSettings()


# =============================================================================
# ENSEMBLE DECISION LOGIC
# =============================================================================

@dataclass
class EnsembleDecision:
    """Result of ensemble evaluation for a single bet."""
    should_bet: bool
    direction: str  # "UNDER", "OVER", or None
    units: float
    source: EdgeSource
    lvt_confidence: float
    player_bias_confidence: float
    combined_confidence: float
    reasoning: str


def get_units_for_source(market: str, source: EdgeSource) -> float:
    """Get unit sizing based on edge source and market."""
    if source in (EdgeSource.CONFLICT, EdgeSource.NEITHER):
        return 0.0

    config = MARKET_ENSEMBLE_CONFIG.get(market)
    if not config:
        return 0.0

    if source == EdgeSource.BOTH:
        return config.both_agree_units
    elif source == EdgeSource.LVT_ONLY:
        return config.lvt_only_units
    elif source == EdgeSource.PLAYER_BIAS_ONLY:
        return config.player_bias_only_units

    return 0.0


def get_combined_confidence(
    market: str,
    lvt_confidence: float,
    player_bias_confidence: float,
    source: EdgeSource
) -> float:
    """Calculate combined confidence based on edge weights."""
    config = MARKET_ENSEMBLE_CONFIG.get(market)
    if not config:
        return max(lvt_confidence, player_bias_confidence)

    if source == EdgeSource.BOTH:
        # Weighted average when both agree
        return (
            config.lvt_weight * lvt_confidence +
            config.player_bias_weight * player_bias_confidence
        )
    elif source == EdgeSource.LVT_ONLY:
        return lvt_confidence
    elif source == EdgeSource.PLAYER_BIAS_ONLY:
        return player_bias_confidence

    return 0.0


def get_market_config(market: str) -> MarketEnsembleConfig:
    """Get ensemble config for a market."""
    if market not in MARKET_ENSEMBLE_CONFIG:
        raise ValueError(f"Market {market} not supported for ensemble")
    return MARKET_ENSEMBLE_CONFIG[market]


# =============================================================================
# VERSION INFO
# =============================================================================

ENSEMBLE_VERSION = "edge_v1"
