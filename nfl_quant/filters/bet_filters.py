"""
Bet Filtering Module

Based on walk-forward recalibration analysis (2025-12-07):
- High confidence (>60%) = +4.5% ROI
- Receptions market, lines 3-5 = +3.3% ROI
- OVER bets are fundamentally broken (-12.6% ROI)

This module implements hard filters to only take profitable bet types.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BetFilterConfig:
    """Configuration for bet filtering."""

    # Confidence thresholds
    min_confidence: float = 0.60  # Only take bets with >60% model confidence

    # Market restrictions
    allowed_markets: tuple = ('player_receptions',)  # Only receptions market

    # Line restrictions for receptions
    receptions_min_line: float = 2.5
    receptions_max_line: float = 5.5

    # Volatility threshold (skip high-variance players)
    max_player_cv: float = 0.50  # Coefficient of variation threshold

    # Side restrictions
    allow_over: bool = False  # OVER bets are -12.6% ROI, disabled by default
    allow_under: bool = True

    # OVER-specific confidence range (backtest showed 60-65% is profitable for OVER)
    over_min_confidence: Optional[float] = None  # If set, OVER requires this minimum
    over_max_confidence: Optional[float] = None  # If set, OVER requires this maximum

    # Game context
    min_game_total: Optional[float] = 40.0  # Skip low-scoring game projections
    max_game_total: Optional[float] = 55.0  # Skip shootout projections (too volatile)


def should_take_bet(
    market: str,
    line: float,
    confidence: float,
    side: str,
    player_cv: Optional[float] = None,
    game_total: Optional[float] = None,
    config: Optional[BetFilterConfig] = None
) -> tuple[bool, str]:
    """
    Determine if a bet should be taken based on filter criteria.

    Args:
        market: Market type (e.g., 'player_receptions')
        line: The betting line
        confidence: Model confidence (0-1)
        side: 'OVER' or 'UNDER'
        player_cv: Player's coefficient of variation (std/mean)
        game_total: Vegas game total
        config: Filter configuration

    Returns:
        Tuple of (should_take: bool, reason: str)
    """
    if config is None:
        config = BetFilterConfig()

    # 1. Confidence filter
    if confidence < config.min_confidence:
        return False, f"Low confidence ({confidence:.1%} < {config.min_confidence:.0%})"

    # 2. Market filter
    if market not in config.allowed_markets:
        return False, f"Market not allowed ({market})"

    # 3. Side filter
    if side == 'OVER':
        if not config.allow_over:
            return False, "OVER bets disabled"
        # Check OVER-specific confidence range if configured
        if config.over_min_confidence is not None and confidence < config.over_min_confidence:
            return False, f"OVER confidence too low ({confidence:.1%} < {config.over_min_confidence:.0%})"
        if config.over_max_confidence is not None and confidence >= config.over_max_confidence:
            return False, f"OVER confidence too high ({confidence:.1%} >= {config.over_max_confidence:.0%})"
    if side == 'UNDER' and not config.allow_under:
        return False, "UNDER bets disabled"

    # 4. Line restrictions (market-specific)
    if market == 'player_receptions':
        if line < config.receptions_min_line:
            return False, f"Line too low ({line} < {config.receptions_min_line})"
        if line > config.receptions_max_line:
            return False, f"Line too high ({line} > {config.receptions_max_line})"

    # 5. Volatility filter
    if player_cv is not None and player_cv > config.max_player_cv:
        return False, f"Player too volatile (CV={player_cv:.2f} > {config.max_player_cv})"

    # 6. Game total filter
    if game_total is not None:
        if config.min_game_total and game_total < config.min_game_total:
            return False, f"Game total too low ({game_total} < {config.min_game_total})"
        if config.max_game_total and game_total > config.max_game_total:
            return False, f"Game total too high ({game_total} > {config.max_game_total})"

    return True, "Passed all filters"


def calculate_player_volatility(trailing_values: list) -> Optional[float]:
    """
    Calculate coefficient of variation for a player's recent stats.

    Args:
        trailing_values: List of recent stat values (at least 3 required)

    Returns:
        Coefficient of variation (std/mean), or None if insufficient data
    """
    if len(trailing_values) < 3:
        return None

    values = np.array(trailing_values)
    mean = np.mean(values)

    if mean <= 0:
        return None

    std = np.std(values)
    return std / mean


def get_dynamic_confidence_threshold(player_cv: float, base_threshold: float = 0.60) -> float:
    """
    Adjust confidence threshold based on player volatility.

    High-variance players require higher confidence to bet.

    Args:
        player_cv: Player's coefficient of variation
        base_threshold: Base confidence threshold

    Returns:
        Adjusted confidence threshold
    """
    # Linear adjustment: +10% threshold for each 0.1 CV above 0.3
    cv_adjustment = max(0, (player_cv - 0.30) * 1.0)
    adjusted = base_threshold + cv_adjustment

    # Cap at 80%
    return min(0.80, adjusted)


# Pre-configured filter profiles
# Based on walk-forward recalibration analysis (2025-12-07)

# ELITE: Lines 0-3 only - 81.2% win rate, +55.1% ROI (n=16)
# Role players with defined scheme roles
ELITE_FILTER = BetFilterConfig(
    min_confidence=0.60,
    allowed_markets=('player_receptions',),
    receptions_min_line=0.5,
    receptions_max_line=3.0,
    max_player_cv=0.50,
    allow_over=False,
    allow_under=True,
)

# CONSERVATIVE: >50% confidence, all XGBoost classifier markets
# Updated 2025-12-08: Added player_reception_yds to allowed markets
# Updated 2025-12-08: Relaxed volatility filter from 0.50 to 0.65
# Updated 2025-12-08: Lowered confidence threshold from 60% to 50%
# Updated 2025-12-08: Enabled OVER bets in 60-65% confidence range (+5.5% ROI in backtest)
# Updated 2025-12-28: Aligned with CLASSIFIER_MARKETS from model_config.py
# - pass_attempts and pass_completions: RE-ENABLED (Dec 2025)
# - pass_yds: EXCLUDED (-15.8% ROI in holdout, failing both directions)
CONSERVATIVE_FILTER = BetFilterConfig(
    min_confidence=0.50,
    allowed_markets=(
        'player_receptions',
        'player_reception_yds',
        # player_rush_yds DISABLED for now
        'player_rush_attempts',
        'player_pass_attempts',      # Re-enabled Dec 2025 per model_config.py
        'player_pass_completions',   # Re-enabled Dec 2025 per model_config.py
        # player_pass_yds EXCLUDED: -15.8% ROI in holdout
    ),
    receptions_min_line=0.5,
    receptions_max_line=15.0,  # Effectively no line restriction
    max_player_cv=0.65,
    allow_over=True,  # Enabled with confidence range restriction
    allow_under=True,
    over_min_confidence=0.60,  # OVER only profitable at 60-65%
    over_max_confidence=0.65,  # Cap at 65% (higher confidence OVER bets lose money)
)

# MODERATE: Receptions only, slightly lower confidence
MODERATE_FILTER = BetFilterConfig(
    min_confidence=0.58,
    allowed_markets=('player_receptions',),
    receptions_min_line=0.5,
    receptions_max_line=15.0,
    max_player_cv=0.55,
    allow_over=False,
    allow_under=True,
)

# AGGRESSIVE: Lower thresholds (NOT RECOMMENDED based on backtest)
AGGRESSIVE_FILTER = BetFilterConfig(
    min_confidence=0.55,
    allowed_markets=('player_receptions', 'player_reception_yds'),
    receptions_min_line=0.5,
    receptions_max_line=15.0,
    max_player_cv=0.60,
    allow_over=False,
    allow_under=True,
)
