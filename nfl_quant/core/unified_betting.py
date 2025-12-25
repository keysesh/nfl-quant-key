#!/usr/bin/env python3
"""
NFL QUANT - Unified Betting Calculations

Standardizes edge calculation, Kelly criterion, and confidence tiering
across ALL bet types (player props and game lines).

This module ensures consistency throughout the entire pipeline.
"""

from enum import Enum
from typing import Optional
import numpy as np

from nfl_quant.betting.kelly_criterion import (
    calculate_kelly_fraction as _canonical_kelly,
    calculate_fractional_kelly as _canonical_fractional_kelly,
    american_to_decimal,
)


class ConfidenceTier(str, Enum):
    """Unified confidence tiers across all bet types."""
    ELITE = "ELITE"      # Top picks with exceptional edge
    HIGH = "HIGH"        # Strong picks with solid edge
    STANDARD = "STANDARD"  # Normal picks meeting minimum criteria
    LOW = "LOW"          # Below threshold, use caution


def american_odds_to_implied_prob(american_odds: float) -> float:
    """
    Convert American odds to implied probability (with vig included).

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0.0 to 1.0)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig_two_way(prob_side1: float, prob_side2: float) -> tuple[float, float]:
    """
    Remove vig from two-way market (e.g., over/under, spread).

    Args:
        prob_side1: Implied probability of side 1
        prob_side2: Implied probability of side 2

    Returns:
        Tuple of (fair_prob_side1, fair_prob_side2)
    """
    total = prob_side1 + prob_side2
    if total <= 0:
        return prob_side1, prob_side2

    fair_prob_side1 = prob_side1 / total
    fair_prob_side2 = prob_side2 / total

    return fair_prob_side1, fair_prob_side2


def calculate_edge(model_prob: float, market_implied_prob: float) -> float:
    """
    Calculate edge as the difference between model probability and market probability.

    THIS IS THE STANDARD EDGE CALCULATION FOR ALL BET TYPES.

    Args:
        model_prob: Model's probability of the bet winning (0.0 to 1.0)
        market_implied_prob: Market implied probability (0.0 to 1.0)

    Returns:
        Edge as decimal (e.g., 0.15 = 15% edge)
    """
    return model_prob - market_implied_prob


def calculate_edge_percentage(model_prob: float, market_implied_prob: float) -> float:
    """
    Calculate edge as a percentage.

    Args:
        model_prob: Model's probability of the bet winning (0.0 to 1.0)
        market_implied_prob: Market implied probability (0.0 to 1.0)

    Returns:
        Edge as percentage (e.g., 15.0 = 15% edge)
    """
    return calculate_edge(model_prob, market_implied_prob) * 100


def american_to_decimal_odds(american_odds: float) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 1.91, 2.50)
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_kelly_fraction(
    model_prob: float,
    american_odds: float,
    fractional: float = 0.25
) -> float:
    """
    Calculate Kelly optimal bet sizing.

    Uses quarter Kelly (0.25) by default for risk management.
    Delegates to canonical implementation in nfl_quant.betting.kelly_criterion.

    Args:
        model_prob: Model's probability of winning (0.0 to 1.0)
        american_odds: American odds for the bet
        fractional: Kelly fraction (default 0.25 = quarter Kelly)

    Returns:
        Fraction of bankroll to bet (0.0 to 1.0)
    """
    if model_prob <= 0 or model_prob >= 1:
        return 0.0

    # Delegate to canonical implementation
    kelly = _canonical_fractional_kelly(model_prob, american_odds, fractional)

    # Cap at 10% of bankroll for safety (unified_betting specific limit)
    return min(kelly, 0.10)


def calculate_recommended_units(
    kelly_fraction: float,
    base_units: float = 100.0
) -> float:
    """
    Convert Kelly fraction to recommended bet units.

    Args:
        kelly_fraction: Kelly fraction (0.0 to 1.0)
        base_units: Base unit size (default 100 units = 1% of bankroll each)

    Returns:
        Recommended units to bet
    """
    return round(kelly_fraction * base_units, 1)


def assign_confidence_tier(
    edge_pct: float,
    model_prob: float,
    bet_type: str = "player_prop"
) -> ConfidenceTier:
    """
    Assign confidence tier based purely on model probability (confidence).

    Tiers are based on how confident the model is in its prediction,
    NOT on edge percentage (which can be misleading).

    Args:
        edge_pct: Edge percentage (kept for backwards compatibility, not used)
        model_prob: Model probability (0.0 to 1.0) - PRIMARY SIGNAL
        bet_type: Type of bet (kept for backwards compatibility, not used)

    Returns:
        ConfidenceTier enum value
    """
    # Confidence-based tiering (model_prob is the only signal that matters)
    # ELITE: Very high confidence (70%+)
    if model_prob >= 0.70:
        return ConfidenceTier.ELITE

    # HIGH: High confidence (60-70%)
    if model_prob >= 0.60:
        return ConfidenceTier.HIGH

    # STANDARD: Moderate confidence (55-60%)
    if model_prob >= 0.55:
        return ConfidenceTier.STANDARD

    # LOW: Below actionable threshold (<55%)
    return ConfidenceTier.LOW


def calculate_expected_roi(edge_pct: float, american_odds: float = -110) -> float:
    """
    Calculate expected ROI as: E[ROI] = (model_prob * payout) - 1

    For a fair bet, expected ROI = 0.
    For a +EV bet, expected ROI > 0.

    Args:
        edge_pct: Edge percentage (e.g., 5.0 means 5% edge over fair market prob)
        american_odds: American odds for the bet

    Returns:
        Expected ROI percentage

    Example:
        - Model prob 55%, market prob 50% (5% edge), odds -110
        - Decimal odds = 1.909
        - E[ROI] = 0.55 * 1.909 - 1 = 0.05 = 5%
    """
    # Convert edge_pct back to probability difference
    # edge_pct = (model_prob - market_prob) * 100
    # We need model_prob to calculate expected value
    # Since we don't have it directly, use the approximation:
    # E[ROI] ≈ edge_pct * (decimal_odds - 1) / decimal_odds for symmetric bets
    #
    # More accurate formula using decimal odds:
    decimal_odds = american_to_decimal_odds(american_odds)

    # Expected ROI = edge * (potential_profit / stake)
    # For unit stake, potential_profit = decimal_odds - 1
    # E[ROI] ≈ edge * (decimal_odds - 1) when bet is close to fair
    # This is an approximation that works well for small edges

    return edge_pct * (decimal_odds - 1)


def select_best_side(
    prob_side1: float,
    prob_side2: float,
    odds_side1: float,
    odds_side2: float,
    name_side1: str,
    name_side2: str
) -> dict:
    """
    Select the best side to bet based on edge.

    Args:
        prob_side1: Model probability for side 1
        prob_side2: Model probability for side 2 (should be ~1 - prob_side1)
        odds_side1: American odds for side 1
        odds_side2: American odds for side 2
        name_side1: Name of side 1 (e.g., "OVER", "HOME -3.5")
        name_side2: Name of side 2 (e.g., "UNDER", "AWAY +3.5")

    Returns:
        Dict with selected side information
    """
    # Calculate market implied probabilities
    market_prob1 = american_odds_to_implied_prob(odds_side1)
    market_prob2 = american_odds_to_implied_prob(odds_side2)

    # Remove vig to get fair market probabilities
    fair_market_prob1, fair_market_prob2 = remove_vig_two_way(market_prob1, market_prob2)

    # Calculate edges using fair market probabilities
    edge1 = calculate_edge_percentage(prob_side1, fair_market_prob1)
    edge2 = calculate_edge_percentage(prob_side2, fair_market_prob2)

    # Select side with higher edge
    if edge1 >= edge2:
        selected_prob = prob_side1
        selected_market_prob = fair_market_prob1
        selected_odds = odds_side1
        selected_name = name_side1
        selected_edge = edge1
    else:
        selected_prob = prob_side2
        selected_market_prob = fair_market_prob2
        selected_odds = odds_side2
        selected_name = name_side2
        selected_edge = edge2

    return {
        'pick': selected_name,
        'model_prob': selected_prob,
        'market_prob': selected_market_prob,
        'edge_pct': selected_edge,
        'american_odds': selected_odds,
    }


def create_unified_bet_output(
    bet_id: str,
    bet_type: str,
    pick: str,
    model_prob: float,
    market_prob: float,
    american_odds: float,
    model_projection: Optional[float] = None,
    market_line: Optional[float] = None,
    game: str = "",
    player: str = "",
    team: str = "",
    position: str = "",
    fractional_kelly: float = 0.25
) -> dict:
    """
    Create a unified bet output with all calculated fields.

    This ensures consistent output schema across all bet types.

    Args:
        bet_id: Unique identifier for the bet
        bet_type: Type of bet (player_prop, spread, total, moneyline)
        pick: Selected pick (e.g., "OVER 5.5", "KC -3.5")
        model_prob: Model's probability
        market_prob: Fair market probability (vig removed)
        american_odds: American odds for the bet
        model_projection: Model's projected value (for props/totals)
        market_line: Market line (for props/spreads/totals)
        game: Game string (e.g., "BUF @ KC")
        player: Player name (for props)
        team: Team abbreviation
        position: Position (for props)
        fractional_kelly: Kelly fraction to use

    Returns:
        Unified bet output dictionary
    """
    # Core calculations using unified methods
    edge_pct = calculate_edge_percentage(model_prob, market_prob)
    kelly_fraction = calculate_kelly_fraction(model_prob, american_odds, fractional_kelly)
    recommended_units = calculate_recommended_units(kelly_fraction)
    confidence_tier = assign_confidence_tier(edge_pct, model_prob, bet_type)
    expected_roi = calculate_expected_roi(edge_pct, american_odds)

    return {
        # Identifiers
        'bet_id': bet_id,
        'bet_type': bet_type,
        'game': game,
        'player': player,
        'team': team,
        'position': position,

        # Pick information
        'pick': pick,
        'market_line': market_line,
        'model_projection': model_projection,

        # Probabilities (STANDARDIZED)
        'model_prob': round(model_prob, 4),
        'market_prob': round(market_prob, 4),

        # Edge metrics (STANDARDIZED)
        'edge_pct': round(edge_pct, 2),
        'expected_roi': round(expected_roi, 2),

        # Bet sizing (STANDARDIZED)
        'american_odds': american_odds,
        'kelly_fraction': round(kelly_fraction, 4),
        'recommended_units': recommended_units,

        # Classification (STANDARDIZED)
        'confidence_tier': confidence_tier.value,
    }
