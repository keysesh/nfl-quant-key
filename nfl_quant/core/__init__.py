"""
NFL QUANT Core Module

Provides unified betting calculations and standardized interfaces
for all bet types (player props and game lines).
"""

from .unified_betting import (
    ConfidenceTier,
    american_odds_to_implied_prob,
    remove_vig_two_way,
    calculate_edge,
    calculate_edge_percentage,
    american_to_decimal_odds,
    calculate_kelly_fraction,
    calculate_recommended_units,
    assign_confidence_tier,
    calculate_expected_roi,
    select_best_side,
    create_unified_bet_output,
)

__all__ = [
    'ConfidenceTier',
    'american_odds_to_implied_prob',
    'remove_vig_two_way',
    'calculate_edge',
    'calculate_edge_percentage',
    'american_to_decimal_odds',
    'calculate_kelly_fraction',
    'calculate_recommended_units',
    'assign_confidence_tier',
    'calculate_expected_roi',
    'select_best_side',
    'create_unified_bet_output',
]
