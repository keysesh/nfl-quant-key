"""
NFL QUANT Betting Module

Sportsbook-specific betting rules and prop market logic.
Currently implements DraftKings rules and Kelly Criterion sizing.
"""

from nfl_quant.betting.draftkings_prop_rules import (
    DraftKingsAnytimeTDRules,
    DraftKingsPropActionRules,
    DraftKingsQBTDMarkets,
    calculate_anytime_td_probability,
)

from nfl_quant.betting.kelly_criterion import (
    calculate_kelly_fraction,
    calculate_fractional_kelly,
    calculate_kelly_with_limits,
    optimal_bet_size,
    simulate_kelly_performance,
)

__all__ = [
    'DraftKingsAnytimeTDRules',
    'DraftKingsPropActionRules',
    'DraftKingsQBTDMarkets',
    'calculate_anytime_td_probability',
    'calculate_kelly_fraction',
    'calculate_fractional_kelly',
    'calculate_kelly_with_limits',
    'optimal_bet_size',
    'simulate_kelly_performance',
]
