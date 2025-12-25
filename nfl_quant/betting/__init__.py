"""NFL Quant Betting Module - SNR-based filtering and bet sizing."""

from nfl_quant.betting.snr_filter import (
    SNRFilter,
    BetDecision,
    filter_bets_by_snr,
)

from nfl_quant.betting.risk_of_ruin import (
    calculate_risk_of_ruin,
    calculate_risk_of_ruin_kelly,
    simulate_bankroll_paths,
    find_safe_bet_size,
    analyze_betting_strategy,
    recommend_position_size,
    RiskMetrics,
)

__all__ = [
    # SNR Filter
    'SNRFilter',
    'BetDecision',
    'filter_bets_by_snr',
    # Risk of Ruin (V28)
    'calculate_risk_of_ruin',
    'calculate_risk_of_ruin_kelly',
    'simulate_bankroll_paths',
    'find_safe_bet_size',
    'analyze_betting_strategy',
    'recommend_position_size',
    'RiskMetrics',
]
