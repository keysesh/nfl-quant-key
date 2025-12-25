"""
NFL QUANT - Risk of Ruin Calculator
====================================

Tier 5 Blueprint feature for bankroll management and risk assessment.

Implements:
1. Closed-form risk of ruin for fixed bet sizing
2. Monte Carlo simulation for Kelly sizing
3. Drawdown probability analysis
4. Optimal bet sizing given risk tolerance

Theory:
-------
Risk of Ruin (RoR) is the probability of losing your entire bankroll given:
- Win probability (p)
- Payoff ratio (b, decimal odds - 1)
- Bet size as fraction of bankroll (f)

For fixed bet sizing with even money bets:
    RoR = ((1-p)/p)^(bankroll/bet_size) if p > 0.5 else 1.0

For fractional Kelly with variable odds:
    RoR ≈ (1/e)^(bankroll_units / expected_growth_rate)

Usage:
    from nfl_quant.betting.risk_of_ruin import (
        calculate_risk_of_ruin,
        simulate_bankroll_paths,
        find_safe_bet_size,
    )

    # Fixed bet sizing
    ror = calculate_risk_of_ruin(
        bankroll=1000,
        bet_size=50,
        win_prob=0.55,
        odds=-110
    )

    # Monte Carlo simulation
    paths = simulate_bankroll_paths(
        bankroll=1000,
        bet_size=50,
        win_prob=0.55,
        odds=-110,
        n_bets=100,
        n_simulations=10000
    )
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk assessment metrics for a betting strategy."""
    risk_of_ruin: float          # Probability of losing entire bankroll
    expected_growth: float       # Expected bankroll growth rate
    max_drawdown_50: float       # 50th percentile max drawdown
    max_drawdown_95: float       # 95th percentile max drawdown
    break_even_bets: int         # Bets needed to recover from 50% drawdown
    kelly_fraction: float        # Optimal Kelly bet fraction
    half_kelly_ror: float        # RoR at half Kelly
    quarter_kelly_ror: float     # RoR at quarter Kelly


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds >= 100:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


def kelly_fraction(win_prob: float, decimal_odds: float) -> float:
    """
    Calculate optimal Kelly bet fraction.

    Formula: f* = (p * b - q) / b
    where:
        p = win probability
        q = 1 - p
        b = decimal odds - 1 (payoff ratio)

    Returns:
        Optimal bet fraction (can be negative = no bet)
    """
    if decimal_odds <= 1:
        return 0.0

    b = decimal_odds - 1  # Payoff ratio
    q = 1 - win_prob

    kelly = (win_prob * b - q) / b

    return max(0, kelly)  # Never bet negative


def calculate_risk_of_ruin(
    bankroll: float,
    bet_size: float,
    win_prob: float,
    odds: float,
    odds_format: str = 'american',
) -> float:
    """
    Calculate probability of losing entire bankroll.

    Uses closed-form solution for geometric random walk.

    Args:
        bankroll: Starting bankroll in dollars
        bet_size: Bet size in dollars
        win_prob: True win probability (0 to 1)
        odds: Betting odds (American format by default)
        odds_format: 'american' or 'decimal'

    Returns:
        Probability of ruin (0 to 1)
    """
    if bet_size <= 0 or bankroll <= 0:
        return 0.0

    if bet_size > bankroll:
        return 1.0  # Betting more than bankroll

    # Convert odds
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    # Calculate edge
    implied_prob = decimal_to_implied_prob(decimal_odds)
    edge = win_prob - implied_prob

    if edge <= 0:
        # No edge = eventual ruin is certain
        return 1.0

    if win_prob <= 0.5:
        # Less than 50% win rate = high ruin risk
        return 1.0

    # Number of bet units in bankroll
    units = bankroll / bet_size

    # Closed-form RoR for fixed betting
    # RoR = ((1-p)/p)^n where n = units
    loss_prob = 1 - win_prob
    ratio = loss_prob / win_prob

    if ratio >= 1:
        return 1.0

    ror = math.pow(ratio, units)

    return min(1.0, max(0.0, ror))


def calculate_risk_of_ruin_kelly(
    bankroll_units: float,
    win_prob: float,
    decimal_odds: float,
    kelly_multiplier: float = 1.0,
) -> float:
    """
    Calculate RoR for Kelly betting strategy.

    Kelly betting has theoretically 0% RoR, but fractional Kelly has some risk.

    Args:
        bankroll_units: Bankroll in bet units
        win_prob: True win probability
        decimal_odds: Decimal odds
        kelly_multiplier: Fraction of Kelly (0.5 = half Kelly)

    Returns:
        Estimated RoR
    """
    full_kelly = kelly_fraction(win_prob, decimal_odds)

    if full_kelly <= 0:
        return 1.0  # No edge

    f = full_kelly * kelly_multiplier

    if f <= 0:
        return 0.0  # No betting = no ruin

    if f >= 1:
        return 1.0  # Betting 100% = instant ruin possible

    # Expected growth rate per bet (Kelly criterion formula)
    b = decimal_odds - 1
    growth = win_prob * math.log(1 + f * b) + (1 - win_prob) * math.log(1 - f)

    if growth <= 0:
        return 1.0

    # Approximate RoR using growth rate
    # Lower bound: RoR ≈ exp(-2 * growth * bankroll_units)
    ror = math.exp(-2 * growth * bankroll_units)

    return min(1.0, max(0.0, ror))


def simulate_bankroll_paths(
    bankroll: float,
    bet_size: float,
    win_prob: float,
    odds: float,
    n_bets: int = 100,
    n_simulations: int = 10000,
    odds_format: str = 'american',
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation of bankroll evolution.

    Args:
        bankroll: Starting bankroll
        bet_size: Fixed bet size
        win_prob: True win probability
        odds: Betting odds
        n_bets: Number of bets per simulation
        n_simulations: Number of simulation paths
        odds_format: 'american' or 'decimal'
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results:
        - ruin_rate: Fraction of paths that hit zero
        - mean_final: Average final bankroll
        - median_final: Median final bankroll
        - percentiles: 5th, 25th, 50th, 75th, 95th percentile finals
        - max_drawdowns: Max drawdown for each path
    """
    np.random.seed(seed)

    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    profit_if_win = bet_size * (decimal_odds - 1)
    loss_if_lose = bet_size

    # Initialize tracking arrays
    final_bankrolls = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0

    for sim in range(n_simulations):
        current_bankroll = bankroll
        peak_bankroll = bankroll
        max_dd = 0.0

        for bet in range(n_bets):
            if current_bankroll <= 0:
                ruin_count += 1
                break

            # Simulate bet outcome
            if np.random.random() < win_prob:
                current_bankroll += profit_if_win
            else:
                current_bankroll -= loss_if_lose

            # Track peak and drawdown
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            else:
                drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
                if drawdown > max_dd:
                    max_dd = drawdown

        final_bankrolls[sim] = max(0, current_bankroll)
        max_drawdowns[sim] = max_dd

    return {
        'ruin_rate': ruin_count / n_simulations,
        'mean_final': float(np.mean(final_bankrolls)),
        'median_final': float(np.median(final_bankrolls)),
        'std_final': float(np.std(final_bankrolls)),
        'percentiles': {
            'p5': float(np.percentile(final_bankrolls, 5)),
            'p25': float(np.percentile(final_bankrolls, 25)),
            'p50': float(np.percentile(final_bankrolls, 50)),
            'p75': float(np.percentile(final_bankrolls, 75)),
            'p95': float(np.percentile(final_bankrolls, 95)),
        },
        'max_drawdown_median': float(np.median(max_drawdowns)),
        'max_drawdown_95': float(np.percentile(max_drawdowns, 95)),
        'n_simulations': n_simulations,
        'n_bets': n_bets,
    }


def find_safe_bet_size(
    bankroll: float,
    win_prob: float,
    odds: float,
    max_ror: float = 0.01,
    odds_format: str = 'american',
) -> float:
    """
    Find maximum bet size that keeps RoR below threshold.

    Args:
        bankroll: Starting bankroll
        win_prob: True win probability
        odds: Betting odds
        max_ror: Maximum acceptable risk of ruin (default 1%)
        odds_format: 'american' or 'decimal'

    Returns:
        Maximum safe bet size
    """
    # Binary search for bet size
    low = 0.0
    high = bankroll

    for _ in range(50):  # Max iterations
        mid = (low + high) / 2
        ror = calculate_risk_of_ruin(bankroll, mid, win_prob, odds, odds_format)

        if ror < max_ror:
            low = mid
        else:
            high = mid

        if high - low < 0.01:
            break

    return low


def analyze_betting_strategy(
    bankroll: float,
    bet_size: float,
    win_prob: float,
    odds: float,
    odds_format: str = 'american',
) -> RiskMetrics:
    """
    Comprehensive risk analysis for a betting strategy.

    Args:
        bankroll: Starting bankroll
        bet_size: Fixed bet size
        win_prob: True win probability
        odds: Betting odds
        odds_format: 'american' or 'decimal'

    Returns:
        RiskMetrics with full analysis
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    # Calculate Kelly fraction
    kelly = kelly_fraction(win_prob, decimal_odds)

    # Calculate RoR at different Kelly fractions
    units = bankroll / bet_size if bet_size > 0 else float('inf')

    ror_full = calculate_risk_of_ruin(bankroll, bet_size, win_prob, odds, odds_format)
    ror_half = calculate_risk_of_ruin_kelly(units, win_prob, decimal_odds, 0.5)
    ror_quarter = calculate_risk_of_ruin_kelly(units, win_prob, decimal_odds, 0.25)

    # Expected growth per bet
    b = decimal_odds - 1
    ev_per_bet = win_prob * b - (1 - win_prob)
    expected_growth = ev_per_bet * bet_size / bankroll if bankroll > 0 else 0

    # Simulate for drawdown analysis
    sim = simulate_bankroll_paths(
        bankroll, bet_size, win_prob, odds,
        n_bets=100, n_simulations=1000,
        odds_format=odds_format
    )

    # Bets to recover from 50% drawdown
    if expected_growth > 0:
        break_even = int(math.ceil(0.5 * bankroll / (expected_growth * bankroll)))
    else:
        break_even = float('inf')

    return RiskMetrics(
        risk_of_ruin=ror_full,
        expected_growth=expected_growth,
        max_drawdown_50=sim['max_drawdown_median'],
        max_drawdown_95=sim['max_drawdown_95'],
        break_even_bets=min(break_even, 99999),
        kelly_fraction=kelly,
        half_kelly_ror=ror_half,
        quarter_kelly_ror=ror_quarter,
    )


def recommend_position_size(
    bankroll: float,
    edge: float,
    odds: float,
    risk_tolerance: str = 'moderate',
    odds_format: str = 'american',
) -> dict:
    """
    Recommend bet sizing based on edge and risk tolerance.

    Args:
        bankroll: Current bankroll
        edge: Estimated edge (e.g., 0.05 = 5% edge)
        odds: Betting odds
        risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        odds_format: 'american' or 'decimal'

    Returns:
        Dictionary with recommended bet size and analysis
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    implied_prob = decimal_to_implied_prob(decimal_odds)
    true_prob = implied_prob + edge

    if true_prob <= implied_prob:
        return {
            'recommended_bet': 0.0,
            'reason': 'No edge detected',
            'kelly_fraction': 0.0,
        }

    kelly = kelly_fraction(true_prob, decimal_odds)

    # Kelly multipliers by risk tolerance
    multipliers = {
        'conservative': 0.25,
        'moderate': 0.50,
        'aggressive': 0.75,
    }

    mult = multipliers.get(risk_tolerance, 0.50)
    recommended_fraction = kelly * mult
    recommended_bet = recommended_fraction * bankroll

    # Cap at 5% of bankroll for safety
    max_bet = 0.05 * bankroll
    recommended_bet = min(recommended_bet, max_bet)

    # Calculate RoR at this size
    ror = calculate_risk_of_ruin(bankroll, recommended_bet, true_prob, odds, odds_format)

    return {
        'recommended_bet': round(recommended_bet, 2),
        'fraction_of_bankroll': recommended_fraction,
        'kelly_fraction': kelly,
        'kelly_multiplier': mult,
        'risk_of_ruin': ror,
        'edge': edge,
        'true_prob': true_prob,
        'implied_prob': implied_prob,
    }


def simulate_small_stakes_season(
    initial_bankroll: float = 100.0,
    bet_size: float = 3.50,
    win_prob: float = 0.55,
    bets_per_week: int = 10,
    weeks: int = 17,
    n_simulations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Simulate realistic NFL season for small-stakes bettor.

    Designed for $2-$5 betting with $100 bankroll as per user requirements.

    Args:
        initial_bankroll: Starting bankroll ($100 default)
        bet_size: Average bet size ($3.50 = middle of $2-$5 range)
        win_prob: Expected win rate (55% for validated markets)
        bets_per_week: Bets placed per week (10 default)
        weeks: NFL regular season weeks (17)
        n_simulations: Monte Carlo paths (10000)
        seed: Random seed for reproducibility

    Returns:
        dict with:
        - expected_profit: Mean profit over season
        - probability_profitable: % of seasons in the black
        - worst_case_5pct: 5th percentile final bankroll
        - best_case_95pct: 95th percentile final bankroll
        - median_final: 50th percentile final bankroll
        - max_drawdown_median: Typical max drawdown
        - max_drawdown_worst: 95th percentile max drawdown
    """
    np.random.seed(seed)

    n_bets = bets_per_week * weeks  # ~170 bets per season

    # Simulate outcomes (-110 odds: win = bet_size * 0.909, lose = bet_size)
    win_amount = bet_size * 0.909  # Standard -110 payout
    lose_amount = bet_size

    final_bankrolls = []
    max_drawdowns = []

    for _ in range(n_simulations):
        bankroll = initial_bankroll
        peak = initial_bankroll
        max_dd = 0

        for _ in range(n_bets):
            if bankroll <= 0:
                break

            # Determine if we win this bet
            if np.random.random() < win_prob:
                bankroll += win_amount
            else:
                bankroll -= lose_amount

            # Track drawdown
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        final_bankrolls.append(bankroll)
        max_drawdowns.append(max_dd)

    final_arr = np.array(final_bankrolls)
    dd_arr = np.array(max_drawdowns)

    return {
        'initial_bankroll': initial_bankroll,
        'bet_size': bet_size,
        'win_prob': win_prob,
        'total_bets': n_bets,
        'expected_profit': float(np.mean(final_arr) - initial_bankroll),
        'probability_profitable': float(np.mean(final_arr > initial_bankroll)),
        'probability_ruin': float(np.mean(final_arr <= 0)),
        'worst_case_5pct': float(np.percentile(final_arr, 5)),
        'median_final': float(np.percentile(final_arr, 50)),
        'best_case_95pct': float(np.percentile(final_arr, 95)),
        'max_drawdown_median': float(np.percentile(dd_arr, 50)),
        'max_drawdown_worst': float(np.percentile(dd_arr, 95)),
        'mean_final': float(np.mean(final_arr)),
        'std_final': float(np.std(final_arr)),
    }
