"""
Kelly Criterion Bet Sizing
===========================

Industry standard for optimal bankroll management in sports betting.
Kelly criterion maximizes long-term growth rate while managing risk.

Formula: f* = (bp - q) / b
Where:
    f* = fraction of bankroll to bet
    b = decimal odds - 1 (net odds)
    p = probability of winning
    q = 1 - p (probability of losing)

Fractional Kelly is recommended to reduce variance (typically 25-50% of full Kelly).
"""

import numpy as np
from typing import Dict, Any


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(american_odds))


def calculate_kelly_fraction(win_prob: float, american_odds: float) -> float:
    """
    Calculate optimal Kelly fraction for a single bet.

    Args:
        win_prob: Probability of winning (0-1)
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Fraction of bankroll to bet (0-1)
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0

    # Convert to decimal odds
    decimal_odds = american_to_decimal(american_odds)

    # Kelly formula
    b = decimal_odds - 1.0  # Net odds
    p = win_prob
    q = 1.0 - p

    kelly = (b * p - q) / b

    # Return 0 if negative edge
    return max(0.0, kelly)


def calculate_fractional_kelly(
    win_prob: float,
    american_odds: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate fractional Kelly bet size (more conservative).

    Args:
        win_prob: Probability of winning
        american_odds: American odds
        fraction: Kelly fraction (0.25 = quarter Kelly, recommended)

    Returns:
        Fraction of bankroll to bet
    """
    full_kelly = calculate_kelly_fraction(win_prob, american_odds)
    return full_kelly * fraction


def calculate_kelly_with_limits(
    win_prob: float,
    american_odds: float,
    max_bet_fraction: float = 0.05,
    min_edge_pct: float = 0.05,
    kelly_fraction: float = 0.25
) -> Dict[str, Any]:
    """
    Calculate Kelly bet size with practical limits.

    Args:
        win_prob: Calibrated probability of winning
        american_odds: American odds
        max_bet_fraction: Maximum fraction of bankroll per bet (default 5%)
        min_edge_pct: Minimum edge required to bet (default 5%)
        kelly_fraction: Fraction of Kelly to use (default 25%)

    Returns:
        Dictionary with betting recommendation
    """
    # Calculate implied probability from odds
    decimal_odds = american_to_decimal(american_odds)
    implied_prob = 1.0 / decimal_odds

    # Calculate edge
    edge = win_prob - implied_prob

    # Don't bet if edge is below minimum
    if edge < min_edge_pct:
        return {
            'should_bet': False,
            'reason': f'Edge {edge:.1%} below minimum {min_edge_pct:.1%}',
            'edge': edge,
            'recommended_fraction': 0.0,
            'confidence': 'NO BET'
        }

    # Calculate Kelly fraction
    full_kelly = calculate_kelly_fraction(win_prob, american_odds)
    fractional_kelly = full_kelly * kelly_fraction

    # Apply maximum bet limit
    final_fraction = min(fractional_kelly, max_bet_fraction)

    # Determine confidence level
    if edge >= 0.15:
        confidence = 'HIGH'
    elif edge >= 0.10:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    return {
        'should_bet': True,
        'edge': edge,
        'full_kelly': full_kelly,
        'fractional_kelly': fractional_kelly,
        'recommended_fraction': final_fraction,
        'confidence': confidence,
        'expected_value': edge * final_fraction * 100  # EV per $100 risked
    }


def optimal_bet_size(
    bankroll: float,
    win_prob: float,
    american_odds: float,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 5.0,
    min_edge_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Calculate optimal bet amount in dollars.

    Args:
        bankroll: Total bankroll in dollars
        win_prob: Calibrated probability of winning
        american_odds: American odds
        kelly_fraction: Fraction of Kelly (0.25 recommended)
        max_bet_pct: Maximum bet as percentage of bankroll
        min_edge_pct: Minimum edge percentage to place bet

    Returns:
        Dictionary with bet recommendation
    """
    kelly_result = calculate_kelly_with_limits(
        win_prob=win_prob,
        american_odds=american_odds,
        max_bet_fraction=max_bet_pct / 100.0,
        min_edge_pct=min_edge_pct / 100.0,
        kelly_fraction=kelly_fraction
    )

    if not kelly_result['should_bet']:
        return {
            **kelly_result,
            'bet_amount': 0.0,
            'bankroll': bankroll
        }

    bet_amount = bankroll * kelly_result['recommended_fraction']

    # Round to nearest dollar
    bet_amount = round(bet_amount)

    return {
        **kelly_result,
        'bet_amount': bet_amount,
        'bankroll': bankroll,
        'bet_percentage': (bet_amount / bankroll) * 100
    }


def simulate_kelly_performance(
    predictions: list,
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 5.0,
    min_edge_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Simulate Kelly betting strategy performance on historical predictions.

    Args:
        predictions: List of dicts with 'prob_over_cal', 'edge', 'went_over', 'recommendation', 'over_odds', 'under_odds'
        initial_bankroll: Starting bankroll
        kelly_fraction: Fraction of Kelly to use
        max_bet_pct: Maximum bet percentage
        min_edge_pct: Minimum edge to bet

    Returns:
        Performance metrics
    """
    bankroll = initial_bankroll
    bets_placed = 0
    wins = 0
    total_wagered = 0.0
    total_profit = 0.0

    for pred in predictions:
        if pred.get('edge', 0) < min_edge_pct / 100.0:
            continue

        # Determine which side to bet
        if pred['recommendation'] == 'OVER':
            win_prob = pred['prob_over_cal']
            odds = pred['over_odds']
            won = pred['went_over'] == 1
        else:
            win_prob = 1.0 - pred['prob_over_cal']
            odds = pred['under_odds']
            won = pred['went_over'] == 0

        # Calculate bet size
        bet_info = optimal_bet_size(
            bankroll=bankroll,
            win_prob=win_prob,
            american_odds=odds,
            kelly_fraction=kelly_fraction,
            max_bet_pct=max_bet_pct,
            min_edge_pct=min_edge_pct
        )

        if not bet_info['should_bet']:
            continue

        bet_amount = bet_info['bet_amount']
        total_wagered += bet_amount
        bets_placed += 1

        # Calculate profit/loss
        if won:
            wins += 1
            decimal_odds = american_to_decimal(odds)
            profit = bet_amount * (decimal_odds - 1.0)
        else:
            profit = -bet_amount

        total_profit += profit
        bankroll += profit

        # Stop if bankroll depleted
        if bankroll <= 0:
            break

    win_rate = wins / bets_placed if bets_placed > 0 else 0
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    return {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': total_profit,
        'roi_pct': roi,
        'bets_placed': bets_placed,
        'wins': wins,
        'win_rate': win_rate,
        'total_wagered': total_wagered,
        'bankroll_growth': (bankroll / initial_bankroll - 1) * 100
    }
