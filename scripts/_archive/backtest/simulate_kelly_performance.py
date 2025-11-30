#!/usr/bin/env python3
"""
Simulate Kelly Criterion betting performance on backtest results.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.betting.kelly_criterion import simulate_kelly_performance


def main():
    # Load backtest results
    df = pd.read_csv(Path('data/backtest/real_props_backtest.csv'))

    # Prepare predictions for Kelly simulation
    predictions = []
    for _, row in df.iterrows():
        # Determine recommendation (OVER if calibrated prob > 0.5)
        prob_over = row['prob_over_calibrated']
        if prob_over > 0.5:
            recommendation = 'OVER'
            edge = row['edge_over']
        else:
            recommendation = 'UNDER'
            edge = row['edge_under']

        pred = {
            'prob_over_cal': prob_over,
            'edge': edge,
            'went_over': row['went_over'],
            'recommendation': recommendation,
            'over_odds': row.get('over_odds', -110),
            'under_odds': row.get('under_odds', -110)
        }
        predictions.append(pred)

    print('=' * 60)
    print('KELLY CRITERION PERFORMANCE SIMULATION')
    print('=' * 60)
    print(f'Total predictions: {len(predictions)}')

    # Test different configurations
    print('\nConfiguration Comparison:')
    print('-' * 60)

    for kelly_frac in [0.25, 0.50]:
        for min_edge in [5, 10, 15]:
            result = simulate_kelly_performance(
                predictions,
                initial_bankroll=1000.0,
                kelly_fraction=kelly_frac,
                max_bet_pct=5.0,
                min_edge_pct=min_edge / 100.0
            )

            print(f'Kelly {kelly_frac:.0%}, Min Edge {min_edge}%:')
            print(f'  Bets: {result["bets_placed"]:4d} | '
                  f'Win: {result["win_rate"]:5.1%} | '
                  f'ROI: {result["roi_pct"]:+6.1f}% | '
                  f'${result["initial_bankroll"]:.0f} -> ${result["final_bankroll"]:.0f}')

    # Best configuration
    print('\n' + '=' * 60)
    print('RECOMMENDED: Quarter Kelly, 5% min edge (balanced)')
    print('=' * 60)
    result = simulate_kelly_performance(
        predictions,
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_pct=5.0,
        min_edge_pct=0.05
    )
    print(f'Starting bankroll: ${result["initial_bankroll"]:.0f}')
    print(f'Final bankroll:    ${result["final_bankroll"]:.0f}')
    print(f'Total profit:      ${result["profit"]:.2f}')
    print(f'ROI:               {result["roi_pct"]:.1f}%')
    print(f'Bets placed:       {result["bets_placed"]}')
    print(f'Win rate:          {result["win_rate"]:.1%}')
    print(f'Total wagered:     ${result["total_wagered"]:.2f}')
    print(f'Bankroll growth:   {result["bankroll_growth"]:.1f}%')

    # Conservative high-edge only
    print('\n' + '=' * 60)
    print('CONSERVATIVE: Quarter Kelly, 10% min edge (high confidence)')
    print('=' * 60)
    result = simulate_kelly_performance(
        predictions,
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_pct=5.0,
        min_edge_pct=0.10
    )
    print(f'Starting bankroll: ${result["initial_bankroll"]:.0f}')
    print(f'Final bankroll:    ${result["final_bankroll"]:.0f}')
    print(f'Total profit:      ${result["profit"]:.2f}')
    print(f'ROI:               {result["roi_pct"]:.1f}%')
    print(f'Bets placed:       {result["bets_placed"]}')
    print(f'Win rate:          {result["win_rate"]:.1%}')
    print(f'Total wagered:     ${result["total_wagered"]:.2f}')
    print(f'Bankroll growth:   {result["bankroll_growth"]:.1f}%')


if __name__ == '__main__':
    main()
