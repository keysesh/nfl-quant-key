#!/usr/bin/env python3
"""
Walk-Forward Validation on Full Unbiased Universe

This script performs true walk-forward validation:
1. Train on past weeks
2. Test on future week
3. No look-ahead bias
4. Full universe of props (not cherry-picked)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_unbiased_dataset():
    """Load the unbiased backtest dataset"""
    df = pd.read_parquet(PROJECT_ROOT / 'data' / 'backtest' / 'unbiased_backtest_dataset.parquet')
    print(f"Loaded {len(df)} props from unbiased dataset")
    print(f"Seasons: {sorted(df['season'].unique().tolist())}")
    print(f"Weeks: {sorted(df['week'].unique().tolist())}")
    return df


def calculate_edge(train_data, market):
    """
    Calculate naive model edge based on historical over-hit rates

    This is a simple model that:
    1. Calculates historical over-hit rate for this market
    2. Compares to market implied probability
    3. Returns edge if our prob differs from market
    """
    market_data = train_data[train_data['market'] == market]

    if len(market_data) < 10:
        return None

    # Historical over-hit rate
    actual_over_rate = market_data['over_hit'].mean()

    # Average market implied probability (no-vig)
    avg_market_over_prob = market_data['fair_over_prob'].mean()

    # Edge = our probability - market probability
    edge = actual_over_rate - avg_market_over_prob

    return {
        'over_rate': actual_over_rate,
        'market_over_prob': avg_market_over_prob,
        'edge': edge,
        'sample_size': len(market_data)
    }


def simple_betting_strategy(test_row, edge_info, min_edge=0.05):
    """
    Simple betting strategy based on edge

    Bets over if:
    1. Historical over-hit rate > market implied prob + min_edge

    Bets under if:
    1. Historical under-hit rate > market implied prob + min_edge
    """
    if edge_info is None:
        return None, 0

    over_edge = edge_info['edge']
    under_edge = -edge_info['edge']  # If over has negative edge, under has positive

    # Check for betting opportunity
    if over_edge > min_edge:
        return 'over', over_edge
    elif under_edge > min_edge:
        return 'under', under_edge
    else:
        return None, 0


def calculate_bet_profit(row, bet_direction):
    """Calculate profit for a bet"""
    if bet_direction is None:
        return 0

    if bet_direction == 'over':
        if row['over_hit']:
            # Won
            odds = row['over_odds']
            if odds > 0:
                return odds  # $100 bet returns $odds profit
            else:
                return 100 / abs(odds) * 100
        else:
            return -100  # Lost $100

    elif bet_direction == 'under':
        if row['under_hit']:
            odds = row['under_odds']
            if odds > 0:
                return odds
            else:
                return 100 / abs(odds) * 100
        else:
            return -100

    return 0


def walk_forward_validation(df, min_train_weeks=4, min_edge=0.05):
    """
    Perform walk-forward validation

    For each test week:
    1. Train on all previous weeks
    2. Calculate edges per market
    3. Bet on props with edge > min_edge
    4. Track profit/loss
    """
    results = []
    all_weeks = sorted(df['week'].unique())

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"Min training weeks: {min_train_weeks}")
    print(f"Minimum edge for betting: {min_edge*100:.1f}%")
    print(f"{'='*60}")

    for test_week in all_weeks[min_train_weeks:]:
        # Training data: all weeks before test week
        train_data = df[df['week'] < test_week]
        test_data = df[df['week'] == test_week]

        if len(test_data) == 0:
            continue

        train_weeks = sorted(train_data['week'].unique().tolist())

        print(f"\nWeek {test_week}: Train on weeks {train_weeks}")
        print(f"  Training props: {len(train_data)}")
        print(f"  Testing props: {len(test_data)}")

        # Calculate edges for each market
        market_edges = {}
        for market in df['market'].unique():
            edge_info = calculate_edge(train_data, market)
            if edge_info:
                market_edges[market] = edge_info
                print(f"  {market}: over_rate={edge_info['over_rate']*100:.1f}%, edge={edge_info['edge']*100:.1f}%")

        # Test on week's props
        week_bets = 0
        week_profit = 0
        week_results = []

        for _, row in test_data.iterrows():
            market = row['market']
            edge_info = market_edges.get(market)

            bet_direction, bet_edge = simple_betting_strategy(row, edge_info, min_edge)

            if bet_direction:
                profit = calculate_bet_profit(row, bet_direction)
                week_bets += 1
                week_profit += profit

                week_results.append({
                    'week': test_week,
                    'player': row['player'],
                    'market': market,
                    'line': row['line'],
                    'bet': bet_direction,
                    'edge': bet_edge,
                    'actual': row['actual_stat'],
                    'won': profit > 0,
                    'profit': profit
                })

        roi = (week_profit / (week_bets * 100) * 100) if week_bets > 0 else 0
        hit_rate = sum(1 for r in week_results if r['won']) / len(week_results) * 100 if week_results else 0

        print(f"  Bets placed: {week_bets}")
        print(f"  Hit rate: {hit_rate:.1f}%")
        print(f"  Week profit: ${week_profit:.2f} ({roi:.2f}% ROI)")

        results.append({
            'test_week': test_week,
            'train_weeks': train_weeks,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'bets_placed': week_bets,
            'hit_rate': hit_rate,
            'profit': week_profit,
            'roi': roi,
            'details': week_results
        })

    return results


def summarize_results(results):
    """Summarize walk-forward validation results"""
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*60}")

    total_bets = sum(r['bets_placed'] for r in results)
    total_profit = sum(r['profit'] for r in results)

    if total_bets == 0:
        print("No bets placed!")
        return

    total_roi = total_profit / (total_bets * 100) * 100
    avg_hit_rate = np.mean([r['hit_rate'] for r in results if r['bets_placed'] > 0])

    print(f"\nTotal weeks tested: {len(results)}")
    print(f"Total bets placed: {total_bets}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Overall ROI: {total_roi:.2f}%")
    print(f"Average hit rate: {avg_hit_rate:.1f}%")

    # Week-by-week breakdown
    print("\nWeek-by-week:")
    for r in results:
        profit_str = f"${r['profit']:+.2f}"
        roi_str = f"{r['roi']:+.2f}%"
        print(f"  Week {r['test_week']}: {r['bets_placed']} bets, {r['hit_rate']:.1f}% hit, {profit_str} ({roi_str})")

    # Cumulative P&L
    print("\nCumulative P&L:")
    cumulative = 0
    for r in results:
        cumulative += r['profit']
        print(f"  After Week {r['test_week']}: ${cumulative:.2f}")

    # Market breakdown
    print("\nMarket breakdown:")
    market_results = {}
    for r in results:
        for detail in r['details']:
            market = detail['market']
            if market not in market_results:
                market_results[market] = {'bets': 0, 'wins': 0, 'profit': 0}
            market_results[market]['bets'] += 1
            market_results[market]['wins'] += 1 if detail['won'] else 0
            market_results[market]['profit'] += detail['profit']

    for market, stats in sorted(market_results.items()):
        win_rate = stats['wins'] / stats['bets'] * 100
        roi = stats['profit'] / (stats['bets'] * 100) * 100
        print(f"  {market}:")
        print(f"    Bets: {stats['bets']}, Win rate: {win_rate:.1f}%, ROI: {roi:.2f}%")

    return {
        'total_bets': total_bets,
        'total_profit': total_profit,
        'total_roi': total_roi,
        'avg_hit_rate': avg_hit_rate,
        'market_results': market_results
    }


def run_multiple_strategies(df):
    """Run validation with different edge thresholds"""
    print("\n" + "="*60)
    print("TESTING MULTIPLE EDGE THRESHOLDS")
    print("="*60)

    strategies = []

    for min_edge in [0.0, 0.02, 0.05, 0.08, 0.10]:
        print(f"\n--- Min Edge: {min_edge*100:.0f}% ---")
        results = walk_forward_validation(df, min_train_weeks=4, min_edge=min_edge)

        if results:
            total_bets = sum(r['bets_placed'] for r in results)
            total_profit = sum(r['profit'] for r in results)

            if total_bets > 0:
                roi = total_profit / (total_bets * 100) * 100
                strategies.append({
                    'min_edge': min_edge,
                    'bets': total_bets,
                    'profit': total_profit,
                    'roi': roi
                })

    # Compare strategies
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")

    for s in strategies:
        print(f"Min Edge {s['min_edge']*100:.0f}%: {s['bets']} bets, ${s['profit']:.2f} profit, {s['roi']:.2f}% ROI")

    return strategies


def main():
    print("="*60)
    print("WALK-FORWARD VALIDATION ON FULL UNIVERSE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # Load data
    df = load_unbiased_dataset()

    # Run main validation
    results = walk_forward_validation(df, min_train_weeks=4, min_edge=0.05)
    summary = summarize_results(results)

    # Test multiple strategies
    strategies = run_multiple_strategies(df)

    # Save results
    output_dir = PROJECT_ROOT / 'data' / 'backtest'

    # Save detailed results
    all_bets = []
    for r in results:
        all_bets.extend(r['details'])

    if all_bets:
        bets_df = pd.DataFrame(all_bets)
        bets_df.to_csv(output_dir / 'walk_forward_all_bets.csv', index=False)
        print(f"\n✅ Saved all bets: {output_dir / 'walk_forward_all_bets.csv'}")

    # Save summary
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_props': len(df),
        'weeks_tested': len(results),
        **summary,
        'strategies': strategies
    }

    with open(output_dir / 'walk_forward_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float64, np.int64)) else x)
    print(f"✅ Saved summary: {output_dir / 'walk_forward_summary.json'}")


if __name__ == '__main__':
    main()
