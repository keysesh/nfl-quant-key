#!/usr/bin/env python3
"""
Edge & Confidence Threshold Optimization Tool

Tests different combinations of edge thresholds and confidence levels to find
the optimal settings for maximum win rate and ROI.

This tool helps answer:
- What edge threshold yields the best win rate?
- What confidence level should we require?
- Which combinations maximize ROI?
- What's the trade-off between volume and win rate?

Usage:
    python scripts/analyze/optimize_edge_confidence.py --weeks 1-10
    python scripts/analyze/optimize_edge_confidence.py --full
    python scripts/analyze/optimize_edge_confidence.py --weeks 1-10 --by-market
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.validate.validate_framework_2025 import (
    load_recommendations,
    load_actual_stats,
    load_game_results,
    match_player_bet_to_actual,
    match_game_bet_to_actual,
    calculate_profit,
)


def load_comprehensive_backtest_data() -> pd.DataFrame:
    """Load comprehensive backtest data (13,915 bets)."""
    backtest_file = Path('reports/detailed_bet_analysis_weekall.csv')

    if not backtest_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(backtest_file)

    # Standardize column names
    if 'bet_won' in df.columns:
        df['won'] = df['bet_won'].astype(bool)

    # Convert edge to percentage if needed
    if 'edge' in df.columns:
        # Check if edge is already percentage (0-1) or decimal (0-100)
        if df['edge'].max() <= 1.0:
            df['edge_pct'] = df['edge'] * 100
        else:
            df['edge_pct'] = df['edge']

    # Ensure model_prob is 0-1 range
    if 'model_prob' in df.columns:
        if df['model_prob'].max() > 1.0:
            df['model_prob'] = df['model_prob'] / 100

    # Profit is already calculated, but ensure it's in dollars
    if 'profit' in df.columns:
        # Profit appears to be in units, convert to dollars if needed
        if df['profit'].abs().max() < 10:
            df['profit'] = df['profit'] * 100  # Convert units to dollars

    return df


def collect_all_bet_results(weeks: List[int] = None, season: int = 2025, use_comprehensive: bool = False) -> pd.DataFrame:
    """Collect all bet results for analysis."""

    # Try comprehensive backtest data first if requested
    if use_comprehensive:
        df = load_comprehensive_backtest_data()
        if not df.empty:
            return df

    # Fallback to week-by-week collection
    if weeks is None:
        weeks = []

    all_results = []

    for week in weeks:
        recommendations = load_recommendations(week)
        if recommendations.empty:
            continue

        actual_stats = load_actual_stats(week, season)
        game_results = load_game_results(week, season)

        if actual_stats.empty and game_results.empty:
            continue

        for _, bet in recommendations.iterrows():
            market = bet.get('market', '')

            if 'player_' in market:
                result = match_player_bet_to_actual(bet, actual_stats, week)
            elif market in ['game_total', 'spread', 'moneyline']:
                result = match_game_bet_to_actual(bet, game_results, week)
            else:
                continue

            if result:
                result['week'] = week
                result['model_projection'] = bet.get('model_projection', np.nan)
                result['edge_pct'] = bet.get('edge_pct', np.nan)
                result['model_prob'] = bet.get('model_prob', np.nan)
                result['odds'] = bet.get('odds', np.nan)
                result['market'] = market
                all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Calculate profit
    df['profit'] = df.apply(
        lambda row: calculate_profit(row['odds'], 100.0) if row['won'] else -100.0,
        axis=1
    )

    return df


def test_threshold_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Test different edge and confidence threshold combinations."""
    if df.empty:
        return pd.DataFrame()

    # Define test ranges
    edge_thresholds = [0, 2, 5, 10, 15, 20, 25, 30]
    confidence_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    results = []

    for edge_thresh in edge_thresholds:
        for conf_thresh in confidence_thresholds:
            # Filter bets by thresholds
            filtered = df[
                (df['edge_pct'] >= edge_thresh) &
                (df['model_prob'] >= conf_thresh)
            ].copy()

            if len(filtered) == 0:
                continue

            # Calculate metrics
            total_bets = len(filtered)
            wins = filtered['won'].sum()
            losses = total_bets - wins
            win_rate = wins / total_bets if total_bets > 0 else 0

            total_profit = filtered['profit'].sum()
            total_wagered = total_bets * 100.0
            roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

            avg_edge = filtered['edge_pct'].mean()
            avg_prob = filtered['model_prob'].mean()

            results.append({
                'edge_threshold': edge_thresh,
                'confidence_threshold': conf_thresh,
                'total_bets': total_bets,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate * 100,
                'total_profit': total_profit,
                'roi': roi,
                'avg_edge': avg_edge,
                'avg_prob': avg_prob,
            })

    return pd.DataFrame(results)


def test_by_market(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Test thresholds separately for each market."""
    if df.empty or 'market' not in df.columns:
        return {}

    market_results = {}

    for market in df['market'].unique():
        market_df = df[df['market'] == market]

        if len(market_df) < 5:  # Need minimum sample size
            continue

        market_results[market] = test_threshold_combinations(market_df)

    return market_results


def find_optimal_thresholds(results_df: pd.DataFrame) -> Dict:
    """Find optimal threshold combinations."""
    if results_df.empty:
        return {}

    optimal = {}

    # Highest win rate (with minimum 10 bets)
    min_bets = 10
    valid_results = results_df[results_df['total_bets'] >= min_bets]

    if len(valid_results) > 0:
        best_wr = valid_results.loc[valid_results['win_rate'].idxmax()]
        optimal['highest_win_rate'] = {
            'edge_threshold': best_wr['edge_threshold'],
            'confidence_threshold': best_wr['confidence_threshold'],
            'win_rate': best_wr['win_rate'],
            'roi': best_wr['roi'],
            'total_bets': best_wr['total_bets'],
        }

    # Highest ROI (with minimum 10 bets)
    if len(valid_results) > 0:
        best_roi = valid_results.loc[valid_results['roi'].idxmax()]
        optimal['highest_roi'] = {
            'edge_threshold': best_roi['edge_threshold'],
            'confidence_threshold': best_roi['confidence_threshold'],
            'win_rate': best_roi['win_rate'],
            'roi': best_roi['roi'],
            'total_bets': best_roi['total_bets'],
        }

    # Best balance (win rate > 55% and ROI > 10%)
    balanced = valid_results[
        (valid_results['win_rate'] >= 55) &
        (valid_results['roi'] >= 10)
    ]

    if len(balanced) > 0:
        # Score = win_rate * 0.6 + roi * 0.4 (weighted)
        balanced['score'] = balanced['win_rate'] * 0.6 + balanced['roi'] * 0.4
        best_balanced = balanced.loc[balanced['score'].idxmax()]
        optimal['best_balanced'] = {
            'edge_threshold': best_balanced['edge_threshold'],
            'confidence_threshold': best_balanced['confidence_threshold'],
            'win_rate': best_balanced['win_rate'],
            'roi': best_balanced['roi'],
            'total_bets': best_balanced['total_bets'],
            'score': best_balanced['score'],
        }

    # Maximum volume (most bets with win rate > 52.4%)
    profitable = valid_results[valid_results['win_rate'] >= 52.4]

    if len(profitable) > 0:
        max_volume = profitable.loc[profitable['total_bets'].idxmax()]
        optimal['max_volume'] = {
            'edge_threshold': max_volume['edge_threshold'],
            'confidence_threshold': max_volume['confidence_threshold'],
            'win_rate': max_volume['win_rate'],
            'roi': max_volume['roi'],
            'total_bets': max_volume['total_bets'],
        }

    return optimal


def print_optimization_results(results_df: pd.DataFrame, optimal: Dict):
    """Print comprehensive optimization results."""
    print("\n" + "="*80)
    print("EDGE & CONFIDENCE THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    print()

    if results_df.empty:
        print("⚠️  No results to analyze")
        return

    # Summary table
    print("TOP 10 COMBINATIONS BY WIN RATE:")
    print("-"*100)
    print(f"{'Edge':<8} {'Conf':<8} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'ROI':<10} {'Profit':<12}")
    print("-"*100)

    top_wr = results_df.nlargest(10, 'win_rate')
    for _, row in top_wr.iterrows():
        print(
            f"{row['edge_threshold']:>5}%   {row['confidence_threshold']:>5.0%}   "
            f"{int(row['total_bets']):<8} {int(row['wins']):<8} "
            f"{row['win_rate']:>10.1f}%  {row['roi']:>8.1f}%  ${row['total_profit']:>10,.2f}"
        )

    print()

    print("TOP 10 COMBINATIONS BY ROI:")
    print("-"*100)
    print(f"{'Edge':<8} {'Conf':<8} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'ROI':<10} {'Profit':<12}")
    print("-"*100)

    top_roi = results_df.nlargest(10, 'roi')
    for _, row in top_roi.iterrows():
        print(
            f"{row['edge_threshold']:>5}%   {row['confidence_threshold']:>5.0%}   "
            f"{int(row['total_bets']):<8} {int(row['wins']):<8} "
            f"{row['win_rate']:>10.1f}%  {row['roi']:>8.1f}%  ${row['total_profit']:>10,.2f}"
        )

    print()

    # Optimal recommendations
    if optimal:
        print("="*80)
        print("OPTIMAL THRESHOLD RECOMMENDATIONS")
        print("="*80)
        print()

        if 'highest_win_rate' in optimal:
            opt = optimal['highest_win_rate']
            print("🏆 HIGHEST WIN RATE:")
            print(f"   Edge Threshold: {opt['edge_threshold']}%+")
            print(f"   Confidence Threshold: {opt['confidence_threshold']:.0%}+")
            print(f"   Win Rate: {opt['win_rate']:.1f}%")
            print(f"   ROI: {opt['roi']:.1f}%")
            print(f"   Sample Size: {opt['total_bets']} bets")
            print()

        if 'highest_roi' in optimal:
            opt = optimal['highest_roi']
            print("💰 HIGHEST ROI:")
            print(f"   Edge Threshold: {opt['edge_threshold']}%+")
            print(f"   Confidence Threshold: {opt['confidence_threshold']:.0%}+")
            print(f"   Win Rate: {opt['win_rate']:.1f}%")
            print(f"   ROI: {opt['roi']:.1f}%")
            print(f"   Sample Size: {opt['total_bets']} bets")
            print()

        if 'best_balanced' in optimal:
            opt = optimal['best_balanced']
            print("⚖️  BEST BALANCED (Win Rate >55% + ROI >10%):")
            print(f"   Edge Threshold: {opt['edge_threshold']}%+")
            print(f"   Confidence Threshold: {opt['confidence_threshold']:.0%}+")
            print(f"   Win Rate: {opt['win_rate']:.1f}%")
            print(f"   ROI: {opt['roi']:.1f}%")
            print(f"   Sample Size: {opt['total_bets']} bets")
            print()

        if 'max_volume' in optimal:
            opt = optimal['max_volume']
            print("📊 MAXIMUM VOLUME (Most bets with win rate >52.4%):")
            print(f"   Edge Threshold: {opt['edge_threshold']}%+")
            print(f"   Confidence Threshold: {opt['confidence_threshold']:.0%}+")
            print(f"   Win Rate: {opt['win_rate']:.1f}%")
            print(f"   ROI: {opt['roi']:.1f}%")
            print(f"   Sample Size: {opt['total_bets']} bets")
            print()

    # Heatmap data (for visualization)
    print("="*80)
    print("WIN RATE HEATMAP (Edge Threshold vs Confidence Threshold)")
    print("="*80)
    print()

    # Create pivot table
    pivot_wr = results_df.pivot_table(
        values='win_rate',
        index='edge_threshold',
        columns='confidence_threshold',
        aggfunc='mean'
    )

    print("Win Rate (%) by Threshold Combination:")
    print(pivot_wr.round(1).to_string())
    print()

    pivot_roi = results_df.pivot_table(
        values='roi',
        index='edge_threshold',
        columns='confidence_threshold',
        aggfunc='mean'
    )

    print("ROI (%) by Threshold Combination:")
    print(pivot_roi.round(1).to_string())
    print()


def print_market_specific_results(market_results: Dict[str, pd.DataFrame]):
    """Print market-specific optimization results."""
    if not market_results:
        return

    print("\n" + "="*80)
    print("MARKET-SPECIFIC OPTIMIZATION RESULTS")
    print("="*80)
    print()

    for market, results_df in market_results.items():
        if results_df.empty:
            continue

        optimal = find_optimal_thresholds(results_df)

        print(f"📊 {market.upper()}:")
        print("-"*80)

        if 'best_balanced' in optimal:
            opt = optimal['best_balanced']
            print(f"  Recommended Edge: {opt['edge_threshold']}%+")
            print(f"  Recommended Confidence: {opt['confidence_threshold']:.0%}+")
            print(f"  Expected Win Rate: {opt['win_rate']:.1f}%")
            print(f"  Expected ROI: {opt['roi']:.1f}%")
            print(f"  Sample Size: {opt['total_bets']} bets")
        else:
            # Fallback to highest win rate
            if 'highest_win_rate' in optimal:
                opt = optimal['highest_win_rate']
                print(f"  Recommended Edge: {opt['edge_threshold']}%+")
                print(f"  Recommended Confidence: {opt['confidence_threshold']:.0%}+")
                print(f"  Expected Win Rate: {opt['win_rate']:.1f}%")
                print(f"  Expected ROI: {opt['roi']:.1f}%")

        print()


def main():
    parser = argparse.ArgumentParser(description='Optimize edge and confidence thresholds')
    parser.add_argument('--weeks', type=str, help='Weeks to analyze (e.g., "1-10" or "10")')
    parser.add_argument('--full', action='store_true', help='Analyze all available weeks')
    parser.add_argument('--comprehensive', action='store_true', help='Use comprehensive backtest data (13,915 bets)')
    parser.add_argument('--by-market', action='store_true', help='Analyze by market type')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    # Determine weeks
    if args.full:
        weeks = list(range(1, 11))
    elif args.weeks:
        if '-' in args.weeks:
            start, end = map(int, args.weeks.split('-'))
            weeks = list(range(start, end + 1))
        else:
            weeks = [int(w) for w in args.weeks.split(',')]
    else:
        weeks = [10]

    print("="*80)
    print("EDGE & CONFIDENCE THRESHOLD OPTIMIZATION")
    print("="*80)

    if args.comprehensive:
        print("Using comprehensive backtest data (13,915 bets)")
    else:
        print(f"Analyzing weeks: {weeks}")
    print()

    # Collect all bet results
    bet_results = collect_all_bet_results(weeks if not args.comprehensive else None, use_comprehensive=args.comprehensive)

    if bet_results.empty:
        print("⚠️  No bet results found for analysis")
        return

    print(f"✅ Collected {len(bet_results)} bet results")

    # Test threshold combinations
    print("\nTesting threshold combinations...")
    results_df = test_threshold_combinations(bet_results)

    if results_df.empty:
        print("⚠️  No valid threshold combinations found")
        return

    print(f"✅ Tested {len(results_df)} threshold combinations")

    # Find optimal thresholds
    optimal = find_optimal_thresholds(results_df)

    # Print results
    print_optimization_results(results_df, optimal)

    # Market-specific analysis
    if args.by_market:
        print("\nAnalyzing by market type...")
        market_results = test_by_market(bet_results)
        print_market_specific_results(market_results)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f'reports/edge_confidence_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"\n✅ Results saved to: {output_file}")

    # Save optimal recommendations
    if optimal:
        rec_file = output_file.parent / f"optimal_thresholds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(rec_file, 'w') as f:
            f.write("OPTIMAL THRESHOLD RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            for key, opt in optimal.items():
                f.write(f"{key.upper().replace('_', ' ')}:\n")
                f.write(f"  Edge Threshold: {opt['edge_threshold']}%+\n")
                f.write(f"  Confidence Threshold: {opt['confidence_threshold']:.0%}+\n")
                f.write(f"  Win Rate: {opt['win_rate']:.1f}%\n")
                f.write(f"  ROI: {opt['roi']:.1f}%\n")
                f.write(f"  Sample Size: {opt['total_bets']} bets\n\n")

        print(f"✅ Optimal thresholds saved to: {rec_file}")


if __name__ == '__main__':
    main()
