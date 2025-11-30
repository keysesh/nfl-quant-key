#!/usr/bin/env python3
"""
Market Performance Analysis Tool

Analyzes historical performance by market type to identify:
- Which markets have highest win rates
- Optimal edge thresholds per market
- ROI by market
- Confidence levels that work best

Usage:
    python scripts/analyze/market_performance_analysis.py --weeks 1-10
    python scripts/analyze/market_performance_analysis.py --full
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.validate.validate_framework_2025 import (
    load_recommendations,
    load_actual_stats,
    load_game_results,
    match_player_bet_to_actual,
    match_game_bet_to_actual,
    american_to_decimal,
    calculate_profit,
)


def analyze_market_performance(weeks: List[int], season: int = 2025) -> pd.DataFrame:
    """Analyze performance by market type."""
    print("="*80)
    print("MARKET PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Analyzing weeks: {weeks}")
    print()

    all_bet_results = []

    for week in weeks:
        # Load recommendations
        recommendations = load_recommendations(week)
        if recommendations.empty:
            continue

        # Load actuals
        actual_stats = load_actual_stats(week, season)
        game_results = load_game_results(week, season)

        if actual_stats.empty and game_results.empty:
            continue

        # Match bets to actuals
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
                all_bet_results.append(result)

    if not all_bet_results:
        print("⚠️  No bet results found")
        return pd.DataFrame()

    return pd.DataFrame(all_bet_results)


def calculate_market_metrics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive metrics by market."""
    if df.empty:
        return {}

    metrics = {}

    # Overall by market
    market_summary = df.groupby('market').agg({
        'won': ['count', 'sum', 'mean'],
        'edge_pct': ['mean', 'std'],
        'model_prob': ['mean', 'std'],
    }).round(2)

    market_summary.columns = ['bets', 'wins', 'win_rate', 'avg_edge', 'std_edge', 'avg_prob', 'std_prob']
    market_summary['losses'] = market_summary['bets'] - market_summary['wins']

    # Calculate ROI
    df['profit'] = df.apply(
        lambda row: calculate_profit(row['odds'], 100.0) if row['won'] else -100.0,
        axis=1
    )

    market_roi = df.groupby('market')['profit'].agg(['sum', 'mean']).round(2)
    market_summary['total_profit'] = market_roi['sum']
    market_summary['avg_profit'] = market_roi['mean']
    market_summary['roi_pct'] = (market_summary['total_profit'] / (market_summary['bets'] * 100) * 100).round(2)

    metrics['by_market'] = market_summary

    # By edge threshold
    edge_bins = [0, 5, 10, 15, 20, 30, 100]
    edge_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30%+']

    df['edge_bin'] = pd.cut(df['edge_pct'], bins=edge_bins, labels=edge_labels)

    edge_summary = df.groupby('edge_bin').agg({
        'won': ['count', 'sum', 'mean'],
        'profit': 'sum',
    }).round(2)

    edge_summary.columns = ['bets', 'wins', 'win_rate', 'total_profit']
    edge_summary['roi_pct'] = (edge_summary['total_profit'] / (edge_summary['bets'] * 100) * 100).round(2)

    metrics['by_edge'] = edge_summary

    # By probability bin
    prob_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob_labels = ['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90%+']

    df['prob_bin'] = pd.cut(df['model_prob'], bins=prob_bins, labels=prob_labels)

    prob_summary = df.groupby('prob_bin').agg({
        'won': ['count', 'sum', 'mean'],
        'profit': 'sum',
    }).round(2)

    prob_summary.columns = ['bets', 'wins', 'win_rate', 'total_profit']
    prob_summary['roi_pct'] = (prob_summary['total_profit'] / (prob_summary['bets'] * 100) * 100).round(2)

    metrics['by_probability'] = prob_summary

    # Combined: Market + Edge threshold
    market_edge = df.groupby(['market', 'edge_bin']).agg({
        'won': ['count', 'sum', 'mean'],
        'profit': 'sum',
    }).round(2)

    market_edge.columns = ['bets', 'wins', 'win_rate', 'total_profit']
    market_edge['roi_pct'] = (market_edge['total_profit'] / (market_edge['bets'] * 100) * 100).round(2)

    metrics['by_market_edge'] = market_edge

    return metrics


def print_market_analysis(metrics: Dict):
    """Print comprehensive market analysis."""
    print("\n" + "="*80)
    print("PERFORMANCE BY MARKET TYPE")
    print("="*80)
    print()

    if 'by_market' not in metrics:
        print("⚠️  No market data available")
        return

    market_df = metrics['by_market']

    print(f"{'Market':<30} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'Avg Edge':<12} {'ROI':<10} {'Profit':<12}")
    print("-"*100)

    # Sort by win rate
    market_df_sorted = market_df.sort_values('win_rate', ascending=False)

    for market, row in market_df_sorted.iterrows():
        print(
            f"{market:<30} {int(row['bets']):<8} {int(row['wins']):<8} "
            f"{row['win_rate']*100:>10.1f}%  {row['avg_edge']:>10.1f}%  "
            f"{row['roi_pct']:>8.1f}%  ${row['total_profit']:>10,.2f}"
        )

    print()

    # Best markets
    print("🏆 TOP MARKETS BY WIN RATE:")
    top_markets = market_df_sorted.head(3)
    for market, row in top_markets.iterrows():
        print(f"  {market}: {row['win_rate']*100:.1f}% win rate, {row['roi_pct']:.1f}% ROI")

    print()

    # Performance by edge threshold
    if 'by_edge' in metrics:
        print("="*80)
        print("PERFORMANCE BY EDGE THRESHOLD")
        print("="*80)
        print()

        edge_df = metrics['by_edge']
        print(f"{'Edge Range':<15} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'ROI':<10} {'Profit':<12}")
        print("-"*80)

        for edge_bin, row in edge_df.iterrows():
            print(
                f"{edge_bin:<15} {int(row['bets']):<8} {int(row['wins']):<8} "
                f"{row['win_rate']*100:>10.1f}%  {row['roi_pct']:>8.1f}%  ${row['total_profit']:>10,.2f}"
            )

        print()

    # Performance by probability
    if 'by_probability' in metrics:
        print("="*80)
        print("PERFORMANCE BY MODEL CONFIDENCE")
        print("="*80)
        print()

        prob_df = metrics['by_probability']
        print(f"{'Confidence':<15} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'ROI':<10} {'Profit':<12}")
        print("-"*80)

        for prob_bin, row in prob_df.iterrows():
            print(
                f"{prob_bin:<15} {int(row['bets']):<8} {int(row['wins']):<8} "
                f"{row['win_rate']*100:>10.1f}%  {row['roi_pct']:>8.1f}%  ${row['total_profit']:>10,.2f}"
            )

        print()

    # Optimal thresholds per market
    if 'by_market_edge' in metrics:
        print("="*80)
        print("OPTIMAL EDGE THRESHOLDS BY MARKET")
        print("="*80)
        print()

        market_edge_df = metrics['by_market_edge']

        for market in market_df_sorted.index[:5]:  # Top 5 markets
            market_data = market_edge_df.loc[market]
            profitable = market_data[market_data['roi_pct'] > 0]

            if len(profitable) > 0:
                best_edge = profitable.loc[profitable['win_rate'].idxmax()]
                print(f"{market}:")
                print(f"  Best Edge Range: {best_edge.name}")
                print(f"  Win Rate: {best_edge['win_rate']*100:.1f}%")
                print(f"  ROI: {best_edge['roi_pct']:.1f}%")
                print(f"  Sample Size: {int(best_edge['bets'])} bets")
                print()


def generate_recommendations_report(metrics: Dict) -> str:
    """Generate actionable recommendations."""
    lines = []
    lines.append("="*80)
    lines.append("ACTIONABLE RECOMMENDATIONS")
    lines.append("="*80)
    lines.append("")

    if 'by_market' not in metrics:
        lines.append("⚠️  Insufficient data for recommendations")
        return "\n".join(lines)

    market_df = metrics['by_market']

    # Tier 1: High confidence markets
    tier1 = market_df[
        (market_df['win_rate'] >= 0.60) &
        (market_df['roi_pct'] >= 10) &
        (market_df['bets'] >= 5)
    ].sort_values('win_rate', ascending=False)

    if len(tier1) > 0:
        lines.append("✅ TIER 1: HIGH CONFIDENCE MARKETS (Always Bet)")
        lines.append("-"*80)
        for market, row in tier1.iterrows():
            lines.append(f"  {market}:")
            lines.append(f"    Win Rate: {row['win_rate']*100:.1f}%")
            lines.append(f"    ROI: {row['roi_pct']:.1f}%")
            lines.append(f"    Recommended Edge: {max(5, row['avg_edge']):.0f}%+")
            lines.append(f"    Sample Size: {int(row['bets'])} bets")
            lines.append()

    # Tier 2: Medium confidence
    tier2 = market_df[
        (market_df['win_rate'] >= 0.55) &
        (market_df['roi_pct'] >= 5) &
        (market_df['bets'] >= 3)
    ].sort_values('win_rate', ascending=False)

    if len(tier2) > 0:
        lines.append("⚠️  TIER 2: MEDIUM CONFIDENCE (Be Selective)")
        lines.append("-"*80)
        for market, row in tier2.iterrows():
            if market in tier1.index:
                continue
            lines.append(f"  {market}:")
            lines.append(f"    Win Rate: {row['win_rate']*100:.1f}%")
            lines.append(f"    ROI: {row['roi_pct']:.1f}%")
            lines.append(f"    Recommended Edge: {max(10, row['avg_edge']*1.5):.0f}%+")
            lines.append(f"    Sample Size: {int(row['bets'])} bets")
            lines.append()

    # Tier 3: Avoid
    tier3 = market_df[
        (market_df['win_rate'] < 0.52) |
        (market_df['roi_pct'] < 0)
    ].sort_values('win_rate', ascending=True)

    if len(tier3) > 0:
        lines.append("❌ TIER 3: AVOID (Do Not Bet)")
        lines.append("-"*80)
        for market, row in tier3.iterrows():
            lines.append(f"  {market}:")
            lines.append(f"    Win Rate: {row['win_rate']*100:.1f}%")
            lines.append(f"    ROI: {row['roi_pct']:.1f}%")
            lines.append(f"    Reason: Below breakeven")
            lines.append()

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze market performance')
    parser.add_argument('--weeks', type=str, help='Weeks to analyze (e.g., "1-10" or "10")')
    parser.add_argument('--full', action='store_true', help='Analyze all available weeks')
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
        weeks = [10]  # Default: current week

    # Analyze
    bet_results = analyze_market_performance(weeks)

    if bet_results.empty:
        print("⚠️  No bet results found for analysis")
        return

    print(f"✅ Analyzed {len(bet_results)} bets across {len(weeks)} weeks")

    # Calculate metrics
    metrics = calculate_market_metrics(bet_results)

    # Print analysis
    print_market_analysis(metrics)

    # Generate recommendations
    recommendations = generate_recommendations_report(metrics)
    print(recommendations)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f'reports/market_performance_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("MARKET PERFORMANCE ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Weeks Analyzed: {weeks}")
    report.append(f"Total Bets: {len(bet_results)}")
    report.append("")
    report.append(str(metrics['by_market']))
    report.append("")
    report.append(recommendations)

    output_file.write_text("\n".join(report))
    print(f"\n✅ Report saved to: {output_file}")


if __name__ == '__main__':
    main()
