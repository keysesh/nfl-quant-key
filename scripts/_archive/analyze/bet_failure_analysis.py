#!/usr/bin/env python3
"""
Bet Failure Analysis Tool

Analyzes why bets failed to identify:
- Common failure patterns
- Market-specific failure reasons
- Edge vs actual outcome discrepancies
- Model calibration issues

Usage:
    python scripts/analyze/bet_failure_analysis.py --weeks 1-10
    python scripts/analyze/bet_failure_analysis.py --full
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
)


def analyze_failed_bets(weeks: List[int], season: int = 2025) -> pd.DataFrame:
    """Analyze failed bets in detail."""
    print("="*80)
    print("BET FAILURE ANALYSIS")
    print("="*80)
    print(f"Analyzing weeks: {weeks}")
    print()
    
    all_bet_results = []
    
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
                all_bet_results.append(result)
    
    if not all_bet_results:
        print("⚠️  No bet results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_bet_results)
    
    # Add failure analysis columns
    df['missed_by'] = np.nan
    df['failure_reason'] = ''
    
    for idx, row in df.iterrows():
        if not row['won']:
            line = row.get('line', np.nan)
            actual = row.get('actual_value', np.nan)
            pick = str(row.get('pick', '')).lower()
            
            if pd.notna(line) and pd.notna(actual):
                if 'under' in pick:
                    missed_by = actual - line  # How much over the line
                    df.at[idx, 'missed_by'] = missed_by
                    if missed_by <= 0.5:
                        df.at[idx, 'failure_reason'] = 'Close miss (<0.5)'
                    elif missed_by <= 2.0:
                        df.at[idx, 'failure_reason'] = 'Moderate miss (0.5-2.0)'
                    else:
                        df.at[idx, 'failure_reason'] = 'Large miss (>2.0)'
                elif 'over' in pick:
                    missed_by = line - actual  # How much under the line
                    df.at[idx, 'missed_by'] = missed_by
                    if missed_by <= 0.5:
                        df.at[idx, 'failure_reason'] = 'Close miss (<0.5)'
                    elif missed_by <= 2.0:
                        df.at[idx, 'failure_reason'] = 'Moderate miss (0.5-2.0)'
                    else:
                        df.at[idx, 'failure_reason'] = 'Large miss (>2.0)'
    
    return df


def analyze_failure_patterns(df: pd.DataFrame) -> Dict:
    """Identify patterns in failed bets."""
    failed = df[df['won'] == False].copy()
    
    if failed.empty:
        return {}
    
    patterns = {}
    
    # By market
    patterns['by_market'] = failed.groupby('market').agg({
        'won': 'count',
        'missed_by': ['mean', 'std', 'min', 'max'],
        'edge_pct': 'mean',
        'model_prob': 'mean',
    }).round(2)
    
    # By failure reason
    if 'failure_reason' in failed.columns:
        patterns['by_reason'] = failed.groupby('failure_reason').agg({
            'won': 'count',
            'missed_by': 'mean',
            'edge_pct': 'mean',
        }).round(2)
    
    # By edge threshold
    failed['edge_bin'] = pd.cut(
        failed['edge_pct'],
        bins=[0, 5, 10, 15, 20, 100],
        labels=['0-5%', '5-10%', '10-15%', '15-20%', '20%+']
    )
    
    patterns['by_edge'] = failed.groupby('edge_bin').agg({
        'won': 'count',
        'missed_by': 'mean',
    }).round(2)
    
    # Model projection vs actual
    if 'model_projection' in failed.columns:
        failed['projection_error'] = abs(failed['model_projection'] - failed['actual_value'])
        patterns['projection_accuracy'] = {
            'mean_error': failed['projection_error'].mean(),
            'median_error': failed['projection_error'].median(),
            'std_error': failed['projection_error'].std(),
        }
    
    return patterns


def print_failure_analysis(df: pd.DataFrame, patterns: Dict):
    """Print detailed failure analysis."""
    failed = df[df['won'] == False].copy()
    
    if failed.empty:
        print("✅ No failed bets to analyze!")
        return
    
    print("\n" + "="*80)
    print(f"FAILED BETS ANALYSIS ({len(failed)} failures)")
    print("="*80)
    print()
    
    # Overall failure rate
    total_bets = len(df)
    failure_rate = len(failed) / total_bets * 100
    print(f"Overall Failure Rate: {failure_rate:.1f}% ({len(failed)}/{total_bets})")
    print()
    
    # By market
    if 'by_market' in patterns:
        print("FAILURES BY MARKET:")
        print("-"*80)
        market_failures = patterns['by_market']
        print(f"{'Market':<30} {'Failures':<12} {'Avg Miss':<12} {'Avg Edge':<12}")
        print("-"*80)
        
        for market, row in market_failures.iterrows():
            print(
                f"{market:<30} {int(row[('won', 'count')]):<12} "
                f"{row[('missed_by', 'mean')]:>10.2f}  {row[('edge_pct', 'mean')]:>10.1f}%"
            )
        print()
    
    # By failure reason
    if 'by_reason' in patterns:
        print("FAILURES BY SEVERITY:")
        print("-"*80)
        reason_failures = patterns['by_reason']
        print(f"{'Reason':<25} {'Count':<10} {'Avg Miss':<12} {'Avg Edge':<12}")
        print("-"*80)
        
        for reason, row in reason_failures.iterrows():
            print(
                f"{reason:<25} {int(row['won']):<10} "
                f"{row['missed_by']:>10.2f}  {row['edge_pct']:>10.1f}%"
            )
        print()
    
    # By edge threshold
    if 'by_edge' in patterns:
        print("FAILURES BY EDGE THRESHOLD:")
        print("-"*80)
        edge_failures = patterns['by_edge']
        print(f"{'Edge Range':<15} {'Failures':<12} {'Avg Miss':<12}")
        print("-"*80)
        
        for edge_bin, row in edge_failures.iterrows():
            print(
                f"{edge_bin:<15} {int(row['won']):<12} {row['missed_by']:>10.2f}"
            )
        print()
    
    # Model projection accuracy
    if 'projection_accuracy' in patterns:
        print("MODEL PROJECTION ACCURACY (Failed Bets):")
        print("-"*80)
        accuracy = patterns['projection_accuracy']
        print(f"Mean Error: {accuracy['mean_error']:.2f}")
        print(f"Median Error: {accuracy['median_error']:.2f}")
        print(f"Std Deviation: {accuracy['std_error']:.2f}")
        print()
    
    # Detailed failure examples
    print("EXAMPLE FAILED BETS:")
    print("-"*80)
    print(f"{'Player/Game':<25} {'Market':<20} {'Pick':<15} {'Line':<8} {'Actual':<10} {'Miss':<10} {'Edge':<10}")
    print("-"*80)
    
    for idx, row in failed.head(10).iterrows():
        player = row.get('player', row.get('game', 'N/A'))
        market = row.get('market', 'N/A')
        pick = row.get('pick', 'N/A')
        line = row.get('line', np.nan)
        actual = row.get('actual_value', np.nan)
        missed = row.get('missed_by', np.nan)
        edge = row.get('edge_pct', np.nan)
        
        print(
            f"{str(player)[:24]:<25} {market[:19]:<20} {pick[:14]:<15} "
            f"{line:>7.1f}  {actual:>9.1f}  {missed:>9.2f}  {edge:>9.1f}%"
        )
    print()


def generate_insights(df: pd.DataFrame, patterns: Dict) -> str:
    """Generate actionable insights from failure analysis."""
    failed = df[df['won'] == False].copy()
    
    if failed.empty:
        return "✅ No failures to analyze - all bets won!"
    
    insights = []
    insights.append("="*80)
    insights.append("KEY INSIGHTS FROM FAILURE ANALYSIS")
    insights.append("="*80)
    insights.append("")
    
    # Market-specific insights
    if 'by_market' in patterns:
        market_failures = patterns['by_market']
        
        # Find markets with high failure rates
        total_by_market = df.groupby('market').size()
        failure_by_market = failed.groupby('market').size()
        failure_rates = (failure_by_market / total_by_market * 100).sort_values(ascending=False)
        
        insights.append("⚠️  MARKETS WITH HIGHEST FAILURE RATES:")
        for market, rate in failure_rates.head(3).items():
            insights.append(f"  {market}: {rate:.1f}% failure rate")
        insights.append("")
        
        # Find markets with large misses
        large_misses = market_failures[market_failures[('missed_by', 'mean')] > 2.0]
        if len(large_misses) > 0:
            insights.append("⚠️  MARKETS WITH LARGE AVERAGE MISSES (>2.0):")
            for market, row in large_misses.iterrows():
                insights.append(f"  {market}: Avg miss of {row[('missed_by', 'mean')]:.2f}")
            insights.append("")
    
    # Edge threshold insights
    if 'by_edge' in patterns:
        edge_failures = patterns['by_edge']
        
        # Find edge ranges with high failure rates
        total_by_edge = df.groupby(pd.cut(df['edge_pct'], bins=[0, 5, 10, 15, 20, 100], labels=['0-5%', '5-10%', '10-15%', '15-20%', '20%+'])).size()
        failure_by_edge = failed.groupby('edge_bin').size()
        edge_failure_rates = (failure_by_edge / total_by_edge * 100).sort_values(ascending=False)
        
        insights.append("📊 FAILURE RATES BY EDGE THRESHOLD:")
        for edge_bin, rate in edge_failure_rates.items():
            insights.append(f"  {edge_bin}: {rate:.1f}% failure rate")
        insights.append("")
    
    # Recommendations
    insights.append("💡 RECOMMENDATIONS:")
    insights.append("-"*80)
    
    # Check if high-edge bets fail more
    if 'by_edge' in patterns:
        high_edge_failures = edge_failures.loc[edge_failures.index.isin(['15-20%', '20%+']), 'won'].sum() if '15-20%' in edge_failures.index or '20%+' in edge_failures.index else 0
        if high_edge_failures > 0:
            insights.append("  ⚠️  High-edge bets (>15%) are failing - may indicate overconfidence")
            insights.append("     Consider: Lower edge threshold or improve model calibration")
    
    # Check projection accuracy
    if 'projection_accuracy' in patterns:
        mean_error = patterns['projection_accuracy']['mean_error']
        if mean_error > 5.0:
            insights.append(f"  ⚠️  Large projection errors (mean: {mean_error:.2f})")
            insights.append("     Consider: Improve model accuracy or increase uncertainty estimates")
    
    insights.append("")
    
    return "\n".join(insights)


def main():
    parser = argparse.ArgumentParser(description='Analyze bet failures')
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
        weeks = [10]
    
    # Analyze
    bet_results = analyze_failed_bets(weeks)
    
    if bet_results.empty:
        print("⚠️  No bet results found for analysis")
        return
    
    print(f"✅ Analyzed {len(bet_results)} bets")
    
    # Calculate patterns
    patterns = analyze_failure_patterns(bet_results)
    
    # Print analysis
    print_failure_analysis(bet_results, patterns)
    
    # Generate insights
    insights = generate_insights(bet_results, patterns)
    print(insights)
    
    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f'reports/bet_failure_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("BET FAILURE ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Weeks Analyzed: {weeks}")
    report.append(f"Total Bets: {len(bet_results)}")
    report.append(f"Failed Bets: {len(bet_results[bet_results['won'] == False])}")
    report.append("")
    report.append(insights)
    
    output_file.write_text("\n".join(report))
    print(f"\n✅ Report saved to: {output_file}")


if __name__ == '__main__':
    main()

