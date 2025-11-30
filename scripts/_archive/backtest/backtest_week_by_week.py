#!/usr/bin/env python3
"""
Week-by-Week Backtest Validation

Backtests each week individually to:
1. Identify week-specific performance issues
2. Validate calibration consistency across weeks
3. Check data quality by week
4. Ensure readiness for Week 9

This helps catch issues before Week 9 betting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def american_to_prob(american_odds):
    """Convert American odds to implied probability."""
    if pd.isna(american_odds):
        return 0.5
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_brier_score(y_true, y_pred):
    """Calculate Brier score (lower is better)."""
    return np.mean((y_pred - y_true) ** 2)


def backtest_week(df, week_num):
    """Backtest a single week."""
    week_data = df[df['week'] == week_num].copy()

    if len(week_data) == 0:
        return None

    # Calculate metrics
    total_bets = len(week_data)
    wins = week_data['bet_won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0

    # Calculate ROI (assuming $1 bets)
    # Profit = wins * (decimal_odds - 1) - losses * 1
    # For simplicity, assume -110 odds (1.909 decimal)
    profit_per_win = 0.909  # -110 odds
    profit = (wins * profit_per_win) - losses
    roi = (profit / total_bets) * 100 if total_bets > 0 else 0

    # Calculate calibration metrics
    if 'model_prob' in week_data.columns:
        probs = week_data['model_prob'].values
        outcomes = week_data['bet_won'].astype(int).values
        brier = calculate_brier_score(outcomes, probs)
    else:
        brier = None

    # Check win rate by probability bin
    prob_bins = {}
    if 'model_prob' in week_data.columns:
        for bin_min in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
            bin_max = bin_min + 0.1 if bin_min < 0.9 else 1.0
            bin_data = week_data[
                (week_data['model_prob'] >= bin_min) &
                (week_data['model_prob'] < bin_max)
            ]
            if len(bin_data) > 0:
                bin_wins = bin_data['bet_won'].sum()
                bin_win_rate = bin_wins / len(bin_data)
                avg_prob = bin_data['model_prob'].mean()
                prob_bins[f"{bin_min:.0%}-{bin_max:.0%}"] = {
                    'count': len(bin_data),
                    'predicted': avg_prob,
                    'actual': bin_win_rate,
                    'error': abs(bin_win_rate - avg_prob)
                }

    return {
        'week': week_num,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit': profit,
        'roi': roi,
        'brier_score': brier,
        'prob_bins': prob_bins
    }


def validate_week_preparation(week_num, backtest_results):
    """Validate that a week is ready for betting."""
    if week_num not in backtest_results:
        return False, "No backtest data for this week"

    week_result = backtest_results[week_num]

    issues = []

    # Check minimum bets
    if week_result['total_bets'] < 10:
        issues.append(f"Too few bets ({week_result['total_bets']})")

    # Check win rate calibration
    if week_result['brier_score'] and week_result['brier_score'] > 0.25:
        issues.append(f"Poor calibration (Brier: {week_result['brier_score']:.3f})")

    # Check probability bin calibration
    for bin_name, bin_data in week_result['prob_bins'].items():
        if bin_data['error'] > 0.15:  # More than 15% error
            issues.append(f"Poor calibration in {bin_name} bin (error: {bin_data['error']:.1%})")

    if issues:
        return False, "; ".join(issues)

    return True, "Ready"


def main():
    """Run week-by-week backtest validation."""
    print("=" * 80)
    print("📊 WEEK-BY-WEEK BACKTEST VALIDATION")
    print("=" * 80)
    print(f"Purpose: Validate each week individually to ensure Week 9 readiness")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load backtest data
    backtest_file = Path('reports/BACKTEST_WEEKS_1_8_VALIDATION.csv')
    if not backtest_file.exists():
        print(f"❌ Backtest data not found: {backtest_file}")
        print("   Run backtest first or check file location")
        return

    df = pd.read_csv(backtest_file)
    print(f"✅ Loaded {len(df):,} backtest records")

    # Get available weeks
    available_weeks = sorted(df['week'].unique())
    print(f"📅 Available weeks: {available_weeks}\n")

    # Backtest each week
    print("=" * 80)
    print("WEEK-BY-WEEK PERFORMANCE")
    print("=" * 80)
    print()

    results = {}
    for week in available_weeks:
        result = backtest_week(df, week)
        if result:
            results[week] = result

            # Print summary
            print(f"📅 WEEK {week}")
            print("-" * 80)
            print(f"   Bets:        {result['total_bets']:4d}")
            print(f"   Wins:        {result['wins']:4d}")
            print(f"   Losses:      {result['losses']:4d}")
            print(f"   Win Rate:    {result['win_rate']:6.1%}")
            print(f"   Profit:      ${result['profit']:+.2f}")
            print(f"   ROI:         {result['roi']:+6.1f}%")

            if result['brier_score']:
                print(f"   Brier Score: {result['brier_score']:.4f}")

            # Print probability bin calibration
            if result['prob_bins']:
                print(f"\n   📊 Calibration by Probability Bin:")
                for bin_name, bin_data in sorted(result['prob_bins'].items()):
                    print(f"      {bin_name:8s} | Pred: {bin_data['predicted']:5.1%} | "
                          f"Actual: {bin_data['actual']:5.1%} | "
                          f"Error: {bin_data['error']:+5.1%} | "
                          f"Count: {bin_data['count']:3d}")

            print()

    # Overall summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print()

    total_bets = sum(r['total_bets'] for r in results.values())
    total_wins = sum(r['wins'] for r in results.values())
    total_profit = sum(r['profit'] for r in results.values())
    overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

    print(f"Total Bets:     {total_bets:5d}")
    print(f"Total Wins:     {total_wins:5d}")
    print(f"Overall Win Rate: {overall_win_rate:6.1%}")
    print(f"Total Profit:   ${total_profit:+.2f}")
    print(f"Overall ROI:    {overall_roi:+6.1f}%")
    print()

    # Week-by-week ROI trend
    print("📈 ROI Trend by Week:")
    for week in sorted(results.keys()):
        roi = results[week]['roi']
        status = "✅" if roi > 0 else "❌" if roi < -5 else "⚠️"
        print(f"   Week {week}: {roi:+6.1f}% {status}")
    print()

    # Validate Week 9 preparation
    print("=" * 80)
    print("WEEK 9 PREPARATION VALIDATION")
    print("=" * 80)
    print()

    # Use most recent week as proxy for Week 9 readiness
    if results:
        latest_week = max(results.keys())
        latest_result = results[latest_week]

        print(f"Latest Week Available: Week {latest_week}")
        print(f"Win Rate: {latest_result['win_rate']:.1%}")
        print(f"ROI: {latest_result['roi']:+.1f}%")

        if latest_result['brier_score']:
            print(f"Brier Score: {latest_result['brier_score']:.4f}")
            if latest_result['brier_score'] < 0.22:
                print("✅ Calibration is good")
            else:
                print("⚠️  Calibration needs improvement")

        # Check for consistency
        avg_roi = np.mean([r['roi'] for r in results.values()])
        std_roi = np.std([r['roi'] for r in results.values()])

        print(f"\n📊 Consistency Check:")
        print(f"   Average ROI: {avg_roi:+.1f}%")
        print(f"   ROI Std Dev: {std_roi:.1f}%")

        if std_roi > 20:
            print("   ⚠️  High variance - performance inconsistent across weeks")
        else:
            print("   ✅ Performance is consistent across weeks")

        # Predictions for Week 9
        print(f"\n🎯 Week 9 Predictions (based on historical performance):")
        print(f"   Expected Win Rate: {overall_win_rate:.1%}")
        print(f"   Expected ROI: {avg_roi:+.1f}%")
        print(f"   Risk: {'Low' if std_roi < 15 else 'Medium' if std_roi < 25 else 'High'}")

    # Save detailed results
    results_df = pd.DataFrame([
        {
            'week': r['week'],
            'total_bets': r['total_bets'],
            'wins': r['wins'],
            'losses': r['losses'],
            'win_rate': r['win_rate'],
            'profit': r['profit'],
            'roi': r['roi'],
            'brier_score': r['brier_score']
        }
        for r in results.values()
    ])

    output_file = Path('reports/week_by_week_backtest_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved: {output_file}")
    print()

    # Final recommendation
    print("=" * 80)
    print("✅ RECOMMENDATION")
    print("=" * 80)

    if overall_win_rate > 0.65 and overall_roi > 10:
        print("✅ Week 9 READY - Historical performance is strong")
    elif overall_win_rate > 0.60 and overall_roi > 5:
        print("⚠️  Week 9 READY WITH CAUTION - Performance is acceptable")
    else:
        print("❌ Week 9 NOT READY - Review model and calibration")

    print()


if __name__ == '__main__':
    main()
