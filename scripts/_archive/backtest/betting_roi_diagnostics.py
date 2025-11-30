"""
Betting ROI Diagnostics - Deep dive into model performance issues

Analyzes:
- Why RB rushing yards shows negative ROI despite good coverage
- Model probability calibration (predicted vs actual win rates)
- Edge distribution and bet selection patterns
- Missing stat types (receiving_yards, TDs)
- Optimal edge thresholds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List


def analyze_calibration(bets_df: pd.DataFrame):
    """
    Analyze model probability calibration.

    Well-calibrated model: When model says 60% prob, actual win rate should be ~60%
    """
    print("="*80)
    print("PROBABILITY CALIBRATION ANALYSIS")
    print("="*80)
    print()

    # Bin probabilities
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    bets_df['prob_bin'] = pd.cut(bets_df['prob_model'], bins=bins)

    print("Model Probability vs Actual Win Rate:")
    print()

    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        type_bets = bets_df[bets_df['stat_type'] == stat_type]
        if len(type_bets) == 0:
            continue

        print(f"📊 {stat_type.upper()}:")

        calibration = type_bets.groupby('prob_bin').agg({
            'won': ['count', 'sum', 'mean'],
            'prob_model': 'mean',
            'profit': 'sum'
        })

        if len(calibration) > 0:
            print(f"  {'Prob Range':<15} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Avg Prob':<10} {'Profit':<10}")
            print(f"  {'-'*70}")

            for idx, row in calibration.iterrows():
                n_bets = int(row[('won', 'count')])
                n_wins = int(row[('won', 'sum')])
                win_rate = row[('won', 'mean')] * 100
                avg_prob = row[('prob_model', 'mean')] * 100
                profit = row[('profit', 'sum')]

                calibration_diff = win_rate - avg_prob
                diff_indicator = "✅" if abs(calibration_diff) < 5 else "⚠️" if abs(calibration_diff) < 10 else "❌"

                print(f"  {str(idx):<15} {n_bets:<6} {n_wins:<6} {win_rate:6.1f}% {avg_prob:8.1f}% ${profit:>8.0f} {diff_indicator}")

        print()


def analyze_edge_distribution(bets_df: pd.DataFrame):
    """Analyze distribution of edges and which ones are profitable."""
    print("="*80)
    print("EDGE DISTRIBUTION ANALYSIS")
    print("="*80)
    print()

    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        type_bets = bets_df[bets_df['stat_type'] == stat_type]
        if len(type_bets) == 0:
            continue

        print(f"📊 {stat_type.upper()}:")

        # Bin by edge
        edge_bins = [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
        type_bets['edge_bin'] = pd.cut(type_bets['edge'], bins=edge_bins)

        edge_analysis = type_bets.groupby('edge_bin').agg({
            'won': ['count', 'sum', 'mean'],
            'edge': 'mean',
            'profit': 'sum'
        })

        print(f"  {'Edge Range':<15} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Avg Edge':<10} {'Profit':<10} {'ROI%':<8}")
        print(f"  {'-'*85}")

        for idx, row in edge_analysis.iterrows():
            n_bets = int(row[('won', 'count')])
            n_wins = int(row[('won', 'sum')])
            win_rate = row[('won', 'mean')] * 100
            avg_edge = row[('edge', 'mean')] * 100
            profit = row[('profit', 'sum')]
            roi = (profit / (n_bets * 100)) * 100 if n_bets > 0 else 0

            print(f"  {str(idx):<15} {n_bets:<6} {n_wins:<6} {win_rate:6.1f}% {avg_edge:8.1f}% ${profit:>8.0f} {roi:>6.1f}%")

        print()


def analyze_over_under_bias(bets_df: pd.DataFrame):
    """Analyze over/under selection bias."""
    print("="*80)
    print("OVER/UNDER BET SELECTION ANALYSIS")
    print("="*80)
    print()

    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        type_bets = bets_df[bets_df['stat_type'] == stat_type]
        if len(type_bets) == 0:
            continue

        print(f"📊 {stat_type.upper()}:")

        side_analysis = type_bets.groupby('side').agg({
            'won': ['count', 'sum', 'mean'],
            'edge': 'mean',
            'profit': 'sum'
        })

        print(f"  {'Side':<8} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Avg Edge':<10} {'Profit':<10} {'ROI%':<8}")
        print(f"  {'-'*70}")

        for side, row in side_analysis.iterrows():
            n_bets = int(row[('won', 'count')])
            n_wins = int(row[('won', 'sum')])
            win_rate = row[('won', 'mean')] * 100
            avg_edge = row[('edge', 'mean')] * 100
            profit = row[('profit', 'sum')]
            roi = (profit / (n_bets * 100)) * 100 if n_bets > 0 else 0

            bias_indicator = "⚠️ HEAVY BIAS" if n_bets > type_bets.shape[0] * 0.8 else ""

            print(f"  {side:<8} {n_bets:<6} {n_wins:<6} {win_rate:6.1f}% {avg_edge:8.1f}% ${profit:>8.0f} {roi:>6.1f}% {bias_indicator}")

        print()


def analyze_missing_stat_types():
    """Analyze why receiving_yards and TDs don't appear in results."""
    print("="*80)
    print("MISSING STAT TYPES INVESTIGATION")
    print("="*80)
    print()

    # Check predictions file
    pred_file = Path("reports/v3_backtest_predictions.csv")
    if not pred_file.exists():
        print("❌ Predictions file not found")
        return

    preds = pd.read_csv(pred_file)

    print("📊 Prediction Coverage by Stat Type:")
    stat_counts = preds['stat_type'].value_counts()
    for stat_type, count in stat_counts.items():
        print(f"  {stat_type}: {count} predictions")

    print()

    # Check historical props
    print("📊 Historical Props Coverage:")
    from pathlib import Path
    import glob

    all_props = []
    for week in range(1, 9):
        props_file = Path(f"data/historical/backfill/player_props_history_2025*T000000Z.csv")
        matching_files = glob.glob(str(props_file))

        if matching_files:
            df = pd.read_csv(matching_files[0])
            all_props.append(df)

    if all_props:
        props_df = pd.concat(all_props, ignore_index=True)
        market_counts = props_df['market'].value_counts()

        print("\n  Historical Props by Market:")
        for market, count in market_counts.items():
            print(f"    {market}: {count} props")

    print()

    # Map to stat types
    PROP_TYPE_MAP = {
        'player_pass_yds': 'passing_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rec_yds': 'receiving_yards',
    }

    print("📊 Why receiving_yards might be missing:")
    print("  Possible reasons:")
    print("  1. Historical props file doesn't have 'player_rec_yds' market")
    print("  2. Player name matching failed (different formats)")
    print("  3. Edge threshold too high (no receiving bets > 5% edge)")
    print()


def test_edge_thresholds(bets_df: pd.DataFrame):
    """Test different edge thresholds to find optimal strategy."""
    print("="*80)
    print("EDGE THRESHOLD OPTIMIZATION")
    print("="*80)
    print()

    # Load all potential bets (before filtering by edge)
    # This requires re-running analysis with edge threshold = 0
    print("⚠️  Note: This analysis uses current bets (5% edge threshold)")
    print("    To test lower thresholds, re-run betting_roi_analysis.py with --min-edge 0.01")
    print()

    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        type_bets = bets_df[bets_df['stat_type'] == stat_type]
        if len(type_bets) == 0:
            continue

        print(f"📊 {stat_type.upper()}:")

        # Test different thresholds
        thresholds = [0.05, 0.07, 0.10, 0.15, 0.20]

        print(f"  {'Threshold':<12} {'Bets':<6} {'Win%':<8} {'Profit':<10} {'ROI%':<8}")
        print(f"  {'-'*50}")

        for threshold in thresholds:
            filtered = type_bets[type_bets['edge'] >= threshold]

            if len(filtered) == 0:
                print(f"  {threshold*100:>5.0f}% {0:<11} -")
                continue

            n_bets = len(filtered)
            win_rate = filtered['won'].mean() * 100
            profit = filtered['profit'].sum()
            roi = (profit / (n_bets * 100)) * 100

            print(f"  {threshold*100:>5.0f}% {n_bets:<11} {win_rate:6.1f}% ${profit:>8.0f} {roi:>6.1f}%")

        print()


def main():
    # Load detailed results
    results_file = Path("reports/betting_roi_detailed.csv")

    if not results_file.exists():
        print("❌ Detailed results file not found")
        print("Run: python scripts/backtest/betting_roi_analysis.py first")
        return

    bets_df = pd.read_csv(results_file)

    print("\n")
    print("="*80)
    print("BETTING ROI DIAGNOSTICS")
    print("="*80)
    print(f"\nAnalyzing {len(bets_df)} bets from weeks 1-8")
    print(f"Stat types: {bets_df['stat_type'].unique()}")
    print()

    # Run all analyses
    analyze_over_under_bias(bets_df)
    analyze_calibration(bets_df)
    analyze_edge_distribution(bets_df)
    test_edge_thresholds(bets_df)
    analyze_missing_stat_types()

    print("\n" + "="*80)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*80)
    print("""
Based on the diagnostic analysis above:

1. **Over/Under Bias**: Check if model heavily favors one side
   - If 80%+ of bets are UNDER, model may be systematically underestimating

2. **Calibration Issues**: Compare predicted probability to actual win rate
   - If model says 70% but wins 55%, model is overconfident
   - This means edges are inflated (not real)

3. **Edge Distribution**: Which edge ranges are profitable?
   - May need to raise threshold above 5% for rushing yards
   - Passing yards working well suggests it's stat-specific

4. **Missing Receiving Yards**: Investigate why no receiving props matched
   - Likely prop market naming issue or no props available

5. **TDs**: Check if TD predictions exist in backtest output
   - May need to add TD market support to ROI analysis
    """)


if __name__ == '__main__':
    main()
