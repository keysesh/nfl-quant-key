#!/usr/bin/env python3
"""
Analyze why receptions (+7% ROI) work but reception yards (-1% ROI) don't

Key questions:
1. Is the model overconfident on yards?
2. Is variance underestimated for yards?
3. Are there specific player types where yards fail?
4. Is yards-per-reception (YPR) the issue?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_backtest_data():
    """Load historical simulated props for analysis"""
    props_file = Path("data/calibration/historical_props_simulated.parquet")
    df = pd.read_parquet(props_file)

    # Filter to reception-related markets
    rec_df = df[df['market'].isin(['player_receptions', 'player_reception_yds'])].copy()
    rec_df = rec_df[rec_df['model_prob_raw'].notna()]

    return rec_df

def analyze_calibration_quality(df):
    """Compare calibration between receptions and reception yards"""

    print("=" * 80)
    print("CALIBRATION QUALITY COMPARISON")
    print("=" * 80)
    print()

    for market in ['player_receptions', 'player_reception_yds']:
        market_df = df[df['market'] == market]

        print(f"\n{market}:")
        print(f"  Total props: {len(market_df):,}")

        # Expected vs actual win rate
        expected = market_df['model_prob_raw'].mean()
        actual = market_df['bet_won'].mean()
        bias = (expected - actual) * 100

        print(f"  Expected win rate: {expected*100:.1f}%")
        print(f"  Actual win rate: {actual*100:.1f}%")
        confidence_label = "OVERCONFIDENT" if bias > 0 else "UNDERCONFIDENT"
        print(f"  Bias: {bias:+.1f}% ({confidence_label})")

        # Analyze by confidence bins
        print(f"\n  By confidence level:")
        bins = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
        for low, high in bins:
            bin_df = market_df[(market_df['model_prob_raw'] >= low) &
                              (market_df['model_prob_raw'] < high)]
            if len(bin_df) > 0:
                exp = bin_df['model_prob_raw'].mean()
                act = bin_df['bet_won'].mean()
                diff = (exp - act) * 100
                print(f"    {low:.0%}-{high:.0%}: n={len(bin_df):5,} | Exp={exp*100:.1f}% Act={act*100:.1f}% Diff={diff:+.1f}%")

def analyze_variance(df):
    """Check if variance is properly estimated"""

    print("\n" + "=" * 80)
    print("VARIANCE ANALYSIS")
    print("=" * 80)
    print()

    for market in ['player_receptions', 'player_reception_yds']:
        market_df = df[df['market'] == market]

        print(f"\n{market}:")

        # Model's predicted variance
        model_std = market_df['model_std'].mean()

        # Actual variance (difference between projection and actual)
        market_df['error'] = market_df['actual_value'] - market_df['model_projection']
        actual_std = market_df['error'].std()

        print(f"  Model predicted std: {model_std:.2f}")
        print(f"  Actual error std: {actual_std:.2f}")
        print(f"  Ratio (actual/predicted): {actual_std/model_std:.2f}x")

        if actual_std > model_std * 1.2:
            print(f"  ⚠️  Model UNDERESTIMATES variance (overconfident)")
        elif actual_std < model_std * 0.8:
            print(f"  ⚠️  Model OVERESTIMATES variance (underconfident)")
        else:
            print(f"  ✅ Variance estimation reasonable")

def analyze_by_position(df):
    """Check if specific positions/roles are problematic"""

    print("\n" + "=" * 80)
    print("PERFORMANCE BY POSITION")
    print("=" * 80)
    print()

    for market in ['player_receptions', 'player_reception_yds']:
        market_df = df[df['market'] == market]

        print(f"\n{market}:")

        position_stats = market_df.groupby('position').agg({
            'bet_won': 'mean',
            'model_prob_raw': 'mean',
            'player_name': 'count'
        }).reset_index()
        position_stats.columns = ['position', 'actual_win_rate', 'expected_win_rate', 'count']
        position_stats['bias'] = (position_stats['expected_win_rate'] - position_stats['actual_win_rate']) * 100

        for _, row in position_stats.iterrows():
            if row['count'] >= 100:  # Only show positions with enough data
                print(f"  {row['position']}: n={int(row['count']):5,} | "
                      f"Exp={row['expected_win_rate']*100:.1f}% Act={row['actual_win_rate']*100:.1f}% "
                      f"Bias={row['bias']:+.1f}%")

def analyze_ypr_issue(df):
    """Analyze yards-per-reception variance"""

    print("\n" + "=" * 80)
    print("YARDS-PER-RECEPTION ANALYSIS")
    print("=" * 80)
    print()

    # Filter to props where we have both receptions and yards
    rec_df = df[df['market'] == 'player_receptions'].copy()
    yds_df = df[df['market'] == 'player_reception_yds'].copy()

    # Merge on player/week/season
    merged = rec_df.merge(
        yds_df,
        on=['player_name', 'season', 'week'],
        suffixes=('_rec', '_yds')
    )

    if len(merged) > 0:
        # Calculate actual YPR
        merged['actual_ypr'] = merged['actual_value_yds'] / merged['actual_value_rec'].clip(lower=1)

        # Calculate implied YPR from model
        merged['model_ypr'] = merged['model_projection_yds'] / merged['model_projection_rec'].clip(lower=1)

        print(f"Props with both receptions and yards: {len(merged):,}")
        print()
        print(f"Model YPR:")
        print(f"  Mean: {merged['model_ypr'].mean():.2f} yards/reception")
        print(f"  Std: {merged['model_ypr'].std():.2f}")
        print()
        print(f"Actual YPR:")
        print(f"  Mean: {merged['actual_ypr'].mean():.2f} yards/reception")
        print(f"  Std: {merged['actual_ypr'].std():.2f}")
        print()

        # Check if variance in YPR is underestimated
        ypr_error = merged['actual_ypr'] - merged['model_ypr']
        print(f"YPR Prediction Error:")
        print(f"  Mean error: {ypr_error.mean():+.2f} yards/rec")
        print(f"  Std error: {ypr_error.std():.2f} yards/rec")

        if abs(ypr_error.mean()) > 1.0:
            print(f"  ⚠️  Systematic bias in YPR prediction!")

def main():
    print("=" * 80)
    print("ANALYZING: WHY RECEPTIONS WORK BUT RECEPTION YARDS DON'T")
    print("=" * 80)
    print()

    # Load data
    print("Loading historical simulation data...")
    df = load_backtest_data()
    print(f"Loaded {len(df):,} props")
    print()

    # Run analyses
    analyze_calibration_quality(df)
    analyze_variance(df)
    analyze_by_position(df)
    analyze_ypr_issue(df)

    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Based on the analysis above, the likely issues are:")
    print()
    print("1. VARIANCE UNDERESTIMATION")
    print("   - Reception yards have higher game-to-game variance")
    print("   - Model may not capture YPR volatility well")
    print()
    print("2. POSITION-SPECIFIC ISSUES")
    print("   - Check if certain positions (WR vs TE vs RB) perform worse")
    print("   - May need position-specific YPR models")
    print()
    print("3. MISSING FEATURES")
    print("   - Opponent secondary quality (CB rankings)")
    print("   - Target depth (deep threats vs possession receivers)")
    print("   - Weather conditions")
    print()
    print("Next steps:")
    print("  1. Increase variance in reception yards simulator")
    print("  2. Add opponent secondary quality feature")
    print("  3. Consider separate models for WR/TE/RB receiving yards")
    print()

if __name__ == "__main__":
    main()
