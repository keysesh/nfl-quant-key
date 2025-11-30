#!/usr/bin/env python3
"""
Identify sources of model overconfidence.

This script analyzes when the model makes extreme predictions and whether
they're justified or indicate systematic overconfidence.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def analyze_extreme_predictions(df: pd.DataFrame):
    """Identify patterns in overly confident predictions."""
    print("=" * 80)
    print("ANALYZING MODEL OVERCONFIDENCE")
    print("=" * 80)
    print()

    # Define confidence buckets
    df['confidence_bucket'] = pd.cut(
        df['model_prob'],
        bins=[0, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 1.0],
        labels=['Very Low (0-20%)', 'Low (20-30%)', 'Below Avg (30-40%)',
                'Slightly Below (40-45%)', 'Neutral (45-55%)',
                'Slightly Above (55-60%)', 'Above Avg (60-70%)',
                'High (70-80%)', 'Very High (80-100%)']
    )

    print("Win rate by confidence level:")
    print()
    print(f"{'Confidence Bucket':<25s} | {'Count':>6s} | {'Pred':>7s} | {'Actual':>7s} | {'Error':>7s}")
    print("-" * 80)

    for bucket in df['confidence_bucket'].cat.categories:
        bucket_df = df[df['confidence_bucket'] == bucket]
        if len(bucket_df) == 0:
            continue

        count = len(bucket_df)
        mean_pred = bucket_df['model_prob'].mean()
        actual_rate = bucket_df['bet_outcome'].mean()
        error = actual_rate - mean_pred

        status = "✓" if abs(error) < 0.1 else "✗"
        print(f"{bucket:<25s} | {count:6d} | {mean_pred:7.3f} | {actual_rate:7.3f} | {error:+7.3f} {status}")

    print()
    print()

    # Focus on extreme predictions
    print("=" * 80)
    print("EXTREME PREDICTIONS ANALYSIS")
    print("=" * 80)
    print()

    # Very confident OVER predictions
    very_high = df[df['model_prob'] >= 0.8].copy()
    if len(very_high) > 0:
        print(f"Very Confident OVER predictions (>=80%): {len(very_high)}")
        print(f"  Average predicted: {very_high['model_prob'].mean():.3f}")
        print(f"  Actual win rate: {very_high['bet_outcome'].mean():.3f}")
        print(f"  Calibration error: {very_high['bet_outcome'].mean() - very_high['model_prob'].mean():+.3f}")
        print()

        # Breakdown by market
        print("  Breakdown by market:")
        for market in very_high['market'].value_counts().head(5).index:
            market_df = very_high[very_high['market'] == market]
            print(f"    {market:<25s}: {len(market_df):3d} bets, {market_df['bet_outcome'].mean():.3f} win rate")
        print()

        # Breakdown by position
        print("  Breakdown by position:")
        for pos in very_high['position'].value_counts().head(5).index:
            pos_df = very_high[very_high['position'] == pos]
            print(f"    {pos:<5s}: {len(pos_df):3d} bets, {pos_df['bet_outcome'].mean():.3f} win rate")
        print()

    # Very confident UNDER predictions
    very_low = df[df['model_prob'] <= 0.2].copy()
    if len(very_low) > 0:
        print(f"Very Confident UNDER predictions (<=20%): {len(very_low)}")
        print(f"  Average predicted: {very_low['model_prob'].mean():.3f}")
        print(f"  Actual win rate: {very_low['bet_outcome'].mean():.3f}")
        print(f"  Calibration error: {very_low['bet_outcome'].mean() - very_low['model_prob'].mean():+.3f}")
        print()

        # Breakdown by market
        print("  Breakdown by market:")
        for market in very_low['market'].value_counts().head(5).index:
            market_df = very_low[very_low['market'] == market]
            print(f"    {market:<25s}: {len(market_df):3d} bets, {market_df['bet_outcome'].mean():.3f} win rate")
        print()

        # Breakdown by position
        print("  Breakdown by position:")
        for pos in very_low['position'].value_counts().head(5).index:
            pos_df = very_low[very_low['position'] == pos]
            print(f"    {pos:<5s}: {len(pos_df):3d} bets, {pos_df['bet_outcome'].mean():.3f} win rate")
        print()

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Calculate error by confidence
    very_high_error = very_high['bet_outcome'].mean() - very_high['model_prob'].mean() if len(very_high) > 0 else 0
    very_low_error = very_low['bet_outcome'].mean() - very_low['model_prob'].mean() if len(very_low) > 0 else 0

    if very_high_error < -0.15:
        print("HIGH CONFIDENCE OVERS:")
        print("  Model is severely overconfident when predicting OVER")
        print("  Recommendations:")
        print("  - Add uncertainty to simulation variance")
        print("  - Check for correlated features inflating confidence")
        print("  - Increase player performance variance in simulations")
        print()

    if very_low_error > 0.15:
        print("HIGH CONFIDENCE UNDERS:")
        print("  Model is severely overconfident when predicting UNDER")
        print("  Recommendations:")
        print("  - Review low-probability logic (are you too aggressive?)")
        print("  - Check if you're underestimating variance for low-usage players")
        print("  - Consider that bookmakers set favorable lines for unders")
        print()

    neutral = df[(df['model_prob'] >= 0.4) & (df['model_prob'] <= 0.6)]
    neutral_error = abs(neutral['bet_outcome'].mean() - neutral['model_prob'].mean())

    if neutral_error < 0.05:
        print("NEUTRAL PREDICTIONS:")
        print("  Model is well-calibrated in the 40-60% range")
        print("  This suggests the core model logic is sound")
        print("  Focus calibration efforts on extreme predictions only")
        print()


def main():
    backtest_file = 'reports/framework_backtest_weeks_1_7_fixed.csv'
    df = pd.read_csv(backtest_file)

    # Remove NaN
    df = df[df['model_prob'].notna()].copy()

    print(f"Analyzing {len(df):,} predictions from weeks 1-7")
    print()

    analyze_extreme_predictions(df)


if __name__ == "__main__":
    main()
