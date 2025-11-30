#!/usr/bin/env python3
"""
Analyze whether market-specific calibrators would improve performance.

This script:
1. Loads backtest data with outcomes by market
2. Calculates calibration metrics by market type
3. Recommends whether to use separate calibrators per market
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_calibration_by_market(df: pd.DataFrame):
    """Analyze calibration performance for each market separately."""

    print("=" * 100)
    print("MARKET-SPECIFIC CALIBRATION ANALYSIS")
    print("=" * 100)
    print()

    markets = df['market'].unique()

    results = []

    for market in markets:
        market_df = df[df['market'] == market]

        if len(market_df) < 50:  # Need sufficient data
            continue

        # Calculate calibration metrics
        raw_probs = market_df['model_prob_raw'].values
        cal_probs = market_df['model_prob'].values
        outcomes = market_df['bet_won'].values

        # Overall metrics
        brier_raw = brier_score_loss(outcomes, raw_probs)
        brier_cal = brier_score_loss(outcomes, cal_probs)

        mace_raw = abs(raw_probs.mean() - outcomes.mean())
        mace_cal = abs(cal_probs.mean() - outcomes.mean())

        # Bin-specific analysis
        bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        bin_maces = []

        for low, high in bins:
            mask = (cal_probs >= low) & (cal_probs < high)
            if mask.sum() > 10:
                bin_mace = abs(cal_probs[mask].mean() - outcomes[mask].mean())
                bin_maces.append(bin_mace)

        avg_bin_mace = np.mean(bin_maces) if bin_maces else 0.0
        max_bin_mace = np.max(bin_maces) if bin_maces else 0.0

        results.append({
            'market': market,
            'count': len(market_df),
            'hit_rate': outcomes.mean(),
            'avg_raw_prob': raw_probs.mean(),
            'avg_cal_prob': cal_probs.mean(),
            'brier_raw': brier_raw,
            'brier_cal': brier_cal,
            'brier_improvement': brier_raw - brier_cal,
            'mace_raw': mace_raw,
            'mace_cal': mace_cal,
            'mace_improvement': mace_raw - mace_cal,
            'avg_bin_mace': avg_bin_mace,
            'max_bin_mace': max_bin_mace
        })

    results_df = pd.DataFrame(results).sort_values('count', ascending=False)

    print(f"{'Market':<25} | {'Count':>6} | {'Hit Rate':>9} | {'MACE (Cal)':>11} | {'Max Bin MACE':>13} | {'Brier Improv':>14}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        status = "✅" if row['mace_cal'] < 0.05 else ("⚠️ " if row['mace_cal'] < 0.10 else "❌")
        print(f"{row['market']:<25} | {row['count']:6d} | {row['hit_rate']:9.1%} | {row['mace_cal']:11.4f} | {row['max_bin_mace']:13.4f} | {row['brier_improvement']:+14.4f} {status}")

    print()

    # Analysis
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print()

    # Check variance in calibration quality
    mace_std = results_df['mace_cal'].std()
    mace_range = results_df['mace_cal'].max() - results_df['mace_cal'].min()

    print(f"MACE Standard Deviation across markets: {mace_std:.4f}")
    print(f"MACE Range (max - min): {mace_range:.4f}")
    print()

    if mace_range > 0.05:
        print("✅ RECOMMENDATION: Use market-specific calibrators")
        print()
        print("RATIONALE:")
        print(f"  - Significant variation in calibration quality (range = {mace_range:.4f})")
        print(f"  - Some markets perform much worse than others")
        print(f"  - Market-specific calibrators could improve overall accuracy")
        print()
        print("IMPLEMENTATION:")
        print("  1. Train separate calibrators for each major market:")
        for _, row in results_df.head(5).iterrows():
            print(f"     - {row['market']}: {row['count']} samples")
        print("  2. Save as: configs/calibrator_{market}.json")
        print("  3. Load appropriate calibrator based on market in prediction pipeline")
    else:
        print("➖ RECOMMENDATION: Continue using single unified calibrator")
        print()
        print("RATIONALE:")
        print(f"  - Low variation in calibration quality (range = {mace_range:.4f})")
        print(f"  - Unified calibrator works well across all markets")
        print(f"  - Complexity of market-specific calibrators not justified")

    print()

    # Save results
    output_file = Path('reports/market_specific_calibration_analysis.csv')
    results_df.to_csv(output_file, index=False)
    print(f"💾 Detailed results saved to: {output_file}")
    print()

    return results_df


def main():
    print("=" * 100)
    print("MARKET-SPECIFIC CALIBRATION ANALYSIS")
    print("=" * 100)
    print()

    # Load backtest data
    backtest_file = Path('reports/FRESH_BACKTEST_WEEKS_1_8_CALIBRATED.csv')

    if not backtest_file.exists():
        print(f"ERROR: Backtest file not found: {backtest_file}")
        return

    print(f"Loading backtest data from: {backtest_file}")
    df = pd.read_csv(backtest_file)
    print(f"Loaded {len(df):,} predictions")
    print()

    # Verify required columns
    required_cols = ['market', 'model_prob', 'model_prob_raw', 'bet_won']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return

    # Analyze
    results_df = analyze_calibration_by_market(df)

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
