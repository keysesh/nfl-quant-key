#!/usr/bin/env python3
"""
Analyze why hit rate is low in backtests.

Investigates:
1. Model calibration (predicted vs actual win rates)
2. Probability distribution of model_prob
3. Edge distribution
4. Market efficiency (are lines accurate?)
5. Simulation quality (distribution width)
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.backtest.backtest_player_props import run_backtest, summarize
from nfl_quant.utils.season_utils import get_current_season


def analyze_calibration(results: pd.DataFrame):
    """Analyze model calibration - are predictions accurate?"""
    print("=" * 80)
    print("📊 MODEL CALIBRATION ANALYSIS")
    print("=" * 80)
    print()

    if 'model_prob' not in results.columns:
        print("⚠️  No model_prob column found")
        return

    # Bin by predicted probability
    bins = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "70-75%"),
        (0.70, 0.75, "70-75%"),
        (0.75, 0.80, "75-80%"),
        (0.80, 0.85, "80-85%"),
        (0.85, 0.90, "85-90%"),
        (0.90, 0.95, "90-95%"),
        (0.95, 1.00, "95%+"),
    ]

    print(f"{'Predicted Range':<15} {'Bets':<8} {'Predicted WR':<15} {'Actual WR':<15} {'Error':<12} {'ROI':<10}")
    print("-" * 80)

    for bin_min, bin_max, label in bins:
        bin_data = results[
            (results['model_prob'] >= bin_min) &
            (results['model_prob'] < bin_max)
        ]

        if len(bin_data) > 0:
            predicted = bin_data['model_prob'].mean()
            actual = bin_data['bet_won'].mean()
            error = actual - predicted
            roi = bin_data['unit_return'].mean()

            print(f"{label:<15} {len(bin_data):<8} {predicted:>14.1%} {actual:>14.1%} {error:>+11.1%} {roi:>+9.1%}")

    print()


def analyze_edge_distribution(results: pd.DataFrame):
    """Analyze edge distribution."""
    print("=" * 80)
    print("💰 EDGE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    if 'edge' not in results.columns:
        print("⚠️  No edge column found")
        return

    print(f"Edge Statistics:")
    print(f"  Mean:    {results['edge'].mean():.2%}")
    print(f"  Median:  {results['edge'].median():.2%}")
    print(f"  Min:     {results['edge'].min():.2%}")
    print(f"  Max:     {results['edge'].max():.2%}")
    print(f"  Std Dev: {results['edge'].std():.2%}")
    print()

    # Edge bins
    edge_bins = [
        (0.03, 0.05, "3-5%"),
        (0.05, 0.10, "5-10%"),
        (0.10, 0.20, "10-20%"),
        (0.20, 0.30, "20-30%"),
        (0.30, 1.00, "30%+"),
    ]

    print(f"{'Edge Range':<15} {'Bets':<8} {'Hit Rate':<12} {'ROI':<10}")
    print("-" * 80)

    for bin_min, bin_max, label in edge_bins:
        bin_data = results[
            (results['edge'] >= bin_min) &
            (results['edge'] < bin_max)
        ]

        if len(bin_data) > 0:
            hit_rate = bin_data['bet_won'].mean()
            roi = bin_data['unit_return'].mean()

            print(f"{label:<15} {len(bin_data):<8} {hit_rate:>11.1%} {roi:>+9.1%}")

    print()


def analyze_market_efficiency(results: pd.DataFrame):
    """Analyze market efficiency - are lines accurate?"""
    print("=" * 80)
    print("📈 MARKET EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()

    if 'implied_prob' not in results.columns:
        print("⚠️  No implied_prob column found")
        return

    # What's the actual win rate at different implied probabilities?
    bins = [
        (0.40, 0.45, "40-45%"),
        (0.45, 0.50, "45-50%"),
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
    ]

    print(f"{'Implied Prob':<15} {'Bets':<8} {'Implied':<12} {'Actual WR':<12} {'Error':<12}")
    print("-" * 80)

    for bin_min, bin_max, label in bins:
        bin_data = results[
            (results['implied_prob'] >= bin_min) &
            (results['implied_prob'] < bin_max)
        ]

        if len(bin_data) > 0:
            implied = bin_data['implied_prob'].mean()
            actual = bin_data['bet_won'].mean()
            error = actual - implied

            print(f"{label:<15} {len(bin_data):<8} {implied:>11.1%} {actual:>11.1%} {error:>+11.1%}")

    print()


def analyze_by_market(results: pd.DataFrame):
    """Analyze performance by market type."""
    print("=" * 80)
    print("🎯 PERFORMANCE BY MARKET")
    print("=" * 80)
    print()

    if 'market' not in results.columns:
        print("⚠️  No market column found")
        return

    print(f"{'Market':<25} {'Bets':<8} {'Hit Rate':<12} {'Avg Edge':<12} {'ROI':<10}")
    print("-" * 80)

    for market in sorted(results['market'].unique()):
        market_data = results[results['market'] == market]
        hit_rate = market_data['bet_won'].mean()
        avg_edge = market_data['edge'].mean()
        roi = market_data['unit_return'].mean()

        print(f"{market:<25} {len(market_data):<8} {hit_rate:>11.1%} {avg_edge:>11.1%} {roi:>+9.1%}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest hit rate")
    parser.add_argument("--results-file", type=Path, help="Path to saved results CSV")
    parser.add_argument("--start-week", type=int, default=1)
    parser.add_argument("--end-week", type=int, default=7)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    if args.results_file and args.results_file.exists():
        print(f"📂 Loading results from {args.results_file}")
        results = pd.read_csv(args.results_file)
    else:
        print("🔍 Running backtest...")
        results = run_backtest(
            start_week=args.start_week,
            end_week=args.end_week,
            season=get_current_season(),
            prop_files_dir=Path("data/historical"),
            min_edge=args.min_edge,
            min_confidence=args.min_confidence,
        )

    print(f"\n📊 Analyzing {len(results)} bets")
    print()

    # Overall summary
    summarize(results)
    print()

    # Detailed analyses
    analyze_calibration(results)
    analyze_edge_distribution(results)
    analyze_market_efficiency(results)
    analyze_by_market(results)

    # Key insights
    print("=" * 80)
    print("🔍 KEY INSIGHTS")
    print("=" * 80)
    print()

    overall_hit_rate = results['bet_won'].mean()
    avg_model_prob = results['model_prob'].mean()
    calibration_error = overall_hit_rate - avg_model_prob

    print(f"Overall Hit Rate:      {overall_hit_rate:.1%}")
    print(f"Average Model Prob:    {avg_model_prob:.1%}")
    print(f"Calibration Error:     {calibration_error:+.1%}")
    print()

    if calibration_error < -0.05:
        print("⚠️  Model is OVERCONFIDENT (predicting higher win rate than actual)")
        print("   Suggestions:")
        print("   - Add calibration layer (isotonic regression)")
        print("   - Increase simulation variance")
        print("   - Apply shrinkage to probabilities")
    elif calibration_error > 0.05:
        print("⚠️  Model is UNDERCONFIDENT (predicting lower win rate than actual)")
        print("   Suggestions:")
        print("   - Reduce uncertainty in simulations")
        print("   - Check if model is too conservative")
    else:
        print("✅ Model is well-calibrated")


if __name__ == "__main__":
    main()
