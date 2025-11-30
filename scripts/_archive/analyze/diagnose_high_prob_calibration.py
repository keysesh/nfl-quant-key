#!/usr/bin/env python3
"""
Deep diagnostic analysis of high-probability calibration issues.

This script specifically analyzes the 70%+ probability range where
calibrated probabilities show ~51% hit rate despite 70-80% predicted probabilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_calibration_by_bin(df: pd.DataFrame):
    """Analyze calibration performance across probability bins."""
    print("=" * 80)
    print("CALIBRATION ANALYSIS BY PROBABILITY BIN")
    print("=" * 80)
    print()

    # Define probability bins
    bins = [
        (0.00, 0.40, "Very Low (0-40%)"),
        (0.40, 0.50, "Low (40-50%)"),
        (0.50, 0.55, "Slightly Above (50-55%)"),
        (0.55, 0.60, "Medium (55-60%)"),
        (0.60, 0.65, "Medium-High (60-65%)"),
        (0.65, 0.70, "High (65-70%)"),
        (0.70, 0.75, "Very High (70-75%)"),
        (0.75, 0.80, "Extremely High (75-80%)"),
        (0.80, 1.00, "Ultra High (80-100%)"),
    ]

    print(f"{'Probability Range':<25} | {'Count':>6} | {'Raw Prob':>9} | {'Cal Prob':>9} | {'Hit Rate':>9} | {'MACE':>9} | {'Edge Calc':>10}")
    print("-" * 110)

    results = []

    for low, high, label in bins:
        # Filter by CALIBRATED probability (this is what we're betting on)
        mask = (df['model_prob'] >= low) & (df['model_prob'] < high)
        bin_df = df[mask]

        if len(bin_df) == 0:
            continue

        count = len(bin_df)
        avg_raw_prob = bin_df['model_prob_raw'].mean()
        avg_cal_prob = bin_df['model_prob'].mean()
        hit_rate = bin_df['bet_won'].mean()
        mace = abs(avg_cal_prob - hit_rate)
        avg_edge = bin_df['edge'].mean() if 'edge' in bin_df.columns else 0.0

        # Determine status
        if mace > 0.10:
            status = "SEVERE"
        elif mace > 0.05:
            status = "BAD"
        elif mace > 0.03:
            status = "WARN"
        else:
            status = "OK"

        print(f"{label:<25} | {count:6d} | {avg_raw_prob:9.4f} | {avg_cal_prob:9.4f} | {hit_rate:9.4f} | {mace:9.4f} | {avg_edge:+9.4f} [{status}]")

        results.append({
            'bin_label': label,
            'bin_low': low,
            'bin_high': high,
            'count': count,
            'avg_raw_prob': avg_raw_prob,
            'avg_cal_prob': avg_cal_prob,
            'hit_rate': hit_rate,
            'mace': mace,
            'avg_edge': avg_edge,
            'status': status
        })

    print()

    # Focus on the problematic range (70%+)
    print("=" * 80)
    print("FOCUS: HIGH-PROBABILITY BINS (70%+)")
    print("=" * 80)
    print()

    high_prob_mask = df['model_prob'] >= 0.70
    high_prob_df = df[high_prob_mask]

    if len(high_prob_df) > 0:
        print(f"Total predictions with calibrated prob >= 70%: {len(high_prob_df)}")
        print(f"Average raw probability: {high_prob_df['model_prob_raw'].mean():.4f}")
        print(f"Average calibrated probability: {high_prob_df['model_prob'].mean():.4f}")
        print(f"Actual hit rate: {high_prob_df['bet_won'].mean():.4f}")
        print(f"Expected hits: {high_prob_df['model_prob'].sum():.1f}")
        print(f"Actual hits: {high_prob_df['bet_won'].sum():.0f}")
        print(f"Hit rate deficit: {(high_prob_df['bet_won'].mean() - high_prob_df['model_prob'].mean()):.4f}")
        print()

        # Detailed breakdown by 5% increments
        print("5% incremental bins:")
        print(f"{'Range':>12} | {'Count':>6} | {'Raw':>8} | {'Cal':>8} | {'Hit%':>8} | {'Error':>8}")
        print("-" * 60)

        for prob_start in np.arange(0.70, 1.00, 0.05):
            prob_end = prob_start + 0.05
            mask = (high_prob_df['model_prob'] >= prob_start) & (high_prob_df['model_prob'] < prob_end)
            bin_data = high_prob_df[mask]

            if len(bin_data) > 0:
                error = bin_data['bet_won'].mean() - bin_data['model_prob'].mean()
                print(f"{prob_start:.2f}-{prob_end:.2f} | {len(bin_data):6d} | {bin_data['model_prob_raw'].mean():8.4f} | {bin_data['model_prob'].mean():8.4f} | {bin_data['bet_won'].mean():8.4f} | {error:+8.4f}")

        print()
    else:
        print("No predictions with calibrated probability >= 70%")
        print()

    return pd.DataFrame(results)


def analyze_raw_vs_calibrated_mapping(df: pd.DataFrame, calibrator_path: str):
    """Analyze the isotonic regression mapping from raw to calibrated probabilities."""
    print("=" * 80)
    print("ISOTONIC REGRESSION CURVE ANALYSIS")
    print("=" * 80)
    print()

    # Load calibrator
    with open(calibrator_path) as f:
        cal_params = json.load(f)

    # Handle both old and new format
    x_key = 'X_thresholds' if 'X_thresholds' in cal_params else 'x'
    y_key = 'y_thresholds' if 'y_thresholds' in cal_params else 'y'

    x_thresh = np.array(cal_params[x_key])
    y_thresh = np.array(cal_params[y_key])

    print(f"Calibrator has {len(x_thresh)} calibration points")
    print()

    # Show the calibration curve, focusing on high probabilities
    print("Calibration curve (focusing on raw prob >= 70%):")
    print(f"{'Raw Prob':>10} | {'Calibrated':>10} | {'Adjustment':>12} | {'Samples in Training':>20}")
    print("-" * 60)

    for i in range(len(x_thresh)):
        raw_p = x_thresh[i]
        cal_p = y_thresh[i]

        if raw_p >= 0.70:
            adjustment = cal_p - raw_p

            # Count how many training samples fell in this range
            if i < len(x_thresh) - 1:
                mask = (df['model_prob_raw'] >= x_thresh[i]) & (df['model_prob_raw'] < x_thresh[i+1])
            else:
                mask = df['model_prob_raw'] >= x_thresh[i]

            sample_count = mask.sum()

            print(f"{raw_p:10.4f} | {cal_p:10.4f} | {adjustment:+12.4f} | {sample_count:20d}")

    print()

    # Identify the problem range
    print("=" * 80)
    print("ROOT CAUSE IDENTIFICATION")
    print("=" * 80)
    print()

    # Find where severe adjustments occur
    severe_adjustments = []
    for i in range(len(x_thresh)):
        if x_thresh[i] >= 0.70:
            adjustment = y_thresh[i] - x_thresh[i]
            if adjustment < -0.10:  # More than 10% reduction
                severe_adjustments.append((x_thresh[i], y_thresh[i], adjustment))

    if severe_adjustments:
        print("SEVERE ADJUSTMENTS DETECTED:")
        for raw_p, cal_p, adj in severe_adjustments:
            print(f"  Raw {raw_p:.4f} → Calibrated {cal_p:.4f} (adjustment: {adj:+.4f})")
        print()
        print("This indicates isotonic regression is applying excessive shrinkage")
        print("in the high-probability range.")
        print()
    else:
        print("No severe adjustments detected in calibration curve.")
        print()

    # Check training data distribution
    print("Training data distribution in high-probability range:")
    for prob_start in np.arange(0.70, 1.00, 0.10):
        prob_end = prob_start + 0.10
        mask = (df['model_prob_raw'] >= prob_start) & (df['model_prob_raw'] < prob_end)
        count = mask.sum()
        if count > 0:
            win_rate = df[mask]['bet_won'].mean()
            print(f"  Raw prob {prob_start:.2f}-{prob_end:.2f}: {count:4d} samples, {win_rate:.4f} win rate")

    print()


def calculate_optimal_shrinkage(df: pd.DataFrame):
    """Calculate what shrinkage factor would optimize high-probability calibration."""
    print("=" * 80)
    print("OPTIMAL SHRINKAGE CALCULATION")
    print("=" * 80)
    print()

    # Focus on high-probability predictions
    high_prob_mask = df['model_prob_raw'] >= 0.70
    high_prob_df = df[high_prob_mask]

    if len(high_prob_df) == 0:
        print("No high-probability raw predictions to analyze.")
        return

    print(f"Analyzing {len(high_prob_df)} predictions with raw prob >= 70%")
    print()

    # Try different shrinkage factors
    print("Testing different shrinkage strategies:")
    print(f"{'Strategy':30s} | {'Avg Pred':>9} | {'Hit Rate':>9} | {'MACE':>9} | {'Brier':>9}")
    print("-" * 75)

    raw_probs = high_prob_df['model_prob_raw'].values
    outcomes = high_prob_df['bet_won'].values
    current_cal = high_prob_df['model_prob'].values

    # Current calibration
    brier_current = brier_score_loss(outcomes, current_cal)
    mace_current = abs(current_cal.mean() - outcomes.mean())
    print(f"{'Current Isotonic (>70% only)':30s} | {current_cal.mean():9.4f} | {outcomes.mean():9.4f} | {mace_current:9.4f} | {brier_current:9.4f}")

    # Raw (no calibration)
    brier_raw = brier_score_loss(outcomes, raw_probs)
    mace_raw = abs(raw_probs.mean() - outcomes.mean())
    print(f"{'Raw (no calibration)':30s} | {raw_probs.mean():9.4f} | {outcomes.mean():9.4f} | {mace_raw:9.4f} | {brier_raw:9.4f}")

    # Try different linear shrinkage factors
    best_mace = float('inf')
    best_shrinkage = 0

    for shrinkage in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # Apply shrinkage: pull toward 0.5
        calibrated = 0.5 + (raw_probs - 0.5) * shrinkage
        calibrated = np.clip(calibrated, 0, 1)

        brier = brier_score_loss(outcomes, calibrated)
        mace = abs(calibrated.mean() - outcomes.mean())

        print(f"{'Linear shrinkage ' + str(shrinkage):30s} | {calibrated.mean():9.4f} | {outcomes.mean():9.4f} | {mace:9.4f} | {brier:9.4f}")

        if mace < best_mace:
            best_mace = mace
            best_shrinkage = shrinkage

    print()
    print(f"RECOMMENDATION: Use linear shrinkage factor of {best_shrinkage} for raw prob >= 70%")
    print(f"This would reduce MACE from {mace_current:.4f} to {best_mace:.4f}")
    print()

    return best_shrinkage


def plot_calibration_diagnostics(df: pd.DataFrame, output_dir: Path = Path("reports")):
    """Create detailed diagnostic plots."""
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Raw vs Calibrated scatter
    ax = axes[0, 0]
    ax.scatter(df['model_prob_raw'], df['model_prob'], alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No calibration (y=x)')
    ax.set_xlabel('Raw Model Probability', fontsize=12)
    ax.set_ylabel('Calibrated Probability', fontsize=12)
    ax.set_title('Isotonic Calibration Mapping', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight the problematic region
    ax.axhspan(0.70, 0.80, alpha=0.2, color='red', label='Problematic range')
    ax.axvspan(0.70, 1.00, alpha=0.1, color='orange')

    # Plot 2: Calibration curve by bin
    ax = axes[0, 1]

    # Calculate calibration by bins
    bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_hit_rates = []
    bin_avg_probs = []

    for i in range(len(bins) - 1):
        mask = (df['model_prob'] >= bins[i]) & (df['model_prob'] < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_hit_rates.append(df[mask]['bet_won'].mean())
            bin_avg_probs.append(df[mask]['model_prob'].mean())

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax.scatter(bin_avg_probs, bin_hit_rates, s=100, alpha=0.7, label='Actual Calibration')
    ax.set_xlabel('Predicted Probability (Calibrated)', fontsize=12)
    ax.set_ylabel('Actual Win Rate', fontsize=12)
    ax.set_title('Calibration Quality (Post-Isotonic)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution comparison
    ax = axes[1, 0]
    ax.hist(df['model_prob_raw'], bins=30, alpha=0.5, label='Raw Probabilities', edgecolor='black')
    ax.hist(df['model_prob'], bins=30, alpha=0.5, label='Calibrated Probabilities', edgecolor='black')
    ax.axvline(0.70, color='red', linestyle='--', linewidth=2, label='70% threshold')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Raw vs Calibrated Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: MACE by probability bin
    ax = axes[1, 1]

    bin_ranges = []
    mace_values = []

    for prob_start in np.arange(0.40, 1.00, 0.05):
        prob_end = prob_start + 0.05
        mask = (df['model_prob'] >= prob_start) & (df['model_prob'] < prob_end)
        if mask.sum() > 10:  # At least 10 samples
            mace = abs(df[mask]['model_prob'].mean() - df[mask]['bet_won'].mean())
            bin_ranges.append(f"{prob_start:.2f}")
            mace_values.append(mace)

    colors = ['red' if m > 0.10 else 'orange' if m > 0.05 else 'green' for m in mace_values]
    ax.bar(range(len(bin_ranges)), mace_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Probability Bin (Calibrated)', fontsize=12)
    ax.set_ylabel('Mean Absolute Calibration Error', fontsize=12)
    ax.set_title('Calibration Error by Bin\n(Red > 10%, Orange > 5%, Green ≤ 5%)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(bin_ranges)))
    ax.set_xticklabels(bin_ranges, rotation=45)
    ax.axhline(0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = output_dir / 'high_prob_calibration_diagnostics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plots saved to: {plot_path}")
    print()


def main():
    print("=" * 80)
    print("HIGH-PROBABILITY CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    print()

    # Load backtest data
    backtest_file = Path('reports/FRESH_BACKTEST_WEEKS_1_8_CALIBRATED.csv')
    calibrator_file = Path('configs/calibrator.json')

    if not backtest_file.exists():
        print(f"ERROR: Backtest file not found: {backtest_file}")
        return

    if not calibrator_file.exists():
        print(f"ERROR: Calibrator file not found: {calibrator_file}")
        return

    print(f"Loading backtest data from: {backtest_file}")
    df = pd.read_csv(backtest_file)
    print(f"Loaded {len(df):,} predictions")
    print()

    # Verify required columns
    required_cols = ['model_prob', 'model_prob_raw', 'bet_won']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Step 1: Analyze calibration by bin
    bin_results = analyze_calibration_by_bin(df)

    # Step 2: Analyze isotonic regression mapping
    analyze_raw_vs_calibrated_mapping(df, calibrator_file)

    # Step 3: Calculate optimal shrinkage
    optimal_shrinkage = calculate_optimal_shrinkage(df)

    # Step 4: Create diagnostic plots
    plot_calibration_diagnostics(df)

    # Save results
    bin_results.to_csv('reports/calibration_bin_analysis.csv', index=False)
    print(f"Bin analysis saved to: reports/calibration_bin_analysis.csv")
    print()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
