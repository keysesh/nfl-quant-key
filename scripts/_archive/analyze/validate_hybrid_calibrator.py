#!/usr/bin/env python3
"""
Validate the hybrid calibration approach on existing backtest data.

This script:
1. Loads the existing backtest data with raw probabilities
2. Applies the new hybrid calibration approach
3. Compares old vs new calibration performance
4. Generates validation plots
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator


def compare_calibration_methods(df: pd.DataFrame):
    """Compare old isotonic vs new hybrid calibration."""
    print("=" * 80)
    print("CALIBRATION METHOD COMPARISON")
    print("=" * 80)
    print()

    # Load old calibrator
    old_calibrator = NFLProbabilityCalibrator()
    old_calibrator.load('configs/calibrator.json')

    # Create new hybrid calibrator (with aggressive shrinkage for high probs)
    new_calibrator = NFLProbabilityCalibrator(
        high_prob_threshold=0.70,
        high_prob_shrinkage=0.3
    )
    new_calibrator.load('configs/calibrator.json')

    # Apply both calibrations
    raw_probs = df['model_prob_raw'].values
    outcomes = df['bet_won'].values

    # Old calibration (what's currently in model_prob column)
    old_cal = df['model_prob'].values

    # New hybrid calibration
    new_cal = new_calibrator.transform(raw_probs)

    # Overall metrics
    print("OVERALL PERFORMANCE:")
    print(f"{'Method':<30} | {'Brier Score':>12} | {'Mean Prob':>10} | {'Hit Rate':>10} | {'MACE':>10}")
    print("-" * 75)

    brier_raw = brier_score_loss(outcomes, raw_probs)
    mace_raw = abs(raw_probs.mean() - outcomes.mean())
    print(f"{'Raw (no calibration)':<30} | {brier_raw:12.4f} | {raw_probs.mean():10.4f} | {outcomes.mean():10.4f} | {mace_raw:10.4f}")

    brier_old = brier_score_loss(outcomes, old_cal)
    mace_old = abs(old_cal.mean() - outcomes.mean())
    print(f"{'Old Isotonic':<30} | {brier_old:12.4f} | {old_cal.mean():10.4f} | {outcomes.mean():10.4f} | {mace_old:10.4f}")

    brier_new = brier_score_loss(outcomes, new_cal)
    mace_new = abs(new_cal.mean() - outcomes.mean())
    print(f"{'New Hybrid (iso<70%, shrink>=70%)':<30} | {brier_new:12.4f} | {new_cal.mean():10.4f} | {outcomes.mean():10.4f} | {mace_new:10.4f}")

    print()

    # High-probability performance (70%+)
    print("=" * 80)
    print("HIGH-PROBABILITY RANGE (Raw >= 70%)")
    print("=" * 80)
    print()

    high_mask = raw_probs >= 0.70
    if high_mask.sum() > 0:
        print(f"Total predictions with raw prob >= 70%: {high_mask.sum()}")
        print()

        print(f"{'Method':<30} | {'Brier Score':>12} | {'Mean Prob':>10} | {'Hit Rate':>10} | {'MACE':>10}")
        print("-" * 75)

        high_outcomes = outcomes[high_mask]

        high_raw = raw_probs[high_mask]
        brier_hr = brier_score_loss(high_outcomes, high_raw)
        mace_hr = abs(high_raw.mean() - high_outcomes.mean())
        print(f"{'Raw':<30} | {brier_hr:12.4f} | {high_raw.mean():10.4f} | {high_outcomes.mean():10.4f} | {mace_hr:10.4f}")

        high_old = old_cal[high_mask]
        brier_ho = brier_score_loss(high_outcomes, high_old)
        mace_ho = abs(high_old.mean() - high_outcomes.mean())
        print(f"{'Old Isotonic':<30} | {brier_ho:12.4f} | {high_old.mean():10.4f} | {high_outcomes.mean():10.4f} | {mace_ho:10.4f}")

        high_new = new_cal[high_mask]
        brier_hn = brier_score_loss(high_outcomes, high_new)
        mace_hn = abs(high_new.mean() - high_outcomes.mean())
        print(f"{'New Hybrid':<30} | {brier_hn:12.4f} | {high_new.mean():10.4f} | {high_outcomes.mean():10.4f} | {mace_hn:10.4f}")

        print()
        print(f"MACE Improvement: {mace_ho:.4f} → {mace_hn:.4f} ({(mace_hn - mace_ho):.4f})")
        print(f"Brier Improvement: {brier_ho:.4f} → {brier_hn:.4f} ({(brier_hn - brier_ho):.4f})")
        print()

    # Detailed bin analysis
    print("=" * 80)
    print("BIN-BY-BIN COMPARISON (Raw Probability Bins)")
    print("=" * 80)
    print()

    bins = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]

    for low, high in bins:
        mask = (raw_probs >= low) & (raw_probs < high)
        if mask.sum() < 10:
            continue

        print(f"\nRaw Probability: {low:.2f} - {high:.2f} ({mask.sum()} samples)")
        print(f"{'Method':<30} | {'Avg Prob':>10} | {'Hit Rate':>10} | {'MACE':>10}")
        print("-" * 60)

        bin_outcomes = outcomes[mask]

        bin_old = old_cal[mask]
        mace_bo = abs(bin_old.mean() - bin_outcomes.mean())
        print(f"{'Old Isotonic':<30} | {bin_old.mean():10.4f} | {bin_outcomes.mean():10.4f} | {mace_bo:10.4f}")

        bin_new = new_cal[mask]
        mace_bn = abs(bin_new.mean() - bin_outcomes.mean())
        print(f"{'New Hybrid':<30} | {bin_new.mean():10.4f} | {bin_outcomes.mean():10.4f} | {mace_bn:10.4f}")

        improvement = mace_bo - mace_bn
        if improvement > 0:
            print(f"  ✅ Improvement: {improvement:+.4f}")
        elif improvement < 0:
            print(f"  ⚠️  Degradation: {improvement:+.4f}")
        else:
            print(f"  ➖ No change")

    return {
        'raw_probs': raw_probs,
        'old_cal': old_cal,
        'new_cal': new_cal,
        'outcomes': outcomes
    }


def plot_calibration_comparison(data: dict, output_dir: Path = Path("reports")):
    """Create comparison plots."""
    output_dir.mkdir(exist_ok=True, parents=True)

    raw_probs = data['raw_probs']
    old_cal = data['old_cal']
    new_cal = data['new_cal']
    outcomes = data['outcomes']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Calibration curve comparison
    ax = axes[0, 0]

    # Calculate calibration curves
    bins = np.linspace(0, 1, 11)
    bin_centers_old = []
    bin_hit_rates_old = []
    bin_centers_new = []
    bin_hit_rates_new = []

    for i in range(len(bins) - 1):
        # Old calibration
        mask_old = (old_cal >= bins[i]) & (old_cal < bins[i+1])
        if mask_old.sum() > 0:
            bin_centers_old.append(old_cal[mask_old].mean())
            bin_hit_rates_old.append(outcomes[mask_old].mean())

        # New calibration
        mask_new = (new_cal >= bins[i]) & (new_cal < bins[i+1])
        if mask_new.sum() > 0:
            bin_centers_new.append(new_cal[mask_new].mean())
            bin_hit_rates_new.append(outcomes[mask_new].mean())

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    ax.scatter(bin_centers_old, bin_hit_rates_old, s=100, alpha=0.7, label='Old Isotonic', marker='o')
    ax.scatter(bin_centers_new, bin_hit_rates_new, s=100, alpha=0.7, label='New Hybrid', marker='s')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Win Rate', fontsize=12)
    ax.set_title('Calibration Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution comparison
    ax = axes[0, 1]
    ax.hist(old_cal, bins=30, alpha=0.5, label='Old Isotonic', edgecolor='black')
    ax.hist(new_cal, bins=30, alpha=0.5, label='New Hybrid', edgecolor='black')
    ax.axvline(0.70, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Calibrated Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Calibrated Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Raw to calibrated mapping
    ax = axes[1, 0]
    ax.scatter(raw_probs, old_cal, alpha=0.3, s=10, label='Old Isotonic')
    ax.scatter(raw_probs, new_cal, alpha=0.3, s=10, label='New Hybrid')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No calibration', alpha=0.5)
    ax.axvline(0.70, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Shrinkage threshold')
    ax.set_xlabel('Raw Probability', fontsize=12)
    ax.set_ylabel('Calibrated Probability', fontsize=12)
    ax.set_title('Calibration Mapping Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: MACE by bin
    ax = axes[1, 1]

    bin_ranges = []
    mace_old = []
    mace_new = []

    for prob_start in np.arange(0.50, 1.00, 0.10):
        prob_end = prob_start + 0.10
        mask = (raw_probs >= prob_start) & (raw_probs < prob_end)
        if mask.sum() > 10:
            bin_ranges.append(f"{prob_start:.1f}-{prob_end:.1f}")

            mace_o = abs(old_cal[mask].mean() - outcomes[mask].mean())
            mace_n = abs(new_cal[mask].mean() - outcomes[mask].mean())

            mace_old.append(mace_o)
            mace_new.append(mace_n)

    x = np.arange(len(bin_ranges))
    width = 0.35

    ax.bar(x - width/2, mace_old, width, label='Old Isotonic', alpha=0.7)
    ax.bar(x + width/2, mace_new, width, label='New Hybrid', alpha=0.7)
    ax.set_xlabel('Raw Probability Bin', fontsize=12)
    ax.set_ylabel('Mean Absolute Calibration Error', fontsize=12)
    ax.set_title('MACE by Raw Probability Bin', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_ranges)
    ax.axhline(0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = output_dir / 'hybrid_calibration_validation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation plots saved to: {plot_path}")
    print()


def main():
    print("=" * 80)
    print("HYBRID CALIBRATION VALIDATION")
    print("=" * 80)
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
    required_cols = ['model_prob', 'model_prob_raw', 'bet_won']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return

    # Compare calibration methods
    data = compare_calibration_methods(df)

    # Create validation plots
    plot_calibration_comparison(data)

    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()

    print("SUMMARY:")
    print("✅ Hybrid calibration applies aggressive shrinkage (0.3) for raw prob >= 70%")
    print("✅ This should significantly reduce overconfidence in high-probability predictions")
    print("✅ Check the validation plots to see calibration curve improvement")
    print()


if __name__ == "__main__":
    main()
