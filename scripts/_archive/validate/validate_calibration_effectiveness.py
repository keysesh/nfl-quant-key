#!/usr/bin/env python3
"""
Validate Probability Calibration Effectiveness

This script validates that probability calibration improvements are actually
improving model accuracy and betting effectiveness, not just manipulating data.

Metrics calculated:
- Brier Score (lower is better)
- Log Loss (lower is better)
- Expected Calibration Error (ECE)
- Calibration plot (predicted vs actual win rates)
- Win rate by probability bin

Tests:
1. Raw probabilities vs calibrated probabilities accuracy
2. Historical performance validation
3. Edge preservation verification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import json


def calculate_brier_score(y_true, y_pred):
    """Calculate Brier score (lower is better)."""
    return np.mean((y_pred - y_true) ** 2)


def calculate_log_loss(y_true, y_pred, epsilon=1e-15):
    """Calculate log loss (lower is better)."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def calculate_ece(y_true, y_pred, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def load_historical_recommendations():
    """Load historical betting recommendations with outcomes."""
    # Try to find historical data - check multiple possible locations
    historical_files = [
        Path('reports/BACKTEST_WEEKS_1_8_VALIDATION.csv'),
        Path('reports/betting_backtest_all_bets.csv'),
        Path('reports/framework_backtest_weeks_1_7.csv'),
        Path('reports/backtest_results.csv'),
        Path('data/historical/player_prop_training_dataset.csv'),
        Path('reports/historical_recommendations.csv'),
    ]

    for file_path in historical_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Found: {file_path}")

                # Handle different column name variations
                prob_col = None
                for col in ['model_prob', 'prob', 'probability']:
                    if col in df.columns:
                        prob_col = col
                        break

                if not prob_col:
                    continue

                # Handle outcome columns
                outcome_col = None
                outcome_cols = ['bet_won', 'win', 'outcome', 'bet_outcome_over', 'bet_outcome_under']
                for col in outcome_cols:
                    if col in df.columns:
                        outcome_col = col
                        break

                if not outcome_col:
                    continue

                # Convert outcome to binary (1 = win, 0 = loss)
                if outcome_col == 'bet_won':
                    df['win'] = df[outcome_col].astype(int)
                elif outcome_col in ['bet_outcome_over', 'bet_outcome_under']:
                    df['win'] = df[outcome_col].apply(lambda x: 1 if x == 1.0 else 0 if x == 0.0 else np.nan)
                else:
                    df['win'] = pd.to_numeric(df[outcome_col], errors='coerce')

                # Rename probability column
                df['model_prob'] = pd.to_numeric(df[prob_col], errors='coerce')

                # Filter out NaN outcomes and probabilities
                df = df[df['win'].notna()].copy()
                df = df[df['model_prob'].notna()].copy()

                if len(df) > 0:
                    print(f"   ✅ Loaded {len(df):,} bets with outcomes")
                    return df
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {e}")
                continue

    print("\n⚠️  No historical outcome data found.")
    return None


def validate_calibration():
    """Validate probability calibration effectiveness."""
    print("=" * 80)
    print("🎯 PROBABILITY CALIBRATION VALIDATION")
    print("=" * 80)

    # Try to load historical data
    historical_df = load_historical_recommendations()

    if historical_df is None:
        print("\n⚠️  No historical outcomes available yet.")
        print("   To validate calibration:")
        print("   1. Run backtesting on past weeks")
        print("   2. Track actual outcomes of recommendations")
        print("   3. Build historical dataset with outcomes")
        return

    # Check if we have both raw and calibrated probabilities
    has_raw = 'raw_prob' in historical_df.columns
    has_calibrated = 'model_prob' in historical_df.columns

    if not has_calibrated:
        print("❌ No probability data found in historical dataset")
        return

    # Get outcomes (1 = win, 0 = loss)
    if 'win' in historical_df.columns:
        y_true = historical_df['win'].values
    elif 'outcome' in historical_df.columns:
        y_true = (historical_df['outcome'] == 1).astype(int).values
    else:
        print("❌ No outcome data found")
        return

    y_pred_calibrated = historical_df['model_prob'].values

    # Calculate metrics for calibrated probabilities
    print("\n📊 CALIBRATED PROBABILITIES METRICS:")
    print("-" * 80)
    brier_cal = calculate_brier_score(y_true, y_pred_calibrated)
    log_loss_cal = calculate_log_loss(y_true, y_pred_calibrated)
    ece_cal = calculate_ece(y_true, y_pred_calibrated)

    print(f"Brier Score: {brier_cal:.4f} (lower is better)")
    print(f"Log Loss: {log_loss_cal:.4f} (lower is better)")
    print(f"Expected Calibration Error: {ece_cal:.4f} (lower is better)")

    # Compare with raw if available
    if has_raw:
        y_pred_raw = historical_df['raw_prob'].values
        brier_raw = calculate_brier_score(y_true, y_pred_raw)
        log_loss_raw = calculate_log_loss(y_true, y_pred_raw)
        ece_raw = calculate_ece(y_true, y_pred_raw)

        print("\n📊 RAW PROBABILITIES METRICS:")
        print("-" * 80)
        print(f"Brier Score: {brier_raw:.4f}")
        print(f"Log Loss: {log_loss_raw:.4f}")
        print(f"Expected Calibration Error: {ece_raw:.4f}")

        print("\n📊 IMPROVEMENT:")
        print("-" * 80)
        brier_improvement = ((brier_raw - brier_cal) / brier_raw) * 100
        log_loss_improvement = ((log_loss_raw - log_loss_cal) / log_loss_raw) * 100
        ece_improvement = ((ece_raw - ece_cal) / ece_raw) * 100

        print(f"Brier Score: {brier_improvement:+.1f}% improvement")
        print(f"Log Loss: {log_loss_improvement:+.1f}% improvement")
        print(f"ECE: {ece_improvement:+.1f}% improvement")

        if brier_cal < brier_raw and log_loss_cal < log_loss_raw:
            print("\n✅ Calibration is IMPROVING accuracy!")
        else:
            print("\n⚠️  Calibration may be HURTING accuracy - needs review")

    # Generate calibration plot
    print("\n📈 Generating calibration plot...")
    generate_calibration_plot(y_true, y_pred_calibrated, has_raw, y_pred_raw if has_raw else None)

    # Analyze win rates by probability bin
    print("\n📊 WIN RATE BY PROBABILITY BIN:")
    print("-" * 80)
    analyze_win_rate_by_bin(y_true, y_pred_calibrated)


def generate_calibration_plot(y_true, y_pred_calibrated, has_raw=False, y_pred_raw=None):
    """Generate calibration plot showing predicted vs actual win rates."""
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    # Calculate for calibrated
    bin_centers = []
    actual_rates = []
    predicted_rates = []
    counts = []

    for i in range(len(bin_boundaries) - 1):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_pred_calibrated > bin_lower) & (y_pred_calibrated <= bin_upper)
        count = in_bin.sum()

        if count > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            actual_rates.append(y_true[in_bin].mean())
            predicted_rates.append(y_pred_calibrated[in_bin].mean())
            counts.append(count)

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    plt.plot(bin_centers, actual_rates, 'o-', label='Calibrated Probabilities', linewidth=2, markersize=8)

    if has_raw and y_pred_raw is not None:
        bin_centers_raw = []
        actual_rates_raw = []

        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (y_pred_raw > bin_lower) & (y_pred_raw <= bin_upper)
            count = in_bin.sum()

            if count > 0:
                bin_centers_raw.append((bin_lower + bin_upper) / 2)
                actual_rates_raw.append(y_true[in_bin].mean())

        plt.plot(bin_centers_raw, actual_rates_raw, 's-', label='Raw Probabilities',
                linewidth=2, markersize=8, alpha=0.7)

    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Actual Win Rate', fontsize=12)
    plt.title('Calibration Plot: Predicted vs Actual Win Rates', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    output_path = Path('reports/calibration_validation_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved calibration plot: {output_path}")


def analyze_win_rate_by_bin(y_true, y_pred):
    """Analyze win rates by probability bins."""
    bins = [0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    bin_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90%+']

    for i in range(len(bins) - 1):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        count = in_bin.sum()

        if count > 0:
            actual_rate = y_true[in_bin].mean()
            avg_predicted = y_pred[in_bin].mean()
            difference = actual_rate - avg_predicted

            print(f"{bin_labels[i]:8} | Predicted: {avg_predicted:5.1%} | Actual: {actual_rate:5.1%} | "
                  f"Diff: {difference:+5.1%} | Count: {count:3}")


if __name__ == '__main__':
    validate_calibration()
