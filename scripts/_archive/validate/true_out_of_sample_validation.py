#!/usr/bin/env python3
"""
True Out-of-Sample Validation (Leave-One-Week-Out)

This script performs rigorous cross-validation by:
1. Training calibrators on N-1 weeks
2. Testing on the held-out week
3. Repeating for each week
4. Reporting true OOS performance metrics

This eliminates lookahead bias and provides realistic performance estimates.
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression


def train_isotonic_calibrator(train_probs, train_outcomes):
    """Train isotonic regression calibrator."""
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    ir.fit(train_probs, train_outcomes)

    # Extract thresholds for JSON storage
    X_thresh = ir.X_thresholds_.tolist()
    y_thresh = ir.y_thresholds_.tolist()

    return {
        'X_thresholds': X_thresh,
        'y_thresholds': y_thresh,
        'high_prob_threshold': 0.75,
        'high_prob_shrinkage': 0.3
    }


def apply_calibration(prob, calibrator):
    """Apply isotonic calibration to a single probability."""
    X = np.array(calibrator['X_thresholds'])
    y = np.array(calibrator['y_thresholds'])

    idx = np.searchsorted(X, prob)
    if idx == 0:
        return y[0]
    elif idx >= len(X):
        return y[-1]
    else:
        x0, x1 = X[idx-1], X[idx]
        y0, y1 = y[idx-1], y[idx]
        return y0 + (prob - x0) * (y1 - y0) / (x1 - x0)


def leave_one_week_out_validation():
    """Perform true OOS validation."""
    print("=" * 80)
    print("TRUE OUT-OF-SAMPLE VALIDATION (Leave-One-Week-Out)")
    print("=" * 80)
    print()

    # Load backtest data
    base_dir = Path(__file__).parent.parent.parent
    backtest_file = base_dir / "models" / "calibration" / "backtest_2025.csv"

    if not backtest_file.exists():
        print("ERROR: Backtest data not found")
        return None

    df = pd.read_csv(backtest_file)
    weeks = sorted(df['week'].unique())

    print(f"Total predictions: {len(df):,}")
    print(f"Weeks available: {weeks}")
    print()

    # Store results
    oos_results = []

    print("Running leave-one-week-out cross-validation...")
    print("-" * 80)

    for test_week in weeks:
        # Split data
        train_df = df[df['week'] != test_week].copy()
        test_df = df[df['week'] == test_week].copy()

        # Train calibrator on training weeks only
        train_probs = train_df['predicted_prob_over'].values
        train_outcomes = train_df['hit_over'].values

        calibrator = train_isotonic_calibrator(train_probs, train_outcomes)

        # Apply calibration to test week
        test_df['calibrated_prob'] = test_df['predicted_prob_over'].apply(
            lambda p: apply_calibration(p, calibrator)
        )

        # Calculate metrics for test week
        raw_brier = np.mean((test_df['predicted_prob_over'] - test_df['hit_over']) ** 2)
        cal_brier = np.mean((test_df['calibrated_prob'] - test_df['hit_over']) ** 2)

        # Correlation
        raw_corr = test_df['predicted_prob_over'].corr(test_df['hit_over'])
        cal_corr = test_df['calibrated_prob'].corr(test_df['hit_over'])

        # Simulated ROI (at 10% edge threshold)
        market_prob = 0.5  # Assuming -110 both ways
        edge_threshold = 0.10

        qualified_bets = test_df[test_df['calibrated_prob'] - market_prob > edge_threshold]
        if len(qualified_bets) > 0:
            wins = qualified_bets['hit_over'].sum()
            losses = len(qualified_bets) - wins
            profit = (wins * 100) - (losses * 110)
            roi = profit / (len(qualified_bets) * 110) * 100
        else:
            roi = 0.0
            wins = 0
            losses = 0

        result = {
            'week': int(test_week),
            'n_predictions': int(len(test_df)),
            'n_training': int(len(train_df)),
            'raw_brier': float(round(raw_brier, 4)),
            'calibrated_brier': float(round(cal_brier, 4)),
            'brier_improvement_pct': float(round((raw_brier - cal_brier) / raw_brier * 100, 2)),
            'raw_correlation': float(round(raw_corr, 4)),
            'calibrated_correlation': float(round(cal_corr, 4)),
            'qualified_bets': int(len(qualified_bets)),
            'wins': int(wins),
            'losses': int(losses),
            'roi_pct': float(round(roi, 2))
        }

        oos_results.append(result)

        print(f"  Week {test_week}: N={len(test_df):4d}, "
              f"Raw Brier={raw_brier:.4f}, "
              f"Cal Brier={cal_brier:.4f} ({result['brier_improvement_pct']:+.1f}%), "
              f"ROI={roi:+.1f}%")

    print()

    # Aggregate statistics
    print("=" * 80)
    print("AGGREGATED OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 80)

    # Average metrics
    avg_raw_brier = np.mean([r['raw_brier'] for r in oos_results])
    avg_cal_brier = np.mean([r['calibrated_brier'] for r in oos_results])
    avg_improvement = np.mean([r['brier_improvement_pct'] for r in oos_results])

    total_qualified = sum([r['qualified_bets'] for r in oos_results])
    total_wins = sum([r['wins'] for r in oos_results])
    total_losses = sum([r['losses'] for r in oos_results])

    if total_qualified > 0:
        overall_profit = (total_wins * 100) - (total_losses * 110)
        overall_roi = overall_profit / (total_qualified * 110) * 100
    else:
        overall_roi = 0.0

    print(f"\nMean Raw Brier Score: {avg_raw_brier:.4f}")
    print(f"Mean Calibrated Brier Score: {avg_cal_brier:.4f}")
    print(f"Mean Improvement: {avg_improvement:.2f}%")
    print()

    print(f"Total Qualified Bets (10% edge): {total_qualified}")
    print(f"Total Wins: {total_wins}")
    print(f"Total Losses: {total_losses}")
    print(f"Win Rate: {total_wins/total_qualified*100:.1f}%")
    print(f"OVERALL TRUE OOS ROI: {overall_roi:+.2f}%")
    print()

    # Worst-case week
    worst_week = min(oos_results, key=lambda x: x['roi_pct'])
    best_week = max(oos_results, key=lambda x: x['roi_pct'])

    print(f"Best Week: Week {best_week['week']} (ROI: {best_week['roi_pct']:+.1f}%)")
    print(f"Worst Week: Week {worst_week['week']} (ROI: {worst_week['roi_pct']:+.1f}%)")
    print()

    # Compare to in-sample metrics
    print("=" * 80)
    print("COMPARISON: IN-SAMPLE vs TRUE OUT-OF-SAMPLE")
    print("=" * 80)

    # In-sample (training on all data)
    in_sample_calibrator = train_isotonic_calibrator(
        df['predicted_prob_over'].values,
        df['hit_over'].values
    )
    df['in_sample_cal'] = df['predicted_prob_over'].apply(
        lambda p: apply_calibration(p, in_sample_calibrator)
    )
    in_sample_brier = np.mean((df['in_sample_cal'] - df['hit_over']) ** 2)

    print(f"In-Sample Calibrated Brier: {in_sample_brier:.4f} (optimistic)")
    print(f"True OOS Calibrated Brier: {avg_cal_brier:.4f} (realistic)")
    print(f"Bias (In-Sample - OOS): {in_sample_brier - avg_cal_brier:+.4f}")
    print()

    if in_sample_brier < avg_cal_brier:
        print("WARNING: In-sample performance is better than OOS - some overfitting present")
    else:
        print("Good: OOS performance similar to in-sample - minimal overfitting")

    # Save results
    output_file = base_dir / "reports" / "true_oos_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'avg_raw_brier': float(avg_raw_brier),
                'avg_calibrated_brier': float(avg_cal_brier),
                'avg_improvement_pct': float(avg_improvement),
                'total_qualified_bets': int(total_qualified),
                'total_wins': int(total_wins),
                'total_losses': int(total_losses),
                'overall_roi_pct': float(overall_roi),
                'in_sample_brier': float(round(in_sample_brier, 4))
            },
            'weekly_results': oos_results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return oos_results


if __name__ == '__main__':
    leave_one_week_out_validation()
