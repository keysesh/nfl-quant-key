#!/usr/bin/env python3
"""
Retrain Calibrators on 2025 Season Data

Uses walk-forward validation on 2025 weeks 1-10 to train
calibrators that are appropriate for Week 11 predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
import nfl_quant.data.dynamic_parameters as dp

PROJECT_ROOT = Path(__file__).parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "models" / "calibration"


def load_2025_data():
    """Load 2025 season data only."""
    dp._provider_instance = None  # Reset to reload
    provider = dp.get_parameter_provider()
    df = provider.weekly_data

    # Filter to 2025 only
    df_2025 = df[df['season'] == 2025].copy()
    print(f"Loaded 2025 data: {len(df_2025)} rows")
    print(f"Weeks available: {sorted(df_2025['week'].unique())}")

    return df_2025


def generate_backtest_predictions(df):
    """
    Generate predictions for weeks 5-10 using walk-forward validation.
    For each week N, use weeks 1 to N-1 to predict.
    """
    print("\n=== GENERATING 2025 BACKTEST PREDICTIONS ===")

    results = []
    markets = {
        'receptions': 'receptions',
        'receiving_yards': 'receiving_yards',
        'rushing_yards': 'rushing_yards',
        'carries': 'carries',
        'targets': 'targets',
    }

    # Typical lines for each market
    typical_lines = {
        'receptions': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        'receiving_yards': [30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5],
        'rushing_yards': [40.5, 50.5, 60.5, 70.5, 80.5],
        'carries': [10.5, 12.5, 14.5, 16.5, 18.5],
        'targets': [4.5, 5.5, 6.5, 7.5, 8.5],
    }

    # Walk-forward: predict weeks 5-10 (need at least 4 weeks history)
    for predict_week in range(5, 11):
        print(f"\nPredicting Week {predict_week}...")

        # Get players who played this week
        week_data = df[df['week'] == predict_week]
        n_predictions = 0

        for _, row in week_data.iterrows():
            player_name = row['player_name']
            team = row['recent_team']
            position = row['position']

            # Get historical data (weeks 1 to predict_week-1)
            hist_data = df[
                (df['player_name'] == player_name) &
                (df['recent_team'] == team) &
                (df['week'] < predict_week)
            ]

            if len(hist_data) < 3:  # Need at least 3 prior games
                continue

            # Get last 4 weeks
            recent = hist_data.nlargest(4, 'week')
            n_games = len(recent)

            # Get position averages (starter-level)
            pos_data = df[(df['position'] == position) & (df['week'] < predict_week)]

            for market_name, col in markets.items():
                if col not in recent.columns or col not in row.index:
                    continue

                actual_val = row[col]
                if pd.isna(actual_val):
                    continue

                # Calculate shrunk mean
                raw_mean = recent[col].mean()

                # Starter-level position mean
                median_val = pos_data[col].quantile(0.5)
                starter_data = pos_data[pos_data[col] >= median_val]
                pos_mean = starter_data[col].mean() if len(starter_data) > 0 else pos_data[col].mean()

                # Bayesian shrinkage
                SHRINKAGE_STRENGTH = 3.0
                player_weight = n_games / (n_games + SHRINKAGE_STRENGTH)
                shrunk_mean = player_weight * raw_mean + (1 - player_weight) * pos_mean

                # Calculate std
                if n_games > 1:
                    raw_std = recent[col].std()
                else:
                    raw_std = shrunk_mean * 0.5

                # Variance inflation (based on actual population CV)
                if shrunk_mean > 0:
                    player_cvs = df.groupby('player_name')[col].agg(['mean', 'std'])
                    player_cvs = player_cvs[player_cvs['mean'] > 0.1]
                    if len(player_cvs) > 0:
                        actual_cv = (player_cvs['std'] / player_cvs['mean']).median()
                    else:
                        actual_cv = 0.8

                    current_cv = raw_std / shrunk_mean if shrunk_mean > 0 else 0.5
                    inflation = min(3.0, max(1.0, actual_cv / current_cv)) if current_cv > 0 else 1.5
                    adjusted_std = raw_std * max(inflation, 1.2)
                else:
                    adjusted_std = 0.5

                # Generate predictions for relevant lines
                for line in typical_lines.get(market_name, []):
                    # Only predict if line is near the player's mean
                    if shrunk_mean > 0.1 and 0.3 * shrunk_mean <= line <= 3.0 * shrunk_mean:
                        # Monte Carlo simulation
                        samples = np.random.normal(shrunk_mean, adjusted_std, 5000)
                        samples = np.maximum(0, samples)
                        prob_over = np.mean(samples > line)

                        # Actual outcome
                        hit_over = 1 if actual_val > line else 0

                        results.append({
                            'player_name': player_name,
                            'team': team,
                            'position': position,
                            'week': predict_week,
                            'market': market_name,
                            'line': line,
                            'predicted_prob_over': prob_over,
                            'actual': actual_val,
                            'hit_over': hit_over,
                        })
                        n_predictions += 1

        print(f"  Generated {n_predictions} predictions for week {predict_week}")

    results_df = pd.DataFrame(results)
    print(f"\nTotal predictions: {len(results_df)}")
    return results_df


def train_calibrators(results_df):
    """Train isotonic calibrators on the backtest results."""
    print("\n=== TRAINING CALIBRATORS ===")

    raw_probs = results_df['predicted_prob_over'].values
    actuals = results_df['hit_over'].astype(float).values

    # Overall calibrator
    print("Training overall calibrator...")
    overall_cal = NFLProbabilityCalibrator(
        high_prob_threshold=0.70,
        high_prob_shrinkage=0.3
    )
    overall_cal.fit(raw_probs, actuals)
    overall_cal.save(str(CALIBRATION_DIR / "overall_calibrator.json"))

    # Check calibration quality
    calibrated_probs = overall_cal.transform(raw_probs)

    # Expected Calibration Error
    def calc_ece(probs, actuals, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_pred = probs[mask].mean()
                bin_actual = actuals[mask].mean()
                ece += (mask.sum() / len(probs)) * abs(bin_pred - bin_actual)
        return ece

    ece_raw = calc_ece(raw_probs, actuals)
    ece_cal = calc_ece(calibrated_probs, actuals)

    print(f"  ECE Raw: {ece_raw:.4f}")
    print(f"  ECE Calibrated: {ece_cal:.4f}")
    print(f"  Improvement: {(ece_raw - ece_cal)/ece_raw*100:.1f}%")

    # Show calibration curve
    print("\nCalibration curve:")
    for raw_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cal_p = overall_cal.transform(np.array([raw_p]))[0]
        print(f"  Raw {raw_p:.1f} -> Calibrated {cal_p:.3f}")

    # Position-specific calibrators
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_mask = results_df['position'] == position
        if pos_mask.sum() >= 50:
            pos_raw = raw_probs[pos_mask]
            pos_actuals = actuals[pos_mask]

            pos_cal = NFLProbabilityCalibrator()
            pos_cal.fit(pos_raw, pos_actuals)
            pos_cal.save(str(CALIBRATION_DIR / f"position_{position}_calibrator.json"))
            print(f"\nTrained {position} calibrator ({pos_mask.sum()} samples)")

    # Market-specific calibrators
    for market in results_df['market'].unique():
        market_mask = results_df['market'] == market
        if market_mask.sum() >= 50:
            market_raw = raw_probs[market_mask]
            market_actuals = actuals[market_mask]

            market_cal = NFLProbabilityCalibrator()
            market_cal.fit(market_raw, market_actuals)
            market_cal.save(str(CALIBRATION_DIR / f"market_{market}_calibrator.json"))
            print(f"Trained {market} calibrator ({market_mask.sum()} samples)")

    return overall_cal


def analyze_model_accuracy(results_df):
    """Check if projections are actually predictive."""
    print("\n=== MODEL ACCURACY ANALYSIS ===")

    # Check by market
    for market in results_df['market'].unique():
        market_data = results_df[results_df['market'] == market]
        hit_rate = market_data['hit_over'].mean()
        mean_prob = market_data['predicted_prob_over'].mean()

        print(f"\n{market}:")
        print(f"  N predictions: {len(market_data)}")
        print(f"  Mean P(Over): {mean_prob:.3f}")
        print(f"  Actual hit rate: {hit_rate:.3f}")
        print(f"  Calibration error: {abs(mean_prob - hit_rate):.3f}")

    # Overall
    print(f"\nOverall:")
    print(f"  Mean predicted P(Over): {results_df['predicted_prob_over'].mean():.3f}")
    print(f"  Actual over rate: {results_df['hit_over'].mean():.3f}")


def main():
    print("=" * 70)
    print("RETRAINING CALIBRATORS ON 2025 SEASON DATA")
    print("=" * 70)

    # Load data
    df = load_2025_data()

    # Generate backtest predictions
    results_df = generate_backtest_predictions(df)

    if len(results_df) == 0:
        print("ERROR: No predictions generated")
        return

    # Save backtest results
    results_df.to_csv(CALIBRATION_DIR / "backtest_2025.csv", index=False)

    # Analyze model accuracy
    analyze_model_accuracy(results_df)

    # Train calibrators
    train_calibrators(results_df)

    print("\n" + "=" * 70)
    print("CALIBRATORS RETRAINED ON 2025 DATA")
    print("=" * 70)


if __name__ == "__main__":
    main()
