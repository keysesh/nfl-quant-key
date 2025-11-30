#!/usr/bin/env python3
"""
Prepare Comprehensive Calibration Training Data
================================================

Combines historical model predictions with actual outcomes to create
comprehensive training dataset for calibrator retraining.

Inputs:
- Historical player prop predictions (with raw probabilities)
- Historical odds data
- Actual game outcomes

Output:
- reports/comprehensive_calibration_training_data.csv
  Columns: week, player, market, line, model_prob_raw, implied_prob, bet_won

Usage:
    python scripts/data/prepare_comprehensive_calibration_training_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def load_all_available_backtest_data() -> pd.DataFrame:
    """
    Load all available backtest data sources that have both raw probabilities and outcomes.
    """

    data_sources = [
        'reports/FRESH_BACKTEST_WEEKS_1_8_CALIBRATED.csv',
        # Add other sources here if they have model_prob_raw + bet_won
    ]

    dfs = []

    for source in data_sources:
        path = Path(source)
        if path.exists():
            try:
                df = pd.read_csv(path)

                required_cols = ['market', 'model_prob_raw', 'bet_won']
                if all(col in df.columns for col in required_cols):
                    # Standardize columns
                    df_standardized = df[['week', 'player', 'market', 'line',
                                          'model_prob_raw', 'bet_won']].copy()

                    # Add implied_prob if available
                    if 'implied_prob' in df.columns:
                        df_standardized['implied_prob'] = df['implied_prob']
                    elif 'american_price' in df.columns:
                        df_standardized['implied_prob'] = df['american_price'].apply(american_to_prob)
                    else:
                        df_standardized['implied_prob'] = 0.5

                    dfs.append(df_standardized)
                    print(f"  ✅ Loaded {len(df)} samples from {source}")
                else:
                    print(f"  ⚠️  Skipped {source} (missing required columns)")

            except Exception as e:
                print(f"  ❌ Error loading {source}: {e}")

    if not dfs:
        raise ValueError("No valid training data sources found")

    # Combine all sources
    combined_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate
    combined_df = combined_df.drop_duplicates(subset=['player', 'week', 'market', 'line'])

    return combined_df


def load_historical_predictions_and_outcomes() -> pd.DataFrame:
    """
    Alternative approach: Load historical model predictions and match with actual stats.

    This can expand the training dataset beyond just the bets we actually made.
    """

    # Look for historical prediction files
    prediction_files = list(Path('data').glob('model_predictions_week*.csv'))

    if not prediction_files:
        print("  ⚠️  No historical prediction files found")
        return pd.DataFrame()

    print(f"  Found {len(prediction_files)} prediction files")

    # Load actual stats
    actual_stats_file = Path('data/processed/actual_stats_2025_weeks_1_8.csv')
    if not actual_stats_file.exists():
        print("  ⚠️  Actual stats file not found")
        return pd.DataFrame()

    actual_stats = pd.read_csv(actual_stats_file)
    print(f"  ✅ Loaded {len(actual_stats)} player-week actual stats")

    # TODO: Implement matching logic
    # This would involve:
    # 1. For each prediction, generate probability for multiple lines
    # 2. Match with actual stats to determine if Over/Under hit
    # 3. Create training samples

    # For now, return empty - we'll use the existing backtest data
    return pd.DataFrame()


def main():
    print("=" * 100)
    print("PREPARING COMPREHENSIVE CALIBRATION TRAINING DATA")
    print("=" * 100)
    print()

    # Method 1: Load existing backtest data with raw probs + outcomes
    print("📊 Loading existing backtest data...")
    backtest_data = load_all_available_backtest_data()
    print(f"✅ Loaded {len(backtest_data):,} training samples from backtest data")
    print()

    # Method 2: Generate additional samples from historical predictions (future enhancement)
    print("🔍 Checking for additional historical prediction data...")
    historical_data = load_historical_predictions_and_outcomes()

    if len(historical_data) > 0:
        print(f"✅ Generated {len(historical_data):,} additional samples")
        combined_data = pd.concat([backtest_data, historical_data], ignore_index=True)
    else:
        print("  ℹ️  No additional data generated (using backtest data only)")
        combined_data = backtest_data

    # Deduplicate
    combined_data = combined_data.drop_duplicates(subset=['player', 'week', 'market', 'line'])

    print()
    print("=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
    print()
    print(f"Total samples: {len(combined_data):,}")
    print(f"Weeks covered: {sorted(combined_data['week'].unique())}")
    print()
    print("Market breakdown:")
    for market, count in combined_data['market'].value_counts().items():
        print(f"  {market:<30}: {count:6,} samples")
    print()

    # Save
    output_file = Path('reports/comprehensive_calibration_training_data.csv')
    combined_data.to_csv(output_file, index=False)
    print(f"💾 Saved to: {output_file}")
    print()

    # Statistics
    print("=" * 100)
    print("TRAINING DATA STATISTICS")
    print("=" * 100)
    print()
    print(f"Raw probability range: {combined_data['model_prob_raw'].min():.4f} - {combined_data['model_prob_raw'].max():.4f}")
    print(f"Raw probability mean: {combined_data['model_prob_raw'].mean():.4f}")
    print(f"Win rate: {combined_data['bet_won'].mean():.2%}")
    print()

    print("NEXT STEPS:")
    print("1. Review the training data quality")
    print("2. Retrain calibrators using this comprehensive dataset:")
    print("   python scripts/train/train_market_specific_calibrators.py")
    print()


if __name__ == "__main__":
    main()
