#!/usr/bin/env python3
"""
Analyze systematic bias in ALL player stat predictions.

Checks:
- Rushing yards
- Receiving yards
- Passing yards
- Receptions
- Rushing TDs
- Receiving TDs
- Passing TDs

Outputs calibration factors for each stat type.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load NFLverse data
nflverse_path = Path('data/nflverse_cache/stats_player_week_2025.csv')
nflverse_df = pd.read_csv(nflverse_path, low_memory=False)

print("="*80)
print("COMPREHENSIVE STAT BIAS ANALYSIS - WEEKS 1-8")
print("="*80)

# Load predictions from backtest
# We need to run predictions for each week and compare to actuals

from scripts.backtest.backtest_td_predictions_integrated import (
    generate_predictions_integrated,
    load_schedule
)

results = {
    'rushing_yards': {'predicted': [], 'actual': []},
    'receiving_yards': {'predicted': [], 'actual': []},
    'passing_yards': {'predicted': [], 'actual': []},
    'receptions': {'predicted': [], 'actual': []},
    'rushing_tds': {'predicted': [], 'actual': []},
    'receiving_tds': {'predicted': [], 'actual': []},
    'passing_tds': {'predicted': [], 'actual': []},
}

print("\nProcessing weeks 1-8...")

for week in range(1, 9):
    print(f"\nWeek {week}...")

    # Generate predictions
    df = generate_predictions_integrated(week, nflverse_df)

    if df.empty:
        continue

    # Get actuals from NFLverse
    week_actuals = nflverse_df[nflverse_df['week'] == week].copy()

    # Merge predictions with actuals
    merged = df.merge(
        week_actuals,
        left_on='player_name',
        right_on='player_display_name',
        how='inner'
    )

    print(f"  Matched {len(merged)} players")

    # Collect stats by position
    for _, row in merged.iterrows():
        position = row.get('position', 'UNK')

        # Rushing stats (RB primarily)
        if position in ['RB', 'QB']:
            pred_rush_yds = row.get('rushing_yards_mean', 0)
            actual_rush_yds = row.get('rushing_yards', 0)
            pred_rush_tds = row.get('rushing_tds_mean', 0)
            actual_rush_tds = row.get('rushing_tds', 0)

            if pred_rush_yds > 0 or actual_rush_yds > 0:
                results['rushing_yards']['predicted'].append(pred_rush_yds)
                results['rushing_yards']['actual'].append(actual_rush_yds)

            if pred_rush_tds > 0 or actual_rush_tds > 0:
                results['rushing_tds']['predicted'].append(pred_rush_tds)
                results['rushing_tds']['actual'].append(actual_rush_tds)

        # Receiving stats (WR, TE, RB)
        if position in ['WR', 'TE', 'RB']:
            pred_rec_yds = row.get('receiving_yards_mean', 0)
            actual_rec_yds = row.get('receiving_yards', 0)
            pred_rec = row.get('receptions_mean', 0)
            actual_rec = row.get('receptions', 0)
            pred_rec_tds = row.get('receiving_tds_mean', 0)
            actual_rec_tds = row.get('receiving_tds', 0)

            if pred_rec_yds > 0 or actual_rec_yds > 0:
                results['receiving_yards']['predicted'].append(pred_rec_yds)
                results['receiving_yards']['actual'].append(actual_rec_yds)

            if pred_rec > 0 or actual_rec > 0:
                results['receptions']['predicted'].append(pred_rec)
                results['receptions']['actual'].append(actual_rec)

            if pred_rec_tds > 0 or actual_rec_tds > 0:
                results['receiving_tds']['predicted'].append(pred_rec_tds)
                results['receiving_tds']['actual'].append(actual_rec_tds)

        # Passing stats (QB only)
        if position == 'QB':
            pred_pass_yds = row.get('passing_yards_mean', 0)
            actual_pass_yds = row.get('passing_yards', 0)
            pred_pass_tds = row.get('passing_tds_mean', 0)
            actual_pass_tds = row.get('passing_tds', 0)

            if pred_pass_yds > 0 or actual_pass_yds > 0:
                results['passing_yards']['predicted'].append(pred_pass_yds)
                results['passing_yards']['actual'].append(actual_pass_yds)

            if pred_pass_tds > 0 or actual_pass_tds > 0:
                results['passing_tds']['predicted'].append(pred_pass_tds)
                results['passing_tds']['actual'].append(actual_pass_tds)

print("\n" + "="*80)
print("RESULTS - SYSTEMATIC BIAS ANALYSIS")
print("="*80)

calibration_factors = {}

for stat_name, data in results.items():
    if len(data['predicted']) == 0:
        continue

    pred = np.array(data['predicted'])
    actual = np.array(data['actual'])

    # Calculate metrics
    mean_pred = np.mean(pred)
    mean_actual = np.mean(actual)
    bias = mean_actual - mean_pred
    bias_pct = (bias / mean_actual * 100) if mean_actual > 0 else 0

    # Calculate calibration factor (multiplicative)
    calibration_factor = mean_actual / mean_pred if mean_pred > 0 else 1.0

    # RMSE
    rmse = np.sqrt(np.mean((pred - actual)**2))

    # MAE
    mae = np.mean(np.abs(pred - actual))

    print(f"\n{stat_name.upper().replace('_', ' ')}:")
    print(f"  Samples: {len(pred)}")
    print(f"  Mean Predicted: {mean_pred:.2f}")
    print(f"  Mean Actual: {mean_actual:.2f}")
    print(f"  Bias: {bias:+.2f} ({bias_pct:+.1f}%)")
    print(f"  Calibration Factor: {calibration_factor:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")

    calibration_factors[stat_name] = calibration_factor

    # Flag if significant bias
    if abs(bias_pct) > 5:
        print(f"  ⚠️  SIGNIFICANT BIAS - NEEDS CALIBRATION")
    else:
        print(f"  ✅ Well calibrated")

print("\n" + "="*80)
print("CALIBRATION FACTORS SUMMARY")
print("="*80)

for stat_name, factor in calibration_factors.items():
    bias_pct = (factor - 1.0) * 100
    print(f"{stat_name:20s}: {factor:6.4f}  ({bias_pct:+6.2f}%)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("\nStats needing calibration (|bias| > 5%):")
for stat_name, factor in calibration_factors.items():
    bias_pct = abs((factor - 1.0) * 100)
    if bias_pct > 5:
        print(f"  - {stat_name}: {factor:.4f}x adjustment needed")

print("\n✅ Analysis complete")
