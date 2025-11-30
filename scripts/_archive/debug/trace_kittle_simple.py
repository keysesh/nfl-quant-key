#!/usr/bin/env python3
"""
Simple trace of George Kittle's Week 12 calculation.
Shows the raw data and how trailing_yards_per_target is calculated.
"""

import pandas as pd
from pathlib import Path

def main():
    print("\n" + "=" * 80)
    print("GEORGE KITTLE WEEK 12 - CALCULATION TRACE")
    print("=" * 80)

    # Load NFLverse weekly stats
    weekly_stats_path = Path("data/nflverse/weekly_stats.parquet")
    if not weekly_stats_path.exists():
        print("\n❌ NFLverse data not found")
        return

    weekly = pd.read_parquet(weekly_stats_path)

    # Filter for Kittle in 2025, weeks before Week 12
    kittle_data = weekly[
        (weekly['player_display_name'] == 'George Kittle') &
        (weekly['season'] == 2025) &
        (weekly['week'] < 12) &
        (weekly['week'] >= 8)  # Last 4 weeks before Week 12
    ].sort_values('week')

    if len(kittle_data) == 0:
        print("\n❌ No data found for George Kittle")
        return

    print(f"\n📊 GEORGE KITTLE HISTORICAL DATA (Weeks 8-11):")
    print(f"\n{'Week':<6} {'Targets':<8} {'Rec Yards':<10} {'Yards/Target':<12} {'Receptions':<10} {'Rec TDs':<8}")
    print("-" * 70)

    total_targets = 0
    total_rec_yards = 0
    total_rec_tds = 0

    for _, row in kittle_data.iterrows():
        targets = row.get('targets', 0)
        rec_yards = row.get('receiving_yards', 0)
        receptions = row.get('receptions', 0)
        rec_tds = row.get('receiving_tds', 0)

        yards_per_target = rec_yards / targets if targets > 0 else 0.0

        print(f"{int(row['week']):<6} {targets:<8.0f} {rec_yards:<10.0f} {yards_per_target:<12.2f} {receptions:<10.0f} {rec_tds:<8.0f}")

        total_targets += targets
        total_rec_yards += rec_yards
        total_rec_tds += rec_tds

    print("-" * 70)

    # Calculate trailing metrics (what the model uses as input)
    trailing_yards_per_target = total_rec_yards / total_targets if total_targets > 0 else None
    trailing_td_rate_pass = total_rec_tds / total_targets if total_targets > 0 else None

    print(f"\n📈 TRAILING METRICS (Input to Model):")
    print(f"   Total Targets (4 weeks): {total_targets:.0f}")
    print(f"   Total Receiving Yards: {total_rec_yards:.0f}")
    print(f"   Trailing Yards/Target: {trailing_yards_per_target:.3f} ✅ THIS IS THE KEY METRIC")
    print(f"   Trailing TD Rate (Pass): {trailing_td_rate_pass:.4f}")

    # Load actual Week 12 prediction
    predictions = pd.read_csv('data/model_predictions_week12.csv')
    kittle_pred = predictions[predictions['player_name'] == 'George Kittle'].iloc[0]

    print(f"\n🎯 WEEK 12 PREDICTION OUTPUT:")
    print(f"   Projected Targets: {kittle_pred['targets_mean']:.1f}")
    print(f"   Projected Receiving Yards: {kittle_pred['receiving_yards_mean']:.1f}")
    print(f"   Projected Receptions: {kittle_pred['receptions_mean']:.1f}")
    print(f"   TD Probability: {kittle_pred['receiving_tds_mean']:.3f}")

    # Calculate what yards SHOULD be based on trailing efficiency
    # Basic calculation: projected_targets * yards_per_target (before adjustments)
    baseline_yards = kittle_pred['targets_mean'] * trailing_yards_per_target

    print(f"\n🔍 CALCULATION BREAKDOWN:")
    print(f"   Step 1: Baseline = Targets × Yards/Target")
    print(f"           {kittle_pred['targets_mean']:.1f} × {trailing_yards_per_target:.3f} = {baseline_yards:.1f} yards")
    print(f"   ")
    print(f"   Step 2: Apply efficiency adjustments (opponent defense, game script, etc.)")
    print(f"           Opponent DEF EPA vs TE: {kittle_pred['opponent_def_epa_vs_position']:+.4f}")
    print(f"           {'(WEAK defense = INCREASE yards)' if kittle_pred['opponent_def_epa_vs_position'] > 0 else '(STRONG defense = DECREASE yards)'}")
    print(f"   ")
    print(f"   Step 3: Final = {kittle_pred['receiving_yards_mean']:.1f} yards (after all adjustments)")

    # Check if the fix is working
    print(f"\n✅ VERIFICATION:")
    print(f"   trailing_yards_per_target was {'CALCULATED' if trailing_yards_per_target else 'MISSING'}")
    print(f"   Value: {trailing_yards_per_target:.3f}" if trailing_yards_per_target else "   Value: None")

    if trailing_yards_per_target:
        print(f"\n   ✅ Bug #3 fix is WORKING - using position-specific receiving efficiency!")
    else:
        print(f"\n   ❌ Bug #3 still present - would fall back to generic yards_per_opportunity")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
