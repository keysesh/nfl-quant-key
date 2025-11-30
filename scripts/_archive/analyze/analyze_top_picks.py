#!/usr/bin/env python3
"""
Analyze what happens if we only take the top N most confident picks
instead of all picks that meet the minimum thresholds.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ANALYZING TOP PICKS - DOES QUALITY > QUANTITY?")
print("=" * 80)
print()

# Load framework backtest data
df = pd.read_csv('reports/framework_backtest_weeks_1_7_fixed.csv')
fw_df = df[df['model_prob'].notna()].copy()

print(f"✅ Loaded {len(fw_df):,} bets with model predictions")
print()

# Apply the new filters (edge >= 10%, confidence >= 75%)
fw_df = fw_df[(fw_df['edge'] >= 0.10) & (fw_df['model_prob'] >= 0.75)].copy()

print(f"✅ After new filters (edge≥10%, conf≥75%): {len(fw_df):,} bets")
print()

# Calculate weekly volume
bets_per_week = len(fw_df) / 7
print(f"📊 Average bets per week: {bets_per_week:.0f}")
print()

# Calculate win rate and ROI for the full dataset
total_bets = len(fw_df)
total_wins = fw_df['bet_won'].sum()
total_roi = fw_df['unit_return'].sum()
win_rate = total_wins / total_bets
avg_roi = total_roi / total_bets

print("📈 CURRENT APPROACH (All bets meeting thresholds):")
print(f"   Total bets: {total_bets:,}")
print(f"   Win rate: {win_rate:.1%}")
print(f"   Total ROI: {total_roi:.2f} units")
print(f"   Average ROI per bet: {avg_roi:.2%}")
print()

# Now analyze what happens if we only take the TOP N most confident picks
print("🎯 WHAT IF WE ONLY TAKE THE TOP N MOST CONFIDENT PICKS?")
print()
print(f"{'Top N':<10} {'Bets':<10} {'Win Rate':<12} {'Total ROI':<12} {'Avg ROI':<12}")
print("-" * 70)

# Test different top N values
for top_n in [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 630]:
    # Get top N by confidence
    top_bets = fw_df.nlargest(top_n, 'model_prob')

    n_bets = len(top_bets)
    n_wins = top_bets['bet_won'].sum()
    wr = n_wins / n_bets if n_bets > 0 else 0
    n_roi = top_bets['unit_return'].sum()
    avg_r = n_roi / n_bets if n_bets > 0 else 0

    print(f"{top_n:<10} {n_bets:<10} {wr:<12.1%} {n_roi:<12.2f} {avg_r:<12.2%}")

print()

# Let's also look at this per week to understand the pattern better
print("📅 WHAT HAPPENS PER WEEK?")
print()
print(f"{'Week':<8} {'All Bets':<12} {'Win Rate':<12} {'Top 10':<12} {'Win Rate':<12}")
print("-" * 70)

for week in sorted(fw_df['week'].unique()):
    week_data = fw_df[fw_df['week'] == week].copy()

    all_bets = len(week_data)
    all_wins = week_data['bet_won'].sum()
    all_wr = all_wins / all_bets if all_bets > 0 else 0

    top_10 = week_data.nlargest(10, 'model_prob')
    top_wins = top_10['bet_won'].sum() if len(top_10) > 0 else 0
    top_wr = top_wins / len(top_10) if len(top_10) > 0 else 0

    print(f"Week {week:<6} {all_bets:<12} {all_wr:<12.1%} {len(top_10):<12} {top_wr:<12.1%}")

print()

# Now let's see confidence distribution
print("📊 CONFIDENCE DISTRIBUTION OF ALL BETS")
print()
print(f"{'Conf Range':<15} {'Bets':<10} {'Win Rate':<12} {'Avg ROI':<12}")
print("-" * 60)

ranges = [
    (0.75, 0.80, "75-80%"),
    (0.80, 0.85, "80-85%"),
    (0.85, 0.90, "85-90%"),
    (0.90, 0.95, "90-95%"),
    (0.95, 1.00, "95-100%"),
]

for low, high, label in ranges:
    subset = fw_df[(fw_df['model_prob'] >= low) & (fw_df['model_prob'] < high)]

    if len(subset) > 0:
        n_bets = len(subset)
        n_wins = subset['bet_won'].sum()
        wr = n_wins / n_bets
        avg_r = subset['unit_return'].sum() / n_bets

        print(f"{label:<15} {n_bets:<10} {wr:<12.1%} {avg_r:<12.2%}")

print()

# Recommendation
print("=" * 80)
print("💡 RECOMMENDATION")
print("=" * 80)

# Find optimal trade-off
optimal_n = None
best_total_roi = -999

for top_n in [20, 30, 50, 75, 100]:
    top_bets = fw_df.nlargest(top_n, 'model_prob')
    total_roi = top_bets['unit_return'].sum()
    win_rate = top_bets['bet_won'].sum() / len(top_bets)

    # Weight by win rate and total ROI
    score = win_rate * 100 + total_roi / 5

    if total_roi > best_total_roi:
        best_total_roi = total_roi
        optimal_n = top_n

print(f"\n✅ Best strategy: Top {optimal_n} most confident bets per week")
print(f"   Expected: ~{optimal_n} bets/week instead of {bets_per_week:.0f} bets/week")
print(f"   This would reduce volume but improve quality")
print()
