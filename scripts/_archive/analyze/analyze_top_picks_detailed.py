#!/usr/bin/env python3
"""
Detailed analysis of betting quality vs quantity trade-off
"""

import pandas as pd
import numpy as np

df = pd.read_csv('reports/framework_backtest_weeks_1_7_fixed.csv')
fw_df = df[df['model_prob'].notna()].copy()

# Apply new filters
fw_df = fw_df[(fw_df['edge'] >= 0.10) & (fw_df['model_prob'] >= 0.75)].copy()

print("=" * 80)
print("DETAILED QUALITY vs QUANTITY ANALYSIS")
print("=" * 80)
print()

# Sort by confidence
fw_df_sorted = fw_df.sort_values('model_prob', ascending=False).reset_index(drop=True)

print("🔍 DETAILED BREAKDOWN BY BET RANK (Most Confident First)")
print()
print(f"{'Rank Range':<15} {'Bets':<10} {'Win Rate':<12} {'Avg ROI':<15} {'Total ROI':<15}")
print("-" * 80)

ranges = [
    (1, 10, "1-10"),
    (11, 20, "11-20"),
    (21, 30, "21-30"),
    (31, 50, "31-50"),
    (51, 75, "51-75"),
    (76, 100, "76-100"),
    (101, 150, "101-150"),
    (151, 200, "151-200"),
    (201, 300, "201-300"),
    (301, 500, "301-500"),
    (501, 630, "501-630"),
]

results = []

for start, end, label in ranges:
    if end <= len(fw_df_sorted):
        subset = fw_df_sorted.iloc[start-1:end]

        n_bets = len(subset)
        n_wins = subset['bet_won'].sum()
        wr = n_wins / n_bets if n_bets > 0 else 0
        avg_r = subset['unit_return'].sum() / n_bets if n_bets > 0 else 0
        total_r = subset['unit_return'].sum()

        results.append((label, n_bets, wr, avg_r, total_r))

        print(f"{label:<15} {n_bets:<10} {wr:<12.1%} {avg_r:<15.2%} {total_r:<15.2f}")

print()

# Find the sweet spot
print("=" * 80)
print("📊 FINDING THE SWEET SPOT")
print("=" * 80)
print()

print(f"{'Top N':<10} {'Win Rate':<12} {'Avg ROI':<15} {'Total ROI':<15} {'Score*':<15}")
print("-" * 80)

best_score = -999
best_n = None

for top_n in [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 630]:
    top_bets = fw_df_sorted.head(top_n)

    n_bets = len(top_bets)
    n_wins = top_bets['bet_won'].sum()
    wr = n_wins / n_bets if n_bets > 0 else 0
    avg_r = top_bets['unit_return'].sum() / n_bets if n_bets > 0 else 0
    total_r = top_bets['unit_return'].sum()

    # Score = win rate * 10 + total ROI / 10 (to balance both metrics)
    score = wr * 10 + total_r / 10

    if score > best_score:
        best_score = score
        best_n = top_n

    print(f"{top_n:<10} {wr:<12.1%} {avg_r:<15.2%} {total_r:<15.2f} {score:<15.2f}")

print()
print("*Score = win_rate × 10 + total_ROI / 10")
print()

# Show the best recommendation
best_bets = fw_df_sorted.head(best_n)
best_wins = best_bets['bet_won'].sum()
best_wr = best_wins / len(best_bets)
best_avg_roi = best_bets['unit_return'].sum() / len(best_bets)
best_total_roi = best_bets['unit_return'].sum()

print("=" * 80)
print("🎯 RECOMMENDATION")
print("=" * 80)
print()
print(f"✅ OPTIMAL STRATEGY: Top {best_n} most confident bets")
print()
print(f"Current approach (all 630 bets):")
print(f"  Win Rate: 54.9%")
print(f"  Avg ROI: 4.85%")
print(f"  Total ROI: 30.55 units")
print(f"  Volume: ~90 bets/week")
print()
print(f"Recommended approach (top {best_n} bets):")
print(f"  Win Rate: {best_wr:.1%}")
print(f"  Avg ROI: {best_avg_roi:.2%}")
print(f"  Total ROI: {best_total_roi:.2f} units")
print(f"  Volume: ~{best_n//7:.0f} bets/week")
print()
print(f"📈 IMPROVEMENT:")
print(f"  Win Rate: {best_wr - 0.549:.1%}")
print(f"  Avg ROI: {best_avg_roi - 0.0485:.2%}")
print(f"  Volume reduction: {(630-best_n)/630:.0%} fewer bets")
print()

# Show weekly breakdown for recommended strategy
print("=" * 80)
print(f"📅 WEEKLY BREAKDOWN (Top {best_n} bets)")
print("=" * 80)
print()

for week in sorted(fw_df['week'].unique()):
    week_data = fw_df[fw_df['week'] == week].sort_values('model_prob', ascending=False)

    if len(week_data) >= best_n:
        top_bets = week_data.head(best_n)
    else:
        top_bets = week_data

    n_bets = len(top_bets)
    n_wins = top_bets['bet_won'].sum()
    wr = n_wins / n_bets if n_bets > 0 else 0
    roi = top_bets['unit_return'].sum()

    print(f"Week {week}: {n_bets:2} bets | WR: {wr:5.1%} | ROI: {roi:+7.2f}")

print()
