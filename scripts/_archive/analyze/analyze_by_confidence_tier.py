#!/usr/bin/env python3
"""
Analyze performance by confidence tiers to find optimal strategy
"""

import pandas as pd
import numpy as np

df = pd.read_csv('reports/framework_backtest_weeks_1_7_fixed.csv')
fw_df = df[df['model_prob'].notna()].copy()

print("=" * 80)
print("CONFIDENCE TIER ANALYSIS")
print("=" * 80)
print()

# Apply new filters
fw_df = fw_df[(fw_df['edge'] >= 0.10) & (fw_df['model_prob'] >= 0.75)].copy()

# Sort by confidence
fw_df_sorted = fw_df.sort_values('model_prob', ascending=False).reset_index(drop=True)

# Define confidence tiers
print("🔍 PERFORMANCE BY CONFIDENCE TIER")
print()
print(f"{'Confidence Tier':<20} {'Bets':<10} {'Win Rate':<12} {'Avg ROI':<15} {'Total ROI':<15}")
print("-" * 85)

# Test different confidence cutoffs
tiers = [
    (0.95, 1.0, "95-100% (Very High)"),
    (0.90, 0.95, "90-95% (High)"),
    (0.85, 0.90, "85-90% (Medium-High)"),
    (0.80, 0.85, "80-85% (Medium)"),
    (0.75, 0.80, "75-80% (Medium-Low)"),
]

cumulative_results = []

for low, high, label in tiers:
    subset = fw_df_sorted[(fw_df_sorted['model_prob'] >= low) & (fw_df_sorted['model_prob'] < high)]

    if len(subset) > 0:
        n_bets = len(subset)
        n_wins = subset['bet_won'].sum()
        wr = n_wins / n_bets
        avg_r = subset['unit_return'].sum() / n_bets
        total_r = subset['unit_return'].sum()

        cumulative_results.append({
            'tier': label,
            'low': low,
            'high': high,
            'bets': n_bets,
            'win_rate': wr,
            'avg_roi': avg_r,
            'total_roi': total_r
        })

        print(f"{label:<20} {n_bets:<10} {wr:<12.1%} {avg_r:<15.2%} {total_r:<15.2f}")

print()

# Now find the optimal strategy by testing different combinations
print("=" * 80)
print("🎯 FINDING OPTIMAL STRATEGY")
print("=" * 80)
print()

best_strategy = None
best_score = -999

print(f"{'Strategy':<30} {'Bets':<10} {'Win Rate':<12} {'Avg ROI':<12} {'Total ROI':<15}")
print("-" * 90)

# Test different combinations
strategies = [
    ("95%+ only", 0.95, 1.0, None),
    ("90%+ only", 0.90, 1.0, None),
    ("85%+ only", 0.85, 1.0, None),
    ("Top 50 bets", None, None, 50),
    ("Top 100 bets", None, None, 100),
    ("Top 200 bets", None, None, 200),
    ("All meets thresholds", 0.75, 1.0, None),
]

for strategy_name, min_conf, max_conf, top_n in strategies:
    if top_n is not None:
        # Fixed number strategy
        subset = fw_df_sorted.head(top_n)
    else:
        # Confidence range strategy
        subset = fw_df_sorted[(fw_df_sorted['model_prob'] >= min_conf) &
                              (fw_df_sorted['model_prob'] < max_conf)]

    if len(subset) == 0:
        continue

    n_bets = len(subset)
    n_wins = subset['bet_won'].sum()
    wr = n_wins / n_bets
    avg_r = subset['unit_return'].sum() / n_bets
    total_r = subset['unit_return'].sum()

    # Score: balance win rate and efficiency (ROI per bet)
    # Prefer higher win rate but penalize very low volumes
    if n_bets < 50:
        # Too few bets = not reliable
        score = wr * 10 - 5
    else:
        score = wr * 10 + avg_r * 100

    if score > best_score:
        best_score = score
        best_strategy = {
            'name': strategy_name,
            'bets': n_bets,
            'win_rate': wr,
            'avg_roi': avg_r,
            'total_roi': total_r,
            'min_conf': min_conf,
            'max_conf': max_conf,
            'top_n': top_n,
        }

    print(f"{strategy_name:<30} {n_bets:<10} {wr:<12.1%} {avg_r:<12.2%} {total_r:<15.2f}")

print()

# Show recommendation
if best_strategy:
    print("=" * 80)
    print("✅ RECOMMENDED STRATEGY")
    print("=" * 80)
    print()
    print(f"Strategy: {best_strategy['name']}")
    print(f"Expected bets per week: {best_strategy['bets'] / 7:.0f}")
    print(f"Win rate: {best_strategy['win_rate']:.1%}")
    print(f"Average ROI: {best_strategy['avg_roi']:.2%}")
    print(f"Total ROI (7 weeks): {best_strategy['total_roi']:.2f} units")
    print()

    # Compare to current approach
    current_avg_roi = fw_df['unit_return'].sum() / len(fw_df)
    current_wr = fw_df['bet_won'].sum() / len(fw_df)

    avg_roi_diff = (best_strategy['avg_roi'] - current_avg_roi)
    wr_diff = (best_strategy['win_rate'] - current_wr)

    print("📊 IMPROVEMENT vs Current Approach:")
    print(f"  Win rate: {wr_diff:+.1%}")
    print(f"  Average ROI: {avg_roi_diff:+.2%}")
    print(f"  Volume: {best_strategy['bets']}/{len(fw_df)} bets ({((len(fw_df)-best_strategy['bets'])/len(fw_df))*100:.0f}% fewer)")
    print()
