#!/usr/bin/env python3
"""Analyze win rate by different thresholds"""

import pandas as pd

df = pd.read_csv('reports/framework_backtest_weeks_1_7_fixed.csv')
fw_df = df[df['model_prob'].notna()].copy()

print('=' * 80)
print('WIN RATE ANALYSIS BY THRESHOLD')
print('=' * 80)
print()

# Test different edge thresholds
print('1. Filtering by MINIMUM EDGE:')
print('{:<10} {:>10} {:>10} {:>10} {:>10}'.format(
    'Min Edge', 'Bets', 'Win Rate', 'Total ROI', 'Avg ROI'))
print('-' * 60)

for thresh in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
    filtered = fw_df[fw_df['edge'] >= thresh].copy()
    if len(filtered) > 0:
        wr = filtered['bet_won'].sum() / len(filtered)
        total_roi = filtered['unit_return'].sum()
        avg_roi = total_roi / len(filtered)
        print('{:<10} {:>10} {:>10.1%} {:>10.2f} {:>10.2%}'.format(
            f'{thresh:.0%}', len(filtered), wr, total_roi, avg_roi))

print()
print('2. Filtering by MINIMUM CONFIDENCE:')
print('{:<10} {:>10} {:>10} {:>10} {:>10}'.format(
    'Min Conf', 'Bets', 'Win Rate', 'Total ROI', 'Avg ROI'))
print('-' * 60)

for conf in [0.55, 0.60, 0.65, 0.70, 0.75]:
    filtered = fw_df[fw_df['model_prob'] >= conf].copy()
    if len(filtered) > 0:
        wr = filtered['bet_won'].sum() / len(filtered)
        total_roi = filtered['unit_return'].sum()
        avg_roi = total_roi / len(filtered)
        print('{:<10} {:>10} {:>10.1%} {:>10.2f} {:>10.2%}'.format(
            f'{conf:.0%}', len(filtered), wr, total_roi, avg_roi))

print()
print('3. COMBINED FILTERING (Edge + Confidence):')
print('{:<20} {:>10} {:>10} {:>10} {:>10}'.format(
    'Thresholds', 'Bets', 'Win Rate', 'Total ROI', 'Avg ROI'))
print('-' * 75)

combos = [
    (0.03, 0.60),
    (0.05, 0.60),
    (0.05, 0.65),
    (0.07, 0.65),
    (0.07, 0.70),
    (0.10, 0.70),
    (0.10, 0.75),
]

for edge_thresh, conf_thresh in combos:
    filtered = fw_df[(fw_df['edge'] >= edge_thresh) &
                     (fw_df['model_prob'] >= conf_thresh)].copy()
    if len(filtered) > 0:
        wr = filtered['bet_won'].sum() / len(filtered)
        total_roi = filtered['unit_return'].sum()
        avg_roi = total_roi / len(filtered)
        print('{:<20} {:>10} {:>10.1%} {:>10.2f} {:>10.2%}'.format(
            f'Edge≥{edge_thresh:.0%}, Conf≥{conf_thresh:.0%}',
            len(filtered), wr, total_roi, avg_roi))

print()
print('4. BY MARKET:')
for market in sorted(fw_df['market'].unique()):
    market_data = fw_df[fw_df['market'] == market]
    wr = market_data['bet_won'].sum() / len(market_data)
    roi = market_data['unit_return'].sum()
    print(f'  {market:25s}: {len(market_data):4} bets | WR: {wr:5.1%} | ROI: {roi:+7.2f}')

print()
print('=' * 80)
print('RECOMMENDATIONS:')
print('=' * 80)

# Find best combination
best_combos = []
for edge_thresh, conf_thresh in combos:
    filtered = fw_df[(fw_df['edge'] >= edge_thresh) &
                     (fw_df['model_prob'] >= conf_thresh)].copy()
    if len(filtered) > 0:
        wr = filtered['bet_won'].sum() / len(filtered)
        roi = filtered['unit_return'].sum() / len(filtered)
        best_combos.append((edge_thresh, conf_thresh, wr, roi, len(filtered)))

# Sort by win rate
best_combos.sort(key=lambda x: x[2], reverse=True)

print('\nTop 3 Combinations by Win Rate:')
for i, (edge, conf, wr, roi, num) in enumerate(best_combos[:3], 1):
    print(f'{i}. Edge ≥ {edge:.0%}, Confidence ≥ {conf:.0%}')
    print(f'   {num} bets | WR: {wr:.1%} | ROI: {roi:+.2%}')

