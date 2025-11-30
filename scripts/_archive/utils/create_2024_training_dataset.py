#!/usr/bin/env python3
"""
Match 2024 props from Odds API with actual player stats to create training data.
Then combine with 2025 data and retrain calibrator.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("=" * 80)
print("CREATING 2024 PROP TRAINING DATASET")
print("=" * 80)
print()

# 1. Load 2024 props from Odds API
print("📂 Loading 2024 props from Odds API...")
props_2024 = pd.read_csv('data/historical/player_props_2024_from_odds_api.csv')
print(f"   ✅ Loaded {len(props_2024):,} props")
print()

# 2. Load 2024 player stats
print("📂 Loading 2024 player stats...")
stats_2024 = pd.read_csv('data/historical/player_stats_2024.csv')
print(f"   ✅ Loaded {len(stats_2024):,} player-week records")
print()

# 3. Parse stats JSON field
print("📊 Parsing player stats...")
stats_expanded = []
for _, row in stats_2024.iterrows():
    try:
        stats_dict = json.loads(row['stats']) if isinstance(row['stats'], str) else row['stats']
        if isinstance(stats_dict, dict):
            record = {
                'week': row['week'],
                'player_id': row.get('player'),
                'player_name': stats_dict.get('player_name', ''),
                'team': stats_dict.get('team', ''),
                'pass_yds': stats_dict.get('pass_yds', 0),
                'pass_td': stats_dict.get('pass_td', 0),
                'rush_yds': stats_dict.get('rush_yds', 0),
                'rec': stats_dict.get('rec', 0),
                'rec_yds': stats_dict.get('rec_yds', 0),
            }
            stats_expanded.append(record)
    except:
        continue

stats_df = pd.DataFrame(stats_expanded)
print(f"   ✅ Parsed {len(stats_df):,} player-week stats")
print()

# 4. Map prop markets to stat fields
market_to_stat = {
    'player_pass_yds': ('pass_yds', 'Passing Yards'),
    'player_pass_tds': ('pass_td', 'Passing TDs'),
    'player_rush_yds': ('rush_yds', 'Rushing Yards'),
    'player_receptions': ('rec', 'Receptions'),
    'player_reception_yds': ('rec_yds', 'Receiving Yards'),
}

# 5. Extract week from commence_time
print("📅 Extracting weeks from game dates...")
props_2024['commence_dt'] = pd.to_datetime(props_2024['commence_time'])

# Map dates to weeks (approximate)
def date_to_week_2024(date):
    """Map 2024 game date to week number."""
    date = pd.to_datetime(date)
    # 2024 Week 1 started around Sep 5
    week_1_start = pd.to_datetime('2024-09-05')
    days_diff = (date - week_1_start).days
    week = (days_diff // 7) + 1
    return max(1, min(18, week))  # Clamp to 1-18

props_2024['week'] = props_2024['commence_dt'].apply(date_to_week_2024)
print(f"   ✅ Mapped props to weeks {props_2024['week'].min()}-{props_2024['week'].max()}")
print()

# 6. Match props with stats to calculate outcomes
print("🔗 Matching props with actual stats...")
matched_props = []

for _, prop in props_2024.iterrows():
    player_name = prop['player']
    week = prop['week']
    market = prop['market']
    line = prop['line']

    if pd.isna(line) or market not in market_to_stat:
        continue

    stat_field, _ = market_to_stat[market]

    # Find matching player stat
    # Simple name matching (case-insensitive)
    player_stats = stats_df[
        (stats_df['week'] == week) &
        (stats_df['player_name'].str.lower() == player_name.lower())
    ]

    if len(player_stats) > 0:
        actual_value = player_stats.iloc[0][stat_field]

        # Calculate outcome
        bet_outcome_over = 1.0 if actual_value > line else 0.0
        bet_outcome_under = 1.0 if actual_value < line else 0.0

        matched_props.append({
            'season': 2024,
            'week': week,
            'player': player_name,
            'team': player_stats.iloc[0]['team'],
            'market': market,
            'line': line,
            'price': prop['price'],
            'bookmaker': prop['bookmaker'],
            'actual_value': actual_value,
            'bet_outcome_over': bet_outcome_over,
            'bet_outcome_under': bet_outcome_under,
            'commence_time': prop['commence_time'],
        })

matched_df = pd.DataFrame(matched_props)
print(f"   ✅ Matched {len(matched_df):,} props with actual outcomes")
print(f"   Match rate: {len(matched_df)/len(props_2024):.1%}")
print()

# 7. Load 2025 data
print("📂 Loading 2025 training data...")
df_2025 = pd.read_csv('data/historical/player_prop_training_dataset.csv')
df_2025['season'] = 2025
print(f"   ✅ Loaded {len(df_2025):,} props from 2025")
print()

# 8. Combine datasets
print("🔗 Combining 2024 + 2025 data...")
combined = pd.concat([matched_df, df_2025], ignore_index=True)
print(f"   ✅ Combined dataset: {len(combined):,} total props")
print()

# 9. Save combined dataset
output_path = Path('data/historical/player_prop_training_dataset_combined.csv')
combined.to_csv(output_path, index=False)
print(f"✅ Saved to: {output_path}")
print()

# 10. Summary
print("📊 FINAL DATASET SUMMARY:")
print()
print(f"   Total props: {len(combined):,}")
print(f"   2024 season: {len(combined[combined['season']==2024]):,} props")
print(f"   2025 season: {len(combined[combined['season']==2025]):,} props")
print()

# Filter to props with outcomes
with_outcomes = combined[combined['bet_outcome_over'].notna()]
print(f"   With outcomes: {len(with_outcomes):,}")
print(f"   2024: {len(with_outcomes[with_outcomes['season']==2024]):,}")
print(f"   2025: {len(with_outcomes[with_outcomes['season']==2025]):,}")
print()

print("   Markets:")
for market in with_outcomes['market'].value_counts().head(10).items():
    print(f"      {market[0]:<25} {market[1]:>6,} props")
print()

print("🎯 Next step:")
print("   Run: python retrain_calibrator_full_history.py")
print("   This will train calibrator on full 2024+2025 dataset")
