#!/usr/bin/env python3
"""
Diagnose why snap_share is 0 for all players in model predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from nfl_quant.utils.unified_integration import calculate_snap_share_from_data

# Load current predictions
df = pd.read_csv('data/model_predictions_week11.csv')

print("=== SNAP SHARE DIAGNOSTIC ===\n")

# Test on a few known players
test_players = [
    ('Patrick Mahomes', 'QB', 'KC', 11),
    ('Josh Allen', 'QB', 'BUF', 11),
    ('Christian McCaffrey', 'RB', 'SF', 11),
    ('Tyreek Hill', 'WR', 'MIA', 11),
]

for player_name, position, team, week in test_players:
    print(f"\nTesting: {player_name} ({position}, {team})")

    snap_share = calculate_snap_share_from_data(
        player_name=player_name,
        position=position,
        team=team,
        week=week,
        season=2025,
        lookback_weeks=4
    )

    print(f"  Result: {snap_share}")
    if snap_share is None:
        print("  ❌ Returned None")
    elif snap_share == 0:
        print("  ⚠️  Returned 0")
    else:
        print(f"  ✅ Returned {snap_share:.1%}")

# Check PBP data availability
print("\n=== PBP DATA CHECK ===")
pbp_path = Path('data/processed/pbp_2025.parquet')
if pbp_path.exists():
    pbp_df = pd.read_parquet(pbp_path)
    print(f"✅ PBP file exists: {len(pbp_df)} rows")
    print(f"Columns: {list(pbp_df.columns[:10])}...")

    # Check if player data exists
    if 'passer_player_name' in pbp_df.columns:
        print(f"\nSample passers: {pbp_df['passer_player_name'].dropna().unique()[:5]}")
    if 'rusher_player_name' in pbp_df.columns:
        print(f"Sample rushers: {pbp_df['rusher_player_name'].dropna().unique()[:5]}")
    if 'receiver_player_name' in pbp_df.columns:
        print(f"Sample receivers: {pbp_df['receiver_player_name'].dropna().unique()[:5]}")
else:
    print(f"❌ PBP file not found: {pbp_path}")
