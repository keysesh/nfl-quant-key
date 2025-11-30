#!/usr/bin/env python3
"""
Build historical player stats from Sleeper data (2025 only)

NOTE: This script is DEPRECATED in favor of create_unified_historical_player_stats.py
which combines 2025 Sleeper + historical NFLverse/Sleeper data.

This script is kept for backward compatibility but should be replaced with
the unified version for better data coverage.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

def build_player_stats_history():
    """Build player stats history from weeks 1-7"""

    print("="*80)
    print("BUILDING HISTORICAL PLAYER STATS")
    print("="*80)

    stats_dir = Path("data/sleeper_stats")
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    all_stats = []

    for stats_file in sorted(stats_dir.glob("stats_week*_2025.csv")):
        match = re.search(r"stats_week(\d+)_", stats_file.name)
        if not match:
            continue
        week = int(match.group(1))
        try:
            df = pd.read_csv(stats_file)
        except EmptyDataError:
            continue
        df["week"] = week
        all_stats.append(df)

    if not all_stats:
        raise ValueError(f"No Sleeper stats files found in {stats_dir}")

    combined = pd.concat(all_stats, ignore_index=True)
    print(f"\nLoaded {len(combined)} player-week records")

    # Build historical database
    player_db = {}

    # Pre-calculate team totals per week for accurate share calculations
    print("\nCalculating team totals per week...")
    team_week_totals = {}
    for week in combined['week'].unique():
        week_data = combined[combined['week'] == week]
        for team in week_data['team'].dropna().unique():
            team_week_data = week_data[week_data['team'] == team]
            
            # Calculate team total targets (sum of all receiving targets for WR, TE, RB)
            team_total_targets = team_week_data[
                team_week_data['position'].isin(['WR', 'TE', 'RB'])
            ]['rec_tgt'].sum()
            
            # Calculate team total carries (sum of all rush attempts for RB, QB)
            team_total_carries = team_week_data[
                team_week_data['position'].isin(['RB', 'QB'])
            ]['rush_att'].sum()
            
            key = (team, week)
            team_week_totals[key] = {
                'total_targets': team_total_targets,
                'total_carries': team_total_carries
            }
    
    print(f"   Calculated totals for {len(team_week_totals)} team-week combinations")

    for player_name in combined['player_name'].unique():
        player_weeks = combined[combined['player_name'] == player_name].sort_values('week')

        if len(player_weeks) == 0:
            continue

        position = player_weeks.iloc[0]['position']
        team = player_weeks.iloc[0]['team']  # Get team from first week

        # Calculate trailing stats
        historical_stats = {
            'total_completions': 0,
            'total_attempts': 0,
            'total_pass_yd': 0,
            'total_rush_yd': 0,
            'total_rush_att': 0,
            'total_rec': 0,
            'total_rec_yd': 0,
            'total_rec_tgt': 0,
            'weeks_played': 0
        }

        # Calculate team totals for this player's weeks
        total_team_targets = 0
        total_team_carries = 0

        for idx, row in player_weeks.iterrows():
            historical_stats['total_completions'] += row['pass_cmp']
            historical_stats['total_attempts'] += row['pass_att']
            historical_stats['total_pass_yd'] += row['pass_yd']
            historical_stats['total_rush_yd'] += row['rush_yd']
            historical_stats['total_rush_att'] += row['rush_att']
            historical_stats['total_rec'] += row['rec']
            historical_stats['total_rec_yd'] += row['rec_yd']
            historical_stats['total_rec_tgt'] += row['rec_tgt']
            historical_stats['weeks_played'] += 1
            
            # Accumulate team totals for this week
            week = row['week']
            team_key = (team, week)
            if team_key in team_week_totals:
                total_team_targets += team_week_totals[team_key]['total_targets']
                total_team_carries += team_week_totals[team_key]['total_carries']

        # Calculate averages
        weeks = historical_stats['weeks_played']

        if weeks > 0:
            # Derive historical features
            trailing_snap_share = 0.6  # Default (would need snap count data to calculate properly)

            # Calculate target share from ACTUAL team totals (not hardcoded)
            # Target share = player targets / team total targets across all weeks
            if historical_stats['total_rec_tgt'] > 0 and total_team_targets > 0:
                trailing_target_share = historical_stats['total_rec_tgt'] / total_team_targets
            elif position in ['RB', 'WR', 'TE']:
                # If player has 0 targets but is a receiving position, set to 0 (not None)
                # This allows the system to know the player exists but has no receiving role
                trailing_target_share = 0.0
            else:
                trailing_target_share = None

            # Calculate carry share from ACTUAL team totals (not hardcoded)
            # Carry share = player carries / team total carries across all weeks
            if historical_stats['total_rush_att'] > 0 and total_team_carries > 0:
                trailing_carry_share = historical_stats['total_rush_att'] / total_team_carries
            elif position == 'RB':
                # If RB has 0 carries, set to 0 (not None)
                trailing_carry_share = 0.0
            else:
                trailing_carry_share = None

            if historical_stats['total_attempts'] > 0:
                trailing_yards_per_opportunity = historical_stats['total_pass_yd'] / historical_stats['total_attempts']
            elif historical_stats['total_rush_att'] > 0:
                trailing_yards_per_opportunity = historical_stats['total_rush_yd'] / historical_stats['total_rush_att']
            elif historical_stats['total_rec_tgt'] > 0:
                trailing_yards_per_opportunity = historical_stats['total_rec_yd'] / historical_stats['total_rec_tgt']
            else:
                trailing_yards_per_opportunity = 0.0  # Changed from 5.0 - use 0 if no data

            # TD rate (simplified)
            total_tds = sum(player_weeks['pass_td']) + sum(player_weeks['rush_td']) + sum(player_weeks['rec_td'])
            opportunities = historical_stats['total_attempts'] + historical_stats['total_rush_att'] + historical_stats['total_rec_tgt']
            trailing_td_rate = total_tds / max(opportunities, 1)

            player_db[player_name] = {
                'position': position,
                'team': player_weeks.iloc[-1]['team'],  # Latest team
                'weeks_played': weeks,
                'trailing_snap_share': trailing_snap_share,
                'trailing_target_share': trailing_target_share,
                'trailing_carry_share': trailing_carry_share,
                'trailing_yards_per_opportunity': trailing_yards_per_opportunity,
                'trailing_td_rate': min(trailing_td_rate, 0.15),  # Cap at 15%
                'avg_pass_yd': historical_stats['total_pass_yd'] / weeks if weeks > 0 else 0,
                'avg_rush_yd': historical_stats['total_rush_yd'] / weeks if weeks > 0 else 0,
                'avg_rec_yd': historical_stats['total_rec_yd'] / weeks if weeks > 0 else 0,
            }

    # Save to file
    output_file = Path('data/historical_player_stats.json')
    with open(output_file, 'w') as f:
        json.dump(player_db, f, indent=2)

    print(f"\n✅ Created historical player database: {len(player_db)} players")
    print(f"   Saved to {output_file}")

    # Show sample
    print("\n📊 Sample players:")
    for i, (name, stats) in enumerate(list(player_db.items())[:5]):
        print(f"  {name} ({stats['position']}):")
        print(f"    Weeks: {stats['weeks_played']}, Yards/opp: {stats['trailing_yards_per_opportunity']:.2f}")
        if stats.get('trailing_target_share') is not None:
            print(f"    Target share: {stats['trailing_target_share']:.1%}")
        if stats.get('trailing_carry_share') is not None:
            print(f"    Carry share: {stats['trailing_carry_share']:.1%}")

    print("\n" + "="*80)
    print("✅ Historical stats complete!")
    print("="*80)

    return player_db

if __name__ == '__main__':
    build_player_stats_history()
