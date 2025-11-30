#!/usr/bin/env python3
"""
Rebuild player database from ALL available sources:
- DraftKings props (103 players tonight)
- Sleeper weekly stats (all active players)
- Historical player stats
"""

import pandas as pd
import json
from pathlib import Path
from glob import glob


def load_players_from_props():
    """Get all players from current DraftKings props"""
    props_file = 'data/nfl_player_props_draftkings.csv'

    if not Path(props_file).exists():
        return {}

    props = pd.read_csv(props_file)
    players = {}

    for player_name in props['player_name'].unique():
        if pd.isna(player_name):
            continue

        # Skip defense/special teams
        if 'D/ST' in player_name or 'Defense' in player_name:
            continue

        player_props = props[props['player_name'] == player_name]

        # Infer position from markets
        markets = set(player_props['market'].unique())

        if 'player_pass_yds' in markets or 'player_pass_tds' in markets:
            position = 'QB'
        elif 'player_rush_yds' in markets and 'player_reception_yds' in markets:
            position = 'RB'
        elif 'player_reception_yds' in markets or 'player_receptions' in markets:
            # Could be WR or TE
            # Check if they have rush attempts (more likely RB)
            if 'player_rush_yds' in markets:
                position = 'RB'
            else:
                position = 'WR'  # Default to WR, will refine from Sleeper
        elif 'player_kicking_points' in markets:
            position = 'K'
        else:
            position = 'UNKNOWN'

        # Get team from home_team or away_team
        team = None
        if not player_props.empty:
            # Try to find player's team (they'll be consistently home or away in their games)
            home_teams = player_props['home_team'].unique()
            away_teams = player_props['away_team'].unique()

            # Simple heuristic: if player appears for one team consistently
            if len(home_teams) == 1 and len(away_teams) > 1:
                team = home_teams[0]
            elif len(away_teams) == 1 and len(home_teams) > 1:
                team = away_teams[0]

        players[player_name] = {
            'position': position,
            'team': team,
            'markets': list(markets),
            'source': 'draftkings_props'
        }

    return players


def load_players_from_sleeper():
    """Get players from Sleeper stats files"""
    players = {}

    # Get most recent week
    stats_files = sorted(glob('data/sleeper_stats/stats_week*_2025.csv'), reverse=True)

    if not stats_files:
        return players

    # Load most recent week
    latest_stats = pd.read_csv(stats_files[0])

    for _, row in latest_stats.iterrows():
        player_name = row['player_name']

        if pd.isna(player_name):
            continue

        # Skip if already in database with good info
        if player_name in players and players[player_name]['position'] != 'UNKNOWN':
            # Update team if missing
            if not players[player_name].get('team'):
                players[player_name]['team'] = row['team']
            continue

        players[player_name] = {
            'position': row['position'],
            'team': row['team'],
            'player_id': row.get('player_id'),
            'source': 'sleeper_stats'
        }

    return players


def merge_player_databases(props_players, sleeper_players):
    """Merge player databases, prioritizing better information"""
    merged = {}

    # Start with props players (they're actively bet on)
    for name, data in props_players.items():
        merged[name] = data.copy()

    # Enhance with Sleeper data
    for name, data in sleeper_players.items():
        if name in merged:
            # Update missing fields
            if merged[name].get('position') == 'UNKNOWN' or not merged[name].get('position'):
                merged[name]['position'] = data['position']

            if not merged[name].get('team'):
                merged[name]['team'] = data['team']

            if not merged[name].get('player_id'):
                merged[name]['player_id'] = data.get('player_id')

            merged[name]['sleeper_verified'] = True
        else:
            # Add new player from Sleeper
            merged[name] = data.copy()

    return merged


def add_default_trailing_stats(players):
    """
    Add trailing stats for players from calculated historical data.
    
    IMPORTANT: No hardcoded defaults. All stats come from calculated historical data.
    If no historical data exists, stats remain None or 0.0.
    """

    # Load historical stats (from create_historical_player_stats.py)
    historical_stats_file = 'data/historical_player_stats.json'
    historical_stats = {}
    
    if Path(historical_stats_file).exists():
        with open(historical_stats_file) as f:
            historical_stats = json.load(f)
        print(f"   ✅ Loaded {len(historical_stats)} players from historical_player_stats.json")

    # Load week-specific stats if available (alternative source)
    week_stats_file = 'data/week_specific_trailing_stats.json'
    week_stats = {}
    
    if Path(week_stats_file).exists():
        with open(week_stats_file) as f:
            week_stats = json.load(f)
        print(f"   ✅ Loaded {len(week_stats)} players from week_specific_trailing_stats.json")

    # Add calculated stats (no hardcoded defaults)
    for name, data in players.items():
        # Priority: week_stats > historical_stats > no stats (leave as None/0)
        if name in week_stats:
            # Use actual trailing stats from week_stats
            data.update(week_stats[name])
        elif name in historical_stats:
            # Use historical stats (calculated from actual team totals)
            hist = historical_stats[name]
            data.update({
                'trailing_snap_share': hist.get('trailing_snap_share'),
                'trailing_target_share': hist.get('trailing_target_share'),  # May be None or 0.0
                'trailing_carry_share': hist.get('trailing_carry_share'),  # May be None or 0.0
                'trailing_yards_per_opportunity': hist.get('trailing_yards_per_opportunity'),
                'trailing_td_rate': hist.get('trailing_td_rate'),
            })
        # If no historical data exists, leave stats as None/0
        # This is intentional - we don't want to use hardcoded defaults

    return players


def save_player_database(players, output_file='data/player_database.json'):
    """Save player database to JSON"""

    # Sort by player name
    sorted_players = dict(sorted(players.items()))

    with open(output_file, 'w') as f:
        json.dump(sorted_players, f, indent=2)

    print(f"✅ Saved {len(sorted_players)} players to {output_file}")


def main():
    print("="*80)
    print("🔄 REBUILDING PLAYER DATABASE")
    print("="*80)
    print()

    # Load from props
    print("1. Loading players from DraftKings props...")
    props_players = load_players_from_props()
    print(f"   ✅ Found {len(props_players)} players in props")

    # Load from Sleeper
    print("\n2. Loading players from Sleeper stats...")
    sleeper_players = load_players_from_sleeper()
    print(f"   ✅ Found {len(sleeper_players)} players in Sleeper")

    # Merge
    print("\n3. Merging databases...")
    merged_players = merge_player_databases(props_players, sleeper_players)
    print(f"   ✅ Merged to {len(merged_players)} total players")

    # Add trailing stats from unified historical data
    print("\n4. Adding trailing stats from unified historical data...")
    print("   (Combines 2025 Sleeper + historical NFLverse/Sleeper)")
    complete_players = add_default_trailing_stats(merged_players)

    # Save
    print("\n5. Saving to file...")
    save_player_database(complete_players)

    # Summary
    print("\n"+"="*80)
    print("📊 PLAYER DATABASE SUMMARY")
    print("="*80)

    position_counts = {}
    for data in complete_players.values():
        pos = data.get('position', 'UNKNOWN')
        position_counts[pos] = position_counts.get(pos, 0) + 1

    print("\nBy Position:")
    for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count} players")

    # Show players with props
    with_props = sum(1 for d in complete_players.values() if d.get('source') == 'draftkings_props')
    print(f"\nPlayers with active props: {with_props}")

    print("\n✅ COMPLETE!")


if __name__ == "__main__":
    main()
