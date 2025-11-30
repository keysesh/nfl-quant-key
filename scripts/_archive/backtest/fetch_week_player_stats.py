#!/usr/bin/env python3
"""
Fetch actual player stats for a given NFL week.
Uses nflverse play-by-play data to extract player performances.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.season_config import infer_season_from_week

def fetch_week_player_stats(week: int, season: int = None):
    """
    Fetch player stats for a specific week from nflverse.

    Args:
        week: NFL week number
        season: NFL season year (defaults to inferred from week)

    Returns:
        DataFrame with player stats
    """
    if season is None:
        season = infer_season_from_week(week)
        print(f"ℹ️  No season specified, inferred season {season} for week {week}")

    print(f"Fetching Week {week} player stats for {season} season...")

    # Load play-by-play data
    url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'

    try:
        print(f"Loading play-by-play data from nflverse...")
        pbp = pd.read_parquet(url)

        # Filter to specific week
        week_pbp = pbp[pbp['week'] == week].copy()
        print(f"Loaded {len(week_pbp):,} plays from Week {week}")

        # Extract passing stats
        passing_stats = week_pbp[week_pbp['pass_attempt'] == 1].groupby(['passer_player_id', 'passer_player_name', 'posteam']).agg({
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum',
            'complete_pass': 'sum',
            'pass_attempt': 'count'
        }).reset_index()
        passing_stats.columns = ['player_id', 'player_name', 'team', 'passing_yards', 'passing_tds', 'interceptions', 'completions', 'attempts']
        passing_stats['position'] = 'QB'

        # Extract rushing stats
        rushing_stats = week_pbp[week_pbp['rush_attempt'] == 1].groupby(['rusher_player_id', 'rusher_player_name', 'posteam']).agg({
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
            'rush_attempt': 'count'
        }).reset_index()
        rushing_stats.columns = ['player_id', 'player_name', 'team', 'rushing_yards', 'rushing_tds', 'rushing_attempts']

        # Extract receiving stats
        receiving_stats = week_pbp[week_pbp['pass_attempt'] == 1].groupby(['receiver_player_id', 'receiver_player_name', 'posteam']).agg({
            'receiving_yards': 'sum',
            'pass_touchdown': 'sum',  # receiving TD
            'complete_pass': 'sum'  # receptions
        }).reset_index()
        receiving_stats.columns = ['player_id', 'player_name', 'team', 'receiving_yards', 'receiving_tds', 'receptions']

        # Combine all stats
        all_players = []

        # Process passing
        for _, row in passing_stats.iterrows():
            all_players.append({
                'player_name': row['player_name'],
                'player_id': row['player_id'],
                'team': row['team'],
                'position': 'QB',
                'passing_yards': row['passing_yards'],
                'passing_tds': row['passing_tds'],
                'interceptions': row['interceptions'],
                'completions': row['completions'],
                'attempts': row['attempts'],
                'rushing_yards': 0,
                'rushing_tds': 0,
                'rushing_attempts': 0,
                'receiving_yards': 0,
                'receiving_tds': 0,
                'receptions': 0
            })

        # Process rushing (merge with passing for QBs)
        for _, row in rushing_stats.iterrows():
            existing = next((p for p in all_players if p['player_id'] == row['player_id']), None)
            if existing:
                existing['rushing_yards'] = row['rushing_yards']
                existing['rushing_tds'] = row['rushing_tds']
                existing['rushing_attempts'] = row['rushing_attempts']
            else:
                all_players.append({
                    'player_name': row['player_name'],
                    'player_id': row['player_id'],
                    'team': row['team'],
                    'position': 'RB',  # Assume RB for now
                    'passing_yards': 0,
                    'passing_tds': 0,
                    'interceptions': 0,
                    'completions': 0,
                    'attempts': 0,
                    'rushing_yards': row['rushing_yards'],
                    'rushing_tds': row['rushing_tds'],
                    'rushing_attempts': row['rushing_attempts'],
                    'receiving_yards': 0,
                    'receiving_tds': 0,
                    'receptions': 0
                })

        # Process receiving
        for _, row in receiving_stats.iterrows():
            existing = next((p for p in all_players if p['player_id'] == row['player_id']), None)
            if existing:
                existing['receiving_yards'] = row['receiving_yards']
                existing['receiving_tds'] = row['receiving_tds']
                existing['receptions'] = row['receptions']
            else:
                all_players.append({
                    'player_name': row['player_name'],
                    'player_id': row['player_id'],
                    'team': row['team'],
                    'position': 'WR/TE',  # Assume receiver
                    'passing_yards': 0,
                    'passing_tds': 0,
                    'interceptions': 0,
                    'completions': 0,
                    'attempts': 0,
                    'rushing_yards': 0,
                    'rushing_tds': 0,
                    'rushing_attempts': 0,
                    'receiving_yards': row['receiving_yards'],
                    'receiving_tds': row['receiving_tds'],
                    'receptions': row['receptions']
                })

        df = pd.DataFrame(all_players)

        # Save to file
        output_dir = Path(f'data/results/{season}')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'week{week}_player_stats.csv'
        df.to_csv(output_file, index=False)

        print(f"\n✅ Player stats saved to: {output_file}")
        print(f"Total players with stats: {len(df)}")
        print(f"\nBreakdown:")
        print(f"  Players with passing stats: {(df['passing_yards'] > 0).sum()}")
        print(f"  Players with rushing stats: {(df['rushing_yards'] > 0).sum()}")
        print(f"  Players with receiving stats: {(df['receiving_yards'] > 0).sum()}")

        return df

    except Exception as e:
        print(f"❌ Error fetching player stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    week = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    season = int(sys.argv[2]) if len(sys.argv) > 2 else None  # Auto-infer

    df = fetch_week_player_stats(week, season)

    if df is not None and not df.empty:
        print("\n" + "="*80)
        print("Sample player stats:")
        print("="*80)
        print(df.head(10).to_string(index=False))
