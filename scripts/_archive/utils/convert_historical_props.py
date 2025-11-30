#!/usr/bin/env python3
"""
Convert historical props from The Odds API format to our standard format.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
HIST_DIR = DATA_DIR / 'historical'


def convert_historical_props():
    """Convert all historical props to standard format."""

    # Find all historical prop files
    hist_files = list(HIST_DIR.glob('player_props_history_*.csv'))

    if not hist_files:
        print("No historical prop files found")
        return

    all_converted = []

    for f in hist_files:
        print(f"Converting {f.name}...")
        df = pd.read_csv(f)

        # Market type mapping
        market_map = {
            'player_pass_yds': 'passing_yards',
            'player_rush_yds': 'rushing_yards',
            'player_reception_yds': 'receiving_yards',
            'player_receptions': 'receptions',
        }

        # Group by player, market, line to get over/under pairs
        # Each row has over or under, we need to combine them

        for (player, market, line), group in df.groupby(['player', 'market', 'line']):
            if market not in market_map:
                continue

            over_row = group[group['prop_type'] == 'over']
            under_row = group[group['prop_type'] == 'under']

            if len(over_row) == 0 or len(under_row) == 0:
                continue

            # Get first occurrence
            over_row = over_row.iloc[0]
            under_row = under_row.iloc[0]

            # Extract week from game_id (format: YYYYMMDD_AWAY_HOME)
            game_id = over_row['game_id']

            all_converted.append({
                'player': player,
                'stat_type': market_map[market],
                'line': line,
                'over_odds': over_row['price'],
                'under_odds': under_row['price'],
                'game_id': game_id,
                'away_team': over_row['away_team'],
                'home_team': over_row['home_team'],
                'commence_time': over_row['commence_time'],
                'source_file': f.stem,
            })

    if not all_converted:
        print("No props converted")
        return

    converted_df = pd.DataFrame(all_converted)

    # Remove duplicates
    converted_df = converted_df.drop_duplicates(subset=['player', 'stat_type', 'line'])

    # Save
    output_file = DATA_DIR / 'historical_player_props_converted.csv'
    converted_df.to_csv(output_file, index=False)

    print(f"\nConverted {len(converted_df)} unique props")
    print(f"Saved to {output_file}")
    print(f"\nBy stat type:")
    print(converted_df['stat_type'].value_counts())


if __name__ == '__main__':
    convert_historical_props()
