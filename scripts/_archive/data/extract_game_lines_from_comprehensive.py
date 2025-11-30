#!/usr/bin/env python3
"""
Extract Game Lines from Comprehensive Odds Files

The comprehensive odds files (odds_week*_comprehensive.csv) contain both
game lines AND player props. This script extracts just the game lines
(spread, total, moneyline) and saves them in the correct format.

Usage:
    python scripts/data/extract_game_lines_from_comprehensive.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent


def extract_game_lines_from_comprehensive():
    """Extract game lines from all comprehensive odds files."""

    # Load NFLverse schedule
    schedules_file = PROJECT_ROOT / "data/nflverse/schedules.parquet"
    schedules = pd.read_parquet(schedules_file)
    schedules_2025 = schedules[schedules['season'] == 2025].copy()

    # Find comprehensive odds files
    comprehensive_files = sorted((PROJECT_ROOT / "data").glob("odds_week*_comprehensive.csv"))

    if not comprehensive_files:
        print("No comprehensive odds files found")
        return

    print(f"Found {len(comprehensive_files)} comprehensive odds files")

    all_game_lines = []

    for comp_file in comprehensive_files:
        print(f"\nProcessing {comp_file.name}...")
        df = pd.read_csv(comp_file)

        # Filter to game lines only
        game_line_markets = ['spreads', 'totals', 'h2h']
        game_lines = df[df['market'].isin(game_line_markets)].copy()

        if len(game_lines) == 0:
            print(f"  No game lines found")
            continue

        # Convert to our standard format
        standardized = []

        for _, row in game_lines.iterrows():
            # Determine side
            if row['market'] == 'h2h':
                side = row['direction']  # 'home' or 'away'
                point = None
                market = 'moneyline'
            elif row['market'] == 'spreads':
                side = row['direction']
                point = row['line']
                market = 'spread'
            elif row['market'] == 'totals':
                side = row['direction']  # 'over' or 'under'
                point = row['line']
                market = 'total'
            else:
                continue

            standardized.append({
                'game_id': row['game_id'],
                'away_team': row['away_team'],
                'home_team': row['home_team'],
                'commence_time': row['commence_time'],
                'sportsbook': row['sportsbook'],
                'market': market,
                'side': side,
                'point': point,
                'price': row['market_odds'],
                'season': 2025,
                'week': None,  # Will be filled from schedule
                'archived_at': datetime.now().isoformat()
            })

        if standardized:
            game_lines_df = pd.DataFrame(standardized)

            # Match to correct week from schedule
            for game_id in game_lines_df['game_id'].unique():
                game_data = game_lines_df[game_lines_df['game_id'] == game_id].iloc[0]
                home_team = game_data['home_team']
                away_team = game_data['away_team']

                match = schedules_2025[
                    (schedules_2025['home_team'] == home_team) &
                    (schedules_2025['away_team'] == away_team)
                ]

                if len(match) > 0:
                    correct_week = int(match.iloc[0]['week'])
                    game_lines_df.loc[game_lines_df['game_id'] == game_id, 'week'] = correct_week
                    print(f"  {away_team}@{home_team} → Week {correct_week}")

            # Only keep games with matched weeks
            game_lines_df = game_lines_df[game_lines_df['week'].notna()].copy()
            all_game_lines.append(game_lines_df)

    if not all_game_lines:
        print("\nNo game lines extracted")
        return

    # Combine all game lines
    combined = pd.concat(all_game_lines, ignore_index=True)
    combined['week'] = combined['week'].astype(int)

    print(f"\n{'='*70}")
    print(f"Total game line odds extracted: {len(combined)}")
    print(f"Weeks covered: {sorted(combined['week'].unique())}")

    # Save by week
    output_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"
    output_dir.mkdir(parents=True, exist_ok=True)

    for week in sorted(combined['week'].unique()):
        week_odds = combined[combined['week'] == week]
        output_file = output_dir / f"game_lines_2025_week{week:02d}.csv"

        # Load existing if present and combine
        if output_file.exists():
            existing = pd.read_csv(output_file)
            week_odds = pd.concat([existing, week_odds], ignore_index=True)
            # Remove duplicates (same game_id, market, side)
            week_odds = week_odds.drop_duplicates(
                subset=['game_id', 'market', 'side'],
                keep='first'
            )

        week_odds.to_csv(output_file, index=False)
        games_count = week_odds['game_id'].nunique()
        print(f"Week {week:2d}: ✅ Saved {len(week_odds)} odds ({games_count} games) to {output_file.name}")

    print(f"\n✅ Extraction complete!")


if __name__ == "__main__":
    extract_game_lines_from_comprehensive()
