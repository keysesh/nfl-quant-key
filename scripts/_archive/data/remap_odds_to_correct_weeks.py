#!/usr/bin/env python3
"""
Remap Odds to Correct Weeks

The raw DraftKings odds files (odds_week8_draftkings.csv, etc.) contain games
from multiple weeks due to being captured at different times. This script:
1. Loads ALL raw odds files
2. Matches each game to the correct week using the NFLverse schedule
3. Re-archives odds into the correct week files

Usage:
    python scripts/data/remap_odds_to_correct_weeks.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent


def remap_odds_to_correct_weeks():
    """Remap all odds files to correct weeks based on schedule."""

    # Load NFLverse schedule
    schedules_file = PROJECT_ROOT / "data/nflverse/schedules.parquet"
    schedules = pd.read_parquet(schedules_file)
    schedules_2025 = schedules[schedules['season'] == 2025].copy()

    # Find all raw DraftKings odds files
    odds_files = sorted((PROJECT_ROOT / "data").glob("odds_week*_draftkings.csv"))

    if not odds_files:
        print("No DraftKings odds files found")
        return

    print(f"Found {len(odds_files)} odds files to process")

    # Collect all odds
    all_odds = []

    for odds_file in odds_files:
        print(f"\nProcessing {odds_file.name}...")
        df = pd.read_csv(odds_file)

        # Only keep game lines (spread, total, moneyline)
        game_line_markets = ['spread', 'total', 'moneyline']
        df = df[df['market'].isin(game_line_markets)].copy()

        # Match each game to correct week from schedule
        for game_id in df['game_id'].unique():
            game_odds = df[df['game_id'] == game_id].copy()

            # Extract teams from game odds
            home_team = game_odds['home_team'].iloc[0]
            away_team = game_odds['away_team'].iloc[0]

            # Find in schedule
            match = schedules_2025[
                (schedules_2025['home_team'] == home_team) &
                (schedules_2025['away_team'] == away_team)
            ]

            if len(match) > 0:
                correct_week = int(match.iloc[0]['week'])
                # Update week in odds
                game_odds['week'] = correct_week
                game_odds['season'] = 2025
                game_odds['archived_at'] = datetime.now().isoformat()
                all_odds.append(game_odds)
                print(f"  {away_team}@{home_team} → Week {correct_week}")
            else:
                print(f"  ⚠️  {away_team}@{home_team} NOT FOUND in schedule")

    if not all_odds:
        print("\nNo odds matched to schedule")
        return

    # Combine all odds
    combined_odds = pd.concat(all_odds, ignore_index=True)

    print(f"\n{'='*70}")
    print(f"Total odds collected: {len(combined_odds)}")
    print(f"Weeks covered: {sorted(combined_odds['week'].unique())}")

    # Save by week
    output_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"
    output_dir.mkdir(parents=True, exist_ok=True)

    for week in sorted(combined_odds['week'].unique()):
        week_odds = combined_odds[combined_odds['week'] == week]
        output_file = output_dir / f"game_lines_2025_week{week:02d}.csv"
        week_odds.to_csv(output_file, index=False)

        games_count = week_odds['game_id'].nunique()
        print(f"Week {week:2d}: ✅ Saved {len(week_odds)} odds ({games_count} games) to {output_file.name}")

    print(f"\n✅ Remapping complete!")


if __name__ == "__main__":
    remap_odds_to_correct_weeks()
