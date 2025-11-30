#!/usr/bin/env python3
"""
Archive Game Line Odds

Extracts and archives spread, total, and moneyline odds from current week
DraftKings odds files into a historical format for backtesting.

This creates a historical record of game lines that can be used to validate
predictions against actual market conditions.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season, get_current_week

PROJECT_ROOT = Path(__file__).parent.parent.parent


def extract_game_lines_from_week(season: int, week: int) -> pd.DataFrame:
    """
    Extract game lines (spread, total, moneyline) from a week's odds file.

    Args:
        season: Season year
        week: Week number

    Returns:
        DataFrame with game line odds
    """
    # Check multiple possible file locations
    possible_files = [
        PROJECT_ROOT / f"data/odds_week{week}_draftkings.csv",
        PROJECT_ROOT / f"data/odds/week_{week}_game_lines.csv",
        PROJECT_ROOT / f"data/draftkings_week{week}.csv",
    ]

    odds_file = None
    for file_path in possible_files:
        if file_path.exists():
            odds_file = file_path
            break

    if odds_file is None:
        raise FileNotFoundError(f"No odds file found for season {season} week {week}")

    print(f"Reading odds from: {odds_file}")
    df = pd.read_csv(odds_file)

    # Filter to game lines only (spread, total, moneyline)
    game_line_markets = ['spread', 'total', 'moneyline']
    game_lines = df[df['market'].isin(game_line_markets)].copy()

    # Add metadata
    game_lines['season'] = season
    game_lines['week'] = week
    game_lines['archived_at'] = datetime.now().isoformat()

    return game_lines


def archive_current_week():
    """Archive the current week's game line odds."""
    season = get_current_season()
    week = get_current_week()

    print(f"Archiving game lines for {season} Week {week}")

    try:
        game_lines = extract_game_lines_from_week(season, week)

        if len(game_lines) == 0:
            print("Warning: No game lines found in odds file")
            return

        # Save to historical directory
        output_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"game_lines_{season}_week{week:02d}.csv"
        game_lines.to_csv(output_file, index=False)

        print(f"✅ Archived {len(game_lines)} game line odds to: {output_file}")

        # Count by market type
        market_counts = game_lines['market'].value_counts()
        print("\nBreakdown by market:")
        for market, count in market_counts.items():
            print(f"  {market}: {count}")

        # Count unique games
        unique_games = game_lines['game_id'].nunique()
        print(f"\nTotal games: {unique_games}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error archiving odds: {e}")


def archive_historical_weeks(season: int, start_week: int, end_week: int):
    """
    Archive multiple weeks of historical odds.

    Args:
        season: Season year
        start_week: First week to archive
        end_week: Last week to archive
    """
    print(f"Archiving game lines for {season} Weeks {start_week}-{end_week}")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for week in range(start_week, end_week + 1):
        try:
            game_lines = extract_game_lines_from_week(season, week)

            if len(game_lines) == 0:
                print(f"Week {week}: No game lines found")
                fail_count += 1
                continue

            # Save to historical directory
            output_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"game_lines_{season}_week{week:02d}.csv"
            game_lines.to_csv(output_file, index=False)

            unique_games = game_lines['game_id'].nunique()
            print(f"Week {week:2d}: ✅ Archived {len(game_lines)} odds ({unique_games} games)")
            success_count += 1

        except FileNotFoundError:
            print(f"Week {week:2d}: ⚠️  Odds file not found")
            fail_count += 1
        except Exception as e:
            print(f"Week {week:2d}: ❌ Error: {e}")
            fail_count += 1

    print("\n" + "=" * 70)
    print(f"Archive complete: {success_count} weeks successful, {fail_count} failed")


def load_historical_game_lines(season: int, week: int) -> pd.DataFrame:
    """
    Load archived game line odds for a specific week.

    Args:
        season: Season year
        week: Week number

    Returns:
        DataFrame with game line odds
    """
    file_path = PROJECT_ROOT / "data" / "historical" / "game_lines" / f"game_lines_{season}_week{week:02d}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"No archived game lines found for {season} Week {week}")

    return pd.read_csv(file_path)


def create_unified_game_lines_file(season: int):
    """
    Combine all archived game lines for a season into one file.

    Args:
        season: Season year
    """
    print(f"Creating unified game lines file for {season}")

    game_lines_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"

    if not game_lines_dir.exists():
        print(f"No game lines directory found at {game_lines_dir}")
        return

    # Find all files for this season
    pattern = f"game_lines_{season}_week*.csv"
    files = sorted(game_lines_dir.glob(pattern))

    if not files:
        print(f"No game line files found for season {season}")
        return

    print(f"Found {len(files)} week files")

    # Combine all weeks
    all_weeks = []
    for file_path in files:
        df = pd.read_csv(file_path)
        all_weeks.append(df)

    combined = pd.concat(all_weeks, ignore_index=True)

    # Save unified file
    output_file = game_lines_dir / f"game_lines_{season}_all_weeks.csv"
    combined.to_csv(output_file, index=False)

    print(f"✅ Created unified file: {output_file}")
    print(f"   Total odds: {len(combined)}")
    print(f"   Total games: {combined['game_id'].nunique()}")
    print(f"   Weeks covered: {combined['week'].min()} - {combined['week'].max()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Archive game line odds for backtesting")
    parser.add_argument("--current", action="store_true", help="Archive current week")
    parser.add_argument("--season", type=int, help="Season year")
    parser.add_argument("--start-week", type=int, help="Start week")
    parser.add_argument("--end-week", type=int, help="End week")
    parser.add_argument("--unified", action="store_true", help="Create unified season file")

    args = parser.parse_args()

    if args.current:
        archive_current_week()
    elif args.unified and args.season:
        create_unified_game_lines_file(args.season)
    elif args.season and args.start_week and args.end_week:
        archive_historical_weeks(args.season, args.start_week, args.end_week)
    else:
        print("Usage:")
        print("  Archive current week:")
        print("    python scripts/data/archive_game_line_odds.py --current")
        print()
        print("  Archive historical weeks:")
        print("    python scripts/data/archive_game_line_odds.py --season 2025 --start-week 1 --end-week 10")
        print()
        print("  Create unified file:")
        print("    python scripts/data/archive_game_line_odds.py --season 2025 --unified")
