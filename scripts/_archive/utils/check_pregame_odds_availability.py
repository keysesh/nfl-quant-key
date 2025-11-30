#!/usr/bin/env python3
"""
Check Pre-Game Odds Availability
=================================

This script helps determine if you have usable pre-game odds in your data,
even when you've fetched odds multiple times (both pre-game and during games).

Usage:
    python scripts/utils/check_pregame_odds_availability.py --week 9
    python scripts/utils/check_pregame_odds_availability.py --week 9 --show-games
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.odds import (
    load_game_status_map,
    filter_pregame_odds
)


def check_pregame_odds_by_game(odds_df: pd.DataFrame, week: int):
    """Check pre-game odds availability by game.

    Args:
        odds_df: Raw odds DataFrame
        week: NFL week number

    Returns:
        Dictionary with per-game analysis
    """
    # Load game status
    game_status_map = load_game_status_map(week)

    # Apply filtering to get valid pre-game odds
    valid_pregame = filter_pregame_odds(odds_df, week=week)

    # Analyze by game
    game_analysis = {}

    # Group raw odds by game
    if 'home_team' in odds_df.columns and 'away_team' in odds_df.columns:
        for _, row in odds_df.iterrows():
            home = row.get('home_team', '')
            away = row.get('away_team', '')
            if home and away:
                game_key = f"{away} @ {home}"

                if game_key not in game_analysis:
                    game_analysis[game_key] = {
                        'total_raw_odds': 0,
                        'valid_pregame_odds': 0,
                        'home_team': home,
                        'away_team': away,
                        'commence_times': set(),
                        'fetch_times': set()
                    }

                game_analysis[game_key]['total_raw_odds'] += 1

                # Track timing
                if 'commence_time' in row.index and pd.notna(row['commence_time']):
                    game_analysis[game_key]['commence_times'].add(str(row['commence_time'])[:16])
                if 'fetch_timestamp' in row.index and pd.notna(row['fetch_timestamp']):
                    game_analysis[game_key]['fetch_times'].add(str(row['fetch_timestamp'])[:16])

    # Count valid pre-game odds per game
    if not valid_pregame.empty and 'home_team' in valid_pregame.columns and 'away_team' in valid_pregame.columns:
        for _, row in valid_pregame.iterrows():
            home = row.get('home_team', '')
            away = row.get('away_team', '')
            if home and away:
                game_key = f"{away} @ {home}"
                if game_key in game_analysis:
                    game_analysis[game_key]['valid_pregame_odds'] += 1

    return game_analysis


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check if pre-game odds are available in your data'
    )
    parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='NFL week number to check'
    )
    parser.add_argument(
        '--show-games',
        action='store_true',
        help='Show detailed breakdown by game'
    )
    parser.add_argument(
        '--show-fetch-times',
        action='store_true',
        help='Show when odds were fetched for each game'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PRE-GAME ODDS AVAILABILITY CHECK")
    print("=" * 80)
    print(f"Week: {args.week}")
    print()

    # Load odds data
    odds_file = Path('data/nfl_player_props_draftkings.csv')

    if not odds_file.exists():
        print(f"❌ Odds file not found: {odds_file}")
        print(f"\n   Available odds files:")
        for f in Path('data').glob('*odds*.csv'):
            print(f"      • {f}")
        return

    print(f"📂 Loading odds from: {odds_file}")
    odds_df = pd.read_csv(odds_file)
    print(f"   ✅ Loaded {len(odds_df):,} total odds records")

    # Check for pre-game odds
    print(f"\n🔍 Filtering for valid pre-game odds...")
    valid_pregame = filter_pregame_odds(odds_df, week=args.week)

    total = len(odds_df)
    valid = len(valid_pregame)
    rejected = total - valid

    print(f"\n📊 Overall Results:")
    print(f"   Total odds in file:      {total:,}")
    print(f"   Valid pre-game odds:     {valid:,} ({valid/total*100:.1f}%)")
    print(f"   Rejected (in-game/stale): {rejected:,} ({rejected/total*100:.1f}%)")

    if valid > 0:
        print(f"\n✅ GOOD NEWS: You have {valid:,} usable pre-game odds!")
        print(f"   These odds were fetched BEFORE games started")
        print(f"   You can use these for predictions")
    else:
        print(f"\n❌ NO PRE-GAME ODDS FOUND")
        print(f"   All {total:,} odds appear to be from in-game fetches")
        print(f"   You need to fetch odds BEFORE games start")

    # Game-by-game analysis
    if args.show_games or valid == 0:
        print(f"\n📋 Game-by-Game Breakdown:")
        print("-" * 80)

        game_analysis = check_pregame_odds_by_game(odds_df, args.week)

        if not game_analysis:
            print("   ⚠️  No games found in odds data")
        else:
            games_with_pregame = 0
            games_without_pregame = 0

            for game_key, stats in sorted(game_analysis.items()):
                has_pregame = stats['valid_pregame_odds'] > 0
                if has_pregame:
                    games_with_pregame += 1
                    status_icon = "✅"
                else:
                    games_without_pregame += 1
                    status_icon = "❌"

                print(f"\n   {status_icon} {game_key}")
                print(f"      Total odds:       {stats['total_raw_odds']:,}")
                print(f"      Pre-game odds:    {stats['valid_pregame_odds']:,}")

                if args.show_fetch_times:
                    if stats['fetch_times']:
                        print(f"      Fetched at:       {', '.join(sorted(stats['fetch_times']))}")
                    if stats['commence_times']:
                        print(f"      Kickoff time:     {', '.join(sorted(stats['commence_times']))}")

            print(f"\n   Summary:")
            print(f"      Games with pre-game odds:    {games_with_pregame}")
            print(f"      Games without pre-game odds: {games_without_pregame}")

    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 80)

    if valid > 0:
        print(f"   ✓ You can proceed with predictions")
        print(f"   ✓ {valid:,} valid pre-game odds available")
        print(f"\n   Next step:")
        print(f"      python scripts/predict/generate_current_week_recommendations.py {args.week}")
    else:
        print(f"   ✗ No usable pre-game odds found")
        print(f"   ✗ Need to fetch odds BEFORE games start")
        print(f"\n   How to fix:")
        print(f"      1. Fetch odds 2-4 hours before first game:")
        print(f"         python scripts/fetch/fetch_nfl_player_props.py")
        print(f"      2. Run this check again:")
        print(f"         python scripts/utils/check_pregame_odds_availability.py --week {args.week}")
        print(f"      3. Then generate predictions:")
        print(f"         python scripts/predict/generate_current_week_recommendations.py {args.week}")

    # Show timing analysis if we have fetch_timestamp
    if 'fetch_timestamp' in odds_df.columns and odds_df['fetch_timestamp'].notna().any():
        print(f"\n⏰ Fetch Timing Analysis:")
        print("-" * 80)

        # Parse timestamps
        fetch_times = []
        for ts in odds_df['fetch_timestamp'].dropna().unique():
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    fetch_times.append(dt)
            except:
                pass

        if fetch_times:
            fetch_times.sort()
            print(f"   First fetch: {fetch_times[0].strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"   Last fetch:  {fetch_times[-1].strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"   Total fetches: {len(fetch_times)}")

            # Check if any fetches were before games
            if valid > 0:
                print(f"\n   ✅ At least one fetch was BEFORE games started")
                print(f"      → These odds are usable for predictions")
            else:
                print(f"\n   ⚠️  All fetches appear to be DURING or AFTER games")
                print(f"      → Schedule next fetch earlier")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
