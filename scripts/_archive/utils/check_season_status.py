#!/usr/bin/env python3
"""
NFL Season Status Checker
==========================

Shows current NFL week, completed games, upcoming games, and data availability.

Usage:
    python scripts/utils/check_season_status.py [--week WEEK] [--season SEASON]

Example:
    python scripts/utils/check_season_status.py
    python scripts/utils/check_season_status.py --week 10 --season 2025
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season


class SeasonStatusChecker:
    """Check NFL season status and data availability."""

    def __init__(self, week: Optional[int] = None, season: Optional[int] = None):
        self.week = week
        self.season = season if season is not None else get_current_season()
        self.base_dir = Path(Path.cwd())
        self.now = datetime.now(timezone.utc)

    def determine_current_week(self) -> int:
        """Determine current NFL week based on game schedules."""
        # Try to load from NFLverse games
        games_file = self.base_dir / 'data/nflverse/games.parquet'

        if not games_file.exists():
            # Try PBP data to infer week
            pbp_file = self.base_dir / f'data/nflverse/pbp_{self.season}.parquet'
            if pbp_file.exists():
                df = pd.read_parquet(pbp_file)
                if 'week' in df.columns:
                    latest_week = int(df['week'].max())
                    # If we have data for this week and it's recent, we're likely in that week
                    return latest_week

        # Fallback: estimate based on date
        # NFL season typically starts first week of September
        season_start = datetime(self.season, 9, 1, tzinfo=timezone.utc)
        weeks_elapsed = (self.now - season_start).days // 7

        # NFL weeks typically start on Tuesday/Wednesday
        # Adjust based on day of week
        day_of_week = self.now.weekday()  # 0=Monday, 6=Sunday
        if day_of_week >= 1:  # Tuesday-Sunday
            weeks_elapsed += 1

        # Cap at reasonable range
        current_week = min(max(1, weeks_elapsed), 18)

        return current_week

    def check_completed_games(self, week: int) -> Dict:
        """Check which games have been completed."""
        print("="*80)
        print("COMPLETED GAMES CHECK")
        print("="*80)

        results = {
            'week': week,
            'completed': [],
            'upcoming': [],
            'in_progress': [],
            'game_details': []
        }

        # Check PBP data for completed games
        pbp_file = self.base_dir / f'data/nflverse/pbp_{self.season}.parquet'
        if pbp_file.exists():
            df = pd.read_parquet(pbp_file)
            week_games = df[df['week'] == week] if 'week' in df.columns else pd.DataFrame()

            if not week_games.empty and 'game_id' in week_games.columns:
                game_ids = week_games['game_id'].unique()
                results['completed'] = [str(gid) for gid in game_ids]

                # Get game details
                if 'posteam' in week_games.columns and 'defteam' in week_games.columns:
                    game_details = []
                    for game_id in game_ids:
                        game_plays = week_games[week_games['game_id'] == game_id]
                        if not game_plays.empty:
                            # Try to get teams from plays
                            teams = set()
                            if 'posteam' in game_plays.columns:
                                teams.update(game_plays['posteam'].dropna().unique())
                            if 'defteam' in game_plays.columns:
                                teams.update(game_plays['defteam'].dropna().unique())

                            if len(teams) >= 2:
                                team_list = sorted(list(teams))
                                game_details.append({
                                    'game_id': str(game_id),
                                    'away_team': team_list[0],
                                    'home_team': team_list[1] if len(team_list) > 1 else team_list[0],
                                    'plays': len(game_plays)
                                })

                    results['game_details'] = game_details

                print(f"\nWeek {week} Completed Games: {len(game_ids)} games")
                if results['game_details']:
                    print("\nCompleted Games:")
                    for game in results['game_details'][:15]:
                        print(f"  ✅ {game['away_team']} @ {game['home_team']} ({game['plays']} plays)")
                else:
                    print("  (Game details not available)")

        # Check Sleeper stats for completed games
        sleeper_file = self.base_dir / f'data/sleeper_stats/stats_week{week}_{self.season}.csv'
        if sleeper_file.exists():
            df = pd.read_csv(sleeper_file)
            if 'team' in df.columns:
                teams_with_stats = df['team'].dropna().unique()
                teams_with_stats = [t for t in teams_with_stats if str(t) != 'nan' and str(t) != '']
                print(f"\nWeek {week} Teams with Stats: {len(teams_with_stats)} teams")
                print(f"  Teams: {', '.join(sorted(teams_with_stats)[:10])}")
                results['has_stats'] = True
            else:
                results['has_stats'] = False
        else:
            results['has_stats'] = False
            print(f"\n⚠️  No Sleeper stats found for Week {week}")

        return results

    def check_upcoming_games(self, week: int, completed_game_ids: List[str] = None) -> Dict:
        """Check upcoming games from odds data."""
        print("\n" + "="*80)
        print("UPCOMING GAMES CHECK")
        print("="*80)

        results = {
            'week': week,
            'games': [],
            'total_games': 0,
            'upcoming': [],
            'completed': []
        }

        completed_game_ids = completed_game_ids or []

        # Check odds file
        odds_file = self.base_dir / f'data/odds_week{week}_draftkings.csv'
        if odds_file.exists():
            df = pd.read_csv(odds_file)

            # Remove duplicates (odds file may have multiple rows per game)
            if 'home_team' in df.columns and 'away_team' in df.columns:
                df_unique = df.drop_duplicates(subset=['home_team', 'away_team'])
            else:
                df_unique = df

            results['total_games'] = len(df_unique)
            results['games'] = df_unique.to_dict('records')

            print(f"\nWeek {week} Games in Odds Data: {len(df_unique)} unique games")

            # Categorize games
            upcoming_games = []
            completed_games = []

            # Show games with start times
            if 'commence_time' in df_unique.columns:
                print("\nGame Schedule:")
                for _, row in df_unique.iterrows():
                    away = row.get('away_team', '?')
                    home = row.get('home_team', '?')
                    game = f"{away} @ {home}"
                    commence_str = row.get('commence_time', '')

                    # Parse commence time
                    try:
                        if commence_str:
                            commence = pd.to_datetime(commence_str)
                            if commence < self.now:
                                status = "⏰ Started"
                                completed_games.append(game)
                            else:
                                status = "📅 Upcoming"
                                upcoming_games.append(game)
                        else:
                            status = "❓ Unknown"
                            upcoming_games.append(game)
                    except:
                        status = "❓ Unknown"
                        upcoming_games.append(game)

                    print(f"  {status} {game:30s} - {commence_str}")

                results['upcoming'] = upcoming_games
                results['completed'] = completed_games

                print(f"\nSummary:")
                print(f"  📅 Upcoming: {len(upcoming_games)} games")
                print(f"  ⏰ Started/Completed: {len(completed_games)} games")
        else:
            print(f"\n⚠️  No odds data found for Week {week}")

        return results

    def check_data_availability(self) -> Dict:
        """Check what data we have for each week."""
        print("\n" + "="*80)
        print("DATA AVAILABILITY BY WEEK")
        print("="*80)

        results = {}

        # Check Sleeper stats
        print("\nSleeper Stats:")
        sleeper_files = sorted(self.base_dir.glob(f'data/sleeper_stats/stats_week*_{self.season}.csv'))
        sleeper_weeks = sorted([int(f.stem.split('_')[1].replace('week', '')) for f in sleeper_files])

        for week in range(1, 19):
            has_data = week in sleeper_weeks
            status = "✅" if has_data else "❌"
            results[f'week_{week}_sleeper'] = has_data
            if sleeper_weeks and week <= max(sleeper_weeks) + 2:  # Show up to 2 weeks ahead
                print(f"  Week {week:2d}: {status} {'Available' if has_data else 'Missing'}")
            elif not sleeper_weeks and week <= 3:  # If no data, show first few weeks
                print(f"  Week {week:2d}: {status} {'Available' if has_data else 'Missing'}")

        # Check NFLverse PBP
        print("\nNFLverse PBP Data:")
        pbp_file = self.base_dir / f'data/nflverse/pbp_{self.season}.parquet'
        if pbp_file.exists():
            df = pd.read_parquet(pbp_file)
            if 'week' in df.columns:
                pbp_weeks = sorted(df['week'].unique().tolist())
                for week in range(1, 19):
                    has_data = week in pbp_weeks
                    status = "✅" if has_data else "❌"
                    results[f'week_{week}_pbp'] = has_data
                    if pbp_weeks and week <= max(pbp_weeks) + 2:
                        print(f"  Week {week:2d}: {status} {'Available' if has_data else 'Missing'}")
                    elif not pbp_weeks and week <= 3:
                        print(f"  Week {week:2d}: {status} {'Available' if has_data else 'Missing'}")

        return results

    def generate_summary(self) -> Dict:
        """Generate comprehensive season status summary."""
        print("="*80)
        print("NFL SEASON STATUS CHECKER")
        print("="*80)
        print(f"Season: {self.season}")
        print(f"Current Date: {self.now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*80)

        # Determine current week
        if self.week is None:
            current_week = self.determine_current_week()
            print(f"\n📅 Estimated Current Week: {current_week}")
        else:
            current_week = self.week
            print(f"\n📅 Checking Week: {current_week}")

        # Check completed games
        completed = self.check_completed_games(current_week)

        # Check upcoming games (pass completed game IDs to avoid duplicates)
        upcoming = self.check_upcoming_games(current_week, completed.get('completed', []))

        # Check data availability
        data_availability = self.check_data_availability()

        # Generate summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        print(f"\nCurrent Week: {current_week}")
        print(f"Completed Games (Week {current_week}): {len(completed.get('completed', []))} games")
        print(f"Upcoming Games (Week {current_week}): {upcoming.get('total_games', 0)} games")
        print(f"Week {current_week} Stats Available: {'✅ Yes' if completed.get('has_stats') else '❌ No'}")

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)

        if not completed.get('has_stats'):
            print(f"\n⚠️  Week {current_week} stats not yet available")
            print(f"   Action: Wait for games to complete, then fetch:")
            print(f"   python scripts/fetch/fetch_sleeper_week9_stats.py --week {current_week} --season {self.season}")

        if upcoming.get('total_games', 0) > 0:
            print(f"\n✅ Week {current_week} games scheduled")
            print(f"   Ready to generate predictions for {upcoming.get('total_games')} games")

        # Show what weeks we can predict
        sleeper_week_files = list(self.base_dir.glob(f'data/sleeper_stats/stats_week*_{self.season}.csv'))
        latest_sleeper_week = max([int(f.stem.split('_')[1].replace('week', ''))
                                   for f in sleeper_week_files]) if sleeper_week_files else 0

        if latest_sleeper_week > 0:
            print(f"\n📊 Latest Stats Available: Week {latest_sleeper_week}")
            print(f"   Can generate predictions for Week {latest_sleeper_week + 1} and beyond")
            print(f"   (Using Week {latest_sleeper_week} stats for trailing averages)")

        return {
            'current_week': current_week,
            'completed_games': completed,
            'upcoming_games': upcoming,
            'data_availability': data_availability
        }


def main():
    parser = argparse.ArgumentParser(description='Check NFL season status')
    parser.add_argument('--week', type=int, help='Specific week to check (default: auto-detect)')
    parser.add_argument('--season', type=int, default=None,
                        help=f'Season year (default: auto-detect, currently {get_current_season()})')

    args = parser.parse_args()

    checker = SeasonStatusChecker(args.week, args.season)
    summary = checker.generate_summary()

    sys.exit(0)


if __name__ == '__main__':
    main()
