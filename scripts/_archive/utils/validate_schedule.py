#!/usr/bin/env python3
"""
Schedule validation utilities to prevent generating picks for incorrect games.

Validates that requested games actually exist in the NFL schedule before
generating predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from nfl_quant.utils.season_config import get_current_nfl_season
from nfl_quant.utils.nflverse_loader import load_schedules

class ScheduleValidator:
    """Validates games against official NFL schedule."""

    def __init__(self, season: Optional[int] = None):
        """
        Initialize with season year.

        Args:
            season: NFL season year. If None, automatically detects current season.
        """
        self.season = season if season is not None else get_current_nfl_season()
        self._schedule = None

    def load_schedule(self) -> pd.DataFrame:
        """Load NFL schedule for the season."""
        if self._schedule is None:
            print(f"Loading {self.season} NFL schedule...")
            self._schedule = load_schedules(seasons=self.season)
            print(f"✅ Loaded schedule: {len(self._schedule)} games")
        return self._schedule

    def get_week_games(self, week: int) -> pd.DataFrame:
        """Get all games for a specific week."""
        schedule = self.load_schedule()
        week_games = schedule[schedule['week'] == week].copy()
        return week_games[['week', 'gameday', 'away_team', 'home_team',
                          'away_score', 'home_score', 'game_id']]

    def validate_game(self, away_team: str, home_team: str, week: int) -> Tuple[bool, str]:
        """
        Validate that a specific game exists in the schedule.

        Returns:
            (is_valid, message)
        """
        schedule = self.load_schedule()

        # Look for exact match
        game = schedule[(schedule['week'] == week) &
                       (schedule['away_team'] == away_team) &
                       (schedule['home_team'] == home_team)]

        if len(game) > 0:
            game_row = game.iloc[0]
            game_date = game_row['gameday']
            return True, f"✅ Valid: {away_team} @ {home_team} on {game_date}"

        # Check if teams are playing each other but with reversed home/away
        reversed_game = schedule[(schedule['week'] == week) &
                                (schedule['away_team'] == home_team) &
                                (schedule['home_team'] == away_team)]

        if len(reversed_game) > 0:
            return False, f"❌ Invalid: {home_team} @ {away_team} exists, but you specified {away_team} @ {home_team}"

        # Check if either team is playing in that week
        away_team_games = schedule[(schedule['week'] == week) &
                                   ((schedule['away_team'] == away_team) |
                                    (schedule['home_team'] == away_team))]

        home_team_games = schedule[(schedule['week'] == week) &
                                   ((schedule['away_team'] == home_team) |
                                    (schedule['home_team'] == home_team))]

        if len(away_team_games) > 0:
            actual_game = away_team_games.iloc[0]
            actual_opponent = actual_game['home_team'] if actual_game['away_team'] == away_team else actual_game['away_team']
            return False, f"❌ Invalid: {away_team} plays {actual_opponent} in Week {week}, not {home_team}"

        if len(home_team_games) > 0:
            actual_game = home_team_games.iloc[0]
            actual_opponent = actual_game['home_team'] if actual_game['away_team'] == home_team else actual_game['away_team']
            return False, f"❌ Invalid: {home_team} plays {actual_opponent} in Week {week}, not {away_team}"

        # Neither team is playing in that week (bye week)
        return False, f"❌ Invalid: Neither {away_team} nor {home_team} is playing in Week {week} (possible bye week)"

    def list_week_games(self, week: int, format: str = 'simple') -> None:
        """
        Print all games for a specific week.

        Args:
            week: Week number
            format: 'simple', 'detailed', or 'numbered'
        """
        games = self.get_week_games(week)

        if len(games) == 0:
            print(f"\n❌ No games found for Week {week}")
            return

        print(f"\n📅 Week {week} Games ({len(games)} total):")
        print("="*80)

        for idx, (_, game) in enumerate(games.iterrows(), 1):
            away = game['away_team']
            home = game['home_team']
            gameday = game['gameday']

            if format == 'numbered':
                print(f"{idx:2d}. {away} @ {home} ({gameday})")
            elif format == 'detailed':
                score_away = game.get('away_score', 'N/A')
                score_home = game.get('home_score', 'N/A')
                if pd.notna(score_away) and pd.notna(score_home):
                    print(f"{away} @ {home}: {score_away}-{score_home} ({gameday})")
                else:
                    print(f"{away} @ {home} ({gameday}) - Not played yet")
            else:  # simple
                print(f"  {away} @ {home}")

        print("="*80)

    def suggest_games(self, week: int, team: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Suggest valid games for prediction.

        Args:
            week: Week number
            team: Optional team filter (returns games involving this team)

        Returns:
            List of (away_team, home_team) tuples
        """
        games = self.get_week_games(week)

        if team:
            games = games[(games['away_team'] == team) | (games['home_team'] == team)]

        return [(row['away_team'], row['home_team']) for _, row in games.iterrows()]

    def validate_predictions_file(self, picks_file: str, week: int) -> Tuple[bool, Dict]:
        """
        Validate that a predictions file references valid games.

        Args:
            picks_file: Path to CSV with predictions
            week: Expected week number

        Returns:
            (is_valid, details_dict)
        """
        try:
            df = pd.read_csv(picks_file)
        except Exception as e:
            return False, {'error': f"Could not read file: {e}"}

        if 'game' not in df.columns:
            return False, {'error': "No 'game' column in predictions file"}

        # Extract unique games from predictions
        unique_games = df['game'].dropna().unique()

        results = {
            'total_picks': len(df),
            'unique_games': len(unique_games),
            'valid_games': [],
            'invalid_games': [],
            'validation_details': []
        }

        for game_str in unique_games:
            if '@' not in str(game_str):
                results['invalid_games'].append(game_str)
                results['validation_details'].append({
                    'game': game_str,
                    'valid': False,
                    'reason': 'Invalid format (missing @)'
                })
                continue

            parts = game_str.split('@')
            if len(parts) != 2:
                results['invalid_games'].append(game_str)
                results['validation_details'].append({
                    'game': game_str,
                    'valid': False,
                    'reason': 'Invalid format (cannot parse teams)'
                })
                continue

            away_team = parts[0].strip()
            home_team = parts[1].strip()

            is_valid, message = self.validate_game(away_team, home_team, week)

            if is_valid:
                results['valid_games'].append(game_str)
            else:
                results['invalid_games'].append(game_str)

            results['validation_details'].append({
                'game': game_str,
                'valid': is_valid,
                'message': message
            })

        all_valid = len(results['invalid_games']) == 0
        return all_valid, results

def main():
    """CLI for schedule validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate NFL schedule and games')
    parser.add_argument('--week', type=int, required=True, help='Week number')
    parser.add_argument('--season', type=int, default=None,
                       help=f'Season year (default: auto-detect current season, currently {get_current_nfl_season()})')
    parser.add_argument('--list-games', action='store_true', help='List all games in week')
    parser.add_argument('--validate-game', nargs=2, metavar=('AWAY', 'HOME'),
                       help='Validate specific game (e.g., --validate-game PHI DAL)')
    parser.add_argument('--validate-file', type=str, help='Validate predictions file')
    parser.add_argument('--team', type=str, help='Filter games by team')

    args = parser.parse_args()

    validator = ScheduleValidator(season=args.season)

    if args.list_games:
        validator.list_week_games(args.week, format='numbered')

    if args.validate_game:
        away_team, home_team = args.validate_game
        is_valid, message = validator.validate_game(away_team, home_team, args.week)
        print(f"\n{message}")
        return 0 if is_valid else 1

    if args.validate_file:
        is_valid, details = validator.validate_predictions_file(args.validate_file, args.week)
        print(f"\n📋 Validation Results for: {args.validate_file}")
        print("="*80)
        print(f"Total Picks: {details['total_picks']}")
        print(f"Unique Games: {details['unique_games']}")
        print(f"Valid Games: {len(details['valid_games'])}")
        print(f"Invalid Games: {len(details['invalid_games'])}")
        print()

        for detail in details['validation_details']:
            status = "✅" if detail['valid'] else "❌"
            print(f"{status} {detail['game']}")
            if 'message' in detail:
                print(f"   {detail['message']}")

        return 0 if is_valid else 1

    if args.team:
        games = validator.suggest_games(args.week, args.team)
        print(f"\n📅 Week {args.week} games involving {args.team}:")
        for away, home in games:
            print(f"  {away} @ {home}")

    if not (args.list_games or args.validate_game or args.validate_file or args.team):
        validator.list_week_games(args.week)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
