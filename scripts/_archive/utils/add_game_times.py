#!/usr/bin/env python3
"""Add game time data to unified recommendations from odds API"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season


def load_game_times_from_odds(week: int = None):
    """
    Load game times from odds CSV files.

    Args:
        week: Week number (optional, tries multiple files)

    Returns:
        DataFrame with game, commence_time mapping
    """
    game_times = []

    # Try multiple possible file locations
    possible_files = []

    if week:
        possible_files.extend([
            Path(f'data/odds_week{week}_draftkings.csv'),
            Path(f'data/odds_week{week}.csv'),
        ])

    # Also try player props file (has commence_time)
    possible_files.extend([
        Path('data/nfl_player_props_draftkings.csv'),
    ])

    # Try any odds files
    data_dir = Path('data')
    if data_dir.exists():
        for odds_file in data_dir.glob('odds_week*.csv'):
            possible_files.append(odds_file)

    for file_path in possible_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)

                # Check if commence_time column exists
                if 'commence_time' in df.columns:
                    # Extract unique game-time mappings
                    if 'home_team' in df.columns and 'away_team' in df.columns:
                        # Format: "Away @ Home"
                        df['game'] = df['away_team'] + ' @ ' + df['home_team']
                    elif 'game' in df.columns:
                        # Already formatted
                        pass
                    else:
                        # Try to construct from available columns
                        continue

                    game_time_df = df[
                        ['game', 'commence_time']
                    ].drop_duplicates()
                    game_times.append(game_time_df)
                    print(
                        f"✅ Loaded game times from {file_path.name}: "
                        f"{len(game_time_df)} games"
                    )

            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                continue

    if not game_times:
        print("⚠️  No game time data found in odds files")
        return pd.DataFrame()

    # Combine all game times
    combined = pd.concat(
        game_times, ignore_index=True
    ).drop_duplicates(subset=['game'])
    return combined


def fetch_game_times_from_api(week: int = None):
    """
    Fetch game times directly from The Odds API.

    Args:
        week: Week number (optional)

    Returns:
        DataFrame with game, commence_time mapping
    """
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv('ODDS_API_KEY')

    if not API_KEY:
        print("⚠️  ODDS_API_KEY not found - cannot fetch from API")
        return pd.DataFrame()

    try:
        url = (
            "https://api.the-odds-api.com/v4/sports/"
            "americanfootball_nfl/odds"
        )
        params = {
            'apiKey': API_KEY,
            'regions': 'us',
            'markets': 'spreads',
            'oddsFormat': 'american',
            'bookmakers': 'draftkings'
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        games = response.json()

        game_times = []
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time']

            # Convert team names to abbreviations for consistency
            from scripts.fetch.fetch_live_odds import convert_team_name
            home_abbr = convert_team_name(home_team)
            away_abbr = convert_team_name(away_team)

            game_str = f"{away_abbr} @ {home_abbr}"

            game_times.append({
                'game': game_str,
                'commence_time': commence_time,
                'home_team': home_abbr,
                'away_team': away_abbr
            })

        df = pd.DataFrame(game_times)
        print(f"✅ Fetched {len(df)} game times from API")
        return df

    except Exception as e:
        print(f"⚠️  Error fetching from API: {e}")
        return pd.DataFrame()


def fetch_game_times_from_nflverse(week: int = None, season: int = None):
    """
    Fetch game times from nflverse schedules using nflreadpy.
    Python equivalent of nflreadr::load_schedules() from R.

    Reference: https://nflfastr.com/reference/fast_scraper_schedules.html

    Args:
        week: Week number (optional, filters to specific week)
        season: Season year (default: auto-detect current season)

    Returns:
        DataFrame with game, commence_time mapping
    """
    if season is None:
        season = get_current_season()
    try:
        from nfl_quant.utils.nflverse_loader import load_schedules
        from datetime import datetime

        print(f"📅 Fetching schedules from nflverse (season {season})...")

        # Load schedules from R-fetched data
        schedules_df = load_schedules(seasons=season)

        if schedules_df.empty:
            print("⚠️  No schedule data returned from nflverse")
            return pd.DataFrame()

        # Filter by week if specified
        if week is not None and 'week' in schedules_df.columns:
            schedules_df = schedules_df[schedules_df['week'] == week]

        # Check required columns
        required_cols = ['gameday', 'gametime', 'away_team', 'home_team']
        missing_cols = [
            col for col in required_cols if col not in schedules_df.columns
        ]
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(schedules_df.columns)}")
            return pd.DataFrame()

        # Convert team names to abbreviations for consistency
        from scripts.fetch.fetch_live_odds import convert_team_name

        game_times = []
        for _, row in schedules_df.iterrows():
            home_abbr = convert_team_name(row['home_team'])
            away_abbr = convert_team_name(row['away_team'])

            # Construct game string in format "Away @ Home"
            game_str = f"{away_abbr} @ {home_abbr}"

            # Convert gameday + gametime to ISO format (commence_time)
            # nflverse format: gameday='2025-09-04', gametime='20:20'
            gameday = str(row['gameday'])
            gametime = str(row['gametime'])

            # Parse the date and time
            try:
                # nflverse format: YYYY-MM-DD and HH:MM (24-hour)
                if gametime and gametime != 'nan':
                    # Combine date and time: "2025-09-04 20:20"
                    dt_str = f"{gameday} {gametime}"
                    try:
                        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
                        # Convert to ISO format with UTC
                        commence_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    except ValueError:
                        # Try with seconds
                        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                        commence_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                else:
                    # No time, use midnight
                    commence_time = f"{gameday}T00:00:00Z"

                game_times.append({
                    'game': game_str,
                    'commence_time': commence_time,
                    'home_team': home_abbr,
                    'away_team': away_abbr
                })
            except Exception as e:
                print(f"⚠️  Error parsing date/time for {game_str}: {e}")
                continue

        if not game_times:
            print("⚠️  No valid game times found in schedule")
            return pd.DataFrame()

        df = pd.DataFrame(game_times).drop_duplicates(subset=['game'])
        print(f"✅ Fetched {len(df)} game times from nflverse schedules")
        return df

    except ImportError:
        print(
            "⚠️  nflreadpy not installed - install with: pip install nflreadpy"
        )
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Error fetching from nflverse: {e}")
        return pd.DataFrame()


def add_game_times_to_recommendations(week: int = None):
    """
    Add game time data to unified recommendations CSV.

    Args:
        week: Week number (optional)

    Returns:
        Updated DataFrame with game_time column
    """
    recs_path = Path('reports/unified_betting_recommendations.csv')

    if not recs_path.exists():
        print(f"❌ {recs_path} not found")
        return None

    df = pd.read_csv(recs_path)
    print(f"✅ Loaded {len(df)} recommendations")

    # Check if game_time already exists
    if 'game_time' in df.columns or 'commence_time' in df.columns:
        print("✅ Game times already present in recommendations")
        return df

    # Try to load from existing files first
    game_times_df = load_game_times_from_odds(week)

    # If not found, try fetching from Odds API
    if game_times_df.empty:
        print("📡 Fetching game times from Odds API...")
        game_times_df = fetch_game_times_from_api(week)

    # If still not found, try fetching from nflverse schedules
    if game_times_df.empty:
        print("📅 Fetching game times from nflverse schedules...")
        # Try current season and previous season as fallback
        current_season = get_current_season()
        game_times_df = fetch_game_times_from_nflverse(week, season=current_season)
        if game_times_df.empty:
            game_times_df = fetch_game_times_from_nflverse(week, season=current_season - 1)

    if game_times_df.empty:
        print("⚠️  No game time data available")
        df['game_time'] = None
        df['commence_time'] = None
        return df

    # Merge game times
    # Match by game string
    df = df.merge(
        game_times_df[['game', 'commence_time']],
        on='game',
        how='left'
    )

    # Rename commence_time to game_time for consistency
    if 'commence_time' in df.columns:
        df['game_time'] = df['commence_time']

    # Count matches
    matched = df['game_time'].notna().sum()
    pct = matched / len(df) * 100
    print(f"✅ Matched game times for {matched}/{len(df)} bets ({pct:.1f}%)")

    # Save updated recommendations
    df.to_csv(recs_path, index=False)
    print(f"💾 Saved updated recommendations with game times to {recs_path}")

    return df


if __name__ == '__main__':
    import sys

    week = None
    if len(sys.argv) > 1:
        try:
            week = int(sys.argv[1])
        except ValueError:
            print("Usage: python add_game_times.py [week_number]")
            sys.exit(1)

    print("="*80)
    print("ADDING GAME TIMES TO UNIFIED RECOMMENDATIONS")
    print("="*80)
    print()

    add_game_times_to_recommendations(week)
