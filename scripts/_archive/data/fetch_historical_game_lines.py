#!/usr/bin/env python3
"""
Fetch Historical Game Line Odds

Retrieves historical spread, total, and moneyline odds for past NFL games.
This fills gaps in our historical data to enable comprehensive backtesting.

Data Sources:
1. The Odds API (primary) - https://the-odds-api.com/
2. Local archives if available

Usage:
    # Fetch specific week
    python scripts/data/fetch_historical_game_lines.py --season 2025 --week 1

    # Fetch multiple weeks
    python scripts/data/fetch_historical_game_lines.py --season 2025 --start-week 1 --end-week 7

    # Use API key from environment
    export ODDS_API_KEY="your_api_key_here"
"""

import os
import sys
import argparse
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_api_key_from_config() -> Optional[str]:
    """Load API key from config file if available."""
    config_file = PROJECT_ROOT / "configs" / "odds_api_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            return config.get('api_key')
    return None


def get_nfl_game_ids_for_week(season: int, week: int) -> pd.DataFrame:
    """
    Get NFL game IDs and teams from NFLverse schedules.

    Args:
        season: Season year
        week: Week number

    Returns:
        DataFrame with game_id, home_team, away_team, game_date
    """
    schedules_file = PROJECT_ROOT / "data/nflverse/schedules.parquet"
    if not schedules_file.exists():
        raise FileNotFoundError(f"Schedules file not found: {schedules_file}")

    df = pd.read_parquet(schedules_file)
    df = df[(df['season'] == season) & (df['week'] == week)].copy()

    if len(df) == 0:
        raise ValueError(f"No games found for {season} week {week}")

    return df[['game_id', 'home_team', 'away_team', 'gameday']].copy()


def fetch_odds_api_historical(
    api_key: str,
    sport: str = "americanfootball_nfl",
    regions: str = "us",
    markets: str = "spreads,totals,h2h",
    date: Optional[str] = None
) -> List[dict]:
    """
    Fetch historical odds from The Odds API.

    Args:
        api_key: The Odds API key
        sport: Sport identifier
        regions: Regions to fetch odds for (us, uk, eu, au)
        markets: Comma-separated markets (spreads, totals, h2h)
        date: ISO date string (YYYY-MM-DD) - fetches historical odds near this date

    Returns:
        List of game odds dictionaries

    Note:
        - Free tier: 500 requests/month
        - Historical data may require paid plan
        - Docs: https://the-odds-api.com/liveapi/guides/v4/
    """
    base_url = "https://api.the-odds-api.com/v4"

    if date:
        # Historical endpoint (may require paid plan)
        url = f"{base_url}/historical/sports/{sport}/odds"
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "date": date,
            "oddsFormat": "american",
            "dateFormat": "iso"
        }
    else:
        # Current odds endpoint (free tier)
        url = f"{base_url}/sports/{sport}/odds"
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso"
        }

    print(f"Fetching from: {url}")
    print(f"Params: {params}")

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    # Check remaining requests
    remaining = response.headers.get('x-requests-remaining', 'unknown')
    print(f"API requests remaining: {remaining}")

    result = response.json()

    # Historical endpoint returns {data: [...], timestamp: ...}
    # Current endpoint returns [...]
    if isinstance(result, dict) and 'data' in result:
        return result['data']
    else:
        return result


def convert_odds_api_to_our_format(
    api_data: List[dict],
    season: int,
    week: int,
    schedules_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert Odds API format to our standardized format.

    Args:
        api_data: Raw data from Odds API
        season: Season year
        week: Week number
        schedules_df: DataFrame with NFLverse game IDs

    Returns:
        DataFrame in our standardized format
    """
    rows = []

    for game in api_data:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

        # Try to match to NFLverse game_id
        # Odds API uses full team names, NFLverse uses abbreviations
        # This mapping might need adjustment
        team_abbrev_map = {
            'Arizona Cardinals': 'ARI',
            'Atlanta Falcons': 'ATL',
            'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF',
            'Carolina Panthers': 'CAR',
            'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LA',
            'Miami Dolphins': 'MIA',
            'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE',
            'New Orleans Saints': 'NO',
            'New York Giants': 'NYG',
            'New York Jets': 'NYJ',
            'Philadelphia Eagles': 'PHI',
            'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA',
            'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN',
            'Washington Commanders': 'WAS',
        }

        home_abbr = team_abbrev_map.get(home_team, home_team)
        away_abbr = team_abbrev_map.get(away_team, away_team)

        # Create game_id
        game_id = f"{season}_{week:02d}_{away_abbr}_{home_abbr}"

        # Process bookmakers
        for bookmaker in game.get('bookmakers', []):
            sportsbook = bookmaker.get('key', 'unknown')

            if sportsbook != 'draftkings':
                continue  # Only keep DraftKings for consistency

            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')

                if market_key == 'spreads':
                    # Spread odds
                    for outcome in market.get('outcomes', []):
                        side = 'home' if outcome['name'] == home_team else 'away'
                        point = outcome.get('point', 0)
                        price = outcome.get('price', -110)

                        rows.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': sportsbook,
                            'market': 'spread',
                            'side': side,
                            'point': point,
                            'price': price,
                            'season': season,
                            'week': week,
                            'archived_at': datetime.now().isoformat()
                        })

                elif market_key == 'totals':
                    # Total odds
                    for outcome in market.get('outcomes', []):
                        side = outcome['name'].lower()  # 'Over' or 'Under'
                        point = outcome.get('point', 0)
                        price = outcome.get('price', -110)

                        rows.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': sportsbook,
                            'market': 'total',
                            'side': side.lower(),
                            'point': point,
                            'price': price,
                            'season': season,
                            'week': week,
                            'archived_at': datetime.now().isoformat()
                        })

                elif market_key == 'h2h':
                    # Moneyline odds
                    for outcome in market.get('outcomes', []):
                        side = 'home' if outcome['name'] == home_team else 'away'
                        price = outcome.get('price', 0)

                        rows.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': sportsbook,
                            'market': 'moneyline',
                            'side': side,
                            'point': None,
                            'price': price,
                            'season': season,
                            'week': week,
                            'archived_at': datetime.now().isoformat()
                        })

    return pd.DataFrame(rows)


def fetch_and_archive_week(season: int, week: int, api_key: Optional[str] = None):
    """
    Fetch and archive historical game lines for a specific week.

    Args:
        season: Season year
        week: Week number
        api_key: The Odds API key (or from environment)
    """
    print(f"\n{'=' * 70}")
    print(f"Fetching Historical Game Lines: {season} Week {week}")
    print(f"{'=' * 70}")

    # Get API key from multiple sources
    if api_key is None:
        api_key = os.getenv('ODDS_API_KEY')

    if api_key is None:
        api_key = load_api_key_from_config()

    if not api_key:
        print("❌ Error: No API key provided")
        print("Set ODDS_API_KEY environment variable or pass --api-key")
        print("\nTo get an API key:")
        print("1. Sign up at https://the-odds-api.com/")
        print("2. Free tier: 500 requests/month")
        print("3. Historical data may require paid plan")
        return

    # Get NFLverse schedule for this week
    try:
        schedules_df = get_nfl_game_ids_for_week(season, week)
        print(f"Found {len(schedules_df)} games in NFLverse schedule")
    except Exception as e:
        print(f"❌ Error loading schedule: {e}")
        return

    # Get game date for historical fetch (use first game's date)
    game_date = schedules_df['gameday'].iloc[0]
    print(f"Game date: {game_date}")

    # Convert date to ISO timestamp with time (noon on game day)
    # The Odds API historical endpoint requires full timestamp
    if isinstance(game_date, str):
        historical_timestamp = f"{game_date}T12:00:00Z"
    else:
        historical_timestamp = f"{game_date.strftime('%Y-%m-%d')}T12:00:00Z"
    print(f"Historical timestamp: {historical_timestamp}")

    # Fetch from Odds API
    try:
        print("\nFetching from The Odds API...")
        api_data = fetch_odds_api_historical(
            api_key=api_key,
            date=historical_timestamp  # This gets historical odds near this timestamp
        )
        print(f"Retrieved {len(api_data)} games from API")
    except Exception as e:
        print(f"❌ Error fetching from API: {e}")
        print("\nNote: Historical data may require a paid plan")
        print("Free tier only provides current/upcoming games")
        return

    if len(api_data) == 0:
        print("⚠️  No data returned from API")
        return

    # Convert to our format
    odds_df = convert_odds_api_to_our_format(api_data, season, week, schedules_df)

    if len(odds_df) == 0:
        print("⚠️  No odds converted to our format")
        return

    # Save to archive
    output_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"game_lines_{season}_week{week:02d}.csv"
    odds_df.to_csv(output_file, index=False)

    print(f"\n✅ Archived {len(odds_df)} odds to: {output_file}")
    print(f"   Games: {odds_df['game_id'].nunique()}")
    print(f"   Markets: {odds_df['market'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical NFL game line odds")
    parser.add_argument("--season", type=int, default=get_current_season(),
                       help="Season year")
    parser.add_argument("--week", type=int, help="Single week to fetch")
    parser.add_argument("--start-week", type=int, help="Start week for range")
    parser.add_argument("--end-week", type=int, help="End week for range")
    parser.add_argument("--api-key", type=str, help="The Odds API key")

    args = parser.parse_args()

    if args.week:
        # Single week
        fetch_and_archive_week(args.season, args.week, args.api_key)
    elif args.start_week and args.end_week:
        # Range of weeks
        for week in range(args.start_week, args.end_week + 1):
            fetch_and_archive_week(args.season, week, args.api_key)
    else:
        print("Usage:")
        print("  Single week:")
        print("    python scripts/data/fetch_historical_game_lines.py --season 2025 --week 1")
        print()
        print("  Multiple weeks:")
        print("    python scripts/data/fetch_historical_game_lines.py --season 2025 --start-week 1 --end-week 7")
        print()
        print("  With API key:")
        print("    python scripts/data/fetch_historical_game_lines.py --week 1 --api-key YOUR_KEY")
        print()
        print("Environment:")
        print("    export ODDS_API_KEY=your_key_here")


if __name__ == "__main__":
    main()
