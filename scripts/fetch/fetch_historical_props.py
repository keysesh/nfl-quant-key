#!/usr/bin/env python3
"""
Fetch Historical NFL Player Props from The Odds API

Uses the historical odds endpoint to backfill training data for additional markets:
- player_rush_attempts
- player_pass_completions
- player_pass_attempts
- player_pass_tds
- player_rush_tds
- player_reception_tds

API Endpoint: /v4/historical/sports/{sport}/events/{eventId}/odds
Docs: https://the-odds-api.com/liveapi/guides/v4/#historical-odds

Usage:
    python scripts/fetch/fetch_historical_props.py --weeks 1-13 --season 2025
    python scripts/fetch/fetch_historical_props.py --weeks 5-10 --season 2025 --dry-run
"""

import os
import sys
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent

# NFL 2025 Schedule - Week start dates (Thursday of each week)
NFL_2025_WEEK_DATES = {
    1: '2025-09-04',
    2: '2025-09-11',
    3: '2025-09-18',
    4: '2025-09-25',
    5: '2025-10-02',
    6: '2025-10-09',
    7: '2025-10-16',
    8: '2025-10-23',
    9: '2025-10-30',
    10: '2025-11-06',
    11: '2025-11-13',
    12: '2025-11-20',
    13: '2025-11-27',
    14: '2025-12-04',
    15: '2025-12-11',
    16: '2025-12-18',
    17: '2025-12-25',
    18: '2026-01-01',
}

# All player prop markets to fetch
ALL_PROP_MARKETS = [
    # Core markets (already have historical)
    'player_pass_yds',
    'player_rush_yds',
    'player_receptions',
    'player_reception_yds',
    # Additional markets (need historical)
    'player_pass_tds',
    'player_pass_attempts',
    'player_pass_completions',
    'player_pass_interceptions',
    'player_rush_attempts',
    'player_rush_tds',
    'player_reception_tds',
    # Combo markets
    'player_anytime_td',
    'player_1st_td',
]


def fetch_historical_events(api_key: str, date: str) -> list:
    """
    Fetch historical NFL events for a specific date.

    Args:
        api_key: The Odds API key
        date: Date in YYYY-MM-DD format

    Returns:
        List of events with timestamp info
    """
    url = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events"

    params = {
        'apiKey': api_key,
        'date': f'{date}T12:00:00Z',  # Noon UTC on that date
        'dateFormat': 'iso'
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  Error fetching events for {date}: {response.status_code}")
        return [], None

    data = response.json()
    timestamp = data.get('timestamp')
    return data.get('data', []), timestamp


def fetch_historical_event_odds(api_key: str, event_id: str, date: str, markets: list) -> dict:
    """
    Fetch historical player props for a specific event.

    Args:
        api_key: The Odds API key
        event_id: Event ID from the events endpoint
        date: Date in ISO format for the historical snapshot
        markets: List of market keys to fetch

    Returns:
        Event odds data or None
    """
    url = f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}/odds"

    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'date': date,
        'bookmakers': 'draftkings'  # Only DraftKings
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"      API Error {response.status_code}: {response.text[:200]}")
        return None

    return response.json()


def get_week_dates(week: int) -> tuple:
    """Get the start and end dates for a given NFL week."""
    start_date = NFL_2025_WEEK_DATES.get(week)
    if not start_date:
        return None, None

    # Week runs Thursday through Monday (5 days)
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days=4)

    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def extract_props_from_response(data: dict, week: int, season: int) -> list:
    """Extract prop data from API response."""
    props = []

    if not data or 'data' not in data:
        return props

    event_data = data['data']
    bookmakers = event_data.get('bookmakers', [])

    # Find DraftKings
    dk_books = [b for b in bookmakers if b['key'] == 'draftkings']
    if not dk_books:
        return props

    dk = dk_books[0]

    for market in dk.get('markets', []):
        market_key = market['key']

        for outcome in market.get('outcomes', []):
            prop = {
                'season': season,
                'week': week,
                'event_id': event_data.get('id'),
                'home_team': event_data.get('home_team'),
                'away_team': event_data.get('away_team'),
                'commence_time': event_data.get('commence_time'),
                'market': market_key,
                'player': outcome.get('description', ''),
                'outcome_type': outcome.get('name'),  # Over/Under
                'line': outcome.get('point'),
                'odds': outcome.get('price'),
            }
            props.append(prop)

    return props


def fetch_week_props(api_key: str, week: int, season: int = 2025, dry_run: bool = False) -> pd.DataFrame:
    """
    Fetch all player props for a given week.

    Args:
        api_key: The Odds API key
        week: NFL week number
        season: NFL season year
        dry_run: If True, don't make API calls

    Returns:
        DataFrame of props
    """
    print(f"\n{'='*60}")
    print(f"Week {week} ({season})")
    print(f"{'='*60}")

    start_date, end_date = get_week_dates(week)
    if not start_date:
        print(f"  No schedule data for week {week}")
        return pd.DataFrame()

    print(f"  Date range: {start_date} to {end_date}")

    if dry_run:
        print("  [DRY RUN] Would fetch events and props")
        return pd.DataFrame()

    # Fetch events for the week (use Sunday as reference)
    sunday = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=3)
    sunday_str = sunday.strftime('%Y-%m-%dT18:00:00Z')

    events, timestamp = fetch_historical_events(api_key, sunday_str.split('T')[0])
    print(f"  Found {len(events)} events (snapshot: {timestamp})")

    all_props = []

    for event in events:
        event_id = event.get('id')
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        commence = event.get('commence_time', '')

        print(f"    {away} @ {home}")

        # Fetch props for this event using historical EVENT ODDS endpoint
        # Use commence time minus 2 hours for "closing" lines
        try:
            commence_dt = datetime.fromisoformat(commence.replace('Z', '+00:00'))
            snapshot_dt = commence_dt - timedelta(hours=2)
            # Format as YYYY-MM-DDTHH:MM:SSZ (API requires Z format, not +00:00)
            snapshot_time = snapshot_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception as e:
            print(f"      Date parse error: {e}")
            snapshot_time = sunday_str

        data = fetch_historical_event_odds(api_key, event_id, snapshot_time, ALL_PROP_MARKETS)

        if data:
            props = extract_props_from_response(data, week, season)
            all_props.extend(props)
            print(f"      -> {len(props)} prop lines")
        else:
            print(f"      -> No props available")

        # Rate limiting - important for historical API
        time.sleep(0.5)

    if all_props:
        df = pd.DataFrame(all_props)
        print(f"\n  Total props: {len(df)}")
        print(f"  Markets: {df['market'].value_counts().to_dict()}")
        return df

    return pd.DataFrame()


def match_with_actuals(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match props with actual results from NFLverse data.

    Args:
        props_df: DataFrame of historical props

    Returns:
        DataFrame with actuals and under_hit columns
    """
    print("\nMatching with NFLverse actuals...")

    # Load NFLverse stats
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv'
    if not stats_path.exists():
        print(f"  Warning: {stats_path} not found")
        return props_df

    stats = pd.read_csv(stats_path, low_memory=False)

    # Normalize player names
    from nfl_quant.utils.player_names import normalize_player_name
    props_df['player_norm'] = props_df['player'].apply(normalize_player_name)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

    # Map market to stat column
    market_to_stat = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_completions': 'completions',
        'player_pass_attempts': 'attempts',
        'player_pass_tds': 'passing_tds',
        'player_rush_tds': 'rushing_tds',
        'player_reception_tds': 'receiving_tds',
    }

    # Merge actuals
    results = []

    for _, prop in props_df.iterrows():
        market = prop['market']
        stat_col = market_to_stat.get(market)

        if not stat_col:
            continue

        # Find matching stat
        player_stats = stats[
            (stats['player_norm'] == prop['player_norm']) &
            (stats['season'] == prop['season']) &
            (stats['week'] == prop['week'])
        ]

        if len(player_stats) == 0:
            continue

        actual = player_stats[stat_col].iloc[0] if stat_col in player_stats.columns else None

        if actual is None or pd.isna(actual):
            continue

        result = prop.to_dict()
        result['actual'] = actual
        result['under_hit'] = 1 if actual < prop['line'] else 0
        results.append(result)

    if results:
        df = pd.DataFrame(results)
        print(f"  Matched {len(df)} props with actuals")
        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Fetch historical NFL player props')
    parser.add_argument('--weeks', type=str, required=True, help='Week range (e.g., 1-13 or 5-10)')
    parser.add_argument('--season', type=int, default=2025, help='NFL season year')
    parser.add_argument('--dry-run', action='store_true', help='Preview without API calls')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    api_key = os.getenv('ODDS_API_KEY')
    if not api_key and not args.dry_run:
        print("Error: ODDS_API_KEY not found in .env file")
        sys.exit(1)

    # Parse week range
    if '-' in args.weeks:
        start_week, end_week = map(int, args.weeks.split('-'))
        weeks = list(range(start_week, end_week + 1))
    else:
        weeks = [int(args.weeks)]

    print("="*80)
    print("HISTORICAL PROPS FETCHER")
    print("="*80)
    print(f"Season: {args.season}")
    print(f"Weeks: {weeks}")
    print(f"Dry run: {args.dry_run}")

    # Fetch props for each week
    all_dfs = []

    for week in weeks:
        df = fetch_week_props(api_key, week, args.season, args.dry_run)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo props fetched")
        return

    # Combine all weeks
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"TOTAL: {len(combined)} prop lines fetched")
    print(f"{'='*80}")

    # Show market breakdown
    print("\nMarket breakdown:")
    print(combined['market'].value_counts().to_string())

    # Match with actuals
    if not args.dry_run:
        matched = match_with_actuals(combined)

        if len(matched) > 0:
            # Save to file
            output_path = args.output or PROJECT_ROOT / 'data' / 'backtest' / f'historical_props_{args.season}_expanded.csv'
            matched.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")

            # Show sample
            print("\nSample matched props:")
            print(matched.head(10).to_string())

    print("\n" + "="*80)
    print("Done! Check API usage at: https://the-odds-api.com/account/")
    print("="*80)


if __name__ == '__main__':
    main()
