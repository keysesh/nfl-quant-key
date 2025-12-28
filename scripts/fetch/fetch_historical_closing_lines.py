#!/usr/bin/env python3
"""
Fetch Historical Closing Lines for CLV Analysis - 2025 Season

Uses The Odds API historical endpoint to fetch player prop closing lines
(5 minutes before kickoff) for calculating Closing Line Value (CLV).

CLV is the gold standard for validating betting model edge:
- Positive CLV = Model beats the efficient closing market
- Consistent positive CLV = Real edge, not luck

API Endpoint: /v4/historical/sports/{sport}/events/{eventId}/odds
Docs: https://the-odds-api.com/liveapi/guides/v4/#historical-odds

Usage:
    # Fetch closing lines for specific weeks
    python scripts/fetch/fetch_historical_closing_lines.py --weeks 1-17

    # Dry run (preview without API calls)
    python scripts/fetch/fetch_historical_closing_lines.py --weeks 1-5 --dry-run

    # Custom offset from kickoff (default 5 minutes)
    python scripts/fetch/fetch_historical_closing_lines.py --weeks 1-5 --minutes-before 10

Cost: 10 API credits per event per market per region
      2025 season weeks 1-17: ~136 games x 6 markets = 8,160 credits
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
import json

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

# Core player prop markets for CLV tracking
CLV_MARKETS = [
    'player_pass_yds',
    'player_rush_yds',
    'player_receptions',
    'player_reception_yds',
    'player_rush_attempts',
    'player_pass_attempts',
]


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def devig_power(over_odds: float, under_odds: float) -> tuple:
    """
    De-vig using power method (more accurate than proportional).

    Returns:
        Tuple of (no_vig_over, no_vig_under) probabilities
    """
    p_over = american_to_implied(over_odds)
    p_under = american_to_implied(under_odds)
    total = p_over + p_under

    return p_over / total, p_under / total


def fetch_historical_events(api_key: str, date: str) -> tuple:
    """
    Fetch historical NFL events for a specific date.

    Args:
        api_key: The Odds API key
        date: Date in YYYY-MM-DD format

    Returns:
        Tuple of (events list, timestamp)
    """
    url = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events"

    params = {
        'apiKey': api_key,
        'date': f'{date}T12:00:00Z',
        'dateFormat': 'iso'
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  Error fetching events for {date}: {response.status_code}")
        return [], None

    data = response.json()
    timestamp = data.get('timestamp')
    return data.get('data', []), timestamp


def fetch_closing_line_odds(
    api_key: str,
    event_id: str,
    commence_time: str,
    minutes_before: int = 5
) -> dict:
    """
    Fetch player props at closing time (minutes before kickoff).

    Args:
        api_key: The Odds API key
        event_id: Event ID from the events endpoint
        commence_time: ISO format commence time
        minutes_before: Minutes before kickoff to capture (default 5)

    Returns:
        Event odds data or None
    """
    # Calculate closing time
    try:
        commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        closing_dt = commence_dt - timedelta(minutes=minutes_before)
        closing_time = closing_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"      Date parse error: {e}")
        return None

    url = f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}/odds"

    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(CLV_MARKETS),
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'date': closing_time,
        'bookmakers': 'draftkings'
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"      API Error {response.status_code}: {response.text[:200]}")
        return None

    return response.json()


def extract_closing_lines(data: dict, week: int, season: int) -> list:
    """
    Extract closing line data from API response with de-vigged probabilities.

    Returns:
        List of closing line records
    """
    lines = []

    if not data or 'data' not in data:
        return lines

    event_data = data['data']
    snapshot_timestamp = data.get('timestamp')
    bookmakers = event_data.get('bookmakers', [])

    # Find DraftKings
    dk_books = [b for b in bookmakers if b['key'] == 'draftkings']
    if not dk_books:
        return lines

    dk = dk_books[0]

    for market in dk.get('markets', []):
        market_key = market['key']
        outcomes = market.get('outcomes', [])

        # Group outcomes by player (Over/Under pairs)
        player_outcomes = {}
        for outcome in outcomes:
            player = outcome.get('description', '')
            if player not in player_outcomes:
                player_outcomes[player] = {}

            side = outcome.get('name', '').lower()  # 'Over' or 'Under'
            player_outcomes[player][side] = {
                'odds': outcome.get('price'),
                'line': outcome.get('point')
            }

        # Create records with de-vigged probabilities
        for player, sides in player_outcomes.items():
            if 'over' not in sides or 'under' not in sides:
                continue

            over_odds = sides['over']['odds']
            under_odds = sides['under']['odds']
            line = sides['over']['line']  # Same line for both

            # De-vig
            no_vig_over, no_vig_under = devig_power(over_odds, under_odds)

            record = {
                'season': season,
                'week': week,
                'event_id': event_data.get('id'),
                'home_team': event_data.get('home_team'),
                'away_team': event_data.get('away_team'),
                'commence_time': event_data.get('commence_time'),
                'closing_timestamp': snapshot_timestamp,
                'market': market_key,
                'player_name': player,
                'line': line,
                'over_odds': over_odds,
                'under_odds': under_odds,
                'no_vig_over': round(no_vig_over, 4),
                'no_vig_under': round(no_vig_under, 4),
            }
            lines.append(record)

    return lines


def fetch_week_closing_lines(
    api_key: str,
    week: int,
    season: int = 2025,
    minutes_before: int = 5,
    dry_run: bool = False
) -> pd.DataFrame:
    """
    Fetch closing lines for all games in a given week.

    Args:
        api_key: The Odds API key
        week: NFL week number
        season: NFL season year
        minutes_before: Minutes before kickoff to capture
        dry_run: If True, don't make API calls

    Returns:
        DataFrame of closing lines
    """
    print(f"\n{'='*60}")
    print(f"Week {week} ({season}) - Closing Lines ({minutes_before} min before kickoff)")
    print(f"{'='*60}")

    start_date = NFL_2025_WEEK_DATES.get(week)
    if not start_date:
        print(f"  No schedule data for week {week}")
        return pd.DataFrame()

    # Check if this week has already passed (2025 season starts Sep 4)
    today = datetime.now()
    week_start = datetime.strptime(start_date, '%Y-%m-%d')

    if week_start > today:
        print(f"  Week {week} hasn't happened yet (starts {start_date})")
        return pd.DataFrame()

    if dry_run:
        print("  [DRY RUN] Would fetch events and closing lines")
        return pd.DataFrame()

    # Fetch events for Sunday of the week (most games)
    sunday = week_start + timedelta(days=3)
    events, timestamp = fetch_historical_events(api_key, sunday.strftime('%Y-%m-%d'))
    print(f"  Found {len(events)} events (snapshot: {timestamp})")

    all_lines = []

    for event in events:
        event_id = event.get('id')
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        commence = event.get('commence_time', '')

        print(f"    {away} @ {home}")

        # Fetch closing lines
        data = fetch_closing_line_odds(api_key, event_id, commence, minutes_before)

        if data:
            lines = extract_closing_lines(data, week, season)
            all_lines.extend(lines)
            print(f"      -> {len(lines)} closing lines")
        else:
            print(f"      -> No data available")

        # Rate limiting
        time.sleep(0.5)

    if all_lines:
        df = pd.DataFrame(all_lines)
        print(f"\n  Total: {len(df)} closing lines")
        print(f"  Markets: {df['market'].value_counts().to_dict()}")
        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Fetch historical closing lines for CLV analysis (2025 season)'
    )
    parser.add_argument(
        '--weeks',
        type=str,
        required=True,
        help='Week range (e.g., 1-17 or 5-10)'
    )
    parser.add_argument(
        '--minutes-before',
        type=int,
        default=5,
        help='Minutes before kickoff to capture (default: 5)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without API calls'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path'
    )
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

    print("=" * 80)
    print("HISTORICAL CLOSING LINES FETCHER - 2025 Season CLV")
    print("=" * 80)
    print(f"Weeks: {weeks}")
    print(f"Minutes before kickoff: {args.minutes_before}")
    print(f"Markets: {CLV_MARKETS}")
    print(f"Dry run: {args.dry_run}")

    # Fetch closing lines for each week
    all_dfs = []

    for week in weeks:
        df = fetch_week_closing_lines(
            api_key,
            week,
            season=2025,
            minutes_before=args.minutes_before,
            dry_run=args.dry_run
        )
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo closing lines fetched")
        if not args.dry_run:
            print("(2025 season may not have started yet)")
        return

    # Combine all weeks
    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"TOTAL: {len(combined)} closing lines fetched")
    print(f"{'='*80}")

    # Show summary
    print("\nMarket breakdown:")
    print(combined['market'].value_counts().to_string())

    print("\nWeek breakdown:")
    print(combined.groupby('week').size().to_string())

    # Save to file
    output_path = args.output or PROJECT_ROOT / 'data' / 'odds' / 'historical_closing_lines_2025.csv'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Show sample
    print("\nSample closing lines:")
    sample_cols = ['week', 'player_name', 'market', 'line', 'no_vig_over', 'no_vig_under']
    print(combined[sample_cols].head(10).to_string())

    print("\n" + "=" * 80)
    print("Done! Check API usage at: https://the-odds-api.com/account/")
    print("=" * 80)


if __name__ == '__main__':
    main()
