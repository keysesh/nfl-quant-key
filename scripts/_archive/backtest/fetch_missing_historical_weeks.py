#!/usr/bin/env python3
"""
Fetch missing historical NFL player props by week.
Uses the new API key: 73ec9367021badb173a0b68c35af818f

This script fetches historical odds week-by-week to build a complete dataset.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest' / 'historical_by_week'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# New API key provided by user
API_KEY = "73ec9367021badb173a0b68c35af818f"

# NFL 2025 Season Week Start Dates
# These are snapshot dates to fetch historical odds
NFL_2025_WEEKS = {
    1: "2025-09-07T12:00:00Z",   # Week 1: Sep 4-8
    2: "2025-09-14T12:00:00Z",   # Week 2: Sep 11-15
    3: "2025-09-21T12:00:00Z",   # Week 3: Sep 18-22
    4: "2025-09-28T12:00:00Z",   # Week 4: Sep 25-29
    5: "2025-10-05T12:00:00Z",   # Week 5: Oct 2-6
    6: "2025-10-12T12:00:00Z",   # Week 6: Oct 9-13
    7: "2025-10-19T12:00:00Z",   # Week 7: Oct 16-20
    8: "2025-10-26T12:00:00Z",   # Week 8: Oct 23-27
    9: "2025-11-02T12:00:00Z",   # Week 9: Oct 30-Nov 3
    10: "2025-11-09T12:00:00Z",  # Week 10: Nov 6-10
    11: "2025-11-16T12:00:00Z",  # Week 11: Nov 13-17
}

# Player prop markets to fetch
PROP_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
]


def fetch_historical_events(snapshot_date: str) -> list:
    """Fetch NFL events that existed at the snapshot date."""
    url = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events"
    params = {
        "apiKey": API_KEY,
        "date": snapshot_date,
    }

    print(f"  Fetching events for {snapshot_date}...")
    resp = requests.get(url, params=params, timeout=60)

    if resp.status_code == 401:
        print(f"    ERROR: Invalid API key or quota exceeded")
        return []

    if resp.status_code == 422:
        print(f"    ERROR: Historical data not available for this date")
        return []

    resp.raise_for_status()

    data = resp.json()
    events = data.get("data", data)

    # Check remaining quota
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"    API Quota: {remaining} remaining, {used} used")

    return events if isinstance(events, list) else []


def fetch_event_props(event_id: str, snapshot_date: str) -> dict:
    """Fetch player props for a single event."""
    url = f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "date": snapshot_date,
        "regions": "us",
        "markets": ",".join(PROP_MARKETS),
        "bookmakers": "draftkings",
        "oddsFormat": "american",
    }

    resp = requests.get(url, params=params, timeout=60)

    if resp.status_code != 200:
        return {}

    return resp.json()


def parse_props_from_response(event_data: dict, event_info: dict, week: int) -> list:
    """Parse player props from API response."""
    props = []

    # Handle historical API wrapper
    if "data" in event_data:
        event_data = event_data["data"]

    bookmakers = event_data.get("bookmakers", [])

    for book in bookmakers:
        if book.get("key") != "draftkings":
            continue

        for market in book.get("markets", []):
            market_key = market.get("key")
            last_update = market.get("last_update")

            for outcome in market.get("outcomes", []):
                props.append({
                    "week": week,
                    "event_id": event_info["id"],
                    "home_team": event_info["home_team"],
                    "away_team": event_info["away_team"],
                    "commence_time": event_info["commence_time"],
                    "market": market_key,
                    "player": outcome.get("description", ""),
                    "prop_type": outcome.get("name", "").lower(),
                    "line": outcome.get("point"),
                    "american_odds": outcome.get("price"),
                    "market_last_update": last_update,
                })

    return props


def fetch_week_props(week: int, snapshot_date: str) -> pd.DataFrame:
    """Fetch all player props for a single week."""
    print(f"\n{'='*60}")
    print(f"WEEK {week}")
    print(f"{'='*60}")

    # Get events for this week
    events = fetch_historical_events(snapshot_date)

    if not events:
        print(f"  No events found for week {week}")
        return pd.DataFrame()

    print(f"  Found {len(events)} events")

    all_props = []

    for i, event in enumerate(events):
        event_id = event.get("id")
        home = event.get("home_team", "?")
        away = event.get("away_team", "?")

        print(f"  [{i+1}/{len(events)}] {away} @ {home}...", end=" ")

        try:
            props_data = fetch_event_props(event_id, snapshot_date)
            props = parse_props_from_response(props_data, event, week)
            all_props.extend(props)
            print(f"{len(props)} props")
        except Exception as e:
            print(f"ERROR: {e}")

        # Rate limiting
        time.sleep(0.5)

    if not all_props:
        return pd.DataFrame()

    df = pd.DataFrame(all_props)
    print(f"\n  Total: {len(df)} props for week {week}")
    print(f"  Unique players: {df['player'].nunique()}")

    return df


def main():
    print("="*70)
    print("FETCHING HISTORICAL DRAFTKINGS PLAYER PROPS")
    print("API Key:", f"{API_KEY[:8]}...{API_KEY[-4:]}")
    print("="*70)

    # First, check API quota
    print("\nChecking API access...")
    test_events = fetch_historical_events("2025-09-07T12:00:00Z")

    if not test_events:
        print("Cannot access historical API. Check API key and quota.")
        return

    print(f"API access confirmed. Found {len(test_events)} test events.\n")

    # Fetch each week
    all_weeks = []

    for week, snapshot in NFL_2025_WEEKS.items():
        try:
            week_df = fetch_week_props(week, snapshot)

            if not week_df.empty:
                # Save individual week
                week_file = OUTPUT_DIR / f"week_{week}_props.csv"
                week_df.to_csv(week_file, index=False)
                print(f"  Saved to: {week_file}")

                all_weeks.append(week_df)

            # Wait between weeks
            time.sleep(2)

        except Exception as e:
            print(f"  ERROR fetching week {week}: {e}")
            continue

    # Combine all weeks
    if all_weeks:
        combined = pd.concat(all_weeks, ignore_index=True)
        combined_file = DATA_DIR / 'backtest' / 'all_historical_props_2025.csv'
        combined.to_csv(combined_file, index=False)

        print(f"\n{'='*70}")
        print(f"COMPLETE!")
        print(f"{'='*70}")
        print(f"Total props: {len(combined)}")
        print(f"Total weeks: {combined['week'].nunique()}")
        print(f"Props per week:")
        for week in sorted(combined['week'].unique()):
            count = len(combined[combined['week'] == week])
            print(f"  Week {week}: {count} props")
        print(f"\nSaved to: {combined_file}")
    else:
        print("\nNo data fetched. Check API quota.")


if __name__ == "__main__":
    main()
