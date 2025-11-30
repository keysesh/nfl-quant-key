#!/usr/bin/env python3
"""
Fetch NFL player props from The Odds API using the event-specific endpoint.
Uses the correct API structure: /v4/sports/{sport}/events/{eventId}/odds
"""

import os
import sys
import requests
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_nfl_events(api_key):
    """
    Step 1: Get list of upcoming NFL events with their IDs
    """
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"

    params = {
        'apiKey': api_key,
        'dateFormat': 'iso'
    }

    print(f"Fetching NFL events...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Error fetching events: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    events = response.json()
    print(f"✅ Found {len(events)} NFL events")

    return events


def fetch_event_player_props(event_id, api_key, markets):
    """
    Step 2: Fetch player props for a specific event
    """
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"

    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None, response.status_code, response.text

    return response.json(), 200, None


def main():
    api_key = os.getenv('ODDS_API_KEY')

    if not api_key:
        print("❌ Error: ODDS_API_KEY not found in .env file")
        sys.exit(1)

    print("=" * 80)
    print("NFL PLAYER PROPS FETCHER")
    print("=" * 80)

    # Define ALL available player prop markets (comprehensive list)
    prop_markets = [
        # PASSING - All available passing markets
        'player_pass_yds',
        'player_pass_tds',
        'player_pass_attempts',
        'player_pass_completions',
        'player_pass_interceptions',
        'player_pass_longest_completion',

        # RUSHING - All available rushing markets
        'player_rush_yds',
        'player_rush_attempts',
        'player_rush_tds',
        'player_rush_longest',

        # RECEIVING - All available receiving markets
        'player_receptions',
        'player_reception_yds',
        'player_reception_tds',
        'player_reception_longest',

        # SCORING - All touchdown and scoring markets
        'player_anytime_td',
        'player_1st_td',
        'player_last_td',
        # Note: player_2+_td and player_3+_td are not valid markets (causes 422 errors)

        # COMBO STATS - Combined rushing + receiving
        'player_rush_reception_yds',
        'player_rush_reception_tds',
        'player_pass_rush_reception_yds',
        'player_pass_rush_reception_tds',

        # KICKING - Field goals and PATs
        'player_field_goals',
        'player_kicking_points',
        'player_pats',

        # DEFENSE - Tackles, sacks, interceptions
        'player_sacks',
        'player_solo_tackles',
        'player_tackles_assists',
        'player_defensive_interceptions',

        # ALTERNATE LINES (if supported by API)
        # Note: These may need to be requested as separate markets
    ]

    print(f"\nTarget markets: {', '.join(prop_markets)}")
    print()

    # Step 1: Get events
    events = fetch_nfl_events(api_key)

    if not events:
        print("❌ Failed to fetch events")
        sys.exit(1)

    # Step 2: Fetch player props for each event
    all_props = []
    errors = []

    for i, event in enumerate(events, 1):  # Fetch ALL events
        event_id = event['id']
        home_team = event['home_team']
        away_team = event['away_team']
        commence_time = event['commence_time']

        print(f"\n[{i}/{len(events)}] {away_team} @ {home_team}")
        print(f"Event ID: {event_id}")
        print(f"Commence: {commence_time}")

        data, status_code, error_msg = fetch_event_player_props(event_id, api_key, prop_markets)

        if status_code != 200:
            error_info = {
                'event': f"{away_team} @ {home_team}",
                'event_id': event_id,
                'status_code': status_code,
                'error': error_msg
            }
            errors.append(error_info)
            print(f"  ❌ Error {status_code}: {error_msg[:100]}")
            continue

        # Extract DraftKings props
        bookmakers = data.get('bookmakers', [])
        dk_books = [b for b in bookmakers if b['key'] == 'draftkings']

        if not dk_books:
            print(f"  ⚠️  No DraftKings data available")
            continue

        dk = dk_books[0]
        markets = dk.get('markets', [])

        if not markets:
            print(f"  ⚠️  No player prop markets available")
            continue

        print(f"  ✅ Found {len(markets)} markets")

        # Process each market
        for market in markets:
            market_key = market['key']
            outcomes = market.get('outcomes', [])

            print(f"     - {market_key}: {len(outcomes)} outcomes")

            for outcome in outcomes:
                # Calculate metadata for odds filtering
                from datetime import datetime, timezone
                fetch_timestamp = datetime.now(timezone.utc).isoformat()

                try:
                    commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    current_time = datetime.now(timezone.utc)
                    minutes_to_kickoff = (commence_dt - current_time).total_seconds() / 60
                except:
                    minutes_to_kickoff = None

                prop_data = {
                    'event_id': event_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': commence_time,
                    'fetch_timestamp': fetch_timestamp,
                    'minutes_to_kickoff': minutes_to_kickoff,
                    'market': market_key,
                    'player_name': outcome.get('description', 'N/A'),
                    'outcome_type': outcome.get('name'),  # Over/Under or Yes/No
                    'line': outcome.get('point', None),
                    'odds': outcome.get('price'),
                    'last_update': market.get('last_update')
                }
                all_props.append(prop_data)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_props:
        df = pd.DataFrame(all_props)

        print(f"\n✅ Successfully fetched {len(all_props)} player prop lines")
        print(f"   Unique players: {df['player_name'].nunique()}")
        print(f"   Markets: {df['market'].nunique()}")

        # Save to CSV
        output_file = 'data/nfl_player_props_draftkings.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")

        # Show sample
        print("\n📊 Sample Props:")
        print(df.head(10).to_string(index=False))

        # Market breakdown
        print("\n📈 Market Breakdown:")
        market_counts = df.groupby('market').size().sort_values(ascending=False)
        for market, count in market_counts.items():
            print(f"   {market}: {count} lines")

    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)}")
        for err in errors[:3]:  # Show first 3 errors
            print(f"   - {err['event']}: Status {err['status_code']}")

    # Check API usage
    print("\n" + "=" * 80)
    print("Done! Check your API usage at: https://the-odds-api.com/account/")
    print("=" * 80)


if __name__ == "__main__":
    main()
