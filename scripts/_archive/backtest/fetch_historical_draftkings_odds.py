#!/usr/bin/env python3
"""
Historical DraftKings Odds Fetcher for NFL 2024-2025 Season

Fetches historical odds snapshots from The Odds API for backtesting purposes.
Uses the Historical Event Odds API to get pregame odds for completed games.

IMPORTANT: Historical API costs 10 credits per market per region, so this is expensive.
We'll fetch odds snapshots taken ~1-2 hours before kickoff for each game.

API Endpoints Used:
- GET /v4/historical/sports/{sport}/events - Get historical events (FREE)
- GET /v4/historical/sports/{sport}/events/{eventId}/odds - Get historical odds for specific event
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

API_KEY = os.getenv('ODDS_API_KEY')
if not API_KEY:
    raise ValueError("ODDS_API_KEY not found in .env file")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# NFL 2024-2025 Season weeks and their date ranges
# Each week tuple: (week_num, start_date, end_date)
NFL_2024_WEEKS = [
    (1, "2024-09-05", "2024-09-09"),
    (2, "2024-09-12", "2024-09-16"),
    (3, "2024-09-19", "2024-09-23"),
    (4, "2024-09-26", "2024-09-30"),
    (5, "2024-10-03", "2024-10-07"),
    (6, "2024-10-10", "2024-10-14"),
    (7, "2024-10-17", "2024-10-21"),
    (8, "2024-10-24", "2024-10-28"),
    (9, "2024-10-31", "2024-11-04"),
    (10, "2024-11-07", "2024-11-11"),
    (11, "2024-11-14", "2024-11-18"),
    (12, "2024-11-21", "2024-11-25"),
    (13, "2024-11-28", "2024-12-02"),
    (14, "2024-12-05", "2024-12-09"),
    (15, "2024-12-12", "2024-12-16"),
    (16, "2024-12-19", "2024-12-23"),
    (17, "2024-12-25", "2024-12-29"),
    (18, "2025-01-04", "2025-01-05"),
]

# 2025 Season
NFL_2025_WEEKS = [
    (1, "2025-09-04", "2025-09-08"),
    (2, "2025-09-11", "2025-09-15"),
    (3, "2025-09-18", "2025-09-22"),
    (4, "2025-09-25", "2025-09-29"),
    (5, "2025-10-02", "2025-10-06"),
    (6, "2025-10-09", "2025-10-13"),
    (7, "2025-10-16", "2025-10-20"),
    (8, "2025-10-23", "2025-10-27"),
    (9, "2025-10-30", "2025-11-03"),
    (10, "2025-11-06", "2025-11-10"),
    (11, "2025-11-13", "2025-11-17"),
]


class QuotaTracker:
    """Track API quota usage with budget limits"""
    def __init__(self, budget_limit: int = 5000):
        self.total_used = 0
        self.session_used = 0
        self.remaining = None
        self.budget_limit = budget_limit
        self.calls = []

    def update(self, response, cost_estimate: int = 0):
        """Update quota from response headers"""
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        last = response.headers.get('x-requests-last')

        actual_cost = int(float(last)) if last else cost_estimate

        self.remaining = int(float(remaining)) if remaining else None
        self.total_used = int(float(used)) if used else self.total_used
        self.session_used += actual_cost

        self.calls.append({
            'cost': actual_cost,
            'remaining': self.remaining,
            'timestamp': datetime.now()
        })

        print(f"  💰 Cost: {actual_cost} | Session Total: {self.session_used} | Remaining: {remaining}")

        # Check budget
        if self.session_used >= self.budget_limit:
            raise BudgetExceededError(f"Session budget of {self.budget_limit} credits exceeded!")

    def can_afford(self, estimated_cost: int) -> bool:
        """Check if we can afford this call within budget"""
        return (self.session_used + estimated_cost) <= self.budget_limit

    def summary(self):
        """Print quota summary"""
        print("\n" + "="*80)
        print("📊 QUOTA SUMMARY")
        print("="*80)
        print(f"Total API calls: {len(self.calls)}")
        print(f"Session credits used: {self.session_used}")
        print(f"Budget limit: {self.budget_limit}")
        print(f"Remaining in budget: {self.budget_limit - self.session_used}")
        print(f"API credits remaining: {self.remaining}")
        if self.calls:
            print(f"Average cost per call: {sum(c['cost'] for c in self.calls) / len(self.calls):.1f}")
        print("="*80)


class BudgetExceededError(Exception):
    pass


def convert_team_name(full_name: str) -> str:
    """Convert full team name to abbreviation"""
    team_map = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }
    return team_map.get(full_name, full_name[:3].upper())


def fetch_historical_events(date_iso: str) -> list:
    """
    Fetch historical events for a specific date
    Cost: FREE (doesn't count against quota)

    Args:
        date_iso: ISO8601 timestamp like "2024-10-20T12:00:00Z"
    """
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"

    params = {
        'apiKey': API_KEY,
        'date': date_iso,
        'dateFormat': 'iso'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        # Response has wrapper with timestamp and data
        if isinstance(data, dict) and 'data' in data:
            events = data['data']
            actual_timestamp = data.get('timestamp', date_iso)
            print(f"  📅 Snapshot timestamp: {actual_timestamp}")
            return events
        else:
            return data

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error fetching historical events: {e}")
        return []


def fetch_historical_event_odds(
    event_id: str,
    date_iso: str,
    quota: QuotaTracker,
    markets: list = None
) -> dict:
    """
    Fetch historical odds for a specific event at a specific time
    Cost: 10 credits per market per region

    Args:
        event_id: API event ID
        date_iso: ISO8601 timestamp for snapshot
        quota: QuotaTracker instance
        markets: List of markets to fetch (default: player props)
    """
    if markets is None:
        # Key player prop markets
        markets = [
            'player_pass_yds',
            'player_pass_tds',
            'player_rush_yds',
            'player_receptions',
            'player_reception_yds',
            'player_anytime_td'
        ]

    # Calculate cost estimate
    estimated_cost = len(markets) * 10  # 10 per market, 1 region

    if not quota.can_afford(estimated_cost):
        print(f"  ⚠️  Cannot afford {estimated_cost} credits (budget: {quota.budget_limit - quota.session_used} remaining)")
        return None

    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"

    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
        'bookmakers': 'draftkings',
        'date': date_iso,
        'dateFormat': 'iso'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        quota.update(response, cost_estimate=estimated_cost)

        data = response.json()

        # Response has wrapper
        if isinstance(data, dict) and 'data' in data:
            actual_timestamp = data.get('timestamp', date_iso)
            return {
                'snapshot_timestamp': actual_timestamp,
                'event_data': data['data']
            }
        else:
            return {'snapshot_timestamp': date_iso, 'event_data': data}

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error fetching historical odds: {e}")
        return None


def parse_player_props_from_event(event_data: dict, snapshot_timestamp: str) -> list:
    """
    Parse player prop odds from event data

    Returns list of records with:
    - event_id, game_id, commence_time, teams
    - player, stat_type, line, over_odds, under_odds
    - snapshot_timestamp for when odds were captured
    """
    records = []

    if not event_data:
        return records

    # Parse event info
    event_id = event_data.get('id', '')
    home_team = event_data.get('home_team', '')
    away_team = event_data.get('away_team', '')
    commence_time = event_data.get('commence_time', '')

    home_abbr = convert_team_name(home_team)
    away_abbr = convert_team_name(away_team)

    # Extract date from commence_time to create game_id
    try:
        dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        date_str = dt.strftime('%Y%m%d')
        game_id = f"{date_str}_{away_abbr}_{home_abbr}"
    except:
        game_id = f"UNKNOWN_{away_abbr}_{home_abbr}"

    # Find DraftKings
    draftkings = None
    for bookmaker in event_data.get('bookmakers', []):
        if bookmaker['key'] == 'draftkings':
            draftkings = bookmaker
            break

    if not draftkings:
        return records

    # Map market keys to our stat types
    stat_type_map = {
        'player_pass_yds': 'passing_yards',
        'player_pass_tds': 'passing_tds',
        'player_pass_completions': 'passing_completions',
        'player_rush_yds': 'rushing_yards',
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_anytime_td': 'anytime_td',
        'player_1st_td': 'first_td'
    }

    # Parse markets
    for market in draftkings.get('markets', []):
        market_key = market['key']
        stat_type = stat_type_map.get(market_key)

        if not stat_type:
            continue

        market_last_update = market.get('last_update', snapshot_timestamp)

        # Handle TD props (binary) vs stat props (over/under)
        is_td_market = 'td' in market_key

        if is_td_market:
            # TD props: each outcome is a player with their odds
            for outcome in market['outcomes']:
                player_name = outcome.get('description') or outcome.get('name', 'Unknown')

                records.append({
                    'event_id': event_id,
                    'game_id': game_id,
                    'commence_time': commence_time,
                    'away_team': away_abbr,
                    'home_team': home_abbr,
                    'bookmaker_key': 'draftkings',
                    'bookmaker_title': 'DraftKings',
                    'market': market_key,
                    'player': player_name,
                    'prop_type': 'yes',
                    'line': None,
                    'price': outcome['price'],
                    'american_price': outcome['price'],
                    'decimal_price': None,
                    'snapshot_timestamp': snapshot_timestamp,
                    'bookmaker_last_update': draftkings.get('last_update', ''),
                    'market_last_update': market_last_update,
                    'retrieved_at': datetime.now(timezone.utc).isoformat()
                })
        else:
            # Over/Under props: group by player
            player_props = {}
            for outcome in market['outcomes']:
                player_name = outcome.get('description', 'Unknown')
                direction = outcome['name'].lower()  # 'over' or 'under'

                if player_name not in player_props:
                    player_props[player_name] = {'line': outcome.get('point', 0)}

                player_props[player_name][direction] = outcome['price']

            # Create records for each player
            for player_name, props in player_props.items():
                if 'over' in props and 'under' in props:
                    # Create separate over and under records
                    for direction in ['over', 'under']:
                        records.append({
                            'event_id': event_id,
                            'game_id': game_id,
                            'commence_time': commence_time,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'bookmaker_key': 'draftkings',
                            'bookmaker_title': 'DraftKings',
                            'market': market_key,
                            'player': player_name,
                            'prop_type': direction,
                            'line': props['line'],
                            'price': props[direction],
                            'american_price': props[direction],
                            'decimal_price': None,
                            'snapshot_timestamp': snapshot_timestamp,
                            'bookmaker_last_update': draftkings.get('last_update', ''),
                            'market_last_update': market_last_update,
                            'retrieved_at': datetime.now(timezone.utc).isoformat()
                        })

    return records


def fetch_week_historical_odds(
    week_num: int,
    week_start: str,
    week_end: str,
    quota: QuotaTracker,
    season: int = 2024,
    hours_before_kickoff: int = 2
) -> pd.DataFrame:
    """
    Fetch historical odds for all games in a week

    Strategy:
    1. Get list of events from middle of week
    2. For each event, fetch odds snapshot ~2 hours before kickoff

    Args:
        week_num: NFL week number
        week_start: Start date "YYYY-MM-DD"
        week_end: End date "YYYY-MM-DD"
        quota: QuotaTracker instance
        season: NFL season year
        hours_before_kickoff: How many hours before kickoff to fetch snapshot
    """
    print(f"\n{'='*80}")
    print(f"📅 FETCHING WEEK {week_num} ({season} Season)")
    print(f"   Date range: {week_start} to {week_end}")
    print(f"{'='*80}")

    # Step 1: Get events from middle of the week
    # Use the end of the week to ensure we capture all games
    events_snapshot_date = f"{week_end}T23:59:59Z"

    print(f"\n📋 Fetching events for week {week_num}...")
    events = fetch_historical_events(events_snapshot_date)

    if not events:
        print(f"  ⚠️  No events found for week {week_num}")
        return pd.DataFrame()

    # Filter events to those in our date range
    week_start_dt = datetime.fromisoformat(f"{week_start}T00:00:00+00:00")
    week_end_dt = datetime.fromisoformat(f"{week_end}T23:59:59+00:00")

    week_events = []
    for event in events:
        try:
            commence_dt = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            if week_start_dt <= commence_dt <= week_end_dt:
                week_events.append(event)
        except:
            continue

    print(f"  ✅ Found {len(week_events)} games in week {week_num}")

    if not week_events:
        return pd.DataFrame()

    # Step 2: Fetch odds for each event
    all_props = []

    for i, event in enumerate(week_events, 1):
        event_id = event['id']
        home_team = event['home_team']
        away_team = event['away_team']
        commence_time = event['commence_time']

        print(f"\n  [{i}/{len(week_events)}] {away_team} @ {home_team}")
        print(f"       Kickoff: {commence_time}")

        # Calculate snapshot time (X hours before kickoff)
        try:
            commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            snapshot_dt = commence_dt - timedelta(hours=hours_before_kickoff)
            snapshot_iso = snapshot_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except:
            print(f"       ⚠️  Could not parse commence time, skipping")
            continue

        print(f"       Fetching odds snapshot from: {snapshot_iso}")

        # Fetch historical odds
        odds_data = fetch_historical_event_odds(event_id, snapshot_iso, quota)

        if not odds_data:
            print(f"       ⚠️  No odds data returned")
            continue

        # Parse player props
        props = parse_player_props_from_event(
            odds_data['event_data'],
            odds_data['snapshot_timestamp']
        )

        print(f"       ✅ Parsed {len(props)} prop records")
        all_props.extend(props)

        # Rate limiting
        if i < len(week_events):
            time.sleep(1)  # Be respectful to the API

    # Convert to DataFrame
    if all_props:
        df = pd.DataFrame(all_props)
        print(f"\n  📊 Week {week_num} Summary:")
        print(f"     Total prop records: {len(df)}")
        print(f"     Unique players: {df['player'].nunique()}")
        print(f"     Markets: {df['market'].unique().tolist()}")
        return df
    else:
        return pd.DataFrame()


def fetch_all_historical_odds(
    season: int = 2024,
    start_week: int = 1,
    end_week: int = 18,
    budget_limit: int = 5000,
    hours_before_kickoff: int = 2,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Fetch historical odds for multiple weeks

    Args:
        season: NFL season (2024 or 2025)
        start_week: First week to fetch
        end_week: Last week to fetch
        budget_limit: Maximum API credits to use
        hours_before_kickoff: Hours before game to fetch snapshot
        output_dir: Directory to save output
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'historical'
    output_dir.mkdir(parents=True, exist_ok=True)

    quota = QuotaTracker(budget_limit=budget_limit)

    # Select weeks based on season
    if season == 2024:
        weeks = NFL_2024_WEEKS
    else:
        weeks = NFL_2025_WEEKS

    print("\n" + "="*80)
    print(f"🏈 NFL HISTORICAL ODDS FETCHER - {season} Season")
    print("="*80)
    print(f"Weeks: {start_week} to {end_week}")
    print(f"Budget: {budget_limit} credits")
    print(f"Snapshot: {hours_before_kickoff} hours before kickoff")
    print(f"Output: {output_dir}")
    print("="*80)

    all_data = []

    for week_num, week_start, week_end in weeks:
        if week_num < start_week or week_num > end_week:
            continue

        try:
            week_df = fetch_week_historical_odds(
                week_num,
                week_start,
                week_end,
                quota,
                season,
                hours_before_kickoff
            )

            if not week_df.empty:
                # Add week number
                week_df['season'] = season
                week_df['week'] = week_num
                all_data.append(week_df)

                # Save intermediate results
                week_file = output_dir / f'historical_odds_{season}_week{week_num:02d}.csv'
                week_df.to_csv(week_file, index=False)
                print(f"\n  💾 Saved: {week_file}")

        except BudgetExceededError as e:
            print(f"\n❌ {e}")
            print("Stopping to preserve budget.")
            break
        except Exception as e:
            print(f"\n❌ Error processing week {week_num}: {e}")
            continue

    # Combine all weeks
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = output_dir / f'historical_odds_{season}_weeks{start_week}-{end_week}.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"\n💾 Saved combined data: {combined_file}")

        # Print final summary
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY")
        print("="*80)
        print(f"Season: {season}")
        print(f"Weeks fetched: {combined_df['week'].unique().tolist()}")
        print(f"Total games: {combined_df['game_id'].nunique()}")
        print(f"Total prop records: {len(combined_df)}")
        print(f"Unique players: {combined_df['player'].nunique()}")
        print(f"\nMarkets breakdown:")
        for market in combined_df['market'].unique():
            count = len(combined_df[combined_df['market'] == market])
            print(f"  - {market}: {count} records")

        quota.summary()

        return combined_df
    else:
        print("\n⚠️  No data fetched")
        quota.summary()
        return pd.DataFrame()


def estimate_cost(num_weeks: int = 18, games_per_week: int = 16, markets: int = 6) -> int:
    """
    Estimate API cost for fetching historical odds

    Cost formula: 10 credits per market per region per event
    """
    cost_per_game = markets * 10  # 10 credits per market, 1 region
    total_cost = num_weeks * games_per_week * cost_per_game
    return total_cost


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch historical DraftKings odds')
    parser.add_argument('--season', type=int, default=2024, help='NFL season year')
    parser.add_argument('--start-week', type=int, default=1, help='First week to fetch')
    parser.add_argument('--end-week', type=int, default=18, help='Last week to fetch')
    parser.add_argument('--budget', type=int, default=5000, help='Max API credits to use')
    parser.add_argument('--hours-before', type=int, default=2, help='Hours before kickoff for snapshot')
    parser.add_argument('--estimate-only', action='store_true', help='Only estimate cost, do not fetch')

    args = parser.parse_args()

    if args.estimate_only:
        weeks_to_fetch = args.end_week - args.start_week + 1
        estimated = estimate_cost(weeks_to_fetch, games_per_week=16, markets=6)
        print(f"\n📊 COST ESTIMATE")
        print(f"Weeks: {weeks_to_fetch}")
        print(f"Games per week: ~16")
        print(f"Markets per game: 6")
        print(f"Cost per game: 60 credits (6 markets × 10 credits)")
        print(f"\nTotal estimated cost: {estimated:,} credits")
        print(f"\n⚠️  This is an upper bound. Actual cost may be lower if:")
        print(f"    - Some games don't have all markets")
        print(f"    - Bye weeks have fewer games")
        print(f"    - Empty responses don't charge")
    else:
        fetch_all_historical_odds(
            season=args.season,
            start_week=args.start_week,
            end_week=args.end_week,
            budget_limit=args.budget,
            hours_before_kickoff=args.hours_before
        )
