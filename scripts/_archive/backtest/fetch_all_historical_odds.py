#!/usr/bin/env python3
"""
Fetch ALL Historical DraftKings Odds for 2024 and 2025 Seasons

Uses The Odds API Historical endpoints to get pregame odds for every game.
Supports multiple API keys to maximize available credits.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API keys
API_KEYS = [
    "73ec9367021badb173a0b68c35af818f",  # Key 2 - 93k credits
    "1fa38c2a5b8df1b50ad9be8887386f04",  # Key 1 - 3.8k credits
]

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# NFL Season schedules
NFL_WEEKS = {
    2024: [
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
    ],
    2025: [
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
    ],
}


class MultiKeyQuotaManager:
    """Manage multiple API keys and their quotas"""

    def __init__(self, api_keys):
        self.keys = api_keys
        self.current_key_idx = 0
        self.key_usage = {key: 0 for key in api_keys}
        self.key_remaining = {}
        self.total_used = 0

    def get_current_key(self):
        return self.keys[self.current_key_idx]

    def update_quota(self, response):
        key = self.get_current_key()
        remaining = float(response.headers.get('x-requests-remaining', 0))
        used = float(response.headers.get('x-requests-used', 0))
        last = float(response.headers.get('x-requests-last', 0))

        self.key_remaining[key] = remaining
        self.key_usage[key] = used
        self.total_used += last

        print(f"    [Key {self.current_key_idx+1}] Cost: {last:.0f} | Remaining: {remaining:.0f}")

        # Switch key if running low
        if remaining < 100 and self.current_key_idx < len(self.keys) - 1:
            print(f"    ⚠️  Switching to next API key...")
            self.current_key_idx += 1

    def summary(self):
        print(f"\n{'='*60}")
        print("API QUOTA SUMMARY")
        print(f"{'='*60}")
        for i, key in enumerate(self.keys):
            remaining = self.key_remaining.get(key, "Unknown")
            used = self.key_usage.get(key, 0)
            print(f"Key {i+1} ({key[:8]}...): Used {used}, Remaining {remaining}")
        print(f"Total credits used this session: {self.total_used:.0f}")
        print(f"{'='*60}")


def convert_team_name(full_name):
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


def fetch_historical_events(date_iso, api_key):
    """Get events for a historical date (FREE - no quota cost)"""
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    params = {
        'apiKey': api_key,
        'date': date_iso,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, dict) and 'data' in data:
        return data['data'], data.get('timestamp')
    return data, date_iso


def fetch_historical_event_odds(event_id, date_iso, quota_mgr, markets):
    """
    Fetch historical odds for a specific event
    Cost: 10 credits per market
    """
    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"

    params = {
        'apiKey': quota_mgr.get_current_key(),
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
        'bookmakers': 'draftkings',
        'date': date_iso,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    quota_mgr.update_quota(response)

    data = response.json()
    if isinstance(data, dict) and 'data' in data:
        return data['data'], data.get('timestamp')
    return data, date_iso


def parse_props_from_event(event_data, snapshot_timestamp, season, week):
    """Parse player props from event odds data"""
    records = []

    if not event_data:
        return records

    event_id = event_data.get('id', '')
    home_team = event_data.get('home_team', '')
    away_team = event_data.get('away_team', '')
    commence_time = event_data.get('commence_time', '')

    home_abbr = convert_team_name(home_team)
    away_abbr = convert_team_name(away_team)

    # Create game_id
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

    # Market mapping
    stat_type_map = {
        'player_pass_yds': 'passing_yards',
        'player_pass_tds': 'passing_tds',
        'player_rush_yds': 'rushing_yards',
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_anytime_td': 'anytime_td',
    }

    for market in draftkings.get('markets', []):
        market_key = market['key']
        stat_type = stat_type_map.get(market_key)
        if not stat_type:
            continue

        market_last_update = market.get('last_update', snapshot_timestamp)
        is_td_market = 'td' in market_key

        if is_td_market:
            for outcome in market['outcomes']:
                player_name = outcome.get('description') or outcome.get('name', 'Unknown')
                records.append({
                    'event_id': event_id,
                    'game_id': game_id,
                    'season': season,
                    'week': week,
                    'commence_time': commence_time,
                    'away_team': away_abbr,
                    'home_team': home_abbr,
                    'market': market_key,
                    'player': player_name,
                    'prop_type': 'yes',
                    'line': None,
                    'price': outcome['price'],
                    'snapshot_timestamp': snapshot_timestamp,
                    'market_last_update': market_last_update,
                })
        else:
            # Over/Under props
            player_props = {}
            for outcome in market['outcomes']:
                player_name = outcome.get('description', 'Unknown')
                direction = outcome['name'].lower()
                if player_name not in player_props:
                    player_props[player_name] = {'line': outcome.get('point', 0)}
                player_props[player_name][direction] = outcome['price']

            for player_name, props in player_props.items():
                if 'over' in props and 'under' in props:
                    for direction in ['over', 'under']:
                        records.append({
                            'event_id': event_id,
                            'game_id': game_id,
                            'season': season,
                            'week': week,
                            'commence_time': commence_time,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'market': market_key,
                            'player': player_name,
                            'prop_type': direction,
                            'line': props['line'],
                            'price': props[direction],
                            'snapshot_timestamp': snapshot_timestamp,
                            'market_last_update': market_last_update,
                        })

    return records


def fetch_season(season, quota_mgr, output_dir, markets, hours_before=2):
    """Fetch all historical odds for a season"""
    print(f"\n{'='*60}")
    print(f"FETCHING {season} SEASON")
    print(f"{'='*60}")

    weeks = NFL_WEEKS.get(season, [])
    all_data = []

    for week_num, week_start, week_end in weeks:
        print(f"\n📅 Week {week_num} ({week_start} to {week_end})")

        # Skip future weeks
        today = datetime.now().strftime('%Y-%m-%d')
        if week_start > today:
            print(f"  ⏭️ Skipping future week")
            continue

        # Get events using a date DURING the week (not end)
        # Use middle of week to ensure events are captured
        week_start_dt = datetime.fromisoformat(f"{week_start}T00:00:00")
        week_end_dt = datetime.fromisoformat(f"{week_end}T23:59:59")
        mid_week_dt = week_start_dt + (week_end_dt - week_start_dt) / 2
        events_date = mid_week_dt.strftime('%Y-%m-%dT12:00:00Z')

        events, _ = fetch_historical_events(events_date, quota_mgr.get_current_key())

        # Filter to this week's games
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

        print(f"  Found {len(week_events)} games")

        week_props = []
        for i, event in enumerate(week_events, 1):
            event_id = event['id']
            home = event['home_team']
            away = event['away_team']
            commence_time = event['commence_time']

            # Calculate snapshot time (hours before kickoff)
            try:
                commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                snapshot_dt = commence_dt - timedelta(hours=hours_before)
                snapshot_iso = snapshot_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                continue

            print(f"  [{i}/{len(week_events)}] {away} @ {home}")

            try:
                event_data, actual_snapshot = fetch_historical_event_odds(
                    event_id, snapshot_iso, quota_mgr, markets
                )

                props = parse_props_from_event(event_data, actual_snapshot, season, week_num)
                week_props.extend(props)
                print(f"    ✓ {len(props)} props")

            except Exception as e:
                print(f"    ✗ Error: {e}")

            time.sleep(0.3)  # Rate limiting

        # Save week data
        if week_props:
            week_df = pd.DataFrame(week_props)
            week_file = output_dir / f'historical_odds_{season}_week{week_num:02d}.csv'
            week_df.to_csv(week_file, index=False)
            print(f"  💾 Saved {len(week_df)} props to {week_file.name}")
            all_data.extend(week_props)

    # Save season data
    if all_data:
        season_df = pd.DataFrame(all_data)
        season_file = output_dir / f'historical_odds_{season}_complete.csv'
        season_df.to_csv(season_file, index=False)
        print(f"\n✅ Season {season}: {len(season_df)} total props saved")
        return season_df

    return pd.DataFrame()


def main():
    print("="*60)
    print("FETCHING ALL HISTORICAL DRAFTKINGS ODDS")
    print("2024 and 2025 NFL Seasons")
    print("="*60)

    # Initialize quota manager
    quota_mgr = MultiKeyQuotaManager(API_KEYS)

    # Output directory
    output_dir = PROJECT_ROOT / 'data' / 'historical'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markets to fetch
    markets = [
        'player_pass_yds',
        'player_rush_yds',
        'player_receptions',
        'player_reception_yds',
        'player_pass_tds',
        'player_anytime_td',
    ]

    print(f"\nMarkets: {markets}")
    print(f"Cost per game: {len(markets) * 10} credits")

    all_seasons_data = []

    # Fetch 2024 season
    df_2024 = fetch_season(2024, quota_mgr, output_dir, markets)
    if not df_2024.empty:
        all_seasons_data.append(df_2024)

    # Fetch 2025 season
    df_2025 = fetch_season(2025, quota_mgr, output_dir, markets)
    if not df_2025.empty:
        all_seasons_data.append(df_2025)

    # Combine all data
    if all_seasons_data:
        combined_df = pd.concat(all_seasons_data, ignore_index=True)
        combined_file = output_dir / 'historical_odds_2024_2025_complete.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✅ Combined dataset: {len(combined_df)} total props")
        print(f"   Saved to: {combined_file}")

        # Summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total props fetched: {len(combined_df)}")
        print(f"Unique games: {combined_df['game_id'].nunique()}")
        print(f"Seasons: {combined_df['season'].unique().tolist()}")
        for season in combined_df['season'].unique():
            season_data = combined_df[combined_df['season'] == season]
            print(f"  {season}: {len(season_data)} props, {season_data['game_id'].nunique()} games")

    quota_mgr.summary()


if __name__ == '__main__':
    main()
