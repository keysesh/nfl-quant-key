"""
Comprehensive NFL Odds Fetcher - Game Lines + Player Props
Efficiently fetches all odds data needed for the pipeline with quota tracking
"""
import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time

# Load API key from .env
load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

if not API_KEY:
    raise ValueError("ODDS_API_KEY not found in .env file")

BASE_URL = "https://api.the-odds-api.com/v4"

class QuotaTracker:
    """Track API quota usage"""
    def __init__(self):
        self.total_used = 0
        self.remaining = None
        self.calls = []

    def update(self, response, cost_estimate):
        """Update quota from response headers"""
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        last = response.headers.get('x-requests-last')

        self.remaining = int(float(remaining)) if remaining else None
        self.total_used = int(float(used)) if used else self.total_used

        self.calls.append({
            'cost': int(float(last)) if last else cost_estimate,
            'remaining': self.remaining,
            'timestamp': datetime.now()
        })

        print(f"  💰 Quota - Used: {last or cost_estimate} | Total Used: {used} | Remaining: {remaining}")

    def summary(self):
        """Print quota summary"""
        print("\n" + "="*80)
        print("📊 QUOTA SUMMARY")
        print("="*80)
        print(f"Total API calls: {len(self.calls)}")
        print(f"Total credits used: {self.total_used}")
        print(f"Remaining credits: {self.remaining}")
        print(f"Average cost per call: {sum(c['cost'] for c in self.calls) / len(self.calls):.1f}")
        print("="*80)

quota = QuotaTracker()


def fetch_game_lines(week: int = None):
    """
    Fetch game lines (spreads, totals, moneylines) for upcoming NFL games
    Cost: 3 credits (3 markets × 1 region)
    """
    print("\n🏈 FETCHING GAME LINES (Spreads, Totals, Moneylines)")
    print("-" * 80)

    url = f"{BASE_URL}/sports/americanfootball_nfl/odds"

    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',  # 3 markets
        'oddsFormat': 'american',
        'bookmakers': 'draftkings'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        quota.update(response, cost_estimate=3)

        games = response.json()

        if not games:
            print("  ⚠️  No upcoming games found")
            return pd.DataFrame()

        print(f"  ✅ Found {len(games)} games")

        # Parse game lines
        odds_data = []

        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            game_id = game['id']
            commence_time = game['commence_time']

            # Convert to abbreviations
            home_abbr = convert_team_name(home_team)
            away_abbr = convert_team_name(away_team)

            # Create our game_id format
            if week:
                our_game_id = f"2025_{week:02d}_{away_abbr}_{home_abbr}"
            else:
                our_game_id = f"2025_XX_{away_abbr}_{home_abbr}"

            # Calculate metadata for odds filtering
            from datetime import datetime, timezone
            fetch_timestamp = datetime.now(timezone.utc).isoformat()

            try:
                commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                minutes_to_kickoff = (commence_dt - current_time).total_seconds() / 60
            except:
                minutes_to_kickoff = None

            # Find DraftKings
            draftkings = None
            for bookmaker in game.get('bookmakers', []):
                if bookmaker['key'] == 'draftkings':
                    draftkings = bookmaker
                    break

            if not draftkings:
                continue

            # Extract markets
            for market in draftkings.get('markets', []):
                market_key = market['key']

                for outcome in market['outcomes']:
                    record = {
                        'api_event_id': game_id,
                        'game_id': our_game_id,
                        'away_team': away_abbr,
                        'home_team': home_abbr,
                        'commence_time': commence_time,
                        'fetch_timestamp': fetch_timestamp,
                        'minutes_to_kickoff': minutes_to_kickoff,
                        'sportsbook': 'draftkings',
                        'bet_type': 'Game Line',
                        'market': market_key,
                        'player': None,
                        'stat_type': None,
                    }

                    if market_key == 'spreads':
                        record['market_name'] = f"{'Home' if outcome['name'] == home_team else 'Away'} Spread"
                        record['line'] = outcome.get('point', 0)
                        record['market_odds'] = outcome['price']
                        record['direction'] = 'home' if outcome['name'] == home_team else 'away'

                    elif market_key == 'totals':
                        record['market_name'] = f"{outcome['name'].title()}"
                        record['line'] = outcome.get('point', 0)
                        record['market_odds'] = outcome['price']
                        record['direction'] = outcome['name'].lower()

                    elif market_key == 'h2h':
                        record['market_name'] = f"{'Home' if outcome['name'] == home_team else 'Away'} ML"
                        record['line'] = None
                        record['market_odds'] = outcome['price']
                        record['direction'] = 'home' if outcome['name'] == home_team else 'away'

                    odds_data.append(record)

        df = pd.DataFrame(odds_data)
        print(f"  ✅ Parsed {len(df)} game line records")

        return df

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()


def fetch_player_props(event_ids: list, week: int = None):
    """
    Fetch player props for specific events
    Cost: Variable depending on markets and events

    Key player prop markets for NFL:
    - player_pass_yds (QB passing yards)
    - player_pass_tds (QB passing TDs)
    - player_rush_yds (RB/QB rushing yards)
    - player_receptions (WR/TE/RB receptions)
    - player_reception_yds (WR/TE/RB receiving yards)
    """
    print("\n🎯 FETCHING PLAYER PROPS")
    print("-" * 80)

    if not event_ids:
        print("  ⚠️  No event IDs provided")
        return pd.DataFrame()

    # Key markets we care about
    prop_markets = [
        'player_pass_yds',
        'player_pass_tds',
        'player_pass_completions',
        'player_pass_interceptions',
        'player_rush_yds',
        'player_rush_attempts',
        'player_receptions',
        'player_reception_yds',
        'player_anytime_td',
        'player_1st_td',
        'player_field_goals',
        'player_kicking_points'
    ]

    all_props = []

    for i, event_id in enumerate(event_ids, 1):
        print(f"\n  [{i}/{len(event_ids)}] Fetching props for event {event_id[:8]}...")

        url = f"{BASE_URL}/sports/americanfootball_nfl/events/{event_id}/odds"

        params = {
            'apiKey': API_KEY,
            'regions': 'us',
            'markets': ','.join(prop_markets),
            'oddsFormat': 'american',
            'bookmakers': 'draftkings'
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            # Cost = markets returned × regions
            # We'll see actual cost from headers
            quota.update(response, cost_estimate=len(prop_markets))

            data = response.json()

            if not data:
                print(f"    ⚠️  No data returned")
                continue

            # Parse event info
            home_team = data['home_team']
            away_team = data['away_team']
            home_abbr = convert_team_name(home_team)
            away_abbr = convert_team_name(away_team)
            commence_time = data['commence_time']

            if week:
                our_game_id = f"2025_{week:02d}_{away_abbr}_{home_abbr}"
            else:
                our_game_id = f"2025_XX_{away_abbr}_{home_abbr}"

            # Calculate metadata for odds filtering
            from datetime import datetime, timezone
            fetch_timestamp = datetime.now(timezone.utc).isoformat()

            try:
                commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                minutes_to_kickoff = (commence_dt - current_time).total_seconds() / 60
            except:
                minutes_to_kickoff = None

            # Find DraftKings bookmaker
            draftkings = None
            for bookmaker in data.get('bookmakers', []):
                if bookmaker['key'] == 'draftkings':
                    draftkings = bookmaker
                    break

            if not draftkings:
                print(f"    ⚠️  No DraftKings odds")
                continue

            props_found = 0

            # Parse player prop markets
            for market in draftkings.get('markets', []):
                market_key = market['key']

                # Map market key to our stat types
                stat_type_map = {
                    'player_pass_yds': 'passing_yards',
                    'player_pass_tds': 'passing_tds',
                    'player_pass_completions': 'passing_completions',
                    'player_pass_interceptions': 'passing_interceptions',
                    'player_rush_yds': 'rushing_yards',
                    'player_rush_attempts': 'rushing_attempts',
                    'player_receptions': 'receptions',
                    'player_reception_yds': 'receiving_yards',
                    'player_anytime_td': 'anytime_td',
                    'player_1st_td': 'first_td',
                    'player_field_goals': 'field_goals',
                    'player_kicking_points': 'kicking_points'
                }

                stat_type = stat_type_map.get(market_key)
                if not stat_type:
                    continue

                # Handle TD props (binary) vs stat props (over/under)
                is_td_market = 'td' in market_key

                if is_td_market:
                    # TD props are binary (player name with odds)
                    for outcome in market['outcomes']:
                        player_name = outcome.get('description') or outcome.get('name', 'Unknown')
                        props_found += 1
                        all_props.append({
                            'api_event_id': event_id,
                            'game_id': our_game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'fetch_timestamp': fetch_timestamp,
                            'minutes_to_kickoff': minutes_to_kickoff,
                            'sportsbook': 'draftkings',
                            'bet_type': 'Player Prop',
                            'market': market_key,
                            'market_name': f"{player_name} {stat_type.replace('_', ' ').title()}",
                            'player': player_name,
                            'stat_type': stat_type,
                            'line': None,  # No line for TD props
                            'market_odds': outcome['price'],
                            'over_odds': None,
                            'under_odds': None,
                            'direction': 'yes'  # To score
                        })
                else:
                    # Group outcomes by player (Over/Under pairs)
                    player_props = {}
                    for outcome in market['outcomes']:
                        player_name = outcome.get('description', 'Unknown')
                        direction = outcome['name'].lower()  # 'over' or 'under'

                        if player_name not in player_props:
                            player_props[player_name] = {}

                        player_props[player_name][direction] = {
                            'line': outcome.get('point', 0),
                            'odds': outcome['price']
                        }

                    # Create records for each player
                    for player_name, directions in player_props.items():
                        if 'over' in directions and 'under' in directions:
                            # We have both sides
                            props_found += 1
                            all_props.append({
                                'api_event_id': event_id,
                                'game_id': our_game_id,
                                'away_team': away_abbr,
                                'home_team': home_abbr,
                                'commence_time': commence_time,
                                'fetch_timestamp': fetch_timestamp,
                                'minutes_to_kickoff': minutes_to_kickoff,
                                'sportsbook': 'draftkings',
                                'bet_type': 'Player Prop',
                                'market': market_key,
                                'market_name': f"{player_name} {stat_type.replace('_', ' ').title()}",
                                'player': player_name,
                                'stat_type': stat_type,
                                'line': directions['over']['line'],
                                'market_odds': None,
                                'over_odds': directions['over']['odds'],
                                'under_odds': directions['under']['odds'],
                                'direction': None  # Both available
                            })

            print(f"    ✅ Found {props_found} player props")

            # Rate limiting - be nice to the API
            if i < len(event_ids):
                time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"    ❌ Error: {e}")
            continue

    df = pd.DataFrame(all_props)
    if not df.empty:
        print(f"\n  ✅ Total player props: {len(df)}")
    else:
        print(f"\n  ⚠️  No player props fetched")

    return df


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


def main(week: int = None, fetch_props: bool = True):
    """
    Main orchestrator for fetching all odds data

    Args:
        week: NFL week number
        fetch_props: Whether to fetch player props (costs more quota)
    """
    print("\n" + "="*80)
    print("🏈 NFL COMPREHENSIVE ODDS FETCHER")
    print("="*80)
    print(f"Week: {week if week else 'Auto-detect'}")
    print(f"Fetch Player Props: {fetch_props}")
    print(f"API Key: {API_KEY[:8]}...")
    print("="*80)

    # Step 1: Fetch game lines (cheap - 3 credits)
    game_lines_df = fetch_game_lines(week)

    if game_lines_df.empty:
        print("\n❌ No game lines found. Exiting.")
        return

    # Step 2: Fetch player props (expensive - depends on number of events and markets)
    player_props_df = pd.DataFrame()

    if fetch_props:
        # Get unique event IDs from game lines
        event_ids = game_lines_df['api_event_id'].unique().tolist()
        print(f"\n📋 Found {len(event_ids)} events to fetch props for")

        # Estimate cost
        prop_markets_count = 12  # We're fetching 12 player prop markets
        estimated_cost = len(event_ids) * prop_markets_count
        print(f"⚠️  Estimated cost for props: ~{estimated_cost} credits")
        print(f"   (Actual cost may be lower if some markets have no data)")

        response = input(f"\n🤔 Continue with player props? [Y/n]: ")
        if response.lower() in ['', 'y', 'yes']:
            player_props_df = fetch_player_props(event_ids, week)
        else:
            print("⏭️  Skipping player props")

    # Step 3: Combine and save
    print("\n📦 SAVING DATA")
    print("-" * 80)

    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    # Save game lines
    if not game_lines_df.empty:
        if week:
            game_lines_file = output_dir / f'odds_week{week}_game_lines.csv'
        else:
            game_lines_file = output_dir / f'odds_game_lines_{datetime.now().strftime("%Y%m%d")}.csv'

        game_lines_df.to_csv(game_lines_file, index=False)
        print(f"  ✅ Game lines saved: {game_lines_file}")

    # Save player props
    if not player_props_df.empty:
        if week:
            props_file = output_dir / f'odds_week{week}_player_props.csv'
        else:
            props_file = output_dir / f'odds_player_props_{datetime.now().strftime("%Y%m%d")}.csv'

        player_props_df.to_csv(props_file, index=False)
        print(f"  ✅ Player props saved: {props_file}")

    # Save combined format for pipeline compatibility
    if not game_lines_df.empty or not player_props_df.empty:
        # Combine into unified format
        combined_df = pd.concat([game_lines_df, player_props_df], ignore_index=True)

        if week:
            combined_file = output_dir / f'odds_week{week}_comprehensive.csv'
        else:
            combined_file = output_dir / f'odds_comprehensive_{datetime.now().strftime("%Y%m%d")}.csv'

        combined_df.to_csv(combined_file, index=False)
        print(f"  ✅ Combined data saved: {combined_file}")

    # Print quota summary
    quota.summary()

    # Print data summary
    print("\n" + "="*80)
    print("📊 DATA SUMMARY")
    print("="*80)
    print(f"Games found: {game_lines_df['game_id'].nunique() if not game_lines_df.empty else 0}")
    print(f"Game line records: {len(game_lines_df)}")
    print(f"Player prop records: {len(player_props_df)}")
    print(f"Total records: {len(game_lines_df) + len(player_props_df)}")
    print("="*80)

    print("\n✅ SUCCESS! All data fetched and saved.")


if __name__ == '__main__':
    import sys

    # Parse arguments
    week = None
    fetch_props = True

    if len(sys.argv) > 1:
        try:
            week = int(sys.argv[1])
        except ValueError:
            print("Usage: python fetch_comprehensive_odds.py [week_number] [--no-props]")
            sys.exit(1)

    if '--no-props' in sys.argv:
        fetch_props = False

    main(week, fetch_props)
