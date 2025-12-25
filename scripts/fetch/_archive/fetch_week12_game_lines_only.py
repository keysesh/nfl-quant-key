"""
Fetch ONLY game lines for Week 12 (no player props)
Quick 3-credit API call to get spreads, totals, moneylines
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.data_matching import normalize_team_name

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

if not API_KEY:
    raise ValueError("ODDS_API_KEY not found in .env file")

BASE_URL = "https://api.the-odds-api.com/v4"

print("\n" + "="*80)
print("🏈 FETCH WEEK 12 GAME LINES ONLY")
print("="*80)
print(f"  API Key: {API_KEY[:12]}...")
print(f"  Estimated cost: 3 credits")
print("="*80)

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

    # Check quota
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    last = response.headers.get('x-requests-last')

    print(f"\n💰 Quota - Used: {last} | Total Used: {used} | Remaining: {remaining}")

    games = response.json()

    if not games:
        print("\n⚠️  No upcoming games found")
        sys.exit(1)

    print(f"\n✅ Found {len(games)} games")

    # Parse game lines - using nflverse format
    odds_data = []

    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        game_id = game['id']
        commence_time = game['commence_time']

        # Convert to abbreviations using data_matching utility
        try:
            home_abbr = normalize_team_name(home_team)
            away_abbr = normalize_team_name(away_team)
        except ValueError as e:
            print(f"  ⚠️  Skipping game {away_team} @ {home_team}: {e}")
            continue

        # Create game_id in nflverse format
        our_game_id = f"2025_12_{away_abbr}_{home_abbr}"

        # Metadata
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
                if market_key == 'spreads':
                    side = 'home' if outcome['name'] == home_team else 'away'
                    odds_data.append({
                        'game_id': our_game_id,
                        'away_team': away_abbr,
                        'home_team': home_abbr,
                        'commence_time': commence_time,
                        'sportsbook': 'draftkings',
                        'market': 'spread',
                        'side': side,
                        'point': outcome.get('point'),
                        'price': outcome['price'],
                        'week': 12,
                        'season': 2025,
                        'archived_at': fetch_timestamp
                    })

                elif market_key == 'totals':
                    odds_data.append({
                        'game_id': our_game_id,
                        'away_team': away_abbr,
                        'home_team': home_abbr,
                        'commence_time': commence_time,
                        'sportsbook': 'draftkings',
                        'market': 'total',
                        'side': outcome['name'].lower(),  # 'over' or 'under'
                        'point': outcome.get('point'),
                        'price': outcome['price'],
                        'week': 12,
                        'season': 2025,
                        'archived_at': fetch_timestamp
                    })

                elif market_key == 'h2h':
                    side = 'home' if outcome['name'] == home_team else 'away'
                    odds_data.append({
                        'game_id': our_game_id,
                        'away_team': away_abbr,
                        'home_team': home_abbr,
                        'commence_time': commence_time,
                        'sportsbook': 'draftkings',
                        'market': 'moneyline',
                        'side': side,
                        'point': None,
                        'price': outcome['price'],
                        'week': 12,
                        'season': 2025,
                        'archived_at': fetch_timestamp
                    })

    df = pd.DataFrame(odds_data)

    if df.empty:
        print("\n⚠️  No game lines parsed")
        sys.exit(1)

    print(f"\n✅ Parsed {len(df)} game line records from {df['game_id'].nunique()} games")

    # Save to historical folder
    output_dir = Path('data/historical/game_lines')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'game_lines_2025_week12.csv'
    df.to_csv(output_file, index=False)

    print(f"\n💾 Saved to: {output_file}")
    print(f"\nSample games:")
    for game_id in df['game_id'].unique()[:5]:
        game_df = df[df['game_id'] == game_id]
        spread = game_df[game_df['market'] == 'spread']
        total = game_df[game_df['market'] == 'total']
        ml = game_df[game_df['market'] == 'moneyline']

        # Get spread line (home)
        spread_line = spread[spread['side'] == 'home']['point'].values[0] if len(spread) > 0 else None
        total_line = total[total['side'] == 'over']['point'].values[0] if len(total) > 0 else None
        home_ml = ml[ml['side'] == 'home']['price'].values[0] if len(ml) > 0 else None

        away, home = game_id.split('_')[2], game_id.split('_')[3]
        print(f"  {away} @ {home}: {spread_line:+.1f} / O/U {total_line} / ML {home_ml:+d}")

    print("\n✅ Done!")

except requests.exceptions.RequestException as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
