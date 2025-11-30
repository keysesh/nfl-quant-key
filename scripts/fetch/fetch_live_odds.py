"""
Fetch live NFL odds from The Odds API (DraftKings only)
"""
import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

if not API_KEY:
    raise ValueError("ODDS_API_KEY not found in .env file")

def fetch_nfl_odds_draftkings(week: int = None):
    """
    Fetch NFL odds from DraftKings via The Odds API
    
    Args:
        week: NFL week number (optional, for filename)
    """
    print("🏈 Fetching NFL odds from DraftKings...")
    print(f"API Key: {API_KEY[:8]}...")
    
    # The Odds API endpoint for NFL
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',  # US bookmakers
        'markets': 'spreads,totals,h2h',  # Spread, totals (O/U), moneyline
        'oddsFormat': 'american',  # -110, +150, etc.
        'bookmakers': 'draftkings'  # DraftKings only
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # Check remaining requests
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        print(f"✅ API Response: {response.status_code}")
        print(f"📊 Requests - Used: {used} | Remaining: {remaining}")
        
        games = response.json()
        
        if not games:
            print("⚠️  No games found. NFL might be off-season or no upcoming games.")
            return None
        
        print(f"📋 Found {len(games)} NFL games with odds")
        
        # Parse odds data
        odds_data = []
        
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time']
            
            # Convert team names to abbreviations (simple mapping)
            home_abbr = convert_team_name(home_team)
            away_abbr = convert_team_name(away_team)
            
            # Create game_id format: 2025_08_AWAY_HOME
            if week:
                game_id = f"2025_{week:02d}_{away_abbr}_{home_abbr}"
            else:
                game_id = f"2025_XX_{away_abbr}_{home_abbr}"
            
            # Find DraftKings bookmaker
            draftkings = None
            for bookmaker in game.get('bookmakers', []):
                if bookmaker['key'] == 'draftkings':
                    draftkings = bookmaker
                    break
            
            if not draftkings:
                print(f"⚠️  No DraftKings odds for {away_team} @ {home_team}")
                continue
            
            # Extract markets
            spreads = None
            totals = None
            h2h = None
            
            for market in draftkings.get('markets', []):
                if market['key'] == 'spreads':
                    spreads = market
                elif market['key'] == 'totals':
                    totals = market
                elif market['key'] == 'h2h':
                    h2h = market
            
            # Parse spreads
            if spreads:
                for outcome in spreads['outcomes']:
                    if outcome['name'] == home_team:
                        odds_data.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': 'draftkings',
                            'market': 'spread',
                            'side': 'home',
                            'point': outcome.get('point', 0),
                            'price': outcome['price']
                        })
                    elif outcome['name'] == away_team:
                        odds_data.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': 'draftkings',
                            'market': 'spread',
                            'side': 'away',
                            'point': outcome.get('point', 0),
                            'price': outcome['price']
                        })
            
            # Parse totals
            if totals:
                for outcome in totals['outcomes']:
                    odds_data.append({
                        'game_id': game_id,
                        'away_team': away_abbr,
                        'home_team': home_abbr,
                        'commence_time': commence_time,
                        'sportsbook': 'draftkings',
                        'market': 'total',
                        'side': outcome['name'].lower(),  # 'over' or 'under'
                        'point': outcome.get('point', 0),
                        'price': outcome['price']
                    })
            
            # Parse moneyline (h2h)
            if h2h:
                for outcome in h2h['outcomes']:
                    if outcome['name'] == home_team:
                        odds_data.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': 'draftkings',
                            'market': 'moneyline',
                            'side': 'home',
                            'point': None,
                            'price': outcome['price']
                        })
                    elif outcome['name'] == away_team:
                        odds_data.append({
                            'game_id': game_id,
                            'away_team': away_abbr,
                            'home_team': home_abbr,
                            'commence_time': commence_time,
                            'sportsbook': 'draftkings',
                            'market': 'moneyline',
                            'side': 'away',
                            'point': None,
                            'price': outcome['price']
                        })
        
        # Create DataFrame
        df = pd.DataFrame(odds_data)
        
        if df.empty:
            print("⚠️  No odds data extracted")
            return None
        
        # Display summary
        print("\n📊 ODDS SUMMARY:")
        print("="*80)
        for game_id in df['game_id'].unique():
            game_odds = df[df['game_id'] == game_id]
            away = game_odds['away_team'].iloc[0]
            home = game_odds['home_team'].iloc[0]
            
            print(f"\n{away} @ {home}")
            
            # Spread
            home_spread = game_odds[(game_odds['market'] == 'spread') & (game_odds['side'] == 'home')]
            away_spread = game_odds[(game_odds['market'] == 'spread') & (game_odds['side'] == 'away')]
            if not home_spread.empty:
                hs_point = home_spread['point'].iloc[0]
                hs_price = home_spread['price'].iloc[0]
                as_point = away_spread['point'].iloc[0]
                as_price = away_spread['price'].iloc[0]
                print(f"  Spread: {home} {hs_point:+.1f} ({hs_price:+d}) | {away} {as_point:+.1f} ({as_price:+d})")
            
            # Total
            over = game_odds[(game_odds['market'] == 'total') & (game_odds['side'] == 'over')]
            under = game_odds[(game_odds['market'] == 'total') & (game_odds['side'] == 'under')]
            if not over.empty:
                o_point = over['point'].iloc[0]
                o_price = over['price'].iloc[0]
                u_price = under['price'].iloc[0]
                print(f"  Total: O {o_point:.1f} ({o_price:+d}) | U {o_point:.1f} ({u_price:+d})")
            
            # Moneyline
            home_ml = game_odds[(game_odds['market'] == 'moneyline') & (game_odds['side'] == 'home')]
            away_ml = game_odds[(game_odds['market'] == 'moneyline') & (game_odds['side'] == 'away')]
            if not home_ml.empty:
                hml_price = home_ml['price'].iloc[0]
                aml_price = away_ml['price'].iloc[0]
                print(f"  Moneyline: {home} {hml_price:+d} | {away} {aml_price:+d}")
        
        # Save to CSV
        output_dir = Path('data')
        output_dir.mkdir(exist_ok=True)
        
        if week:
            output_file = output_dir / f'odds_week{week}_draftkings.csv'
        else:
            output_file = output_dir / f'odds_draftkings_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved odds to: {output_file}")
        
        # Also create simplified format for compatibility
        simple_odds = []
        for _, row in df.iterrows():
            if row['market'] == 'spread':
                simple_odds.append({
                    'game_id': row['game_id'],
                    'side': f"{row['side']}_spread",
                    'american_odds': row['price'],
                    'point': row['point']
                })
            elif row['market'] == 'total':
                simple_odds.append({
                    'game_id': row['game_id'],
                    'side': row['side'],
                    'american_odds': row['price'],
                    'point': row['point']
                })
        
        simple_df = pd.DataFrame(simple_odds)
        if week:
            simple_file = output_dir / f'odds_week{week}.csv'
            simple_df.to_csv(simple_file, index=False)
            print(f"✅ Saved simplified format to: {simple_file}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching odds: {e}")
        return None


def convert_team_name(full_name: str) -> str:
    """Convert full team name to abbreviation"""
    team_map = {
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
        'Los Angeles Rams': 'LAR',
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
        'Washington Commanders': 'WAS'
    }
    return team_map.get(full_name, full_name[:3].upper())


if __name__ == '__main__':
    import sys
    
    # Get week from command line or use None
    week = None
    if len(sys.argv) > 1:
        try:
            week = int(sys.argv[1])
        except ValueError:
            print("Usage: python fetch_live_odds.py [week_number]")
            sys.exit(1)
    
    print(f"🏈 NFL ODDS FETCHER - DraftKings Only")
    print(f"=" * 80)
    if week:
        print(f"Week: {week}")
    else:
        print("Week: Auto-detect upcoming games")
    print()
    
    df = fetch_nfl_odds_draftkings(week)
    
    if df is not None:
        print(f"\n✅ SUCCESS! Fetched {len(df)} odds records")
    else:
        print("\n❌ Failed to fetch odds")







