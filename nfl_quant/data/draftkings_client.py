"""
DraftKings NFL Props Client - Odds API Integration
API Key: 73ec9367021badb173a0b68c35af818f

Integrated into NFL QUANT system for:
1. Live odds fetching
2. Closing line capture
3. Edge detection against model predictions
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import requests
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "73ec9367021badb173a0b68c35af818f"
DRAFTKINGS = "draftkings"

# Core markets we track
CORE_MARKETS = [
    'player_pass_yds',
    'player_rush_yds',
    'player_reception_yds',
    'player_receptions',
    'player_pass_tds',
    'player_rush_attempts',
    'player_anytime_td',
]


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def calculate_ev(prob: float, odds: int) -> float:
    """Calculate expected value of a bet."""
    decimal = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
    return prob * (decimal - 1) - (1 - prob)


def kelly(prob: float, odds: int, mult: float = 0.25) -> float:
    """Calculate Kelly criterion bet size (quarter Kelly by default)."""
    decimal = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
    k = ((decimal - 1) * prob - (1 - prob)) / (decimal - 1)
    return max(0, min(k * mult, 0.10))  # Cap at 10%


@dataclass
class PropLine:
    """Represents a single prop line from DraftKings."""
    event_id: str
    commence_time: str
    home_team: str
    away_team: str
    market: str
    player_name: str
    line: float
    over_price: int
    under_price: int
    fetch_timestamp: str = ""

    def __post_init__(self):
        if not self.fetch_timestamp:
            self.fetch_timestamp = datetime.utcnow().isoformat()

    @property
    def no_vig_over(self) -> float:
        """Calculate no-vig probability for OVER."""
        o = american_to_prob(self.over_price)
        u = american_to_prob(self.under_price)
        return o / (o + u)

    @property
    def no_vig_under(self) -> float:
        """Calculate no-vig probability for UNDER."""
        return 1 - self.no_vig_over

    @property
    def vig_pct(self) -> float:
        """Calculate the vig percentage."""
        o = american_to_prob(self.over_price)
        u = american_to_prob(self.under_price)
        return (o + u - 1) * 100


class DKClient:
    """DraftKings odds client using the-odds-api.com"""

    BASE = "https://api.the-odds-api.com/v4"
    SPORT = "americanfootball_nfl"

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        self.remaining = None
        self.used = None

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make API request with error handling."""
        params['apiKey'] = self.api_key
        try:
            r = requests.get(f"{self.BASE}/{endpoint}", params=params, timeout=30)
            self.remaining = r.headers.get('x-requests-remaining')
            self.used = r.headers.get('x-requests-used')
            logger.info(f"API quota: {self.remaining} remaining, {self.used} used")

            if r.status_code != 200:
                logger.error(f"API error {r.status_code}: {r.text}")
                return {}
            return r.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {}

    def get_events(self) -> list:
        """Get all upcoming NFL events."""
        return self._get(f"sports/{self.SPORT}/events", {})

    def get_props(self, event_id: str, markets: list = None) -> List[PropLine]:
        """Get props for a specific event."""
        if markets is None:
            markets = CORE_MARKETS

        data = self._get(f"sports/{self.SPORT}/events/{event_id}/odds", {
            'bookmakers': DRAFTKINGS,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
        })
        return self._parse(data) if data else []

    def get_all_props(self, markets: list = None) -> pd.DataFrame:
        """Get all props for all upcoming games."""
        if markets is None:
            markets = CORE_MARKETS

        events = self.get_events()
        logger.info(f"Found {len(events)} upcoming NFL games")

        props = []
        for e in events:
            event_props = self.get_props(e['id'], markets)
            props.extend(event_props)
            logger.info(f"  {e['away_team']} @ {e['home_team']}: {len(event_props)} props")
            time.sleep(0.3)  # Rate limiting

        if not props:
            logger.warning("No props found")
            return pd.DataFrame()

        df = pd.DataFrame([asdict(p) for p in props])
        df['no_vig_over'] = df.apply(
            lambda r: american_to_prob(r['over_price']) /
                     (american_to_prob(r['over_price']) + american_to_prob(r['under_price'])),
            axis=1
        )
        df['no_vig_under'] = 1 - df['no_vig_over']

        return df

    def get_historical_props(self, event_id: str, date: str, markets: list = None) -> List[PropLine]:
        """Get historical odds for an event at a specific date/time."""
        if markets is None:
            markets = CORE_MARKETS

        data = self._get(f"historical/sports/{self.SPORT}/events/{event_id}/odds", {
            'bookmakers': DRAFTKINGS,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'date': date,
        })
        return self._parse(data.get('data', {})) if data else []

    def _parse(self, data: dict) -> List[PropLine]:
        """Parse API response into PropLine objects."""
        props = []

        for bk in data.get('bookmakers', []):
            if bk.get('key') != DRAFTKINGS:
                continue

            for mkt in bk.get('markets', []):
                # Group outcomes by player
                players = {}
                for o in mkt.get('outcomes', []):
                    player = o.get('description')
                    if player not in players:
                        players[player] = {}
                    players[player][o.get('name')] = o

                # Create PropLine for each player with both Over and Under
                for player, dirs in players.items():
                    if 'Over' in dirs and 'Under' in dirs:
                        props.append(PropLine(
                            event_id=data.get('id', ''),
                            commence_time=data.get('commence_time', ''),
                            home_team=data.get('home_team', ''),
                            away_team=data.get('away_team', ''),
                            market=mkt.get('key'),
                            player_name=player,
                            line=dirs['Over'].get('point', 0),
                            over_price=dirs['Over'].get('price', -110),
                            under_price=dirs['Under'].get('price', -110),
                        ))

        return props


def find_edges(
    model_preds: pd.DataFrame,
    min_edge: float = 0.03,
    min_ev: float = 0.02,
    variance_inflation: float = 1.25
) -> pd.DataFrame:
    """
    Find betting edges by comparing model predictions to DraftKings lines.

    Args:
        model_preds: DataFrame with columns:
            - player_name
            - stat_type: 'pass_yds', 'rush_yds', 'rec_yds', 'receptions'
            - predicted_value: mean prediction
            - predicted_std: standard deviation
        min_edge: Minimum edge threshold (default 3%)
        min_ev: Minimum expected value threshold (default 2%)
        variance_inflation: Factor to inflate std (addresses underestimation issue)

    Returns:
        DataFrame of edges sorted by EV
    """
    client = DKClient()
    dk = client.get_all_props()

    if dk.empty:
        logger.warning("No DraftKings props available")
        return pd.DataFrame()

    # Map stat types to market keys
    stat_map = {
        'pass_yds': 'player_pass_yds',
        'rush_yds': 'player_rush_yds',
        'rec_yds': 'player_reception_yds',
        'receptions': 'player_receptions',
        'pass_tds': 'player_pass_tds',
        'rush_attempts': 'player_rush_attempts',
        'receiving_yards': 'player_reception_yds',
        'rushing_yards': 'player_rush_yds',
        'passing_yards': 'player_pass_yds',
    }

    edges = []

    for _, pred in model_preds.iterrows():
        stat_type = pred.get('stat_type', '')
        mkt = stat_map.get(stat_type, stat_type)

        mean = pred['predicted_value']
        std = pred['predicted_std'] * variance_inflation  # Apply inflation fix

        if std <= 0:
            logger.warning(f"Zero std for {pred['player_name']}, skipping")
            continue

        # Find matching DK lines (case-insensitive)
        matches = dk[
            (dk['player_name'].str.lower() == pred['player_name'].lower()) &
            (dk['market'] == mkt)
        ]

        for _, line in matches.iterrows():
            # Calculate model probability of OVER
            model_over = 1 - stats.norm.cdf(line['line'], mean, std)
            over_edge = model_over - line['no_vig_over']
            over_ev = calculate_ev(model_over, line['over_price'])

            # Check OVER edge
            if over_edge >= min_edge and over_ev >= min_ev:
                edges.append({
                    'player': pred['player_name'],
                    'market': mkt,
                    'line': line['line'],
                    'direction': 'OVER',
                    'odds': line['over_price'],
                    'model_prob': round(model_over, 3),
                    'dk_prob': round(line['no_vig_over'], 3),
                    'edge': round(over_edge, 3),
                    'ev': round(over_ev, 3),
                    'kelly': round(kelly(model_over, line['over_price']), 4),
                    'matchup': f"{line['away_team']} @ {line['home_team']}",
                    'commence_time': line['commence_time'],
                })

            # Calculate model probability of UNDER
            model_under = 1 - model_over
            under_edge = model_under - line['no_vig_under']
            under_ev = calculate_ev(model_under, line['under_price'])

            # Check UNDER edge
            if under_edge >= min_edge and under_ev >= min_ev:
                edges.append({
                    'player': pred['player_name'],
                    'market': mkt,
                    'line': line['line'],
                    'direction': 'UNDER',
                    'odds': line['under_price'],
                    'model_prob': round(model_under, 3),
                    'dk_prob': round(line['no_vig_under'], 3),
                    'edge': round(under_edge, 3),
                    'ev': round(under_ev, 3),
                    'kelly': round(kelly(model_under, line['under_price']), 4),
                    'matchup': f"{line['away_team']} @ {line['home_team']}",
                    'commence_time': line['commence_time'],
                })

    if not edges:
        logger.info("No edges found meeting criteria")
        return pd.DataFrame()

    result = pd.DataFrame(edges).sort_values('ev', ascending=False)
    logger.info(f"Found {len(result)} edges (min_edge={min_edge}, min_ev={min_ev})")

    return result


def save_current_odds(output_dir: Path = None) -> Path:
    """
    Fetch and save current DraftKings odds to CSV.
    Use for closing line capture before games start.

    Returns:
        Path to saved CSV file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'odds'

    output_dir.mkdir(parents=True, exist_ok=True)

    client = DKClient()
    df = client.get_all_props()

    if df.empty:
        logger.warning("No odds to save")
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'draftkings_props_{timestamp}.csv'
    df.to_csv(filename, index=False)

    logger.info(f"Saved {len(df)} props to {filename}")
    return filename


if __name__ == "__main__":
    # Test the client
    print("=" * 60)
    print("DraftKings NFL Props Client - Test")
    print("=" * 60)

    client = DKClient()

    print("\nFetching NFL events...")
    events = client.get_events()
    print(f"Found {len(events)} upcoming games:")

    for e in events[:5]:
        print(f"  {e['away_team']} @ {e['home_team']} - {e['commence_time']}")

    if events:
        print(f"\nFetching props for first game...")
        props = client.get_props(events[0]['id'])
        print(f"Found {len(props)} props")

        for p in props[:5]:
            print(f"  {p.player_name}: {p.market} {p.line} (O:{p.over_price}/U:{p.under_price})")

    print(f"\nAPI quota remaining: {client.remaining}")
