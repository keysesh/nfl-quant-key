"""
Line movement scraper using The Odds API.

Fetches current odds every 15 minutes and stores in CLV database.
"""

import os
import requests
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class LineMovementTracker:
    """Track line movements from sportsbooks via The Odds API."""

    def __init__(self, config_path: str = 'configs/odds_api_config.json'):
        """Initialize tracker with API configuration."""
        # Load API key from .env file (matches existing system)
        load_dotenv()
        self.api_key = os.getenv('ODDS_API_KEY')

        if not self.api_key:
            raise ValueError("ODDS_API_KEY not found in .env file")

        # Load config for other settings
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.base_url = self.config.get('base_url', 'https://api.the-odds-api.com/v4')
            self.sport = self.config.get('sport', 'americanfootball_nfl')
        else:
            self.base_url = 'https://api.the-odds-api.com/v4'
            self.sport = 'americanfootball_nfl'

        self.db = sqlite3.connect('data/clv_tracking.db')

    def fetch_current_odds(self) -> dict:
        """Fetch current odds from The Odds API."""
        url = f"{self.base_url}/sports/{self.sport}/odds/"

        params = {
            'apiKey': self.api_key,
            'regions': self.config.get('regions', 'us'),
            'markets': ','.join(self.config.get('markets', ['spreads', 'totals'])),
            'oddsFormat': 'american'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Log API usage
            remaining = response.headers.get('x-requests-remaining')
            logger.info(f"API requests remaining: {remaining}")

            return data

        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return None

    def store_line_movements(self, odds_data: list):
        """Store odds data in database."""
        if not odds_data:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now()

        for game in odds_data:
            game_id = game['id']

            for bookmaker in game.get('bookmakers', []):
                sportsbook = bookmaker['key']

                for market in bookmaker.get('markets', []):
                    market_type = market['key']

                    for outcome in market.get('outcomes', []):
                        cursor.execute("""
                            INSERT INTO line_movements (
                                game_id, market_type, timestamp, sportsbook,
                                line_value, over_odds, under_odds, home_odds, away_odds
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            game_id,
                            market_type,
                            timestamp,
                            sportsbook,
                            outcome.get('point'),
                            outcome.get('price') if outcome.get('name') == 'Over' else None,
                            outcome.get('price') if outcome.get('name') == 'Under' else None,
                            outcome.get('price') if outcome.get('name') == game['home_team'] else None,
                            outcome.get('price') if outcome.get('name') == game['away_team'] else None
                        ))

        self.db.commit()
        logger.info(f"Stored {len(odds_data)} games worth of line movements")

    def get_closing_line(self, game_id: str, market_type: str):
        """Get closing line (30 min before kickoff)."""
        cursor = self.db.cursor()

        # Query lines from 25-35 minutes before game
        # This is the industry standard for "closing line"
        query = """
            SELECT line_value, over_odds, under_odds, home_odds, away_odds
            FROM line_movements
            WHERE game_id = ? AND market_type = ?
            ORDER BY ABS(julianday('now') - julianday(timestamp)) ASC
            LIMIT 1
        """

        cursor.execute(query, (game_id, market_type))
        return cursor.fetchone()

    def run_continuous(self, interval_minutes: int = 15):
        """Run continuous line tracking."""
        logger.info(f"Starting continuous line tracking (every {interval_minutes} min)")

        while True:
            try:
                odds_data = self.fetch_current_odds()
                if odds_data:
                    self.store_line_movements(odds_data)

                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Stopping line tracker")
                break
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                time.sleep(60)  # Wait 1 min before retry

    def close(self):
        """Close database connection."""
        self.db.close()
