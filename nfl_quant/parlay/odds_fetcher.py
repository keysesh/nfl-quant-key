"""
Parlay Odds Fetcher
===================

Fetches and calculates parlay odds from various sources.

Methods:
1. Calculate from individual leg odds (multiply decimal odds)
2. Fetch from Odds API if available (future feature)

The Odds API primarily provides individual prop/game odds.
Custom parlay pricing is typically not available via API.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from pathlib import Path


@dataclass
class LegOdds:
    """Represents odds for a single parlay leg."""
    player: str
    market: str
    direction: str  # "OVER" or "UNDER"
    line: float
    american_odds: int
    decimal_odds: float
    implied_prob: float


@dataclass
class ParlayOdds:
    """Represents calculated parlay odds."""
    legs: List[LegOdds]
    calculated_american: int
    calculated_decimal: float
    calculated_implied_prob: float
    source: str  # "calculated" or "api"


class ParlayOddsFetcher:
    """
    Fetch and calculate parlay odds.

    Primary method: Calculate from leg multiplication.
    Future: API integration if sportsbooks expose parlay pricing.
    """

    def __init__(self, api_key: str = None, preferred_sportsbook: str = "fanduel"):
        """
        Initialize the odds fetcher.

        Args:
            api_key: Odds API key (from env if not provided)
            preferred_sportsbook: Default sportsbook for odds
        """
        self.api_key = api_key or os.environ.get('ODDS_API_KEY', '')
        self.preferred_sportsbook = preferred_sportsbook
        self.base_url = "https://api.the-odds-api.com/v4"

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal."""
        if american > 0:
            return 1.0 + (american / 100.0)
        else:
            return 1.0 + (100.0 / abs(american))

    @staticmethod
    def decimal_to_american(decimal: float) -> int:
        """Convert decimal odds to American."""
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))

    @staticmethod
    def decimal_to_implied_prob(decimal: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1.0 / decimal if decimal > 0 else 0.0

    def calculate_parlay_odds(self, leg_odds: List[int]) -> Tuple[int, float, float]:
        """
        Calculate parlay odds from individual leg American odds.

        Standard parlay calculation: multiply decimal odds.

        Args:
            leg_odds: List of American odds for each leg

        Returns:
            (american_odds, decimal_odds, implied_probability)
        """
        if not leg_odds:
            return 0, 1.0, 1.0

        # Multiply decimal odds
        decimal_product = 1.0
        for odds in leg_odds:
            decimal_product *= self.american_to_decimal(odds)

        # Convert to American
        american = self.decimal_to_american(decimal_product)

        # Calculate implied probability
        implied_prob = self.decimal_to_implied_prob(decimal_product)

        return american, decimal_product, implied_prob

    def create_leg_odds(
        self,
        player: str,
        market: str,
        direction: str,
        line: float,
        american_odds: int
    ) -> LegOdds:
        """Create a LegOdds object with all derived values."""
        decimal = self.american_to_decimal(american_odds)
        implied = self.decimal_to_implied_prob(decimal)

        return LegOdds(
            player=player,
            market=market,
            direction=direction,
            line=line,
            american_odds=american_odds,
            decimal_odds=round(decimal, 4),
            implied_prob=round(implied, 4)
        )

    def calculate_parlay(self, legs: List[LegOdds]) -> ParlayOdds:
        """
        Calculate parlay odds from leg odds objects.

        Args:
            legs: List of LegOdds objects

        Returns:
            ParlayOdds with calculated values
        """
        leg_american = [leg.american_odds for leg in legs]
        american, decimal, implied = self.calculate_parlay_odds(leg_american)

        return ParlayOdds(
            legs=legs,
            calculated_american=american,
            calculated_decimal=round(decimal, 4),
            calculated_implied_prob=round(implied, 4),
            source="calculated"
        )

    def fetch_live_odds(
        self,
        sport: str = "americanfootball_nfl",
        markets: str = "player_props"
    ) -> Optional[Dict]:
        """
        Fetch live odds from The Odds API.

        Note: This fetches individual prop odds, not parlay-specific pricing.
        Parlay odds are calculated from individual legs.

        Args:
            sport: Sport key
            markets: Market types to fetch

        Returns:
            API response dict or None if failed
        """
        if not self.api_key:
            print("Warning: No ODDS_API_KEY configured")
            return None

        try:
            url = f"{self.base_url}/sports/{sport}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': markets,
                'bookmakers': self.preferred_sportsbook
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            print(f"Error fetching odds: {e}")
            return None

    def get_leg_odds_from_props(
        self,
        props_data: Dict,
        player: str,
        market: str,
        direction: str
    ) -> Optional[int]:
        """
        Extract odds for a specific leg from props data.

        Args:
            props_data: Props data from edge recommendations or API
            player: Player name
            market: Market type (e.g., player_receptions)
            direction: "OVER" or "UNDER"

        Returns:
            American odds or None if not found
        """
        # This would parse the props data structure
        # For now, return standard -110 as default
        return -110

    def estimate_parlay_with_vig(
        self,
        leg_odds: List[int],
        vig_adjustment: float = 0.02
    ) -> Tuple[int, float, float]:
        """
        Estimate parlay odds with vig adjustment.

        Some books add extra vig to parlays beyond leg multiplication.
        This estimates that adjustment.

        Args:
            leg_odds: List of American odds
            vig_adjustment: Per-leg vig adjustment (default 2%)

        Returns:
            (american_odds, decimal_odds, implied_probability)
        """
        american, decimal, implied = self.calculate_parlay_odds(leg_odds)

        # Adjust for additional parlay vig
        num_legs = len(leg_odds)
        total_vig_adjustment = 1 - (vig_adjustment * num_legs)

        adjusted_decimal = decimal * total_vig_adjustment
        adjusted_american = self.decimal_to_american(adjusted_decimal)
        adjusted_implied = self.decimal_to_implied_prob(adjusted_decimal)

        return adjusted_american, adjusted_decimal, adjusted_implied


def test_odds_fetcher():
    """Test the odds fetcher."""
    fetcher = ParlayOddsFetcher()

    # Test parlay odds calculation
    leg_odds = [-110, -115, -105, -120]

    american, decimal, implied = fetcher.calculate_parlay_odds(leg_odds)
    print(f"4-leg parlay from {leg_odds}:")
    print(f"  American: {american:+d}")
    print(f"  Decimal: {decimal:.4f}")
    print(f"  Implied Prob: {implied:.2%}")
    print()

    # Test with vig adjustment
    american_vig, decimal_vig, implied_vig = fetcher.estimate_parlay_with_vig(
        leg_odds, vig_adjustment=0.02
    )
    print(f"With 2% per-leg vig adjustment:")
    print(f"  American: {american_vig:+d}")
    print(f"  Decimal: {decimal_vig:.4f}")
    print(f"  Implied Prob: {implied_vig:.2%}")
    print()

    # Test creating leg odds objects
    leg1 = fetcher.create_leg_odds(
        player="Travis Kelce",
        market="player_receptions",
        direction="UNDER",
        line=5.5,
        american_odds=-115
    )
    print(f"Leg: {leg1.player} {leg1.direction} {leg1.line} {leg1.market}")
    print(f"  American: {leg1.american_odds:+d}")
    print(f"  Decimal: {leg1.decimal_odds:.4f}")
    print(f"  Implied: {leg1.implied_prob:.2%}")


if __name__ == "__main__":
    test_odds_fetcher()
