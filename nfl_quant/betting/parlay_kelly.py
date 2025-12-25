"""
Parlay Kelly Criterion
======================

Extension of Kelly criterion for multi-leg parlays.
Key adjustments:
1. Variance penalty by number of legs (more legs = more conservative)
2. Lower maximum stake (0.5x straight bet max)
3. Correlation-adjusted probability input

Formula: f* = (bp - q) / b
Where:
    f* = fraction of bankroll to bet
    b = decimal odds - 1 (net odds)
    p = win probability (correlation-adjusted)
    q = 1 - p
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# Import base Kelly functions
try:
    from .kelly_criterion import american_to_decimal, calculate_kelly_fraction
except ImportError:
    from nfl_quant.betting.kelly_criterion import american_to_decimal, calculate_kelly_fraction


@dataclass
class ParlayKellyResult:
    """Result of parlay Kelly calculation."""
    should_bet: bool
    edge: float
    full_kelly: float
    variance_adjusted_kelly: float
    recommended_fraction: float
    recommended_stake: float
    num_legs: int
    variance_penalty: float
    confidence: str
    expected_value: float
    reason: Optional[str] = None


class ParlayKelly:
    """
    Kelly criterion with variance adjustment for multi-leg parlays.

    Variance increases exponentially with legs, so Kelly fraction
    should decrease accordingly.
    """

    # Variance penalty by number of legs
    VARIANCE_PENALTY = {
        2: 0.85,  # 2-leg: 85% of base Kelly
        3: 0.70,  # 3-leg: 70% of base Kelly
        4: 0.55,  # 4-leg: 55% of base Kelly
        5: 0.40,  # 5-leg: 40% of base Kelly (if ever used)
    }

    def __init__(
        self,
        base_kelly_fraction: float = 0.15,  # More conservative than single bets (0.25)
        max_bet_fraction: float = 0.025,  # 2.5% max for parlays (vs 5% for straights)
        min_edge_pct: float = 0.05,  # 5% minimum edge for parlays
        base_unit_size: float = 5.0
    ):
        """
        Initialize parlay Kelly calculator.

        Args:
            base_kelly_fraction: Base Kelly fraction before variance adjustment
            max_bet_fraction: Maximum fraction of bankroll per parlay
            min_edge_pct: Minimum edge required to recommend a parlay
            base_unit_size: Dollar value per unit
        """
        self.base_kelly_fraction = base_kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge_pct = min_edge_pct
        self.base_unit_size = base_unit_size

    def get_variance_penalty(self, num_legs: int) -> float:
        """
        Get variance penalty for given number of legs.

        Args:
            num_legs: Number of parlay legs

        Returns:
            Penalty multiplier (0-1)
        """
        if num_legs < 2:
            return 1.0
        return self.VARIANCE_PENALTY.get(num_legs, 0.35)  # 35% for 6+ legs

    def calculate_parlay_kelly(
        self,
        adjusted_prob: float,
        parlay_odds: int,
        num_legs: int,
        bankroll: float = 1000.0
    ) -> ParlayKellyResult:
        """
        Calculate Kelly bet size for a parlay.

        Args:
            adjusted_prob: Correlation-adjusted joint probability
            parlay_odds: American odds for the parlay
            num_legs: Number of legs in the parlay
            bankroll: Current bankroll

        Returns:
            ParlayKellyResult with betting recommendation
        """
        # Convert odds
        decimal_odds = american_to_decimal(parlay_odds)
        implied_prob = 1.0 / decimal_odds

        # Calculate edge
        edge = adjusted_prob - implied_prob

        # Check minimum edge
        if edge < self.min_edge_pct:
            return ParlayKellyResult(
                should_bet=False,
                edge=edge,
                full_kelly=0.0,
                variance_adjusted_kelly=0.0,
                recommended_fraction=0.0,
                recommended_stake=0.0,
                num_legs=num_legs,
                variance_penalty=self.get_variance_penalty(num_legs),
                confidence='NO BET',
                expected_value=0.0,
                reason=f'Edge {edge:.1%} below minimum {self.min_edge_pct:.1%}'
            )

        # Calculate full Kelly
        full_kelly = calculate_kelly_fraction(adjusted_prob, parlay_odds)

        if full_kelly <= 0:
            return ParlayKellyResult(
                should_bet=False,
                edge=edge,
                full_kelly=full_kelly,
                variance_adjusted_kelly=0.0,
                recommended_fraction=0.0,
                recommended_stake=0.0,
                num_legs=num_legs,
                variance_penalty=self.get_variance_penalty(num_legs),
                confidence='NO BET',
                expected_value=0.0,
                reason='Negative Kelly (no edge after vig)'
            )

        # Apply base fraction
        fractional_kelly = full_kelly * self.base_kelly_fraction

        # Apply variance penalty
        variance_penalty = self.get_variance_penalty(num_legs)
        variance_adjusted_kelly = fractional_kelly * variance_penalty

        # Apply maximum cap
        final_fraction = min(variance_adjusted_kelly, self.max_bet_fraction)

        # Calculate stake
        stake = bankroll * final_fraction
        stake = round(stake, 2)

        # Determine confidence level
        if edge >= 0.15:
            confidence = 'HIGH'
        elif edge >= 0.10:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # Calculate EV
        expected_value = stake * (decimal_odds * adjusted_prob - 1)

        return ParlayKellyResult(
            should_bet=True,
            edge=edge,
            full_kelly=full_kelly,
            variance_adjusted_kelly=variance_adjusted_kelly,
            recommended_fraction=final_fraction,
            recommended_stake=stake,
            num_legs=num_legs,
            variance_penalty=variance_penalty,
            confidence=confidence,
            expected_value=expected_value
        )

    def calculate_units(
        self,
        adjusted_prob: float,
        parlay_odds: int,
        num_legs: int,
        max_units: float = 0.5  # Max 0.5 units for parlays
    ) -> float:
        """
        Calculate recommended units for a parlay.

        Simpler interface that returns just the unit sizing.

        Args:
            adjusted_prob: Correlation-adjusted probability
            parlay_odds: American odds
            num_legs: Number of legs
            max_units: Maximum units to bet

        Returns:
            Recommended units (0 to max_units)
        """
        result = self.calculate_parlay_kelly(
            adjusted_prob=adjusted_prob,
            parlay_odds=parlay_odds,
            num_legs=num_legs,
            bankroll=1000.0  # Use normalized bankroll
        )

        if not result.should_bet:
            return 0.0

        # Convert fraction to units (normalized to 1000 bankroll with $5 unit)
        units = (result.recommended_stake / self.base_unit_size)

        # Cap at max_units
        return min(units, max_units)


def calculate_parlay_odds(leg_odds: list) -> int:
    """
    Calculate parlay odds from individual leg odds.

    Args:
        leg_odds: List of American odds for each leg

    Returns:
        Parlay odds in American format
    """
    if not leg_odds:
        return 0

    # Convert to decimal, multiply, convert back
    decimal_product = 1.0
    for odds in leg_odds:
        decimal_product *= american_to_decimal(odds)

    # Convert back to American
    if decimal_product >= 2.0:
        return int((decimal_product - 1) * 100)
    else:
        return int(-100 / (decimal_product - 1))


def test_parlay_kelly():
    """Test the parlay Kelly calculator."""
    kelly = ParlayKelly()

    # Test 2-leg parlay
    result = kelly.calculate_parlay_kelly(
        adjusted_prob=0.35,
        parlay_odds=300,  # +300
        num_legs=2,
        bankroll=1000.0
    )
    print(f"2-leg parlay (+300, 35% prob):")
    print(f"  should_bet={result.should_bet}, edge={result.edge:.1%}")
    print(f"  full_kelly={result.full_kelly:.4f}")
    print(f"  variance_penalty={result.variance_penalty}")
    print(f"  recommended_stake=${result.recommended_stake:.2f}")
    print(f"  confidence={result.confidence}")
    print()

    # Test 4-leg parlay
    result = kelly.calculate_parlay_kelly(
        adjusted_prob=0.12,
        parlay_odds=800,  # +800
        num_legs=4,
        bankroll=1000.0
    )
    print(f"4-leg parlay (+800, 12% prob):")
    print(f"  should_bet={result.should_bet}, edge={result.edge:.1%}")
    print(f"  full_kelly={result.full_kelly:.4f}")
    print(f"  variance_penalty={result.variance_penalty}")
    print(f"  recommended_stake=${result.recommended_stake:.2f}")
    print(f"  confidence={result.confidence}")
    print()

    # Test units calculation
    units = kelly.calculate_units(
        adjusted_prob=0.35,
        parlay_odds=300,
        num_legs=2
    )
    print(f"Recommended units for 2-leg +300: {units:.2f}u")

    # Test parlay odds calculation
    leg_odds = [-110, -115, -105]
    combined = calculate_parlay_odds(leg_odds)
    print(f"\nParlay odds for {leg_odds}: {combined:+d}")


if __name__ == "__main__":
    test_parlay_kelly()
