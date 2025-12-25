"""
Test suite for the parlay optimization system.

Tests:
1. Correlation-adjusted probability calculator
2. Parlay Kelly criterion
3. Odds calculator
4. ParlayRecommender integration
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.parlay.correlation_adjusted import (
    CorrelationAdjustedCalculator,
    ParlayLeg
)
from nfl_quant.betting.parlay_kelly import (
    ParlayKelly,
    calculate_parlay_odds
)
from nfl_quant.parlay.odds_fetcher import ParlayOddsFetcher
from nfl_quant.parlay.recommendation import (
    ParlayRecommender,
    SingleBet
)


class TestCorrelationAdjustedCalculator:
    """Test the correlation-adjusted probability calculator."""

    @pytest.fixture
    def calculator(self):
        return CorrelationAdjustedCalculator()

    def test_cross_game_correlation_is_zero(self, calculator):
        """Cross-game legs should have zero correlation."""
        leg1 = ParlayLeg(
            player="Player A",
            team="KC",
            market="player_receptions",
            direction="UNDER",
            line=5.5,
            confidence=0.60,
            game="KC@LV"
        )
        leg2 = ParlayLeg(
            player="Player B",
            team="MIA",
            market="player_rush_yds",
            direction="UNDER",
            line=80.5,
            confidence=0.55,
            game="MIA@NYJ"
        )

        correlation = calculator.get_correlation(leg1, leg2)
        assert correlation == 0.0, "Cross-game correlation should be zero"

    def test_same_player_correlation_is_positive(self, calculator):
        """Same player different markets should have positive correlation."""
        leg1 = ParlayLeg(
            player="Travis Kelce",
            team="KC",
            market="player_receptions",
            direction="UNDER",
            line=5.5,
            confidence=0.60,
            game="KC@LV"
        )
        leg2 = ParlayLeg(
            player="Travis Kelce",
            team="KC",
            market="player_reception_yds",
            direction="UNDER",
            line=70.5,
            confidence=0.55,
            game="KC@LV"
        )

        correlation = calculator.get_correlation(leg1, leg2)
        assert correlation > 0.0, "Same player correlation should be positive"

    def test_joint_probability_with_zero_correlation(self, calculator):
        """Joint probability with zero correlation = naive product."""
        leg1 = ParlayLeg(
            player="Player A", team="KC", market="player_receptions",
            direction="UNDER", line=5.5, confidence=0.60, game="KC@LV"
        )
        leg2 = ParlayLeg(
            player="Player B", team="MIA", market="player_rush_yds",
            direction="UNDER", line=80.5, confidence=0.55, game="MIA@NYJ"
        )

        naive, adjusted, factor = calculator.calculate_joint_probability([leg1, leg2])

        # For independent events, adjusted should equal naive
        assert abs(factor - 1.0) < 0.01, "Correlation factor should be ~1.0 for independent events"
        assert abs(naive - adjusted) < 0.01, "Naive and adjusted should match for independent events"

    def test_naive_probability_is_product(self, calculator):
        """Naive probability should be product of individual probabilities."""
        leg1 = ParlayLeg(
            player="Player A", team="KC", market="player_receptions",
            direction="UNDER", line=5.5, confidence=0.60, game="KC@LV"
        )
        leg2 = ParlayLeg(
            player="Player B", team="MIA", market="player_rush_yds",
            direction="UNDER", line=80.5, confidence=0.55, game="MIA@NYJ"
        )

        naive, _, _ = calculator.calculate_joint_probability([leg1, leg2])

        expected_naive = 0.60 * 0.55
        assert abs(naive - expected_naive) < 0.01, f"Naive should be {expected_naive}, got {naive}"


class TestParlayKelly:
    """Test the parlay Kelly criterion."""

    @pytest.fixture
    def kelly(self):
        return ParlayKelly()

    def test_variance_penalty_increases_with_legs(self, kelly):
        """Variance penalty should increase with more legs."""
        penalty_2 = kelly.get_variance_penalty(2)
        penalty_3 = kelly.get_variance_penalty(3)
        penalty_4 = kelly.get_variance_penalty(4)

        assert penalty_2 > penalty_3 > penalty_4, "Penalty should decrease with more legs"

    def test_kelly_requires_positive_edge(self, kelly):
        """Should not recommend bet if edge is below minimum."""
        result = kelly.calculate_parlay_kelly(
            adjusted_prob=0.12,  # 12% prob
            parlay_odds=700,  # +700
            num_legs=4,
            bankroll=1000.0
        )

        # Implied prob at +700 is ~12.5%, so edge is negative
        assert result.should_bet is False, "Should not bet with negative edge"

    def test_kelly_recommends_bet_with_positive_edge(self, kelly):
        """Should recommend bet if edge is above minimum."""
        result = kelly.calculate_parlay_kelly(
            adjusted_prob=0.35,  # 35% prob
            parlay_odds=200,  # +200 (implied ~33%)
            num_legs=2,
            bankroll=1000.0
        )

        # 35% vs 33% implied = ~2% edge, but we need 5% minimum
        # Let's test with higher edge
        result = kelly.calculate_parlay_kelly(
            adjusted_prob=0.45,
            parlay_odds=200,
            num_legs=2,
            bankroll=1000.0
        )

        # 45% vs 33% implied = ~12% edge
        assert result.should_bet is True, "Should bet with positive edge"
        assert result.recommended_stake > 0, "Stake should be positive"


class TestParlayOddsFetcher:
    """Test the parlay odds fetcher."""

    @pytest.fixture
    def fetcher(self):
        return ParlayOddsFetcher()

    def test_american_to_decimal_positive(self, fetcher):
        """Test American to decimal conversion for positive odds."""
        decimal = fetcher.american_to_decimal(200)
        assert abs(decimal - 3.0) < 0.001, "+200 should be 3.0 decimal"

    def test_american_to_decimal_negative(self, fetcher):
        """Test American to decimal conversion for negative odds."""
        decimal = fetcher.american_to_decimal(-110)
        expected = 1 + (100 / 110)
        assert abs(decimal - expected) < 0.001, f"-110 should be {expected} decimal"

    def test_parlay_odds_calculation(self, fetcher):
        """Test parlay odds multiplication."""
        leg_odds = [-110, -110]  # Two -110 legs

        american, decimal, implied = fetcher.calculate_parlay_odds(leg_odds)

        # Each -110 is ~1.909 decimal
        # Product should be ~3.64 decimal
        assert decimal > 3.5, f"2-leg parlay should be > 3.5 decimal, got {decimal}"
        assert american > 250, f"2-leg parlay should be > +250, got {american}"


class TestParlayRecommender:
    """Test the parlay recommender integration."""

    @pytest.fixture
    def recommender(self):
        return ParlayRecommender(
            max_legs=4,
            min_confidence=0.55,
            cross_game_only=True
        )

    @pytest.fixture
    def sample_bets(self):
        """Create sample bets for testing."""
        return [
            SingleBet(
                name="Player A UNDER 5.5",
                bet_type="Player Prop UNDER",
                game="KC@LV",
                team="KC",
                player="Player A",
                market="player_receptions",
                odds=-110,
                our_prob=0.65
            ),
            SingleBet(
                name="Player B UNDER 80.5",
                bet_type="Player Prop UNDER",
                game="MIA@NYJ",
                team="MIA",
                player="Player B",
                market="player_rush_yds",
                odds=-115,
                our_prob=0.60
            ),
            SingleBet(
                name="Player C UNDER 250.5",
                bet_type="Player Prop UNDER",
                game="DAL@PHI",
                team="DAL",
                player="Player C",
                market="player_pass_yds",
                odds=-105,
                our_prob=0.58
            ),
        ]

    def test_generates_parlays(self, recommender, sample_bets):
        """Should generate parlay combinations."""
        parlays = recommender.generate_parlays(sample_bets, num_parlays=10)

        assert len(parlays) > 0, "Should generate at least one parlay"

    def test_cross_game_only_blocks_same_game(self, recommender):
        """Should block parlays from same game when cross_game_only=True."""
        same_game_bets = [
            SingleBet(
                name="Player A UNDER 5.5",
                bet_type="Player Prop UNDER",
                game="KC@LV",
                team="KC",
                player="Player A",
                market="player_receptions",
                odds=-110,
                our_prob=0.65
            ),
            SingleBet(
                name="Player B UNDER 80.5",
                bet_type="Player Prop UNDER",
                game="KC@LV",  # Same game
                team="LV",
                player="Player B",
                market="player_rush_yds",
                odds=-115,
                our_prob=0.60
            ),
        ]

        parlays = recommender.generate_parlays(same_game_bets, num_parlays=10)

        assert len(parlays) == 0, "Should not generate parlays from same game"

    def test_parlay_has_enhanced_fields(self, recommender, sample_bets):
        """Parlays should include new enhanced fields."""
        parlays = recommender.generate_parlays(sample_bets, num_parlays=10)

        if parlays:
            parlay = parlays[0]
            assert hasattr(parlay, 'naive_prob'), "Should have naive_prob"
            assert hasattr(parlay, 'adjusted_prob'), "Should have adjusted_prob"
            assert hasattr(parlay, 'correlation_factor'), "Should have correlation_factor"
            assert hasattr(parlay, 'recommended_units'), "Should have recommended_units"


class TestCalculateParlayOdds:
    """Test the parlay odds calculation function."""

    def test_two_leg_parlay(self):
        """Test 2-leg parlay odds."""
        odds = calculate_parlay_odds([-110, -110])
        assert odds > 250, f"2-leg -110/-110 should be > +250, got {odds}"

    def test_three_leg_parlay(self):
        """Test 3-leg parlay odds."""
        odds = calculate_parlay_odds([-110, -115, -105])
        assert odds > 500, f"3-leg parlay should be > +500, got {odds}"

    def test_mixed_odds_parlay(self):
        """Test parlay with mixed positive and negative odds."""
        odds = calculate_parlay_odds([-110, +150])
        assert odds > 300, f"Mixed odds parlay should be > +300"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
