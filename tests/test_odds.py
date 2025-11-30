"""Tests for odds and Kelly sizing utilities."""

import pytest

from nfl_quant.utils.odds import OddsEngine


class TestOddsConversion:
    """Test odds conversion functions."""

    def test_american_to_implied_prob_negative_odds(self) -> None:
        """Test conversion of negative American odds."""
        # -110 should be ~0.524
        prob = OddsEngine.american_to_implied_prob(-110)
        assert 0.52 < prob < 0.53

    def test_american_to_implied_prob_positive_odds(self) -> None:
        """Test conversion of positive American odds."""
        # +110 should be ~0.476
        prob = OddsEngine.american_to_implied_prob(110)
        assert 0.47 < prob < 0.48

    def test_american_to_implied_prob_even_odds(self) -> None:
        """Test -100 odds = 0.5 implied."""
        prob = OddsEngine.american_to_implied_prob(-100)
        assert prob == 0.5

    def test_implied_prob_to_american_roundtrip(self) -> None:
        """Test round-trip conversion."""
        original_odds = -110
        prob = OddsEngine.american_to_implied_prob(original_odds)
        converted_back = OddsEngine.implied_prob_to_american(prob)
        assert converted_back == original_odds


class TestEVCalculation:
    """Test expected value calculations."""

    def test_ev_positive(self) -> None:
        """Test positive EV scenario."""
        # Win prob 55%, implied prob 50%
        ev = OddsEngine.calculate_ev(0.55, 0.50)
        assert ev > 0

    def test_ev_negative(self) -> None:
        """Test negative EV scenario."""
        # Win prob 40%, implied prob 50%
        ev = OddsEngine.calculate_ev(0.40, 0.50)
        assert ev < 0

    def test_ev_zero(self) -> None:
        """Test break-even EV."""
        # Win prob = implied prob
        ev = OddsEngine.calculate_ev(0.50, 0.50)
        assert ev == 0.0


class TestKellySizing:
    """Test Kelly Criterion calculations."""

    def test_kelly_positive_ev(self) -> None:
        """Test Kelly sizing with positive EV."""
        kelly = OddsEngine.kelly_fraction(0.60, 0.50, kelly_fraction=1.0)
        assert kelly > 0

    def test_kelly_negative_ev(self) -> None:
        """Test Kelly sizing with negative EV."""
        kelly = OddsEngine.kelly_fraction(0.40, 0.50, kelly_fraction=1.0)
        assert kelly <= 0

    def test_kelly_fraction_scaling(self) -> None:
        """Test Kelly fraction scaling."""
        kelly_full = OddsEngine.kelly_fraction(0.60, 0.50, kelly_fraction=1.0)
        kelly_half = OddsEngine.kelly_fraction(0.60, 0.50, kelly_fraction=0.5)
        assert kelly_half == kelly_full * 0.5

    def test_kelly_bounds(self) -> None:
        """Test Kelly stays in reasonable bounds."""
        kelly = OddsEngine.kelly_fraction(0.99, 0.01, kelly_fraction=1.0)
        assert kelly <= 0.25  # Should be capped


class TestBetSizing:
    """Test complete bet sizing workflow."""

    def test_bet_sizing_returns_valid_object(self) -> None:
        """Test bet sizing returns valid BetSizing object."""
        engine = OddsEngine()
        bet = engine.size_bet(
            game_id="test_1",
            side="home_spread",
            american_odds=-110,
            win_prob=0.55,
            bankroll=10000,
            kelly_fraction=0.5,
            max_bet_pct=5.0,
        )
        assert bet.game_id == "test_1"
        assert bet.american_odds == -110
        assert 0 <= bet.kelly_pct <= 100
        assert bet.suggested_bet_amount >= 0

    def test_bet_sizing_respects_max_pct(self) -> None:
        """Test bet sizing respects max bet percentage."""
        engine = OddsEngine()
        bankroll = 10000
        max_pct = 2.0
        bet = engine.size_bet(
            game_id="test_1",
            side="home_spread",
            american_odds=-110,
            win_prob=0.95,  # Very high win prob
            bankroll=bankroll,
            kelly_fraction=1.0,  # Full Kelly
            max_bet_pct=max_pct,
        )
        assert bet.suggested_bet_amount <= bankroll * (max_pct / 100)

    def test_no_bet_on_negative_ev(self) -> None:
        """Test no bet is suggested on negative EV."""
        engine = OddsEngine()
        bet = engine.size_bet(
            game_id="test_1",
            side="home_spread",
            american_odds=-110,
            win_prob=0.40,  # Below implied prob
            bankroll=10000,
            kelly_fraction=0.5,
            max_bet_pct=5.0,
        )
        assert bet.suggested_bet_amount == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
