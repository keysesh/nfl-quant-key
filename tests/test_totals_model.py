#!/usr/bin/env python3
"""
Unit Tests for Totals Model

Tests the V29 totals model for:
1. Sign convention correctness
2. Directional behavior
3. Weather adjustments
4. Regression test cases

Run with:
    pytest tests/test_totals_model.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.edges.game_line_edge import GameLineEdge


class TestTotalsSignConventions:
    """Test that sign conventions are correct for totals."""

    @pytest.fixture
    def edge(self):
        """Create a fresh GameLineEdge instance."""
        return GameLineEdge()

    @pytest.fixture
    def baseline_epa(self):
        """Baseline EPA dict representing average team."""
        return {
            'off_epa': 0.0,
            'def_epa_allowed': 0.0,
            'def_epa': 0.0,
            'pace': 60.0,
            'games': 6
        }

    def test_increasing_off_epa_increases_total(self, edge, baseline_epa):
        """Increasing offensive EPA should increase model total."""
        # Baseline: both teams at 0 EPA
        home_base = baseline_epa.copy()
        away_base = baseline_epa.copy()

        _, _, _, debug_base = edge.calculate_total_edge(
            home_base, away_base, market_total=45.0
        )
        base_total = debug_base['clipped_model_total']

        # Increase home team's offensive EPA
        home_good_offense = baseline_epa.copy()
        home_good_offense['off_epa'] = 0.10  # Good offense

        _, _, _, debug_good = edge.calculate_total_edge(
            home_good_offense, away_base, market_total=45.0
        )
        good_offense_total = debug_good['clipped_model_total']

        assert good_offense_total > base_total, (
            f"Increasing off_epa should increase total: "
            f"base={base_total:.1f}, with good offense={good_offense_total:.1f}"
        )

    def test_increasing_def_epa_allowed_increases_total(self, edge, baseline_epa):
        """Increasing defensive EPA allowed (bad defense) should increase total."""
        home_base = baseline_epa.copy()
        away_base = baseline_epa.copy()

        _, _, _, debug_base = edge.calculate_total_edge(
            home_base, away_base, market_total=45.0
        )
        base_total = debug_base['clipped_model_total']

        # Increase home team's def_epa_allowed (bad defense)
        home_bad_defense = baseline_epa.copy()
        home_bad_defense['def_epa_allowed'] = 0.10  # Bad defense
        home_bad_defense['def_epa'] = 0.10

        _, _, _, debug_bad_def = edge.calculate_total_edge(
            home_bad_defense, away_base, market_total=45.0
        )
        bad_defense_total = debug_bad_def['clipped_model_total']

        assert bad_defense_total > base_total, (
            f"Increasing def_epa_allowed (bad defense) should increase total: "
            f"base={base_total:.1f}, with bad defense={bad_defense_total:.1f}"
        )

    def test_good_defense_decreases_total(self, edge, baseline_epa):
        """Good defense (negative def_epa_allowed) should decrease total."""
        home_base = baseline_epa.copy()
        away_base = baseline_epa.copy()

        _, _, _, debug_base = edge.calculate_total_edge(
            home_base, away_base, market_total=45.0
        )
        base_total = debug_base['clipped_model_total']

        # Good defense = negative def_epa_allowed
        home_good_defense = baseline_epa.copy()
        home_good_defense['def_epa_allowed'] = -0.10  # Good defense
        home_good_defense['def_epa'] = -0.10

        _, _, _, debug_good_def = edge.calculate_total_edge(
            home_good_defense, away_base, market_total=45.0
        )
        good_defense_total = debug_good_def['clipped_model_total']

        assert good_defense_total < base_total, (
            f"Good defense (negative def_epa_allowed) should decrease total: "
            f"base={base_total:.1f}, with good defense={good_defense_total:.1f}"
        )


class TestTotalsWeatherAdjustments:
    """Test weather adjustments for totals."""

    @pytest.fixture
    def edge(self):
        return GameLineEdge()

    @pytest.fixture
    def average_epa(self):
        return {
            'off_epa': 0.0,
            'def_epa_allowed': 0.0,
            'def_epa': 0.0,
            'pace': 60.0,
            'games': 6
        }

    def test_dome_increases_total(self, edge, average_epa):
        """Dome games should project higher totals."""
        home = average_epa.copy()
        away = average_epa.copy()

        # Outdoor game
        _, _, _, debug_outdoor = edge.calculate_total_edge(
            home, away, market_total=45.0,
            is_dome=False, temperature=50.0
        )
        outdoor_total = debug_outdoor['clipped_model_total']

        # Dome game
        _, _, _, debug_dome = edge.calculate_total_edge(
            home, away, market_total=45.0,
            is_dome=True, temperature=72.0
        )
        dome_total = debug_dome['clipped_model_total']

        assert dome_total > outdoor_total, (
            f"Dome should increase total: outdoor={outdoor_total:.1f}, dome={dome_total:.1f}"
        )

    def test_cold_decreases_total(self, edge, average_epa):
        """Cold weather should decrease totals."""
        home = average_epa.copy()
        away = average_epa.copy()

        # Warm outdoor game
        _, _, _, debug_warm = edge.calculate_total_edge(
            home, away, market_total=45.0,
            is_dome=False, temperature=60.0
        )
        warm_total = debug_warm['clipped_model_total']

        # Cold game
        _, _, _, debug_cold = edge.calculate_total_edge(
            home, away, market_total=45.0,
            is_dome=False, temperature=25.0  # Below 32F
        )
        cold_total = debug_cold['clipped_model_total']

        assert cold_total < warm_total, (
            f"Cold weather should decrease total: warm={warm_total:.1f}, cold={cold_total:.1f}"
        )


class TestTotalsRegressionCases:
    """Regression tests for specific scenarios."""

    @pytest.fixture
    def edge(self):
        return GameLineEdge()

    def test_high_scoring_environment(self, edge):
        """
        REGRESSION TEST: Two good offenses + two bad defenses
        should result in HIGHER projected total than average.

        This was the original bug where the differential formula
        would cancel out in high-scoring environments.
        """
        # Two good offenses, two bad defenses
        home_epa = {
            'off_epa': 0.10,       # Good offense
            'def_epa_allowed': 0.10,  # Bad defense (allows points)
            'def_epa': 0.10,
            'pace': 65.0,          # Slightly above average pace
            'games': 6
        }
        away_epa = {
            'off_epa': 0.10,       # Good offense
            'def_epa_allowed': 0.10,  # Bad defense
            'def_epa': 0.10,
            'pace': 65.0,
            'games': 6
        }

        # Average teams
        avg_epa = {
            'off_epa': 0.0,
            'def_epa_allowed': 0.0,
            'def_epa': 0.0,
            'pace': 60.0,
            'games': 6
        }

        # Get model totals
        _, _, _, debug_high = edge.calculate_total_edge(
            home_epa, away_epa, market_total=50.0
        )
        high_scoring_total = debug_high['clipped_model_total']

        _, _, _, debug_avg = edge.calculate_total_edge(
            avg_epa, avg_epa.copy(), market_total=45.0
        )
        average_total = debug_avg['clipped_model_total']

        assert high_scoring_total > average_total, (
            f"High scoring environment (good O, bad D) should project higher: "
            f"high={high_scoring_total:.1f}, avg={average_total:.1f}"
        )

        # The total should be notably higher, not just marginally
        diff = high_scoring_total - average_total
        assert diff > 2.0, (
            f"High scoring environment should be at least 2 pts higher: "
            f"diff={diff:.1f}"
        )

    def test_low_scoring_environment(self, edge):
        """Good defenses + bad offenses should project lower total."""
        # Two bad offenses, two good defenses
        home_epa = {
            'off_epa': -0.10,      # Bad offense
            'def_epa_allowed': -0.10,  # Good defense (limits points)
            'def_epa': -0.10,
            'pace': 55.0,          # Slow pace
            'games': 6
        }
        away_epa = {
            'off_epa': -0.10,
            'def_epa_allowed': -0.10,
            'def_epa': -0.10,
            'pace': 55.0,
            'games': 6
        }

        avg_epa = {
            'off_epa': 0.0,
            'def_epa_allowed': 0.0,
            'def_epa': 0.0,
            'pace': 60.0,
            'games': 6
        }

        _, _, _, debug_low = edge.calculate_total_edge(
            home_epa, away_epa, market_total=40.0
        )
        low_scoring_total = debug_low['clipped_model_total']

        _, _, _, debug_avg = edge.calculate_total_edge(
            avg_epa, avg_epa.copy(), market_total=45.0
        )
        average_total = debug_avg['clipped_model_total']

        assert low_scoring_total < average_total, (
            f"Low scoring environment should project lower: "
            f"low={low_scoring_total:.1f}, avg={average_total:.1f}"
        )

    def test_pace_affects_total(self, edge):
        """Higher pace should result in higher projected total."""
        # High pace teams
        high_pace = {
            'off_epa': 0.0,
            'def_epa_allowed': 0.0,
            'def_epa': 0.0,
            'pace': 70.0,  # High pace
            'games': 6
        }

        # Low pace teams
        low_pace = {
            'off_epa': 0.0,
            'def_epa_allowed': 0.0,
            'def_epa': 0.0,
            'pace': 50.0,  # Low pace
            'games': 6
        }

        _, _, _, debug_high = edge.calculate_total_edge(
            high_pace, high_pace.copy(), market_total=50.0
        )
        high_pace_total = debug_high['clipped_model_total']

        _, _, _, debug_low = edge.calculate_total_edge(
            low_pace, low_pace.copy(), market_total=40.0
        )
        low_pace_total = debug_low['clipped_model_total']

        assert high_pace_total > low_pace_total, (
            f"Higher pace should project higher total: "
            f"high pace={high_pace_total:.1f}, low pace={low_pace_total:.1f}"
        )


class TestTotalsEdgeDirection:
    """Test that edge direction is correct."""

    @pytest.fixture
    def edge(self):
        return GameLineEdge()

    def test_over_direction_when_model_higher(self, edge):
        """When model total > market total, should recommend OVER."""
        # High-scoring teams
        high_epa = {
            'off_epa': 0.15,
            'def_epa_allowed': 0.10,
            'def_epa': 0.10,
            'pace': 65.0,
            'games': 6
        }

        # Use low market total to ensure model > market
        direction, edge_pct, confidence, debug = edge.calculate_total_edge(
            high_epa, high_epa.copy(), market_total=38.0  # Low market
        )

        if direction is not None:
            assert direction == 'OVER', (
                f"Model total ({debug['clipped_model_total']:.1f}) > "
                f"market (38.0) should be OVER, got {direction}"
            )

    def test_under_direction_when_model_lower(self, edge):
        """When model total < market total, should recommend UNDER."""
        # Low-scoring teams
        low_epa = {
            'off_epa': -0.15,
            'def_epa_allowed': -0.10,
            'def_epa': -0.10,
            'pace': 55.0,
            'games': 6
        }

        # Use high market total to ensure model < market
        direction, edge_pct, confidence, debug = edge.calculate_total_edge(
            low_epa, low_epa.copy(), market_total=55.0  # High market
        )

        if direction is not None:
            assert direction == 'UNDER', (
                f"Model total ({debug['clipped_model_total']:.1f}) < "
                f"market (55.0) should be UNDER, got {direction}"
            )


class TestTotalsDebugInfo:
    """Test that debug info is correctly populated."""

    @pytest.fixture
    def edge(self):
        return GameLineEdge()

    def test_debug_info_contains_required_fields(self, edge):
        """Debug info should contain all required fields."""
        epa = {
            'off_epa': 0.05,
            'def_epa_allowed': 0.02,
            'def_epa': 0.02,
            'pace': 62.0,
            'games': 6
        }

        _, _, _, debug = edge.calculate_total_edge(
            epa, epa.copy(), market_total=45.0,
            is_dome=True, temperature=72.0, wind_speed=5.0
        )

        required_fields = [
            'plays_total', 'ppp_baseline', 'ppp_epa_adj', 'ppp_weather_adj',
            'combined_off_epa', 'combined_def_epa_allowed',
            'raw_model_total', 'clipped_model_total', 'market_total',
            'edge_pts', 'was_clipped', 'is_dome', 'temperature', 'wind_speed'
        ]

        for field in required_fields:
            assert field in debug, f"Debug info missing field: {field}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
