#!/usr/bin/env python3
"""
Tests for the centralized FeatureEngine.

These tests ensure:
1. No data leakage in feature calculations
2. Consistent calculations across all usage
3. Proper handling of edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.features.core import (
    FeatureEngine,
    get_feature_engine,
    calculate_trailing_stat,
    calculate_tprr,
    calculate_yards_per_route,
    estimate_routes_run,
    calculate_route_participation,
    calculate_all_trailing_stats,
)


class TestTrailingStats:
    """Test trailing stat calculations."""

    def test_trailing_stat_no_leakage(self):
        """Verify shift(1) is applied to prevent data leakage."""
        # Create test data
        df = pd.DataFrame({
            'player_norm': ['player_a'] * 5,
            'season': [2025] * 5,
            'week': [1, 2, 3, 4, 5],
            'receptions': [5, 6, 7, 8, 9],
        })

        engine = FeatureEngine(cache_enabled=False)

        # With no_leakage=True (default), week 5 should NOT see week 5's value
        trailing = engine.calculate_trailing_stat(
            df, 'receptions', no_leakage=True
        )

        # Week 1 should be NaN (no prior data)
        assert pd.isna(trailing.iloc[0]), "Week 1 should be NaN with no_leakage"

        # Week 5's trailing should be calculated from weeks 1-4, not including week 5
        # EWMA of [5, 6, 7, 8] ≈ 7.0 (weighted toward recent)
        week5_trailing = trailing.iloc[4]
        assert week5_trailing < 8.5, f"Week 5 trailing ({week5_trailing}) should not include week 5's value of 9"

    def test_trailing_stat_with_leakage_disabled(self):
        """Verify no shift when no_leakage=False."""
        df = pd.DataFrame({
            'player_norm': ['player_a'] * 5,
            'season': [2025] * 5,
            'week': [1, 2, 3, 4, 5],
            'receptions': [5, 6, 7, 8, 9],
        })

        engine = FeatureEngine(cache_enabled=False)

        # Without no_leakage, week 5 SHOULD see its own value
        trailing = engine.calculate_trailing_stat(
            df, 'receptions', no_leakage=False
        )

        # Week 5's value should include week 5 (EWMA weighted ~7.9)
        week5_trailing = trailing.iloc[4]
        # With EWMA span=4, the weight on week 5 is ~40%, so value is ~7.9
        assert week5_trailing > 7.5, "Without shift, trailing should include current week"

    def test_trailing_stat_multiple_players(self):
        """Verify trailing stats are calculated per-player."""
        df = pd.DataFrame({
            'player_norm': ['player_a', 'player_a', 'player_b', 'player_b'],
            'season': [2025, 2025, 2025, 2025],
            'week': [1, 2, 1, 2],
            'receptions': [5, 10, 20, 25],
        })

        engine = FeatureEngine(cache_enabled=False)
        trailing = engine.calculate_trailing_stat(df, 'receptions', no_leakage=True)

        # Player A week 2 should only see player A's data
        # Player B week 2 should only see player B's data
        player_a_week2 = trailing.iloc[1]
        player_b_week2 = trailing.iloc[3]

        assert player_a_week2 < 15, f"Player A trailing ({player_a_week2}) should be based on their data only"
        assert player_b_week2 > 15, f"Player B trailing ({player_b_week2}) should be based on their data only"


class TestDefenseEPA:
    """Test defense EPA calculations."""

    def test_defense_epa_no_leakage(self):
        """Verify defense EPA uses shift(1)."""
        # Create mock PBP data
        pbp = pd.DataFrame({
            'defteam': ['KC'] * 100 + ['KC'] * 100,
            'season': [2025] * 200,
            'week': [1] * 100 + [2] * 100,
            'play_type': ['run'] * 200,
            'epa': [-0.1] * 100 + [-0.2] * 100,  # Week 1: -0.1, Week 2: -0.2
        })

        engine = FeatureEngine(cache_enabled=False)
        def_epa = engine.calculate_defense_epa(pbp, 'run', no_leakage=True)

        # Week 1 should have NaN trailing (no prior data)
        week1 = def_epa[def_epa['week'] == 1]['trailing_def_epa'].values[0]
        assert pd.isna(week1), "Week 1 should have NaN trailing EPA"

        # Week 2's trailing should only see week 1's data
        week2 = def_epa[def_epa['week'] == 2]['trailing_def_epa'].values[0]
        assert abs(week2 - (-0.1)) < 0.01, f"Week 2 trailing EPA ({week2}) should be week 1's value (-0.1)"


class TestV12Features:
    """Test V12 feature calculations."""

    def test_calculate_v12_features(self):
        """Verify V12 feature calculations."""
        engine = FeatureEngine(cache_enabled=False)

        features = engine.calculate_v12_features(
            line=6.5,
            trailing_stat=5.0,
            player_under_rate=0.6,
            player_bias=-0.5,
            market_under_rate=0.55
        )

        # Check all 12 features are present
        expected_features = [
            'line_vs_trailing', 'line_level', 'line_in_sweet_spot',
            'player_under_rate', 'player_bias', 'market_under_rate',
            'LVT_x_player_tendency', 'LVT_x_player_bias', 'LVT_x_regime',
            'LVT_in_sweet_spot', 'market_bias_strength', 'player_market_aligned'
        ]

        for feat in expected_features:
            assert feat in features, f"Missing feature: {feat}"

        # Verify calculations
        # line_vs_trailing uses percentage method: (line - trailing) / trailing
        # (6.5 - 5.0) / 5.0 = 0.3
        assert features['line_vs_trailing'] == 0.3  # (6.5 - 5.0) / 5.0
        assert features['line_level'] == 6.5
        # line_in_sweet_spot uses smooth gaussian falloff, 6.5 is near center (4.5)
        assert 0.5 < features['line_in_sweet_spot'] <= 1.0  # 6.5 is in sweet spot range

    def test_player_under_rate_uses_prior_data_only(self):
        """Verify player_under_rate only uses prior weeks."""
        # Need enough data for rolling window (min_games=3 by default)
        hist_odds = pd.DataFrame({
            'player_norm': ['player_a'] * 6,
            'global_week': [1, 2, 3, 4, 5, 6],
            'under_hit': [1, 1, 1, 0, 0, 0],  # 50% overall
            'actual_stat': [5, 6, 7, 8, 9, 10],
            'line': [6, 6, 6, 6, 6, 6],
        })

        engine = FeatureEngine(cache_enabled=False)

        # Get under rate for week 7 (should only see weeks 1-6)
        under_rate = engine.calculate_player_under_rate(
            hist_odds, 'player_a', target_global_week=7
        )

        assert 0.4 <= under_rate <= 0.6, f"Under rate ({under_rate}) should be ~0.5"

        # Get under rate for week 4 (should only see weeks 1-3, all hits)
        under_rate_w4 = engine.calculate_player_under_rate(
            hist_odds, 'player_a', target_global_week=4
        )

        assert under_rate_w4 == 1.0, f"Week 4 under rate ({under_rate_w4}) should be 1.0 (weeks 1-3 all hit)"


class TestRouteMetrics:
    """Test route metric calculations."""

    def test_calculate_tprr(self):
        """Test TPRR calculation."""
        assert calculate_tprr(8, 30) == pytest.approx(0.267, rel=0.01)
        assert calculate_tprr(0, 30) == 0.0
        assert calculate_tprr(10, 0) == 0.0

    def test_calculate_yards_per_route(self):
        """Test Y/RR calculation."""
        assert calculate_yards_per_route(75, 30) == 2.5
        assert calculate_yards_per_route(0, 30) == 0.0
        assert calculate_yards_per_route(100, 0) == 0.0

    def test_estimate_routes_run(self):
        """Test routes run estimation."""
        # 50 snaps, 35 pass attempts out of 65 total plays
        routes = estimate_routes_run(50, 35, 65)
        assert routes == pytest.approx(26.92, rel=0.01)

        # Edge cases
        assert estimate_routes_run(50, 35, 0) == 0.0

    def test_route_participation(self):
        """Test route participation calculation."""
        assert calculate_route_participation(30, 40) == 0.75
        assert calculate_route_participation(50, 40) == 1.0  # Capped at 1.0
        assert calculate_route_participation(30, 0) == 0.0


class TestBatchCalculations:
    """Test batch feature calculations."""

    def test_calculate_all_trailing_stats(self):
        """Test batch trailing stats calculation."""
        df = pd.DataFrame({
            'player_norm': ['a', 'a', 'a', 'b', 'b', 'b'],
            'season': [2025] * 6,
            'week': [1, 2, 3, 1, 2, 3],
            'receptions': [5, 6, 7, 10, 11, 12],
            'receiving_yards': [50, 60, 70, 100, 110, 120],
        })

        result = calculate_all_trailing_stats(
            df,
            player_col='player_norm',
            stat_cols=['receptions', 'receiving_yards']
        )

        assert 'trailing_receptions' in result.columns
        assert 'trailing_receiving_yards' in result.columns

        # Verify no leakage (week 1 should be NaN)
        week1_a = result[(result['player_norm'] == 'a') & (result['week'] == 1)]['trailing_receptions'].values[0]
        assert pd.isna(week1_a), "Week 1 should have NaN trailing"


class TestLineVsTrailing:
    """Test LVT calculations."""

    def test_lvt_difference(self):
        """Test LVT with difference method."""
        engine = FeatureEngine(cache_enabled=False)

        assert engine.calculate_line_vs_trailing(6.5, 5.0, 'difference') == 1.5
        assert engine.calculate_line_vs_trailing(5.0, 6.5, 'difference') == -1.5

    def test_lvt_ratio(self):
        """Test LVT with ratio method."""
        engine = FeatureEngine(cache_enabled=False)

        assert engine.calculate_line_vs_trailing(6.5, 5.0, 'ratio') == 1.3
        assert engine.calculate_line_vs_trailing(5.0, 10.0, 'ratio') == 0.5

    def test_lvt_edge_cases(self):
        """Test LVT with edge cases."""
        engine = FeatureEngine(cache_enabled=False)

        assert engine.calculate_line_vs_trailing(6.5, 0, 'difference') == 0.0
        assert engine.calculate_line_vs_trailing(6.5, np.nan, 'ratio') == 0.0


class TestCaching:
    """Test caching behavior."""

    def test_cache_enabled(self):
        """Verify caching works correctly."""
        engine = FeatureEngine(cache_enabled=True)

        # First call should cache
        # (We can't easily test this without loading real data,
        #  but we can verify the cache dict exists)
        assert engine._pbp_cache == {}
        assert engine._defense_epa_cache == {}

    def test_clear_cache(self):
        """Test cache clearing."""
        engine = FeatureEngine(cache_enabled=True)
        engine._defense_epa_cache['test'] = pd.DataFrame()

        engine.clear_cache()

        assert engine._defense_epa_cache == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
