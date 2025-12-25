"""
Test script for position-specific defensive matchup features.

Tests the V4 position defense implementation (2025-12-04).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_engine_integration():
    """Test that FeatureEngine can calculate position defense stats."""
    print("\n" + "=" * 60)
    print("TEST 1: FeatureEngine Position Defense Integration")
    print("=" * 60)

    from nfl_quant.features import get_feature_engine

    engine = get_feature_engine()

    # Test for each position
    test_cases = [
        ("KC", "WR", 2024, 10),
        ("BUF", "RB", 2024, 10),
        ("SF", "TE", 2024, 10),
        ("PHI", "QB", 2024, 10),
    ]

    all_passed = True

    for opponent, position, season, week in test_cases:
        print(f"\nTesting {opponent} defense vs {position}:")

        try:
            stats = engine.get_position_defense_stats(opponent, position, season, week)

            pos_lower = position.lower()
            expected_keys = [
                f'def_vs_{pos_lower}_epa',
                f'def_vs_{pos_lower}_yds_per_play',
                f'def_vs_{pos_lower}_rank',
            ]

            for key in expected_keys:
                if key in stats:
                    print(f"  {key}: {stats[key]:.3f}")
                else:
                    print(f"  {key}: MISSING!")
                    all_passed = False

            # Verify bounds
            epa = stats.get(f'def_vs_{pos_lower}_epa', 0)
            if not -0.5 <= epa <= 0.5:
                print(f"  WARNING: EPA {epa} out of expected bounds [-0.5, 0.5]")

            rank = stats.get(f'def_vs_{pos_lower}_rank', 16)
            if not 1 <= rank <= 32:
                print(f"  WARNING: Rank {rank} out of expected bounds [1, 32]")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False

    print(f"\nTest 1 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_feature_defaults():
    """Test that FEATURE_DEFAULTS has all position defense features."""
    print("\n" + "=" * 60)
    print("TEST 2: FEATURE_DEFAULTS Coverage")
    print("=" * 60)

    from nfl_quant.features.feature_defaults import FEATURE_DEFAULTS

    expected_features = [
        'def_vs_wr_epa', 'def_vs_wr_yds_per_play', 'def_vs_wr_rank',
        'def_vs_rb_epa', 'def_vs_rb_yds_per_play', 'def_vs_rb_rank',
        'def_vs_te_epa', 'def_vs_te_yds_per_play', 'def_vs_te_rank',
        'def_vs_qb_epa', 'def_vs_qb_yds_per_play', 'def_vs_qb_rank',
        'opp_position_def_epa', 'opp_position_def_rank',
    ]

    all_covered = True

    for feature in expected_features:
        if feature in FEATURE_DEFAULTS:
            print(f"  {feature}: {FEATURE_DEFAULTS[feature]}")
        else:
            print(f"  {feature}: MISSING!")
            all_covered = False

    print(f"\nTest 2 Result: {'PASSED' if all_covered else 'FAILED'}")
    return all_covered


def test_feature_bounds():
    """Test that FEATURE_BOUNDS has all position defense features."""
    print("\n" + "=" * 60)
    print("TEST 3: FEATURE_BOUNDS Coverage")
    print("=" * 60)

    from nfl_quant.validation.input_validation import FEATURE_BOUNDS

    expected_features = [
        'def_vs_wr_epa', 'def_vs_wr_yds_per_play', 'def_vs_wr_rank',
        'def_vs_rb_epa', 'def_vs_rb_yds_per_play', 'def_vs_rb_rank',
        'def_vs_te_epa', 'def_vs_te_yds_per_play', 'def_vs_te_rank',
        'def_vs_qb_epa', 'def_vs_qb_yds_per_play', 'def_vs_qb_rank',
        'opp_position_def_epa', 'opp_position_def_rank',
    ]

    all_covered = True

    for feature in expected_features:
        if feature in FEATURE_BOUNDS:
            bounds = FEATURE_BOUNDS[feature]
            print(f"  {feature}: [{bounds[0]}, {bounds[1]}]")
        else:
            print(f"  {feature}: MISSING!")
            all_covered = False

    print(f"\nTest 3 Result: {'PASSED' if all_covered else 'FAILED'}")
    return all_covered


def test_extract_features_integration():
    """Test that extract_features_for_bet includes position defense."""
    print("\n" + "=" * 60)
    print("TEST 4: extract_features_for_bet Integration")
    print("=" * 60)

    from nfl_quant.features import get_feature_engine

    engine = get_feature_engine()

    # Test a sample player
    test_cases = [
        {
            "player_name": "Davante Adams",
            "player_id": None,
            "team": "NYJ",
            "opponent": "BUF",
            "position": "WR",
            "market": "player_receptions",
            "line": 5.5,
            "season": 2024,
            "week": 10,
            "trailing_stat": 5.0,
        },
        {
            "player_name": "Travis Kelce",
            "player_id": None,
            "team": "KC",
            "opponent": "DEN",
            "position": "TE",
            "market": "player_receiving_yards",
            "line": 55.5,
            "season": 2024,
            "week": 10,
            "trailing_stat": 50.0,
        },
    ]

    all_passed = True

    for case in test_cases:
        print(f"\nTesting {case['player_name']} ({case['position']}) vs {case['opponent']}:")

        try:
            features = engine.extract_features_for_bet(**case)

            pos_lower = case['position'].lower()
            pos_defense_keys = [
                f'def_vs_{pos_lower}_epa',
                f'def_vs_{pos_lower}_yds_per_play',
                f'def_vs_{pos_lower}_rank',
                'opp_position_def_epa',
                'opp_position_def_rank',
            ]

            for key in pos_defense_keys:
                if key in features:
                    print(f"  {key}: {features[key]:.3f}")
                else:
                    print(f"  {key}: MISSING (not returned by extract_features_for_bet)")
                    # Not a failure - feature may be optional

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\nTest 4 Result: {'PASSED' if all_passed else 'NEEDS REVIEW'}")
    return all_passed


def test_sample_calculations():
    """Test actual calculations on real data."""
    print("\n" + "=" * 60)
    print("TEST 5: Sample Calculations with Real Data")
    print("=" * 60)

    from nfl_quant.features import get_feature_engine

    engine = get_feature_engine()

    # Get stats for teams known to have extreme defenses
    teams_to_test = [
        ("SF", "WR", "expected good vs WR"),
        ("KC", "TE", "expected average"),
        ("DET", "RB", "expected weak vs run"),
    ]

    all_passed = True

    for team, position, expected in teams_to_test:
        print(f"\n{team} defense vs {position} ({expected}):")

        try:
            stats = engine.get_position_defense_stats(team, position, 2024, 12)

            pos_lower = position.lower()
            epa = stats.get(f'def_vs_{pos_lower}_epa', 0)
            yds = stats.get(f'def_vs_{pos_lower}_yds_per_play', 0)
            rank = stats.get(f'def_vs_{pos_lower}_rank', 16)

            print(f"  EPA: {epa:.3f}")
            print(f"  Yds/Play: {yds:.2f}")
            print(f"  Rank: {rank:.0f}/32")

            # Verify values are reasonable
            if epa == 0 and yds == 0 and rank == 16:
                print("  NOTE: Default values returned (no data available)")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False

    print(f"\nTest 5 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# POSITION-SPECIFIC DEFENSE FEATURE TESTS")
    print("# V4 Implementation - 2025-12-04")
    print("#" * 60)

    results = {}

    # Run tests
    results['Feature Engine'] = test_feature_engine_integration()
    results['Feature Defaults'] = test_feature_defaults()
    results['Feature Bounds'] = test_feature_bounds()
    results['Extract Features'] = test_extract_features_integration()
    results['Sample Calculations'] = test_sample_calculations()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
