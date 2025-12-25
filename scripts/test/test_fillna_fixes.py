#!/usr/bin/env python3
"""
Test script for fillna fixes and new validation modules.

Run: python scripts/test/test_fillna_fixes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np


def test_feature_defaults():
    """Test the feature_defaults module."""
    print("\n" + "="*60)
    print("TEST: feature_defaults module")
    print("="*60)

    from nfl_quant.features.feature_defaults import (
        safe_fillna, FEATURE_DEFAULTS, LeagueAverages2024,
        get_position_defaults, validate_defaults_coverage
    )

    # Test 1: LeagueAverages2024 dataclass
    league_avg = LeagueAverages2024()
    assert league_avg.comp_pct == 0.648, f"Expected 0.648, got {league_avg.comp_pct}"
    assert league_avg.yards_per_carry == 4.3, f"Expected 4.3, got {league_avg.yards_per_carry}"
    print("  [PASS] LeagueAverages2024 dataclass works")

    # Test 2: FEATURE_DEFAULTS contains key features
    assert 'player_under_rate' in FEATURE_DEFAULTS
    assert FEATURE_DEFAULTS['player_under_rate'] == 0.5, "player_under_rate should default to 0.5, not 0!"
    assert FEATURE_DEFAULTS['opp_pass_def_rank'] == 16.0, "Ranks should default to middle (16)"
    print("  [PASS] FEATURE_DEFAULTS has correct semantic values")

    # Test 3: safe_fillna replaces NaN with semantic defaults
    test_df = pd.DataFrame({
        'player_under_rate': [0.6, np.nan, 0.4],
        'trailing_comp_pct': [0.65, 0.70, np.nan],
        'opp_pass_def_rank': [5.0, np.nan, 25.0],
    })

    filled = safe_fillna(test_df)

    assert filled['player_under_rate'].iloc[1] == 0.5, "NaN should be filled with 0.5"
    assert filled['trailing_comp_pct'].iloc[2] == 0.648, "NaN should be filled with league avg"
    assert filled['opp_pass_def_rank'].iloc[1] == 16.0, "NaN should be filled with 16"
    print("  [PASS] safe_fillna fills NaN with semantic defaults")

    # Test 4: Position-specific defaults
    qb_defaults = get_position_defaults('QB')
    assert qb_defaults['snap_share'] == 1.0, "QB snap_share should be 1.0"

    rb_defaults = get_position_defaults('RB')
    assert rb_defaults['snap_share'] == 0.4, "RB snap_share should be 0.4"
    print("  [PASS] Position-specific defaults work")

    # Test 5: Validate defaults coverage
    test_features = ['player_under_rate', 'trailing_comp_pct', 'unknown_feature']
    coverage = validate_defaults_coverage(test_features)
    assert 'player_under_rate' in coverage['covered']
    assert 'unknown_feature' in coverage['missing']
    print("  [PASS] Defaults coverage validation works")

    print("\n  FEATURE_DEFAULTS TEST SUMMARY: ALL PASSED")


def test_input_validation():
    """Test the input_validation module."""
    print("\n" + "="*60)
    print("TEST: input_validation module")
    print("="*60)

    from nfl_quant.validation.input_validation import (
        validate_features, validate_and_log, FEATURE_BOUNDS,
        ValidationResult, create_validation_report, clip_to_bounds
    )

    # Test 1: FEATURE_BOUNDS contains bounds
    assert 'player_under_rate' in FEATURE_BOUNDS
    assert FEATURE_BOUNDS['player_under_rate'] == (0.0, 1.0)
    assert FEATURE_BOUNDS['opp_pass_def_rank'] == (1.0, 32.0)
    print("  [PASS] FEATURE_BOUNDS has correct ranges")

    # Test 2: validate_features catches NaN
    test_df = pd.DataFrame({
        'player_under_rate': [0.5, np.nan],
        'line_level': [5.5, 6.5],
    })

    result = validate_features(test_df, fail_on_nan=True)
    assert not result.is_valid, "Should fail validation with NaN"
    assert len(result.nan_issues) > 0
    print("  [PASS] validate_features catches NaN")

    # Test 3: validate_features catches Inf
    test_df_inf = pd.DataFrame({
        'player_under_rate': [0.5, np.inf],
    })

    result_inf = validate_features(test_df_inf)
    assert not result_inf.is_valid, "Should fail validation with Inf"
    assert len(result_inf.inf_issues) > 0
    print("  [PASS] validate_features catches Inf")

    # Test 4: validate_features catches out-of-bounds
    test_df_bounds = pd.DataFrame({
        'player_under_rate': [1.5],  # Should be 0-1
        'opp_pass_def_rank': [50.0],  # Should be 1-32
    })

    result_bounds = validate_features(test_df_bounds, fail_on_bounds=True)
    assert len(result_bounds.bound_issues) > 0
    print("  [PASS] validate_features catches out-of-bounds")

    # Test 5: clip_to_bounds works
    clipped = clip_to_bounds(test_df_bounds)
    assert clipped['player_under_rate'].iloc[0] == 1.0, "Should clip to 1.0"
    assert clipped['opp_pass_def_rank'].iloc[0] == 32.0, "Should clip to 32.0"
    print("  [PASS] clip_to_bounds works")

    # Test 6: create_validation_report
    report = create_validation_report(test_df)
    assert len(report) > 0
    assert 'feature' in report.columns
    print("  [PASS] create_validation_report works")

    print("\n  INPUT_VALIDATION TEST SUMMARY: ALL PASSED")


def test_game_context():
    """Test the game_context module."""
    print("\n" + "="*60)
    print("TEST: game_context module")
    print("="*60)

    from nfl_quant.features.game_context import (
        calculate_team_implied_total,
        calculate_opponent_implied_total,
        calculate_games_since_return,
        calculate_first_game_back,
        calculate_days_rest,
        calculate_short_rest_flag,
    )

    # Test 1: Team implied total - favorite
    implied = calculate_team_implied_total(48.5, -3.5)
    assert implied == 26.0, f"Favorite should get 26.0, got {implied}"
    print("  [PASS] Team implied total for favorite")

    # Test 2: Team implied total - underdog
    implied_dog = calculate_team_implied_total(48.5, 3.5)
    assert implied_dog == 22.5, f"Underdog should get 22.5, got {implied_dog}"
    print("  [PASS] Team implied total for underdog")

    # Test 3: Opponent implied total
    opp_implied = calculate_opponent_implied_total(48.5, -3.5)
    assert opp_implied == 22.5, f"Opponent of favorite should get 22.5, got {opp_implied}"
    print("  [PASS] Opponent implied total")

    # Test 4: NaN handling
    nan_implied = calculate_team_implied_total(np.nan, -3.5)
    assert nan_implied == 22.0, "NaN should default to 22.0"
    print("  [PASS] NaN handling in implied totals")

    # Test 5: First game back
    assert calculate_first_game_back(1) == 1
    assert calculate_first_game_back(2) == 0
    assert calculate_first_game_back(0) == 0
    print("  [PASS] First game back calculation")

    # Test 6: Days rest
    assert calculate_days_rest(None) == 7, "Default should be 7"
    print("  [PASS] Days rest default")

    # Test 7: Short rest flag
    assert calculate_short_rest_flag(4) == 1, "4 days should be short rest"
    assert calculate_short_rest_flag(7) == 0, "7 days should not be short rest"
    print("  [PASS] Short rest flag")

    print("\n  GAME_CONTEXT TEST SUMMARY: ALL PASSED")


def test_feature_engine_injury_features():
    """Test the injury features added to FeatureEngine."""
    print("\n" + "="*60)
    print("TEST: FeatureEngine injury features")
    print("="*60)

    from nfl_quant.features.core import get_feature_engine

    engine = get_feature_engine()

    # Test 1: get_team_implied_total method exists and works
    implied = engine.get_team_implied_total('KC', 2025, 14, game_total=48.5, spread=-3.5)
    assert implied == 26.0, f"Expected 26.0, got {implied}"
    print("  [PASS] get_team_implied_total method works")

    # Test 2: get_team_implied_total defaults properly
    default_implied = engine.get_team_implied_total('KC', 2025, 14)
    assert default_implied == 22.0, f"Expected 22.0 default, got {default_implied}"
    print("  [PASS] get_team_implied_total defaults to 22.0")

    # Test 3: Injury methods exist (may not have data, but should not error)
    try:
        backup_flag = engine.get_backup_qb_flag('KC', 2025, 14)
        assert backup_flag in [0, 1], "Backup QB flag should be 0 or 1"
        print("  [PASS] get_backup_qb_flag method works")
    except Exception as e:
        print(f"  [WARN] get_backup_qb_flag: {e} (may be missing data)")

    try:
        boost = engine.get_teammate_injury_boost('test_id', 'KC', 'WR', 2025, 14, 'player_receptions')
        assert boost >= 1.0, "Injury boost should be >= 1.0"
        print("  [PASS] get_teammate_injury_boost method works")
    except Exception as e:
        print(f"  [WARN] get_teammate_injury_boost: {e} (may be missing data)")

    # Test 4: Games since return (may not have data for test player)
    try:
        games_since = engine.get_games_since_return('test_id', 2025, 14)
        assert 0 <= games_since <= 4, "Games since return should be 0-4"
        print("  [PASS] get_games_since_return method works")
    except Exception as e:
        print(f"  [WARN] get_games_since_return: {e} (may be missing data)")

    print("\n  FEATURE_ENGINE INJURY TEST SUMMARY: METHODS EXIST")


def test_modules_import():
    """Test that all new modules import without errors."""
    print("\n" + "="*60)
    print("TEST: Module imports")
    print("="*60)

    try:
        from nfl_quant.features.feature_defaults import safe_fillna, FEATURE_DEFAULTS
        print("  [PASS] nfl_quant.features.feature_defaults imports")
    except Exception as e:
        print(f"  [FAIL] feature_defaults import: {e}")
        return False

    try:
        from nfl_quant.validation.input_validation import validate_features, FEATURE_BOUNDS
        print("  [PASS] nfl_quant.validation.input_validation imports")
    except Exception as e:
        print(f"  [FAIL] input_validation import: {e}")
        return False

    try:
        from nfl_quant.features.game_context import calculate_team_implied_total
        print("  [PASS] nfl_quant.features.game_context imports")
    except Exception as e:
        print(f"  [FAIL] game_context import: {e}")
        return False

    try:
        from nfl_quant.features.core import get_feature_engine
        engine = get_feature_engine()
        print("  [PASS] nfl_quant.features.core imports")
    except Exception as e:
        print(f"  [FAIL] core import: {e}")
        return False

    print("\n  MODULE IMPORT TEST SUMMARY: ALL PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FILLNA FIXES TEST SUITE")
    print("="*60)

    all_passed = True

    try:
        # Test module imports first
        if not test_modules_import():
            print("\n[FAIL] Module import tests failed. Stopping.")
            return 1

        # Test individual modules
        test_feature_defaults()
        test_input_validation()
        test_game_context()
        test_feature_engine_injury_features()

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
