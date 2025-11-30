#!/usr/bin/env python3
"""
NFL QUANT - Game Lines System Integration Test

Validates that all components of the game lines system are working correctly:
1. Master file exists and has correct format
2. Centralized loader works properly
3. Backtest can load and use data
4. Coverage is complete for target weeks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from nfl_quant.data.game_lines_loader import (
    load_game_lines,
    get_coverage_summary,
    validate_game_lines_format,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_master_file_exists():
    """Test that master file exists."""
    print("\n" + "="*70)
    print("TEST 1: Master File Existence")
    print("="*70)

    master_file = PROJECT_ROOT / "data/historical/game_lines_master_2025.csv"

    if master_file.exists():
        print(f"✅ Master file exists: {master_file}")
        file_size = master_file.stat().st_size / 1024  # KB
        print(f"   File size: {file_size:.1f} KB")
        return True
    else:
        print(f"❌ Master file not found: {master_file}")
        return False


def test_master_file_format():
    """Test that master file has correct format."""
    print("\n" + "="*70)
    print("TEST 2: Master File Format")
    print("="*70)

    try:
        df = load_game_lines(season=2025, source="master")

        print(f"✅ Loaded {len(df)} records from master file")
        print(f"   Unique games: {df['game_id'].nunique()}")
        print(f"   Weeks covered: {sorted(df['week'].unique())}")
        print(f"   Markets: {df['market'].value_counts().to_dict()}")

        # Validate format
        validate_game_lines_format(df)
        print(f"✅ Format validation passed")

        # Check required columns
        required_cols = ['season', 'week', 'home_team', 'away_team', 'market', 'side', 'price']
        missing = set(required_cols) - set(df.columns)
        if missing:
            print(f"❌ Missing columns: {missing}")
            return False

        print(f"✅ All required columns present")
        return True

    except Exception as e:
        print(f"❌ Error loading master file: {e}")
        return False


def test_loader_functions():
    """Test all loader functions work correctly."""
    print("\n" + "="*70)
    print("TEST 3: Loader Functions")
    print("="*70)

    tests_passed = 0
    total_tests = 4

    # Test 1: Load all weeks
    try:
        df = load_game_lines(season=2025)
        print(f"✅ Load all weeks: {len(df)} records")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Load all weeks failed: {e}")

    # Test 2: Load specific week
    try:
        df = load_game_lines(season=2025, weeks=10)
        print(f"✅ Load week 10: {len(df)} records")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Load week 10 failed: {e}")

    # Test 3: Load multiple weeks
    try:
        df = load_game_lines(season=2025, weeks=[8, 9, 10])
        print(f"✅ Load weeks 8-10: {len(df)} records")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Load weeks 8-10 failed: {e}")

    # Test 4: Get coverage summary
    try:
        summary = get_coverage_summary(season=2025, source="master")
        print(f"✅ Coverage summary: {len(summary)} weeks")
        print(f"   Sample: {summary.head(3).to_dict('records')}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Coverage summary failed: {e}")

    print(f"\n📊 Loader tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_coverage_completeness():
    """Test that coverage is complete for expected weeks."""
    print("\n" + "="*70)
    print("TEST 4: Coverage Completeness")
    print("="*70)

    try:
        summary = get_coverage_summary(season=2025, source="master")

        # Check each week has games
        weeks_with_zero_games = summary[summary['games_count'] == 0]
        if len(weeks_with_zero_games) > 0:
            print(f"⚠️  Weeks with no games: {weeks_with_zero_games['week'].tolist()}")
        else:
            print(f"✅ All weeks have games")

        # Check all weeks have all 3 markets
        complete_weeks = []
        incomplete_weeks = []

        for _, row in summary.iterrows():
            week = row['week']
            markets = row['markets']
            required_markets = {'spread', 'total', 'moneyline'}

            if required_markets.issubset(set(markets)):
                complete_weeks.append(week)
            else:
                incomplete_weeks.append((week, set(markets)))

        print(f"✅ Complete weeks (all 3 markets): {len(complete_weeks)}")
        if incomplete_weeks:
            print(f"⚠️  Incomplete weeks:")
            for week, markets in incomplete_weeks:
                missing = required_markets - markets
                print(f"     Week {week}: missing {missing}")

        return len(incomplete_weeks) == 0

    except Exception as e:
        print(f"❌ Coverage check failed: {e}")
        return False


def test_backtest_integration():
    """Test that backtest can load and use the data."""
    print("\n" + "="*70)
    print("TEST 5: Backtest Integration")
    print("="*70)

    try:
        # Load schedule
        schedules_file = PROJECT_ROOT / "data/nflverse/schedules.parquet"
        if not schedules_file.exists():
            print(f"⚠️  Schedule file not found: {schedules_file}")
            return False

        schedules = pd.read_parquet(schedules_file)
        test_weeks = [8, 9, 10]
        games = schedules[
            (schedules['season'] == 2025) &
            (schedules['week'].isin(test_weeks))
        ]

        print(f"✅ Loaded schedule: {len(games)} games for weeks {test_weeks}")

        # Load odds
        odds_df = load_game_lines(season=2025, weeks=test_weeks)
        print(f"✅ Loaded odds: {len(odds_df)} records")

        # Test merge
        unique_games_in_odds = odds_df[['home_team', 'away_team', 'week']].drop_duplicates()
        print(f"   Unique game matchups in odds: {len(unique_games_in_odds)}")

        # Count how many games can be matched
        matched_count = 0
        for _, game in games.iterrows():
            match = odds_df[
                (odds_df['week'] == game['week']) &
                (odds_df['home_team'] == game['home_team']) &
                (odds_df['away_team'] == game['away_team'])
            ]
            if len(match) > 0:
                matched_count += 1

        match_rate = (matched_count / len(games)) * 100
        print(f"✅ Match rate: {matched_count}/{len(games)} ({match_rate:.1f}%)")

        if match_rate >= 95:
            print(f"✅ Excellent match rate!")
            return True
        elif match_rate >= 80:
            print(f"⚠️  Good match rate, but could be better")
            return True
        else:
            print(f"❌ Low match rate - needs investigation")
            return False

    except Exception as e:
        print(f"❌ Backtest integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("NFL QUANT - GAME LINES SYSTEM INTEGRATION TESTS")
    print("="*70)

    results = []

    results.append(("Master File Exists", test_master_file_exists()))
    results.append(("Master File Format", test_master_file_format()))
    results.append(("Loader Functions", test_loader_functions()))
    results.append(("Coverage Completeness", test_coverage_completeness()))
    results.append(("Backtest Integration", test_backtest_integration()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")

    print("="*70)
    print(f"OVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All systems operational!")
        return True
    else:
        print("⚠️  Some tests failed - review output above")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
