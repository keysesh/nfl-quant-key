#!/usr/bin/env python3
"""
Smoke Test: Injury Loader + Policy

Tests that:
1. get_injuries returns contract columns and valid statuses
2. apply_injury_policy blocks OUT/DOUBTFUL correctly and never boosts
3. Failure path triggers strict/conservative behaviors

Usage:
    python scripts/debug/smoke_injuries.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_injury_loader():
    """Test the injury loader."""
    print("\n" + "="*60)
    print("TEST 1: Injury Loader")
    print("="*60)

    from nfl_quant.data.injury_loader import (
        get_injuries,
        InjuryDataError,
        get_injury_freshness,
        REQUIRED_COLUMNS,
    )

    # Check freshness
    freshness = get_injury_freshness()
    print(f"\nCache status: {freshness.get('status', 'UNKNOWN')}")
    if 'hours_old' in freshness:
        print(f"  Hours old: {freshness['hours_old']}")
    if 'record_count' in freshness:
        print(f"  Records: {freshness['record_count']}")

    # Load injuries (force refresh to test API)
    try:
        df = get_injuries(refresh=True)
        print(f"\n✅ Loaded {len(df)} injury records")
    except InjuryDataError as e:
        print(f"\n❌ InjuryDataError: {e}")
        return False

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        print(f"\n❌ Missing required columns: {missing_cols}")
        return False
    print(f"✅ All required columns present: {REQUIRED_COLUMNS}")

    # Check status values
    if len(df) > 0:
        valid_statuses = {'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'ACTIVE', 'UNKNOWN'}
        statuses = set(df['status'].unique())
        invalid = statuses - valid_statuses
        if invalid:
            print(f"\n❌ Invalid status values: {invalid}")
            return False
        print(f"✅ Status values valid: {statuses}")

        # Check risk scores
        if df['risk_score'].min() < 0 or df['risk_score'].max() > 1:
            print(f"\n❌ Risk score out of range [0,1]")
            return False
        print(f"✅ Risk scores in valid range [0, 1]")

        # Show sample
        print(f"\nSample injuries:")
        for _, row in df.head(5).iterrows():
            print(f"  {row['player_name']} ({row['team']}): {row['status']} (risk={row['risk_score']:.2f})")

    return True


def test_injury_policy():
    """Test the injury policy module."""
    print("\n" + "="*60)
    print("TEST 2: Injury Policy")
    print("="*60)

    from nfl_quant.policy.injury_policy import (
        apply_injury_policy,
        InjuryMode,
        get_injury_summary,
    )
    from nfl_quant.data.injury_loader import get_injuries, InjuryDataError

    # Create sample recommendations
    sample_recs = pd.DataFrame([
        {'player': 'Josh Allen', 'team': 'BUF', 'direction': 'OVER', 'market': 'player_pass_yds', 'combined_confidence': 0.65},
        {'player': 'Jalen Hurts', 'team': 'PHI', 'direction': 'UNDER', 'market': 'player_rush_yds', 'combined_confidence': 0.62},
        {'player': 'Test Player OUT', 'team': 'NYG', 'direction': 'OVER', 'market': 'player_receptions', 'combined_confidence': 0.58},
        {'player': 'Test Player UNDER', 'team': 'DAL', 'direction': 'UNDER', 'market': 'player_receptions', 'combined_confidence': 0.60},
    ])

    # Create test injury data
    test_injuries = pd.DataFrame([
        {'source': 'test', 'updated_at': datetime.now().isoformat(), 'player_key': '1',
         'player_name': 'Test Player OUT', 'team': 'NYG', 'pos': 'WR', 'status': 'OUT', 'risk_score': 1.0},
        {'source': 'test', 'updated_at': datetime.now().isoformat(), 'player_key': '2',
         'player_name': 'Jalen Hurts', 'team': 'PHI', 'pos': 'QB', 'status': 'QUESTIONABLE', 'risk_score': 0.5},
    ])

    print(f"\nInput: {len(sample_recs)} recommendations")
    print(f"Injuries: {len(test_injuries)} players")

    # Apply policy
    result = apply_injury_policy(sample_recs, test_injuries, mode=InjuryMode.CONSERVATIVE)

    print(f"\nOutput: {len(result)} recommendations (blocked {len(sample_recs) - len(result)})")

    # Check that OUT player was blocked
    out_player_in_result = 'Test Player OUT' in result['player'].values
    if out_player_in_result:
        print(f"\n❌ OUT player should have been blocked!")
        return False
    print(f"✅ OUT player was blocked correctly")

    # Check that confidence was never boosted
    for idx, row in result.iterrows():
        orig_conf = sample_recs[sample_recs['player'] == row['player']]['combined_confidence'].values[0]
        if row['combined_confidence'] > orig_conf:
            print(f"\n❌ SAFETY VIOLATION: Confidence boosted from {orig_conf} to {row['combined_confidence']}")
            return False
    print(f"✅ No confidence boosts (safety check passed)")

    # Get summary
    summary = get_injury_summary(result)
    print(f"\nPolicy summary: {summary}")

    # Test CONSERVATIVE mode with missing injury data
    print("\n--- Testing CONSERVATIVE mode with missing data ---")
    result_no_injuries = apply_injury_policy(sample_recs, None, mode=InjuryMode.CONSERVATIVE)
    over_bets = len(result_no_injuries[result_no_injuries['direction'] == 'OVER'])
    if over_bets > 0:
        print(f"❌ CONSERVATIVE mode should block all OVERs when no injury data")
        return False
    print(f"✅ CONSERVATIVE mode blocked all OVERs when no injury data")

    # Test STRICT mode
    print("\n--- Testing STRICT mode with missing data ---")
    try:
        apply_injury_policy(sample_recs, None, mode=InjuryMode.STRICT)
        print(f"❌ STRICT mode should raise ValueError when no injury data")
        return False
    except ValueError as e:
        print(f"✅ STRICT mode raised ValueError: {e}")

    return True


def test_depth_chart_loader():
    """Test the depth chart loader."""
    print("\n" + "="*60)
    print("TEST 3: Depth Chart Loader")
    print("="*60)

    from nfl_quant.data.depth_chart_loader import (
        get_depth_charts,
        get_starters,
        get_depth_chart_freshness,
    )

    # Check freshness
    freshness = get_depth_chart_freshness()
    print(f"\nCache status:")
    for name, status in freshness.items():
        print(f"  {name}: {status.get('status', 'UNKNOWN')}")

    # Load depth charts for week 18
    print(f"\n--- Loading Week 18 depth charts ---")
    try:
        df = get_depth_charts(season=2025, week=18)
        print(f"✅ Loaded {len(df):,} rows for {df['team'].nunique()} teams")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # Check coverage
    if df['team'].nunique() < 30:
        print(f"❌ Insufficient team coverage: {df['team'].nunique()}/32")
        return False
    print(f"✅ Team coverage: {df['team'].nunique()}/32")

    # Test starters
    print(f"\n--- Testing starters for BUF and PHI ---")
    for team in ['BUF', 'PHI']:
        starters = get_starters(team=team, week=18)
        print(f"\n{team}:")
        for pos, player in starters.items():
            print(f"  {pos}: {player}")

    return True


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("NFL QUANT - INJURY & DEPTH CHART SMOKE TEST")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    results = {}

    # Test 1: Injury Loader
    results['injury_loader'] = test_injury_loader()

    # Test 2: Injury Policy
    results['injury_policy'] = test_injury_policy()

    # Test 3: Depth Chart Loader
    results['depth_chart_loader'] = test_depth_chart_loader()

    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
