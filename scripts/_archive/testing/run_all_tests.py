#!/usr/bin/env python3
"""
Full Framework Test Runner
===========================

Master script that runs all testing tools:
1. Data completeness check
2. Sleeper vs NFLverse comparison
3. Full framework end-to-end test
4. System validation

Usage:
    python scripts/testing/run_all_tests.py [week] [--season YEAR] [--dry-run]

Example:
    python scripts/testing/run_all_tests.py 11 --season 2025 --dry-run
"""

import sys
from pathlib import Path
import subprocess
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_command(script: str, args: list = None, description: str = ""):
    """Run a test script and return success status."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)

    cmd = [sys.executable, script] + (args or [])
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, check=False)
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {description}")
        return success
    except Exception as e:
        print(f"\n❌ ERROR: {description} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all framework tests')
    parser.add_argument('week', type=int, help='Week number to test')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (skip actual execution)')
    parser.add_argument('--skip-data-check', action='store_true', help='Skip data completeness check')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip Sleeper vs NFLverse comparison')
    parser.add_argument('--skip-framework', action='store_true', help='Skip full framework test')
    parser.add_argument('--skip-validation', action='store_true', help='Skip system validation')

    args = parser.parse_args()

    print("="*80)
    print("FULL FRAMEWORK TEST RUNNER")
    print("="*80)
    print(f"Week: {args.week}")
    print(f"Season: {args.season}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FULL EXECUTION'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    test_results = {}

    # Test 1: Data Completeness Check
    if not args.skip_data_check:
        test_results['data_completeness'] = run_command(
            'scripts/testing/check_data_completeness.py',
            ['--week', str(args.week), '--season', str(args.season)],
            'Data Completeness Check'
        )

    # Test 2: Sleeper vs NFLverse Comparison
    if not args.skip_comparison:
        test_results['data_comparison'] = run_command(
            'scripts/testing/compare_sleeper_nflverse.py',
            [str(args.week), '--season', str(args.season)],
            'Sleeper vs NFLverse Comparison'
        )

    # Test 3: Full Framework Test
    if not args.skip_framework:
        framework_args = [str(args.week), '--season', str(args.season)]
        if args.dry_run:
            framework_args.append('--dry-run')
        test_results['framework_test'] = run_command(
            'scripts/testing/test_full_framework.py',
            framework_args,
            'Full Framework End-to-End Test'
        )

    # Test 4: System Validation
    if not args.skip_validation:
        test_results['system_validation'] = run_command(
            'scripts/integration/validate_complete_system.py',
            [],
            'Complete System Validation'
        )

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")

    total_passed = sum(test_results.values())
    total_tests = len(test_results)

    print(f"\n  Overall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED - Framework is ready!")
    elif total_passed >= total_tests * 0.75:
        print("\n⚠️  MOST TESTS PASSED - Review failures above")
    else:
        print("\n❌ MULTIPLE TESTS FAILED - Framework needs attention")

    print("="*80)

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == '__main__':
    main()
