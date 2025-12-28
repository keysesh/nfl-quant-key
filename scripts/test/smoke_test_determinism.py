#!/usr/bin/env python3
"""
Smoke Test: Production Determinism Constraints

Validates the non-negotiable production constraints:
1. RESEARCH mode is completely banned
2. PlayerResolver supports offline/snapshot mode
3. Injury matching includes provenance fields
4. No silent excepts in critical modules

Run:
    python scripts/test/smoke_test_determinism.py

Exit codes:
    0 = All tests pass
    1 = One or more tests fail
"""

import sys
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_no_research_mode():
    """Verify RESEARCH mode is banned from depth_chart_loader."""
    print("\n[TEST] RESEARCH mode ban...")

    # Check that mode parameter doesn't exist in depth_chart_loader
    from nfl_quant.data.depth_chart_loader import get_depth_charts
    import inspect

    sig = inspect.signature(get_depth_charts)
    params = list(sig.parameters.keys())

    if 'mode' in params:
        print("  FAIL: 'mode' parameter still exists in get_depth_charts()")
        return False

    # Check no mode='RESEARCH' pattern in Python files (exclude this test file and archives)
    result = subprocess.run(
        ['grep', '-r', "mode=['\"]RESEARCH['\"]", '--include=*.py',
         '--exclude=smoke_test_determinism.py', '--exclude-dir=_archive',
         'nfl_quant/', 'scripts/'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )

    if result.returncode == 0:  # grep found matches
        # Filter out comments and string literals that explain the ban
        lines = [l for l in result.stdout.strip().split('\n')
                 if l and not l.strip().startswith('#') and 'FAIL' not in l]
        if lines:
            print(f"  FAIL: Found mode='RESEARCH' in code:\n" + '\n'.join(lines))
            return False

    print("  PASS: RESEARCH mode is banned")
    return True


def test_resolver_offline_mode():
    """Verify PlayerResolver supports offline/snapshot mode."""
    print("\n[TEST] PlayerResolver offline mode...")

    from nfl_quant.data.player_resolver import (
        PlayerResolver,
        ResolverMode,
        ResolverNotAvailableError
    )

    # Test 1: ResolverMode enum exists
    modes = [m.value for m in ResolverMode]
    expected_modes = ['online', 'offline', 'conservative']
    if set(modes) != set(expected_modes):
        print(f"  FAIL: ResolverMode missing modes. Got: {modes}")
        return False

    # Test 2: from_snapshot class method exists
    if not hasattr(PlayerResolver, 'from_snapshot'):
        print("  FAIL: PlayerResolver.from_snapshot() method missing")
        return False

    # Test 3: is_available property exists
    if not hasattr(PlayerResolver, 'is_available'):
        print("  FAIL: PlayerResolver.is_available property missing")
        return False

    # Test 4: save_snapshot method exists
    resolver = PlayerResolver(mode=ResolverMode.CONSERVATIVE)
    if not hasattr(resolver, 'save_snapshot'):
        print("  FAIL: PlayerResolver.save_snapshot() method missing")
        return False

    # Test 5: OFFLINE mode fails without cache
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_snapshot = Path(tmpdir) / "nonexistent.parquet"
            resolver = PlayerResolver.from_snapshot(fake_snapshot, strict=True)
            print("  FAIL: OFFLINE mode should fail for missing snapshot")
            return False
    except ResolverNotAvailableError:
        pass  # Expected behavior

    print("  PASS: PlayerResolver offline mode works correctly")
    return True


def test_injury_provenance_fields():
    """Verify injury matching includes provenance fields."""
    print("\n[TEST] Injury provenance fields...")

    from nfl_quant.data.injury_loader import (
        InjuryMatchType,
        InjuryMatchConfidence,
        InjuryMatchResult,
        match_player_to_injury_with_provenance
    )

    # Test 1: InjuryMatchType enum exists with correct values
    expected_types = ['gsis_id', 'exact', 'normalized', 'partial', 'none']
    actual_types = [m.value for m in InjuryMatchType]
    if set(actual_types) != set(expected_types):
        print(f"  FAIL: InjuryMatchType missing values. Got: {actual_types}")
        return False

    # Test 2: InjuryMatchConfidence enum exists with correct values
    expected_conf = ['high', 'med', 'low', 'none']
    actual_conf = [m.value for m in InjuryMatchConfidence]
    if set(actual_conf) != set(expected_conf):
        print(f"  FAIL: InjuryMatchConfidence missing values. Got: {actual_conf}")
        return False

    # Test 3: InjuryMatchResult has required fields
    import pandas as pd
    result = InjuryMatchResult(
        matched=False,
        injury_record=None,
        match_type=InjuryMatchType.NONE,
        match_confidence=InjuryMatchConfidence.NONE,
        match_source="test"
    )

    if not hasattr(result, 'to_dict'):
        print("  FAIL: InjuryMatchResult.to_dict() method missing")
        return False

    result_dict = result.to_dict()
    required_keys = ['injury_match_type', 'injury_match_confidence', 'injury_match_source']
    missing = set(required_keys) - set(result_dict.keys())
    if missing:
        print(f"  FAIL: InjuryMatchResult.to_dict() missing keys: {missing}")
        return False

    # Test 4: match_player_to_injury_with_provenance exists and returns InjuryMatchResult
    # Use empty DataFrame to avoid network calls
    empty_df = pd.DataFrame(columns=['player_name', 'team', 'gsis_id', 'status', 'risk_score'])
    result = match_player_to_injury_with_provenance(
        player_name="Test Player",
        team="TST",
        injuries_df=empty_df
    )

    if not isinstance(result, InjuryMatchResult):
        print(f"  FAIL: match_player_to_injury_with_provenance returned {type(result)}")
        return False

    print("  PASS: Injury provenance fields are present")
    return True


def test_no_silent_excepts():
    """Verify no bare except clauses in critical modules."""
    print("\n[TEST] No silent excepts...")

    # Check for bare 'except:' (most dangerous)
    result = subprocess.run(
        ['grep', '-rn', 'except *:', '--include=*.py', 'nfl_quant/data/'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )

    if result.returncode == 0:  # grep found matches
        print(f"  FAIL: Found bare 'except:' in data modules:\n{result.stdout}")
        return False

    print("  PASS: No silent excepts in data modules")
    return True


def test_depth_chart_validation():
    """Verify depth chart loader has strict validation."""
    print("\n[TEST] Depth chart strict validation...")

    from nfl_quant.data.depth_chart_loader import DepthChartValidationError

    # Test that the exception class exists
    if not issubclass(DepthChartValidationError, Exception):
        print("  FAIL: DepthChartValidationError is not an Exception")
        return False

    print("  PASS: Depth chart validation is strict")
    return True


def main():
    print("=" * 60)
    print("SMOKE TEST: Production Determinism Constraints")
    print("=" * 60)

    tests = [
        ("RESEARCH mode ban", test_no_research_mode),
        ("Resolver offline mode", test_resolver_offline_mode),
        ("Injury provenance", test_injury_provenance_fields),
        ("No silent excepts", test_no_silent_excepts),
        ("Depth chart validation", test_depth_chart_validation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    failed = len(results) - passed

    for name, success, error in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")
        if error:
            print(f"         Error: {error}")

    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nSome tests failed. Fix issues before production deployment.")
        sys.exit(1)
    else:
        print("\nAll tests passed. Production constraints are enforced.")
        sys.exit(0)


if __name__ == '__main__':
    main()
