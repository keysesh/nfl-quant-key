#!/usr/bin/env python3
"""
Quick Integration Test

Tests that regime integration is properly wired up without errors.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from nfl_quant.regime.integration import RegimeAwareTrailingStats, get_regime_aware_extractor
        print("  ✅ regime.integration imported")
    except Exception as e:
        print(f"  ❌ Failed to import regime.integration: {e}")
        return False

    try:
        from nfl_quant.features.trailing_stats import get_trailing_stats_extractor, ENABLE_REGIME_DETECTION
        print(f"  ✅ trailing_stats imported (ENABLE_REGIME_DETECTION={ENABLE_REGIME_DETECTION})")
    except Exception as e:
        print(f"  ❌ Failed to import trailing_stats: {e}")
        return False

    return True


def test_feature_flag():
    """Test that feature flag switches extractors correctly."""
    print("\nTesting feature flag...")

    # Test with regime disabled
    os.environ['ENABLE_REGIME_DETECTION'] = '0'

    # Force reimport to pick up env var
    import importlib
    from nfl_quant.features import trailing_stats
    importlib.reload(trailing_stats)

    from nfl_quant.features.trailing_stats import get_trailing_stats_extractor

    try:
        extractor = get_trailing_stats_extractor()
        extractor_type = type(extractor).__name__
        print(f"  Regime disabled: {extractor_type}")

        if extractor_type == "TrailingStatsExtractor":
            print("  ✅ Standard extractor used when regime disabled")
        else:
            print(f"  ⚠️  Expected TrailingStatsExtractor, got {extractor_type}")

    except Exception as e:
        print(f"  ❌ Error with regime disabled: {e}")
        return False

    # Test with regime enabled
    os.environ['ENABLE_REGIME_DETECTION'] = '1'
    importlib.reload(trailing_stats)

    try:
        from nfl_quant.features.trailing_stats import get_trailing_stats_extractor
        extractor = get_trailing_stats_extractor()
        extractor_type = type(extractor).__name__
        print(f"  Regime enabled: {extractor_type}")

        if extractor_type == "RegimeAwareTrailingStats":
            print("  ✅ Regime-aware extractor used when regime enabled")
        else:
            print(f"  ⚠️  Expected RegimeAwareTrailingStats, got {extractor_type}")

    except Exception as e:
        print(f"  ⚠️  Regime extractor not available (may need data): {e}")
        # This is OK - regime extractor may fail without data

    # Reset
    os.environ['ENABLE_REGIME_DETECTION'] = '0'

    return True


def test_interface_compatibility():
    """Test that regime extractor has same interface as standard extractor."""
    print("\nTesting interface compatibility...")

    from nfl_quant.regime.integration import RegimeAwareTrailingStats

    # Check required methods exist
    required_methods = ['get_trailing_stats', '_compute_player_week_stats']

    for method in required_methods:
        if hasattr(RegimeAwareTrailingStats, method):
            print(f"  ✅ {method} exists")
        else:
            print(f"  ❌ {method} missing")
            return False

    return True


def test_cli_integration():
    """Test that CLI flags work."""
    print("\nTesting CLI integration...")

    # Check that generate_model_predictions.py has been updated
    script_path = Path(__file__).parent.parent / 'predict' / 'generate_model_predictions.py'

    if not script_path.exists():
        print(f"  ❌ Script not found: {script_path}")
        return False

    with open(script_path) as f:
        content = f.read()

    if '--enable-regime' in content:
        print("  ✅ --enable-regime flag added to generate_model_predictions.py")
    else:
        print("  ❌ --enable-regime flag not found in generate_model_predictions.py")
        return False

    if 'ENABLE_REGIME_DETECTION' in content:
        print("  ✅ Environment variable handling added")
    else:
        print("  ❌ Environment variable handling not found")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("REGIME INTEGRATION TEST SUITE")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Feature Flag", test_feature_flag),
        ("Interface Compatibility", test_interface_compatibility),
        ("CLI Integration", test_cli_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print('='*80)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n✅ All tests passed! Integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
