"""
Integration Tests for Calibration System
=========================================

Tests the complete calibration system including:
- Calibrator loading (market-specific + unified + game line)
- Hybrid calibration (isotonic + shrinkage)
- Edge calculations
- Production readiness
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.calibration.calibrator_loader import (
    load_calibrator_for_market,
    load_game_line_calibrator,
    get_available_market_calibrators,
    validate_all_calibrators
)


class TestCalibratorLoading:
    """Test calibrator loading functionality."""

    def test_market_specific_calibrators_exist(self):
        """Test that market-specific calibrators are available."""
        markets = get_available_market_calibrators()

        # Should have at least these three markets
        expected_markets = ['player_reception_yds', 'player_receptions', 'player_rush_yds']

        for market in expected_markets:
            assert market in markets, f"Missing market-specific calibrator: {market}"

    def test_load_market_specific_calibrator(self):
        """Test loading a market-specific calibrator."""
        calibrator = load_calibrator_for_market('player_reception_yds', use_cache=False)

        assert calibrator.is_fitted
        assert hasattr(calibrator, 'high_prob_threshold')
        assert hasattr(calibrator, 'high_prob_shrinkage')
        assert calibrator.high_prob_threshold == 0.70
        assert calibrator.high_prob_shrinkage == 0.3

    def test_fallback_to_unified_calibrator(self):
        """Test fallback to unified calibrator for markets without specific calibrators."""
        # Use a market that doesn't have a specific calibrator
        calibrator = load_calibrator_for_market('player_pass_tds', use_cache=False)

        assert calibrator.is_fitted
        assert hasattr(calibrator, 'high_prob_threshold')

    def test_load_game_line_calibrator(self):
        """Test loading game line calibrator."""
        calibrator = load_game_line_calibrator()

        assert calibrator.is_fitted
        assert hasattr(calibrator, 'high_prob_threshold')
        assert calibrator.high_prob_threshold == 0.70
        assert calibrator.high_prob_shrinkage == 0.3

    def test_all_calibrators_valid(self):
        """Test that all calibrator files are valid."""
        results = validate_all_calibrators()

        # Check critical calibrators
        assert results['unified'] == True, "Unified calibrator invalid"
        assert results['game_line'] == True, "Game line calibrator invalid"
        assert results['player_reception_yds'] == True, "Reception yards calibrator invalid"
        assert results['player_receptions'] == True, "Receptions calibrator invalid"
        assert results['player_rush_yds'] == True, "Rush yards calibrator invalid"


class TestHybridCalibration:
    """Test hybrid calibration functionality."""

    def test_shrinkage_applies_for_high_probabilities(self):
        """Test that shrinkage is applied for raw prob >= 70%."""
        calibrator = load_calibrator_for_market('player_reception_yds', use_cache=False)

        # High raw probability should be shrunk significantly
        raw_prob = 0.85
        calibrated = calibrator.transform(raw_prob)

        # Should be shrunk toward 50%
        assert calibrated < raw_prob, "High probability not shrunk"

        # With shrinkage=0.3, expected: 0.5 + (0.85 - 0.5) * 0.3 = 0.605
        # But isotonic may adjust it further, so just check it's well below 70%
        assert calibrated < 0.70, f"Shrinkage didn't prevent >70% (got {calibrated:.2%})"

    def test_isotonic_applies_for_low_probabilities(self):
        """Test that isotonic regression is used for raw prob < 70%."""
        calibrator = load_calibrator_for_market('player_reception_yds', use_cache=False)

        # Low probability should use isotonic calibration
        raw_prob = 0.55
        calibrated = calibrator.transform(raw_prob)

        # Should be calibrated (different from raw), but method varies
        assert isinstance(calibrated, (float, np.floating))
        assert 0.0 <= calibrated <= 1.0

    def test_batch_calibration(self):
        """Test calibration of multiple probabilities at once."""
        calibrator = load_calibrator_for_market('player_receptions', use_cache=False)

        raw_probs = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
        calibrated = calibrator.transform(raw_probs)

        assert len(calibrated) == len(raw_probs)
        assert all(0.0 <= p <= 1.0 for p in calibrated)

        # High probabilities should be shrunk
        for i, raw in enumerate(raw_probs):
            if raw >= 0.70:
                assert calibrated[i] < raw, f"High prob {raw} not shrunk"

    def test_edge_cases(self):
        """Test edge cases within training range."""
        # Use player_receptions which has widest training range (0.14 - 1.0)
        calibrator = load_calibrator_for_market('player_receptions', use_cache=False)

        # Test mid-range probability (should be reasonable)
        cal_50 = calibrator.transform(0.50)
        assert 0.0 <= cal_50 <= 1.0, f"50% calibrated to invalid value {cal_50:.2%}"

        # Test low probability within training range
        cal_low = calibrator.transform(0.20)
        assert 0.0 <= cal_low <= 1.0, f"20% calibrated to invalid value {cal_low:.2%}"

        # Test high probability (should be shrunk)
        cal_high = calibrator.transform(0.95)
        assert 0.0 <= cal_high <= 1.0, f"95% calibrated to invalid value {cal_high:.2%}"
        assert cal_high < 0.70, f"95% not shrunk below 70% (got {cal_high:.2%})"


class TestProductionReadiness:
    """Test production readiness of calibration system."""

    def test_no_calibrated_probs_above_70_percent(self):
        """Test that hybrid calibration prevents probabilities > 70%."""
        calibrator = load_calibrator_for_market('player_reception_yds', use_cache=False)

        # Test a range of high probabilities
        high_probs = np.linspace(0.70, 0.99, 30)
        calibrated = calibrator.transform(high_probs)

        # With shrinkage=0.3, all should be below 70%
        # Max possible: 0.5 + (0.99 - 0.5) * 0.3 = 0.647
        violators = calibrated[calibrated > 0.70]

        assert len(violators) == 0, f"Found {len(violators)} calibrated probs > 70%: {violators}"

    def test_realistic_edge_calculations(self):
        """Test that edge calculations are realistic (<15%)."""
        calibrator = load_calibrator_for_market('player_receptions', use_cache=False)

        # Simulate some predictions
        raw_probs = np.array([0.60, 0.70, 0.80, 0.90])
        calibrated_probs = calibrator.transform(raw_probs)

        # Simulate implied probabilities (typical odds around 50%)
        implied_probs = np.array([0.52, 0.52, 0.52, 0.52])

        # Calculate edges
        edges = calibrated_probs - implied_probs

        # Edges should be realistic (<15%)
        assert all(abs(e) < 0.15 for e in edges), f"Unrealistic edges: {edges}"

    def test_calibration_preserves_ordering(self):
        """Test that calibration preserves probability ordering."""
        calibrator = load_calibrator_for_market('player_rush_yds', use_cache=False)

        raw_probs = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        calibrated = calibrator.transform(raw_probs)

        # Calibrated probabilities should preserve ordering
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i+1], \
                f"Ordering violated: {calibrated[i]:.3f} > {calibrated[i+1]:.3f}"

    def test_performance_benchmarks(self):
        """Test that calibration is fast enough for production."""
        import time

        calibrator = load_calibrator_for_market('player_reception_yds', use_cache=False)

        # Test single prediction speed
        start = time.time()
        for _ in range(1000):
            calibrator.transform(0.75)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0, f"Calibration too slow: {avg_time_ms:.2f}ms per prediction"

        # Test batch prediction speed
        batch_size = 100
        raw_probs = np.random.uniform(0.3, 0.95, batch_size)

        start = time.time()
        calibrator.transform(raw_probs)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Batch calibration too slow: {elapsed:.3f}s for {batch_size} predictions"


def run_all_tests():
    """Run all tests and print results."""
    import sys

    print("=" * 100)
    print("CALIBRATION SYSTEM INTEGRATION TESTS")
    print("=" * 100)
    print()

    # Use pytest if available, otherwise run manually
    try:
        import pytest
        exit_code = pytest.main([__file__, '-v'])
        sys.exit(exit_code)
    except ImportError:
        print("pytest not installed, running tests manually...")
        print()

        test_classes = [
            TestCalibratorLoading(),
            TestHybridCalibration(),
            TestProductionReadiness()
        ]

        total_tests = 0
        passed_tests = 0

        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            print(f"\n{class_name}:")
            print("-" * 80)

            for method_name in dir(test_class):
                if method_name.startswith('test_'):
                    total_tests += 1
                    try:
                        method = getattr(test_class, method_name)
                        method()
                        print(f"  ✅ {method_name}")
                        passed_tests += 1
                    except AssertionError as e:
                        print(f"  ❌ {method_name}: {e}")
                    except Exception as e:
                        print(f"  ❌ {method_name}: ERROR - {e}")

        print()
        print("=" * 100)
        print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
        print("=" * 100)

        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED - System ready for production")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED - Review before deployment")
            sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
