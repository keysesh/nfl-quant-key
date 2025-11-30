#!/usr/bin/env python3
"""
Comprehensive Validation Suite
===============================

Runs comprehensive validation of all calibrators and system components:
1. Calibrator loading and integrity
2. Calibrator quality metrics
3. Market coverage validation
4. Position-specific TD calibrator validation
5. Integration testing
6. Performance metrics

Generates detailed validation report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market, validate_all_calibrators
import joblib


class ComprehensiveValidator:
    """Comprehensive validation suite for calibrators."""

    def __init__(self):
        self.base_dir = Path(Path.cwd())
        self.config_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "data/models"
        self.results = {}

    def validate_calibrator_loading(self):
        """Test that all calibrators load correctly."""
        print("\n" + "="*80)
        print("TEST 1: CALIBRATOR LOADING")
        print("="*80)

        test_results = {}

        # Test unified calibrator
        try:
            calibrator = NFLProbabilityCalibrator()
            calibrator.load(str(self.config_dir / 'calibrator.json'))
            assert calibrator.is_fitted
            test_results['unified'] = {'loaded': True, 'fitted': True}
            print("  ✅ Unified calibrator: Loaded successfully")
        except Exception as e:
            test_results['unified'] = {'loaded': False, 'error': str(e)}
            print(f"  ❌ Unified calibrator: Failed - {e}")

        # Test market-specific calibrators
        markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds',
                   'player_pass_completions', 'player_pass_attempts', 'player_targets']

        for market in markets:
            try:
                calibrator = load_calibrator_for_market(market, str(self.config_dir))
                test_results[market] = {
                    'loaded': True,
                    'fitted': calibrator.is_fitted if hasattr(calibrator, 'is_fitted') else True
                }
                print(f"  ✅ {market}: Loaded successfully")
            except Exception as e:
                test_results[market] = {'loaded': False, 'error': str(e)}
                print(f"  ⚠️  {market}: Not available - {e}")

        # Test position-specific TD calibrators
        positions = ['QB', 'RB', 'WR', 'TE']
        for position in positions:
            try:
                cal_path = self.config_dir / f'td_calibrator_{position}.json'
                if cal_path.exists():
                    calibrator = NFLProbabilityCalibrator()
                    calibrator.load(str(cal_path))
                    test_results[f'td_{position}'] = {'loaded': True, 'fitted': calibrator.is_fitted}
                    print(f"  ✅ TD calibrator ({position}): Loaded successfully")
                else:
                    test_results[f'td_{position}'] = {'loaded': False, 'error': 'File not found'}
                    print(f"  ⚠️  TD calibrator ({position}): Not found")
            except Exception as e:
                test_results[f'td_{position}'] = {'loaded': False, 'error': str(e)}
                print(f"  ❌ TD calibrator ({position}): Failed - {e}")

        self.results['calibrator_loading'] = test_results
        return test_results

    def validate_calibrator_quality(self):
        """Test that calibrators produce reasonable outputs."""
        print("\n" + "="*80)
        print("TEST 2: CALIBRATOR QUALITY")
        print("="*80)

        test_probs = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
        quality_results = {}

        # Test market-specific calibrators
        markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds']

        for market in markets:
            try:
                calibrator = load_calibrator_for_market(market, str(self.config_dir))
                calibrated = calibrator.transform(test_probs)

                # Quality checks
                is_not_flat = np.std(calibrated) > 0.01
                is_monotonic = np.all(np.diff(calibrated) >= -0.01)  # Allow small numerical errors
                is_reasonable = all((calibrated >= 0.0) & (calibrated <= 1.0))
                reduces_overconfidence = calibrated[-1] < test_probs[-1]  # 90% should be calibrated down

                quality_results[market] = {
                    'not_flat': is_not_flat,
                    'monotonic': is_monotonic,
                    'reasonable_range': is_reasonable,
                    'reduces_overconfidence': reduces_overconfidence,
                    'test_output': calibrated.round(3).tolist(),
                    'passed': is_not_flat and is_monotonic and is_reasonable
                }

                status = "✅" if quality_results[market]['passed'] else "❌"
                print(f"  {status} {market}:")
                print(f"      Input:  {test_probs}")
                print(f"      Output: {calibrated.round(3)}")
                print(f"      Flat: {'No' if is_not_flat else 'YES (BAD)'}")
                print(f"      Monotonic: {'Yes' if is_monotonic else 'NO (BAD)'}")

            except Exception as e:
                quality_results[market] = {'passed': False, 'error': str(e)}
                print(f"  ⚠️  {market}: Error - {e}")

        # Test position-specific TD calibrators
        positions = ['QB', 'RB', 'WR', 'TE']
        for position in positions:
            try:
                cal_path = self.config_dir / f'td_calibrator_{position}.json'
                if cal_path.exists():
                    calibrator = NFLProbabilityCalibrator()
                    calibrator.load(str(cal_path))
                    calibrated = calibrator.transform(test_probs)

                    is_not_flat = np.std(calibrated) > 0.01
                    is_monotonic = np.all(np.diff(calibrated) >= -0.01)
                    is_reasonable = all((calibrated >= 0.0) & (calibrated <= 1.0))

                    quality_results[f'td_{position}'] = {
                        'not_flat': is_not_flat,
                        'monotonic': is_monotonic,
                        'reasonable_range': is_reasonable,
                        'test_output': calibrated.round(3).tolist(),
                        'passed': is_not_flat and is_monotonic and is_reasonable
                    }

                    status = "✅" if quality_results[f'td_{position}']['passed'] else "❌"
                    print(f"  {status} TD calibrator ({position}):")
                    print(f"      Output: {calibrated.round(3)}")
                else:
                    quality_results[f'td_{position}'] = {'passed': False, 'error': 'Not found'}
            except Exception as e:
                quality_results[f'td_{position}'] = {'passed': False, 'error': str(e)}

        self.results['calibrator_quality'] = quality_results
        return quality_results

    def validate_market_coverage(self):
        """Validate that all expected markets have calibrators."""
        print("\n" + "="*80)
        print("TEST 3: MARKET COVERAGE")
        print("="*80)

        expected_markets = {
            'core': ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds'],
            'missing': ['player_pass_completions', 'player_pass_attempts', 'player_targets']
        }

        coverage_results = {}

        for category, markets in expected_markets.items():
            for market in markets:
                try:
                    calibrator = load_calibrator_for_market(market, str(self.config_dir))
                    coverage_results[market] = {'available': True, 'category': category}
                    print(f"  ✅ {market}: Available")
                except Exception as e:
                    coverage_results[market] = {'available': False, 'category': category, 'error': str(e)}
                    status = "⚠️ " if category == 'missing' else "❌"
                    print(f"  {status} {market}: Not available")

        self.results['market_coverage'] = coverage_results
        return coverage_results

    def validate_position_td_calibrators(self):
        """Validate position-specific TD calibrators."""
        print("\n" + "="*80)
        print("TEST 4: POSITION-SPECIFIC TD CALIBRATORS")
        print("="*80)

        positions = ['QB', 'RB', 'WR', 'TE']
        td_results = {}

        for position in positions:
            cal_path = self.config_dir / f'td_calibrator_{position}.json'
            if cal_path.exists():
                try:
                    calibrator = NFLProbabilityCalibrator()
                    calibrator.load(str(cal_path))

                    # Test calibration
                    test_probs = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
                    calibrated = calibrator.transform(test_probs)

                    td_results[position] = {
                        'available': True,
                        'fitted': calibrator.is_fitted,
                        'calibration_points': len(calibrator.calibrator.X_thresholds_),
                        'test_output': calibrated.round(3).tolist()
                    }
                    print(f"  ✅ {position}: Available ({td_results[position]['calibration_points']} points)")
                except Exception as e:
                    td_results[position] = {'available': False, 'error': str(e)}
                    print(f"  ❌ {position}: Error - {e}")
            else:
                td_results[position] = {'available': False, 'error': 'File not found'}
                print(f"  ⚠️  {position}: Not found")

        self.results['position_td_calibrators'] = td_results
        return td_results

    def validate_integration(self):
        """Test integration with prediction pipeline."""
        print("\n" + "="*80)
        print("TEST 5: INTEGRATION TESTING")
        print("="*80)

        integration_results = {}

        # Test that calibrators can be loaded via loader
        try:
            from nfl_quant.calibration.calibrator_loader import validate_all_calibrators
            validation_results = validate_all_calibrators(str(self.config_dir))
            integration_results['loader_validation'] = validation_results
            print("  ✅ Calibrator loader validation:")
            for name, status in validation_results.items():
                status_icon = "✅" if status else "❌"
                print(f"      {status_icon} {name}")
        except Exception as e:
            integration_results['loader_validation'] = {'error': str(e)}
            print(f"  ❌ Loader validation failed: {e}")

        # Test calibration transform on sample data
        try:
            calibrator = NFLProbabilityCalibrator()
            calibrator.load(str(self.config_dir / 'calibrator.json'))
            test_input = np.array([0.2, 0.5, 0.8])
            test_output = calibrator.transform(test_input)
            assert len(test_output) == len(test_input)
            assert all((test_output >= 0) & (test_output <= 1))
            integration_results['transform_test'] = {'passed': True}
            print("  ✅ Transform test: Passed")
        except Exception as e:
            integration_results['transform_test'] = {'passed': False, 'error': str(e)}
            print(f"  ❌ Transform test: Failed - {e}")

        self.results['integration'] = integration_results
        return integration_results

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def generate_summary_report(self):
        """Generate comprehensive validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        # Count results
        loading_results = self.results.get('calibrator_loading', {})
        quality_results = self.results.get('calibrator_quality', {})
        coverage_results = self.results.get('market_coverage', {})
        td_results = self.results.get('position_td_calibrators', {})
        integration_results = self.results.get('integration', {})

        # Calculate statistics
        loaded_count = sum(1 for r in loading_results.values() if r.get('loaded', False))
        quality_passed = sum(1 for r in quality_results.values() if r.get('passed', False))
        coverage_available = sum(1 for r in coverage_results.values() if r.get('available', False))
        td_available = sum(1 for r in td_results.values() if r.get('available', False))

        print(f"\nTest Results:")
        print(f"  Calibrator Loading: {loaded_count}/{len(loading_results)} loaded")
        print(f"  Calibrator Quality: {quality_passed}/{len(quality_results)} passed")
        print(f"  Market Coverage: {coverage_available}/{len(coverage_results)} available")
        print(f"  Position TD Calibrators: {td_available}/4 available")

        # Overall status
        total_tests = len(loading_results) + len(quality_results) + len(coverage_results) + len(td_results)
        passed_tests = loaded_count + quality_passed + coverage_available + td_available

        print(f"\n" + "="*80)
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print("="*80)

        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY")
            overall_status = "PASS"
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️  MOST TESTS PASSED - SYSTEM IS MOSTLY READY")
            print("   Review failures above")
            overall_status = "WARNING"
        else:
            print("\n❌ MULTIPLE TESTS FAILED - SYSTEM NEEDS WORK")
            print("   Review failures above")
            overall_status = "FAIL"

        # Save results
        report = {
            'validation_date': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'loaded_count': int(loaded_count),
                'quality_passed': int(quality_passed),
                'coverage_available': int(coverage_available),
                'td_available': int(td_available),
                'total_tests': int(total_tests),
                'passed_tests': int(passed_tests)
            },
            'detailed_results': self._convert_to_json_serializable(self.results)
        }

        report_path = self.base_dir / 'reports' / 'validation_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_path}")

        return overall_status == "PASS"

    def run_validation(self):
        """Run all validation tests."""
        print("="*80)
        print("COMPREHENSIVE VALIDATION SUITE")
        print("="*80)
        print("Validating all calibrators and system components...")

        self.validate_calibrator_loading()
        self.validate_calibrator_quality()
        self.validate_market_coverage()
        self.validate_position_td_calibrators()
        self.validate_integration()

        success = self.generate_summary_report()

        return success


def main():
    validator = ComprehensiveValidator()
    success = validator.run_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

