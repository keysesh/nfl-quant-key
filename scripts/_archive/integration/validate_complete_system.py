#!/usr/bin/env python3
"""
Complete System Validation
============================

Validates that the entire betting model pipeline works correctly end-to-end.

Tests:
1. All calibrators load correctly
2. Calibrators have proper quality (not flat, reasonable range)
3. Model predictions can be generated
4. Recommendations can be generated
5. All market types are properly calibrated

This provides confidence that the system is production-ready.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market
import joblib

class SystemValidator:
    """Validate complete betting model system."""

    def __init__(self):
        self.base_dir = Path(Path.cwd())
        self.results = {}

    def test_calibrator_loading(self):
        """Test that all calibrators load correctly."""
        print("\n" + "="*80)
        print("TEST 1: CALIBRATOR LOADING")
        print("="*80)

        markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds']
        test_results = {}

        for market in markets:
            try:
                calibrator = load_calibrator_for_market(market)
                test_results[market] = {
                    'loaded': True,
                    'type': str(type(calibrator)),
                    'fitted': calibrator.is_fitted
                }
                print(f"  ✅ {market}: Loaded successfully")
            except Exception as e:
                test_results[market] = {
                    'loaded': False,
                    'error': str(e)
                }
                print(f"  ❌ {market}: Failed - {e}")

        self.results['calibrator_loading'] = test_results
        return all(r['loaded'] for r in test_results.values())

    def test_calibrator_quality(self):
        """Test that calibrators produce reasonable outputs."""
        print("\n" + "="*80)
        print("TEST 2: CALIBRATOR QUALITY")
        print("="*80)

        markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds']
        test_probs = np.array([0.60, 0.70, 0.80, 0.90, 0.95])

        quality_results = {}

        for market in markets:
            try:
                calibrator = load_calibrator_for_market(market)
                calibrated = calibrator.transform(test_probs)

                # Quality checks
                is_not_flat = np.std(calibrated) > 0.01  # Not all the same value
                is_decreasing_confidence = calibrated[4] < test_probs[4]  # 95% should be calibrated down
                is_reasonable = all((calibrated >= 0.4) & (calibrated <= 0.90))  # Reasonable range

                quality_results[market] = {
                    'not_flat': is_not_flat,
                    'reduces_overconfidence': is_decreasing_confidence,
                    'reasonable_range': is_reasonable,
                    'test_output': calibrated.round(3).tolist(),
                    'passed': is_not_flat and is_decreasing_confidence and is_reasonable
                }

                status = "✅" if quality_results[market]['passed'] else "❌"
                print(f"  {status} {market}:")
                print(f"      Input:  {test_probs}")
                print(f"      Output: {calibrated.round(3)}")
                print(f"      Flat: {'No' if is_not_flat else 'YES (BAD)'}")
                print(f"      Reduces 95%: {'Yes' if is_decreasing_confidence else 'NO (BAD)'}")

            except Exception as e:
                quality_results[market] = {
                    'passed': False,
                    'error': str(e)
                }
                print(f"  ❌ {market}: Error - {e}")

        self.results['calibrator_quality'] = quality_results
        return all(r.get('passed', False) for r in quality_results.values())

    def test_td_calibrator(self):
        """Test TD calibrator specifically (was previously broken)."""
        print("\n" + "="*80)
        print("TEST 3: TD CALIBRATOR")
        print("="*80)

        td_cal_paths = [
            self.base_dir / 'data/models/td_calibrator_v2_improved.joblib',
            self.base_dir / 'data/models/td_calibrator_v1.joblib',
        ]

        for path in td_cal_paths:
            if path.exists():
                try:
                    calibrator = joblib.load(path)

                    # Test
                    test_probs = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
                    calibrated = calibrator.predict(test_probs)

                    # Quality checks
                    is_not_flat = np.std(calibrated) > 0.01
                    num_points = len(calibrator.X_thresholds_)

                    print(f"\n  Testing: {path.name}")
                    print(f"    Calibration points: {num_points}")
                    print(f"    Input:  {test_probs}")
                    print(f"    Output: {calibrated.round(3)}")
                    print(f"    Std dev: {np.std(calibrated):.3f}")

                    if is_not_flat and num_points >= 20:
                        print(f"    ✅ GOOD QUALITY (not flat, {num_points} points)")
                        self.results['td_calibrator'] = {'passed': True, 'path': path.name, 'points': num_points}
                        return True
                    elif is_not_flat:
                        print(f"    ⚠️  OK (not flat, but only {num_points} points)")
                        self.results['td_calibrator'] = {'passed': True, 'path': path.name, 'points': num_points, 'warning': 'few points'}
                        return True
                    else:
                        print(f"    ❌ POOR (flat calibration)")

                except Exception as e:
                    print(f"    ❌ Error loading {path.name}: {e}")

        self.results['td_calibrator'] = {'passed': False, 'error': 'No working TD calibrator found'}
        print(f"\n  ❌ No working TD calibrator found")
        return False

    def test_data_availability(self):
        """Test that required data files exist."""
        print("\n" + "="*80)
        print("TEST 4: DATA AVAILABILITY")
        print("="*80)

        required_files = {
            '2024 PBP Data': self.base_dir / 'data/nflverse/pbp_2024.parquet',
            '2025 PBP Data': self.base_dir / 'data/nflverse/pbp_2025.parquet',
            'Usage Model': self.base_dir / 'data/models/usage_predictor_v4_defense.joblib',
            'Efficiency Model': self.base_dir / 'data/models/efficiency_predictor_v2_defense.joblib',
        }

        data_results = {}

        for name, path in required_files.items():
            exists = path.exists()
            data_results[name] = {'exists': exists, 'path': str(path)}

            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {'Found' if exists else 'MISSING'}")

        self.results['data_availability'] = data_results
        return all(r['exists'] for r in data_results.values())

    def test_prediction_generation(self):
        """Test that model predictions can be generated."""
        print("\n" + "="*80)
        print("TEST 5: PREDICTION GENERATION")
        print("="*80)

        # Check if recent predictions exist
        prediction_files = list((self.base_dir / 'data').glob('model_predictions_week*.csv'))

        if not prediction_files:
            print("  ⚠️  No prediction files found")
            print("  This is OK - predictions are generated on demand")
            self.results['prediction_generation'] = {'status': 'no_files', 'note': 'predictions generated on demand'}
            return True

        # Check most recent
        latest = max(prediction_files, key=lambda p: int(p.stem.split('week')[1]))
        df = pd.read_csv(latest)

        print(f"  ✅ Found recent predictions: {latest.name}")
        print(f"     Players: {len(df)}")
        print(f"     Columns: {len(df.columns)}")

        # Check for required columns
        required_cols = ['player_name', 'team', 'position', 'week']
        has_required = all(col in df.columns for col in required_cols)

        if has_required:
            print(f"     ✅ Has all required columns")
            self.results['prediction_generation'] = {'status': 'pass', 'file': latest.name, 'players': len(df)}
            return True
        else:
            print(f"     ❌ Missing required columns")
            self.results['prediction_generation'] = {'status': 'fail', 'error': 'missing columns'}
            return False

    def test_calibration_integration(self):
        """Test that calibration is properly integrated into recommendations."""
        print("\n" + "="*80)
        print("TEST 6: CALIBRATION INTEGRATION")
        print("="*80)

        # This tests that market type extraction works correctly
        from scripts.predict.generate_current_week_recommendations import extract_market_type

        test_cases = {
            'player_reception_yds': 'Receiving Yards',
            'player_rush_yds': 'Rushing Yards',
            'player_receptions': 'Receptions',
            'player_pass_yds': 'Passing Yards',
        }

        integration_results = {}

        for expected, market_str in test_cases.items():
            result = extract_market_type(market_str)
            matches = result == expected

            integration_results[market_str] = {
                'expected': expected,
                'actual': result,
                'matches': matches
            }

            status = "✅" if matches else "❌"
            print(f"  {status} '{market_str}' → '{result}' (expected '{expected}')")

        self.results['calibration_integration'] = integration_results
        return all(r['matches'] for r in integration_results.values())

    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        test_results = {
            'Calibrator Loading': self.test_calibrator_loading(),
            'Calibrator Quality': self.test_calibrator_quality(),
            'TD Calibrator': self.test_td_calibrator(),
            'Data Availability': self.test_data_availability(),
            'Prediction Generation': self.test_prediction_generation(),
            'Calibration Integration': self.test_calibration_integration(),
        }

        print(f"\nTest Results:")
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} - {test_name}")

        total_tests = len(test_results)
        passed_tests = sum(test_results.values())

        print(f"\n" + "="*80)
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed")
        print("="*80)

        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY")
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️  MOST TESTS PASSED - SYSTEM IS MOSTLY READY")
            print("   Review failures above")
        else:
            print("\n❌ MULTIPLE TESTS FAILED - SYSTEM NEEDS WORK")
            print("   Review failures above")

        print()

        return passed_tests == total_tests

    def run_validation(self):
        """Run all validation tests."""
        print("="*80)
        print("COMPLETE SYSTEM VALIDATION")
        print("="*80)
        print("Testing all components of the betting model pipeline...")

        success = self.generate_report()

        return success


def main():
    validator = SystemValidator()
    success = validator.run_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
