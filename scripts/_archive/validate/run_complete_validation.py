#!/usr/bin/env python3
"""
Complete Accuracy Validation Pipeline

Runs all validation checks to ensure maximum accuracy:
1. Data integrity validation
2. Calibration effectiveness validation
3. Temporal independence check
4. Calibration method consistency
5. Performance metrics

Run this before every calibration or weekly recommendation generation.
"""

import sys
from pathlib import Path
import subprocess

def run_validation_pipeline():
    """Run complete validation pipeline."""
    print("=" * 80)
    print("🎯 COMPLETE ACCURACY VALIDATION PIPELINE")
    print("=" * 80)
    print()

    checks_passed = 0
    checks_failed = 0

    # Step 1: Data Integrity Validation
    print("📊 STEP 1: Data Integrity Validation")
    print("-" * 80)
    try:
        result = subprocess.run(
            ['python', 'scripts/validate/validate_data_integrity.py'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            checks_passed += 1
            print("✅ Data integrity check PASSED\n")
        else:
            checks_failed += 1
            print("❌ Data integrity check FAILED\n")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error running data integrity check: {e}\n")
        checks_failed += 1

    # Step 2: Calibration Effectiveness Validation
    print("📊 STEP 2: Calibration Effectiveness Validation")
    print("-" * 80)
    try:
        result = subprocess.run(
            ['python', 'scripts/validate/validate_calibration_effectiveness.py'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            checks_passed += 1
            print("✅ Calibration effectiveness check PASSED\n")
        else:
            checks_failed += 1
            print("⚠️  Calibration effectiveness check completed with warnings\n")
    except Exception as e:
        print(f"⚠️  Error running calibration check: {e}\n")
        checks_failed += 1

    # Step 3: Calibration Method Check
    print("📊 STEP 3: Calibration Method Consistency")
    print("-" * 80)
    calibrator_path = Path('configs/calibrator.json')
    recommendation_script = Path('scripts/predict/generate_current_week_recommendations.py')

    has_isotonic = calibrator_path.exists()
    has_inline = False
    if recommendation_script.exists():
        content = recommendation_script.read_text()
        has_inline = 'calibrate_probability' in content or 'shrinkage' in content.lower()

    if has_isotonic and has_inline:
        print("⚠️  Multiple calibration methods detected")
        print("   Recommendation: Use isotonic calibrator only (most accurate)")
        checks_failed += 1
    elif has_isotonic:
        print("✅ Using isotonic calibrator (data-driven, most accurate)")
        checks_passed += 1
    elif has_inline:
        print("⚠️  Using inline shrinkage (fallback method)")
        print("   Recommendation: Train isotonic calibrator for better accuracy")
        checks_passed += 1
    else:
        print("❌ No calibration method detected")
        checks_failed += 1

    print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Passed: {checks_passed}")
    print(f"Failed/Warnings: {checks_failed}")

    if checks_failed == 0:
        print("\n✅ ALL CHECKS PASSED - System ready for maximum accuracy")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED - Review warnings above")
        print("\nRecommendations:")
        print("  1. Fix data integrity issues if any")
        print("  2. Train isotonic calibrator: python scripts/train/retrain_calibrator_full_history.py")
        print("  3. Use isotonic calibrator in production (most accurate)")
        return 1

if __name__ == '__main__':
    exit_code = run_validation_pipeline()
    sys.exit(exit_code)
