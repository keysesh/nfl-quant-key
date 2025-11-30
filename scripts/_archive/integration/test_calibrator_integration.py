#!/usr/bin/env python3
"""
Test Calibrator Integration
============================

Quick test to verify market-specific calibrators load and work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market

def test_calibrator_loading():
    """Test loading all market-specific calibrators."""
    print("="*80)
    print("TESTING CALIBRATOR INTEGRATION")
    print("="*80)

    markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds']
    test_probs = np.array([0.60, 0.70, 0.80, 0.90, 0.95])

    print(f"\nTest probabilities: {test_probs}")
    print("\n" + "-"*80)

    all_passed = True

    for market in markets:
        try:
            print(f"\n{market}:")
            calibrator = load_calibrator_for_market(market)

            # Test calibration
            calibrated = calibrator.transform(test_probs)

            print(f"  ✓ Loaded successfully")
            print(f"  Calibrated: {calibrated.round(3)}")
            print(f"  90% → {calibrated[3]:.1%}")
            print(f"  95% → {calibrated[4]:.1%}")

            # Validate calibration quality
            if calibrated[4] >= 0.85:
                print(f"  ⚠️  WARNING: 95% calibration seems high ({calibrated[4]:.1%})")
                all_passed = False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CALIBRATORS WORKING CORRECTLY")
    else:
        print("❌ SOME CALIBRATORS FAILED")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = test_calibrator_loading()
    sys.exit(0 if success else 1)
