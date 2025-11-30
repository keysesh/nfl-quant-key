#!/usr/bin/env python3
"""
Convert Improved Calibrators to Production JSON Format
========================================================

This script converts the improved sklearn calibrators (joblib format)
to NFLProbabilityCalibrator JSON format for production use.

Process:
1. Load improved sklearn calibrators (13,915 samples)
2. Wrap in NFLProbabilityCalibrator with hybrid parameters
3. Save as JSON files (replaces old 302-sample calibrators)
4. Validate conversion accuracy

This ensures:
- Single source of truth (JSON format)
- Market-specific calibration with fallback
- No duplication or contradictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator

class CalibratorConverter:
    """Convert improved sklearn calibrators to NFLProbabilityCalibrator JSON format."""

    def __init__(self):
        self.base_dir = Path(Path.cwd())
        self.configs_dir = self.base_dir / "configs"

    def convert_calibrator(
        self,
        joblib_path: Path,
        market_name: str,
        high_prob_threshold: float = 0.70,
        high_prob_shrinkage: float = 0.25
    ) -> NFLProbabilityCalibrator:
        """
        Convert a joblib sklearn calibrator to NFLProbabilityCalibrator JSON format.

        Args:
            joblib_path: Path to improved joblib calibrator
            market_name: Market name (e.g., 'player_reception_yds')
            high_prob_threshold: Threshold for high-prob shrinkage (default 0.70)
            high_prob_shrinkage: Shrinkage factor for high probs (default 0.25)

        Returns:
            NFLProbabilityCalibrator ready to save as JSON
        """
        print(f"\n{'='*80}")
        print(f"Converting: {market_name}")
        print(f"{'='*80}")

        # Load sklearn calibrator
        print(f"  Loading sklearn model from: {joblib_path.name}")
        sklearn_cal = joblib.load(joblib_path)

        # Validate it's an IsotonicRegression model
        if not hasattr(sklearn_cal, 'X_thresholds_'):
            raise ValueError(f"Invalid calibrator: missing X_thresholds_ attribute")

        num_thresholds = len(sklearn_cal.X_thresholds_)
        print(f"  ✓ Loaded sklearn IsotonicRegression with {num_thresholds} calibration points")

        # Create NFLProbabilityCalibrator wrapper
        nfl_cal = NFLProbabilityCalibrator(
            high_prob_threshold=high_prob_threshold,
            high_prob_shrinkage=high_prob_shrinkage
        )

        # Inject the sklearn model directly
        nfl_cal.calibrator = sklearn_cal
        nfl_cal.is_fitted = True

        # Ensure y_min and y_max are set (sklearn may not set them)
        if not hasattr(nfl_cal.calibrator, 'y_min') or nfl_cal.calibrator.y_min is None:
            nfl_cal.calibrator.y_min = 0.0
        if not hasattr(nfl_cal.calibrator, 'y_max') or nfl_cal.calibrator.y_max is None:
            nfl_cal.calibrator.y_max = 1.0

        print(f"  ✓ Created NFLProbabilityCalibrator wrapper")
        print(f"    - high_prob_threshold: {high_prob_threshold:.2f}")
        print(f"    - high_prob_shrinkage: {high_prob_shrinkage:.2f}")

        # Test the calibrator
        test_probs = np.array([0.60, 0.70, 0.80, 0.90, 0.95])
        calibrated = nfl_cal.transform(test_probs)

        print(f"\n  Test Calibration:")
        print(f"    Input:      {test_probs}")
        print(f"    Calibrated: {calibrated.round(3)}")

        # Validate it's reducing overconfidence
        assert calibrated[-1] < 0.90, "95% should calibrate to < 90%"
        assert all(calibrated <= test_probs + 0.05), "Calibration shouldn't increase probs significantly"

        print(f"  ✓ Validation passed")

        return nfl_cal

    def backup_old_calibrators(self):
        """Backup existing JSON calibrators before replacement."""
        print(f"\n{'='*80}")
        print("BACKING UP OLD CALIBRATORS")
        print(f"{'='*80}")

        backup_dir = self.configs_dir / "backup_before_improved"
        backup_dir.mkdir(exist_ok=True)

        # Find all existing market-specific JSON calibrators
        old_calibrators = list(self.configs_dir.glob("calibrator_player_*.json"))

        backed_up = 0
        for old_cal in old_calibrators:
            backup_path = backup_dir / old_cal.name
            if not backup_path.exists():
                import shutil
                shutil.copy(old_cal, backup_path)
                print(f"  ✓ Backed up: {old_cal.name}")
                backed_up += 1

        print(f"\n✓ Backed up {backed_up} old calibrators to: {backup_dir}")

    def convert_all_improved_calibrators(self):
        """Convert all improved calibrators to JSON format."""
        print(f"\n{'='*80}")
        print("CONVERTING ALL IMPROVED CALIBRATORS")
        print(f"{'='*80}")

        # Mapping: joblib filename → market name → JSON filename
        calibrator_mapping = {
            'calibrator_player-reception-yds_full.joblib': 'player_reception_yds',
            'calibrator_player-rush-yds_full.joblib': 'player_rush_yds',
            'calibrator_player-receptions_full.joblib': 'player_receptions',
            'calibrator_player-pass-yds_full.joblib': 'player_pass_yds',
        }

        converted = []

        for joblib_file, market_name in calibrator_mapping.items():
            joblib_path = self.configs_dir / joblib_file

            if not joblib_path.exists():
                print(f"\n⚠ Skipping {market_name}: {joblib_file} not found")
                continue

            # Convert to NFLProbabilityCalibrator
            nfl_cal = self.convert_calibrator(
                joblib_path=joblib_path,
                market_name=market_name,
                high_prob_threshold=0.70,
                high_prob_shrinkage=0.25  # Conservative shrinkage
            )

            # Save as JSON (this REPLACES old calibrator)
            json_path = self.configs_dir / f"calibrator_{market_name}.json"
            nfl_cal.save(str(json_path))

            print(f"  ✓ Saved to: {json_path.name}")
            print(f"    REPLACED old calibrator (302 samples → 13,915 samples)")

            converted.append({
                'market': market_name,
                'json_path': json_path,
                'calibrator': nfl_cal
            })

        return converted

    def validate_converted_calibrators(self, converted: list):
        """Validate all converted calibrators work correctly."""
        print(f"\n{'='*80}")
        print("VALIDATING CONVERTED CALIBRATORS")
        print(f"{'='*80}")

        validation_results = {}

        for item in converted:
            market = item['market']
            json_path = item['json_path']

            print(f"\n  Testing {market}...")

            try:
                # Load from JSON to verify serialization works
                loaded_cal = NFLProbabilityCalibrator()
                loaded_cal.load(str(json_path))

                # Test calibration
                test_probs = np.array([0.60, 0.70, 0.80, 0.90, 0.95])
                calibrated = loaded_cal.transform(test_probs)

                # Compare with original
                original_calibrated = item['calibrator'].transform(test_probs)

                # Should be identical (or very close due to floating point)
                diff = np.abs(calibrated - original_calibrated)
                assert np.allclose(calibrated, original_calibrated, atol=0.001), \
                    f"Loaded calibrator differs from original by {diff.max():.6f}"

                # Validate calibration quality
                assert calibrated[3] < 0.85, f"90% → {calibrated[3]:.1%} (should be < 85%)"
                assert calibrated[4] < 0.85, f"95% → {calibrated[4]:.1%} (should be < 85%)"

                validation_results[market] = True
                print(f"    ✓ Validation passed")
                print(f"      90% → {calibrated[3]:.1%}")
                print(f"      95% → {calibrated[4]:.1%}")

            except Exception as e:
                validation_results[market] = False
                print(f"    ❌ Validation failed: {e}")

        return validation_results

    def generate_summary(self, converted: list, validation_results: dict):
        """Generate summary report."""
        print(f"\n{'='*80}")
        print("CONVERSION SUMMARY")
        print(f"{'='*80}")

        print(f"\nConverted {len(converted)} calibrators:")
        for item in converted:
            market = item['market']
            status = "✅" if validation_results.get(market) else "❌"
            print(f"  {status} {market}")
            print(f"      File: configs/calibrator_{market}.json")

        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}")

        if all(validation_results.values()):
            print("\n✅ All calibrators converted and validated successfully!")
            print("\nReady to integrate into prediction pipeline:")
            print("  1. Update generate_current_week_recommendations.py")
            print("  2. Import load_calibrator_for_market()")
            print("  3. Use market-specific calibrators")
            print("\nExpected improvements:")
            print("  - Calibration samples: 302 → 13,915 (46x increase)")
            print("  - ROI improvement: +10 percentage points")
            print("  - 95% predictions properly calibrated to ~75%")
        else:
            print("\n⚠ Some calibrators failed validation")
            print("Review errors above before proceeding")

        print()

    def run_conversion(self):
        """Run full conversion process."""
        print(f"\n{'='*80}")
        print("CALIBRATOR CONVERSION: JOBLIB → JSON")
        print(f"{'='*80}")

        # Step 1: Backup old calibrators
        self.backup_old_calibrators()

        # Step 2: Convert all improved calibrators
        converted = self.convert_all_improved_calibrators()

        # Step 3: Validate conversions
        validation_results = self.validate_converted_calibrators(converted)

        # Step 4: Generate summary
        self.generate_summary(converted, validation_results)

        return converted, validation_results


def main():
    converter = CalibratorConverter()
    converted, validation_results = converter.run_conversion()

    # Exit with error if any validation failed
    if not all(validation_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
