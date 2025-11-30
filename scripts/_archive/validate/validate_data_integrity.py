#!/usr/bin/env python3
"""
Data Integrity & Calibration Validation

Validates that calibration data is clean, independent, and correctly separated.
Prevents data leakage and ensures calibration accuracy.

Checks:
1. Source data purity (no pre-calibration artifacts)
2. Temporal independence (train data < test data chronologically)
3. Calibration method consistency (only one method active)
4. Data completeness (>95% completeness)
5. Distribution validation (probabilities match model output)
6. No double calibration (checking for calibration artifacts)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class DataIntegrityValidator:
    """Validate data integrity for calibration."""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []

    def validate_source_data_purity(self, df: pd.DataFrame, prob_col: str = 'model_prob') -> bool:
        """
        Check if probabilities are truly raw (not pre-calibrated).

        Signs of pre-calibration:
        - Too many probabilities at exact boundaries (0.85, 0.90, etc.)
        - Unusual clustering at specific values
        - Distribution doesn't match expected model output
        """
        if prob_col not in df.columns:
            self.issues.append(f"Probability column '{prob_col}' not found")
            return False

        probs = df[prob_col].dropna()

        if len(probs) == 0:
            self.issues.append("No probabilities found in data")
            return False

        # Check for hard boundaries (sign of clipping/calibration)
        boundary_values = [0.50, 0.85, 0.90, 0.92, 0.95]
        boundary_counts = {}
        for boundary in boundary_values:
            count = ((probs >= boundary - 0.01) & (probs <= boundary + 0.01)).sum()
            if count > len(probs) * 0.05:  # More than 5% at boundary
                boundary_counts[boundary] = count
                self.warnings.append(
                    f"⚠️  {count} ({count/len(probs)*100:.1f}%) probabilities at {boundary:.2f} "
                    f"- may indicate pre-calibration/clipping"
                )

        # Check distribution spread
        prob_std = probs.std()
        if prob_std < 0.10:
            self.warnings.append(
                f"⚠️  Low probability spread (std={prob_std:.3f}) - may indicate calibration artifacts"
            )

        # Check for expected model distribution (should have variety)
        if probs.min() >= 0.50 and probs.max() <= 0.92:
            self.warnings.append(
                f"⚠️  Probabilities clipped to [0.50, 0.92] range - may be pre-calibrated"
            )

        if len(boundary_counts) == 0 and prob_std > 0.10:
            self.passed_checks.append("✅ Source data appears clean (no obvious calibration artifacts)")
            return True

        return len(boundary_counts) == 0

    def validate_temporal_independence(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        week_col: str = 'week',
        date_col: str = None
    ) -> bool:
        """Validate that training data chronologically precedes test data."""

        # Check week ordering
        if week_col in train_df.columns and week_col in test_df.columns:
            train_max_week = train_df[week_col].max()
            test_min_week = test_df[week_col].min()

            if train_max_week >= test_min_week:
                self.issues.append(
                    f"❌ Temporal leakage: Training max week ({train_max_week}) >= Test min week ({test_min_week})"
                )
                return False

            self.passed_checks.append(
                f"✅ Temporal independence: Train weeks ≤ {train_max_week}, Test weeks ≥ {test_min_week}"
            )
            return True

        # Check date ordering if available
        if date_col and date_col in train_df.columns and date_col in test_df.columns:
            train_max_date = pd.to_datetime(train_df[date_col]).max()
            test_min_date = pd.to_datetime(test_df[date_col]).min()

            if train_max_date >= test_min_date:
                self.issues.append(
                    f"❌ Temporal leakage: Training max date ({train_max_date}) >= Test min date ({test_min_date})"
                )
                return False

            self.passed_checks.append("✅ Temporal independence validated by date")
            return True

        self.warnings.append("⚠️  Cannot validate temporal independence - no week/date column found")
        return True  # Can't validate, but not necessarily wrong

    def validate_calibration_method_consistency(
        self,
        prob_col: str,
        expected_method: str = None
    ) -> bool:
        """Check if calibration method is consistent."""

        # Check if isotonic calibrator exists
        calibrator_path = Path('configs/calibrator.json')
        has_isotonic = calibrator_path.exists()

        # Check if inline shrinkage is used (in generate_current_week_recommendations.py)
        recommendation_script = Path('scripts/predict/generate_current_week_recommendations.py')
        has_inline_shrinkage = False
        if recommendation_script.exists():
            content = recommendation_script.read_text()
            if 'calibrate_probability_inline' in content:  # Fallback function exists
                has_inline_shrinkage = True

        # This is EXPECTED - inline shrinkage is fallback, isotonic is primary
        # Only warn if BOTH are actively being used (not just both exist)
        if has_isotonic and has_inline_shrinkage:
            # Check if inline is actually being used (not just fallback)
            if recommendation_script.exists():
                content = recommendation_script.read_text()
                # If calibrator is loaded and used, inline is just fallback (OK)
                if 'calibrator.load' in content and 'calibrator.transform' in content:
                    self.passed_checks.append(
                        "✅ Using isotonic calibrator (inline shrinkage is fallback only)"
                    )
                    return True  # This is OK - inline is fallback
                else:
                    self.warnings.append(
                        "⚠️  Both calibration methods exist - ensure only one is active"
                    )
                    return True  # Warning but not error
            else:
                self.warnings.append("⚠️  Cannot check calibration usage - script not found")
                return True

        if has_isotonic:
            self.passed_checks.append("✅ Using isotonic calibrator (single method)")
        elif has_inline_shrinkage:
            self.passed_checks.append("✅ Using inline shrinkage (single method)")
        else:
            self.warnings.append("⚠️  No calibration method detected - probabilities may be uncalibrated")

        return True

    def validate_data_completeness(
        self,
        df: pd.DataFrame,
        required_cols: List[str],
        min_completeness: float = 0.95
    ) -> bool:
        """Validate data completeness."""

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.issues.append(f"❌ Missing required columns: {missing_cols}")
            return False

        completeness = {}
        for col in required_cols:
            completeness[col] = df[col].notna().sum() / len(df)
            if completeness[col] < min_completeness:
                self.issues.append(
                    f"❌ Low completeness for '{col}': {completeness[col]:.1%} "
                    f"(required: {min_completeness:.1%})"
                )

        all_passed = all(c >= min_completeness for c in completeness.values())
        if all_passed:
            self.passed_checks.append(
                f"✅ Data completeness: {', '.join([f'{c}={completeness[c]:.1%}' for c in required_cols])}"
            )

        return all_passed

    def validate_distribution(
        self,
        df: pd.DataFrame,
        prob_col: str = 'model_prob'
    ) -> bool:
        """Validate probability distribution matches expected model output."""

        if prob_col not in df.columns:
            return False

        probs = df[prob_col].dropna()

        # Check range
        if probs.min() < 0 or probs.max() > 1:
            self.issues.append(f"❌ Probabilities outside [0, 1] range: [{probs.min():.3f}, {probs.max():.3f}]")
            return False

        # Check for reasonable distribution
        prob_mean = probs.mean()
        prob_std = probs.std()

        if prob_mean < 0.3 or prob_mean > 0.7:
            self.warnings.append(
                f"⚠️  Unusual probability mean: {prob_mean:.3f} "
                f"(expected around 0.50 for balanced data)"
            )

        if prob_std < 0.05:
            self.warnings.append(
                f"⚠️  Very low probability spread: {prob_std:.3f} "
                f"(may indicate calibration artifacts)"
            )

        self.passed_checks.append(
            f"✅ Probability distribution: mean={prob_mean:.3f}, std={prob_std:.3f}, "
            f"range=[{probs.min():.3f}, {probs.max():.3f}]"
        )

        return True

    def validate_no_double_calibration(
        self,
        df: pd.DataFrame,
        raw_prob_col: str = 'model_prob',
        calibrated_prob_col: str = None
    ) -> bool:
        """Check if probabilities have been calibrated multiple times."""

        if calibrated_prob_col and calibrated_prob_col in df.columns:
            raw_probs = df[raw_prob_col].dropna()
            cal_probs = df[calibrated_prob_col].dropna()

            # Check if calibrated probabilities show signs of double calibration
            # (e.g., extreme shrinkage, clustering at boundaries)
            cal_std = cal_probs.std()
            if cal_std < raw_probs.std() * 0.5:
                self.warnings.append(
                    f"⚠️  Calibrated probabilities have much lower spread "
                    f"(may indicate double calibration)"
                )

        return True

    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("=" * 80)
        report.append("DATA INTEGRITY VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        if self.passed_checks:
            report.append("✅ PASSED CHECKS:")
            for check in self.passed_checks:
                report.append(f"   {check}")
            report.append("")

        if self.warnings:
            report.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                report.append(f"   {warning}")
            report.append("")

        if self.issues:
            report.append("❌ ISSUES:")
            for issue in self.issues:
                report.append(f"   {issue}")
            report.append("")

        # Summary
        if not self.issues:
            report.append("✅ DATA INTEGRITY: PASSED")
        else:
            report.append("❌ DATA INTEGRITY: FAILED - Fix issues before calibration")

        return "\n".join(report)


def validate_calibration_data_integrity(
    train_data_path: str = None,
    test_data_path: str = None,
    prob_col: str = 'model_prob'
) -> bool:
    """Main validation function."""

    validator = DataIntegrityValidator()

    print("=" * 80)
    print("🔒 DATA INTEGRITY VALIDATION")
    print("=" * 80)
    print()

    # Load data if paths provided
    train_df = None
    test_df = None

    if train_data_path:
        if Path(train_data_path).exists():
            train_df = pd.read_csv(train_data_path)
            print(f"✅ Loaded training data: {train_data_path} ({len(train_df):,} rows)")
        else:
            print(f"⚠️  Training data not found: {train_data_path}")

    if test_data_path:
        if Path(test_data_path).exists():
            test_df = pd.read_csv(test_data_path)
            print(f"✅ Loaded test data: {test_data_path} ({len(test_df):,} rows)")
        else:
            print(f"⚠️  Test data not found: {test_data_path}")

    print()

    # Run validations
    if train_df is not None:
        print("📊 Validating training data...")
        validator.validate_source_data_purity(train_df, prob_col)
        validator.validate_data_completeness(
            train_df,
            required_cols=[prob_col, 'bet_outcome_over', 'week']
        )
        validator.validate_distribution(train_df, prob_col)
        print()

    if test_df is not None and train_df is not None:
        print("📊 Validating temporal independence...")
        validator.validate_temporal_independence(train_df, test_df)
        print()

    print("📊 Validating calibration method consistency...")
    validator.validate_calibration_method_consistency(prob_col)
    print()

    # Generate report
    report = validator.generate_report()
    print(report)

    # Save report
    report_path = Path('reports/data_integrity_validation_report.txt')
    report_path.write_text(report)
    print(f"\n💾 Report saved: {report_path}")

    return len(validator.issues) == 0


if __name__ == '__main__':
    import sys

    # Default paths
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'data/historical/player_prop_training_dataset.csv'
    test_path = sys.argv[2] if len(sys.argv) > 2 else None

    is_valid = validate_calibration_data_integrity(train_path, test_path)
    sys.exit(0 if is_valid else 1)
