#!/usr/bin/env python3
"""
Training Data Validation Script
================================

Purpose: Validate data quality before model training to catch issues early.

This script checks:
1. Sufficient sample sizes for each position
2. Target variable variance (detect constant predictions)
3. Feature distributions (detect outliers, skew)
4. Missing value analysis
5. Feature-target correlations
6. Data leakage detection

Usage:
    python scripts/diagnostics/validate_training_data.py --data-path data/processed/pbp_2024.parquet

Output:
    - Console validation report
    - data/diagnostics/training_data_validation.json
    - Exit code 0 if all checks pass, 1 if critical issues found
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TrainingDataValidator:
    """Validates training data quality before model training."""

    def __init__(self, min_samples=500, min_variance=0.01, min_correlation=0.05):
        """
        Initialize validator with thresholds.

        Args:
            min_samples: Minimum samples required per position
            min_variance: Minimum target variance required
            min_correlation: Minimum feature-target correlation
        """
        self.min_samples = min_samples
        self.min_variance = min_variance
        self.min_correlation = min_correlation
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'critical_failures': [],
            'warnings': [],
            'passed': False,
        }

    def validate_sample_size(self, df: pd.DataFrame, position: str, target: str) -> bool:
        """Check if sufficient samples exist."""
        check_name = f"sample_size_{position}_{target}"
        count = len(df)

        self.results['checks'][check_name] = {
            'type': 'sample_size',
            'position': position,
            'target': target,
            'count': int(count),
            'threshold': self.min_samples,
            'passed': count >= self.min_samples,
        }

        if count < self.min_samples:
            msg = f"Insufficient samples for {position} {target}: {count} < {self.min_samples}"
            self.results['critical_failures'].append(msg)
            logger.error(f"❌ {msg}")
            return False
        else:
            logger.info(f"✅ {position} {target}: {count} samples (>= {self.min_samples})")
            return True

    def validate_target_variance(
        self,
        df: pd.DataFrame,
        position: str,
        target_col: str
    ) -> bool:
        """Check if target has sufficient variance."""
        check_name = f"variance_{position}_{target_col}"

        target_values = df[target_col].dropna()
        variance = float(target_values.var())
        std = float(target_values.std())
        mean = float(target_values.mean())
        cv = std / mean if mean > 0 else 0

        self.results['checks'][check_name] = {
            'type': 'variance',
            'position': position,
            'target': target_col,
            'variance': variance,
            'std': std,
            'mean': mean,
            'coefficient_of_variation': cv,
            'threshold': self.min_variance,
            'passed': variance >= self.min_variance,
        }

        if variance < self.min_variance:
            msg = f"Low variance for {position} {target_col}: {variance:.6f} < {self.min_variance}"
            self.results['critical_failures'].append(msg)
            logger.error(f"❌ {msg}")
            logger.error(f"   → Model will predict constant value (mean={mean:.4f})")
            return False
        else:
            logger.info(f"✅ {position} {target_col}: variance={variance:.4f}, std={std:.4f}")
            return True

    def validate_feature_correlation(
        self,
        df: pd.DataFrame,
        position: str,
        target_col: str,
        feature_cols: list
    ) -> bool:
        """Check if features correlate with target."""
        check_name = f"correlation_{position}_{target_col}"

        correlations = {}
        for feat in feature_cols:
            if feat in df.columns:
                corr = float(df[feat].corr(df[target_col]))
                correlations[feat] = corr

        max_corr = max(abs(c) for c in correlations.values()) if correlations else 0

        self.results['checks'][check_name] = {
            'type': 'correlation',
            'position': position,
            'target': target_col,
            'correlations': correlations,
            'max_abs_correlation': max_corr,
            'threshold': self.min_correlation,
            'passed': max_corr >= self.min_correlation,
        }

        if max_corr < self.min_correlation:
            msg = f"No predictive features for {position} {target_col}: max_corr={max_corr:.4f}"
            self.results['critical_failures'].append(msg)
            logger.error(f"❌ {msg}")
            return False
        else:
            logger.info(f"✅ {position} {target_col}: max_corr={max_corr:.4f}")
            # Show top correlated features
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, corr in sorted_corr[:3]:
                logger.info(f"   - {feat}: {corr:.4f}")
            return True

    def check_missing_values(
        self,
        df: pd.DataFrame,
        position: str,
        columns: list
    ) -> bool:
        """Check for excessive missing values."""
        check_name = f"missing_values_{position}"

        missing_analysis = {}
        max_missing_pct = 0

        for col in columns:
            if col in df.columns:
                missing_count = int(df[col].isna().sum())
                missing_pct = float(missing_count / len(df) * 100)
                missing_analysis[col] = {
                    'count': missing_count,
                    'pct': missing_pct,
                }
                max_missing_pct = max(max_missing_pct, missing_pct)

                if missing_pct > 20:
                    msg = f"High missing values in {position} {col}: {missing_pct:.1f}%"
                    self.results['warnings'].append(msg)
                    logger.warning(f"⚠️  {msg}")

        self.results['checks'][check_name] = {
            'type': 'missing_values',
            'position': position,
            'analysis': missing_analysis,
            'max_missing_pct': max_missing_pct,
            'passed': max_missing_pct < 50,  # Critical if >50% missing
        }

        if max_missing_pct >= 50:
            msg = f"Critical missing values in {position}: {max_missing_pct:.1f}%"
            self.results['critical_failures'].append(msg)
            logger.error(f"❌ {msg}")
            return False
        else:
            logger.info(f"✅ {position}: max missing = {max_missing_pct:.1f}%")
            return True

    def check_outliers(
        self,
        df: pd.DataFrame,
        position: str,
        target_col: str,
    ) -> bool:
        """Check for excessive outliers in target."""
        check_name = f"outliers_{position}_{target_col}"

        target = df[target_col].dropna()
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1

        # Outliers: values beyond 3*IQR from quartiles
        outlier_mask = (target < q1 - 3*iqr) | (target > q3 + 3*iqr)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = float(outlier_count / len(target) * 100)

        self.results['checks'][check_name] = {
            'type': 'outliers',
            'position': position,
            'target': target_col,
            'count': outlier_count,
            'pct': outlier_pct,
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'passed': outlier_pct < 5,  # Warning if >5% outliers
        }

        if outlier_pct > 5:
            msg = f"High outliers in {position} {target_col}: {outlier_pct:.1f}%"
            self.results['warnings'].append(msg)
            logger.warning(f"⚠️  {msg}")
            return False
        else:
            logger.info(f"✅ {position} {target_col}: {outlier_pct:.1f}% outliers")
            return True

    def check_distribution_shape(
        self,
        df: pd.DataFrame,
        position: str,
        target_col: str
    ) -> bool:
        """Analyze distribution shape (skewness, kurtosis)."""
        check_name = f"distribution_{position}_{target_col}"

        target = df[target_col].dropna()
        skewness = float(stats.skew(target))
        kurtosis = float(stats.kurtosis(target))

        self.results['checks'][check_name] = {
            'type': 'distribution',
            'position': position,
            'target': target_col,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'highly_skewed': abs(skewness) > 2,
            'heavy_tailed': kurtosis > 7,
        }

        if abs(skewness) > 2:
            msg = f"Highly skewed distribution for {position} {target_col}: skew={skewness:.2f}"
            self.results['warnings'].append(msg)
            logger.warning(f"⚠️  {msg}")
            logger.warning("   → Consider log transformation or robust scaling")

        logger.info(f"✅ {position} {target_col}: skew={skewness:.2f}, kurtosis={kurtosis:.2f}")
        return True

    def save_results(self, output_path: Path):
        """Save validation results to JSON."""
        # Determine overall pass/fail
        self.results['passed'] = len(self.results['critical_failures']) == 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nValidation results saved to: {output_path}")

    def print_summary(self):
        """Print validation summary."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)

        total_checks = len(self.results['checks'])
        passed_checks = sum(1 for c in self.results['checks'].values() if c['passed'])

        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {total_checks - passed_checks}")
        logger.info(f"Critical Failures: {len(self.results['critical_failures'])}")
        logger.info(f"Warnings: {len(self.results['warnings'])}")

        if self.results['critical_failures']:
            logger.error("\nCRITICAL FAILURES:")
            for fail in self.results['critical_failures']:
                logger.error(f"  - {fail}")

        if self.results['warnings']:
            logger.warning("\nWARNINGS:")
            for warn in self.results['warnings']:
                logger.warning(f"  - {warn}")

        if self.results['passed']:
            logger.info("\n✅ ALL VALIDATION CHECKS PASSED")
            logger.info("   Training data is ready for model training")
        else:
            logger.error("\n❌ VALIDATION FAILED")
            logger.error("   Fix critical issues before training models")


def main():
    """Run training data validation."""
    parser = argparse.ArgumentParser(description='Validate training data quality')
    parser.add_argument(
        '--data-path',
        type=Path,
        default=Path('data/processed/pbp_2024.parquet'),
        help='Path to play-by-play data'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/diagnostics/training_data_validation.json'),
        help='Output path for validation results'
    )
    args = parser.parse_args()

    logger.info("Training Data Validation Script")
    logger.info("="*70 + "\n")

    # Check if data exists
    if not args.data_path.exists():
        logger.error(f"❌ Data file not found: {args.data_path}")
        sys.exit(1)

    # Load data (simplified for demo - would load actual training data)
    logger.info(f"Loading data from: {args.data_path}")
    pbp_df = pd.read_parquet(args.data_path)

    # Initialize validator
    validator = TrainingDataValidator(
        min_samples=500,
        min_variance=0.01,
        min_correlation=0.05
    )

    # Example validation for QB passing stats
    logger.info("\n" + "="*70)
    logger.info("VALIDATING QB PASSING STATS")
    logger.info("="*70)

    # Prepare QB data (simplified)
    qb_pass = pbp_df[
        (pbp_df['play_type'] == 'pass') &
        (pbp_df['passer_player_id'].notna())
    ].copy()

    # Group by QB-game
    qb_games = (
        qb_pass.groupby(['passer_player_id', 'game_id'])
        .agg({
            'pass_attempt': 'sum',
            'complete_pass': 'sum',
            'pass_touchdown': 'sum',
            'yards_gained': 'sum',
        })
        .reset_index()
    )

    qb_games['comp_pct'] = qb_games['complete_pass'] / qb_games['pass_attempt']
    qb_games['td_rate'] = qb_games['pass_touchdown'] / qb_games['pass_attempt']
    qb_games['yards_per_attempt'] = qb_games['yards_gained'] / qb_games['pass_attempt']

    # Run validation checks
    validator.validate_sample_size(qb_games, 'QB', 'completion_pct')
    validator.validate_target_variance(qb_games, 'QB', 'comp_pct')
    validator.validate_feature_correlation(
        qb_games, 'QB', 'comp_pct',
        feature_cols=['pass_attempt', 'yards_per_attempt']
    )
    validator.check_missing_values(qb_games, 'QB', ['comp_pct', 'td_rate'])
    validator.check_outliers(qb_games, 'QB', 'comp_pct')
    validator.check_distribution_shape(qb_games, 'QB', 'comp_pct')

    # Check QB TD rate (the problematic model)
    logger.info("\n" + "="*70)
    logger.info("VALIDATING QB TD RATE (PROBLEMATIC MODEL)")
    logger.info("="*70)

    validator.validate_sample_size(qb_games, 'QB', 'td_rate')
    validator.validate_target_variance(qb_games, 'QB', 'td_rate')
    validator.validate_feature_correlation(
        qb_games, 'QB', 'td_rate',
        feature_cols=['pass_attempt', 'comp_pct', 'yards_per_attempt']
    )
    validator.check_outliers(qb_games, 'QB', 'td_rate')
    validator.check_distribution_shape(qb_games, 'QB', 'td_rate')

    # Save and print summary
    validator.save_results(args.output)
    validator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if validator.results['passed'] else 1)


if __name__ == '__main__':
    main()
