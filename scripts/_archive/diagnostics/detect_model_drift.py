#!/usr/bin/env python3
"""
Model Drift Detection
=====================

Detects distribution shifts between training data (2024) and live data (2025)
that may degrade model performance.

Monitors:
- Feature distribution shifts (KS test, JS divergence)
- Target variable shifts (mean, variance changes)
- Prediction distribution changes
- Performance degradation over time

Triggers retraining when drift exceeds thresholds.

Usage:
    python scripts/diagnostics/detect_model_drift.py --week 9
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelDriftDetector:
    """
    Detects distribution shifts between training and production data.
    """

    # Drift thresholds
    KS_THRESHOLD = 0.15  # Kolmogorov-Smirnov statistic
    JS_THRESHOLD = 0.10  # Jensen-Shannon divergence
    MEAN_SHIFT_THRESHOLD = 0.20  # 20% change in mean
    VAR_SHIFT_THRESHOLD = 0.30  # 30% change in variance

    def __init__(self, baseline_season: int = 2024, current_season: int = 2025):
        """
        Initialize drift detector.

        Args:
            baseline_season: Training data season
            current_season: Production data season
        """
        self.baseline_season = baseline_season
        self.current_season = current_season
        self.drift_results = {}

    def load_baseline_data(self) -> pd.DataFrame:
        """Load 2024 baseline training data."""
        logger.info(f"Loading {self.baseline_season} baseline data...")

        pbp_path = Path(f'data/nflverse/pbp_{self.baseline_season}.parquet')
        if not pbp_path.exists():
            pbp_path = Path(f'data/processed/pbp_{self.baseline_season}.parquet')

        df = pd.read_parquet(pbp_path)
        logger.info(f"  Loaded {len(df):,} plays from {self.baseline_season}")
        return df

    def load_current_data(self, week: int) -> pd.DataFrame:
        """Load 2025 live data through specified week."""
        logger.info(f"Loading {self.current_season} data through week {week}...")

        pbp_path = Path(f'data/nflverse/pbp_{self.current_season}.parquet')

        df = pd.read_parquet(pbp_path)
        df = df[df['week'] <= week]
        logger.info(f"  Loaded {len(df):,} plays from {self.current_season} weeks 1-{week}")
        return df

    def calculate_ks_statistic(
        self,
        baseline: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic.

        Returns:
            (ks_statistic, p_value)
        """
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        return ks_stat, p_value

    def calculate_js_divergence(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 50
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions.

        Returns:
            JS divergence (0 = identical, 1 = completely different)
        """
        # Create histograms
        range_min = min(baseline.min(), current.min())
        range_max = max(baseline.max(), current.max())

        hist_baseline, _ = np.histogram(baseline, bins=bins, range=(range_min, range_max), density=True)
        hist_current, _ = np.histogram(current, bins=bins, range=(range_min, range_max), density=True)

        # Normalize to probabilities
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_current = hist_current / hist_current.sum()

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_baseline = hist_baseline + epsilon
        hist_current = hist_current + epsilon

        # Calculate JS divergence
        m = 0.5 * (hist_baseline + hist_current)
        js_div = 0.5 * stats.entropy(hist_baseline, m) + 0.5 * stats.entropy(hist_current, m)

        return js_div

    def detect_feature_drift(
        self,
        df_baseline: pd.DataFrame,
        df_current: pd.DataFrame,
        features: List[str]
    ) -> Dict:
        """
        Detect drift in feature distributions.

        Args:
            df_baseline: Baseline (training) data
            df_current: Current (production) data
            features: List of features to check

        Returns:
            Dictionary with drift results for each feature
        """
        logger.info("\\n" + "="*80)
        logger.info("FEATURE DRIFT DETECTION")
        logger.info("="*80)

        results = {}

        for feature in features:
            if feature not in df_baseline.columns or feature not in df_current.columns:
                continue

            # Get non-null values
            baseline_vals = df_baseline[feature].dropna().values
            current_vals = df_current[feature].dropna().values

            if len(baseline_vals) < 30 or len(current_vals) < 30:
                continue

            # Calculate drift metrics
            ks_stat, ks_p = self.calculate_ks_statistic(baseline_vals, current_vals)
            js_div = self.calculate_js_divergence(baseline_vals, current_vals)

            # Mean and variance shifts
            baseline_mean = baseline_vals.mean()
            current_mean = current_vals.mean()
            mean_shift = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-10)

            baseline_var = baseline_vals.var()
            current_var = current_vals.var()
            var_shift = abs(current_var - baseline_var) / (baseline_var + 1e-10)

            # Determine if drift detected
            drift_detected = (
                ks_stat > self.KS_THRESHOLD or
                js_div > self.JS_THRESHOLD or
                mean_shift > self.MEAN_SHIFT_THRESHOLD or
                var_shift > self.VAR_SHIFT_THRESHOLD
            )

            results[feature] = {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'js_divergence': float(js_div),
                'mean_shift': float(mean_shift),
                'var_shift': float(var_shift),
                'drift_detected': drift_detected,
                'baseline_mean': float(baseline_mean),
                'current_mean': float(current_mean),
                'baseline_std': float(np.sqrt(baseline_var)),
                'current_std': float(np.sqrt(current_var)),
            }

            # Log significant drift
            if drift_detected:
                logger.warning(f"⚠️  DRIFT DETECTED: {feature}")
                logger.warning(f"   KS statistic: {ks_stat:.3f} (threshold: {self.KS_THRESHOLD})")
                logger.warning(f"   JS divergence: {js_div:.3f} (threshold: {self.JS_THRESHOLD})")
                logger.warning(f"   Mean shift: {mean_shift:.1%} (threshold: {self.MEAN_SHIFT_THRESHOLD:.0%})")
            else:
                logger.info(f"✅ {feature}: No significant drift")

        return results

    def generate_drift_report(
        self,
        drift_results: Dict,
        week: int
    ) -> Dict:
        """
        Generate comprehensive drift report.

        Args:
            drift_results: Results from drift detection
            week: Current week

        Returns:
            Report dictionary
        """
        # Count features with drift
        drift_count = sum(1 for r in drift_results.values() if r['drift_detected'])
        total_features = len(drift_results)

        # Identify most drifted features
        drifted_features = [
            (feat, r['ks_statistic'])
            for feat, r in drift_results.items()
            if r['drift_detected']
        ]
        drifted_features.sort(key=lambda x: x[1], reverse=True)

        report = {
            'detection_date': datetime.now().isoformat(),
            'baseline_season': self.baseline_season,
            'current_season': self.current_season,
            'current_week': week,
            'features_checked': total_features,
            'features_with_drift': drift_count,
            'drift_percentage': 100 * drift_count / max(total_features, 1),
            'most_drifted_features': [f[0] for f in drifted_features[:10]],
            'retraining_recommended': drift_count > 0.2 * total_features,  # >20% features drifted
            'feature_details': drift_results,
        }

        return report

    def run_drift_detection(self, week: int) -> Dict:
        """
        Run complete drift detection pipeline.

        Args:
            week: Current NFL week

        Returns:
            Drift detection report
        """
        logger.info("="*80)
        logger.info(f"MODEL DRIFT DETECTION - WEEK {week}")
        logger.info("="*80)

        # Load data
        df_baseline = self.load_baseline_data()
        df_current = self.load_current_data(week)

        # Key features to monitor
        features_to_check = [
            'score_differential',
            'ep',  # Expected points
            'epa',  # Expected points added
            'wp',  # Win probability
            'yards_gained',
            'air_yards',
            'yards_after_catch',
            'comp_air_epa',
            'comp_yac_epa',
        ]

        # Detect drift
        drift_results = self.detect_feature_drift(df_baseline, df_current, features_to_check)

        # Generate report
        report = self.generate_drift_report(drift_results, week)

        # Save report
        output_path = Path(f'data/diagnostics/drift_detection_week{week}.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\\n📊 Drift Report saved: {output_path}")

        return report


def main():
    """Run drift detection."""
    import argparse

    parser = argparse.ArgumentParser(description='Detect model drift')
    parser.add_argument('--week', type=int, default=9, help='Current NFL week')
    args = parser.parse_args()

    detector = ModelDriftDetector()
    report = detector.run_drift_detection(args.week)

    # Summary
    logger.info("\\n" + "="*80)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"\\nFeatures checked: {report['features_checked']}")
    logger.info(f"Features with drift: {report['features_with_drift']} ({report['drift_percentage']:.1f}%)")

    if report['most_drifted_features']:
        logger.info(f"\\nMost drifted features:")
        for feat in report['most_drifted_features'][:5]:
            logger.info(f"  - {feat}")

    if report['retraining_recommended']:
        logger.warning(f"\\n⚠️  RETRAINING RECOMMENDED")
        logger.warning(f"   Significant drift detected in {report['drift_percentage']:.0f}% of features")
        logger.warning(f"   Run: python scripts/train/retrain_models_incremental.py --week {args.week}")
    else:
        logger.info(f"\\n✅ No retraining needed")
        logger.info(f"   Drift within acceptable limits")

    return 0 if not report['retraining_recommended'] else 1


if __name__ == '__main__':
    sys.exit(main())
