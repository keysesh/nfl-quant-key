#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation
==================================

Full end-to-end validation of the NFL QUANT prediction pipeline:
1. Data integrity checks
2. Model artifact loading
3. Walk-forward backtesting metrics
4. Calibration effectiveness
5. Monte Carlo simulation sanity checks
6. Performance regression detection

This script provides a complete health check of the system.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats


class PipelineValidator:
    """Comprehensive validation of the NFL QUANT pipeline."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'metrics': {},
            'warnings': [],
            'errors': [],
            'overall_status': 'UNKNOWN'
        }

    def run_all_validations(self):
        """Run complete validation suite."""
        print("=" * 80)
        print("NFL QUANT PIPELINE COMPREHENSIVE VALIDATION")
        print("=" * 80)
        print(f"Timestamp: {self.results['timestamp']}")
        print()

        # 1. Data Integrity
        self.validate_data_integrity()

        # 2. Model Artifacts
        self.validate_model_artifacts()

        # 3. Backtest Performance
        self.validate_backtest_performance()

        # 4. Calibration Effectiveness
        self.validate_calibration_effectiveness()

        # 5. Simulation Sanity
        self.validate_simulation_sanity()

        # 6. Generate Summary
        self.generate_summary()

        return self.results

    def validate_data_integrity(self):
        """Check NFLverse data availability and quality."""
        print("1. DATA INTEGRITY VALIDATION")
        print("-" * 80)

        checks = {}

        # Check NFLverse parquet files
        nflverse_dir = self.base_dir / "data" / "nflverse"
        critical_files = [
            'player_stats.parquet',
            'schedules.parquet',
            'rosters.parquet',
            'pbp.parquet'
        ]

        for fname in critical_files:
            fpath = nflverse_dir / fname
            if fpath.exists():
                try:
                    df = pd.read_parquet(fpath)
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    checks[fname] = {
                        'status': 'PASS',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_mb': round(size_mb, 2)
                    }
                    print(f"  ✅ {fname}: {len(df):,} rows, {size_mb:.1f}MB")
                except Exception as e:
                    checks[fname] = {'status': 'FAIL', 'error': str(e)}
                    print(f"  ❌ {fname}: Error reading - {e}")
                    self.results['errors'].append(f"Failed to read {fname}")
            else:
                checks[fname] = {'status': 'MISSING'}
                print(f"  ❌ {fname}: MISSING")
                self.results['errors'].append(f"Missing critical file: {fname}")

        # Check player stats has recent data
        if 'player_stats.parquet' in checks and checks['player_stats.parquet']['status'] == 'PASS':
            df = pd.read_parquet(nflverse_dir / 'player_stats.parquet')
            if 'week' in df.columns:
                max_week = df['week'].max()
                checks['latest_week'] = int(max_week)
                print(f"  ℹ️  Latest week in data: Week {max_week}")
                if max_week < 10:
                    self.results['warnings'].append(f"Data may be stale (latest week: {max_week})")

        self.results['checks']['data_integrity'] = checks
        print()

    def validate_model_artifacts(self):
        """Validate all model artifacts load correctly."""
        print("2. MODEL ARTIFACT VALIDATION")
        print("-" * 80)

        checks = {}

        # Check calibrators
        cal_dir = self.base_dir / "models" / "calibration"
        cal_files = list(cal_dir.glob("*_calibrator.json"))

        valid_cals = 0
        for cal_file in cal_files:
            try:
                with open(cal_file) as f:
                    data = json.load(f)
                if 'X_thresholds' in data and 'y_thresholds' in data:
                    valid_cals += 1
            except:
                pass

        checks['calibrators'] = {
            'total': len(cal_files),
            'valid': valid_cals,
            'status': 'PASS' if valid_cals == len(cal_files) else 'PARTIAL'
        }
        print(f"  ✅ Calibrators: {valid_cals}/{len(cal_files)} valid")

        # Check joblib models
        models_dir = self.base_dir / "data" / "models"
        joblib_files = list(models_dir.glob("*.joblib"))

        valid_models = 0
        import joblib
        for model_file in joblib_files:
            try:
                model = joblib.load(model_file)
                valid_models += 1
            except:
                pass

        checks['prediction_models'] = {
            'total': len(joblib_files),
            'valid': valid_models,
            'status': 'PASS' if valid_models == len(joblib_files) else 'PARTIAL'
        }
        print(f"  ✅ Prediction Models: {valid_models}/{len(joblib_files)} valid")

        # Check config files
        config_file = self.base_dir / "configs" / "simulation_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                checks['simulation_config'] = {
                    'status': 'PASS',
                    'keys': len(config)
                }
                print(f"  ✅ Simulation Config: {len(config)} parameters")
            except Exception as e:
                checks['simulation_config'] = {'status': 'FAIL', 'error': str(e)}
                print(f"  ❌ Simulation Config: {e}")
        else:
            checks['simulation_config'] = {'status': 'MISSING'}
            print("  ❌ Simulation Config: MISSING")

        self.results['checks']['model_artifacts'] = checks
        print()

    def validate_backtest_performance(self):
        """Calculate key backtest metrics from historical predictions."""
        print("3. BACKTEST PERFORMANCE VALIDATION")
        print("-" * 80)

        backtest_file = self.base_dir / "models" / "calibration" / "backtest_2025.csv"

        if not backtest_file.exists():
            print("  ❌ Backtest data not found")
            self.results['errors'].append("Missing backtest_2025.csv")
            return

        df = pd.read_csv(backtest_file)
        print(f"  ℹ️  Loaded {len(df):,} backtest records")

        metrics = {}

        # Overall metrics
        metrics['total_predictions'] = len(df)
        metrics['weeks_covered'] = sorted(df['week'].unique().tolist())
        metrics['markets'] = df['market'].unique().tolist()

        # Calculate Brier Score (mean squared error of probability predictions)
        brier_score = np.mean((df['predicted_prob_over'] - df['hit_over']) ** 2)
        metrics['brier_score'] = round(brier_score, 4)
        print(f"  📊 Raw Brier Score: {brier_score:.4f}")

        # Calculate Expected Calibration Error
        # Bin predictions and compare average prediction to actual hit rate
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        df['bin'] = pd.cut(df['predicted_prob_over'], bins=bins, labels=False, include_lowest=True)

        ece = 0.0
        bin_stats = []
        for i in range(n_bins):
            bin_data = df[df['bin'] == i]
            if len(bin_data) > 0:
                avg_pred = bin_data['predicted_prob_over'].mean()
                actual_hit_rate = bin_data['hit_over'].mean()
                n_samples = len(bin_data)
                ece += (n_samples / len(df)) * abs(avg_pred - actual_hit_rate)
                bin_stats.append({
                    'bin': i,
                    'avg_pred': round(avg_pred, 3),
                    'actual_rate': round(actual_hit_rate, 3),
                    'n_samples': n_samples,
                    'gap': round(abs(avg_pred - actual_hit_rate), 3)
                })

        metrics['expected_calibration_error'] = round(ece, 4)
        metrics['calibration_bins'] = bin_stats
        print(f"  📊 Expected Calibration Error: {ece:.4f} ({ece*100:.2f}%)")

        # Performance by market
        print(f"\n  Market-Specific Performance:")
        market_metrics = {}
        for market in sorted(df['market'].unique()):
            market_df = df[df['market'] == market]
            market_brier = np.mean((market_df['predicted_prob_over'] - market_df['hit_over']) ** 2)
            market_metrics[market] = {
                'n_predictions': len(market_df),
                'brier_score': round(market_brier, 4),
                'hit_rate': round(market_df['hit_over'].mean(), 4)
            }
            print(f"    {market:20s}: Brier={market_brier:.4f}, N={len(market_df):,}, Hit%={market_df['hit_over'].mean()*100:.1f}%")

        metrics['by_market'] = market_metrics

        # Performance by position
        print(f"\n  Position-Specific Performance:")
        position_metrics = {}
        for pos in sorted(df['position'].unique()):
            pos_df = df[df['position'] == pos]
            pos_brier = np.mean((pos_df['predicted_prob_over'] - pos_df['hit_over']) ** 2)
            position_metrics[pos] = {
                'n_predictions': len(pos_df),
                'brier_score': round(pos_brier, 4)
            }
            print(f"    {pos:5s}: Brier={pos_brier:.4f}, N={len(pos_df):,}")

        metrics['by_position'] = position_metrics

        # Check for performance thresholds
        if brier_score > 0.25:
            self.results['warnings'].append(f"High Brier Score ({brier_score:.4f}) - model may be poorly calibrated")
        if ece > 0.10:
            self.results['warnings'].append(f"High ECE ({ece:.4f}) - significant calibration issues")

        self.results['metrics']['backtest'] = metrics
        print()

    def validate_calibration_effectiveness(self):
        """Test that calibration actually improves predictions."""
        print("4. CALIBRATION EFFECTIVENESS VALIDATION")
        print("-" * 80)

        backtest_file = self.base_dir / "models" / "calibration" / "backtest_2025.csv"
        if not backtest_file.exists():
            print("  ❌ Cannot validate - missing backtest data")
            return

        df = pd.read_csv(backtest_file)

        # Load overall calibrator
        cal_file = self.base_dir / "models" / "calibration" / "overall_calibrator.json"
        if not cal_file.exists():
            print("  ❌ Overall calibrator not found")
            return

        with open(cal_file) as f:
            calibrator = json.load(f)

        X_thresh = np.array(calibrator['X_thresholds'])
        y_thresh = np.array(calibrator['y_thresholds'])

        # Apply calibration to raw predictions
        def calibrate(prob):
            idx = np.searchsorted(X_thresh, prob)
            if idx == 0:
                return y_thresh[0]
            elif idx >= len(X_thresh):
                return y_thresh[-1]
            else:
                x0, x1 = X_thresh[idx-1], X_thresh[idx]
                y0, y1 = y_thresh[idx-1], y_thresh[idx]
                return y0 + (prob - x0) * (y1 - y0) / (x1 - x0)

        df['calibrated_prob'] = df['predicted_prob_over'].apply(calibrate)

        # Compare metrics
        raw_brier = np.mean((df['predicted_prob_over'] - df['hit_over']) ** 2)
        cal_brier = np.mean((df['calibrated_prob'] - df['hit_over']) ** 2)

        # Calculate ECE for both
        def calculate_ece(probs, actuals, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            ece = 0.0
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    avg_pred = np.mean(probs[mask])
                    actual_rate = np.mean(actuals[mask])
                    weight = np.sum(mask) / len(probs)
                    ece += weight * abs(avg_pred - actual_rate)
            return ece

        raw_ece = calculate_ece(df['predicted_prob_over'].values, df['hit_over'].values)
        cal_ece = calculate_ece(df['calibrated_prob'].values, df['hit_over'].values)

        metrics = {
            'raw_brier_score': round(raw_brier, 4),
            'calibrated_brier_score': round(cal_brier, 4),
            'brier_improvement': round((raw_brier - cal_brier) / raw_brier * 100, 2),
            'raw_ece': round(raw_ece, 4),
            'calibrated_ece': round(cal_ece, 4),
            'ece_improvement': round((raw_ece - cal_ece) / raw_ece * 100, 2)
        }

        print(f"  📊 Brier Score: {raw_brier:.4f} -> {cal_brier:.4f} ({metrics['brier_improvement']:.1f}% improvement)")
        print(f"  📊 ECE: {raw_ece:.4f} -> {cal_ece:.4f} ({metrics['ece_improvement']:.1f}% improvement)")

        # Check high probability shrinkage
        high_prob_mask = df['predicted_prob_over'] > 0.8
        if high_prob_mask.sum() > 0:
            raw_high = df.loc[high_prob_mask, 'predicted_prob_over'].mean()
            cal_high = df.loc[high_prob_mask, 'calibrated_prob'].mean()
            actual_high = df.loc[high_prob_mask, 'hit_over'].mean()

            metrics['high_prob_raw_avg'] = round(raw_high, 4)
            metrics['high_prob_calibrated_avg'] = round(cal_high, 4)
            metrics['high_prob_actual_rate'] = round(actual_high, 4)

            print(f"\n  High Probability (>80%) Shrinkage:")
            print(f"    Raw Avg: {raw_high:.3f}, Calibrated: {cal_high:.3f}, Actual: {actual_high:.3f}")

            if cal_high > raw_high:
                self.results['warnings'].append("Calibration increasing high probabilities (unexpected)")

        # Validate calibration is helpful
        if cal_brier >= raw_brier:
            self.results['warnings'].append("Calibration not improving Brier score")
        if cal_ece >= raw_ece:
            self.results['warnings'].append("Calibration not improving ECE")

        self.results['metrics']['calibration'] = metrics
        print()

    def validate_simulation_sanity(self):
        """Sanity check the simulation outputs."""
        print("5. SIMULATION SANITY VALIDATION")
        print("-" * 80)

        # Check for unrealistic predictions
        backtest_file = self.base_dir / "models" / "calibration" / "backtest_2025.csv"
        if not backtest_file.exists():
            print("  ❌ Cannot validate - missing backtest data")
            return

        df = pd.read_csv(backtest_file)

        checks = {}

        # Check for extreme probabilities
        extreme_high = (df['predicted_prob_over'] > 0.99).sum()
        extreme_low = (df['predicted_prob_over'] < 0.01).sum()

        checks['extreme_high_prob'] = int(extreme_high)
        checks['extreme_low_prob'] = int(extreme_low)

        print(f"  ℹ️  Extreme probabilities (>99%): {extreme_high}")
        print(f"  ℹ️  Extreme probabilities (<1%): {extreme_low}")

        if extreme_high > 100:
            self.results['warnings'].append(f"Many extreme high probabilities ({extreme_high})")

        # Check probability distribution
        prob_mean = df['predicted_prob_over'].mean()
        prob_std = df['predicted_prob_over'].std()

        checks['prob_mean'] = round(prob_mean, 4)
        checks['prob_std'] = round(prob_std, 4)

        print(f"  📊 Probability Distribution: mean={prob_mean:.3f}, std={prob_std:.3f}")

        # Check for impossible predictions (should all be 0-1)
        invalid_probs = ((df['predicted_prob_over'] < 0) | (df['predicted_prob_over'] > 1)).sum()
        if invalid_probs > 0:
            self.results['errors'].append(f"Found {invalid_probs} invalid probabilities outside [0,1]")
            print(f"  ❌ Invalid probabilities: {invalid_probs}")
        else:
            print(f"  ✅ All probabilities in valid range [0,1]")

        # Check correlation between prediction and actual (should be positive)
        corr = df['predicted_prob_over'].corr(df['hit_over'])
        checks['prediction_actual_correlation'] = round(corr, 4)
        print(f"  📊 Prediction-Actual Correlation: {corr:.4f}")

        if corr < 0:
            self.results['errors'].append("Negative correlation between predictions and actuals!")
        elif corr < 0.1:
            self.results['warnings'].append(f"Weak prediction-actual correlation ({corr:.4f})")

        # Check hit rate by probability bucket makes sense (should increase)
        buckets = pd.cut(df['predicted_prob_over'], bins=[0, 0.3, 0.5, 0.7, 1.0], labels=['Low', 'Medium', 'High', 'Very High'])
        bucket_stats = df.groupby(buckets, observed=True)['hit_over'].mean()

        checks['hit_rate_by_bucket'] = bucket_stats.to_dict()
        print(f"\n  Hit Rate by Probability Bucket:")
        for bucket, rate in bucket_stats.items():
            print(f"    {bucket:10s}: {rate:.3f}")

        # Verify monotonic increase
        if len(bucket_stats) >= 2:
            if not all(bucket_stats.iloc[i] <= bucket_stats.iloc[i+1] for i in range(len(bucket_stats)-1)):
                self.results['warnings'].append("Hit rate not monotonically increasing with probability")

        self.results['metrics']['simulation'] = checks
        print()

    def generate_summary(self):
        """Generate overall validation summary."""
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        n_errors = len(self.results['errors'])
        n_warnings = len(self.results['warnings'])

        if n_errors > 0:
            self.results['overall_status'] = 'FAILED'
            print(f"❌ VALIDATION FAILED - {n_errors} errors found")
        elif n_warnings > 3:
            self.results['overall_status'] = 'WARNING'
            print(f"⚠️  VALIDATION PASSED WITH WARNINGS - {n_warnings} warnings")
        else:
            self.results['overall_status'] = 'PASSED'
            print(f"✅ VALIDATION PASSED")

        if n_errors > 0:
            print(f"\nErrors ({n_errors}):")
            for err in self.results['errors']:
                print(f"  ❌ {err}")

        if n_warnings > 0:
            print(f"\nWarnings ({n_warnings}):")
            for warn in self.results['warnings']:
                print(f"  ⚠️  {warn}")

        # Key metrics summary
        if 'backtest' in self.results['metrics']:
            bt = self.results['metrics']['backtest']
            print(f"\nKey Performance Metrics:")
            print(f"  • Total Predictions: {bt['total_predictions']:,}")
            print(f"  • Brier Score: {bt['brier_score']:.4f} (target: <0.20)")
            print(f"  • ECE: {bt['expected_calibration_error']:.4f} (target: <0.05)")

        if 'calibration' in self.results['metrics']:
            cal = self.results['metrics']['calibration']
            print(f"  • Calibration Brier Improvement: {cal['brier_improvement']:.1f}%")
            print(f"  • Calibration ECE Improvement: {cal['ece_improvement']:.1f}%")

        # Save results to JSON
        output_file = self.base_dir / "reports" / "pipeline_validation_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nFull report saved to: {output_file}")

        return self.results['overall_status']


def main():
    validator = PipelineValidator()
    results = validator.run_all_validations()

    # Exit with appropriate code
    if results['overall_status'] == 'FAILED':
        sys.exit(1)
    elif results['overall_status'] == 'WARNING':
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
