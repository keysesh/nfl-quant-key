#!/usr/bin/env python3
"""
Calibration Quality Monitoring Script
======================================

Monitors calibration quality metrics and alerts if degradation detected.

Usage:
    python scripts/monitor/monitor_calibration_quality.py
    python scripts/monitor/monitor_calibration_quality.py --week 9
    python scripts/monitor/monitor_calibration_quality.py --alert-threshold 0.05
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.calibrator_loader import (
    load_calibrator_for_market,
    get_available_market_calibrators,
    validate_all_calibrators
)


class CalibrationMonitor:
    """Monitors calibration quality and generates alerts."""

    def __init__(self, alert_threshold_mace: float = 0.05, alert_threshold_brier: float = 0.30):
        self.alert_threshold_mace = alert_threshold_mace
        self.alert_threshold_brier = alert_threshold_brier
        self.alerts = []

    def load_recent_results(self, week: int = None) -> pd.DataFrame:
        """Load recent backtest results for monitoring."""

        # Try to load week-specific results
        if week:
            week_file = f'reports/detailed_bet_analysis_week{week}.csv'
            if Path(week_file).exists():
                return pd.read_csv(week_file)

        # Fall back to comprehensive results
        result_files = [
            'reports/FRESH_BACKTEST_WEEKS_1_8_CALIBRATED.csv',
            'reports/detailed_bet_analysis_weekall.csv'
        ]

        for file_path in result_files:
            path = Path(file_path)
            if path.exists():
                df = pd.read_csv(path)

                # Filter to week if specified
                if week and 'week' in df.columns:
                    df = df[df['week'] == week]

                return df

        raise FileNotFoundError("No recent results found for monitoring")

    def calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate calibration metrics."""

        from sklearn.metrics import brier_score_loss

        # Remove NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            return None

        # Calculate metrics
        brier = brier_score_loss(y_true, y_pred)
        mace = abs(y_pred.mean() - y_true.mean())

        # Bin-wise calibration
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_metrics = []

        for i in range(len(bins) - 1):
            bin_mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
            if bin_mask.sum() > 0:
                bin_mace = abs(y_pred[bin_mask].mean() - y_true[bin_mask].mean())
                bin_metrics.append({
                    'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                    'count': int(bin_mask.sum()),
                    'avg_pred': float(y_pred[bin_mask].mean()),
                    'avg_actual': float(y_true[bin_mask].mean()),
                    'mace': float(bin_mace)
                })

        return {
            'brier_score': float(brier),
            'mace': float(mace),
            'samples': len(y_true),
            'win_rate': float(y_true.mean()),
            'avg_prob': float(y_pred.mean()),
            'bin_metrics': bin_metrics
        }

    def monitor_market_calibration(
        self,
        df: pd.DataFrame,
        market: str
    ) -> Dict:
        """Monitor calibration for a specific market."""

        market_df = df[df['market'] == market].copy()

        if len(market_df) == 0:
            return None

        # Determine which probability column to use
        if 'model_prob' in market_df.columns:
            y_pred = market_df['model_prob'].values
        elif 'model_prob_raw' in market_df.columns:
            # Apply calibration
            calibrator = load_calibrator_for_market(market)
            y_pred = calibrator.transform(market_df['model_prob_raw'].values)
        else:
            return None

        y_true = market_df['bet_won'].values

        metrics = self.calculate_calibration_metrics(y_true, y_pred)

        if metrics:
            metrics['market'] = market

            # Check for alerts
            if metrics['mace'] > self.alert_threshold_mace:
                self.alerts.append({
                    'severity': 'HIGH',
                    'market': market,
                    'issue': f"MACE ({metrics['mace']:.4f}) exceeds threshold ({self.alert_threshold_mace})",
                    'metric': 'MACE',
                    'value': metrics['mace'],
                    'threshold': self.alert_threshold_mace
                })

            if metrics['brier_score'] > self.alert_threshold_brier:
                self.alerts.append({
                    'severity': 'MEDIUM',
                    'market': market,
                    'issue': f"Brier score ({metrics['brier_score']:.4f}) exceeds threshold ({self.alert_threshold_brier})",
                    'metric': 'Brier',
                    'value': metrics['brier_score'],
                    'threshold': self.alert_threshold_brier
                })

            # Check for high-probability bin issues
            for bin_metric in metrics['bin_metrics']:
                if bin_metric['avg_pred'] >= 0.70 and bin_metric['mace'] > 0.10:
                    self.alerts.append({
                        'severity': 'HIGH',
                        'market': market,
                        'issue': f"High-prob bin {bin_metric['bin']} has MACE {bin_metric['mace']:.4f}",
                        'metric': 'Bin MACE',
                        'value': bin_metric['mace'],
                        'threshold': 0.10,
                        'bin': bin_metric['bin']
                    })

        return metrics

    def generate_report(self, week: int = None):
        """Generate calibration quality report."""

        print("=" * 100)
        print("CALIBRATION QUALITY MONITORING REPORT")
        print("=" * 100)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if week:
            print(f"Week: {week}")
        print()

        # Load data
        print("📊 Loading recent results...")
        try:
            df = self.load_recent_results(week)
            print(f"✅ Loaded {len(df):,} bets")
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return

        # Check that required columns exist
        if 'bet_won' not in df.columns:
            print("❌ Error: 'bet_won' column not found in results")
            return

        if 'market' not in df.columns:
            print("❌ Error: 'market' column not found in results")
            return

        print()

        # Monitor each market
        print("=" * 100)
        print("MARKET-SPECIFIC CALIBRATION QUALITY")
        print("=" * 100)
        print()

        markets = df['market'].unique()
        all_metrics = []

        for market in sorted(markets):
            print(f"📈 {market}")
            print("-" * 80)

            metrics = self.monitor_market_calibration(df, market)

            if metrics:
                all_metrics.append(metrics)

                print(f"  Samples: {metrics['samples']:,}")
                print(f"  Brier Score: {metrics['brier_score']:.4f}")
                print(f"  MACE: {metrics['mace']:.4f}")
                print(f"  Win Rate: {metrics['win_rate']:.2%}")
                print(f"  Avg Predicted: {metrics['avg_prob']:.2%}")

                if metrics['bin_metrics']:
                    print(f"\n  Bin-wise calibration:")
                    print(f"  {'Bin':<12} | {'Count':>6} | {'Avg Pred':>10} | {'Avg Actual':>12} | {'MACE':>8}")
                    print(f"  {'-'*60}")
                    for bin_m in metrics['bin_metrics']:
                        print(f"  {bin_m['bin']:<12} | {bin_m['count']:6d} | {bin_m['avg_pred']:10.2%} | {bin_m['avg_actual']:12.2%} | {bin_m['mace']:8.4f}")

                print()
            else:
                print(f"  ⚠️  No data available")
                print()

        # Alerts
        if self.alerts:
            print("=" * 100)
            print(f"⚠️  ALERTS ({len(self.alerts)})")
            print("=" * 100)
            print()

            for alert in self.alerts:
                severity_icon = "🔴" if alert['severity'] == 'HIGH' else "🟡"
                print(f"{severity_icon} [{alert['severity']}] {alert['market']}")
                print(f"   {alert['issue']}")
                print()

        else:
            print("=" * 100)
            print("✅ NO ALERTS - All calibration metrics within acceptable ranges")
            print("=" * 100)
            print()

        # Summary
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print()

        if all_metrics:
            avg_brier = np.mean([m['brier_score'] for m in all_metrics])
            avg_mace = np.mean([m['mace'] for m in all_metrics])
            total_samples = sum([m['samples'] for m in all_metrics])

            print(f"Markets monitored: {len(all_metrics)}")
            print(f"Total samples: {total_samples:,}")
            print(f"Average Brier Score: {avg_brier:.4f}")
            print(f"Average MACE: {avg_mace:.4f}")
            print(f"Alerts: {len(self.alerts)}")
            print()

            if avg_mace < 0.03:
                print("✅ Excellent calibration quality (MACE < 3%)")
            elif avg_mace < 0.05:
                print("✅ Good calibration quality (MACE < 5%)")
            elif avg_mace < 0.10:
                print("⚠️  Acceptable calibration quality (MACE < 10%)")
            else:
                print("❌ Poor calibration quality (MACE > 10%) - Consider retraining")

        # Save report
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'week': week,
            'metrics': all_metrics,
            'alerts': self.alerts,
            'thresholds': {
                'mace': self.alert_threshold_mace,
                'brier': self.alert_threshold_brier
            }
        }

        report_file = Path('reports/calibration_monitoring_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Full report saved to: {report_file}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Monitor calibration quality and generate alerts'
    )
    parser.add_argument(
        '--week',
        type=int,
        help='Specific week to monitor (default: all available data)'
    )
    parser.add_argument(
        '--alert-threshold',
        type=float,
        default=0.05,
        help='MACE threshold for alerts (default: 0.05)'
    )
    parser.add_argument(
        '--brier-threshold',
        type=float,
        default=0.30,
        help='Brier score threshold for alerts (default: 0.30)'
    )

    args = parser.parse_args()

    monitor = CalibrationMonitor(
        alert_threshold_mace=args.alert_threshold,
        alert_threshold_brier=args.brier_threshold
    )

    monitor.generate_report(week=args.week)

    # Exit with error code if alerts found
    sys.exit(1 if monitor.alerts else 0)


if __name__ == "__main__":
    main()
