#!/usr/bin/env python3
"""
Calibration Analysis
====================

Analyzes predicted probabilities vs actual hit rates to identify calibration issues.
Creates calibration curves and identifies which markets need calibration correction.

Usage:
    python scripts/backtest/analyze_calibration.py --bet-log reports/backtest_bet_log_20251118_222905.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """Analyzes betting calibration and identifies correction opportunities."""

    def __init__(self, bet_log_path: Path):
        self.bet_log_path = bet_log_path
        self.bets = None
        self.calibration_data = {}

    def load_bet_log(self):
        """Load bet log CSV."""
        logger.info(f"Loading bet log from {self.bet_log_path}...")
        self.bets = pd.read_csv(self.bet_log_path)
        logger.info(f"  ✓ Loaded {len(self.bets):,} bets")

    def analyze_overall_calibration(self) -> Dict:
        """Analyze overall calibration across all bets."""
        logger.info("\n" + "="*80)
        logger.info("OVERALL CALIBRATION ANALYSIS")
        logger.info("="*80 + "\n")

        # Create probability bins
        bins = [0.0, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        self.bets['prob_bin'] = pd.cut(self.bets['model_prob'], bins=bins, include_lowest=True)

        # Calculate hit rate by bin
        calibration_by_bin = self.bets.groupby('prob_bin', observed=True).agg({
            'hit': ['count', 'sum', 'mean'],
            'model_prob': 'mean',
            'edge': 'mean',
            'profit': 'sum'
        }).reset_index()

        calibration_by_bin.columns = ['prob_bin', 'count', 'wins', 'hit_rate', 'avg_predicted_prob', 'avg_edge', 'total_profit']

        logger.info("Calibration by Predicted Probability Bin:\n")
        logger.info(f"{'Predicted Prob':>15} | {'Count':>6} | {'Wins':>5} | {'Hit Rate':>8} | {'Avg Edge':>8} | {'Profit':>8}")
        logger.info("-" * 80)

        for _, row in calibration_by_bin.iterrows():
            logger.info(
                f"{str(row['prob_bin']):>15} | "
                f"{row['count']:6.0f} | "
                f"{row['wins']:5.0f} | "
                f"{row['hit_rate']:7.1%} | "
                f"{row['avg_edge']:7.1%} | "
                f"${row['total_profit']:7.2f}"
            )

        # Calculate calibration error
        calibration_error = np.abs(calibration_by_bin['hit_rate'] - calibration_by_bin['avg_predicted_prob']).mean()
        logger.info(f"\n📊 Mean Calibration Error: {calibration_error:.1%}")

        # Brier score (lower is better, 0 = perfect)
        brier_score = np.mean((self.bets['model_prob'] - self.bets['hit']) ** 2)
        logger.info(f"📊 Brier Score: {brier_score:.4f} (0 = perfect, 0.25 = random)")

        return {
            'calibration_by_bin': calibration_by_bin,
            'calibration_error': calibration_error,
            'brier_score': brier_score
        }

    def analyze_by_market(self) -> Dict:
        """Analyze calibration by market type."""
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION BY MARKET")
        logger.info("="*80 + "\n")

        market_results = {}

        for market in self.bets['market'].unique():
            market_bets = self.bets[self.bets['market'] == market].copy()

            if len(market_bets) < 20:
                continue  # Skip markets with too few bets

            # Create bins
            bins = [0.0, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
            market_bets['prob_bin'] = pd.cut(market_bets['model_prob'], bins=bins, include_lowest=True)

            # Calculate hit rate by bin
            calibration = market_bets.groupby('prob_bin', observed=True).agg({
                'hit': ['count', 'mean'],
                'model_prob': 'mean'
            }).reset_index()

            calibration.columns = ['prob_bin', 'count', 'hit_rate', 'avg_predicted_prob']

            # Calculate metrics
            calibration_error = np.abs(calibration['hit_rate'] - calibration['avg_predicted_prob']).mean()
            brier_score = np.mean((market_bets['model_prob'] - market_bets['hit']) ** 2)

            logger.info(f"\n{market.upper()}:")
            logger.info(f"  Total Bets: {len(market_bets)}")
            logger.info(f"  Overall Hit Rate: {market_bets['hit'].mean():.1%}")
            logger.info(f"  Avg Predicted Prob: {market_bets['model_prob'].mean():.1%}")
            logger.info(f"  Calibration Error: {calibration_error:.1%}")
            logger.info(f"  Brier Score: {brier_score:.4f}")
            logger.info(f"  ROI: {(market_bets['profit'].sum() / len(market_bets)) * 100:.1f}%")

            market_results[market] = {
                'calibration': calibration,
                'calibration_error': calibration_error,
                'brier_score': brier_score,
                'count': len(market_bets),
                'hit_rate': market_bets['hit'].mean(),
                'avg_predicted_prob': market_bets['model_prob'].mean()
            }

        return market_results

    def analyze_high_confidence_bets(self):
        """Analyze performance of high-confidence bets (>70% predicted probability)."""
        logger.info("\n" + "="*80)
        logger.info("HIGH-CONFIDENCE BETS ANALYSIS (Predicted Prob > 70%)")
        logger.info("="*80 + "\n")

        high_conf_bets = self.bets[self.bets['model_prob'] > 0.70].copy()

        if len(high_conf_bets) == 0:
            logger.warning("⚠️  No high-confidence bets found")
            return

        logger.info(f"Total High-Confidence Bets: {len(high_conf_bets)}")
        logger.info(f"Hit Rate: {high_conf_bets['hit'].mean():.1%}")
        logger.info(f"Expected Hit Rate: {high_conf_bets['model_prob'].mean():.1%}")
        logger.info(f"ROI: {(high_conf_bets['profit'].sum() / len(high_conf_bets)) * 100:.1f}%")
        logger.info(f"Total Profit: ${high_conf_bets['profit'].sum():.2f}")

        # Break down by probability range
        logger.info("\nBy Probability Range:")
        for lower, upper in [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.0)]:
            range_bets = high_conf_bets[
                (high_conf_bets['model_prob'] >= lower) &
                (high_conf_bets['model_prob'] < upper)
            ]

            if len(range_bets) > 0:
                logger.info(
                    f"  {lower:.0%}-{upper:.0%}: "
                    f"{len(range_bets):3d} bets, "
                    f"{range_bets['hit'].mean():.1%} hit rate, "
                    f"{(range_bets['profit'].sum() / len(range_bets)) * 100:+6.1f}% ROI"
                )

    def identify_systematic_biases(self):
        """Identify systematic biases in predictions."""
        logger.info("\n" + "="*80)
        logger.info("SYSTEMATIC BIAS ANALYSIS")
        logger.info("="*80 + "\n")

        # Over/Under bias
        over_bets = self.bets[self.bets['side'] == 'OVER']
        under_bets = self.bets[self.bets['side'] == 'UNDER']

        logger.info("OVER vs UNDER Performance:")
        logger.info(f"  OVER: {len(over_bets)} bets, {over_bets['hit'].mean():.1%} hit rate, {(over_bets['profit'].sum() / len(over_bets)) * 100:+.1f}% ROI")
        logger.info(f"  UNDER: {len(under_bets)} bets, {under_bets['hit'].mean():.1%} hit rate, {(under_bets['profit'].sum() / len(under_bets)) * 100:+.1f}% ROI")

        # Position bias (only if position column exists)
        if 'position' in self.bets.columns:
            logger.info("\nPerformance by Position:")
            for position in ['QB', 'RB', 'WR', 'TE']:
                pos_bets = self.bets[self.bets['position'] == position]
                if len(pos_bets) > 0:
                    logger.info(
                        f"  {position}: {len(pos_bets):3d} bets, "
                        f"{pos_bets['hit'].mean():.1%} hit rate, "
                        f"{(pos_bets['profit'].sum() / len(pos_bets)) * 100:+6.1f}% ROI"
                    )

        # Edge range bias
        logger.info("\nPerformance by Edge Range:")
        for lower, upper in [(0.03, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]:
            edge_bets = self.bets[
                (self.bets['edge'] >= lower) &
                (self.bets['edge'] < upper)
            ]

            if len(edge_bets) > 0:
                logger.info(
                    f"  {lower:.0%}-{upper:.0%}: "
                    f"{len(edge_bets):3d} bets, "
                    f"{edge_bets['hit'].mean():.1%} hit rate, "
                    f"{(edge_bets['profit'].sum() / len(edge_bets)) * 100:+6.1f}% ROI"
                )

    def create_calibration_curve(self, output_path: Path = None):
        """Create calibration curve plot."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING CALIBRATION CURVE")
        logger.info("="*80 + "\n")

        # Overall calibration curve
        prob_true, prob_pred = calibration_curve(
            self.bets['hit'],
            self.bets['model_prob'],
            n_bins=10
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)

        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Actual Win Rate', fontsize=12)
        ax.set_title('Calibration Curve (Overall)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if output_path is None:
            output_path = project_root / 'reports/calibration_curve.png'

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"✓ Saved calibration curve: {output_path}")
        plt.close()

    def generate_calibration_recommendations(self):
        """Generate recommendations for calibration improvements."""
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION RECOMMENDATIONS")
        logger.info("="*80 + "\n")

        # Calculate overall metrics
        overall_hit_rate = self.bets['hit'].mean()
        overall_predicted_prob = self.bets['model_prob'].mean()

        logger.info("🎯 Key Findings:\n")

        # Finding 1: Overall calibration
        if abs(overall_hit_rate - overall_predicted_prob) > 0.05:
            logger.info(
                f"1. MAJOR CALIBRATION ISSUE: Model predicts {overall_predicted_prob:.1%} average "
                f"win rate but actual is {overall_hit_rate:.1%}"
            )
            logger.info("   → Implement isotonic calibration on historical data\n")
        else:
            logger.info(
                f"1. Overall calibration is reasonable: {overall_predicted_prob:.1%} predicted vs "
                f"{overall_hit_rate:.1%} actual"
            )
            logger.info("   → Focus on specific market calibration\n")

        # Finding 2: High-confidence bets
        high_conf = self.bets[self.bets['model_prob'] > 0.70]
        if len(high_conf) > 0:
            high_conf_hit_rate = high_conf['hit'].mean()
            high_conf_expected = high_conf['model_prob'].mean()

            if high_conf_hit_rate < 0.70:
                logger.info(
                    f"2. HIGH-CONFIDENCE BETS UNDERPERFORM: {high_conf_hit_rate:.1%} actual vs "
                    f"{high_conf_expected:.1%} predicted"
                )
                logger.info("   → Model is overconfident, apply shrinkage to high probabilities\n")
            else:
                logger.info(f"2. High-confidence bets performing well: {high_conf_hit_rate:.1%} hit rate")
                logger.info("   → Maintain or increase stakes on high-confidence picks\n")

        # Finding 3: Negative ROI markets
        logger.info("3. MARKET-SPECIFIC ISSUES:\n")
        for market in self.bets['market'].unique():
            market_bets = self.bets[self.bets['market'] == market]
            market_roi = (market_bets['profit'].sum() / len(market_bets)) * 100

            if market_roi < -10:
                logger.info(
                    f"   ⚠️  {market}: {market_roi:+.1f}% ROI - "
                    f"CONSIDER EXCLUDING OR RECALIBRATING"
                )
            elif market_roi > 5:
                logger.info(f"   ✅ {market}: {market_roi:+.1f}% ROI - PERFORMING WELL")

        logger.info("\n📋 Recommended Actions:\n")
        logger.info("1. Implement isotonic calibration using Weeks 5-10 as training data")
        logger.info("2. Validate calibration on Week 11 (hold-out set)")
        logger.info("3. Consider excluding passing_yards market (if ROI < -20%)")
        logger.info("4. Apply separate calibration per market type")
        logger.info("5. Re-run backtest with calibrated probabilities")

    def run_full_analysis(self):
        """Run complete calibration analysis."""
        self.load_bet_log()
        self.analyze_overall_calibration()
        self.analyze_by_market()
        self.analyze_high_confidence_bets()
        self.identify_systematic_biases()
        self.create_calibration_curve()
        self.generate_calibration_recommendations()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze betting calibration and identify correction opportunities"
    )
    parser.add_argument(
        '--bet-log',
        type=str,
        required=True,
        help="Path to bet log CSV file"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = CalibrationAnalyzer(Path(args.bet_log))

    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
