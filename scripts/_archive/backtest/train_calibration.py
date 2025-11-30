#!/usr/bin/env python3
"""
Train Isotonic Calibration
===========================

Trains market-specific isotonic calibrators using historical bet data.
Uses Weeks 5-10 as training, Week 11 as validation.

Usage:
    python scripts/backtest/train_calibration.py --bet-log reports/backtest_bet_log_20251118_222905.csv
"""

import argparse
import logging
import sys
import pickle
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationTrainer:
    """Trains isotonic calibrators for betting probabilities."""

    def __init__(self, bet_log_path: Path, train_weeks: list, val_week: int):
        self.bet_log_path = bet_log_path
        self.train_weeks = train_weeks
        self.val_week = val_week
        self.bets = None
        self.calibrators = {}

    def load_bet_log(self):
        """Load bet log CSV."""
        logger.info(f"Loading bet log from {self.bet_log_path}...")
        self.bets = pd.read_csv(self.bet_log_path)
        logger.info(f"  ✓ Loaded {len(self.bets):,} bets")

    def train_overall_calibrator(self) -> IsotonicRegression:
        """Train overall calibrator on all markets combined."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING OVERALL CALIBRATOR")
        logger.info("="*80 + "\n")

        # Filter to training weeks
        train_bets = self.bets[self.bets['week'].isin(self.train_weeks)].copy()
        logger.info(f"Training on {len(train_bets):,} bets from weeks {self.train_weeks}")

        # Train isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(train_bets['model_prob'], train_bets['hit'])

        # Evaluate on training set
        train_calibrated = calibrator.predict(train_bets['model_prob'])
        train_brier_before = brier_score_loss(train_bets['hit'], train_bets['model_prob'])
        train_brier_after = brier_score_loss(train_bets['hit'], train_calibrated)

        logger.info(f"📊 Training Set Performance:")
        logger.info(f"   Brier Score Before: {train_brier_before:.4f}")
        logger.info(f"   Brier Score After:  {train_brier_after:.4f}")
        logger.info(f"   Improvement: {((train_brier_before - train_brier_after) / train_brier_before) * 100:.1f}%")

        # Evaluate on validation set
        val_bets = self.bets[self.bets['week'] == self.val_week].copy()
        if len(val_bets) > 0:
            val_calibrated = calibrator.predict(val_bets['model_prob'])
            val_brier_before = brier_score_loss(val_bets['hit'], val_bets['model_prob'])
            val_brier_after = brier_score_loss(val_bets['hit'], val_calibrated)

            logger.info(f"\n📊 Validation Set Performance (Week {self.val_week}):")
            logger.info(f"   Brier Score Before: {val_brier_before:.4f}")
            logger.info(f"   Brier Score After:  {val_brier_after:.4f}")
            logger.info(f"   Improvement: {((val_brier_before - val_brier_after) / val_brier_before) * 100:.1f}%")

        return calibrator

    def train_market_calibrators(self) -> Dict[str, IsotonicRegression]:
        """Train separate calibrator for each market."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING MARKET-SPECIFIC CALIBRATORS")
        logger.info("="*80 + "\n")

        calibrators = {}
        train_bets = self.bets[self.bets['week'].isin(self.train_weeks)].copy()
        val_bets = self.bets[self.bets['week'] == self.val_week].copy()

        for market in train_bets['market'].unique():
            market_train = train_bets[train_bets['market'] == market]
            market_val = val_bets[val_bets['market'] == market]

            if len(market_train) < 20:
                logger.warning(f"⚠️  {market}: Only {len(market_train)} training samples, skipping")
                continue

            logger.info(f"\n{market.upper()}:")
            logger.info(f"  Training samples: {len(market_train)}")
            logger.info(f"  Validation samples: {len(market_val)}")

            # Train calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(market_train['model_prob'], market_train['hit'])

            # Evaluate training
            train_calibrated = calibrator.predict(market_train['model_prob'])
            train_brier_before = brier_score_loss(market_train['hit'], market_train['model_prob'])
            train_brier_after = brier_score_loss(market_train['hit'], train_calibrated)

            logger.info(f"  Training Brier: {train_brier_before:.4f} → {train_brier_after:.4f} ({((train_brier_before - train_brier_after) / train_brier_before) * 100:+.1f}%)")

            # Evaluate validation
            if len(market_val) > 0:
                val_calibrated = calibrator.predict(market_val['model_prob'])
                val_brier_before = brier_score_loss(market_val['hit'], market_val['model_prob'])
                val_brier_after = brier_score_loss(market_val['hit'], val_calibrated)

                logger.info(f"  Validation Brier: {val_brier_before:.4f} → {val_brier_after:.4f} ({((val_brier_before - val_brier_after) / val_brier_before) * 100:+.1f}%)")

                # Calculate ROI before/after
                val_bets_calibrated = market_val.copy()
                val_bets_calibrated['calibrated_prob'] = val_calibrated

                # Recalculate edge with calibrated probabilities
                # For simplicity, just compare ROI with original vs calibrated hit rates
                roi_before = (market_val['profit'].sum() / len(market_val)) * 100
                logger.info(f"  Validation ROI (original): {roi_before:+.1f}%")

            calibrators[market] = calibrator

        return calibrators

    def train_side_adjusted_calibrators(self) -> Dict[str, Dict[str, IsotonicRegression]]:
        """Train separate calibrators for OVER vs UNDER (addresses systematic bias)."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING SIDE-ADJUSTED CALIBRATORS (OVER/UNDER)")
        logger.info("="*80 + "\n")

        calibrators = {}
        train_bets = self.bets[self.bets['week'].isin(self.train_weeks)].copy()
        val_bets = self.bets[self.bets['week'] == self.val_week].copy()

        for side in ['OVER', 'UNDER']:
            side_train = train_bets[train_bets['side'] == side]
            side_val = val_bets[val_bets['side'] == side]

            logger.info(f"\n{side}:")
            logger.info(f"  Training samples: {len(side_train)}")
            logger.info(f"  Validation samples: {len(side_val)}")

            if len(side_train) < 20:
                logger.warning(f"  ⚠️  Too few samples, skipping")
                continue

            # Train calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(side_train['model_prob'], side_train['hit'])

            # Evaluate
            train_calibrated = calibrator.predict(side_train['model_prob'])
            train_brier_before = brier_score_loss(side_train['hit'], side_train['model_prob'])
            train_brier_after = brier_score_loss(side_train['hit'], train_calibrated)

            logger.info(f"  Training Brier: {train_brier_before:.4f} → {train_brier_after:.4f}")

            if len(side_val) > 0:
                val_calibrated = calibrator.predict(side_val['model_prob'])
                val_brier_before = brier_score_loss(side_val['hit'], side_val['model_prob'])
                val_brier_after = brier_score_loss(side_val['hit'], val_calibrated)

                logger.info(f"  Validation Brier: {val_brier_before:.4f} → {val_brier_after:.4f}")

                roi_before = (side_val['profit'].sum() / len(side_val)) * 100
                logger.info(f"  Validation ROI (original): {roi_before:+.1f}%")

            calibrators[side] = calibrator

        return calibrators

    def save_calibrators(self, output_dir: Path):
        """Save calibrators to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*80)
        logger.info("SAVING CALIBRATORS")
        logger.info("="*80 + "\n")

        # Save overall calibrator
        if 'overall' in self.calibrators:
            path = output_dir / 'calibrator_overall.pkl'
            with open(path, 'wb') as f:
                pickle.dump(self.calibrators['overall'], f)
            logger.info(f"✓ Saved overall calibrator: {path}")

        # Save market calibrators
        if 'market' in self.calibrators:
            for market, calibrator in self.calibrators['market'].items():
                safe_name = market.lower().replace('_', '').replace(' ', '')
                path = output_dir / f'calibrator_{safe_name}.pkl'
                with open(path, 'wb') as f:
                    pickle.dump(calibrator, f)
                logger.info(f"✓ Saved {market} calibrator: {path}")

        # Save side calibrators
        if 'side' in self.calibrators:
            for side, calibrator in self.calibrators['side'].items():
                path = output_dir / f'calibrator_side_{side.lower()}.pkl'
                with open(path, 'wb') as f:
                    pickle.dump(calibrator, f)
                logger.info(f"✓ Saved {side} calibrator: {path}")

    def train_all(self):
        """Train all calibrators."""
        self.load_bet_log()

        # Train overall calibrator
        self.calibrators['overall'] = self.train_overall_calibrator()

        # Train market-specific calibrators
        self.calibrators['market'] = self.train_market_calibrators()

        # Train side-adjusted calibrators
        self.calibrators['side'] = self.train_side_adjusted_calibrators()

        # Save all calibrators
        output_dir = project_root / 'data/calibration'
        self.save_calibrators(output_dir)

        logger.info("\n" + "="*80)
        logger.info("✅ CALIBRATION TRAINING COMPLETE")
        logger.info("="*80 + "\n")
        logger.info("Next steps:")
        logger.info("1. Re-run backtest with calibrated probabilities")
        logger.info("2. Compare ROI before vs after calibration")
        logger.info("3. Decide which calibration strategy to use (overall/market/side)")


def main():
    parser = argparse.ArgumentParser(
        description="Train isotonic calibrators for betting probabilities"
    )
    parser.add_argument(
        '--bet-log',
        type=str,
        required=True,
        help="Path to bet log CSV file"
    )
    parser.add_argument(
        '--train-weeks',
        type=int,
        nargs='+',
        default=[5, 6, 7, 8, 9, 10],
        help="Weeks to use for training (default: 5-10)"
    )
    parser.add_argument(
        '--val-week',
        type=int,
        default=11,
        help="Week to use for validation (default: 11)"
    )

    args = parser.parse_args()

    # Create trainer
    trainer = CalibrationTrainer(
        Path(args.bet_log),
        args.train_weeks,
        args.val_week
    )

    # Train all calibrators
    trainer.train_all()


if __name__ == "__main__":
    main()
