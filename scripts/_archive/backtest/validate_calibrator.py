#!/usr/bin/env python3
"""
Validate TD Calibrator on held-out weeks 7-8
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss, log_loss
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.backtest.backtest_td_predictions_integrated import (
    generate_predictions_integrated,
    enhance_td_predictions_backtest,
    normalize_tds_backtest
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_calibrator(weeks: list, calibrator_path: str):
    """
    Validate calibrator on held-out weeks.

    Args:
        weeks: List of weeks to validate on (should be different from training)
        calibrator_path: Path to trained calibrator
    """
    # Load calibrator
    calibrator = joblib.load(calibrator_path)
    logger.info(f"✅ Loaded calibrator from {calibrator_path}")

    # Load NFLverse data
    nflverse_path = Path('data/nflverse_cache/stats_player_week_2025.csv')
    nflverse_df = pd.read_csv(nflverse_path, low_memory=False)
    logger.info(f"✅ Loaded NFLverse data: {len(nflverse_df)} rows")

    all_predictions = []
    all_actuals = []

    for week in weeks:
        logger.info(f"\nValidating week {week}...")

        # Generate predictions
        df = generate_predictions_integrated(week, nflverse_df)
        if df.empty:
            logger.warning(f"No predictions for week {week}")
            continue

        # Enhance with TD predictions
        df = enhance_td_predictions_backtest(df, week)

        # Normalize
        df = normalize_tds_backtest(df, week)

        # Load actuals from NFLverse
        week_actuals = nflverse_df[nflverse_df['week'] == week].copy()
        if week_actuals.empty:
            logger.warning(f"No actuals for week {week}")
            continue

        # Calculate actual TD scored
        week_actuals['actual_td'] = (
            (week_actuals.get('rushing_tds', 0) > 0) |
            (week_actuals.get('receiving_tds', 0) > 0) |
            (week_actuals.get('passing_tds', 0) > 0)
        ).astype(int)

        # Merge
        merged = df.merge(
            week_actuals[['player_display_name', 'actual_td']],
            left_on='player_name',
            right_on='player_display_name',
            how='inner'
        )

        # Calculate raw predicted TD probability (clipped)
        merged['raw_td_prob'] = np.clip(
            merged.get('rushing_tds_mean', 0) +
            merged.get('receiving_tds_mean', 0) +
            merged.get('passing_tds_mean', 0),
            0.0, 0.95
        )

        # Apply calibrator
        merged['calibrated_td_prob'] = calibrator.predict(merged['raw_td_prob'].values)

        all_predictions.append(merged)
        all_actuals.append(merged['actual_td'].values)

        logger.info(f"  Week {week}: {len(merged)} predictions")
        logger.info(f"    Actual TD rate: {merged['actual_td'].mean():.1%}")
        logger.info(f"    Raw predicted rate: {merged['raw_td_prob'].mean():.1%}")
        logger.info(f"    Calibrated predicted rate: {merged['calibrated_td_prob'].mean():.1%}")

    # Combine all weeks
    combined = pd.concat(all_predictions, ignore_index=True)

    logger.info("\n" + "="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)
    logger.info(f"\nTotal predictions: {len(combined)}")
    logger.info(f"Actual TD rate: {combined['actual_td'].mean():.1%}")
    logger.info(f"Raw predicted rate: {combined['raw_td_prob'].mean():.1%}")
    logger.info(f"Calibrated predicted rate: {combined['calibrated_td_prob'].mean():.1%}")

    # Calculate metrics
    y_true = combined['actual_td'].values
    y_raw = combined['raw_td_prob'].values
    y_cal = combined['calibrated_td_prob'].values

    brier_raw = brier_score_loss(y_true, y_raw)
    brier_cal = brier_score_loss(y_true, y_cal)

    try:
        logloss_raw = log_loss(y_true, y_raw)
        logloss_cal = log_loss(y_true, y_cal)
    except:
        logloss_raw = None
        logloss_cal = None

    logger.info(f"\nBrier Score:")
    logger.info(f"  Raw predictions:        {brier_raw:.4f}")
    logger.info(f"  Calibrated predictions: {brier_cal:.4f}")
    logger.info(f"  Improvement:            {brier_raw - brier_cal:.4f} ({(brier_raw-brier_cal)/brier_raw*100:.1f}%)")

    if logloss_raw and logloss_cal:
        logger.info(f"\nLog Loss:")
        logger.info(f"  Raw predictions:        {logloss_raw:.4f}")
        logger.info(f"  Calibrated predictions: {logloss_cal:.4f}")
        logger.info(f"  Improvement:            {logloss_raw - logloss_cal:.4f}")

    logger.info("\n" + "="*80)

    return combined


if __name__ == '__main__':
    calibrator_path = 'data/models/td_calibrator_v1.joblib'
    validation_weeks = [7, 8]

    logger.info("="*80)
    logger.info("VALIDATING TD CALIBRATOR ON HELD-OUT WEEKS")
    logger.info("="*80)
    logger.info(f"\nCalibrator: {calibrator_path}")
    logger.info(f"Validation weeks: {validation_weeks}")
    logger.info(f"(Training was on weeks 1-6)")

    results = validate_calibrator(validation_weeks, calibrator_path)

    logger.info("\n✅ Validation complete")
