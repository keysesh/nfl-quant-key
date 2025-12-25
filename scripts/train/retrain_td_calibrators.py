#!/usr/bin/env python3
"""
Retrain Efficiency Predictor with TD Calibration.

Key fix: Ensure team-level TD predictions match historical averages.

Historical baselines (2024 NFL):
- Rushing TDs per team per game: 0.96
- Receiving TDs per team per game: 1.48
- Total TDs per team per game: 2.44

The issue: When summing player TDs, we get ~3.7 instead of ~2.4 (54% over-prediction).

Solution: Apply a calibration factor to TD rate predictions that ensures team totals match.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.config_enhanced import config
from nfl_quant.utils.season_utils import get_training_seasons, get_current_season

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Historical TD rates per team per game (from 2024 data analysis)
HISTORICAL_RUSH_TDS_PER_TEAM = 0.96
HISTORICAL_REC_TDS_PER_TEAM = 1.48
HISTORICAL_TOTAL_TDS_PER_TEAM = 2.44


def calculate_td_calibration_factors(predictions_df: pd.DataFrame) -> dict:
    """
    Calculate calibration factors to bring TD predictions in line with historical rates.

    Returns dict with 'rush_td_factor' and 'rec_td_factor'.
    """
    # Group by team and sum TDs
    team_tds = predictions_df.groupby('team').agg({
        'rushing_tds_mean': 'sum',
        'receiving_tds_mean': 'sum'
    }).reset_index()

    avg_rush_tds = team_tds['rushing_tds_mean'].mean()
    avg_rec_tds = team_tds['receiving_tds_mean'].mean()

    # Calculate calibration factors
    rush_factor = HISTORICAL_RUSH_TDS_PER_TEAM / avg_rush_tds if avg_rush_tds > 0 else 1.0
    rec_factor = HISTORICAL_REC_TDS_PER_TEAM / avg_rec_tds if avg_rec_tds > 0 else 1.0

    logger.info(f"TD Calibration Analysis:")
    logger.info(f"  Current avg rushing TDs per team: {avg_rush_tds:.2f}")
    logger.info(f"  Target (historical): {HISTORICAL_RUSH_TDS_PER_TEAM:.2f}")
    logger.info(f"  Rush TD calibration factor: {rush_factor:.3f}")
    logger.info(f"")
    logger.info(f"  Current avg receiving TDs per team: {avg_rec_tds:.2f}")
    logger.info(f"  Target (historical): {HISTORICAL_REC_TDS_PER_TEAM:.2f}")
    logger.info(f"  Rec TD calibration factor: {rec_factor:.3f}")

    return {
        'rush_td_factor': rush_factor,
        'rec_td_factor': rec_factor,
        'historical_rush_tds': HISTORICAL_RUSH_TDS_PER_TEAM,
        'historical_rec_tds': HISTORICAL_REC_TDS_PER_TEAM
    }


def prepare_efficiency_training_data() -> pd.DataFrame:
    """Prepare training data from PBP with proper TD rate calculations."""
    logger.info("Preparing efficiency training data...")

    seasons = get_training_seasons()
    all_data = []

    for season in seasons:
        pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
        if not pbp_path.exists():
            logger.warning(f"PBP data not found for {season}, skipping")
            continue

        pbp = pd.read_parquet(pbp_path)
        logger.info(f"Loaded {len(pbp)} plays from {season}")

        # Process rushers
        rush_plays = pbp[
            (pbp['play_type'] == 'run') &
            (pbp['rusher_player_id'].notna())
        ].copy()

        rush_data = rush_plays.groupby(['rusher_player_id', 'week', 'game_id', 'posteam']).agg({
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
            'rush_attempt': 'sum'
        }).reset_index()

        rush_data.columns = ['player_id', 'week', 'game_id', 'team', 'rushing_yards', 'rushing_tds', 'carries']
        rush_data['season'] = season
        rush_data['td_type'] = 'rush'

        # Calculate TD rate
        rush_data['td_rate'] = rush_data['rushing_tds'] / rush_data['carries'].clip(lower=1)

        # Filter to meaningful samples (at least 5 carries)
        rush_data = rush_data[rush_data['carries'] >= 5]

        all_data.append(rush_data)

        # Process receivers
        pass_plays = pbp[
            (pbp['play_type'] == 'pass') &
            (pbp['receiver_player_id'].notna())
        ].copy()

        rec_data = pass_plays.groupby(['receiver_player_id', 'week', 'game_id', 'posteam']).agg({
            'receiving_yards': 'sum',
            'pass_touchdown': 'sum',
            'complete_pass': 'sum'
        }).reset_index()

        rec_data.columns = ['player_id', 'week', 'game_id', 'team', 'receiving_yards', 'receiving_tds', 'receptions']
        rec_data['season'] = season
        rec_data['td_type'] = 'rec'

        # Calculate targets (complete + incomplete passes to this receiver)
        targets = pass_plays.groupby(['receiver_player_id', 'week', 'game_id'])['play_id'].count().reset_index()
        targets.columns = ['player_id', 'week', 'game_id', 'targets']
        rec_data = rec_data.merge(targets, on=['player_id', 'week', 'game_id'], how='left')

        rec_data['td_rate'] = rec_data['receiving_tds'] / rec_data['targets'].clip(lower=1)

        # Filter to meaningful samples (at least 3 targets)
        rec_data = rec_data[rec_data['targets'] >= 3]

        all_data.append(rec_data)

    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total training samples: {len(df)}")

    return df


def train_td_rate_model(train_df: pd.DataFrame, td_type: str) -> XGBRegressor:
    """Train TD rate model with conservative predictions."""
    logger.info(f"Training TD rate model for {td_type}...")

    df = train_df[train_df['td_type'] == td_type].copy()

    # Features: trailing TD rate, usage, and week
    df = df.sort_values(['player_id', 'season', 'week'])
    df['trailing_td_rate'] = df.groupby('player_id')['td_rate'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )

    # Add trailing usage
    if td_type == 'rush':
        df['trailing_usage'] = df.groupby('player_id')['carries'].transform(
            lambda x: x.rolling(4, min_periods=1).mean().shift(1)
        )
    else:
        df['trailing_usage'] = df.groupby('player_id')['targets'].transform(
            lambda x: x.rolling(4, min_periods=1).mean().shift(1)
        )

    # Drop rows without trailing data
    df = df.dropna(subset=['trailing_td_rate', 'trailing_usage'])

    if len(df) < 100:
        logger.warning(f"Not enough samples for {td_type} TD model")
        return None

    feature_cols = ['trailing_td_rate', 'trailing_usage', 'week']
    X = df[feature_cols]
    y = df['td_rate']

    # Use conservative hyperparameters to prevent overfitting
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,  # Shallow trees for conservative predictions
        learning_rate=0.05,
        min_child_weight=10,  # Higher = more conservative
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(X, y)

    # Evaluate
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R2: {r2:.4f}")
    logger.info(f"  Mean actual TD rate: {y.mean():.4f}")
    logger.info(f"  Mean predicted TD rate: {preds.mean():.4f}")

    return model


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("RETRAINING EFFICIENCY MODELS WITH TD CALIBRATION")
    logger.info("=" * 60)

    # Load current predictions to calculate calibration factors
    current_season = get_current_season()
    current_week = 12  # Adjust as needed

    predictions_path = Path(f'data/model_predictions_week{current_week}.csv')
    if predictions_path.exists():
        current_preds = pd.read_csv(predictions_path)
        calibration = calculate_td_calibration_factors(current_preds)

        # Save calibration factors
        calibration_path = Path('data/models/td_calibration_factors.joblib')
        joblib.dump(calibration, calibration_path)
        logger.info(f"\nSaved calibration factors to {calibration_path}")
    else:
        logger.warning(f"No current predictions found at {predictions_path}")
        calibration = None

    # Prepare training data
    train_df = prepare_efficiency_training_data()

    # Train models
    rush_model = train_td_rate_model(train_df, 'rush')
    rec_model = train_td_rate_model(train_df, 'rec')

    # Save models
    if rush_model:
        rush_path = Path('data/models/td_rate_rush_calibrated.joblib')
        joblib.dump(rush_model, rush_path)
        logger.info(f"Saved rushing TD rate model to {rush_path}")

    if rec_model:
        rec_path = Path('data/models/td_rate_rec_calibrated.joblib')
        joblib.dump(rec_model, rec_path)
        logger.info(f"Saved receiving TD rate model to {rec_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TD CALIBRATION COMPLETE")
    logger.info("=" * 60)

    if calibration:
        logger.info(f"\nTo apply calibration to predictions:")
        logger.info(f"  Rush TDs *= {calibration['rush_td_factor']:.3f}")
        logger.info(f"  Rec TDs *= {calibration['rec_td_factor']:.3f}")

        # Show expected impact
        logger.info(f"\nExpected impact:")
        logger.info(f"  Before: ~3.76 TDs per team")
        logger.info(f"  After: ~{3.76 * (calibration['rush_td_factor'] + calibration['rec_td_factor'])/2:.2f} TDs per team")
        logger.info(f"  Target: 2.44 TDs per team")


if __name__ == '__main__':
    main()
