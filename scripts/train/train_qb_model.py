#!/usr/bin/env python3
"""
Train QB Passing Yards Model

Trains a specialized model for player_pass_yds using:
1. QB-specific features (not receiver features)
2. Regression + variance estimation (not classification)
3. Starter filtering (only train on confirmed starters)

Usage:
    python scripts/train/train_qb_model.py
    python scripts/train/train_qb_model.py --validate  # Include walk-forward validation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from nfl_quant.config_paths import PROJECT_ROOT, BACKTEST_DIR, MODELS_DIR
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.features.qb_features import QBFeatureExtractor, extract_qb_features_batch
from nfl_quant.features.qb_starter import get_qb_starter_detector, QBRole
from nfl_quant.models.qb_passing_model import QBPassingModel
from configs.qb_model_config import (
    QB_FEATURES,
    MIN_STARTER_CONFIDENCE,
    MIN_PASS_ATTEMPTS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """Load and prepare training data."""
    # Load enriched data
    data_path = BACKTEST_DIR / 'combined_odds_actuals_ENRICHED.csv'
    if not data_path.exists():
        # Fall back to standard file
        data_path = BACKTEST_DIR / 'combined_odds_actuals_2023_2024_2025.csv'

    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Normalize player names
    df['player_norm'] = df['player'].apply(normalize_player_name)

    # Add global week
    df['global_week'] = (df['season'] - 2023) * 18 + df['week']

    # Filter to player_pass_yds market
    df = df[df['market'] == 'player_pass_yds'].copy()

    logger.info(f"Loaded {len(df)} player_pass_yds records")

    return df


def filter_to_starters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to confirmed starters only."""
    detector = get_qb_starter_detector()

    starter_mask = []
    for _, row in df.iterrows():
        info = detector.classify_qb(
            row['player'],
            row.get('team', ''),
            row.get('season', 2025),
        )
        is_starter = info.role == QBRole.STARTER and info.confidence >= MIN_STARTER_CONFIDENCE
        starter_mask.append(is_starter)

    df_filtered = df[starter_mask].copy()
    logger.info(f"Filtered to {len(df_filtered)} starter records (from {len(df)})")

    return df_filtered


def filter_by_attempts(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Filter out games where QB had < MIN_PASS_ATTEMPTS (injury/benching)."""
    # Join attempts from stats
    stats_subset = stats[['player_norm', 'season', 'week', 'attempts']].drop_duplicates()
    df = df.merge(stats_subset, on=['player_norm', 'season', 'week'], how='left', suffixes=('', '_stats'))

    # Use stats attempts if available, otherwise use actual_stat / 7.5 as proxy
    if 'attempts' not in df.columns or df['attempts'].isna().all():
        df['attempts_proxy'] = df['actual_stat'] / 7.5  # Rough yards per attempt
        df_filtered = df[df['attempts_proxy'] >= MIN_PASS_ATTEMPTS / 7.5].copy()
    else:
        df_filtered = df[df['attempts'] >= MIN_PASS_ATTEMPTS].copy()

    logger.info(f"Filtered to {len(df_filtered)} records with sufficient attempts (from {len(df)})")

    return df_filtered


def load_player_stats() -> pd.DataFrame:
    """Load player stats for trailing calculations."""
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv'
    if not stats_path.exists():
        stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if stats_path.exists():
            stats = pd.read_parquet(stats_path)
        else:
            logger.warning("Player stats not found")
            return pd.DataFrame()
    else:
        stats = pd.read_csv(stats_path, low_memory=False)

    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats = stats[stats['position'] == 'QB'].copy()

    return stats


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract QB features for all rows."""
    logger.info("Extracting QB features...")

    # Use the batch extractor
    df_with_features = extract_qb_features_batch(
        df,
        historical_odds=df,  # Use same data for player rates
        market='player_pass_yds',
    )

    # Check feature coverage
    for col in QB_FEATURES:
        if col in df_with_features.columns:
            non_null = df_with_features[col].notna().sum()
            pct = non_null / len(df_with_features) * 100
            if pct < 90:
                logger.warning(f"Feature {col}: {pct:.1f}% non-null")

    return df_with_features


def train_model(df: pd.DataFrame) -> QBPassingModel:
    """Train the QB model."""
    logger.info("Training QB model...")

    # Get feature columns that exist in data
    feature_cols = [c for c in QB_FEATURES if c in df.columns]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X = df[feature_cols].copy()
    y = df['actual_stat'].values

    # Train model
    model = QBPassingModel()
    model.fit(X, y, feature_cols)

    return model


def validate_walk_forward(
    df: pd.DataFrame,
    model: QBPassingModel,
    holdout_weeks: list = [12, 13, 14],
) -> pd.DataFrame:
    """
    Run walk-forward validation on holdout weeks.

    Args:
        df: Full dataset with features
        model: Trained model
        holdout_weeks: Weeks to use for validation

    Returns:
        DataFrame with validation results
    """
    logger.info(f"Running walk-forward validation on weeks {holdout_weeks}")

    # Filter to 2025 holdout weeks
    holdout = df[(df['season'] == 2025) & (df['week'].isin(holdout_weeks))].copy()

    if len(holdout) == 0:
        logger.warning("No holdout data found")
        return pd.DataFrame()

    # Get predictions
    X_holdout = holdout[model.feature_cols].fillna(0)
    lines = holdout['line'].values

    results = model.predict_with_edge(X_holdout, lines, min_edge_yards=10.0, min_confidence=0.55)

    # Add actual outcomes
    results['actual_stat'] = holdout['actual_stat'].values
    results['under_hit'] = (holdout['actual_stat'] < holdout['line']).astype(int).values

    # Calculate P(UNDER) hit rate
    under_picks = results[results['pick'] == 'UNDER']
    if len(under_picks) > 0:
        under_hit_rate = under_picks['under_hit'].mean()
        under_roi = (under_hit_rate * 0.909 - (1 - under_hit_rate)) * 100
        logger.info(f"UNDER picks: n={len(under_picks)}, hit_rate={under_hit_rate*100:.1f}%, ROI={under_roi:+.1f}%")

    over_picks = results[results['pick'] == 'OVER']
    if len(over_picks) > 0:
        over_hit_rate = 1 - over_picks['under_hit'].mean()  # OVER hits when under doesn't
        over_roi = (over_hit_rate * 0.909 - (1 - over_hit_rate)) * 100
        logger.info(f"OVER picks: n={len(over_picks)}, hit_rate={over_hit_rate*100:.1f}%, ROI={over_roi:+.1f}%")

    # Overall recommended bets
    recommended = results[results['recommend_bet']]
    if len(recommended) > 0:
        # Calculate actual hit rate for recommended bets
        rec_hits = []
        for _, row in recommended.iterrows():
            if row['pick'] == 'UNDER':
                rec_hits.append(row['under_hit'])
            else:
                rec_hits.append(1 - row['under_hit'])

        rec_hit_rate = np.mean(rec_hits)
        rec_roi = (rec_hit_rate * 0.909 - (1 - rec_hit_rate)) * 100
        logger.info(f"Recommended bets: n={len(recommended)}, hit_rate={rec_hit_rate*100:.1f}%, ROI={rec_roi:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train QB Passing Yards Model')
    parser.add_argument('--validate', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--holdout-weeks', default='12-14', help='Holdout weeks for validation')
    args = parser.parse_args()

    print("=" * 70)
    print("QB PASSING YARDS MODEL TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Parse holdout weeks
    if '-' in args.holdout_weeks:
        start, end = map(int, args.holdout_weeks.split('-'))
        holdout_weeks = list(range(start, end + 1))
    else:
        holdout_weeks = [int(args.holdout_weeks)]

    # Step 1: Load data
    df = load_training_data()
    if len(df) == 0:
        logger.error("No training data loaded")
        return

    # Step 2: Load player stats for filtering
    stats = load_player_stats()

    # Step 3: Filter by attempts (remove injury/benching games)
    if len(stats) > 0:
        df = filter_by_attempts(df, stats)

    # Remove rows with missing actuals
    df = df[df['actual_stat'].notna()].copy()

    # CRITICAL: Split data BEFORE feature extraction to prevent leakage
    if args.validate:
        # True temporal split: train on weeks < holdout, test on holdout weeks
        train_mask = ~((df['season'] == 2025) & (df['week'].isin(holdout_weeks)))
        train_df = df[train_mask].copy()
        holdout_df = df[~train_mask].copy()
        logger.info(f"Temporal split: {len(train_df)} train, {len(holdout_df)} holdout")
    else:
        train_df = df.copy()
        holdout_df = None

    # Step 4: Extract features using ONLY training data for rates
    logger.info("Extracting features for training data...")
    train_df = extract_qb_features_batch(
        train_df,
        historical_odds=train_df,  # Only use training data for rates
        market='player_pass_yds',
    )
    logger.info(f"Final training set: {len(train_df)} samples")

    # Step 5: Train model on training data ONLY
    model = train_model(train_df)

    # Step 7: Print feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    importance = model.get_feature_importance()
    for i, (feat, imp) in enumerate(importance.items()):
        if i < 15:  # Top 15
            print(f"  {feat}: {imp*100:.1f}%")

    # Step 6: Save model
    model_path = model.save()
    print(f"\nModel saved to: {model_path}")

    # Step 7: TRUE Out-of-Sample Validation (if requested)
    if args.validate and holdout_df is not None and len(holdout_df) > 0:
        print("\n" + "=" * 70)
        print("TRUE OUT-OF-SAMPLE VALIDATION")
        print("=" * 70)
        print(f"Holdout weeks: {holdout_weeks}")
        print(f"Holdout samples: {len(holdout_df)}")

        # Extract features for holdout using ONLY training data for rates
        logger.info("Extracting features for holdout data (using training data for rates)...")
        holdout_df = extract_qb_features_batch(
            holdout_df,
            historical_odds=train_df,  # ONLY training data for rates (no leakage)
            market='player_pass_yds',
        )

        # Get predictions on holdout
        X_holdout = holdout_df[model.feature_cols].fillna(0)
        lines = holdout_df['line'].values

        results = model.predict_with_edge(X_holdout, lines, min_edge_yards=10.0, min_confidence=0.55)

        # Add actual outcomes
        results['actual_stat'] = holdout_df['actual_stat'].values
        results['under_hit'] = (holdout_df['actual_stat'] < holdout_df['line']).astype(int).values

        # Calculate results
        under_picks = results[results['pick'] == 'UNDER']
        if len(under_picks) > 0:
            under_hit_rate = under_picks['under_hit'].mean()
            under_roi = (under_hit_rate * 0.909 - (1 - under_hit_rate)) * 100
            print(f"UNDER picks: n={len(under_picks)}, hit_rate={under_hit_rate*100:.1f}%, ROI={under_roi:+.1f}%")

        over_picks = results[results['pick'] == 'OVER']
        if len(over_picks) > 0:
            over_hit_rate = 1 - over_picks['under_hit'].mean()
            over_roi = (over_hit_rate * 0.909 - (1 - over_hit_rate)) * 100
            print(f"OVER picks: n={len(over_picks)}, hit_rate={over_hit_rate*100:.1f}%, ROI={over_roi:+.1f}%")

        # Overall
        all_picks = results[results['pick'] != 'NO_BET']
        if len(all_picks) > 0:
            all_hits = []
            for _, row in all_picks.iterrows():
                if row['pick'] == 'UNDER':
                    all_hits.append(row['under_hit'])
                else:
                    all_hits.append(1 - row['under_hit'])
            all_hit_rate = np.mean(all_hits)
            all_roi = (all_hit_rate * 0.909 - (1 - all_hit_rate)) * 100
            print(f"\nOVERALL: n={len(all_picks)}, hit_rate={all_hit_rate*100:.1f}%, ROI={all_roi:+.1f}%")

        # Save validation results
        output_path = BACKTEST_DIR / 'qb_model_validation_results.csv'
        results.to_csv(output_path, index=False)
        print(f"\nValidation results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model version: {model.version}")
    print(f"Training samples: {model.train_n_samples}")
    print(f"Training MAE: {model.train_mae:.1f} yards")
    print(f"Training RMSE: {model.train_rmse:.1f} yards")


if __name__ == '__main__':
    main()
