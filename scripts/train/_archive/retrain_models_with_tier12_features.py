#!/usr/bin/env python3
"""
Retrain ML Models with TIER 1 & TIER 2 Features

This script retrains UsagePredictor and EfficiencyPredictor models with enhanced features:
- TIER 1: EWMA weighting, regime detection, game script
- TIER 2: NGS metrics, situational EPA

Uses temporal cross-validation to prevent overfitting and validate improvements.

Usage:
    # Retrain usage predictor (targets/carries) for WR
    python scripts/train/retrain_models_with_tier12_features.py --position WR --model usage

    # Retrain efficiency predictor (yards/TD) for RB
    python scripts/train/retrain_models_with_tier12_features.py --position RB --model efficiency

    # Retrain all positions for both models
    python scripts/train/retrain_models_with_tier12_features.py --all

    # Use temporal CV to validate
    python scripts/train/retrain_models_with_tier12_features.py --position WR --model usage --validate

Expected MAE improvement: -17% to -30%
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple
import pickle
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.features.tier1_2_integration import (
    extract_all_tier1_2_features,
    get_feature_columns_for_position,
    validate_features
)
from nfl_quant.validation import (
    TemporalCrossValidator,
    evaluate_model_temporal
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nfl_quant.utils.season_utils import get_current_season, get_training_seasons

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE COLUMNS FOR EACH POSITION/MODEL
# ============================================================================

def get_usage_feature_cols(position: str) -> List[str]:
    """
    Get feature columns for UsagePredictor (targets/carries).

    These features predict VOLUME (how many opportunities a player gets).
    """
    # Base features (all positions)
    features = [
        'week',

        # Baseline trailing stats
        'trailing_snap_share',
        'trailing_target_share',  # WR/TE/RB
        'trailing_carry_share',   # RB/QB

        # TIER 1: Regime detection
        'weeks_since_regime_change',
        'is_in_regime',
        'regime_confidence',

        # TIER 1: Game script
        'usage_when_leading',
        'usage_when_trailing',
        'usage_when_close',
        'game_script_sensitivity',

        # TIER 2: Situational EPA (opponent defense strength)
        'redzone_epa',
        'third_down_epa',
        'two_minute_epa',
    ]

    # Position-specific additions
    if position == 'RB':
        features.append('goalline_epa')  # RBs get more goalline touches

    return features


def get_efficiency_feature_cols(position: str) -> List[str]:
    """
    Get feature columns for EfficiencyPredictor (yards/catch, TDs).

    These features predict EFFICIENCY (what player does with opportunities).
    """
    # Base features
    features = [
        'week',
        'trailing_snap_share',
    ]

    # Position-specific NGS and efficiency features
    if position in ['WR', 'TE']:
        features.extend([
            'trailing_target_share',

            # TIER 2: NGS receiving metrics
            'avg_separation',
            'avg_cushion',
            'avg_yac_above_expectation',
            'catch_percentage',
            'avg_intended_air_yards',
            'percent_share_of_intended_air_yards',

            # TIER 2: Situational EPA
            'redzone_epa',
            'two_minute_epa',
        ])

    elif position == 'RB':
        features.extend([
            'trailing_carry_share',

            # TIER 2: NGS rushing metrics
            'rush_yards_over_expected_per_att',
            'efficiency',
            'percent_attempts_gte_eight_defenders',
            'avg_time_to_los',

            # TIER 2: Situational EPA
            'redzone_epa',
            'goalline_epa',
        ])

    elif position == 'QB':
        features.extend([
            # TIER 2: NGS passing metrics
            'avg_time_to_throw',
            'completion_pct_above_exp',
            'aggressiveness',
            'avg_air_yards_to_sticks',

            # TIER 2: Situational EPA
            'redzone_epa',
            'third_down_epa',
            'two_minute_epa',
        ])

    return features


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_historical_data(
    position: str,
    season: int = None,
    min_week: int = 1,
    max_week: int = None,
    seasons: List[int] = None
) -> pd.DataFrame:
    """
    Load historical player data with TIER 1 & 2 features.

    Args:
        position: Player position (QB, RB, WR, TE)
        season: Season year (used if seasons not specified, auto-detected if None)
        min_week: First week to include
        max_week: Last week to include (None = all available)
        seasons: List of seasons to load (default: auto-detected via get_training_seasons)

    Returns:
        DataFrame with features and targets
    """
    logger.info(f"\nLoading historical data for {position}...")

    # Use multiple seasons by default for better training data
    if seasons is None:
        if season is None:
            season = get_current_season()
        seasons = get_training_seasons(season, include_current=True)

    logger.info(f"  Loading data from seasons: {seasons}")

    # Load play-by-play data from ALL seasons
    pbp_dfs = []
    for s in seasons:
        pbp_path = PROJECT_ROOT / f'data/nflverse/pbp_{s}.parquet'
        if pbp_path.exists():
            df = pd.read_parquet(pbp_path)
            pbp_dfs.append(df)
            logger.info(f"  ✓ Loaded {len(df):,} plays from pbp_{s}.parquet")
        else:
            logger.warning(f"  ⚠️  PBP data not found for {s}: {pbp_path}")

    if not pbp_dfs:
        raise FileNotFoundError(
            f"No PBP data found for any season in {seasons}\n"
            f"Run: python scripts/fetch/pull_2024_season_data.py"
        )

    pbp_df = pd.concat(pbp_dfs, ignore_index=True)
    logger.info(f"  ✓ Combined {len(pbp_df):,} total plays from {len(pbp_dfs)} seasons")

    # Load weekly player stats from ALL seasons
    # First try season-specific files, then fall back to weekly_stats.parquet
    weekly_dfs = []
    for s in seasons:
        weekly_path = PROJECT_ROOT / f'data/nflverse/weekly_{s}.parquet'
        if weekly_path.exists():
            df = pd.read_parquet(weekly_path)
            if 'season' not in df.columns:
                df['season'] = s  # Add season column if missing
            weekly_dfs.append(df)
            logger.info(f"  ✓ Loaded {len(df):,} player-weeks from weekly_{s}.parquet")
        else:
            logger.debug(f"  weekly_{s}.parquet not found, will try weekly_stats.parquet")

    # If no season-specific files found, try weekly_stats.parquet (contains all seasons)
    if not weekly_dfs:
        weekly_path = PROJECT_ROOT / 'data/nflverse/weekly_stats.parquet'
        if weekly_path.exists():
            df = pd.read_parquet(weekly_path)
            weekly_dfs.append(df)
            logger.info(f"  ✓ Loaded {len(df):,} player-weeks from weekly_stats.parquet")
        else:
            raise FileNotFoundError(
                f"No weekly stats found for any season\n"
                f"Run: python scripts/fetch/pull_2024_season_data.py"
            )

    weekly_df = pd.concat(weekly_dfs, ignore_index=True)

    # Filter to requested seasons and position
    weekly_df = weekly_df[
        (weekly_df['season'].isin(seasons)) &
        (weekly_df['position'] == position) &
        (weekly_df['week'] >= min_week)
    ].copy()

    if max_week is not None:
        weekly_df = weekly_df[weekly_df['week'] <= max_week]

    logger.info(f"  ✓ Filtered to {len(weekly_df):,} player-weeks for {position} across {weekly_df['season'].nunique()} seasons")

    # Extract TIER 1 & 2 features for each player-week
    all_rows = []

    for _, row in weekly_df.iterrows():
        player_name = row['player_display_name']
        team = row['team'] if 'team' in row else row.get('recent_team', 'UNK')
        week = row['week']
        player_season = row['season']  # Use the actual season from the row

        # Skip if not enough history (need at least 4 weeks)
        if week < 5:
            continue

        # Get opponent (would need schedule data - using placeholder)
        opponent = "UNK"  # TODO: Load from schedule

        try:
            # Extract all TIER 1 & 2 features
            features = extract_all_tier1_2_features(
                player_name=player_name,
                position=position,
                team=team,
                opponent=opponent,
                current_week=week,
                season=player_season,  # Use correct season for this player-week
                pbp_df=pbp_df,
                use_ewma=True,
                use_regime=True,
                use_game_script=True,
                use_ngs=True,
                use_situational_epa=True
            )

            # Validate features
            features = validate_features(features, position)

            # Add targets (actual values from this week)
            targets = {
                'player_name': player_name,
                'team': team,
                'week': week,
                'season': player_season,  # Store correct season
            }

            # Usage targets
            if position in ['WR', 'TE', 'RB']:
                targets['actual_targets'] = row.get('targets', 0)
                targets['actual_receptions'] = row.get('receptions', 0)

            if position in ['RB', 'QB']:
                targets['actual_carries'] = row.get('carries', 0)

            # Efficiency targets
            targets['actual_receiving_yards'] = row.get('receiving_yards', 0)
            targets['actual_rushing_yards'] = row.get('rushing_yards', 0)
            targets['actual_receiving_tds'] = row.get('receiving_tds', 0)
            targets['actual_rushing_tds'] = row.get('rushing_tds', 0)

            # Combine
            row_data = {**features, **targets}
            all_rows.append(row_data)

        except Exception as e:
            logger.debug(f"  ✗ Skipping {player_name} week {week}: {e}")
            continue

    df = pd.DataFrame(all_rows)
    logger.info(f"  ✓ Extracted features for {len(df):,} player-weeks")
    logger.info(f"     Weeks: {df['week'].min()}-{df['week'].max()}")
    logger.info(f"     Unique players: {df['player_name'].nunique()}")

    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_usage_model(
    position: str,
    df: pd.DataFrame,
    target_col: str,
    validate: bool = False
) -> XGBRegressor:
    """
    Train UsagePredictor with TIER 1 & 2 features.

    Args:
        position: Player position
        df: Historical data with features
        target_col: Target column name ('actual_targets' or 'actual_carries')
        validate: Use temporal cross-validation

    Returns:
        Trained XGBRegressor model
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training UsagePredictor for {position} - {target_col}")
    logger.info(f"{'='*80}")

    # Get feature columns
    feature_cols = get_usage_feature_cols(position)

    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)

    if missing_cols:
        logger.warning(f"  ⚠️  Missing features: {missing_cols}")

    logger.info(f"  Using {len(available_cols)} features:")
    for col in available_cols:
        logger.info(f"    - {col}")

    # Prepare data
    X = df[available_cols].copy()
    y = df[target_col].copy()

    # Remove rows with missing targets
    mask = ~y.isna() & (y >= 0)
    X = X[mask]
    y = y[mask]

    logger.info(f"\n  Training samples: {len(X):,}")
    logger.info(f"  Target mean: {y.mean():.2f}")
    logger.info(f"  Target std: {y.std():.2f}")

    # XGBoost hyperparameters (tuned for usage prediction)
    params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    if validate:
        # Temporal cross-validation
        logger.info(f"\n  Running temporal cross-validation...")

        results = evaluate_model_temporal(
            model=XGBRegressor(**params),
            df=df[mask].copy(),
            X_cols=available_cols,
            y_col=target_col,
            week_col='week',
            strategy='expanding',
            min_train_weeks=4
        )

        logger.info(f"\n  Temporal CV Results:")
        logger.info(f"    Mean MAE:  {results['mae'].mean():.3f} ± {results['mae'].std():.3f}")
        logger.info(f"    Mean RMSE: {results['rmse'].mean():.3f} ± {results['rmse'].std():.3f}")
        logger.info(f"    Mean R²:   {results['r2'].mean():.3f} ± {results['r2'].std():.3f}")

        # Show per-week performance
        logger.info(f"\n  Per-Week Performance:")
        for _, row in results.iterrows():
            logger.info(
                f"    Week {int(row['test_week']):2d}: "
                f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, R²={row['r2']:.3f}"
            )

    # Train final model on all data
    logger.info(f"\n  Training final model on all {len(X):,} samples...")

    model = XGBRegressor(**params)
    model.fit(X, y)

    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': available_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    logger.info(f"\n  Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"    {i+1}. {row['feature']:40s} {row['importance']:.4f}")

    # Final evaluation
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    logger.info(f"\n  Final Model Performance:")
    logger.info(f"    MAE:  {mae:.3f}")
    logger.info(f"    RMSE: {rmse:.3f}")

    logger.info(f"\n  ✅ Model training complete")

    return model


def train_efficiency_model(
    position: str,
    df: pd.DataFrame,
    target_col: str,
    validate: bool = False
) -> XGBRegressor:
    """
    Train EfficiencyPredictor with TIER 1 & 2 features.

    Args:
        position: Player position
        df: Historical data with features
        target_col: Target column name (yards or TD rate)
        validate: Use temporal cross-validation

    Returns:
        Trained XGBRegressor model
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training EfficiencyPredictor for {position} - {target_col}")
    logger.info(f"{'='*80}")

    # Get feature columns
    feature_cols = get_efficiency_feature_cols(position)

    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)

    if missing_cols:
        logger.warning(f"  ⚠️  Missing features: {missing_cols}")

    logger.info(f"  Using {len(available_cols)} features:")
    for col in available_cols:
        logger.info(f"    - {col}")

    # Prepare data
    X = df[available_cols].copy()
    y = df[target_col].copy()

    # Remove rows with missing targets
    mask = ~y.isna() & (y >= 0)
    X = X[mask]
    y = y[mask]

    logger.info(f"\n  Training samples: {len(X):,}")
    logger.info(f"  Target mean: {y.mean():.2f}")
    logger.info(f"  Target std: {y.std():.2f}")

    # XGBoost hyperparameters (tuned for efficiency prediction)
    params = {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'gamma': 0.05,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    if validate:
        # Temporal cross-validation
        logger.info(f"\n  Running temporal cross-validation...")

        results = evaluate_model_temporal(
            model=XGBRegressor(**params),
            df=df[mask].copy(),
            X_cols=available_cols,
            y_col=target_col,
            week_col='week',
            strategy='expanding',
            min_train_weeks=4
        )

        logger.info(f"\n  Temporal CV Results:")
        logger.info(f"    Mean MAE:  {results['mae'].mean():.3f} ± {results['mae'].std():.3f}")
        logger.info(f"    Mean RMSE: {results['rmse'].mean():.3f} ± {results['rmse'].std():.3f}")
        logger.info(f"    Mean R²:   {results['r2'].mean():.3f} ± {results['r2'].std():.3f}")

    # Train final model
    logger.info(f"\n  Training final model on all {len(X):,} samples...")

    model = XGBRegressor(**params)
    model.fit(X, y)

    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': available_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    logger.info(f"\n  Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"    {i+1}. {row['feature']:40s} {row['importance']:.4f}")

    # Final evaluation
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    logger.info(f"\n  Final Model Performance:")
    logger.info(f"    MAE:  {mae:.3f}")
    logger.info(f"    RMSE: {rmse:.3f}")

    logger.info(f"\n  ✅ Model training complete")

    return model


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model: XGBRegressor, position: str, model_type: str, target: str):
    """Save trained model to disk."""
    output_dir = PROJECT_ROOT / 'models' / 'tier12'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_type}_{position}_{target}_tier12_{timestamp}.pkl"
    output_path = output_dir / filename

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"\n  💾 Model saved: {output_path}")

    # Also save as "latest"
    latest_path = output_dir / f"{model_type}_{position}_{target}_tier12_latest.pkl"
    with open(latest_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"  💾 Latest saved: {latest_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Retrain models with TIER 1 & 2 features')
    parser.add_argument('--position', type=str, choices=['QB', 'RB', 'WR', 'TE'],
                       help='Position to train')
    parser.add_argument('--model', type=str, choices=['usage', 'efficiency', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--validate', action='store_true',
                       help='Run temporal cross-validation')
    parser.add_argument('--all', action='store_true',
                       help='Train all positions')
    parser.add_argument('--season', type=int, default=None,
                       help='Season year (default: auto-detect current season)')
    parser.add_argument('--min-week', type=int, default=1,
                       help='First week to include')
    parser.add_argument('--max-week', type=int, default=None,
                       help='Last week to include')

    args = parser.parse_args()

    positions = ['QB', 'RB', 'WR', 'TE'] if args.all else [args.position]

    if not args.all and not args.position:
        parser.error("Either --position or --all must be specified")

    for position in positions:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING POSITION: {position}")
        logger.info(f"{'='*80}")

        # Load historical data
        df = load_historical_data(
            position=position,
            season=args.season,
            min_week=args.min_week,
            max_week=args.max_week
        )

        # Train usage model
        if args.model in ['usage', 'both']:
            if position in ['WR', 'TE', 'RB']:
                model = train_usage_model(
                    position=position,
                    df=df,
                    target_col='actual_targets',
                    validate=args.validate
                )
                save_model(model, position, 'usage', 'targets')

            if position in ['RB', 'QB']:
                model = train_usage_model(
                    position=position,
                    df=df,
                    target_col='actual_carries',
                    validate=args.validate
                )
                save_model(model, position, 'usage', 'carries')

        # Train efficiency model
        if args.model in ['efficiency', 'both']:
            if position in ['WR', 'TE', 'RB']:
                model = train_efficiency_model(
                    position=position,
                    df=df,
                    target_col='actual_receiving_yards',
                    validate=args.validate
                )
                save_model(model, position, 'efficiency', 'rec_yards')

            if position in ['RB', 'QB']:
                model = train_efficiency_model(
                    position=position,
                    df=df,
                    target_col='actual_rushing_yards',
                    validate=args.validate
                )
                save_model(model, position, 'efficiency', 'rush_yards')

    logger.info(f"\n{'='*80}")
    logger.info("✅ ALL MODELS RETRAINED WITH TIER 1 & 2 FEATURES")
    logger.info(f"{'='*80}")
    logger.info("\n📋 Next steps:")
    logger.info("1. Compare new vs old model performance")
    logger.info("2. Backtest on historical data")
    logger.info("3. Update production models")
    logger.info("4. Retrain calibrators with new model outputs")


if __name__ == '__main__':
    main()
