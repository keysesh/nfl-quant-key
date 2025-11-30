#!/usr/bin/env python3
"""
Train Usage Predictor V4 - With Defensive Features.

Key improvements over V3:
1. Includes opponent defensive EPA as feature
2. Includes opponent defensive rank
3. Includes team pace metrics
4. Better generalization to unseen matchups

Features:
- trailing_snaps
- trailing_attempts
- trailing_carries
- week
- opponent_def_epa (NEW)
- opponent_def_rank (NEW)
- team_pace (NEW)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Import centralized config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.config_enhanced import config
from nfl_quant.utils.season_utils import get_training_seasons

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model hyperparameters from config
_model_params = config.models.usage_predictor


def prepare_qb_usage_data_with_defense(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare QB usage data with defensive features."""
    logger.info("Preparing QB usage data with defense...")

    # Step 1: Get passing stats
    passing_stats = (
        pbp_df[pbp_df['play_type'] == 'pass']
        .groupby(['passer_player_id', 'passer_player_name', 'posteam', 'defteam', 'week'])
        .size()
        .reset_index(name='pass_attempts')
    )

    # Step 2: Get rushing stats
    rushing_stats = (
        pbp_df[pbp_df['play_type'] == 'run']
        .groupby(['rusher_player_id', 'rusher_player_name', 'posteam', 'defteam', 'week'])
        .size()
        .reset_index(name='rush_attempts')
    )

    # Step 3: Merge on player + week + opponent
    qb_games = passing_stats.merge(
        rushing_stats,
        left_on=['passer_player_id', 'passer_player_name', 'posteam', 'defteam', 'week'],
        right_on=['rusher_player_id', 'rusher_player_name', 'posteam', 'defteam', 'week'],
        how='left'
    )

    # Rename columns
    qb_games['player_id'] = qb_games['passer_player_id']
    qb_games['player_name'] = qb_games['passer_player_name']
    qb_games['team'] = qb_games['posteam']
    qb_games['opponent'] = qb_games['defteam']

    # Fill NaN rush attempts with 0
    qb_games['rush_attempts'] = qb_games['rush_attempts'].fillna(0)

    # Calculate snaps, attempts, carries
    qb_games['snaps'] = qb_games['pass_attempts'] + qb_games['rush_attempts']
    qb_games['attempts'] = qb_games['pass_attempts']
    qb_games['carries'] = qb_games['rush_attempts']
    qb_games['position'] = 'QB'

    logger.info(f"  Found {len(qb_games)} QB-game samples")
    logger.info(f"  Mean pass attempts: {qb_games['pass_attempts'].mean():.1f}")
    logger.info(f"  Mean rush attempts: {qb_games['rush_attempts'].mean():.1f}")

    return qb_games[['player_id', 'team', 'opponent', 'week', 'position', 'snaps', 'attempts', 'carries']]


def prepare_rb_usage_data_with_defense(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare RB usage data with defensive features."""
    logger.info("Preparing RB usage data with defense...")

    # Rushing stats
    rushing_stats = (
        pbp_df[pbp_df['play_type'] == 'run']
        .groupby(['rusher_player_id', 'rusher_player_name', 'posteam', 'defteam', 'week'])
        .size()
        .reset_index(name='rush_attempts')
    )

    # Receiving stats (targets)
    receiving_stats = (
        pbp_df[pbp_df['receiver_player_id'].notna()]
        .groupby(['receiver_player_id', 'receiver_player_name', 'posteam', 'defteam', 'week'])
        .size()
        .reset_index(name='targets')
    )

    # Merge
    rb_games = rushing_stats.merge(
        receiving_stats,
        left_on=['rusher_player_id', 'rusher_player_name', 'posteam', 'defteam', 'week'],
        right_on=['receiver_player_id', 'receiver_player_name', 'posteam', 'defteam', 'week'],
        how='left'
    )

    rb_games['player_id'] = rb_games['rusher_player_id']
    rb_games['player_name'] = rb_games['rusher_player_name']
    rb_games['team'] = rb_games['posteam']
    rb_games['opponent'] = rb_games['defteam']
    rb_games['targets'] = rb_games['targets'].fillna(0)

    # Filter to significant RBs (10+ carries per game)
    rb_games = rb_games[rb_games['rush_attempts'] >= 5].copy()

    rb_games['position'] = 'RB'
    rb_games['snaps'] = rb_games['rush_attempts'] + rb_games['targets']
    rb_games['attempts'] = rb_games['targets']
    rb_games['carries'] = rb_games['rush_attempts']

    logger.info(f"  Found {len(rb_games)} RB-game samples")
    logger.info(f"  Mean carries: {rb_games['carries'].mean():.1f}")
    logger.info(f"  Mean targets: {rb_games['targets'].mean():.1f}")

    return rb_games[['player_id', 'team', 'opponent', 'week', 'position', 'snaps', 'attempts', 'carries']]


def prepare_wr_usage_data_with_defense(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare WR usage data with defensive features."""
    logger.info("Preparing WR/TE usage data with defense...")

    receiving_stats = (
        pbp_df[pbp_df['receiver_player_id'].notna()]
        .groupby(['receiver_player_id', 'receiver_player_name', 'posteam', 'defteam', 'week'])
        .size()
        .reset_index(name='targets')
    )

    wr_games = receiving_stats[receiving_stats['targets'] >= 2].copy()

    wr_games['player_id'] = wr_games['receiver_player_id']
    wr_games['player_name'] = wr_games['receiver_player_name']
    wr_games['team'] = wr_games['posteam']
    wr_games['opponent'] = wr_games['defteam']

    wr_games['position'] = 'WR'
    wr_games['snaps'] = wr_games['targets']
    wr_games['attempts'] = wr_games['targets']
    wr_games['carries'] = 0

    logger.info(f"  Found {len(wr_games)} WR/TE-game samples")
    logger.info(f"  Mean targets: {wr_games['targets'].mean():.1f}")

    return wr_games[['player_id', 'team', 'opponent', 'week', 'position', 'snaps', 'attempts', 'carries']]


def add_defensive_features(df: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent defensive features."""
    logger.info("Adding defensive features...")

    # Calculate defensive EPA by team and week
    # Pass defense
    pass_def = (
        pbp_df[pbp_df['play_type'] == 'pass']
        .groupby(['defteam', 'week'])
        ['epa']
        .mean()
        .reset_index()
        .rename(columns={'defteam': 'opponent', 'epa': 'opp_pass_def_epa'})
    )

    # Rush defense
    rush_def = (
        pbp_df[pbp_df['play_type'] == 'run']
        .groupby(['defteam', 'week'])
        ['epa']
        .mean()
        .reset_index()
        .rename(columns={'defteam': 'opponent', 'epa': 'opp_rush_def_epa'})
    )

    # Merge defensive stats
    df = df.merge(pass_def, on=['opponent', 'week'], how='left')
    df = df.merge(rush_def, on=['opponent', 'week'], how='left')

    # Fill missing with league average (0.0)
    df['opp_pass_def_epa'] = df['opp_pass_def_epa'].fillna(0.0)
    df['opp_rush_def_epa'] = df['opp_rush_def_epa'].fillna(0.0)

    # Calculate defensive rank (within each week)
    df['opp_pass_def_rank'] = df.groupby('week')['opp_pass_def_epa'].rank(ascending=True)
    df['opp_rush_def_rank'] = df.groupby('week')['opp_rush_def_epa'].rank(ascending=True)

    logger.info(f"  Added defensive features for {len(df)} rows")

    return df


def add_team_pace(df: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Add team pace features."""
    logger.info("Adding team pace features...")

    # Calculate plays per game by team and week
    team_pace = (
        pbp_df[pbp_df['play_type'].isin(['pass', 'run'])]
        .groupby(['posteam', 'game_id', 'week'])
        .size()
        .reset_index(name='plays')
    )

    team_pace_avg = (
        team_pace
        .groupby(['posteam', 'week'])
        ['plays']
        .mean()
        .reset_index()
        .rename(columns={'posteam': 'team', 'plays': 'team_pace'})
    )

    # Merge
    df = df.merge(team_pace_avg, on=['team', 'week'], how='left')
    df['team_pace'] = df['team_pace'].fillna(65.0)  # League average

    logger.info(f"  Added pace features for {len(df)} rows")

    return df


def add_trailing_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing 4-week averages."""
    logger.info("Calculating trailing averages...")

    df = df.sort_values(['player_id', 'week'])

    # Trailing usage
    df['trailing_snaps'] = df.groupby('player_id')['snaps'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )
    df['trailing_attempts'] = df.groupby('player_id')['attempts'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )
    df['trailing_carries'] = df.groupby('player_id')['carries'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )

    # Trailing defensive EPA (rolling average of opponents faced)
    df['trailing_opp_pass_def_epa'] = df.groupby('player_id')['opp_pass_def_epa'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )
    df['trailing_opp_rush_def_epa'] = df.groupby('player_id')['opp_rush_def_epa'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )

    # Drop rows without trailing data
    df = df[df['trailing_snaps'].notna()].copy()

    return df


def train_position_models_v4(train_df: pd.DataFrame, position: str):
    """Train XGBoost models for a position with defensive features."""
    logger.info(f"Training V4 models for {position}...")

    pos_df = train_df[train_df['position'] == position].copy()

    if len(pos_df) < 50:
        logger.warning(f"  Only {len(pos_df)} samples for {position} - skipping")
        return None

    # Features based on position
    if position == 'QB':
        feature_cols = [
            'trailing_snaps',
            'trailing_attempts',
            'trailing_carries',
            'week',
            'opp_pass_def_epa',          # NEW
            'opp_pass_def_rank',          # NEW
            'trailing_opp_pass_def_epa',  # NEW
            'team_pace',                   # NEW
        ]
    elif position == 'RB':
        feature_cols = [
            'trailing_snaps',
            'trailing_attempts',
            'trailing_carries',
            'week',
            'opp_rush_def_epa',          # NEW
            'opp_rush_def_rank',          # NEW
            'trailing_opp_rush_def_epa',  # NEW
            'team_pace',                   # NEW
        ]
    else:  # WR
        feature_cols = [
            'trailing_snaps',
            'trailing_attempts',
            'trailing_carries',
            'week',
            'opp_pass_def_epa',          # NEW
            'opp_pass_def_rank',          # NEW
            'trailing_opp_pass_def_epa',  # NEW
            'team_pace',                   # NEW
        ]

    X = pos_df[feature_cols]

    models = {}

    for target in ['snaps', 'attempts', 'carries']:
        y = pos_df[target]

        # FIX: Add sample weights to prevent over-regression of high performers
        # High performers (top 20%) get 2x weight, top 10% get 3x weight
        # This prevents the model from regressing star players to the mean
        percentile_80 = y.quantile(0.80)
        percentile_90 = y.quantile(0.90)
        sample_weights = np.ones(len(y))
        sample_weights[y >= percentile_80] = 2.0  # Top 20% get 2x weight
        sample_weights[y >= percentile_90] = 3.0  # Top 10% get 3x weight

        logger.info(f"  {target}: Weighted training - top 20% (>={percentile_80:.1f}) 2x, top 10% (>={percentile_90:.1f}) 3x")

        # Train
        # Load hyperparameters from config
        model = XGBRegressor(
            n_estimators=_model_params['n_estimators'],
            max_depth=_model_params['max_depth'],
            learning_rate=_model_params['learning_rate'],
            objective='reg:squarederror',
            random_state=_model_params['random_state']
        )
        model.fit(X, y, sample_weight=sample_weights)
        models[target] = model

        # Evaluate
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        logger.info(f"  {target}: MAE={mae:.2f}, R2={r2:.3f}")

    logger.info(f"  ✓ Trained 3 models for {position} ({len(pos_df)} samples)")

    return models


def main():
    """Main training pipeline."""
    print("="*80)
    print("TRAINING USAGE PREDICTOR V4 WITH DEFENSIVE FEATURES")
    print("="*80)
    print()

    # Use NFLverse parquet files (R-generated canonical source)
    nflverse_dir = Path('data/nflverse')
    processed_dir = Path('data/processed')
    seasons = sorted(get_training_seasons())

    logger.info("Loading NFLverse PBP data...")
    pbp_frames = []
    for season in seasons:
        candidates = [
            nflverse_dir / f'pbp_{season}.parquet',
            processed_dir / f'pbp_{season}.parquet',
        ]
        season_frame = None
        for path in candidates:
            if path.exists():
                season_frame = pd.read_parquet(path)
                logger.info(f"  Loaded {len(season_frame):,} plays from {season} ({path.name})")
                break
        if season_frame is None:
            logger.warning(f"  ⚠️  Missing play-by-play data for {season}")
            continue
        pbp_frames.append(season_frame)

    if not pbp_frames:
        print("❌ No NFLverse play-by-play files found. Run data ingestion first.")
        return

    if len(pbp_frames) == 1:
        pbp_df = pbp_frames[0]
    else:
        pbp_df = pd.concat(pbp_frames, ignore_index=True)
        logger.info(f"  Combined: {len(pbp_df):,} total plays across seasons {seasons}")

    print()

    # Prepare data for each position
    qb_data = prepare_qb_usage_data_with_defense(pbp_df)
    rb_data = prepare_rb_usage_data_with_defense(pbp_df)
    wr_data = prepare_wr_usage_data_with_defense(pbp_df)

    # Combine
    all_data = pd.concat([qb_data, rb_data, wr_data], ignore_index=True)
    logger.info(f"Total samples: {len(all_data)}")
    print()

    # Add defensive features
    all_data = add_defensive_features(all_data, pbp_df)

    # Add team pace
    all_data = add_team_pace(all_data, pbp_df)

    # Add trailing features
    train_df = add_trailing_features_v4(all_data)

    logger.info(f"Training samples after trailing features: {len(train_df)}")
    print()

    # Train models for each position
    all_models = {}
    for position in ['QB', 'RB', 'WR']:
        models = train_position_models_v4(train_df, position)
        if models:
            all_models[position] = models
        print()

    # Save
    output_path = Path('data/models/usage_predictor_v4_defense.joblib')
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Get feature columns for each position
    feature_cols_by_position = {
        'QB': ['trailing_snaps', 'trailing_attempts', 'trailing_carries', 'week',
               'opp_pass_def_epa', 'opp_pass_def_rank', 'trailing_opp_pass_def_epa', 'team_pace'],
        'RB': ['trailing_snaps', 'trailing_attempts', 'trailing_carries', 'week',
               'opp_rush_def_epa', 'opp_rush_def_rank', 'trailing_opp_rush_def_epa', 'team_pace'],
        'WR': ['trailing_snaps', 'trailing_attempts', 'trailing_carries', 'week',
               'opp_pass_def_epa', 'opp_pass_def_rank', 'trailing_opp_pass_def_epa', 'team_pace'],
    }

    model_bundle = {
        'models': all_models,
        'feature_cols': feature_cols_by_position,
        'trained_date': datetime.now().isoformat(),
        'training_samples': len(train_df),
        'version': 'v4_with_defense',
    }

    joblib.dump(model_bundle, output_path)

    print("="*80)
    print(f"✅ SAVED TO: {output_path}")
    print("="*80)
    print()
    print("Summary:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Positions: {list(all_models.keys())}")
    print(f"  Features: {len(feature_cols_by_position['QB'])} (includes defensive + pace)")
    print()
    print("New features added:")
    print("  ✓ opponent_def_epa")
    print("  ✓ opponent_def_rank")
    print("  ✓ trailing_opponent_def_epa")
    print("  ✓ team_pace")
    print()


if __name__ == '__main__':
    main()
