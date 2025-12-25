#!/usr/bin/env python3
"""
Train Efficiency Predictor V2 - With Defensive Features.

Key improvements over V1:
1. Includes opponent defensive EPA as feature
2. Includes opponent defensive rank
3. Includes team pace metrics
4. Better generalization to unseen matchups

Predicts:
- yards_per_target (WR/TE/RB)
- yards_per_carry (RB/QB)
- completion_pct (QB)
- yards_per_completion (QB)
- td_rate_pass (QB/WR/TE/RB)
- td_rate_rush (RB/QB)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Import centralized config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.config_enhanced import config
from nfl_quant.utils.season_utils import get_training_seasons

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model hyperparameters from config
_model_params = config.models.efficiency_predictor


def prepare_qb_efficiency_with_defense(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare QB efficiency data with defensive features."""
    logger.info("Preparing QB efficiency data with defense...")

    pass_plays = pbp_df[pbp_df['play_type'] == 'pass'].copy()

    qb_efficiency = []
    for (player_id, week, defteam), group in pass_plays.groupby(['passer_player_id', 'week', 'defteam']):
        if pd.isna(player_id):
            continue

        attempts = len(group)
        if attempts < 5:
            continue

        completions = group['complete_pass'].sum()
        passing_yards = group['passing_yards'].sum()
        pass_tds = group['pass_touchdown'].sum()

        comp_pct = completions / attempts if attempts > 0 else 0
        yards_per_completion = passing_yards / completions if completions > 0 else 0
        td_rate = pass_tds / attempts if attempts > 0 else 0

        # Get team info
        posteam = group['posteam'].iloc[0]

        qb_efficiency.append({
            'player_id': player_id,
            'week': week,
            'team': posteam,
            'opponent': defteam,
            'position': 'QB',
            'attempts': attempts,
            'comp_pct': comp_pct,
            'yards_per_completion': yards_per_completion,
            'td_rate_pass': td_rate,
        })

    df = pd.DataFrame(qb_efficiency)
    logger.info(f"  Found {len(df)} QB-game efficiency samples")
    logger.info(f"  Mean comp %: {df['comp_pct'].mean():.3f}")
    logger.info(f"  Mean yds/comp: {df['yards_per_completion'].mean():.2f}")

    return df


def prepare_receiver_efficiency_with_defense(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare WR/TE/RB receiving efficiency with defensive features."""
    logger.info("Preparing receiver efficiency data with defense...")

    pass_plays = pbp_df[pbp_df['receiver_player_id'].notna()].copy()

    rec_efficiency = []
    for (player_id, week, defteam), group in pass_plays.groupby(['receiver_player_id', 'week', 'defteam']):
        targets = len(group)
        if targets < 3:
            continue

        receptions = group['complete_pass'].sum()
        rec_yards = group['receiving_yards'].sum()
        rec_tds = group['pass_touchdown'].sum()

        yards_per_target = rec_yards / targets if targets > 0 else 0
        td_rate = rec_tds / targets if targets > 0 else 0

        # Get team info
        posteam = group['posteam'].iloc[0]

        rec_efficiency.append({
            'player_id': player_id,
            'week': week,
            'team': posteam,
            'opponent': defteam,
            'position': 'WR',
            'targets': targets,
            'yards_per_target': yards_per_target,
            'td_rate_pass': td_rate,
        })

    df = pd.DataFrame(rec_efficiency)
    logger.info(f"  Found {len(df)} receiver-game efficiency samples")
    logger.info(f"  Mean yds/target: {df['yards_per_target'].mean():.2f}")

    return df


def prepare_rusher_efficiency_with_defense(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare RB/QB rushing efficiency with defensive features."""
    logger.info("Preparing rusher efficiency data with defense...")

    rush_plays = pbp_df[pbp_df['play_type'] == 'run'].copy()

    rush_efficiency = []
    for (player_id, week, defteam), group in rush_plays.groupby(['rusher_player_id', 'week', 'defteam']):
        if pd.isna(player_id):
            continue

        carries = len(group)
        if carries < 3:
            continue

        rush_yards = group['rushing_yards'].sum()
        rush_tds = group['rush_touchdown'].sum()

        yards_per_carry = rush_yards / carries if carries > 0 else 0
        td_rate = rush_tds / carries if carries > 0 else 0

        # Get team info
        posteam = group['posteam'].iloc[0]

        rush_efficiency.append({
            'player_id': player_id,
            'week': week,
            'team': posteam,
            'opponent': defteam,
            'position': 'RB',
            'carries': carries,
            'yards_per_carry': yards_per_carry,
            'td_rate_rush': td_rate,
        })

    df = pd.DataFrame(rush_efficiency)
    logger.info(f"  Found {len(df)} rusher-game efficiency samples")
    logger.info(f"  Mean yds/carry: {df['yards_per_carry'].mean():.2f}")

    return df


def add_defensive_features(df: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent defensive features."""
    logger.info("Adding defensive features...")

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

    # Merge
    df = df.merge(pass_def, on=['opponent', 'week'], how='left')
    df = df.merge(rush_def, on=['opponent', 'week'], how='left')

    # Fill missing
    df['opp_pass_def_epa'] = df['opp_pass_def_epa'].fillna(0.0)
    df['opp_rush_def_epa'] = df['opp_rush_def_epa'].fillna(0.0)

    # Calculate ranks
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
    df['team_pace'] = df['team_pace'].fillna(65.0)

    logger.info(f"  Added pace features for {len(df)} rows")

    return df


def add_trailing_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing 4-week averages for efficiency."""
    logger.info("Calculating trailing averages...")

    df = df.sort_values(['player_id', 'week'])

    # Trailing efficiency metrics
    for col in ['comp_pct', 'yards_per_completion', 'yards_per_target', 'yards_per_carry', 'td_rate_pass', 'td_rate_rush']:
        if col in df.columns:
            df[f'trailing_{col}'] = df.groupby('player_id')[col].transform(
                lambda x: x.rolling(4, min_periods=1).mean().shift(1)
            )

    # Trailing defensive EPA
    df['trailing_opp_pass_def_epa'] = df.groupby('player_id')['opp_pass_def_epa'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )
    df['trailing_opp_rush_def_epa'] = df.groupby('player_id')['opp_rush_def_epa'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )

    # Fill missing trailing features with appropriate defaults by position
    # RBs who don't catch passes will have NaN for trailing_yards_per_target
    # WRs will have NaN for trailing_yards_per_carry
    # This is OK - we'll handle it when filtering for each position's models

    # Drop rows that don't have enough history (first game for each player)
    # We need at least one of the key trailing metrics
    df = df[
        df['trailing_opp_pass_def_epa'].notna() |
        df['trailing_opp_rush_def_epa'].notna()
    ].copy()

    logger.info(f"  Samples after trailing features: {len(df)}")

    return df


def train_efficiency_models_v2(train_df: pd.DataFrame, metric_name: str, feature_cols: list):
    """Train XGBoost model for an efficiency metric."""
    logger.info(f"Training V2 model for {metric_name}...")

    # Filter to rows with this metric
    valid_df = train_df[train_df[metric_name].notna()].copy()

    if len(valid_df) < 50:
        logger.warning(f"  Only {len(valid_df)} samples for {metric_name} - skipping")
        return None

    X = valid_df[feature_cols]
    y = valid_df[metric_name]

    # FIX: Add sample weights to prevent over-regression of high performers
    # High performers (top 20%) get 2x weight, top 10% get 3x weight
    # This prevents the model from regressing star players to the mean
    percentile_80 = y.quantile(0.80)
    percentile_90 = y.quantile(0.90)
    sample_weights = np.ones(len(y))
    sample_weights[y >= percentile_80] = 2.0  # Top 20% get 2x weight
    sample_weights[y >= percentile_90] = 3.0  # Top 10% get 3x weight

    logger.info(f"  {metric_name}: Weighted training - top 20% (>={percentile_80:.2f}) 2x, top 10% (>={percentile_90:.2f}) 3x")

    # FIX (Dec 7, 2025): Add MONOTONE CONSTRAINTS to force model to respect trailing metrics
    # Without this, the model over-weights opponent features and regresses toward mean
    # Example: trailing Y/T = 7.0 was predicting 10.0 (43% inflation!)
    # Constraint = 1 means: as feature increases, prediction should increase
    # Constraint = 0 means: no constraint
    monotone_constraints = [0] * len(feature_cols)  # Default: no constraints

    # Apply positive monotone constraint on trailing efficiency metrics
    for i, col in enumerate(feature_cols):
        if col in ['trailing_yards_per_target', 'trailing_yards_per_carry',
                   'trailing_yards_per_completion', 'trailing_comp_pct']:
            monotone_constraints[i] = 1  # Positive: more trailing -> more predicted
            logger.info(f"  {metric_name}: Monotone constraint +1 on {col} (index {i})")

    # Convert to tuple format for XGBoost
    monotone_constraints_str = '(' + ','.join(str(x) for x in monotone_constraints) + ')'
    logger.info(f"  {metric_name}: Monotone constraints = {monotone_constraints_str}")

    # Train
    # Load hyperparameters from config
    model = XGBRegressor(
        n_estimators=_model_params['n_estimators'],
        max_depth=_model_params['max_depth'],
        learning_rate=_model_params['learning_rate'],
        objective='reg:squarederror',
        random_state=_model_params['random_state'],
        monotone_constraints=tuple(monotone_constraints)  # NEW: Force respect for trailing metrics
    )
    model.fit(X, y, sample_weight=sample_weights)

    # Evaluate
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    logger.info(f"  {metric_name}: MAE={mae:.3f}, R2={r2:.3f} ({len(valid_df)} samples)")

    return model


def main():
    """Main training pipeline."""
    print("="*80)
    print("TRAINING EFFICIENCY PREDICTOR V2 WITH DEFENSIVE FEATURES")
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

    # Prepare efficiency data
    qb_eff = prepare_qb_efficiency_with_defense(pbp_df)
    rec_eff = prepare_receiver_efficiency_with_defense(pbp_df)
    rush_eff = prepare_rusher_efficiency_with_defense(pbp_df)

    # Combine (merge QB passing with QB rushing, RB rushing with receiving)
    logger.info("Merging multi-stat positions...")

    # Get all unique player IDs
    all_qb_ids = set(qb_eff['player_id'].unique())
    all_rb_ids = set(rush_eff['player_id'].unique()) - all_qb_ids  # RBs = rushers who aren't QBs
    all_wr_ids = set(rec_eff['player_id'].unique()) - all_rb_ids  # WRs = receivers who aren't RBs

    logger.info(f"  Identified {len(all_qb_ids)} QBs, {len(all_rb_ids)} RBs, {len(all_wr_ids)} WRs")

    # For QBs: merge passing and rushing stats
    qb_rush = rush_eff[rush_eff['player_id'].isin(all_qb_ids)].copy()
    qb_combined = qb_eff.merge(
        qb_rush[['player_id', 'week', 'yards_per_carry', 'td_rate_rush']],
        on=['player_id', 'week'],
        how='left'
    )
    qb_combined['position'] = 'QB'
    logger.info(f"  QB combined: {len(qb_combined)} samples")

    # For RBs: merge rushing and receiving stats (THIS IS THE KEY FIX)
    rb_rush = rush_eff[rush_eff['player_id'].isin(all_rb_ids)].copy()
    rb_rec = rec_eff[rec_eff['player_id'].isin(all_rb_ids)].copy()

    rb_combined = rb_rush.merge(
        rb_rec[['player_id', 'week', 'yards_per_target', 'td_rate_pass']],
        on=['player_id', 'week'],
        how='left',
        suffixes=('', '_rec')
    )
    rb_combined['position'] = 'RB'
    logger.info(f"  RB combined: {len(rb_combined)} samples")

    # WRs: just receiving
    wr_eff = rec_eff[rec_eff['player_id'].isin(all_wr_ids)].copy()
    wr_eff['position'] = 'WR'
    logger.info(f"  WR receiving: {len(wr_eff)} samples")

    # Combine all
    all_data = pd.concat([qb_combined, rb_combined, wr_eff], ignore_index=True)
    logger.info(f"Total samples after combining: {len(all_data)}")
    logger.info(f"  QB: {len(all_data[all_data['position'] == 'QB'])}")
    logger.info(f"  RB: {len(all_data[all_data['position'] == 'RB'])}")
    logger.info(f"  WR: {len(all_data[all_data['position'] == 'WR'])}")
    print()

    # Add defensive features
    all_data = add_defensive_features(all_data, pbp_df)

    # Add team pace
    all_data = add_team_pace(all_data, pbp_df)

    # Add trailing features
    train_df = add_trailing_features_v2(all_data)

    logger.info(f"Training samples after trailing features: {len(train_df)}")
    print()

    # Train models for each efficiency metric
    all_models = {}

    # QB-specific features
    qb_feature_cols = [
        'week',
        'trailing_comp_pct',
        'trailing_yards_per_completion',
        'trailing_td_rate_pass',
        'opp_pass_def_epa',
        'opp_pass_def_rank',
        'trailing_opp_pass_def_epa',
        'team_pace',
    ]

    # QB models
    qb_df = train_df[train_df['position'] == 'QB'].copy()
    for metric in ['comp_pct', 'yards_per_completion', 'td_rate_pass']:
        model = train_efficiency_models_v2(qb_df, metric, qb_feature_cols)
        if model:
            all_models[f'QB_{metric}'] = model

    # QB rushing
    qb_rush_features = [
        'week',
        'trailing_yards_per_carry',
        'trailing_td_rate_rush',
        'opp_rush_def_epa',
        'opp_rush_def_rank',
        'trailing_opp_rush_def_epa',
        'team_pace',
    ]
    model = train_efficiency_models_v2(qb_df, 'yards_per_carry', qb_rush_features)
    if model:
        all_models['QB_yards_per_carry'] = model
    model = train_efficiency_models_v2(qb_df, 'td_rate_rush', qb_rush_features)
    if model:
        all_models['QB_td_rate_rush'] = model

    print()

    # RB features
    rb_feature_cols = [
        'week',
        'trailing_yards_per_carry',
        'trailing_yards_per_target',
        'trailing_td_rate_rush',
        'trailing_td_rate_pass',
        'opp_rush_def_epa',
        'opp_rush_def_rank',
        'opp_pass_def_epa',
        'trailing_opp_rush_def_epa',
        'team_pace',
    ]

    # RB models
    rb_df = train_df[train_df['position'] == 'RB'].copy()
    for metric in ['yards_per_carry', 'yards_per_target', 'td_rate_rush', 'td_rate_pass']:
        model = train_efficiency_models_v2(rb_df, metric, rb_feature_cols)
        if model:
            all_models[f'RB_{metric}'] = model

    print()

    # WR features
    wr_feature_cols = [
        'week',
        'trailing_yards_per_target',
        'trailing_td_rate_pass',
        'opp_pass_def_epa',
        'opp_pass_def_rank',
        'trailing_opp_pass_def_epa',
        'team_pace',
    ]

    # WR models
    wr_df = train_df[train_df['position'] == 'WR'].copy()
    for metric in ['yards_per_target', 'td_rate_pass']:
        model = train_efficiency_models_v2(wr_df, metric, wr_feature_cols)
        if model:
            all_models[f'WR_{metric}'] = model

    print()

    # Save
    output_path = Path('data/models/efficiency_predictor_v2_defense.joblib')
    output_path.parent.mkdir(exist_ok=True, parents=True)

    model_bundle = {
        'models': all_models,
        'feature_cols': {
            'QB': qb_feature_cols,
            'QB_rush': qb_rush_features,
            'RB': rb_feature_cols,
            'WR': wr_feature_cols,
        },
        'trained_date': datetime.now().isoformat(),
        'training_samples': len(train_df),
        'version': 'v2_with_defense',
    }

    joblib.dump(model_bundle, output_path)

    print("="*80)
    print(f"✅ SAVED TO: {output_path}")
    print("="*80)
    print()
    print("Summary:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Models trained: {len(all_models)}")
    print(f"  Positions: QB, RB, WR")
    print()
    print("New features added:")
    print("  ✓ opponent_def_epa")
    print("  ✓ opponent_def_rank")
    print("  ✓ trailing_opponent_def_epa")
    print("  ✓ team_pace")
    print()


if __name__ == '__main__':
    main()
