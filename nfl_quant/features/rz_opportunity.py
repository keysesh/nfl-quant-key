"""
Red Zone Opportunity Features for ATTD Prediction

Computes trailing RZ carries/targets from PBP data.
These are the most predictive features for Anytime TD:
- RBs with trailing_rz_carries >= 3.0: 53.9% TD rate (vs 24% base)
- WRs with trailing_rz_targets >= 2.0: 48.8% TD rate (vs 18% base)

Usage:
    rz_opps = compute_rz_opportunity_features(season=2024)
    df = merge_rz_opportunity(df, rz_opps)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def compute_rz_opportunity_features(
    season: int,
    ewma_span: int = 4,
) -> pd.DataFrame:
    """
    Compute trailing RZ opportunity features from PBP.

    These features track a player's historical red zone usage,
    which strongly predicts future TD scoring.

    Args:
        season: Season year
        ewma_span: EWMA span for trailing average

    Returns:
        DataFrame with:
        - player_id, week
        - trailing_rz_carries: Trailing EWMA of RZ carries
        - trailing_rz_targets: Trailing EWMA of RZ targets
        - trailing_rz_touches: Combined RZ opportunities
    """
    logger.info(f"Computing RZ opportunity features for {season}...")

    # Load PBP data
    pbp_paths = [
        Path(f'data/nflverse/pbp_{season}.parquet'),
        Path('data/nflverse/pbp.parquet'),
    ]

    pbp = None
    for path in pbp_paths:
        if path.exists():
            pbp = pd.read_parquet(path)
            if 'season' in pbp.columns:
                pbp = pbp[pbp['season'] == season]
            logger.info(f"  Loaded PBP from {path}: {len(pbp):,} plays")
            break

    if pbp is None or len(pbp) == 0:
        logger.warning(f"No PBP data for {season}")
        return pd.DataFrame()

    # Filter to red zone plays
    rz = pbp[
        (pbp['yardline_100'] <= 20) &
        (pbp['play_type'].isin(['pass', 'run']))
    ].copy()

    logger.info(f"  Red zone plays: {len(rz):,}")

    if len(rz) == 0:
        return pd.DataFrame()

    # Compute RZ carries per player-week
    rz_rush = rz[rz['play_type'] == 'run'].copy()
    if 'rusher_player_id' in rz_rush.columns:
        rz_carries = (
            rz_rush.groupby(['rusher_player_id', 'week'])
            .size()
            .reset_index(name='rz_carries')
        )
        rz_carries.columns = ['player_id', 'week', 'rz_carries']
    else:
        rz_carries = pd.DataFrame(columns=['player_id', 'week', 'rz_carries'])

    # Compute RZ targets per player-week
    rz_pass = rz[rz['play_type'] == 'pass'].copy()
    if 'receiver_player_id' in rz_pass.columns:
        rz_pass = rz_pass.dropna(subset=['receiver_player_id'])
        rz_targets = (
            rz_pass.groupby(['receiver_player_id', 'week'])
            .size()
            .reset_index(name='rz_targets')
        )
        rz_targets.columns = ['player_id', 'week', 'rz_targets']
    else:
        rz_targets = pd.DataFrame(columns=['player_id', 'week', 'rz_targets'])

    # Combine
    rz_data = pd.merge(rz_carries, rz_targets, on=['player_id', 'week'], how='outer')
    rz_data = rz_data.fillna(0)
    rz_data = rz_data.sort_values(['player_id', 'week'])

    # Compute trailing features with shift(1) to prevent leakage
    rz_data['trailing_rz_carries'] = (
        rz_data.groupby('player_id')['rz_carries']
        .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
    ).fillna(0)

    rz_data['trailing_rz_targets'] = (
        rz_data.groupby('player_id')['rz_targets']
        .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
    ).fillna(0)

    # Combined RZ touches
    rz_data['trailing_rz_touches'] = (
        rz_data['trailing_rz_carries'] + rz_data['trailing_rz_targets']
    )

    # Add season
    rz_data['season'] = season

    logger.info(f"  Computed RZ features for {rz_data['player_id'].nunique()} players")

    return rz_data[['player_id', 'season', 'week',
                    'trailing_rz_carries', 'trailing_rz_targets', 'trailing_rz_touches',
                    'rz_carries', 'rz_targets']]


def merge_rz_opportunity(
    df: pd.DataFrame,
    rz_opps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge RZ opportunity features into main DataFrame.

    Args:
        df: Main DataFrame with player_id/season/week
        rz_opps: RZ opportunity features

    Returns:
        DataFrame with RZ features added
    """
    feature_cols = ['trailing_rz_carries', 'trailing_rz_targets', 'trailing_rz_touches']

    if rz_opps.empty:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df

    merge_cols = ['player_id', 'season', 'week']

    # Select needed columns
    rz_subset = rz_opps[merge_cols + feature_cols].drop_duplicates()

    # Check if columns already exist (from previous season merge)
    # If so, drop them first to avoid suffix issues
    existing_cols = [c for c in feature_cols if c in df.columns]
    if existing_cols:
        # Keep existing values, only fill in missing
        temp_df = df.merge(rz_subset, on=merge_cols, how='left', suffixes=('', '_new'))
        for col in feature_cols:
            new_col = f'{col}_new'
            if new_col in temp_df.columns:
                # Prefer new value if available, otherwise keep existing
                temp_df[col] = temp_df[new_col].combine_first(temp_df[col])
                temp_df = temp_df.drop(columns=[new_col])
        df = temp_df
    else:
        df = df.merge(rz_subset, on=merge_cols, how='left')

    # Fill missing with 0
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0.0

    return df


def load_rz_opportunity_all_seasons(
    seasons: List[int] = None,
) -> pd.DataFrame:
    """
    Load RZ opportunity features for multiple seasons.

    Args:
        seasons: List of seasons (default: [2023, 2024, 2025])

    Returns:
        Combined DataFrame with all seasons
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    all_data = []
    for season in seasons:
        try:
            rz_data = compute_rz_opportunity_features(season)
            if not rz_data.empty:
                all_data.append(rz_data)
        except Exception as e:
            logger.warning(f"Failed to compute RZ features for {season}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()
