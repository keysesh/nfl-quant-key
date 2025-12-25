"""
Player-specific Red Zone TD Conversion Rate

Computes trailing TD rate per RZ snap/opportunity for each player.
This is a critical feature for TD prediction - some players convert
RZ touches at 2x the rate of others.

Key Features:
- rz_td_per_snap: TDs / RZ snaps (trailing EWMA)
- rz_td_per_carry: TDs / RZ carries (for rushers)
- rz_td_per_target: TDs / RZ targets (for receivers)

Usage:
    rz_rates = compute_player_rz_td_rates(pbp, season=2024)
    df = merge_rz_td_rates(df, rz_rates)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Actual RZ TD rates computed from PBP data (2023-2024 seasons)
# These replace the old incorrect defaults (0.12, 0.08)
ACTUAL_RZ_RUSH_TD_RATE = 0.172   # 17.2% of RZ carries result in TD (was 0.12)
ACTUAL_RZ_REC_TD_RATE = 0.214   # 21.4% of RZ targets result in TD (was 0.08)
ACTUAL_RZ_TD_PER_SNAP = 0.19    # Combined RZ TD rate per snap

# Goal-line specific rates (yardline <= 5)
ACTUAL_GL_RUSH_TD_RATE = 0.394  # 39.4% of goal-line carries result in TD
ACTUAL_GL_REC_TD_RATE = 0.357   # 35.7% of goal-line targets result in TD


def compute_player_rz_td_rates(
    pbp: pd.DataFrame,
    season: int,
    weeks: Optional[List[int]] = None,
    ewma_span: int = 4,
) -> pd.DataFrame:
    """
    Compute player's TD conversion rate in red zone.

    Args:
        pbp: Play-by-play DataFrame with yardline_100, touchdown, player IDs
        season: Season year
        weeks: Optional list of weeks to include (default: all)
        ewma_span: EWMA span for trailing average (default: 4 games)

    Returns:
        DataFrame with columns:
        - player_id
        - season
        - week
        - rz_td_per_snap: TDs / RZ snaps (trailing EWMA)
        - rz_td_per_carry: TDs / RZ carries (for rushers)
        - rz_td_per_target: TDs / RZ targets (for receivers)
        - rz_snaps: Total RZ snaps
        - rz_tds: Total RZ TDs
    """
    logger.info(f"Computing player RZ TD rates for {season}...")

    # Filter to season
    df = pbp[pbp['season'] == season].copy()

    if weeks is not None:
        df = df[df['week'].isin(weeks)]

    if len(df) == 0:
        logger.warning(f"No PBP data for season {season}")
        return pd.DataFrame()

    # Filter to red zone plays (yardline_100 <= 20)
    rz = df[
        (df['yardline_100'] <= 20) &
        (df['play_type'].isin(['pass', 'run']))
    ].copy()

    logger.info(f"  Red zone plays: {len(rz):,}")

    # Get rushing TD rates
    rush_rates = _compute_rush_rz_rates(rz, ewma_span)

    # Get receiving TD rates
    rec_rates = _compute_receiving_rz_rates(rz, ewma_span)

    # Combine (outer join to get all players)
    if len(rush_rates) > 0 and len(rec_rates) > 0:
        combined = pd.merge(
            rush_rates,
            rec_rates,
            on=['player_id', 'season', 'week'],
            how='outer',
            suffixes=('', '_rec')
        )
    elif len(rush_rates) > 0:
        combined = rush_rates
    elif len(rec_rates) > 0:
        combined = rec_rates
    else:
        return pd.DataFrame()

    # Fill NaN with 0 for missing rates
    rate_cols = ['rz_td_per_carry', 'rz_td_per_target', 'rz_rush_tds', 'rz_rec_tds',
                 'rz_carries', 'rz_targets']
    for col in rate_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # Compute combined RZ TD rate (per snap)
    rz_carries = combined['rz_carries'] if 'rz_carries' in combined.columns else 0
    rz_targets = combined['rz_targets'] if 'rz_targets' in combined.columns else 0
    rz_rush_tds = combined['rz_rush_tds'] if 'rz_rush_tds' in combined.columns else 0
    rz_rec_tds = combined['rz_rec_tds'] if 'rz_rec_tds' in combined.columns else 0

    combined['rz_snaps'] = rz_carries + rz_targets
    combined['rz_tds'] = rz_rush_tds + rz_rec_tds
    combined['rz_td_per_snap'] = np.where(
        combined['rz_snaps'] > 0,
        combined['rz_tds'] / combined['rz_snaps'],
        0
    )

    logger.info(f"  Computed RZ TD rates for {combined['player_id'].nunique()} players")

    return combined


def _compute_rush_rz_rates(rz: pd.DataFrame, ewma_span: int) -> pd.DataFrame:
    """Compute rushing TD rates in red zone per player."""
    # Filter to rushing plays
    rush = rz[rz['play_type'] == 'run'].copy()

    if len(rush) == 0:
        return pd.DataFrame()

    # Identify rusher (use rusher_player_id or rusher_id)
    if 'rusher_player_id' in rush.columns:
        rush['player_id'] = rush['rusher_player_id']
    elif 'rusher_id' in rush.columns:
        rush['player_id'] = rush['rusher_id']
    else:
        logger.warning("No rusher ID column found in PBP")
        return pd.DataFrame()

    # Drop plays with no player ID
    rush = rush.dropna(subset=['player_id'])

    # Group by player/week
    weekly = rush.groupby(['player_id', 'season', 'week']).agg({
        'rush_touchdown': 'sum',  # TDs this week
    }).reset_index()

    # Add carry counts
    carry_counts = rush.groupby(['player_id', 'season', 'week']).size().reset_index(name='rz_carries')
    weekly = weekly.merge(carry_counts, on=['player_id', 'season', 'week'])

    weekly.rename(columns={'rush_touchdown': 'rz_rush_tds'}, inplace=True)

    # Sort for EWMA calculation
    weekly = weekly.sort_values(['player_id', 'season', 'week'])

    # Compute trailing EWMA with shift(1) to prevent leakage
    weekly['rz_td_per_carry'] = (
        weekly.groupby('player_id')
        .apply(lambda x: (x['rz_rush_tds'] / x['rz_carries'].clip(lower=1))
               .shift(1).ewm(span=ewma_span, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Leave NaN from first weeks - XGBoost handles missing data natively
    # For prediction, merge_rz_td_rates will fill with actual averages
    # DO NOT fill here - let the model learn from missing data patterns

    return weekly[['player_id', 'season', 'week', 'rz_td_per_carry', 'rz_rush_tds', 'rz_carries']]


def _compute_receiving_rz_rates(rz: pd.DataFrame, ewma_span: int) -> pd.DataFrame:
    """Compute receiving TD rates in red zone per player."""
    # Filter to passing plays
    passing = rz[rz['play_type'] == 'pass'].copy()

    if len(passing) == 0:
        return pd.DataFrame()

    # Identify receiver (use receiver_player_id or receiver_id)
    if 'receiver_player_id' in passing.columns:
        passing['player_id'] = passing['receiver_player_id']
    elif 'receiver_id' in passing.columns:
        passing['player_id'] = passing['receiver_id']
    else:
        logger.warning("No receiver ID column found in PBP")
        return pd.DataFrame()

    # Drop plays with no player ID (incomplete passes, sacks)
    passing = passing.dropna(subset=['player_id'])

    # Group by player/week
    weekly = passing.groupby(['player_id', 'season', 'week']).agg({
        'pass_touchdown': 'sum',  # TDs this week
    }).reset_index()

    # Add target counts
    target_counts = passing.groupby(['player_id', 'season', 'week']).size().reset_index(name='rz_targets')
    weekly = weekly.merge(target_counts, on=['player_id', 'season', 'week'])

    weekly.rename(columns={'pass_touchdown': 'rz_rec_tds'}, inplace=True)

    # Sort for EWMA calculation
    weekly = weekly.sort_values(['player_id', 'season', 'week'])

    # Compute trailing EWMA with shift(1) to prevent leakage
    weekly['rz_td_per_target'] = (
        weekly.groupby('player_id')
        .apply(lambda x: (x['rz_rec_tds'] / x['rz_targets'].clip(lower=1))
               .shift(1).ewm(span=ewma_span, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Leave NaN from first weeks - XGBoost handles missing data natively
    # For prediction, merge_rz_td_rates will fill with actual averages
    # DO NOT fill here - let the model learn from missing data patterns

    return weekly[['player_id', 'season', 'week', 'rz_td_per_target', 'rz_rec_tds', 'rz_targets']]


def merge_rz_td_rates(
    df: pd.DataFrame,
    rz_rates: pd.DataFrame,
    player_id_col: str = 'player_id',
    for_training: bool = True,
) -> pd.DataFrame:
    """
    Merge RZ TD rates into main DataFrame.

    Args:
        df: Main DataFrame with player/season/week columns
        rz_rates: RZ TD rates from compute_player_rz_td_rates()
        player_id_col: Column name for player ID in df
        for_training: If True, leave NaN for XGBoost to handle natively.
                     If False, fill with computed league averages for prediction.

    Returns:
        DataFrame with RZ TD rate features added
    """
    if rz_rates.empty:
        logger.warning("Empty RZ rates, adding default values")
        if for_training:
            # Leave NaN for training - XGBoost handles missing data natively
            df['rz_td_per_snap'] = np.nan
            df['rz_td_per_carry'] = np.nan
            df['rz_td_per_target'] = np.nan
        else:
            # Use actual computed league averages for prediction
            df['rz_td_per_snap'] = ACTUAL_RZ_TD_PER_SNAP
            df['rz_td_per_carry'] = ACTUAL_RZ_RUSH_TD_RATE
            df['rz_td_per_target'] = ACTUAL_RZ_REC_TD_RATE
        return df

    merge_cols = [player_id_col, 'season', 'week']
    rate_cols = ['rz_td_per_snap', 'rz_td_per_carry', 'rz_td_per_target']

    # Rename player_id if needed
    rz_rates_copy = rz_rates.copy()
    if player_id_col != 'player_id' and 'player_id' in rz_rates_copy.columns:
        rz_rates_copy.rename(columns={'player_id': player_id_col}, inplace=True)

    # Select only needed columns
    available_cols = [c for c in merge_cols + rate_cols if c in rz_rates_copy.columns]
    rz_rates_copy = rz_rates_copy[available_cols].drop_duplicates()

    df = df.merge(rz_rates_copy, on=merge_cols, how='left')

    if for_training:
        # Leave NaN for training - XGBoost handles missing data natively
        # Only ensure columns exist
        if 'rz_td_per_snap' not in df.columns:
            df['rz_td_per_snap'] = np.nan
        if 'rz_td_per_carry' not in df.columns:
            df['rz_td_per_carry'] = np.nan
        if 'rz_td_per_target' not in df.columns:
            df['rz_td_per_target'] = np.nan
    else:
        # Fill missing with actual computed league averages for prediction
        if 'rz_td_per_snap' in df.columns:
            df['rz_td_per_snap'] = df['rz_td_per_snap'].fillna(ACTUAL_RZ_TD_PER_SNAP)
        else:
            df['rz_td_per_snap'] = ACTUAL_RZ_TD_PER_SNAP

        if 'rz_td_per_carry' in df.columns:
            df['rz_td_per_carry'] = df['rz_td_per_carry'].fillna(ACTUAL_RZ_RUSH_TD_RATE)
        else:
            df['rz_td_per_carry'] = ACTUAL_RZ_RUSH_TD_RATE

        if 'rz_td_per_target' in df.columns:
            df['rz_td_per_target'] = df['rz_td_per_target'].fillna(ACTUAL_RZ_REC_TD_RATE)
        else:
            df['rz_td_per_target'] = ACTUAL_RZ_REC_TD_RATE

    return df


def load_and_compute_rz_td_rates(
    season: int,
    weeks: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load PBP data and compute RZ TD rates.

    Convenience function that handles data loading.

    Args:
        season: Season year
        weeks: Optional list of weeks

    Returns:
        DataFrame with RZ TD rates
    """
    # Try to load PBP data (cascading path lookup)
    pbp_paths = [
        Path(f'data/nflverse/pbp_{season}.parquet'),
        Path('data/nflverse/pbp.parquet'),
        Path(f'data/processed/pbp_{season}.parquet'),
    ]

    pbp = None
    for path in pbp_paths:
        if path.exists():
            logger.info(f"Loading PBP from {path}")
            pbp = pd.read_parquet(path)
            break

    if pbp is None:
        logger.error("No PBP data found")
        return pd.DataFrame()

    return compute_player_rz_td_rates(pbp, season, weeks)
