"""
Goal Line Role Detection

Identifies the primary goal-line player (yardline <= 5) for each team.
This is critical for TD prediction - primary GL backs score 3x more TDs.

Key Features:
- gl_carries: Goal-line carries per player-week
- gl_carry_share: Player's share of team's GL carries
- gl_targets: Goal-line targets per player-week
- gl_target_share: Player's share of team's GL targets
- is_primary_gl_back: Binary flag for #1 GL back on team

Usage:
    gl_roles = compute_goal_line_roles(pbp, season=2024)
    df = merge_goal_line_features(df, gl_roles)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def compute_goal_line_roles(
    pbp: pd.DataFrame,
    season: int,
    weeks: Optional[List[int]] = None,
    ewma_span: int = 4,
) -> pd.DataFrame:
    """
    Identify goal-line role for each player.

    Goal line is defined as yardline_100 <= 5 (within 5 yards of end zone).

    Args:
        pbp: Play-by-play DataFrame
        season: Season year
        weeks: Optional list of weeks to include
        ewma_span: EWMA span for trailing average

    Returns:
        DataFrame with columns:
        - player_id
        - season
        - week
        - gl_carries: Goal-line carries this week
        - gl_carry_share: Share of team's GL carries (trailing EWMA)
        - gl_targets: Goal-line targets this week
        - gl_target_share: Share of team's GL targets (trailing EWMA)
        - is_primary_gl_back: Binary flag (highest GL carry share on team)
        - is_primary_gl_receiver: Binary flag (highest GL target share on team)
    """
    logger.info(f"Computing goal-line roles for {season}...")

    # Filter to season
    df = pbp[pbp['season'] == season].copy()

    if weeks is not None:
        df = df[df['week'].isin(weeks)]

    if len(df) == 0:
        logger.warning(f"No PBP data for season {season}")
        return pd.DataFrame()

    # Filter to goal line plays (yardline_100 <= 5)
    gl = df[
        (df['yardline_100'] <= 5) &
        (df['play_type'].isin(['pass', 'run']))
    ].copy()

    logger.info(f"  Goal-line plays: {len(gl):,}")

    if len(gl) == 0:
        return pd.DataFrame()

    # Compute rush GL roles
    rush_roles = _compute_rush_gl_roles(gl, ewma_span)

    # Compute receiving GL roles
    rec_roles = _compute_receiving_gl_roles(gl, ewma_span)

    # Combine
    if len(rush_roles) > 0 and len(rec_roles) > 0:
        combined = pd.merge(
            rush_roles,
            rec_roles,
            on=['player_id', 'season', 'week', 'posteam'],
            how='outer'
        )
    elif len(rush_roles) > 0:
        combined = rush_roles
    elif len(rec_roles) > 0:
        combined = rec_roles
    else:
        return pd.DataFrame()

    # Fill NaN
    for col in ['gl_carries', 'gl_carry_share', 'gl_targets', 'gl_target_share',
                'is_primary_gl_back', 'is_primary_gl_receiver']:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # Add combined GL opportunity share
    combined['gl_opportunity_share'] = (
        combined.get('gl_carry_share', 0) * 0.6 +  # Weight rushing more heavily
        combined.get('gl_target_share', 0) * 0.4
    )

    logger.info(f"  Computed GL roles for {combined['player_id'].nunique()} players")

    return combined


def _compute_rush_gl_roles(gl: pd.DataFrame, ewma_span: int) -> pd.DataFrame:
    """Compute goal-line rushing roles per player."""
    # Filter to rushing plays
    rush = gl[gl['play_type'] == 'run'].copy()

    if len(rush) == 0:
        return pd.DataFrame()

    # Identify rusher
    if 'rusher_player_id' in rush.columns:
        rush['player_id'] = rush['rusher_player_id']
    elif 'rusher_id' in rush.columns:
        rush['player_id'] = rush['rusher_id']
    else:
        logger.warning("No rusher ID column found")
        return pd.DataFrame()

    rush = rush.dropna(subset=['player_id'])

    # Count GL carries per player-team-week
    player_gl = rush.groupby(['player_id', 'posteam', 'season', 'week']).size().reset_index(name='gl_carries')

    # Count team GL carries per team-week
    team_gl = rush.groupby(['posteam', 'season', 'week']).size().reset_index(name='team_gl_carries')

    # Merge to get share
    player_gl = player_gl.merge(team_gl, on=['posteam', 'season', 'week'])
    player_gl['gl_carry_share_raw'] = player_gl['gl_carries'] / player_gl['team_gl_carries']

    # Sort for EWMA
    player_gl = player_gl.sort_values(['player_id', 'season', 'week'])

    # Compute trailing EWMA with shift(1)
    player_gl['gl_carry_share'] = (
        player_gl.groupby('player_id')['gl_carry_share_raw']
        .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
    )
    player_gl['gl_carry_share'] = player_gl['gl_carry_share'].fillna(0)

    # Identify primary GL back per team-week (highest share)
    player_gl['gl_carry_rank'] = player_gl.groupby(['posteam', 'season', 'week'])['gl_carry_share'].rank(
        method='first', ascending=False
    )
    player_gl['is_primary_gl_back'] = (player_gl['gl_carry_rank'] == 1).astype(int)

    return player_gl[['player_id', 'posteam', 'season', 'week', 'gl_carries',
                      'gl_carry_share', 'is_primary_gl_back']]


def _compute_receiving_gl_roles(gl: pd.DataFrame, ewma_span: int) -> pd.DataFrame:
    """Compute goal-line receiving roles per player."""
    # Filter to passing plays
    passing = gl[gl['play_type'] == 'pass'].copy()

    if len(passing) == 0:
        return pd.DataFrame()

    # Identify receiver
    if 'receiver_player_id' in passing.columns:
        passing['player_id'] = passing['receiver_player_id']
    elif 'receiver_id' in passing.columns:
        passing['player_id'] = passing['receiver_id']
    else:
        logger.warning("No receiver ID column found")
        return pd.DataFrame()

    passing = passing.dropna(subset=['player_id'])

    # Count GL targets per player-team-week
    player_gl = passing.groupby(['player_id', 'posteam', 'season', 'week']).size().reset_index(name='gl_targets')

    # Count team GL targets per team-week
    team_gl = passing.groupby(['posteam', 'season', 'week']).size().reset_index(name='team_gl_targets')

    # Merge to get share
    player_gl = player_gl.merge(team_gl, on=['posteam', 'season', 'week'])
    player_gl['gl_target_share_raw'] = player_gl['gl_targets'] / player_gl['team_gl_targets']

    # Sort for EWMA
    player_gl = player_gl.sort_values(['player_id', 'season', 'week'])

    # Compute trailing EWMA with shift(1)
    player_gl['gl_target_share'] = (
        player_gl.groupby('player_id')['gl_target_share_raw']
        .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
    )
    player_gl['gl_target_share'] = player_gl['gl_target_share'].fillna(0)

    # Identify primary GL receiver per team-week
    player_gl['gl_target_rank'] = player_gl.groupby(['posteam', 'season', 'week'])['gl_target_share'].rank(
        method='first', ascending=False
    )
    player_gl['is_primary_gl_receiver'] = (player_gl['gl_target_rank'] == 1).astype(int)

    return player_gl[['player_id', 'posteam', 'season', 'week', 'gl_targets',
                      'gl_target_share', 'is_primary_gl_receiver']]


def merge_goal_line_features(
    df: pd.DataFrame,
    gl_roles: pd.DataFrame,
    player_id_col: str = 'player_id',
) -> pd.DataFrame:
    """
    Merge goal-line features into main DataFrame.

    Args:
        df: Main DataFrame with player/season/week columns
        gl_roles: GL roles from compute_goal_line_roles()
        player_id_col: Column name for player ID in df

    Returns:
        DataFrame with GL features added
    """
    if gl_roles.empty:
        logger.warning("Empty GL roles, adding default values")
        df['gl_carry_share'] = 0.0
        df['gl_target_share'] = 0.0
        df['gl_opportunity_share'] = 0.0
        df['is_primary_gl_back'] = 0
        return df

    merge_cols = [player_id_col, 'season', 'week']
    feature_cols = ['gl_carry_share', 'gl_target_share', 'gl_opportunity_share',
                    'is_primary_gl_back', 'is_primary_gl_receiver']

    # Rename player_id if needed
    gl_copy = gl_roles.copy()
    if player_id_col != 'player_id' and 'player_id' in gl_copy.columns:
        gl_copy.rename(columns={'player_id': player_id_col}, inplace=True)

    # Select only needed columns
    available_cols = [c for c in merge_cols + feature_cols if c in gl_copy.columns]
    gl_copy = gl_copy[available_cols].drop_duplicates()

    df = df.merge(gl_copy, on=merge_cols, how='left')

    # Fill missing with 0
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def load_and_compute_goal_line_roles(
    season: int,
    weeks: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load PBP data and compute goal-line roles.

    Convenience function that handles data loading.

    Args:
        season: Season year
        weeks: Optional list of weeks

    Returns:
        DataFrame with GL roles
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

    return compute_goal_line_roles(pbp, season, weeks)
