"""
Position Role Detection for V24 Defensive Matchup Analysis.

Determines player position roles (WR1/WR2/WR3, RB1/RB2, TE1) using:
- depth_charts.parquet: depth_team column (1=starter)
- weekly_stats.parquet: target_share for ranking within team
- ngs_receiving.parquet: avg_cushion for slot detection

All lookups use week-1 data to prevent data leakage.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Module-level caches
_DEPTH_CHARTS_CACHE = None
_NGS_RECEIVING_CACHE = None
_POSITION_ROLES_CACHE = None  # Computed position roles by team/week


def load_depth_charts(force_reload: bool = False) -> pd.DataFrame:
    """
    Load depth charts using canonical loader with caching.

    Returns:
        DataFrame with columns: gsis_id, club_code, week, season, position, depth_team
    """
    global _DEPTH_CHARTS_CACHE

    if _DEPTH_CHARTS_CACHE is not None and not force_reload:
        return _DEPTH_CHARTS_CACHE

    try:
        from nfl_quant.data.depth_chart_loader import get_depth_charts
        df = get_depth_charts()
    except Exception as e:
        logger.warning(f"Failed to load depth charts: {e}")
        return pd.DataFrame()

    # Rename club_code to team for consistency with other datasets
    if 'club_code' in df.columns and 'team' not in df.columns:
        df = df.rename(columns={'club_code': 'team'})

    # Ensure depth_team is integer where possible
    if 'depth_team' in df.columns:
        df['depth_team'] = pd.to_numeric(df['depth_team'], errors='coerce')

    _DEPTH_CHARTS_CACHE = df
    logger.info(f"Loaded depth charts: {len(df):,} records")

    return df


def load_ngs_receiving(force_reload: bool = False) -> pd.DataFrame:
    """
    Load NGS receiving data for slot detection.

    Key columns: player_gsis_id, avg_cushion, avg_intended_air_yards, week, season
    """
    global _NGS_RECEIVING_CACHE

    if _NGS_RECEIVING_CACHE is not None and not force_reload:
        return _NGS_RECEIVING_CACHE

    path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_receiving.parquet'

    if not path.exists():
        logger.warning(f"NGS receiving not found at {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    _NGS_RECEIVING_CACHE = df
    logger.info(f"Loaded NGS receiving: {len(df):,} records")

    return df


def get_player_position_role(
    player_id: str,
    team: str,
    season: int,
    week: int,
    position: str,
    weekly_stats: pd.DataFrame = None,
    depth_charts: pd.DataFrame = None
) -> Dict:
    """
    Determine player's position role (WR1/WR2/WR3, RB1/RB2, TE1).

    Uses week-1 data to prevent leakage.

    Args:
        player_id: Player GSIS ID
        team: Team abbreviation
        season: Season year
        week: Current week (will lookup week-1)
        position: Position (WR, RB, TE)
        weekly_stats: Optional pre-loaded weekly stats
        depth_charts: Optional pre-loaded depth charts

    Returns:
        Dict with:
        - position_role: str ('WR1', 'WR2', 'WR3', 'RB1', 'RB2', 'TE1', 'UNKNOWN')
        - pos_rank: int (1, 2, 3, or 99 for unknown)
        - is_starter: bool
        - depth_team: int (from depth chart)
    """
    result = {
        'position_role': 'UNKNOWN',
        'pos_rank': 99,
        'is_starter': False,
        'depth_team': 99
    }

    if position not in ['WR', 'RB', 'TE']:
        return result

    # Use week-1 for no leakage
    lookup_week = week - 1 if week > 1 else 1

    # Load data if not provided
    if depth_charts is None:
        depth_charts = load_depth_charts()

    if len(depth_charts) == 0:
        # Fall back to target share ranking
        return _get_role_from_target_share(
            player_id, team, season, lookup_week, position, weekly_stats
        )

    # Get depth chart info for this player
    player_depth = depth_charts[
        (depth_charts['gsis_id'] == player_id) &
        (depth_charts['team'] == team) &
        (depth_charts['season'] == season) &
        (depth_charts['week'] == lookup_week) &
        (depth_charts['position'] == position)
    ]

    if len(player_depth) > 0:
        depth_team_val = player_depth['depth_team'].iloc[0]
        result['depth_team'] = int(depth_team_val) if pd.notna(depth_team_val) else 99

    # For WR: multiple players can be depth_team=1, so rank by target share
    # For RB/TE: typically clearer hierarchy
    if position == 'WR':
        return _get_wr_role_with_target_share(
            player_id, team, season, lookup_week, weekly_stats, depth_charts, result
        )
    else:
        # RB/TE: use depth_team more directly, fall back to stats
        return _get_rb_te_role(
            player_id, team, season, lookup_week, position, weekly_stats, depth_charts, result
        )


def _get_wr_role_with_target_share(
    player_id: str,
    team: str,
    season: int,
    week: int,
    weekly_stats: pd.DataFrame,
    depth_charts: pd.DataFrame,
    result: Dict
) -> Dict:
    """
    Determine WR role using target share ranking within team.

    WR1 = highest target share, WR2 = second highest, etc.
    """
    if weekly_stats is None:
        # Load weekly stats
        ws_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if ws_path.exists():
            weekly_stats = pd.read_parquet(ws_path)
        else:
            # Can't determine role without stats
            if result['depth_team'] <= 3:
                result['position_role'] = f"WR{result['depth_team']}"
                result['pos_rank'] = result['depth_team']
                result['is_starter'] = result['depth_team'] == 1
            return result

    # Get team WRs from recent weeks (last 4 weeks before current)
    team_wrs = weekly_stats[
        (weekly_stats['team'] == team) &
        (weekly_stats['season'] == season) &
        (weekly_stats['week'] <= week) &
        (weekly_stats['week'] > max(0, week - 4)) &
        (weekly_stats['position'] == 'WR')
    ]

    if len(team_wrs) == 0:
        # No stats available, use depth chart
        if result['depth_team'] <= 3:
            result['position_role'] = f"WR{result['depth_team']}"
            result['pos_rank'] = result['depth_team']
            result['is_starter'] = result['depth_team'] == 1
        return result

    # Calculate average target share per player
    avg_target_share = team_wrs.groupby('player_id')['target_share'].mean().sort_values(ascending=False)

    # Find player's rank
    if player_id in avg_target_share.index:
        rank = list(avg_target_share.index).index(player_id) + 1
        rank = min(rank, 3)  # Cap at WR3
        result['position_role'] = f"WR{rank}"
        result['pos_rank'] = rank
        result['is_starter'] = rank == 1
    elif result['depth_team'] <= 3:
        result['position_role'] = f"WR{result['depth_team']}"
        result['pos_rank'] = result['depth_team']
        result['is_starter'] = result['depth_team'] == 1

    return result


def _get_rb_te_role(
    player_id: str,
    team: str,
    season: int,
    week: int,
    position: str,
    weekly_stats: pd.DataFrame,
    depth_charts: pd.DataFrame,
    result: Dict
) -> Dict:
    """
    Determine RB/TE role using depth chart and usage stats.
    """
    # Use depth_team if available
    if result['depth_team'] <= 2:
        result['position_role'] = f"{position}{result['depth_team']}"
        result['pos_rank'] = result['depth_team']
        result['is_starter'] = result['depth_team'] == 1
        return result

    # Fall back to usage ranking
    if weekly_stats is None:
        ws_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if ws_path.exists():
            weekly_stats = pd.read_parquet(ws_path)
        else:
            return result

    stat_col = 'carries' if position == 'RB' else 'targets'

    team_players = weekly_stats[
        (weekly_stats['team'] == team) &
        (weekly_stats['season'] == season) &
        (weekly_stats['week'] <= week) &
        (weekly_stats['week'] > max(0, week - 4)) &
        (weekly_stats['position'] == position)
    ]

    if len(team_players) == 0:
        return result

    # Rank by usage
    avg_usage = team_players.groupby('player_id')[stat_col].mean().sort_values(ascending=False)

    if player_id in avg_usage.index:
        rank = list(avg_usage.index).index(player_id) + 1
        rank = min(rank, 2)  # Cap at position2
        result['position_role'] = f"{position}{rank}"
        result['pos_rank'] = rank
        result['is_starter'] = rank == 1

    return result


def _get_role_from_target_share(
    player_id: str,
    team: str,
    season: int,
    week: int,
    position: str,
    weekly_stats: pd.DataFrame
) -> Dict:
    """
    Fallback: determine role from target share when depth chart unavailable.
    """
    result = {
        'position_role': 'UNKNOWN',
        'pos_rank': 99,
        'is_starter': False,
        'depth_team': 99
    }

    if weekly_stats is None:
        ws_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if ws_path.exists():
            weekly_stats = pd.read_parquet(ws_path)
        else:
            return result

    stat_col = 'carries' if position == 'RB' else 'target_share'

    team_players = weekly_stats[
        (weekly_stats['team'] == team) &
        (weekly_stats['season'] == season) &
        (weekly_stats['week'] <= week) &
        (weekly_stats['week'] > max(0, week - 4)) &
        (weekly_stats['position'] == position)
    ]

    if len(team_players) == 0:
        return result

    avg_stat = team_players.groupby('player_id')[stat_col].mean().sort_values(ascending=False)

    if player_id in avg_stat.index:
        rank = list(avg_stat.index).index(player_id) + 1
        max_rank = 3 if position == 'WR' else 2
        rank = min(rank, max_rank)
        result['position_role'] = f"{position}{rank}"
        result['pos_rank'] = rank
        result['is_starter'] = rank == 1

    return result


def is_slot_receiver(
    player_id: str,
    season: int,
    week: int = None,
    ngs_data: pd.DataFrame = None,
    cushion_threshold: float = 5.5,
    adot_threshold: float = 8.0
) -> Tuple[bool, float]:
    """
    Detect if receiver primarily aligns in slot using NGS metrics.

    Slot receivers typically have:
    - avg_cushion > 5.5 yards (more space at snap)
    - avg_intended_air_yards < 8.0 (shorter routes)

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Optional specific week to check (uses trailing if None)
        ngs_data: Optional pre-loaded NGS data
        cushion_threshold: Cushion yards threshold for slot
        adot_threshold: aDOT threshold for slot

    Returns:
        Tuple of (is_slot: bool, slot_alignment_pct: float)
    """
    if ngs_data is None:
        ngs_data = load_ngs_receiving()

    if len(ngs_data) == 0:
        return False, 0.0

    # Filter to player
    if week is not None:
        # Use trailing weeks (week-1 and prior)
        player_ngs = ngs_data[
            (ngs_data['player_gsis_id'] == player_id) &
            (ngs_data['season'] == season) &
            (ngs_data['week'] < week) &
            (ngs_data['week'] > max(0, week - 5))
        ]
    else:
        player_ngs = ngs_data[
            (ngs_data['player_gsis_id'] == player_id) &
            (ngs_data['season'] == season)
        ]

    if len(player_ngs) == 0:
        return False, 0.0

    # Calculate metrics
    avg_cushion = player_ngs['avg_cushion'].mean()
    avg_adot = player_ngs['avg_intended_air_yards'].mean() if 'avg_intended_air_yards' in player_ngs.columns else 10.0

    # Slot detection
    is_slot = avg_cushion > cushion_threshold and avg_adot < adot_threshold

    # Calculate slot alignment percentage (what % of weeks had slot-like metrics)
    slot_weeks = (
        (player_ngs['avg_cushion'] > cushion_threshold) &
        (player_ngs.get('avg_intended_air_yards', pd.Series([10.0] * len(player_ngs))) < adot_threshold)
    )
    slot_pct = slot_weeks.mean() if len(slot_weeks) > 0 else 0.0

    return is_slot, float(slot_pct)


def get_all_position_roles_vectorized(
    weekly_stats: pd.DataFrame,
    depth_charts: pd.DataFrame = None,
    season: int = None
) -> pd.DataFrame:
    """
    Vectorized calculation of position roles for all players.

    More efficient for batch processing during training.

    Args:
        weekly_stats: Weekly stats DataFrame
        depth_charts: Optional depth charts DataFrame
        season: Optional filter to specific season

    Returns:
        DataFrame with player_id, team, week, season, position_role, pos_rank, is_starter
    """
    if depth_charts is None:
        depth_charts = load_depth_charts()

    # Filter to season if specified
    if season is not None:
        weekly_stats = weekly_stats[weekly_stats['season'] == season]

    # Focus on WR, RB, TE
    positions = ['WR', 'RB', 'TE']
    stats_filtered = weekly_stats[weekly_stats['position'].isin(positions)].copy()

    if len(stats_filtered) == 0:
        return pd.DataFrame()

    # Calculate trailing target share rank within team/position for each week
    # This is the most robust way to determine role

    results = []

    for (team, season_val, position), group in stats_filtered.groupby(['team', 'season', 'position']):
        weeks = sorted(group['week'].unique())

        for week in weeks:
            # Use trailing 4 weeks (excluding current)
            trailing = group[
                (group['week'] < week) &
                (group['week'] >= max(1, week - 4))
            ]

            if len(trailing) == 0:
                continue

            # Rank by usage
            if position == 'WR':
                stat_col = 'target_share'
            elif position == 'RB':
                stat_col = 'carries'
            else:  # TE
                stat_col = 'targets'

            avg_usage = trailing.groupby('player_id')[stat_col].mean().sort_values(ascending=False)

            max_rank = 3 if position == 'WR' else 2

            for rank_idx, player_id in enumerate(avg_usage.index[:max_rank], 1):
                results.append({
                    'player_id': player_id,
                    'team': team,
                    'week': week,
                    'season': season_val,
                    'position': position,
                    'position_role': f"{position}{rank_idx}",
                    'pos_rank': rank_idx,
                    'is_starter': rank_idx == 1
                })

    if len(results) == 0:
        return pd.DataFrame()

    return pd.DataFrame(results)


def add_slot_detection_vectorized(
    roles_df: pd.DataFrame,
    ngs_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Add slot detection to position roles DataFrame.

    Args:
        roles_df: DataFrame with position roles
        ngs_data: Optional NGS receiving data

    Returns:
        DataFrame with is_slot_receiver and slot_alignment_pct columns added
    """
    if ngs_data is None:
        ngs_data = load_ngs_receiving()

    roles_df = roles_df.copy()
    roles_df['is_slot_receiver'] = False
    roles_df['slot_alignment_pct'] = 0.0

    if len(ngs_data) == 0:
        return roles_df

    # Filter to WR only
    wr_mask = roles_df['position'] == 'WR'

    if not wr_mask.any():
        return roles_df

    # Calculate avg cushion per player/season
    ngs_avg = ngs_data.groupby(['player_gsis_id', 'season']).agg({
        'avg_cushion': 'mean',
        'avg_intended_air_yards': 'mean'
    }).reset_index()

    ngs_avg['is_slot'] = (
        (ngs_avg['avg_cushion'] > 5.5) &
        (ngs_avg['avg_intended_air_yards'] < 8.0)
    )

    # Calculate slot percentage
    slot_pct = ngs_data.copy()
    slot_pct['is_slot_week'] = (
        (slot_pct['avg_cushion'] > 5.5) &
        (slot_pct['avg_intended_air_yards'] < 8.0)
    )
    slot_pct_agg = slot_pct.groupby(['player_gsis_id', 'season'])['is_slot_week'].mean().reset_index()
    slot_pct_agg = slot_pct_agg.rename(columns={'is_slot_week': 'slot_alignment_pct'})

    # Merge
    ngs_avg = ngs_avg.merge(slot_pct_agg, on=['player_gsis_id', 'season'], how='left')
    ngs_avg['slot_alignment_pct'] = ngs_avg['slot_alignment_pct'].fillna(0.0)

    # Join to roles
    roles_df = roles_df.merge(
        ngs_avg[['player_gsis_id', 'season', 'is_slot', 'slot_alignment_pct']].rename(
            columns={'player_gsis_id': 'player_id', 'is_slot': 'is_slot_ngs', 'slot_alignment_pct': 'slot_pct_ngs'}
        ),
        on=['player_id', 'season'],
        how='left'
    )

    # Update slot columns where we have NGS data
    ngs_mask = roles_df['is_slot_ngs'].notna()
    roles_df.loc[ngs_mask & wr_mask, 'is_slot_receiver'] = roles_df.loc[ngs_mask & wr_mask, 'is_slot_ngs'].fillna(False)
    roles_df.loc[ngs_mask & wr_mask, 'slot_alignment_pct'] = roles_df.loc[ngs_mask & wr_mask, 'slot_pct_ngs'].fillna(0.0)

    # Clean up temp columns
    roles_df = roles_df.drop(columns=['is_slot_ngs', 'slot_pct_ngs'], errors='ignore')

    return roles_df


def clear_caches():
    """Clear all module-level caches."""
    global _DEPTH_CHARTS_CACHE, _NGS_RECEIVING_CACHE, _POSITION_ROLES_CACHE
    _DEPTH_CHARTS_CACHE = None
    _NGS_RECEIVING_CACHE = None
    _POSITION_ROLES_CACHE = None
    logger.info("Position role caches cleared")


# Encoding for model features
POSITION_ROLE_ENCODING = {
    'WR1': 1, 'WR2': 2, 'WR3': 3,
    'RB1': 1, 'RB2': 2,
    'TE1': 1, 'TE2': 2,
    'UNKNOWN': 99
}


def encode_position_role(role: str) -> int:
    """Encode position role to integer for model."""
    return POSITION_ROLE_ENCODING.get(role, 99)
