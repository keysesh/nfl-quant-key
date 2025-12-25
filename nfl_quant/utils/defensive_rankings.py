"""
Defensive Rankings Module.

Calculates league-wide defensive rankings (#1-32) for:
- Pass Defense (EPA)
- Rush Defense (EPA)
- Total Defense (EPA)

Rankings are based on EPA per play from play-by-play data.
Lower EPA allowed = better defense = higher rank (closer to #1).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Module-level cache
_RANKINGS_CACHE = {}
_PBP_CACHE = None


def load_pbp(season: int = 2025, force_reload: bool = False) -> pd.DataFrame:
    """Load play-by-play data with caching."""
    global _PBP_CACHE

    cache_key = f"pbp_{season}"
    if _PBP_CACHE is not None and cache_key in _PBP_CACHE and not force_reload:
        return _PBP_CACHE[cache_key]

    # Try multiple paths (cascading lookup per CLAUDE.md)
    paths = [
        PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet',
        PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{season}.parquet',
        PROJECT_ROOT / 'data' / 'processed' / f'pbp_{season}.parquet'
    ]

    for path in paths:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                # Filter to requested season
                if 'season' in df.columns:
                    df = df[df['season'] == season]
                if _PBP_CACHE is None:
                    _PBP_CACHE = {}
                _PBP_CACHE[cache_key] = df
                logger.info(f"Loaded PBP from {path}: {len(df)} plays")
                return df
            except Exception as e:
                logger.warning(f"Failed to load PBP from {path}: {e}")
                continue

    logger.warning("No PBP data found")
    return pd.DataFrame()


def calculate_defensive_rankings(
    season: int,
    week: int
) -> pd.DataFrame:
    """
    Calculate defensive rankings 1-32 for all teams.

    Rankings are based on EPA per play allowed (lower = better = rank 1).

    Args:
        season: Season year
        week: Calculate rankings up to this week (exclusive)

    Returns:
        DataFrame with columns:
        - team
        - pass_def_epa, pass_def_rank (1-32)
        - rush_def_epa, rush_def_rank (1-32)
        - total_def_epa, total_def_rank (1-32)
    """
    cache_key = f"rankings_{season}_{week}"

    if cache_key in _RANKINGS_CACHE:
        return _RANKINGS_CACHE[cache_key]

    pbp = load_pbp(season)

    if len(pbp) == 0:
        return pd.DataFrame()

    # Filter to regular plays before the specified week
    plays = pbp[
        (pbp['week'] < week) &
        (pbp['play_type'].isin(['pass', 'run'])) &
        (pbp['epa'].notna()) &
        (pbp['defteam'].notna())
    ].copy()

    if len(plays) == 0:
        return pd.DataFrame()

    # Calculate EPA per play by defense
    # Pass defense
    pass_plays = plays[plays['play_type'] == 'pass']
    pass_def = pass_plays.groupby('defteam').agg({
        'epa': 'mean',
        'play_id': 'count'
    }).rename(columns={'epa': 'pass_def_epa', 'play_id': 'pass_plays'})

    # Rush defense
    rush_plays = plays[plays['play_type'] == 'run']
    rush_def = rush_plays.groupby('defteam').agg({
        'epa': 'mean',
        'play_id': 'count'
    }).rename(columns={'epa': 'rush_def_epa', 'play_id': 'rush_plays'})

    # Total defense
    total_def = plays.groupby('defteam').agg({
        'epa': 'mean',
        'play_id': 'count'
    }).rename(columns={'epa': 'total_def_epa', 'play_id': 'total_plays'})

    # Merge all
    rankings = pass_def.join(rush_def, how='outer').join(total_def, how='outer')
    rankings = rankings.reset_index().rename(columns={'defteam': 'team'})

    # Calculate ranks (1 = best defense = lowest EPA allowed)
    # ascending=True because lower EPA = better defense
    rankings['pass_def_rank'] = rankings['pass_def_epa'].rank(ascending=True, method='min').astype(int)
    rankings['rush_def_rank'] = rankings['rush_def_epa'].rank(ascending=True, method='min').astype(int)
    rankings['total_def_rank'] = rankings['total_def_epa'].rank(ascending=True, method='min').astype(int)

    # Sort by total defense rank
    rankings = rankings.sort_values('total_def_rank')

    _RANKINGS_CACHE[cache_key] = rankings

    return rankings


def get_team_defensive_ranks(
    team: str,
    season: int,
    week: int
) -> Dict:
    """
    Get defensive rankings for a specific team.

    Args:
        team: Team abbreviation
        season: Season year
        week: Current week

    Returns:
        Dict with:
        - pass_def_rank: int (1-32)
        - rush_def_rank: int (1-32)
        - total_def_rank: int (1-32)
        - pass_def_epa: float
        - rush_def_epa: float
        - total_def_epa: float
        - has_data: bool
    """
    rankings = calculate_defensive_rankings(season, week)

    if len(rankings) == 0:
        return _empty_def_ranks()

    team_row = rankings[rankings['team'] == team]

    if len(team_row) == 0:
        return _empty_def_ranks()

    row = team_row.iloc[0]

    return {
        'pass_def_rank': int(row['pass_def_rank']) if pd.notna(row['pass_def_rank']) else 16,
        'rush_def_rank': int(row['rush_def_rank']) if pd.notna(row['rush_def_rank']) else 16,
        'total_def_rank': int(row['total_def_rank']) if pd.notna(row['total_def_rank']) else 16,
        'pass_def_epa': float(row['pass_def_epa']) if pd.notna(row['pass_def_epa']) else 0.0,
        'rush_def_epa': float(row['rush_def_epa']) if pd.notna(row['rush_def_epa']) else 0.0,
        'total_def_epa': float(row['total_def_epa']) if pd.notna(row['total_def_epa']) else 0.0,
        'has_data': True
    }


def format_def_rank_class(rank: int) -> str:
    """
    Get CSS class for defensive rank badge.

    Args:
        rank: Defensive rank (1-32)

    Returns:
        CSS class name: 'def-elite', 'def-average', or 'def-poor'
    """
    if rank <= 10:
        return 'def-elite'  # Top 10 defenses
    elif rank <= 22:
        return 'def-average'  # Middle tier
    else:
        return 'def-poor'  # Bottom 10 defenses


def _empty_def_ranks() -> Dict:
    """Return empty defensive ranks."""
    return {
        'pass_def_rank': 16,
        'rush_def_rank': 16,
        'total_def_rank': 16,
        'pass_def_epa': 0.0,
        'rush_def_epa': 0.0,
        'total_def_epa': 0.0,
        'has_data': False
    }


def clear_caches():
    """Clear module-level caches."""
    global _RANKINGS_CACHE, _PBP_CACHE
    _RANKINGS_CACHE = {}
    _PBP_CACHE = None
    logger.info("Defensive rankings caches cleared")
