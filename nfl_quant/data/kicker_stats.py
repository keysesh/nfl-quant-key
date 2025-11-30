"""
Kicker Stats Collection from nflverse

Extract kicker historical statistics for prop predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def get_kicker_historical_stats(pbp_df: pd.DataFrame, seasons: List[int] = None) -> pd.DataFrame:
    """
    Extract kicker historical stats from play-by-play data.
    
    Args:
        pbp_df: Play-by-play DataFrame from nflverse
        seasons: List of seasons to use (None = use all in pbp_df)
    
    Returns:
        DataFrame with weekly kicker stats and trailing averages
    """
    
    if pbp_df is None or len(pbp_df) == 0:
        logger.warning("No PBP data provided")
        return pd.DataFrame()
    
    # Filter to specific seasons if provided
    if seasons:
        pbp_df = pbp_df[pbp_df['season'].isin(seasons)]
    
    # Filter to kicking plays only
    kicks = pbp_df[pbp_df['play_type'].isin(['field_goal', 'extra_point'])].copy()
    
    if len(kicks) == 0:
        logger.warning("No kicking plays found in PBP data")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(kicks)} kicking plays")
    
    # Aggregate by kicker per game
    kicker_weekly = []
    
    for (kicker_id, kicker_name, team, season, week), group in kicks.groupby([
        'kicker_player_id',
        'kicker_player_name', 
        'posteam',
        'season',
        'week'
    ]):
        
        # Field Goals
        fg_attempts = len(group[group['play_type'] == 'field_goal'])
        fg_made = (group[group['play_type'] == 'field_goal']['field_goal_result'] == 'made').sum()
        
        # Extra Points
        xp_attempts = len(group[group['play_type'] == 'extra_point'])
        xp_made = (group[group['play_type'] == 'extra_point']['extra_point_result'] == 'good').sum()
        
        # Average kick distance
        kick_distances = group['kick_distance'].dropna()
        avg_distance = kick_distances.mean() if len(kick_distances) > 0 else 0
        
        kicker_weekly.append({
            'kicker_player_id': kicker_id,
            'kicker_player_name': kicker_name,
            'team': team,
            'season': season,
            'week': week,
            'fg_attempts': fg_attempts,
            'fg_made': fg_made,
            'xp_attempts': xp_attempts,
            'xp_made': xp_made,
            'avg_distance': avg_distance,
            'kicking_points': fg_made * 3 + xp_made,
        })
    
    df = pd.DataFrame(kicker_weekly)
    
    if len(df) == 0:
        logger.warning("No kicker stats aggregated")
        return pd.DataFrame()
    
    logger.info(f"Aggregated stats for {df['kicker_player_id'].nunique()} unique kickers")
    
    # Calculate trailing averages for each kicker
    df = df.sort_values(['kicker_player_id', 'season', 'week'])
    
    # Trailing FG attempts per game
    df['trailing_fg_attempts_per_game'] = (
        df.groupby('kicker_player_id')['fg_attempts']
        .transform(lambda x: x.rolling(4, min_periods=2).mean())
    )
    
    # Trailing FG percentage
    df['trailing_fg_pct'] = df.groupby('kicker_player_id').apply(
        lambda x: x['fg_made'].rolling(4, min_periods=2).sum() / 
                  x['fg_attempts'].rolling(4, min_periods=2).sum()
        if x['fg_attempts'].rolling(4, min_periods=2).sum().iloc[-1] > 0 else 0.8
    ).reset_index(level=0, drop=True).fillna(0.8)
    
    # Trailing XP per game
    df['trailing_xp_per_game'] = (
        df.groupby('kicker_player_id')['xp_attempts']
        .transform(lambda x: x.rolling(4, min_periods=2).mean())
    )
    
    # Trailing kicking points per game
    df['trailing_kicking_points_per_game'] = (
        df.groupby('kicker_player_id')['kicking_points']
        .transform(lambda x: x.rolling(4, min_periods=2).mean())
    )
    
    logger.info(f"Calculated trailing averages for kickers")
    
    return df


def get_kicker_stats_for_player(kicker_name: str, team: str, df: pd.DataFrame) -> Optional[Dict]:
    """
    Get most recent kicker stats for a specific player.
    
    Args:
        kicker_name: Kicker's name
        team: Team abbreviation
        df: Kicker stats DataFrame from get_kicker_historical_stats()
    
    Returns:
        Dictionary with trailing stats or None if not found
    """
    
    if df is None or len(df) == 0:
        return None
    
    # Find kicker's most recent stats
    kicker_data = df[(df['kicker_player_name'] == kicker_name) & 
                     (df['team'] == team)].copy()
    
    if len(kicker_data) == 0:
        logger.warning(f"Kicker {kicker_name} ({team}) not found in stats")
        return None
    
    # Get most recent week
    most_recent = kicker_data.sort_values(['season', 'week']).iloc[-1]
    
    return {
        'kicker_name': kicker_name,
        'team': team,
        'trailing_fg_attempts_per_game': most_recent.get('trailing_fg_attempts_per_game', 2.0),
        'trailing_fg_pct': most_recent.get('trailing_fg_pct', 0.85),
        'trailing_xp_per_game': most_recent.get('trailing_xp_per_game', 3.0),
        'trailing_kicking_points_per_game': most_recent.get('trailing_kicking_points_per_game', 9.0),
        'weeks_played': len(kicker_data),
    }




