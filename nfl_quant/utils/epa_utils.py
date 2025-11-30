#!/usr/bin/env python3
"""
Centralized EPA Calculation Utilities

Provides consistent EPA calculation with regression to mean across the entire system.
This ensures that all components (game line predictions, player props, etc.) use
the same EPA methodology.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def regress_epa_to_mean(
    raw_epa: float,
    sample_size: int,
    league_mean: float = 0.0,
    regression_factor: float = 0.5
) -> float:
    """
    Regress EPA values toward league mean to account for sample size variance.

    This is critical for preventing extreme predictions. In sports analytics,
    small sample sizes lead to noisy EPA estimates. We regress toward league average.

    Args:
        raw_epa: Raw EPA per play from team data
        sample_size: Number of games or plays in sample
        league_mean: League average EPA (0.0 by definition)
        regression_factor: Base regression weight (0.5 = 50% toward mean with 10 samples)

    Returns:
        Regressed EPA value closer to league mean
    """
    # More samples = less regression needed
    # With 10 samples, we regress 50% toward mean
    # With 8 samples, we regress 62.5% toward mean
    # With 16 samples, we regress 31.25% toward mean
    base_samples = 10
    regression_weight = regression_factor * (base_samples / max(sample_size, 1))
    regression_weight = min(regression_weight, 0.75)  # Cap at 75% regression

    regressed_epa = raw_epa * (1 - regression_weight) + league_mean * regression_weight

    return regressed_epa


def calculate_team_defensive_epa(
    pbp_df: pd.DataFrame,
    team: str,
    weeks: int = 10,
    season: Optional[int] = None
) -> dict:
    """
    Calculate team defensive EPA metrics with regression to mean.

    Args:
        pbp_df: Play-by-play DataFrame
        team: Team abbreviation
        weeks: Number of recent weeks to include
        season: Season to filter (if None, uses max season)

    Returns:
        dict with overall, pass, and rush defensive EPA
    """
    if season is not None:
        pbp_df = pbp_df[pbp_df['season'] == season].copy()

    # Filter to relevant weeks
    max_week = pbp_df['week'].max()
    recent_weeks = range(max(1, max_week - weeks + 1), max_week + 1)

    # Defensive plays (when team is defending)
    def_plays = pbp_df[
        (pbp_df['defteam'] == team) &
        (pbp_df['week'].isin(recent_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    if len(def_plays) == 0:
        return {
            'def_epa_allowed': 0.0,
            'pass_def_epa': 0.0,
            'rush_def_epa': 0.0,
            'sample_games': 0
        }

    # Count sample size
    sample_games = len(def_plays['game_id'].unique())

    # Calculate RAW EPA per play
    raw_total_epa = def_plays['epa'].mean()

    # Pass defense
    pass_def = def_plays[def_plays['play_type'] == 'pass']
    raw_pass_epa = pass_def['epa'].mean() if len(pass_def) > 0 else 0.0

    # Rush defense
    rush_def = def_plays[def_plays['play_type'] == 'run']
    raw_rush_epa = rush_def['epa'].mean() if len(rush_def) > 0 else 0.0

    # CRITICAL: Regress toward league mean
    # Positive EPA = bad defense (opponents score well)
    # Negative EPA = good defense (opponents score poorly)
    def_epa_allowed = regress_epa_to_mean(raw_total_epa, sample_games)
    pass_def_epa = regress_epa_to_mean(raw_pass_epa, sample_games)
    rush_def_epa = regress_epa_to_mean(raw_rush_epa, sample_games)

    return {
        'def_epa_allowed': float(def_epa_allowed),
        'pass_def_epa': float(pass_def_epa),
        'rush_def_epa': float(rush_def_epa),
        'sample_games': sample_games
    }


def calculate_team_offensive_epa(
    pbp_df: pd.DataFrame,
    team: str,
    weeks: int = 10,
    season: Optional[int] = None
) -> dict:
    """
    Calculate team offensive EPA metrics with regression to mean.

    Args:
        pbp_df: Play-by-play DataFrame
        team: Team abbreviation
        weeks: Number of recent weeks to include
        season: Season to filter (if None, uses max season)

    Returns:
        dict with overall, pass, and rush offensive EPA
    """
    if season is not None:
        pbp_df = pbp_df[pbp_df['season'] == season].copy()

    # Filter to relevant weeks
    max_week = pbp_df['week'].max()
    recent_weeks = range(max(1, max_week - weeks + 1), max_week + 1)

    # Offensive plays
    off_plays = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(recent_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    if len(off_plays) == 0:
        return {
            'off_epa': 0.0,
            'pass_off_epa': 0.0,
            'rush_off_epa': 0.0,
            'sample_games': 0
        }

    # Count sample size
    sample_games = len(off_plays['game_id'].unique())

    # Calculate RAW EPA per play
    raw_total_epa = off_plays['epa'].mean()

    # Pass offense
    pass_off = off_plays[off_plays['play_type'] == 'pass']
    raw_pass_epa = pass_off['epa'].mean() if len(pass_off) > 0 else 0.0

    # Rush offense
    rush_off = off_plays[off_plays['play_type'] == 'run']
    raw_rush_epa = rush_off['epa'].mean() if len(rush_off) > 0 else 0.0

    # CRITICAL: Regress toward league mean
    off_epa = regress_epa_to_mean(raw_total_epa, sample_games)
    pass_off_epa = regress_epa_to_mean(raw_pass_epa, sample_games)
    rush_off_epa = regress_epa_to_mean(raw_rush_epa, sample_games)

    return {
        'off_epa': float(off_epa),
        'pass_off_epa': float(pass_off_epa),
        'rush_off_epa': float(rush_off_epa),
        'sample_games': sample_games
    }


def calculate_all_team_defensive_epa(
    pbp_df: pd.DataFrame,
    weeks: int = 10,
    season: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate defensive EPA for all teams with regression to mean.

    Args:
        pbp_df: Play-by-play DataFrame
        weeks: Number of recent weeks to include
        season: Season to filter

    Returns:
        DataFrame with team, def_epa_allowed, pass_def_epa, rush_def_epa, sample_games
    """
    if season is not None:
        pbp_df = pbp_df[pbp_df['season'] == season].copy()

    teams = pbp_df['defteam'].dropna().unique()
    results = []

    for team in teams:
        epa_data = calculate_team_defensive_epa(pbp_df, team, weeks, season=None)
        results.append({
            'team': team,
            'def_epa_allowed': epa_data['def_epa_allowed'],
            'pass_def_epa': epa_data['pass_def_epa'],
            'rush_def_epa': epa_data['rush_def_epa'],
            'sample_games': epa_data['sample_games']
        })

    df = pd.DataFrame(results)
    logger.info(f"Calculated defensive EPA for {len(df)} teams with regression to mean")
    return df


def calculate_position_specific_defensive_epa(
    pbp_df: pd.DataFrame,
    opponent: str,
    position: str,
    weeks: int = 10,
    season: Optional[int] = None
) -> float:
    """
    Get opponent defensive EPA for a specific position WITH regression to mean.

    Args:
        pbp_df: Play-by-play DataFrame
        opponent: Opponent team abbreviation
        position: Player position ('QB', 'RB', 'WR', 'TE')
        weeks: Number of recent weeks
        season: Season to filter

    Returns:
        Defensive EPA (positive = bad defense, negative = good defense)
    """
    if season is not None:
        pbp_df = pbp_df[pbp_df['season'] == season].copy()

    # Filter to relevant weeks
    max_week = pbp_df['week'].max()
    recent_weeks = range(max(1, max_week - weeks + 1), max_week + 1)

    # Get position-specific defensive EPA
    if position in ['QB', 'WR', 'TE']:
        # Pass defense
        def_plays = pbp_df[
            (pbp_df['play_type'] == 'pass') &
            (pbp_df['defteam'] == opponent) &
            (pbp_df['week'].isin(recent_weeks))
        ]
    elif position == 'RB':
        # Rush defense (and passing to RBs)
        def_plays = pbp_df[
            (pbp_df['defteam'] == opponent) &
            (pbp_df['week'].isin(recent_weeks))
        ]
    else:
        return 0.0

    if len(def_plays) == 0:
        return 0.0

    sample_games = len(def_plays['game_id'].unique())
    raw_epa = def_plays['epa'].mean()

    # CRITICAL: Regress toward league mean
    regressed_epa = regress_epa_to_mean(raw_epa, sample_games)

    return float(regressed_epa)


def calculate_situational_epa(
    pbp_df: pd.DataFrame,
    team: str,
    situation: str = 'redzone',
    play_type: str = 'all',
    weeks: int = 10,
    season: Optional[int] = None,
    is_defense: bool = True
) -> float:
    """
    Calculate situational EPA for specific game contexts.

    Situations:
    - 'redzone': Inside opponent's 20-yard line
    - 'goalline': Inside opponent's 5-yard line
    - 'third_down': Third down conversions
    - 'two_minute': Last 2 minutes of half
    - 'fourth_quarter': Fourth quarter only

    Args:
        pbp_df: Play-by-play DataFrame
        team: Team abbreviation
        situation: Situation type (see above)
        play_type: 'pass', 'run', or 'all'
        weeks: Number of recent weeks to include
        season: Season to filter
        is_defense: If True, calculates defensive EPA. If False, offensive EPA.

    Returns:
        Situational EPA (regressed to mean)
    """
    if season is not None:
        pbp_df = pbp_df[pbp_df['season'] == season].copy()

    # Filter to relevant weeks
    max_week = pbp_df['week'].max()
    recent_weeks = range(max(1, max_week - weeks + 1), max_week + 1)

    # Start with team filter
    if is_defense:
        plays = pbp_df[
            (pbp_df['defteam'] == team) &
            (pbp_df['week'].isin(recent_weeks))
        ].copy()
    else:
        plays = pbp_df[
            (pbp_df['posteam'] == team) &
            (pbp_df['week'].isin(recent_weeks))
        ].copy()

    # Apply play type filter
    if play_type == 'pass':
        plays = plays[plays['play_type'] == 'pass']
    elif play_type == 'run':
        plays = plays[plays['play_type'] == 'run']
    elif play_type == 'all':
        plays = plays[plays['play_type'].isin(['pass', 'run'])]

    # Apply situation filter
    if situation == 'redzone':
        # Inside 20-yard line
        plays = plays[plays['yardline_100'] <= 20]
    elif situation == 'goalline':
        # Inside 5-yard line
        plays = plays[plays['yardline_100'] <= 5]
    elif situation == 'third_down':
        # Third down only
        plays = plays[plays['down'] == 3]
    elif situation == 'two_minute':
        # Last 2 minutes of half (either half)
        plays = plays[
            ((plays['qtr'].isin([2])) & (plays['half_seconds_remaining'] <= 120)) |
            ((plays['qtr'].isin([4])) & (plays['game_seconds_remaining'] <= 120))
        ]
    elif situation == 'fourth_quarter':
        # Fourth quarter only
        plays = plays[plays['qtr'] == 4]

    if len(plays) == 0:
        return 0.0

    # Calculate EPA
    sample_games = len(plays['game_id'].unique())
    raw_epa = plays['epa'].mean()

    # Regress to mean (more aggressive regression for situational stats)
    regressed_epa = regress_epa_to_mean(
        raw_epa,
        sample_games,
        regression_factor=0.6  # More regression for smaller samples
    )

    return float(regressed_epa)


def get_all_situational_epa_features(
    pbp_df: pd.DataFrame,
    team: str,
    position: str,
    weeks: int = 10,
    season: Optional[int] = None,
    is_defense: bool = True
) -> dict:
    """
    Get all relevant situational EPA features for a team/position.

    Args:
        pbp_df: Play-by-play DataFrame
        team: Team abbreviation
        position: Player position (QB, RB, WR, TE)
        weeks: Number of recent weeks
        season: Season to filter
        is_defense: If True, gets defensive EPA. If False, offensive EPA.

    Returns:
        Dict with situational EPA features
    """
    play_type = 'pass' if position in ['QB', 'WR', 'TE'] else 'all'

    features = {
        'redzone_epa': calculate_situational_epa(
            pbp_df, team, 'redzone', play_type, weeks, season, is_defense
        ),
        'third_down_epa': calculate_situational_epa(
            pbp_df, team, 'third_down', play_type, weeks, season, is_defense
        ),
        'two_minute_epa': calculate_situational_epa(
            pbp_df, team, 'two_minute', play_type, weeks, season, is_defense
        ),
    }

    # Add goalline for RBs (TD predictions)
    if position == 'RB':
        features['goalline_epa'] = calculate_situational_epa(
            pbp_df, team, 'goalline', 'run', weeks, season, is_defense
        )

    return features
