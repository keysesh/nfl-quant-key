"""
Opponent Context Feature Builder for V23.

Extracts opponent defense features for ALL positions:
- Pass defense metrics (for QBs, WRs, TEs)
- Rush defense metrics (for RBs)
- Game context (spread, total, implied score)

This module fills the gap where opponent features were HARDCODED to league averages
instead of using actual opponent-specific data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def calculate_opponent_trailing_defense(
    weekly_stats: pd.DataFrame,
    n_weeks: int = 4
) -> pd.DataFrame:
    """
    Calculate trailing opponent defense stats from game logs.

    Uses weekly_stats to calculate what each team's defense allowed in recent games.
    This data already has 'opponent_team' column we can group by.

    Returns DataFrame with columns:
    - team (the defense), week, season
    - opp_pass_yds_allowed_trailing (pass yards allowed by this defense)
    - opp_rush_yds_allowed_trailing (rush yards allowed by this defense)
    - opp_receptions_allowed_trailing (receptions allowed by this defense)
    - opp_pass_def_vs_avg (z-score relative to league average)
    - opp_rush_def_vs_avg (z-score relative to league average)
    """
    logger.info("Calculating opponent trailing defense stats...")

    # Aggregate offensive stats by opponent (what defense allowed)
    # Group by opponent_team to get what that DEFENSE allowed
    offense_cols = [
        'passing_yards', 'rushing_yards', 'receptions',
        'receiving_yards', 'passing_tds', 'rushing_tds',
        'completions', 'attempts', 'targets', 'carries'
    ]

    # Filter to columns that exist
    available_cols = [c for c in offense_cols if c in weekly_stats.columns]

    if 'opponent_team' not in weekly_stats.columns:
        logger.error("weekly_stats missing opponent_team column!")
        return pd.DataFrame()

    # Aggregate by opponent and game
    defense_allowed = weekly_stats.groupby(
        ['opponent_team', 'week', 'season']
    )[available_cols].sum().reset_index()

    # Rename opponent_team to team (this is the defense)
    defense_allowed = defense_allowed.rename(columns={'opponent_team': 'team'})

    # Add column aliases
    if 'passing_yards' in defense_allowed.columns:
        defense_allowed['pass_yds_allowed'] = defense_allowed['passing_yards']
    if 'rushing_yards' in defense_allowed.columns:
        defense_allowed['rush_yds_allowed'] = defense_allowed['rushing_yards']
    if 'receptions' in defense_allowed.columns:
        defense_allowed['receptions_allowed'] = defense_allowed['receptions']
    if 'receiving_yards' in defense_allowed.columns:
        defense_allowed['rec_yds_allowed'] = defense_allowed['receiving_yards']

    # Sort for rolling calculation
    defense_allowed = defense_allowed.sort_values(['team', 'season', 'week'])

    # Calculate trailing averages (shift by 1 to avoid leakage)
    trailing_cols = [
        'pass_yds_allowed', 'rush_yds_allowed', 'receptions_allowed', 'rec_yds_allowed'
    ]

    for col in trailing_cols:
        if col in defense_allowed.columns:
            defense_allowed[f'{col}_trailing'] = (
                defense_allowed.groupby('team')[col]
                .transform(lambda x: x.shift(1).ewm(span=n_weeks, min_periods=1).mean())
            )

    # Calculate vs league average (for comparison)
    for col in ['pass_yds_allowed', 'rush_yds_allowed']:
        if col in defense_allowed.columns:
            trailing_col = f'{col}_trailing'
            if trailing_col in defense_allowed.columns:
                league_avg = defense_allowed[col].mean()
                league_std = defense_allowed[col].std()
                if league_std > 0:
                    defense_allowed[f'{col.replace("_allowed", "_def_vs_avg")}'] = (
                        (defense_allowed[trailing_col] - league_avg) / league_std
                    )
                else:
                    defense_allowed[f'{col.replace("_allowed", "_def_vs_avg")}'] = 0

    logger.info(f"Calculated defense stats for {defense_allowed['team'].nunique()} teams")

    return defense_allowed


def add_opponent_features_to_data(
    df: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    week_col: str = 'week',
    season_col: str = 'season',
    opponent_col: str = 'opponent'
) -> pd.DataFrame:
    """
    Add opponent context features to a DataFrame.

    Joins opponent defense trailing stats based on week/season/opponent.

    Args:
        df: DataFrame to add features to (e.g., odds or training data)
        weekly_stats: Weekly stats with opponent_team column
        week_col: Column name for week in df
        season_col: Column name for season in df
        opponent_col: Column name for opponent team in df

    Returns:
        DataFrame with opponent features added
    """
    if opponent_col not in df.columns:
        logger.warning(f"Missing {opponent_col} column, cannot add opponent features")
        return df

    # Calculate opponent defense stats
    opp_defense = calculate_opponent_trailing_defense(weekly_stats)

    if len(opp_defense) == 0:
        logger.warning("Could not calculate opponent defense stats")
        return df

    # Select columns to join
    join_cols = ['team', 'week', 'season']
    feature_cols = [c for c in opp_defense.columns if 'trailing' in c or 'vs_avg' in c]

    opp_features = opp_defense[join_cols + feature_cols].copy()

    # Rename for clarity (these are OPPONENT defense stats)
    rename_map = {c: f'opp_{c}' for c in feature_cols if not c.startswith('opp_')}
    opp_features = opp_features.rename(columns=rename_map)

    # Also rename 'team' to match the opponent column for joining
    opp_features = opp_features.rename(columns={'team': opponent_col})

    # Merge
    df_with_opp = df.merge(
        opp_features,
        on=[opponent_col, week_col, season_col],
        how='left'
    )

    # Fill missing with league averages
    for col in df_with_opp.columns:
        if 'opp_' in col and 'trailing' in col:
            df_with_opp[col] = df_with_opp[col].fillna(df_with_opp[col].median())
        elif 'vs_avg' in col:
            df_with_opp[col] = df_with_opp[col].fillna(0)  # Neutral

    logger.info(f"Added {len(feature_cols)} opponent features")

    return df_with_opp


def get_team_defense_epa() -> pd.DataFrame:
    """
    Load pre-computed team defense EPA data.

    Returns DataFrame with team, def_epa_allowed, defensive_plays.
    """
    epa_path = PROJECT_ROOT / 'data' / 'nflverse' / 'team_defensive_epa.parquet'

    if epa_path.exists():
        return pd.read_parquet(epa_path)
    else:
        logger.warning(f"Team defense EPA file not found at {epa_path}")
        return pd.DataFrame()


def add_defense_epa_feature(
    df: pd.DataFrame,
    opponent_col: str = 'opponent'
) -> pd.DataFrame:
    """
    Add team defense EPA feature from pre-computed file.

    IMPORTANT: Does NOT fill missing values with defaults.
    XGBoost handles NaN natively - let it learn when data is missing.

    Args:
        df: DataFrame with opponent column
        opponent_col: Name of opponent team column

    Returns:
        DataFrame with opp_def_epa column added (NaN where unavailable)
    """
    # Initialize as NaN (not 0.0!)
    if 'opp_def_epa' not in df.columns:
        df['opp_def_epa'] = np.nan

    def_epa = get_team_defense_epa()

    if len(def_epa) == 0:
        # Return with NaN, not 0.0
        return df

    if opponent_col not in df.columns:
        return df

    # Rename for join
    def_epa = def_epa.rename(columns={'team': opponent_col, 'def_epa_allowed': 'opp_def_epa'})

    # Only keep opponent column and the EPA column
    join_cols = [opponent_col, 'opp_def_epa']
    available_cols = [c for c in join_cols if c in def_epa.columns]

    if len(available_cols) < 2:
        return df

    df = df.merge(
        def_epa[available_cols],
        on=opponent_col,
        how='left',
        suffixes=('_old', '')
    )

    # If there was an old column, drop it
    if 'opp_def_epa_old' in df.columns:
        df = df.drop(columns=['opp_def_epa_old'])

    # DO NOT fill missing - let XGBoost handle NaN natively

    return df


def get_market_specific_opponent_adjustment(
    market: str,
    opp_pass_def: float,
    opp_rush_def: float
) -> float:
    """
    Get the appropriate opponent defense adjustment for a specific market.

    Args:
        market: Market type (player_pass_yds, player_rush_yds, etc.)
        opp_pass_def: Opponent pass defense vs average (-1 to +1)
        opp_rush_def: Opponent rush defense vs average (-1 to +1)

    Returns:
        Adjustment factor (-1 to +1 scale, positive = good matchup for player)
    """
    MARKET_DEFENSE_MAP = {
        # Pass-based markets use pass defense
        'player_pass_yds': opp_pass_def,
        'player_receptions': opp_pass_def,
        'player_reception_yds': opp_pass_def,
        'player_pass_completions': opp_pass_def,
        'player_pass_attempts': opp_pass_def,
        # Rush-based markets use rush defense
        'player_rush_yds': opp_rush_def,
        'player_rush_attempts': opp_rush_def,
        # Mixed markets
        'player_anytime_td': (opp_pass_def + opp_rush_def) / 2,
        'player_pass_tds': opp_pass_def,
        'player_rush_tds': opp_rush_def,
    }

    return MARKET_DEFENSE_MAP.get(market, 0.0)


# New features for V23
# Note: Column names must match what calculate_opponent_trailing_defense produces
V23_OPPONENT_FEATURES = [
    'opp_pass_yds_allowed_trailing',
    'opp_rush_yds_allowed_trailing',
    'opp_receptions_allowed_trailing',
    'opp_rec_yds_allowed_trailing',
    'opp_pass_yds_def_vs_avg',  # z-score vs league average
    'opp_rush_yds_def_vs_avg',  # z-score vs league average
    'opp_def_epa',
]
