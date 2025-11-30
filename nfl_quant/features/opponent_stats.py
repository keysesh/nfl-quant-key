#!/usr/bin/env python3
"""
NFL QUANT - Opponent-Aware Statistics Module

Calculates player performance statistics against specific opponents
with TIME-AWARE lookback to prevent data leakage.

Key Features:
- vs_opponent_divergence: How much player over/under-performs vs specific opponent
- vs_opponent_games: Number of prior games vs this opponent (confidence metric)
- LVT_x_vs_opponent: Interaction feature for V13 model

CRITICAL: All calculations use shift(1) to ensure we only use prior games.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def calculate_vs_opponent_stats_time_aware(
    stats: pd.DataFrame,
    stat_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    Calculate player performance vs specific opponents with TIME-AWARE lookback.

    This function computes rolling statistics of how a player performs against
    each opponent, using ONLY games that occurred BEFORE the current game.

    Args:
        stats: DataFrame with player stats including:
            - player_norm: Normalized player name
            - opponent_team: Opponent team abbreviation
            - season: NFL season year
            - week: NFL week number
            - Various stat columns (receptions, receiving_yards, etc.)
        stat_cols: List of stat columns to calculate (defaults to common stats)

    Returns:
        DataFrame with additional columns:
            - vs_opp_avg_{stat}: Rolling mean vs this opponent (prior games only)
            - vs_opp_games_{stat}: Count of prior games vs this opponent

    CRITICAL: Uses shift(1) to prevent data leakage (no future information)
    """
    if stat_cols is None:
        stat_cols = ['receptions', 'receiving_yards', 'rushing_yards',
                     'passing_yards', 'completions', 'passing_completions']

    # Filter to only stat columns that exist
    stat_cols = [c for c in stat_cols if c in stats.columns]

    if 'opponent_team' not in stats.columns:
        logger.warning("opponent_team column not found - cannot calculate opponent stats")
        return stats

    logger.info("Calculating TIME-AWARE vs-opponent statistics...")

    # Sort by player and time
    stats = stats.sort_values(['player_norm', 'season', 'week']).copy()

    # Create global week for proper ordering
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    result_rows = []

    for (player, opp), group in stats.groupby(['player_norm', 'opponent_team']):
        group = group.sort_values('global_week').copy()

        for stat_col in stat_cols:
            if stat_col not in group.columns:
                continue

            # Calculate expanding mean (only prior games)
            # shift(1) ensures we don't include current game
            group[f'vs_opp_avg_{stat_col}'] = (
                group[stat_col].expanding(min_periods=1).mean().shift(1)
            )

            # Count of prior games vs this opponent
            group[f'vs_opp_games_{stat_col}'] = (
                group[stat_col].expanding().count().shift(1).fillna(0)
            )

        result_rows.append(group)

    if result_rows:
        result = pd.concat(result_rows, ignore_index=True)
        logger.info(f"  Calculated opponent stats for {len(result)} player-game rows")
        return result

    return stats


def calculate_opponent_divergence(
    df: pd.DataFrame,
    market: str,
    trailing_col: str
) -> pd.DataFrame:
    """
    Calculate how much player diverges from their average when facing specific opponent.

    Args:
        df: DataFrame with vs_opp_avg columns and trailing stats
        market: Market type (player_receptions, player_rush_yds, etc.)
        trailing_col: Column name for player's overall trailing average

    Returns:
        DataFrame with:
            - vs_opponent_divergence: (vs_opp_avg - trailing) / trailing
              Positive = player overperforms vs this opponent
              Negative = player underperforms vs this opponent
            - vs_opponent_games: Number of prior matchups (for confidence)
            - vs_opponent_confidence: Scaled confidence (0-1) based on sample size
    """
    # Map market to stat column
    market_to_stat = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        'player_pass_completions': 'completions',
    }

    stat_col = market_to_stat.get(market)
    if stat_col is None:
        logger.warning(f"Unknown market {market} for opponent divergence")
        df['vs_opponent_divergence'] = 0.0
        df['vs_opponent_games'] = 0
        df['vs_opponent_confidence'] = 0.0
        return df

    vs_opp_col = f'vs_opp_avg_{stat_col}'
    vs_opp_games_col = f'vs_opp_games_{stat_col}'

    if vs_opp_col not in df.columns:
        logger.warning(f"Column {vs_opp_col} not found - no opponent stats available")
        df['vs_opponent_divergence'] = 0.0
        df['vs_opponent_games'] = 0
        df['vs_opponent_confidence'] = 0.0
        return df

    # Calculate divergence: (vs_opp_avg - trailing) / trailing
    # Clamp to [-1, +1] range
    df['vs_opponent_divergence'] = (
        (df[vs_opp_col] - df[trailing_col]) / df[trailing_col].clip(lower=0.1)
    ).clip(lower=-1, upper=1).fillna(0)

    # Get games count
    if vs_opp_games_col in df.columns:
        df['vs_opponent_games'] = df[vs_opp_games_col].fillna(0)
    else:
        df['vs_opponent_games'] = 0

    # Confidence: need 2+ games for signal, scales to 1.0 at 4+ games
    df['vs_opponent_confidence'] = (
        (df['vs_opponent_games'] - 1).clip(lower=0) / 3
    ).clip(upper=1)

    return df


def create_opponent_interaction_features(
    df: pd.DataFrame,
    market: str
) -> pd.DataFrame:
    """
    Create interaction features between LVT and opponent divergence.

    Args:
        df: DataFrame with line_vs_trailing and vs_opponent_divergence
        market: Market type for logging

    Returns:
        DataFrame with:
            - LVT_x_vs_opponent: Key interaction feature
              Negative when:
                - LVT > 0 (line above trailing, suggesting UNDER)
                - BUT divergence > 0 (player overperforms vs opponent)
              This REDUCES the UNDER signal when opponent history suggests OVER
    """
    if 'line_vs_trailing' not in df.columns:
        logger.warning("line_vs_trailing not found - cannot create opponent interactions")
        df['LVT_x_vs_opponent'] = 0.0
        df['trailing_opp_adjusted'] = df.get('trailing_stat', 0)
        df['line_vs_trailing_opp_adj'] = df.get('line_vs_trailing', 0)
        return df

    # LVT_x_vs_opponent: Modulates LVT based on opponent history
    # If player OVERPERFORMS vs opponent (divergence > 0):
    #   - LVT > 0 (line above trailing) gets reduced (less confident in UNDER)
    #   - This captures cases like "Lamar goes UNDER 64% overall but OVER vs CIN"
    df['LVT_x_vs_opponent'] = (
        df['line_vs_trailing'] *
        (-df['vs_opponent_divergence'].fillna(0)) *
        df['vs_opponent_confidence'].fillna(0)
    )

    # Opponent-adjusted trailing
    trailing_col = df.columns[df.columns.str.startswith('trailing_')].tolist()
    if trailing_col:
        trailing_col = trailing_col[0]
        df['trailing_opp_adjusted'] = (
            df[trailing_col] * (1 + df['vs_opponent_divergence'].fillna(0))
        )
        df['line_vs_trailing_opp_adj'] = (
            df['line'] - df['trailing_opp_adjusted']
        )
    else:
        df['trailing_opp_adjusted'] = 0
        df['line_vs_trailing_opp_adj'] = df['line_vs_trailing']

    return df


def get_opponent_conflict_flag(
    row: pd.Series,
    divergence_threshold: float = 0.15
) -> Tuple[bool, str]:
    """
    Check if there's a conflict between model signal and opponent history.

    Args:
        row: DataFrame row with pick, vs_opponent_divergence, vs_opponent_games
        divergence_threshold: Minimum divergence to flag (default 15%)

    Returns:
        Tuple of (is_conflict, reason_string)

    Example:
        If model says UNDER but player averages 25% more vs this opponent,
        returns (True, "Player +25% vs opponent (3 games)")
    """
    divergence = row.get('vs_opponent_divergence', 0)
    games = row.get('vs_opponent_games', 0)
    pick = str(row.get('pick', '')).lower()

    # Need at least 2 prior games for reliable signal
    if games < 2:
        return (False, "")

    is_under = 'under' in pick
    is_over = 'over' in pick

    # Conflict: Model says UNDER but player overperforms vs this opponent
    if is_under and divergence > divergence_threshold:
        return (True, f"⚠️ Player +{divergence*100:.0f}% vs opponent ({int(games)} games)")

    # Conflict: Model says OVER but player underperforms vs this opponent
    if is_over and divergence < -divergence_threshold:
        return (True, f"⚠️ Player {divergence*100:.0f}% vs opponent ({int(games)} games)")

    return (False, "")


def add_opponent_features_to_predictions(
    predictions_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    week: int,
    season: int = 2025
) -> pd.DataFrame:
    """
    Add opponent-aware features to model predictions for a specific week.

    This is the main entry point for integrating opponent stats into predictions.

    Args:
        predictions_df: DataFrame from model_predictions_weekX.csv
        stats_df: Player stats with opponent_team column
        week: Current week number
        season: Current season

    Returns:
        predictions_df with additional opponent columns
    """
    logger.info(f"Adding opponent features for Week {week}...")

    # Calculate opponent stats from historical data
    stats_with_opp = calculate_vs_opponent_stats_time_aware(stats_df)

    # For each prediction, find their upcoming opponent and historical stats
    for idx, row in predictions_df.iterrows():
        player_norm = row.get('player_norm', row.get('player_name', '').lower())
        opponent = row.get('opponent', row.get('opponent_team', ''))

        if not opponent:
            continue

        # Look up historical performance vs this opponent
        player_vs_opp = stats_with_opp[
            (stats_with_opp['player_norm'] == player_norm) &
            (stats_with_opp['opponent_team'] == opponent) &
            (stats_with_opp['global_week'] < (season - 2023) * 18 + week)
        ]

        if len(player_vs_opp) == 0:
            continue

        # Get most recent prior game vs this opponent
        latest = player_vs_opp.sort_values('global_week').iloc[-1]

        # Add opponent features to predictions
        for col in latest.index:
            if col.startswith('vs_opp_'):
                predictions_df.loc[idx, col] = latest[col]

    return predictions_df
