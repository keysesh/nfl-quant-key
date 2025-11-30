"""
Prediction Helpers - Bridge between FeatureEngine and prediction scripts.

This module provides helper functions that prediction scripts can use
to calculate trailing stats consistently with training.

Usage:
    from nfl_quant.features.prediction_helpers import calculate_player_trailing_stats

    trailing = calculate_player_trailing_stats(player_df)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from nfl_quant.features.core import get_feature_engine

logger = logging.getLogger(__name__)


def calculate_player_trailing_stats(
    player_df: pd.DataFrame,
    use_ewma: bool = True,
    span: int = 4
) -> Dict[str, float]:
    """
    Calculate trailing stats for a single player's historical data.

    THIS REPLACES inline EWMA calculations in prediction scripts.

    Args:
        player_df: DataFrame with player's weekly stats (sorted by week)
        use_ewma: If True, use EWMA. If False, use simple mean.
        span: EWMA span (default 4)

    Returns:
        Dictionary with trailing stats:
        - avg_targets, avg_receptions, avg_rec_yards
        - avg_carries, avg_rush_yards, avg_pass_yards
    """
    if len(player_df) == 0:
        return {
            'avg_targets': 0.0,
            'avg_receptions': 0.0,
            'avg_rec_yards': 0.0,
            'avg_carries': 0.0,
            'avg_rush_yards': 0.0,
            'avg_pass_yards': 0.0,
        }

    # Sort by week to ensure proper calculation
    df = player_df.sort_values('week')

    result = {}

    stat_mappings = [
        ('targets', 'avg_targets'),
        ('receptions', 'avg_receptions'),
        ('receiving_yards', 'avg_rec_yards'),
        ('carries', 'avg_carries'),
        ('rushing_yards', 'avg_rush_yards'),
        ('passing_yards', 'avg_pass_yards'),
    ]

    for col, result_key in stat_mappings:
        if col in df.columns:
            if use_ewma:
                # EWMA matching training: span=4, min_periods=1
                # Get the LAST value (most recent weighted average)
                result[result_key] = float(
                    df[col].ewm(span=span, min_periods=1).mean().iloc[-1]
                )
            else:
                result[result_key] = float(df[col].mean())
        else:
            result[result_key] = 0.0

    return result


def calculate_batch_trailing_stats(
    stats_df: pd.DataFrame,
    current_week: int,
    player_col: str = 'player_display_name'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate trailing stats for all players in batch.

    This is more efficient than calling calculate_player_trailing_stats
    for each player individually.

    Args:
        stats_df: DataFrame with all player stats
        current_week: Current week number (excludes this week)
        player_col: Column identifying players

    Returns:
        Dictionary mapping player_name to their trailing stats
    """
    # Filter to weeks before current
    prior_stats = stats_df[stats_df['week'] < current_week].copy()

    if len(prior_stats) == 0:
        return {}

    result = {}

    # Group by player
    for player_name, player_df in prior_stats.groupby(player_col):
        result[player_name] = calculate_player_trailing_stats(player_df)

    return result


def get_player_trailing_for_prediction(
    player_name: str,
    team: str,
    position: str,
    season: int,
    week: int,
    stats_df: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Get trailing stats for a specific player for prediction.

    This is the main entry point for prediction scripts.

    Args:
        player_name: Player name
        team: Team abbreviation
        position: Player position
        season: Season year
        week: Week number (predictions for this week)
        stats_df: Optional pre-loaded stats DataFrame

    Returns:
        Dictionary with trailing stats
    """
    from nfl_quant.utils.player_names import normalize_player_name

    if stats_df is None:
        # Load from NFLverse
        engine = get_feature_engine()
        stats_df = engine._load_weekly_stats()

    # Normalize player name
    player_norm = normalize_player_name(player_name)

    # Filter to player's prior weeks this season
    if 'player_norm' not in stats_df.columns:
        stats_df = stats_df.copy()
        if 'player_display_name' in stats_df.columns:
            stats_df['player_norm'] = stats_df['player_display_name'].apply(normalize_player_name)
        elif 'player_name' in stats_df.columns:
            stats_df['player_norm'] = stats_df['player_name'].apply(normalize_player_name)

    player_stats = stats_df[
        (stats_df['player_norm'] == player_norm) &
        (stats_df['season'] == season) &
        (stats_df['week'] < week)
    ]

    if len(player_stats) == 0:
        logger.debug(f"No prior stats for {player_name} in {season} week {week}")
        return {
            'avg_targets': 0.0,
            'avg_receptions': 0.0,
            'avg_rec_yards': 0.0,
            'avg_carries': 0.0,
            'avg_rush_yards': 0.0,
            'avg_pass_yards': 0.0,
        }

    return calculate_player_trailing_stats(player_stats)


def calculate_v12_prediction_features(
    player_name: str,
    line: float,
    trailing_stat: float,
    market: str,
    historical_odds: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Calculate V12 features for a prediction.

    This provides the same features as training, ensuring consistency.

    Args:
        player_name: Player name
        line: Vegas line for this prop
        trailing_stat: Player's trailing average for this stat
        market: Market type (player_receptions, etc.)
        historical_odds: Optional historical odds for betting features

    Returns:
        Dictionary with V12 features
    """
    engine = get_feature_engine()

    # Calculate LVT
    lvt = engine.calculate_line_vs_trailing(line, trailing_stat, method='difference')

    # Basic features always available
    features = {
        'line_vs_trailing': lvt,
        'line_level': line,
        'line_in_sweet_spot': 1.0 if 3.5 <= line <= 7.5 else 0.0,
    }

    # If we have historical odds, calculate betting features
    if historical_odds is not None and len(historical_odds) > 0:
        from nfl_quant.utils.player_names import normalize_player_name
        player_norm = normalize_player_name(player_name)

        # Need a target global_week - use max + 1 (current week)
        max_week = historical_odds['global_week'].max()
        target_week = max_week + 1

        player_under_rate = engine.calculate_player_under_rate(
            historical_odds, player_norm, target_week
        )
        player_bias = engine.calculate_player_bias(
            historical_odds, player_norm, target_week
        )
        market_under_rate = engine.calculate_market_under_rate(
            historical_odds, target_week
        )

        # Add all V12 features
        features.update({
            'player_under_rate': player_under_rate,
            'player_bias': player_bias,
            'market_under_rate': market_under_rate,
            'LVT_x_player_tendency': lvt * (player_under_rate - 0.5),
            'LVT_x_player_bias': lvt * player_bias,
            'LVT_x_regime': lvt * (market_under_rate - 0.5),
            'LVT_in_sweet_spot': lvt * features['line_in_sweet_spot'],
            'market_bias_strength': abs(market_under_rate - 0.5) * 2,
            'player_market_aligned': (player_under_rate - 0.5) * (market_under_rate - 0.5),
        })
    else:
        # Use neutral defaults
        features.update({
            'player_under_rate': 0.5,
            'player_bias': 0.0,
            'market_under_rate': 0.5,
            'LVT_x_player_tendency': 0.0,
            'LVT_x_player_bias': 0.0,
            'LVT_x_regime': 0.0,
            'LVT_in_sweet_spot': lvt * features['line_in_sweet_spot'],
            'market_bias_strength': 0.0,
            'player_market_aligned': 0.0,
        })

    return features
