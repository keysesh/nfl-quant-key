#!/usr/bin/env python3
"""
Defensive Stats Integration Helper

Integrates defensive statistics into player prop predictions.
This fixes the issue where opponent defense is currently defaulting to 0.0.

IMPORTANT: Applies regression to mean for defensive EPA to account for sample size variance.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

from nfl_quant.utils.epa_utils import regress_epa_to_mean

logger = logging.getLogger(__name__)


def get_defensive_epa_for_player(
    opponent: str,
    position: str,
    week: int,
    pbp_path: Optional[Path] = None,
    season: Optional[int] = None
) -> float:
    """
    Get opponent defensive EPA for a specific position.

    Args:
        opponent: Opponent team abbreviation (e.g., 'WAS')
        position: Player position ('QB', 'RB', 'WR', 'TE')
        week: Current week
        pbp_path: Path to play-by-play parquet file
        season: Season to use (defaults to current season)

    Returns:
        Defensive EPA (negative = good defense, positive = bad defense)
    """
    if pbp_path is None:
        if season is None:
            from nfl_quant.utils.season_utils import get_current_season
            season = get_current_season()
        pbp_path = Path(f'data/processed/pbp_{season}.parquet')

    if not pbp_path.exists():
        logger.warning(f"PBP file not found: {pbp_path}")
        return 0.0  # Default neutral

    try:
        # Load PBP data
        pbp_df = pd.read_parquet(pbp_path)

        # Use trailing 4 weeks for defensive stats
        start_week = max(1, week - 4)
        end_week = week - 1

        if start_week > end_week:
            # Not enough data yet, use league average
            return 0.0

        # Filter to relevant weeks
        relevant_pbp = pbp_df[
            (pbp_df['week'] >= start_week) &
            (pbp_df['week'] <= end_week)
        ]

        # Get position-specific defensive EPA
        if position in ['QB', 'WR', 'TE']:
            # Pass defense
            pass_def = relevant_pbp[
                (relevant_pbp['play_type'] == 'pass') &
                (relevant_pbp['defteam'] == opponent)
            ]

            if len(pass_def) == 0:
                logger.debug(f"No pass defense data for {opponent}")
                return 0.0

            # Average EPA per pass play (negative = good defense)
            raw_pass_epa = pass_def['epa'].mean()
            # CRITICAL: Regress to mean to account for sample size variance
            sample_games = len(pass_def['game_id'].unique())
            regressed_epa = regress_epa_to_mean(raw_pass_epa, sample_games)
            return float(regressed_epa)

        elif position == 'RB':
            # Rush defense
            rush_def = relevant_pbp[
                (relevant_pbp['play_type'] == 'run') &
                (relevant_pbp['defteam'] == opponent)
            ]

            if len(rush_def) == 0:
                logger.debug(f"No rush defense data for {opponent}")
                return 0.0

            # Average EPA per rush play (negative = good defense)
            raw_rush_epa = rush_def['epa'].mean()
            # CRITICAL: Regress to mean to account for sample size variance
            sample_games = len(rush_def['game_id'].unique())
            regressed_epa = regress_epa_to_mean(raw_rush_epa, sample_games)
            return float(regressed_epa)

        else:
            return 0.0

    except Exception as e:
        logger.warning(f"Error calculating defensive EPA for {opponent}: {e}")
        return 0.0


def get_defensive_stats_batch(
    players: pd.DataFrame,
    week: int,
    pbp_path: Optional[Path] = None,
    season: Optional[int] = None
) -> Dict[str, float]:
    """
    Get defensive stats for multiple players at once (more efficient).

    Args:
        players: DataFrame with columns ['opponent', 'position']
        week: Current week
        pbp_path: Path to play-by-play parquet file
        season: Season to use (defaults to current season)

    Returns:
        Dictionary mapping (opponent, position) -> defensive EPA
    """
    if pbp_path is None:
        if season is None:
            from nfl_quant.utils.season_utils import get_current_season
            season = get_current_season()
        pbp_path = Path(f'data/processed/pbp_{season}.parquet')

    if not pbp_path.exists():
        logger.warning(f"PBP file not found: {pbp_path}")
        return {}

    try:
        # Load PBP data once
        pbp_df = pd.read_parquet(pbp_path)

        # Use trailing 4 weeks
        start_week = max(1, week - 4)
        end_week = week - 1

        if start_week > end_week:
            return {}

        relevant_pbp = pbp_df[
            (pbp_df['week'] >= start_week) &
            (pbp_df['week'] <= end_week)
        ]

        defensive_stats = {}

        # Get unique opponent/position combinations
        unique_combos = players[['opponent', 'position']].drop_duplicates()

        for _, row in unique_combos.iterrows():
            opponent = row['opponent']
            position = row['position']

            if position in ['QB', 'WR', 'TE']:
                # Pass defense
                pass_def = relevant_pbp[
                    (relevant_pbp['play_type'] == 'pass') &
                    (relevant_pbp['defteam'] == opponent)
                ]

                if len(pass_def) > 0:
                    raw_epa = float(pass_def['epa'].mean())
                    # CRITICAL: Regress to mean
                    sample_games = len(pass_def['game_id'].unique())
                    regressed_epa = regress_epa_to_mean(raw_epa, sample_games)
                    defensive_stats[(opponent, position)] = regressed_epa
                else:
                    defensive_stats[(opponent, position)] = 0.0

            elif position == 'RB':
                # Rush defense
                rush_def = relevant_pbp[
                    (relevant_pbp['play_type'] == 'run') &
                    (relevant_pbp['defteam'] == opponent)
                ]

                if len(rush_def) > 0:
                    raw_epa = float(rush_def['epa'].mean())
                    # CRITICAL: Regress to mean
                    sample_games = len(rush_def['game_id'].unique())
                    regressed_epa = regress_epa_to_mean(raw_epa, sample_games)
                    defensive_stats[(opponent, position)] = regressed_epa
                else:
                    defensive_stats[(opponent, position)] = 0.0

            else:
                defensive_stats[(opponent, position)] = 0.0

        return defensive_stats

    except Exception as e:
        logger.warning(f"Error calculating batch defensive stats: {e}")
        return {}


def integrate_defensive_stats_into_predictions(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    week: int
) -> pd.DataFrame:
    """
    Integrate defensive stats into prediction DataFrame.

    Adds columns:
    - opponent_def_epa: Defensive EPA for opponent
    - opponent_def_epa_vs_position: Position-specific defensive EPA

    Args:
        predictions_df: DataFrame with predictions
        odds_df: DataFrame with odds (contains opponent info)
        week: Current week

    Returns:
        DataFrame with defensive stats added
    """
    logger.info("Integrating defensive statistics...")

    # Merge opponent info from odds if available
    has_opponent = 'opponent' not in predictions_df.columns
    odds_has_opponent = 'opponent' in odds_df.columns
    if has_opponent and odds_has_opponent:
        # Try to match by player name
        merged = predictions_df.merge(
            odds_df[['player_name', 'opponent']].drop_duplicates(),
            on='player_name',
            how='left'
        )
        predictions_df['opponent'] = merged['opponent']

    # Get defensive stats for all unique opponent/position combos
    has_opponent_col = 'opponent' in predictions_df.columns
    has_position_col = 'position' in predictions_df.columns
    if has_opponent_col and has_position_col:
        defensive_stats = get_defensive_stats_batch(
            predictions_df[['opponent', 'position']].drop_duplicates(),
            week
        )

        # Apply defensive stats
        predictions_df['opponent_def_epa_vs_position'] = predictions_df.apply(
            lambda row: defensive_stats.get(
                (row.get('opponent', ''), row.get('position', '')),
                0.0
            ),
            axis=1
        )

        stats_count = len(defensive_stats)
        logger.info(
            f"   ✅ Added defensive stats for {stats_count} "
            f"opponent/position combinations"
        )
    else:
        logger.warning(
            "   ⚠️  Missing 'opponent' or 'position' columns - "
            "cannot add defensive stats"
        )
        predictions_df['opponent_def_epa_vs_position'] = 0.0

    return predictions_df
