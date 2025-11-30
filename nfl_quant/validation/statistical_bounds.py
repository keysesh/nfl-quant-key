#!/usr/bin/env python3
"""
Statistical Bounds Validation

Validates projections against historical performance to prevent impossible predictions.
Per framework rules (CLAUDE.md): Projections >3σ from historical mean must be flagged/capped.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_player_historical_stats(
    player_name: str,
    position: str,
    stat_columns: list = None
) -> pd.DataFrame:
    """
    Load historical stats for a player.

    Args:
        player_name: Player name
        position: Position (QB, RB, WR, TE)
        stat_columns: Columns to load (default: all relevant for position)

    Returns:
        DataFrame with player's historical game stats
    """
    weekly_path = PROJECT_ROOT / 'data/nflverse/weekly_stats.parquet'
    if not weekly_path.exists():
        logger.warning(f"Weekly stats not found at {weekly_path}")
        return pd.DataFrame()

    weekly = pd.read_parquet(weekly_path)

    # Filter to player
    player_data = weekly[
        (weekly['player_display_name'].str.contains(player_name.split()[0], case=False, na=False)) &
        (weekly['player_display_name'].str.contains(player_name.split()[-1], case=False, na=False)) &
        (weekly['position'] == position)
    ].copy()

    if stat_columns:
        available_cols = [c for c in stat_columns if c in player_data.columns]
        return player_data[available_cols + ['season', 'week']]

    return player_data


def validate_projection(
    player_name: str,
    position: str,
    stat_name: str,
    projection: float,
    sigma_threshold: float = 3.0,
    min_games: int = 5
) -> Dict[str, any]:
    """
    Validate a projection against historical performance.

    Args:
        player_name: Player name
        position: Position
        stat_name: Stat being projected (e.g., 'receptions', 'rushing_yards')
        projection: Projected value
        sigma_threshold: Z-score threshold (default 3.0σ per framework)
        min_games: Minimum historical games required

    Returns:
        Dict with validation results:
        - is_valid: bool
        - z_score: float (how many σ from mean)
        - career_mean: float
        - career_max: float
        - recommendation: str (ACCEPT/FLAG/CAP)
        - capped_value: float (if CAP recommended)
    """
    result = {
        'is_valid': True,
        'z_score': 0.0,
        'career_mean': 0.0,
        'career_std': 0.0,
        'career_max': 0.0,
        'career_games': 0,
        'recommendation': 'ACCEPT',
        'capped_value': projection,
        'message': ''
    }

    # Load historical data
    historical = load_player_historical_stats(player_name, position, [stat_name])

    if historical.empty or stat_name not in historical.columns:
        result['message'] = f"No historical data for {player_name} {stat_name}"
        logger.debug(result['message'])
        return result

    stats = historical[stat_name].dropna()

    if len(stats) < min_games:
        result['message'] = f"Insufficient history ({len(stats)} games, need {min_games})"
        logger.debug(result['message'])
        return result

    # Calculate historical statistics
    result['career_mean'] = stats.mean()
    result['career_std'] = stats.std()
    result['career_max'] = stats.max()
    result['career_games'] = len(stats)

    # Calculate Z-score
    if result['career_std'] > 0:
        result['z_score'] = (projection - result['career_mean']) / result['career_std']
    else:
        # No variance = all games had same value
        result['z_score'] = 0.0 if projection == result['career_mean'] else float('inf')

    # Check against threshold
    if abs(result['z_score']) > sigma_threshold:
        # Projection is extreme
        if projection > result['career_max']:
            # Projection exceeds career maximum
            result['is_valid'] = False
            result['recommendation'] = 'CAP'
            # Cap to career max + 1σ (allow some upside but not extreme)
            result['capped_value'] = min(projection, result['career_max'] + result['career_std'])
            result['message'] = (
                f"Projection {projection:.1f} exceeds career max {result['career_max']:.1f} "
                f"({result['z_score']:.1f}σ from mean). Recommend cap to {result['capped_value']:.1f}"
            )
            logger.warning(f"{player_name}: {result['message']}")
        else:
            # Within career range but >3σ - flag for review
            result['recommendation'] = 'FLAG'
            result['message'] = (
                f"Projection {projection:.1f} is {result['z_score']:.1f}σ from mean "
                f"({result['career_mean']:.1f} ± {result['career_std']:.1f}). Flag for review."
            )
            logger.info(f"{player_name}: {result['message']}")
    else:
        result['message'] = (
            f"Projection {projection:.1f} within normal range "
            f"({result['z_score']:.1f}σ, max {result['career_max']:.1f})"
        )
        logger.debug(f"{player_name}: {result['message']}")

    return result


def validate_player_projections(
    player_name: str,
    position: str,
    projections: Dict[str, float],
    apply_caps: bool = True
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """
    Validate all projections for a player.

    Args:
        player_name: Player name
        position: Position
        projections: Dict of stat_name -> projected_value
        apply_caps: Whether to apply recommended caps

    Returns:
        Tuple of (capped_projections, validation_results)
    """
    # Map common projection names to weekly stats columns
    stat_mapping = {
        'receptions_mean': 'receptions',
        'rec_yards_mean': 'receiving_yards',
        'rec_tds_mean': 'receiving_tds',
        'rush_yards_mean': 'rushing_yards',
        'rush_tds_mean': 'rushing_tds',
        'carries_mean': 'carries',
        'rushing_attempts': 'carries',  # FIX: Map rushing_attempts to carries column
        'pass_yards_mean': 'passing_yards',
        'pass_tds_mean': 'passing_tds',
        'pass_attempts': 'attempts',  # FIX: Map pass_attempts to attempts column
        'attempts_mean': 'attempts',
        'passing_yards': 'passing_yards',  # Additional mappings for consistency
        'receiving_yards': 'receiving_yards',
        'rushing_yards': 'rushing_yards',
        'receiving_tds': 'receiving_tds',
        'rushing_tds': 'rushing_tds',
        'passing_tds': 'passing_tds',
    }

    capped = projections.copy()
    validations = {}

    for proj_name, proj_value in projections.items():
        # Map to historical stat name
        stat_name = stat_mapping.get(proj_name, proj_name.replace('_mean', ''))

        # Validate
        validation = validate_projection(
            player_name=player_name,
            position=position,
            stat_name=stat_name,
            projection=proj_value
        )

        validations[proj_name] = validation

        # Apply cap if recommended
        if apply_caps and validation['recommendation'] == 'CAP':
            capped[proj_name] = validation['capped_value']
            logger.warning(
                f"CAPPED {player_name} {proj_name}: "
                f"{proj_value:.1f} → {validation['capped_value']:.1f} "
                f"({validation['z_score']:.1f}σ, max={validation['career_max']:.1f})"
            )

    return capped, validations


# Convenience functions
def check_projection_sanity(
    player_name: str,
    position: str,
    stat_name: str,
    projection: float
) -> bool:
    """
    Quick sanity check: Is projection within 3σ of historical mean?

    Returns True if sane, False if extreme.
    """
    validation = validate_projection(player_name, position, stat_name, projection)
    return validation['recommendation'] == 'ACCEPT'
