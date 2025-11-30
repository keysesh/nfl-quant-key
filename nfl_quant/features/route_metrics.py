"""
Route-based metrics for NFL QUANT V4.

This module provides functions to calculate and analyze route-based receiving metrics:
- Routes run (estimated from snap counts)
- Route participation (RP)
- Targets per route run (TPRR)
- Yards per route run (Y/RR)

These metrics are key inputs for the V4 systematic overhaul using Negative Binomial
and Lognormal distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def estimate_routes_run(
    snap_count: int,
    team_pass_attempts: int,
    team_total_plays: int
) -> float:
    """
    Estimate routes run from snap counts.

    This is Option A from the V4 specification - estimating routes run
    from snap participation and team pass play rate.

    Formula: routes_run ≈ snaps × (team_pass_attempts / team_plays)

    Args:
        snap_count: Player's offensive snaps
        team_pass_attempts: Team's total pass attempts
        team_total_plays: Team's total offensive plays

    Returns:
        Estimated routes run (float)

    Example:
        >>> estimate_routes_run(50, 35, 65)
        26.92  # 50 snaps × (35/65) = 26.92 routes
    """
    if team_total_plays == 0:
        logger.warning("Team has 0 total plays, returning 0 routes")
        return 0.0

    pass_play_rate = team_pass_attempts / team_total_plays
    estimated_routes = snap_count * pass_play_rate

    return estimated_routes


def calculate_route_participation(
    routes_run: float,
    team_pass_attempts: int
) -> float:
    """
    Calculate route participation percentage (RP).

    RP = routes_run / team_pass_attempts

    This represents what percentage of the team's pass plays the player
    was on the field running a route.

    Args:
        routes_run: Player's routes run
        team_pass_attempts: Team's total pass attempts

    Returns:
        Route participation rate (0.0 to 1.0+, can exceed 1.0 for estimation errors)

    Example:
        >>> calculate_route_participation(30, 40)
        0.75  # Player ran route on 75% of pass plays
    """
    if team_pass_attempts == 0:
        logger.warning("Team has 0 pass attempts, returning 0.0 RP")
        return 0.0

    rp = routes_run / team_pass_attempts

    # Cap at 1.0 (100%) - estimation can occasionally exceed
    if rp > 1.0:
        logger.debug(f"RP {rp:.2f} exceeds 1.0, capping at 1.0")
        rp = 1.0

    return rp


def calculate_tprr(
    targets: int,
    routes_run: float
) -> float:
    """
    Calculate targets per route run (TPRR).

    TPRR = targets / routes_run

    This measures target efficiency - how often the player gets targeted
    when running a route.

    Args:
        targets: Player's targets
        routes_run: Player's routes run

    Returns:
        Targets per route run (typically 0.15 to 0.35 for WR/TE)

    Example:
        >>> calculate_tprr(8, 30)
        0.267  # Player targeted on 26.7% of routes
    """
    if routes_run == 0:
        logger.warning("Player has 0 routes run, returning 0.0 TPRR")
        return 0.0

    tprr = targets / routes_run

    return tprr


def calculate_yrr(
    receiving_yards: int,
    routes_run: float
) -> float:
    """
    Calculate yards per route run (Y/RR).

    Y/RR = receiving_yards / routes_run

    This measures per-route efficiency - how many yards the player generates
    per route run (whether targeted or not).

    Args:
        receiving_yards: Player's receiving yards
        routes_run: Player's routes run

    Returns:
        Yards per route run (typically 1.5 to 3.0 for WR/TE)

    Example:
        >>> calculate_yrr(75, 30)
        2.5  # Player averages 2.5 yards per route
    """
    if routes_run == 0:
        logger.warning("Player has 0 routes run, returning 0.0 Y/RR")
        return 0.0

    yrr = receiving_yards / routes_run

    return yrr


def extract_route_metrics_from_pbp(
    player_name: str,
    position: str,
    team: str,
    season: int,
    week: int,
    pbp_data: pd.DataFrame,
    snap_counts: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract all route-based metrics for a player from PBP and snap data.

    This is the main entry point for route metric extraction in the V4 pipeline.

    Args:
        player_name: Player name (normalized)
        position: Player position (WR, TE, RB)
        team: Team abbreviation
        season: NFL season year
        week: Week number
        pbp_data: Play-by-play dataframe
        snap_counts: Snap counts dataframe

    Returns:
        Dictionary with keys:
            - routes_run: Estimated routes run
            - route_participation: RP (0.0 to 1.0)
            - tprr: Targets per route run
            - yrr: Yards per route run
            - targets: Raw targets (for validation)
            - receiving_yards: Raw receiving yards
            - snaps: Raw snap count

    Example:
        >>> metrics = extract_route_metrics_from_pbp(
        ...     'George Kittle', 'TE', 'SF', 2025, 12, pbp, snaps
        ... )
        >>> metrics['tprr']
        0.25
    """
    from nfl_quant.utils.player_names import normalize_name

    player_normalized = normalize_name(player_name)

    # Get snap count for player
    player_snaps = snap_counts[
        (snap_counts['player'].apply(normalize_name) == player_normalized) &
        (snap_counts['team'] == team) &
        (snap_counts['season'] == season) &
        (snap_counts['week'] == week)
    ]

    if player_snaps.empty:
        logger.warning(f"No snap data for {player_name} Week {week}")
        return {
            'routes_run': 0.0,
            'route_participation': 0.0,
            'tprr': 0.0,
            'yrr': 0.0,
            'targets': 0,
            'receiving_yards': 0,
            'snaps': 0
        }

    snap_count = int(player_snaps['offense_snaps'].iloc[0])

    # Get team pass attempts and total plays from PBP
    team_plays = pbp_data[
        (pbp_data['season'] == season) &
        (pbp_data['week'] == week) &
        (pbp_data['posteam'] == team)
    ]

    team_pass_attempts = len(team_plays[team_plays['pass'] == 1])
    team_total_plays = len(team_plays)

    # Get player targets and receiving yards from PBP
    player_targets = pbp_data[
        (pbp_data['season'] == season) &
        (pbp_data['week'] == week) &
        (pbp_data['receiver_player_name'].apply(normalize_name) == player_normalized)
    ]

    targets = len(player_targets)
    receiving_yards = int(player_targets['receiving_yards'].sum())

    # Calculate route metrics
    routes_run = estimate_routes_run(snap_count, team_pass_attempts, team_total_plays)
    rp = calculate_route_participation(routes_run, team_pass_attempts)
    tprr = calculate_tprr(targets, routes_run)
    yrr = calculate_yrr(receiving_yards, routes_run)

    return {
        'routes_run': routes_run,
        'route_participation': rp,
        'tprr': tprr,
        'yrr': yrr,
        'targets': targets,
        'receiving_yards': receiving_yards,
        'snaps': snap_count
    }


def calculate_trailing_route_metrics(
    player_name: str,
    position: str,
    team: str,
    season: int,
    current_week: int,
    pbp_data: pd.DataFrame,
    snap_counts: pd.DataFrame,
    lookback_weeks: int = 4,
    use_ewma: bool = True,
    ewma_span: int = 4
) -> Dict[str, float]:
    """
    Calculate trailing (EWMA-weighted) route metrics over multiple weeks.

    This function integrates with the V4 EWMA weighting system to calculate
    recent-weighted averages of route metrics.

    Args:
        player_name: Player name (normalized)
        position: Player position
        team: Team abbreviation
        season: NFL season year
        current_week: Current week (metrics calculated before this week)
        pbp_data: Play-by-play dataframe
        snap_counts: Snap counts dataframe
        lookback_weeks: Number of weeks to look back (default 4)
        use_ewma: Use EWMA weighting (True) or simple mean (False)
        ewma_span: EWMA span parameter (default 4 for 40%-27%-18%-12% weights)

    Returns:
        Dictionary with trailing metrics:
            - trailing_routes_run
            - trailing_route_participation
            - trailing_tprr
            - trailing_yrr

    Example:
        >>> trailing = calculate_trailing_route_metrics(
        ...     'George Kittle', 'TE', 'SF', 2025, 12, pbp, snaps
        ... )
        >>> trailing['trailing_tprr']
        0.24  # EWMA-weighted TPRR over last 4 weeks
    """
    # Collect metrics for each week
    weeks_data = []

    for week in range(max(1, current_week - lookback_weeks), current_week):
        metrics = extract_route_metrics_from_pbp(
            player_name, position, team, season, week, pbp_data, snap_counts
        )
        metrics['week'] = week
        weeks_data.append(metrics)

    if not weeks_data:
        logger.warning(f"No route metrics found for {player_name} before Week {current_week}")
        return {
            'trailing_routes_run': 0.0,
            'trailing_route_participation': 0.0,
            'trailing_tprr': 0.0,
            'trailing_yrr': 0.0
        }

    df = pd.DataFrame(weeks_data).sort_values('week')

    # Apply EWMA or simple mean
    if use_ewma and len(df) > 1:
        trailing_routes = df['routes_run'].ewm(span=ewma_span, adjust=False).mean().iloc[-1]
        trailing_rp = df['route_participation'].ewm(span=ewma_span, adjust=False).mean().iloc[-1]
        trailing_tprr = df['tprr'].ewm(span=ewma_span, adjust=False).mean().iloc[-1]
        trailing_yrr = df['yrr'].ewm(span=ewma_span, adjust=False).mean().iloc[-1]
    else:
        trailing_routes = df['routes_run'].mean()
        trailing_rp = df['route_participation'].mean()
        trailing_tprr = df['tprr'].mean()
        trailing_yrr = df['yrr'].mean()

    return {
        'trailing_routes_run': trailing_routes,
        'trailing_route_participation': trailing_rp,
        'trailing_tprr': trailing_tprr,
        'trailing_yrr': trailing_yrr
    }


# Validation functions

def validate_route_metrics(metrics: Dict[str, float], position: str) -> Tuple[bool, str]:
    """
    Validate route metrics against expected ranges by position.

    Args:
        metrics: Route metrics dictionary
        position: Player position

    Returns:
        (is_valid, message) tuple

    Example:
        >>> metrics = {'tprr': 0.85, 'yrr': 12.5}
        >>> validate_route_metrics(metrics, 'WR')
        (False, "TPRR 0.85 exceeds expected range for WR (0.15-0.40)")
    """
    # Expected ranges by position (based on NFL analytics literature)
    ranges = {
        'WR': {
            'tprr': (0.15, 0.35),  # 15-35% of routes result in target
            'yrr': (1.5, 3.5),      # 1.5-3.5 yards per route
            'route_participation': (0.50, 1.0)  # 50-100% of pass plays
        },
        'TE': {
            'tprr': (0.12, 0.30),
            'yrr': (1.0, 3.0),
            'route_participation': (0.40, 0.95)
        },
        'RB': {
            'tprr': (0.08, 0.25),
            'yrr': (0.5, 2.5),
            'route_participation': (0.20, 0.70)
        }
    }

    if position not in ranges:
        return True, "Position not in validation set"

    pos_ranges = ranges[position]

    for metric, (min_val, max_val) in pos_ranges.items():
        if metric not in metrics:
            continue

        value = metrics[metric]

        if value < min_val or value > max_val:
            return False, f"{metric.upper()} {value:.2f} outside expected range for {position} ({min_val}-{max_val})"

    return True, "All metrics within expected ranges"
