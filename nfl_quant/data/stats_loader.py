"""
Universal Stats Loader - unified interface for loading player stats from NFLverse.

This module provides a single API for loading player stats from NFLverse data.

Usage:
    from nfl_quant.data.stats_loader import load_weekly_stats

    # Load weekly stats
    stats = load_weekly_stats(week=1, season=2025)
"""

from pathlib import Path
import pandas as pd
from typing import Optional, Literal
import logging

from .adapters import NFLVerseAdapter

logger = logging.getLogger(__name__)

# Global adapter instance (initialized once)
_nflverse_adapter = None


def get_nflverse_adapter() -> NFLVerseAdapter:
    """Get or create NFLverse adapter instance."""
    global _nflverse_adapter
    if _nflverse_adapter is None:
        _nflverse_adapter = NFLVerseAdapter()
    return _nflverse_adapter


def load_weekly_stats(
    week: int,
    season: int
) -> pd.DataFrame:
    """
    Load weekly player stats in canonical format from NFLverse.

    Args:
        week: Week number (1-18)
        season: Season year (2023, 2024, 2025)

    Returns:
        DataFrame in canonical format with columns:
            player_id, player_name, position, team, season, week,
            passing_yards, passing_attempts, passing_completions, passing_tds, interceptions,
            rushing_yards, rushing_attempts, rushing_tds,
            receptions, receiving_yards, receiving_tds, targets,
            opponent, game_id, source

    Raises:
        FileNotFoundError: If no data available for the given week/season

    Examples:
        >>> stats = load_weekly_stats(week=1, season=2025)
        >>> print(f"Loaded {len(stats)} players")

        >>> # Filter to specific position
        >>> qb_stats = stats[stats['position'] == 'QB']
    """
    adapter = get_nflverse_adapter()

    if not adapter.is_available(week, season):
        raise FileNotFoundError(
            f"No stats data available for week {week}, season {season}. "
            f"Run data fetching scripts to populate NFLverse data."
        )

    logger.debug(f"Loading week {week}, season {season} from NFLverse")
    return adapter.load_weekly_stats(week, season)


def is_data_available(week: int, season: int) -> bool:
    """
    Check if data is available for the given week/season from NFLverse.

    Args:
        week: Week number
        season: Season year

    Returns:
        True if data exists
    """
    return get_nflverse_adapter().is_available(week, season)


def load_season_stats(season: int, weeks: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load stats for multiple weeks in a season.

    Args:
        season: Season year
        weeks: List of week numbers to load (None = all available weeks)

    Returns:
        Combined DataFrame for all weeks
    """
    if weeks is None:
        # Auto-detect available weeks
        weeks = []
        for week in range(1, 19):  # NFL has 18 weeks
            if is_data_available(week, season):
                weeks.append(week)

    if not weeks:
        raise FileNotFoundError(f"No data available for season {season}")

    all_data = []
    for week in weeks:
        # FAIL EXPLICITLY if any week is unavailable
        week_data = load_weekly_stats(week, season)
        all_data.append(week_data)

    if not all_data:
        raise FileNotFoundError(
            f"No data loaded for season {season}. "
            f"Run data fetching scripts to populate data."
        )

    return pd.concat(all_data, ignore_index=True)


def get_player_stats(
    player_name: str,
    week: int,
    season: int,
    team: Optional[str] = None
) -> Optional[pd.Series]:
    """
    Get stats for a specific player in a specific week.

    Args:
        player_name: Player name (case-insensitive, fuzzy match)
        week: Week number
        season: Season year
        team: Team abbreviation (optional, helps with matching)

    Returns:
        Series with player stats or None if not found
    """
    stats = load_weekly_stats(week, season)

    # Normalize player name for matching
    from .adapters.base_adapter import StatsAdapter
    adapter = StatsAdapter()
    normalized_name = adapter.normalize_player_name(player_name)

    # Create normalized column for matching
    stats['normalized_name'] = stats['player_name'].apply(adapter.normalize_player_name)

    # Filter by name
    matches = stats[stats['normalized_name'] == normalized_name]

    # If multiple matches and team provided, filter by team
    if len(matches) > 1 and team:
        team_normalized = adapter.normalize_team_name(team)
        matches = matches[matches['team'] == team_normalized]

    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        logger.warning(f"Multiple matches for {player_name}, returning first")

    return matches.iloc[0]
