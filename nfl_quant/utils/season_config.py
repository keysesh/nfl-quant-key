"""
Central NFL Season Configuration
==================================

Single source of truth for determining the current NFL season.

The NFL season spans two calendar years:
- Regular season: September (Year N) through January (Year N+1)
- Season is named for the year it STARTS in

Examples:
- September 2025 → 2025 season
- January 2026 → 2025 season (playoffs)
- February 2026 → 2025 season (Super Bowl)
"""

from datetime import datetime
from typing import Optional


def get_current_nfl_season(reference_date: Optional[datetime] = None) -> int:
    """
    Determine the current NFL season based on date.

    NFL season logic:
    - January-July: Previous year's season (off-season/draft)
    - August-December: Current year's season

    Args:
        reference_date: Date to check (defaults to today)

    Returns:
        NFL season year (int)

    Examples:
        >>> get_current_nfl_season(datetime(2025, 1, 15))  # Playoffs
        2024
        >>> get_current_nfl_season(datetime(2025, 9, 5))   # Week 1
        2025
        >>> get_current_nfl_season(datetime(2025, 11, 14)) # Week 11
        2025
    """
    if reference_date is None:
        reference_date = datetime.now()

    year = reference_date.year
    month = reference_date.month

    # January through July: off-season, use previous year's season
    if month <= 7:
        return year - 1

    # August through December: current season
    return year


def get_current_nfl_week(reference_date: Optional[datetime] = None) -> Optional[int]:
    """
    Estimate current NFL week based on date.

    Note: This is an approximation. For exact week, query nflverse schedule.

    Args:
        reference_date: Date to check (defaults to today)

    Returns:
        Estimated week number (1-18), or None if off-season
    """
    if reference_date is None:
        reference_date = datetime.now()

    season = get_current_nfl_season(reference_date)

    # Rough approximation: Week 1 typically starts first Thursday after Labor Day
    # (early September, usually Sept 5-11)
    # For 2025 season, assume Week 1 starts Sept 4, 2025

    # If not in season months, return None
    month = reference_date.month
    if month <= 7:  # Off-season
        return None

    # Rough week calculation
    # September 4 = Week 1 start (Thursday)
    # Each week is 7 days
    week_1_start = datetime(season, 9, 4)  # Approximate

    if reference_date < week_1_start:
        return 0  # Preseason

    days_since_week1 = (reference_date - week_1_start).days
    week = (days_since_week1 // 7) + 1

    # Cap at Week 18 (regular season)
    return min(week, 18)


def infer_season_from_week(week: int, reference_date: Optional[datetime] = None) -> int:
    """
    Intelligently infer which season a week number refers to.

    Logic:
    - If current week >= requested week: assume current season
    - If requesting future week: assume current season (might be upcoming)
    - Provides user feedback about inference

    Args:
        week: NFL week number (1-18)
        reference_date: Reference date (defaults to today)

    Returns:
        Inferred season year

    Examples:
        >>> # It's Week 11 in Nov 2025, asking for Week 10
        >>> infer_season_from_week(10, datetime(2025, 11, 14))
        2025  # Recent past week in current season

        >>> # It's Week 5 in Oct 2025, asking for Week 15
        >>> infer_season_from_week(15, datetime(2025, 10, 10))
        2025  # Upcoming week in current season
    """
    current_season = get_current_nfl_season(reference_date)
    current_week = get_current_nfl_week(reference_date)

    # If off-season or preseason, default to current season
    if current_week is None or current_week == 0:
        return current_season

    # If requesting earlier or similar week, assume current season
    # (common case: validating last week's results)
    if week <= current_week + 1:  # Allow slight future (next week predictions)
        return current_season

    # Requesting future week - could be next season or late current season
    # Default to current season (user can override if needed)
    return current_season


def validate_season_consistency(season: int, reference_date: Optional[datetime] = None) -> bool:
    """
    Validate that a provided season matches the current NFL season.

    Args:
        season: Season year to validate
        reference_date: Date to check against (defaults to today)

    Returns:
        True if season matches current NFL season

    Raises:
        ValueError: If season doesn't match current season
    """
    current_season = get_current_nfl_season(reference_date)

    if season != current_season:
        raise ValueError(
            f"Season mismatch: Provided season={season}, but current NFL season={current_season}. "
            f"Are you trying to backtest historical data? If so, explicitly pass season={season}."
        )

    return True


# Default season for all scripts (dynamic)
CURRENT_NFL_SEASON = get_current_nfl_season()
CURRENT_NFL_WEEK = get_current_nfl_week()


if __name__ == "__main__":
    # CLI for checking current season
    import sys

    print(f"Current NFL Season: {CURRENT_NFL_SEASON}")
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Estimated Week: {CURRENT_NFL_WEEK}")

    # Test historical dates
    test_dates = [
        datetime(2025, 1, 15),   # Playoffs
        datetime(2025, 7, 1),    # Off-season
        datetime(2025, 9, 4),    # Week 1
        datetime(2025, 11, 14),  # Week 11
        datetime(2026, 1, 10),   # Next year playoffs
    ]

    print("\nHistorical Season Detection:")
    for date in test_dates:
        season = get_current_nfl_season(date)
        week = get_current_nfl_week(date)
        print(f"  {date.strftime('%Y-%m-%d')} → Season {season}, Week {week}")

    # Test smart week inference
    print("\nSmart Week → Season Inference (from today):")
    for week_num in [1, 5, 10, 11, 15, 18]:
        inferred = infer_season_from_week(week_num)
        print(f"  Week {week_num:2d} → Season {inferred}")
