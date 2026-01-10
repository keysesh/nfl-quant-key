"""
Season detection and handling utilities.

Standardizes season logic across all scripts to match R validation script.

The season detection logic follows NFL calendar:
- NFL regular season: September (Week 1) through early January (Week 18)
- Playoffs: Mid-January through early February (Super Bowl)
- Offseason: February through August

Detection Rule:
- August-December → Current year is the season
- January-July → Previous year is the season (playoffs/offseason)

Examples:
- September 2025 → 2025 season (regular season start)
- November 2025 → 2025 season (mid-season)
- January 2026 → 2025 season (playoffs)
- March 2026 → 2025 season (offseason, training data available)
"""

from datetime import datetime
from typing import Optional, List, Tuple


def get_current_season() -> int:
    """
    Detect current NFL season based on date.

    Logic matches R validation script (validate_week_r.R):
    - August-December → Current year
    - January-July → Previous year

    Returns:
        int: Current NFL season year

    Examples:
        >>> # If today is November 14, 2025
        >>> get_current_season()
        2025

        >>> # If today is January 15, 2026 (playoffs)
        >>> get_current_season()
        2025

        >>> # If today is July 1, 2026 (offseason)
        >>> get_current_season()
        2025
    """
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # NFL season runs Aug-Feb (Aug = start, Jan/Feb = playoffs)
    # This matches the R script logic:
    # current_season <- if (current_month >= 8) current_year else current_year - 1
    if current_month >= 8:
        return current_year
    else:
        return current_year - 1


def get_training_seasons(current_season: Optional[int] = None,
                         include_current: bool = True) -> List[int]:
    """
    Get list of seasons to use for model training.

    Args:
        current_season: Current season (auto-detected if None)
        include_current: Whether to include current season (default: True)

    Returns:
        List[int]: Seasons to include in training (e.g., [2024, 2025])

    Examples:
        >>> # If current is 2025 Week 10
        >>> get_training_seasons()
        [2024, 2025]  # Full 2024 + 2025 weeks 1-9

        >>> # For historical backtest on 2024 only
        >>> get_training_seasons(2024, include_current=False)
        [2023]
    """
    if current_season is None:
        current_season = get_current_season()

    seasons = []

    # Always include previous completed season
    seasons.append(current_season - 1)

    # Optionally include current season (for live predictions)
    if include_current:
        seasons.append(current_season)

    return seasons


def validate_season(season: int, allow_future: bool = False) -> bool:
    """
    Validate if a season is reasonable.

    Args:
        season: Season year to validate
        allow_future: Whether to allow future seasons (default: False)

    Returns:
        bool: True if season is valid

    Examples:
        >>> validate_season(2024)
        True

        >>> validate_season(2008)  # Before reliable data
        False

        >>> validate_season(2026)  # Future
        False

        >>> validate_season(2026, allow_future=True)
        True
    """
    current_season = get_current_season()

    # NFL data available from ~2009 onward (nflverse, nflfastR)
    if season < 2009:
        return False

    # Check if future season
    if season > current_season and not allow_future:
        return False

    # Reasonable upper bound (next season max)
    if season > current_season + 1:
        return False

    return True


def season_to_year_range(season: int) -> Tuple[int, int]:
    """
    Convert season to year range.

    NFL seasons span two calendar years:
    - Regular season: September (year N) to January (year N+1)
    - Playoffs: January-February (year N+1)

    Args:
        season: NFL season year

    Returns:
        Tuple[int, int]: (start_year, end_year) covering regular season + playoffs

    Examples:
        >>> season_to_year_range(2024)
        (2024, 2025)

        >>> season_to_year_range(2025)
        (2025, 2026)
    """
    return (season, season + 1)


def get_current_week(season: Optional[int] = None) -> int:
    """
    Get the current NFL week based on date.

    Uses NFL schedule data to accurately determine the current week.
    Week 1 2025: September 4-8, 2025
    Each subsequent week is roughly 7 days after.

    Args:
        season: Season to check (default: current season)

    Returns:
        int: Current week number (1-18 regular season, 19-22 playoffs)

    Examples:
        >>> # If today is November 16, 2025 (Week 11 games)
        >>> get_current_week()
        11
    """
    if season is None:
        season = get_current_season()

    current_season = get_current_season()

    # If checking a past season, return 18 (season complete)
    if season < current_season:
        return 18

    # If checking a future season, return 0
    if season > current_season:
        return 0

    now = datetime.now()

    # NFL 2025 Season Week 1 starts September 4, 2025 (Thursday)
    # Adjust these dates for each season as needed
    # Week dates are based on the Tuesday start of each game week
    nfl_week_starts = {
        2025: {
            1: datetime(2025, 9, 2),    # Week 1: Sep 4-8
            2: datetime(2025, 9, 9),    # Week 2: Sep 11-15
            3: datetime(2025, 9, 16),   # Week 3: Sep 18-22
            4: datetime(2025, 9, 23),   # Week 4: Sep 25-29
            5: datetime(2025, 9, 30),   # Week 5: Oct 2-6
            6: datetime(2025, 10, 7),   # Week 6: Oct 9-13
            7: datetime(2025, 10, 14),  # Week 7: Oct 16-20
            8: datetime(2025, 10, 21),  # Week 8: Oct 23-27
            9: datetime(2025, 10, 28),  # Week 9: Oct 30-Nov 3
            10: datetime(2025, 11, 4),  # Week 10: Nov 6-10
            11: datetime(2025, 11, 11), # Week 11: Nov 13-17
            12: datetime(2025, 11, 18), # Week 12: Nov 20-24
            13: datetime(2025, 11, 25), # Week 13: Nov 27-Dec 1
            14: datetime(2025, 12, 2),  # Week 14: Dec 4-8
            15: datetime(2025, 12, 9),  # Week 15: Dec 11-15
            16: datetime(2025, 12, 16), # Week 16: Dec 18-22
            17: datetime(2025, 12, 23), # Week 17: Dec 25-29
            18: datetime(2025, 12, 30), # Week 18: Jan 1-5
            # Playoffs
            19: datetime(2026, 1, 7),   # Wild Card: Jan 10-13
            20: datetime(2026, 1, 14),  # Divisional: Jan 17-18
            21: datetime(2026, 1, 21),  # Conference: Jan 25-26
            22: datetime(2026, 2, 4),   # Super Bowl: Feb 8
        },
        2024: {
            1: datetime(2024, 9, 3),
            # ... Similar pattern for 2024
        }
    }

    # Use the schedule if available
    if season in nfl_week_starts:
        week_dates = nfl_week_starts[season]
        current_week = 1

        for week, start_date in sorted(week_dates.items()):
            if now >= start_date:
                current_week = week
            else:
                break

        # Cap at 22 (Super Bowl week) to include playoffs
        return min(current_week, 22)

    # Fallback: estimate based on season start
    season_start = datetime(season, 9, 7)

    if now < season_start:
        return 0

    days_since_start = (now - season_start).days
    weeks_elapsed = (days_since_start // 7) + 1

    return min(max(1, weeks_elapsed), 22)


def get_weeks_completed(season: Optional[int] = None,
                        current_week: Optional[int] = None) -> int:
    """
    Estimate number of weeks completed in a season.

    Args:
        season: Season to check (default: current season)
        current_week: Override current week (for testing)

    Returns:
        int: Number of completed weeks (0-18)

    Note:
        This is an estimate. For exact completion status, check actual game data.
    """
    if season is None:
        season = get_current_season()

    current_season = get_current_season()

    # If checking a past season, all 18 weeks are complete
    if season < current_season:
        return 18

    # If checking a future season, no weeks complete
    if season > current_season:
        return 0

    # For current season, estimate based on date
    if current_week is not None:
        # Week explicitly provided
        return max(0, current_week - 1)

    # Use the new accurate week detection
    return max(0, get_current_week(season) - 1)


def format_season_display(season: int) -> str:
    """
    Format season for display purposes.

    Args:
        season: Season year

    Returns:
        str: Formatted season string

    Examples:
        >>> format_season_display(2024)
        '2024-2025 NFL Season'

        >>> format_season_display(2025)
        '2025-2026 NFL Season'
    """
    start_year, end_year = season_to_year_range(season)
    return f"{start_year}-{end_year} NFL Season"


def is_offseason(season: Optional[int] = None) -> bool:
    """
    Check if we're currently in the offseason for a given season.

    Args:
        season: Season to check (default: current season)

    Returns:
        bool: True if in offseason (Feb-Aug), False if in season (Sep-Jan)

    Examples:
        >>> # If today is March 2026
        >>> is_offseason(2025)
        True  # 2025 season is over

        >>> # If today is November 2025
        >>> is_offseason(2025)
        False  # 2025 season is active
    """
    if season is None:
        season = get_current_season()

    current_season = get_current_season()
    now = datetime.now()
    month = now.month

    # If checking a past season, always offseason
    if season < current_season:
        return True

    # If checking a future season, always offseason (not started)
    if season > current_season:
        return True

    # For current season, check month
    # Season active: August (8) through January (1)
    # Offseason: February (2) through July (7)
    if month >= 2 and month <= 7:
        return True

    return False


# Convenience constants
EARLIEST_SEASON = 2009  # First season with reliable nflverse data
REGULAR_SEASON_WEEKS = 18  # NFL regular season length (as of 2021)
PLAYOFF_WEEKS = 4  # Wild Card, Divisional, Conference Championship, Super Bowl


if __name__ == "__main__":
    # Quick tests
    print("NFL Season Utils - Quick Test")
    print("=" * 60)
    print(f"Current Season: {get_current_season()}")
    print(f"Training Seasons: {get_training_seasons()}")
    print(f"Season Display: {format_season_display(get_current_season())}")
    print(f"Is Offseason: {is_offseason()}")
    print(f"Weeks Completed (estimate): {get_weeks_completed()}")
    print("=" * 60)
    print(f"Season 2024 Valid: {validate_season(2024)}")
    print(f"Season 2008 Valid: {validate_season(2008)}")
    print(f"Season 2026 Valid: {validate_season(2026)}")
    print(f"Season 2026 Valid (allow future): {validate_season(2026, allow_future=True)}")
