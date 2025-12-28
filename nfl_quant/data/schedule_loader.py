"""
Canonical Schedule Data Loader

Single entry point for all schedule/game time data access.
Supports mapping weeks to game dates for depth chart derivation.

Usage:
    from nfl_quant.data.schedule_loader import get_schedule, get_week_kickoff_times

    # Get full schedule
    df = get_schedule(season=2025)

    # Get earliest kickoff time for a week
    kickoffs = get_week_kickoff_times(season=2025, week=18)
    # Returns: {'BUF': datetime(2026, 1, 4, ...), 'PHI': datetime(2026, 1, 4, ...), ...}
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

# Cache for loaded schedule data
_SCHEDULE_CACHE = {}
_CACHE_TIMESTAMPS = {}

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCHEDULE_PATH = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
MAX_CACHE_AGE_HOURS = 24  # Schedules rarely change

# Required columns
REQUIRED_COLUMNS = ['season', 'week', 'gameday', 'home_team', 'away_team']


def get_schedule(
    season: Optional[int] = None,
    week: Optional[int] = None,
    refresh: bool = False
) -> pd.DataFrame:
    """
    Get NFL schedule data.

    Args:
        season: Filter to specific season (None = all seasons)
        week: Filter to specific week (None = all weeks)
        refresh: Force reload from disk

    Returns:
        DataFrame with schedule data

    Raises:
        FileNotFoundError: If schedule file doesn't exist
    """
    global _SCHEDULE_CACHE, _CACHE_TIMESTAMPS

    cache_key = f"schedule_{season or 'all'}_{week or 'all'}"

    # Check cache validity
    if not refresh and cache_key in _SCHEDULE_CACHE:
        cache_time = _CACHE_TIMESTAMPS.get(cache_key)
        if cache_time:
            cache_age = datetime.now() - cache_time
            if cache_age < timedelta(hours=MAX_CACHE_AGE_HOURS):
                return _SCHEDULE_CACHE[cache_key]

    # Check file exists
    if not SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"Schedule file not found: {SCHEDULE_PATH}. "
            "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
        )

    # Load data
    logger.info(f"Loading schedule from {SCHEDULE_PATH}")
    df = pd.read_parquet(SCHEDULE_PATH)
    logger.info(f"Loaded {len(df):,} schedule rows")

    # Validate required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schedule missing required columns: {missing}")

    # Filter to season
    if season is not None:
        df = df[df['season'] == season].copy()
        logger.info(f"Filtered to season {season}: {len(df):,} rows")

    # Filter to week
    if week is not None:
        df = df[df['week'] == week].copy()
        logger.info(f"Filtered to week {week}: {len(df):,} rows")

    # Update cache
    _SCHEDULE_CACHE[cache_key] = df
    _CACHE_TIMESTAMPS[cache_key] = datetime.now()

    return df


def get_week_kickoff_times(
    season: int,
    week: int,
    timezone: str = 'America/New_York'
) -> Dict[str, datetime]:
    """
    Get the earliest kickoff time for each team in a given week.

    Args:
        season: NFL season
        week: NFL week number
        timezone: Timezone for parsing (default ET)

    Returns:
        Dict mapping team abbreviation to game kickoff datetime
    """
    df = get_schedule(season=season, week=week)

    if len(df) == 0:
        logger.warning(f"No games found for season {season} week {week}")
        return {}

    # Parse gameday and gametime into datetime
    tz = pytz.timezone(timezone)
    kickoffs = {}

    for _, row in df.iterrows():
        try:
            # Parse gameday (YYYY-MM-DD format) and gametime (HH:MM format)
            gameday = str(row['gameday'])
            gametime = str(row.get('gametime', '13:00'))  # Default 1 PM

            # Handle missing gametime
            if pd.isna(gametime) or gametime == 'nan':
                gametime = '13:00'

            dt_str = f"{gameday} {gametime}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            dt = tz.localize(dt)

            # Add both teams
            kickoffs[row['home_team']] = dt
            kickoffs[row['away_team']] = dt

        except Exception as e:
            logger.warning(f"Error parsing game time for {row.get('game_id')}: {e}")
            continue

    logger.info(f"Found kickoff times for {len(kickoffs)} teams in week {week}")
    return kickoffs


def get_earliest_kickoff(
    season: int,
    week: int,
    timezone: str = 'America/New_York'
) -> Optional[datetime]:
    """
    Get the earliest game kickoff time for a given week.

    Useful for determining which depth chart snapshot to use.

    Args:
        season: NFL season
        week: NFL week number
        timezone: Timezone for result

    Returns:
        Earliest kickoff datetime, or None if no games found
    """
    kickoffs = get_week_kickoff_times(season=season, week=week, timezone=timezone)

    if not kickoffs:
        return None

    return min(kickoffs.values())


def get_schedule_freshness() -> dict:
    """Get freshness status of schedule data."""
    if not SCHEDULE_PATH.exists():
        return {
            'exists': False,
            'path': str(SCHEDULE_PATH),
            'status': 'MISSING'
        }

    file_age = datetime.now() - datetime.fromtimestamp(SCHEDULE_PATH.stat().st_mtime)
    hours_old = file_age.total_seconds() / 3600

    return {
        'exists': True,
        'path': str(SCHEDULE_PATH),
        'modified': datetime.fromtimestamp(SCHEDULE_PATH.stat().st_mtime).isoformat(),
        'hours_old': round(hours_old, 1),
        'status': 'FRESH' if hours_old < MAX_CACHE_AGE_HOURS else 'STALE',
        'threshold_hours': MAX_CACHE_AGE_HOURS
    }


def clear_schedule_cache() -> None:
    """Clear the schedule cache."""
    global _SCHEDULE_CACHE, _CACHE_TIMESTAMPS
    _SCHEDULE_CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
    logger.info("Cleared schedule cache")
