"""
Canonical Vegas Lines Data Loader

Single entry point for all game lines data access.
Handles spread, total, and moneyline data.

Usage:
    from nfl_quant.data.lines_loader import get_lines, get_game_total

    # Get all lines for a week
    df = get_lines(week=18)

    # Get specific game total
    total = get_game_total(week=18, home_team='BUF', away_team='PHI')
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for loaded lines data
_LINES_CACHE = {}
_CACHE_TIMESTAMPS = {}

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MAX_CACHE_AGE_HOURS = 4
MAX_FILE_AGE_HOURS = 6  # Lines stale faster than PBP


def get_lines(
    week: int,
    season: int = 2025,
    source: str = 'draftkings',
    refresh: bool = False
) -> pd.DataFrame:
    """
    Get Vegas lines for a specific week.

    Args:
        week: NFL week number
        season: NFL season (default 2025)
        source: Odds source (default 'draftkings')
        refresh: Force reload from disk

    Returns:
        DataFrame with lines data

    Raises:
        FileNotFoundError: If lines file doesn't exist
    """
    global _LINES_CACHE, _CACHE_TIMESTAMPS

    cache_key = f"lines_{season}_{week}_{source}"

    # Check cache validity
    if not refresh and cache_key in _LINES_CACHE:
        cache_time = _CACHE_TIMESTAMPS.get(cache_key)
        if cache_time:
            cache_age = datetime.now() - cache_time
            if cache_age < timedelta(hours=MAX_CACHE_AGE_HOURS):
                return _LINES_CACHE[cache_key]

    # Find lines file
    lines_path = DATA_DIR / f'odds_week{week}_{source}.csv'
    simple_path = DATA_DIR / f'odds_week{week}.csv'

    if lines_path.exists():
        df = pd.read_csv(lines_path)
    elif simple_path.exists():
        df = pd.read_csv(simple_path)
    else:
        raise FileNotFoundError(
            f"Lines file not found for week {week}. "
            f"Run 'python scripts/fetch/fetch_live_odds.py {week}' to fetch."
        )

    # Check freshness
    file_age = datetime.now() - datetime.fromtimestamp(
        lines_path.stat().st_mtime if lines_path.exists() else simple_path.stat().st_mtime
    )
    if file_age > timedelta(hours=MAX_FILE_AGE_HOURS):
        logger.warning(
            f"Lines file is {file_age.total_seconds() / 3600:.1f}h old "
            f"(threshold: {MAX_FILE_AGE_HOURS}h). Lines may have moved."
        )

    # Update cache
    _LINES_CACHE[cache_key] = df
    _CACHE_TIMESTAMPS[cache_key] = datetime.now()

    return df


def get_game_total(
    week: int,
    home_team: str,
    away_team: str,
    season: int = 2025
) -> Optional[float]:
    """
    Get the over/under total for a specific game.

    Args:
        week: NFL week number
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: NFL season

    Returns:
        Total points line, or None if not found
    """
    try:
        df = get_lines(week=week, season=season)
    except FileNotFoundError:
        logger.warning(f"No lines data for week {week}")
        return None

    # Build game_id pattern
    game_id = f"{season}_{week}_{away_team}_{home_team}"

    # Look for over line
    over_row = df[(df['game_id'] == game_id) & (df['side'] == 'over')]
    if len(over_row) > 0:
        return over_row.iloc[0]['point']

    # Try alternate format
    if 'total' in df.columns:
        game_row = df[df['game_id'] == game_id]
        if len(game_row) > 0:
            return game_row.iloc[0]['total']

    return None


def get_game_spread(
    week: int,
    home_team: str,
    away_team: str,
    season: int = 2025
) -> Optional[float]:
    """
    Get the spread for a specific game (home team perspective).

    Args:
        week: NFL week number
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: NFL season

    Returns:
        Home team spread (negative = favorite), or None if not found
    """
    try:
        df = get_lines(week=week, season=season)
    except FileNotFoundError:
        logger.warning(f"No lines data for week {week}")
        return None

    game_id = f"{season}_{week}_{away_team}_{home_team}"

    # Look for home spread
    home_spread = df[(df['game_id'] == game_id) & (df['side'] == 'home_spread')]
    if len(home_spread) > 0:
        return home_spread.iloc[0]['point']

    return None


def get_all_game_context(week: int, season: int = 2025) -> Dict[str, Dict]:
    """
    Get game context (spread, total) for all games in a week.

    Returns:
        Dict mapping game_id to {spread, total, home_team, away_team}
    """
    try:
        df = get_lines(week=week, season=season)
    except FileNotFoundError:
        return {}

    context = {}

    # Group by game_id
    for game_id in df['game_id'].unique():
        game_rows = df[df['game_id'] == game_id]

        # Parse team names from game_id: {season}_{week}_{away}_{home}
        parts = game_id.split('_')
        if len(parts) >= 4:
            away_team = parts[2]
            home_team = parts[3]
        else:
            continue

        game_context = {
            'home_team': home_team,
            'away_team': away_team,
            'spread': None,
            'total': None
        }

        # Get spread
        home_spread = game_rows[game_rows['side'] == 'home_spread']
        if len(home_spread) > 0:
            game_context['spread'] = home_spread.iloc[0]['point']

        # Get total
        over = game_rows[game_rows['side'] == 'over']
        if len(over) > 0:
            game_context['total'] = over.iloc[0]['point']

        context[game_id] = game_context

    return context


def get_lines_freshness(week: int, season: int = 2025) -> dict:
    """Get freshness status of lines data for a week."""
    lines_path = DATA_DIR / f'odds_week{week}_draftkings.csv'
    simple_path = DATA_DIR / f'odds_week{week}.csv'

    path = lines_path if lines_path.exists() else simple_path

    if not path.exists():
        return {
            'exists': False,
            'week': week,
            'status': 'MISSING'
        }

    file_age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    hours_old = file_age.total_seconds() / 3600

    return {
        'exists': True,
        'path': str(path),
        'week': week,
        'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        'hours_old': round(hours_old, 1),
        'status': 'FRESH' if hours_old < MAX_FILE_AGE_HOURS else 'STALE',
        'threshold_hours': MAX_FILE_AGE_HOURS
    }


def clear_lines_cache() -> None:
    """Clear the lines cache to force reload."""
    global _LINES_CACHE, _CACHE_TIMESTAMPS
    _LINES_CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
    logger.info("Cleared lines cache")
