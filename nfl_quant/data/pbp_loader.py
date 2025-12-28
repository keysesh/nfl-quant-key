"""
Canonical PBP Data Loader

Single entry point for all play-by-play data access.
Enforces freshness, schema validation, and caching.

Usage:
    from nfl_quant.data.pbp_loader import get_pbp

    # Get current season PBP
    df = get_pbp()

    # Get specific season
    df = get_pbp(season=2024)

    # Force refresh
    df = get_pbp(refresh=True)
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for loaded PBP data
_PBP_CACHE = {}
_CACHE_TIMESTAMPS = {}

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
PBP_PATH = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
MAX_CACHE_AGE_HOURS = 4  # Cache valid for 4 hours during active use
MAX_FILE_AGE_HOURS = 12  # File considered stale after 12 hours

# Required columns for schema validation
REQUIRED_COLUMNS = [
    'season', 'week', 'game_id', 'play_id',
    'posteam', 'defteam', 'play_type',
    'epa', 'success', 'wp', 'wpa'
]


def get_pbp(
    season: Optional[int] = None,
    refresh: bool = False,
    validate: bool = True
) -> pd.DataFrame:
    """
    Get play-by-play data with caching and validation.

    Args:
        season: Filter to specific season (None = all seasons)
        refresh: Force reload from disk
        validate: Run schema validation checks

    Returns:
        DataFrame with PBP data

    Raises:
        FileNotFoundError: If PBP file doesn't exist
        ValueError: If schema validation fails
    """
    global _PBP_CACHE, _CACHE_TIMESTAMPS

    cache_key = f"pbp_{season or 'all'}"

    # Check cache validity
    if not refresh and cache_key in _PBP_CACHE:
        cache_time = _CACHE_TIMESTAMPS.get(cache_key)
        if cache_time:
            cache_age = datetime.now() - cache_time
            if cache_age < timedelta(hours=MAX_CACHE_AGE_HOURS):
                logger.debug(f"Using cached PBP data (age: {cache_age})")
                return _PBP_CACHE[cache_key]

    # Check file exists
    if not PBP_PATH.exists():
        raise FileNotFoundError(
            f"PBP file not found: {PBP_PATH}. "
            "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
        )

    # Check file freshness
    file_age = datetime.now() - datetime.fromtimestamp(PBP_PATH.stat().st_mtime)
    if file_age > timedelta(hours=MAX_FILE_AGE_HOURS):
        logger.warning(
            f"PBP file is {file_age.total_seconds() / 3600:.1f}h old "
            f"(threshold: {MAX_FILE_AGE_HOURS}h). Consider refreshing."
        )

    # Load data
    logger.info(f"Loading PBP data from {PBP_PATH}")
    df = pd.read_parquet(PBP_PATH)
    logger.info(f"Loaded {len(df):,} plays")

    # Filter to season if specified
    if season is not None:
        if 'season' not in df.columns:
            raise ValueError("PBP data missing 'season' column")
        df = df[df['season'] == season].copy()
        logger.info(f"Filtered to season {season}: {len(df):,} plays")

    # Validate schema
    if validate:
        _validate_schema(df)

    # Update cache
    _PBP_CACHE[cache_key] = df
    _CACHE_TIMESTAMPS[cache_key] = datetime.now()

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate PBP DataFrame has required columns and valid data."""
    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"PBP data missing required columns: {missing}")

    # Check for duplicate play keys
    if 'game_id' in df.columns and 'play_id' in df.columns:
        duplicates = df.duplicated(subset=['game_id', 'play_id'], keep=False)
        dup_count = duplicates.sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate (game_id, play_id) pairs")

    # Check EPA range (sanity check)
    if 'epa' in df.columns:
        epa_valid = df['epa'].dropna()
        if len(epa_valid) > 0:
            if epa_valid.min() < -20 or epa_valid.max() > 20:
                logger.warning(
                    f"EPA values outside expected range: "
                    f"[{epa_valid.min():.2f}, {epa_valid.max():.2f}]"
                )


def get_pbp_freshness() -> dict:
    """Get freshness status of PBP data."""
    if not PBP_PATH.exists():
        return {
            'exists': False,
            'path': str(PBP_PATH),
            'status': 'MISSING'
        }

    file_age = datetime.now() - datetime.fromtimestamp(PBP_PATH.stat().st_mtime)
    hours_old = file_age.total_seconds() / 3600

    return {
        'exists': True,
        'path': str(PBP_PATH),
        'modified': datetime.fromtimestamp(PBP_PATH.stat().st_mtime).isoformat(),
        'hours_old': round(hours_old, 1),
        'status': 'FRESH' if hours_old < MAX_FILE_AGE_HOURS else 'STALE',
        'threshold_hours': MAX_FILE_AGE_HOURS
    }


def clear_pbp_cache() -> None:
    """Clear the PBP cache to force reload."""
    global _PBP_CACHE, _CACHE_TIMESTAMPS
    _PBP_CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
    logger.info("Cleared PBP cache")


# Convenience function for quick access
def get_current_season_pbp(refresh: bool = False) -> pd.DataFrame:
    """Get PBP data for the current NFL season only."""
    from nfl_quant.utils.season_utils import get_current_season
    return get_pbp(season=get_current_season(), refresh=refresh)
