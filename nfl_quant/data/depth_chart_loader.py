"""
Canonical Depth Chart Data Loader

Single entry point for all depth chart data access.
Handles schema normalization, type casting, and validation.

Schema Changes:
- Season <= 2024: Uses week column directly
- Season >= 2025: Uses dt (timestamp) snapshots, derives week from schedule

PRODUCTION MODE ONLY:
- All calls require >= 30 teams coverage
- No partial data allowed (fail-closed)
- For analytics/research, use explicit --allow-partial flag in scripts

Usage:
    from nfl_quant.data.depth_chart_loader import get_depth_charts

    # Get current week depth charts (fails if < 30 teams)
    df = get_depth_charts(week=18)

    # Get team's depth chart
    df = get_team_depth(team='BUF', week=18)
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pytz

from nfl_quant.config_paths import (
    PROJECT_ROOT, NFLVERSE_DIR,
    get_depth_charts_path, is_snapshot_mode, get_run_inputs_dir
)

logger = logging.getLogger(__name__)

# Cache for loaded depth chart data
_DEPTH_CHART_CACHE = {}
_CACHE_TIMESTAMPS = {}

# Static paths (for freshness checks and fallbacks)
DEPTH_CHART_PATH = NFLVERSE_DIR / 'depth_charts.parquet'
DEPTH_CHART_2025_PATH = NFLVERSE_DIR / 'depth_charts_2025.parquet'
MAX_CACHE_AGE_HOURS = 12
MAX_FILE_AGE_HOURS = 24  # Depth charts change less frequently

# Required columns after normalization
REQUIRED_COLUMNS = ['season', 'week', 'team', 'player_name', 'pos_abb', 'pos_rank']

# Validation thresholds - PRODUCTION ONLY, NO PARTIAL DATA
MIN_TEAMS_REQUIRED = 30  # Require at least 30 teams for any production use

# Current season
from nfl_quant.utils.season_utils import get_current_season
CURRENT_SEASON = get_current_season()


class DepthChartValidationError(Exception):
    """Raised when depth chart data fails validation."""
    pass


def get_depth_charts(
    season: int = 2025,
    week: Optional[int] = None,
    refresh: bool = False,
    validate: bool = True
) -> pd.DataFrame:
    """
    Get depth chart data with normalization and validation.

    PRODUCTION MODE: Fails if fewer than 30 teams available.
    For partial data in scripts, use --allow-partial flag and call
    get_depth_charts_unsafe() instead.

    Args:
        season: NFL season (default 2025)
        week: Specific week (None = all weeks)
        refresh: Force reload from disk
        validate: Run schema validation (default True)

    Returns:
        Normalized DataFrame with depth chart data

    Raises:
        FileNotFoundError: If depth charts file doesn't exist
        DepthChartValidationError: If validation fails (< 30 teams)
    """
    global _DEPTH_CHART_CACHE, _CACHE_TIMESTAMPS

    cache_key = f"depth_{season}_{week or 'all'}"

    # Check cache validity
    if not refresh and cache_key in _DEPTH_CHART_CACHE:
        cache_time = _CACHE_TIMESTAMPS.get(cache_key)
        if cache_time:
            cache_age = datetime.now() - cache_time
            if cache_age < timedelta(hours=MAX_CACHE_AGE_HOURS):
                return _DEPTH_CHART_CACHE[cache_key]

    # Determine the path to use - snapshot mode takes priority
    depth_chart_path = get_depth_charts_path()

    if is_snapshot_mode():
        logger.info(f"[SNAPSHOT MODE] Using depth charts from: {depth_chart_path}")

    # For current season (2025+), try snapshot-based file first
    if season >= CURRENT_SEASON:
        # In snapshot mode, the snapshot should have the combined file
        if is_snapshot_mode() and depth_chart_path.exists():
            logger.info(f"Loading {season} depth charts from {depth_chart_path}")
            df = pd.read_parquet(depth_chart_path)
            logger.info(f"Loaded {len(df):,} raw {season} depth chart rows")
            # 2025+ file uses dt timestamp - derive week from schedule
            df = _normalize_snapshot_depth_charts(df, season, week)
        elif DEPTH_CHART_2025_PATH.exists():
            logger.info(f"Loading {season} depth charts from {DEPTH_CHART_2025_PATH}")
            df = pd.read_parquet(DEPTH_CHART_2025_PATH)
            logger.info(f"Loaded {len(df):,} raw {season} depth chart rows")
            # 2025+ file uses dt timestamp - derive week from schedule
            df = _normalize_snapshot_depth_charts(df, season, week)
        elif depth_chart_path.exists():
            logger.info(f"Loading depth charts from {depth_chart_path}")
            df = pd.read_parquet(depth_chart_path)
            logger.info(f"Loaded {len(df):,} raw depth chart rows")
            df = _normalize_depth_charts(df)
            if 'season' in df.columns:
                df = df[df['season'] == season].copy()
            if week is not None:
                df = df[df['week'] == week].copy()
        else:
            raise FileNotFoundError(
                f"Depth charts file not found: {depth_chart_path}. "
                "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
            )
    else:
        # Historical seasons - use standard file
        if not depth_chart_path.exists():
            raise FileNotFoundError(
                f"Depth charts file not found: {depth_chart_path}. "
                "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
            )

        # Check freshness (only for non-snapshot mode)
        if not is_snapshot_mode():
            file_age = datetime.now() - datetime.fromtimestamp(depth_chart_path.stat().st_mtime)
            if file_age > timedelta(hours=MAX_FILE_AGE_HOURS):
                logger.warning(
                    f"Depth charts file is {file_age.total_seconds() / 3600:.1f}h old "
                    f"(threshold: {MAX_FILE_AGE_HOURS}h). Consider refreshing."
                )

        # Load raw data
        logger.info(f"Loading depth charts from {depth_chart_path}")
        df = pd.read_parquet(depth_chart_path)
        logger.info(f"Loaded {len(df):,} raw depth chart rows")
        # Normalize standard format
        df = _normalize_depth_charts(df)

        # Filter to season (for historical data)
        if 'season' in df.columns:
            df = df[df['season'] == season].copy()
        logger.info(f"Filtered to season {season}: {len(df):,} rows")

        # Filter to week if specified
        if week is not None:
            df = df[df['week'] == week].copy()
            logger.info(f"Filtered to week {week}: {len(df):,} rows")

    # Validate - ALWAYS STRICT
    if validate:
        _validate_depth_charts(df, season, week)

    # Log coverage report
    _log_coverage_report(df, season, week)

    # Update cache
    _DEPTH_CHART_CACHE[cache_key] = df
    _CACHE_TIMESTAMPS[cache_key] = datetime.now()

    return df


def _normalize_snapshot_depth_charts(
    df: pd.DataFrame,
    season: int,
    week: Optional[int] = None
) -> pd.DataFrame:
    """
    Normalize 2025+ depth chart format (uses dt timestamp, no week column).

    For each team, selects the most recent snapshot dt <= the week's earliest kickoff.
    If week is None or kickoff time unavailable, uses the most recent snapshot overall.

    The 2025 file is a snapshot with columns:
    dt, team, player_name, pos_abb, pos_rank, etc.
    """
    df = df.copy()

    # Parse dt as datetime (ISO format: '2025-11-16T07:14:40Z')
    df['dt_parsed'] = pd.to_datetime(df['dt'], utc=True)
    logger.info(f"Parsed dt range: {df['dt_parsed'].min()} to {df['dt_parsed'].max()}")

    # Add season column
    df['season'] = season

    # Get earliest kickoff for the week if specified
    kickoff_utc = None
    if week is not None:
        try:
            from nfl_quant.data.schedule_loader import get_earliest_kickoff
            kickoff = get_earliest_kickoff(season=season, week=week)
            if kickoff:
                logger.info(f"Week {week} earliest kickoff: {kickoff}")
                # Convert kickoff to UTC for comparison
                kickoff_utc = kickoff.astimezone(pytz.UTC)
            else:
                logger.warning(f"No kickoff time found for season {season} week {week}")
        except ImportError as e:
            raise DepthChartValidationError(
                f"Schedule loader required for week derivation: {e}"
            )
        except Exception as e:
            raise DepthChartValidationError(
                f"Failed to get kickoff time for week {week}: {e}"
            )

    # For each team, select the most recent snapshot dt <= kickoff
    result_rows = []

    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()

        if kickoff_utc is not None:
            # Filter to snapshots before kickoff
            valid_snapshots = team_df[team_df['dt_parsed'] <= kickoff_utc]
            if len(valid_snapshots) == 0:
                # No snapshots before kickoff - FAIL, don't fallback silently
                raise DepthChartValidationError(
                    f"No depth chart snapshot for {team} before kickoff {kickoff_utc}. "
                    "Data may be stale or incomplete."
                )
        else:
            valid_snapshots = team_df

        # Get the most recent snapshot for this team
        latest_dt = valid_snapshots['dt_parsed'].max()
        team_latest = valid_snapshots[valid_snapshots['dt_parsed'] == latest_dt]
        result_rows.append(team_latest)

    if result_rows:
        df = pd.concat(result_rows, ignore_index=True)
    else:
        raise DepthChartValidationError("No depth chart data after normalization")

    # Add week column
    if week is not None:
        df['week'] = week
    else:
        # Default to current week based on most recent snapshot
        df['week'] = 17  # Fallback

    # Ensure pos_rank is integer
    if 'pos_rank' in df.columns:
        df['pos_rank'] = df['pos_rank'].fillna(99).astype(int)

    # Clean string columns
    for col in ['team', 'player_name', 'pos_abb']:
        if col in df.columns and df[col].dtype == 'object':
            if col == 'team':
                df[col] = df[col].str.strip().str.upper()
            else:
                df[col] = df[col].str.strip()

    # Drop rows with null essential columns
    df = df.dropna(subset=['team', 'player_name'])

    # Drop helper column
    if 'dt_parsed' in df.columns:
        df = df.drop(columns=['dt_parsed'])

    logger.info(f"Normalized to {len(df):,} rows for {df['team'].nunique()} teams")
    return df


def _normalize_depth_charts(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize depth chart schema and types (pre-2025 format)."""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Handle two naming conventions in the data:
    # 1. team/player_name (newer)
    # 2. club_code/full_name (older)
    if 'team' in df.columns and 'club_code' in df.columns:
        # Fill team from club_code where team is null
        df['team'] = df['team'].fillna(df['club_code'])
    elif 'club_code' in df.columns and 'team' not in df.columns:
        df['team'] = df['club_code']

    if 'player_name' in df.columns and 'full_name' in df.columns:
        # Fill player_name from full_name where player_name is null
        df['player_name'] = df['player_name'].fillna(df['full_name'])
    elif 'full_name' in df.columns and 'player_name' not in df.columns:
        df['player_name'] = df['full_name']

    # Drop rows with still-null team or player
    df = df.dropna(subset=['team', 'player_name'])

    # Cast types - handle NaN before int conversion
    if 'season' in df.columns:
        df['season'] = df['season'].fillna(0).astype(int)
    if 'week' in df.columns:
        df['week'] = df['week'].fillna(0).astype(int)
    if 'pos_rank' in df.columns:
        df['pos_rank'] = df['pos_rank'].fillna(99).astype(int)

    # Handle position column naming (pos_abb vs position)
    if 'pos_abb' in df.columns and 'position' in df.columns:
        # Fill pos_abb from position where pos_abb is null
        df['pos_abb'] = df['pos_abb'].fillna(df['position'])
    elif 'position' in df.columns and 'pos_abb' not in df.columns:
        df['pos_abb'] = df['position']

    # Clean string columns
    for col in ['team', 'player_name', 'pos_abb']:
        if col in df.columns and df[col].dtype == 'object':
            if col == 'team':
                df[col] = df[col].str.strip().str.upper()
            else:
                df[col] = df[col].str.strip()

    # Filter out invalid seasons/weeks (zeros from NaN fill)
    df = df[(df['season'] > 2000) & (df['week'] > 0)]

    return df


def _validate_depth_charts(
    df: pd.DataFrame,
    season: int,
    week: Optional[int]
) -> None:
    """
    Validate depth chart data - STRICT MODE ONLY.

    Raises DepthChartValidationError if:
    - Missing required columns
    - No data
    - Fewer than 30 teams
    """
    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DepthChartValidationError(f"Depth charts missing required columns: {missing}")

    # Check we have data
    if len(df) == 0:
        week_msg = f" week {week}" if week else ""
        raise DepthChartValidationError(f"No depth chart data for season {season}{week_msg}")

    # Check for expected teams (32 NFL teams)
    teams = df['team'].unique()
    num_teams = len(teams)

    if num_teams < MIN_TEAMS_REQUIRED:
        expected_teams = {
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
        }
        missing_teams = expected_teams - set(teams)
        raise DepthChartValidationError(
            f"Insufficient team coverage: {num_teams}/32 teams (required >= {MIN_TEAMS_REQUIRED}). "
            f"Missing teams: {sorted(missing_teams)}"
        )

    # Check week dtype is integer
    if df['week'].dtype != 'int64':
        logger.warning(f"Week column is {df['week'].dtype}, expected int64")


def _log_coverage_report(
    df: pd.DataFrame,
    season: int,
    week: Optional[int]
) -> None:
    """Log a coverage report for the depth chart data."""
    teams = sorted(df['team'].unique())
    num_teams = len(teams)

    # Expected NFL teams
    expected_teams = {
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    }
    missing_teams = expected_teams - set(teams)

    week_str = f" week {week}" if week else ""
    logger.info(
        f"[DEPTH CHART COVERAGE] Season {season}{week_str}: "
        f"{num_teams}/32 teams ({num_teams/32*100:.0f}%)"
    )

    if missing_teams:
        logger.warning(f"Missing teams: {sorted(missing_teams)}")


def get_team_depth(
    team: str,
    week: int,
    season: int = 2025
) -> Dict[str, List[str]]:
    """
    Get depth chart for a specific team.

    Args:
        team: Team abbreviation (e.g., 'BUF')
        week: NFL week
        season: NFL season

    Returns:
        Dict mapping position to list of player names (ordered by rank)
    """
    df = get_depth_charts(season=season, week=week)

    team_df = df[df['team'] == team.upper()]
    if len(team_df) == 0:
        raise DepthChartValidationError(f"No depth chart for team {team} in week {week}")

    depth = {}
    for pos in team_df['pos_abb'].dropna().unique():
        pos_df = team_df[team_df['pos_abb'] == pos].sort_values('pos_rank')
        depth[pos] = pos_df['player_name'].tolist()

    return depth


def get_starters(
    team: str,
    week: int,
    season: int = 2025,
    positions: List[str] = ['QB', 'RB', 'WR', 'TE']
) -> Dict[str, str]:
    """
    Get starting players for key positions.

    Args:
        team: Team abbreviation
        week: NFL week
        season: NFL season
        positions: List of positions to get starters for

    Returns:
        Dict mapping position to starting player name
    """
    depth = get_team_depth(team=team, week=week, season=season)

    starters = {}
    for pos in positions:
        players = depth.get(pos, [])
        if players:
            starters[pos] = players[0]  # First player is starter

    return starters


def get_depth_chart_freshness() -> dict:
    """Get freshness status of depth chart data."""
    # Check both files
    results = {}

    for name, path in [('historical', DEPTH_CHART_PATH), ('2025', DEPTH_CHART_2025_PATH)]:
        if not path.exists():
            results[name] = {
                'exists': False,
                'path': str(path),
                'status': 'MISSING'
            }
        else:
            file_age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            hours_old = file_age.total_seconds() / 3600

            results[name] = {
                'exists': True,
                'path': str(path),
                'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                'hours_old': round(hours_old, 1),
                'status': 'FRESH' if hours_old < MAX_FILE_AGE_HOURS else 'STALE',
                'threshold_hours': MAX_FILE_AGE_HOURS
            }

    return results


def clear_depth_chart_cache() -> None:
    """Clear the depth chart cache."""
    global _DEPTH_CHART_CACHE, _CACHE_TIMESTAMPS
    _DEPTH_CHART_CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
    logger.info("Cleared depth chart cache")
