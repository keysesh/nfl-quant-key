"""
Centralized Game Line Odds Loader

Provides consistent interface for loading game line odds across all scripts.
Supports both historical (master file) and live (current week) data sources.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_game_lines(
    season: int,
    weeks: Optional[Union[int, List[int]]] = None,
    source: str = "auto"
) -> pd.DataFrame:
    """
    Load game line odds (spread, total, moneyline) with automatic source detection.

    Args:
        season: Season year (e.g., 2025)
        weeks: Single week number, list of weeks, or None for all available
        source: Data source preference:
            - "auto" (default): Try master file first, fall back to individual files
            - "master": Only use master consolidated file
            - "weekly": Only use individual week files
            - "live": Use current live odds file

    Returns:
        DataFrame with columns:
            - season, week, gameday
            - game_id, home_team, away_team
            - market (spread/total/moneyline), side, point, price
            - sportsbook, data_source, collected_at

    Raises:
        FileNotFoundError: If no odds data found for specified parameters

    Examples:
        >>> # Load all weeks from master file (best for backtesting)
        >>> odds = load_game_lines(season=2025)

        >>> # Load specific weeks
        >>> odds = load_game_lines(season=2025, weeks=[8, 9, 10])

        >>> # Load current week for live predictions
        >>> odds = load_game_lines(season=2025, weeks=12, source="live")
    """
    # Normalize weeks to list
    if weeks is None:
        weeks_list = None
    elif isinstance(weeks, int):
        weeks_list = [weeks]
    else:
        weeks_list = list(weeks)

    # Try different sources based on preference
    if source == "auto":
        # Try master file first (most comprehensive)
        try:
            return _load_from_master(season, weeks_list)
        except FileNotFoundError:
            pass

        # Fall back to weekly files
        try:
            return _load_from_weekly_files(season, weeks_list)
        except FileNotFoundError:
            pass

        # Fall back to live file if single current week
        if weeks_list and len(weeks_list) == 1:
            try:
                return _load_from_live(weeks_list[0])
            except FileNotFoundError:
                pass

        raise FileNotFoundError(
            f"No game line odds found for season {season}, weeks {weeks_list}"
        )

    elif source == "master":
        return _load_from_master(season, weeks_list)

    elif source == "weekly":
        return _load_from_weekly_files(season, weeks_list)

    elif source == "live":
        if not weeks_list or len(weeks_list) != 1:
            raise ValueError("source='live' requires exactly one week")
        return _load_from_live(weeks_list[0])

    else:
        raise ValueError(f"Invalid source: {source}. Must be 'auto', 'master', 'weekly', or 'live'")


def _load_from_master(season: int, weeks: Optional[List[int]] = None) -> pd.DataFrame:
    """Load from consolidated master file."""
    master_file = PROJECT_ROOT / "data" / "historical" / f"game_lines_master_{season}.csv"

    if not master_file.exists():
        raise FileNotFoundError(f"Master file not found: {master_file}")

    df = pd.read_csv(master_file)

    # Filter to requested weeks if specified
    if weeks is not None:
        df = df[df['week'].isin(weeks)].copy()

    if len(df) == 0:
        raise FileNotFoundError(f"No data found in master file for weeks {weeks}")

    # Standardize column names
    if 'game_id_nflverse' in df.columns:
        df = df.rename(columns={'game_id_nflverse': 'game_id'})

    return df


def _load_from_weekly_files(season: int, weeks: Optional[List[int]] = None) -> pd.DataFrame:
    """Load from individual week files in historical archive."""
    game_lines_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"

    if not game_lines_dir.exists():
        raise FileNotFoundError(f"Game lines directory not found: {game_lines_dir}")

    # If no weeks specified, try to load all available
    if weeks is None:
        week_files = sorted(game_lines_dir.glob(f"game_lines_{season}_week*.csv"))
        if not week_files:
            raise FileNotFoundError(f"No weekly game line files found for {season}")
    else:
        week_files = [
            game_lines_dir / f"game_lines_{season}_week{week:02d}.csv"
            for week in weeks
        ]

    # Load all matching files
    odds_dfs = []
    for week_file in week_files:
        if week_file.exists():
            df = pd.read_csv(week_file)
            odds_dfs.append(df)

    if not odds_dfs:
        raise FileNotFoundError(f"No game line files found for {season} weeks {weeks}")

    combined = pd.concat(odds_dfs, ignore_index=True)

    # Standardize column names
    if 'game_id_nflverse' in combined.columns:
        combined = combined.rename(columns={'game_id_nflverse': 'game_id'})

    return combined


def _load_from_live(week: int) -> pd.DataFrame:
    """Load current week's live odds file."""
    live_file = PROJECT_ROOT / "data" / f"odds_week{week}.csv"

    if not live_file.exists():
        raise FileNotFoundError(
            f"Live odds file not found: {live_file}\n"
            f"Run fetch_comprehensive_odds.py to fetch current odds"
        )

    df = pd.read_csv(live_file)

    # Filter to game lines only (exclude player props if present)
    game_line_markets = ['spread', 'total', 'moneyline', 'spreads', 'totals', 'h2h']
    if 'market' in df.columns:
        df = df[df['market'].isin(game_line_markets)].copy()

    if len(df) == 0:
        raise FileNotFoundError(f"No game line odds found in {live_file}")

    return df


def get_coverage_summary(season: int, source: str = "master") -> pd.DataFrame:
    """
    Get coverage summary showing which weeks have odds data available.

    Args:
        season: Season year
        source: Data source to check ("master" or "weekly")

    Returns:
        DataFrame with columns: week, games_count, odds_count, markets
    """
    try:
        if source == "master":
            df = _load_from_master(season, weeks=None)
        else:
            df = _load_from_weekly_files(season, weeks=None)

        summary = df.groupby('week').agg({
            'game_id': 'nunique',
            'market': lambda x: list(x.unique())
        }).reset_index()

        summary.columns = ['week', 'games_count', 'markets']
        summary['odds_count'] = df.groupby('week').size().values

        return summary.sort_values('week')

    except FileNotFoundError:
        return pd.DataFrame(columns=['week', 'games_count', 'odds_count', 'markets'])


def validate_game_lines_format(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has expected game lines format.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If DataFrame is missing required columns or has invalid values
    """
    required_columns = [
        'home_team', 'away_team',
        'market', 'side', 'price'
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check valid markets
    valid_markets = {'spread', 'spreads', 'total', 'totals', 'moneyline', 'h2h'}
    invalid_markets = set(df['market'].unique()) - valid_markets
    if invalid_markets:
        raise ValueError(f"Invalid markets found: {invalid_markets}")

    # Check for null values in critical columns
    for col in ['home_team', 'away_team', 'market', 'price']:
        if df[col].isna().any():
            raise ValueError(f"Null values found in column: {col}")

    return True


# Convenience aliases
load_historical_game_lines = load_game_lines  # Alias for clarity
load_live_game_lines = lambda week: load_game_lines(season=2025, weeks=week, source="live")
