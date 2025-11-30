"""
NFLverse Data Loader - File-based (R-fetched data)
====================================================

Centralized utility for loading NFLverse data from R-fetched parquet/CSV files.
This replaces direct nflreadpy calls with file-based loading for better reliability.

Usage:
    from nfl_quant.utils.nflverse_loader import (
        load_schedules,
        load_rosters,
        load_player_stats,
        load_pbp,
    )

    # Load schedules for season
    schedules = load_schedules(season=2025)

    # Load player stats
    stats = load_player_stats(seasons=[2024, 2025])

Data fetching:
    Run: Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last
    This creates parquet files in data/nflverse/
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Default data directory
NFLVERSE_DIR = Path("data/nflverse")


def _load_file(file_patterns: List[Path], description: str = "data") -> pd.DataFrame:
    """Load first available file from list of patterns."""
    for filepath in file_patterns:
        if filepath.exists():
            logger.debug(f"Loading {description} from {filepath}")
            if filepath.suffix == ".parquet":
                return pd.read_parquet(filepath)
            elif filepath.suffix == ".csv":
                return pd.read_csv(filepath, low_memory=False)
            elif filepath.suffix == ".rds":
                logger.warning(f"RDS files require R to read. Use parquet or CSV instead.")
                continue
    raise FileNotFoundError(
        f"{description} not found. Tried: {[str(p) for p in file_patterns]}\n"
        f"Please fetch data with R first:\n"
        f"  Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last"
    )


def load_schedules(
    seasons: Optional[Union[int, List[int]]] = None,
    data_dir: Path = NFLVERSE_DIR
) -> pd.DataFrame:
    """Load NFL schedules from R-fetched data.

    Args:
        seasons: Season(s) to filter (e.g., 2025 or [2024, 2025])
        data_dir: Directory containing NFLverse data files

    Returns:
        DataFrame with schedule data
    """
    file_patterns = [
        data_dir / "schedules.parquet",
        data_dir / "schedules.csv",
        data_dir / "games.parquet",
    ]

    df = _load_file(file_patterns, "schedules")

    # Filter by season if requested
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        if "season" in df.columns:
            df = df[df["season"].isin(seasons)].copy()

    logger.info(f"Loaded {len(df)} schedule records")
    return df


def load_rosters(
    seasons: Optional[Union[int, List[int]]] = None,
    data_dir: Path = NFLVERSE_DIR
) -> pd.DataFrame:
    """Load NFL rosters from R-fetched data.

    Args:
        seasons: Season(s) to filter (e.g., 2025 or [2024, 2025])
        data_dir: Directory containing NFLverse data files

    Returns:
        DataFrame with roster data
    """
    file_patterns = [
        data_dir / "rosters.parquet",
        data_dir / "rosters.csv",
    ]

    df = _load_file(file_patterns, "rosters")

    # Filter by season if requested
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        if "season" in df.columns:
            df = df[df["season"].isin(seasons)].copy()

    logger.info(f"Loaded {len(df)} roster records")
    return df


def load_player_stats(
    seasons: Optional[Union[int, List[int]]] = None,
    data_dir: Path = NFLVERSE_DIR
) -> pd.DataFrame:
    """Load NFL player stats from R-fetched data.

    Args:
        seasons: Season(s) to filter (e.g., 2025 or [2024, 2025])
        data_dir: Directory containing NFLverse data files

    Returns:
        DataFrame with player stats
    """
    file_patterns = [
        data_dir / "player_stats.parquet",
        data_dir / "player_stats.csv",
        data_dir / "weekly_stats.parquet",
    ]

    df = _load_file(file_patterns, "player_stats")

    # Filter by season if requested
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        if "season" in df.columns:
            df = df[df["season"].isin(seasons)].copy()

    logger.info(f"Loaded {len(df)} player stat records")
    return df


def load_pbp(
    seasons: Optional[Union[int, List[int]]] = None,
    data_dir: Path = NFLVERSE_DIR
) -> pd.DataFrame:
    """Load NFL play-by-play data from R-fetched data.

    Args:
        seasons: Season(s) to filter (e.g., 2025 or [2024, 2025])
        data_dir: Directory containing NFLverse data files

    Returns:
        DataFrame with PBP data
    """
    # Try season-specific files first
    if seasons is not None and isinstance(seasons, int):
        season_file = data_dir / f"pbp_{seasons}.parquet"
        if season_file.exists():
            df = pd.read_parquet(season_file)
            logger.info(f"Loaded {len(df)} plays for season {seasons}")
            return df

    file_patterns = [
        data_dir / "pbp.parquet",
        data_dir / "pbp_historical.parquet",
        data_dir / "pbp.csv",
    ]

    df = _load_file(file_patterns, "play-by-play")

    # Filter by season if requested
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        if "season" in df.columns:
            df = df[df["season"].isin(seasons)].copy()

    logger.info(f"Loaded {len(df)} play records")
    return df


def load_injuries(
    data_dir: Path = Path("data/injuries")
) -> pd.DataFrame:
    """Load current injuries from local cache.

    Note: Injuries are fetched separately via scripts/fetch/fetch_injuries_api.py

    Args:
        data_dir: Directory containing injury data

    Returns:
        DataFrame with injury data
    """
    injury_file = data_dir / "current_injuries.csv"
    if not injury_file.exists():
        logger.warning(
            f"No injuries file found at {injury_file}. "
            f"Run: python scripts/fetch/fetch_injuries_api.py"
        )
        return pd.DataFrame()

    df = pd.read_csv(injury_file)
    logger.info(f"Loaded {len(df)} injury records")
    return df


def check_data_availability(data_dir: Path = NFLVERSE_DIR) -> dict:
    """Check which NFLverse data files are available.

    Returns:
        Dictionary with file availability status
    """
    files_to_check = {
        "schedules": ["schedules.parquet", "schedules.csv", "games.parquet"],
        "rosters": ["rosters.parquet", "rosters.csv"],
        "player_stats": ["player_stats.parquet", "player_stats.csv"],
        "pbp": ["pbp.parquet", "pbp_historical.parquet", "pbp.csv"],
    }

    status = {}
    for data_type, filenames in files_to_check.items():
        found = False
        for filename in filenames:
            filepath = data_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                status[data_type] = {"file": str(filepath), "size_mb": round(size_mb, 2)}
                found = True
                break
        if not found:
            status[data_type] = {"file": None, "size_mb": 0}

    return status
