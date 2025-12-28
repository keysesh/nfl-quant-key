"""
Centralized path configuration for NFL QUANT.

This module provides a single source of truth for all file paths,
eliminating hardcoded relative paths that fail when CWD changes.

Snapshot Isolation:
    When a run_id is set, data paths resolve to runs/<run_id>/inputs/
    instead of the global data/nflverse/ directory. This ensures
    pipeline runs use pinned snapshots, not live data.

Usage:
    from nfl_quant.config_paths import (
        PROJECT_ROOT, DATA_DIR, NFLVERSE_DIR, MODELS_DIR,
        get_pbp_file, get_model_file,
        set_run_context, get_run_context, clear_run_context
    )

    # For production runs with snapshot isolation:
    set_run_context("week17_20251227_120000")
    pbp = get_pbp_path()  # Returns runs/<run_id>/inputs/pbp.parquet

    # For development (global paths):
    clear_run_context()
    pbp = get_pbp_path()  # Returns data/nflverse/pbp.parquet
"""

from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# RUN CONTEXT (for snapshot isolation)
# =============================================================================

_run_context: Optional[str] = None  # Current run_id


def set_run_context(run_id: str) -> None:
    """
    Set the current run context for snapshot isolation.

    When set, data path functions return paths under runs/<run_id>/inputs/
    instead of the global data/nflverse/ directory.

    Args:
        run_id: Unique identifier for the run (e.g., "week17_20251227_120000")
    """
    global _run_context
    _run_context = run_id
    # Also set as env var for subprocesses
    os.environ['NFL_QUANT_RUN_ID'] = run_id
    logger.info(f"Run context set: {run_id}")


def get_run_context() -> Optional[str]:
    """Get the current run context (run_id or None if not set)."""
    global _run_context
    # Check env var first (for subprocesses)
    if _run_context is None:
        _run_context = os.environ.get('NFL_QUANT_RUN_ID')
    return _run_context


def clear_run_context() -> None:
    """Clear the run context, reverting to global paths."""
    global _run_context
    _run_context = None
    if 'NFL_QUANT_RUN_ID' in os.environ:
        del os.environ['NFL_QUANT_RUN_ID']
    logger.info("Run context cleared")


def is_snapshot_mode() -> bool:
    """Check if we're in snapshot mode (run context is set)."""
    return get_run_context() is not None


# =============================================================================
# CORE PATHS
# =============================================================================

# Project root - works regardless of current working directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Core directories
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data subdirectories
NFLVERSE_DIR = DATA_DIR / "nflverse"
MODELS_DIR = DATA_DIR / "models"
INJURIES_DIR = DATA_DIR / "injuries"
BACKTEST_DIR = DATA_DIR / "backtest"
HISTORICAL_DIR = DATA_DIR / "historical"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Commonly referenced NFLverse files
WEEKLY_STATS_FILE = NFLVERSE_DIR / "weekly_stats.parquet"
SNAP_COUNTS_FILE = NFLVERSE_DIR / "snap_counts.parquet"
SCHEDULES_FILE = NFLVERSE_DIR / "schedules.parquet"
DEPTH_CHARTS_FILE = NFLVERSE_DIR / "depth_charts.parquet"
ROSTERS_FILE = NFLVERSE_DIR / "rosters.parquet"
INJURIES_FILE = NFLVERSE_DIR / "injuries.parquet"

# Model files
ACTIVE_MODEL_FILE = MODELS_DIR / "active_model.joblib"
USAGE_PREDICTOR_FILE = MODELS_DIR / "usage_predictor_v4_defense.joblib"
EFFICIENCY_PREDICTOR_FILE = MODELS_DIR / "efficiency_predictor_v2_defense.joblib"
TD_POISSON_MODEL_FILE = MODELS_DIR / "td_poisson_model.joblib"


# =============================================================================
# SNAPSHOT-AWARE PATH FUNCTIONS
# =============================================================================

def get_run_dir() -> Optional[Path]:
    """Get the current run directory, if run context is set."""
    run_id = get_run_context()
    if run_id is None:
        return None
    return PROJECT_ROOT / "runs" / run_id


def get_run_inputs_dir() -> Optional[Path]:
    """Get the inputs directory for the current run."""
    run_dir = get_run_dir()
    if run_dir is None:
        return None
    return run_dir / "inputs"


def get_snapshot_path(filename: str) -> Optional[Path]:
    """
    Get the snapshot path for a file if in snapshot mode.

    Returns the path under runs/<run_id>/inputs/<filename> if the file exists,
    None otherwise.
    """
    inputs_dir = get_run_inputs_dir()
    if inputs_dir is None:
        return None

    snapshot_path = inputs_dir / filename
    if snapshot_path.exists():
        return snapshot_path
    return None


def get_nflverse_path(filename: str, require_snapshot: bool = False) -> Path:
    """
    Get path to an NFLverse data file, respecting snapshot mode.

    Args:
        filename: The filename (e.g., 'weekly_stats.parquet')
        require_snapshot: If True and in snapshot mode, raise error if snapshot missing

    Returns:
        Path to the file (snapshot path if available, otherwise global path)

    Raises:
        FileNotFoundError: If require_snapshot is True, in snapshot mode,
                          and snapshot doesn't exist
    """
    # Check for snapshot first
    if is_snapshot_mode():
        snapshot_path = get_snapshot_path(filename)
        if snapshot_path is not None:
            logger.debug(f"Using snapshot: {snapshot_path}")
            return snapshot_path

        if require_snapshot:
            raise FileNotFoundError(
                f"Snapshot isolation violated: {filename} not found in "
                f"{get_run_inputs_dir()}. Phase 0 may have failed to snapshot this file."
            )
        else:
            logger.warning(
                f"Snapshot mode active but {filename} not in snapshot. "
                f"Falling back to global path."
            )

    # Fall back to global path
    return NFLVERSE_DIR / filename


def get_pbp_path(season: int = None, require_snapshot: bool = False) -> Path:
    """
    Get path to play-by-play file, respecting snapshot mode.

    Args:
        season: Season year (if None, uses 'pbp.parquet')
        require_snapshot: If True, require snapshot when in snapshot mode

    Returns:
        Path to the pbp file
    """
    if season:
        filename = f"pbp_{season}.parquet"
    else:
        filename = "pbp.parquet"
    return get_nflverse_path(filename, require_snapshot=require_snapshot)


def get_weekly_stats_path(require_snapshot: bool = False) -> Path:
    """Get path to weekly stats file, respecting snapshot mode."""
    return get_nflverse_path("weekly_stats.parquet", require_snapshot=require_snapshot)


def get_snap_counts_path(require_snapshot: bool = False) -> Path:
    """Get path to snap counts file, respecting snapshot mode."""
    return get_nflverse_path("snap_counts.parquet", require_snapshot=require_snapshot)


def get_depth_charts_path(require_snapshot: bool = False) -> Path:
    """Get path to depth charts file, respecting snapshot mode."""
    return get_nflverse_path("depth_charts.parquet", require_snapshot=require_snapshot)


def get_rosters_path(require_snapshot: bool = False) -> Path:
    """Get path to rosters file, respecting snapshot mode."""
    return get_nflverse_path("rosters.parquet", require_snapshot=require_snapshot)


def get_schedules_path(require_snapshot: bool = False) -> Path:
    """Get path to schedules file, respecting snapshot mode."""
    return get_nflverse_path("schedules.parquet", require_snapshot=require_snapshot)


def get_injuries_path(require_snapshot: bool = False) -> Path:
    """Get path to injuries file, respecting snapshot mode."""
    # Try snapshot path first
    if is_snapshot_mode():
        snapshot_path = get_snapshot_path("injuries.parquet")
        if snapshot_path is not None:
            return snapshot_path
        if require_snapshot:
            raise FileNotFoundError(
                f"Snapshot isolation violated: injuries.parquet not found in "
                f"{get_run_inputs_dir()}."
            )
    return INJURIES_FILE


# =============================================================================
# LEGACY PATH FUNCTIONS (for backward compatibility)
# =============================================================================

def get_pbp_file(season: int) -> Path:
    """Get play-by-play file for a specific season.

    DEPRECATED: Use get_pbp_path(season) for snapshot support.
    """
    return NFLVERSE_DIR / f"pbp_{season}.parquet"


def get_weekly_stats_file(season: int = None) -> Path:
    """Get weekly stats file, optionally for a specific season."""
    if season:
        return NFLVERSE_DIR / f"weekly_stats_{season}.parquet"
    return WEEKLY_STATS_FILE


def get_roster_file(season: int) -> Path:
    """Get roster file for a specific season."""
    return NFLVERSE_DIR / f"rosters_{season}.csv"


def get_injury_file(season: int) -> Path:
    """Get injury file for a specific season."""
    return INJURIES_DIR / f"injuries_{season}.csv"


def get_odds_file(week: int, season: int = None) -> Path:
    """Get odds file for a specific week."""
    return DATA_DIR / f"odds_week{week}.csv"


def get_player_props_file(week: int) -> Path:
    """Get player props file for a specific week."""
    return DATA_DIR / f"odds_player_props_week{week}.csv"


def get_model_file(model_name: str) -> Path:
    """Get path to a model file by name."""
    return MODELS_DIR / f"{model_name}.joblib"


def get_predictions_file(week: int, season: int = None) -> Path:
    """Get model predictions output file for a week."""
    return DATA_DIR / f"model_predictions_week{week}.csv"


def get_recommendations_file() -> Path:
    """Get current week recommendations file."""
    return REPORTS_DIR / "CURRENT_WEEK_RECOMMENDATIONS.csv"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data_dirs() -> None:
    """Ensure all required data directories exist."""
    for dir_path in [
        DATA_DIR, NFLVERSE_DIR, MODELS_DIR, INJURIES_DIR,
        BACKTEST_DIR, HISTORICAL_DIR, OUTPUTS_DIR, REPORTS_DIR
    ]:
        ensure_dir(dir_path)
