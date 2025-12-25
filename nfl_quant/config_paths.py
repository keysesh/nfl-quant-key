"""
Centralized path configuration for NFL QUANT.

This module provides a single source of truth for all file paths,
eliminating hardcoded relative paths that fail when CWD changes.

Usage:
    from nfl_quant.config_paths import (
        PROJECT_ROOT, DATA_DIR, NFLVERSE_DIR, MODELS_DIR,
        get_pbp_file, get_model_file
    )
"""

from pathlib import Path


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


def get_pbp_file(season: int) -> Path:
    """Get play-by-play file for a specific season."""
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
