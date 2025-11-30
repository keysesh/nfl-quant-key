"""
Universal Stats Schema for NFL Player Stats

This module defines the canonical schema that all stats data sources
(NFLverse, Sleeper, etc.) will be transformed into.

Key Design Principles:
1. Source-agnostic: Same schema regardless of data source
2. Complete: Includes all stats needed for prediction and backtesting
3. Extensible: Easy to add new stat types
4. Parquet-first: Optimized for efficient storage and reading
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# Canonical column names (standardized across all sources)
CANONICAL_COLUMNS = [
    # Identifiers
    "player_id",          # Unique player identifier
    "player_name",        # Full display name
    "position",           # QB, RB, WR, TE, etc.
    "team",               # 3-letter team abbreviation
    "season",             # Year (2023, 2024, 2025)
    "week",               # Week number (1-18)

    # Passing stats
    "passing_yards",
    "passing_attempts",
    "passing_completions",
    "passing_tds",
    "interceptions",

    # Rushing stats
    "rushing_yards",
    "rushing_attempts",
    "rushing_tds",

    # Receiving stats
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "targets",

    # Additional fields (optional)
    "opponent",           # Opponent team abbreviation
    "game_id",            # Unique game identifier
    "source",             # Data source (nflverse, sleeper, etc.)
]


@dataclass
class PlayerWeeklyStats:
    """
    Canonical representation of a player's weekly stats.

    This is the universal format that all data sources transform into.
    """
    # Identifiers
    player_id: str
    player_name: str
    position: str
    team: str
    season: int
    week: int

    # Passing stats
    passing_yards: float = 0.0
    passing_attempts: float = 0.0
    passing_completions: float = 0.0
    passing_tds: int = 0
    interceptions: int = 0

    # Rushing stats
    rushing_yards: float = 0.0
    rushing_attempts: float = 0.0
    rushing_tds: int = 0

    # Receiving stats
    receptions: int = 0
    receiving_yards: float = 0.0
    receiving_tds: int = 0
    targets: int = 0

    # Optional fields
    opponent: Optional[str] = None
    game_id: Optional[str] = None
    source: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "position": self.position,
            "team": self.team,
            "season": self.season,
            "week": self.week,
            "passing_yards": self.passing_yards,
            "passing_attempts": self.passing_attempts,
            "passing_completions": self.passing_completions,
            "passing_tds": self.passing_tds,
            "interceptions": self.interceptions,
            "rushing_yards": self.rushing_yards,
            "rushing_attempts": self.rushing_attempts,
            "rushing_tds": self.rushing_tds,
            "receptions": self.receptions,
            "receiving_yards": self.receiving_yards,
            "receiving_tds": self.receiving_tds,
            "targets": self.targets,
            "opponent": self.opponent,
            "game_id": self.game_id,
            "source": self.source,
        }


def validate_stats_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame conforms to the canonical schema.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_columns = [
        "player_id", "player_name", "position", "team", "season", "week",
        "passing_yards", "passing_attempts", "passing_completions", "passing_tds", "interceptions",
        "rushing_yards", "rushing_attempts", "rushing_tds",
        "receptions", "receiving_yards", "receiving_tds", "targets"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate data types
    if not pd.api.types.is_integer_dtype(df['season']):
        raise ValueError("season must be integer")
    if not pd.api.types.is_integer_dtype(df['week']):
        raise ValueError("week must be integer")

    return True


def create_canonical_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Create a canonical stats DataFrame from a list of records.

    Args:
        records: List of dictionaries with stat data

    Returns:
        DataFrame in canonical format
    """
    df = pd.DataFrame(records)

    # Ensure all canonical columns exist
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            if col in ["opponent", "game_id", "source"]:
                df[col] = None
            elif "yards" in col or "attempts" in col or "completions" in col:
                df[col] = 0.0
            else:
                df[col] = 0

    # Reorder columns to match canonical order
    df = df[CANONICAL_COLUMNS]

    # Validate
    validate_stats_dataframe(df)

    return df
