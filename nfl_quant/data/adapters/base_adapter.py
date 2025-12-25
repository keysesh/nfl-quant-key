"""
Base adapter class for stats data sources.

All data source adapters must inherit from this base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from typing import Optional

from nfl_quant.utils.player_names import normalize_player_name as _canonical_normalize


class StatsAdapter(ABC):
    """
    Abstract base class for stats data adapters.

    Each data source (NFLverse, Sleeper, etc.) implements this interface
    to transform their specific format into the canonical format.
    """

    def __init__(self):
        self.source_name = "unknown"

    @abstractmethod
    def load_weekly_stats(self, week: int, season: int, **kwargs) -> pd.DataFrame:
        """
        Load weekly stats for a specific week/season in canonical format.

        Args:
            week: Week number (1-18)
            season: Season year (2023, 2024, 2025)
            **kwargs: Additional source-specific parameters

        Returns:
            DataFrame in canonical format with columns:
                player_id, player_name, position, team, season, week,
                passing_yards, passing_attempts, passing_completions, passing_tds, interceptions,
                rushing_yards, rushing_attempts, rushing_tds,
                receptions, receiving_yards, receiving_tds, targets,
                opponent (optional), game_id (optional), source
        """
        pass

    @abstractmethod
    def is_available(self, week: int, season: int) -> bool:
        """
        Check if data is available for the given week/season.

        Args:
            week: Week number
            season: Season year

        Returns:
            True if data exists, False otherwise
        """
        pass

    def normalize_player_name(self, name: str) -> str:
        """
        Normalize player name for consistent matching across sources.

        Delegates to canonical implementation in nfl_quant.utils.player_names.

        Args:
            name: Raw player name

        Returns:
            Normalized player name
        """
        return _canonical_normalize(name)

    def normalize_team_name(self, team: str) -> str:
        """
        Normalize team abbreviation.

        Args:
            team: Raw team name or abbreviation

        Returns:
            Standardized 3-letter team abbreviation (uppercase)
        """
        if pd.isna(team):
            return "UNK"
        team = str(team).strip().upper()
        # Handle common variations
        team_mapping = {
            "JAC": "JAX",
            "WSH": "WAS",
            "LA": "LAR",  # If ambiguous, default to Rams
        }
        return team_mapping.get(team, team)

    def _ensure_canonical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has all canonical columns with correct types.

        Args:
            df: DataFrame to validate

        Returns:
            DataFrame with all canonical columns
        """
        required_cols = {
            "player_id": str,
            "player_name": str,
            "position": str,
            "team": str,
            "season": int,
            "week": int,
            "passing_yards": float,
            "passing_attempts": float,
            "passing_completions": float,
            "passing_tds": int,
            "interceptions": int,
            "rushing_yards": float,
            "rushing_attempts": float,
            "rushing_tds": int,
            "receptions": int,
            "receiving_yards": float,
            "receiving_tds": int,
            "targets": int,
        }

        optional_cols = {
            "opponent": str,
            "game_id": str,
            "source": str,
        }

        # Add missing required columns with default values
        for col, dtype in required_cols.items():
            if col not in df.columns:
                if dtype == str:
                    df[col] = ""
                elif dtype == int:
                    df[col] = 0
                elif dtype == float:
                    df[col] = 0.0

        # Add missing optional columns
        for col, dtype in optional_cols.items():
            if col not in df.columns:
                if col == "source":
                    df[col] = self.source_name
                else:
                    df[col] = None

        # Convert data types
        for col, dtype in required_cols.items():
            if col in df.columns:
                if dtype == int:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif dtype == float:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
                elif dtype == str:
                    df[col] = df[col].fillna("").astype(str)

        # Add source if not present
        if 'source' not in df.columns or df['source'].isna().all():
            df['source'] = self.source_name

        return df
