"""
NFL Data Fetcher using nflreadpy.

Centralized module for fetching NFL data from nflverse using nflreadpy.
Replaces manual downloads and provides caching.
"""

import nflreadpy as nfl
from nflreadpy.config import update_config
from pathlib import Path
import pandas as pd
from typing import List, Optional
import shutil


class NFLDataFetcher:
    """Fetch NFL data using nflreadpy with caching."""

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize NFL data fetcher.

        Args:
            cache_dir: Directory for filesystem cache. Defaults to .nfl_cache/
            verbose: Show progress messages
        """
        if cache_dir is None:
            cache_dir = str(Path.cwd() / '.nfl_cache')

        self.cache_dir = Path(cache_dir)

        # Ensure cache directory exists (workaround for nflreadpy bug)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure nflreadpy
        update_config(
            cache_mode="filesystem",
            cache_dir=str(self.cache_dir.absolute()),
            verbose=verbose,
            timeout=120
        )
        self.verbose = verbose

        if self.verbose:
            print(f"✅ NFL Data Fetcher initialized")
            print(f"   Cache directory: {self.cache_dir}")

    def get_player_stats(self, seasons: List[int], weeks: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get weekly player statistics.

        Args:
            seasons: List of seasons (e.g., [2024, 2025])
            weeks: Optional list of weeks to filter (e.g., [1, 2, 3])

        Returns:
            pandas DataFrame with player stats
        """
        if self.verbose:
            print(f"\n📊 Fetching player stats for seasons: {seasons}")

        # Load from nflreadpy (returns Polars)
        stats = nfl.load_player_stats(seasons=seasons)

        # Convert to pandas
        stats_df = stats.to_pandas()

        # Filter to specific weeks if requested
        if weeks is not None:
            stats_df = stats_df[stats_df['week'].isin(weeks)]

        if self.verbose:
            print(f"   ✅ Loaded {len(stats_df):,} player-week records")
            print(f"   Weeks: {sorted(stats_df['week'].unique())}")
            print(f"   Players: {stats_df['player_display_name'].nunique()}")

        return stats_df

    def get_schedules(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get game schedules.

        Args:
            seasons: List of seasons

        Returns:
            pandas DataFrame with schedule data
        """
        if self.verbose:
            print(f"\n📅 Fetching schedules for seasons: {seasons}")

        schedules = nfl.load_schedules(seasons=seasons)
        schedules_df = schedules.to_pandas()

        if self.verbose:
            print(f"   ✅ Loaded {len(schedules_df)} games")

        return schedules_df

    def get_rosters(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get player rosters.

        Args:
            seasons: List of seasons

        Returns:
            pandas DataFrame with roster data
        """
        if self.verbose:
            print(f"\n👥 Fetching rosters for seasons: {seasons}")

        rosters = nfl.load_rosters(seasons=seasons)
        rosters_df = rosters.to_pandas()

        if self.verbose:
            print(f"   ✅ Loaded {len(rosters_df):,} roster entries")

        return rosters_df

    def get_injuries(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get injury reports.

        Args:
            seasons: List of seasons

        Returns:
            pandas DataFrame with injury data
        """
        if self.verbose:
            print(f"\n🏥 Fetching injuries for seasons: {seasons}")

        injuries = nfl.load_injuries(seasons=seasons)
        injuries_df = injuries.to_pandas()

        if self.verbose:
            print(f"   ✅ Loaded {len(injuries_df):,} injury reports")

        return injuries_df

    def get_depth_charts(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get depth charts.

        Args:
            seasons: List of seasons

        Returns:
            pandas DataFrame with depth chart data
        """
        if self.verbose:
            print(f"\n📋 Fetching depth charts for seasons: {seasons}")

        depth = nfl.load_depth_charts(seasons=seasons)
        depth_df = depth.to_pandas()

        if self.verbose:
            print(f"   ✅ Loaded {len(depth_df):,} depth chart entries")

        return depth_df

    def get_pbp(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get play-by-play data (use sparingly - large dataset).

        Args:
            seasons: List of seasons

        Returns:
            pandas DataFrame with play-by-play data
        """
        if self.verbose:
            print(f"\n🏈 Fetching play-by-play for seasons: {seasons}")
            print("   ⚠️  Warning: PBP is a large dataset, this may take time...")

        pbp = nfl.load_pbp(seasons=seasons)
        pbp_df = pbp.to_pandas()

        if self.verbose:
            print(f"   ✅ Loaded {len(pbp_df):,} plays")

        return pbp_df

    def get_players(self) -> pd.DataFrame:
        """
        Get player metadata/IDs.

        Returns:
            pandas DataFrame with player info
        """
        if self.verbose:
            print(f"\n🔍 Fetching player metadata...")

        players = nfl.load_players()
        players_df = players.to_pandas()

        if self.verbose:
            print(f"   ✅ Loaded {len(players_df):,} players")

        return players_df

    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cached data.

        Args:
            pattern: Optional pattern to match (e.g., "player_stats")
        """
        if self.verbose:
            print(f"\n🗑️  Clearing cache" + (f" (pattern: {pattern})" if pattern else ""))

        # Clear filesystem cache
        if self.cache_dir.exists():
            if pattern:
                for file in self.cache_dir.glob(f"*{pattern}*"):
                    file.unlink()
            else:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print("   ✅ Cache cleared")

    @staticmethod
    def get_current_season() -> int:
        """Get current NFL season year."""
        return nfl.get_current_season()

    @staticmethod
    def get_current_week() -> int:
        """Get current NFL week."""
        return nfl.get_current_week()


# Convenience functions for quick access
def fetch_player_stats(seasons: List[int], weeks: Optional[List[int]] = None,
                       cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Quick function to fetch player stats.

    Args:
        seasons: List of seasons
        weeks: Optional list of weeks
        cache_dir: Optional cache directory

    Returns:
        pandas DataFrame
    """
    fetcher = NFLDataFetcher(cache_dir=cache_dir, verbose=True)
    return fetcher.get_player_stats(seasons, weeks)


def fetch_current_season_stats(weeks: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Fetch stats for current season.

    Args:
        weeks: Optional list of weeks to filter

    Returns:
        pandas DataFrame
    """
    current_season = nfl.get_current_season()
    return fetch_player_stats([current_season], weeks)


if __name__ == "__main__":
    """Demo usage."""
    print("=" * 80)
    print("NFL DATA FETCHER - DEMO")
    print("=" * 80)

    # Initialize
    fetcher = NFLDataFetcher()

    # Get current season info
    season = fetcher.get_current_season()
    week = fetcher.get_current_week()

    print(f"\n🏈 Current Season: {season}")
    print(f"📅 Current Week: {week}")

    # Fetch player stats for weeks 1-8
    stats = fetcher.get_player_stats(seasons=[season], weeks=list(range(1, 9)))

    print(f"\n✅ Sample data (first 5 rows):")
    print(stats[['player_display_name', 'position', 'week', 'passing_yards',
                 'rushing_yards', 'receiving_yards']].head())

    print("\n" + "=" * 80)
    print("✅ Integration ready! Use this module in your pipeline.")
    print("=" * 80)
