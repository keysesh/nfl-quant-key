"""
NFLverse Data Fetcher - Direct Downloads from GitHub Releases.

Uses direct CSV/Parquet downloads from nflverse-data repository.
Simple, reliable, no dependencies beyond pandas.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional
import time


class NFLverseFetcher:
    """Fetch NFL data directly from nflverse GitHub releases."""

    BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize NFLverse data fetcher.

        Args:
            cache_dir: Directory for local cache. Defaults to data/nflverse_cache/
            verbose: Show progress messages
        """
        if cache_dir is None:
            cache_dir = str(Path.cwd() / 'data' / 'nflverse_cache')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        if self.verbose:
            print(f"✅ NFLverse Fetcher initialized")
            print(f"   Cache directory: {self.cache_dir}")
            print(f"   Source: {self.BASE_URL}")

    def _download_file(self, url: str, filename: str, force_refresh: bool = False) -> Path:
        """
        Download file from URL with caching.

        Args:
            url: Full URL to file
            filename: Local filename
            force_refresh: Skip cache and download fresh

        Returns:
            Path to local file
        """
        local_path = self.cache_dir / filename

        # Use cached version if exists and not forcing refresh
        if local_path.exists() and not force_refresh:
            if self.verbose:
                print(f"   📂 Using cached: {filename}")
            return local_path

        # Download
        if self.verbose:
            print(f"   ⬇️  Downloading: {filename}")

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            # Save to cache
            with open(local_path, 'wb') as f:
                f.write(response.content)

            if self.verbose:
                size_mb = len(response.content) / (1024 * 1024)
                print(f"   ✅ Downloaded: {size_mb:.2f} MB")

            return local_path

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Download failed: {e}")
            raise

    def get_player_stats(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Get weekly player statistics.

        Args:
            seasons: List of seasons (e.g., [2024, 2025])
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with player stats
        """
        if self.verbose:
            print(f"\n📊 Fetching player stats for seasons: {seasons}")

        dfs = []
        for season in seasons:
            filename = f"stats_player_week_{season}.csv"
            url = f"{self.BASE_URL}/stats_player/{filename}"

            try:
                local_file = self._download_file(url, filename, force_refresh)
                df = pd.read_csv(local_file)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to load {season} data: {e}")
                continue

        if not dfs:
            raise ValueError("No data loaded for any season")

        # Combine all seasons
        combined = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            print(f"   ✅ Loaded {len(combined):,} player-week records")
            print(f"   Seasons: {sorted(combined['season'].unique())}")
            print(f"   Weeks: {sorted(combined['week'].unique())}")
            print(f"   Players: {combined['player_display_name'].nunique()}")

        return combined

    def get_schedules(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get game schedules.

        Args:
            seasons: Optional list of seasons to filter
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with schedule data
        """
        if self.verbose:
            print(f"\n📅 Fetching schedules...")

        filename = "games.csv"
        url = f"{self.BASE_URL}/schedules/{filename}"

        local_file = self._download_file(url, filename, force_refresh)
        schedules = pd.read_csv(local_file)

        # Filter to specific seasons if requested
        if seasons is not None:
            schedules = schedules[schedules['season'].isin(seasons)]

        if self.verbose:
            print(f"   ✅ Loaded {len(schedules):,} games")
            if seasons:
                print(f"   Filtered to seasons: {seasons}")

        return schedules

    def get_rosters(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Get player rosters.

        Args:
            seasons: List of seasons
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with roster data
        """
        if self.verbose:
            print(f"\n👥 Fetching rosters for seasons: {seasons}")

        dfs = []
        for season in seasons:
            filename = f"roster_{season}.csv"
            url = f"{self.BASE_URL}/rosters/{filename}"

            try:
                local_file = self._download_file(url, filename, force_refresh)
                df = pd.read_csv(local_file)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to load {season} rosters: {e}")
                continue

        if not dfs:
            raise ValueError("No rosters loaded")

        combined = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            print(f"   ✅ Loaded {len(combined):,} roster entries")

        return combined

    def get_injuries(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Get injury reports.

        Args:
            seasons: List of seasons
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with injury data
        """
        if self.verbose:
            print(f"\n🏥 Fetching injuries for seasons: {seasons}")

        dfs = []
        for season in seasons:
            filename = f"injuries_{season}.csv"
            url = f"{self.BASE_URL}/injuries/{filename}"

            try:
                local_file = self._download_file(url, filename, force_refresh)
                df = pd.read_csv(local_file)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to load {season} injuries: {e}")
                continue

        if not dfs:
            print(f"   ⚠️  No injury data available")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            print(f"   ✅ Loaded {len(combined):,} injury reports")

        return combined

    def get_depth_charts(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Get depth charts.

        Args:
            seasons: List of seasons
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with depth chart data
        """
        if self.verbose:
            print(f"\n📋 Fetching depth charts for seasons: {seasons}")

        dfs = []
        for season in seasons:
            filename = f"depth_charts_{season}.csv"
            url = f"{self.BASE_URL}/depth_charts/{filename}"

            try:
                local_file = self._download_file(url, filename, force_refresh)
                df = pd.read_csv(local_file)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to load {season} depth charts: {e}")
                continue

        if not dfs:
            print(f"   ⚠️  No depth chart data available")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            print(f"   ✅ Loaded {len(combined):,} depth chart entries")

        return combined

    def get_snap_counts(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Get snap counts.

        Args:
            seasons: List of seasons
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with snap count data
        """
        if self.verbose:
            print(f"\n⚡ Fetching snap counts for seasons: {seasons}")

        dfs = []
        for season in seasons:
            filename = f"snap_counts_{season}.csv"
            url = f"{self.BASE_URL}/snap_counts/{filename}"

            try:
                local_file = self._download_file(url, filename, force_refresh)
                df = pd.read_csv(local_file)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to load {season} snap counts: {e}")
                continue

        if not dfs:
            print(f"   ⚠️  No snap count data available")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            print(f"   ✅ Loaded {len(combined):,} snap count records")

        return combined

    def get_players(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get player metadata/IDs.

        Args:
            force_refresh: Force fresh download

        Returns:
            pandas DataFrame with player info
        """
        if self.verbose:
            print(f"\n🔍 Fetching player metadata...")

        filename = "players.csv"
        url = f"{self.BASE_URL}/players/{filename}"

        local_file = self._download_file(url, filename, force_refresh)
        players = pd.read_csv(local_file)

        if self.verbose:
            print(f"   ✅ Loaded {len(players):,} players")

        return players

    def clear_cache(self):
        """Clear all cached files."""
        if self.verbose:
            print(f"\n🗑️  Clearing cache...")

        for file in self.cache_dir.glob("*"):
            if file.is_file():
                file.unlink()

        if self.verbose:
            print(f"   ✅ Cache cleared")

    @staticmethod
    def get_current_season() -> int:
        """Get current NFL season year (approximation based on date)."""
        from datetime import datetime
        now = datetime.now()
        # NFL season starts in September, so if month >= 9, current year is season
        # Otherwise, previous year
        return now.year if now.month >= 9 else now.year - 1

    @staticmethod
    def get_current_week() -> int:
        """
        Estimate current NFL week (approximation based on date).

        Note: This is a rough estimate. For accurate week, use schedule data.
        """
        from datetime import datetime
        now = datetime.now()

        # NFL season typically starts first Thursday after Labor Day (early Sept)
        # Week 1 usually starts around Sept 5-10
        if now.month < 9 or (now.month == 9 and now.day < 5):
            return 0  # Preseason

        # Rough calculation: weeks since Sept 5
        season_start = datetime(now.year, 9, 5)
        days_elapsed = (now - season_start).days
        week = (days_elapsed // 7) + 1

        return min(max(week, 1), 18)  # Cap at weeks 1-18


# Convenience functions
def fetch_player_stats(seasons: List[int], cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Quick function to fetch player stats."""
    fetcher = NFLverseFetcher(cache_dir=cache_dir)
    return fetcher.get_player_stats(seasons)


def fetch_current_season_stats(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Fetch stats for current season."""
    current_season = NFLverseFetcher.get_current_season()
    return fetch_player_stats([current_season], cache_dir)


if __name__ == "__main__":
    """Demo usage."""
    print("=" * 80)
    print("NFLVERSE FETCHER - DEMO")
    print("=" * 80)

    # Initialize
    fetcher = NFLverseFetcher()

    # Get current season info
    season = fetcher.get_current_season()
    week = fetcher.get_current_week()

    print(f"\n🏈 Estimated Current Season: {season}")
    print(f"📅 Estimated Current Week: {week}")

    # Fetch player stats for weeks 1-8
    print(f"\n{'=' * 80}")
    stats = fetcher.get_player_stats(seasons=[season])

    # Filter to weeks 1-8
    stats_1_8 = stats[stats['week'] <= 8]

    print(f"\n✅ Sample data (top 5 QBs Week 8):")
    week8_qbs = stats_1_8[
        (stats_1_8['position'] == 'QB') &
        (stats_1_8['week'] == 8)
    ].nlargest(5, 'passing_yards')

    print(week8_qbs[['player_display_name', 'team', 'week',
                     'passing_yards', 'passing_tds', 'rushing_yards']].to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ Direct downloads working perfectly!")
    print("   Use this fetcher in your pipeline.")
    print("=" * 80)
