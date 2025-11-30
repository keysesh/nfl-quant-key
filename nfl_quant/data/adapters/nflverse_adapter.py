"""
NFLverse stats adapter - transforms NFLverse format to canonical format.

NFLverse Format (Parquet):
  player_id, player_name, player_display_name, position, recent_team, season, week,
  completions, attempts, passing_yards, passing_tds, interceptions,
  carries, rushing_yards, rushing_tds,
  receptions, targets, receiving_yards, receiving_tds,
  opponent_team, ...
"""

from pathlib import Path
import pandas as pd
from typing import Optional
from .base_adapter import StatsAdapter


class NFLVerseAdapter(StatsAdapter):
    """Adapter for NFLverse weekly stats Parquet files."""

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__()
        self.source_name = "nflverse"
        self.data_dir = data_dir or Path("data/nflverse")
        self._cache = {}  # Cache loaded parquet files by season

    def load_weekly_stats(self, week: int, season: int, **kwargs) -> pd.DataFrame:
        """
        Load NFLverse weekly stats and transform to canonical format.

        Args:
            week: Week number (1-18)
            season: Season year
            **kwargs: Not used for NFLverse

        Returns:
            DataFrame in canonical format
        """
        # Load season data (cached)
        season_data = self._load_season_data(season)

        if season_data is None:
            raise FileNotFoundError(f"NFLverse data not found for season {season}")

        # Filter to specific week
        week_data = season_data[season_data['week'] == week].copy()

        if len(week_data) == 0:
            raise ValueError(f"No data found for week {week}, season {season}")

        # Transform to canonical format
        canonical = pd.DataFrame()

        # Identifiers
        canonical["player_id"] = week_data["player_id"].astype(str)
        # NFLverse has both player_name and player_display_name - use display_name
        canonical["player_name"] = week_data.get("player_display_name", week_data.get("player_name", ""))
        canonical["position"] = week_data["position"]
        # Handle both R-generated parquet (team) and legacy NFLverse (recent_team)
        if "recent_team" in week_data.columns:
            canonical["team"] = week_data["recent_team"].apply(self.normalize_team_name)
        elif "team" in week_data.columns:
            canonical["team"] = week_data["team"].apply(self.normalize_team_name)
        else:
            raise KeyError("NFLverse data missing team column (expected 'team' or 'recent_team')")
        canonical["season"] = season
        canonical["week"] = week

        # Passing stats (NFLverse: attempts, completions, passing_yards, passing_tds, interceptions)
        canonical["passing_yards"] = pd.to_numeric(week_data["passing_yards"] if "passing_yards" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0.0)
        canonical["passing_attempts"] = pd.to_numeric(week_data["attempts"] if "attempts" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0.0)
        canonical["passing_completions"] = pd.to_numeric(week_data["completions"] if "completions" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0.0)
        # ALIAS: Also provide 'attempts' and 'completions' for backward compatibility with simulation code
        canonical["attempts"] = canonical["passing_attempts"]
        canonical["completions"] = canonical["passing_completions"]
        canonical["passing_tds"] = pd.to_numeric(week_data["passing_tds"] if "passing_tds" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0).astype(int)
        # Handle both 'interceptions' and 'passing_interceptions' column names
        if "interceptions" in week_data.columns:
            canonical["interceptions"] = pd.to_numeric(week_data["interceptions"], errors='coerce').fillna(0).astype(int)
        elif "passing_interceptions" in week_data.columns:
            canonical["interceptions"] = pd.to_numeric(week_data["passing_interceptions"], errors='coerce').fillna(0).astype(int)
        else:
            canonical["interceptions"] = pd.Series([0] * len(week_data), dtype=int)

        # Rushing stats (NFLverse: carries, rushing_yards, rushing_tds)
        canonical["rushing_yards"] = pd.to_numeric(week_data["rushing_yards"] if "rushing_yards" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0.0)
        canonical["rushing_attempts"] = pd.to_numeric(week_data["carries"] if "carries" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0.0)
        # ALIAS: Also provide 'carries' for backward compatibility with simulation code
        canonical["carries"] = canonical["rushing_attempts"]
        canonical["rushing_tds"] = pd.to_numeric(week_data["rushing_tds"] if "rushing_tds" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0).astype(int)

        # Receiving stats (NFLverse: receptions, targets, receiving_yards, receiving_tds)
        canonical["receptions"] = pd.to_numeric(week_data["receptions"] if "receptions" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0).astype(int)
        canonical["receiving_yards"] = pd.to_numeric(week_data["receiving_yards"] if "receiving_yards" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0.0)
        canonical["receiving_tds"] = pd.to_numeric(week_data["receiving_tds"] if "receiving_tds" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0).astype(int)
        canonical["targets"] = pd.to_numeric(week_data["targets"] if "targets" in week_data.columns else pd.Series([0] * len(week_data)), errors='coerce').fillna(0).astype(int)

        # Optional fields (NFLverse has opponent_team)
        canonical["opponent"] = week_data.get("opponent_team", None)
        if "opponent" in canonical.columns:
            canonical["opponent"] = canonical["opponent"].apply(
                lambda x: self.normalize_team_name(x) if pd.notna(x) else None
            )
        canonical["game_id"] = None  # Could extract from NFLverse if needed
        canonical["source"] = self.source_name

        # Ensure canonical format
        canonical = self._ensure_canonical_columns(canonical)

        return canonical

    def is_available(self, week: int, season: int) -> bool:
        """
        Check if NFLverse stats are available for the given week/season.

        Args:
            week: Week number
            season: Season year

        Returns:
            True if data exists
        """
        try:
            season_data = self._load_season_data(season)
            if season_data is None:
                return False
            return (season_data['week'] == week).any()
        except Exception:
            return False

    def _load_season_data(self, season: int) -> Optional[pd.DataFrame]:
        """
        Load entire season data from parquet file (with caching).

        Priority order (prefers R-generated files):
        1. player_stats.parquet (R-generated, most comprehensive)
        2. weekly_stats.parquet (R-generated alias)
        3. weekly_{season}.parquet (legacy per-season file)
        4. weekly_historical.parquet (legacy combined file)

        Args:
            season: Season year

        Returns:
            DataFrame with full season data or None if not found
        """
        # Check cache first
        if season in self._cache:
            return self._cache[season]

        # Priority 1: R-generated player_stats.parquet (most comprehensive)
        player_stats_file = self.data_dir / "player_stats.parquet"
        if player_stats_file.exists():
            df = pd.read_parquet(player_stats_file)
            if 'season' in df.columns:
                season_df = df[df['season'] == season].copy()
                if len(season_df) > 0:
                    self._cache[season] = season_df
                    return season_df

        # Priority 2: R-generated weekly_stats.parquet (alias)
        weekly_stats_file = self.data_dir / "weekly_stats.parquet"
        if weekly_stats_file.exists():
            df = pd.read_parquet(weekly_stats_file)
            if 'season' in df.columns:
                season_df = df[df['season'] == season].copy()
                if len(season_df) > 0:
                    self._cache[season] = season_df
                    return season_df

        # Priority 3: Legacy per-season parquet file
        weekly_file = self.data_dir / f"weekly_{season}.parquet"
        if weekly_file.exists():
            df = pd.read_parquet(weekly_file)
            self._cache[season] = df
            return df

        # Priority 4: Legacy historical combined file
        historical_file = self.data_dir / "weekly_historical.parquet"
        if historical_file.exists():
            df = pd.read_parquet(historical_file)
            # Filter to specific season
            if 'season' in df.columns:
                season_df = df[df['season'] == season].copy()
                if len(season_df) > 0:
                    self._cache[season] = season_df
                    return season_df

        return None

    def load_all_seasons(self, seasons: list[int]) -> pd.DataFrame:
        """
        Load multiple seasons at once (more efficient for bulk loading).

        Args:
            seasons: List of season years to load

        Returns:
            Combined DataFrame in canonical format
        """
        all_data = []

        for season in seasons:
            season_data = self._load_season_data(season)
            if season_data is not None:
                # Get all weeks for this season
                weeks = sorted(season_data['week'].unique())
                for week in weeks:
                    try:
                        week_canonical = self.load_weekly_stats(week, season)
                        all_data.append(week_canonical)
                    except Exception as e:
                        # Skip weeks with errors
                        continue

        if not all_data:
            raise ValueError(f"No data found for seasons {seasons}")

        return pd.concat(all_data, ignore_index=True)
