"""
NFLverse Data Loader
====================

Centralized access to nflverse-data repository via nflreadpy package.

This module provides:
- Intelligent caching of NFLverse weekly player stats
- Automatic detection of new data availability
- Position-level catch rate calculations
- Data validation and quality checks

Data Sources:
- Primary: nflreadpy.load_player_stats() - Official NFLverse Python package
- Storage: Parquet format for fast I/O and compression
- Update Schedule: ~30 minutes after final game each week

Author: NFL Quant Analytics
Last Updated: 2025-01-10
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class NFLverseDataLoader:
    """
    Centralized loader for NFLverse player statistics.

    Handles caching, data freshness checks, and position-level aggregations.
    Replaces Sleeper API dependencies with NFLverse-only data pipeline.

    Usage:
        loader = NFLverseDataLoader()
        catch_rate = loader.get_position_catch_rate(season=2025, position='WR')
        df = loader.get_weekly_stats(seasons=[2025])
    """

    # Expected catch rate ranges for data validation
    CATCH_RATE_RANGES = {
        'RB': (0.70, 0.85),
        'WR': (0.60, 0.70),
        'TE': (0.65, 0.75),
        'FB': (0.68, 0.78),
    }

    # Minimum targets required for valid catch rate calculation
    MIN_TARGETS_PER_POSITION = 500
    MIN_TARGETS_WARNING_THRESHOLD = 100

    def __init__(self, cache_dir: Path = None):
        """
        Initialize NFLverse data loader with caching support.

        Args:
            cache_dir: Directory for cached data files. Defaults to data/nflverse_cache
        """
        if cache_dir is None:
            self.cache_dir = Path('data/nflverse_cache')
        elif isinstance(cache_dir, str):
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently accessed data
        self._stats_cache: Dict[int, pd.DataFrame] = {}
        self._catch_rates_cache: Dict[Tuple[int, str], float] = {}

        # Metadata tracking
        self._cache_metadata_file = self.cache_dir / 'cache_metadata.json'
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self._cache_metadata_file.exists():
            try:
                with open(self._cache_metadata_file) as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self._cache_metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache metadata: {e}")

    def _get_cache_file(self, season: int, format: str = 'parquet') -> Path:
        """Get cache file path for a given season."""
        return self.cache_dir / f'stats_player_week_{season}.{format}'

    def _is_cache_fresh(self, season: int, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is fresh enough.

        Args:
            season: NFL season year
            max_age_hours: Maximum age in hours before refresh needed

        Returns:
            True if cache is fresh, False if needs refresh
        """
        cache_file = self._get_cache_file(season, 'parquet')
        if not cache_file.exists():
            return False

        # Check file modification time
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - mod_time

        if age > timedelta(hours=max_age_hours):
            logger.info(f"Cache for {season} is {age.total_seconds()/3600:.1f} hours old (max: {max_age_hours})")
            return False

        return True

    def get_weekly_stats(
        self,
        seasons: List[int],
        force_refresh: bool = False,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load weekly player stats with intelligent caching.

        Data is loaded from:
        1. In-memory cache (if available)
        2. Disk cache (Parquet files)
        3. NFLverse API (if cache missing or force_refresh=True)

        Args:
            seasons: List of season years to load (e.g., [2024, 2025])
            force_refresh: Force refresh from NFLverse API even if cache exists
            use_cache: Whether to use cached data at all

        Returns:
            DataFrame with weekly player statistics

        Raises:
            ValueError: If data cannot be loaded from any source
        """
        all_stats = []

        # CANONICAL SOURCE: R-generated parquet file from R/nflreadr
        # This is the ONLY authoritative data source - no nflreadpy fallback
        r_parquet = Path('data/nflverse/player_stats.parquet')
        if not r_parquet.exists():
            raise ValueError(
                f"R-generated parquet not found at {r_parquet}. "
                f"Run 'Rscript scripts/fetch/fetch_nflverse_data.R' first to fetch data via R/nflreadr."
            )

        try:
            logger.debug(f"Loading from R-generated canonical parquet: {r_parquet}")
            all_data = pd.read_parquet(r_parquet)

            # Filter to requested seasons
            missing_seasons = []
            for season in seasons:
                if season in all_data['season'].unique():
                    season_df = all_data[all_data['season'] == season].copy()
                    self._stats_cache[season] = season_df
                    all_stats.append(season_df)
                    logger.debug(f"Loaded {len(season_df)} rows for {season} from R parquet")
                else:
                    missing_seasons.append(season)

            if missing_seasons:
                raise ValueError(
                    f"Seasons {missing_seasons} not found in R parquet. "
                    f"Available seasons: {sorted(all_data['season'].unique())}. "
                    f"Run R fetch script to update data."
                )

            if not all_stats:
                raise ValueError(f"No data loaded for seasons: {seasons}")

            combined = pd.concat(all_stats, ignore_index=True)
            logger.info(f"Loaded {len(combined)} rows for seasons {seasons} from R parquet (canonical source)")
            return combined

        except Exception as e:
            if "not found" in str(e) or "Run R" in str(e):
                raise  # Re-raise our specific errors
            raise ValueError(
                f"Failed to load R-generated parquet: {e}. "
                f"Ensure R/nflreadr data is up to date by running fetch script."
            )

    def get_position_catch_rate(
        self,
        season: int,
        position: str,
        validate: bool = True
    ) -> float:
        """
        Calculate position-average catch rate from actual data.

        Catch rate = Total Receptions / Total Targets for all players at position.

        Args:
            season: NFL season year
            position: Player position (RB, WR, TE, etc.)
            validate: Whether to validate catch rate is within expected range

        Returns:
            Catch rate (0.0 to 1.0)

        Raises:
            ValueError: If insufficient data or invalid catch rate
        """
        # Check cache first
        cache_key = (season, position)
        if cache_key in self._catch_rates_cache:
            return self._catch_rates_cache[cache_key]

        # Load data
        df = self.get_weekly_stats([season])

        # Filter by position
        pos_stats = df[df['position'] == position].copy()

        if len(pos_stats) == 0:
            raise ValueError(
                f"No data found for position {position} in {season}. "
                f"Available positions: {sorted(df['position'].unique())}"
            )

        # Calculate catch rate
        total_receptions = pos_stats['receptions'].sum()
        total_targets = pos_stats['targets'].sum()

        if total_targets == 0:
            raise ValueError(
                f"No targets recorded for {position} in {season}. "
                f"Cannot calculate catch rate."
            )

        # Check minimum sample size
        if total_targets < self.MIN_TARGETS_WARNING_THRESHOLD:
            logger.warning(
                f"Low sample size for {position} in {season}: "
                f"{total_targets} targets (recommend minimum {self.MIN_TARGETS_PER_POSITION})"
            )

        catch_rate = total_receptions / total_targets

        # Validate catch rate is reasonable
        if validate and position in self.CATCH_RATE_RANGES:
            min_rate, max_rate = self.CATCH_RATE_RANGES[position]
            if not (min_rate <= catch_rate <= max_rate):
                logger.warning(
                    f"Catch rate for {position} ({catch_rate:.3f}) outside expected range "
                    f"({min_rate:.3f} to {max_rate:.3f}). "
                    f"Sample: {total_receptions}/{total_targets} targets"
                )

        logger.info(
            f"{position} {season} catch rate: {catch_rate:.3f} "
            f"({total_receptions:.0f}/{total_targets:.0f} targets)"
        )

        # Cache result
        self._catch_rates_cache[cache_key] = catch_rate

        return catch_rate

    def check_for_updates(self, season: int) -> Dict[str, Any]:
        """
        Check if newer data is available than local cache.

        Args:
            season: NFL season year to check

        Returns:
            Dictionary with update status:
            - has_cache: bool
            - cache_age_hours: float
            - cache_weeks: List[int]
            - needs_update: bool (if cache > 24 hours old)
        """
        cache_file = self._get_cache_file(season, 'parquet')

        status = {
            'season': season,
            'has_cache': cache_file.exists(),
            'cache_age_hours': None,
            'cache_weeks': [],
            'needs_update': True,
        }

        if cache_file.exists():
            # Get cache age
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age = datetime.now() - mod_time
            status['cache_age_hours'] = age.total_seconds() / 3600

            # Get weeks from metadata
            if f'{season}_weeks' in self.metadata:
                status['cache_weeks'] = self.metadata[f'{season}_weeks']

            # Determine if update needed (> 24 hours old)
            status['needs_update'] = age > timedelta(hours=24)

        return status

    def clear_cache(self, season: Optional[int] = None):
        """
        Clear cached data.

        Args:
            season: Specific season to clear, or None to clear all
        """
        if season is not None:
            # Clear specific season
            self._stats_cache.pop(season, None)
            self._catch_rates_cache = {
                k: v for k, v in self._catch_rates_cache.items()
                if k[0] != season
            }

            for ext in ['parquet', 'csv']:
                cache_file = self._get_cache_file(season, ext)
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Deleted cache file: {cache_file}")
        else:
            # Clear all
            self._stats_cache.clear()
            self._catch_rates_cache.clear()

            for cache_file in self.cache_dir.glob('stats_player_week_*'):
                cache_file.unlink()
                logger.info(f"Deleted cache file: {cache_file}")

    def get_data_summary(self, season: int) -> Dict[str, Any]:
        """
        Get summary statistics for cached data.

        Args:
            season: NFL season year

        Returns:
            Dictionary with summary stats
        """
        try:
            df = self.get_weekly_stats([season])

            summary = {
                'season': season,
                'total_rows': len(df),
                'weeks': sorted(df['week'].unique().tolist()),
                'week_count': df['week'].nunique(),
                'positions': sorted(df['position'].unique().tolist()),
                'catch_rates': {},
            }

            # Calculate catch rates by position
            for pos in ['RB', 'WR', 'TE']:
                try:
                    rate = self.get_position_catch_rate(season, pos, validate=False)
                    pos_stats = df[df['position'] == pos]
                    summary['catch_rates'][pos] = {
                        'rate': round(rate, 3),
                        'receptions': int(pos_stats['receptions'].sum()),
                        'targets': int(pos_stats['targets'].sum()),
                    }
                except Exception as e:
                    summary['catch_rates'][pos] = {'error': str(e)}

            return summary

        except Exception as e:
            return {
                'season': season,
                'error': str(e)
            }
