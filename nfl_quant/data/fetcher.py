"""Data fetcher for nflverse API."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from nfl_quant.config import settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches NFL data from nflverse API."""

    def __init__(self) -> None:
        """Initialize fetcher with configuration."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.USER_AGENT})
        self.raw_dir = settings.RAW_DATA_DIR
        self.processed_dir = settings.PROCESSED_DATA_DIR

    def _fetch_with_retries(
        self, url: str, max_retries: int = 3, backoff: float = 1.0
    ) -> Optional[dict[str, Any] | list]:
        """Fetch JSON from URL with retry logic.

        Args:
            url: URL to fetch
            max_retries: Maximum retry attempts
            backoff: Backoff multiplier

        Returns:
            Parsed JSON response or None if failed
        """
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = backoff ** attempt
                    logger.warning(
                        f"Fetch failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Fetch failed after {max_retries} attempts: {url}")
                    raise

    def fetch_nflverse_pbp(self, season: int = 2025) -> pd.DataFrame:
        """Fetch play-by-play data from nflverse (via R-fetched files).

        Args:
            season: Season (must be 2025)

        Returns:
            DataFrame with PBP data
        """
        season = settings.validate_season(season)
        logger.info(f"Loading nflverse PBP data for season {season}...")

        # Try to load from R-fetched data first
        nflverse_dir = Path("data/nflverse")

        # Check for season-specific file first
        pbp_file = nflverse_dir / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            pbp_file = nflverse_dir / "pbp.parquet"
        if not pbp_file.exists():
            pbp_file = nflverse_dir / "pbp.csv"

        if pbp_file.exists():
            if pbp_file.suffix == ".parquet":
                pbp = pd.read_parquet(pbp_file)
            else:
                pbp = pd.read_csv(pbp_file, low_memory=False)

            # Filter to requested season if combined file
            if "season" in pbp.columns and pbp["season"].nunique() > 1:
                pbp = pbp[pbp["season"] == season].copy()

            logger.info(f"Loaded {len(pbp)} plays for season {season} from {pbp_file}")

            # Save to processed dir for consistency
            output_file = self.processed_dir / f"pbp_{season}.parquet"
            if not output_file.exists():
                pbp.to_parquet(output_file, index=False)
                logger.info(f"Cached PBP to {output_file}")

            return pbp
        else:
            raise FileNotFoundError(
                f"PBP data not found. Please fetch with R first:\n"
                f"  Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last"
            )

    def fetch_nflverse_team_stats(self, season: int = 2025) -> pd.DataFrame:
        """Fetch team-level stats from nflverse (via R-fetched files or direct download).

        Args:
            season: Season (must be 2025)

        Returns:
            DataFrame with team stats
        """
        season = settings.validate_season(season)
        logger.info(f"Loading nflverse team stats for season {season}...")

        # Try to load from R-fetched data first
        nflverse_dir = Path("data/nflverse")

        # Check for team stats file
        stats_file = nflverse_dir / f"team_stats_{season}.parquet"
        if not stats_file.exists():
            stats_file = nflverse_dir / "team_stats.parquet"
        if not stats_file.exists():
            stats_file = nflverse_dir / "team_stats.csv"

        if stats_file.exists():
            if stats_file.suffix == ".parquet":
                team_stats = pd.read_parquet(stats_file)
            else:
                team_stats = pd.read_csv(stats_file, low_memory=False)

            # Filter to requested season if needed
            if "season" in team_stats.columns:
                team_stats = team_stats[team_stats["season"] == season].copy()

            logger.info(f"Loaded stats for {len(team_stats)} team-weeks for season {season}")

            # Save to processed dir for consistency
            output_file = self.processed_dir / f"team_stats_{season}.parquet"
            if not output_file.exists():
                team_stats.to_parquet(output_file, index=False)
                logger.info(f"Cached team stats to {output_file}")

            return team_stats
        else:
            # Fallback: download directly from NFLverse GitHub
            logger.info("Team stats file not found locally, downloading from NFLverse...")
            url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"
            try:
                team_stats = pd.read_parquet(url)
                if "season" in team_stats.columns:
                    team_stats = team_stats[team_stats["season"] == season].copy()
                logger.info(f"Downloaded {len(team_stats)} team stat records")

                output_file = self.processed_dir / f"team_stats_{season}.parquet"
                team_stats.to_parquet(output_file, index=False)
                logger.info(f"Saved team stats to {output_file}")

                return team_stats
            except Exception as e:
                raise FileNotFoundError(
                    f"Team stats not available. Run R fetch first:\n"
                    f"  Rscript scripts/fetch/fetch_nflverse_data.R\n"
                    f"Error: {e}"
                )

    def fetch_nflverse_schedule(self, season: int = 2025) -> pd.DataFrame:
        """Fetch schedule from nflverse (via R-fetched files or direct download).

        Args:
            season: Season (must be 2025)

        Returns:
            DataFrame with schedule
        """
        season = settings.validate_season(season)
        logger.info(f"Loading nflverse schedule for season {season}...")

        # Try to load from R-fetched data first
        nflverse_dir = Path("data/nflverse")

        # Check for schedules file
        sched_file = nflverse_dir / "schedules.parquet"
        if not sched_file.exists():
            sched_file = nflverse_dir / "schedules.csv"
        if not sched_file.exists():
            sched_file = nflverse_dir / "games.parquet"

        if sched_file.exists():
            if sched_file.suffix == ".parquet":
                schedule = pd.read_parquet(sched_file)
            else:
                schedule = pd.read_csv(sched_file, low_memory=False)

            # Filter to requested season
            if "season" in schedule.columns:
                schedule = schedule[schedule["season"] == season].copy()

            logger.info(f"Loaded {len(schedule)} games for season {season}")

            # Save to processed dir for consistency
            output_file = self.processed_dir / f"schedule_{season}.parquet"
            if not output_file.exists():
                schedule.to_parquet(output_file, index=False)
                logger.info(f"Cached schedule to {output_file}")

            return schedule
        else:
            # Fallback: download directly from NFLverse GitHub
            logger.info("Schedule file not found locally, downloading from NFLverse...")
            url = "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.parquet"
            try:
                schedule = pd.read_parquet(url)
                if "season" in schedule.columns:
                    schedule = schedule[schedule["season"] == season].copy()
                logger.info(f"Downloaded {len(schedule)} games")

                output_file = self.processed_dir / f"schedule_{season}.parquet"
                schedule.to_parquet(output_file, index=False)
                logger.info(f"Saved schedule to {output_file}")

                return schedule
            except Exception as e:
                raise FileNotFoundError(
                    f"Schedule not available. Run R fetch first:\n"
                    f"  Rscript scripts/fetch/fetch_nflverse_data.R\n"
                    f"Error: {e}"
                )

    def load_pbp_parquet(self, season: int = 2025) -> pd.DataFrame:
        """Load PBP from parquet cache.

        Args:
            season: Season to load

        Returns:
            PBP DataFrame
        """
        file_path = self.processed_dir / f"pbp_{season}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"PBP cache not found: {file_path}")

        return pd.read_parquet(file_path)

    def load_team_stats_parquet(self, season: int = 2025) -> pd.DataFrame:
        """Load team stats from parquet cache.

        Args:
            season: Season to load

        Returns:
            Team stats DataFrame
        """
        file_path = self.processed_dir / f"team_stats_{season}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Team stats cache not found: {file_path}")

        return pd.read_parquet(file_path)

    def load_schedule_parquet(self, season: int = 2025) -> pd.DataFrame:
        """Load schedule from parquet cache.

        Args:
            season: Season to load

        Returns:
            Schedule DataFrame
        """
        file_path = self.processed_dir / f"schedule_{season}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Schedule cache not found: {file_path}")

        return pd.read_parquet(file_path)



