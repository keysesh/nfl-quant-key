"""
Extract team-level metrics for player prop predictions.

Includes:
- Offensive pace (plays per game)
- Seconds per play
- Play-type tendencies
- Situational usage patterns
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TeamMetricsExtractor:
    """Extract team-level metrics from play-by-play data."""

    def __init__(self, pbp_path: Path = None, season: Optional[int] = None):
        """
        Initialize with PBP data.

        Args:
            pbp_path: Path to play-by-play parquet file
            season: Season to load (defaults to current season)
        """
        # Use FRESH generic PBP as primary source (not stale season-specific files)
        if pbp_path is None:
            pbp_path = Path('data/nflverse/pbp.parquet')
            if not pbp_path.exists():
                raise FileNotFoundError(
                    f"PBP file not found: {pbp_path}. "
                    "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
                )

        logger.info(f"Loading PBP data from {pbp_path}")
        self.pbp_df = pd.read_parquet(pbp_path)

        # Filter to requested season if provided
        if season is not None and 'season' in self.pbp_df.columns:
            self.pbp_df = self.pbp_df[self.pbp_df['season'] == season]
            logger.info(f"Filtered to season {season}")
        logger.info(f"Loaded {len(self.pbp_df):,} plays")

        # Pre-compute team metrics
        self._compute_team_stats()

    def _compute_team_stats(self):
        """Pre-compute all team-level statistics."""
        logger.info("Computing team statistics...")

        # Offensive plays per game
        plays_per_game = (
            self.pbp_df[self.pbp_df['play_type'].isin(['pass', 'run'])]
            .groupby(['posteam', 'game_id'])
            .size()
            .reset_index(name='plays')
        )

        team_pace = (
            plays_per_game
            .groupby('posteam')
            .agg({
                'plays': ['mean', 'std', 'count']
            })
            .reset_index()
        )
        team_pace.columns = ['team', 'plays_per_game', 'plays_std', 'game_count']

        # Pass rate
        pass_rate = (
            self.pbp_df[self.pbp_df['play_type'].isin(['pass', 'run'])]
            .groupby('posteam')
            .agg({
                'play_type': lambda x: (x == 'pass').mean()
            })
            .reset_index()
        )
        pass_rate.columns = ['team', 'pass_rate']

        # Merge
        self.team_stats = team_pace.merge(pass_rate, on='team')

        logger.info(f"  Computed metrics for {len(self.team_stats)} teams")

    def get_team_pace(
        self,
        team: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get team's offensive pace.

        Args:
            team: Team abbreviation (e.g., 'KC')
            current_week: Current week number
            trailing_weeks: Number of weeks to average

        Returns:
            Dict with pace metrics
        """
        # Week range
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            logger.warning(f"No trailing data for week {current_week}")
            return self._get_default_pace()

        # Filter to trailing weeks
        team_plays = self.pbp_df[
            (self.pbp_df['posteam'] == team) &
            (self.pbp_df['play_type'].isin(['pass', 'run'])) &
            (self.pbp_df['week'] >= start_week) &
            (self.pbp_df['week'] <= end_week)
        ]

        if len(team_plays) == 0:
            logger.warning(f"No data for {team} in weeks {start_week}-{end_week}")
            return self._get_default_pace()

        # Calculate pace
        plays_per_game = (
            team_plays
            .groupby('game_id')
            .size()
            .mean()
        )

        # Pass rate
        pass_rate = (team_plays['play_type'] == 'pass').mean()

        # Plays in neutral game script (within 1 score)
        neutral_plays = team_plays[
            (team_plays['score_differential'].abs() <= 8) |
            (team_plays['score_differential'].isna())
        ]

        neutral_plays_per_game = (
            neutral_plays.groupby('game_id').size().mean()
            if len(neutral_plays) > 0 else plays_per_game
        )

        return {
            'plays_per_game': float(plays_per_game),
            'neutral_plays_per_game': float(neutral_plays_per_game),
            'pass_rate': float(pass_rate),
            'games': len(team_plays['game_id'].unique()),
        }

    def get_combined_pace(
        self,
        offense_team: str,
        defense_team: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get expected combined pace for a matchup.

        Args:
            offense_team: Offensive team
            defense_team: Defensive team
            current_week: Current week
            trailing_weeks: Trailing window

        Returns:
            float: Expected plays for offensive team
        """
        off_pace = self.get_team_pace(offense_team, current_week, trailing_weeks)
        def_pace = self.get_team_pace(defense_team, current_week, trailing_weeks)

        # Defensive pace = plays they allow per game
        # (inverse of their own offensive pace as proxy)
        league_avg = 65.0

        # Weight offensive pace 60%, defensive pace 40%
        expected_pace = (
            0.6 * off_pace['plays_per_game'] +
            0.4 * (2 * league_avg - def_pace['plays_per_game'])
        )

        return float(expected_pace)

    def _get_default_pace(self) -> Dict[str, float]:
        """Get league average pace as fallback."""
        return {
            'plays_per_game': 65.0,
            'neutral_plays_per_game': 65.0,
            'pass_rate': 0.58,
            'games': 0,
        }


# Singleton instance for reuse
_TEAM_METRICS_EXTRACTOR = None


def get_team_metrics_extractor() -> TeamMetricsExtractor:
    """Get or create team metrics extractor singleton."""
    global _TEAM_METRICS_EXTRACTOR
    if _TEAM_METRICS_EXTRACTOR is None:
        _TEAM_METRICS_EXTRACTOR = TeamMetricsExtractor()
    return _TEAM_METRICS_EXTRACTOR
