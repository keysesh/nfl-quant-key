"""
Extract opponent defensive metrics for player prop predictions.

This module calculates defense-specific EPA and efficiency metrics vs different positions:
- Pass defense EPA vs QBs
- Rush defense EPA vs RBs
- Coverage EPA vs WRs/TEs
- Yards allowed per attempt by position
- TD rate allowed by position
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional

from nfl_quant.utils.epa_utils import regress_epa_to_mean

logger = logging.getLogger(__name__)


class DefensiveMetricsExtractor:
    """Extract defensive strength metrics from play-by-play data."""

    def __init__(self, pbp_path: Path = None, season: Optional[int] = None):
        """
        Initialize with PBP data.

        Args:
            pbp_path: Path to play-by-play parquet file
            season: Season to load (defaults to current season)
        """
        # Use nflverse data as single source of truth (includes most recent week)
        if pbp_path is None:
            if season is None:
                from nfl_quant.utils.season_utils import get_current_season
                season = get_current_season()
            pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')

        logger.info(f"Loading PBP data from {pbp_path}")
        self.pbp_df = pd.read_parquet(pbp_path)
        logger.info(f"Loaded {len(self.pbp_df):,} plays")

        # Pre-compute defensive metrics
        self._compute_defensive_stats()

    def _compute_defensive_stats(self):
        """Pre-compute all defensive metrics by team and week."""
        logger.info("Computing defensive statistics...")

        # Pass defense vs QBs
        pass_defense = (
            self.pbp_df[self.pbp_df['play_type'] == 'pass']
            .groupby(['defteam', 'week'])
            .agg({
                'epa': 'mean',                    # EPA per pass play (negative = good defense)
                'passing_yards': 'sum',           # Total passing yards allowed
                'pass_touchdown': 'sum',          # Passing TDs allowed
                'complete_pass': 'sum',           # Completions allowed
                'incomplete_pass': 'sum',         # Incompletions (good)
                'sack': 'sum',                    # Sacks (good defense)
            })
            .reset_index()
        )

        # Calculate attempts and rates
        pass_defense['pass_attempts'] = (
            pass_defense['complete_pass'] +
            pass_defense['incomplete_pass'] +
            pass_defense['sack']
        )
        pass_defense['yards_per_attempt'] = (
            pass_defense['passing_yards'] / pass_defense['pass_attempts']
        ).fillna(0)
        pass_defense['td_rate_allowed'] = (
            pass_defense['pass_touchdown'] / pass_defense['pass_attempts']
        ).fillna(0)
        pass_defense['completion_pct_allowed'] = (
            pass_defense['complete_pass'] /
            (pass_defense['complete_pass'] + pass_defense['incomplete_pass'])
        ).fillna(0)

        # Rush defense vs RBs
        rush_defense = (
            self.pbp_df[self.pbp_df['play_type'] == 'run']
            .groupby(['defteam', 'week'])
            .agg({
                'epa': 'mean',                    # EPA per rush play
                'rushing_yards': 'sum',           # Total rushing yards allowed
                'rush_touchdown': 'sum',          # Rushing TDs allowed
            })
            .reset_index()
        )

        # Calculate rush attempts and rates
        rush_attempts = (
            self.pbp_df[self.pbp_df['play_type'] == 'run']
            .groupby(['defteam', 'week'])
            .size()
            .reset_index(name='rush_attempts')
        )
        rush_defense = rush_defense.merge(rush_attempts, on=['defteam', 'week'])

        rush_defense['yards_per_carry_allowed'] = (
            rush_defense['rushing_yards'] / rush_defense['rush_attempts']
        ).fillna(0)
        rush_defense['td_rate_allowed'] = (
            rush_defense['rush_touchdown'] / rush_defense['rush_attempts']
        ).fillna(0)

        # Store for lookup
        self.pass_defense_stats = pass_defense
        self.rush_defense_stats = rush_defense

        logger.info(f"  Pass defense: {len(pass_defense)} team-weeks")
        logger.info(f"  Rush defense: {len(rush_defense)} team-weeks")

    def get_defense_vs_position(
        self,
        defense_team: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get defensive strength vs specific position.

        Args:
            defense_team: Team abbreviation (e.g., 'KC')
            position: Player position ('QB', 'RB', 'WR', 'TE')
            current_week: Current week number
            trailing_weeks: Number of weeks to average (default 4)

        Returns:
            Dict with defensive metrics:
            - epa_per_play: Lower is better defense (negative = good)
            - yards_per_attempt: Lower is better
            - td_rate_allowed: Lower is better
            - etc.
        """
        # Week range for trailing stats
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            logger.warning(f"No trailing data for week {current_week}")
            return self._get_default_defense_stats(position)

        # Get position-specific defensive stats
        if position == 'QB':
            # Pass defense
            defense_data = self.pass_defense_stats[
                (self.pass_defense_stats['defteam'] == defense_team) &
                (self.pass_defense_stats['week'] >= start_week) &
                (self.pass_defense_stats['week'] <= end_week)
            ]

            if len(defense_data) == 0:
                logger.warning(f"No pass defense data for {defense_team} in weeks {start_week}-{end_week}")
                return self._get_default_defense_stats(position)

            # CRITICAL: Apply regression to mean for EPA based on sample size
            raw_epa = float(defense_data['epa'].mean())
            sample_weeks = len(defense_data)
            regressed_epa = regress_epa_to_mean(raw_epa, sample_weeks)

            return {
                'epa_per_play': regressed_epa,
                'yards_per_attempt': float(defense_data['yards_per_attempt'].mean()),
                'td_rate_allowed': float(defense_data['td_rate_allowed'].mean()),
                'completion_pct_allowed': float(defense_data['completion_pct_allowed'].mean()),
                'sacks_per_game': float(defense_data['sack'].mean()),
            }

        elif position == 'RB':
            # Rush defense
            defense_data = self.rush_defense_stats[
                (self.rush_defense_stats['defteam'] == defense_team) &
                (self.rush_defense_stats['week'] >= start_week) &
                (self.rush_defense_stats['week'] <= end_week)
            ]

            if len(defense_data) == 0:
                logger.warning(f"No rush defense data for {defense_team} in weeks {start_week}-{end_week}")
                return self._get_default_defense_stats(position)

            # CRITICAL: Apply regression to mean for EPA based on sample size
            raw_epa = float(defense_data['epa'].mean())
            sample_weeks = len(defense_data)
            regressed_epa = regress_epa_to_mean(raw_epa, sample_weeks)

            return {
                'epa_per_play': regressed_epa,
                'yards_per_carry_allowed': float(defense_data['yards_per_carry_allowed'].mean()),
                'td_rate_allowed': float(defense_data['td_rate_allowed'].mean()),
            }

        elif position in ['WR', 'TE']:
            # Pass defense (coverage)
            defense_data = self.pass_defense_stats[
                (self.pass_defense_stats['defteam'] == defense_team) &
                (self.pass_defense_stats['week'] >= start_week) &
                (self.pass_defense_stats['week'] <= end_week)
            ]

            if len(defense_data) == 0:
                logger.warning(f"No coverage data for {defense_team} in weeks {start_week}-{end_week}")
                return self._get_default_defense_stats(position)

            # CRITICAL: Apply regression to mean for EPA based on sample size
            raw_epa = float(defense_data['epa'].mean())
            sample_weeks = len(defense_data)
            regressed_epa = regress_epa_to_mean(raw_epa, sample_weeks)

            return {
                'epa_per_play': regressed_epa,
                'yards_per_attempt': float(defense_data['yards_per_attempt'].mean()),
                'td_rate_allowed': float(defense_data['td_rate_allowed'].mean()),
                'completion_pct_allowed': float(defense_data['completion_pct_allowed'].mean()),
            }

        else:
            raise ValueError(f"Unsupported position: {position}")

    def _get_default_defense_stats(self, position: str) -> Dict[str, float]:
        """
        Get league average defensive stats as fallback.

        Args:
            position: Player position

        Returns:
            Dict with league average defensive metrics
        """
        if position == 'QB':
            return {
                'epa_per_play': 0.0,              # League average
                'yards_per_attempt': 6.5,
                'td_rate_allowed': 0.04,
                'completion_pct_allowed': 0.64,
                'sacks_per_game': 2.3,
            }
        elif position == 'RB':
            return {
                'epa_per_play': 0.0,
                'yards_per_carry_allowed': 4.3,
                'td_rate_allowed': 0.05,
            }
        elif position in ['WR', 'TE']:
            return {
                'epa_per_play': 0.0,
                'yards_per_attempt': 6.5,
                'td_rate_allowed': 0.04,
                'completion_pct_allowed': 0.64,
            }
        else:
            return {'epa_per_play': 0.0}

    def get_defense_rank(
        self,
        defense_team: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> int:
        """
        Get defensive rank vs position (1 = best defense, 32 = worst).

        Args:
            defense_team: Team abbreviation
            position: Player position
            current_week: Current week
            trailing_weeks: Trailing window

        Returns:
            int: Rank (1-32)
        """
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if position in ['QB', 'WR', 'TE']:
            defense_data = self.pass_defense_stats[
                (self.pass_defense_stats['week'] >= start_week) &
                (self.pass_defense_stats['week'] <= end_week)
            ]

            # Calculate average EPA per team
            team_epa = defense_data.groupby('defteam')['epa'].mean().sort_values()

        elif position == 'RB':
            defense_data = self.rush_defense_stats[
                (self.rush_defense_stats['week'] >= start_week) &
                (self.rush_defense_stats['week'] <= end_week)
            ]

            # Calculate average EPA per team
            team_epa = defense_data.groupby('defteam')['epa'].mean().sort_values()

        else:
            return 16  # Middle rank as default

        # Get rank (1 = lowest EPA = best defense)
        ranks = {team: rank + 1 for rank, team in enumerate(team_epa.index)}

        return ranks.get(defense_team, 16)


# Singleton instance for reuse
_DEFENSIVE_EXTRACTOR = None


def get_defensive_metrics_extractor() -> DefensiveMetricsExtractor:
    """Get or create defensive metrics extractor singleton."""
    global _DEFENSIVE_EXTRACTOR
    if _DEFENSIVE_EXTRACTOR is None:
        _DEFENSIVE_EXTRACTOR = DefensiveMetricsExtractor()
    return _DEFENSIVE_EXTRACTOR
