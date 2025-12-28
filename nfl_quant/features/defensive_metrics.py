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
        # Use FRESH generic PBP as primary source (not stale season-specific files)
        # See .claude/rules/data-freshness.md - NO FALLBACK to stale files
        if pbp_path is None:
            # Primary: fresh generic file updated by R script daily
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
            logger.info(f"Filtered to season {season}: {len(self.pbp_df):,} plays")
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


    def get_defense_vs_wr_depth(
        self,
        defense_team: str,
        depth_rank: int,
        current_week: int,
        trailing_weeks: int = 4,
        weekly_stats: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Get defensive performance against WRs by depth chart position.

        WR1 is identified as the WR with highest target share on each opposing team.
        This addresses the gap where all WRs were grouped together in defense stats.

        Args:
            defense_team: Team abbreviation (e.g., 'HOU')
            depth_rank: 1 = WR1, 2 = WR2, 3 = WR3
            current_week: Current week number
            trailing_weeks: Number of weeks to average
            weekly_stats: Optional pre-loaded weekly stats DataFrame

        Returns:
            Dict with WR depth-specific defensive metrics:
            - avg_receptions_allowed: Avg receptions allowed to WR at this depth
            - over_7_rate: % of WRs at this depth who hit 7+ receptions
            - catch_rate_allowed: Catch rate allowed to WR at this depth
        """
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            return self._get_default_wr_depth_stats()

        # Load weekly stats if not provided
        if weekly_stats is None:
            try:
                weekly_stats = pd.read_parquet('data/nflverse/weekly_stats.parquet')
            except FileNotFoundError:
                logger.warning("weekly_stats.parquet not found")
                return self._get_default_wr_depth_stats()

        # Filter to WRs in the trailing window
        ws = weekly_stats[
            (weekly_stats['position'] == 'WR') &
            (weekly_stats['week'] >= start_week) &
            (weekly_stats['week'] <= end_week) &
            (weekly_stats['season'] == 2024)  # Current season
        ].copy()

        if len(ws) == 0:
            return self._get_default_wr_depth_stats()

        # Identify WR1 for each team-week based on target share
        # WR1 = WR with highest targets on the team that week
        ws['targets'] = ws.get('targets', ws.get('receptions', 0) / 0.65)  # Estimate if missing
        ws['receptions'] = ws.get('receptions', 0)

        # Rank WRs within each team-week by targets
        ws['wr_depth_rank'] = ws.groupby(['team', 'week'])['targets'].rank(
            method='first', ascending=False
        )

        # Get WRs at the specified depth rank who played AGAINST the defense_team
        # Find games where defense_team was the opponent
        wr_vs_def = ws[
            (ws['opponent_team'] == defense_team) &
            (ws['wr_depth_rank'] == depth_rank)
        ]

        if len(wr_vs_def) == 0:
            logger.debug(f"No WR{depth_rank} data vs {defense_team}")
            return self._get_default_wr_depth_stats()

        # Calculate defensive metrics against this WR depth
        avg_receptions = float(wr_vs_def['receptions'].mean())
        over_7_count = (wr_vs_def['receptions'] >= 7).sum()
        over_7_rate = over_7_count / len(wr_vs_def) if len(wr_vs_def) > 0 else 0.5

        # Calculate catch rate if we have targets
        if 'targets' in wr_vs_def.columns and wr_vs_def['targets'].sum() > 0:
            catch_rate = wr_vs_def['receptions'].sum() / wr_vs_def['targets'].sum()
        else:
            catch_rate = 0.631  # Actual WR catch rate from data (was 0.65)

        return {
            'avg_receptions_allowed': avg_receptions,
            'over_7_rate': over_7_rate,
            'catch_rate_allowed': catch_rate,
            'sample_size': len(wr_vs_def),
        }

    def _get_default_wr_depth_stats(self) -> Dict[str, float]:
        """Get league average WR1 defensive stats."""
        return {
            'avg_receptions_allowed': 5.5,  # League avg for WR1
            'over_7_rate': 0.40,            # ~40% hit 7+ in typical week
            'catch_rate_allowed': 0.631,    # Actual WR catch rate from data (was 0.65)
            'sample_size': 0,
        }

    def get_rz_td_defense(
        self,
        defense_team: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get opponent's red zone TD defense metrics.

        Computes RZ-specific TD rates from PBP data (yardline_100 <= 20).
        Used to replace hardcoded league averages in TD Poisson model.

        Args:
            defense_team: Team abbreviation (e.g., 'KC')
            current_week: Current week number
            trailing_weeks: Number of weeks to average (default 4)

        Returns:
            Dict with RZ-specific TD rates:
            - rz_pass_td_rate: Pass TDs allowed / RZ pass attempts
            - rz_rush_td_rate: Rush TDs allowed / RZ rush attempts
            - rz_total_td_rate: Total TDs allowed / RZ plays
            - sample_size: Number of RZ plays in sample
        """
        # Week range for trailing stats (only use prior weeks)
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            logger.warning(f"No trailing data for week {current_week}")
            return self._get_default_rz_td_defense()

        # Filter to red zone plays (yardline_100 <= 20)
        rz_plays = self.pbp_df[
            (self.pbp_df['yardline_100'] <= 20) &
            (self.pbp_df['defteam'] == defense_team) &
            (self.pbp_df['week'] >= start_week) &
            (self.pbp_df['week'] <= end_week) &
            (self.pbp_df['play_type'].isin(['pass', 'run']))
        ]

        if len(rz_plays) == 0:
            logger.warning(f"No RZ plays for {defense_team} in weeks {start_week}-{end_week}")
            return self._get_default_rz_td_defense()

        # Pass plays in red zone
        rz_pass = rz_plays[rz_plays['play_type'] == 'pass']
        rz_pass_attempts = len(rz_pass)
        rz_pass_tds = rz_pass['pass_touchdown'].sum() if 'pass_touchdown' in rz_pass.columns else 0
        rz_pass_td_rate = rz_pass_tds / rz_pass_attempts if rz_pass_attempts > 0 else 0.0

        # Rush plays in red zone
        rz_rush = rz_plays[rz_plays['play_type'] == 'run']
        rz_rush_attempts = len(rz_rush)
        rz_rush_tds = rz_rush['rush_touchdown'].sum() if 'rush_touchdown' in rz_rush.columns else 0
        rz_rush_td_rate = rz_rush_tds / rz_rush_attempts if rz_rush_attempts > 0 else 0.0

        # Total RZ TD rate
        total_rz_plays = len(rz_plays)
        total_rz_tds = rz_pass_tds + rz_rush_tds
        rz_total_td_rate = total_rz_tds / total_rz_plays if total_rz_plays > 0 else 0.0

        # Apply regression to mean for small samples
        # League average RZ TD rate is ~0.20 (20% of RZ plays result in TDs)
        league_avg_rz_td = 0.20
        min_plays_for_full_weight = 40  # ~10 RZ plays per game * 4 weeks

        if total_rz_plays < min_plays_for_full_weight:
            weight = total_rz_plays / min_plays_for_full_weight
            # Actual league averages from PBP data: rush=17.2%, pass=21.4%
            rz_pass_td_rate = weight * rz_pass_td_rate + (1 - weight) * 0.214  # Actual: 21.4%
            rz_rush_td_rate = weight * rz_rush_td_rate + (1 - weight) * 0.172  # Actual: 17.2%
            rz_total_td_rate = weight * rz_total_td_rate + (1 - weight) * league_avg_rz_td

        return {
            'rz_pass_td_rate': float(rz_pass_td_rate),
            'rz_rush_td_rate': float(rz_rush_td_rate),
            'rz_total_td_rate': float(rz_total_td_rate),
            'sample_size': int(total_rz_plays),
        }

    def _get_default_rz_td_defense(self) -> Dict[str, float]:
        """Get league average RZ TD defense stats as fallback.

        Actual rates from PBP data (2023-2024):
        - RZ Rush TD rate: 17.2% (was incorrectly 12%)
        - RZ Pass TD rate: 21.4% (was incorrectly 8%)
        """
        return {
            'rz_pass_td_rate': 0.214,   # 21.4% of RZ pass plays result in TD
            'rz_rush_td_rate': 0.172,   # 17.2% of RZ rush plays result in TD
            'rz_total_td_rate': 0.20,   # ~20% overall RZ TD rate
            'sample_size': 0,
        }


# Singleton instance for reuse
_DEFENSIVE_EXTRACTOR = None


def get_defensive_metrics_extractor() -> DefensiveMetricsExtractor:
    """Get or create defensive metrics extractor singleton."""
    global _DEFENSIVE_EXTRACTOR
    if _DEFENSIVE_EXTRACTOR is None:
        _DEFENSIVE_EXTRACTOR = DefensiveMetricsExtractor()
    return _DEFENSIVE_EXTRACTOR


def get_wr1_defense_feature(
    defense_team: str,
    current_week: int,
    trailing_weeks: int = 4
) -> float:
    """
    Get defense vs WR1 receptions allowed feature.

    Convenience function for feature extraction.

    Args:
        defense_team: Opposing team abbreviation
        current_week: Current week
        trailing_weeks: Trailing window

    Returns:
        Average receptions allowed to opponent WR1s
    """
    extractor = get_defensive_metrics_extractor()
    stats = extractor.get_defense_vs_wr_depth(
        defense_team=defense_team,
        depth_rank=1,  # WR1
        current_week=current_week,
        trailing_weeks=trailing_weeks
    )
    return stats['avg_receptions_allowed']
