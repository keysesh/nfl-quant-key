"""
Snap Count Feature Extraction

Uses actual snap participation data instead of inferring from PBP.
Critical for:
- Usage prediction (how many opportunities will player get)
- Role emergence detection (backup becoming starter)
- Workload trends (increasing/decreasing involvement)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SnapCountFeatureExtractor:
    """Extract snap count features for usage prediction."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize with snap counts data.

        Args:
            data_dir: Path to nflverse data directory
        """
        if data_dir is None:
            data_dir = Path('data/nflverse')

        self.data_dir = Path(data_dir)
        self.snap_counts = None
        self._load_data()

    def _load_data(self):
        """Load snap counts parquet file."""
        snap_path = self.data_dir / 'snap_counts.parquet'
        if snap_path.exists():
            self.snap_counts = pd.read_parquet(snap_path)
            logger.info(f"Loaded snap counts: {len(self.snap_counts):,} records")
        else:
            logger.warning(f"Snap counts data not found at {snap_path}")

    def get_player_snap_features(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get snap participation features for a player.

        Args:
            player_name: Player name
            season: Season year
            week: Current week
            trailing_weeks: Weeks to average

        Returns:
            Dict with snap features
        """
        features = {
            'avg_offense_snaps': 0.0,
            'avg_offense_pct': 0.0,
            'snap_trend': 0.0,  # Positive = increasing, negative = decreasing
            'snap_volatility': 0.0,  # Standard deviation of snap %
            'recent_offense_pct': 0.0,  # Most recent week
            'snap_share_vs_team_avg': 1.0,  # Ratio to team average
            'is_primary_option': False,  # >50% snap share
            'role_change_detected': False,  # Significant change in usage
        }

        if self.snap_counts is None:
            logger.debug("Snap counts data not available")
            return features

        # Filter to player and trailing weeks
        player_data = self.snap_counts[
            (self.snap_counts['player'].str.contains(player_name, case=False, na=False)) &
            (self.snap_counts['season'] == season) &
            (self.snap_counts['week'] < week) &
            (self.snap_counts['week'] >= max(1, week - trailing_weeks))
        ].sort_values('week')

        if len(player_data) == 0:
            # Try partial name match (last name)
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                player_data = self.snap_counts[
                    (self.snap_counts['player'].str.contains(last_name, case=False, na=False)) &
                    (self.snap_counts['season'] == season) &
                    (self.snap_counts['week'] < week) &
                    (self.snap_counts['week'] >= max(1, week - trailing_weeks))
                ].sort_values('week')

        if len(player_data) == 0:
            logger.debug(f"No snap count data for {player_name}")
            return features

        # Extract snap metrics
        if 'offense_snaps' in player_data.columns:
            features['avg_offense_snaps'] = player_data['offense_snaps'].mean()

        if 'offense_pct' in player_data.columns:
            snap_pcts = player_data['offense_pct'].values
            features['avg_offense_pct'] = np.mean(snap_pcts)
            features['snap_volatility'] = np.std(snap_pcts) if len(snap_pcts) > 1 else 0.0
            features['recent_offense_pct'] = snap_pcts[-1] if len(snap_pcts) > 0 else 0.0

            # Calculate trend (slope of snap % over weeks)
            if len(snap_pcts) >= 2:
                weeks = np.arange(len(snap_pcts))
                trend = np.polyfit(weeks, snap_pcts, 1)[0]  # Linear slope
                features['snap_trend'] = trend

            # Is primary option (>50% snaps)
            features['is_primary_option'] = features['avg_offense_pct'] > 0.50

            # Detect role change (>10% change from first to last week in window)
            if len(snap_pcts) >= 2:
                change = snap_pcts[-1] - snap_pcts[0]
                features['role_change_detected'] = abs(change) > 0.10

        logger.debug(f"Snap features for {player_name}: {features}")
        return features

    def get_team_snap_distribution(
        self,
        team: str,
        position: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Get team's snap distribution for a position group.

        Args:
            team: Team abbreviation (e.g., 'KC')
            position: Position group (e.g., 'WR', 'RB')
            season: Season year
            week: Current week

        Returns:
            Dict with team snap distribution info
        """
        distribution = {
            'team_total_snaps': 0,
            'position_total_snaps': 0,
            'num_players_with_snaps': 0,
            'concentration': 0.0,  # 0-1, higher = more concentrated to top player
            'top_player_pct': 0.0,
        }

        if self.snap_counts is None:
            return distribution

        # Get team's most recent week data
        team_data = self.snap_counts[
            (self.snap_counts['team'] == team) &
            (self.snap_counts['season'] == season) &
            (self.snap_counts['week'] == week - 1) &  # Previous week
            (self.snap_counts['position'] == position)
        ]

        if len(team_data) == 0:
            return distribution

        distribution['position_total_snaps'] = team_data['offense_snaps'].sum()
        distribution['num_players_with_snaps'] = len(team_data)

        if distribution['position_total_snaps'] > 0:
            snap_shares = team_data['offense_pct'].values
            distribution['top_player_pct'] = snap_shares.max()

            # Herfindahl index for concentration
            hhi = np.sum((snap_shares / snap_shares.sum()) ** 2) if snap_shares.sum() > 0 else 0
            distribution['concentration'] = hhi

        return distribution

    def detect_emerging_players(
        self,
        team: str,
        position: str,
        season: int,
        week: int,
        min_increase: float = 0.15
    ) -> List[Dict]:
        """
        Detect players with rapidly increasing snap shares.

        Args:
            team: Team abbreviation
            position: Position
            season: Season year
            week: Current week
            min_increase: Minimum snap % increase to flag

        Returns:
            List of dicts with emerging player info
        """
        emerging = []

        if self.snap_counts is None:
            return emerging

        # Get all players for this team/position
        team_pos_data = self.snap_counts[
            (self.snap_counts['team'] == team) &
            (self.snap_counts['position'] == position) &
            (self.snap_counts['season'] == season) &
            (self.snap_counts['week'] < week)
        ]

        if len(team_pos_data) == 0:
            return emerging

        # Group by player and calculate trends
        for player_name in team_pos_data['player'].unique():
            player_weeks = team_pos_data[
                team_pos_data['player'] == player_name
            ].sort_values('week')

            if len(player_weeks) >= 3:  # Need at least 3 weeks
                snap_pcts = player_weeks['offense_pct'].values
                first_avg = np.mean(snap_pcts[:2])  # First 2 weeks
                recent_avg = np.mean(snap_pcts[-2:])  # Last 2 weeks

                increase = recent_avg - first_avg

                if increase >= min_increase:
                    emerging.append({
                        'player': player_name,
                        'team': team,
                        'position': position,
                        'snap_increase': increase,
                        'current_snap_pct': snap_pcts[-1],
                        'weeks_trending_up': len([i for i in range(1, len(snap_pcts))
                                                   if snap_pcts[i] > snap_pcts[i - 1]])
                    })

        # Sort by increase
        emerging.sort(key=lambda x: x['snap_increase'], reverse=True)
        return emerging

    def calculate_workload_sustainability(
        self,
        player_name: str,
        position: str,
        season: int,
        week: int
    ) -> float:
        """
        Calculate how sustainable current workload is (injury/fatigue risk).

        Args:
            player_name: Player name
            position: Position
            season: Season year
            week: Current week

        Returns:
            Sustainability score (0-1, lower = higher injury risk)
        """
        features = self.get_player_snap_features(player_name, season, week)

        snap_pct = features['avg_offense_pct']
        snaps = features['avg_offense_snaps']

        # Position-specific thresholds
        if position == 'RB':
            # High touch count RBs have injury risk
            if snap_pct > 0.75 or snaps > 55:
                return 0.5  # High workload, injury risk
            elif snap_pct > 0.60:
                return 0.7
            else:
                return 0.9

        elif position in ['WR', 'TE']:
            # WRs/TEs can sustain high snap counts
            if snap_pct > 0.90:
                return 0.8
            else:
                return 0.95

        elif position == 'QB':
            return 0.95  # QBs typically play every snap

        return 0.85  # Default

    def get_usage_prediction_features(
        self,
        player_name: str,
        position: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Get comprehensive features for usage/opportunity prediction.

        Combines snap counts with trend analysis.
        NO HARDCODED RATES - computes opportunity rates from actual NFLverse data.

        Args:
            player_name: Player name
            position: Position
            season: Season year
            week: Current week

        Returns:
            Features dict optimized for usage prediction
        """
        snap_features = self.get_player_snap_features(player_name, season, week)

        # Calculate expected opportunities based on snap share
        # CRITICAL: Compute position-specific rates from actual NFLverse data
        opportunity_rates = self._get_position_opportunity_rates(position, season)

        if position == 'RB':
            expected_carries = snap_features['avg_offense_snaps'] * opportunity_rates['carries_per_snap']
            expected_targets = snap_features['avg_offense_snaps'] * opportunity_rates['targets_per_snap']
            snap_features['expected_carries'] = expected_carries
            snap_features['expected_targets_rb'] = expected_targets

        elif position in ['WR', 'TE']:
            expected_targets = snap_features['avg_offense_snaps'] * opportunity_rates['targets_per_snap']
            snap_features['expected_targets'] = expected_targets

        elif position == 'QB':
            expected_attempts = snap_features['avg_offense_snaps'] * opportunity_rates['attempts_per_snap']
            snap_features['expected_pass_attempts'] = expected_attempts

        # Add sustainability metric
        snap_features['workload_sustainability'] = self.calculate_workload_sustainability(
            player_name, position, season, week
        )

        return snap_features

    def _get_position_opportunity_rates(self, position: str, season: int) -> Dict[str, float]:
        """
        Calculate position-specific opportunity rates from actual NFLverse data.

        NO HARDCODED RATES - uses actual snap counts and weekly stats.

        Args:
            position: Player position
            season: Season year

        Returns:
            Dict with opportunity rates per snap
        """
        # Load weekly stats to compute actual rates
        weekly_path = self.data_dir / 'weekly_stats.parquet'
        if not weekly_path.exists():
            raise FileNotFoundError(
                f"Weekly stats not found at {weekly_path}. "
                f"NO HARDCODED DEFAULTS - run data/fetch_nflverse_r.R to get actual data."
            )

        weekly = pd.read_parquet(weekly_path)

        # Use previous season for more complete data, or current season if available
        prior_season = season - 1
        season_data = weekly[weekly['season'] == prior_season]

        if season_data.empty:
            # Try current season
            season_data = weekly[weekly['season'] == season]
            if season_data.empty:
                raise ValueError(
                    f"No data found for season {prior_season} or {season}. "
                    f"NO HARDCODED DEFAULTS - ensure NFLverse data is available."
                )

        if self.snap_counts is None or self.snap_counts.empty:
            raise ValueError(
                "Snap counts data not loaded. "
                "NO HARDCODED DEFAULTS - ensure snap_counts.parquet exists."
            )

        # Merge snap counts with weekly stats to compute rates
        snap_season = self.snap_counts[self.snap_counts['season'] == prior_season]
        if snap_season.empty:
            snap_season = self.snap_counts[self.snap_counts['season'] == season]
            if snap_season.empty:
                raise ValueError(
                    f"No snap count data for season {prior_season} or {season}. "
                    f"NO HARDCODED DEFAULTS."
                )

        pos_snap = snap_season[snap_season['position'] == position]
        pos_stats = season_data[season_data['position'] == position]

        if pos_snap.empty or pos_stats.empty:
            raise ValueError(
                f"No {position} data found for computing opportunity rates. "
                f"NO HARDCODED DEFAULTS."
            )

        # Compute average rates from actual data
        rates = {}

        if position == 'RB':
            # Calculate carries per snap and targets per snap for RBs
            total_snaps = pos_snap['offense_snaps'].sum()
            total_carries = pos_stats['carries'].sum()
            total_targets = pos_stats['targets'].sum()

            if total_snaps == 0:
                raise ValueError(f"No RB snaps found. NO HARDCODED DEFAULTS.")

            rates['carries_per_snap'] = total_carries / total_snaps
            rates['targets_per_snap'] = total_targets / total_snaps

            logger.debug(f"RB rates from actual data: {rates['carries_per_snap']:.3f} carries/snap, "
                        f"{rates['targets_per_snap']:.3f} targets/snap")

        elif position in ['WR', 'TE']:
            # Calculate targets per snap for WR/TE
            total_snaps = pos_snap['offense_snaps'].sum()
            total_targets = pos_stats['targets'].sum()

            if total_snaps == 0:
                raise ValueError(f"No {position} snaps found. NO HARDCODED DEFAULTS.")

            rates['targets_per_snap'] = total_targets / total_snaps

            logger.debug(f"{position} rates from actual data: {rates['targets_per_snap']:.3f} targets/snap")

        elif position == 'QB':
            # Calculate pass attempts per snap for QBs
            total_snaps = pos_snap['offense_snaps'].sum()
            total_attempts = pos_stats['attempts'].sum()

            if total_snaps == 0:
                raise ValueError(f"No QB snaps found. NO HARDCODED DEFAULTS.")

            rates['attempts_per_snap'] = total_attempts / total_snaps

            logger.debug(f"QB rates from actual data: {rates['attempts_per_snap']:.3f} attempts/snap")

        else:
            raise ValueError(
                f"Unknown position: {position}. "
                f"NO HARDCODED DEFAULTS - position must be QB, RB, WR, or TE."
            )

        return rates


# Convenience function
def get_snap_features(
    player_name: str,
    season: int = 2025,
    week: int = 11,
    trailing_weeks: int = 4,
    data_dir: Path = None
) -> Dict[str, float]:
    """
    Quick function to get snap count features.

    Args:
        player_name: Player name
        season: Season year
        week: Current week
        trailing_weeks: Weeks to average
        data_dir: Data directory path

    Returns:
        Snap features dict
    """
    extractor = SnapCountFeatureExtractor(data_dir)
    return extractor.get_player_snap_features(player_name, season, week, trailing_weeks)
