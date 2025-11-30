"""
Next Gen Stats Feature Extraction Module

Extracts advanced metrics from NFL Next Gen Stats data:
- QB: Time to throw, completion % above expectation, aggressiveness
- WR/TE: Separation, cushion, YAC above expectation
- RB: Rush yards over expected, efficiency, time to LOS

These metrics isolate player skill from situational factors.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class NGSFeatureExtractor:
    """Extract Next Gen Stats features for player prop predictions."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize with NGS data directory.

        Args:
            data_dir: Path to nflverse data directory (default: data/nflverse)
        """
        if data_dir is None:
            data_dir = Path('data/nflverse')

        self.data_dir = Path(data_dir)
        self.ngs_passing = None
        self.ngs_receiving = None
        self.ngs_rushing = None
        self._load_data()

    def _load_data(self):
        """Load NGS parquet files if available."""
        # Try to load NGS passing data
        passing_path = self.data_dir / 'ngs_passing.parquet'
        if passing_path.exists():
            self.ngs_passing = pd.read_parquet(passing_path)
            logger.info(f"Loaded NGS passing: {len(self.ngs_passing):,} records")
        else:
            # Try historical fallback
            hist_path = self.data_dir / 'ngs_passing_historical.parquet'
            if hist_path.exists():
                self.ngs_passing = pd.read_parquet(hist_path)
                logger.info(f"Loaded NGS passing (historical): {len(self.ngs_passing):,} records")
            else:
                logger.warning(f"NGS passing data not found at {passing_path}")

        # Try to load NGS receiving data
        receiving_path = self.data_dir / 'ngs_receiving.parquet'
        if receiving_path.exists():
            self.ngs_receiving = pd.read_parquet(receiving_path)
            logger.info(f"Loaded NGS receiving: {len(self.ngs_receiving):,} records")
        else:
            hist_path = self.data_dir / 'ngs_receiving_historical.parquet'
            if hist_path.exists():
                self.ngs_receiving = pd.read_parquet(hist_path)
                logger.info(f"Loaded NGS receiving (historical): {len(self.ngs_receiving):,} records")
            else:
                logger.warning(f"NGS receiving data not found at {receiving_path}")

        # Try to load NGS rushing data
        rushing_path = self.data_dir / 'ngs_rushing.parquet'
        if rushing_path.exists():
            self.ngs_rushing = pd.read_parquet(rushing_path)
            logger.info(f"Loaded NGS rushing: {len(self.ngs_rushing):,} records")
        else:
            hist_path = self.data_dir / 'ngs_rushing_historical.parquet'
            if hist_path.exists():
                self.ngs_rushing = pd.read_parquet(hist_path)
                logger.info(f"Loaded NGS rushing (historical): {len(self.ngs_rushing):,} records")
            else:
                logger.warning(f"NGS rushing data not found at {rushing_path}")

    def get_qb_features(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get QB advanced metrics from NGS data.

        Args:
            player_name: QB name (e.g., "Patrick Mahomes")
            season: Season year
            week: Current week number
            trailing_weeks: Number of weeks to average

        Returns:
            Dict with QB NGS features
        """
        features = {
            'avg_time_to_throw': 2.7,  # League average default
            'completion_pct_above_exp': 0.0,
            'aggressiveness': 0.18,  # League average ~18%
            'avg_air_yards_to_sticks': 0.0,
            'passer_rating_ngs': 90.0,
            'expected_completion_pct': 0.65,
        }

        if self.ngs_passing is None:
            logger.debug("NGS passing data not available, using defaults")
            return features

        # Filter to player and trailing weeks
        player_data = self.ngs_passing[
            (self.ngs_passing['player_display_name'].str.contains(player_name, case=False, na=False)) &
            (self.ngs_passing['season'] == season) &
            (self.ngs_passing['week'] < week) &
            (self.ngs_passing['week'] >= max(1, week - trailing_weeks))
        ]

        if len(player_data) == 0:
            # Try partial name match
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                player_data = self.ngs_passing[
                    (self.ngs_passing['player_display_name'].str.contains(last_name, case=False, na=False)) &
                    (self.ngs_passing['season'] == season) &
                    (self.ngs_passing['week'] < week) &
                    (self.ngs_passing['week'] >= max(1, week - trailing_weeks))
                ]

        if len(player_data) == 0:
            logger.debug(f"No NGS passing data found for {player_name}")
            return features

        # Extract averages
        if 'avg_time_to_throw' in player_data.columns:
            features['avg_time_to_throw'] = player_data['avg_time_to_throw'].mean()

        if 'completion_percentage_above_expectation' in player_data.columns:
            features['completion_pct_above_exp'] = player_data['completion_percentage_above_expectation'].mean()

        if 'aggressiveness' in player_data.columns:
            features['aggressiveness'] = player_data['aggressiveness'].mean()

        if 'avg_air_yards_to_sticks' in player_data.columns:
            features['avg_air_yards_to_sticks'] = player_data['avg_air_yards_to_sticks'].mean()

        if 'passer_rating' in player_data.columns:
            features['passer_rating_ngs'] = player_data['passer_rating'].mean()

        if 'expected_completion_percentage' in player_data.columns:
            features['expected_completion_pct'] = player_data['expected_completion_percentage'].mean()

        logger.debug(f"NGS QB features for {player_name}: {features}")
        return features

    def get_receiver_features(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get WR/TE advanced metrics from NGS data.

        Args:
            player_name: Receiver name
            season: Season year
            week: Current week number
            trailing_weeks: Number of weeks to average

        Returns:
            Dict with receiver NGS features
        """
        features = {
            'avg_separation': 2.5,  # League average ~2.5 yards
            'avg_cushion': 6.0,  # League average ~6 yards
            'avg_yac_above_expectation': 0.0,
            'catch_percentage': 0.65,  # League average
            'avg_intended_air_yards': 10.0,
            'percent_share_of_intended_air_yards': 0.15,
        }

        if self.ngs_receiving is None:
            logger.debug("NGS receiving data not available, using defaults")
            return features

        # Filter to player and trailing weeks
        player_data = self.ngs_receiving[
            (self.ngs_receiving['player_display_name'].str.contains(player_name, case=False, na=False)) &
            (self.ngs_receiving['season'] == season) &
            (self.ngs_receiving['week'] < week) &
            (self.ngs_receiving['week'] >= max(1, week - trailing_weeks))
        ]

        if len(player_data) == 0:
            # Try partial name match
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                player_data = self.ngs_receiving[
                    (self.ngs_receiving['player_display_name'].str.contains(last_name, case=False, na=False)) &
                    (self.ngs_receiving['season'] == season) &
                    (self.ngs_receiving['week'] < week) &
                    (self.ngs_receiving['week'] >= max(1, week - trailing_weeks))
                ]

        if len(player_data) == 0:
            logger.debug(f"No NGS receiving data found for {player_name}")
            return features

        # Extract averages
        if 'avg_separation' in player_data.columns:
            features['avg_separation'] = player_data['avg_separation'].mean()

        if 'avg_cushion' in player_data.columns:
            features['avg_cushion'] = player_data['avg_cushion'].mean()

        if 'avg_yac_above_expectation' in player_data.columns:
            features['avg_yac_above_expectation'] = player_data['avg_yac_above_expectation'].mean()

        if 'catch_percentage' in player_data.columns:
            features['catch_percentage'] = player_data['catch_percentage'].mean()

        if 'avg_intended_air_yards' in player_data.columns:
            features['avg_intended_air_yards'] = player_data['avg_intended_air_yards'].mean()

        if 'percent_share_of_intended_air_yards' in player_data.columns:
            features['percent_share_of_intended_air_yards'] = player_data['percent_share_of_intended_air_yards'].mean()

        logger.debug(f"NGS receiver features for {player_name}: {features}")
        return features

    def get_rusher_features(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get RB advanced metrics from NGS data.

        Args:
            player_name: RB name
            season: Season year
            week: Current week number
            trailing_weeks: Number of weeks to average

        Returns:
            Dict with RB NGS features
        """
        features = {
            'rush_yards_over_expected_per_att': 0.0,  # League average = 0
            'efficiency': 4.0,  # Lower = more north/south running
            'percent_attempts_gte_eight_defenders': 0.25,  # ~25% of runs face 8+ box
            'avg_time_to_los': 2.5,  # Seconds to line of scrimmage
            'expected_rush_yards': 4.0,  # League average
            'rush_pct_over_expected': 0.0,
        }

        if self.ngs_rushing is None:
            logger.debug("NGS rushing data not available, using defaults")
            return features

        # Filter to player and trailing weeks
        player_data = self.ngs_rushing[
            (self.ngs_rushing['player_display_name'].str.contains(player_name, case=False, na=False)) &
            (self.ngs_rushing['season'] == season) &
            (self.ngs_rushing['week'] < week) &
            (self.ngs_rushing['week'] >= max(1, week - trailing_weeks))
        ]

        if len(player_data) == 0:
            # Try partial name match
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                player_data = self.ngs_rushing[
                    (self.ngs_rushing['player_display_name'].str.contains(last_name, case=False, na=False)) &
                    (self.ngs_rushing['season'] == season) &
                    (self.ngs_rushing['week'] < week) &
                    (self.ngs_rushing['week'] >= max(1, week - trailing_weeks))
                ]

        if len(player_data) == 0:
            logger.debug(f"No NGS rushing data found for {player_name}")
            return features

        # Extract averages
        if 'rush_yards_over_expected_per_att' in player_data.columns:
            features['rush_yards_over_expected_per_att'] = player_data['rush_yards_over_expected_per_att'].mean()

        if 'efficiency' in player_data.columns:
            features['efficiency'] = player_data['efficiency'].mean()

        if 'percent_attempts_gte_eight_defenders' in player_data.columns:
            features['percent_attempts_gte_eight_defenders'] = player_data['percent_attempts_gte_eight_defenders'].mean()

        if 'avg_time_to_los' in player_data.columns:
            features['avg_time_to_los'] = player_data['avg_time_to_los'].mean()

        if 'expected_rush_yards' in player_data.columns:
            features['expected_rush_yards'] = player_data['expected_rush_yards'].mean()

        if 'rush_pct_over_expected' in player_data.columns:
            features['rush_pct_over_expected'] = player_data['rush_pct_over_expected'].mean()

        logger.debug(f"NGS rusher features for {player_name}: {features}")
        return features

    def get_player_features(
        self,
        player_name: str,
        position: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get all NGS features for a player based on position.

        Args:
            player_name: Player name
            position: Position (QB, RB, WR, TE)
            season: Season year
            week: Current week
            trailing_weeks: Weeks to average

        Returns:
            Combined NGS features dict
        """
        if position == 'QB':
            return self.get_qb_features(player_name, season, week, trailing_weeks)
        elif position == 'RB':
            # RBs can have both rushing and receiving features
            rush_features = self.get_rusher_features(player_name, season, week, trailing_weeks)
            rec_features = self.get_receiver_features(player_name, season, week, trailing_weeks)
            # Prefix receiving features for RBs
            rb_rec_features = {f'rb_rec_{k}': v for k, v in rec_features.items()}
            return {**rush_features, **rb_rec_features}
        elif position in ['WR', 'TE']:
            return self.get_receiver_features(player_name, season, week, trailing_weeks)
        else:
            logger.warning(f"Unknown position {position} for NGS features")
            return {}

    def calculate_skill_score(
        self,
        player_name: str,
        position: str,
        season: int,
        week: int
    ) -> float:
        """
        Calculate a composite skill score isolating player talent from situation.

        Higher score = more skilled player (talent > situation)
        Lower score = more situational player (opportunity > skill)

        Args:
            player_name: Player name
            position: Position
            season: Season year
            week: Current week

        Returns:
            Skill score from -1 to +1 (normalized)
        """
        features = self.get_player_features(player_name, position, season, week)

        if not features:
            return 0.0

        if position == 'QB':
            # QB skill = CPOE + low time to throw + high passer rating
            cpoe_score = features.get('completion_pct_above_exp', 0) / 10  # Normalize
            ttt_score = (2.7 - features.get('avg_time_to_throw', 2.7)) / 0.5  # Lower is better
            rating_score = (features.get('passer_rating_ngs', 90) - 90) / 20
            return np.clip((cpoe_score + ttt_score + rating_score) / 3, -1, 1)

        elif position == 'RB':
            # RB skill = RYOE + efficiency
            ryoe_score = features.get('rush_yards_over_expected_per_att', 0) / 2
            efficiency_score = (4.0 - features.get('efficiency', 4.0)) / 1.0  # Lower is better
            return np.clip((ryoe_score + efficiency_score) / 2, -1, 1)

        elif position in ['WR', 'TE']:
            # WR/TE skill = separation + YAC above expected
            sep_score = (features.get('avg_separation', 2.5) - 2.5) / 1.0
            yac_score = features.get('avg_yac_above_expectation', 0) / 2
            return np.clip((sep_score + yac_score) / 2, -1, 1)

        return 0.0

    def get_regression_indicators(
        self,
        player_name: str,
        position: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Identify if player is due for positive or negative regression.

        Returns:
            Dict with regression indicators:
            - positive_regression: Player underperforming skill (buy low)
            - negative_regression: Player overperforming skill (sell high)
        """
        features = self.get_player_features(player_name, position, season, week)

        indicators = {
            'positive_regression_indicator': 0.0,  # Likely to improve
            'negative_regression_indicator': 0.0,  # Likely to regress
            'sustainable_production': 0.5,  # 0-1 scale
        }

        if position == 'QB':
            # High CPOE but low actual completion % = positive regression
            cpoe = features.get('completion_pct_above_exp', 0)
            if cpoe > 3:  # Very skilled
                indicators['sustainable_production'] = 0.8
            elif cpoe < -3:  # Overperforming
                indicators['negative_regression_indicator'] = 0.6

        elif position == 'RB':
            ryoe = features.get('rush_yards_over_expected_per_att', 0)
            if ryoe > 1.0:  # Elite skill
                indicators['sustainable_production'] = 0.9
            elif ryoe < -1.0:  # Bad skill, will regress down
                indicators['negative_regression_indicator'] = 0.7

        elif position in ['WR', 'TE']:
            yac_above = features.get('avg_yac_above_expectation', 0)
            separation = features.get('avg_separation', 2.5)

            if yac_above > 2 and separation > 3:
                indicators['sustainable_production'] = 0.85
            elif yac_above < -2:
                indicators['negative_regression_indicator'] = 0.6

        return indicators


# Convenience function for quick access
def get_ngs_features(
    player_name: str,
    position: str,
    season: int = 2025,
    week: int = 11,
    data_dir: Path = None
) -> Dict[str, float]:
    """
    Quick function to get NGS features for a player.

    Args:
        player_name: Player name
        position: QB, RB, WR, or TE
        season: Season year
        week: Current week
        data_dir: Path to nflverse data

    Returns:
        Dict of NGS features
    """
    extractor = NGSFeatureExtractor(data_dir)
    return extractor.get_player_features(player_name, position, season, week)
