"""
Fantasy Football Opportunity Feature Extraction

Uses expected fantasy points data to identify:
- Regression candidates (over/underperforming expected)
- Sustainable vs lucky production
- True opportunity quality vs results

Key concept: Expected FP based on opportunity (targets, carries, red zone)
vs actual FP reveals skill and luck components.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class FFOpportunityFeatureExtractor:
    """Extract expected fantasy points and regression features."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize with FF Opportunity data.

        Args:
            data_dir: Path to nflverse data directory
        """
        if data_dir is None:
            data_dir = Path('data/nflverse')

        self.data_dir = Path(data_dir)
        self.ff_opportunity = None
        self._load_data()

    def _load_data(self):
        """Load FF opportunity parquet file."""
        opp_path = self.data_dir / 'ff_opportunity.parquet'
        if opp_path.exists():
            self.ff_opportunity = pd.read_parquet(opp_path)
            logger.info(f"Loaded FF opportunity: {len(self.ff_opportunity):,} records")
        else:
            logger.warning(f"FF opportunity data not found at {opp_path}")
            # Try to find it in subdirectories
            for subdir in ['weekly', 'historical']:
                alt_path = self.data_dir / subdir / 'ff_opportunity.parquet'
                if alt_path.exists():
                    self.ff_opportunity = pd.read_parquet(alt_path)
                    logger.info(f"Loaded FF opportunity from {alt_path}: {len(self.ff_opportunity):,} records")
                    break

    def get_player_opportunity_features(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get expected fantasy points features for a player.

        Args:
            player_name: Player name
            season: Season year
            week: Current week
            trailing_weeks: Weeks to average

        Returns:
            Dict with opportunity features
        """
        features = {
            'total_fantasy_points_exp': 0.0,
            'total_fantasy_points_actual': 0.0,
            'fantasy_points_diff': 0.0,  # Actual - Expected (+ = overperforming)
            'rush_fantasy_points_exp': 0.0,
            'rec_fantasy_points_exp': 0.0,
            'pass_fantasy_points_exp': 0.0,
            'rush_fp_diff': 0.0,
            'rec_fp_diff': 0.0,
            'pass_fp_diff': 0.0,
            'opportunity_quality': 0.5,  # 0-1 scale
            'efficiency_vs_expected': 1.0,  # Ratio of actual/expected
        }

        if self.ff_opportunity is None:
            logger.debug("FF opportunity data not available")
            return features

        # Filter to player and trailing weeks
        player_data = self.ff_opportunity[
            (self.ff_opportunity['player_name'].str.contains(player_name, case=False, na=False)) &
            (self.ff_opportunity['season'] == season) &
            (self.ff_opportunity['week'] < week) &
            (self.ff_opportunity['week'] >= max(1, week - trailing_weeks))
        ]

        if len(player_data) == 0:
            # Try partial name match
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                player_data = self.ff_opportunity[
                    (self.ff_opportunity['player_name'].str.contains(last_name, case=False, na=False)) &
                    (self.ff_opportunity['season'] == season) &
                    (self.ff_opportunity['week'] < week) &
                    (self.ff_opportunity['week'] >= max(1, week - trailing_weeks))
                ]

        if len(player_data) == 0:
            logger.debug(f"No FF opportunity data for {player_name}")
            return features

        # Extract expected and actual fantasy points
        if 'total_fantasy_points_exp' in player_data.columns:
            features['total_fantasy_points_exp'] = player_data['total_fantasy_points_exp'].mean()

        if 'total_fantasy_points' in player_data.columns:
            features['total_fantasy_points_actual'] = player_data['total_fantasy_points'].mean()

        # Calculate differential
        if features['total_fantasy_points_exp'] > 0:
            features['fantasy_points_diff'] = (
                features['total_fantasy_points_actual'] - features['total_fantasy_points_exp']
            )
            features['efficiency_vs_expected'] = (
                features['total_fantasy_points_actual'] / features['total_fantasy_points_exp']
            )

        # Component breakdowns
        if 'rush_fantasy_points_exp' in player_data.columns:
            features['rush_fantasy_points_exp'] = player_data['rush_fantasy_points_exp'].mean()
            if 'rush_fantasy_points' in player_data.columns:
                actual_rush = player_data['rush_fantasy_points'].mean()
                features['rush_fp_diff'] = actual_rush - features['rush_fantasy_points_exp']

        if 'rec_fantasy_points_exp' in player_data.columns:
            features['rec_fantasy_points_exp'] = player_data['rec_fantasy_points_exp'].mean()
            if 'rec_fantasy_points' in player_data.columns:
                actual_rec = player_data['rec_fantasy_points'].mean()
                features['rec_fp_diff'] = actual_rec - features['rec_fantasy_points_exp']

        if 'pass_fantasy_points_exp' in player_data.columns:
            features['pass_fantasy_points_exp'] = player_data['pass_fantasy_points_exp'].mean()
            if 'pass_fantasy_points' in player_data.columns:
                actual_pass = player_data['pass_fantasy_points'].mean()
                features['pass_fp_diff'] = actual_pass - features['pass_fantasy_points_exp']

        # Opportunity quality score (based on expected FP relative to position)
        # Higher expected FP = higher quality opportunity
        exp_fp = features['total_fantasy_points_exp']
        if exp_fp >= 20:  # Elite opportunity
            features['opportunity_quality'] = 0.9
        elif exp_fp >= 15:  # Good opportunity
            features['opportunity_quality'] = 0.75
        elif exp_fp >= 10:  # Average opportunity
            features['opportunity_quality'] = 0.6
        elif exp_fp >= 5:  # Low opportunity
            features['opportunity_quality'] = 0.4
        else:
            features['opportunity_quality'] = 0.2

        logger.debug(f"FF opportunity features for {player_name}: {features}")
        return features

    def identify_regression_candidates(
        self,
        season: int,
        week: int,
        min_games: int = 3,
        position_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Identify players due for positive or negative regression.

        Args:
            season: Season year
            week: Current week
            min_games: Minimum games to qualify
            position_filter: Optional position filter (QB, RB, WR, TE)

        Returns:
            DataFrame with regression candidates
        """
        if self.ff_opportunity is None:
            logger.warning("FF opportunity data not available for regression analysis")
            return pd.DataFrame()

        # Aggregate player data
        player_agg = (
            self.ff_opportunity[
                (self.ff_opportunity['season'] == season) &
                (self.ff_opportunity['week'] < week)
            ]
            .groupby('player_name')
            .agg({
                'total_fantasy_points_exp': 'mean',
                'total_fantasy_points': 'mean',
                'week': 'count'
            })
            .reset_index()
        )

        player_agg.columns = ['player_name', 'avg_exp_fp', 'avg_actual_fp', 'games_played']

        # Filter by minimum games
        player_agg = player_agg[player_agg['games_played'] >= min_games]

        # Calculate differentials
        player_agg['fp_diff'] = player_agg['avg_actual_fp'] - player_agg['avg_exp_fp']
        player_agg['efficiency_ratio'] = player_agg['avg_actual_fp'] / player_agg['avg_exp_fp'].replace(0, 1)

        # Classify regression type
        def classify_regression(row):
            if row['fp_diff'] > 3:  # Overperforming by 3+ FP/game
                return 'NEGATIVE_REGRESSION'  # Sell high
            elif row['fp_diff'] < -3:  # Underperforming by 3+ FP/game
                return 'POSITIVE_REGRESSION'  # Buy low
            else:
                return 'STABLE'

        player_agg['regression_type'] = player_agg.apply(classify_regression, axis=1)

        # Sort by magnitude of differential
        player_agg = player_agg.sort_values('fp_diff', key=abs, ascending=False)

        return player_agg

    def get_betting_adjustment(
        self,
        player_name: str,
        season: int,
        week: int,
        market_type: str  # 'passing_yards', 'rushing_yards', 'receiving_yards', etc.
    ) -> float:
        """
        Get a betting line adjustment based on regression analysis.

        Positive = player likely to exceed expectations (bet OVER)
        Negative = player likely to underperform (bet UNDER)

        Args:
            player_name: Player name
            season: Season year
            week: Current week
            market_type: Type of prop market

        Returns:
            Adjustment factor (-1 to +1 scale)
        """
        features = self.get_player_opportunity_features(player_name, season, week)

        # Base adjustment on efficiency vs expected
        eff_ratio = features.get('efficiency_vs_expected', 1.0)

        # If player is overperforming (ratio > 1.1), expect regression down
        # If underperforming (ratio < 0.9), expect regression up
        if eff_ratio > 1.15:
            adjustment = -0.3  # Strong negative regression expected (bet UNDER)
        elif eff_ratio > 1.05:
            adjustment = -0.15  # Mild negative regression
        elif eff_ratio < 0.85:
            adjustment = 0.3  # Strong positive regression expected (bet OVER)
        elif eff_ratio < 0.95:
            adjustment = 0.15  # Mild positive regression
        else:
            adjustment = 0.0  # Stable performance

        # Market-specific adjustments
        if market_type in ['rushing_yards', 'player_rush_yds']:
            rush_diff = features.get('rush_fp_diff', 0)
            if rush_diff > 2:
                adjustment -= 0.2  # Overperforming rushing
            elif rush_diff < -2:
                adjustment += 0.2  # Underperforming rushing

        elif market_type in ['receiving_yards', 'player_reception_yds', 'receptions']:
            rec_diff = features.get('rec_fp_diff', 0)
            if rec_diff > 2:
                adjustment -= 0.2  # Overperforming receiving
            elif rec_diff < -2:
                adjustment += 0.2  # Underperforming receiving

        return np.clip(adjustment, -1.0, 1.0)

    def calculate_sustainable_production_score(
        self,
        player_name: str,
        season: int,
        week: int
    ) -> float:
        """
        Calculate how sustainable a player's production is (0-1 scale).

        High score = production based on skill and opportunity
        Low score = production based on luck/variance

        Args:
            player_name: Player name
            season: Season year
            week: Current week

        Returns:
            Sustainability score (0-1)
        """
        features = self.get_player_opportunity_features(player_name, season, week)

        eff_ratio = features.get('efficiency_vs_expected', 1.0)
        opp_quality = features.get('opportunity_quality', 0.5)

        # Sustainable production = good opportunity + performing near expectation
        if 0.9 <= eff_ratio <= 1.1:
            # Performing as expected - sustainable
            sustainability = 0.8 * opp_quality
        elif 0.8 <= eff_ratio <= 1.2:
            # Slight deviation - mostly sustainable
            sustainability = 0.6 * opp_quality
        else:
            # Large deviation - less sustainable (will regress)
            sustainability = 0.3 * opp_quality

        # High opportunity quality adds to sustainability
        sustainability = min(1.0, sustainability + opp_quality * 0.2)

        return sustainability


# Convenience function
def get_regression_adjustment(
    player_name: str,
    market_type: str,
    season: int = 2025,
    week: int = 11,
    data_dir: Path = None
) -> float:
    """
    Quick function to get regression-based betting adjustment.

    Args:
        player_name: Player name
        market_type: Prop market type
        season: Season year
        week: Current week
        data_dir: Data directory path

    Returns:
        Adjustment factor (-1 to +1)
    """
    extractor = FFOpportunityFeatureExtractor(data_dir)
    return extractor.get_betting_adjustment(player_name, season, week, market_type)
