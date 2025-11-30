"""
Bayesian Shrinkage Integration
================================

Integrates Bayesian shrinkage into the prediction pipeline to handle
small-sample adjustments automatically.

This module provides wrapper functions that can be dropped into existing
prediction code with minimal changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from nfl_quant.models.bayesian_shrinkage import BayesianShrinker, load_position_priors
from nfl_quant.utils.season_utils import get_current_season, get_training_seasons

logger = logging.getLogger(__name__)


class ShrinkagePredictor:
    """
    Wrapper that adds Bayesian shrinkage to predictions.

    Usage:
        shrinkage = ShrinkagePredictor()
        df_predictions = shrinkage.apply_shrinkage(
            df=df_predictions,
            position='RB',
            current_week=9
        )
    """

    def __init__(self, season: Optional[int] = None):
        """
        Initialize shrinkage predictors for each position.

        Args:
            season: Season to load priors from (defaults to current season - 1)
        """
        self.shrinkers = {}
        self.season = season
        self._load_priors()

    def _load_priors(self):
        """Load position-specific priors."""
        # Use training seasons to get the most recent completed season
        if self.season is None:
            training_seasons = get_training_seasons()
            self.season = training_seasons[0] if training_seasons else get_current_season() - 1

        for position in ['QB', 'RB', 'WR']:
            shrinker = BayesianShrinker(position=position)
            priors = load_position_priors(position, season=self.season)

            for stat_name, prior in priors.items():
                shrinker.set_prior(stat_name, prior['mean'], prior['std'])

            self.shrinkers[position] = shrinker
            logger.info(f"Loaded {len(priors)} priors for {position} (season={self.season})")

    def apply_shrinkage(
        self,
        df: pd.DataFrame,
        position: str,
        current_week: int = 9,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply Bayesian shrinkage to player predictions.

        Args:
            df: DataFrame with player predictions
            position: Player position (QB, RB, WR)
            current_week: Current NFL week (for sample size calculation)
            season: Season to use (defaults to current season)

        Returns:
            DataFrame with shrunk estimates added
        """
        if position not in self.shrinkers:
            logger.warning(f"No shrinkage available for {position}")
            return df

        shrinker = self.shrinkers[position]

        # Auto-detect season if not provided
        if season is None:
            season = get_current_season()

        # Calculate sample size (weeks completed in current season)
        weeks_completed = current_week - 1  # Week 9 = 8 weeks of data

        # Add sample size column if not present
        games_played_col = f'games_played_{season}'
        if games_played_col not in df.columns:
            # Assume players have played most weeks (conservative estimate)
            df[games_played_col] = max(1, weeks_completed - 1)

        # Apply shrinkage to each relevant stat
        stat_mappings = self._get_stat_mappings(position)

        for col_name, stat_type in stat_mappings.items():
            if col_name in df.columns:
                logger.info(f"Applying shrinkage to {col_name} (n={len(df)})")

                df = shrinker.shrink_dataframe(
                    df=df,
                    stat_col=col_name,
                    sample_size_col=games_played_col,
                    stat_name=stat_type,
                    output_col=f"{col_name}_shrunk"
                )

        return df

    def _get_stat_mappings(self, position: str) -> Dict[str, str]:
        """
        Map DataFrame column names to shrinkage stat types.

        Returns:
            Dictionary mapping column names to stat types
        """
        if position == 'QB':
            return {
                'comp_pct': 'comp_pct',
                'yards_per_completion': 'yards_per_completion',
                'yards_per_carry': 'yards_per_carry',
                'td_rate_pass': 'td_rate',
                'td_rate_rush': 'td_rate',
            }
        elif position == 'RB':
            return {
                'yards_per_carry': 'yards_per_carry',
                'yards_per_target': 'yards_per_target',
                'td_rate_rush': 'td_rate',
                'td_rate_pass': 'td_rate',
            }
        elif position == 'WR':
            return {
                'yards_per_target': 'yards_per_target',
                'td_rate_pass': 'td_rate',
            }
        else:
            return {}


def apply_early_season_shrinkage(
    df: pd.DataFrame,
    current_week: int,
    position_col: str = 'position',
    season: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to apply shrinkage to all positions in a DataFrame.

    Args:
        df: DataFrame with predictions for multiple positions
        current_week: Current NFL week
        position_col: Column containing position labels
        season: Season to use (defaults to current season)

    Returns:
        DataFrame with shrunk estimates added for all positions
    """
    if season is None:
        season = get_current_season()

    if current_week >= 10:
        logger.info(f"Week {current_week}: Late season, shrinkage less critical")
    else:
        logger.info(f"Week {current_week}: Early season, applying aggressive shrinkage")

    shrinkage = ShrinkagePredictor(season=season)

    results = []
    for position in df[position_col].unique():
        df_pos = df[df[position_col] == position].copy()
        df_pos = shrinkage.apply_shrinkage(df_pos, position, current_week, season)
        results.append(df_pos)

    return pd.concat(results, ignore_index=True)


def should_use_shrinkage(current_week: int, player_games: int) -> bool:
    """
    Determine if shrinkage should be applied based on context.

    Args:
        current_week: Current NFL week
        player_games: Number of games player has played

    Returns:
        True if shrinkage should be applied
    """
    # Always shrink with < 30 observations
    if player_games < 30:
        return True

    # Early season (weeks 1-6): Shrink everyone
    if current_week <= 6:
        return True

    # Mid-season (weeks 7-12): Shrink if < 50 observations
    if current_week <= 12 and player_games < 50:
        return True

    # Late season: Only shrink backups
    if player_games < 30:
        return True

    return False


# Example integration into generate_model_predictions.py
def example_integration():
    """
    Example showing how to integrate into generate_model_predictions.py
    """
    print("="*80)
    print("SHRINKAGE INTEGRATION EXAMPLE")
    print("="*80)

    # Get current season dynamically
    current_season = get_current_season()
    games_played_col = f'games_played_{current_season}'

    # Simulated predictions DataFrame
    df_predictions = pd.DataFrame({
        'player': ['Patrick Mahomes', 'Backup QB', 'Derrick Henry', 'Practice Squad RB'],
        'position': ['QB', 'QB', 'RB', 'RB'],
        'yards_per_carry': [5.5, 7.2, 4.8, 6.5],  # Backup/PS have inflated stats
        'td_rate_rush': [0.08, 0.12, 0.03, 0.08],
        games_played_col: [8, 2, 8, 1],  # Small samples for backups
    })

    print("\nBefore Shrinkage:")
    print(df_predictions[['player', 'position', 'yards_per_carry', 'td_rate_rush', games_played_col]])

    # Apply shrinkage
    current_season = get_current_season()
    shrinkage = ShrinkagePredictor(season=current_season)
    df_predictions = apply_early_season_shrinkage(
        df=df_predictions,
        current_week=9,
        position_col='position',
        season=current_season
    )

    print("\nAfter Shrinkage:")
    print(df_predictions[[
        'player', 'position',
        'yards_per_carry', 'yards_per_carry_shrunk',
        'td_rate_rush', 'td_rate_rush_shrunk',
        games_played_col
    ]])

    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("\nBackup QB/RB with inflated stats (small samples) get pulled toward")
    print("league average, while starters with large samples remain unchanged.")
    print("\nThis prevents overfitting to lucky performances in limited snaps.")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Run example
    example_integration()
