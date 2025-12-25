"""
Player-level feature engineering for props prediction.

Extracts usage metrics (snaps, targets, carries) and efficiency metrics
(yards per opportunity) from play-by-play data.
"""

import logging
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class PlayerFeatureEngineer:
    """Generate player-level features for props prediction."""

    def __init__(self, pbp_df: pd.DataFrame):
        """
        Initialize with play-by-play data.

        Args:
            pbp_df: nflfastR play-by-play DataFrame
        """
        self.pbp = pbp_df
        self.player_features = None

    def generate_features(self, week: int) -> pd.DataFrame:
        """
        Generate player features for upcoming week using historical data.

        Args:
            week: Week to generate features for (uses weeks 1 to week-1 as history)

        Returns:
            DataFrame with player features
        """
        logger.info(f"Generating player features for week {week}...")

        # Use only historical data (weeks before target week)
        historical_pbp = self.pbp[self.pbp['week'] < week].copy()

        if len(historical_pbp) == 0:
            raise ValueError(f"No historical data available for week {week}")

        # Generate features for different position groups
        qb_features = self._generate_qb_features(historical_pbp)
        rb_features = self._generate_rb_features(historical_pbp)
        wr_te_features = self._generate_wr_te_features(historical_pbp)

        # Combine all features
        all_features = pd.concat([qb_features, rb_features, wr_te_features], ignore_index=True)

        logger.info(f"Generated features for {len(all_features)} players")

        self.player_features = all_features
        return all_features

    def _generate_qb_features(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Generate QB-specific features."""
        # Filter to passing plays
        pass_plays = pbp[pbp['play_type'] == 'pass'].copy()

        # Group by QB (passer_player_id)
        qb_stats = pass_plays.groupby('passer_player_id').agg({
            'complete_pass': ['sum', 'count', 'mean'],  # Completions, attempts, completion %
            'passing_yards': ['sum', 'mean'],            # Total yards, yards per attempt
            'pass_touchdown': 'sum',                      # Passing TDs
            'interception': 'sum',                        # INTs
            'epa': ['mean', 'std'],                       # EPA per play, variance
            'air_yards': ['sum', 'mean'],                 # Air yards
            'yards_after_catch': 'sum',                   # YAC
            'qb_hit': 'sum',                              # Times hit
            'sack': 'sum',                                # Sacks taken
        }).reset_index()

        # Flatten column names
        qb_stats.columns = ['_'.join(col).strip('_') for col in qb_stats.columns.values]
        qb_stats = qb_stats.rename(columns={'passer_player_id': 'player_id'})

        # Add position
        qb_stats['position'] = 'QB'

        # Calculate derived features
        qb_stats['attempts'] = qb_stats['complete_pass_count']
        qb_stats['completions'] = qb_stats['complete_pass_sum']
        qb_stats['comp_pct'] = qb_stats['complete_pass_mean']
        qb_stats['pass_yards_per_attempt'] = qb_stats['passing_yards_mean']
        qb_stats['pass_tds'] = qb_stats['pass_touchdown_sum']
        qb_stats['interceptions'] = qb_stats['interception_sum']
        qb_stats['epa_per_play'] = qb_stats['epa_mean']

        return qb_stats

    def _generate_rb_features(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Generate RB-specific features."""
        # Rushing plays
        rush_plays = pbp[pbp['play_type'] == 'run'].copy()

        rush_stats = rush_plays.groupby('rusher_player_id').agg({
            'rushing_yards': ['sum', 'mean', 'count'],  # Total, YPC, carries
            'rush_touchdown': 'sum',                     # Rush TDs
            'epa': ['mean', 'std'],                      # EPA per carry
        }).reset_index()

        rush_stats.columns = ['_'.join(col).strip('_') for col in rush_stats.columns.values]
        rush_stats = rush_stats.rename(columns={'rusher_player_id': 'player_id'})

        # Receiving (RBs also catch passes)
        pass_plays = pbp[pbp['play_type'] == 'pass'].copy()

        rec_stats = pass_plays.groupby('receiver_player_id').agg({
            'complete_pass': 'sum',                      # Receptions
            'receiving_yards': ['sum', 'mean'],          # Rec yards, yards per reception
            'pass_touchdown': 'sum',                      # Rec TDs
            'air_yards': 'mean',                         # Average depth of target
        }).reset_index()

        rec_stats.columns = ['_'.join(col).strip('_') for col in rec_stats.columns.values]
        rec_stats = rec_stats.rename(columns={'receiver_player_id': 'player_id'})

        # Merge rushing and receiving
        rb_stats = rush_stats.merge(rec_stats, on='player_id', how='outer', suffixes=('_rush', '_rec'))

        rb_stats['position'] = 'RB'

        # Derived features
        rb_stats['carries'] = rb_stats['rushing_yards_count']
        rb_stats['rush_yards'] = rb_stats['rushing_yards_sum']
        rb_stats['yards_per_carry'] = rb_stats['rushing_yards_mean']
        rb_stats['rush_tds'] = rb_stats['rush_touchdown_sum']
        rb_stats['receptions'] = rb_stats['complete_pass_sum']
        rb_stats['rec_yards'] = rb_stats['receiving_yards_sum']
        rb_stats['rec_tds'] = rb_stats['pass_touchdown_sum']

        return rb_stats

    def _generate_wr_te_features(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Generate WR/TE-specific features."""
        pass_plays = pbp[pbp['play_type'] == 'pass'].copy()

        rec_stats = pass_plays.groupby('receiver_player_id').agg({
            'complete_pass': ['sum', 'count'],           # Receptions, targets
            'receiving_yards': ['sum', 'mean'],          # Rec yards, yards per reception
            'pass_touchdown': 'sum',                      # Rec TDs
            'air_yards': ['sum', 'mean'],                # Total air yards, aDOT
            'yards_after_catch': ['sum', 'mean'],        # YAC
            'epa': ['mean', 'std'],                      # EPA per target
        }).reset_index()

        rec_stats.columns = ['_'.join(col).strip('_') for col in rec_stats.columns.values]
        rec_stats = rec_stats.rename(columns={'receiver_player_id': 'player_id'})

        # Position will be determined from roster data
        rec_stats['position'] = 'WR'  # Default, will be updated with actual roster data

        # Derived features
        rec_stats['targets'] = rec_stats['complete_pass_count']
        rec_stats['receptions'] = rec_stats['complete_pass_sum']
        rec_stats['catch_rate'] = rec_stats['receptions'] / rec_stats['targets']
        rec_stats['rec_yards'] = rec_stats['receiving_yards_sum']
        rec_stats['yards_per_reception'] = rec_stats['receiving_yards_mean']
        rec_stats['rec_tds'] = rec_stats['pass_touchdown_sum']
        rec_stats['adot'] = rec_stats['air_yards_mean']  # Average depth of target
        rec_stats['yac_per_rec'] = rec_stats['yards_after_catch_mean']

        return rec_stats

    def apply_regularization(self, features_df: pd.DataFrame, stat_col: str, alpha: float = 1.5) -> pd.Series:
        """
        Apply Ridge regression shrinkage to player stats (same as team EPA).

        Shrinks individual player stats toward position mean to reduce noise.

        Args:
            features_df: Player features DataFrame
            stat_col: Column name to regularize
            alpha: Ridge regularization strength

        Returns:
            Regularized stat values
        """
        # Get valid values (drop NaN)
        valid_data = features_df[[stat_col, 'position']].dropna()

        if len(valid_data) == 0:
            return features_df[stat_col]

        # Calculate position means (priors)
        position_means = valid_data.groupby('position')[stat_col].mean()

        # Shrink toward position mean using Ridge
        ridge = Ridge(alpha=alpha)

        # Feature: current stat value
        X = valid_data[[stat_col]].values
        # Target: keep original (Ridge will shrink automatically)
        y = valid_data[stat_col].values

        ridge.fit(X, y)

        # Apply to all players
        regularized = features_df[stat_col].copy()
        for idx, row in features_df.iterrows():
            if pd.notna(row[stat_col]):
                original_val = row[stat_col]
                position_mean = position_means.get(row['position'], 0)
                # Shrink toward mean
                regularized.iloc[idx] = ridge.coef_[0] * original_val + ridge.intercept_

        return regularized

    def save_features(self, filepath: str):
        """Save generated features to CSV."""
        if self.player_features is None:
            raise ValueError("No features generated yet. Call generate_features() first.")

        self.player_features.to_csv(filepath, index=False)
        logger.info(f"Saved player features to {filepath}")
