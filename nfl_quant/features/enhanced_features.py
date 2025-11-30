"""
Enhanced Feature Engineering for NFL Player Props

Implements:
1. Exponentially Weighted Moving Averages (EWMA) with multiple spans
2. Player-specific historical features
3. Opponent-adjusted metrics by position
4. Trend analysis (current vs season average)
5. Efficiency metrics with proper variance handling
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlayerFeatures:
    """Container for player features."""
    # Usage features (EWMA spans)
    targets_ewma_3: float = 0.0
    targets_ewma_5: float = 0.0
    targets_ewma_10: float = 0.0
    targets_season_avg: float = 0.0
    targets_trend: float = 0.0  # ewma_3 / season_avg

    carries_ewma_3: float = 0.0
    carries_ewma_5: float = 0.0
    carries_ewma_10: float = 0.0
    carries_season_avg: float = 0.0
    carries_trend: float = 0.0

    pass_attempts_ewma_3: float = 0.0
    pass_attempts_ewma_5: float = 0.0
    pass_attempts_ewma_10: float = 0.0
    pass_attempts_season_avg: float = 0.0
    pass_attempts_trend: float = 0.0

    # Efficiency features
    yards_per_target_ewma_3: float = 0.0
    yards_per_target_ewma_5: float = 0.0
    yards_per_target_season: float = 0.0

    yards_per_carry_ewma_3: float = 0.0
    yards_per_carry_ewma_5: float = 0.0
    yards_per_carry_season: float = 0.0

    yards_per_attempt_ewma_3: float = 0.0
    yards_per_attempt_ewma_5: float = 0.0
    yards_per_attempt_season: float = 0.0

    completion_rate_ewma_3: float = 0.0
    completion_rate_ewma_5: float = 0.0
    completion_rate_season: float = 0.0

    # TD rates
    receiving_td_rate_season: float = 0.0
    rushing_td_rate_season: float = 0.0
    passing_td_rate_season: float = 0.0

    # Volume metrics
    snap_share_ewma_3: float = 0.0
    target_share_ewma_3: float = 0.0
    carry_share_ewma_3: float = 0.0

    # Consistency (variance)
    targets_std: float = 0.0
    carries_std: float = 0.0
    yards_per_target_std: float = 0.0

    # Home/Away splits
    home_targets_avg: float = 0.0
    away_targets_avg: float = 0.0
    home_away_diff: float = 0.0

    # Games played
    games_played: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'targets_ewma_3': self.targets_ewma_3,
            'targets_ewma_5': self.targets_ewma_5,
            'targets_ewma_10': self.targets_ewma_10,
            'targets_season_avg': self.targets_season_avg,
            'targets_trend': self.targets_trend,
            'carries_ewma_3': self.carries_ewma_3,
            'carries_ewma_5': self.carries_ewma_5,
            'carries_ewma_10': self.carries_ewma_10,
            'carries_season_avg': self.carries_season_avg,
            'carries_trend': self.carries_trend,
            'pass_attempts_ewma_3': self.pass_attempts_ewma_3,
            'pass_attempts_ewma_5': self.pass_attempts_ewma_5,
            'pass_attempts_ewma_10': self.pass_attempts_ewma_10,
            'pass_attempts_season_avg': self.pass_attempts_season_avg,
            'pass_attempts_trend': self.pass_attempts_trend,
            'yards_per_target_ewma_3': self.yards_per_target_ewma_3,
            'yards_per_target_ewma_5': self.yards_per_target_ewma_5,
            'yards_per_target_season': self.yards_per_target_season,
            'yards_per_carry_ewma_3': self.yards_per_carry_ewma_3,
            'yards_per_carry_ewma_5': self.yards_per_carry_ewma_5,
            'yards_per_carry_season': self.yards_per_carry_season,
            'yards_per_attempt_ewma_3': self.yards_per_attempt_ewma_3,
            'yards_per_attempt_ewma_5': self.yards_per_attempt_ewma_5,
            'yards_per_attempt_season': self.yards_per_attempt_season,
            'completion_rate_ewma_3': self.completion_rate_ewma_3,
            'completion_rate_ewma_5': self.completion_rate_ewma_5,
            'completion_rate_season': self.completion_rate_season,
            'receiving_td_rate_season': self.receiving_td_rate_season,
            'rushing_td_rate_season': self.rushing_td_rate_season,
            'passing_td_rate_season': self.passing_td_rate_season,
            'snap_share_ewma_3': self.snap_share_ewma_3,
            'target_share_ewma_3': self.target_share_ewma_3,
            'carry_share_ewma_3': self.carry_share_ewma_3,
            'targets_std': self.targets_std,
            'carries_std': self.carries_std,
            'yards_per_target_std': self.yards_per_target_std,
            'home_targets_avg': self.home_targets_avg,
            'away_targets_avg': self.away_targets_avg,
            'home_away_diff': self.home_away_diff,
            'games_played': self.games_played,
        }


@dataclass
class OpponentFeatures:
    """Opponent defensive strength features."""
    # EPA allowed (lower = better defense)
    pass_epa_allowed: float = 0.0
    rush_epa_allowed: float = 0.0

    # Yards allowed per play
    pass_yards_per_attempt_allowed: float = 0.0
    rush_yards_per_carry_allowed: float = 0.0
    receiving_yards_per_target_allowed: float = 0.0

    # TD rates allowed
    pass_td_rate_allowed: float = 0.0
    rush_td_rate_allowed: float = 0.0
    receiving_td_rate_allowed: float = 0.0

    # Position-specific (WR/RB/TE)
    wr_yards_per_target_allowed: float = 0.0
    rb_yards_per_target_allowed: float = 0.0
    te_yards_per_target_allowed: float = 0.0

    # Pressure and coverage
    pressure_rate: float = 0.0
    sack_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'opp_pass_epa_allowed': self.pass_epa_allowed,
            'opp_rush_epa_allowed': self.rush_epa_allowed,
            'opp_pass_yards_per_attempt_allowed': self.pass_yards_per_attempt_allowed,
            'opp_rush_yards_per_carry_allowed': self.rush_yards_per_carry_allowed,
            'opp_receiving_yards_per_target_allowed': self.receiving_yards_per_target_allowed,
            'opp_pass_td_rate_allowed': self.pass_td_rate_allowed,
            'opp_rush_td_rate_allowed': self.rush_td_rate_allowed,
            'opp_receiving_td_rate_allowed': self.receiving_td_rate_allowed,
            'opp_wr_yards_per_target_allowed': self.wr_yards_per_target_allowed,
            'opp_rb_yards_per_target_allowed': self.rb_yards_per_target_allowed,
            'opp_te_yards_per_target_allowed': self.te_yards_per_target_allowed,
            'opp_pressure_rate': self.pressure_rate,
            'opp_sack_rate': self.sack_rate,
        }


@dataclass
class VegasFeatures:
    """Vegas line features."""
    game_total: float = 45.0  # Over/under
    team_total: float = 22.5  # Implied team score
    spread: float = 0.0  # Point spread (negative = favorite)
    spread_abs: float = 0.0  # Absolute spread
    is_favorite: bool = False
    is_home: bool = False

    # Derived
    implied_possessions: float = 10.0
    pace_adjustment: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'vegas_game_total': self.game_total,
            'vegas_team_total': self.team_total,
            'vegas_spread': self.spread,
            'vegas_spread_abs': self.spread_abs,
            'vegas_is_favorite': int(self.is_favorite),
            'vegas_is_home': int(self.is_home),
            'vegas_implied_possessions': self.implied_possessions,
            'vegas_pace_adjustment': self.pace_adjustment,
        }


class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor with EWMA, opponent adjustments, and Vegas features.
    """

    def __init__(self, stats_df: pd.DataFrame = None, pbp_df: pd.DataFrame = None):
        """
        Initialize with player stats and play-by-play data.

        Args:
            stats_df: Player weekly stats (from NFLverse)
            pbp_df: Play-by-play data (for detailed metrics)
        """
        self.stats_df = stats_df
        self.pbp_df = pbp_df

        # Pre-compute aggregations
        if stats_df is not None:
            self._precompute_team_stats()
            self._precompute_defensive_stats()

        logger.info("Enhanced feature extractor initialized")

    def _precompute_team_stats(self):
        """Compute team-level aggregations for share calculations."""
        if self.stats_df is None:
            return

        # Team targets per week
        self.team_targets = (
            self.stats_df.groupby(['team', 'season', 'week'])['targets']
            .sum()
            .reset_index(name='team_targets')
        )

        # Team carries per week
        self.team_carries = (
            self.stats_df.groupby(['team', 'season', 'week'])['carries']
            .sum()
            .reset_index(name='team_carries')
        )

        # Team pass attempts per week
        self.team_pass_attempts = (
            self.stats_df.groupby(['team', 'season', 'week'])['attempts']
            .sum()
            .reset_index(name='team_attempts')
        )

        logger.info(f"Pre-computed team stats: {len(self.team_targets)} team-weeks")

    def _precompute_defensive_stats(self):
        """Compute defensive strength metrics."""
        if self.stats_df is None:
            return

        # This will be populated from opponent stats
        # For each defense, calculate what they allow
        self.defense_stats = {}

        # Group by opponent_team to see what each defense allows
        for team in self.stats_df['team'].unique():
            # Games where this team was the opponent
            opp_games = self.stats_df[self.stats_df['opponent_team'] == team]

            if len(opp_games) == 0:
                continue

            self.defense_stats[team] = {
                'pass_yards_allowed_per_game': opp_games['passing_yards'].mean(),
                'rush_yards_allowed_per_game': opp_games['rushing_yards'].mean(),
                'receiving_yards_allowed_per_game': opp_games['receiving_yards'].mean(),
                'targets_allowed_per_game': opp_games['targets'].mean(),
                'carries_allowed_per_game': opp_games['carries'].mean(),
                'passing_tds_allowed': opp_games['passing_tds'].mean(),
                'rushing_tds_allowed': opp_games['rushing_tds'].mean(),
                'receiving_tds_allowed': opp_games['receiving_tds'].mean(),
            }

        logger.info(f"Pre-computed defensive stats for {len(self.defense_stats)} teams")

    def _calculate_ewma(self, series: pd.Series, span: int) -> float:
        """
        Calculate exponentially weighted moving average.

        Args:
            series: Time series of values (ordered by week)
            span: EWMA span parameter

        Returns:
            EWMA value (most recent weighted average)
        """
        if len(series) == 0:
            return 0.0

        if len(series) == 1:
            return float(series.iloc[0])

        # Calculate EWMA - the last value is the most recent weighted average
        ewma = series.ewm(span=span, adjust=True).mean()
        return float(ewma.iloc[-1])

    def get_player_features(
        self,
        player_name: str,
        player_id: str,
        team: str,
        position: str,
        season: int,
        week: int,
        is_home: bool = False
    ) -> PlayerFeatures:
        """
        Extract comprehensive player features up to (but not including) current week.

        Args:
            player_name: Player display name
            player_id: Player ID
            team: Player's team
            position: Player position (QB, RB, WR, TE)
            season: Current season
            week: Current week (features use weeks 1 to week-1)
            is_home: Whether this is a home game

        Returns:
            PlayerFeatures with EWMA and seasonal metrics
        """
        features = PlayerFeatures()

        if self.stats_df is None:
            logger.warning("No stats data available")
            return features

        # Get player's historical games (before current week)
        player_data = self.stats_df[
            (self.stats_df['player_id'] == player_id) &
            (self.stats_df['season'] == season) &
            (self.stats_df['week'] < week)
        ].sort_values('week')

        if len(player_data) == 0:
            logger.warning(f"No historical data for {player_name} before week {week}")
            return features

        features.games_played = len(player_data)

        # EWMA for targets (receiving)
        if 'targets' in player_data.columns:
            targets = player_data['targets'].fillna(0)
            features.targets_ewma_3 = self._calculate_ewma(targets, span=3)
            features.targets_ewma_5 = self._calculate_ewma(targets, span=5)
            features.targets_ewma_10 = self._calculate_ewma(targets, span=10)
            features.targets_season_avg = float(targets.mean())
            features.targets_std = float(targets.std()) if len(targets) > 1 else 0.0

            if features.targets_season_avg > 0:
                features.targets_trend = features.targets_ewma_3 / features.targets_season_avg
            else:
                features.targets_trend = 1.0

        # EWMA for carries (rushing)
        if 'carries' in player_data.columns:
            carries = player_data['carries'].fillna(0)
            features.carries_ewma_3 = self._calculate_ewma(carries, span=3)
            features.carries_ewma_5 = self._calculate_ewma(carries, span=5)
            features.carries_ewma_10 = self._calculate_ewma(carries, span=10)
            features.carries_season_avg = float(carries.mean())
            features.carries_std = float(carries.std()) if len(carries) > 1 else 0.0

            if features.carries_season_avg > 0:
                features.carries_trend = features.carries_ewma_3 / features.carries_season_avg
            else:
                features.carries_trend = 1.0

        # EWMA for pass attempts (QB)
        if 'attempts' in player_data.columns and position == 'QB':
            attempts = player_data['attempts'].fillna(0)
            features.pass_attempts_ewma_3 = self._calculate_ewma(attempts, span=3)
            features.pass_attempts_ewma_5 = self._calculate_ewma(attempts, span=5)
            features.pass_attempts_ewma_10 = self._calculate_ewma(attempts, span=10)
            features.pass_attempts_season_avg = float(attempts.mean())

            if features.pass_attempts_season_avg > 0:
                features.pass_attempts_trend = features.pass_attempts_ewma_3 / features.pass_attempts_season_avg
            else:
                features.pass_attempts_trend = 1.0

        # Efficiency metrics: yards per target
        if 'receiving_yards' in player_data.columns and 'targets' in player_data.columns:
            # Calculate per-game yards per target
            mask = player_data['targets'] > 0
            if mask.sum() > 0:
                ypt = player_data.loc[mask, 'receiving_yards'] / player_data.loc[mask, 'targets']
                features.yards_per_target_ewma_3 = self._calculate_ewma(ypt, span=3)
                features.yards_per_target_ewma_5 = self._calculate_ewma(ypt, span=5)
                features.yards_per_target_season = float(ypt.mean())
                features.yards_per_target_std = float(ypt.std()) if len(ypt) > 1 else 0.0

        # Efficiency metrics: yards per carry
        if 'rushing_yards' in player_data.columns and 'carries' in player_data.columns:
            mask = player_data['carries'] > 0
            if mask.sum() > 0:
                ypc = player_data.loc[mask, 'rushing_yards'] / player_data.loc[mask, 'carries']
                features.yards_per_carry_ewma_3 = self._calculate_ewma(ypc, span=3)
                features.yards_per_carry_ewma_5 = self._calculate_ewma(ypc, span=5)
                features.yards_per_carry_season = float(ypc.mean())

        # Efficiency metrics: yards per attempt (QB)
        if 'passing_yards' in player_data.columns and 'attempts' in player_data.columns:
            mask = player_data['attempts'] > 0
            if mask.sum() > 0:
                ypa = player_data.loc[mask, 'passing_yards'] / player_data.loc[mask, 'attempts']
                features.yards_per_attempt_ewma_3 = self._calculate_ewma(ypa, span=3)
                features.yards_per_attempt_ewma_5 = self._calculate_ewma(ypa, span=5)
                features.yards_per_attempt_season = float(ypa.mean())

        # Completion rate (QB)
        if 'completions' in player_data.columns and 'attempts' in player_data.columns:
            mask = player_data['attempts'] > 0
            if mask.sum() > 0:
                comp_rate = player_data.loc[mask, 'completions'] / player_data.loc[mask, 'attempts']
                features.completion_rate_ewma_3 = self._calculate_ewma(comp_rate, span=3)
                features.completion_rate_ewma_5 = self._calculate_ewma(comp_rate, span=5)
                features.completion_rate_season = float(comp_rate.mean())

        # TD rates
        if 'receiving_tds' in player_data.columns and 'targets' in player_data.columns:
            total_rec_tds = player_data['receiving_tds'].sum()
            total_targets = player_data['targets'].sum()
            if total_targets > 0:
                features.receiving_td_rate_season = total_rec_tds / total_targets

        if 'rushing_tds' in player_data.columns and 'carries' in player_data.columns:
            total_rush_tds = player_data['rushing_tds'].sum()
            total_carries = player_data['carries'].sum()
            if total_carries > 0:
                features.rushing_td_rate_season = total_rush_tds / total_carries

        if 'passing_tds' in player_data.columns and 'attempts' in player_data.columns:
            total_pass_tds = player_data['passing_tds'].sum()
            total_attempts = player_data['attempts'].sum()
            if total_attempts > 0:
                features.passing_td_rate_season = total_pass_tds / total_attempts

        # Share metrics (need team totals)
        if hasattr(self, 'team_targets') and 'targets' in player_data.columns:
            # Merge with team totals
            merged = player_data.merge(
                self.team_targets,
                on=['team', 'season', 'week'],
                how='left'
            )
            if 'team_targets' in merged.columns:
                mask = merged['team_targets'] > 0
                if mask.sum() > 0:
                    target_share = merged.loc[mask, 'targets'] / merged.loc[mask, 'team_targets']
                    features.target_share_ewma_3 = self._calculate_ewma(target_share, span=3)

        if hasattr(self, 'team_carries') and 'carries' in player_data.columns:
            merged = player_data.merge(
                self.team_carries,
                on=['team', 'season', 'week'],
                how='left'
            )
            if 'team_carries' in merged.columns:
                mask = merged['team_carries'] > 0
                if mask.sum() > 0:
                    carry_share = merged.loc[mask, 'carries'] / merged.loc[mask, 'team_carries']
                    features.carry_share_ewma_3 = self._calculate_ewma(carry_share, span=3)

        # Home/Away splits (need game location info)
        # This would require additional data about game location
        # For now, use placeholder
        features.home_targets_avg = features.targets_season_avg
        features.away_targets_avg = features.targets_season_avg
        features.home_away_diff = 0.0

        return features

    def get_opponent_features(
        self,
        opponent_team: str,
        position: str,
        season: int,
        week: int
    ) -> OpponentFeatures:
        """
        Get opponent defensive strength features.

        Args:
            opponent_team: Opponent team abbreviation
            position: Player position to adjust for
            season: Current season
            week: Current week

        Returns:
            OpponentFeatures with defensive metrics
        """
        features = OpponentFeatures()

        if not hasattr(self, 'defense_stats') or opponent_team not in self.defense_stats:
            logger.warning(f"No defensive stats for {opponent_team}")
            return features

        defense = self.defense_stats[opponent_team]

        # Basic defensive metrics
        features.pass_yards_per_attempt_allowed = defense.get('pass_yards_allowed_per_game', 0) / 35  # Approx attempts
        features.rush_yards_per_carry_allowed = defense.get('rush_yards_allowed_per_game', 0) / 25  # Approx carries
        features.receiving_yards_per_target_allowed = defense.get('receiving_yards_allowed_per_game', 0) / defense.get('targets_allowed_per_game', 1)

        # TD rates
        features.pass_td_rate_allowed = defense.get('passing_tds_allowed', 0) / 35
        features.rush_td_rate_allowed = defense.get('rushing_tds_allowed', 0) / 25
        features.receiving_td_rate_allowed = defense.get('receiving_tds_allowed', 0) / defense.get('targets_allowed_per_game', 1)

        # Position-specific (placeholder - would need more detailed data)
        if position == 'WR':
            features.wr_yards_per_target_allowed = features.receiving_yards_per_target_allowed * 1.1  # WRs slightly more efficient
        elif position == 'RB':
            features.rb_yards_per_target_allowed = features.receiving_yards_per_target_allowed * 0.8  # RBs shorter routes
        elif position == 'TE':
            features.te_yards_per_target_allowed = features.receiving_yards_per_target_allowed * 1.0  # TEs middle ground

        return features

    def get_vegas_features(
        self,
        game_total: float = 45.0,
        spread: float = 0.0,
        is_home: bool = False
    ) -> VegasFeatures:
        """
        Calculate Vegas-derived features.

        Args:
            game_total: Over/under line
            spread: Point spread (negative for favorite)
            is_home: Whether team is home

        Returns:
            VegasFeatures with derived metrics
        """
        features = VegasFeatures()

        features.game_total = game_total
        features.spread = spread
        features.spread_abs = abs(spread)
        features.is_home = is_home
        features.is_favorite = spread < 0

        # Team total = (game_total + spread) / 2 for favorite
        # Team total = (game_total - spread) / 2 for underdog
        if spread < 0:  # Favorite
            features.team_total = (game_total - spread) / 2
        else:  # Underdog
            features.team_total = (game_total - spread) / 2

        # Implied possessions (average NFL game has ~12 possessions per team)
        # Higher scoring games = more possessions typically
        avg_possessions = 12.0
        avg_total = 45.0
        features.implied_possessions = avg_possessions * (game_total / avg_total)

        # Pace adjustment (higher totals = faster pace)
        features.pace_adjustment = game_total / avg_total

        return features

    def extract_all_features(
        self,
        player_name: str,
        player_id: str,
        team: str,
        opponent_team: str,
        position: str,
        season: int,
        week: int,
        is_home: bool = False,
        game_total: float = 45.0,
        spread: float = 0.0
    ) -> Dict:
        """
        Extract all features for a player-game combination.

        Returns:
            Dictionary with all feature values
        """
        # Player features
        player_feats = self.get_player_features(
            player_name, player_id, team, position, season, week, is_home
        )

        # Opponent features
        opp_feats = self.get_opponent_features(opponent_team, position, season, week)

        # Vegas features
        vegas_feats = self.get_vegas_features(game_total, spread, is_home)

        # Combine all
        all_features = {}
        all_features.update(player_feats.to_dict())
        all_features.update(opp_feats.to_dict())
        all_features.update(vegas_feats.to_dict())

        # Add interaction features
        all_features['targets_x_vegas_pace'] = player_feats.targets_ewma_3 * vegas_feats.pace_adjustment
        all_features['carries_x_vegas_pace'] = player_feats.carries_ewma_3 * vegas_feats.pace_adjustment
        all_features['trend_x_opp_adjustment'] = player_feats.targets_trend * (1 + opp_feats.receiving_yards_per_target_allowed / 10)

        return all_features


def load_enhanced_extractor(season: int = 2025) -> EnhancedFeatureExtractor:
    """
    Load enhanced feature extractor with pre-loaded data.

    Args:
        season: Season to load stats for

    Returns:
        EnhancedFeatureExtractor instance
    """
    # Load NFLverse stats
    stats_path = Path(f'data/nflverse/player_stats_{season}_2025.csv')
    if not stats_path.exists():
        stats_path = Path('data/nflverse/player_stats_2024_2025.csv')

    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        logger.info(f"Loaded {len(stats_df)} player stat records")
    else:
        logger.warning(f"Stats file not found: {stats_path}")
        stats_df = None

    # Load PBP if available
    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
    if pbp_path.exists():
        pbp_df = pd.read_parquet(pbp_path)
        logger.info(f"Loaded {len(pbp_df)} PBP plays")
    else:
        logger.warning(f"PBP file not found: {pbp_path}")
        pbp_df = None

    return EnhancedFeatureExtractor(stats_df=stats_df, pbp_df=pbp_df)
