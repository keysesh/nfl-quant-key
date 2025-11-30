"""
Extract actual trailing statistics from play-by-play data for predictor inputs.

This solves the feature mismatch problem:
- Predictors were trained on actual per-game counts (Josh Allen: 31 attempts/game)
- We were passing shares × constants (0.97 × 70 = 67.9)
- This module extracts real trailing stats from PBP data
"""

import pandas as pd
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Feature flag for regime detection
# Set via environment variable: ENABLE_REGIME_DETECTION=1
ENABLE_REGIME_DETECTION = os.environ.get('ENABLE_REGIME_DETECTION', '0') == '1'


class TrailingStatsExtractor:
    """Extract trailing 4-week averages from play-by-play data."""

    def __init__(self, pbp_path: Path = None, season: Optional[int] = None):
        """
        Initialize with PBP data.

        Args:
            pbp_path: Path to play-by-play parquet file
            season: Season to load (defaults to current season)
        """
        if pbp_path is None:
            if season is None:
                from nfl_quant.utils.season_utils import get_current_season
                season = get_current_season()
            pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')

        logger.info(f"Loading PBP data from {pbp_path}")
        self.pbp_df = pd.read_parquet(pbp_path)
        logger.info(f"Loaded {len(self.pbp_df):,} plays")

        # Pre-compute player-week aggregations for fast lookup
        self._compute_player_week_stats()

    def _compute_player_week_stats(self):
        """Pre-compute all player-week statistics."""
        logger.info("Computing player-week statistics...")

        # QB passing stats
        qb_passing = (
            self.pbp_df[self.pbp_df['play_type'] == 'pass']
            .groupby(['passer_player_id', 'passer_player_name', 'week'])
            .size()
            .reset_index(name='pass_attempts')
        )

        # QB/RB rushing stats
        rushing = (
            self.pbp_df[self.pbp_df['play_type'] == 'run']
            .groupby(['rusher_player_id', 'rusher_player_name', 'week'])
            .size()
            .reset_index(name='rush_attempts')
        )

        # WR/TE/RB receiving stats (targets)
        receiving = (
            self.pbp_df[self.pbp_df['receiver_player_id'].notna()]
            .groupby(['receiver_player_id', 'receiver_player_name', 'week'])
            .size()
            .reset_index(name='targets')
        )

        # Store for lookup
        self.qb_passing_stats = qb_passing
        self.rushing_stats = rushing
        self.receiving_stats = receiving

        logger.info(f"  QB passing: {len(qb_passing)} player-weeks")
        logger.info(f"  Rushing: {len(rushing)} player-weeks")
        logger.info(f"  Receiving: {len(receiving)} player-weeks")

    def _calculate_weighted_average(
        self,
        values: pd.Series,
        use_ewma: bool = True,
        span: int = 4
    ) -> float:
        """
        Calculate weighted average of values.

        Args:
            values: Series of values to average
            use_ewma: Use exponential weighting
            span: Span for EWMA (default 4)

        Returns:
            Weighted average value
        """
        if len(values) == 0:
            return 0.0

        if use_ewma:
            # EWMA: Recent values weighted more heavily
            # With span=4: Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
            alpha = 2.0 / (span + 1)  # Smoothing factor
            weights = []
            for i in range(len(values)):
                weight = alpha * (1 - alpha) ** i
                weights.append(weight)

            # Reverse weights (most recent gets highest weight)
            weights = list(reversed(weights))

            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Calculate weighted average
            weighted_avg = sum(v * w for v, w in zip(values, weights))
            return float(weighted_avg)
        else:
            # Simple mean
            return float(values.mean())

    def get_trailing_stats(
        self,
        player_name: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4,
        use_ewma: bool = True
    ) -> Dict[str, float]:
        """
        Get trailing N-week averages for a player.

        Args:
            player_name: Player name (e.g., "J.Allen")
            position: Position (QB, RB, WR, TE)
            current_week: Current week number
            trailing_weeks: Number of weeks to average (default 4)
            use_ewma: Use exponential weighting (default True)
                     If True, recent weeks weighted higher:
                       Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
                     If False, all weeks weighted equally (25% each)

        Returns:
            Dict with trailing_snaps, trailing_attempts, trailing_carries
        """
        # Week range for trailing stats
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            # No trailing data available (Week 1)
            logger.warning(f"No trailing data for {player_name} in week {current_week}")
            return {
                'trailing_snaps': 0.0,
                'trailing_attempts': 0.0,
                'trailing_carries': 0.0,
            }

        # Position-specific stats
        if position == 'QB':
            # Get passing attempts
            qb_data = self.qb_passing_stats[
                (self.qb_passing_stats['passer_player_name'] == player_name) &
                (self.qb_passing_stats['week'] >= start_week) &
                (self.qb_passing_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            # Get rushing attempts
            rush_data = self.rushing_stats[
                (self.rushing_stats['rusher_player_name'] == player_name) &
                (self.rushing_stats['week'] >= start_week) &
                (self.rushing_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            trailing_attempts = self._calculate_weighted_average(
                qb_data['pass_attempts'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(qb_data) > 0 else 0.0

            trailing_carries = self._calculate_weighted_average(
                rush_data['rush_attempts'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rush_data) > 0 else 0.0

            trailing_snaps = trailing_attempts + trailing_carries

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),
                'trailing_carries': float(trailing_carries),
            }

        elif position == 'RB':
            # Get rushing attempts
            rush_data = self.rushing_stats[
                (self.rushing_stats['rusher_player_name'] == player_name) &
                (self.rushing_stats['week'] >= start_week) &
                (self.rushing_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            # Get targets
            rec_data = self.receiving_stats[
                (self.receiving_stats['receiver_player_name'] == player_name) &
                (self.receiving_stats['week'] >= start_week) &
                (self.receiving_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            trailing_carries = self._calculate_weighted_average(
                rush_data['rush_attempts'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rush_data) > 0 else 0.0

            trailing_attempts = self._calculate_weighted_average(
                rec_data['targets'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rec_data) > 0 else 0.0  # attempts = targets for RB

            trailing_snaps = trailing_carries + trailing_attempts

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),  # targets
                'trailing_carries': float(trailing_carries),
            }

        elif position in ['WR', 'TE']:
            # Get targets
            rec_data = self.receiving_stats[
                (self.receiving_stats['receiver_player_name'] == player_name) &
                (self.receiving_stats['week'] >= start_week) &
                (self.receiving_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            trailing_attempts = self._calculate_weighted_average(
                rec_data['targets'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rec_data) > 0 else 0.0  # attempts = targets for WR/TE

            trailing_snaps = trailing_attempts  # Approximation

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),  # targets
                'trailing_carries': 0.0,
            }

        else:
            raise ValueError(f"Unsupported position: {position}")

    def get_regime_features(
        self,
        player_name: str,
        team: str,
        current_week: int,
        position: str
    ) -> Dict[str, float]:
        """
        Extract regime-related features for ML models.

        Args:
            player_name: Player name
            team: Team abbreviation
            current_week: Current week number
            position: Player position

        Returns:
            Dict with regime features:
            - weeks_since_regime_change: Weeks since last QB/coaching change
            - is_in_regime: 1.0 if in an active regime, 0.0 otherwise
            - regime_confidence: Confidence score (0-1) if in regime
        """
        if not ENABLE_REGIME_DETECTION:
            # Regime detection disabled - return neutral features
            return {
                'weeks_since_regime_change': 999.0,  # No regime
                'is_in_regime': 0.0,
                'regime_confidence': 0.0,
            }

        try:
            # Use regime-aware extractor to detect regime
            from ..regime.integration import get_regime_aware_extractor
            regime_extractor = get_regime_aware_extractor(enable_regime=True)

            # Get regime info for player's team
            regime_info = regime_extractor._detect_team_regime(team, current_week)

            if regime_info is None:
                # No active regime
                return {
                    'weeks_since_regime_change': 999.0,  # No regime
                    'is_in_regime': 0.0,
                    'regime_confidence': 0.0,
                }

            # Active regime detected
            weeks_since_change = current_week - regime_info['start_week']

            return {
                'weeks_since_regime_change': float(weeks_since_change),
                'is_in_regime': 1.0,
                'regime_confidence': float(regime_info['confidence']),
            }

        except Exception as e:
            logger.warning(f"Error extracting regime features for {player_name}: {e}")
            # Return neutral features on error
            return {
                'weeks_since_regime_change': 999.0,
                'is_in_regime': 0.0,
                'regime_confidence': 0.0,
            }

    def get_game_script_features(
        self,
        player_name: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Extract game script-related features from historical data.

        Args:
            player_name: Player name
            position: Player position
            current_week: Current week number
            trailing_weeks: Number of weeks to look back

        Returns:
            Dict with game script features:
            - usage_when_leading: Avg usage when team leading by 7+
            - usage_when_trailing: Avg usage when team trailing by 7+
            - usage_when_close: Avg usage when within 1 score
            - game_script_sensitivity: Std dev of usage across game scripts
        """
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            return {
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            }

        # Filter plays for this player in the trailing window
        # This requires score differential data from PBP
        player_plays = self.pbp_df[
            (self.pbp_df['week'] >= start_week) &
            (self.pbp_df['week'] <= end_week) &
            (
                (self.pbp_df['passer_player_name'] == player_name) |
                (self.pbp_df['rusher_player_name'] == player_name) |
                (self.pbp_df['receiver_player_name'] == player_name)
            )
        ].copy()

        if len(player_plays) == 0:
            return {
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            }

        # Calculate score differential (positive = team leading)
        # score_differential may already exist in PBP data
        if 'score_differential' in player_plays.columns:
            player_plays['score_diff'] = player_plays['score_differential']
        elif 'posteam_score' in player_plays.columns and 'defteam_score' in player_plays.columns:
            player_plays['score_diff'] = player_plays['posteam_score'] - player_plays['defteam_score']
        else:
            # Can't calculate game script features without score data
            logger.debug(f"No score data available for game script features for {player_name}")
            return {
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            }

        # Group plays by game script
        leading_plays = player_plays[player_plays['score_diff'] >= 7]
        trailing_plays = player_plays[player_plays['score_diff'] <= -7]
        close_plays = player_plays[player_plays['score_diff'].abs() < 7]

        # Count plays per game script per week
        def count_plays_per_week(plays_df):
            if len(plays_df) == 0:
                return 0.0
            # Group by week and count, then average
            weekly_counts = plays_df.groupby('week').size()
            return float(weekly_counts.mean())

        usage_leading = count_plays_per_week(leading_plays)
        usage_trailing = count_plays_per_week(trailing_plays)
        usage_close = count_plays_per_week(close_plays)

        # Calculate sensitivity (variance in usage)
        usage_values = [usage_leading, usage_trailing, usage_close]
        usage_values = [v for v in usage_values if v > 0]
        if len(usage_values) >= 2:
            sensitivity = float(pd.Series(usage_values).std())
        else:
            sensitivity = 0.0

        return {
            'usage_when_leading': usage_leading,
            'usage_when_trailing': usage_trailing,
            'usage_when_close': usage_close,
            'game_script_sensitivity': sensitivity,
        }


# Singleton instance for reuse
_EXTRACTOR = None


def get_trailing_stats_extractor() -> TrailingStatsExtractor:
    """
    Get or create trailing stats extractor singleton.

    If ENABLE_REGIME_DETECTION=1, returns regime-aware extractor instead.
    """
    global _EXTRACTOR

    if ENABLE_REGIME_DETECTION:
        # Use regime-aware extractor (drop-in replacement)
        try:
            from ..regime.integration import get_regime_aware_extractor
            logger.info("✓ Using regime-aware trailing stats extractor")
            return get_regime_aware_extractor(enable_regime=True)
        except ImportError as e:
            logger.warning(f"Regime detection not available: {e}")
            logger.warning("Falling back to standard 4-week extractor")
            # Fall through to standard extractor

    # Standard extractor
    if _EXTRACTOR is None:
        _EXTRACTOR = TrailingStatsExtractor()
        if ENABLE_REGIME_DETECTION:
            logger.info("Using standard 4-week trailing stats extractor (fallback)")
        else:
            logger.info("Using standard 4-week trailing stats extractor")
    return _EXTRACTOR
