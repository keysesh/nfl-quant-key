"""
Regime Detection Integration Module

Provides drop-in replacements for standard trailing stats calculation
that automatically detect and adjust for regime changes.

This module enables non-invasive integration with the existing pipeline:
- Same interface as TrailingStatsExtractor
- Automatic regime detection
- Graceful fallback to standard 4-week windows
- Feature flag for easy enable/disable
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

from .detector import RegimeDetector
from .projections import RegimeAwareProjector
from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class RegimeAwareTrailingStats:
    """
    Drop-in replacement for TrailingStatsExtractor that uses regime detection.

    Usage:
        # Instead of:
        extractor = TrailingStatsExtractor()
        stats = extractor.get_trailing_stats(player, position, week)

        # Use:
        extractor = RegimeAwareTrailingStats()
        stats = extractor.get_trailing_stats(player, position, week)
    """

    def __init__(
        self,
        pbp_path: Path = None,
        season: int = 2025,
        enable_regime_detection: bool = True,
        fallback_to_standard: bool = True,
        min_regime_confidence: float = 0.7,
    ):
        """
        Initialize regime-aware trailing stats extractor.

        Args:
            pbp_path: Path to play-by-play parquet file
            season: Current season year
            enable_regime_detection: If False, uses standard 4-week windows
            fallback_to_standard: If True, falls back to 4-week on errors
            min_regime_confidence: Minimum confidence to use regime (0.7 = 70%)
        """
        self.season = season
        self.enable_regime_detection = enable_regime_detection
        self.fallback_to_standard = fallback_to_standard
        self.min_regime_confidence = min_regime_confidence

        # Load PBP data
        if pbp_path is None:
            pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')

        logger.info(f"Loading PBP data from {pbp_path}")
        self.pbp_df = pd.read_parquet(pbp_path)
        logger.info(f"Loaded {len(self.pbp_df):,} plays")

        # Initialize regime detection components
        if self.enable_regime_detection:
            logger.info("Initializing regime detection system...")
            self.detector = RegimeDetector()
            self.projector = RegimeAwareProjector(detector=self.detector)
            self.fetcher = DataFetcher()

            # Cache for regime detections (team -> detection result)
            self._regime_cache = {}
            logger.info("✓ Regime detection system initialized")
        else:
            logger.info("Regime detection disabled - using standard 4-week windows")
            self.detector = None
            self.projector = None
            self.fetcher = None

        # Pre-compute player-week stats (same as standard extractor)
        self._compute_player_week_stats()

    def _compute_player_week_stats(self):
        """Pre-compute all player-week statistics (same as standard extractor)."""
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

    def _get_player_team(self, player_name: str, week: int) -> Optional[str]:
        """Get player's team for a given week."""
        # Try to extract from PBP data
        player_plays = self.pbp_df[
            (self.pbp_df['week'] <= week) &
            (
                (self.pbp_df['passer_player_name'] == player_name) |
                (self.pbp_df['rusher_player_name'] == player_name) |
                (self.pbp_df['receiver_player_name'] == player_name)
            )
        ]

        if len(player_plays) > 0:
            # Get most recent team
            most_recent = player_plays.nlargest(1, 'week')
            return most_recent.iloc[0]['posteam']

        return None

    def _detect_team_regime(
        self,
        team: str,
        current_week: int,
    ) -> Optional[Dict]:
        """
        Detect active regime for a team.

        Returns:
            Dict with regime info or None if no regime detected
        """
        # Check cache
        cache_key = f"{team}_{current_week}"
        if cache_key in self._regime_cache:
            return self._regime_cache[cache_key]

        try:
            # Load player stats if needed
            player_stats_df = self._load_player_stats(current_week)

            # Detect regimes
            result = self.detector.detect_all_regimes(
                team=team,
                current_week=current_week,
                season=self.season,
                pbp_df=self.pbp_df,
                player_stats_df=player_stats_df,
            )

            # Check if there's an active regime with sufficient confidence
            if result.has_active_regime:
                regime = result.active_regime
                if regime.trigger.confidence >= self.min_regime_confidence:
                    regime_info = {
                        'detected': True,
                        'type': regime.trigger.type.value,
                        'description': regime.trigger.description,
                        'start_week': regime.start_week,
                        'games_in_regime': regime.games_in_regime,
                        'confidence': regime.trigger.confidence,
                        'affected_players': [p.player_id for p in regime.affected_players],
                    }
                    self._regime_cache[cache_key] = regime_info
                    return regime_info

            # No regime detected
            self._regime_cache[cache_key] = None
            return None

        except Exception as e:
            logger.warning(f"Error detecting regime for {team}: {e}")
            if self.fallback_to_standard:
                logger.info(f"Falling back to standard window for {team}")
                return None
            raise

    def _load_player_stats(self, max_week: int) -> pd.DataFrame:
        """Load player stats up to specified week."""
        # Try nflverse cache first (legacy location)
        stats_file = Path(f'data/nflverse_cache/stats_player_week_{self.season}.csv')
        if stats_file.exists():
            df = pd.read_csv(stats_file)
            return df[df['week'] < max_week]

        # Try R-fetched data (new location)
        try:
            from nfl_quant.utils.nflverse_loader import load_player_stats
            df = load_player_stats(seasons=self.season)
            return df[df['week'] < max_week]
        except Exception as e:
            raise FileNotFoundError(
                f"No player stats found for season {self.season}. "
                f"Failed to load from NFLverse: {e}\n"
                f"Run: Rscript scripts/fetch/fetch_nflverse_data.R"
            )

    def _calculate_regime_aware_window(
        self,
        player_name: str,
        team: str,
        current_week: int,
    ) -> Tuple[int, int, Optional[Dict]]:
        """
        Calculate optimal trailing window based on regime detection.

        Returns:
            Tuple of (start_week, end_week, regime_info)
        """
        # Detect regime for player's team
        regime_info = self._detect_team_regime(team, current_week)

        if regime_info is None:
            # No regime - use standard 4-week window
            start_week = max(1, current_week - 4)
            end_week = current_week - 1
            return start_week, end_week, None

        # Regime detected - adjust window to regime start
        regime_start = regime_info['start_week']
        start_week = regime_start
        end_week = current_week - 1

        # Ensure minimum sample size (at least 2 games)
        if end_week - start_week + 1 < 2:
            logger.info(
                f"Regime window too small for {player_name} ({end_week - start_week + 1} games), "
                "falling back to standard 4-week window"
            )
            start_week = max(1, current_week - 4)
            return start_week, end_week, None

        logger.debug(
            f"{player_name}: Using regime window {start_week}-{end_week} "
            f"({regime_info['type']}, {regime_info['games_in_regime']} games)"
        )

        return start_week, end_week, regime_info

    def get_trailing_stats(
        self,
        player_name: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4,  # Kept for interface compatibility
    ) -> Dict[str, float]:
        """
        Get trailing averages for a player (regime-aware or standard).

        Args:
            player_name: Player name (e.g., "J.Allen")
            position: Position (QB, RB, WR, TE)
            current_week: Current week number
            trailing_weeks: Ignored if regime detection is enabled

        Returns:
            Dict with trailing_snaps, trailing_attempts, trailing_carries, and metadata
        """
        # Get player's team
        team = self._get_player_team(player_name, current_week)
        if team is None:
            logger.warning(f"Could not find team for {player_name}")
            # Fall back to standard calculation
            return self._get_standard_trailing_stats(
                player_name, position, current_week, trailing_weeks
            )

        # Calculate window (regime-aware or standard)
        if self.enable_regime_detection:
            try:
                start_week, end_week, regime_info = self._calculate_regime_aware_window(
                    player_name, team, current_week
                )
            except Exception as e:
                logger.warning(f"Error calculating regime window for {player_name}: {e}")
                if self.fallback_to_standard:
                    return self._get_standard_trailing_stats(
                        player_name, position, current_week, trailing_weeks
                    )
                raise
        else:
            # Standard 4-week window
            start_week = max(1, current_week - trailing_weeks)
            end_week = current_week - 1
            regime_info = None

        if end_week < 1:
            # No trailing data available (Week 1)
            logger.warning(f"No trailing data for {player_name} in week {current_week}")
            return {
                'trailing_snaps': 0.0,
                'trailing_attempts': 0.0,
                'trailing_carries': 0.0,
                'regime_detected': False,
                'window_weeks': 0,
            }

        # Extract stats using the calculated window
        stats = self._extract_stats_for_window(
            player_name, position, start_week, end_week
        )

        # Add regime metadata
        if regime_info:
            stats['regime_detected'] = True
            stats['regime_type'] = regime_info['type']
            stats['regime_confidence'] = regime_info['confidence']
            stats['regime_start_week'] = regime_info['start_week']
            stats['regime_games'] = regime_info['games_in_regime']
        else:
            stats['regime_detected'] = False

        stats['window_start'] = start_week
        stats['window_end'] = end_week
        stats['window_weeks'] = end_week - start_week + 1

        return stats

    def _get_standard_trailing_stats(
        self,
        player_name: str,
        position: str,
        current_week: int,
        trailing_weeks: int,
    ) -> Dict[str, float]:
        """Get standard trailing stats (fallback method)."""
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            return {
                'trailing_snaps': 0.0,
                'trailing_attempts': 0.0,
                'trailing_carries': 0.0,
                'regime_detected': False,
                'window_weeks': 0,
            }

        stats = self._extract_stats_for_window(
            player_name, position, start_week, end_week
        )
        stats['regime_detected'] = False
        stats['window_weeks'] = end_week - start_week + 1

        return stats

    def _extract_stats_for_window(
        self,
        player_name: str,
        position: str,
        start_week: int,
        end_week: int,
    ) -> Dict[str, float]:
        """
        Extract stats for a specific week window.

        (Same logic as original TrailingStatsExtractor)
        """
        # Position-specific stats
        if position == 'QB':
            # Get passing attempts
            qb_data = self.qb_passing_stats[
                (self.qb_passing_stats['passer_player_name'] == player_name) &
                (self.qb_passing_stats['week'] >= start_week) &
                (self.qb_passing_stats['week'] <= end_week)
            ]

            # Get rushing attempts
            rush_data = self.rushing_stats[
                (self.rushing_stats['rusher_player_name'] == player_name) &
                (self.rushing_stats['week'] >= start_week) &
                (self.rushing_stats['week'] <= end_week)
            ]

            trailing_attempts = qb_data['pass_attempts'].mean() if len(qb_data) > 0 else 0.0
            trailing_carries = rush_data['rush_attempts'].mean() if len(rush_data) > 0 else 0.0
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
            ]

            # Get targets
            rec_data = self.receiving_stats[
                (self.receiving_stats['receiver_player_name'] == player_name) &
                (self.receiving_stats['week'] >= start_week) &
                (self.receiving_stats['week'] <= end_week)
            ]

            trailing_carries = rush_data['rush_attempts'].mean() if len(rush_data) > 0 else 0.0
            trailing_attempts = rec_data['targets'].mean() if len(rec_data) > 0 else 0.0
            trailing_snaps = trailing_carries + trailing_attempts

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),
                'trailing_carries': float(trailing_carries),
            }

        elif position in ['WR', 'TE']:
            # Get targets
            rec_data = self.receiving_stats[
                (self.receiving_stats['receiver_player_name'] == player_name) &
                (self.receiving_stats['week'] >= start_week) &
                (self.receiving_stats['week'] <= end_week)
            ]

            trailing_attempts = rec_data['targets'].mean() if len(rec_data) > 0 else 0.0
            trailing_snaps = trailing_attempts

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),
                'trailing_carries': 0.0,
            }

        else:
            raise ValueError(f"Unsupported position: {position}")


# Singleton instance for reuse
_REGIME_EXTRACTOR = None


def get_regime_aware_extractor(enable_regime: bool = True) -> RegimeAwareTrailingStats:
    """
    Get or create regime-aware trailing stats extractor singleton.

    Args:
        enable_regime: If True, uses regime detection. If False, uses standard 4-week windows.

    Returns:
        RegimeAwareTrailingStats instance
    """
    global _REGIME_EXTRACTOR
    if _REGIME_EXTRACTOR is None or _REGIME_EXTRACTOR.enable_regime_detection != enable_regime:
        _REGIME_EXTRACTOR = RegimeAwareTrailingStats(enable_regime_detection=enable_regime)
    return _REGIME_EXTRACTOR
