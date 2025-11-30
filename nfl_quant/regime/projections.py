"""
Regime-Aware Projection System

Integrates regime detection with player projections using dynamic windowing
and weighted blending strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .schemas import (
    Regime,
    RegimeWindow,
    RegimeMetrics,
    ProjectionAdjustment,
    RegimeType,
)
from .detector import RegimeDetector
from .metrics import RegimeMetricsCalculator


class RegimeAwareProjector:
    """
    Main projection system that incorporates regime detection.

    Features:
    - Dynamic window segmentation (uses only regime-relevant weeks)
    - Regime-weighted blending for small samples
    - Bayesian shrinkage toward positional baselines
    - Cross-season regime tracking
    """

    def __init__(
        self,
        detector: Optional[RegimeDetector] = None,
        calculator: Optional[RegimeMetricsCalculator] = None,
        default_trailing_weeks: int = 4,
        min_regime_sample: int = 2,
        blend_threshold_weeks: int = 4,
    ):
        """
        Initialize projector.

        Args:
            detector: RegimeDetector instance (creates new if None)
            calculator: RegimeMetricsCalculator instance (creates new if None)
            default_trailing_weeks: Default window if no regime detected
            min_regime_sample: Minimum weeks to use regime data
            blend_threshold_weeks: Weeks before using 100% regime data
        """
        self.detector = detector or RegimeDetector()
        self.calculator = calculator or RegimeMetricsCalculator()
        self.default_trailing_weeks = default_trailing_weeks
        self.min_regime_sample = min_regime_sample
        self.blend_threshold_weeks = blend_threshold_weeks

    def get_regime_adjusted_window(
        self,
        player_name: str,
        team: str,
        current_week: int,
        season: int,
        pbp_df: pd.DataFrame,
        player_stats_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[int], Dict]:
        """
        Get the optimal week window for calculating trailing stats.

        Logic:
        - If regime detected and sample >= min_regime_sample: Use regime weeks only
        - If regime detected but sample < min: Blend regime + pre-regime
        - If no regime detected: Use default trailing window

        Args:
            player_name: Player name
            team: Team abbreviation
            current_week: Current week number
            season: Season year
            pbp_df: Play-by-play data
            player_stats_df: Weekly player stats

        Returns:
            (weeks_to_use, metadata_dict)
        """
        # Detect regimes for team
        detection_result = self.detector.detect_all_regimes(
            team=team,
            current_week=current_week,
            season=season,
            pbp_df=pbp_df,
            player_stats_df=player_stats_df,
        )

        metadata = {
            "regime_detected": False,
            "regime_type": None,
            "regime_id": None,
            "regime_confidence": 1.0,
            "regime_start_week": None,
            "weeks_in_regime": 0,
            "blending_applied": False,
            "new_regime_weight": 1.0,
            "old_regime_weight": 0.0,
        }

        # Check if player affected by any regime
        if not detection_result.has_active_regime:
            # No regime - use default trailing window
            weeks_to_use = list(
                range(
                    max(1, current_week - self.default_trailing_weeks),
                    current_week,
                )
            )
            return weeks_to_use, metadata

        active_regime = detection_result.active_regime

        # Check if player is affected
        if player_name not in active_regime.affected_players:
            # Player not affected - use default
            weeks_to_use = list(
                range(
                    max(1, current_week - self.default_trailing_weeks),
                    current_week,
                )
            )
            return weeks_to_use, metadata

        # Player IS affected by regime
        regime_window = self.detector.get_regime_window(active_regime, current_week)

        metadata["regime_detected"] = True
        metadata["regime_type"] = active_regime.trigger.type.value
        metadata["regime_id"] = active_regime.regime_id
        metadata["regime_confidence"] = active_regime.trigger.confidence
        metadata["regime_start_week"] = active_regime.start_week
        metadata["weeks_in_regime"] = regime_window.num_weeks

        # Decision logic
        if regime_window.num_weeks >= self.blend_threshold_weeks:
            # Sufficient sample - use regime data only
            weeks_to_use = regime_window.weeks
            metadata["new_regime_weight"] = 1.0
            metadata["old_regime_weight"] = 0.0

        elif regime_window.num_weeks >= self.min_regime_sample:
            # Moderate sample - blend regime with pre-regime
            regime_weeks = regime_window.weeks

            # Pre-regime weeks
            pre_regime_start = max(
                1, active_regime.start_week - self.default_trailing_weeks
            )
            pre_regime_weeks = list(
                range(pre_regime_start, active_regime.start_week)
            )

            # Combine (regime weeks will be weighted more heavily)
            weeks_to_use = pre_regime_weeks + regime_weeks

            # Calculate weights
            new_weight, old_weight = self._calculate_blend_weights(
                regime_window.num_weeks
            )

            metadata["blending_applied"] = True
            metadata["new_regime_weight"] = new_weight
            metadata["old_regime_weight"] = old_weight

        else:
            # Insufficient regime sample - use default with warning
            weeks_to_use = list(
                range(
                    max(1, current_week - self.default_trailing_weeks),
                    current_week,
                )
            )
            metadata["insufficient_sample_warning"] = True

        return weeks_to_use, metadata

    def _calculate_blend_weights(
        self, weeks_in_new_regime: int
    ) -> Tuple[float, float]:
        """
        Calculate blending weights between new and old regime.

        Strategy:
        - 2 weeks: 60% new, 40% old
        - 3 weeks: 80% new, 20% old
        - 4+ weeks: 100% new, 0% old

        Args:
            weeks_in_new_regime: Number of weeks in new regime

        Returns:
            (new_regime_weight, old_regime_weight)
        """
        if weeks_in_new_regime >= 4:
            return 1.0, 0.0
        elif weeks_in_new_regime == 3:
            return 0.8, 0.2
        elif weeks_in_new_regime == 2:
            return 0.6, 0.4
        else:
            return 0.5, 0.5

    def apply_regime_weighted_blend(
        self,
        new_regime_value: float,
        old_regime_value: float,
        new_weight: float,
        old_weight: float,
    ) -> float:
        """
        Apply weighted blend between regime values.

        Args:
            new_regime_value: Value from new regime period
            old_regime_value: Value from old regime period
            new_weight: Weight for new regime (0-1)
            old_weight: Weight for old regime (0-1)

        Returns:
            Blended value
        """
        return (new_regime_value * new_weight) + (old_regime_value * old_weight)

    def get_regime_specific_stats(
        self,
        player_name: str,
        player_id: str,
        position: str,
        team: str,
        current_week: int,
        season: int,
        pbp_df: pd.DataFrame,
        player_stats_df: pd.DataFrame,
    ) -> Dict:
        """
        Calculate regime-specific trailing stats for a player.

        Returns dictionary with regime-aware trailing averages:
        - snaps, targets, carries (usage)
        - yards per opportunity (efficiency)
        - regime metadata

        Args:
            player_name: Player name
            player_id: Player ID
            position: Position
            team: Team abbreviation
            current_week: Current week
            season: Season year
            pbp_df: Play-by-play data
            player_stats_df: Player weekly stats

        Returns:
            Dictionary with regime-adjusted stats
        """
        # Get regime-adjusted window
        weeks_to_use, metadata = self.get_regime_adjusted_window(
            player_name=player_name,
            team=team,
            current_week=current_week,
            season=season,
            pbp_df=pbp_df,
            player_stats_df=player_stats_df,
        )

        # Filter stats to those weeks
        regime_stats = player_stats_df[
            (player_stats_df["player_name"] == player_name)
            & (player_stats_df["week"].isin(weeks_to_use))
        ].copy()

        if len(regime_stats) == 0:
            # No data - return zeros
            return {
                "snaps_per_game": 0.0,
                "targets_per_game": 0.0,
                "carries_per_game": 0.0,
                "yards_per_target": 0.0,
                "yards_per_carry": 0.0,
                "catch_rate": 0.0,
                "games_played": 0,
                **metadata,
            }

        # Calculate trailing stats
        games_played = len(regime_stats)

        # Snaps
        snaps_per_game = (
            regime_stats["snaps"].sum() / games_played
            if "snaps" in regime_stats.columns
            else 0.0
        )

        # Targets (WR/TE/RB)
        targets_per_game = (
            regime_stats["targets"].sum() / games_played
            if "targets" in regime_stats.columns
            else 0.0
        )

        # Carries (RB/QB)
        carries_per_game = (
            regime_stats["rush_attempts"].sum() / games_played
            if "rush_attempts" in regime_stats.columns
            else 0.0
        )

        # Yards per target
        total_targets = regime_stats["targets"].sum() if "targets" in regime_stats.columns else 0
        total_rec_yards = regime_stats["rec_yards"].sum() if "rec_yards" in regime_stats.columns else 0
        yards_per_target = (
            total_rec_yards / total_targets if total_targets > 0 else 0.0
        )

        # Yards per carry
        total_carries = regime_stats["rush_attempts"].sum() if "rush_attempts" in regime_stats.columns else 0
        total_rush_yards = regime_stats["rush_yards"].sum() if "rush_yards" in regime_stats.columns else 0
        yards_per_carry = (
            total_rush_yards / total_carries if total_carries > 0 else 0.0
        )

        # Catch rate
        total_receptions = regime_stats["receptions"].sum() if "receptions" in regime_stats.columns else 0
        catch_rate = total_receptions / total_targets if total_targets > 0 else 0.0

        return {
            "snaps_per_game": snaps_per_game,
            "targets_per_game": targets_per_game,
            "carries_per_game": carries_per_game,
            "yards_per_target": yards_per_target,
            "yards_per_carry": yards_per_carry,
            "catch_rate": catch_rate,
            "games_played": games_played,
            **metadata,
        }

    def create_projection_adjustment(
        self,
        player_name: str,
        market: str,
        base_projection: float,
        regime_metrics: RegimeMetrics,
        previous_metrics: Optional[RegimeMetrics],
    ) -> ProjectionAdjustment:
        """
        Create a projection adjustment based on regime change.

        Args:
            player_name: Player name
            market: Market type
            base_projection: Baseline projection
            regime_metrics: Current regime metrics
            previous_metrics: Previous regime metrics (if exists)

        Returns:
            ProjectionAdjustment object
        """
        # Calculate regime multiplier
        regime_multiplier = 1.0

        if previous_metrics:
            # Determine which metric to use based on market
            if "reception" in market or "rec_" in market:
                # Use target share
                if (
                    regime_metrics.usage.target_share
                    and previous_metrics.usage.target_share
                    and previous_metrics.usage.target_share > 0
                ):
                    regime_multiplier = (
                        regime_metrics.usage.target_share
                        / previous_metrics.usage.target_share
                    )

            elif "rush" in market:
                # Use carry/touch share
                if (
                    regime_metrics.usage.touch_share
                    and previous_metrics.usage.touch_share
                    and previous_metrics.usage.touch_share > 0
                ):
                    regime_multiplier = (
                        regime_metrics.usage.touch_share
                        / previous_metrics.usage.touch_share
                    )

            elif "pass" in market:
                # Use snap share or pass rate
                if previous_metrics.usage.snap_share > 0:
                    regime_multiplier = (
                        regime_metrics.usage.snap_share
                        / previous_metrics.usage.snap_share
                    )

        adjusted_projection = base_projection * regime_multiplier

        # Calculate weights
        new_weight, old_weight = self._calculate_blend_weights(
            regime_metrics.weeks_in_regime
        )

        # Confidence
        confidence = regime_metrics.sample_quality
        confidence_map = {"excellent": 0.95, "good": 0.85, "fair": 0.70, "poor": 0.50}
        confidence_value = confidence_map.get(confidence, 0.70)

        adjustment_reason = f"Regime change detected ({regime_metrics.regime_id})"

        return ProjectionAdjustment(
            player_name=player_name,
            market=market,
            base_projection=base_projection,
            regime_multiplier=regime_multiplier,
            adjusted_projection=adjusted_projection,
            adjustment_reason=adjustment_reason,
            confidence=confidence_value,
            new_regime_weight=new_weight,
            old_regime_weight=old_weight,
        )

    def batch_process_players(
        self,
        players: List[Dict],
        current_week: int,
        season: int,
        pbp_df: pd.DataFrame,
        player_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process multiple players to get regime-adjusted stats.

        Args:
            players: List of player dicts with keys: player_name, player_id, position, team
            current_week: Current week
            season: Season year
            pbp_df: Play-by-play data
            player_stats_df: Player stats

        Returns:
            DataFrame with regime-adjusted stats for all players
        """
        results = []

        for player in players:
            try:
                stats = self.get_regime_specific_stats(
                    player_name=player["player_name"],
                    player_id=player.get("player_id", ""),
                    position=player["position"],
                    team=player["team"],
                    current_week=current_week,
                    season=season,
                    pbp_df=pbp_df,
                    player_stats_df=player_stats_df,
                )

                results.append(
                    {
                        "player_name": player["player_name"],
                        "player_id": player.get("player_id", ""),
                        "position": player["position"],
                        "team": player["team"],
                        **stats,
                    }
                )

            except Exception as e:
                print(f"Error processing {player['player_name']}: {e}")
                continue

        return pd.DataFrame(results)


class RegimeWindowManager:
    """
    Manages dynamic window segmentation for regime analysis.

    Handles:
    - Cross-season regime tracking
    - Multi-regime player handling (trades, etc.)
    - Injury-adjusted windows
    """

    def __init__(self):
        pass

    def get_cross_season_window(
        self,
        regime: Regime,
        current_week: int,
        current_season: int,
        previous_season_pbp: Optional[pd.DataFrame] = None,
    ) -> RegimeWindow:
        """
        Get regime window that may span seasons.

        Example: QB/OC started Week 15 last year, still active this year
        -> Include last year's weeks 15-18 + this year's weeks 1-current

        Args:
            regime: Regime object
            current_week: Current week in current season
            current_season: Current season
            previous_season_pbp: Previous season PBP data (optional)

        Returns:
            RegimeWindow potentially spanning multiple seasons
        """
        if regime.season == current_season:
            # Same season - simple case
            weeks = list(range(regime.start_week, current_week))
            return RegimeWindow(
                regime_id=regime.regime_id,
                start_week=regime.start_week,
                end_week=current_week,
                weeks=weeks,
                season=current_season,
            )

        elif regime.season == current_season - 1 and previous_season_pbp is not None:
            # Cross-season regime
            # Last season weeks
            last_season_weeks = list(range(regime.start_week, 19))  # Assume 18 reg season weeks

            # This season weeks
            this_season_weeks = list(range(1, current_week))

            # Combine
            all_weeks = last_season_weeks + this_season_weeks

            return RegimeWindow(
                regime_id=regime.regime_id,
                start_week=regime.start_week,
                end_week=current_week,
                weeks=all_weeks,
                season=regime.season,  # Original season
            )

        else:
            # Regime too old - just use current season
            weeks = list(range(1, current_week))
            return RegimeWindow(
                regime_id=regime.regime_id,
                start_week=1,
                end_week=current_week,
                weeks=weeks,
                season=current_season,
            )

    def get_injury_adjusted_window(
        self,
        regime: Regime,
        player_name: str,
        current_week: int,
        injury_data: pd.DataFrame,
    ) -> RegimeWindow:
        """
        Adjust regime window to exclude injury weeks and include ramp-up.

        Args:
            regime: Regime object
            player_name: Player name
            current_week: Current week
            injury_data: DataFrame with columns: player_name, week, injury_status

        Returns:
            Injury-adjusted RegimeWindow
        """
        base_window = self.detector.get_regime_window(regime, current_week)

        # Get player injuries in regime period
        player_injuries = injury_data[
            (injury_data["player_name"] == player_name)
            & (injury_data["week"].isin(base_window.weeks))
        ]

        if len(player_injuries) == 0:
            return base_window  # No injuries

        # Exclude weeks marked as "Out" or "Doubtful"
        injury_weeks = player_injuries[
            player_injuries["injury_status"].isin(["Out", "Doubtful"])
        ]["week"].tolist()

        adjusted_weeks = [w for w in base_window.weeks if w not in injury_weeks]

        return RegimeWindow(
            regime_id=regime.regime_id,
            start_week=base_window.start_week,
            end_week=base_window.end_week,
            weeks=adjusted_weeks,
            season=regime.season,
        )
