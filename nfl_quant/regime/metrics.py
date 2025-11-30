"""
Regime-Specific Metric Calculation

Calculates usage, efficiency, and context metrics for players within specific regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .schemas import (
    Regime,
    RegimeWindow,
    RegimeMetrics,
    UsageMetrics,
    EfficiencyMetrics,
    ContextMetrics,
    RegimeComparison,
    RegimeImpact,
    RegimeType,
)


class RegimeMetricsCalculator:
    """
    Calculates regime-specific metrics for players and teams.

    Provides:
    - Usage metrics (snaps, targets, carries, routes)
    - Efficiency metrics (yards per opportunity, catch rate, etc.)
    - Context metrics (team pass rate, game script, opponent quality)
    - Comparison between regimes
    - Statistical significance testing
    """

    def __init__(self):
        pass

    def calculate_player_regime_metrics(
        self,
        player_name: str,
        player_id: str,
        position: str,
        team: str,
        regime: Regime,
        regime_window: RegimeWindow,
        pbp_df: pd.DataFrame,
        player_stats_df: pd.DataFrame,
    ) -> RegimeMetrics:
        """
        Calculate complete metrics for a player in a specific regime.

        Args:
            player_name: Player's name
            player_id: Unique player identifier
            position: Player position (QB, RB, WR, TE)
            team: Team abbreviation
            regime: Regime object
            regime_window: Time window for analysis
            pbp_df: Play-by-play DataFrame
            player_stats_df: Weekly player stats DataFrame

        Returns:
            RegimeMetrics with usage, efficiency, and context
        """
        # Filter data to regime window
        regime_pbp = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["week"].isin(regime_window.weeks))
        ].copy()

        regime_stats = player_stats_df[
            (player_stats_df["player_name"] == player_name)
            & (player_stats_df["week"].isin(regime_window.weeks))
        ].copy()

        # Calculate metrics based on position
        usage = self._calculate_usage_metrics(
            player_name, position, regime_pbp, regime_stats
        )

        efficiency = self._calculate_efficiency_metrics(
            player_name, position, regime_pbp, regime_stats
        )

        context = self._calculate_context_metrics(
            team, regime_window, regime_pbp, pbp_df
        )

        games_played = len(regime_stats["week"].unique())
        games_missed = regime_window.num_weeks - games_played

        return RegimeMetrics(
            player_name=player_name,
            player_id=player_id,
            position=position,
            team=team,
            regime_id=regime.regime_id,
            usage=usage,
            efficiency=efficiency,
            context=context,
            weeks_in_regime=regime_window.num_weeks,
            games_played=games_played,
            games_missed=games_missed,
        )

    def _calculate_usage_metrics(
        self,
        player_name: str,
        position: str,
        regime_pbp: pd.DataFrame,
        regime_stats: pd.DataFrame,
    ) -> UsageMetrics:
        """Calculate usage metrics from play-by-play and stats."""
        # Get player's plays
        player_plays = self._get_player_plays(player_name, regime_pbp)

        # Snaps
        snaps_per_game = len(player_plays) / max(len(regime_stats), 1)

        # Team offensive snaps
        team_snaps = len(regime_pbp)
        snap_share = len(player_plays) / team_snaps if team_snaps > 0 else 0.0

        # Position-specific usage
        if position in ["WR", "TE"]:
            # Targets
            targets = player_plays[player_plays["receiver_player_name"] == player_name]
            targets_per_game = len(targets) / max(len(regime_stats), 1)

            team_targets = regime_pbp[regime_pbp["play_type"] == "pass"]
            target_share = len(targets) / len(team_targets) if len(team_targets) > 0 else 0.0

            # Routes
            # Estimate routes as snaps on passing plays
            passing_plays = regime_pbp[regime_pbp["play_type"] == "pass"]
            routes_per_game = len(passing_plays) / max(len(regime_stats), 1)
            route_participation = routes_per_game / (len(passing_plays) / max(len(regime_stats), 1)) if len(passing_plays) > 0 else 0.0

            # Red zone targets
            rz_targets = targets[targets["yardline_100"] <= 20]
            team_rz_targets = team_targets[team_targets["yardline_100"] <= 20]
            redzone_target_share = (
                len(rz_targets) / len(team_rz_targets)
                if len(team_rz_targets) > 0
                else 0.0
            )

            return UsageMetrics(
                snap_share=snap_share,
                snaps_per_game=snaps_per_game,
                target_share=target_share,
                targets_per_game=targets_per_game,
                routes_per_game=routes_per_game,
                route_participation=route_participation,
                redzone_target_share=redzone_target_share,
            )

        elif position == "RB":
            # Carries
            carries = player_plays[player_plays["rusher_player_name"] == player_name]
            carries_per_game = len(carries) / max(len(regime_stats), 1)

            team_carries = regime_pbp[regime_pbp["play_type"] == "run"]
            carry_share = len(carries) / len(team_carries) if len(team_carries) > 0 else 0.0

            # Targets (receiving work)
            targets = player_plays[player_plays["receiver_player_name"] == player_name]
            targets_per_game = len(targets) / max(len(regime_stats), 1)

            team_targets = regime_pbp[regime_pbp["play_type"] == "pass"]
            target_share = len(targets) / len(team_targets) if len(team_targets) > 0 else 0.0

            # Touch share (carries + targets)
            touches = len(carries) + len(targets)
            team_touches = len(team_carries) + len(team_targets)
            touch_share = touches / team_touches if team_touches > 0 else 0.0

            # Red zone carries
            rz_carries = carries[carries["yardline_100"] <= 20]
            team_rz_carries = team_carries[team_carries["yardline_100"] <= 20]
            redzone_carry_share = (
                len(rz_carries) / len(team_rz_carries)
                if len(team_rz_carries) > 0
                else 0.0
            )

            # Goal line carries
            gl_carries = carries[carries["yardline_100"] <= 5]
            team_gl_carries = team_carries[team_carries["yardline_100"] <= 5]
            goalline_carry_share = (
                len(gl_carries) / len(team_gl_carries)
                if len(team_gl_carries) > 0
                else 0.0
            )

            return UsageMetrics(
                snap_share=snap_share,
                snaps_per_game=snaps_per_game,
                target_share=target_share,
                targets_per_game=targets_per_game,
                touch_share=touch_share,
                carries_per_game=carries_per_game,
                redzone_carry_share=redzone_carry_share,
                goalline_carry_share=goalline_carry_share,
            )

        elif position == "QB":
            # QB usage is mostly snaps
            return UsageMetrics(
                snap_share=snap_share,
                snaps_per_game=snaps_per_game,
            )

        else:
            # Default
            return UsageMetrics(
                snap_share=snap_share,
                snaps_per_game=snaps_per_game,
            )

    def _calculate_efficiency_metrics(
        self,
        player_name: str,
        position: str,
        regime_pbp: pd.DataFrame,
        regime_stats: pd.DataFrame,
    ) -> EfficiencyMetrics:
        """Calculate efficiency metrics."""
        player_plays = self._get_player_plays(player_name, regime_pbp)

        if position in ["WR", "TE"]:
            # Receiving efficiency
            receptions = player_plays[
                (player_plays["receiver_player_name"] == player_name)
                & (player_plays["complete_pass"] == 1)
            ]

            targets = player_plays[player_plays["receiver_player_name"] == player_name]

            catch_rate = len(receptions) / len(targets) if len(targets) > 0 else 0.0

            yards_per_target = (
                targets["yards_gained"].mean() if len(targets) > 0 else 0.0
            )

            yards_after_catch_per_reception = (
                receptions["yards_after_catch"].mean() if len(receptions) > 0 else 0.0
            )

            average_depth_of_target = (
                targets["air_yards"].mean() if len(targets) > 0 else 0.0
            )

            # Yards per route run (estimate)
            # Routes ≈ snaps on passing plays
            passing_plays = regime_pbp[regime_pbp["play_type"] == "pass"]
            routes = len(passing_plays) / max(len(regime_stats), 1)
            total_yards = targets["yards_gained"].sum()
            yards_per_route_run = (
                total_yards / (routes * len(regime_stats))
                if routes > 0 and len(regime_stats) > 0
                else 0.0
            )

            # Success rate
            success_plays = targets[targets["success"] == 1]
            success_rate = len(success_plays) / len(targets) if len(targets) > 0 else 0.0

            # EPA
            epa_per_play = targets["epa"].mean() if len(targets) > 0 else 0.0

            return EfficiencyMetrics(
                yards_per_route_run=yards_per_route_run,
                yards_per_target=yards_per_target,
                catch_rate=catch_rate,
                yards_after_catch_per_reception=yards_after_catch_per_reception,
                average_depth_of_target=average_depth_of_target,
                success_rate=success_rate,
                epa_per_play=epa_per_play,
            )

        elif position == "RB":
            # Rushing efficiency
            carries = player_plays[player_plays["rusher_player_name"] == player_name]
            yards_per_carry = (
                carries["yards_gained"].mean() if len(carries) > 0 else 0.0
            )

            # Receiving efficiency
            targets = player_plays[player_plays["receiver_player_name"] == player_name]
            yards_per_reception = (
                targets["yards_gained"].mean() if len(targets) > 0 else 0.0
            )

            # Success rate
            all_touches = pd.concat([carries, targets])
            success_plays = all_touches[all_touches["success"] == 1]
            success_rate = (
                len(success_plays) / len(all_touches) if len(all_touches) > 0 else 0.0
            )

            # EPA
            epa_per_play = all_touches["epa"].mean() if len(all_touches) > 0 else 0.0

            return EfficiencyMetrics(
                yards_per_carry=yards_per_carry,
                yards_per_reception=yards_per_reception,
                success_rate=success_rate,
                epa_per_play=epa_per_play,
            )

        elif position == "QB":
            # Passing efficiency
            passes = player_plays[player_plays["passer_player_name"] == player_name]

            completions = passes[passes["complete_pass"] == 1]
            completion_percentage = (
                len(completions) / len(passes) if len(passes) > 0 else 0.0
            )

            yards_per_attempt = passes["yards_gained"].mean() if len(passes) > 0 else 0.0

            yards_per_completion = (
                completions["yards_gained"].mean() if len(completions) > 0 else 0.0
            )

            # Rushing efficiency
            rushes = player_plays[player_plays["rusher_player_name"] == player_name]
            yards_per_carry = rushes["yards_gained"].mean() if len(rushes) > 0 else 0.0

            # Success rate
            success_plays = passes[passes["success"] == 1]
            success_rate = len(success_plays) / len(passes) if len(passes) > 0 else 0.0

            # EPA
            epa_per_play = passes["epa"].mean() if len(passes) > 0 else 0.0

            return EfficiencyMetrics(
                completion_percentage=completion_percentage,
                yards_per_attempt=yards_per_attempt,
                yards_per_completion=yards_per_completion,
                yards_per_carry=yards_per_carry,
                success_rate=success_rate,
                epa_per_play=epa_per_play,
            )

        else:
            return EfficiencyMetrics()

    def _calculate_context_metrics(
        self,
        team: str,
        regime_window: RegimeWindow,
        regime_pbp: pd.DataFrame,
        full_pbp_df: pd.DataFrame,
    ) -> ContextMetrics:
        """Calculate team context metrics during regime."""
        # Team pass rate
        team_pass_rate = (
            (regime_pbp["play_type"] == "pass").sum() / len(regime_pbp)
            if len(regime_pbp) > 0
            else 0.0
        )

        # Points per game
        games = regime_pbp.groupby("game_id")
        points_per_game = 0.0
        if len(games) > 0:
            game_points = []
            for game_id, game_df in games:
                # Get final score for this team
                team_score = game_df["total_home_score"].iloc[-1] if game_df["home_team"].iloc[0] == team else game_df["total_away_score"].iloc[-1]
                game_points.append(team_score)
            points_per_game = np.mean(game_points) if game_points else 0.0

        # Plays per game
        plays_per_game = len(regime_pbp) / regime_window.num_weeks if regime_window.num_weeks > 0 else 0.0

        # Average game script (point differential)
        game_scripts = []
        for game_id, game_df in games:
            is_home = game_df["home_team"].iloc[0] == team
            if is_home:
                diff = game_df["score_differential"].mean()
            else:
                diff = -game_df["score_differential"].mean()
            game_scripts.append(diff)

        average_game_script = np.mean(game_scripts) if game_scripts else 0.0

        # Home/away split
        home_games = regime_pbp[regime_pbp["home_team"] == team]["game_id"].nunique()
        away_games = regime_pbp[regime_pbp["away_team"] == team]["game_id"].nunique()

        # Opponent defensive quality
        opponents = []
        for game_id in regime_pbp["game_id"].unique():
            game_df = regime_pbp[regime_pbp["game_id"] == game_id]
            if game_df["home_team"].iloc[0] == team:
                opponent = game_df["away_team"].iloc[0]
            else:
                opponent = game_df["home_team"].iloc[0]
            opponents.append(opponent)

        # Calculate avg opponent defensive EPA
        opp_defensive_epas = []
        for opp in opponents:
            opp_def_plays = full_pbp_df[
                (full_pbp_df["defteam"] == opp)
                & (full_pbp_df["week"].isin(regime_window.weeks))
            ]
            if len(opp_def_plays) > 0:
                opp_defensive_epas.append(-opp_def_plays["epa"].mean())  # Negative EPA is good for defense

        avg_opponent_defensive_epa = (
            np.mean(opp_defensive_epas) if opp_defensive_epas else None
        )

        return ContextMetrics(
            team_pass_rate=team_pass_rate,
            team_points_per_game=points_per_game,
            team_plays_per_game=plays_per_game,
            average_game_script=average_game_script,
            home_games=home_games,
            away_games=away_games,
            avg_opponent_defensive_epa=avg_opponent_defensive_epa,
        )

    def compare_regimes(
        self,
        player_name: str,
        position: str,
        team: str,
        current_metrics: RegimeMetrics,
        previous_metrics: Optional[RegimeMetrics],
    ) -> RegimeComparison:
        """
        Compare current regime to previous regime.

        Args:
            player_name: Player name
            position: Player position
            team: Team abbreviation
            current_metrics: Metrics from current regime
            previous_metrics: Metrics from previous regime (None if first regime)

        Returns:
            RegimeComparison with deltas
        """
        if previous_metrics is None:
            return RegimeComparison(
                player_name=player_name,
                position=position,
                team=team,
                previous_regime_id=None,
                current_regime_id=current_metrics.regime_id,
            )

        # Calculate deltas
        target_share_delta = None
        snap_share_delta = None
        touch_share_delta = None
        yards_per_route_delta = None
        yards_per_carry_delta = None
        catch_rate_delta = None

        # Usage deltas
        if current_metrics.usage.target_share and previous_metrics.usage.target_share:
            target_share_delta = (
                current_metrics.usage.target_share - previous_metrics.usage.target_share
            )

        snap_share_delta = (
            current_metrics.usage.snap_share - previous_metrics.usage.snap_share
        )

        if current_metrics.usage.touch_share and previous_metrics.usage.touch_share:
            touch_share_delta = (
                current_metrics.usage.touch_share - previous_metrics.usage.touch_share
            )

        # Efficiency deltas
        if (
            current_metrics.efficiency.yards_per_route_run
            and previous_metrics.efficiency.yards_per_route_run
        ):
            yards_per_route_delta = (
                current_metrics.efficiency.yards_per_route_run
                - previous_metrics.efficiency.yards_per_route_run
            )

        if (
            current_metrics.efficiency.yards_per_carry
            and previous_metrics.efficiency.yards_per_carry
        ):
            yards_per_carry_delta = (
                current_metrics.efficiency.yards_per_carry
                - previous_metrics.efficiency.yards_per_carry
            )

        if (
            current_metrics.efficiency.catch_rate
            and previous_metrics.efficiency.catch_rate
        ):
            catch_rate_delta = (
                current_metrics.efficiency.catch_rate
                - previous_metrics.efficiency.catch_rate
            )

        # Statistical significance test (t-test on key metric)
        # Use target share for WR/TE, carry share for RB
        is_significant = False
        p_value = None

        # Note: Would need raw data arrays for proper t-test
        # For now, use heuristic: >10% change with 3+ games = significant
        if current_metrics.games_played >= 3 and previous_metrics.games_played >= 3:
            if position in ["WR", "TE"] and target_share_delta:
                is_significant = abs(target_share_delta) > 0.10
            elif position == "RB" and touch_share_delta:
                is_significant = abs(touch_share_delta) > 0.10

        return RegimeComparison(
            player_name=player_name,
            position=position,
            team=team,
            previous_regime_id=previous_metrics.regime_id,
            current_regime_id=current_metrics.regime_id,
            target_share_delta=target_share_delta,
            snap_share_delta=snap_share_delta,
            touch_share_delta=touch_share_delta,
            yards_per_route_delta=yards_per_route_delta,
            yards_per_carry_delta=yards_per_carry_delta,
            catch_rate_delta=catch_rate_delta,
            is_statistically_significant=is_significant,
            p_value=p_value,
        )

    def quantify_regime_impact(
        self,
        player_name: str,
        regime: Regime,
        market: str,
        baseline_projection: float,
        regime_metrics: RegimeMetrics,
        previous_metrics: Optional[RegimeMetrics],
        positional_baseline: float,
    ) -> RegimeImpact:
        """
        Quantify impact of regime change on projections.

        Args:
            player_name: Player name
            regime: Regime object
            market: Market type (e.g., "player_pass_yds")
            baseline_projection: Original projection without regime adjustment
            regime_metrics: Current regime metrics
            previous_metrics: Previous regime metrics
            positional_baseline: League average for position/role

        Returns:
            RegimeImpact with adjusted projections
        """
        # Calculate regime multiplier based on metrics
        regime_multiplier = 1.0

        if previous_metrics:
            # Compare usage/efficiency
            if "receptions" in market or "reception" in market:
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
                if (
                    regime_metrics.usage.touch_share
                    and previous_metrics.usage.touch_share
                    and previous_metrics.usage.touch_share > 0
                ):
                    regime_multiplier = (
                        regime_metrics.usage.touch_share
                        / previous_metrics.usage.touch_share
                    )

        # Confidence factor based on sample size
        confidence_factor = regime.sample_size_confidence

        # Raw impact
        raw_impact = baseline_projection * (regime_multiplier - 1.0)

        # Confidence-adjusted impact
        adjusted_impact = raw_impact * confidence_factor

        # Regime-adjusted projection
        regime_adjusted_projection = baseline_projection + adjusted_impact

        # Apply regression to mean (shrinkage toward positional baseline)
        # Shrinkage strength inversely proportional to games
        shrinkage_factor = 1.0 / (1.0 + regime_metrics.games_played)

        final_projection = (
            regime_adjusted_projection * (1 - shrinkage_factor)
            + positional_baseline * shrinkage_factor
        )

        return RegimeImpact(
            player_name=player_name,
            regime_id=regime.regime_id,
            market=market,
            baseline_projection=baseline_projection,
            regime_adjusted_projection=regime_adjusted_projection,
            absolute_impact=adjusted_impact,
            relative_impact_pct=(regime_multiplier - 1.0) * 100,
            raw_impact=raw_impact,
            confidence_factor=confidence_factor,
            adjusted_impact=adjusted_impact,
            positional_baseline=positional_baseline,
            shrinkage_factor=shrinkage_factor,
            final_projection=final_projection,
            sample_size=regime_metrics.games_played,
            regime_type=regime.trigger.type,
        )

    def _get_player_plays(
        self, player_name: str, pbp_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Get all plays involving a player (pass, rush, or receiving)."""
        player_plays = pbp_df[
            (pbp_df["passer_player_name"] == player_name)
            | (pbp_df["rusher_player_name"] == player_name)
            | (pbp_df["receiver_player_name"] == player_name)
        ].copy()

        return player_plays
