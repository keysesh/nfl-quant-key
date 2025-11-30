"""
Regime Change Reporting System

Generates human-readable reports and betting recommendations based on regime analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json

from .schemas import (
    Regime,
    RegimeMetrics,
    RegimeComparison,
    RegimeDetectionResult,
    TeamRegimeSummary,
    PlayerRegimeProfile,
    BettingRecommendation,
    RegimeType,
)


class RegimeReportGenerator:
    """
    Generates comprehensive reports for regime analysis.

    Output formats:
    - Markdown player reports
    - Team-level summaries
    - CSV exports
    - JSON API responses
    """

    def __init__(self):
        pass

    def generate_player_report(
        self,
        profile: PlayerRegimeProfile,
        current_week: int,
        opponent: str,
        projections: Dict[str, float],
        betting_lines: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate detailed player projection report.

        Args:
            profile: PlayerRegimeProfile with regime data
            current_week: Current week
            opponent: Opponent team abbreviation
            projections: Dict of market -> projection value
            betting_lines: Dict of market -> line value (optional)

        Returns:
            Formatted markdown report
        """
        regime = profile.current_regime
        metrics = profile.regime_metrics
        comparison = profile.regime_comparison

        # Header
        report = f"""
Player: {profile.player_name}
Team: {profile.team}
Position: {profile.position}
Current Regime: {regime.trigger.description} (Weeks {regime.start_week}-{current_week}, {regime.season})

"""

        # Regime-Adjusted Projections
        report += "## Regime-Adjusted Projections (Week {week} vs {opp})\n".format(
            week=current_week, opp=opponent
        )

        for market, proj in projections.items():
            market_display = market.replace("player_", "").replace("_", " ").title()

            # Get line if available
            line_str = ""
            if betting_lines and market in betting_lines:
                line = betting_lines[market]
                line_str = f" (Line: {line})"

            report += f"├─ {market_display}: {proj:.1f}{line_str}\n"

        report += "\n"

        # Key Regime Metrics
        report += "## Key Regime Metrics\n"

        if profile.position in ["WR", "TE"]:
            report += f"├─ Target Share: {metrics.usage.target_share:.1%}"
            if comparison and comparison.target_share_delta:
                delta_str = f" ({comparison.target_share_delta:+.1%})"
                report += delta_str
            report += "\n"

            if metrics.efficiency.yards_per_route_run:
                report += f"├─ Yards Per Route: {metrics.efficiency.yards_per_route_run:.2f}"
                if comparison and comparison.yards_per_route_delta:
                    delta_str = f" ({comparison.yards_per_route_delta:+.2f})"
                    report += delta_str
                report += "\n"

            if metrics.usage.targets_per_game:
                report += f"├─ Targets Per Game: {metrics.usage.targets_per_game:.1f}\n"

        elif profile.position == "RB":
            report += f"├─ Touch Share: {metrics.usage.touch_share:.1%}"
            if comparison and comparison.touch_share_delta:
                delta_str = f" ({comparison.touch_share_delta:+.1%})"
                report += delta_str
            report += "\n"

            if metrics.efficiency.yards_per_carry:
                report += f"├─ Yards Per Carry: {metrics.efficiency.yards_per_carry:.2f}"
                if comparison and comparison.yards_per_carry_delta:
                    delta_str = f" ({comparison.yards_per_carry_delta:+.2f})"
                    report += delta_str
                report += "\n"

            if metrics.usage.carries_per_game:
                report += f"├─ Carries Per Game: {metrics.usage.carries_per_game:.1f}\n"

        # Team context
        report += f"├─ Team Pass Rate: {metrics.context.team_pass_rate:.1%}\n"
        report += f"└─ Team Points Per Game: {metrics.context.team_points_per_game:.1f}\n"

        report += "\n"

        # Regime Change Impact
        if comparison and comparison.previous_regime_id:
            report += "## Regime Change Impact\n"

            impact_desc = self._describe_regime_impact(profile, comparison)
            report += impact_desc + "\n\n"

        # Sample Quality Warning
        if metrics.sample_quality in ["fair", "poor"]:
            report += "## ⚠️ Sample Size Warning\n"
            report += f"Only {metrics.games_played} games in current regime. "
            report += "Projections may be less reliable.\n\n"

        return report

    def generate_team_summary(
        self,
        team: str,
        season: int,
        current_week: int,
        detection_result: RegimeDetectionResult,
        player_metrics: List[RegimeMetrics],
        pbp_df: pd.DataFrame,
    ) -> TeamRegimeSummary:
        """
        Generate team-level regime summary.

        Args:
            team: Team abbreviation
            season: Season year
            current_week: Current week
            detection_result: Regime detection results
            player_metrics: List of player regime metrics
            pbp_df: Play-by-play data for team metrics

        Returns:
            TeamRegimeSummary object
        """
        if not detection_result.has_active_regime:
            # No regime - return neutral summary
            return TeamRegimeSummary(
                team=team,
                season=season,
                current_week=current_week,
                active_regime=None,
                regime_start_week=1,
                weeks_in_regime=current_week - 1,
                impact_level="LOW",
            )

        regime = detection_result.active_regime

        # Calculate team metric changes
        pre_regime_pbp = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["week"] < regime.start_week)
        ]

        regime_pbp = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["week"] >= regime.start_week)
            & (pbp_df["week"] < current_week)
        ]

        pass_rate_change = None
        pace_change = None
        points_per_game_change = None
        epa_change = None

        if len(pre_regime_pbp) > 0 and len(regime_pbp) > 0:
            # Pass rate
            pre_pass_rate = (pre_regime_pbp["play_type"] == "pass").mean()
            regime_pass_rate = (regime_pbp["play_type"] == "pass").mean()
            pass_rate_change = regime_pass_rate - pre_pass_rate

            # Pace (plays per game)
            pre_pace = len(pre_regime_pbp) / pre_regime_pbp["game_id"].nunique()
            regime_pace = len(regime_pbp) / regime_pbp["game_id"].nunique()
            pace_change = regime_pace - pre_pace

            # Points per game
            pre_ppg = self._calculate_ppg(team, pre_regime_pbp)
            regime_ppg = self._calculate_ppg(team, regime_pbp)
            points_per_game_change = regime_ppg - pre_ppg

            # EPA
            pre_epa = pre_regime_pbp["epa"].mean()
            regime_epa = regime_pbp["epa"].mean()
            epa_change = regime_epa - pre_epa

        # Position group impacts
        wr_metrics = [m for m in player_metrics if m.position == "WR"]
        te_metrics = [m for m in player_metrics if m.position == "TE"]
        rb_metrics = [m for m in player_metrics if m.position == "RB"]

        wr_target_change = (
            np.mean([m.usage.target_share for m in wr_metrics if m.usage.target_share])
            if wr_metrics
            else None
        )

        te_target_change = (
            np.mean([m.usage.target_share for m in te_metrics if m.usage.target_share])
            if te_metrics
            else None
        )

        rb_touch_change = (
            np.mean([m.usage.touch_share for m in rb_metrics if m.usage.touch_share])
            if rb_metrics
            else None
        )

        # Determine impact level
        impact_level = self._determine_impact_level(regime, pass_rate_change, epa_change)

        # Players to target/fade
        players_to_target = []
        players_to_fade = []

        for metrics in player_metrics:
            # Simple heuristic: increased usage = target, decreased = fade
            if metrics.usage.target_share and metrics.usage.target_share > 0.20:
                players_to_target.append(metrics.player_name)
            elif metrics.usage.touch_share and metrics.usage.touch_share > 0.40:
                players_to_target.append(metrics.player_name)

        return TeamRegimeSummary(
            team=team,
            season=season,
            current_week=current_week,
            active_regime=regime,
            regime_start_week=regime.start_week,
            weeks_in_regime=current_week - regime.start_week,
            pass_rate_change=pass_rate_change,
            pace_change=pace_change,
            points_per_game_change=points_per_game_change,
            epa_per_play_change=epa_change,
            wr_target_share_change=wr_target_change,
            te_target_share_change=te_target_change,
            rb_touch_share_change=rb_touch_change,
            players_to_target=players_to_target,
            players_to_fade=players_to_fade,
            impact_level=impact_level,
        )

    def generate_team_summary_markdown(
        self, summary: TeamRegimeSummary
    ) -> str:
        """
        Generate markdown report for team summary.

        Args:
            summary: TeamRegimeSummary object

        Returns:
            Formatted markdown string
        """
        if not summary.active_regime:
            return f"{summary.team} - No Active Regime Detected\n"

        regime = summary.active_regime

        report = f"""
{summary.team} - Regime Change Analysis

Active Regime: {regime.trigger.description} (Week {summary.regime_start_week}-{summary.current_week}, {summary.season})
Trigger Event: {regime.trigger.description}
Impact Level: {summary.impact_level}

"""

        # Team metric changes
        if any([summary.pass_rate_change, summary.pace_change, summary.points_per_game_change]):
            report += "## Team Metric Changes\n"

            if summary.pass_rate_change is not None:
                report += f"├─ Pass Rate: {summary.pass_rate_change:+.1%}\n"

            if summary.pace_change is not None:
                report += f"├─ Pace: {summary.pace_change:+.1f} plays/game\n"

            if summary.points_per_game_change is not None:
                report += f"├─ Points/Game: {summary.points_per_game_change:+.1f}\n"

            if summary.epa_per_play_change is not None:
                report += f"└─ EPA/Play: {summary.epa_per_play_change:+.2f}\n"

            report += "\n"

        # Players to target
        if summary.players_to_target:
            report += "## Players to Target\n"
            for i, player in enumerate(summary.players_to_target, 1):
                report += f"{i}. {player}\n"
            report += "\n"

        # Players to fade
        if summary.players_to_fade:
            report += "## Players to Fade\n"
            for i, player in enumerate(summary.players_to_fade, 1):
                report += f"{i}. {player}\n"
            report += "\n"

        return report

    def export_to_csv(
        self,
        player_profiles: List[PlayerRegimeProfile],
        output_path: str,
    ) -> None:
        """
        Export player regime data to CSV.

        Args:
            player_profiles: List of PlayerRegimeProfile objects
            output_path: Path to save CSV
        """
        rows = []

        for profile in player_profiles:
            metrics = profile.regime_metrics
            regime = profile.current_regime
            comparison = profile.regime_comparison

            row = {
                "player_name": profile.player_name,
                "player_id": profile.player_id,
                "position": profile.position,
                "team": profile.team,
                "regime_id": regime.regime_id,
                "regime_type": regime.trigger.type.value,
                "regime_description": regime.trigger.description,
                "regime_start_week": regime.start_week,
                "weeks_in_regime": metrics.weeks_in_regime,
                "games_played": metrics.games_played,
                "sample_quality": metrics.sample_quality,
                # Usage
                "snap_share": metrics.usage.snap_share,
                "target_share": metrics.usage.target_share,
                "touch_share": metrics.usage.touch_share,
                # Efficiency
                "yards_per_route": metrics.efficiency.yards_per_route_run,
                "yards_per_carry": metrics.efficiency.yards_per_carry,
                "catch_rate": metrics.efficiency.catch_rate,
                # Context
                "team_pass_rate": metrics.context.team_pass_rate,
                "team_ppg": metrics.context.team_points_per_game,
                # Comparison
                "target_share_delta": comparison.target_share_delta if comparison else None,
                "snap_share_delta": comparison.snap_share_delta if comparison else None,
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def export_to_json(
        self,
        detection_results: Dict[str, RegimeDetectionResult],
        output_path: str,
    ) -> None:
        """
        Export regime detection results to JSON.

        Args:
            detection_results: Dict of team -> RegimeDetectionResult
            output_path: Path to save JSON
        """
        output = {}

        for team, result in detection_results.items():
            output[team] = {
                "team": result.team,
                "season": result.season,
                "current_week": result.current_week,
                "has_active_regime": result.has_active_regime,
                "regime_count": result.regime_count,
                "affected_players": result.affected_players,
            }

            if result.active_regime:
                regime = result.active_regime
                output[team]["active_regime"] = {
                    "regime_id": regime.regime_id,
                    "type": regime.trigger.type.value,
                    "description": regime.trigger.description,
                    "confidence": regime.trigger.confidence,
                    "start_week": regime.start_week,
                    "games_in_regime": regime.games_in_regime,
                }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    def _describe_regime_impact(
        self, profile: PlayerRegimeProfile, comparison: RegimeComparison
    ) -> str:
        """Generate human-readable impact description."""
        impact_parts = []

        # Usage impact
        if comparison.target_share_delta:
            pct = comparison.target_share_delta * 100
            direction = "increased" if pct > 0 else "decreased"
            impact_parts.append(f"Target share {direction} by {abs(pct):.1f}%")

        if comparison.snap_share_delta:
            pct = comparison.snap_share_delta * 100
            direction = "increased" if pct > 0 else "decreased"
            impact_parts.append(f"Snap share {direction} by {abs(pct):.1f}%")

        # Efficiency impact
        if comparison.yards_per_route_delta:
            direction = "increased" if comparison.yards_per_route_delta > 0 else "decreased"
            impact_parts.append(
                f"Yards per route {direction} by {abs(comparison.yards_per_route_delta):.2f}"
            )

        if not impact_parts:
            return "Regime change detected, analyzing impact..."

        return f"{profile.player_name}'s production {' and '.join(impact_parts)} under new regime."

    def _calculate_ppg(self, team: str, pbp_df: pd.DataFrame) -> float:
        """Calculate points per game for a team."""
        games = pbp_df.groupby("game_id")
        points = []

        for game_id, game_df in games:
            is_home = game_df["home_team"].iloc[0] == team
            if is_home:
                team_score = game_df["total_home_score"].iloc[-1]
            else:
                team_score = game_df["total_away_score"].iloc[-1]
            points.append(team_score)

        return np.mean(points) if points else 0.0

    def _determine_impact_level(
        self,
        regime: Regime,
        pass_rate_change: Optional[float],
        epa_change: Optional[float],
    ) -> str:
        """Determine if regime has HIGH, MEDIUM, or LOW impact."""
        # QB changes are always HIGH impact
        if regime.trigger.type in [RegimeType.QB_CHANGE, RegimeType.QB_RETURN]:
            return "HIGH"

        # Coaching changes are MEDIUM-HIGH
        if regime.trigger.type == RegimeType.COACHING_CHANGE:
            return "MEDIUM"

        # Scheme changes depend on magnitude
        if pass_rate_change and abs(pass_rate_change) > 0.10:
            return "HIGH"

        if epa_change and abs(epa_change) > 0.15:
            return "HIGH"

        # Default
        return "MEDIUM"


class BettingRecommendationGenerator:
    """
    Generates betting recommendations with regime context.
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        min_confidence: float = 0.65,
    ):
        """
        Initialize recommendation generator.

        Args:
            min_edge: Minimum edge % to recommend bet
            min_confidence: Minimum confidence to recommend bet
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence

    def generate_recommendations(
        self,
        predictions_df: pd.DataFrame,
        lines_df: pd.DataFrame,
        regime_profiles: Dict[str, PlayerRegimeProfile],
    ) -> List[BettingRecommendation]:
        """
        Generate betting recommendations with regime flags.

        Args:
            predictions_df: Model predictions
            lines_df: Current betting lines
            regime_profiles: Dict of player_name -> PlayerRegimeProfile

        Returns:
            List of BettingRecommendation objects
        """
        recommendations = []

        # Merge predictions with lines
        merged = predictions_df.merge(
            lines_df,
            on=["player_name", "market"],
            how="inner",
        )

        for _, row in merged.iterrows():
            player_name = row["player_name"]
            market = row["market"]

            model_prob = row["over_probability"]
            line_price = row["over_price"]  # American odds

            # Convert to probability
            market_prob = self._american_to_prob(line_price)

            edge = model_prob - market_prob

            if edge < self.min_edge:
                continue  # Not enough edge

            # Get regime context
            regime_type = None
            regime_weeks = None
            regime_confidence = None
            regime_impact_pct = None
            regime_flag = None
            sample_warning = False

            if player_name in regime_profiles:
                profile = regime_profiles[player_name]
                regime = profile.current_regime
                metrics = profile.regime_metrics

                regime_type = regime.trigger.type
                regime_weeks = metrics.weeks_in_regime
                regime_confidence = regime.trigger.confidence

                # Sample size warning
                if metrics.sample_quality in ["fair", "poor"]:
                    sample_warning = True

                # Regime flag
                if regime_type == RegimeType.QB_CHANGE:
                    regime_flag = "🔄 New QB"
                elif regime_type == RegimeType.COACHING_CHANGE:
                    regime_flag = "🔄 New Coach"
                elif sample_warning:
                    regime_flag = "⚠️ Small Sample"

                # Regime impact
                if profile.regime_comparison:
                    comp = profile.regime_comparison
                    if comp.target_share_delta:
                        regime_impact_pct = comp.target_share_delta * 100

            # Kelly sizing (simplified)
            decimal_odds = self._american_to_decimal(line_price)
            kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
            scaled_kelly = kelly * 0.25  # Quarter Kelly

            # Placeholder bet size (would use bankroll in real system)
            bet_size = max(10, min(100, scaled_kelly * 1000))

            # Expected value
            potential_profit = bet_size * (decimal_odds - 1)
            expected_value = (model_prob * potential_profit) - ((1 - model_prob) * bet_size)

            # Confidence badge
            confidence_badge = "high" if model_prob > 0.70 else "medium" if model_prob > 0.60 else "low"

            rec = BettingRecommendation(
                player_name=player_name,
                team=row.get("team", ""),
                position=row.get("position", ""),
                opponent=row.get("opponent", ""),
                market=market,
                line=row["line"],
                side="over" if edge > 0 else "under",
                model_probability=model_prob,
                market_probability=market_prob,
                edge_pct=edge * 100,
                regime_type=regime_type,
                regime_weeks=regime_weeks,
                regime_confidence=regime_confidence,
                regime_impact_pct=regime_impact_pct,
                kelly_fraction=scaled_kelly,
                bet_size=bet_size,
                expected_value=expected_value,
                confidence_badge=confidence_badge,
                sample_size_warning=sample_warning,
                regime_flag=regime_flag,
            )

            recommendations.append(rec)

        return recommendations

    def _american_to_decimal(self, odds: float) -> float:
        """Convert American odds to decimal."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    def _american_to_prob(self, odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
