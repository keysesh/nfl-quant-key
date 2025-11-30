"""
Regime Change Detection Engine

Multi-dimensional detection system for identifying QB changes, coaching shifts,
roster impacts, and scheme changes that affect player projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats

from .schemas import (
    Regime,
    RegimeType,
    RegimeTrigger,
    QBChangeDetails,
    CoachingChangeDetails,
    RosterImpactDetails,
    SchemeChangeDetails,
    RegimeDetectionResult,
    RegimeWindow,
)


class RegimeDetector:
    """
    Main regime detection engine.

    Detects:
    - QB changes (injury, benching, return)
    - Coaching changes (OC, HC, play-caller)
    - Roster impact events (WR1 emergence, RB committees, O-line injuries)
    - Scheme changes (pass rate, tempo, formations)
    """

    def __init__(
        self,
        min_sample_size: int = 2,
        qb_snap_threshold: float = 0.5,
        target_share_threshold: float = 0.25,
        scheme_change_window: int = 3,
        scheme_change_threshold: float = 0.10,
    ):
        """
        Initialize detector with thresholds.

        Args:
            min_sample_size: Minimum games to consider regime valid
            qb_snap_threshold: Minimum snap % to be considered starter
            target_share_threshold: Target share for WR1 emergence
            scheme_change_window: Games for moving average
            scheme_change_threshold: Minimum % change to flag scheme shift
        """
        self.min_sample_size = min_sample_size
        self.qb_snap_threshold = qb_snap_threshold
        self.target_share_threshold = target_share_threshold
        self.scheme_change_window = scheme_change_window
        self.scheme_change_threshold = scheme_change_threshold

    def detect_all_regimes(
        self,
        team: str,
        current_week: int,
        season: int,
        pbp_df: pd.DataFrame,
        player_stats_df: Optional[pd.DataFrame] = None,
        coaching_data: Optional[pd.DataFrame] = None,
    ) -> RegimeDetectionResult:
        """
        Run all detection methods and return comprehensive result.

        Args:
            team: Team abbreviation (e.g., "KC")
            current_week: Current NFL week
            season: Season year
            pbp_df: Play-by-play DataFrame
            player_stats_df: Weekly player stats (optional)
            coaching_data: Coaching staff data (optional)

        Returns:
            RegimeDetectionResult with all detected regimes
        """
        regimes = []
        affected_players = set()

        # 1. QB Change Detection
        qb_regime = self.detect_qb_changes(team, current_week, season, pbp_df)
        if qb_regime and qb_regime.trigger.type != RegimeType.STABLE:
            regimes.append(qb_regime)
            affected_players.update(qb_regime.affected_players)

        # 2. Coaching Changes
        if coaching_data is not None:
            coaching_regime = self.detect_coaching_changes(
                team, current_week, season, coaching_data, pbp_df
            )
            if coaching_regime:
                regimes.append(coaching_regime)
                affected_players.update(coaching_regime.affected_players)

        # 3. Roster Impact Events
        if player_stats_df is not None:
            roster_regimes = self.detect_roster_impacts(
                team, current_week, season, player_stats_df, pbp_df
            )
            regimes.extend(roster_regimes)
            for regime in roster_regimes:
                affected_players.update(regime.affected_players)

        # 4. Scheme Changes
        scheme_regimes = self.detect_scheme_changes(
            team, current_week, season, pbp_df
        )
        regimes.extend(scheme_regimes)
        for regime in scheme_regimes:
            affected_players.update(regime.affected_players)

        # Determine active regime (most recent)
        active_regime = None
        if regimes:
            active_regime = max(regimes, key=lambda r: r.start_week)

        return RegimeDetectionResult(
            team=team,
            season=season,
            current_week=current_week,
            regimes_detected=regimes,
            active_regime=active_regime,
            affected_players=list(affected_players),
        )

    def detect_qb_changes(
        self,
        team: str,
        current_week: int,
        season: int,
        pbp_df: pd.DataFrame,
    ) -> Optional[Regime]:
        """
        Detect QB regime changes.

        Identifies:
        - Starter switches (planned or injury)
        - Backup QB elevated to starter
        - Veteran/rookie returns from injury

        Args:
            team: Team abbreviation
            current_week: Current week number
            season: Season year
            pbp_df: Play-by-play DataFrame with columns:
                - posteam, week, play_type, passer_player_name, qb_dropback

        Returns:
            Regime object if change detected, None if stable
        """
        # Filter to team's pass plays
        team_passes = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["play_type"] == "pass")
            & (pbp_df["week"] < current_week)
        ].copy()

        if len(team_passes) == 0:
            return None

        # Get QB with most dropbacks per week
        qb_by_week = (
            team_passes.groupby(["week", "passer_player_name"])
            .size()
            .reset_index(name="dropbacks")
        )

        # Get primary QB per week (most dropbacks)
        primary_qbs = (
            qb_by_week.sort_values("dropbacks", ascending=False)
            .groupby("week")
            .first()
            .reset_index()
        )

        if len(primary_qbs) == 0:
            return None

        # Sort by week
        primary_qbs = primary_qbs.sort_values("week")

        weeks = primary_qbs["week"].tolist()
        qbs = primary_qbs["passer_player_name"].tolist()
        dropbacks = primary_qbs["dropbacks"].tolist()

        # Find most recent QB change
        regime_start_week = weeks[0]
        current_qb = qbs[-1]
        previous_qb = None
        change_reason = "planned"

        for i in range(len(weeks) - 1, 0, -1):
            if qbs[i] != qbs[i - 1]:
                regime_start_week = weeks[i]
                previous_qb = qbs[i - 1]

                # Determine change reason
                # If previous QB had been consistent, likely injury
                if i > 1 and qbs[i - 1] == qbs[i - 2]:
                    change_reason = "injury"
                # If current QB appeared earlier, it's a return
                elif current_qb in qbs[:i]:
                    change_reason = "return"
                else:
                    change_reason = "benching"

                break

        # Calculate regime weeks
        regime_weeks = [w for w in weeks if w >= regime_start_week]
        games_in_regime = len(regime_weeks)

        # If no change or very recent (< min_sample), return stable
        if previous_qb is None:
            regime_type = RegimeType.STABLE
            confidence = 1.0
        elif games_in_regime < self.min_sample_size:
            regime_type = RegimeType.QB_CHANGE
            confidence = 0.40  # Low confidence with small sample
        else:
            regime_type = (
                RegimeType.QB_RETURN if change_reason == "return" else RegimeType.QB_CHANGE
            )
            # Confidence grows with sample size
            confidence = min(1.0, 0.50 + (games_in_regime * 0.15))

        # Calculate passing efficiency delta if we have previous regime data
        passing_epa_delta = None
        completion_pct_delta = None

        if previous_qb:
            # Previous regime stats
            prev_regime_passes = team_passes[
                (team_passes["passer_player_name"] == previous_qb)
                & (team_passes["week"] < regime_start_week)
            ]

            # Current regime stats
            curr_regime_passes = team_passes[
                (team_passes["passer_player_name"] == current_qb)
                & (team_passes["week"] >= regime_start_week)
            ]

            if len(prev_regime_passes) > 0 and len(curr_regime_passes) > 0:
                prev_epa = prev_regime_passes["epa"].mean()
                curr_epa = curr_regime_passes["epa"].mean()
                passing_epa_delta = curr_epa - prev_epa

                # Completion percentage
                prev_comp_pct = (
                    prev_regime_passes["complete_pass"].sum() / len(prev_regime_passes)
                )
                curr_comp_pct = (
                    curr_regime_passes["complete_pass"].sum() / len(curr_regime_passes)
                )
                completion_pct_delta = curr_comp_pct - prev_comp_pct

        # Get all offensive players for this team
        affected_players = self._get_offensive_players(team, pbp_df)

        # Create QB change details
        qb_details = QBChangeDetails(
            previous_qb=previous_qb,
            current_qb=current_qb,
            change_reason=change_reason,
            games_missed=0,  # Could calculate from injury data if available
            passing_efficiency_delta=passing_epa_delta,
            completion_pct_delta=completion_pct_delta,
        )

        regime_id = f"{season}_{team}_{regime_type.value}_{regime_start_week}"

        trigger_description = (
            f"{current_qb} started as QB"
            if not previous_qb
            else f"{current_qb} replaced {previous_qb} ({change_reason})"
        )

        return Regime(
            regime_id=regime_id,
            team=team,
            season=season,
            start_week=regime_start_week,
            end_week=None,  # Still active
            trigger=RegimeTrigger(
                type=regime_type,
                description=trigger_description,
                confidence=confidence,
            ),
            details=qb_details,
            affected_players=affected_players,
            games_in_regime=games_in_regime,
            is_active=True,
        )

    def detect_coaching_changes(
        self,
        team: str,
        current_week: int,
        season: int,
        coaching_data: pd.DataFrame,
        pbp_df: pd.DataFrame,
    ) -> Optional[Regime]:
        """
        Detect offensive coordinator or play-caller changes.

        Args:
            team: Team abbreviation
            current_week: Current week
            season: Season year
            coaching_data: DataFrame with columns: team, week, oc, hc, playcaller
            pbp_df: Play-by-play data for context

        Returns:
            Regime if coaching change detected
        """
        team_coaching = coaching_data[
            (coaching_data["team"] == team)
            & (coaching_data["season"] == season)
            & (coaching_data["week"] < current_week)
        ].sort_values("week")

        if len(team_coaching) < 2:
            return None

        # Check for OC change
        oc_change_week = None
        prev_oc = None
        curr_oc = None

        for i in range(1, len(team_coaching)):
            if team_coaching.iloc[i]["oc"] != team_coaching.iloc[i - 1]["oc"]:
                oc_change_week = team_coaching.iloc[i]["week"]
                prev_oc = team_coaching.iloc[i - 1]["oc"]
                curr_oc = team_coaching.iloc[i]["oc"]
                break

        if oc_change_week is None:
            return None  # No coaching change detected

        games_in_regime = current_week - oc_change_week
        confidence = min(1.0, 0.60 + (games_in_regime * 0.10))

        # Analyze scheme shift
        pre_regime = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["week"] < oc_change_week)
        ]
        post_regime = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["week"] >= oc_change_week)
            & (pbp_df["week"] < current_week)
        ]

        philosophy_shift = None
        if len(pre_regime) > 0 and len(post_regime) > 0:
            pre_pass_rate = (pre_regime["play_type"] == "pass").mean()
            post_pass_rate = (post_regime["play_type"] == "pass").mean()

            if post_pass_rate > pre_pass_rate + 0.05:
                philosophy_shift = "pass_heavy"
            elif post_pass_rate < pre_pass_rate - 0.05:
                philosophy_shift = "run_heavy"
            else:
                philosophy_shift = "balanced"

        coaching_details = CoachingChangeDetails(
            previous_oc=prev_oc,
            current_oc=curr_oc,
            change_type="oc",
            offensive_philosophy_shift=philosophy_shift,
        )

        affected_players = self._get_offensive_players(team, pbp_df)

        regime_id = f"{season}_{team}_coaching_change_{oc_change_week}"

        return Regime(
            regime_id=regime_id,
            team=team,
            season=season,
            start_week=oc_change_week,
            end_week=None,
            trigger=RegimeTrigger(
                type=RegimeType.COACHING_CHANGE,
                description=f"OC changed from {prev_oc} to {curr_oc}",
                confidence=confidence,
            ),
            details=coaching_details,
            affected_players=affected_players,
            games_in_regime=games_in_regime,
            is_active=True,
        )

    def detect_roster_impacts(
        self,
        team: str,
        current_week: int,
        season: int,
        player_stats_df: pd.DataFrame,
        pbp_df: pd.DataFrame,
    ) -> List[Regime]:
        """
        Detect roster impact events:
        - WR1 emergence (sustained 25%+ target share)
        - RB committee formation
        - O-line injuries affecting efficiency
        - TE becoming focal point

        Args:
            team: Team abbreviation
            current_week: Current week
            season: Season year
            player_stats_df: Weekly player stats
            pbp_df: Play-by-play data

        Returns:
            List of roster impact regimes
        """
        regimes = []

        team_stats = player_stats_df[
            (player_stats_df["team"] == team)
            & (player_stats_df["season"] == season)
            & (player_stats_df["week"] < current_week)
        ].copy()

        if len(team_stats) == 0:
            return regimes

        # 1. WR1 Emergence Detection
        wr_regime = self._detect_wr1_emergence(team, current_week, season, team_stats)
        if wr_regime:
            regimes.append(wr_regime)

        # 2. RB Committee Detection
        rb_regime = self._detect_rb_committee(team, current_week, season, team_stats)
        if rb_regime:
            regimes.append(rb_regime)

        # 3. TE Focal Point Detection
        te_regime = self._detect_te_focal(team, current_week, season, team_stats)
        if te_regime:
            regimes.append(te_regime)

        return regimes

    def _detect_wr1_emergence(
        self,
        team: str,
        current_week: int,
        season: int,
        team_stats: pd.DataFrame,
    ) -> Optional[Regime]:
        """Detect if a new WR1 has emerged (25%+ target share for 3+ weeks)."""
        wr_stats = team_stats[team_stats["position"] == "WR"].copy()

        if len(wr_stats) == 0:
            return None

        # Calculate weekly team targets
        weekly_team_targets = (
            wr_stats.groupby("week")["targets"].sum().reset_index(name="team_targets")
        )

        wr_stats = wr_stats.merge(weekly_team_targets, on="week")
        wr_stats["target_share"] = wr_stats["targets"] / wr_stats["team_targets"]

        # Find players with 25%+ target share
        dominant_wrs = wr_stats[wr_stats["target_share"] >= self.target_share_threshold]

        if len(dominant_wrs) == 0:
            return None

        # Group by player and find consecutive weeks
        for player in dominant_wrs["player_name"].unique():
            player_data = dominant_wrs[dominant_wrs["player_name"] == player].sort_values(
                "week"
            )

            # Check for 3+ consecutive weeks
            weeks = player_data["week"].tolist()
            if len(weeks) >= 3:
                # Check if consecutive
                consecutive_weeks = []
                for i, week in enumerate(weeks):
                    if i == 0 or week == weeks[i - 1] + 1:
                        consecutive_weeks.append(week)
                    else:
                        consecutive_weeks = [week]

                    if len(consecutive_weeks) >= 3:
                        regime_start_week = consecutive_weeks[0]
                        avg_target_share = player_data[
                            player_data["week"] >= regime_start_week
                        ]["target_share"].mean()

                        roster_details = RosterImpactDetails(
                            position="WR",
                            affected_players=[player],
                            impact_type="emergence",
                            target_share_changes={player: avg_target_share},
                        )

                        regime_id = f"{season}_{team}_wr1_emergence_{regime_start_week}"

                        return Regime(
                            regime_id=regime_id,
                            team=team,
                            season=season,
                            start_week=regime_start_week,
                            end_week=None,
                            trigger=RegimeTrigger(
                                type=RegimeType.ROSTER_WR1_EMERGENCE,
                                description=f"{player} emerged as WR1 ({avg_target_share:.1%} target share)",
                                confidence=0.80,
                            ),
                            details=roster_details,
                            affected_players=[player],
                            games_in_regime=len(consecutive_weeks),
                            is_active=True,
                        )

        return None

    def _detect_rb_committee(
        self,
        team: str,
        current_week: int,
        season: int,
        team_stats: pd.DataFrame,
    ) -> Optional[Regime]:
        """Detect RB committee (no RB with >60% carry share for 3+ weeks)."""
        rb_stats = team_stats[team_stats["position"] == "RB"].copy()

        if len(rb_stats) == 0:
            return None

        # Calculate weekly team carries
        weekly_team_carries = (
            rb_stats.groupby("week")["rush_attempts"]
            .sum()
            .reset_index(name="team_carries")
        )

        rb_stats = rb_stats.merge(weekly_team_carries, on="week")
        rb_stats["carry_share"] = rb_stats["rush_attempts"] / rb_stats["team_carries"]

        # Find weeks where NO RB has >60% carry share
        committee_weeks = []
        for week in rb_stats["week"].unique():
            week_rbs = rb_stats[rb_stats["week"] == week]
            max_share = week_rbs["carry_share"].max()

            if max_share < 0.60:
                committee_weeks.append(week)

        # Need 3+ consecutive committee weeks
        if len(committee_weeks) < 3:
            return None

        committee_weeks.sort()

        # Find start of current committee period
        regime_start_week = committee_weeks[-3]  # Start of last 3-week stretch

        # Get RBs in committee
        committee_rbs = rb_stats[rb_stats["week"] >= regime_start_week]
        rb_names = committee_rbs.groupby("player_name")["carry_share"].mean()
        rb_names = rb_names[rb_names > 0.20].index.tolist()  # At least 20% share

        if len(rb_names) < 2:
            return None  # Not really a committee

        carry_changes = rb_names_dict = {
            name: committee_rbs[committee_rbs["player_name"] == name][
                "carry_share"
            ].mean()
            for name in rb_names
        }

        roster_details = RosterImpactDetails(
            position="RB",
            affected_players=rb_names,
            impact_type="committee",
            touch_share_changes=carry_changes,
        )

        regime_id = f"{season}_{team}_rb_committee_{regime_start_week}"

        return Regime(
            regime_id=regime_id,
            team=team,
            season=season,
            start_week=regime_start_week,
            end_week=None,
            trigger=RegimeTrigger(
                type=RegimeType.ROSTER_RB_COMMITTEE,
                description=f"RB committee formed: {', '.join(rb_names)}",
                confidence=0.75,
            ),
            details=roster_details,
            affected_players=rb_names,
            games_in_regime=len(committee_weeks) - 2,
            is_active=True,
        )

    def _detect_te_focal(
        self,
        team: str,
        current_week: int,
        season: int,
        team_stats: pd.DataFrame,
    ) -> Optional[Regime]:
        """Detect TE becoming primary receiving option (>20% target share for 3+ weeks)."""
        te_stats = team_stats[team_stats["position"] == "TE"].copy()

        if len(te_stats) == 0:
            return None

        # Calculate weekly team targets (all positions)
        weekly_team_targets = (
            team_stats.groupby("week")["targets"].sum().reset_index(name="team_targets")
        )

        te_stats = te_stats.merge(weekly_team_targets, on="week")
        te_stats["target_share"] = te_stats["targets"] / te_stats["team_targets"]

        # Find TEs with 20%+ target share for 3+ consecutive weeks
        focal_tes = te_stats[te_stats["target_share"] >= 0.20]

        if len(focal_tes) == 0:
            return None

        for player in focal_tes["player_name"].unique():
            player_data = focal_tes[focal_tes["player_name"] == player].sort_values("week")

            weeks = player_data["week"].tolist()
            if len(weeks) >= 3:
                # Check consecutive
                consecutive_weeks = []
                for i, week in enumerate(weeks):
                    if i == 0 or week == weeks[i - 1] + 1:
                        consecutive_weeks.append(week)
                    else:
                        consecutive_weeks = [week]

                    if len(consecutive_weeks) >= 3:
                        regime_start_week = consecutive_weeks[0]
                        avg_target_share = player_data[
                            player_data["week"] >= regime_start_week
                        ]["target_share"].mean()

                        roster_details = RosterImpactDetails(
                            position="TE",
                            affected_players=[player],
                            impact_type="emergence",
                            target_share_changes={player: avg_target_share},
                        )

                        regime_id = f"{season}_{team}_te_focal_{regime_start_week}"

                        return Regime(
                            regime_id=regime_id,
                            team=team,
                            season=season,
                            start_week=regime_start_week,
                            end_week=None,
                            trigger=RegimeTrigger(
                                type=RegimeType.ROSTER_TE_FOCAL,
                                description=f"{player} became focal TE ({avg_target_share:.1%} target share)",
                                confidence=0.80,
                            ),
                            details=roster_details,
                            affected_players=[player],
                            games_in_regime=len(consecutive_weeks),
                            is_active=True,
                        )

        return None

    def detect_scheme_changes(
        self,
        team: str,
        current_week: int,
        season: int,
        pbp_df: pd.DataFrame,
    ) -> List[Regime]:
        """
        Detect scheme changes:
        - Pass rate shifts (>10% change over 3-week MA)
        - Tempo changes (plays per game variance)
        - Formation changes

        Args:
            team: Team abbreviation
            current_week: Current week
            season: Season year
            pbp_df: Play-by-play DataFrame

        Returns:
            List of scheme change regimes
        """
        regimes = []

        team_pbp = pbp_df[
            (pbp_df["posteam"] == team)
            & (pbp_df["week"] < current_week)
        ].copy()

        if len(team_pbp) == 0:
            return regimes

        # 1. Pass Rate Shift Detection
        pass_rate_regime = self._detect_pass_rate_shift(
            team, current_week, season, team_pbp
        )
        if pass_rate_regime:
            regimes.append(pass_rate_regime)

        # 2. Tempo Change Detection
        tempo_regime = self._detect_tempo_change(
            team, current_week, season, team_pbp
        )
        if tempo_regime:
            regimes.append(tempo_regime)

        return regimes

    def _detect_pass_rate_shift(
        self,
        team: str,
        current_week: int,
        season: int,
        team_pbp: pd.DataFrame,
    ) -> Optional[Regime]:
        """Detect significant pass rate changes."""
        # Calculate weekly pass rate
        weekly_pass_rate = (
            team_pbp.groupby("week")
            .apply(lambda x: (x["play_type"] == "pass").mean())
            .reset_index(name="pass_rate")
        )

        if len(weekly_pass_rate) < 6:
            return None  # Need at least 6 weeks

        # Calculate 3-week moving average
        weekly_pass_rate["ma3"] = (
            weekly_pass_rate["pass_rate"].rolling(window=self.scheme_change_window).mean()
        )

        # Find significant shifts (>10% change)
        for i in range(self.scheme_change_window, len(weekly_pass_rate) - 1):
            prev_ma = weekly_pass_rate.iloc[i - 1]["ma3"]
            curr_ma = weekly_pass_rate.iloc[i]["ma3"]

            if pd.isna(prev_ma) or pd.isna(curr_ma):
                continue

            pct_change = (curr_ma - prev_ma) / prev_ma

            if abs(pct_change) >= self.scheme_change_threshold:
                regime_start_week = weekly_pass_rate.iloc[i]["week"]
                games_in_regime = current_week - regime_start_week

                if games_in_regime < 2:
                    continue  # Too recent

                scheme_details = SchemeChangeDetails(
                    metric_changed="pass_rate",
                    previous_value=prev_ma,
                    current_value=curr_ma,
                    percent_change=pct_change,
                    moving_average_window=self.scheme_change_window,
                )

                affected_players = self._get_offensive_players(team, team_pbp)

                regime_id = f"{season}_{team}_pass_rate_shift_{regime_start_week}"

                direction = "increased" if pct_change > 0 else "decreased"

                return Regime(
                    regime_id=regime_id,
                    team=team,
                    season=season,
                    start_week=regime_start_week,
                    end_week=None,
                    trigger=RegimeTrigger(
                        type=RegimeType.SCHEME_PASS_RATE_SHIFT,
                        description=f"Pass rate {direction} by {abs(pct_change):.1%}",
                        confidence=0.75,
                    ),
                    details=scheme_details,
                    affected_players=affected_players,
                    games_in_regime=games_in_regime,
                    is_active=True,
                )

        return None

    def _detect_tempo_change(
        self,
        team: str,
        current_week: int,
        season: int,
        team_pbp: pd.DataFrame,
    ) -> Optional[Regime]:
        """Detect significant tempo (pace) changes."""
        # Calculate plays per game
        plays_per_game = (
            team_pbp.groupby(["week", "game_id"])
            .size()
            .reset_index(name="plays")
            .groupby("week")["plays"]
            .mean()
            .reset_index(name="plays_per_game")
        )

        if len(plays_per_game) < 6:
            return None

        # Calculate 3-week MA
        plays_per_game["ma3"] = (
            plays_per_game["plays_per_game"]
            .rolling(window=self.scheme_change_window)
            .mean()
        )

        # Detect shifts >10%
        for i in range(self.scheme_change_window, len(plays_per_game) - 1):
            prev_ma = plays_per_game.iloc[i - 1]["ma3"]
            curr_ma = plays_per_game.iloc[i]["ma3"]

            if pd.isna(prev_ma) or pd.isna(curr_ma):
                continue

            pct_change = (curr_ma - prev_ma) / prev_ma

            if abs(pct_change) >= self.scheme_change_threshold:
                regime_start_week = plays_per_game.iloc[i]["week"]
                games_in_regime = current_week - regime_start_week

                if games_in_regime < 2:
                    continue

                scheme_details = SchemeChangeDetails(
                    metric_changed="tempo",
                    previous_value=prev_ma,
                    current_value=curr_ma,
                    percent_change=pct_change,
                    moving_average_window=self.scheme_change_window,
                )

                affected_players = self._get_offensive_players(team, team_pbp)

                regime_id = f"{season}_{team}_tempo_change_{regime_start_week}"

                direction = "increased" if pct_change > 0 else "decreased"

                return Regime(
                    regime_id=regime_id,
                    team=team,
                    season=season,
                    start_week=regime_start_week,
                    end_week=None,
                    trigger=RegimeTrigger(
                        type=RegimeType.SCHEME_TEMPO_CHANGE,
                        description=f"Tempo {direction} by {abs(pct_change):.1%} ({curr_ma:.1f} plays/game)",
                        confidence=0.70,
                    ),
                    details=scheme_details,
                    affected_players=affected_players,
                    games_in_regime=games_in_regime,
                    is_active=True,
                )

        return None

    def _get_offensive_players(
        self, team: str, pbp_df: pd.DataFrame
    ) -> List[str]:
        """Extract list of offensive players for a team from PBP data."""
        team_offense = pbp_df[pbp_df["posteam"] == team]

        players = set()

        # Add passers
        if "passer_player_name" in team_offense.columns:
            players.update(
                team_offense["passer_player_name"].dropna().unique()
            )

        # Add receivers
        if "receiver_player_name" in team_offense.columns:
            players.update(
                team_offense["receiver_player_name"].dropna().unique()
            )

        # Add rushers
        if "rusher_player_name" in team_offense.columns:
            players.update(
                team_offense["rusher_player_name"].dropna().unique()
            )

        return list(players)

    def get_regime_window(
        self, regime: Regime, current_week: int
    ) -> RegimeWindow:
        """
        Get the time window for regime-specific analysis.

        Args:
            regime: Regime object
            current_week: Current week number

        Returns:
            RegimeWindow with weeks to use for analysis
        """
        end_week = regime.end_week if regime.end_week else current_week

        weeks = list(range(regime.start_week, end_week))

        return RegimeWindow(
            regime_id=regime.regime_id,
            start_week=regime.start_week,
            end_week=end_week,
            weeks=weeks,
            season=regime.season,
        )

    def validate_regime_sample_size(
        self, regime: Regime, min_games: int = 2
    ) -> Tuple[bool, str]:
        """
        Validate if regime has sufficient sample size.

        Args:
            regime: Regime to validate
            min_games: Minimum required games

        Returns:
            (is_valid, warning_message)
        """
        if regime.games_in_regime >= min_games:
            return True, ""

        warning = (
            f"Regime '{regime.regime_id}' has only {regime.games_in_regime} game(s). "
            f"Minimum {min_games} recommended for reliable analysis."
        )

        return False, warning
