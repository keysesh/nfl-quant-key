"""Feature engineering engine for deriving team-level metrics."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from nfl_quant.config import settings
from nfl_quant.schemas import TeamWeekFeatures

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Derives team-level features from PBP and team stats."""

    def __init__(self) -> None:
        """Initialize feature engine."""
        self.epa_threshold = settings.EPA_THRESHOLD
        self.explosive_air_yards = settings.EXPLOSIVE_PLAY_AIR_YARDS_THRESHOLD
        self.explosive_total_yards = settings.EXPLOSIVE_PLAY_TOTAL_YARDS_THRESHOLD
        self.redzone_line = settings.REDZONE_YARD_LINE

    def derive_success_rate(
        self, pbp: pd.DataFrame, team: str, is_offense: bool, play_type: Optional[str] = None
    ) -> float:
        """Derive success rate (EPA > 0) for a team.

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation
            is_offense: True for offense, False for defense
            play_type: Optional filter ("pass" or "rush")

        Returns:
            Success rate as proportion
        """
        if is_offense:
            mask = pbp["posteam"] == team
        else:
            mask = pbp["defteam"] == team

        # Filter by play type if specified
        if play_type == "pass":
            mask &= pbp["play_type"] == "pass"
        elif play_type == "rush":
            mask &= pbp["play_type"] == "run"

        filtered = pbp[mask]
        if len(filtered) == 0:
            return 0.0

        successful = (filtered["epa"] > self.epa_threshold).sum()
        return float(successful / len(filtered))

    def derive_epa_per_play(
        self, pbp: pd.DataFrame, team: str, is_offense: bool, play_type: Optional[str] = None
    ) -> float:
        """Derive EPA per play for a team.

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation
            is_offense: True for offense, False for defense
            play_type: Optional filter ("pass" or "rush")

        Returns:
            EPA per play
        """
        if is_offense:
            mask = pbp["posteam"] == team
        else:
            mask = pbp["defteam"] == team

        # Filter by play type if specified
        if play_type == "pass":
            mask &= pbp["play_type"] == "pass"
        elif play_type == "rush":
            mask &= pbp["play_type"] == "run"

        filtered = pbp[mask]
        if len(filtered) == 0:
            return 0.0

        return float(filtered["epa"].mean())

    def derive_proe(self, pbp: pd.DataFrame, team: str) -> float:
        """Derive PROE (Pass Rate Over Expected) for a team.

        Args:
            pbp: Play-by-play DataFrame with xpass column
            team: Team abbreviation

        Returns:
            PROE = actual_pass_rate - expected_pass_rate
        """
        mask = pbp["posteam"] == team

        filtered = pbp[mask]
        if len(filtered) == 0:
            return 0.0

        # Actual pass rate
        pass_plays = (filtered["play_type"] == "pass").sum()
        actual_pass_rate = pass_plays / len(filtered) if len(filtered) > 0 else 0.0

        # Expected pass rate (xpass)
        if "xpass" in filtered.columns:
            expected_pass_rate = filtered["xpass"].mean()
        else:
            logger.warning("xpass column not found in PBP; PROE will be 0")
            return 0.0

        return float(actual_pass_rate - expected_pass_rate)

    def derive_neutral_pace(self, pbp: pd.DataFrame, team: str) -> float:
        """Derive neutral pace (seconds per play) excluding garbage time.

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation

        Returns:
            Median seconds per play (neutral script)
        """
        mask = pbp["posteam"] == team

        # Exclude garbage time (score diff > 21) and two-minute drills
        filtered = pbp[mask].copy()
        filtered["score_diff"] = (filtered["home_score"] - filtered["away_score"]).abs()
        filtered = filtered[filtered["score_diff"] <= 21]

        # Exclude two-minute drill
        if "seconds_remaining" in filtered.columns:
            filtered = filtered[filtered["seconds_remaining"] > 120]

        # Exclude kneel downs, spikes
        if "play_type" in filtered.columns:
            filtered = filtered[~filtered["play_type"].isin(["kneel", "no_play"])]

        if len(filtered) == 0:
            return 0.0

        if "seconds" in filtered.columns:
            return float(filtered["seconds"].median())
        else:
            return 0.0

    def derive_explosive_rate(
        self, pbp: pd.DataFrame, team: str, is_offense: bool
    ) -> float:
        """Derive explosive play rate (≥15 air yards or ≥20 total yards).

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation
            is_offense: True for offense, False for defense

        Returns:
            Explosive rate as proportion
        """
        if is_offense:
            mask = pbp["posteam"] == team
        else:
            mask = pbp["defteam"] == team

        filtered = pbp[mask].copy()
        if len(filtered) == 0:
            return 0.0

        explosive = False
        if "air_yards" in filtered.columns:
            explosive |= filtered["air_yards"] >= self.explosive_air_yards
        if "yards_gained" in filtered.columns:
            explosive |= filtered["yards_gained"] >= self.explosive_total_yards

        explosive_count = explosive.sum()
        return float(explosive_count / len(filtered))

    def derive_redzone_td_rate(
        self, pbp: pd.DataFrame, team: str, is_offense: bool
    ) -> float:
        """Derive red-zone TD rate (plays inside opponent 20 yard line).

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation
            is_offense: True for offense (TD%), False for defense (prevention %)

        Returns:
            TD rate within red zone
        """
        if is_offense:
            mask = pbp["posteam"] == team
        else:
            mask = pbp["defteam"] == team

        filtered = pbp[mask].copy()

        # Red zone: yardline <= 20 (offense perspective)
        if "yardline_100" in filtered.columns:
            rz_mask = filtered["yardline_100"] <= self.redzone_line
        else:
            logger.warning("yardline_100 column not found; red-zone rate will be 0")
            return 0.0

        rz_plays = filtered[rz_mask]
        if len(rz_plays) == 0:
            return 0.0

        if is_offense:
            td_mask = rz_plays["touchdown"] == 1
        else:
            td_mask = rz_plays["touchdown"] == 0

        td_count = td_mask.sum()
        return float(td_count / len(rz_plays))

    def derive_conversion_rates(
        self, pbp: pd.DataFrame, team: str, is_offense: bool, down: int
    ) -> float:
        """Derive 3rd/4th down conversion rates.

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation
            is_offense: True for offense, False for defense
            down: Down number (3 or 4)

        Returns:
            Conversion rate
        """
        if is_offense:
            mask = pbp["posteam"] == team
        else:
            mask = pbp["defteam"] == team

        filtered = pbp[mask].copy()
        filtered = filtered[filtered["down"] == down]

        if len(filtered) == 0:
            return 0.0

        if is_offense:
            conversions = (filtered["first_down"] == 1).sum()
        else:
            conversions = (filtered["first_down"] == 0).sum()

        return float(conversions / len(filtered))

    def derive_pressure_rate(self, pbp: pd.DataFrame, team: str) -> float:
        """Derive pressure/sack rate proxy for defense.

        Args:
            pbp: Play-by-play DataFrame
            team: Team abbreviation

        Returns:
            Pressure/sack rate as proportion
        """
        mask = pbp["defteam"] == team
        filtered = pbp[mask].copy()

        # Pass plays only
        filtered = filtered[filtered["play_type"] == "pass"]

        if len(filtered) == 0:
            return 0.0

        # Count sacks + pressures (if available)
        pressure_count = 0
        if "sack" in filtered.columns:
            pressure_count += (filtered["sack"] == 1).sum()
        if "pressure" in filtered.columns:
            pressure_count += (filtered["pressure"] == 1).sum()

        return float(pressure_count / len(filtered))

    def derive_team_week_features(
        self,
        pbp: pd.DataFrame,
        team: str,
        week: int,
        season: int = 2025,
        is_offense: bool = True,
    ) -> TeamWeekFeatures:
        """Derive all team-week features.

        Args:
            pbp: Play-by-play DataFrame filtered to week
            team: Team abbreviation
            week: Week number
            season: Season (must be 2025)
            is_offense: True for offense, False for defense

        Returns:
            TeamWeekFeatures object
        """
        season = settings.validate_season(season)

        epa_per_play = self.derive_epa_per_play(pbp, team, is_offense)
        passing_epa = self.derive_epa_per_play(pbp, team, is_offense, play_type="pass")
        rushing_epa = self.derive_epa_per_play(pbp, team, is_offense, play_type="rush")

        success_rate_overall = self.derive_success_rate(pbp, team, is_offense)
        success_rate_pass = self.derive_success_rate(pbp, team, is_offense, play_type="pass")
        success_rate_rush = self.derive_success_rate(pbp, team, is_offense, play_type="rush")

        proe = self.derive_proe(pbp, team) if is_offense else 0.0
        neutral_pace = self.derive_neutral_pace(pbp, team)

        explosive_rate = self.derive_explosive_rate(pbp, team, is_offense)
        redzone_td_rate = self.derive_redzone_td_rate(pbp, team, is_offense)
        third_down_rate = self.derive_conversion_rates(pbp, team, is_offense, down=3)
        fourth_down_rate = self.derive_conversion_rates(pbp, team, is_offense, down=4)
        pressure_rate = (
            self.derive_pressure_rate(pbp, team) if not is_offense else None
        )

        # Get play count
        if is_offense:
            mask = pbp["posteam"] == team
        else:
            mask = pbp["defteam"] == team
        play_count = mask.sum()

        # Estimate game count from plays (rough estimate: ~70 plays per game)
        game_count = max(1, play_count // 70)

        return TeamWeekFeatures(
            season=season,
            week=week,
            team=team,
            is_offense=is_offense,
            epa_per_play=epa_per_play,
            passing_epa=passing_epa,
            rushing_epa=rushing_epa,
            success_rate_overall=success_rate_overall,
            success_rate_pass=success_rate_pass,
            success_rate_rush=success_rate_rush,
            proe=proe,
            neutral_pace=neutral_pace,
            explosive_rate=explosive_rate,
            redzone_td_rate=redzone_td_rate,
            third_down_conv_rate=third_down_rate,
            fourth_down_conv_rate=fourth_down_rate,
            pressure_rate=pressure_rate,
            play_count=int(play_count),
            game_count=int(game_count),
            is_rolling_avg=False,
        )

    def validate_features_completeness(
        self, features: TeamWeekFeatures, min_games: int = 3
    ) -> bool:
        """Validate that features have sufficient data and no NaNs.

        Args:
            features: TeamWeekFeatures to validate
            min_games: Minimum games required

        Returns:
            True if valid, False otherwise
        """
        if features.game_count < min_games:
            logger.warning(
                f"{features.team} only has {features.game_count} games "
                f"(min required: {min_games})"
            )
            return False

        # Check for NaNs (except optional fields)
        required_fields = [
            "epa_per_play",
            "passing_epa",
            "rushing_epa",
            "success_rate_overall",
            "neutral_pace",
        ]

        for field in required_fields:
            value = getattr(features, field)
            if pd.isna(value):
                logger.warning(f"{features.team} has NaN in {field}")
                return False

        return True



