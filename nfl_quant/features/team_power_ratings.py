"""
Team Power Ratings Module
=========================
Comprehensive team strength system combining:
- Elo ratings (60% weight) - FiveThirtyEight methodology
- EPA-based power (30% weight) - Offensive/Defensive efficiency
- Strength of Schedule (10% weight) - Opponent-adjusted metrics

This provides a unified power rating for game line predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

from .team_strength import EnhancedEloCalculator
from ..utils.epa_utils import (
    calculate_team_offensive_epa,
    calculate_team_defensive_epa,
    regress_epa_to_mean
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TeamPowerRatings:
    """
    Unified team power rating system.

    Combines Elo, EPA, and strength of schedule into a single
    power rating that can be used for game line predictions.
    """

    # Weighting factors
    ELO_WEIGHT = 0.60
    EPA_WEIGHT = 0.30
    SOS_WEIGHT = 0.10

    # EPA to points conversion (calibrated)
    EPA_TO_POINTS_BASE = 3.5

    # League averages
    LEAGUE_AVG_PPG = 22.5
    LEAGUE_AVG_TOTAL = 45.0

    def __init__(self):
        self.elo_calculator = EnhancedEloCalculator()
        self._pbp_df: Optional[pd.DataFrame] = None
        self._power_cache: Dict[Tuple[int, int], Dict[str, Dict]] = {}

    def _load_pbp(self) -> pd.DataFrame:
        """Load play-by-play data."""
        if self._pbp_df is not None:
            return self._pbp_df

        pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
        if not pbp_path.exists():
            raise FileNotFoundError(f"PBP data not found at {pbp_path}")

        self._pbp_df = pd.read_parquet(pbp_path)
        return self._pbp_df

    def calculate_team_power(
        self,
        team: str,
        season: int,
        week: int,
        lookback_weeks: int = 6
    ) -> Dict:
        """
        Calculate comprehensive power rating for a team.

        Args:
            team: Team abbreviation
            season: Season year
            week: Target week (uses data BEFORE this week)
            lookback_weeks: Weeks to look back for EPA

        Returns:
            Dict with power components and combined rating
        """
        # 1. Get Elo rating
        elo_ratings = self.elo_calculator.calculate_elo_through_week(season, week)
        team_elo = elo_ratings.get(team, 1505)

        # Normalize Elo to 0-100 scale (1300-1700 range)
        elo_normalized = (team_elo - 1300) / 4  # 1300=0, 1700=100
        elo_normalized = max(0, min(100, elo_normalized))

        # 2. Get EPA metrics
        pbp = self._load_pbp()
        pbp_season = pbp[pbp['season'] == season].copy()

        # Only use weeks before target week
        pbp_filtered = pbp_season[pbp_season['week'] < week]

        off_epa = calculate_team_offensive_epa(pbp_filtered, team, lookback_weeks)
        def_epa = calculate_team_defensive_epa(pbp_filtered, team, lookback_weeks)

        # Net EPA = Offense - Defense (defense positive = bad)
        net_epa = off_epa['off_epa'] - def_epa['def_epa_allowed']

        # Normalize EPA to 0-100 scale (-0.3 to +0.3 range)
        epa_normalized = (net_epa + 0.3) / 0.6 * 100
        epa_normalized = max(0, min(100, epa_normalized))

        # 3. Calculate Strength of Schedule
        sos = self._calculate_strength_of_schedule(team, season, week, elo_ratings)

        # Normalize SOS to 0-100 scale
        sos_normalized = (sos - 1400) / 2  # 1400=0, 1600=100
        sos_normalized = max(0, min(100, sos_normalized))

        # 4. Combine into unified power rating
        combined_power = (
            self.ELO_WEIGHT * elo_normalized +
            self.EPA_WEIGHT * epa_normalized +
            self.SOS_WEIGHT * sos_normalized
        )

        # Calculate expected points per game
        expected_ppg = self.LEAGUE_AVG_PPG + (net_epa * 60 * self.EPA_TO_POINTS_BASE)

        return {
            'team': team,
            'power_rating': combined_power,
            'elo': team_elo,
            'elo_normalized': elo_normalized,
            'net_epa': net_epa,
            'off_epa': off_epa['off_epa'],
            'def_epa': def_epa['def_epa_allowed'],
            'pass_off_epa': off_epa['pass_off_epa'],
            'rush_off_epa': off_epa['rush_off_epa'],
            'pass_def_epa': def_epa['pass_def_epa'],
            'rush_def_epa': def_epa['rush_def_epa'],
            'epa_normalized': epa_normalized,
            'sos': sos,
            'sos_normalized': sos_normalized,
            'expected_ppg': expected_ppg,
            'sample_games': min(off_epa['sample_games'], def_epa['sample_games'])
        }

    def _calculate_strength_of_schedule(
        self,
        team: str,
        season: int,
        week: int,
        elo_ratings: Dict[str, float]
    ) -> float:
        """Calculate strength of schedule based on opponents' Elo."""
        schedules = self.elo_calculator.schedules

        # Get games played before target week
        team_games = schedules[
            (schedules['season'] == season) &
            (schedules['week'] < week) &
            ((schedules['home_team'] == team) | (schedules['away_team'] == team)) &
            (schedules['home_score'].notna())
        ]

        if len(team_games) == 0:
            return 1505  # League average

        opponent_elos = []
        for _, game in team_games.iterrows():
            opponent = game['away_team'] if game['home_team'] == team else game['home_team']
            opp_elo = elo_ratings.get(opponent, 1505)
            opponent_elos.append(opp_elo)

        return np.mean(opponent_elos) if opponent_elos else 1505

    def get_all_team_power_ratings(
        self,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Get power ratings for all teams entering a week.

        Returns DataFrame sorted by power rating.
        """
        cache_key = (season, week)
        if cache_key in self._power_cache:
            ratings = self._power_cache[cache_key]
        else:
            # Get all teams for season
            teams = self.elo_calculator._get_teams_for_season(season)

            ratings = {}
            for team in teams:
                try:
                    ratings[team] = self.calculate_team_power(team, season, week)
                except Exception as e:
                    logger.warning(f"Error calculating power for {team}: {e}")

            self._power_cache[cache_key] = ratings

        # Convert to DataFrame
        records = list(ratings.values())
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values('power_rating', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        return df

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        home_field_advantage: float = 2.5
    ) -> Dict:
        """
        Predict game outcome using power ratings.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Game week
            home_field_advantage: Points added for home team

        Returns:
            Dict with predictions (spread, total, win probability)
        """
        home_power = self.calculate_team_power(home_team, season, week)
        away_power = self.calculate_team_power(away_team, season, week)

        # Calculate spread from power differential
        # Power rating difference of 10 ≈ 3 points
        power_diff = home_power['power_rating'] - away_power['power_rating']
        raw_spread = power_diff * 0.3  # Convert to points

        # Add home field advantage
        fair_spread = raw_spread + home_field_advantage

        # Calculate total from expected PPG
        home_ppg = home_power['expected_ppg']
        away_ppg = away_power['expected_ppg']

        # Adjust for opponent defense
        home_adj_ppg = home_ppg - (away_power['def_epa'] * 30)
        away_adj_ppg = away_ppg - (home_power['def_epa'] * 30)

        fair_total = home_adj_ppg + away_adj_ppg

        # Win probability from Elo
        home_elo_adj = home_power['elo'] + (home_field_advantage * 25)  # 25 Elo per point
        away_elo_adj = away_power['elo']

        win_prob = 1 / (1 + 10 ** ((away_elo_adj - home_elo_adj) / 400))

        return {
            'home_team': home_team,
            'away_team': away_team,
            'fair_spread': round(fair_spread, 1),
            'fair_total': round(fair_total, 1),
            'home_win_prob': round(win_prob, 3),
            'home_power': home_power['power_rating'],
            'away_power': away_power['power_rating'],
            'home_expected_ppg': round(home_adj_ppg, 1),
            'away_expected_ppg': round(away_adj_ppg, 1),
            'power_diff': round(power_diff, 1),
            'home_elo': home_power['elo'],
            'away_elo': away_power['elo'],
            'confidence': self._calculate_confidence(home_power, away_power)
        }

    def _calculate_confidence(
        self,
        home_power: Dict,
        away_power: Dict
    ) -> str:
        """Calculate confidence level based on sample sizes."""
        min_games = min(home_power['sample_games'], away_power['sample_games'])

        if min_games >= 8:
            return 'HIGH'
        elif min_games >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def compare_to_vegas(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        vegas_spread: float,
        vegas_total: float
    ) -> Dict:
        """
        Compare model predictions to Vegas lines.

        Args:
            home_team: Home team
            away_team: Away team
            season: Season
            week: Week
            vegas_spread: Vegas spread (negative = home favored)
            vegas_total: Vegas total

        Returns:
            Dict with edges and recommendations
        """
        prediction = self.predict_game(home_team, away_team, season, week)

        # Calculate edges
        spread_edge = prediction['fair_spread'] - vegas_spread
        total_edge = prediction['fair_total'] - vegas_total

        # Determine recommendations
        spread_rec = None
        if abs(spread_edge) >= 2.0:  # 2+ point edge
            if spread_edge > 0:
                spread_rec = f"{home_team} +{abs(vegas_spread)}" if vegas_spread < 0 else f"{away_team} -{vegas_spread}"
            else:
                spread_rec = f"{home_team} {vegas_spread}" if vegas_spread < 0 else f"{away_team} +{abs(vegas_spread)}"

        total_rec = None
        if abs(total_edge) >= 3.0:  # 3+ point edge
            total_rec = 'OVER' if total_edge > 0 else 'UNDER'

        return {
            **prediction,
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
            'spread_edge': round(spread_edge, 1),
            'total_edge': round(total_edge, 1),
            'spread_edge_pct': round(spread_edge / max(abs(vegas_spread), 1) * 100, 1),
            'total_edge_pct': round(total_edge / vegas_total * 100, 1),
            'spread_recommendation': spread_rec,
            'total_recommendation': total_rec,
            'has_spread_edge': abs(spread_edge) >= 2.0,
            'has_total_edge': abs(total_edge) >= 3.0
        }

    def generate_power_rankings_report(
        self,
        season: int,
        week: int
    ) -> str:
        """Generate a formatted power rankings report."""
        df = self.get_all_team_power_ratings(season, week)

        lines = [
            f"{'='*60}",
            f"NFL POWER RANKINGS - Season {season}, Week {week}",
            f"{'='*60}",
            "",
            f"{'Rank':<5} {'Team':<5} {'Power':<8} {'Elo':<6} {'Net EPA':<9} {'SOS':<6}",
            f"{'-'*45}",
        ]

        for _, row in df.iterrows():
            lines.append(
                f"{int(row['rank']):<5} {row['team']:<5} {row['power_rating']:.1f}    "
                f"{int(row['elo']):<6} {row['net_epa']:+.3f}    {int(row['sos']):<6}"
            )

        lines.append("")
        lines.append("Legend:")
        lines.append("  Power: Combined rating (0-100)")
        lines.append("  Elo: FiveThirtyEight-style rating")
        lines.append("  Net EPA: Offensive EPA - Defensive EPA allowed")
        lines.append("  SOS: Strength of Schedule (avg opponent Elo)")

        return "\n".join(lines)


# Convenience function for quick access
def get_team_power_ratings(season: int, week: int) -> pd.DataFrame:
    """Get team power ratings DataFrame."""
    tpr = TeamPowerRatings()
    return tpr.get_all_team_power_ratings(season, week)


def predict_game(
    home_team: str,
    away_team: str,
    season: int,
    week: int
) -> Dict:
    """Quick game prediction."""
    tpr = TeamPowerRatings()
    return tpr.predict_game(home_team, away_team, season, week)
