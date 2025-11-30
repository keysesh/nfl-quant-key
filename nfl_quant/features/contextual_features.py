"""
Contextual Features Module - Rest, Travel, Bye, Division, etc.

Research-backed contextual adjustments with small, data-driven priors:
1. Rest days (short rest, bye week, extra rest)
2. Travel distance and time zones
3. Divisional games
4. Primetime games (SNF, MNF, TNF)
5. Altitude
6. Home field advantage

Key Research Findings:
- Short rest (TNF, <6 days): -2% to -3% EPA, increased injuries
- Bye week rest: +1% to +2% EPA (slight refresh benefit)
- Cross-country travel (3+ time zones): -1% to -2% EPA
- Divisional games: -0.5 to -1 point in total (familiarity, lower variance)
- Primetime: No consistent effect (narrative bias)
- Altitude (Denver): -1% to -2% passing EPA for visiting teams
- Home field advantage: +1 to +2.5 points (team-varying)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ContextualConfig:
    """Configuration for contextual adjustments."""

    # Rest adjustments (EPA multipliers)
    short_rest_epa_penalty: float = -0.03  # -3% for <6 days rest
    thursday_night_epa_penalty: float = -0.025  # -2.5% for TNF specifically
    bye_week_epa_boost: float = 0.015  # +1.5% coming off bye
    extra_rest_epa_boost: float = 0.01  # +1% for 8+ days rest

    # Travel adjustments
    # Research: minimal effect for short travel, -1 to -2% for coast-to-coast
    travel_distance_threshold_miles: int = 1500  # >1500 miles = "long travel"
    long_travel_epa_penalty: float = -0.015  # -1.5%

    timezone_change_threshold: int = 2  # ≥2 time zones = jet lag
    timezone_epa_penalty_per_zone: float = -0.005  # -0.5% per time zone

    # Divisional game adjustments
    divisional_total_adjustment: float = -0.5  # -0.5 points (lower variance)
    divisional_spread_tightening: float = 0.80  # Spread becomes 80% of normal (closer games)

    # Primetime adjustments (minimal - mostly narrative)
    sunday_night_adjustment: float = 0.0  # No effect
    monday_night_adjustment: float = 0.0  # No effect
    thursday_night_adjustment: float = -0.025  # Same as short rest

    # Altitude adjustment (Denver-specific)
    altitude_threshold_feet: int = 4000  # >4000 ft = "high altitude"
    altitude_visiting_epa_penalty: float = -0.015  # -1.5% for visitors

    # Home field advantage (team-varying, but use league average as prior)
    hfa_points_average: float = 1.5  # Average HFA ~1.5 points
    hfa_range: Tuple[float, float] = (0.5, 2.5)  # Min/max HFA

    # Confidence intervals (for small sample warnings)
    min_games_for_team_hfa: int = 10  # Need 10+ games to estimate team-specific HFA


class ContextualFeatureEngine:
    """
    Calculates contextual feature adjustments for NFL games.

    Uses research-backed priors with conservative effect sizes.
    """

    def __init__(self, config: Optional[ContextualConfig] = None):
        """
        Initialize contextual feature engine.

        Args:
            config: Contextual configuration (uses defaults if None)
        """
        self.config = config or ContextualConfig()
        self.team_locations = self._load_team_locations()
        self.team_stadiums = self._load_stadium_info()

    def _load_team_locations(self) -> Dict[str, Dict[str, float]]:
        """Load team locations (lat, lon, timezone)."""
        return {
            'ARI': {'lat': 33.5276, 'lon': -112.2626, 'tz': 'America/Phoenix', 'tz_offset': -7},
            'ATL': {'lat': 33.7555, 'lon': -84.4009, 'tz': 'America/New_York', 'tz_offset': -5},
            'BAL': {'lat': 39.2780, 'lon': -76.6227, 'tz': 'America/New_York', 'tz_offset': -5},
            'BUF': {'lat': 42.7738, 'lon': -78.7870, 'tz': 'America/New_York', 'tz_offset': -5},
            'CAR': {'lat': 35.2258, 'lon': -80.8528, 'tz': 'America/New_York', 'tz_offset': -5},
            'CHI': {'lat': 41.8623, 'lon': -87.6167, 'tz': 'America/Chicago', 'tz_offset': -6},
            'CIN': {'lat': 39.0954, 'lon': -84.5161, 'tz': 'America/New_York', 'tz_offset': -5},
            'CLE': {'lat': 41.5061, 'lon': -81.6995, 'tz': 'America/New_York', 'tz_offset': -5},
            'DAL': {'lat': 32.7480, 'lon': -97.0934, 'tz': 'America/Chicago', 'tz_offset': -6},
            'DEN': {'lat': 39.7439, 'lon': -105.0201, 'tz': 'America/Denver', 'tz_offset': -7},
            'DET': {'lat': 42.3400, 'lon': -83.0456, 'tz': 'America/New_York', 'tz_offset': -5},
            'GB': {'lat': 44.5013, 'lon': -88.0622, 'tz': 'America/Chicago', 'tz_offset': -6},
            'HOU': {'lat': 29.6847, 'lon': -95.4107, 'tz': 'America/Chicago', 'tz_offset': -6},
            'IND': {'lat': 39.7601, 'lon': -86.1639, 'tz': 'America/New_York', 'tz_offset': -5},
            'JAX': {'lat': 30.3240, 'lon': -81.6373, 'tz': 'America/New_York', 'tz_offset': -5},
            'KC': {'lat': 39.0489, 'lon': -94.4839, 'tz': 'America/Chicago', 'tz_offset': -6},
            'LAC': {'lat': 33.8636, 'lon': -118.2390, 'tz': 'America/Los_Angeles', 'tz_offset': -8},
            'LAR': {'lat': 33.9535, 'lon': -118.3390, 'tz': 'America/Los_Angeles', 'tz_offset': -8},
            'LV': {'lat': 36.0909, 'lon': -115.1833, 'tz': 'America/Los_Angeles', 'tz_offset': -8},
            'MIA': {'lat': 25.9580, 'lon': -80.2389, 'tz': 'America/New_York', 'tz_offset': -5},
            'MIN': {'lat': 44.9738, 'lon': -93.2577, 'tz': 'America/Chicago', 'tz_offset': -6},
            'NE': {'lat': 42.0909, 'lon': -71.2643, 'tz': 'America/New_York', 'tz_offset': -5},
            'NO': {'lat': 29.9511, 'lon': -90.0812, 'tz': 'America/Chicago', 'tz_offset': -6},
            'NYG': {'lat': 40.8135, 'lon': -74.0745, 'tz': 'America/New_York', 'tz_offset': -5},
            'NYJ': {'lat': 40.8135, 'lon': -74.0745, 'tz': 'America/New_York', 'tz_offset': -5},
            'PHI': {'lat': 39.9008, 'lon': -75.1675, 'tz': 'America/New_York', 'tz_offset': -5},
            'PIT': {'lat': 40.4468, 'lon': -80.0158, 'tz': 'America/New_York', 'tz_offset': -5},
            'SEA': {'lat': 47.5952, 'lon': -122.3316, 'tz': 'America/Los_Angeles', 'tz_offset': -8},
            'SF': {'lat': 37.4032, 'lon': -121.9698, 'tz': 'America/Los_Angeles', 'tz_offset': -8},
            'TB': {'lat': 27.9759, 'lon': -82.5033, 'tz': 'America/New_York', 'tz_offset': -5},
            'TEN': {'lat': 36.1665, 'lon': -86.7713, 'tz': 'America/Chicago', 'tz_offset': -6},
            'WAS': {'lat': 38.9076, 'lon': -76.8645, 'tz': 'America/New_York', 'tz_offset': -5},
        }

    def _load_stadium_info(self) -> Dict[str, Dict[str, any]]:
        """Load stadium info (elevation, etc.)."""
        return {
            'DEN': {'elevation_feet': 5280, 'name': 'Empower Field'},  # Mile High
            'ARI': {'elevation_feet': 1132, 'name': 'State Farm Stadium'},
            'LV': {'elevation_feet': 2001, 'name': 'Allegiant Stadium'},
            # Most other stadiums are near sea level (<500 ft)
        }

    def calculate_rest_adjustment(
        self,
        team: str,
        days_since_last_game: int,
        is_coming_off_bye: bool = False
    ) -> Dict[str, float]:
        """
        Calculate rest-based EPA adjustment.

        Args:
            team: Team abbreviation
            days_since_last_game: Days since last game
            is_coming_off_bye: Whether team is coming off bye week

        Returns:
            Dictionary with:
                - epa_multiplier: float
                - rest_category: str
                - adjustment_pct: float (percentage)
        """
        if is_coming_off_bye:
            return {
                'epa_multiplier': 1.0 + self.config.bye_week_epa_boost,
                'rest_category': 'bye_week',
                'adjustment_pct': self.config.bye_week_epa_boost * 100
            }

        if days_since_last_game < 6:
            # Short rest (Thursday night)
            return {
                'epa_multiplier': 1.0 + self.config.short_rest_epa_penalty,
                'rest_category': 'short_rest',
                'adjustment_pct': self.config.short_rest_epa_penalty * 100
            }

        if days_since_last_game >= 8:
            # Extra rest
            return {
                'epa_multiplier': 1.0 + self.config.extra_rest_epa_boost,
                'rest_category': 'extra_rest',
                'adjustment_pct': self.config.extra_rest_epa_boost * 100
            }

        # Normal rest (7 days)
        return {
            'epa_multiplier': 1.0,
            'rest_category': 'normal',
            'adjustment_pct': 0.0
        }

    def calculate_travel_adjustment(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Calculate travel-based EPA adjustment for away team.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dictionary with:
                - epa_multiplier: float (for away team)
                - travel_distance_miles: float
                - timezone_change: int
                - adjustment_pct: float
        """
        # Get locations
        home_loc = self.team_locations.get(home_team, {})
        away_loc = self.team_locations.get(away_team, {})

        if not home_loc or not away_loc:
            return {
                'epa_multiplier': 1.0,
                'travel_distance_miles': 0.0,
                'timezone_change': 0,
                'adjustment_pct': 0.0
            }

        # Calculate distance (Haversine formula)
        distance_miles = self._calculate_distance(
            away_loc['lat'], away_loc['lon'],
            home_loc['lat'], home_loc['lon']
        )

        # Calculate timezone change
        tz_change = abs(home_loc['tz_offset'] - away_loc['tz_offset'])

        # Calculate adjustment
        adjustment = 0.0

        # Long travel penalty
        if distance_miles > self.config.travel_distance_threshold_miles:
            adjustment += self.config.long_travel_epa_penalty

        # Timezone penalty
        if tz_change >= self.config.timezone_change_threshold:
            adjustment += self.config.timezone_epa_penalty_per_zone * tz_change

        return {
            'epa_multiplier': 1.0 + adjustment,
            'travel_distance_miles': distance_miles,
            'timezone_change': tz_change,
            'adjustment_pct': adjustment * 100
        }

    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points (Haversine formula)."""
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in miles
        radius_miles = 3959.0

        return radius_miles * c

    def calculate_divisional_adjustment(
        self,
        home_team: str,
        away_team: str,
        is_divisional: bool
    ) -> Dict[str, float]:
        """
        Calculate divisional game adjustments.

        Args:
            home_team: Home team
            away_team: Away team
            is_divisional: Whether this is a divisional matchup

        Returns:
            Dictionary with:
                - total_adjustment: float (points)
                - spread_multiplier: float
                - is_divisional: bool
        """
        if not is_divisional:
            return {
                'total_adjustment': 0.0,
                'spread_multiplier': 1.0,
                'is_divisional': False
            }

        return {
            'total_adjustment': self.config.divisional_total_adjustment,
            'spread_multiplier': self.config.divisional_spread_tightening,
            'is_divisional': True
        }

    def calculate_altitude_adjustment(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Calculate altitude adjustment for visiting team.

        Args:
            home_team: Home team
            away_team: Away team

        Returns:
            Dictionary with:
                - epa_multiplier: float (for away team)
                - elevation_feet: int
                - is_high_altitude: bool
        """
        stadium_info = self.team_stadiums.get(home_team, {})
        elevation = stadium_info.get('elevation_feet', 0)

        if elevation < self.config.altitude_threshold_feet:
            return {
                'epa_multiplier': 1.0,
                'elevation_feet': elevation,
                'is_high_altitude': False
            }

        # Visiting team penalty for high altitude (Denver)
        return {
            'epa_multiplier': 1.0 + self.config.altitude_visiting_epa_penalty,
            'elevation_feet': elevation,
            'is_high_altitude': True
        }

    def calculate_all_contextual_adjustments(
        self,
        home_team: str,
        away_team: str,
        home_days_rest: int,
        away_days_rest: int,
        is_divisional: bool,
        home_coming_off_bye: bool = False,
        away_coming_off_bye: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all contextual adjustments for a game.

        Args:
            home_team: Home team
            away_team: Away team
            home_days_rest: Days since home team's last game
            away_days_rest: Days since away team's last game
            is_divisional: Whether divisional matchup
            home_coming_off_bye: Home team off bye
            away_coming_off_bye: Away team off bye

        Returns:
            Dictionary with 'home' and 'away' team adjustments
        """
        # Home team adjustments
        home_rest = self.calculate_rest_adjustment(
            team=home_team,
            days_since_last_game=home_days_rest,
            is_coming_off_bye=home_coming_off_bye
        )

        home_altitude = {'epa_multiplier': 1.0, 'elevation_feet': 0, 'is_high_altitude': False}  # No altitude penalty for home

        # Away team adjustments
        away_rest = self.calculate_rest_adjustment(
            team=away_team,
            days_since_last_game=away_days_rest,
            is_coming_off_bye=away_coming_off_bye
        )

        away_travel = self.calculate_travel_adjustment(
            home_team=home_team,
            away_team=away_team
        )

        away_altitude = self.calculate_altitude_adjustment(
            home_team=home_team,
            away_team=away_team
        )

        # Game-level adjustments
        divisional = self.calculate_divisional_adjustment(
            home_team=home_team,
            away_team=away_team,
            is_divisional=is_divisional
        )

        # Combine multiplicative adjustments
        home_combined_multiplier = home_rest['epa_multiplier']
        away_combined_multiplier = (
            away_rest['epa_multiplier'] *
            away_travel['epa_multiplier'] *
            away_altitude['epa_multiplier']
        )

        return {
            'home': {
                'epa_multiplier': home_combined_multiplier,
                'rest': home_rest,
                'altitude': home_altitude,
            },
            'away': {
                'epa_multiplier': away_combined_multiplier,
                'rest': away_rest,
                'travel': away_travel,
                'altitude': away_altitude,
            },
            'game': {
                'divisional': divisional,
                'total_adjustment': divisional['total_adjustment']
            }
        }


# Example usage and testing
if __name__ == '__main__':
    # Initialize engine
    engine = ContextualFeatureEngine()

    print("=== Contextual Feature Adjustments ===\n")

    # Example 1: Thursday Night Football
    print("1. Thursday Night Football (Short Rest)")
    rest_adj = engine.calculate_rest_adjustment(
        team='KC',
        days_since_last_game=4,
        is_coming_off_bye=False
    )
    print(f"  EPA multiplier: {rest_adj['epa_multiplier']:.4f}")
    print(f"  Rest category: {rest_adj['rest_category']}")
    print(f"  Adjustment: {rest_adj['adjustment_pct']:+.1f}%\n")

    # Example 2: Coming off bye
    print("2. Coming Off Bye Week")
    bye_adj = engine.calculate_rest_adjustment(
        team='SF',
        days_since_last_game=14,
        is_coming_off_bye=True
    )
    print(f"  EPA multiplier: {bye_adj['epa_multiplier']:.4f}")
    print(f"  Adjustment: {bye_adj['adjustment_pct']:+.1f}%\n")

    # Example 3: Cross-country travel
    print("3. Cross-Country Travel (NE @ LAR)")
    travel_adj = engine.calculate_travel_adjustment(
        home_team='LAR',
        away_team='NE'
    )
    print(f"  Travel distance: {travel_adj['travel_distance_miles']:.0f} miles")
    print(f"  Timezone change: {travel_adj['timezone_change']} zones")
    print(f"  EPA multiplier: {travel_adj['epa_multiplier']:.4f}")
    print(f"  Adjustment: {travel_adj['adjustment_pct']:+.1f}%\n")

    # Example 4: Denver altitude
    print("4. Altitude Adjustment (MIA @ DEN)")
    altitude_adj = engine.calculate_altitude_adjustment(
        home_team='DEN',
        away_team='MIA'
    )
    print(f"  Elevation: {altitude_adj['elevation_feet']:,} ft")
    print(f"  High altitude: {altitude_adj['is_high_altitude']}")
    print(f"  EPA multiplier: {altitude_adj['epa_multiplier']:.4f}\n")

    # Example 5: Divisional game
    print("5. Divisional Game (KC @ DEN)")
    div_adj = engine.calculate_divisional_adjustment(
        home_team='DEN',
        away_team='KC',
        is_divisional=True
    )
    print(f"  Is divisional: {div_adj['is_divisional']}")
    print(f"  Total adjustment: {div_adj['total_adjustment']:+.1f} points")
    print(f"  Spread multiplier: {div_adj['spread_multiplier']:.2f}\n")

    # Example 6: Full game context
    print("6. Full Game Context (SEA @ NE, TNF, Divisional=False)")
    full_adj = engine.calculate_all_contextual_adjustments(
        home_team='NE',
        away_team='SEA',
        home_days_rest=4,  # TNF
        away_days_rest=4,
        is_divisional=False,
        home_coming_off_bye=False,
        away_coming_off_bye=False
    )
    print(f"  Home (NE) EPA multiplier: {full_adj['home']['epa_multiplier']:.4f}")
    print(f"  Away (SEA) EPA multiplier: {full_adj['away']['epa_multiplier']:.4f}")
    print(f"  Away travel: {full_adj['away']['travel']['travel_distance_miles']:.0f} miles")
    print(f"  Game total adjustment: {full_adj['game']['total_adjustment']:+.1f} points")
