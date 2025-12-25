"""
Enhanced Weather Features Module V2 - Research-Backed Wind Buckets
====================================================================

Improvements over V1:
1. Granular wind buckets (0-10, 10-15, 15-20, 20+ mph) with empirical impacts
2. Deep target share adjustments (wind depresses deep passing)
3. CPOE (Completion % Over Expected) adjustments by wind bucket
4. Interaction effects (wind + cold, wind + precipitation)
5. Dome advantage quantified vs outdoor

Research-Backed Impacts (from Advanced Football Analytics):
- Wind 10-15 mph: -3% passing EPA, -5% deep target share (20+ air yards)
- Wind 15-20 mph: -8% passing EPA, -12% deep target share, +3% rush rate
- Wind 20+ mph: -15% passing EPA, -25% deep target share, +8% rush rate
- Cold (<25°F) + Wind (15+ mph): Additional -5% passing EPA (stacking)
- Dome advantage: +2% completion%, +0.3 yards/attempt vs outdoor

Temperature Effects:
- Extreme cold (<25°F): -6% passing EPA, -3% completion%
- Cold (25-32°F): -4% passing EPA, -2% completion%
- Comfortable (45-75°F): Baseline (no adjustment)
- Extreme heat (>90°F): -2% passing EPA (rare, mostly fatigue)

Precipitation Effects:
- Light rain/snow: -2% completion%, +2% fumble rate
- Moderate: -5% completion%, +4% fumble rate
- Heavy: -8% completion%, +6% fumble rate, -10% deep targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindBucket:
    """Wind speed bucket with associated impacts."""
    name: str
    min_mph: float
    max_mph: float
    passing_epa_multiplier: float
    completion_pct_adjustment: float  # Percentage points
    deep_target_share_multiplier: float  # % of targets 20+ air yards
    rush_rate_boost: float  # Percentage point increase


@dataclass
class TemperatureBucket:
    """Temperature bucket with impacts."""
    name: str
    min_temp: float
    max_temp: float
    passing_epa_multiplier: float
    completion_pct_adjustment: float


@dataclass
class PrecipitationLevel:
    """Precipitation level with impacts."""
    name: str
    passing_epa_multiplier: float  # NEW: direct impact on passing volume/efficiency
    completion_pct_adjustment: float
    fumble_rate_multiplier: float
    deep_target_multiplier: float
    rush_boost: float  # NEW: teams run more in bad weather


class WeatherAdjusterV2:
    """
    Enhanced weather adjuster with research-backed wind buckets.
    """

    # Wind buckets (research-backed from Advanced Football Analytics)
    WIND_BUCKETS = [
        WindBucket(
            name='calm',
            min_mph=0.0,
            max_mph=10.0,
            passing_epa_multiplier=1.00,
            completion_pct_adjustment=0.0,
            deep_target_share_multiplier=1.00,
            rush_rate_boost=0.00
        ),
        WindBucket(
            name='moderate',
            min_mph=10.0,
            max_mph=15.0,
            passing_epa_multiplier=0.97,  # -3%
            completion_pct_adjustment=-1.0,  # -1 pct point
            deep_target_share_multiplier=0.95,  # -5%
            rush_rate_boost=0.02  # +2 pct points
        ),
        WindBucket(
            name='high',
            min_mph=15.0,
            max_mph=20.0,
            passing_epa_multiplier=0.92,  # -8%
            completion_pct_adjustment=-2.5,  # -2.5 pct points
            deep_target_share_multiplier=0.88,  # -12%
            rush_rate_boost=0.05  # +5 pct points
        ),
        WindBucket(
            name='extreme',
            min_mph=20.0,
            max_mph=100.0,  # No upper limit
            passing_epa_multiplier=0.85,  # -15%
            completion_pct_adjustment=-4.0,  # -4 pct points
            deep_target_share_multiplier=0.75,  # -25%
            rush_rate_boost=0.08  # +8 pct points
        ),
    ]

    # Temperature buckets
    TEMP_BUCKETS = [
        TemperatureBucket(
            name='extreme_cold',
            min_temp=-50.0,
            max_temp=25.0,
            passing_epa_multiplier=0.94,  # -6%
            completion_pct_adjustment=-3.0
        ),
        TemperatureBucket(
            name='cold',
            min_temp=25.0,
            max_temp=32.0,
            passing_epa_multiplier=0.96,  # -4%
            completion_pct_adjustment=-2.0
        ),
        TemperatureBucket(
            name='cool',
            min_temp=32.0,
            max_temp=45.0,
            passing_epa_multiplier=0.98,  # -2%
            completion_pct_adjustment=-1.0
        ),
        TemperatureBucket(
            name='comfortable',
            min_temp=45.0,
            max_temp=75.0,
            passing_epa_multiplier=1.00,  # Baseline
            completion_pct_adjustment=0.0
        ),
        TemperatureBucket(
            name='warm',
            min_temp=75.0,
            max_temp=90.0,
            passing_epa_multiplier=0.99,  # -1% (slight)
            completion_pct_adjustment=0.0
        ),
        TemperatureBucket(
            name='extreme_heat',
            min_temp=90.0,
            max_temp=150.0,
            passing_epa_multiplier=0.98,  # -2% (fatigue)
            completion_pct_adjustment=-1.0
        ),
    ]

    # Precipitation levels - RAIN
    PRECIP_LEVELS = {
        'none': PrecipitationLevel(
            name='none',
            passing_epa_multiplier=1.00,
            completion_pct_adjustment=0.0,
            fumble_rate_multiplier=1.00,
            deep_target_multiplier=1.00,
            rush_boost=0.00
        ),
        'light': PrecipitationLevel(
            name='light',
            passing_epa_multiplier=0.97,  # -3% passing
            completion_pct_adjustment=-2.0,
            fumble_rate_multiplier=1.03,
            deep_target_multiplier=0.95,
            rush_boost=0.02  # +2% rush rate
        ),
        'moderate': PrecipitationLevel(
            name='moderate',
            passing_epa_multiplier=0.94,  # -6% passing
            completion_pct_adjustment=-5.0,
            fumble_rate_multiplier=1.05,
            deep_target_multiplier=0.85,
            rush_boost=0.05  # +5% rush rate
        ),
        'heavy': PrecipitationLevel(
            name='heavy',
            passing_epa_multiplier=0.90,  # -10% passing
            completion_pct_adjustment=-8.0,
            fumble_rate_multiplier=1.08,
            deep_target_multiplier=0.75,
            rush_boost=0.08  # +8% rush rate
        ),
    }

    # Precipitation levels - SNOW (worse than rain - visibility, grip, footing)
    SNOW_LEVELS = {
        'none': PrecipitationLevel(
            name='none',
            passing_epa_multiplier=1.00,
            completion_pct_adjustment=0.0,
            fumble_rate_multiplier=1.00,
            deep_target_multiplier=1.00,
            rush_boost=0.00
        ),
        'light': PrecipitationLevel(
            name='light_snow',
            passing_epa_multiplier=0.94,  # -6% passing (worse than light rain)
            completion_pct_adjustment=-3.0,
            fumble_rate_multiplier=1.04,
            deep_target_multiplier=0.90,
            rush_boost=0.04  # +4% rush rate
        ),
        'moderate': PrecipitationLevel(
            name='moderate_snow',
            passing_epa_multiplier=0.88,  # -12% passing
            completion_pct_adjustment=-7.0,
            fumble_rate_multiplier=1.08,
            deep_target_multiplier=0.75,
            rush_boost=0.08  # +8% rush rate
        ),
        'heavy': PrecipitationLevel(
            name='heavy_snow',
            passing_epa_multiplier=0.80,  # -20% passing (major impact)
            completion_pct_adjustment=-12.0,
            fumble_rate_multiplier=1.12,
            deep_target_multiplier=0.60,
            rush_boost=0.12  # +12% rush rate
        ),
    }

    # Dome advantage
    DOME_ADVANTAGE = {
        'completion_pct_boost': 2.0,  # +2 pct points
        'yards_per_attempt_boost': 0.3,  # +0.3 yards
        'passing_epa_multiplier': 1.03  # +3%
    }

    def __init__(self):
        """Initialize enhanced weather adjuster."""
        self.stadium_roofs = self._load_stadium_roofs()

    def _load_stadium_roofs(self) -> Dict[str, str]:
        """Load stadium roof types."""
        return {
            'ARI': 'retractable',
            'ATL': 'retractable',
            'BAL': 'outdoor',
            'BUF': 'outdoor',
            'CAR': 'outdoor',
            'CHI': 'outdoor',
            'CIN': 'outdoor',
            'CLE': 'outdoor',
            'DAL': 'retractable',
            'DEN': 'outdoor',
            'DET': 'dome',
            'GB': 'outdoor',
            'HOU': 'retractable',
            'IND': 'dome',
            'JAX': 'outdoor',
            'KC': 'outdoor',
            'LAC': 'outdoor',
            'LAR': 'dome',
            'LV': 'dome',
            'MIA': 'outdoor',
            'MIN': 'dome',
            'NE': 'outdoor',
            'NO': 'dome',
            'NYG': 'outdoor',
            'NYJ': 'outdoor',
            'PHI': 'outdoor',
            'PIT': 'outdoor',
            'SEA': 'outdoor',
            'SF': 'outdoor',
            'TB': 'outdoor',
            'TEN': 'outdoor',
            'WAS': 'outdoor',
        }

    def get_wind_bucket(self, wind_mph: float) -> WindBucket:
        """
        Get wind bucket for given wind speed.

        Args:
            wind_mph: Wind speed in mph

        Returns:
            WindBucket object
        """
        for bucket in self.WIND_BUCKETS:
            if bucket.min_mph <= wind_mph < bucket.max_mph:
                return bucket

        # Default to extreme if > 20 mph
        return self.WIND_BUCKETS[-1]

    def get_temperature_bucket(self, temp_f: float) -> TemperatureBucket:
        """
        Get temperature bucket for given temperature.

        Args:
            temp_f: Temperature in Fahrenheit

        Returns:
            TemperatureBucket object
        """
        for bucket in self.TEMP_BUCKETS:
            if bucket.min_temp <= temp_f < bucket.max_temp:
                return bucket

        # Default to comfortable if out of range
        return self.TEMP_BUCKETS[3]

    def get_precipitation_level(
        self,
        precip_prob: float,
        precip_intensity: Optional[str] = None,
        precip_type: Optional[str] = None
    ) -> PrecipitationLevel:
        """
        Get precipitation level.

        Args:
            precip_prob: Probability of precipitation (0-1)
            precip_intensity: Optional intensity ('light', 'moderate', 'heavy')
            precip_type: Optional type ('rain', 'snow', None)

        Returns:
            PrecipitationLevel object
        """
        if precip_prob < 0.20:
            return self.PRECIP_LEVELS['none']

        # Choose the right table (snow is worse than rain)
        is_snow = precip_type and 'snow' in precip_type.lower()
        precip_table = self.SNOW_LEVELS if is_snow else self.PRECIP_LEVELS

        if precip_intensity:
            return precip_table.get(precip_intensity, precip_table['light'])

        # Infer from probability
        if precip_prob < 0.50:
            return precip_table['light']
        elif precip_prob < 0.75:
            return precip_table['moderate']
        else:
            return precip_table['heavy']

    def calculate_weather_adjustments(
        self,
        team: str,
        wind_mph: float = 0.0,
        temp_f: float = 65.0,
        precip_prob: float = 0.0,
        precip_intensity: Optional[str] = None,
        precip_type: Optional[str] = None,
        is_dome: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive weather adjustments.

        Args:
            team: Team abbreviation
            wind_mph: Wind speed in mph
            temp_f: Temperature in Fahrenheit
            precip_prob: Precipitation probability (0-1)
            precip_intensity: Optional precipitation intensity ('light', 'moderate', 'heavy')
            precip_type: Optional precipitation type ('rain', 'snow')
            is_dome: Override for dome status (None = use stadium data)

        Returns:
            Dictionary of adjustment multipliers:
                - passing_epa_multiplier
                - completion_pct_adjustment (percentage points)
                - deep_target_share_multiplier
                - rush_rate_boost (percentage points)
                - yards_per_attempt_adjustment
                - fumble_rate_multiplier
        """
        # Determine if dome game
        if is_dome is None:
            roof_type = self.stadium_roofs.get(team, 'outdoor')
            is_dome = roof_type in ['dome', 'retractable']
        else:
            is_dome = is_dome

        # If dome, minimal weather impact
        if is_dome:
            return {
                'passing_epa_multiplier': self.DOME_ADVANTAGE['passing_epa_multiplier'],
                'completion_pct_adjustment': self.DOME_ADVANTAGE['completion_pct_boost'],
                'deep_target_share_multiplier': 1.00,
                'rush_rate_boost': 0.00,
                'yards_per_attempt_adjustment': self.DOME_ADVANTAGE['yards_per_attempt_boost'],
                'fumble_rate_multiplier': 1.00,
                'is_dome': True,
                'wind_bucket': 'dome',
                'temp_bucket': 'comfortable',
                'precip_level': 'none'
            }

        # Get wind bucket
        wind_bucket = self.get_wind_bucket(wind_mph)

        # Get temperature bucket
        temp_bucket = self.get_temperature_bucket(temp_f)

        # Get precipitation level (snow vs rain matters now!)
        precip_level = self.get_precipitation_level(precip_prob, precip_intensity, precip_type)

        # Base adjustments - NOW INCLUDES PRECIPITATION EPA IMPACT
        passing_epa_mult = (
            wind_bucket.passing_epa_multiplier *
            temp_bucket.passing_epa_multiplier *
            precip_level.passing_epa_multiplier  # NEW: precip directly impacts passing
        )
        completion_pct_adj = (
            wind_bucket.completion_pct_adjustment +
            temp_bucket.completion_pct_adjustment +
            precip_level.completion_pct_adjustment
        )
        deep_target_mult = (
            wind_bucket.deep_target_share_multiplier *
            precip_level.deep_target_multiplier
        )
        # Combine wind and precip rush boosts
        rush_rate_boost = wind_bucket.rush_rate_boost + precip_level.rush_boost

        # Interaction effect: Wind + Cold (stacks)
        if wind_mph >= 15.0 and temp_f < 25.0:
            passing_epa_mult *= 0.95  # Additional -5%
            completion_pct_adj -= 1.0  # Additional -1 pct point

        # Interaction effect: Wind + Precipitation (stacks)
        if wind_mph >= 15.0 and precip_prob > 0.5:
            passing_epa_mult *= 0.97  # Additional -3%
            deep_target_mult *= 0.90  # Additional -10% deep targets

        # Interaction effect: Snow + Cold (extra brutal)
        is_snow = precip_type and 'snow' in precip_type.lower()
        if is_snow and temp_f < 25.0 and precip_prob > 0.5:
            passing_epa_mult *= 0.95  # Additional -5% for snow + extreme cold
            rush_rate_boost += 0.05  # +5% more rushing

        # Yards per attempt adjustment (derived from EPA and completion%)
        # Simplified: EPA impact translates roughly 1:1 to yards/attempt
        yards_per_attempt_adj = (passing_epa_mult - 1.0) * 11.0  # ~11 yards baseline

        return {
            'passing_epa_multiplier': passing_epa_mult,
            'completion_pct_adjustment': completion_pct_adj,
            'deep_target_share_multiplier': deep_target_mult,
            'rush_rate_boost': rush_rate_boost,
            'yards_per_attempt_adjustment': yards_per_attempt_adj,
            'fumble_rate_multiplier': precip_level.fumble_rate_multiplier,
            'is_dome': False,
            'wind_bucket': wind_bucket.name,
            'temp_bucket': temp_bucket.name,
            'precip_level': precip_level.name
        }

    def apply_to_player_projection(
        self,
        position: str,
        base_projection: Dict[str, float],
        weather_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply weather adjustments to player projection.

        Args:
            position: Player position (QB, RB, WR, TE)
            base_projection: Base projection dict (yards, attempts, etc.)
            weather_adjustments: Weather adjustments from calculate_weather_adjustments

        Returns:
            Adjusted projection dict
        """
        adjusted = base_projection.copy()

        # QB adjustments
        if position == 'QB':
            # Passing yards
            if 'passing_yards' in adjusted:
                adjusted['passing_yards'] *= weather_adjustments['passing_epa_multiplier']

            # Completion percentage
            if 'completion_pct' in adjusted:
                adjusted['completion_pct'] = min(
                    0.95,
                    adjusted['completion_pct'] + (weather_adjustments['completion_pct_adjustment'] / 100.0)
                )

            # Yards per attempt
            if 'yards_per_attempt' in adjusted:
                adjusted['yards_per_attempt'] += weather_adjustments['yards_per_attempt_adjustment']

            # Rushing (slight benefit from bad weather)
            if 'rushing_yards' in adjusted:
                rush_multiplier = 1.0 + (weather_adjustments['rush_rate_boost'] * 0.5)
                adjusted['rushing_yards'] *= rush_multiplier

        # RB adjustments
        elif position == 'RB':
            # Rushing benefits from wind
            if 'rushing_yards' in adjusted:
                rush_multiplier = 1.0 + (weather_adjustments['rush_rate_boost'] * 0.3)
                adjusted['rushing_yards'] *= rush_multiplier

            # Receiving hurt by wind
            if 'receiving_yards' in adjusted:
                adjusted['receiving_yards'] *= weather_adjustments['passing_epa_multiplier']

            # Fumbles
            if 'fumble_rate' in adjusted:
                adjusted['fumble_rate'] *= weather_adjustments['fumble_rate_multiplier']

        # WR/TE adjustments
        elif position in ['WR', 'TE']:
            # Receiving yards
            if 'receiving_yards' in adjusted:
                adjusted['receiving_yards'] *= weather_adjustments['passing_epa_multiplier']

            # Deep targets (WR-specific)
            if position == 'WR' and 'deep_target_share' in adjusted:
                adjusted['deep_target_share'] *= weather_adjustments['deep_target_share_multiplier']

            # Receptions (completion% impact)
            if 'receptions' in adjusted and 'targets' in adjusted:
                base_catch_rate = adjusted['receptions'] / adjusted['targets'] if adjusted['targets'] > 0 else 0.65
                adj_catch_rate = base_catch_rate + (weather_adjustments['completion_pct_adjustment'] / 100.0)
                adj_catch_rate = np.clip(adj_catch_rate, 0.30, 0.95)
                adjusted['receptions'] = adjusted['targets'] * adj_catch_rate

        return adjusted


# Example usage and testing
if __name__ == '__main__':
    # Initialize adjuster
    adjuster = WeatherAdjusterV2()

    print("=== Enhanced Weather Adjustments V2 ===\n")

    # Example 1: Calm dome game
    print("1. Dome Game (LAR)")
    adjustments = adjuster.calculate_weather_adjustments(
        team='LAR',
        wind_mph=0.0,
        temp_f=72.0,
        precip_prob=0.0
    )
    print(f"  Passing EPA multiplier: {adjustments['passing_epa_multiplier']:.3f}")
    print(f"  Completion % adjustment: {adjustments['completion_pct_adjustment']:+.1f} pct points")
    print(f"  Is dome: {adjustments['is_dome']}\n")

    # Example 2: Moderate wind outdoor
    print("2. Moderate Wind (GB, 12 mph wind, 45°F)")
    adjustments = adjuster.calculate_weather_adjustments(
        team='GB',
        wind_mph=12.0,
        temp_f=45.0,
        precip_prob=0.0
    )
    print(f"  Wind bucket: {adjustments['wind_bucket']}")
    print(f"  Passing EPA multiplier: {adjustments['passing_epa_multiplier']:.3f}")
    print(f"  Completion % adjustment: {adjustments['completion_pct_adjustment']:+.1f} pct points")
    print(f"  Deep target share multiplier: {adjustments['deep_target_share_multiplier']:.3f}")
    print(f"  Rush rate boost: {adjustments['rush_rate_boost']:+.1%}\n")

    # Example 3: Extreme wind + cold (Buffalo)
    print("3. Extreme Wind + Cold (BUF, 22 mph wind, 18°F)")
    adjustments = adjuster.calculate_weather_adjustments(
        team='BUF',
        wind_mph=22.0,
        temp_f=18.0,
        precip_prob=0.0
    )
    print(f"  Wind bucket: {adjustments['wind_bucket']}")
    print(f"  Temp bucket: {adjustments['temp_bucket']}")
    print(f"  Passing EPA multiplier: {adjustments['passing_epa_multiplier']:.3f}")
    print(f"  Completion % adjustment: {adjustments['completion_pct_adjustment']:+.1f} pct points")
    print(f"  Deep target share multiplier: {adjustments['deep_target_share_multiplier']:.3f}")
    print(f"  Rush rate boost: {adjustments['rush_rate_boost']:+.1%}\n")

    # Example 4: Wind + Heavy Rain
    print("4. Wind + Heavy Rain (SEA, 16 mph wind, 50°F, heavy rain)")
    adjustments = adjuster.calculate_weather_adjustments(
        team='SEA',
        wind_mph=16.0,
        temp_f=50.0,
        precip_prob=0.85,
        precip_intensity='heavy'
    )
    print(f"  Wind bucket: {adjustments['wind_bucket']}")
    print(f"  Precip level: {adjustments['precip_level']}")
    print(f"  Passing EPA multiplier: {adjustments['passing_epa_multiplier']:.3f}")
    print(f"  Completion % adjustment: {adjustments['completion_pct_adjustment']:+.1f} pct points")
    print(f"  Deep target multiplier: {adjustments['deep_target_share_multiplier']:.3f}")
    print(f"  Fumble rate multiplier: {adjustments['fumble_rate_multiplier']:.3f}\n")

    # Example 5: Apply to player projection
    print("5. Apply to WR Projection")
    base_proj = {
        'receiving_yards': 75.0,
        'targets': 8.0,
        'receptions': 5.2,  # 65% catch rate
        'deep_target_share': 0.25
    }

    # High wind scenario
    weather_adj = adjuster.calculate_weather_adjustments(
        team='DEN',
        wind_mph=18.0,
        temp_f=40.0
    )

    adjusted_proj = adjuster.apply_to_player_projection(
        position='WR',
        base_projection=base_proj,
        weather_adjustments=weather_adj
    )

    print(f"  Base receiving yards: {base_proj['receiving_yards']:.1f}")
    print(f"  Adjusted receiving yards: {adjusted_proj['receiving_yards']:.1f}")
    print(f"  Base receptions: {base_proj['receptions']:.1f}")
    print(f"  Adjusted receptions: {adjusted_proj['receptions']:.1f}")
    print(f"  Base deep target share: {base_proj['deep_target_share']:.1%}")
    print(f"  Adjusted deep target share: {adjusted_proj['deep_target_share']:.1%}")
