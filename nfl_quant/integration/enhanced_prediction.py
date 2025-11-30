#!/usr/bin/env python3
"""
Enhanced Prediction Data Structures

Contains dataclasses for predictions that include all contextual features:
- Base statistics
- Defensive matchup quality
- Weather/environment adjustments
- Rest/travel context
- Snap count trends
- Injury impact
- Next Gen Stats
- Target share velocity
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DefensiveMatchupFeatures:
    """Features related to opponent defensive quality."""
    opponent_team: str = ""
    opponent_def_epa: float = 0.0  # Defensive EPA allowed (higher = worse defense)
    def_epa_rank: int = 16  # 1 = best defense, 32 = worst
    matchup_multiplier: float = 1.0  # >1 = favorable matchup, <1 = tough matchup
    historical_vs_opponent: float = 0.0  # Avg performance vs this opponent


@dataclass
class WeatherFeatures:
    """Weather and environment features."""
    temperature: float = 70.0  # Fahrenheit
    wind_speed: float = 0.0  # MPH
    precipitation: str = "none"  # none, light, moderate, heavy
    is_dome: bool = False
    wind_bucket: str = "calm"  # calm, moderate, high, extreme
    passing_epa_multiplier: float = 1.0
    deep_target_multiplier: float = 1.0
    rush_boost: float = 0.0


@dataclass
class RestTravelFeatures:
    """Rest days and travel context."""
    days_rest: int = 7
    is_short_rest: bool = False  # <6 days
    is_bye_week_return: bool = False
    travel_distance_miles: float = 0.0
    timezone_change: int = 0
    is_home_game: bool = True
    is_divisional_game: bool = False
    is_primetime: bool = False
    rest_epa_multiplier: float = 1.0
    travel_epa_multiplier: float = 1.0


@dataclass
class SnapCountFeatures:
    """Snap participation and usage trends."""
    avg_offense_pct: float = 0.0  # Average snap percentage
    recent_offense_pct: float = 0.0  # Most recent week
    snap_trend: float = 0.0  # Positive = increasing usage
    snap_volatility: float = 0.0
    is_primary_option: bool = False  # >50% snaps
    role_change_detected: bool = False
    snap_share_multiplier: float = 1.0


@dataclass
class InjuryImpactFeatures:
    """Injury-related adjustments."""
    player_injury_status: str = "healthy"  # healthy, questionable, doubtful, out
    player_game_probability: float = 1.0
    teammates_out: list = field(default_factory=list)  # List of OUT teammates
    target_share_boost: float = 0.0  # Additional target share due to injuries
    carry_share_boost: float = 0.0  # Additional carry share
    injury_redistribution_multiplier: float = 1.0


@dataclass
class TargetShareFeatures:
    """Target share trends and velocity."""
    current_target_share: float = 0.0
    target_share_trend: float = 0.0  # Week-over-week change
    target_share_percentile: float = 0.5  # Percentile within position
    air_yards_share: float = 0.0
    red_zone_target_share: float = 0.0


@dataclass
class NGSFeatures:
    """Next Gen Stats advanced metrics."""
    # QB metrics
    qb_time_to_throw: float = 0.0
    qb_cpoe: float = 0.0  # Completion % over expected
    qb_aggressiveness: float = 0.0

    # WR/TE metrics
    avg_separation: float = 0.0  # Yards of separation at catch
    avg_cushion: float = 0.0  # Yards of cushion at snap
    yac_over_expected: float = 0.0

    # RB metrics
    rush_yards_over_expected: float = 0.0
    time_to_los: float = 0.0  # Time to line of scrimmage

    ngs_skill_multiplier: float = 1.0


@dataclass
class QBConnectionFeatures:
    """QB-WR/TE chemistry features."""
    qb_connection_targets: int = 0
    qb_connection_completions: int = 0
    qb_connection_rate: float = 0.0
    qb_connection_multiplier: float = 1.0
    avg_air_yards: float = 0.0
    is_primary_target: bool = False


@dataclass
class HistoricalMatchupFeatures:
    """Historical performance vs specific opponent."""
    vs_opponent_avg: float = 0.0
    vs_opponent_games: int = 0
    vs_opponent_multiplier: float = 1.0


@dataclass
class TeamPaceFeatures:
    """Team pace and play volume."""
    team_plays_per_game: float = 65.0
    opponent_plays_per_game: float = 65.0
    expected_game_pace: float = 65.0
    pace_multiplier: float = 1.0  # >1 = more opportunities


@dataclass
class AllFeatures:
    """Container for all feature categories."""
    defensive_matchup: DefensiveMatchupFeatures = field(default_factory=DefensiveMatchupFeatures)
    weather: WeatherFeatures = field(default_factory=WeatherFeatures)
    rest_travel: RestTravelFeatures = field(default_factory=RestTravelFeatures)
    snap_counts: SnapCountFeatures = field(default_factory=SnapCountFeatures)
    injury_impact: InjuryImpactFeatures = field(default_factory=InjuryImpactFeatures)
    target_share: TargetShareFeatures = field(default_factory=TargetShareFeatures)
    ngs: NGSFeatures = field(default_factory=NGSFeatures)
    team_pace: TeamPaceFeatures = field(default_factory=TeamPaceFeatures)
    qb_connection: QBConnectionFeatures = field(default_factory=QBConnectionFeatures)
    historical_matchup: HistoricalMatchupFeatures = field(default_factory=HistoricalMatchupFeatures)


@dataclass
class EnhancedPrediction:
    """
    Enhanced prediction with all contextual features.

    This replaces the basic ProductionPrediction with full feature visibility.
    """
    # Core prediction info
    player_name: str
    position: str
    team: str
    opponent: str
    market: str
    line: float
    week: int
    season: int

    # Base statistics (from trailing stats)
    base_mean: float
    base_std: float

    # Adjusted statistics (after all features applied)
    adjusted_mean: float
    adjusted_std: float

    # Probability estimates
    raw_prob_over: float
    calibrated_prob_over: float

    # Market comparison
    market_prob_over: float = 0.5
    edge_over: float = 0.0
    edge_under: float = 0.0

    # Confidence
    confidence_tier: str = "MEDIUM"
    data_quality_score: float = 0.0

    # Feature contributions (how much each feature moved the prediction)
    feature_contributions: Dict[str, float] = field(default_factory=dict)

    # All features
    features: AllFeatures = field(default_factory=AllFeatures)

    # Metadata
    prediction_timestamp: str = ""
    model_version: str = "v2.0-enhanced"
    calibrator_version: str = ""

    def get_total_adjustment_multiplier(self) -> float:
        """Calculate the total adjustment from all features."""
        multipliers = [
            self.features.defensive_matchup.matchup_multiplier,
            self.features.weather.passing_epa_multiplier,
            self.features.rest_travel.rest_epa_multiplier,
            self.features.rest_travel.travel_epa_multiplier,
            self.features.snap_counts.snap_share_multiplier,
            self.features.injury_impact.injury_redistribution_multiplier,
            self.features.ngs.ngs_skill_multiplier,
            self.features.team_pace.pace_multiplier,
            self.features.qb_connection.qb_connection_multiplier,
            self.features.historical_matchup.vs_opponent_multiplier,
        ]

        total = 1.0
        for mult in multipliers:
            total *= mult

        return total

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of key feature impacts."""
        return {
            'base_mean': self.base_mean,
            'adjusted_mean': self.adjusted_mean,
            'total_adjustment': self.adjusted_mean / self.base_mean if self.base_mean > 0 else 1.0,
            'defensive_matchup': self.features.defensive_matchup.matchup_multiplier,
            'weather_impact': self.features.weather.passing_epa_multiplier,
            'rest_impact': self.features.rest_travel.rest_epa_multiplier,
            'snap_trend': self.features.snap_counts.snap_trend,
            'injury_boost': self.features.injury_impact.injury_redistribution_multiplier,
            'opponent_def_epa': self.features.defensive_matchup.opponent_def_epa,
            'wind_speed': self.features.weather.wind_speed,
            'days_rest': self.features.rest_travel.days_rest,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'player_name': self.player_name,
            'position': self.position,
            'team': self.team,
            'opponent': self.opponent,
            'market': self.market,
            'line': self.line,
            'week': self.week,
            'season': self.season,
            'base_mean': self.base_mean,
            'adjusted_mean': self.adjusted_mean,
            'base_std': self.base_std,
            'adjusted_std': self.adjusted_std,
            'raw_prob_over': self.raw_prob_over,
            'calibrated_prob_over': self.calibrated_prob_over,
            'market_prob_over': self.market_prob_over,
            'edge_over': self.edge_over,
            'edge_under': self.edge_under,
            'confidence_tier': self.confidence_tier,
            'data_quality_score': self.data_quality_score,
            'feature_contributions': self.feature_contributions,
            'total_adjustment_multiplier': self.get_total_adjustment_multiplier(),
            **self.get_feature_summary(),
            'model_version': self.model_version,
            'prediction_timestamp': self.prediction_timestamp,
        }
