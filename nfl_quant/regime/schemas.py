"""
Regime Change Detection Schemas

Data models for regime detection, tracking, and impact analysis.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class RegimeType(str, Enum):
    """Types of regime changes detected."""
    QB_CHANGE = "qb_change"
    QB_RETURN = "qb_return"
    COACHING_CHANGE = "coaching_change"
    PLAYCALLER_CHANGE = "playcaller_change"
    ROSTER_WR1_EMERGENCE = "roster_wr1_emergence"
    ROSTER_RB_COMMITTEE = "roster_rb_committee"
    ROSTER_OLINE_INJURY = "roster_oline_injury"
    ROSTER_TE_FOCAL = "roster_te_focal"
    SCHEME_PASS_RATE_SHIFT = "scheme_pass_rate_shift"
    SCHEME_TEMPO_CHANGE = "scheme_tempo_change"
    SCHEME_FORMATION_CHANGE = "scheme_formation_change"
    STABLE = "stable"


class RegimeTrigger(BaseModel):
    """Details about what triggered a regime change."""
    type: RegimeType
    description: str = Field(..., description="Human-readable description of trigger")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    detected_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class QBChangeDetails(BaseModel):
    """Detailed information about QB changes."""
    previous_qb: Optional[str] = None
    current_qb: str
    change_reason: Literal["injury", "benching", "return", "planned"] = "planned"
    games_missed: int = 0
    passing_efficiency_delta: Optional[float] = None  # EPA delta
    completion_pct_delta: Optional[float] = None


class CoachingChangeDetails(BaseModel):
    """Detailed information about coaching changes."""
    previous_oc: Optional[str] = None
    current_oc: Optional[str] = None
    previous_hc: Optional[str] = None
    current_hc: Optional[str] = None
    previous_playcaller: Optional[str] = None
    current_playcaller: Optional[str] = None
    change_type: Literal["oc", "hc", "playcaller"]
    offensive_philosophy_shift: Optional[str] = None  # "pass_heavy", "run_heavy", "balanced"


class RosterImpactDetails(BaseModel):
    """Detailed information about roster changes."""
    position: str
    affected_players: List[str] = Field(default_factory=list)
    impact_type: Literal["emergence", "injury", "committee", "role_change"]
    snap_share_changes: Dict[str, float] = Field(default_factory=dict)  # player -> delta
    target_share_changes: Dict[str, float] = Field(default_factory=dict)
    touch_share_changes: Dict[str, float] = Field(default_factory=dict)


class SchemeChangeDetails(BaseModel):
    """Detailed information about scheme changes."""
    metric_changed: Literal["pass_rate", "tempo", "formation", "redzone_philosophy"]
    previous_value: float
    current_value: float
    percent_change: float
    moving_average_window: int = 3  # Games used to detect shift
    statistical_significance: Optional[float] = None  # p-value if tested


class Regime(BaseModel):
    """Complete regime definition with all metadata."""
    regime_id: str = Field(..., description="Unique identifier: {season}_{team}_{type}_{start_week}")
    team: str
    season: int
    start_week: int
    end_week: Optional[int] = None  # None if still active

    trigger: RegimeTrigger
    details: Optional[Any] = None  # Union of QB/Coaching/Roster/Scheme details

    affected_players: List[str] = Field(default_factory=list)
    games_in_regime: int = 0
    is_active: bool = True

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def sample_size_confidence(self) -> float:
        """Calculate confidence based on games in regime."""
        if self.games_in_regime >= 4:
            return 1.0
        elif self.games_in_regime >= 3:
            return 0.85
        elif self.games_in_regime >= 2:
            return 0.65
        else:
            return 0.40


class RegimeWindow(BaseModel):
    """Time window for regime-specific analysis."""
    regime_id: str
    start_week: int
    end_week: int
    weeks: List[int] = Field(default_factory=list)
    season: int

    @property
    def num_weeks(self) -> int:
        return len(self.weeks)

    @property
    def is_sufficient_sample(self) -> bool:
        """Minimum 2 weeks required."""
        return self.num_weeks >= 2


class UsageMetrics(BaseModel):
    """Player usage metrics within a regime."""
    snap_share: float = Field(..., ge=0.0, le=1.0)
    snaps_per_game: float = Field(..., ge=0.0)

    # Position-specific
    target_share: Optional[float] = Field(None, ge=0.0, le=1.0)
    targets_per_game: Optional[float] = Field(None, ge=0.0)
    touch_share: Optional[float] = Field(None, ge=0.0, le=1.0)
    carries_per_game: Optional[float] = Field(None, ge=0.0)

    # Red zone
    redzone_target_share: Optional[float] = Field(None, ge=0.0, le=1.0)
    redzone_carry_share: Optional[float] = Field(None, ge=0.0, le=1.0)
    goalline_carry_share: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Routes (WR/TE)
    routes_per_game: Optional[float] = Field(None, ge=0.0)
    route_participation: Optional[float] = Field(None, ge=0.0, le=1.0)


class EfficiencyMetrics(BaseModel):
    """Player efficiency metrics within a regime."""
    # WR/TE
    yards_per_route_run: Optional[float] = None
    yards_per_target: Optional[float] = None
    catch_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    yards_after_catch_per_reception: Optional[float] = None
    average_depth_of_target: Optional[float] = None

    # RB
    yards_per_carry: Optional[float] = None
    yards_per_reception: Optional[float] = None
    broken_tackles_per_touch: Optional[float] = None

    # QB
    completion_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    yards_per_attempt: Optional[float] = None
    yards_per_completion: Optional[float] = None

    # Universal
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    epa_per_play: Optional[float] = None
    td_rate: Optional[float] = Field(None, ge=0.0)


class ContextMetrics(BaseModel):
    """Team context metrics during a regime."""
    team_pass_rate: float = Field(..., ge=0.0, le=1.0)
    team_points_per_game: float = Field(..., ge=0.0)
    team_plays_per_game: float = Field(..., ge=0.0)
    average_game_script: float  # Positive = leading, negative = trailing

    # Home/away
    home_games: int = 0
    away_games: int = 0

    # Opponent quality
    avg_opponent_defensive_epa: Optional[float] = None
    avg_opponent_defensive_rank: Optional[float] = None


class RegimeMetrics(BaseModel):
    """Complete metrics for a player in a regime."""
    player_name: str
    player_id: str
    position: str
    team: str
    regime_id: str

    usage: UsageMetrics
    efficiency: EfficiencyMetrics
    context: ContextMetrics

    weeks_in_regime: int
    games_played: int
    games_missed: int = 0

    @property
    def sample_quality(self) -> Literal["excellent", "good", "fair", "poor"]:
        """Assess sample size quality."""
        if self.games_played >= 4:
            return "excellent"
        elif self.games_played >= 3:
            return "good"
        elif self.games_played >= 2:
            return "fair"
        else:
            return "poor"


class RegimeComparison(BaseModel):
    """Comparison between two regimes for a player."""
    player_name: str
    position: str
    team: str

    previous_regime_id: Optional[str] = None
    current_regime_id: str

    # Usage deltas
    target_share_delta: Optional[float] = None
    snap_share_delta: Optional[float] = None
    touch_share_delta: Optional[float] = None

    # Efficiency deltas
    yards_per_route_delta: Optional[float] = None
    yards_per_carry_delta: Optional[float] = None
    catch_rate_delta: Optional[float] = None

    # Fantasy impact
    ppg_delta: Optional[float] = None
    ppg_delta_pct: Optional[float] = None

    # Statistical significance
    is_statistically_significant: bool = False
    p_value: Optional[float] = None
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None


class RegimeImpact(BaseModel):
    """Quantified impact of regime change on projections."""
    player_name: str
    regime_id: str
    market: str  # "player_pass_yds", "player_receptions", etc.

    # Absolute impact
    baseline_projection: float
    regime_adjusted_projection: float
    absolute_impact: float

    # Relative impact
    relative_impact_pct: float

    # Confidence-adjusted
    raw_impact: float
    confidence_factor: float
    adjusted_impact: float

    # Regression to mean
    positional_baseline: float
    shrinkage_factor: float
    final_projection: float

    # Metadata
    sample_size: int
    regime_type: RegimeType
    application_date: datetime = Field(default_factory=datetime.now)


class ProjectionAdjustment(BaseModel):
    """Adjustment to apply to base projection."""
    player_name: str
    market: str

    base_projection: float
    regime_multiplier: float
    adjusted_projection: float

    adjustment_reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Blending weights
    new_regime_weight: float = Field(..., ge=0.0, le=1.0)
    old_regime_weight: float = Field(..., ge=0.0, le=1.0)

    @field_validator('new_regime_weight', 'old_regime_weight')
    @classmethod
    def weights_sum_to_one(cls, v, info):
        if 'new_regime_weight' in info.data:
            total = info.data['new_regime_weight'] + v
            if not 0.99 <= total <= 1.01:  # Allow small floating point error
                raise ValueError("Weights must sum to 1.0")
        return v


class BettingRecommendation(BaseModel):
    """Enhanced betting recommendation with regime context."""
    player_name: str
    team: str
    position: str
    opponent: str

    market: str
    line: float
    side: Literal["over", "under"]

    # Probabilities
    model_probability: float = Field(..., ge=0.0, le=1.0)
    market_probability: float = Field(..., ge=0.0, le=1.0)
    edge_pct: float

    # Regime context
    regime_type: Optional[RegimeType] = None
    regime_weeks: Optional[int] = None
    regime_confidence: Optional[float] = None
    regime_impact_pct: Optional[float] = None

    # Bet sizing
    kelly_fraction: float
    bet_size: float
    expected_value: float

    # Metadata
    confidence_badge: Literal["high", "medium", "low"] = "medium"
    sample_size_warning: bool = False
    regime_flag: Optional[str] = None  # "🔄 New QB", "⚠️ Small sample", etc.


class RegimeDetectionResult(BaseModel):
    """Result from regime detection process."""
    team: str
    season: int
    current_week: int

    regimes_detected: List[Regime] = Field(default_factory=list)
    active_regime: Optional[Regime] = None

    affected_players: List[str] = Field(default_factory=list)
    detection_timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def has_active_regime(self) -> bool:
        return self.active_regime is not None

    @property
    def regime_count(self) -> int:
        return len(self.regimes_detected)


class PlayerRegimeProfile(BaseModel):
    """Complete regime profile for a player."""
    player_name: str
    player_id: str
    position: str
    team: str

    current_regime: Regime
    regime_metrics: RegimeMetrics
    regime_comparison: Optional[RegimeComparison] = None

    # Historical regimes
    previous_regimes: List[Regime] = Field(default_factory=list)

    # Projection impact
    projection_adjustments: List[ProjectionAdjustment] = Field(default_factory=list)

    @property
    def is_regime_stable(self) -> bool:
        """Check if in stable regime with sufficient sample."""
        return (
            self.current_regime.trigger.type == RegimeType.STABLE
            or self.current_regime.games_in_regime >= 4
        )


class TeamRegimeSummary(BaseModel):
    """Team-wide regime analysis summary."""
    team: str
    season: int
    current_week: int

    active_regime: Regime
    regime_start_week: int
    weeks_in_regime: int

    # Team metric changes
    pass_rate_change: Optional[float] = None
    pace_change: Optional[float] = None
    points_per_game_change: Optional[float] = None
    epa_per_play_change: Optional[float] = None

    # Position group impacts
    wr_target_share_change: Optional[float] = None
    te_target_share_change: Optional[float] = None
    rb_touch_share_change: Optional[float] = None

    # Players to target/fade
    players_to_target: List[str] = Field(default_factory=list)
    players_to_fade: List[str] = Field(default_factory=list)

    # Impact level
    impact_level: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
