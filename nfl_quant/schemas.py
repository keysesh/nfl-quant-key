"""Pydantic schemas for strict data contracts across the pipeline."""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Game(BaseModel):
    """NFL game metadata."""

    game_id: str
    season: int
    week: int
    game_date: str
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    is_completed: bool = False

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: int) -> int:
        """Ensure season is 2025."""
        if v != 2025:
            raise ValueError(f"Only 2025 season supported, got {v}")
        return v


class TeamWeekFeatures(BaseModel):
    """Team-level features for a given week."""

    season: int
    week: int
    team: str
    is_offense: bool  # True for offensive stats, False for defensive

    # Core EPA metrics
    epa_per_play: float
    passing_epa: float
    rushing_epa: float

    # Success rate metrics
    success_rate_overall: float
    success_rate_pass: float
    success_rate_rush: float

    # Advanced metrics
    proe: float  # Pass Rate Over Expected
    neutral_pace: float  # seconds per play

    # Play-type metrics
    explosive_rate: float  # Share of explosive plays
    redzone_td_rate: Optional[float] = None
    third_down_conv_rate: Optional[float] = None
    fourth_down_conv_rate: Optional[float] = None

    # Pressure/sack proxy
    pressure_rate: Optional[float] = None

    # Metadata
    play_count: int
    game_count: int  # Games included in rolling window
    is_rolling_avg: bool = False

    model_config = ConfigDict(frozen=False)


class InjuryImpact(BaseModel):
    """Injury impact on team EPAs."""

    team: str
    week: int
    total_impact_offensive_epa: float
    total_impact_defensive_epa: float
    injury_count: int
    missing_qb: bool
    missing_ol_count: int
    player_impacts: list[dict] = Field(default_factory=list)


class SimulationInput(BaseModel):
    """Input for Monte Carlo simulation."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Offensive EPA
    home_offensive_epa: float
    away_offensive_epa: float

    # Defensive EPA
    home_defensive_epa: float

    # NEW: Game context factors
    is_divisional: bool = False
    is_primetime: bool = False
    game_type: str = 'Regular'  # 'Regular', 'SNF', 'MNF', 'TNF'

    # NEW: Weather factors
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    is_dome: bool = False

    # NEW: Injury adjustments (applied to EPA)
    home_injury_offensive_adjustment: float = 0.0
    home_injury_defensive_adjustment: float = 0.0
    away_injury_offensive_adjustment: float = 0.0
    away_injury_defensive_adjustment: float = 0.0
    away_defensive_epa: float

    # Pace adjustment
    home_pace: float
    away_pace: float

    # Other contextual
    home_is_favored: Optional[bool] = None

    model_config = ConfigDict(frozen=True)


class SimulationOutput(BaseModel):
    """Output from Monte Carlo simulation."""

    game_id: str
    trial_count: int
    seed: int

    # Win probabilities
    home_win_prob: float
    away_win_prob: float
    tie_prob: float

    # Fair prices
    fair_spread: float  # Negative = away favored
    fair_total: float

    # Score distributions (percentiles)
    home_score_median: float
    away_score_median: float
    home_score_std: float
    away_score_std: float

    # Total distribution
    total_median: float
    total_std: float
    total_p5: Optional[float] = None  # 5th percentile
    total_p25: Optional[float] = None  # 25th percentile
    total_p50: Optional[float] = None  # 50th percentile (median)
    total_p75: Optional[float] = None  # 75th percentile
    total_p95: Optional[float] = None  # 95th percentile

    # Pace (seconds per play) - used for player prop predictions
    pace: Optional[float] = None  # Average pace (home_pace + away_pace) / 2
    home_pace: Optional[float] = None  # Home team pace
    away_pace: Optional[float] = None  # Away team pace

    model_config = ConfigDict(frozen=True)


class OddsRecord(BaseModel):
    """Odds record for a single side of a game."""

    game_id: str
    side: str  # "home_spread", "away_spread", "over", "under"
    american_odds: int
    fair_odds: Optional[int] = None

    @field_validator("american_odds")
    @classmethod
    def validate_odds(cls, v: int) -> int:
        """American odds must be non-zero integer."""
        if v == 0:
            raise ValueError("American odds cannot be zero")
        return v


class BetSizing(BaseModel):
    """Bet sizing recommendation."""

    game_id: str
    side: str
    american_odds: int
    fair_odds: Optional[int] = None
    implied_prob: float
    win_prob: float
    ev_pct: float

    kelly_fraction: float
    kelly_pct: float
    suggested_bet_amount: float
    max_suggested_bet: float

    potential_win: float
    potential_loss: float

    model_config = ConfigDict(frozen=True)


class ExportRecord(BaseModel):
    """Full export record for a week's bets."""

    game_id: str
    week: int
    home_team: str
    away_team: str
    side: str

    american_odds: int
    implied_prob: float
    win_prob: float
    ev_pct: float

    kelly_pct: float
    suggested_bet_amount: float
    potential_win: float

    injury_count_home: int
    injury_count_away: int
    injury_impact_home_epa: float
    injury_impact_away_epa: float

    fair_spread: float
    fair_total: float

    home_epa_off: float
    home_epa_def: float
    away_epa_off: float
    away_epa_def: float

    export_timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class PlayerPropInput(BaseModel):
    """Input for player prop prediction."""

    player_id: str
    player_name: str
    team: str
    position: str  # QB, RB, WR, TE, K
    week: int
    opponent: str

    # Game context (from game simulation)
    projected_team_total: float
    projected_opponent_total: float
    projected_game_script: float  # Expected point differential
    projected_pace: float  # Seconds per play

    # Historical player stats (trailing 4 weeks)
    trailing_snap_share: float
    trailing_target_share: Optional[float] = None
    trailing_carry_share: Optional[float] = None
    trailing_yards_per_opportunity: float
    trailing_td_rate: float
    # QB-specific efficiency metrics (for trained model)
    trailing_comp_pct: Optional[float] = None
    trailing_yards_per_completion: Optional[float] = None
    trailing_td_rate_pass: Optional[float] = None
    trailing_yards_per_carry: Optional[float] = None
    trailing_td_rate_rush: Optional[float] = None
    trailing_yards_per_target: Optional[float] = None  # Receiving efficiency (WR/TE/RB)

    # Actual historical averages (for use when trailing_target_share is 0.0)
    # These come from unified historical stats and represent actual per-game averages
    avg_rec_yd: Optional[float] = None  # Actual average receiving yards per game
    avg_rec_tgt: Optional[float] = None  # Actual average targets per game (if available)
    avg_rush_yd: Optional[float] = None  # Actual average rushing yards per game

    # Opponent defense strength
    opponent_def_epa_vs_position: float

    # NEW: Kicker-specific fields (only populated when position='K')
    trailing_fg_attempts_per_game: Optional[float] = None
    trailing_fg_pct: Optional[float] = None
    trailing_xp_per_game: Optional[float] = None
    team_implied_total: Optional[float] = None  # Get from Odds API
    opponent_def_rank_points: Optional[int] = None  # Optional enhancement

    # NEW: Matchup context
    opponent_cb_rank: Optional[float] = None  # CB rank vs WR type (0-1, higher = better CB)
    opponent_lb_rank: Optional[float] = None  # LB rank vs RB type
    historical_yards_vs_opponent: Optional[float] = None  # Avg yards vs this opponent
    historical_target_share_vs_opponent: Optional[float] = None  # Avg target share vs opponent
    matchup_strength_score: Optional[float] = None  # 0-1 score of matchup difficulty

    # NEW: QB connection context
    starting_qb_id: Optional[str] = None
    starting_qb_name: Optional[str] = None
    qb_injury_status: Optional[str] = None  # 'healthy', 'injured', 'questionable', 'doubtful'
    target_share_with_starting_qb: Optional[float] = None  # Historical target share with starting QB
    target_share_with_backup_qb: Optional[float] = None  # Historical target share with backup QB
    qb_style_match_score: Optional[float] = None  # 0-1 compatibility score
    yards_per_target_with_qb: Optional[float] = None  # Historical efficiency with QB

    # NEW: Situational context
    teammate_injury_multiplier: Optional[float] = None  # Multiplier when key teammates injured
    lineup_change_multiplier: Optional[float] = None  # Multiplier for lineup changes
    opportunity_multiplier: Optional[float] = None  # Combined opportunity multiplier

    # NEW: Red Zone and Goal Line Factors
    redzone_target_share: Optional[float] = None  # Red zone target share (for WR/TE/RB)
    redzone_carry_share: Optional[float] = None  # Red zone carry share (for RB)
    goalline_carry_share: Optional[float] = None  # Goal line carry share (for RB, inside 5 yards)

    # NEW: Game Context Factors
    is_divisional_game: Optional[bool] = None  # Whether game is divisional
    is_primetime_game: Optional[bool] = None  # Whether game is primetime (SNF/MNF/TNF)
    primetime_type: Optional[str] = None  # 'SNF', 'MNF', 'TNF', or None
    is_high_altitude: Optional[bool] = None  # Whether game at high altitude stadium
    elevation_feet: Optional[int] = None  # Stadium elevation in feet
    field_surface: Optional[str] = None  # 'turf' or 'grass'
    home_field_advantage_points: Optional[float] = None  # Team-specific HFA in points

    # NEW: Weather Factors (from unified integration)
    weather_total_adjustment: Optional[float] = None  # Total weather adjustment multiplier
    weather_passing_adjustment: Optional[float] = None  # Passing-specific weather adjustment

    # NEW: Contextual Factors (from unified integration)
    rest_epa_adjustment: Optional[float] = None  # Rest-based EPA adjustment
    travel_epa_adjustment: Optional[float] = None  # Travel-based EPA adjustment
    is_coming_off_bye: Optional[bool] = None  # Whether team coming off bye week
    altitude_epa_adjustment: Optional[float] = None  # Altitude-based EPA adjustment

    # NEW: Team Usage (from game simulations)
    projected_team_pass_attempts: Optional[float] = None  # Team pass attempts from simulation
    projected_team_rush_attempts: Optional[float] = None  # Team rush attempts from simulation
    projected_team_targets: Optional[float] = None  # Team targets from simulation

    # NEW: Injury Status (from unified integration)
    injury_qb_status: Optional[str] = None  # QB injury status ('healthy', 'injured', 'questionable')
    injury_wr1_status: Optional[str] = None  # WR1 injury status
    injury_rb1_status: Optional[str] = None  # RB1 injury status

    model_config = ConfigDict(frozen=True)


class PlayerPropOutput(BaseModel):
    """Output from player prop simulation."""

    player_id: str
    player_name: str
    position: str
    prop_type: str  # e.g., "passing_yards", "rushing_yards", "receptions"

    trial_count: int
    seed: int

    # Stat distribution
    median_stat: float
    mean_stat: float
    std_stat: float
    p10_stat: float  # 10th percentile
    p90_stat: float  # 90th percentile

    # Prop line probabilities (if line provided)
    prop_line: Optional[float] = None
    over_prob_raw: Optional[float] = None
    over_prob_calibrated: Optional[float] = None

    model_config = ConfigDict(frozen=True)


# Contextual factors schemas for matchups, QB connections, and situational context.


class MatchupRecord(BaseModel):
    """Historical individual matchup performance."""

    offense_player_id: str
    offense_player_name: str
    offense_position: str
    offense_team: str
    defense_player_id: Optional[str] = None  # None if team-level
    defense_player_name: Optional[str] = None
    defense_position: Optional[str] = None  # CB, LB, S, etc.
    defense_team: str
    week: int
    season: int
    game_id: str

    # Matchup-specific stats
    targets_against: int = 0  # For WR vs CB
    receptions_against: int = 0
    yards_against: float = 0.0
    tds_against: int = 0
    air_yards_per_target: float = 0.0
    catch_rate: float = 0.0

    # Usage context
    qb_player_id: Optional[str] = None  # QB throwing to WR
    qb_name: Optional[str] = None
    qb_attempts: int = 0  # Total QB attempts in game

    # Outcome metrics
    target_share_vs_matchup: float = 0.0  # % of targets when matched up
    yards_per_target_vs_matchup: float = 0.0
    catch_rate_vs_matchup: float = 0.0


class QBConnectionRecord(BaseModel):
    """Historical QB-WR connection performance."""

    qb_player_id: str
    qb_name: str
    qb_team: str
    wr_player_id: str
    wr_name: str
    wr_position: str  # WR, TE, RB
    week: int
    season: int
    game_id: str

    # Connection-specific stats
    qb_attempts: int = 0
    wr_targets: int = 0
    wr_target_share: float = 0.0  # targets / qb_attempts
    wr_receptions: int = 0
    wr_receiving_yards: float = 0.0
    wr_receiving_tds: int = 0

    # Efficiency metrics
    yards_per_target: float = 0.0
    catch_rate: float = 0.0
    adot: float = 0.0  # Average depth of target
    air_yards_per_target: float = 0.0

    # Context
    qb_injury_status: str = 'healthy'  # 'healthy', 'injured', 'questionable'
    wr_injury_status: str = 'healthy'
    game_script: float = 0.0  # Score differential


class SituationalContext(BaseModel):
    """Game-specific situational factors."""

    game_id: str
    team: str
    opponent: str
    week: int
    season: int

    # Key player statuses
    starting_qb: str
    backup_qb: Optional[str] = None
    qb_injury_status: str = 'healthy'  # 'healthy', 'injured', 'questionable', 'doubtful'

    # Key skill position availability
    top_wr_1: Optional[str] = None
    top_wr_1_status: str = 'active'  # 'active', 'injured', 'doubtful', 'out'
    top_wr_2: Optional[str] = None
    top_wr_2_status: str = 'active'
    top_rb: Optional[str] = None
    top_rb_status: str = 'active'

    # Expected usage adjustments
    expected_target_share_multiplier: float = 1.0  # For WRs when QB changes
    expected_carry_share_multiplier: float = 1.0  # For RBs when RB1 injured
    expected_snap_share_multiplier: float = 1.0  # For backups when starter out


class MatchupHistory(BaseModel):
    """Aggregated matchup history for a player."""

    player_name: str
    position: str

    # Team-level opponent history
    vs_opponent_team: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Individual matchup history
    vs_specific_cb: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vs_specific_lb: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # QB connection history
    qb_connections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Sample sizes
    total_games: int = 0
    games_vs_opponent: int = 0
