"""
Contextual factors schemas for matchups, QB connections, and situational context.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


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






















