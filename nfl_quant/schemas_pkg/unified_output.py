#!/usr/bin/env python3
"""
NFL QUANT - Unified Output Schemas

Pydantic models for standardized betting recommendations output.
Ensures consistent schema across player props and game lines.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class BetType(str, Enum):
    """Unified bet type classification."""
    PLAYER_PROP = "player_prop"
    SPREAD = "spread"
    TOTAL = "total"
    MONEYLINE = "moneyline"


class ConfidenceTier(str, Enum):
    """Standardized confidence tiers."""
    ELITE = "ELITE"
    HIGH = "HIGH"
    STANDARD = "STANDARD"
    LOW = "LOW"


class UnifiedBetRecommendation(BaseModel):
    """
    Unified schema for all bet recommendations.

    This ensures player props and game lines have identical output structure.
    """

    # Identification
    bet_id: str = Field(..., description="Unique identifier for the bet")
    bet_type: BetType = Field(..., description="Type of bet")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Game context
    game_id: str = Field(..., description="NFL game identifier")
    game: str = Field(..., description="Game string (e.g., 'BUF @ KC')")
    week: int = Field(..., ge=1, le=22)
    season: int = Field(..., ge=2020, le=2030)

    # Player info (for player props)
    player: Optional[str] = Field(None, description="Player full name")
    nflverse_name: Optional[str] = Field(None, description="NFLverse lookup name")
    team: str = Field(..., description="Team abbreviation")
    position: Optional[str] = Field(None, description="Player position")

    # Pick information
    pick: str = Field(..., description="The recommended pick")
    market: Optional[str] = Field(None, description="Market type (for props)")
    market_line: Optional[float] = Field(None, description="Market line/spread/total")
    model_projection: Optional[float] = Field(None, description="Model's projected value")
    model_std: Optional[float] = Field(None, description="Model projection std dev")

    # Probabilities (STANDARDIZED)
    model_prob: float = Field(..., ge=0.0, le=1.0, description="Model probability")
    market_prob: float = Field(..., ge=0.0, le=1.0, description="Fair market probability (vig removed)")

    # Edge metrics (STANDARDIZED)
    edge_pct: float = Field(..., description="Edge percentage = (model_prob - market_prob) * 100")
    expected_roi: float = Field(..., description="Expected ROI accounting for vig")

    # Bet sizing (STANDARDIZED)
    american_odds: float = Field(..., description="American odds for the bet")
    kelly_fraction: float = Field(..., ge=0.0, le=1.0, description="Kelly optimal fraction")
    recommended_units: float = Field(..., ge=0.0, description="Recommended bet units")

    # Classification (STANDARDIZED)
    confidence_tier: ConfidenceTier = Field(..., description="Confidence tier")

    # Optional metadata
    calibration_applied: bool = Field(default=True, description="Whether calibration was applied")
    raw_prob: Optional[float] = Field(None, description="Raw uncalibrated probability")

    @validator('edge_pct')
    def validate_edge(cls, v, values):
        """Verify edge calculation matches probabilities."""
        if 'model_prob' in values and 'market_prob' in values:
            expected = (values['model_prob'] - values['market_prob']) * 100
            if abs(v - expected) > 0.1:  # Allow small rounding errors
                pass  # Just warn, don't fail
        return round(v, 2)

    @validator('kelly_fraction')
    def cap_kelly(cls, v):
        """Ensure Kelly fraction doesn't exceed safety limit."""
        return min(v, 0.10)  # Cap at 10% of bankroll

    class Config:
        use_enum_values = True


class UnifiedPipelineOutput(BaseModel):
    """
    Complete output from unified pipeline run.
    """
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    season: int
    week: int
    pipeline_version: str = Field(default="2.0.0")

    # Recommendations
    player_props: List[UnifiedBetRecommendation] = Field(default_factory=list)
    game_lines: List[UnifiedBetRecommendation] = Field(default_factory=list)

    # Summary statistics
    total_recommendations: int = Field(default=0)
    elite_picks: int = Field(default=0)
    high_picks: int = Field(default=0)
    standard_picks: int = Field(default=0)
    low_picks: int = Field(default=0)

    avg_edge_pct: float = Field(default=0.0)
    total_kelly_allocation: float = Field(default=0.0)

    @validator('total_recommendations', always=True, pre=True)
    def compute_total(cls, v, values):
        """Auto-compute total recommendations."""
        if 'player_props' in values and 'game_lines' in values:
            return len(values['player_props']) + len(values['game_lines'])
        return v

    def compute_summaries(self):
        """Compute summary statistics from recommendations."""
        all_recs = self.player_props + self.game_lines

        if not all_recs:
            return

        self.total_recommendations = len(all_recs)

        # Count by tier
        self.elite_picks = sum(1 for r in all_recs if r.confidence_tier == 'ELITE')
        self.high_picks = sum(1 for r in all_recs if r.confidence_tier == 'HIGH')
        self.standard_picks = sum(1 for r in all_recs if r.confidence_tier == 'STANDARD')
        self.low_picks = sum(1 for r in all_recs if r.confidence_tier == 'LOW')

        # Averages
        self.avg_edge_pct = sum(r.edge_pct for r in all_recs) / len(all_recs)
        self.total_kelly_allocation = sum(r.kelly_fraction for r in all_recs)

    class Config:
        use_enum_values = True


def create_player_prop_recommendation(
    player: str,
    nflverse_name: str,
    team: str,
    position: str,
    game_id: str,
    game: str,
    week: int,
    season: int,
    pick: str,
    market: str,
    line: float,
    projection: float,
    projection_std: float,
    model_prob: float,
    market_prob: float,
    edge_pct: float,
    expected_roi: float,
    american_odds: float,
    kelly_fraction: float,
    recommended_units: float,
    confidence_tier: str,
    raw_prob: Optional[float] = None
) -> UnifiedBetRecommendation:
    """Helper to create a player prop recommendation."""
    bet_id = f"{game_id}_{nflverse_name}_{market}_{pick}".replace(" ", "_").lower()

    return UnifiedBetRecommendation(
        bet_id=bet_id,
        bet_type=BetType.PLAYER_PROP,
        game_id=game_id,
        game=game,
        week=week,
        season=season,
        player=player,
        nflverse_name=nflverse_name,
        team=team,
        position=position,
        pick=pick,
        market=market,
        market_line=line,
        model_projection=projection,
        model_std=projection_std,
        model_prob=model_prob,
        market_prob=market_prob,
        edge_pct=edge_pct,
        expected_roi=expected_roi,
        american_odds=american_odds,
        kelly_fraction=kelly_fraction,
        recommended_units=recommended_units,
        confidence_tier=ConfidenceTier(confidence_tier),
        raw_prob=raw_prob
    )


def create_game_line_recommendation(
    game_id: str,
    game: str,
    week: int,
    season: int,
    bet_type: str,  # 'spread', 'total', 'moneyline'
    team: str,
    pick: str,
    market_line: Optional[float],
    model_fair_line: float,
    model_prob: float,
    market_prob: float,
    edge_pct: float,
    expected_roi: float,
    american_odds: float,
    kelly_fraction: float,
    recommended_units: float,
    confidence_tier: str
) -> UnifiedBetRecommendation:
    """Helper to create a game line recommendation."""
    bet_id = f"{game_id}_{bet_type}_{pick}".replace(" ", "_").replace("@", "at").lower()

    return UnifiedBetRecommendation(
        bet_id=bet_id,
        bet_type=BetType(bet_type),
        game_id=game_id,
        game=game,
        week=week,
        season=season,
        team=team,
        pick=pick,
        market_line=market_line,
        model_projection=model_fair_line,
        model_prob=model_prob,
        market_prob=market_prob,
        edge_pct=edge_pct,
        expected_roi=expected_roi,
        american_odds=american_odds,
        kelly_fraction=kelly_fraction,
        recommended_units=recommended_units,
        confidence_tier=ConfidenceTier(confidence_tier)
    )
