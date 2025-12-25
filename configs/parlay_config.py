"""
Parlay Optimization Configuration
=================================

Centralized configuration for the parlay optimization system.
Cross-game parlays only (no SGP), max 4 legs.
"""

from dataclasses import dataclass
from typing import Dict
from pathlib import Path


@dataclass(frozen=True)
class ParlayConfig:
    """Configuration for parlay generation and optimization."""

    # Correlation thresholds
    max_correlation_for_inclusion: float = 0.70  # Block parlays with correlation > 70%

    # Leg limits (cross-game only, no SGP)
    min_legs: int = 2
    max_legs: int = 4
    cross_game_only: bool = True  # Enforce different games for each leg

    # Sizing (conservative for parlays)
    max_parlay_units: float = 0.5  # 0.5x straight bet max
    parlay_kelly_fraction: float = 0.15  # More conservative than single bets (0.25)
    base_unit_size: float = 5.0  # Dollar value per unit

    # Confidence thresholds
    min_leg_confidence: float = 0.55  # Minimum confidence per leg
    min_combined_confidence: float = 0.10  # Minimum joint probability (after correlation adjustment)

    # Edge requirements
    min_edge_per_leg: float = 0.02  # 2% minimum edge per leg
    min_parlay_edge: float = 0.05  # 5% minimum edge for the parlay

    # Odds API
    preferred_sportsbook: str = "fanduel"
    fallback_to_calculated: bool = True  # Use calculated odds if API unavailable

    # Output
    num_parlays_to_generate: int = 50  # Generate top N parlays
    num_parlays_to_recommend: int = 10  # Recommend top N after filtering


@dataclass(frozen=True)
class VariancePenalty:
    """Kelly variance penalty by number of legs."""

    penalties: Dict[int, float] = None

    def __post_init__(self):
        if self.penalties is None:
            object.__setattr__(self, 'penalties', {
                2: 0.85,  # 2-leg: 85% of base Kelly
                3: 0.70,  # 3-leg: 70% of base Kelly
                4: 0.55,  # 4-leg: 55% of base Kelly
            })

    def get_penalty(self, num_legs: int) -> float:
        """Get variance penalty for given number of legs."""
        if num_legs < 2:
            return 1.0
        return self.penalties.get(num_legs, 0.40)  # Default to 40% for 5+ legs


# Default configuration instance
PARLAY_CONFIG = ParlayConfig()
VARIANCE_PENALTY = VariancePenalty()


# File paths
CORRELATION_MATRIX_PATH = Path("data/correlations/empirical_matrix.json")
PARLAY_REPORTS_DIR = Path("reports")


def get_parlay_output_path(week: int, season: int = 2025) -> Path:
    """Get output path for parlay recommendations."""
    return PARLAY_REPORTS_DIR / f"parlay_recommendations_week{week}_{season}.csv"


def get_edge_recommendations_path(week: int, season: int = 2025) -> Path:
    """Get path to edge recommendations input file."""
    return PARLAY_REPORTS_DIR / f"edge_recommendations_week{week}_{season}.csv"
