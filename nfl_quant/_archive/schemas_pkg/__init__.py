"""
NFL QUANT Schemas Package

Additional Pydantic models for data validation and standardized output.

Note: Core schemas (TeamWeekFeatures, SimulationInput, etc.) are in nfl_quant.schemas (file).
This package (nfl_quant.schemas_pkg/) contains additional schema modules including unified output.
"""

# Import unified output schemas
from .unified_output import (
    BetType,
    ConfidenceTier,
    UnifiedBetRecommendation,
    UnifiedPipelineOutput,
    create_player_prop_recommendation,
    create_game_line_recommendation,
)

# Import matchup schemas
try:
    from .matchup_schemas import *
except ImportError:
    pass

__all__ = [
    'BetType',
    'ConfidenceTier',
    'UnifiedBetRecommendation',
    'UnifiedPipelineOutput',
    'create_player_prop_recommendation',
    'create_game_line_recommendation',
]
