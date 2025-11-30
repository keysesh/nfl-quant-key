"""
NFL Regime Change Detection System

A sophisticated multi-dimensional detection system that identifies QB changes,
coaching shifts, roster impacts, and scheme changes to dynamically adjust
player projections.

Main Components:
- RegimeDetector: Detects all types of regime changes
- RegimeMetricsCalculator: Calculates regime-specific player metrics
- RegimeAwareProjector: Integrates regimes into projections
- RegimeReportGenerator: Generates reports and recommendations

Example Usage:
    from nfl_quant.regime import RegimeDetector, RegimeAwareProjector
    from nfl_quant.utils.season_utils import get_current_season

    # Initialize
    detector = RegimeDetector()
    projector = RegimeAwareProjector(detector=detector)

    # Detect regimes for a team
    result = detector.detect_all_regimes(
        team="KC",
        current_week=9,
        season=get_current_season(),
        pbp_df=pbp_data,
        player_stats_df=player_stats,
    )

    # Get regime-adjusted stats for a player
    stats = projector.get_regime_specific_stats(
        player_name="Patrick Mahomes",
        player_id="pm123",
        position="QB",
        team="KC",
        current_week=9,
        season=get_current_season(),
        pbp_df=pbp_data,
        player_stats_df=player_stats,
    )
"""

from .detector import RegimeDetector
from .metrics import RegimeMetricsCalculator
from .projections import RegimeAwareProjector, RegimeWindowManager
from .reports import RegimeReportGenerator, BettingRecommendationGenerator
from .schemas import (
    # Enums
    RegimeType,
    # Core models
    Regime,
    RegimeTrigger,
    RegimeWindow,
    RegimeDetectionResult,
    # Details
    QBChangeDetails,
    CoachingChangeDetails,
    RosterImpactDetails,
    SchemeChangeDetails,
    # Metrics
    UsageMetrics,
    EfficiencyMetrics,
    ContextMetrics,
    RegimeMetrics,
    RegimeComparison,
    RegimeImpact,
    # Profiles
    PlayerRegimeProfile,
    TeamRegimeSummary,
    # Projections
    ProjectionAdjustment,
    BettingRecommendation,
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "RegimeDetector",
    "RegimeMetricsCalculator",
    "RegimeAwareProjector",
    "RegimeWindowManager",
    "RegimeReportGenerator",
    "BettingRecommendationGenerator",
    # Enums
    "RegimeType",
    # Core models
    "Regime",
    "RegimeTrigger",
    "RegimeWindow",
    "RegimeDetectionResult",
    # Details
    "QBChangeDetails",
    "CoachingChangeDetails",
    "RosterImpactDetails",
    "SchemeChangeDetails",
    # Metrics
    "UsageMetrics",
    "EfficiencyMetrics",
    "ContextMetrics",
    "RegimeMetrics",
    "RegimeComparison",
    "RegimeImpact",
    # Profiles
    "PlayerRegimeProfile",
    "TeamRegimeSummary",
    # Projections
    "ProjectionAdjustment",
    "BettingRecommendation",
]
