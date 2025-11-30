"""
NFL QUANT Feature Engineering

This module provides the SINGLE SOURCE OF TRUTH for all feature calculations.
All scripts must import from here - no inline feature calculations allowed.

Usage:
    from nfl_quant.features import get_feature_engine

    engine = get_feature_engine()
    features = engine.extract_features_for_bet(...)

Or use convenience functions:
    from nfl_quant.features import (
        calculate_trailing_stat,
        get_rush_defense_epa,
        get_pass_defense_epa
    )
"""

from nfl_quant.features.core import (
    # Core engine
    FeatureEngine,
    get_feature_engine,

    # Trailing stats
    calculate_trailing_stat,
    calculate_all_trailing_stats,

    # Defense EPA
    get_rush_defense_epa,
    get_pass_defense_epa,

    # Route metrics
    calculate_tprr,
    calculate_yards_per_route,
    estimate_routes_run,
    calculate_route_participation,
)



