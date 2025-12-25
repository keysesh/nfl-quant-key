"""
NFL QUANT Configuration Package.

Provides centralized configuration for all model and pipeline settings.
All version info and features are defined in model_config.py.
"""
from configs.model_config import (
    # Version info
    MODEL_VERSION,
    MODEL_VERSION_FULL,
    MODEL_VERSION_LOWER,
    # Feature configuration
    FEATURES,
    FEATURE_COUNT,
    CURRENT_FEATURE_COLS,
    # Backward-compat aliases (will deprecate)
    V12_FEATURE_COLS,
    V17_FEATURE_COLS,
    V18_FEATURE_COLS,
    V19_FEATURE_COLS,
    # Sweet spot configuration
    SWEET_SPOT_PARAMS,
    smooth_sweet_spot,
    # Feature flags
    FEATURE_FLAGS,
    # Market configuration
    SUPPORTED_MARKETS,
    CLASSIFIER_MARKETS,
    SIMULATOR_MARKETS,
    MARKET_TO_STAT,
    MARKET_TO_PREDICTION_COLS,
    is_market_enabled,
    get_enabled_markets,
    get_market_disabled_reason,
    # Path helpers
    get_model_path,
    get_active_model_path,
    get_versioned_model_path,
    # Model params
    MODEL_PARAMS,
    # Validation helpers
    get_monotonic_constraints,
    validate_features,
)

__all__ = [
    # Version info
    'MODEL_VERSION',
    'MODEL_VERSION_FULL',
    'MODEL_VERSION_LOWER',
    # Feature configuration
    'FEATURES',
    'FEATURE_COUNT',
    'CURRENT_FEATURE_COLS',
    # Backward-compat aliases
    'V12_FEATURE_COLS',
    'V17_FEATURE_COLS',
    'V18_FEATURE_COLS',
    'V19_FEATURE_COLS',
    # Sweet spot
    'SWEET_SPOT_PARAMS',
    'smooth_sweet_spot',
    # Feature flags
    'FEATURE_FLAGS',
    # Markets
    'SUPPORTED_MARKETS',
    'CLASSIFIER_MARKETS',
    'SIMULATOR_MARKETS',
    'MARKET_TO_STAT',
    'MARKET_TO_PREDICTION_COLS',
    'is_market_enabled',
    'get_enabled_markets',
    'get_market_disabled_reason',
    # Paths
    'get_model_path',
    'get_active_model_path',
    'get_versioned_model_path',
    # Params
    'MODEL_PARAMS',
    # Validation
    'get_monotonic_constraints',
    'validate_features',
]
