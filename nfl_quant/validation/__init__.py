"""
Validation and calibration metrics for NFL predictions.
"""

from .calibration_metrics import CalibrationMetrics
from .temporal_cv import (
    TemporalCrossValidator,
    TemporalHyperparameterTuner,
    CVFold,
    evaluate_model_temporal,
    get_optimal_window_size,
)
from .statistical_bounds import (
    validate_projection,
    validate_player_projections,
    check_projection_sanity,
)

__all__ = [
    'CalibrationMetrics',
    'TemporalCrossValidator',
    'TemporalHyperparameterTuner',
    'CVFold',
    'evaluate_model_temporal',
    'get_optimal_window_size',
    'validate_projection',
    'validate_player_projections',
    'check_projection_sanity',
]
