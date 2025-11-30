"""
NFL Quant Models
"""

from .usage_predictor import UsagePredictor
from .efficiency_predictor import EfficiencyPredictor
from .td_predictor import TouchdownPredictor
from .ensemble_predictor import (
    EnsemblePredictor,
    create_simple_ensemble,
    train_stacking_ensemble,
    compute_ensemble_weights_from_cv
)

__all__ = [
    'UsagePredictor',
    'EfficiencyPredictor',
    'TouchdownPredictor',
    'EnsemblePredictor',
    'create_simple_ensemble',
    'train_stacking_ensemble',
    'compute_ensemble_weights_from_cv',
]
