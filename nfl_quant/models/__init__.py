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
from .qb_passing_model import QBPassingModel, load_qb_model
from .elo_ratings import (
    EloRatingSystem,
    load_elo_ratings,
    initialize_elo_from_nflverse,
    get_elo_system,
)
from .baseline_regression import (
    BaselineLogistic,
    BaselineLinear,
    ModelComparison,
    compare_to_xgboost,
    train_baseline_for_market,
)

__all__ = [
    'UsagePredictor',
    'EfficiencyPredictor',
    'TouchdownPredictor',
    'EnsemblePredictor',
    'create_simple_ensemble',
    'train_stacking_ensemble',
    'compute_ensemble_weights_from_cv',
    'QBPassingModel',
    'load_qb_model',
    # Elo rating system
    'EloRatingSystem',
    'load_elo_ratings',
    'initialize_elo_from_nflverse',
    'get_elo_system',
    # Baseline regression (V28)
    'BaselineLogistic',
    'BaselineLinear',
    'ModelComparison',
    'compare_to_xgboost',
    'train_baseline_for_market',
]
