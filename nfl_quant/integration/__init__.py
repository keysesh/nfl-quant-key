"""
NFL QUANT Integration Module.

Provides bridges between different prediction systems:
- ModelSimulatorBridge: Combines MC simulator with XGBoost classifier
- td_model_bridge: Poisson model for TD props
"""

from nfl_quant.integration.model_simulator_bridge import (
    ModelSimulatorBridge,
    get_unified_prediction,
)

from nfl_quant.integration.td_model_bridge import (
    get_td_prediction,
    is_td_market,
    TD_MARKETS,
)

__all__ = [
    'ModelSimulatorBridge',
    'get_unified_prediction',
    'get_td_prediction',
    'is_td_market',
    'TD_MARKETS',
]
