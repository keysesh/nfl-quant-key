"""
NFL QUANT Edge Detection Module

This module implements the two-edge betting strategy:
1. LVT Edge: Statistical reversion (line vs trailing performance)
2. Player Bias Edge: Player-specific tendencies

The edges are kept separate to preserve signal purity.
See: validate_two_edge_hypothesis.py for validation of independence.
"""

from .base_edge import BaseEdge
from .lvt_edge import LVTEdge
from .player_bias_edge import PlayerBiasEdge
from .ensemble import EdgeEnsemble
from .edge_calibrator import EdgeCalibrator

__all__ = [
    'BaseEdge',
    'LVTEdge',
    'PlayerBiasEdge',
    'EdgeEnsemble',
    'EdgeCalibrator',
]
