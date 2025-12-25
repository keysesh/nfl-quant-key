"""NFL Quantitative Analytics Pipeline.

A production-grade Python package for NFL data fetching, feature engineering,
Monte Carlo simulation, odds pricing, and bet sizing using nflverse.
"""

__version__ = "0.1.0"
__author__ = "NFL Quant Team"

from nfl_quant.data.fetcher import DataFetcher
from nfl_quant.features.core import FeatureEngine
from nfl_quant.simulation.simulator import MonteCarloSimulator

__all__ = ["DataFetcher", "FeatureEngine", "MonteCarloSimulator"]



