"""
Player Simulator - Main Entry Point

Uses V4 backend with proper statistical distributions:
- NegativeBinomial for usage (overdispersed counts)
- Lognormal for efficiency (positive, right-skewed)
- Gaussian Copula for correlation

This is THE simulator to use. No version numbers needed.
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from nfl_quant.config_paths import USAGE_PREDICTOR_FILE, EFFICIENCY_PREDICTOR_FILE
from nfl_quant.simulation.player_simulator_v4 import PlayerSimulatorV4
from nfl_quant.models.usage_predictor import UsagePredictor
from nfl_quant.models.efficiency_predictor import EfficiencyPredictor
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator

logger = logging.getLogger(__name__)


class PlayerSimulator(PlayerSimulatorV4):
    """
    Main player simulator using V4 backend.

    Features:
    - NegativeBinomial for targets/carries/attempts (overdispersed counts)
    - Lognormal for yards per opportunity (positive, right-skewed)
    - Gaussian Copula for target-efficiency correlation
    - Proper variance modeling (CV ~0.4-0.5, not 0.95)
    """

    def __init__(
        self,
        usage_predictor: UsagePredictor,
        efficiency_predictor: EfficiencyPredictor,
        trials: Optional[int] = None,
        seed: Optional[int] = None,
        calibrator: Optional[NFLProbabilityCalibrator] = None,
        td_calibrator: Optional[object] = None,
        **kwargs
    ):
        # V4 doesn't use calibrator/td_calibrator in constructor, ignore them
        super().__init__(
            usage_predictor=usage_predictor,
            efficiency_predictor=efficiency_predictor,
            trials=trials or 50000,
            seed=seed or 42,
        )

        logger.info("PlayerSimulator initialized (NegBin + Lognormal + Copula)")

    # simulate_player() is inherited from PlayerSimulatorV4


def load_predictors(
    usage_model_path: Optional[str] = None,
    efficiency_model_path: Optional[str] = None
) -> Tuple[UsagePredictor, EfficiencyPredictor]:
    """Load usage and efficiency predictors (backward compatible)."""
    if usage_model_path is None:
        usage_model_path = USAGE_PREDICTOR_FILE
    if efficiency_model_path is None:
        efficiency_model_path = EFFICIENCY_PREDICTOR_FILE

    usage_predictor = UsagePredictor()
    usage_predictor.load(str(usage_model_path))

    efficiency_predictor = EfficiencyPredictor()
    efficiency_predictor.load(str(efficiency_model_path))

    logger.info(f"Loaded usage predictor from {usage_model_path}")
    logger.info(f"Loaded efficiency predictor from {efficiency_model_path}")

    return usage_predictor, efficiency_predictor


__all__ = ['PlayerSimulator', 'load_predictors']
