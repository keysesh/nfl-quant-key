"""Enhanced Configuration System - Centralized JSON-based configuration loading.

This module extends the base config.py with JSON configuration loading
for all hardcoded values across the system.

Usage:
    from nfl_quant.config_enhanced import config

    # Access simulation parameters
    trials = config.simulation.trials.default_trials
    td_rate = config.simulation.game_simulation.base_td_rate

    # Access model hyperparameters
    learning_rate = config.models.usage_predictor.learning_rate
    max_depth = config.models.efficiency_predictor.max_depth

    # Access TD prediction parameters
    qb_td_rate = config.td_prediction.baseline_td_rates['QB']['passing_td_per_attempt']

    # Access betting thresholds
    min_confidence = config.betting.confidence_levels.high
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from nfl_quant.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    game_simulation: Dict[str, Any] = field(default_factory=dict)
    game_adjustments: Dict[str, Any] = field(default_factory=dict)
    weather_adjustments: Dict[str, Any] = field(default_factory=dict)
    player_simulation_variance: Dict[str, Any] = field(default_factory=dict)
    mean_adjustments: Dict[str, Any] = field(default_factory=dict)
    completion_bounds: Dict[str, Any] = field(default_factory=dict)
    blending_weights: Dict[str, Any] = field(default_factory=dict)
    matchup_adjustments: Dict[str, Any] = field(default_factory=dict)
    calibration_dampening: Dict[str, Any] = field(default_factory=dict)
    trials: Dict[str, Any] = field(default_factory=dict)
    league_defaults: Dict[str, Any] = field(default_factory=dict)
    game_line_betting: Dict[str, Any] = field(default_factory=dict)  # Added for game line recommendations


@dataclass
class ModelHyperparamsConfig:
    """Model training hyperparameters."""
    usage_predictor: Dict[str, Any] = field(default_factory=dict)
    efficiency_predictor: Dict[str, Any] = field(default_factory=dict)
    training_thresholds: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TDPredictionConfig:
    """Touchdown prediction parameters."""
    baseline_td_rates: Dict[str, Any] = field(default_factory=dict)
    red_zone_multipliers: Dict[str, Any] = field(default_factory=dict)
    game_script_sensitivity: Dict[str, Any] = field(default_factory=dict)
    game_script_bounds: Dict[str, Any] = field(default_factory=dict)
    scoring_environment: Dict[str, Any] = field(default_factory=dict)
    shrinkage: Dict[str, Any] = field(default_factory=dict)
    league_averages: Dict[str, Any] = field(default_factory=dict)
    usage_scenarios_by_position: Dict[str, Any] = field(default_factory=dict)
    probability_bounds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering parameters."""
    injury_adjustments: Dict[str, Any] = field(default_factory=dict)
    defensive_matchup_adjustments: Dict[str, Any] = field(default_factory=dict)
    contextual_factors: Dict[str, Any] = field(default_factory=dict)
    correlation_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Data validation thresholds."""
    min_thresholds: Dict[str, Any] = field(default_factory=dict)
    validation_bounds: Dict[str, Any] = field(default_factory=dict)
    activity_thresholds: Dict[str, Any] = field(default_factory=dict)
    team_level_validation: Dict[str, Any] = field(default_factory=dict)
    error_tolerance: Dict[str, Any] = field(default_factory=dict)
    data_quality_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BettingConfig:
    """Betting recommendation thresholds."""
    confidence_levels: Dict[str, float] = field(default_factory=dict)
    edge_requirements: Dict[str, float] = field(default_factory=dict)
    parlay_optimization: Dict[str, Any] = field(default_factory=dict)
    performance_adjustment: Dict[str, Any] = field(default_factory=dict)
    risk_shift_adjustments: Dict[str, Any] = field(default_factory=dict)
    bet_sizing: Dict[str, Any] = field(default_factory=dict)
    recommendation_filters: Dict[str, Any] = field(default_factory=dict)
    clv_tracking: Dict[str, Any] = field(default_factory=dict)


class EnhancedConfig:
    """
    Enhanced configuration system with JSON-based config loading.

    All hardcoded values are now centralized in configs/*.json files.
    """

    def __init__(self):
        """Initialize and load all configuration files."""
        self.configs_dir = settings.CONFIGS_DIR

        # Load all config files
        self.simulation = self._load_simulation_config()
        self.models = self._load_model_hyperparams()
        self.td_prediction = self._load_td_prediction_config()
        self.feature_engineering = self._load_feature_engineering_config()
        self.validation = self._load_validation_config()
        self.betting = self._load_betting_config()

        logger.info("✅ Enhanced configuration loaded successfully")

    def _load_json(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """
        Load a JSON configuration file.

        Args:
            filename: Name of JSON file (e.g., 'simulation_config.json')
            required: If True, raise error if file doesn't exist

        Returns:
            Dictionary of configuration values
        """
        filepath = self.configs_dir / filename

        if not filepath.exists():
            if required:
                raise FileNotFoundError(
                    f"Required config file not found: {filepath}\n"
                    f"Please create this file or run the configuration setup script."
                )
            else:
                logger.warning(f"Optional config file not found: {filepath}")
                return {}

        try:
            with open(filepath) as f:
                data = json.load(f)
                logger.debug(f"Loaded config: {filename}")
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")

    def _load_simulation_config(self) -> SimulationConfig:
        """Load simulation configuration."""
        data = self._load_json('simulation_config.json', required=True)
        return SimulationConfig(
            game_simulation=data.get('game_simulation', {}),
            game_adjustments=data.get('game_adjustments', {}),
            weather_adjustments=data.get('weather_adjustments', {}),
            player_simulation_variance=data.get('player_simulation_variance', {}),
            mean_adjustments=data.get('mean_adjustments', {}),
            completion_bounds=data.get('completion_bounds', {}),
            blending_weights=data.get('blending_weights', {}),
            matchup_adjustments=data.get('matchup_adjustments', {}),
            calibration_dampening=data.get('calibration_dampening', {}),
            trials=data.get('trials', {}),
            league_defaults=data.get('league_defaults', {}),
            game_line_betting=data.get('game_line_betting', {}),
        )

    def _load_model_hyperparams(self) -> ModelHyperparamsConfig:
        """Load model hyperparameters."""
        data = self._load_json('model_hyperparams.json', required=True)
        return ModelHyperparamsConfig(
            usage_predictor=data.get('usage_predictor', {}),
            efficiency_predictor=data.get('efficiency_predictor', {}),
            training_thresholds=data.get('training_thresholds', {}),
            validation=data.get('validation', {}),
        )

    def _load_td_prediction_config(self) -> TDPredictionConfig:
        """Load TD prediction configuration."""
        data = self._load_json('td_prediction_config.json', required=True)
        return TDPredictionConfig(
            baseline_td_rates=data.get('baseline_td_rates', {}),
            red_zone_multipliers=data.get('red_zone_multipliers', {}),
            game_script_sensitivity=data.get('game_script_sensitivity', {}),
            game_script_bounds=data.get('game_script_bounds', {}),
            scoring_environment=data.get('scoring_environment', {}),
            shrinkage=data.get('shrinkage', {}),
            league_averages=data.get('league_averages', {}),
            usage_scenarios_by_position=data.get('usage_scenarios_by_position', {}),
            probability_bounds=data.get('probability_bounds', {}),
        )

    def _load_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """Load feature engineering configuration."""
        data = self._load_json('feature_engineering_config.json', required=True)
        return FeatureEngineeringConfig(
            injury_adjustments=data.get('injury_adjustments', {}),
            defensive_matchup_adjustments=data.get('defensive_matchup_adjustments', {}),
            contextual_factors=data.get('contextual_factors', {}),
            correlation_factors=data.get('correlation_factors', {}),
        )

    def _load_validation_config(self) -> ValidationConfig:
        """Load validation configuration."""
        data = self._load_json('validation_thresholds.json', required=True)
        return ValidationConfig(
            min_thresholds=data.get('min_thresholds', {}),
            validation_bounds=data.get('validation_bounds', {}),
            activity_thresholds=data.get('activity_thresholds', {}),
            team_level_validation=data.get('team_level_validation', {}),
            error_tolerance=data.get('error_tolerance', {}),
            data_quality_flags=data.get('data_quality_flags', {}),
        )

    def _load_betting_config(self) -> BettingConfig:
        """Load betting configuration."""
        data = self._load_json('betting_thresholds.json', required=True)
        return BettingConfig(
            confidence_levels=data.get('confidence_levels', {}),
            edge_requirements=data.get('edge_requirements', {}),
            parlay_optimization=data.get('parlay_optimization', {}),
            performance_adjustment=data.get('performance_adjustment', {}),
            risk_shift_adjustments=data.get('risk_shift_adjustments', {}),
            bet_sizing=data.get('bet_sizing', {}),
            recommendation_filters=data.get('recommendation_filters', {}),
            clv_tracking=data.get('clv_tracking', {}),
        )

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.

        Args:
            path: Dot-separated path (e.g., 'simulation.trials.default_trials')
            default: Default value if path not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('simulation.trials.default_trials')
            50000
            >>> config.get('models.usage_predictor.learning_rate')
            0.1
        """
        parts = path.split('.')
        value = self

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return default

            if value is None:
                return default

        return value

    def reload(self):
        """Reload all configuration files from disk."""
        logger.info("Reloading configuration from disk...")
        self.__init__()

    def validate(self) -> bool:
        """
        Validate all configuration values.

        Returns:
            True if all validations pass

        Raises:
            ValueError: If any validation fails
        """
        # Check simulation config
        assert self.simulation.trials['default_trials'] > 0, "default_trials must be positive"
        assert 0 <= self.simulation.game_simulation['base_td_rate'] <= 1, "TD rate must be probability"

        # Check model hyperparams
        assert self.models.usage_predictor['learning_rate'] > 0, "learning_rate must be positive"
        assert self.models.usage_predictor['max_depth'] > 0, "max_depth must be positive"

        # Check TD prediction
        for pos, rates in self.td_prediction.baseline_td_rates.items():
            for rate_name, rate_value in rates.items():
                if rate_name.startswith('_'):
                    continue
                assert 0 <= rate_value <= 1, f"{pos} {rate_name} must be probability"

        # Check betting thresholds
        for level, conf in self.td_prediction.baseline_td_rates.items():
            if level.startswith('_'):
                continue
            assert 0 <= conf <= 1, f"Confidence level {level} must be probability"

        logger.info("✅ Configuration validation passed")
        return True


# Global config instance
config = EnhancedConfig()

# Backwards compatibility: expose commonly used values at module level
DEFAULT_TRIALS = config.simulation.trials.get('default_trials', 50000)
DEFAULT_SEED = config.simulation.trials.get('default_seed', 42)
