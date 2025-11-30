"""
Centralized Configuration Loader

Eliminates hardcoded values by loading all thresholds from config files.
"""

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class ConfigLoader:
    """
    Load and provide access to all configuration values.
    Eliminates hardcoded magic numbers throughout codebase.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.configs_dir = PROJECT_ROOT / 'configs'
        self._configs = {}
        self._load_all_configs()
        self._initialized = True

    def _load_all_configs(self):
        """Load all configuration files."""
        config_files = [
            'betting_config.json',
            'validation_config.json',
            'model_hyperparameters.json',
            'feature_engineering_config.json',
            'simulation_config.json',
        ]

        for config_file in config_files:
            path = self.configs_dir / config_file
            if path.exists():
                with open(path) as f:
                    config_name = config_file.replace('.json', '')
                    self._configs[config_name] = json.load(f)
                    logger.debug(f"Loaded {config_name}")
            else:
                logger.warning(f"Config file not found: {path}")

    def get(self, config_name: str, key_path: str = None, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            config_name: Name of config file (without .json)
            key_path: Dot-separated path to value (e.g., 'bankroll_management.max_bet_pct')
            default: Default value if not found

        Returns:
            Configuration value
        """
        if config_name not in self._configs:
            logger.warning(f"Config {config_name} not loaded")
            return default

        if key_path is None:
            return self._configs[config_name]

        # Navigate nested keys
        value = self._configs[config_name]
        for key in key_path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    # Convenience methods for common values

    def get_edge_threshold(self, market: str = 'default') -> float:
        """Get minimum edge threshold for a market."""
        thresholds = {
            'player_reception_yds': 0.03,
            'player_receptions': 0.03,
            'player_rush_yds': 0.04,
            'player_pass_yds': 0.05,
            'player_pass_tds': 0.08,
            'player_anytime_td': 0.08,
            'default': 0.03,
        }
        # Override with config if available
        min_edge = self.get('betting_config', 'bankroll_management.min_edge_required', 0.03)
        return thresholds.get(market, min_edge)

    def get_confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence tier thresholds."""
        return {
            'high': 0.08,  # 8% edge
            'medium': 0.05,  # 5% edge
            'low': 0.03,  # 3% edge
        }

    def get_kelly_fraction(self) -> float:
        """Get Kelly criterion fraction."""
        return self.get('betting_config', 'bet_sizing.kelly_fraction', 0.25)

    def get_max_bet_percentage(self) -> float:
        """Get maximum bet percentage of bankroll."""
        return self.get('betting_config', 'bankroll_management.max_bet_pct', 0.05)

    def get_validation_thresholds(self) -> Dict:
        """Get statistical validation thresholds."""
        return {
            'alpha': self.get('validation_config', 'statistical_tests.alpha', 0.05),
            'min_sample_size': self.get('validation_config', 'statistical_tests.min_sample_size', 200),
            'min_bets': self.get('validation_config', 'statistical_tests.min_bets_for_validation', 100),
        }

    def get_model_params(self, model_type: str = 'lightgbm') -> Dict:
        """Get model hyperparameters."""
        return self.get('model_hyperparameters', model_type, {})

    def get_simulation_params(self) -> Dict:
        """Get simulation parameters."""
        return self.get('simulation_config', None, {})


# Singleton accessor
_config_loader = None


def get_config() -> ConfigLoader:
    """Get singleton ConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


# Convenience functions to replace hardcoded values

def get_edge_threshold(market: str = 'default') -> float:
    """Get minimum edge threshold for market."""
    return get_config().get_edge_threshold(market)


def get_confidence_level(edge: float) -> str:
    """
    Determine confidence level based on edge.
    Replaces hardcoded '0.08 for high, 0.05 for medium' etc.
    """
    thresholds = get_config().get_confidence_thresholds()

    if edge >= thresholds['high']:
        return 'High'
    elif edge >= thresholds['medium']:
        return 'Medium'
    elif edge >= thresholds['low']:
        return 'Low'
    else:
        return 'No Bet'


def get_kelly_fraction() -> float:
    """Get Kelly fraction from config."""
    return get_config().get_kelly_fraction()


def get_validation_alpha() -> float:
    """Get statistical significance alpha level."""
    return get_config().get_validation_thresholds()['alpha']
