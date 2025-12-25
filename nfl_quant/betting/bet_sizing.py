"""
Bet Sizing with Kelly Criterion and Risk Management

Integrates:
1. Kelly criterion for optimal bet sizing
2. Configuration-driven thresholds (no hardcoding)
3. Risk management controls
"""

import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class BetSizer:
    """
    Calculate optimal bet sizes using Kelly criterion and config-driven parameters.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Load betting configuration."""
        if config_path is None:
            config_path = PROJECT_ROOT / 'configs' / 'betting_config.json'

        self.config = self._load_config(config_path)
        self.bankroll = self.config['bankroll_management']['initial_bankroll']
        self.kelly_fraction = self.config['bet_sizing']['kelly_fraction']
        self.max_bet_pct = self.config['bankroll_management']['max_bet_pct']
        self.min_edge_required = self.config['bankroll_management']['min_edge_required']
        self.unit_size = self.config['bet_sizing']['unit_size']

        logger.info(f"BetSizer initialized: Kelly={self.kelly_fraction}, MaxBet={self.max_bet_pct*100}%")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from JSON file."""
        if not config_path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return self._default_config()

        with open(config_path) as f:
            return json.load(f)

    def _default_config(self) -> Dict:
        """Default configuration if file not found."""
        return {
            'bankroll_management': {
                'initial_bankroll': 10000,
                'max_bet_pct': 0.05,
                'kelly_fraction': 0.25,
                'min_edge_required': 0.03,
            },
            'bet_sizing': {
                'kelly_fraction': 0.25,
                'max_units': 5,
                'min_units': 1,
                'unit_size': 100,
            },
            'model_deployment': {
                'high_confidence_max_stake': 3,
                'medium_confidence_max_stake': 2,
                'low_confidence_max_stake': 1,
            }
        }

    def american_to_decimal(self, odds: int) -> float:
        """Convert American odds to decimal odds."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    def calculate_kelly_bet(
        self,
        win_prob: float,
        odds: int,
        edge: float,
        confidence_level: str = 'medium'
    ) -> Dict:
        """
        Calculate optimal bet size using Kelly criterion.

        Args:
            win_prob: Model's probability of winning
            odds: American odds
            edge: Edge vs fair probability
            confidence_level: 'high', 'medium', or 'low'

        Returns:
            Dictionary with bet sizing information
        """
        # Check minimum edge requirement
        if edge < self.min_edge_required:
            return {
                'should_bet': False,
                'reason': f'Edge {edge:.2%} below minimum {self.min_edge_required:.2%}',
                'kelly_fraction': 0,
                'bet_amount': 0,
                'units': 0,
            }

        # Calculate full Kelly
        decimal_odds = self.american_to_decimal(odds)
        b = decimal_odds - 1  # Net profit per unit wagered
        p = win_prob
        q = 1 - p

        full_kelly = (p * b - q) / b if b > 0 else 0

        # Apply fractional Kelly
        kelly_bet_pct = max(0, full_kelly * self.kelly_fraction)

        # Apply confidence-based limits
        confidence_limits = self.config['model_deployment']
        if confidence_level == 'high':
            max_units = confidence_limits.get('high_confidence_max_stake', 3)
        elif confidence_level == 'medium':
            max_units = confidence_limits.get('medium_confidence_max_stake', 2)
        else:
            max_units = confidence_limits.get('low_confidence_max_stake', 1)

        # Cap Kelly bet at max percentage and confidence limit
        kelly_bet_pct = min(kelly_bet_pct, self.max_bet_pct)
        bet_amount = self.bankroll * kelly_bet_pct
        units = bet_amount / self.unit_size

        # Apply unit limits
        units = min(units, max_units)
        units = max(units, 0)
        bet_amount = units * self.unit_size

        return {
            'should_bet': units > 0,
            'reason': 'Kelly criterion bet' if units > 0 else 'Kelly too small',
            'full_kelly': full_kelly,
            'kelly_fraction_used': self.kelly_fraction,
            'kelly_bet_pct': kelly_bet_pct,
            'bet_amount': bet_amount,
            'units': units,
            'confidence_level': confidence_level,
            'edge': edge,
            'expected_value': (p * decimal_odds) - 1,
        }

    def get_edge_threshold_for_market(self, market: str) -> float:
        """Get minimum edge threshold for a specific market.

        Loads from configs/betting_config.json market_edge_thresholds section.
        """
        # Load from config first (preferred)
        market_thresholds = self.config.get('market_edge_thresholds', {})

        # Fallback defaults only if config is missing
        if not market_thresholds:
            market_thresholds = {
                'player_reception_yds': 0.03,
                'player_receptions': 0.03,
                'player_rush_yds': 0.04,
                'player_pass_yds': 0.05,
                'player_pass_tds': 0.08,
                'player_anytime_td': 0.08,
                'player_rush_attempts': 0.03,
                'default': 0.03,
            }

        default_threshold = market_thresholds.get('default', self.min_edge_required)
        return market_thresholds.get(market, default_threshold)

    def get_confidence_tier(self, edge: float, market: str = 'player_receptions') -> str:
        """
        Determine confidence tier based on edge and market.

        Uses config-driven thresholds, not hardcoded values.
        """
        min_edge = self.get_edge_threshold_for_market(market)

        # Tier thresholds relative to minimum edge
        high_threshold = min_edge * 2.0  # 2x minimum = high confidence
        medium_threshold = min_edge * 1.3  # 1.3x minimum = medium confidence

        if edge >= high_threshold:
            return 'high'
        elif edge >= medium_threshold:
            return 'medium'
        elif edge >= min_edge:
            return 'low'
        else:
            return 'no_bet'

    def update_bankroll(self, new_bankroll: float):
        """Update bankroll after wins/losses."""
        self.bankroll = new_bankroll
        logger.info(f"Bankroll updated to ${self.bankroll:.2f}")


# Singleton instance for easy access
_bet_sizer = None


def get_bet_sizer() -> BetSizer:
    """Get or create singleton BetSizer instance."""
    global _bet_sizer
    if _bet_sizer is None:
        _bet_sizer = BetSizer()
    return _bet_sizer


def calculate_bet_size(
    win_prob: float,
    odds: int,
    edge: float,
    market: str = 'player_receptions'
) -> Dict:
    """
    Convenience function to calculate bet size.

    Args:
        win_prob: Model's win probability
        odds: American odds
        edge: Edge vs fair probability
        market: Market type for threshold determination

    Returns:
        Bet sizing information
    """
    sizer = get_bet_sizer()
    confidence = sizer.get_confidence_tier(edge, market)
    return sizer.calculate_kelly_bet(win_prob, odds, edge, confidence)
