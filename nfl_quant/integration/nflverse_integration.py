"""
NFLverse Integration Layer

Connects the new NFLverse feature extractors with the existing prediction pipeline.
This module bridges the gap between:
- Old: PlayerSimulator, TrailingStatsExtractor, EPA features
- New: NGS features, FF Opportunity, Snap Counts, Position-Market Calibration

Usage:
    from nfl_quant.integration.nflverse_integration import NFLverseIntegrator

    integrator = NFLverseIntegrator()
    enhanced_prediction = integrator.enhance_player_prediction(
        player_name, position, team, market_type, base_prediction, season, week
    )
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


class NFLverseIntegrator:
    """
    Integrates new NFLverse features with existing prediction pipeline.

    This class provides a single interface for:
    1. Adding advanced features to existing predictions
    2. Applying position-specific calibration
    3. Computing regression adjustments
    4. Calculating proper bet sizing
    """

    def __init__(self, data_dir: Path = None, config_dir: Path = None):
        """
        Initialize the integrator.

        Args:
            data_dir: Path to nflverse data
            config_dir: Path to config files
        """
        if data_dir is None:
            data_dir = Path('data/nflverse')
        if config_dir is None:
            config_dir = Path('configs')

        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)

        # Lazy-loaded feature extractors
        self._ngs_extractor = None
        self._ff_opp_extractor = None
        self._snap_extractor = None
        self._hist_provider = None
        self._calibrator = None
        self._thresholds = None

        # Cache for performance
        self._feature_cache = {}

        logger.info("NFLverseIntegrator initialized")

    @property
    def ngs_extractor(self):
        """Lazy load NGS feature extractor."""
        if self._ngs_extractor is None:
            try:
                from nfl_quant.features.ngs_features import NGSFeatureExtractor
                self._ngs_extractor = NGSFeatureExtractor(self.data_dir)
            except Exception as e:
                logger.warning(f"Could not load NGS extractor: {e}")
                self._ngs_extractor = "unavailable"
        return self._ngs_extractor if self._ngs_extractor != "unavailable" else None

    @property
    def ff_opp_extractor(self):
        """Lazy load FF Opportunity extractor."""
        if self._ff_opp_extractor is None:
            try:
                from nfl_quant.features.ff_opportunity_features import FFOpportunityFeatureExtractor
                self._ff_opp_extractor = FFOpportunityFeatureExtractor(self.data_dir)
            except Exception as e:
                logger.warning(f"Could not load FF Opportunity extractor: {e}")
                self._ff_opp_extractor = "unavailable"
        return self._ff_opp_extractor if self._ff_opp_extractor != "unavailable" else None

    @property
    def snap_extractor(self):
        """Lazy load Snap Count extractor."""
        if self._snap_extractor is None:
            try:
                from nfl_quant.features.snap_count_features import SnapCountFeatureExtractor
                self._snap_extractor = SnapCountFeatureExtractor(self.data_dir)
            except Exception as e:
                logger.warning(f"Could not load Snap Count extractor: {e}")
                self._snap_extractor = "unavailable"
        return self._snap_extractor if self._snap_extractor != "unavailable" else None

    @property
    def hist_provider(self):
        """Lazy load Historical Baseline provider."""
        if self._hist_provider is None:
            try:
                from nfl_quant.features.historical_baseline import HistoricalBaselineProvider
                self._hist_provider = HistoricalBaselineProvider(self.data_dir)
            except Exception as e:
                logger.warning(f"Could not load Historical Baseline provider: {e}")
                self._hist_provider = "unavailable"
        return self._hist_provider if self._hist_provider != "unavailable" else None

    @property
    def calibrator(self):
        """Lazy load Position-Market Calibrator."""
        if self._calibrator is None:
            try:
                from nfl_quant.calibration.position_market_calibrator import PositionMarketCalibrator
                cal_path = self.config_dir / 'position_market_calibrator.json'
                if cal_path.exists():
                    cal = PositionMarketCalibrator()
                    cal.load(cal_path)
                    self._calibrator = cal
                else:
                    logger.warning(f"Calibrator not found at {cal_path}")
                    self._calibrator = "unavailable"
            except Exception as e:
                logger.warning(f"Could not load calibrator: {e}")
                self._calibrator = "unavailable"
        return self._calibrator if self._calibrator != "unavailable" else None

    @property
    def thresholds(self):
        """Lazy load position-market thresholds."""
        if self._thresholds is None:
            thresh_path = self.config_dir / 'position_market_thresholds.json'
            if thresh_path.exists():
                with open(thresh_path, 'r') as f:
                    self._thresholds = json.load(f)
            else:
                self._thresholds = {}
        return self._thresholds

    def enhance_player_prediction(
        self,
        player_name: str,
        position: str,
        team: str,
        market_type: str,
        base_prediction: float,
        season: int,
        week: int
    ) -> Dict[str, Any]:
        """
        Enhance a base prediction with all NFLverse features.

        Args:
            player_name: Player name
            position: Position (QB, RB, WR, TE)
            team: Team abbreviation
            market_type: Market type (player_pass_yds, etc.)
            base_prediction: Base model prediction (e.g., projected yards)
            season: Current season
            week: Current week

        Returns:
            Dict with enhanced prediction and all features
        """
        cache_key = f"{player_name}_{position}_{market_type}_{season}_{week}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        result = {
            'player_name': player_name,
            'position': position,
            'team': team,
            'market_type': market_type,
            'base_prediction': base_prediction,
            'adjusted_prediction': base_prediction,
            'features': {},
        }

        # 1. Add NGS features
        if self.ngs_extractor:
            try:
                ngs_features = self.ngs_extractor.get_player_features(
                    player_name, position, season, week
                )
                result['features'].update({f'ngs_{k}': v for k, v in ngs_features.items()})

                # Get skill score
                skill_score = self.ngs_extractor.calculate_skill_score(
                    player_name, position, season, week
                )
                result['features']['ngs_skill_score'] = skill_score
            except Exception as e:
                logger.debug(f"NGS features error for {player_name}: {e}")

        # 2. Add FF Opportunity features
        if self.ff_opp_extractor:
            try:
                opp_features = self.ff_opp_extractor.get_player_opportunity_features(
                    player_name, season, week
                )
                result['features'].update({f'ffo_{k}': v for k, v in opp_features.items()})

                # Get regression adjustment
                regression_adj = self.ff_opp_extractor.get_betting_adjustment(
                    player_name, season, week, market_type
                )
                result['features']['ffo_regression_adjustment'] = regression_adj
            except Exception as e:
                logger.debug(f"FF Opportunity error for {player_name}: {e}")

        # 3. Add Snap Count features
        if self.snap_extractor:
            try:
                snap_features = self.snap_extractor.get_usage_prediction_features(
                    player_name, position, season, week
                )
                result['features'].update({f'snap_{k}': v for k, v in snap_features.items()})
            except Exception as e:
                logger.debug(f"Snap count error for {player_name}: {e}")

        # 4. Apply mean bias correction
        bias_correction = self._get_mean_bias_correction(position, market_type)
        result['adjusted_prediction'] = base_prediction + bias_correction
        result['features']['bias_correction'] = bias_correction

        # 5. Apply NGS-based adjustment
        skill_score = result['features'].get('ngs_skill_score', 0.0)
        skill_factor = 1.0 + (skill_score * 0.05)  # Up to ±5%
        result['adjusted_prediction'] *= skill_factor

        # 6. Apply regression adjustment
        regression_adj = result['features'].get('ffo_regression_adjustment', 0.0)
        regression_factor = 1.0 + (regression_adj * 0.1)  # Up to ±10%
        result['adjusted_prediction'] *= regression_factor

        # 7. Apply snap trend adjustment
        snap_trend = result['features'].get('snap_snap_trend', 0.0)
        if snap_trend > 0.05:  # Increasing usage
            result['adjusted_prediction'] *= 1.05
        elif snap_trend < -0.05:  # Decreasing usage
            result['adjusted_prediction'] *= 0.95

        self._feature_cache[cache_key] = result
        return result

    def _get_mean_bias_correction(self, position: str, market_type: str) -> float:
        """Get mean bias correction for position-market."""
        if self.calibrator:
            return self.calibrator.get_mean_bias_correction(position, market_type)

        # Fallback to config
        corrections = self.thresholds.get('mean_bias_corrections', {})
        key = f"{position}_{market_type}"
        return corrections.get(key, 0.0)

    def calibrate_probability(
        self,
        raw_prob: float,
        position: str,
        market_type: str
    ) -> float:
        """
        Calibrate raw model probability.

        Args:
            raw_prob: Raw model probability
            position: Player position
            market_type: Market type

        Returns:
            Calibrated probability
        """
        if self.calibrator:
            return self.calibrator.transform(raw_prob, position, market_type)

        # Fallback shrinkage
        return 0.5 + (raw_prob - 0.5) * 0.6

    def get_edge_threshold(self, position: str, market_type: str) -> float:
        """Get minimum edge threshold for position-market."""
        if self.calibrator:
            return self.calibrator.get_edge_threshold(position, market_type)

        # Fallback to config
        edge_thresholds = self.thresholds.get('edge_thresholds', {})
        if position in edge_thresholds and market_type in edge_thresholds[position]:
            return edge_thresholds[position][market_type].get('min_edge', 0.10)

        # Default thresholds
        defaults = {
            'player_pass_yds': 0.05,
            'player_rush_yds': 0.20,
            'player_reception_yds': 0.10,
            'player_receptions': 0.10,
            'player_pass_tds': 0.10,
            'player_anytime_td': 0.20,
        }
        return defaults.get(market_type, 0.10)

    def get_historical_baseline(
        self,
        player_name: str,
        market_type: str,
        line: float,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """Get historical baseline for a player prop."""
        if self.hist_provider:
            return self.hist_provider.get_historical_baseline(
                player_name, market_type, line, season, week
            )

        # Default empty baseline
        return {
            'hist_count': 0,
            'hist_over_count': 0,
            'hist_over_rate': 0.0,
            'hist_avg': 0.0,
            'hist_avg_vs_line': 0.0,
            'hist_std': 0.0,
        }

    def calculate_kelly_fraction(
        self,
        calibrated_prob: float,
        american_odds: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly criterion bet sizing.

        Args:
            calibrated_prob: Calibrated win probability
            american_odds: American odds (e.g., -110, +150)
            fraction: Kelly fraction (default 0.25 for quarter Kelly)

        Returns:
            Fraction of bankroll to bet
        """
        # Convert American odds to decimal
        if american_odds > 0:
            decimal_odds = 1 + (american_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(american_odds))

        b = decimal_odds - 1
        p = calibrated_prob
        q = 1 - p

        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p - q) / b if b > 0 else 0

        # Apply fraction and cap
        kelly_fraction = kelly * fraction
        kelly_fraction = min(kelly_fraction, 0.05)  # Max 5% of bankroll
        kelly_fraction = max(kelly_fraction, 0)

        return kelly_fraction

    def should_bet(
        self,
        edge_pct: float,
        position: str,
        market_type: str,
        calibrated_prob: float
    ) -> bool:
        """
        Determine if a bet should be placed based on thresholds.

        Args:
            edge_pct: Edge percentage (as decimal, e.g., 0.10 for 10%)
            position: Player position
            market_type: Market type
            calibrated_prob: Calibrated probability

        Returns:
            True if bet meets criteria, False otherwise
        """
        min_threshold = self.get_edge_threshold(position, market_type)

        # Must meet minimum edge threshold
        if edge_pct < min_threshold:
            return False

        # Must have reasonable calibrated probability
        if calibrated_prob < 0.51:
            return False

        return True

    def get_confidence_tier(
        self,
        edge_pct: float,
        position: str,
        market_type: str,
        calibrated_prob: float
    ) -> str:
        """
        Get confidence tier for a bet.

        Args:
            edge_pct: Edge percentage
            position: Player position
            market_type: Market type
            calibrated_prob: Calibrated probability

        Returns:
            'High', 'Medium', or 'Low'
        """
        threshold = self.get_edge_threshold(position, market_type)

        if edge_pct >= threshold * 1.5 and calibrated_prob >= 0.55:
            return 'High'
        elif edge_pct >= threshold and calibrated_prob >= 0.52:
            return 'Medium'
        else:
            return 'Low'


# Global instance for easy access
_integrator = None


def get_integrator() -> NFLverseIntegrator:
    """Get global NFLverseIntegrator instance."""
    global _integrator
    if _integrator is None:
        _integrator = NFLverseIntegrator()
    return _integrator


def enhance_prediction(
    player_name: str,
    position: str,
    team: str,
    market_type: str,
    base_prediction: float,
    season: int = 2025,
    week: int = 11
) -> Dict[str, Any]:
    """
    Quick function to enhance a single prediction.

    Args:
        player_name: Player name
        position: Position
        team: Team
        market_type: Market type
        base_prediction: Base prediction
        season: Season
        week: Week

    Returns:
        Enhanced prediction dict
    """
    integrator = get_integrator()
    return integrator.enhance_player_prediction(
        player_name, position, team, market_type, base_prediction, season, week
    )
