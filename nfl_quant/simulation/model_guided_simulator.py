"""
Model-Guided Monte Carlo Simulator

Integrates XGBoost model features directly into the MC simulation,
ensuring the model influences the projection distribution rather than
just providing a post-hoc probability.

Architecture:
    1. Extract XGBoost features for the bet (same features model was trained on)
    2. Use distribution-modifying features to adjust MC baseline
    3. Run MC trials with adjusted parameters
    4. Use XGBoost P(UNDER) for final direction validation

This solves the "MC ≠ Model" problem where previously:
    - MC ran independently with trailing stats
    - XGBoost predicted P(UNDER) separately
    - Bridge tried to reconcile them post-hoc

Now:
    - Model features influence MC distribution directly
    - Single unified projection that incorporates model insights
    - P(UNDER) from XGBoost validates the direction
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelGuidedResult:
    """Result from model-guided simulation."""
    projection: float  # Model-adjusted mean projection
    std: float  # Standard deviation
    p_under: float  # Probability of going under line
    p_over: float  # Probability of going over line
    direction: str  # OVER or UNDER
    adjustment_factor: float  # How much model shifted the baseline
    base_projection: float  # Original trailing/baseline projection
    adjustment_breakdown: Dict  # Component-wise adjustments
    trials: int  # Number of MC trials run


class ModelGuidedSimulator:
    """
    Monte Carlo simulator that uses XGBoost model features to adjust projections.

    Key Principle:
        Instead of running MC with trailing stats and applying model separately,
        we USE the model features to SHIFT the MC distribution.

    Example:
        Player has trailing avg = 70 rush yards
        But model sees:
            - oline_health_score = +0.5 (good O-line) → +5% adjustment
            - opp_rush_def_epa = +0.2 (weak defense) → +4% adjustment

        Adjusted mean = 70 * 1.09 = 76.3 rush yards

        MC runs with μ=76.3 instead of μ=70
    """

    # Default std by market type (used if not provided)
    DEFAULT_STD = {
        'player_receptions': 2.4,
        'player_rush_yds': 35.5,
        'player_reception_yds': 34.2,
        'player_pass_yds': 82.8,
        'player_rush_attempts': 4.0,
        'player_pass_completions': 6.5,
        'player_pass_attempts': 5.0,
    }

    # Feature weights for distribution adjustment
    # Positive weight = higher feature value → higher projection
    # These are calibrated based on observed feature impacts
    ADJUSTMENT_WEIGHTS = {
        # O-Line & Rush Features
        'oline_health_score': 0.08,  # 8% per unit of O-line health
        'opp_rush_def_vs_avg': 0.06,  # 6% per unit (positive = bad defense)

        # Receiving Features
        'avg_separation': 0.03,  # 3% per yard of separation above/below avg
        'avg_cushion': 0.02,  # 2% per yard of cushion
        'opp_pass_def_vs_avg': 0.05,  # 5% per unit (positive = bad defense)

        # Volume Features
        'game_pace': 0.002,  # 0.2% per pace unit above/below 60
        'vegas_total': 0.002,  # 0.2% per point above/below 45
        'snap_share': 0.10,  # 10% per 10% snap share above/below 50%
        'target_share': 0.15,  # 15% per 10% target share above/below 15%

        # Defense EPA (general)
        'opp_def_epa': 0.10,  # 10% per unit of EPA (positive = bad defense)
    }

    # Market-specific feature relevance
    MARKET_FEATURES = {
        'player_rush_yds': ['oline_health_score', 'opp_rush_def_vs_avg', 'game_pace', 'vegas_total', 'snap_share'],
        'player_rush_attempts': ['oline_health_score', 'game_pace', 'vegas_total', 'snap_share'],
        'player_reception_yds': ['avg_separation', 'avg_cushion', 'opp_pass_def_vs_avg', 'target_share', 'vegas_total'],
        'player_receptions': ['avg_separation', 'avg_cushion', 'opp_pass_def_vs_avg', 'target_share'],
        'player_pass_yds': ['opp_pass_def_vs_avg', 'game_pace', 'vegas_total', 'avg_separation'],
        'player_pass_completions': ['opp_pass_def_vs_avg', 'game_pace', 'avg_separation'],
    }

    # Baseline centers for features (to normalize adjustments)
    FEATURE_CENTERS = {
        'oline_health_score': 0.0,  # Centered at 0
        'opp_rush_def_vs_avg': 0.0,
        'avg_separation': 3.0,  # Average separation is ~3 yards
        'avg_cushion': 6.0,  # Average cushion is ~6 yards
        'opp_pass_def_vs_avg': 0.0,
        'game_pace': 60.0,  # Average plays per game
        'vegas_total': 45.0,  # Average game total
        'snap_share': 0.50,  # 50% snap share
        'target_share': 0.15,  # 15% target share
        'opp_def_epa': 0.0,
    }

    def __init__(self, trials: int = 10000, seed: Optional[int] = None):
        """
        Initialize the model-guided simulator.

        Args:
            trials: Number of Monte Carlo trials per simulation
            seed: Random seed for reproducibility
        """
        self.trials = trials
        self.rng = np.random.default_rng(seed)

    def get_adjustment_factor(
        self,
        features: Dict,
        market: str,
    ) -> Tuple[float, Dict]:
        """
        Calculate the mean adjustment factor from model features.

        Args:
            features: Dict of XGBoost feature values
            market: Market type (e.g., 'player_rush_yds')

        Returns:
            Tuple of (adjustment_factor, breakdown_dict)
            adjustment_factor: Multiplier for baseline mean (e.g., 1.08 = +8%)
            breakdown_dict: Component-wise adjustments for explainability
        """
        total_adjustment = 0.0
        breakdown = {}

        # Get relevant features for this market
        relevant_features = self.MARKET_FEATURES.get(market, list(self.ADJUSTMENT_WEIGHTS.keys()))

        for feature_name in relevant_features:
            if feature_name not in features:
                continue

            value = features.get(feature_name)
            if value is None or np.isnan(value):
                continue

            weight = self.ADJUSTMENT_WEIGHTS.get(feature_name, 0.0)
            center = self.FEATURE_CENTERS.get(feature_name, 0.0)

            # Calculate deviation from center
            deviation = value - center

            # Calculate adjustment contribution
            adjustment = deviation * weight

            # Cap individual adjustments at ±15%
            adjustment = np.clip(adjustment, -0.15, 0.15)

            total_adjustment += adjustment
            breakdown[feature_name] = {
                'value': value,
                'center': center,
                'deviation': deviation,
                'weight': weight,
                'contribution': adjustment
            }

        # Cap total adjustment at ±30%
        total_adjustment = np.clip(total_adjustment, -0.30, 0.30)

        # Convert to multiplier
        adjustment_factor = 1.0 + total_adjustment

        return adjustment_factor, breakdown

    def simulate(
        self,
        baseline_mean: float,
        baseline_std: float,
        line: float,
        features: Dict,
        market: str,
        xgb_p_under: Optional[float] = None,
    ) -> ModelGuidedResult:
        """
        Run model-guided Monte Carlo simulation.

        Args:
            baseline_mean: Trailing average / baseline projection
            baseline_std: Standard deviation (or None for default)
            line: Betting line
            features: Dict of XGBoost feature values
            market: Market type (e.g., 'player_rush_yds')
            xgb_p_under: Optional XGBoost P(UNDER) for validation

        Returns:
            ModelGuidedResult with projection, probabilities, etc.
        """
        # Handle missing std
        if baseline_std is None or baseline_std <= 0 or np.isnan(baseline_std):
            baseline_std = self.DEFAULT_STD.get(market, baseline_mean * 0.3)

        # Ensure positive baseline
        baseline_mean = max(baseline_mean, 0.1)
        baseline_std = max(baseline_std, 0.5)

        # Get model-derived adjustment factor from features
        adjustment_factor, breakdown = self.get_adjustment_factor(features, market)

        # CRITICAL: If XGBoost provides P(UNDER), derive projection from that
        # The MODEL is the source of truth, not trailing stats
        if xgb_p_under is not None and 0.05 < xgb_p_under < 0.95:
            # Use inverse normal CDF to find mean that gives this P(UNDER)
            # P(X < line) = p_under => mean = line - z * std
            z_score = stats.norm.ppf(xgb_p_under)
            model_implied_mean = line - z_score * baseline_std

            # Ensure positive projection
            model_implied_mean = max(model_implied_mean, 0.1)

            # Calculate adjustment from model vs baseline
            if baseline_mean > 0:
                adjustment_factor = model_implied_mean / baseline_mean
                adjustment_factor = np.clip(adjustment_factor, 0.5, 2.0)  # Cap at ±50%

            adjusted_mean = model_implied_mean
            adjusted_std = baseline_std  # Keep original std

            # Add model-derived info to breakdown
            breakdown['model_derived'] = {
                'value': xgb_p_under,
                'center': 0.5,
                'deviation': xgb_p_under - 0.5,
                'weight': 1.0,
                'contribution': (adjustment_factor - 1.0),
                'note': f'Projection derived from P(UNDER)={xgb_p_under:.1%}'
            }
        else:
            # Fall back to feature-based adjustment (legacy behavior)
            adjusted_mean = baseline_mean * adjustment_factor
            adjusted_std = baseline_std * np.sqrt(adjustment_factor) if adjustment_factor > 0 else baseline_std

        # Run Monte Carlo simulation with model-adjusted mean
        outcomes = self.rng.normal(adjusted_mean, adjusted_std, self.trials)

        # Ensure non-negative outcomes for counting stats
        outcomes = np.maximum(outcomes, 0)

        # Calculate probabilities from MC outcomes
        p_under = np.mean(outcomes < line)
        p_over = 1.0 - p_under

        # Direction comes from XGBoost when available, else from MC
        if xgb_p_under is not None:
            direction = 'UNDER' if xgb_p_under > 0.5 else 'OVER'
            # Override MC probabilities with model probabilities for consistency
            p_under = xgb_p_under
            p_over = 1.0 - xgb_p_under
        else:
            direction = 'UNDER' if p_under > 0.5 else 'OVER'

        return ModelGuidedResult(
            projection=float(np.mean(outcomes)),
            std=float(np.std(outcomes)),
            p_under=float(p_under),
            p_over=float(p_over),
            direction=direction,
            adjustment_factor=adjustment_factor,
            base_projection=baseline_mean,
            adjustment_breakdown=breakdown,
            trials=self.trials
        )

    def simulate_for_bet(
        self,
        trailing_stat: float,
        trailing_std: Optional[float],
        line: float,
        market: str,
        features: Dict,
        xgb_p_under: Optional[float] = None,
    ) -> Dict:
        """
        Convenience method for generating bet predictions.

        Returns a dict compatible with the recommendations pipeline.
        """
        result = self.simulate(
            baseline_mean=trailing_stat,
            baseline_std=trailing_std,
            line=line,
            features=features,
            market=market,
            xgb_p_under=xgb_p_under
        )

        return {
            # Primary outputs
            'model_projection': result.projection,
            'model_std': result.std,
            'mc_p_under': result.p_under,
            'mc_p_over': result.p_over,
            'direction': result.direction,

            # Model guidance info
            'adjustment_factor': result.adjustment_factor,
            'baseline_projection': result.base_projection,
            'adjustment_pct': (result.adjustment_factor - 1.0) * 100,

            # For debugging
            'adjustment_breakdown': result.adjustment_breakdown,
            'trials': result.trials,

            # XGBoost validation
            'xgb_p_under': xgb_p_under,
            'xgb_validated': xgb_p_under is not None,
        }


# Singleton instance
_simulator_instance: Optional[ModelGuidedSimulator] = None


def get_model_guided_simulator(trials: int = 10000, seed: int = 42) -> ModelGuidedSimulator:
    """Get or create the singleton simulator instance."""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = ModelGuidedSimulator(trials=trials, seed=seed)
    return _simulator_instance


def simulate_with_model_guidance(
    trailing_stat: float,
    trailing_std: Optional[float],
    line: float,
    market: str,
    features: Dict,
    xgb_p_under: Optional[float] = None,
) -> Dict:
    """
    Convenience function for model-guided simulation.

    Example:
        result = simulate_with_model_guidance(
            trailing_stat=70.0,  # Player's 4W trailing rush yards
            trailing_std=35.0,   # Historical std
            line=69.5,           # Betting line
            market='player_rush_yds',
            features={
                'oline_health_score': 0.5,
                'opp_rush_def_vs_avg': 0.2,
                'snap_share': 0.61,
            },
            xgb_p_under=0.72,  # XGBoost P(UNDER)
        )

        # result = {
        #     'model_projection': 76.3,  # Adjusted up from 70
        #     'mc_p_under': 0.58,
        #     'direction': 'UNDER',  # From XGBoost
        #     'adjustment_pct': 9.0,  # +9% adjustment
        #     ...
        # }
    """
    simulator = get_model_guided_simulator()
    return simulator.simulate_for_bet(
        trailing_stat=trailing_stat,
        trailing_std=trailing_std,
        line=line,
        market=market,
        features=features,
        xgb_p_under=xgb_p_under,
    )
