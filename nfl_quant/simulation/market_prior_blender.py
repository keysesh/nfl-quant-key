"""
Market Prior Blending for Monte Carlo Simulations.

Blends model-based projections with closing market lines to improve calibration.

Research-backed approach:
- Closing lines contain wisdom of the market (sharp money)
- Pure model-based projections can be miscalibrated
- Optimal blend: λ·model + (1-λ)·market, where λ is learned from backtests
- Typical λ ≈ 0.60-0.75 (60-75% weight on model, 25-40% on market)

Key Applications:
1. Game totals: Blend model total with closing O/U line
2. Team totals: Blend model team score with implied total from spread
3. Player props: Blend model projection with closing prop line (if available)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketBlendConfig:
    """Configuration for market prior blending."""

    # Blend weights (λ = weight on model)
    # Higher λ = trust model more, lower λ = trust market more
    lambda_game_total: float = 0.65  # 65% model, 35% market for game totals
    lambda_team_total: float = 0.60  # 60% model, 40% market for team totals
    lambda_spread: float = 0.55      # 55% model, 45% market for spreads
    lambda_player_props: float = 0.70  # 70% model, 30% market for player props

    # Confidence-based adjustments
    # If model has high confidence (low variance), increase λ
    high_confidence_threshold: float = 0.10  # CV < 0.10 = high confidence
    low_confidence_threshold: float = 0.25   # CV > 0.25 = low confidence
    confidence_lambda_boost: float = 0.10    # +10% weight on model if high confidence
    confidence_lambda_penalty: float = -0.10  # -10% weight on model if low confidence

    # Market staleness adjustments
    # If line is stale (far from closing), reduce market weight
    max_hours_to_kickoff: int = 24  # Lines older than 24h are "stale"
    stale_line_penalty: float = 0.15  # -15% weight on market for stale lines

    # Limits on blend weights
    min_lambda: float = 0.40  # Never go below 40% model weight
    max_lambda: float = 0.85  # Never go above 85% model weight

    # Outlier detection
    # If model and market disagree wildly, flag for review
    outlier_threshold_total: float = 10.0  # >10 points difference in total
    outlier_threshold_spread: float = 6.0  # >6 points difference in spread
    outlier_threshold_player_pct: float = 0.30  # >30% difference in player prop


class MarketPriorBlender:
    """
    Blends model projections with market lines for improved calibration.

    Uses research-backed blending weights learned from historical backtests.
    """

    def __init__(self, config: Optional[MarketBlendConfig] = None):
        """
        Initialize market prior blender.

        Args:
            config: Blend configuration (uses defaults if None)
        """
        self.config = config or MarketBlendConfig()

    def blend_game_total(
        self,
        model_total: float,
        model_std: float,
        market_total: Optional[float],
        hours_to_kickoff: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Blend model game total with market closing line.

        Args:
            model_total: Model-projected game total (points)
            model_std: Model standard deviation (for confidence adjustment)
            market_total: Market closing total (O/U line)
            hours_to_kickoff: Hours until kickoff (for staleness adjustment)

        Returns:
            Dictionary with:
                - blended_total: float (blended projection)
                - lambda_used: float (effective blend weight)
                - model_weight: float (% weight on model)
                - market_weight: float (% weight on market)
                - is_outlier: bool (model and market disagree significantly)
        """
        # If no market line available, use model
        if market_total is None:
            return {
                'blended_total': model_total,
                'lambda_used': 1.0,
                'model_weight': 1.0,
                'market_weight': 0.0,
                'is_outlier': False
            }

        # Calculate effective lambda (model weight)
        lambda_effective = self._calculate_effective_lambda(
            base_lambda=self.config.lambda_game_total,
            model_mean=model_total,
            model_std=model_std,
            hours_to_kickoff=hours_to_kickoff
        )

        # Blend
        blended_total = lambda_effective * model_total + (1.0 - lambda_effective) * market_total

        # Check for outlier
        is_outlier = abs(model_total - market_total) > self.config.outlier_threshold_total

        if is_outlier:
            logger.warning(
                f"Outlier detected: Model total {model_total:.1f} vs Market {market_total:.1f} "
                f"(diff: {abs(model_total - market_total):.1f})"
            )

        return {
            'blended_total': blended_total,
            'lambda_used': lambda_effective,
            'model_weight': lambda_effective,
            'market_weight': 1.0 - lambda_effective,
            'is_outlier': is_outlier
        }

    def blend_team_total(
        self,
        model_team_total: float,
        model_std: float,
        market_spread: Optional[float],
        market_total: Optional[float],
        is_home: bool = True
    ) -> Dict[str, float]:
        """
        Blend model team total with market-implied team total.

        Market-implied team total = (total + spread/2) for home, (total - spread/2) for away

        Args:
            model_team_total: Model-projected team total
            model_std: Model standard deviation
            market_spread: Market closing spread (home team perspective, e.g., -3.5)
            market_total: Market closing total
            is_home: Whether this is the home team

        Returns:
            Dictionary with blended team total and metadata
        """
        # If no market data, use model
        if market_spread is None or market_total is None:
            return {
                'blended_team_total': model_team_total,
                'lambda_used': 1.0,
                'model_weight': 1.0,
                'market_weight': 0.0,
                'is_outlier': False
            }

        # Calculate market-implied team total
        # Total = Home + Away
        # Spread = Home - Away (from home perspective)
        # Solving: Home = (Total + Spread) / 2, Away = (Total - Spread) / 2
        if is_home:
            market_team_total = (market_total + market_spread) / 2.0
        else:
            market_team_total = (market_total - market_spread) / 2.0

        # Calculate effective lambda
        lambda_effective = self._calculate_effective_lambda(
            base_lambda=self.config.lambda_team_total,
            model_mean=model_team_total,
            model_std=model_std,
            hours_to_kickoff=None  # Not used for team totals
        )

        # Blend
        blended_team_total = lambda_effective * model_team_total + (1.0 - lambda_effective) * market_team_total

        # Check for outlier (use total threshold / 2 for team totals)
        is_outlier = abs(model_team_total - market_team_total) > (self.config.outlier_threshold_total / 2.0)

        return {
            'blended_team_total': blended_team_total,
            'lambda_used': lambda_effective,
            'model_weight': lambda_effective,
            'market_weight': 1.0 - lambda_effective,
            'is_outlier': is_outlier,
            'market_implied_team_total': market_team_total
        }

    def blend_spread(
        self,
        model_spread: float,
        model_std: float,
        market_spread: Optional[float],
        hours_to_kickoff: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Blend model spread with market closing line.

        Args:
            model_spread: Model-projected spread (home team perspective, negative = favorite)
            model_std: Model standard deviation
            market_spread: Market closing spread
            hours_to_kickoff: Hours until kickoff

        Returns:
            Dictionary with blended spread and metadata
        """
        if market_spread is None:
            return {
                'blended_spread': model_spread,
                'lambda_used': 1.0,
                'model_weight': 1.0,
                'market_weight': 0.0,
                'is_outlier': False
            }

        # Calculate effective lambda
        lambda_effective = self._calculate_effective_lambda(
            base_lambda=self.config.lambda_spread,
            model_mean=model_spread,
            model_std=model_std,
            hours_to_kickoff=hours_to_kickoff
        )

        # Blend
        blended_spread = lambda_effective * model_spread + (1.0 - lambda_effective) * market_spread

        # Check for outlier
        is_outlier = abs(model_spread - market_spread) > self.config.outlier_threshold_spread

        if is_outlier:
            logger.warning(
                f"Outlier detected: Model spread {model_spread:.1f} vs Market {market_spread:.1f} "
                f"(diff: {abs(model_spread - market_spread):.1f})"
            )

        return {
            'blended_spread': blended_spread,
            'lambda_used': lambda_effective,
            'model_weight': lambda_effective,
            'market_weight': 1.0 - lambda_effective,
            'is_outlier': is_outlier
        }

    def blend_player_prop(
        self,
        model_projection: float,
        model_std: float,
        market_line: Optional[float],
        stat_type: str = 'yards'
    ) -> Dict[str, float]:
        """
        Blend model player prop projection with market line.

        Args:
            model_projection: Model-projected stat value (e.g., 75.5 receiving yards)
            model_std: Model standard deviation
            market_line: Market closing prop line
            stat_type: Type of stat ('yards', 'receptions', 'tds', etc.)

        Returns:
            Dictionary with blended projection and metadata
        """
        if market_line is None:
            return {
                'blended_projection': model_projection,
                'lambda_used': 1.0,
                'model_weight': 1.0,
                'market_weight': 0.0,
                'is_outlier': False
            }

        # Calculate effective lambda
        lambda_effective = self._calculate_effective_lambda(
            base_lambda=self.config.lambda_player_props,
            model_mean=model_projection,
            model_std=model_std,
            hours_to_kickoff=None
        )

        # Blend
        blended_projection = lambda_effective * model_projection + (1.0 - lambda_effective) * market_line

        # Check for outlier (percentage-based for player props)
        if market_line > 0:
            pct_diff = abs(model_projection - market_line) / market_line
            is_outlier = pct_diff > self.config.outlier_threshold_player_pct
        else:
            is_outlier = False

        return {
            'blended_projection': blended_projection,
            'lambda_used': lambda_effective,
            'model_weight': lambda_effective,
            'market_weight': 1.0 - lambda_effective,
            'is_outlier': is_outlier
        }

    def _calculate_effective_lambda(
        self,
        base_lambda: float,
        model_mean: float,
        model_std: float,
        hours_to_kickoff: Optional[float] = None
    ) -> float:
        """
        Calculate effective lambda (model weight) with adjustments.

        Args:
            base_lambda: Base blend weight
            model_mean: Model mean projection
            model_std: Model standard deviation
            hours_to_kickoff: Hours until kickoff (for staleness adjustment)

        Returns:
            Adjusted lambda (clipped to min/max)
        """
        lambda_adj = base_lambda

        # Confidence-based adjustment
        if model_mean > 0:
            cv = model_std / model_mean  # Coefficient of variation

            if cv < self.config.high_confidence_threshold:
                # High confidence: boost model weight
                lambda_adj += self.config.confidence_lambda_boost
            elif cv > self.config.low_confidence_threshold:
                # Low confidence: reduce model weight
                lambda_adj += self.config.confidence_lambda_penalty

        # Staleness adjustment
        if hours_to_kickoff is not None:
            if hours_to_kickoff > self.config.max_hours_to_kickoff:
                # Stale line: reduce market weight (increase model weight)
                lambda_adj += self.config.stale_line_penalty

        # Clip to valid range
        lambda_adj = np.clip(lambda_adj, self.config.min_lambda, self.config.max_lambda)

        return lambda_adj

    def blend_distribution(
        self,
        model_samples: np.ndarray,
        market_value: float,
        lambda_weight: float
    ) -> np.ndarray:
        """
        Blend model distribution samples with market point estimate.

        Shifts distribution mean toward market while preserving shape.

        Args:
            model_samples: Array of Monte Carlo samples from model
            market_value: Market line value
            lambda_weight: Blend weight (0-1, higher = more model weight)

        Returns:
            Blended samples array
        """
        # Calculate model mean
        model_mean = np.mean(model_samples)

        # Calculate blend target
        blend_target = lambda_weight * model_mean + (1.0 - lambda_weight) * market_value

        # Shift distribution to match blend target
        shift = blend_target - model_mean
        blended_samples = model_samples + shift

        return blended_samples


def backtest_blend_weights(
    model_projections: pd.DataFrame,
    market_lines: pd.DataFrame,
    actual_outcomes: pd.DataFrame,
    lambda_range: np.ndarray = np.linspace(0.4, 0.9, 21)
) -> pd.DataFrame:
    """
    Backtest different blend weights to find optimal λ.

    Args:
        model_projections: DataFrame with model projections (columns: game_id, projection)
        market_lines: DataFrame with market lines (columns: game_id, line)
        actual_outcomes: DataFrame with actual outcomes (columns: game_id, actual)
        lambda_range: Array of lambda values to test

    Returns:
        DataFrame with backtest results (lambda, MAE, RMSE, coverage)
    """
    # Merge data
    data = model_projections.merge(market_lines, on='game_id').merge(actual_outcomes, on='game_id')

    results = []

    for lambda_val in lambda_range:
        # Calculate blended projections
        data['blended'] = lambda_val * data['projection'] + (1.0 - lambda_val) * data['line']

        # Calculate error metrics
        data['error'] = data['blended'] - data['actual']
        mae = data['error'].abs().mean()
        rmse = np.sqrt((data['error'] ** 2).mean())

        # Calculate coverage (% within 1 std dev)
        # Approximate std dev from residuals
        std_dev = data['error'].std()
        coverage = ((data['error'].abs()) <= std_dev).mean()

        results.append({
            'lambda': lambda_val,
            'MAE': mae,
            'RMSE': rmse,
            'coverage': coverage,
            'model_weight_pct': lambda_val * 100
        })

    return pd.DataFrame(results)


# Example usage and testing
if __name__ == '__main__':
    # Initialize blender
    blender = MarketPriorBlender()

    print("=== Market Prior Blending Examples ===\n")

    # Example 1: Game total blending
    print("1. Game Total Blending")
    result = blender.blend_game_total(
        model_total=48.5,
        model_std=4.2,
        market_total=45.5,
        hours_to_kickoff=2.0
    )
    print(f"  Model: 48.5 (std: 4.2)")
    print(f"  Market: 45.5")
    print(f"  Blended: {result['blended_total']:.1f}")
    print(f"  Model weight: {result['model_weight']:.1%}")
    print(f"  Market weight: {result['market_weight']:.1%}\n")

    # Example 2: Team total with large disagreement
    print("2. Team Total Blending (Outlier)")
    result = blender.blend_team_total(
        model_team_total=28.0,
        model_std=5.0,
        market_spread=-3.5,
        market_total=45.0,
        is_home=True
    )
    print(f"  Model team total: 28.0")
    print(f"  Market-implied total: {result['market_implied_team_total']:.1f}")
    print(f"  Blended: {result['blended_team_total']:.1f}")
    print(f"  Is outlier: {result['is_outlier']}\n")

    # Example 3: Player prop blending
    print("3. Player Prop Blending")
    result = blender.blend_player_prop(
        model_projection=82.5,
        model_std=18.0,
        market_line=75.5,
        stat_type='yards'
    )
    print(f"  Model: 82.5 yards (std: 18.0)")
    print(f"  Market: 75.5 yards")
    print(f"  Blended: {result['blended_projection']:.1f} yards")
    print(f"  Model weight: {result['model_weight']:.1%}\n")

    # Example 4: Distribution blending
    print("4. Distribution Blending (Monte Carlo samples)")
    np.random.seed(42)
    model_samples = np.random.normal(loc=48.5, scale=4.2, size=10000)
    market_value = 45.5
    lambda_weight = 0.65

    blended_samples = blender.blend_distribution(
        model_samples=model_samples,
        market_value=market_value,
        lambda_weight=lambda_weight
    )

    print(f"  Model samples mean: {model_samples.mean():.1f}")
    print(f"  Market value: {market_value:.1f}")
    print(f"  Blended samples mean: {blended_samples.mean():.1f}")
    print(f"  Expected blend: {lambda_weight * model_samples.mean() + (1-lambda_weight) * market_value:.1f}")
    print(f"  Std preserved: {blended_samples.std():.2f} (original: {model_samples.std():.2f})")
