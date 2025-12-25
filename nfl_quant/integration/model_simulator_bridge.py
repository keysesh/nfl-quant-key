"""
Model-Simulator Bridge V3

Combines Monte Carlo simulator projections with XGBoost classifier probabilities.

Architecture (V3 - MC Mean as Projection):
1. MC simulator outputs projection (mc_mean) from player model
2. XGBoost classifier outputs P(UNDER) based on line-aware features
3. Projection = MC mean (NOT derived from P(UNDER))
4. Direction = from classifier's P(UNDER)
5. P(UNDER) calculated as P(X < line) where X ~ Normal(mc_mean, mc_std)

Why V2 (inverse CDF) was wrong:
- Deriving projection FROM P(UNDER) breaks for "trap lines"
- Example: Player averaging 10 yards, line=38.5, P(UNDER)=85%
  - Inverse CDF asks: "What mean makes 38.5 the 85th percentile?"
  - This produces nonsensical projections (negative or near-zero)
- The correct flow: projection → P(UNDER), not P(UNDER) → projection

The MC simulator already models:
- Usage (targets, carries) from historical data + opponent adjustments
- Efficiency (Y/T, YPC) from model predictions
- Final stat = usage × efficiency

The classifier adds:
- Line-aware context (how far is line from trailing avg?)
- Regime detection, game script, etc.
- Used ONLY for pick direction, not projection derivation
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from scipy import stats
import joblib

from configs.model_config import CLASSIFIER_MARKETS, get_active_model_path

logger = logging.getLogger(__name__)


class ModelSimulatorBridge:
    """
    Bridge between Monte Carlo simulator and XGBoost classifier.

    V3: Uses MC mean as projection, classifier P(UNDER) for direction.

    Key insight: Projection should come FROM the player model, not be
    reverse-engineered from P(UNDER). The inverse CDF approach (V2) failed
    for trap lines where line >> expected value.

    Flow:
    1. MC simulator outputs projection (mc_mean) based on player model
    2. Classifier outputs P(UNDER) based on line-aware features
    3. Use mc_mean as the projection (it's what the model predicts)
    4. Use classifier's P(UNDER) for pick direction
    5. Flag conflicts but don't try to "fix" them with math
    """

    # Default std for different market types (used if MC std unavailable)
    DEFAULT_STD = {
        'player_receptions': 2.4,
        'player_rush_yds': 35.5,
        'player_reception_yds': 34.2,
        'player_pass_yds': 82.8,
        'player_rush_attempts': 4.0,
    }

    # Minimum std to prevent extreme projections
    MIN_STD = 0.5

    # Maximum projection deviation from line (safety bound)
    MAX_DEVIATION_STDS = 2.5  # Don't project more than 2.5σ from line

    # Minimum projection as percentage of line (prevents 0.0 projections)
    MIN_PROJECTION_PCT = 0.15  # Projection must be at least 15% of line

    def __init__(self):
        self.model_data = None
        self._load_model()

    def _load_model(self):
        """Load the active XGBoost model."""
        try:
            model_path = get_active_model_path()
            if Path(model_path).exists():
                self.model_data = joblib.load(model_path)
                logger.info(f"ModelSimulatorBridge: Loaded model {self.model_data.get('version', 'unknown')}")
            else:
                logger.warning(f"Model not found at {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def get_mc_p_under(
        self,
        mc_mean: float,
        mc_std: float,
        line: float
    ) -> float:
        """
        Calculate P(UNDER) implied by Monte Carlo simulation.

        Uses CDF of normal distribution to find probability
        that actual value will be below the line.
        """
        if mc_std <= 0:
            return 1.0 if mc_mean < line else 0.0
        return float(stats.norm.cdf(line, mc_mean, mc_std))

    def derive_projection_from_probability(
        self,
        p_under: float,
        line: float,
        std: float,
        market: str = None,
        trailing_mean: float = None
    ) -> float:
        """
        Derive a projection from P(UNDER) using inverse CDF.

        Math:
            If X ~ Normal(μ, σ) and P(X < line) = p_under
            Then: μ = line - σ × Φ⁻¹(p_under)

        Args:
            p_under: Probability of under
            line: The betting line
            std: Standard deviation (from MC or default)
            market: Market type for default std lookup
            trailing_mean: Player's trailing average (for sanity-checking std)

        Returns:
            Projection that is mathematically consistent with p_under
        """
        # Get appropriate std
        if std <= 0 or np.isnan(std):
            std = self.DEFAULT_STD.get(market, 2.5)
        std = max(std, self.MIN_STD)

        # CRITICAL FIX V2: Cap std relative to BOTH line AND trailing stat
        # The MC std can be unrealistically high (e.g., 48 for a player averaging 10 yards)
        # We cap std at:
        #   1. 80% of line (prevents negative projections mathematically)
        #   2. 100% of trailing mean (std shouldn't exceed player's average)
        # Use the more conservative (smaller) cap
        max_std_for_line = line * 0.8 if line > 0 else float('inf')
        max_std_for_trailing = trailing_mean * 1.0 if trailing_mean and trailing_mean > 0 else float('inf')

        # Use the minimum of both caps (most conservative)
        max_std = min(max_std_for_line, max_std_for_trailing)

        if std > max_std and max_std > 0:
            std = max(max_std, self.MIN_STD)

        # Clamp probability to avoid extreme z-scores
        # 0.01 and 0.99 correspond to about ±2.33 std deviations
        p_clamped = np.clip(p_under, 0.01, 0.99)

        # Inverse CDF: Φ⁻¹(p_under)
        z_score = stats.norm.ppf(p_clamped)

        # Clamp z-score to prevent wild projections
        z_score = np.clip(z_score, -self.MAX_DEVIATION_STDS, self.MAX_DEVIATION_STDS)

        # μ = line - σ × z
        projection = line - std * z_score

        # Ensure projection is positive (stats can't be negative)
        # AND has a minimum floor of 15% of line to prevent 0.0 projections
        # For extreme P(UNDER) values, this gives a realistic low projection
        # Example: line=38.5, p_under=98.5% -> projection >= 5.77 (not 0.0)
        min_projection = max(line * self.MIN_PROJECTION_PCT, 0.1)
        projection = max(projection, min_projection)

        return float(projection)

    def get_unified_prediction(
        self,
        market: str,
        line: float,
        mc_mean: float,
        mc_std: float,
        model_p_under: Optional[float] = None,
    ) -> Dict:
        """
        Get unified prediction using MC mean as projection.

        V3 Logic:
        1. Use MC mean as projection (it's from the player model)
        2. Use classifier's P(UNDER) for pick direction
        3. Calculate MC-implied P(UNDER) for comparison
        4. Flag conflicts but don't try to "fix" projection

        Why V3 (not V2 inverse CDF):
        - V2 tried to derive projection FROM P(UNDER) using inverse CDF
        - This broke for "trap lines" (line >> expected value)
        - Example: Player avg 10 yards, line=38.5, P(UNDER)=85%
          - V2 asked: "What mean makes 38.5 the 85th percentile?"
          - Answer depends on std, produces nonsense for high std
        - V3 just uses the MC projection directly

        Args:
            market: Market type (e.g., 'player_receptions')
            line: Betting line
            mc_mean: Mean projection from MC simulation
            mc_std: Standard deviation from simulation
            model_p_under: XGBoost P(UNDER) if available

        Returns:
            Dict with unified prediction data
        """
        # Get MC-implied P(UNDER) for comparison
        mc_p_under = self.get_mc_p_under(mc_mean, mc_std, line)

        # Determine if classifier model is available
        model_available = (
            market in CLASSIFIER_MARKETS and
            model_p_under is not None and
            self.model_data is not None
        )

        # V3: Use MC mean as projection (NOT derived from P(UNDER))
        # The MC simulator already incorporates player model + matchup adjustments
        projection = mc_mean

        if model_available:
            # Use classifier's P(UNDER) for direction
            p_under_to_use = model_p_under
            model_confidence = abs(model_p_under - 0.5) * 2  # Scale to 0-1
            source = 'classifier_direction_mc_projection'
        else:
            # Fallback to MC-only
            p_under_to_use = mc_p_under
            model_p_under = mc_p_under  # For consistency in output
            model_confidence = 0.0
            source = 'mc_only'

        # Determine directions
        mc_direction = 'UNDER' if mc_p_under > 0.5 else 'OVER'
        model_direction = 'UNDER' if p_under_to_use > 0.5 else 'OVER'

        # Check for conflict between MC projection and classifier direction
        # This is informational - we use classifier for direction, MC for projection
        had_conflict = (mc_direction != model_direction) if model_available else False

        # Log conflicts for debugging
        if had_conflict:
            logger.debug(
                f"Direction conflict for {market}: "
                f"MC projection={mc_mean:.1f} implies {mc_direction}, "
                f"but classifier P(UNDER)={model_p_under:.1%} implies {model_direction}"
            )

        return {
            # Primary outputs (V3: MC projection + classifier direction)
            'ensemble_p_under': p_under_to_use,
            'ensemble_projection': projection,  # V3: Use MC mean directly
            'ensemble_direction': model_direction,

            # MC data (for reference)
            'mc_p_under': mc_p_under,
            'mc_mean': mc_mean,
            'mc_std': mc_std,
            'mc_direction': mc_direction,

            # Model data
            'model_p_under': model_p_under,
            'model_direction': model_direction if model_available else None,
            'model_confidence': model_confidence,
            'model_weight': 1.0 if model_available else 0.0,
            'model_available': model_available,

            # Metadata
            'had_conflict': had_conflict,
            'source': source,
            'line': line,
            'market': market,

            # V3: projection source info
            'projection_source': 'mc_mean',  # Always MC mean in V3
            'projection_note': 'MC simulator output (usage × efficiency)',
        }


# Singleton instance
_bridge_instance: Optional[ModelSimulatorBridge] = None


def get_bridge() -> ModelSimulatorBridge:
    """Get or create the singleton bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ModelSimulatorBridge()
    return _bridge_instance


def get_unified_prediction(
    market: str,
    line: float,
    mc_mean: float,
    mc_std: float,
    model_p_under: Optional[float] = None,
) -> Dict:
    """
    Convenience function to get unified prediction from bridge (V3).

    V3 Architecture:
    - Projection comes from MC simulator (mc_mean)
    - Direction comes from classifier (model_p_under)
    - No inverse CDF derivation (that was V2's bug)

    Example:
        result = get_unified_prediction(
            market='player_receptions',
            line=3.5,
            mc_mean=7.1,  # MC projects 7.1 receptions
            mc_std=2.3,
            model_p_under=0.35  # Classifier says 35% chance under
        )

        # Result:
        # {
        #     'ensemble_p_under': 0.35,  # From classifier
        #     'ensemble_projection': 7.1,  # V3: Direct from MC (not derived!)
        #     'ensemble_direction': 'OVER',  # 0.35 < 0.5 → OVER
        #     'had_conflict': False,  # MC and classifier both say OVER
        #     'projection_source': 'mc_mean',
        #     ...
        # }
    """
    bridge = get_bridge()
    return bridge.get_unified_prediction(
        market=market,
        line=line,
        mc_mean=mc_mean,
        mc_std=mc_std,
        model_p_under=model_p_under,
    )
