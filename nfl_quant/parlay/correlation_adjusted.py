"""
Correlation-Adjusted Probability Calculator
============================================

Calculates joint probabilities for parlay legs accounting for correlation.
Uses Gaussian copula approach for multi-leg combinations.

Key Insight: P(A and B) = P(A) * P(B|A), where P(B|A) depends on correlation.
For uncorrelated events: P(A and B) = P(A) * P(B) (naive multiplication)
For correlated events: Joint probability is higher/lower than naive depending on direction.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class ParlayLeg:
    """Represents a single parlay leg."""
    player: str
    team: str
    market: str
    direction: str  # "OVER" or "UNDER"
    line: float
    confidence: float  # Individual leg probability
    game: str
    odds: Optional[int] = None


class CorrelationAdjustedCalculator:
    """
    Calculate joint probabilities accounting for empirical correlations.

    Uses Gaussian copula to model correlation structure:
    1. Transform marginal probabilities to standard normal
    2. Apply correlation structure
    3. Transform back to joint probability
    """

    # Default correlations when empirical data unavailable
    DEFAULT_CORRELATIONS = {
        # Same-player correlations (strong positive)
        ('player_receptions', 'player_reception_yds'): 0.65,
        ('player_rush_yds', 'player_rush_attempts'): 0.85,
        ('player_pass_yds', 'player_pass_tds'): 0.60,
        ('player_rush_yds', 'player_anytime_td'): 0.50,
        ('player_reception_yds', 'player_anytime_td'): 0.45,

        # Same-team correlations (moderate positive)
        ('player_pass_yds', 'player_reception_yds'): 0.35,

        # Cross-game baseline (near zero)
        ('cross_game', 'cross_game'): 0.0,
    }

    def __init__(self, matrix_path: Path = None):
        """
        Initialize with empirical correlation matrix.

        Args:
            matrix_path: Path to empirical_matrix.json
        """
        self.matrix_path = matrix_path or Path("data/correlations/empirical_matrix.json")
        self.empirical_matrix: Dict = {}
        self._load_matrix()

    def _load_matrix(self):
        """Load empirical correlation matrix from JSON."""
        if self.matrix_path.exists():
            with open(self.matrix_path, 'r') as f:
                self.empirical_matrix = json.load(f)
            print(f"Loaded empirical correlation matrix from {self.matrix_path}")
        else:
            print(f"Warning: No empirical matrix at {self.matrix_path}, using defaults")

    def get_correlation(self, leg1: ParlayLeg, leg2: ParlayLeg) -> float:
        """
        Get correlation between two parlay legs.

        Priority:
        1. Empirical same-player correlation
        2. Empirical same-team correlation
        3. Empirical same-game correlation
        4. Default correlation lookup
        5. 0.0 for cross-game (independence)

        Args:
            leg1: First parlay leg
            leg2: Second parlay leg

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Cross-game = independent
        if leg1.game != leg2.game:
            return 0.0

        # Same player
        if leg1.player == leg2.player:
            return self._get_same_player_correlation(leg1.market, leg2.market)

        # Same team
        if leg1.team == leg2.team:
            return self._get_same_team_correlation(leg1.market, leg2.market)

        # Same game, different teams
        return self._get_same_game_correlation(leg1.market, leg2.market)

    def _get_same_player_correlation(self, market1: str, market2: str) -> float:
        """Get correlation for same-player different markets."""
        key = f"{market1}|{market2}"
        rev_key = f"{market2}|{market1}"

        # Check empirical
        if 'same_player' in self.empirical_matrix:
            if key in self.empirical_matrix['same_player']:
                corr = self.empirical_matrix['same_player'][key]['correlation']
                if not np.isnan(corr):
                    return corr
            if rev_key in self.empirical_matrix['same_player']:
                corr = self.empirical_matrix['same_player'][rev_key]['correlation']
                if not np.isnan(corr):
                    return corr

        # Fall back to defaults
        if (market1, market2) in self.DEFAULT_CORRELATIONS:
            return self.DEFAULT_CORRELATIONS[(market1, market2)]
        if (market2, market1) in self.DEFAULT_CORRELATIONS:
            return self.DEFAULT_CORRELATIONS[(market2, market1)]

        # Default for same player
        return 0.50

    def _get_same_team_correlation(self, market1: str, market2: str) -> float:
        """Get correlation for same-team different players."""
        key = f"{market1}|{market2}"
        rev_key = f"{market2}|{market1}"

        # Check empirical
        if 'same_team' in self.empirical_matrix:
            if key in self.empirical_matrix['same_team']:
                corr = self.empirical_matrix['same_team'][key]['correlation']
                if not np.isnan(corr):
                    return corr
            if rev_key in self.empirical_matrix['same_team']:
                corr = self.empirical_matrix['same_team'][rev_key]['correlation']
                if not np.isnan(corr):
                    return corr

        # Default for same team
        return 0.25

    def _get_same_game_correlation(self, market1: str, market2: str) -> float:
        """Get correlation for same-game different teams."""
        key = f"{market1}|{market2}"
        rev_key = f"{market2}|{market1}"

        # Check empirical
        if 'same_game' in self.empirical_matrix:
            if key in self.empirical_matrix['same_game']:
                corr = self.empirical_matrix['same_game'][key]['correlation']
                if not np.isnan(corr):
                    return corr
            if rev_key in self.empirical_matrix['same_game']:
                corr = self.empirical_matrix['same_game'][rev_key]['correlation']
                if not np.isnan(corr):
                    return corr

        # Default for same game (weak due to game script)
        return 0.10

    def calculate_joint_probability_2leg(
        self,
        prob1: float,
        prob2: float,
        correlation: float
    ) -> float:
        """
        Calculate joint probability for 2 correlated events using Gaussian copula.

        Args:
            prob1: Probability of leg 1 hitting
            prob2: Probability of leg 2 hitting
            correlation: Correlation between outcomes

        Returns:
            Joint probability P(leg1 AND leg2)
        """
        if correlation == 0:
            # Independent
            return prob1 * prob2

        # Clamp probabilities to avoid numerical issues
        prob1 = np.clip(prob1, 0.001, 0.999)
        prob2 = np.clip(prob2, 0.001, 0.999)
        correlation = np.clip(correlation, -0.99, 0.99)

        # Transform to standard normal quantiles
        z1 = norm.ppf(prob1)
        z2 = norm.ppf(prob2)

        # Bivariate normal CDF with correlation
        # P(Z1 < z1 AND Z2 < z2) where Z1, Z2 ~ N(0,1) with corr = rho
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]

        joint_prob = stats.multivariate_normal.cdf([z1, z2], mean=mean, cov=cov)

        return float(joint_prob)

    def calculate_joint_probability(
        self,
        legs: List[ParlayLeg]
    ) -> Tuple[float, float, float]:
        """
        Calculate joint probability for multiple legs.

        For 2 legs: Use bivariate Gaussian copula
        For 3+ legs: Use sequential pairwise approach (approximation)

        Args:
            legs: List of parlay legs with confidence scores

        Returns:
            (naive_prob, adjusted_prob, correlation_factor)
            - naive_prob: Product of individual probabilities
            - adjusted_prob: Correlation-adjusted joint probability
            - correlation_factor: adjusted/naive ratio (1.0 = no adjustment)
        """
        if len(legs) < 2:
            prob = legs[0].confidence if legs else 0.0
            return prob, prob, 1.0

        # Calculate naive probability (product)
        probs = [leg.confidence for leg in legs]
        naive_prob = np.prod(probs)

        # For 2 legs: exact Gaussian copula
        if len(legs) == 2:
            correlation = self.get_correlation(legs[0], legs[1])
            adjusted_prob = self.calculate_joint_probability_2leg(
                probs[0], probs[1], correlation
            )
            correlation_factor = adjusted_prob / naive_prob if naive_prob > 0 else 1.0
            return naive_prob, adjusted_prob, correlation_factor

        # For 3+ legs: sequential pairwise approach
        # This is an approximation - assumes correlation structure is pairwise decomposable
        adjusted_prob = probs[0]

        for i in range(1, len(legs)):
            # Get max correlation with any prior leg
            max_corr = 0.0
            for j in range(i):
                corr = self.get_correlation(legs[j], legs[i])
                if abs(corr) > abs(max_corr):
                    max_corr = corr

            # Apply pairwise adjustment
            if max_corr != 0:
                # Adjust current leg probability based on correlation
                conditional_prob = self._conditional_probability(
                    probs[i], adjusted_prob, max_corr
                )
                adjusted_prob = adjusted_prob * conditional_prob
            else:
                adjusted_prob = adjusted_prob * probs[i]

        correlation_factor = adjusted_prob / naive_prob if naive_prob > 0 else 1.0
        return naive_prob, adjusted_prob, correlation_factor

    def _conditional_probability(
        self,
        prob_b: float,
        prob_a: float,
        correlation: float
    ) -> float:
        """
        Calculate P(B|A) given correlation.

        Uses regression adjustment: P(B|A) ≈ P(B) + correlation * adjustment
        """
        # Simple linear adjustment based on correlation
        # If A happens (implied by prob_a being high), B is more likely if positively correlated
        if prob_a > 0.5:  # A is likely
            if correlation > 0:
                # Positive correlation: B more likely given A
                adjustment = correlation * (1 - prob_b) * 0.3
                return min(prob_b + adjustment, 0.99)
            else:
                # Negative correlation: B less likely given A
                adjustment = abs(correlation) * prob_b * 0.3
                return max(prob_b - adjustment, 0.01)
        else:  # A is unlikely
            return prob_b  # Less adjustment needed

    def get_correlation_summary(self, legs: List[ParlayLeg]) -> Dict:
        """
        Get a summary of correlations in a parlay.

        Args:
            legs: List of parlay legs

        Returns:
            Summary dict with correlation info
        """
        correlations = []
        max_corr = 0.0
        min_corr = 0.0

        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                corr = self.get_correlation(legs[i], legs[j])
                correlations.append({
                    'leg1': f"{legs[i].player} {legs[i].market}",
                    'leg2': f"{legs[j].player} {legs[j].market}",
                    'correlation': round(corr, 3),
                    'relationship': self._get_relationship(legs[i], legs[j])
                })
                max_corr = max(max_corr, corr)
                min_corr = min(min_corr, corr)

        return {
            'pairs': correlations,
            'max_correlation': round(max_corr, 3),
            'min_correlation': round(min_corr, 3),
            'avg_correlation': round(np.mean([c['correlation'] for c in correlations]), 3) if correlations else 0.0
        }

    def _get_relationship(self, leg1: ParlayLeg, leg2: ParlayLeg) -> str:
        """Determine relationship type between two legs."""
        if leg1.game != leg2.game:
            return 'cross_game'
        if leg1.player == leg2.player:
            return 'same_player'
        if leg1.team == leg2.team:
            return 'same_team'
        return 'same_game'


def test_calculator():
    """Test the correlation calculator."""
    calc = CorrelationAdjustedCalculator()

    # Create test legs
    leg1 = ParlayLeg(
        player="Travis Kelce",
        team="KC",
        market="player_receptions",
        direction="UNDER",
        line=5.5,
        confidence=0.65,
        game="KC@LV"
    )

    leg2 = ParlayLeg(
        player="Davante Adams",
        team="LV",
        market="player_reception_yds",
        direction="UNDER",
        line=70.5,
        confidence=0.58,
        game="KC@LV"
    )

    leg3 = ParlayLeg(
        player="De'Von Achane",
        team="MIA",
        market="player_rush_yds",
        direction="UNDER",
        line=85.5,
        confidence=0.60,
        game="MIA@NYJ"
    )

    # 2-leg parlay (same game)
    naive, adjusted, factor = calc.calculate_joint_probability([leg1, leg2])
    print(f"2-leg (same game): naive={naive:.3f}, adjusted={adjusted:.3f}, factor={factor:.3f}")

    # 2-leg parlay (cross game)
    naive, adjusted, factor = calc.calculate_joint_probability([leg1, leg3])
    print(f"2-leg (cross game): naive={naive:.3f}, adjusted={adjusted:.3f}, factor={factor:.3f}")

    # 3-leg parlay
    naive, adjusted, factor = calc.calculate_joint_probability([leg1, leg2, leg3])
    print(f"3-leg: naive={naive:.3f}, adjusted={adjusted:.3f}, factor={factor:.3f}")

    # Get correlation summary
    summary = calc.get_correlation_summary([leg1, leg2, leg3])
    print(f"\nCorrelation summary: {summary}")


if __name__ == "__main__":
    test_calculator()
