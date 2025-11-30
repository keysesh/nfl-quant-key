"""
Red Zone and Field Position-Aware TD Model.

Research-backed touchdown prediction that accounts for:
1. Field position (red zone vs midfield)
2. Goal-to-go situations (inside 5 yards)
3. Team red zone efficiency (offense + defense)
4. Player role in red zone (primary vs secondary target)
5. Historical TD rates with Bayesian shrinkage

Key Statistics (from research):
- 85.7% of rushing TDs occur in red zone (inside 20)
- 69.9% of receiving TDs occur in red zone
- Goal-to-go (inside 5): TD rate 3-4x higher than red zone average
- Primary red zone targets: ~2x TD rate vs secondary options
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RedZoneTDConfig:
    """Configuration for red zone TD modeling."""

    # Base TD rates by position (per opportunity in red zone)
    base_rz_td_rate_qb_pass: float = 0.045  # 4.5% per RZ pass attempt
    base_rz_td_rate_qb_rush: float = 0.15   # 15% per RZ rush (goal line sneaks)
    base_rz_td_rate_rb_rush: float = 0.12   # 12% per RZ carry
    base_rz_td_rate_rb_rec: float = 0.10    # 10% per RZ target
    base_rz_td_rate_wr: float = 0.15        # 15% per RZ target
    base_rz_td_rate_te: float = 0.18        # 18% per RZ target (goal line favorites)

    # Goal-to-go multipliers (inside 5 yards)
    goal_to_go_multiplier_rush: float = 3.0  # 3x base rate
    goal_to_go_multiplier_pass: float = 2.5  # 2.5x base rate

    # Field position TD probability
    # Outside red zone (20+): Very low base rate
    outside_rz_td_rate_pass: float = 0.015  # 1.5% per attempt
    outside_rz_td_rate_rush: float = 0.005  # 0.5% per carry (rare long TDs)

    # Red zone opportunity distribution
    # What % of opportunities occur in red zone vs outside?
    pct_opportunities_in_rz_wr: float = 0.25  # 25% of WR targets in RZ
    pct_opportunities_in_rz_te: float = 0.30  # 30% of TE targets in RZ
    pct_opportunities_in_rz_rb_rush: float = 0.35  # 35% of RB carries in RZ
    pct_opportunities_in_rz_rb_rec: float = 0.20  # 20% of RB targets in RZ
    pct_opportunities_in_rz_qb_rush: float = 0.40  # 40% of QB rushes in RZ (scrambles + sneaks)

    # Goal-to-go opportunity distribution (of red zone opportunities)
    pct_rz_as_goal_to_go: float = 0.30  # 30% of RZ opportunities are goal-to-go

    # Team RZ efficiency adjustments
    # Elite RZ offense: +20% TD rate, Poor RZ offense: -20%
    rz_offense_elite_threshold: float = 0.65  # >65% RZ TD rate = elite
    rz_offense_poor_threshold: float = 0.45   # <45% RZ TD rate = poor
    rz_offense_multiplier_range: Tuple[float, float] = (0.80, 1.20)

    # RZ defense adjustments
    rz_defense_elite_threshold: float = 0.45  # <45% allowed = elite
    rz_defense_poor_threshold: float = 0.65   # >65% allowed = poor
    rz_defense_multiplier_range: Tuple[float, float] = (0.85, 1.15)

    # Player role in red zone
    primary_rz_target_boost: float = 1.50  # +50% for primary RZ weapon
    secondary_rz_target_adjustment: float = 0.80  # -20% for secondary option

    # Bayesian shrinkage (regression to mean)
    # Shrink player sample TD rates toward position baseline
    shrinkage_weight: float = 0.40  # 40% weight on baseline, 60% on sample

    # Minimum opportunities for player-specific rate
    min_opportunities_for_personalization: int = 20


class RedZoneTDModel:
    """
    Field position-aware TD model with red zone specialization.

    Predicts TD probability accounting for:
    - Where opportunities occur (red zone vs midfield)
    - Team red zone efficiency (offense & defense)
    - Player role in red zone formations
    - Historical TD rates with shrinkage
    """

    def __init__(self, config: Optional[RedZoneTDConfig] = None):
        """
        Initialize red zone TD model.

        Args:
            config: Red zone TD configuration (uses defaults if None)
        """
        self.config = config or RedZoneTDConfig()

    def predict_td_probability(
        self,
        position: str,
        opportunities: float,
        team_rz_td_rate: Optional[float] = None,
        opponent_rz_td_rate_allowed: Optional[float] = None,
        player_historical_td_rate: Optional[float] = None,
        player_historical_opportunities: Optional[int] = None,
        player_rz_role: str = 'secondary',
        stat_type: str = 'receiving'  # 'receiving' or 'rushing'
    ) -> Dict[str, float]:
        """
        Predict TD probability for a player.

        Args:
            position: Player position (QB, RB, WR, TE)
            opportunities: Expected opportunities (attempts, targets, carries)
            team_rz_td_rate: Team's red zone TD rate (optional)
            opponent_rz_td_rate_allowed: Opponent's RZ TD rate allowed (optional)
            player_historical_td_rate: Player's historical TD rate per opportunity
            player_historical_opportunities: Player's historical opportunities (for shrinkage)
            player_rz_role: Player's red zone role ('primary', 'secondary', 'tertiary')
            stat_type: Type of stat ('receiving' or 'rushing')

        Returns:
            Dictionary with:
                - td_probability_per_opportunity: float (blended rate)
                - expected_tds: float (opportunities * rate)
                - rz_opportunities: float (expected RZ opportunities)
                - outside_rz_opportunities: float
                - rz_td_rate: float (in RZ)
                - outside_rz_td_rate: float
        """
        # Step 1: Get base TD rates by position and stat type
        base_rz_rate, base_outside_rate = self._get_base_rates(position, stat_type)

        # Step 2: Adjust for team RZ efficiency
        team_multiplier = self._get_team_rz_multiplier(
            team_rz_td_rate=team_rz_td_rate,
            opponent_rz_td_rate_allowed=opponent_rz_td_rate_allowed
        )

        adjusted_rz_rate = base_rz_rate * team_multiplier

        # Step 3: Adjust for player role in RZ
        role_multiplier = self._get_player_role_multiplier(player_rz_role)
        adjusted_rz_rate *= role_multiplier

        # Step 4: Apply Bayesian shrinkage if player historical data available
        if player_historical_td_rate is not None and player_historical_opportunities is not None:
            adjusted_rz_rate = self._apply_shrinkage(
                sample_rate=player_historical_td_rate,
                baseline_rate=adjusted_rz_rate,
                n_opportunities=player_historical_opportunities
            )

        # Step 5: Calculate field position distribution
        pct_in_rz = self._get_rz_opportunity_pct(position, stat_type)
        rz_opportunities = opportunities * pct_in_rz
        outside_opportunities = opportunities * (1.0 - pct_in_rz)

        # Step 6: Apply goal-to-go boost to RZ opportunities
        pct_goal_to_go = self.config.pct_rz_as_goal_to_go
        goal_to_go_multiplier = (
            self.config.goal_to_go_multiplier_rush if stat_type == 'rushing'
            else self.config.goal_to_go_multiplier_pass
        )

        # Weighted average of regular RZ and goal-to-go
        effective_rz_rate = (
            adjusted_rz_rate * (1.0 - pct_goal_to_go) +
            adjusted_rz_rate * goal_to_go_multiplier * pct_goal_to_go
        )

        # Step 7: Calculate expected TDs from each field position zone
        expected_rz_tds = rz_opportunities * effective_rz_rate
        expected_outside_tds = outside_opportunities * base_outside_rate

        total_expected_tds = expected_rz_tds + expected_outside_tds

        # Blended rate (for use in binomial sampling)
        blended_rate = total_expected_tds / opportunities if opportunities > 0 else 0.0

        return {
            'td_probability_per_opportunity': blended_rate,
            'expected_tds': total_expected_tds,
            'rz_opportunities': rz_opportunities,
            'outside_rz_opportunities': outside_opportunities,
            'rz_td_rate': effective_rz_rate,
            'outside_rz_td_rate': base_outside_rate,
            'team_multiplier': team_multiplier,
            'role_multiplier': role_multiplier
        }

    def _get_base_rates(self, position: str, stat_type: str) -> Tuple[float, float]:
        """
        Get base TD rates for position and stat type.

        Args:
            position: Player position
            stat_type: 'receiving' or 'rushing'

        Returns:
            Tuple of (rz_td_rate, outside_rz_td_rate)
        """
        if position == 'QB':
            if stat_type == 'rushing':
                return self.config.base_rz_td_rate_qb_rush, self.config.outside_rz_td_rate_rush
            else:
                return self.config.base_rz_td_rate_qb_pass, self.config.outside_rz_td_rate_pass

        elif position == 'RB':
            if stat_type == 'rushing':
                return self.config.base_rz_td_rate_rb_rush, self.config.outside_rz_td_rate_rush
            else:
                return self.config.base_rz_td_rate_rb_rec, self.config.outside_rz_td_rate_pass

        elif position == 'WR':
            return self.config.base_rz_td_rate_wr, self.config.outside_rz_td_rate_pass

        elif position == 'TE':
            return self.config.base_rz_td_rate_te, self.config.outside_rz_td_rate_pass

        else:
            # Default
            return 0.10, 0.01

    def _get_rz_opportunity_pct(self, position: str, stat_type: str) -> float:
        """
        Get percentage of opportunities that occur in red zone.

        Args:
            position: Player position
            stat_type: 'receiving' or 'rushing'

        Returns:
            Percentage (0.0 to 1.0)
        """
        if position == 'QB':
            if stat_type == 'rushing':
                return self.config.pct_opportunities_in_rz_qb_rush
            else:
                return 0.22  # ~22% of pass attempts in RZ (team-level)

        elif position == 'RB':
            if stat_type == 'rushing':
                return self.config.pct_opportunities_in_rz_rb_rush
            else:
                return self.config.pct_opportunities_in_rz_rb_rec

        elif position == 'WR':
            return self.config.pct_opportunities_in_rz_wr

        elif position == 'TE':
            return self.config.pct_opportunities_in_rz_te

        else:
            return 0.20

    def _get_team_rz_multiplier(
        self,
        team_rz_td_rate: Optional[float],
        opponent_rz_td_rate_allowed: Optional[float]
    ) -> float:
        """
        Calculate team red zone efficiency multiplier.

        Args:
            team_rz_td_rate: Team's RZ TD rate (e.g., 0.55 = 55%)
            opponent_rz_td_rate_allowed: Opponent's RZ TD rate allowed

        Returns:
            Multiplier (0.80 to 1.20)
        """
        multiplier = 1.0

        # Offensive RZ efficiency
        if team_rz_td_rate is not None:
            if team_rz_td_rate >= self.config.rz_offense_elite_threshold:
                # Elite offense
                multiplier *= self.config.rz_offense_multiplier_range[1]
            elif team_rz_td_rate <= self.config.rz_offense_poor_threshold:
                # Poor offense
                multiplier *= self.config.rz_offense_multiplier_range[0]
            else:
                # Linear interpolation
                range_min, range_max = self.config.rz_offense_multiplier_range
                normalized = (team_rz_td_rate - self.config.rz_offense_poor_threshold) / (
                    self.config.rz_offense_elite_threshold - self.config.rz_offense_poor_threshold
                )
                multiplier *= range_min + (range_max - range_min) * normalized

        # Defensive RZ efficiency (inverse: high rate allowed = bad defense)
        if opponent_rz_td_rate_allowed is not None:
            if opponent_rz_td_rate_allowed >= self.config.rz_defense_poor_threshold:
                # Poor defense (easy to score on)
                multiplier *= self.config.rz_defense_multiplier_range[1]
            elif opponent_rz_td_rate_allowed <= self.config.rz_defense_elite_threshold:
                # Elite defense
                multiplier *= self.config.rz_defense_multiplier_range[0]
            else:
                # Linear interpolation
                range_min, range_max = self.config.rz_defense_multiplier_range
                normalized = (opponent_rz_td_rate_allowed - self.config.rz_defense_elite_threshold) / (
                    self.config.rz_defense_poor_threshold - self.config.rz_defense_elite_threshold
                )
                multiplier *= range_min + (range_max - range_min) * normalized

        return np.clip(multiplier, 0.70, 1.40)

    def _get_player_role_multiplier(self, player_rz_role: str) -> float:
        """
        Get player role multiplier for red zone usage.

        Args:
            player_rz_role: 'primary', 'secondary', or 'tertiary'

        Returns:
            Multiplier
        """
        if player_rz_role == 'primary':
            return self.config.primary_rz_target_boost
        elif player_rz_role == 'secondary':
            return self.config.secondary_rz_target_adjustment
        elif player_rz_role == 'tertiary':
            return 0.60  # -40% for tertiary option
        else:
            return 1.0  # Unknown = neutral

    def _apply_shrinkage(
        self,
        sample_rate: float,
        baseline_rate: float,
        n_opportunities: int
    ) -> float:
        """
        Apply Bayesian shrinkage to player sample rate.

        Shrinks toward baseline when sample size is small.

        Args:
            sample_rate: Player's observed TD rate
            baseline_rate: Position baseline rate
            n_opportunities: Number of opportunities in sample

        Returns:
            Shrunk rate
        """
        if n_opportunities < self.config.min_opportunities_for_personalization:
            # Heavy shrinkage for small samples
            weight_on_sample = n_opportunities / self.config.min_opportunities_for_personalization
            weight_on_baseline = 1.0 - weight_on_sample
        else:
            # Use config shrinkage weight
            weight_on_baseline = self.config.shrinkage_weight
            weight_on_sample = 1.0 - weight_on_baseline

        shrunk_rate = (
            weight_on_sample * sample_rate +
            weight_on_baseline * baseline_rate
        )

        return shrunk_rate

    def sample_tds(
        self,
        opportunities: np.ndarray,
        td_rate: float,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample touchdowns using binomial distribution.

        Args:
            opportunities: Array of opportunity counts (or single value)
            td_rate: TD probability per opportunity
            n_samples: Number of samples to generate (if opportunities is scalar)

        Returns:
            Array of TD counts
        """
        if np.isscalar(opportunities):
            # Single value: generate n_samples
            if n_samples is None:
                n_samples = 1
            return np.random.binomial(n=int(opportunities), p=td_rate, size=n_samples)
        else:
            # Array of opportunities: sample each
            return np.array([
                np.random.binomial(n=int(opp), p=td_rate)
                for opp in opportunities
            ])


# Example usage and testing
if __name__ == '__main__':
    # Initialize model
    model = RedZoneTDModel()

    print("=== Red Zone TD Model Examples ===\n")

    # Example 1: Elite WR1 vs weak defense
    print("1. Elite WR1 (8 targets, primary RZ role, team RZ TD% 65%, vs defense allowing 60%)")
    result = model.predict_td_probability(
        position='WR',
        opportunities=8.0,
        team_rz_td_rate=0.65,  # Elite offense
        opponent_rz_td_rate_allowed=0.60,  # Poor defense
        player_rz_role='primary',
        stat_type='receiving'
    )
    print(f"  Expected TDs: {result['expected_tds']:.3f}")
    print(f"  TD probability per target: {result['td_probability_per_opportunity']:.3%}")
    print(f"  RZ opportunities: {result['rz_opportunities']:.2f}")
    print(f"  RZ TD rate: {result['rz_td_rate']:.3%}")
    print(f"  Team multiplier: {result['team_multiplier']:.3f}")
    print(f"  Role multiplier: {result['role_multiplier']:.3f}\n")

    # Example 2: RB1 goal line back
    print("2. RB1 (15 carries, primary RZ role, average team)")
    result = model.predict_td_probability(
        position='RB',
        opportunities=15.0,
        team_rz_td_rate=0.55,  # Average
        player_rz_role='primary',
        stat_type='rushing'
    )
    print(f"  Expected TDs: {result['expected_tds']:.3f}")
    print(f"  TD probability per carry: {result['td_probability_per_opportunity']:.3%}")
    print(f"  RZ carries: {result['rz_opportunities']:.2f}")
    print(f"  RZ TD rate (with goal-to-go boost): {result['rz_td_rate']:.3%}\n")

    # Example 3: TE with historical data
    print("3. TE (6 targets, secondary RZ role, historical: 3 TDs in 40 targets)")
    result = model.predict_td_probability(
        position='TE',
        opportunities=6.0,
        team_rz_td_rate=0.58,
        player_historical_td_rate=3.0 / 40.0,  # 7.5% sample rate
        player_historical_opportunities=40,
        player_rz_role='secondary',
        stat_type='receiving'
    )
    print(f"  Expected TDs: {result['expected_tds']:.3f}")
    print(f"  TD probability per target: {result['td_probability_per_opportunity']:.3%}")
    print(f"  (Shrunk from sample rate 7.5% toward baseline)\n")

    # Example 4: Sample TDs for Monte Carlo
    print("4. Monte Carlo TD sampling (10,000 trials)")
    td_samples = model.sample_tds(
        opportunities=8.0,
        td_rate=0.12,  # 12% per opportunity
        n_samples=10000
    )
    print(f"  Mean TDs: {td_samples.mean():.3f} (expected: {8.0 * 0.12:.3f})")
    print(f"  Std TDs: {td_samples.std():.3f}")
    print(f"  P(0 TDs): {(td_samples == 0).mean():.3%}")
    print(f"  P(1+ TDs): {(td_samples >= 1).mean():.3%}")
    print(f"  P(2+ TDs): {(td_samples >= 2).mean():.3%}")
