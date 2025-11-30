"""
Player correlation matrix for Monte Carlo simulations.

Implements research-backed correlation modeling for same-team players:
- QB-WR/TE correlations (positive for targets/yards)
- RB committee correlations (negative for carries/targets)
- Game script correlations (all players affected by score differential)

Uses Cholesky decomposition for efficient correlated random sampling.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

logger = logging.getLogger(__name__)


@dataclass
class PlayerCorrelationConfig:
    """Configuration for player correlation coefficients."""

    # QB-Receiver correlations (positive)
    qb_wr_targets: float = 0.35  # QB attempts up → WR targets up
    qb_wr_yards: float = 0.45    # QB yards up → WR yards up
    qb_te_targets: float = 0.30  # QB attempts up → TE targets up
    qb_te_yards: float = 0.40    # QB yards up → TE yards up

    # RB committee correlations (negative)
    rb_rb_carries: float = -0.55  # RB1 carries up → RB2 carries down
    rb_rb_targets: float = -0.30  # RB1 targets up → RB2 targets down

    # WR room correlations (weak negative)
    wr_wr_targets: float = -0.15  # WR1 targets up → WR2 targets slightly down

    # Cross-position correlations
    rb_wr_targets: float = -0.10  # RB targets up → WR targets slightly down

    # Game script correlations (applied to all players on same team)
    game_script_correlation: float = 0.25  # All players share game context

    # Minimum correlation (prevent numerical instability)
    min_correlation: float = -0.95
    max_correlation: float = 0.95


class PlayerCorrelationMatrix:
    """
    Builds and applies correlation matrices for same-team player simulations.

    Uses Cholesky decomposition to transform independent random variables
    into correlated random variables with the desired correlation structure.
    """

    def __init__(self, config: Optional[PlayerCorrelationConfig] = None):
        """
        Initialize correlation matrix builder.

        Args:
            config: Correlation configuration (uses defaults if None)
        """
        self.config = config or PlayerCorrelationConfig()

    def build_correlation_matrix(
        self,
        players: List[Dict[str, any]],
        team: str
    ) -> Tuple[np.ndarray, List[Dict[str, any]]]:
        """
        Build correlation matrix for a list of same-team players.

        Args:
            players: List of player dictionaries with keys:
                - player_id: str
                - position: str (QB, RB, WR, TE)
                - role: str (primary, secondary, etc.)
            team: Team abbreviation

        Returns:
            Tuple of (correlation_matrix, ordered_players)
            correlation_matrix: NxN symmetric positive semi-definite matrix
            ordered_players: Players in same order as correlation matrix rows/cols
        """
        n_players = len(players)

        if n_players == 0:
            return np.array([[]]), []

        if n_players == 1:
            return np.array([[1.0]]), players

        # Initialize correlation matrix (start with identity)
        corr_matrix = np.eye(n_players)

        # Fill in pairwise correlations
        for i in range(n_players):
            for j in range(i + 1, n_players):
                player_i = players[i]
                player_j = players[j]

                # Calculate correlation coefficient
                corr = self._calculate_pairwise_correlation(player_i, player_j)

                # Clip to safe bounds
                corr = np.clip(
                    corr,
                    self.config.min_correlation,
                    self.config.max_correlation
                )

                # Symmetric matrix
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Ensure positive semi-definite (required for Cholesky)
        corr_matrix = self._ensure_positive_semidefinite(corr_matrix)

        return corr_matrix, players

    def _calculate_pairwise_correlation(
        self,
        player_i: Dict[str, any],
        player_j: Dict[str, any]
    ) -> float:
        """
        Calculate correlation coefficient between two players.

        Args:
            player_i: First player dictionary
            player_j: Second player dictionary

        Returns:
            Correlation coefficient [-1, 1]
        """
        pos_i = player_i.get('position', '')
        pos_j = player_j.get('position', '')

        # QB-Receiver correlations (positive)
        if pos_i == 'QB' and pos_j in ['WR', 'TE']:
            if player_j.get('stat_type') == 'targets':
                return self.config.qb_wr_targets if pos_j == 'WR' else self.config.qb_te_targets
            elif player_j.get('stat_type') == 'yards':
                return self.config.qb_wr_yards if pos_j == 'WR' else self.config.qb_te_yards
            else:
                return self.config.game_script_correlation

        # Symmetric: WR/TE-QB
        if pos_j == 'QB' and pos_i in ['WR', 'TE']:
            if player_i.get('stat_type') == 'targets':
                return self.config.qb_wr_targets if pos_i == 'WR' else self.config.qb_te_targets
            elif player_i.get('stat_type') == 'yards':
                return self.config.qb_wr_yards if pos_i == 'WR' else self.config.qb_te_yards
            else:
                return self.config.game_script_correlation

        # RB committee (negative)
        if pos_i == 'RB' and pos_j == 'RB':
            stat_i = player_i.get('stat_type', 'carries')
            stat_j = player_j.get('stat_type', 'carries')

            if stat_i == 'carries' and stat_j == 'carries':
                return self.config.rb_rb_carries
            elif 'target' in stat_i or 'target' in stat_j:
                return self.config.rb_rb_targets
            else:
                return self.config.game_script_correlation

        # WR room (weak negative)
        if pos_i == 'WR' and pos_j == 'WR':
            # Primary vs secondary WR compete for targets
            role_i = player_i.get('role', 'secondary')
            role_j = player_j.get('role', 'secondary')

            if role_i == 'primary' and role_j == 'secondary':
                return self.config.wr_wr_targets
            elif role_j == 'primary' and role_i == 'secondary':
                return self.config.wr_wr_targets
            else:
                return self.config.game_script_correlation * 0.5

        # Cross-position (RB vs WR)
        if (pos_i == 'RB' and pos_j in ['WR', 'TE']) or (pos_j == 'RB' and pos_i in ['WR', 'TE']):
            return self.config.rb_wr_targets

        # Default: weak game script correlation
        return self.config.game_script_correlation

    def _ensure_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """
        Ensure correlation matrix is positive semi-definite.

        Uses eigenvalue decomposition to fix negative eigenvalues.

        Args:
            matrix: NxN correlation matrix

        Returns:
            Adjusted positive semi-definite matrix
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Clip negative eigenvalues to small positive value
        min_eigenvalue = 1e-8
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        # Reconstruct matrix
        adjusted = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Ensure diagonal is exactly 1.0 (correlation matrix property)
        diag_indices = np.diag_indices_from(adjusted)
        adjusted[diag_indices] = 1.0

        return adjusted

    def generate_correlated_samples(
        self,
        mean_values: np.ndarray,
        std_values: np.ndarray,
        correlation_matrix: np.ndarray,
        n_samples: int,
        random_state: Optional[int] = None,
        use_poisson: bool = False
    ) -> np.ndarray:
        """
        Generate correlated random samples using Cholesky decomposition.

        Args:
            mean_values: Array of means for each player (length N)
            std_values: Array of standard deviations for each player (length N)
            correlation_matrix: NxN correlation matrix
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            use_poisson: If True, use Poisson distribution for count data (targets, carries, attempts)

        Returns:
            samples: Array of shape (n_samples, N) with correlated values
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_players = len(mean_values)

        if n_players == 0:
            return np.array([])

        if n_players == 1:
            # No correlation needed
            if use_poisson:
                # Use Poisson for count data
                return np.random.poisson(
                    lam=mean_values[0],
                    size=(n_samples, 1)
                ).astype(float)
            else:
                return np.random.normal(
                    mean_values[0],
                    std_values[0],
                    size=(n_samples, 1)
                )

        if use_poisson:
            # Use Gaussian copula to generate correlated Poisson samples
            samples = self._generate_correlated_poisson(
                mean_values=mean_values,
                correlation_matrix=correlation_matrix,
                n_samples=n_samples
            )
        else:
            # Generate independent standard normal samples
            independent_samples = np.random.standard_normal(size=(n_samples, n_players))

            # Compute Cholesky decomposition of correlation matrix
            try:
                L = cholesky(correlation_matrix, lower=True)
            except np.linalg.LinAlgError:
                logger.warning("Cholesky decomposition failed, using independent samples")
                L = np.eye(n_players)

            # Apply Cholesky factor to get correlated samples
            correlated_standard = independent_samples @ L.T

            # Scale and shift to desired means and standard deviations
            samples = correlated_standard * std_values + mean_values

        return samples

    def _generate_correlated_poisson(
        self,
        mean_values: np.ndarray,
        correlation_matrix: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Generate correlated Poisson samples using Gaussian copula.

        This method:
        1. Generates correlated standard normal samples
        2. Transforms to uniform [0,1] via Normal CDF
        3. Transforms to Poisson via inverse Poisson CDF

        This preserves Poisson marginal distributions while maintaining correlations.

        Args:
            mean_values: Array of Poisson means (lambda parameters)
            correlation_matrix: Correlation matrix
            n_samples: Number of samples

        Returns:
            Correlated Poisson samples
        """
        n_players = len(mean_values)

        # Generate correlated standard normal samples
        independent_samples = np.random.standard_normal(size=(n_samples, n_players))

        try:
            L = cholesky(correlation_matrix, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed for Poisson copula, using independent samples")
            L = np.eye(n_players)

        correlated_normal = independent_samples @ L.T

        # Transform to uniform [0,1] via Normal CDF
        uniform_samples = stats.norm.cdf(correlated_normal)

        # Transform to Poisson via inverse CDF (quantile function)
        poisson_samples = np.zeros((n_samples, n_players))
        for i in range(n_players):
            # Use Poisson quantile function (ppf = percent point function = inverse CDF)
            poisson_samples[:, i] = stats.poisson.ppf(uniform_samples[:, i], mu=mean_values[i])

        return poisson_samples

    def apply_team_constraints(
        self,
        samples: np.ndarray,
        players: List[Dict[str, any]],
        team_totals: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply team-level constraints to player samples.

        Ensures that sum of player stats doesn't exceed team totals.
        Uses proportional scaling to maintain correlation structure.

        Args:
            samples: Array of shape (n_samples, n_players) with correlated values
            players: List of player dictionaries
            team_totals: Dict with team-level totals:
                - qb_attempts: Total QB pass attempts
                - rb_carries: Total RB carries
                - team_targets: Total targets to all positions

        Returns:
            Adjusted samples respecting team constraints
        """
        n_samples, n_players = samples.shape
        adjusted_samples = samples.copy()

        # Group players by stat type
        qb_pass_indices = []
        rb_carry_indices = []
        all_target_indices = []

        for i, player in enumerate(players):
            pos = player.get('position', '')
            stat_type = player.get('stat_type', '')

            if pos == 'QB' and 'attempt' in stat_type:
                qb_pass_indices.append(i)

            if pos == 'RB' and 'carr' in stat_type:
                rb_carry_indices.append(i)

            if 'target' in stat_type:
                all_target_indices.append(i)

        # Apply QB attempts constraint
        if qb_pass_indices and 'qb_attempts' in team_totals:
            max_attempts = team_totals['qb_attempts']
            for sample_idx in range(n_samples):
                qb_total = np.sum(adjusted_samples[sample_idx, qb_pass_indices])
                if qb_total > max_attempts:
                    # Scale down proportionally
                    scale_factor = max_attempts / qb_total
                    adjusted_samples[sample_idx, qb_pass_indices] *= scale_factor

        # Apply RB carries constraint
        if rb_carry_indices and 'rb_carries' in team_totals:
            max_carries = team_totals['rb_carries']
            for sample_idx in range(n_samples):
                rb_total = np.sum(adjusted_samples[sample_idx, rb_carry_indices])
                if rb_total > max_carries:
                    # Scale down proportionally
                    scale_factor = max_carries / rb_total
                    adjusted_samples[sample_idx, rb_carry_indices] *= scale_factor

        # Apply total targets constraint
        if all_target_indices and 'team_targets' in team_totals:
            max_targets = team_totals['team_targets']
            for sample_idx in range(n_samples):
                target_total = np.sum(adjusted_samples[sample_idx, all_target_indices])
                if target_total > max_targets:
                    # Scale down proportionally
                    scale_factor = max_targets / target_total
                    adjusted_samples[sample_idx, all_target_indices] *= scale_factor

        # Ensure non-negative values
        adjusted_samples = np.maximum(adjusted_samples, 0)

        return adjusted_samples


def build_team_correlation_matrix(
    team_players: pd.DataFrame,
    stat_type: str = 'targets'
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to build correlation matrix from a DataFrame of team players.

    Args:
        team_players: DataFrame with columns:
            - player_id: str
            - position: str
            - role: str (optional)
        stat_type: Type of stat being correlated ('targets', 'carries', 'yards', etc.)

    Returns:
        Tuple of (correlation_matrix, player_ids)
    """
    builder = PlayerCorrelationMatrix()

    # Convert DataFrame to list of dicts
    players = []
    for _, row in team_players.iterrows():
        players.append({
            'player_id': row.get('player_id', ''),
            'position': row.get('position', ''),
            'role': row.get('role', 'secondary'),
            'stat_type': stat_type
        })

    corr_matrix, ordered_players = builder.build_correlation_matrix(
        players=players,
        team=team_players.iloc[0].get('team', 'UNK') if len(team_players) > 0 else 'UNK'
    )

    player_ids = [p['player_id'] for p in ordered_players]

    return corr_matrix, player_ids


# Example usage and testing
if __name__ == '__main__':
    # Example: Build correlation matrix for a team's players
    players = [
        {'player_id': 'QB1', 'position': 'QB', 'stat_type': 'attempts'},
        {'player_id': 'WR1', 'position': 'WR', 'role': 'primary', 'stat_type': 'targets'},
        {'player_id': 'WR2', 'position': 'WR', 'role': 'secondary', 'stat_type': 'targets'},
        {'player_id': 'RB1', 'position': 'RB', 'role': 'primary', 'stat_type': 'carries'},
        {'player_id': 'RB2', 'position': 'RB', 'role': 'secondary', 'stat_type': 'carries'},
    ]

    builder = PlayerCorrelationMatrix()
    corr_matrix, ordered_players = builder.build_correlation_matrix(players, team='KC')

    print("Correlation Matrix:")
    print(corr_matrix)
    print("\nOrdered Players:")
    for i, p in enumerate(ordered_players):
        print(f"{i}: {p['player_id']} ({p['position']})")

    # Generate correlated samples
    mean_values = np.array([35.0, 8.0, 5.0, 15.0, 8.0])  # QB attempts, WR1 targets, WR2 targets, RB1 carries, RB2 carries
    std_values = np.array([5.0, 2.0, 1.5, 3.0, 2.0])

    samples = builder.generate_correlated_samples(
        mean_values=mean_values,
        std_values=std_values,
        correlation_matrix=corr_matrix,
        n_samples=10000,
        random_state=42
    )

    print("\nSample Statistics:")
    print(f"Means: {samples.mean(axis=0)}")
    print(f"Stds: {samples.std(axis=0)}")
    print(f"\nEmpirical Correlation Matrix:")
    print(np.corrcoef(samples.T))

    # Apply team constraints
    team_totals = {
        'qb_attempts': 38.0,
        'rb_carries': 25.0,
        'team_targets': 35.0
    }

    adjusted_samples = builder.apply_team_constraints(
        samples=samples,
        players=ordered_players,
        team_totals=team_totals
    )

    print("\nAdjusted Sample Statistics (with constraints):")
    print(f"Means: {adjusted_samples.mean(axis=0)}")
    print(f"QB attempts max: {adjusted_samples[:, 0].max()}")
    print(f"Total RB carries max: {(adjusted_samples[:, 3] + adjusted_samples[:, 4]).max()}")
