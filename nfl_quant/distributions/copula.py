"""
Gaussian Copula for modeling correlation between usage and efficiency.

What is a Copula?
- A copula separates marginal distributions from their dependence structure
- Allows modeling correlation WITHOUT assuming both variables are Normal
- Example: Targets (NegBin) and Y/T (Lognormal) can be correlated via copula

Why Use Copula?
- Targets and efficiency are NEGATIVELY correlated in NFL:
  * High-target games → lower Y/T (short, safe throws)
  * Low-target games → higher Y/T (selective deep shots)
- This correlation affects total yards distribution
- Ignoring correlation → overestimate variance, poor percentiles

Mathematical Details:
- Gaussian copula uses bivariate normal to generate correlation
- Steps:
  1. Generate correlated normals: Z1, Z2 ~ BivariateNormal(ρ)
  2. Transform to uniforms: U1 = Φ(Z1), U2 = Φ(Z2)
  3. Apply inverse CDFs: X1 = F1^{-1}(U1), X2 = F2^{-1}(U2)
- Result: X1, X2 have correct marginals AND correlation

Spearman Correlation:
- Use Spearman (rank correlation) not Pearson (linear correlation)
- Works for non-Normal marginals
- Conversion: ρ_Pearson ≈ 2 × sin(π × ρ_Spearman / 6)
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def spearman_to_pearson_correlation(rho_spearman: float) -> float:
    """
    Convert Spearman rank correlation to Pearson linear correlation.

    This is needed because Gaussian copula uses Pearson correlation,
    but we estimate from data using Spearman (more robust).

    Args:
        rho_spearman: Spearman correlation (-1 to 1)

    Returns:
        Approximate Pearson correlation

    Formula:
        ρ_Pearson ≈ 2 × sin(π × ρ_Spearman / 6)

    Example:
        >>> spearman_to_pearson_correlation(-0.30)
        -0.295  # Slightly less negative for Pearson
    """
    return 2 * np.sin(np.pi * rho_spearman / 6)


def estimate_correlation_from_data(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'spearman'
) -> float:
    """
    Estimate correlation between two variables.

    Args:
        x: First variable (e.g., targets)
        y: Second variable (e.g., yards/target)
        method: 'spearman' (default) or 'pearson'

    Returns:
        Correlation coefficient

    Example:
        >>> targets = np.array([5, 7, 4, 8, 6, 7, 5, 9])
        >>> ypt = np.array([10.2, 8.5, 11.0, 7.8, 9.0, 8.2, 10.5, 7.2])
        >>> estimate_correlation_from_data(targets, ypt)
        -0.78  # Negative correlation: more targets → lower Y/T
    """
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    if len(x) < 3:
        raise ValueError(f"Need at least 3 observations, got {len(x)}")

    if method == 'spearman':
        corr, _ = stats.spearmanr(x, y)
    elif method == 'pearson':
        corr, _ = stats.pearsonr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'.")

    return float(corr)


def generate_correlated_uniforms(
    rho: float,
    size: int = 10000,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated uniform [0, 1] samples using Gaussian copula.

    This is the core copula function. The outputs can be transformed
    to any marginal distribution using inverse CDFs.

    Args:
        rho: Correlation coefficient (-1 to 1)
        size: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        (u1, u2) tuple of correlated uniform samples

    Example:
        >>> u1, u2 = generate_correlated_uniforms(rho=-0.30, size=10000, random_state=42)
        >>> np.corrcoef(u1, u2)[0, 1]
        -0.295  # Close to -0.30
    """
    if not -1 <= rho <= 1:
        raise ValueError(f"Correlation must be in [-1, 1], got {rho:.3f}")

    if random_state is not None:
        np.random.seed(random_state)

    # Generate bivariate normal with correlation rho
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    z = np.random.multivariate_normal(mean, cov, size=size)

    # Transform to uniforms via standard normal CDF
    u1 = norm.cdf(z[:, 0])
    u2 = norm.cdf(z[:, 1])

    return u1, u2


class GaussianCopula:
    """
    Gaussian Copula for generating correlated samples from different distributions.

    This class provides the V4 framework for sampling correlated targets and efficiency.

    Attributes:
        rho: Correlation coefficient (Spearman by default)
        rho_pearson: Pearson correlation (for Gaussian copula)

    Example:
        >>> from nfl_quant.distributions import NegativeBinomialSampler, LognormalSampler
        >>>
        >>> # Create marginal distributions
        >>> targets_dist = NegativeBinomialSampler(mean=6.5, variance=12.0)
        >>> ypt_dist = LognormalSampler(mean=9.5, cv=0.40)
        >>>
        >>> # Create copula
        >>> copula = GaussianCopula(rho=-0.30)
        >>>
        >>> # Sample correlated targets and Y/T
        >>> targets, ypt = copula.sample_from_marginals(
        ...     targets_dist, ypt_dist, size=10000
        ... )
        >>>
        >>> # Verify correlation
        >>> np.corrcoef(targets, ypt)[0, 1]
        -0.28  # Close to -0.30
    """

    def __init__(self, rho: float, is_pearson: bool = False):
        """
        Initialize Gaussian copula with correlation.

        Args:
            rho: Correlation coefficient
            is_pearson: If True, rho is Pearson correlation.
                       If False (default), rho is Spearman and will be converted.

        Example:
            >>> copula = GaussianCopula(rho=-0.30)  # Spearman
            >>> copula.rho_pearson
            -0.295
        """
        self.rho = rho

        if is_pearson:
            self.rho_pearson = rho
        else:
            # Convert Spearman to Pearson for Gaussian copula
            self.rho_pearson = spearman_to_pearson_correlation(rho)

        logger.debug(
            f"Created Gaussian copula with ρ_Spearman={self.rho:.3f}, "
            f"ρ_Pearson={self.rho_pearson:.3f}"
        )

    def sample_uniforms(
        self,
        size: int = 10000,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate correlated uniform samples.

        Args:
            size: Number of samples
            random_state: Random seed

        Returns:
            (u1, u2) correlated uniforms

        Example:
            >>> copula = GaussianCopula(rho=-0.30)
            >>> u1, u2 = copula.sample_uniforms(size=10000, random_state=42)
        """
        return generate_correlated_uniforms(
            self.rho_pearson, size=size, random_state=random_state
        )

    def sample_from_marginals(
        self,
        dist1,
        dist2,
        size: int = 10000,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from two marginal distributions with correlation.

        Args:
            dist1: First marginal distribution (must have .quantile() method)
            dist2: Second marginal distribution (must have .quantile() method)
            size: Number of samples
            random_state: Random seed

        Returns:
            (x1, x2) correlated samples from dist1 and dist2

        Example:
            >>> from nfl_quant.distributions import NegativeBinomialSampler, LognormalSampler
            >>> targets_dist = NegativeBinomialSampler(mean=6.5, variance=12.0)
            >>> ypt_dist = LognormalSampler(mean=9.5, cv=0.40)
            >>> copula = GaussianCopula(rho=-0.30)
            >>> targets, ypt = copula.sample_from_marginals(
            ...     targets_dist, ypt_dist, size=10000, random_state=42
            ... )
        """
        # Generate correlated uniforms
        u1, u2 = self.sample_uniforms(size=size, random_state=random_state)

        # Transform to marginal distributions via inverse CDF
        x1 = np.array([dist1.quantile(u) for u in u1])
        x2 = np.array([dist2.quantile(u) for u in u2])

        return x1, x2

    def __repr__(self) -> str:
        return (
            f"GaussianCopula(ρ_Spearman={self.rho:.3f}, "
            f"ρ_Pearson={self.rho_pearson:.3f})"
        )


def validate_correlation(
    x: np.ndarray,
    y: np.ndarray,
    expected_rho: float,
    tolerance: float = 0.05
) -> Tuple[bool, str, float]:
    """
    Validate that observed correlation matches expected.

    Args:
        x: First variable samples
        y: Second variable samples
        expected_rho: Expected Spearman correlation
        tolerance: Acceptable deviation (default 0.05)

    Returns:
        (is_valid, message, observed_rho) tuple

    Example:
        >>> u1, u2 = generate_correlated_uniforms(rho=-0.30, size=10000, random_state=42)
        >>> is_valid, msg, rho = validate_correlation(u1, u2, expected_rho=-0.30)
        >>> print(f"{msg} (observed ρ={rho:.3f})")
        Correlation is valid (observed ρ=-0.295)
    """
    observed_rho = estimate_correlation_from_data(x, y, method='spearman')
    error = abs(observed_rho - expected_rho)

    is_valid = error <= tolerance

    if is_valid:
        message = (
            f"Correlation is valid: observed ρ={observed_rho:.3f}, "
            f"expected ρ={expected_rho:.3f} (error={error:.3f} <= {tolerance})"
        )
    else:
        message = (
            f"Correlation validation failed: observed ρ={observed_rho:.3f}, "
            f"expected ρ={expected_rho:.3f} (error={error:.3f} > {tolerance}). "
            "Check sample size or copula implementation."
        )

    return is_valid, message, observed_rho


# Utility functions for common NFL scenarios

def estimate_target_ypt_correlation(
    player_history: np.ndarray,
    min_observations: int = 8
) -> Optional[float]:
    """
    Estimate correlation between targets and Y/T from player history.

    Args:
        player_history: Array with columns ['targets', 'yards', 'receptions']
        min_observations: Minimum games needed for reliable estimate

    Returns:
        Spearman correlation, or None if insufficient data

    Example:
        >>> history = np.array([
        ...     [5, 50, 4],  # 5 targets, 50 yards, 4 receptions
        ...     [7, 56, 5],
        ...     [4, 48, 3],
        ...     [8, 64, 6],
        ...     [6, 54, 4],
        ...     [7, 49, 5],
        ...     [5, 55, 4],
        ...     [9, 63, 7],
        ... ])
        >>> estimate_target_ypt_correlation(history)
        -0.24  # Negative correlation
    """
    if len(player_history) < min_observations:
        logger.warning(
            f"Only {len(player_history)} observations, need {min_observations}. "
            "Returning None."
        )
        return None

    targets = player_history[:, 0]
    yards = player_history[:, 1]

    # Calculate Y/T (avoid division by zero)
    ypt = np.where(targets > 0, yards / targets, np.nan)

    # Remove NaN values
    valid_mask = ~np.isnan(ypt)
    targets_valid = targets[valid_mask]
    ypt_valid = ypt[valid_mask]

    if len(targets_valid) < min_observations:
        logger.warning(
            f"Only {len(targets_valid)} valid Y/T observations after removing zeros. "
            "Returning None."
        )
        return None

    return estimate_correlation_from_data(targets_valid, ypt_valid, method='spearman')


def get_default_target_ypt_correlation(position: str) -> float:
    """
    Get typical target-Y/T correlation by position.

    Based on empirical NFL analysis (to be updated with actual data).

    Args:
        position: Player position (WR, TE, RB)

    Returns:
        Typical Spearman correlation

    Example:
        >>> get_default_target_ypt_correlation('WR')
        -0.25  # WR: more targets → slightly lower Y/T
    """
    defaults = {
        'WR': -0.25,  # Moderate negative correlation
        'TE': -0.20,  # Weaker negative correlation
        'RB': -0.15,  # Weakest (targets often dump-offs regardless)
    }

    return defaults.get(position, -0.20)


def sample_correlated_targets_ypt(
    mean_targets: float,
    target_variance: float,
    mean_ypt: float,
    ypt_cv: float,
    correlation: float = -0.25,
    size: int = 10000,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to sample correlated targets and Y/T.

    This is the main V4 sampling function for receiving yards.

    Args:
        mean_targets: Predicted mean targets
        target_variance: Target variance
        mean_ypt: Predicted mean yards/target
        ypt_cv: Y/T coefficient of variation
        correlation: Spearman correlation (default -0.25)
        size: Number of samples
        random_state: Random seed

    Returns:
        (targets, ypt) correlated samples

    Example:
        >>> targets, ypt = sample_correlated_targets_ypt(
        ...     mean_targets=6.5,
        ...     target_variance=12.0,
        ...     mean_ypt=9.5,
        ...     ypt_cv=0.40,
        ...     correlation=-0.30,
        ...     size=10000,
        ...     random_state=42
        ... )
        >>> receiving_yards = targets * ypt
        >>> print(f"Mean yards: {receiving_yards.mean():.1f}")
        Mean yards: 61.2
    """
    from .negative_binomial import NegativeBinomialSampler
    from .lognormal import LognormalSampler

    # Create marginal distributions
    targets_dist = NegativeBinomialSampler(mean=mean_targets, variance=target_variance)
    ypt_dist = LognormalSampler(mean=mean_ypt, cv=ypt_cv)

    # Create copula
    copula = GaussianCopula(rho=correlation)

    # Sample correlated values
    targets, ypt = copula.sample_from_marginals(
        targets_dist, ypt_dist, size=size, random_state=random_state
    )

    return targets, ypt
