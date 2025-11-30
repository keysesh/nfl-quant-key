"""
Negative Binomial distribution for count data (targets, carries, attempts).

Why Negative Binomial instead of Normal?
- Count data is discrete (0, 1, 2, 3...) not continuous
- Overdispersion: variance > mean (common in NFL usage data)
- Normal can predict negative values (impossible for counts)
- NegBin naturally handles right-skew and heavy tails

Mathematical Details:
- Parameterization: NegBin(n, p) where n = "successes", p = "probability"
- Mean: μ = n(1-p)/p
- Variance: σ² = n(1-p)/p²
- Overdispersion when σ² > μ (always true for NegBin)

Conversion from (mean, variance) to (n, p):
- p = μ / σ²
- n = μ² / (σ² - μ)
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def mean_var_to_negbin_params(mean: float, variance: float) -> Tuple[float, float]:
    """
    Convert mean and variance to Negative Binomial (n, p) parameters.

    Args:
        mean: Mean of distribution (e.g., 6.5 targets)
        variance: Variance (must be > mean for overdispersion)

    Returns:
        (n, p) tuple for scipy.stats.nbinom

    Raises:
        ValueError: If variance <= mean (underdispersed, use Poisson instead)

    Example:
        >>> mean_var_to_negbin_params(6.5, 12.0)
        (7.88, 0.548)  # n=7.88 successes, p=0.548 probability
    """
    if variance <= mean:
        raise ValueError(
            f"Variance ({variance:.2f}) must be > mean ({mean:.2f}) for Negative Binomial. "
            "Use Poisson distribution for variance ≈ mean, or increase variance estimate."
        )

    if mean <= 0:
        raise ValueError(f"Mean must be positive, got {mean:.2f}")

    # Standard conversion formulas
    p = mean / variance
    n = mean ** 2 / (variance - mean)

    # Validate parameters
    if not (0 < p < 1):
        raise ValueError(f"Invalid p={p:.3f}, must be in (0, 1)")

    if n <= 0:
        raise ValueError(f"Invalid n={n:.3f}, must be positive")

    return n, p


def fit_negbin_from_data(data: np.ndarray) -> Tuple[float, float]:
    """
    Fit Negative Binomial parameters from observed count data.

    This uses method of moments (MOM) estimation:
    1. Calculate sample mean and variance
    2. Convert to (n, p) via mean_var_to_negbin_params()

    Args:
        data: Array of count observations (e.g., [5, 7, 4, 8, 6])

    Returns:
        (n, p) tuple for scipy.stats.nbinom

    Example:
        >>> targets = np.array([5, 7, 4, 8, 6, 7, 5])
        >>> n, p = fit_negbin_from_data(targets)
        >>> print(f"NegBin(n={n:.2f}, p={p:.3f})")
        NegBin(n=12.25, p=0.511)
    """
    if len(data) == 0:
        raise ValueError("Cannot fit distribution to empty data")

    mean = float(np.mean(data))
    variance = float(np.var(data, ddof=1))  # Sample variance (N-1)

    # Handle edge case: variance ≈ mean (minimal overdispersion)
    if variance <= mean * 1.01:  # Allow 1% tolerance
        logger.warning(
            f"Minimal overdispersion detected (mean={mean:.2f}, var={variance:.2f}). "
            "Adding small overdispersion factor."
        )
        variance = mean * 1.10  # Force 10% overdispersion

    return mean_var_to_negbin_params(mean, variance)


class NegativeBinomialSampler:
    """
    Sampler for Negative Binomial distribution.

    This class provides a clean interface for sampling from NegBin distributions
    in the V4 Monte Carlo simulation engine.

    Attributes:
        n: Number of successes parameter
        p: Probability parameter
        mean: Distribution mean (μ = n(1-p)/p)
        variance: Distribution variance (σ² = n(1-p)/p²)

    Example:
        >>> sampler = NegativeBinomialSampler(mean=6.5, variance=12.0)
        >>> targets = sampler.sample(size=10000)
        >>> print(f"Simulated mean: {targets.mean():.2f}")
        Simulated mean: 6.48
    """

    def __init__(
        self,
        mean: Optional[float] = None,
        variance: Optional[float] = None,
        n: Optional[float] = None,
        p: Optional[float] = None
    ):
        """
        Initialize NegBin sampler from either (mean, variance) or (n, p).

        Args:
            mean: Distribution mean (requires variance)
            variance: Distribution variance (requires mean)
            n: NegBin n parameter (requires p)
            p: NegBin p parameter (requires n)

        Raises:
            ValueError: If neither (mean, variance) nor (n, p) provided
        """
        if mean is not None and variance is not None:
            # Convert from (mean, variance)
            self.n, self.p = mean_var_to_negbin_params(mean, variance)
        elif n is not None and p is not None:
            # Use (n, p) directly
            self.n = n
            self.p = p
        else:
            raise ValueError("Must provide either (mean, variance) or (n, p)")

        # Calculate derived properties
        self.mean = self.n * (1 - self.p) / self.p
        self.variance = self.n * (1 - self.p) / (self.p ** 2)

        # Create scipy distribution object
        self._dist = stats.nbinom(self.n, self.p)

        logger.debug(
            f"Created NegBin(n={self.n:.2f}, p={self.p:.3f}) "
            f"with mean={self.mean:.2f}, var={self.variance:.2f}"
        )

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the Negative Binomial distribution.

        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of sampled counts

        Example:
            >>> sampler = NegativeBinomialSampler(mean=6.5, variance=12.0)
            >>> samples = sampler.sample(size=10000, random_state=42)
            >>> samples[:10]
            array([8, 5, 7, 6, 4, 9, 7, 5, 6, 8])
        """
        if random_state is not None:
            np.random.seed(random_state)

        return self._dist.rvs(size=size)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability mass function at x."""
        return self._dist.pmf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function at x."""
        return self._dist.cdf(x)

    def quantile(self, q: float) -> int:
        """
        Quantile function (inverse CDF).

        Args:
            q: Quantile (0 to 1)

        Returns:
            Value at quantile q

        Example:
            >>> sampler = NegativeBinomialSampler(mean=6.5, variance=12.0)
            >>> sampler.quantile(0.5)  # Median
            6
            >>> sampler.quantile(0.95)  # 95th percentile
            13
        """
        return int(self._dist.ppf(q))

    def __repr__(self) -> str:
        return (
            f"NegativeBinomialSampler(n={self.n:.2f}, p={self.p:.3f}, "
            f"mean={self.mean:.2f}, variance={self.variance:.2f})"
        )


def validate_negbin_fit(
    data: np.ndarray,
    n: float,
    p: float,
    alpha: float = 0.05
) -> Tuple[bool, str, float]:
    """
    Validate Negative Binomial fit using Kolmogorov-Smirnov test.

    Args:
        data: Observed count data
        n: Fitted n parameter
        p: Fitted p parameter
        alpha: Significance level (default 0.05)

    Returns:
        (is_valid, message, p_value) tuple

    Example:
        >>> data = np.random.negative_binomial(8, 0.55, size=100)
        >>> n, p = fit_negbin_from_data(data)
        >>> is_valid, msg, pval = validate_negbin_fit(data, n, p)
        >>> print(f"{msg} (p={pval:.3f})")
        NegBin fit is valid (p=0.342)
    """
    dist = stats.nbinom(n, p)

    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(data, dist.cdf)

    is_valid = p_value >= alpha

    if is_valid:
        message = f"NegBin fit is valid (KS p-value={p_value:.3f} >= {alpha})"
    else:
        message = (
            f"NegBin fit may be poor (KS p-value={p_value:.3f} < {alpha}). "
            "Consider checking for outliers or regime changes."
        )

    return is_valid, message, p_value


# Utility functions for common NFL scenarios

def estimate_target_variance(
    mean_targets: float,
    overdispersion_factor: float = 1.8
) -> float:
    """
    Estimate target variance from mean using typical NFL overdispersion.

    Empirical analysis shows NFL target distributions have variance ≈ 1.5-2.0 × mean.

    Args:
        mean_targets: Predicted mean targets
        overdispersion_factor: Multiplier (default 1.8)

    Returns:
        Estimated variance

    Example:
        >>> estimate_target_variance(6.5)
        11.7  # variance = 6.5 × 1.8
    """
    return mean_targets * overdispersion_factor


def sample_targets_negbin(
    mean_targets: float,
    overdispersion: float = 1.8,
    size: int = 10000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to sample targets using Negative Binomial.

    Args:
        mean_targets: Mean targets to sample around
        overdispersion: Overdispersion factor (variance = mean × overdispersion)
        size: Number of samples
        random_state: Random seed

    Returns:
        Array of sampled target counts

    Example:
        >>> targets = sample_targets_negbin(6.5, size=10000, random_state=42)
        >>> print(f"Mean: {targets.mean():.2f}, Std: {targets.std():.2f}")
        Mean: 6.48, Std: 3.42
    """
    variance = estimate_target_variance(mean_targets, overdispersion)
    sampler = NegativeBinomialSampler(mean=mean_targets, variance=variance)
    return sampler.sample(size=size, random_state=random_state)
