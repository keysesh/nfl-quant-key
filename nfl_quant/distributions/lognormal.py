"""
Lognormal distribution for efficiency metrics (yards/target, yards/carry).

Why Lognormal instead of Normal?
- Always positive (can't have negative yards/target)
- Right-skewed (big plays create long tail)
- Multiplicative effects (efficiency × usage = output)
- Natural model for ratios and rates

Mathematical Details:
- If Y ~ Lognormal(μ_log, σ_log), then log(Y) ~ Normal(μ_log, σ_log)
- Mean: E[Y] = exp(μ_log + σ_log²/2)
- Variance: Var[Y] = [exp(σ_log²) - 1] × exp(2μ_log + σ_log²)
- Coefficient of Variation: CV = sqrt(exp(σ_log²) - 1)

Conversion from (mean, CV) to (μ_log, σ_log):
- σ_log = sqrt(log(1 + CV²))
- μ_log = log(mean) - σ_log²/2
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def mean_cv_to_lognormal_params(mean: float, cv: float = 0.40) -> Tuple[float, float]:
    """
    Convert mean and coefficient of variation to Lognormal parameters.

    Args:
        mean: Mean of distribution (e.g., 9.5 yards/target)
        cv: Coefficient of variation (std/mean, typically 0.30-0.50 for NFL)
            CV=0.40 means std = 0.40 × mean

    Returns:
        (mu_log, sigma_log) tuple for scipy.stats.lognorm

    Example:
        >>> mean_cv_to_lognormal_params(9.5, 0.40)
        (2.17, 0.385)  # μ_log=2.17, σ_log=0.385
    """
    if mean <= 0:
        raise ValueError(f"Mean must be positive for Lognormal, got {mean:.2f}")

    if cv <= 0:
        raise ValueError(f"CV must be positive, got {cv:.2f}")

    # Standard conversion formulas
    sigma_log = np.sqrt(np.log(1 + cv ** 2))
    mu_log = np.log(mean) - (sigma_log ** 2) / 2

    return mu_log, sigma_log


def fit_lognormal_from_data(data: np.ndarray) -> Tuple[float, float]:
    """
    Fit Lognormal parameters from observed efficiency data.

    This uses method of moments:
    1. Calculate sample mean and CV
    2. Convert to (μ_log, σ_log) via mean_cv_to_lognormal_params()

    Args:
        data: Array of positive values (e.g., [8.5, 9.2, 7.8, 10.1, 9.0])

    Returns:
        (mu_log, sigma_log) tuple for scipy.stats.lognorm

    Example:
        >>> ypt = np.array([8.5, 9.2, 7.8, 10.1, 9.0, 8.8])
        >>> mu_log, sigma_log = fit_lognormal_from_data(ypt)
        >>> print(f"Lognormal(μ={mu_log:.2f}, σ={sigma_log:.3f})")
        Lognormal(μ=2.18, σ=0.102)
    """
    if len(data) == 0:
        raise ValueError("Cannot fit distribution to empty data")

    if np.any(data <= 0):
        raise ValueError("Lognormal requires all positive values")

    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))  # Sample std (N-1)
    cv = std / mean

    # Handle edge case: very low CV (minimal variation)
    if cv < 0.05:
        logger.warning(
            f"Very low CV detected (cv={cv:.3f}). "
            "Setting minimum CV=0.10 to avoid degenerate distribution."
        )
        cv = 0.10

    return mean_cv_to_lognormal_params(mean, cv)


class LognormalSampler:
    """
    Sampler for Lognormal distribution.

    This class provides a clean interface for sampling from Lognormal distributions
    in the V4 Monte Carlo simulation engine.

    Attributes:
        mu_log: Location parameter (log-space mean)
        sigma_log: Scale parameter (log-space std)
        mean: Distribution mean in original space
        median: Distribution median (= exp(mu_log))
        cv: Coefficient of variation

    Example:
        >>> sampler = LognormalSampler(mean=9.5, cv=0.40)
        >>> ypt = sampler.sample(size=10000)
        >>> print(f"Simulated mean: {ypt.mean():.2f}")
        Simulated mean: 9.48
    """

    def __init__(
        self,
        mean: Optional[float] = None,
        cv: Optional[float] = None,
        mu_log: Optional[float] = None,
        sigma_log: Optional[float] = None
    ):
        """
        Initialize Lognormal sampler from either (mean, CV) or (μ_log, σ_log).

        Args:
            mean: Distribution mean (requires cv)
            cv: Coefficient of variation (requires mean)
            mu_log: Log-space mean parameter (requires sigma_log)
            sigma_log: Log-space std parameter (requires mu_log)

        Raises:
            ValueError: If neither (mean, cv) nor (mu_log, sigma_log) provided
        """
        if mean is not None and cv is not None:
            # Convert from (mean, cv)
            self.mu_log, self.sigma_log = mean_cv_to_lognormal_params(mean, cv)
        elif mu_log is not None and sigma_log is not None:
            # Use (μ_log, σ_log) directly
            self.mu_log = mu_log
            self.sigma_log = sigma_log
        else:
            raise ValueError("Must provide either (mean, cv) or (mu_log, sigma_log)")

        # Calculate derived properties
        self.mean = np.exp(self.mu_log + (self.sigma_log ** 2) / 2)
        self.median = np.exp(self.mu_log)
        self.variance = (np.exp(self.sigma_log ** 2) - 1) * np.exp(2 * self.mu_log + self.sigma_log ** 2)
        self.cv = np.sqrt(np.exp(self.sigma_log ** 2) - 1)

        # Create scipy distribution object
        # Note: scipy uses different parameterization: lognorm(s, scale)
        # where s = sigma_log, scale = exp(mu_log)
        self._dist = stats.lognorm(s=self.sigma_log, scale=np.exp(self.mu_log))

        logger.debug(
            f"Created Lognormal(μ_log={self.mu_log:.2f}, σ_log={self.sigma_log:.3f}) "
            f"with mean={self.mean:.2f}, median={self.median:.2f}, cv={self.cv:.2f}"
        )

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the Lognormal distribution.

        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of sampled values (all positive)

        Example:
            >>> sampler = LognormalSampler(mean=9.5, cv=0.40)
            >>> samples = sampler.sample(size=10000, random_state=42)
            >>> samples[:10]
            array([11.2, 8.4, 9.7, 8.9, 7.2, 10.8, 9.3, 8.5, 9.1, 10.4])
        """
        if random_state is not None:
            np.random.seed(random_state)

        return self._dist.rvs(size=size)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function at x."""
        return self._dist.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function at x."""
        return self._dist.cdf(x)

    def quantile(self, q: float) -> float:
        """
        Quantile function (inverse CDF).

        Args:
            q: Quantile (0 to 1)

        Returns:
            Value at quantile q

        Example:
            >>> sampler = LognormalSampler(mean=9.5, cv=0.40)
            >>> sampler.quantile(0.5)  # Median
            9.15
            >>> sampler.quantile(0.95)  # 95th percentile
            15.3
        """
        return float(self._dist.ppf(q))

    def __repr__(self) -> str:
        return (
            f"LognormalSampler(μ_log={self.mu_log:.2f}, σ_log={self.sigma_log:.3f}, "
            f"mean={self.mean:.2f}, cv={self.cv:.2f})"
        )


def validate_lognormal_fit(
    data: np.ndarray,
    mu_log: float,
    sigma_log: float,
    alpha: float = 0.05
) -> Tuple[bool, str, float]:
    """
    Validate Lognormal fit using Kolmogorov-Smirnov test.

    Args:
        data: Observed positive data
        mu_log: Fitted μ_log parameter
        sigma_log: Fitted σ_log parameter
        alpha: Significance level (default 0.05)

    Returns:
        (is_valid, message, p_value) tuple

    Example:
        >>> data = np.random.lognormal(2.17, 0.385, size=100)
        >>> mu_log, sigma_log = fit_lognormal_from_data(data)
        >>> is_valid, msg, pval = validate_lognormal_fit(data, mu_log, sigma_log)
        >>> print(f"{msg} (p={pval:.3f})")
        Lognormal fit is valid (p=0.412)
    """
    dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(data, dist.cdf)

    is_valid = p_value >= alpha

    if is_valid:
        message = f"Lognormal fit is valid (KS p-value={p_value:.3f} >= {alpha})"
    else:
        message = (
            f"Lognormal fit may be poor (KS p-value={p_value:.3f} < {alpha}). "
            "Consider checking for outliers or bimodal distribution."
        )

    return is_valid, message, p_value


# Utility functions for common NFL scenarios

def estimate_ypt_cv_by_position(position: str) -> float:
    """
    Estimate typical CV for yards/target by position.

    Empirical analysis of NFL data shows position-specific CV patterns.

    Args:
        position: Player position (QB, RB, WR, TE)

    Returns:
        Typical CV for that position

    Example:
        >>> estimate_ypt_cv_by_position('WR')
        0.42  # WR have highest CV (big play potential)
    """
    cv_by_position = {
        'WR': 0.42,  # Highest variance (deep threats vs possession)
        'TE': 0.38,  # Moderate variance
        'RB': 0.45,  # High variance (dump-offs vs screens)
        'QB': 0.35,  # Lowest variance (passing efficiency)
    }

    return cv_by_position.get(position, 0.40)  # Default 0.40


def sample_yards_per_target_lognormal(
    mean_ypt: float,
    position: str = 'WR',
    size: int = 10000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to sample yards/target using Lognormal.

    Args:
        mean_ypt: Mean yards per target
        position: Player position (for CV estimation)
        size: Number of samples
        random_state: Random seed

    Returns:
        Array of sampled Y/T values

    Example:
        >>> ypt = sample_yards_per_target_lognormal(9.5, 'WR', size=10000, random_state=42)
        >>> print(f"Mean: {ypt.mean():.2f}, Median: {np.median(ypt):.2f}")
        Mean: 9.48, Median: 9.15
    """
    cv = estimate_ypt_cv_by_position(position)
    sampler = LognormalSampler(mean=mean_ypt, cv=cv)
    return sampler.sample(size=size, random_state=random_state)


def estimate_ypc_cv_by_position(position: str) -> float:
    """
    Estimate typical CV for yards/carry by position.

    Args:
        position: Player position (RB, QB)

    Returns:
        Typical CV for that position

    Example:
        >>> estimate_ypc_cv_by_position('RB')
        0.50  # RB YPC has high variance (big runs vs stuffed)
    """
    cv_by_position = {
        'RB': 0.50,  # High variance (home runs vs contact at LOS)
        'QB': 0.55,  # Very high variance (scrambles, designed runs, sacks)
    }

    return cv_by_position.get(position, 0.50)  # Default 0.50


def sample_yards_per_carry_lognormal(
    mean_ypc: float,
    position: str = 'RB',
    size: int = 10000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to sample yards/carry using Lognormal.

    Args:
        mean_ypc: Mean yards per carry
        position: Player position (for CV estimation)
        size: Number of samples
        random_state: Random seed

    Returns:
        Array of sampled YPC values

    Example:
        >>> ypc = sample_yards_per_carry_lognormal(4.5, 'RB', size=10000, random_state=42)
        >>> print(f"Mean: {ypc.mean():.2f}, P95: {np.percentile(ypc, 95):.2f}")
        Mean: 4.48, P95: 8.92
    """
    cv = estimate_ypc_cv_by_position(position)
    sampler = LognormalSampler(mean=mean_ypc, cv=cv)
    return sampler.sample(size=size, random_state=random_state)
