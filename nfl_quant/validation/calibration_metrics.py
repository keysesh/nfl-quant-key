"""
Calibration and Validation Metrics for Probabilistic Forecasts.

Implements research-backed metrics for evaluating distribution calibration:
1. CRPS (Continuous Ranked Probability Score) - proper scoring rule
2. PIT (Probability Integral Transform) histograms - visual calibration check
3. Calibration curves - binned reliability diagrams
4. Brier score - for binary outcomes (over/under)
5. Log-likelihood score

Key Concepts:
- Well-calibrated forecast: Predicted probabilities match observed frequencies
- CRPS: Measures distance between predicted distribution and actual outcome
  - Lower is better
  - Units: Same as the forecast variable (e.g., points, yards)
- PIT: If model is calibrated, PIT should be uniform [0,1]
- Proper scoring rules: Cannot be gamed by hedging
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


class CalibrationMetrics:
    """
    Calculates calibration metrics for probabilistic forecasts.
    """

    @staticmethod
    def crps(
        forecast_samples: np.ndarray,
        actual_outcome: float
    ) -> float:
        """
        Calculate Continuous Ranked Probability Score (CRPS).

        CRPS measures the distance between the forecast distribution
        and the actual outcome. Lower is better.

        Formula (empirical):
        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        where X, X' are independent samples from forecast distribution, y is actual

        Args:
            forecast_samples: Array of forecast samples (e.g., Monte Carlo)
            actual_outcome: Actual observed outcome

        Returns:
            CRPS value (lower is better)
        """
        n = len(forecast_samples)

        # Term 1: E[|X - y|]
        term1 = np.mean(np.abs(forecast_samples - actual_outcome))

        # Term 2: 0.5 * E[|X - X'|]
        # Approximate by pairwise differences (computationally expensive for large n)
        # Use subsample if n > 1000
        if n > 1000:
            subsample_size = 1000
            indices = np.random.choice(n, subsample_size, replace=False)
            samples_sub = forecast_samples[indices]
        else:
            samples_sub = forecast_samples

        # Pairwise differences
        pairwise_diffs = np.abs(samples_sub[:, None] - samples_sub[None, :])
        term2 = 0.5 * np.mean(pairwise_diffs)

        crps = term1 - term2

        return crps

    @staticmethod
    def crps_normal(
        mean: float,
        std: float,
        actual_outcome: float
    ) -> float:
        """
        Calculate CRPS for normal distribution (closed form).

        Faster than empirical CRPS when forecast is normal.

        Formula:
        CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
        where z = (y - μ) / σ, Φ = normal CDF, φ = normal PDF

        Args:
            mean: Forecast mean
            std: Forecast standard deviation
            actual_outcome: Actual observed outcome

        Returns:
            CRPS value
        """
        z = (actual_outcome - mean) / std

        # Normal CDF and PDF
        phi_z = stats.norm.cdf(z)
        pdf_z = stats.norm.pdf(z)

        crps = std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))

        return crps

    @staticmethod
    def pit(
        forecast_samples: np.ndarray,
        actual_outcome: float
    ) -> float:
        """
        Calculate Probability Integral Transform (PIT) value.

        PIT = CDF(actual_outcome)

        For a well-calibrated forecast, PIT values across many forecasts
        should be uniformly distributed [0, 1].

        Args:
            forecast_samples: Array of forecast samples
            actual_outcome: Actual observed outcome

        Returns:
            PIT value (0 to 1)
        """
        # Empirical CDF at actual outcome
        pit_value = np.mean(forecast_samples <= actual_outcome)

        return pit_value

    @staticmethod
    def pit_histogram(
        pit_values: np.ndarray,
        n_bins: int = 10,
        plot: bool = False,
        title: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Create PIT histogram for calibration assessment.

        A well-calibrated forecast should have a uniform PIT histogram.
        Deviations indicate miscalibration:
        - U-shaped: Overdispersed (too much variance)
        - Inverted-U: Underdispersed (too little variance)
        - Left-skewed: Biased low (forecasts too high)
        - Right-skewed: Biased high (forecasts too low)

        Args:
            pit_values: Array of PIT values from multiple forecasts
            n_bins: Number of bins (default 10 for deciles)
            plot: Whether to plot histogram
            title: Optional plot title

        Returns:
            Dictionary with:
                - histogram: Bin counts
                - bin_edges: Bin edges
                - uniformity_pvalue: Chi-square test p-value (H0: uniform)
        """
        # Create histogram
        hist, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))

        # Chi-square test for uniformity
        expected_count = len(pit_values) / n_bins
        chi2_stat = np.sum((hist - expected_count) ** 2 / expected_count)
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)

        # Plot if requested
        if plot:
            plt.figure(figsize=(8, 5))
            plt.hist(pit_values, bins=n_bins, range=(0, 1), edgecolor='black', alpha=0.7)
            plt.axhline(y=len(pit_values) / n_bins, color='r', linestyle='--', label='Expected (uniform)')
            plt.xlabel('PIT Value')
            plt.ylabel('Frequency')
            plt.title(title or 'PIT Histogram (Calibration Check)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return {
            'histogram': hist,
            'bin_edges': bin_edges,
            'uniformity_pvalue': chi2_pvalue,
            'is_uniform': chi2_pvalue > 0.05  # 5% significance level
        }

    @staticmethod
    def calibration_curve(
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_bins: int = 10,
        plot: bool = False,
        title: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create calibration curve (reliability diagram).

        Bins predictions and compares predicted probability to observed frequency.

        Args:
            predicted_probs: Array of predicted probabilities (0-1)
            actual_outcomes: Array of actual binary outcomes (0 or 1)
            n_bins: Number of bins
            plot: Whether to plot calibration curve
            title: Optional plot title

        Returns:
            DataFrame with columns:
                - bin_center: Midpoint of probability bin
                - predicted_prob: Mean predicted probability in bin
                - observed_freq: Observed frequency in bin
                - count: Number of observations in bin
        """
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate statistics per bin
        results = []
        for i in range(n_bins):
            mask = bin_indices == i
            count = mask.sum()

            if count > 0:
                pred_mean = predicted_probs[mask].mean()
                obs_freq = actual_outcomes[mask].mean()
            else:
                pred_mean = np.nan
                obs_freq = np.nan

            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2

            results.append({
                'bin_center': bin_center,
                'predicted_prob': pred_mean,
                'observed_freq': obs_freq,
                'count': count
            })

        df = pd.DataFrame(results)

        # Plot if requested
        if plot:
            plt.figure(figsize=(8, 8))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

            # Remove empty bins
            df_plot = df[df['count'] > 0]

            plt.scatter(
                df_plot['predicted_prob'],
                df_plot['observed_freq'],
                s=df_plot['count'] * 2,  # Size by count
                alpha=0.6,
                label='Observed'
            )

            plt.xlabel('Predicted Probability')
            plt.ylabel('Observed Frequency')
            plt.title(title or 'Calibration Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()

        return df

    @staticmethod
    def brier_score(
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> float:
        """
        Calculate Brier score for binary outcomes.

        Brier score = mean((predicted - actual)²)

        Lower is better. Range: [0, 1]
        - 0 = perfect predictions
        - 1 = worst predictions

        Args:
            predicted_probs: Array of predicted probabilities (0-1)
            actual_outcomes: Array of actual binary outcomes (0 or 1)

        Returns:
            Brier score (lower is better)
        """
        return brier_score_loss(actual_outcomes, predicted_probs)

    @staticmethod
    def log_score(
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> float:
        """
        Calculate log-likelihood score (cross-entropy loss).

        Lower is better.

        Args:
            predicted_probs: Array of predicted probabilities (0-1)
            actual_outcomes: Array of actual binary outcomes (0 or 1)

        Returns:
            Log score (lower is better)
        """
        # Clip to avoid log(0)
        predicted_probs_clipped = np.clip(predicted_probs, 1e-15, 1 - 1e-15)

        return log_loss(actual_outcomes, predicted_probs_clipped)

    @staticmethod
    def coverage_probability(
        forecast_intervals: List[Tuple[float, float]],
        actual_outcomes: np.ndarray,
        confidence_level: float = 0.90
    ) -> Dict[str, float]:
        """
        Calculate coverage probability for prediction intervals.

        For a well-calibrated model, 90% intervals should contain the actual
        outcome 90% of the time.

        Args:
            forecast_intervals: List of (lower, upper) prediction intervals
            actual_outcomes: Array of actual outcomes
            confidence_level: Expected confidence level (e.g., 0.90 for 90%)

        Returns:
            Dictionary with:
                - coverage: Observed coverage rate
                - expected: Expected coverage rate
                - is_calibrated: Whether coverage matches expectation
        """
        n = len(actual_outcomes)
        assert n == len(forecast_intervals), "Mismatch in number of forecasts and actuals"

        # Count how many intervals contain the actual
        contains_actual = np.array([
            lower <= actual <= upper
            for (lower, upper), actual in zip(forecast_intervals, actual_outcomes)
        ])

        observed_coverage = contains_actual.mean()

        # Check if within ±5% of expected
        is_calibrated = abs(observed_coverage - confidence_level) < 0.05

        return {
            'coverage': observed_coverage,
            'expected': confidence_level,
            'is_calibrated': is_calibrated,
            'deviation': observed_coverage - confidence_level
        }


def validate_forecast_distributions(
    forecast_samples_list: List[np.ndarray],
    actual_outcomes: np.ndarray,
    forecast_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Comprehensive validation of forecast distributions.

    Args:
        forecast_samples_list: List of forecast sample arrays (one per forecast)
        actual_outcomes: Array of actual outcomes (same length as forecast_samples_list)
        forecast_names: Optional names for each forecast

    Returns:
        DataFrame with validation metrics per forecast
    """
    n_forecasts = len(forecast_samples_list)
    assert n_forecasts == len(actual_outcomes), "Mismatch in number of forecasts and actuals"

    if forecast_names is None:
        forecast_names = [f"Forecast_{i+1}" for i in range(n_forecasts)]

    metrics_calc = CalibrationMetrics()
    results = []

    for i, (forecast_samples, actual, name) in enumerate(zip(forecast_samples_list, actual_outcomes, forecast_names)):
        # Calculate metrics
        crps = metrics_calc.crps(forecast_samples, actual)
        pit_val = metrics_calc.pit(forecast_samples, actual)

        forecast_mean = np.mean(forecast_samples)
        forecast_std = np.std(forecast_samples)
        forecast_median = np.median(forecast_samples)

        # Error metrics
        mean_error = forecast_mean - actual
        abs_error = abs(mean_error)

        # Percentiles
        p05 = np.percentile(forecast_samples, 5)
        p95 = np.percentile(forecast_samples, 95)
        interval_width = p95 - p05
        in_90_interval = p05 <= actual <= p95

        results.append({
            'forecast_name': name,
            'actual': actual,
            'forecast_mean': forecast_mean,
            'forecast_median': forecast_median,
            'forecast_std': forecast_std,
            'CRPS': crps,
            'PIT': pit_val,
            'mean_error': mean_error,
            'abs_error': abs_error,
            'p05': p05,
            'p95': p95,
            'interval_width': interval_width,
            'in_90_interval': in_90_interval
        })

    return pd.DataFrame(results)


# Example usage and testing
if __name__ == '__main__':
    np.random.seed(42)
    metrics = CalibrationMetrics()

    print("=== Calibration Metrics Examples ===\n")

    # Example 1: CRPS for a forecast
    print("1. CRPS (Continuous Ranked Probability Score)")
    forecast_samples = np.random.normal(loc=45.0, scale=5.0, size=10000)
    actual_outcome = 48.5

    crps_empirical = metrics.crps(forecast_samples, actual_outcome)
    crps_normal = metrics.crps_normal(mean=45.0, std=5.0, actual_outcome=48.5)

    print(f"  Forecast: N(45, 5²), Actual: 48.5")
    print(f"  CRPS (empirical): {crps_empirical:.2f}")
    print(f"  CRPS (normal formula): {crps_normal:.2f}\n")

    # Example 2: PIT values and histogram
    print("2. PIT (Probability Integral Transform)")
    # Simulate 100 forecasts and actuals
    n_forecasts = 100
    forecast_means = np.random.uniform(20, 30, n_forecasts)
    forecast_std = 5.0

    # Generate actuals from same distributions (well-calibrated)
    actuals = np.random.normal(forecast_means, forecast_std)

    # Calculate PIT for each
    pit_values = []
    for mean, actual in zip(forecast_means, actuals):
        forecast_samps = np.random.normal(mean, forecast_std, size=1000)
        pit = metrics.pit(forecast_samps, actual)
        pit_values.append(pit)

    pit_values = np.array(pit_values)

    pit_hist = metrics.pit_histogram(pit_values, n_bins=10, plot=False)
    print(f"  Number of forecasts: {n_forecasts}")
    print(f"  Uniformity test p-value: {pit_hist['uniformity_pvalue']:.3f}")
    print(f"  Is uniform (well-calibrated): {pit_hist['is_uniform']}\n")

    # Example 3: Calibration curve for binary outcomes
    print("3. Calibration Curve (Binary Outcomes)")
    # Simulate Over/Under predictions
    n_bets = 200
    predicted_probs = np.random.uniform(0.3, 0.8, n_bets)

    # Generate actuals with some miscalibration (predictions too high)
    actual_probs = predicted_probs * 0.85  # Biased low
    actual_outcomes = np.random.binomial(1, actual_probs)

    calib_df = metrics.calibration_curve(
        predicted_probs=predicted_probs,
        actual_outcomes=actual_outcomes,
        n_bins=10,
        plot=False
    )

    print(calib_df[calib_df['count'] > 0][['bin_center', 'predicted_prob', 'observed_freq', 'count']].to_string(index=False))
    print()

    # Example 4: Brier score
    print("4. Brier Score")
    brier = metrics.brier_score(predicted_probs, actual_outcomes)
    print(f"  Brier score: {brier:.4f} (lower is better)\n")

    # Example 5: Coverage probability
    print("5. Coverage Probability (90% Intervals)")
    # Generate 90% prediction intervals
    intervals = []
    actuals_for_coverage = []

    for i in range(50):
        mean = np.random.uniform(40, 50)
        std = 5.0

        # 90% interval: [p05, p95]
        p05 = stats.norm.ppf(0.05, loc=mean, scale=std)
        p95 = stats.norm.ppf(0.95, loc=mean, scale=std)

        intervals.append((p05, p95))

        # Generate actual (from same distribution - well-calibrated)
        actual = np.random.normal(mean, std)
        actuals_for_coverage.append(actual)

    actuals_for_coverage = np.array(actuals_for_coverage)

    coverage_result = metrics.coverage_probability(
        forecast_intervals=intervals,
        actual_outcomes=actuals_for_coverage,
        confidence_level=0.90
    )

    print(f"  Expected coverage: {coverage_result['expected']:.1%}")
    print(f"  Observed coverage: {coverage_result['coverage']:.1%}")
    print(f"  Is calibrated: {coverage_result['is_calibrated']}")
    print(f"  Deviation: {coverage_result['deviation']:+.1%}\n")

    # Example 6: Comprehensive validation
    print("6. Comprehensive Validation Report")
    # Generate 5 forecasts
    forecast_samples_list = []
    actuals_comprehensive = []

    for i in range(5):
        mean = 45.0 + np.random.randn() * 3.0
        samples = np.random.normal(mean, 5.0, size=10000)
        forecast_samples_list.append(samples)

        actual = mean + np.random.randn() * 5.0
        actuals_comprehensive.append(actual)

    actuals_comprehensive = np.array(actuals_comprehensive)

    validation_df = validate_forecast_distributions(
        forecast_samples_list=forecast_samples_list,
        actual_outcomes=actuals_comprehensive,
        forecast_names=[f"Game_{i+1}" for i in range(5)]
    )

    print(validation_df[['forecast_name', 'actual', 'forecast_mean', 'CRPS', 'PIT', 'abs_error', 'in_90_interval']].to_string(index=False))
