"""
Advanced probability distributions for NFL QUANT V4.

This package provides distribution functions for the systematic
probabilistic framework:
- Negative Binomial: For count data (targets, carries) with
  overdispersion
- Lognormal: For strictly positive skewed data (yards/target,
  yards/carry)
- Gaussian Copula: For modeling correlation between usage and efficiency

Mathematical Foundation:
- Negative Binomial better models count data than Normal
  (allows variance > mean)
- Lognormal better models efficiency than Normal
  (always positive, right-skewed)
- Copula preserves marginal distributions while modeling dependency
"""

from .negative_binomial import (
    NegativeBinomialSampler,
    mean_var_to_negbin_params,
    fit_negbin_from_data,
    validate_negbin_fit,
    estimate_target_variance
)

from .lognormal import (
    LognormalSampler,
    mean_cv_to_lognormal_params,
    fit_lognormal_from_data,
    validate_lognormal_fit
)

from .copula import (
    GaussianCopula,
    generate_correlated_uniforms,
    estimate_correlation_from_data,
    validate_correlation,
    sample_correlated_targets_ypt,
    get_default_target_ypt_correlation
)

__all__ = [
    # Negative Binomial
    'NegativeBinomialSampler',
    'mean_var_to_negbin_params',
    'fit_negbin_from_data',
    'validate_negbin_fit',
    'estimate_target_variance',

    # Lognormal
    'LognormalSampler',
    'mean_cv_to_lognormal_params',
    'fit_lognormal_from_data',
    'validate_lognormal_fit',

    # Copula
    'GaussianCopula',
    'generate_correlated_uniforms',
    'estimate_correlation_from_data',
    'validate_correlation',
    'sample_correlated_targets_ypt',
    'get_default_target_ypt_correlation',
]
