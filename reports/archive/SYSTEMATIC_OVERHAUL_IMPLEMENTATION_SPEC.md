# NFL QUANT: Systematic Architecture Overhaul - Complete Implementation Specification

**Expert Role**: Senior Quantitative Sports Analytics Architect with expertise in probabilistic modeling, Monte Carlo simulation, and betting systems

**Date**: November 23, 2025
**Status**: 📋 **IMPLEMENTATION SPECIFICATION**
**Scope**: Complete systematic redesign of NFL QUANT prediction pipeline

---

## Executive Summary

This document provides a **complete, systematic implementation plan** to transform the NFL QUANT prediction system from a basic XGBoost + Normal distribution approach to a sophisticated probabilistic modeling framework using Negative Binomial distributions, Lognormal distributions, and Gaussian copula correlation modeling.

**Current State**: Bug-fixed V3 system using Normal distributions
**Target State**: Probabilistic V4 system with advanced distributions and correlation

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Target Architecture Specification](#target-architecture-specification)
3. [Gap Analysis & File Mapping](#gap-analysis--file-mapping)
4. [Implementation Phases](#implementation-phases)
5. [File-by-File Modification Plan](#file-by-file-modification-plan)
6. [Data Requirements](#data-requirements)
7. [Testing & Validation Strategy](#testing--validation-strategy)
8. [Deployment Plan](#deployment-plan)
9. [Rollback Strategy](#rollback-strategy)

---

## 1. Current Architecture Analysis

### 1.1 Current Data Flow

```
NFLverse Data → Trailing Stats (simple mean) → XGBoost Models → Normal Sampling → Mean ± Std Output
```

### 1.2 Current File Architecture

**Core Prediction Files** (DO NOT DELETE):
- `nfl_quant/models/usage_predictor.py` - XGBoost for targets/carries
- `nfl_quant/models/efficiency_predictor.py` - XGBoost for yards/target
- `nfl_quant/simulation/player_simulator_v3_correlated.py` - Monte Carlo with Normal distributions
- `scripts/predict/generate_model_predictions.py` - Main prediction pipeline
- `scripts/predict/generate_unified_recommendations_v3.py` - Betting recommendations

**Supporting Infrastructure** (KEEP & ENHANCE):
- `nfl_quant/features/trailing_stats.py` - Has EWMA but not used in production
- `nfl_quant/utils/epa_utils.py` - Bayesian EPA (good, keep)
- `nfl_quant/calibration/*` - Isotonic calibration (keep for now)

**Files to DELETE** (deprecated):
- `scripts/predict/live_predictions.py` - Deprecated per framework unification
- `scripts/predict/generate_unified_recommendations.py` (v1) - Old
- `scripts/predict/generate_unified_recommendations_v2.py` - Old
- `scripts/predict/generate_unified_recommendations_v3_FIXED_CALIBRATION.py` - Experimental
- `scripts/predict/generate_model_predictions_BACKUP.py` - Backup only
- All calibrator training scripts (12 total) in `scripts/train/retrain_calibrator_*.py`

### 1.3 Current Limitations

| Component | Current | Issue | Target |
|-----------|---------|-------|--------|
| **Data** | Simple 4-week mean | Equal weights | EWMA + routes metrics |
| **Usage Model** | XGBoost → mean | Normal dist | XGBoost → NegBin params |
| **Efficiency Model** | XGBoost → mean | Normal dist | XGBoost → Lognormal params |
| **Correlation** | None | Independent sampling | Gaussian copula |
| **Simulation** | Normal distributions | Can go negative | NegBin + Lognormal |
| **Outputs** | Mean, std only | No percentiles | Full distribution |
| **Betting** | Basic edge | No Kelly | Kelly + risk analysis |

---

## 2. Target Architecture Specification

### 2.1 Target Data Flow

```
NFLverse + Routes Data
  ↓ EWMA weighting
Trailing Stats (4-week exponentially weighted)
  ↓ Feature extraction
XGBoost Models (output distribution parameters)
  ├─ Usage: predict (μ_targets, dispersion) for Negative Binomial
  └─ Efficiency: predict (μ_log_yt, σ_log_yt) for Lognormal
  ↓ Correlation estimation
Gaussian Copula (ρ between targets and Y/T)
  ↓ Monte Carlo (10k trials)
Correlated Sampling
  ├─ Sample targets ~ NegBin(μ, dispersion)
  └─ Sample Y/T ~ Lognormal(μ_log, σ_log) | targets
  ↓ Calculate yards = targets × Y/T
Full Distribution Output
  ├─ Mean, median, mode
  ├─ Percentiles (5th, 25th, 75th, 95th)
  ├─ P(exceeds line)
  └─ Confidence score
  ↓ Betting intelligence
Kelly Fraction + Risk Analysis + Narrative
```

### 2.2 Mathematical Specifications

#### 2.2.1 Negative Binomial for Targets

**Parameterization**: NegBin(n, p) where:
- `n` = number of successes (dispersion parameter)
- `p` = probability of success
- `μ = n(1-p)/p` (mean)
- `σ² = μ + μ²/n` (variance, allows overdispersion)

**Conversion from model output**:
```python
def convert_mean_to_negbin(mean_targets, overdispersion_factor):
    """
    Convert mean prediction to NegBin parameters.

    Args:
        mean_targets: Model-predicted mean (e.g., 6.5 targets)
        overdispersion_factor: Variance/mean ratio (typically 1.2-2.0)

    Returns:
        n, p for scipy.stats.nbinom
    """
    variance = mean_targets * overdispersion_factor
    n = mean_targets ** 2 / (variance - mean_targets)
    p = mean_targets / variance
    return n, p
```

**Why NegBin instead of Normal**:
- Targets are count data (integers only)
- Overdispersion (variance > mean) common in targets
- Cannot go negative (Normal can)
- Better captures "hot hand" / "cold hand" games

#### 2.2.2 Lognormal for Yards/Target

**Parameterization**: Lognormal(μ_log, σ_log) where:
- `μ_log` = mean of log(Y/T)
- `σ_log` = std dev of log(Y/T)
- Back-transform: Y/T = exp(μ_log + σ_log²/2)

**Conversion from model output**:
```python
def convert_mean_to_lognormal(mean_ypt, cv=0.40):
    """
    Convert mean Y/T to lognormal parameters.

    Args:
        mean_ypt: Model-predicted mean Y/T (e.g., 9.5 yards/target)
        cv: Coefficient of variation (std/mean, typically 0.30-0.50)

    Returns:
        mu_log, sigma_log for scipy.stats.lognorm
    """
    variance = (mean_ypt * cv) ** 2
    mu_log = np.log(mean_ypt ** 2 / np.sqrt(variance + mean_ypt ** 2))
    sigma_log = np.sqrt(np.log(1 + variance / mean_ypt ** 2))
    return mu_log, sigma_log
```

**Why Lognormal instead of Normal**:
- Y/T strictly positive (Normal can go negative)
- Right-skewed (big plays create long tail)
- Multiplicative process (yards accumulate multiplicatively)

#### 2.2.3 Gaussian Copula for Correlation

**Purpose**: Model correlation between targets and Y/T

**Observation**: High-target games often have lower Y/T (shorter, safer throws)

**Implementation**:
```python
from scipy.stats import norm, spearmanr

def estimate_correlation(player_history):
    """
    Estimate Spearman correlation between targets and Y/T.

    Args:
        player_history: DataFrame with columns ['targets', 'yards_per_target']

    Returns:
        rho: Correlation coefficient (-1 to 1)
    """
    rho, _ = spearmanr(
        player_history['targets'],
        player_history['yards_per_target']
    )
    return rho

def generate_correlated_uniforms(rho, size=10000):
    """
    Generate correlated uniform samples using Gaussian copula.

    Args:
        rho: Correlation coefficient
        size: Number of samples

    Returns:
        u1, u2: Correlated uniform samples [0, 1]
    """
    # Bivariate normal with correlation rho
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    z = np.random.multivariate_normal(mean, cov, size=size)

    # Transform to uniforms via CDF
    u1 = norm.cdf(z[:, 0])
    u2 = norm.cdf(z[:, 1])

    return u1, u2

def sample_correlated_distributions(mean_targets, overdispersion,
                                   mean_ypt, cv_ypt, rho, size=10000):
    """
    Sample from correlated NegBin and Lognormal.

    Returns:
        targets, yards_per_target: Correlated samples
    """
    # Generate correlated uniforms
    u1, u2 = generate_correlated_uniforms(rho, size)

    # Convert to NegBin parameters
    n, p = convert_mean_to_negbin(mean_targets, overdispersion)

    # Convert to Lognormal parameters
    mu_log, sigma_log = convert_mean_to_lognormal(mean_ypt, cv_ypt)

    # Transform uniforms to marginal distributions
    from scipy.stats import nbinom, lognorm
    targets = nbinom.ppf(u1, n, p)
    yards_per_target = lognorm.ppf(u2, s=sigma_log, scale=np.exp(mu_log))

    return targets, yards_per_target
```

---

## 3. Gap Analysis & File Mapping

### 3.1 New Files to CREATE

| File Path | Purpose | Priority |
|-----------|---------|----------|
| `nfl_quant/distributions/__init__.py` | Distribution utilities package | 🔥 HIGH |
| `nfl_quant/distributions/negative_binomial.py` | NegBin utilities | 🔥 HIGH |
| `nfl_quant/distributions/lognormal.py` | Lognormal utilities | 🔥 HIGH |
| `nfl_quant/distributions/copula.py` | Gaussian copula correlation | 🔥 HIGH |
| `nfl_quant/features/route_metrics.py` | Routes run, TPRR, Y/RR | 🔥 HIGH |
| `nfl_quant/simulation/player_simulator_v4.py` | New simulator with advanced distributions | 🔥 HIGH |
| `nfl_quant/betting/kelly_criterion_advanced.py` | Kelly sizing, risk analysis | ⭐ MEDIUM |
| `nfl_quant/betting/narrative_generator.py` | Upside/downside narratives | ⭐ MEDIUM |
| `nfl_quant/schemas/v4_output.py` | New output schema with percentiles | 🔥 HIGH |
| `scripts/train/train_v4_models.py` | Train models to output distribution params | 🔥 HIGH |
| `scripts/validate/validate_v4_distributions.py` | Validate distribution fits | ⭐ MEDIUM |
| `scripts/debug/compare_v3_v4.py` | Side-by-side comparison | 🚀 LOW |

### 3.2 Existing Files to MODIFY

| File Path | Current Behavior | Required Changes | Priority |
|-----------|-----------------|------------------|----------|
| `nfl_quant/features/trailing_stats.py` | Has EWMA but not used | Make EWMA default in production | 🔥 HIGH |
| `nfl_quant/models/usage_predictor.py` | Outputs mean only | Add variance/dispersion output | 🔥 HIGH |
| `nfl_quant/models/efficiency_predictor.py` | Outputs mean only | Add log-space parameters | 🔥 HIGH |
| `nfl_quant/schemas.py` | Simple mean/std schema | Add percentile fields | 🔥 HIGH |
| `scripts/predict/generate_model_predictions.py` | Uses V3 simulator | Add V4 simulator option | 🔥 HIGH |
| `scripts/predict/generate_unified_recommendations_v3.py` | Basic edge calc | Add Kelly + narratives | ⭐ MEDIUM |
| `scripts/train/train_usage_predictor_v4_with_defense.py` | Trains for mean | Add dispersion target | 🔥 HIGH |
| `scripts/train/train_efficiency_predictor_v2_with_defense.py` | Trains for mean | Add log-space targets | 🔥 HIGH |

### 3.3 Files to DELETE

| File Path | Reason | Safe to Delete? |
|-----------|--------|-----------------|
| `scripts/predict/live_predictions.py` | Deprecated per framework docs | ✅ YES |
| `scripts/predict/generate_unified_recommendations.py` | V1 - superseded by V3 | ✅ YES |
| `scripts/predict/generate_unified_recommendations_v2.py` | V2 - superseded by V3 | ✅ YES |
| `scripts/predict/generate_unified_recommendations_v3_FIXED_CALIBRATION.py` | Experimental - V3 already has shrinkage | ✅ YES |
| `scripts/train/retrain_calibrator_*.py` (12 files) | Deprecated - using 30% shrinkage instead | ⚠️ ARCHIVE FIRST |
| `scripts/predict/generate_model_predictions_BACKUP.py` | Backup only | ✅ YES |

---

## 4. Implementation Phases

### Phase 0: Preparation & Data Enhancement (Week 1)

**Goal**: Source routes run data, set up V4 development environment

**Tasks**:
1. ✅ Create development branch: `git checkout -b feature/v4-probabilistic-distributions`
2. ❌ Source routes run data (choose option):
   - **Option A** (RECOMMENDED): Estimate from `snaps × (team_pass_attempts / team_plays)`
   - **Option B**: Purchase PFF subscription ($200/month)
   - **Option C**: Scrape Pro Football Reference
3. ❌ Create `nfl_quant/features/route_metrics.py`:
   ```python
   def estimate_routes_run(snap_count, team_pass_attempts, team_total_plays):
       """Estimate routes run from snap participation."""
       pass_play_rate = team_pass_attempts / team_total_plays
       routes_run = snap_count * pass_play_rate
       return routes_run

   def calculate_route_participation(routes_run, team_pass_attempts):
       """RP = routes_run / team_pass_attempts"""
       return routes_run / team_pass_attempts if team_pass_attempts > 0 else 0.0

   def calculate_tprr(targets, routes_run):
       """TPRR = targets / routes_run"""
       return targets / routes_run if routes_run > 0 else 0.0

   def calculate_yrr(receiving_yards, routes_run):
       """Y/RR = receiving_yards / routes_run"""
       return receiving_yards / routes_run if routes_run > 0 else 0.0
   ```

4. ❌ Modify `nfl_quant/features/trailing_stats.py`:
   - Change default from `use_ewma=False` to `use_ewma=True` (line ~120)
   - Add routes metrics to trailing stats calculation
   - Validate EWMA weights (40%, 27%, 18%, 12% for weeks N-1 to N-4)

5. ❌ Create validation dataset:
   - Extract Weeks 5-11 actuals from 2025 season
   - Save to `data/validation/weeks_5_11_actuals.parquet`
   - Will use for V4 vs V3 comparison

**Deliverables**:
- `nfl_quant/features/route_metrics.py` (new)
- `nfl_quant/features/trailing_stats.py` (modified: EWMA default)
- `data/validation/weeks_5_11_actuals.parquet` (new)
- Development branch created

**Time Estimate**: 3-5 days
**Blocker**: Routes run data source decision

---

### Phase 1: Distribution Infrastructure (Week 2)

**Goal**: Implement Negative Binomial, Lognormal, and Gaussian copula utilities

**Tasks**:
1. ❌ Create `nfl_quant/distributions/__init__.py`:
   ```python
   from .negative_binomial import NegativeBinomialSampler, mean_var_to_negbin
   from .lognormal import LognormalSampler, mean_cv_to_lognormal
   from .copula import GaussianCopula, generate_correlated_samples

   __all__ = [
       'NegativeBinomialSampler',
       'LognormalSampler',
       'GaussianCopula',
       'mean_var_to_negbin',
       'mean_cv_to_lognormal',
       'generate_correlated_samples'
   ]
   ```

2. ❌ Create `nfl_quant/distributions/negative_binomial.py`:
   ```python
   import numpy as np
   from scipy.stats import nbinom
   from typing import Tuple

   def mean_var_to_negbin(mean: float, variance: float) -> Tuple[float, float]:
       """
       Convert mean and variance to NegBin(n, p) parameters.

       Formula:
           n = mean² / (variance - mean)
           p = mean / variance

       Args:
           mean: Mean of distribution (e.g., 6.5 targets)
           variance: Variance (must be > mean for overdispersion)

       Returns:
           n, p: Parameters for scipy.stats.nbinom

       Raises:
           ValueError: If variance <= mean (underdispersed)
       """
       if variance <= mean:
           raise ValueError(f"Variance ({variance}) must be > mean ({mean}) for NegBin")

       n = mean ** 2 / (variance - mean)
       p = mean / variance
       return n, p

   def fit_negbin_from_data(data: np.ndarray) -> Tuple[float, float]:
       """Fit NegBin parameters from empirical data."""
       mean = np.mean(data)
       variance = np.var(data)
       return mean_var_to_negbin(mean, variance)

   class NegativeBinomialSampler:
       """Monte Carlo sampler for Negative Binomial distribution."""

       def __init__(self, mean: float, overdispersion: float = 1.5):
           """
           Args:
               mean: Mean of distribution
               overdispersion: Variance/mean ratio (typically 1.2-2.0)
           """
           variance = mean * overdispersion
           self.n, self.p = mean_var_to_negbin(mean, variance)
           self.mean = mean
           self.overdispersion = overdispersion

       def sample(self, size: int = 10000, seed: int = None) -> np.ndarray:
           """Generate samples."""
           if seed is not None:
               np.random.seed(seed)
           return nbinom.rvs(self.n, self.p, size=size)

       def percentiles(self, percentiles: list = [5, 25, 50, 75, 95]) -> dict:
           """Calculate percentiles from distribution."""
           return {
               f'p{p}': nbinom.ppf(p/100, self.n, self.p)
               for p in percentiles
           }
   ```

3. ❌ Create `nfl_quant/distributions/lognormal.py`:
   ```python
   import numpy as np
   from scipy.stats import lognorm
   from typing import Tuple

   def mean_cv_to_lognormal(mean: float, cv: float = 0.40) -> Tuple[float, float]:
       """
       Convert mean and coefficient of variation to lognormal parameters.

       Args:
           mean: Mean of distribution (e.g., 9.5 yards/target)
           cv: Coefficient of variation (std/mean, typically 0.30-0.50)

       Returns:
           mu_log, sigma_log: Parameters for lognormal
       """
       variance = (mean * cv) ** 2
       mu_log = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
       sigma_log = np.sqrt(np.log(1 + variance / mean ** 2))
       return mu_log, sigma_log

   class LognormalSampler:
       """Monte Carlo sampler for Lognormal distribution."""

       def __init__(self, mean: float, cv: float = 0.40):
           self.mean = mean
           self.cv = cv
           self.mu_log, self.sigma_log = mean_cv_to_lognormal(mean, cv)

       def sample(self, size: int = 10000, seed: int = None) -> np.ndarray:
           """Generate samples."""
           if seed is not None:
               np.random.seed(seed)
           return lognorm.rvs(
               s=self.sigma_log,
               scale=np.exp(self.mu_log),
               size=size
           )
   ```

4. ❌ Create `nfl_quant/distributions/copula.py`:
   ```python
   import numpy as np
   from scipy.stats import norm, spearmanr

   class GaussianCopula:
       """
       Gaussian copula for modeling correlation between variables.

       Used to generate correlated samples from different marginal distributions.
       """

       def __init__(self, rho: float):
           """
           Args:
               rho: Correlation coefficient (-1 to 1)
           """
           self.rho = rho

       def generate_correlated_uniforms(self, size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
           """
           Generate correlated uniform [0,1] samples.

           Returns:
               u1, u2: Correlated uniform samples
           """
           # Bivariate normal with correlation rho
           mean = [0, 0]
           cov = [[1, self.rho], [self.rho, 1]]
           z = np.random.multivariate_normal(mean, cov, size=size)

           # Transform to uniforms via normal CDF
           u1 = norm.cdf(z[:, 0])
           u2 = norm.cdf(z[:, 1])

           return u1, u2

   def estimate_correlation_from_history(targets_history: np.ndarray,
                                        ypt_history: np.ndarray) -> float:
       """
       Estimate Spearman correlation between targets and Y/T from player history.

       Args:
           targets_history: Historical targets (e.g., last 10 games)
           ypt_history: Historical yards/target

       Returns:
           rho: Spearman correlation coefficient
       """
       rho, _ = spearmanr(targets_history, ypt_history)
       return rho if not np.isnan(rho) else 0.0  # Default to 0 if insufficient data
   ```

**Deliverables**:
- `nfl_quant/distributions/` package (4 files)
- Unit tests for all distribution functions
- Documentation with mathematical formulas

**Time Estimate**: 5-7 days
**Dependencies**: None

---

### Phase 2: Model Retraining for Distribution Parameters (Week 3)

**Goal**: Retrain XGBoost models to output distribution parameters instead of just means

**Current Training Targets**:
```python
# Usage model
y_targets = data['targets']  # Single value

# Efficiency model
y_ypt = data['yards'] / data['targets']  # Single value
```

**New Training Targets**:
```python
# Usage model - output TWO targets
y_mean_targets = data['targets']
y_dispersion = calculate_dispersion(data['targets'])  # Variance/mean ratio

# Efficiency model - output TWO targets
y_log_ypt = np.log(data['yards'] / data['targets'])
y_log_std = calculate_log_std(data['yards'], data['targets'])
```

**Tasks**:
1. ❌ Modify `nfl_quant/models/usage_predictor.py`:
   - Add `predict_distribution_params()` method
   - Returns `(mean_targets, dispersion)` instead of just `mean_targets`
   - Keep existing `predict()` for backward compatibility

2. ❌ Modify `nfl_quant/models/efficiency_predictor.py`:
   - Add `predict_lognormal_params()` method
   - Returns `(mu_log, sigma_log)` instead of just `mean_ypt`

3. ❌ Create `scripts/train/train_v4_models.py`:
   ```python
   # Pseudocode
   def train_usage_v4():
       # Load data
       data = load_training_data()

       # Calculate targets
       targets = data['targets']

       # Calculate empirical dispersion per player
       dispersion = data.groupby('player_id')['targets'].apply(
           lambda x: x.var() / x.mean() if x.mean() > 0 else 1.5
       )

       # Train two models
       model_mean = XGBRegressor().fit(X, targets)
       model_disp = XGBRegressor().fit(X, dispersion)

       # Save both
       joblib.dump((model_mean, model_disp), 'usage_v4_dist_params.joblib')

   def train_efficiency_v4():
       # Load data
       data = load_training_data()

       # Calculate log-space targets
       ypt = data['yards'] / data['targets']
       log_ypt = np.log(ypt + 1e-6)  # Add small constant to avoid log(0)

       # Calculate log-space std per player
       log_std = data.groupby('player_id').apply(
           lambda x: np.std(np.log(x['yards'] / x['targets'] + 1e-6))
       )

       # Train two models
       model_log_mean = XGBRegressor().fit(X, log_ypt)
       model_log_std = XGBRegressor().fit(X, log_std)

       # Save both
       joblib.dump((model_log_mean, model_log_std), 'efficiency_v4_lognormal.joblib')
   ```

4. ❌ Run retraining:
   ```bash
   python scripts/train/train_v4_models.py --all --validate
   ```

5. ❌ Validate new models on Weeks 5-11:
   - Check if distribution parameters make sense
   - Compare V4 distribution predictions to V3 mean predictions
   - Ensure overdispersion reasonable (1.2-2.0 for targets)

**Deliverables**:
- `data/models/usage_v4_dist_params.joblib` (new)
- `data/models/efficiency_v4_lognormal.joblib` (new)
- Validation report comparing V3 vs V4 parameter outputs

**Time Estimate**: 5-7 days
**Dependencies**: Phase 1 complete

---

### Phase 3: Correlation Estimation (Week 4)

**Goal**: Estimate player-specific correlation between targets and Y/T

**Tasks**:
1. ❌ Create `scripts/analysis/estimate_target_ypt_correlation.py`:
   ```python
   def estimate_all_player_correlations():
       """
       Calculate Spearman correlation for each player with 4+ games.

       Output: CSV with columns [player_id, player_name, rho, n_games]
       """
       pbp_data = load_pbp_data()

       correlations = []
       for player_id in pbp_data['player_id'].unique():
           player_games = pbp_data[pbp_data['player_id'] == player_id]

           if len(player_games) < 4:
               continue  # Need minimum 4 games for correlation

           targets = player_games['targets']
           ypt = player_games['receiving_yards'] / player_games['targets']

           rho = spearmanr(targets, ypt).correlation

           correlations.append({
               'player_id': player_id,
               'player_name': player_games.iloc[0]['player_name'],
               'rho': rho,
               'n_games': len(player_games)
           })

       # Save
       pd.DataFrame(correlations).to_csv('data/correlations/target_ypt_correlations.csv')
   ```

2. ❌ Run correlation estimation:
   ```bash
   python scripts/analysis/estimate_target_ypt_correlation.py
   ```

3. ❌ Analyze correlation distributions:
   - Plot histogram of `rho` values
   - Identify typical ranges (expect -0.3 to -0.5 for most players)
   - Identify outliers (players with unusual correlation patterns)

4. ❌ Create fallback strategy for players with insufficient history:
   ```python
   def get_correlation_with_fallback(player_id, position):
       """
       Get player correlation, fall back to position average if needed.
       """
       correlations = pd.read_csv('data/correlations/target_ypt_correlations.csv')

       player_corr = correlations[correlations['player_id'] == player_id]

       if len(player_corr) > 0 and player_corr.iloc[0]['n_games'] >= 4:
           return player_corr.iloc[0]['rho']
       else:
           # Fall back to position average
           position_avg = {
               'WR': -0.35,  # High-target WR games have lower Y/T
               'TE': -0.30,  # Similar pattern
               'RB': -0.25   # Less pronounced for RBs
           }
           return position_avg.get(position, -0.30)
   ```

**Deliverables**:
- `data/correlations/target_ypt_correlations.csv` (new)
- Correlation analysis report with histograms
- Fallback strategy implemented

**Time Estimate**: 3-5 days
**Dependencies**: Historical data available

---

### Phase 4: PlayerSimulatorV4 Implementation (Week 5-6)

**Goal**: Create new simulator using NegBin, Lognormal, and correlation

**Tasks**:
1. ❌ Create `nfl_quant/simulation/player_simulator_v4.py`:
   ```python
   from nfl_quant.distributions import (
       NegativeBinomialSampler,
       LognormalSampler,
       GaussianCopula,
       estimate_correlation_from_history
   )
   from scipy.stats import nbinom, lognorm
   import numpy as np

   class PlayerSimulatorV4:
       """
       Advanced Monte Carlo simulator using:
       - Negative Binomial for targets (count data)
       - Lognormal for yards/target (positive skewed)
       - Gaussian copula for correlation
       """

       def __init__(self, usage_model, efficiency_model, trials=10000, seed=None):
           self.usage_model = usage_model
           self.efficiency_model = efficiency_model
           self.trials = trials
           self.seed = seed

       def simulate_player(self, player_input):
           """
           Run Monte Carlo simulation for a player.

           Args:
               player_input: PlayerPropInput with features

           Returns:
               dict with full distribution outputs
           """
           # Extract features
           features = self._extract_features(player_input)

           # Predict distribution parameters from models
           mean_targets, dispersion = self.usage_model.predict_distribution_params(features)
           mu_log_ypt, sigma_log_ypt = self.efficiency_model.predict_lognormal_params(features)

           # Estimate correlation
           rho = self._get_correlation(player_input)

           # Generate correlated samples
           copula = GaussianCopula(rho)
           u1, u2 = copula.generate_correlated_uniforms(size=self.trials)

           # Transform to marginal distributions
           n, p = mean_var_to_negbin(mean_targets, mean_targets * dispersion)
           targets_samples = nbinom.ppf(u1, n, p)

           ypt_samples = lognorm.ppf(
               u2,
               s=sigma_log_ypt,
               scale=np.exp(mu_log_ypt)
           )

           # Calculate receiving yards
           receiving_yards_samples = targets_samples * ypt_samples

           # Calculate receptions (using position-specific catch rate)
           catch_rate = self._get_catch_rate(player_input.position)
           receptions_samples = np.random.binomial(
               targets_samples.astype(int),
               catch_rate
           )

           # Calculate percentiles
           return {
               'receiving_yards_mean': np.mean(receiving_yards_samples),
               'receiving_yards_median': np.median(receiving_yards_samples),
               'receiving_yards_p5': np.percentile(receiving_yards_samples, 5),
               'receiving_yards_p25': np.percentile(receiving_yards_samples, 25),
               'receiving_yards_p75': np.percentile(receiving_yards_samples, 75),
               'receiving_yards_p95': np.percentile(receiving_yards_samples, 95),
               'receiving_yards_std': np.std(receiving_yards_samples),

               'targets_mean': np.mean(targets_samples),
               'targets_median': np.median(targets_samples),
               'targets_p25': np.percentile(targets_samples, 25),
               'targets_p75': np.percentile(targets_samples, 75),

               'receptions_mean': np.mean(receptions_samples),
               'receptions_median': np.median(receptions_samples),

               # For betting
               'samples': receiving_yards_samples  # Keep for prob calculations
           }

       def _get_correlation(self, player_input):
           """Get target-Y/T correlation with fallback."""
           # Implementation from Phase 3
           pass

       def _get_catch_rate(self, position):
           """Position-specific catch rates from NFLverse data."""
           catch_rates = {
               'WR': 0.627,  # From 2025 season data
               'TE': 0.724,
               'RB': 0.779
           }
           return catch_rates.get(position, 0.65)
   ```

2. ❌ Modify `scripts/predict/generate_model_predictions.py`:
   - Add `--simulator` argument: `v3` or `v4`
   - Default to `v3` for backward compatibility
   - Load V4 models and simulator when `--simulator v4`

3. ❌ Create comparison script `scripts/debug/compare_v3_v4.py`:
   ```python
   def compare_simulators(player_name, week):
       """Run same player through V3 and V4, show differences."""

       # V3 output
       v3_result = run_v3_simulation(player_name, week)

       # V4 output
       v4_result = run_v4_simulation(player_name, week)

       print(f"\n{player_name} Week {week} Comparison:")
       print(f"{'Metric':<30} {'V3':<15} {'V4':<15} {'Diff':<10}")
       print("=" * 70)
       print(f"{'Receiving Yards (mean)':<30} {v3_result['mean']:<15.1f} {v4_result['mean']:<15.1f} {v4_result['mean']-v3_result['mean']:<10.1f}")
       print(f"{'Receiving Yards (median)':<30} {'-':<15} {v4_result['median']:<15.1f}")
       print(f"{'Receiving Yards (25th-75th)':<30} {'-':<15} {v4_result['p25']:<7.1f}-{v4_result['p75']:<7.1f}")
       print(f"{'Std Dev':<30} {v3_result['std']:<15.1f} {v4_result['std']:<15.1f} {v4_result['std']-v3_result['std']:<10.1f}")
   ```

**Deliverables**:
- `nfl_quant/simulation/player_simulator_v4.py` (new, ~500 lines)
- `scripts/predict/generate_model_predictions.py` (modified: V4 option)
- `scripts/debug/compare_v3_v4.py` (new)
- V3 vs V4 comparison report for 10 test players

**Time Estimate**: 7-10 days
**Dependencies**: Phases 1-3 complete

---

### Phase 5: Output Schema & Percentiles (Week 7)

**Goal**: Update all output schemas to include percentiles and confidence metrics

**Tasks**:
1. ❌ Modify `nfl_quant/schemas.py`:
   ```python
   @dataclass
   class V4SimulationOutput:
       """Output schema for PlayerSimulatorV4."""

       # Point estimates
       mean: float
       median: float
       mode: Optional[float] = None

       # Percentiles
       p5: float
       p25: float
       p75: float
       p95: float

       # Spread
       std: float
       iqr: float  # p75 - p25
       cv: float   # Coefficient of variation (std/mean)

       # For betting
       prob_over_line: Optional[float] = None
       confidence_score: Optional[float] = None
       expected_value: Optional[float] = None

       # Distribution samples (for further analysis)
       samples: Optional[np.ndarray] = None
   ```

2. ❌ Update CSV output in `generate_model_predictions.py`:
   - Add columns: `p5`, `p25`, `p75`, `p95`, `median`, `iqr`
   - Keep existing `mean`, `std` for backward compatibility
   - Total columns: ~65 (was 57)

3. ❌ Update dashboard to show percentile ranges:
   - Display confidence intervals (25th-75th)
   - Show "likely range" visually
   - Add tooltips explaining percentiles

**Deliverables**:
- `nfl_quant/schemas.py` (modified: V4SimulationOutput)
- Updated CSV schema documentation
- Dashboard mockup with percentile visualization

**Time Estimate**: 3-4 days
**Dependencies**: Phase 4 complete

---

### Phase 6: Betting Intelligence Layer (Week 8)

**Goal**: Add Kelly criterion, upside/downside analysis, and narrative generation

**Tasks**:
1. ❌ Create `nfl_quant/betting/kelly_criterion_advanced.py`:
   ```python
   def calculate_kelly_fraction(edge: float, win_prob: float,
                                bankroll: float = 1000) -> dict:
       """
       Calculate Kelly criterion bet size.

       Formula:
           f = (p * (b + 1) - 1) / b
       where:
           p = win probability
           b = decimal odds - 1

       Args:
           edge: Model edge (e.g., 0.15 for 15%)
           win_prob: Calibrated win probability
           bankroll: Total bankroll ($)

       Returns:
           dict with kelly_fraction, recommended_stake, risk_category
       """
       # Assume American odds of -110 (1.909 decimal)
       decimal_odds = 1.909
       b = decimal_odds - 1

       # Kelly formula
       kelly_full = (win_prob * (b + 1) - 1) / b

       # Quarter Kelly for safety
       kelly_quarter = kelly_full / 4

       # Cap at 5% of bankroll (conservative)
       kelly_capped = min(kelly_quarter, 0.05)

       recommended_stake = bankroll * kelly_capped

       # Risk categorization
       if kelly_capped < 0:
           risk_category = "NO BET"
       elif kelly_capped < 0.01:
           risk_category = "VERY LOW"
       elif kelly_capped < 0.02:
           risk_category = "LOW"
       elif kelly_capped < 0.03:
           risk_category = "MEDIUM"
       else:
           risk_category = "HIGH"

       return {
           'kelly_fraction': kelly_capped,
           'recommended_stake': recommended_stake,
           'risk_category': risk_category,
           'full_kelly': kelly_full  # For reference
       }
   ```

2. ❌ Create `nfl_quant/betting/narrative_generator.py`:
   ```python
   def generate_upside_downside_narrative(player_name, position, projection,
                                         matchup, line):
       """
       Generate human-readable narrative explaining bet rationale.

       Returns:
           dict with keys: summary, upside_pathways, downside_risks, recommendation
       """
       # Calculate key metrics
       p_over = calculate_prob_over(projection['samples'], line)
       volatility = projection['cv']

       # Generate summary
       if p_over > 0.6:
           lean = "OVER"
           confidence = "HIGH" if p_over > 0.70 else "MEDIUM"
       elif p_over < 0.4:
           lean = "UNDER"
           confidence = "HIGH" if p_over < 0.30 else "MEDIUM"
       else:
           lean = "PASS"
           confidence = "LOW"

       # Upside pathways
       upside = []
       if matchup['opp_def_epa'] > 0.05:
           upside.append(f"Weak defense (+{matchup['opp_def_epa']:.3f} EPA)")
       if projection['p75'] > line * 1.3:
           upside.append(f"75th percentile scenario: {projection['p75']:.1f} {position.lower()} yards")
       if player_name in ["George Kittle", "Travis Kelce"]:  # Example
           upside.append("High-target game possible if game stays competitive")

       # Downside risks
       downside = []
       if matchup['projected_game_script'] < -3:
           downside.append("Blowout risk (team likely trailing)")
       if volatility > 1.0:
           downside.append(f"High variance (CV={volatility:.2f})")
       if projection['p25'] < line * 0.6:
           downside.append(f"25th percentile scenario: only {projection['p25']:.1f} yards")

       return {
           'summary': f"{lean} {line} ({confidence} confidence, {p_over:.0%} probability)",
           'upside_pathways': upside,
           'downside_risks': downside,
           'recommendation': _generate_bet_recommendation(lean, confidence, p_over, volatility)
       }
   ```

3. ❌ Integrate into `generate_unified_recommendations_v3.py`:
   - Add Kelly sizing to each recommendation
   - Add upside/downside narratives
   - Add volatility warnings

**Deliverables**:
- `nfl_quant/betting/kelly_criterion_advanced.py` (new)
- `nfl_quant/betting/narrative_generator.py` (new)
- Updated recommendations CSV with Kelly fractions
- Enhanced dashboard with narratives

**Time Estimate**: 4-6 days
**Dependencies**: Phase 5 complete

---

### Phase 7: Integration Testing & Validation (Week 9)

**Goal**: Comprehensive testing of V4 vs V3, backtest on Weeks 5-11

**Tasks**:
1. ❌ Run V4 on historical Weeks 5-11:
   ```bash
   for week in {5..11}; do
       python scripts/predict/generate_model_predictions.py $week --simulator v4
   done
   ```

2. ❌ Compare V4 vs V3 accuracy:
   - MAE (Mean Absolute Error)
   - Calibration (predicted probabilities vs actual outcomes)
   - Brier score
   - ROI on bets

3. ❌ Validate distribution assumptions:
   - QQ plots for NegBin (targets should follow NegBin)
   - QQ plots for Lognormal (Y/T should follow Lognormal)
   - Correlation estimates vs empirical correlation

4. ❌ Create validation report:
   ```
   V4 Validation Report (Weeks 5-11)

   Accuracy:
   - MAE: V3=12.4, V4=10.1 (-18% improvement ✅)
   - Calibration: V3=15.2%, V4=8.7% (-43% improvement ✅)
   - Brier Score: V3=0.245, V4=0.201 (-18% improvement ✅)

   Betting Performance:
   - ROI: V3=-3.5%, V4=+2.1% (🎯 positive ROI!)
   - Win Rate: V3=51.5%, V4=54.2%
   - Average Edge: V3=8.8%, V4=11.2%

   Distribution Validation:
   - NegBin fit: 92% of players within expected range ✅
   - Lognormal fit: 88% of players within expected range ✅
   - Correlation: empirical ρ=-0.32, predicted ρ=-0.30 ✅
   ```

**Deliverables**:
- `reports/V4_VALIDATION_REPORT.md` (comprehensive)
- Backtest results CSV
- Decision memo: GO/NO-GO for V4 production

**Time Estimate**: 5-7 days
**Dependencies**: All phases complete

---

### Phase 8: Production Deployment (Week 10)

**Goal**: Deploy V4 to production for Week 12+

**Tasks**:
1. ❌ Update default simulator in `generate_model_predictions.py`:
   ```python
   # Change from
   parser.add_argument('--simulator', default='v3', choices=['v3', 'v4'])
   # To
   parser.add_argument('--simulator', default='v4', choices=['v3', 'v4'])
   ```

2. ❌ Update CLAUDE.md documentation:
   - Document V4 architecture
   - Update workflow commands
   - Add troubleshooting section

3. ❌ Archive deprecated files:
   ```bash
   mkdir -p archive/v3_system
   mv scripts/predict/live_predictions.py archive/v3_system/
   mv scripts/predict/generate_unified_recommendations.py archive/v3_system/
   # ... (all deprecated files from Section 3.3)
   ```

4. ❌ Create rollback script:
   ```bash
   # scripts/utils/rollback_to_v3.sh
   #!/bin/bash
   echo "Rolling back to V3 simulator..."
   git checkout feature/v4-probabilistic-distributions~1 -- \
       nfl_quant/simulation/player_simulator.py \
       scripts/predict/generate_model_predictions.py
   echo "✅ Rolled back to V3"
   ```

5. ❌ Monitor Week 12 production run:
   - Watch for errors
   - Compare Week 12 V4 predictions to V3 (generate both)
   - Verify output files have all expected columns

**Deliverables**:
- V4 in production
- Updated CLAUDE.md
- Archived deprecated files
- Rollback script tested

**Time Estimate**: 2-3 days
**Dependencies**: Phase 7 GO decision

---

## 5. Data Requirements

### 5.1 Routes Run Data

**Problem**: NFLverse doesn't provide `routes_run` by default

**Solutions** (in priority order):

#### Option A: Estimate from Snap Counts ✅ RECOMMENDED
```python
def estimate_routes_run(snap_count, team_pass_attempts, team_total_plays):
    """
    Estimate routes run from snap participation.

    Assumption: Player runs a route on every pass play they're on field for.

    Accuracy: ~80-85% vs PFF data
    """
    pass_play_rate = team_pass_attempts / team_total_plays
    estimated_routes = snap_count * pass_play_rate
    return estimated_routes
```

**Pros**: Free, immediate, reasonably accurate
**Cons**: Noisy for blocking TEs, RBs who don't run routes every snap

**Validation**: Spot-check against PFF free samples

#### Option B: Purchase PFF Subscription
**Cost**: ~$200/month
**Accuracy**: 99%+ (gold standard)
**Pros**: Best data quality
**Cons**: Expensive, integration effort

#### Option C: Scrape Pro Football Reference
**Cost**: Free
**Accuracy**: ~95%
**Pros**: No cost
**Cons**: Legal gray area, brittle scraping

**Decision**: Start with Option A, upgrade to Option B if budget allows

### 5.2 Historical Data for Training

**Required**:
- 2022-2024 seasons (3 years)
- Weekly player stats
- Snap counts
- Team-level pass attempts, total plays

**Current Status**: Only have 2025 season (10 weeks)

**Action Required**:
```bash
Rscript scripts/fetch/fetch_nflverse_data.R --seasons 2022,2023,2024
```

---

## 6. Testing & Validation Strategy

### 6.1 Unit Tests

**Create** `tests/test_distributions.py`:
```python
def test_negative_binomial_conversion():
    """Test mean/variance → NegBin(n,p) conversion."""
    mean = 6.5
    variance = 10.0  # Overdispersion = 1.54

    n, p = mean_var_to_negbin(mean, variance)

    # Verify mean
    assert abs(n * (1-p) / p - mean) < 0.01

    # Verify variance
    assert abs(n * (1-p) / p**2 - variance) < 0.01

def test_lognormal_conversion():
    """Test mean/CV → Lognormal params conversion."""
    mean = 9.5
    cv = 0.40

    mu_log, sigma_log = mean_cv_to_lognormal(mean, cv)

    # Verify mean (lognormal mean = exp(μ + σ²/2))
    lognormal_mean = np.exp(mu_log + sigma_log**2 / 2)
    assert abs(lognormal_mean - mean) < 0.01

def test_copula_correlation():
    """Test Gaussian copula generates correct correlation."""
    rho = -0.30
    copula = GaussianCopula(rho)

    u1, u2 = copula.generate_correlated_uniforms(size=100000)

    empirical_rho = spearmanr(u1, u2).correlation
    assert abs(empirical_rho - rho) < 0.01  # Within 1%
```

### 6.2 Integration Tests

**Create** `tests/test_v4_simulator.py`:
```python
def test_v4_simulator_output_schema():
    """Ensure V4 simulator returns all required fields."""
    simulator = PlayerSimulatorV4(usage_model, efficiency_model)

    result = simulator.simulate_player(test_player_input)

    required_fields = [
        'receiving_yards_mean', 'receiving_yards_median',
        'receiving_yards_p5', 'receiving_yards_p25', 'receiving_yards_p75', 'receiving_yards_p95',
        'targets_mean', 'receptions_mean'
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

def test_v4_vs_v3_consistency():
    """V4 mean should be similar to V3 mean (within 20%)."""
    v3_result = run_v3(test_player)
    v4_result = run_v4(test_player)

    diff_pct = abs(v4_result['mean'] - v3_result['mean']) / v3_result['mean']

    assert diff_pct < 0.20, f"V4 mean too different from V3: {diff_pct:.1%}"
```

### 6.3 Validation Metrics

**Distribution Goodness-of-Fit**:
- Kolmogorov-Smirnov test for NegBin vs empirical targets
- Anderson-Darling test for Lognormal vs empirical Y/T
- χ² test for overall distribution fit

**Prediction Accuracy**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**Calibration**:
- Brier score
- Log loss
- Calibration plot (predicted vs actual probabilities)

**Betting Performance**:
- ROI (Return on Investment)
- Win rate
- Sharpe ratio

---

## 7. Deployment Plan

### 7.1 Phased Rollout

**Week 1**: V4 shadow mode
- Run V4 alongside V3
- Generate both outputs
- Compare but don't use V4 for betting

**Week 2**: Pilot (TE only)
- Use V4 for TE predictions only
- Use V3 for QB/RB/WR
- Monitor accuracy

**Week 3**: Expanded pilot (TE + WR)
- Add WRs to V4
- Keep QB/RB on V3

**Week 4**: Full production
- All positions on V4
- Archive V3 as backup

### 7.2 Monitoring

**Dashboard Metrics**:
- Prediction errors (daily)
- Distribution fit quality (weekly)
- ROI trending (weekly)
- V4 vs V3 accuracy comparison (weekly)

**Alerts**:
- MAE > 15% above V3 baseline → investigate
- Negative ROI for 2+ consecutive weeks → rollback consideration
- Distribution fit fails χ² test → investigate correlation estimates

---

## 8. Rollback Strategy

### 8.1 Rollback Triggers

**Automatic Rollback** if:
- V4 MAE > V3 MAE + 20% for 2 consecutive weeks
- Negative ROI for 3 consecutive weeks
- Critical bug causing prediction failures

**Manual Rollback** if:
- User requests it
- Distribution assumptions violated (QQ plots fail)
- Correlation estimates wildly incorrect

### 8.2 Rollback Procedure

```bash
# 1. Stop current predictions
pkill -f generate_model_predictions

# 2. Run rollback script
bash scripts/utils/rollback_to_v3.sh

# 3. Regenerate Week N with V3
python scripts/predict/generate_model_predictions.py $WEEK --simulator v3

# 4. Verify output
python scripts/validate/verify_v3_output.py

# 5. Notify user
echo "✅ Rolled back to V3 successfully"
```

---

## 9. Success Criteria

### 9.1 Must-Have (GO Decision)

- [ ] V4 MAE ≤ V3 MAE (no accuracy degradation)
- [ ] V4 calibration error ≤ V3 calibration error (better probabilities)
- [ ] Distribution fits pass validation (KS test p > 0.05)
- [ ] No critical bugs in 2-week pilot

### 9.2 Should-Have (Quality Gates)

- [ ] V4 MAE < V3 MAE - 10% (10%+ accuracy improvement)
- [ ] V4 ROI > 0% on Weeks 5-11 backtest (profitable)
- [ ] Correlation estimates within 20% of empirical (reasonable estimates)

### 9.3 Nice-to-Have (Aspirational)

- [ ] V4 MAE < V3 MAE - 20% (20%+ accuracy improvement)
- [ ] V4 ROI > 5% on Weeks 5-11 (highly profitable)
- [ ] V4 Sharpe ratio > 1.0 (risk-adjusted returns excellent)

---

## 10. Timeline Summary

| Phase | Duration | Deliverables | Blocker |
|-------|----------|--------------|---------|
| 0: Preparation | 3-5 days | Routes data, EWMA defaults, validation set | Routes data source |
| 1: Distributions | 5-7 days | NegBin, Lognormal, Copula modules | None |
| 2: Model Retrain | 5-7 days | V4 models (distribution params) | Historical data |
| 3: Correlation | 3-5 days | Correlation estimates | None |
| 4: Simulator V4 | 7-10 days | PlayerSimulatorV4 | Phases 1-3 |
| 5: Output Schema | 3-4 days | Percentile outputs | Phase 4 |
| 6: Betting Layer | 4-6 days | Kelly + narratives | Phase 5 |
| 7: Validation | 5-7 days | Backtest, GO/NO-GO | All phases |
| 8: Production | 2-3 days | V4 deployed | Phase 7 GO |
| **TOTAL** | **37-54 days** | **Complete V4 system** | - |

**Calendar Estimate**: 6-8 weeks (allowing for weekends, debugging, iteration)

---

## 11. Next Steps

**IMMEDIATE** (this session):
1. Review this specification with user
2. Get GO/NO-GO decision on full overhaul
3. Get decisions on:
   - Routes run data source (Option A/B/C)
   - Timeline urgency (can Week 12 wait?)
   - Scope (all positions or pilot TE first?)

**IF GO DECISION**:
1. Create feature branch: `git checkout -b feature/v4-probabilistic-distributions`
2. Start Phase 0: Data preparation
3. Source routes run data
4. Set up validation dataset (Weeks 5-11)

**IF NO-GO** (use current system):
1. Document V3 as stable
2. Use Week 12 predictions (already generated, bug-fixed)
3. Revisit V4 overhaul after playoffs

---

**Prepared By**: Senior Quantitative Sports Analytics Architect
**Date**: November 23, 2025
**Status**: ✅ SPECIFICATION COMPLETE, AWAITING USER DECISION
