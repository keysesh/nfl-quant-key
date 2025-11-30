# Model Architecture Overhaul Plan

**Date**: November 23, 2025
**Status**: 📋 PLANNING PHASE
**Scope**: Complete redesign of prediction pipeline

---

## Current vs Proposed Architecture

### Current Architecture (What We Have Now)

```
1. Data: Targets, yards, receptions from NFLverse
2. Usage Model: XGBoost predicts mean targets
3. Efficiency Model: XGBoost predicts mean yards/target
4. Simulation: Normal distributions, 10k trials
5. Output: Mean ± std for yards, targets, receptions
```

**Issues**:
- ✅ Works, but lacks advanced metrics (route participation, TPRR)
- ✅ Uses simple normal distributions (not Negative Binomial or lognormal)
- ✅ No explicit correlation modeling between targets and efficiency
- ✅ Doesn't output percentiles (25th, 75th, etc.)
- ✅ No catch rate issue (we don't double-count)

### Proposed Architecture (Your Design)

```
1. Data: Add route participation, TPRR, Y/RR metrics + EWMA weighting
2. Usage Model: Negative Binomial distribution for targets (better for count data)
3. Efficiency Model: Lognormal distribution for Y/T (better for skewed positive data)
4. Correlation: Model dependency between targets and Y/T
5. Simulation: Correlated sampling from NegBin + Lognormal
6. Output: Mean, median, percentiles (5th, 25th, 75th, 95th)
7. Betting: Kelly fraction, confidence scores, risk analysis
```

---

## Implementation Phases

### Phase 1: Data Enhancement (Estimated: 2-3 days)

**Goal**: Add advanced receiving metrics

**New Data Sources Needed**:
- **Routes Run**: Not in NFLverse by default - need PFF or Pro Football Reference
- **Route Participation**: Calculate from snap counts + pass plays
- **TPRR**: targets / routes_run
- **Y/RR**: receiving_yards / routes_run

**Files to Create**:
```
nfl_quant/features/route_metrics.py
  - calculate_route_participation()
  - calculate_tprr()
  - calculate_yrr()
  - apply_ewma_weights()
```

**Files to Modify**:
```
scripts/predict/generate_model_predictions.py
  - load_route_metrics() in load_trailing_stats()
  - Add TPRR, Y/RR to feature extraction
```

**Challenge**: Routes run data not readily available in NFLverse
- **Option A**: Estimate from snap counts × pass play rate
- **Option B**: Purchase PFF data
- **Option C**: Scrape Pro Football Reference

---

### Phase 2: Statistical Distribution Upgrade (Estimated: 3-4 days)

**Goal**: Replace normal distributions with Negative Binomial + Lognormal

#### 2.1 Negative Binomial for Targets

**Why**: Targets are count data with overdispersion (variance > mean)

**Current**:
```python
# Normal distribution
targets_mean = usage_model.predict(features)
targets_std = empirical_std
samples = np.random.normal(targets_mean, targets_std, 10000)
```

**Proposed**:
```python
# Negative Binomial distribution
from scipy.stats import nbinom

# Predict mean and overdispersion parameter
targets_mean = usage_model.predict(features)
overdispersion = calculate_overdispersion(player_history)

# Convert to NegBin parameters (n, p)
n, p = mean_var_to_negbin_params(targets_mean, overdispersion)

# Sample
samples = nbinom.rvs(n, p, size=10000)
```

**Files to Create**:
```
nfl_quant/distributions/negative_binomial.py
  - fit_negbin_from_data()
  - mean_var_to_negbin_params()
  - NegativeBinomialSampler class
```

#### 2.2 Lognormal for Yards/Target

**Why**: Y/T is strictly positive with right skew (outlier big plays)

**Current**:
```python
# Normal distribution (can go negative!)
ypt_mean = efficiency_model.predict(features)
ypt_std = empirical_std
samples = np.random.normal(ypt_mean, ypt_std, 10000)
samples = np.clip(samples, 0, None)  # Hack to prevent negative
```

**Proposed**:
```python
# Lognormal distribution (always positive, right-skewed)
from scipy.stats import lognorm

# Predict log-space parameters
log_mean, log_std = efficiency_model.predict_lognormal_params(features)

# Sample
samples = lognorm.rvs(s=log_std, scale=np.exp(log_mean), size=10000)
```

**Files to Create**:
```
nfl_quant/distributions/lognormal.py
  - fit_lognormal_from_data()
  - LognormalSampler class
```

---

### Phase 3: Correlation Modeling (Estimated: 2-3 days)

**Goal**: Model correlation between targets and yards/target

**Why**: High-target games often have lower Y/T (shorter, safer throws)

**Proposed Method**: Copula-based correlation

```python
from scipy.stats import norm
from scipy.stats.mstats import spearmanr

# Calculate historical correlation
target_history = player_data['targets']
ypt_history = player_data['yards'] / player_data['targets']
correlation = spearmanr(target_history, ypt_history).correlation

# Generate correlated uniform samples using Gaussian copula
u1, u2 = generate_correlated_uniforms(correlation, size=10000)

# Transform to marginal distributions
targets = nbinom.ppf(u1, n, p)
yards_per_target = lognorm.ppf(u2, s=log_std, scale=np.exp(log_mean))

# Calculate receiving yards
receiving_yards = targets * yards_per_target
```

**Files to Create**:
```
nfl_quant/correlation/copula.py
  - GaussianCopula class
  - generate_correlated_samples()
  - estimate_correlation_from_data()
```

---

### Phase 4: Simulation Engine Rewrite (Estimated: 4-5 days)

**Goal**: Replace current simulator with advanced version

**Files to Create**:
```
nfl_quant/simulation/player_simulator_v4_advanced.py
  - AdvancedPlayerSimulator class
  - Uses NegBin for targets
  - Uses Lognormal for Y/T
  - Applies correlation structure
  - Outputs full distribution (not just mean/std)
```

**New Output Schema**:
```python
@dataclass
class AdvancedSimulationResult:
    # Point estimates
    mean: float
    median: float
    mode: float

    # Percentiles
    p5: float
    p25: float
    p75: float
    p95: float

    # Uncertainty
    std: float
    cv: float  # Coefficient of variation

    # For betting
    prob_over_line: float  # P(X > sportsbook_line)
    confidence_score: float  # Model confidence (0-1)
    kelly_fraction: float  # Recommended bet size

    # Distribution
    samples: np.ndarray  # All 10k samples for analysis
```

---

### Phase 5: Betting Intelligence Layer (Estimated: 2-3 days)

**Goal**: Add sophisticated betting recommendations

**Files to Create**:
```
nfl_quant/betting/advanced_recommendations.py
  - calculate_kelly_fraction()
  - assess_volatility_risk()
  - generate_narrative_explanation()
  - identify_upside_pathways()
  - identify_downside_risks()
```

**Example Output**:
```
George Kittle vs CAR - Receiving Yards (Line: 45.5)

RECOMMENDATION: UNDER 45.5 ❌
  Edge: -8.2%
  Confidence: MEDIUM (72%)
  Kelly Fraction: 0.0% (no bet)

PROJECTION:
  Mean: 27.8 yards
  Median: 24.5 yards
  25th-75th percentile: 15-38 yards

  P(Over 45.5): 22%
  P(Under 45.5): 78% ✅

VOLATILITY: HIGH (CV = 1.18)
  - Wide range of outcomes (5th: 2 yards, 95th: 78 yards)
  - High-variance player (boom/bust potential)

MATCHUP ANALYSIS:
  ✅ Weak defense (CAR +0.0686 EPA vs TE)
  ⚠️  Game script concern (SF likely leads → less passing)
  ⚠️  Route participation may decrease in blowout

DOWNSIDE RISKS:
  - SF blows out CAR early (40% probability)
  - Kittle sees <4 targets in 2nd half
  - CMC dominates red zone (takes TDs)

UPSIDE PATHWAYS:
  - CAR keeps game close (requires CAR offense to perform)
  - Kittle gets 8+ targets (requires competitive game)
  - Big play (>40 yard TD) - 12% probability

BET SIZING: PASS
  - Line too far from median (45.5 vs 24.5)
  - High variance makes this -EV even with edge
```

---

## Critical Decisions Needed

### Decision 1: Routes Run Data Source

**Problem**: NFLverse doesn't have routes_run

**Options**:
1. **Estimate** from snaps × (pass_attempts / total_plays)
   - Pro: Free, immediate
   - Con: Noisy estimate

2. **Purchase PFF data** (~$200/month)
   - Pro: Gold standard data
   - Con: Cost, integration effort

3. **Scrape Pro Football Reference**
   - Pro: Free
   - Con: Legal gray area, brittle

**Recommendation**: Start with Option 1 (estimate), validate with spot-checks, consider PFF later if needed

---

### Decision 2: Model Retraining Strategy

**Problem**: Current models output mean predictions, not distribution parameters

**Options**:
1. **Retrain from scratch** with new targets:
   - Targets model outputs: (mean, overdispersion)
   - Efficiency model outputs: (log_mean, log_std)

2. **Post-process current models**:
   - Keep current mean predictions
   - Estimate variance from historical data

**Recommendation**: Option 2 for Phase 1, Option 1 for Phase 2

---

### Decision 3: Backward Compatibility

**Problem**: Existing pipeline uses current simulator

**Options**:
1. **Replace entirely** - Break existing workflow
2. **Add new pipeline** - Maintain both (v3 and v4)
3. **Gradual migration** - Feature flags to toggle new components

**Recommendation**: Option 2 - Create PlayerSimulatorV4 alongside V3, allow users to choose

---

## Implementation Timeline

### Sprint 1 (Week 1): Data Enhancement
- [ ] Days 1-2: Implement route participation estimation
- [ ] Day 3: Add TPRR, Y/RR calculations
- [ ] Days 4-5: Add EWMA weighting, validate metrics

### Sprint 2 (Week 2): Distribution Upgrade
- [ ] Days 1-2: Implement Negative Binomial for targets
- [ ] Days 3-4: Implement Lognormal for Y/T
- [ ] Day 5: Unit tests, validation

### Sprint 3 (Week 3): Correlation & Simulation
- [ ] Days 1-2: Implement Gaussian copula
- [ ] Days 3-5: Build PlayerSimulatorV4

### Sprint 4 (Week 4): Betting Layer & Testing
- [ ] Days 1-2: Betting intelligence module
- [ ] Days 3-4: End-to-end testing
- [ ] Day 5: Documentation

**Total Estimated Time**: 4 weeks full-time development

---

## Risks & Mitigation

### Risk 1: Routes Run Data Quality
**Impact**: HIGH - Core metric for new model
**Mitigation**: Extensive validation against known players, cross-check with PFF free samples

### Risk 2: Model Performance Degradation
**Impact**: HIGH - New model might be worse
**Mitigation**: A/B test V4 vs V3 on historical data, keep V3 as fallback

### Risk 3: Complexity Increase
**Impact**: MEDIUM - Harder to debug, maintain
**Mitigation**: Comprehensive unit tests, clear documentation, modular design

### Risk 4: Calibration Issues
**Impact**: HIGH - Probabilities must be accurate
**Mitigation**: Extensive backtesting, isotonic calibration on top of new distributions

---

## Prerequisites Before Starting

1. ✅ **Current system working** - Yes (Week 12 ready)
2. ❌ **Routes run data** - Need to source or estimate
3. ❌ **Historical validation data** - Need cleaned dataset for backtesting
4. ❌ **Development environment** - Need staging area separate from production
5. ❌ **Testing framework** - Need comprehensive test suite

---

## Recommendation

Given the scope and complexity, I recommend a **phased approach**:

### Phase 0 (Now - 1 week): Preparation
- Create development branch
- Set up testing framework
- Gather routes run data (estimation method)
- Create validation dataset (Weeks 5-11 with actual outcomes)

### Phase 1 (Weeks 2-3): Proof of Concept
- Implement for ONE player (George Kittle)
- Validate distribution fits
- Compare to current model
- Get user feedback

### Phase 2 (Weeks 4-6): Full Implementation
- Scale to all positions
- Implement correlation
- Build betting layer
- Extensive testing

### Phase 3 (Weeks 7-8): Production Deployment
- A/B testing vs current model
- Gradual rollout
- Monitor performance

**Total Timeline**: 8 weeks for production-ready system

---

## Next Steps

**Immediate actions** (if you want to proceed):

1. **Decide on routes run data source**
   - Estimate from snaps? Purchase PFF? Scrape PFR?

2. **Set up development environment**
   ```bash
   git checkout -b feature/advanced-simulator-v4
   mkdir -p nfl_quant/simulation_v4
   mkdir -p nfl_quant/distributions
   mkdir -p nfl_quant/correlation
   ```

3. **Create proof-of-concept script**
   ```bash
   scripts/experimental/kittle_v4_comparison.py
   # Compare current vs proposed for George Kittle Week 12
   ```

4. **Validate on historical data**
   - Run both models on Weeks 5-11
   - Compare accuracy, calibration
   - Measure improvement

---

## Questions for You

Before I start implementation, I need clarity on:

1. **Timeline**: Do you need this for Week 13 (7 days) or can it wait?
2. **Routes Run**: Which data source should I use?
3. **Scope**: Start with all positions or just TE/WR first?
4. **Risk Tolerance**: Replace current system or run in parallel?
5. **Testing**: How much validation do you want before production?

---

## Status

**Current System**: ✅ WORKING (Week 12 ready, all bugs fixed)
**Proposed System**: 📋 PLANNING PHASE (4-8 weeks to implement)

**My Recommendation**:
- Use current system for Week 12-13
- Start V4 development in parallel
- Target Week 14-15 for V4 pilot
- Full rollout by Week 16 (playoffs)

This gives us time to:
- Implement correctly
- Test thoroughly
- Validate accuracy
- Build confidence

**Your call**: Do you want to proceed with this overhaul, or should we stick with current system and make incremental improvements?

---

**Prepared By**: NFL QUANT Development Team
**Date**: November 23, 2025
**Status**: Awaiting user decision on scope and timeline
