# NFL Player Props Model Reconstruction Plan

**Date**: December 7, 2025
**Status**: CRITICAL - Model Has No Edge
**Priority**: Immediate Action Required

---

## Executive Summary

### The Core Problem

**The model adds negative value. You would be better off just predicting the betting line.**

| Metric | Our Model | Just Use Line | Difference |
|--------|-----------|---------------|------------|
| Weighted MAE | 30.03 | 20.12 | **+9.92 worse** |
| Direction Accuracy | 49.1% | 50.0% | **-0.9% worse** |
| Prediction-Actual Corr | 0.24 avg | 0.47 avg | **-0.23 worse** |

When the model diverges from the line, it is WRONG more often than right. Higher divergence = larger errors.

### Root Causes Identified

1. **Broken Features**: Critical features are 100% null (team_pass_attempts, game_script_dynamic)
2. **Wrong Distribution**: Using normal distribution for count data (receptions) and skewed data (yards)
3. **Single Model Fallacy**: Same features for fundamentally different markets
4. **Systematic Biases**: Rushing +35 yards bias, Passing -72 yards bias
5. **No Game Context**: Vegas spread/total not integrated into predictions

---

## Phase 1: Current State Assessment

### 1.1 Prediction Error by Market

| Market | MAE | MAE % of Line | Bias | Pred-Actual Corr | Direction Acc |
|--------|-----|---------------|------|------------------|---------------|
| Receptions | 1.79 | 48.3% | -0.27 | 0.378 | **49.7%** |
| Rec Yards | 27.48 | 75.8% | +8.24 | 0.351 | 51.4% |
| Rush Yards | 44.70 | **102.9%** | **+34.93** | 0.133 | **46.4%** |
| Pass Yards | 83.99 | 37.4% | **-71.70** | 0.078 | **47.9%** |

**Key Issues:**
- Receptions: WORSE than coin flip (49.7% direction accuracy)
- Rushing yards: MAE exceeds the average line value
- Rushing yards: Systematic OVER-prediction by 35 yards
- Passing yards: Systematic UNDER-prediction by 72 yards
- Passing yards: Near-zero correlation with actual outcomes

### 1.2 Model vs Line Comparison

The betting line is a better predictor than our model in ALL markets:

```
DIVERGENCE FROM LINE → ERROR CORRELATION
  Receptions:      +0.155 (more divergence = more error)
  Receiving Yards: +0.299 (more divergence = more error)
  Rushing Yards:   +0.373 (more divergence = more error)
  Passing Yards:   +0.200 (more divergence = more error)
```

**Interpretation**: When we disagree with the books, WE are wrong.

### 1.3 Structural Failures

**Blowout Prediction Accuracy:**
| Scenario | Receptions | Rec Yards | Rush Yards | Pass Yards |
|----------|------------|-----------|------------|------------|
| Blowout OVER | 37.2% | 87.5% | 96.7% | 0.0% |
| Blowout UNDER | 55.1% | **16.5%** | **4.2%** | 100.0% |

- Model NEVER correctly predicts rushing blowout UNDERs (4.2%)
- Model NEVER correctly predicts passing blowout OVERs (0.0%)

---

## Phase 2: Feature Engineering Gaps

### 2.1 Broken Features (100% Null)

```
❌ team_pass_attempts: 100% null
❌ team_rush_attempts: 100% null
❌ team_targets: 100% null
❌ game_script_dynamic: 100% null
❌ primetime_type: 100% null
```

These are supposed to be critical features but have NO DATA.

### 2.2 Zero-Variance Features (Useless)

```
❌ rest_epa_adjustment: 1 unique value
❌ travel_epa_adjustment: 1 unique value
❌ home_field_advantage_points: 1 unique value
❌ altitude_epa_adjustment: 2 unique values
```

### 2.3 Missing Critical Features by Market

**Receptions:**
- Route participation rate
- Rolling target share
- Slot vs outside alignment %
- Air yards share
- Separation metrics (NGS)
- QB target tendency under pressure

**Receiving Yards:**
- Average depth of target (aDOT)
- YAC over expected
- Yards per target allowed by opponent CB
- Red zone target share (compresses yards)

**Rushing Yards:**
- Offensive line run blocking grade
- Stacked box rate against player
- Yards before contact
- Game script projection (spread-derived)

**Passing Yards:**
- Wind speed (specific, not generic weather)
- Time to throw
- Sack rate faced
- Play action / RPO rate

---

## Phase 3: Distribution Modeling Failures

### 3.1 Normality Violations

ALL markets fail normality tests (Shapiro-Wilk p < 0.01):

| Market | Skewness | Kurtosis | Zero Rate | Correct Distribution |
|--------|----------|----------|-----------|---------------------|
| Receptions | 0.93 | 1.16 | 5.8% | **Negative Binomial** |
| Rec Yards | 1.09 | 1.13 | 8.6% | **Gamma** |
| Rush Yards | 1.65 | 4.19 | - | **Gamma Mixture** |
| Pass Yards | 0.19 | 1.04 | - | Normal w/ variance adj |

### 3.2 Heteroscedasticity Confirmed

Variance scales with prediction magnitude:

| Market | Q4/Q1 Std Ratio | Implication |
|--------|-----------------|-------------|
| Receptions | 1.42x | Higher receptions = more variance |
| Rec Yards | 1.59x | Higher yards = more variance |
| Rush Yards | 1.51x | Higher yards = more variance |

**Current model ignores this** → Probability calculation is wrong for high-volume players.

### 3.3 Fat Tails

| Market | Fat Tail Ratio | Implication |
|--------|----------------|-------------|
| Receptions | 0.88 | Normal tails OK |
| Rec Yards | 1.22 | Moderate fat tails |
| Rush Yards | **2.29** | Heavy fat tails |
| Pass Yards | **2.48** | Heavy fat tails |

Rushing and passing yards have extreme outcomes more often than normal distribution predicts.

---

## Phase 4: Architecture Recommendations

### 4.1 Current vs Proposed Architecture

**Current (Broken):**
```
Single Model → Normal Distribution → Probability
      ↓
Same features for all markets
      ↓
Systematic miscalibration
```

**Proposed (Fixed):**
```
┌─────────────────────────────────────────────────────────────┐
│                    GAME CONTEXT LAYER                       │
│  Vegas Spread → Pass/Rush Split Projection                  │
│  Vegas Total → Pace Projection                              │
│  Weather → Passing Adjustment                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ RECEPTIONS MODEL  │ │ REC YARDS MODEL   │ │ RUSH YARDS MODEL  │
│                   │ │                   │ │                   │
│ Distribution:     │ │ Distribution:     │ │ Distribution:     │
│ Neg Binomial      │ │ Gamma             │ │ Gamma Mixture     │
│                   │ │                   │ │                   │
│ Features:         │ │ Features:         │ │ Features:         │
│ - Target share    │ │ - aDOT            │ │ - OL grade        │
│ - Route rate      │ │ - YAC/Expected    │ │ - Box count       │
│ - Slot %          │ │ - Separation      │ │ - Carry share     │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              VARIANCE SCALING LAYER                         │
│  std = base_std × sqrt(pred_mean / baseline)               │
│  + game_total_adjustment + spread_adjustment                │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Implementation Roadmap

### Priority 1: Quick Wins (1-2 Days)

| Fix | Expected Impact | Effort | Priority |
|-----|-----------------|--------|----------|
| Fix null features (team_pass_attempts, etc.) | +3% direction acc | Low | **P0** |
| Add Vegas spread to predictions | +2% direction acc | Low | **P0** |
| Add Vegas total to predictions | +2% direction acc | Low | **P0** |
| Implement heteroscedastic std | -5% ECE | Medium | **P0** |

### Priority 2: Distribution Fixes (3-5 Days)

| Fix | Expected Impact | Effort | Priority |
|-----|-----------------|--------|----------|
| Negative Binomial for receptions | +4% direction acc | Medium | **P1** |
| Gamma for yards markets | +3% direction acc | Medium | **P1** |
| Zero-inflation handling | +1% direction acc | Low | **P1** |

### Priority 3: Feature Engineering (1-2 Weeks)

| Fix | Expected Impact | Effort | Priority |
|-----|-----------------|--------|----------|
| Add target share (rolling) | +3% for receptions | Medium | **P2** |
| Add aDOT for receivers | +2% for rec yards | Medium | **P2** |
| Add OL grades for rushing | +3% for rush yards | High | **P2** |
| Add weather details | +2% for pass yards | Medium | **P2** |

### Priority 4: Market-Specific Models (2-3 Weeks)

| Fix | Expected Impact | Effort | Priority |
|-----|-----------------|--------|----------|
| Separate model per market | +5% direction acc | High | **P3** |
| Market-specific feature selection | +3% direction acc | Medium | **P3** |
| Ensemble with linear baseline | +2% calibration | Medium | **P3** |

---

## Success Metrics

### Minimum Viable Product (MVP) Targets

| Metric | Current | Target | Threshold to Deploy |
|--------|---------|--------|---------------------|
| Direction Accuracy | 49.1% | **53%+** | 52% |
| MAE vs Line | +9.92 worse | **Equal or better** | +5 or less |
| ECE (calibration) | 0.22 | **<0.05** | <0.10 |
| Model-Actual Corr | 0.24 | **>0.45** | >0.35 |

### Per-Market Targets

| Market | Current Dir Acc | Target | Current Bias | Target Bias |
|--------|----------------|--------|--------------|-------------|
| Receptions | 49.7% | 54% | -0.27 | ±0.10 |
| Rec Yards | 51.4% | 54% | +8.24 | ±5.0 |
| Rush Yards | 46.4% | 52% | +34.93 | ±10.0 |
| Pass Yards | 47.9% | 52% | -71.70 | ±20.0 |

### Validation Protocol

1. **Hold-out Validation**
   - Train on: 2022-2024 + Weeks 1-8 of 2025
   - Test on: Weeks 9+ of 2025
   - Must beat line on hold-out to deploy

2. **Paper Trading Period**
   - 2 weeks of shadow mode before live deployment
   - Log all recommendations, compare to actuals
   - No real capital until validation passes

3. **Rollback Triggers**
   - Direction accuracy drops below 48% for any market
   - ECE increases above 0.15
   - MAE becomes 20%+ worse than line

---

## Immediate Next Steps

### Day 1: Fix Broken Features
```bash
# 1. Investigate why features are null
python scripts/debug/audit_feature_pipeline.py

# 2. Fix feature calculation for:
#    - team_pass_attempts (use schedule/game data)
#    - team_rush_attempts
#    - game_script_dynamic (derive from spread/total)
```

### Day 2: Add Vegas Integration
```bash
# 1. Add spread/total to feature pipeline
# 2. Calculate expected pass/rush ratio from spread
# 3. Add to model predictions
```

### Day 3-5: Distribution Fixes
```bash
# 1. Implement Negative Binomial for receptions
# 2. Implement Gamma for yards
# 3. Add heteroscedastic variance scaling
# 4. Validate on backtest data
```

### Week 2: Market-Specific Models
```bash
# 1. Create separate feature sets per market
# 2. Train market-specific XGBoost models
# 3. Implement market-appropriate distributions
# 4. Run full walk-forward validation
```

---

## Conclusion

The model needs **fundamental reconstruction**, not tuning. The current approach of:
- Same features for all markets
- Normal distribution for all stats
- Ignoring game context
- Broken feature pipeline

...produces predictions that are **worse than the betting line**.

The good news: the path forward is clear. Fix the features, fix the distributions, add game context, and train market-specific models. With these changes, we have a realistic path to 53%+ direction accuracy and genuine edge.

Without these changes, we're betting blind.
