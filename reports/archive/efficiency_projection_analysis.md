# Efficiency Projection Analysis - Conservative YPC & Y/R Investigation
**Date**: November 23, 2025
**Issue**: Models projecting unusually low efficiency metrics

---

## Executive Summary

**CRITICAL FINDINGS**:

1. ✅ **EWMA Snap Share Calculation - WORKING AS DESIGNED**
   - Derrick Henry: 58.7% projected (vs 56.2% simple average, 60.8% recent 4-week average)
   - Uses EWMA (span=4) correctly: 40% weight on most recent week, decreasing exponentially
   - **Status**: NO BUG - this is correct behavior per CLAUDE.md

2. 🚨 **EFFICIENCY PROJECTIONS - SEVERELY UNDERESTIMATING**
   - Mark Andrews: 5.38 Y/R projected vs **8.62 Y/R actual** (-37.7% error)
   - Derrick Henry: 2.61 YPC projected vs **4.86 YPC actual** (-46.3% error)
   - **Root Cause**: Models predicting BELOW their own input features
   - **Status**: 🚨 CRITICAL BUG - Model regression/overfitting issue

---

## Finding #1: EWMA Snap Share Calculation ✅ CORRECT

### Derrick Henry Snap Share Analysis

**Actual Snap Data (from snap_counts.parquet)**:
```
Week 1:  57.0%
Week 2:  61.0%
Week 3:  49.0%
Week 4:  38.0%  (low usage game)
Week 5:  65.0%
Week 6:  49.0%
Week 8:  58.0%
Week 9:  64.0%
Week 10: 67.0%
Week 11: 54.0%
```

**Calculated Averages**:
- Simple Mean (all 10 weeks): **56.2%**
- Recent 4 Weeks Mean (8-11): **60.8%**
- EWMA (span=4): **58.7%** ← Used by model
- **Recommendation Value**: **58.7%** ✅ MATCH

### EWMA Calculation Breakdown

EWMA formula with span=4:
```
Alpha = 2 / (span + 1) = 2 / 5 = 0.40

Weights (most recent to oldest):
Week 11 (54.0%): Start value
Week 10 (67.0%): EWMA = 0.40 × 67.0% + 0.60 × 54.0% = 59.2%
Week 9 (64.0%):  EWMA = 0.40 × 64.0% + 0.60 × 59.2% = 61.1%
Week 8 (58.0%):  EWMA = 0.40 × 58.0% + 0.60 × 61.1% = 59.9%
...
Final EWMA: 58.7%
```

**Interpretation**:
- EWMA gives 40% weight to most recent week, 27% to week N-2, 18% to week N-3, 12% to week N-4
- This correctly balances recent trend (60.8%) with longer-term average (56.2%)
- **58.7% is mathematically correct** per CLAUDE.md TIER 1 enhancements

**Conclusion**: ✅ **NO ACTION NEEDED** - EWMA working as designed

---

## Finding #2: Efficiency Projections - CRITICAL BUG 🚨

### Mark Andrews Receiving Efficiency

**Actual 2025 Performance**:
```
Total Targets: 43
Total Catches: 32
Total Yards: 276
Catch Rate: 74.4%
Actual Y/R: 8.62 yards/reception
Actual Y/Tgt: 6.42 yards/target
```

**Weekly Y/R Breakdown**:
```
Week 1:   5.00 (1 catch)
Week 2:   2.00 (1 catch)
Week 3:  15.17 (6 catches) ← High variance
Week 4:   4.29 (7 catches)
Week 5:  11.00 (2 catches)
Week 6:   6.00 (4 catches)
Week 8:  11.33 (3 catches)
Week 9:  11.00 (2 catches)
Week 10:  4.67 (3 catches)
Week 11: 10.67 (3 catches)

Mean: 8.62 Y/R
Median: 8.50 Y/R
```

**Model Projection**:
- Model Projected Y/R: **5.38**
- Actual Y/R: **8.62**
- **Error: -37.7%** (model underestimates by 3.25 yards/reception)

**Expected Range for TE**: 8-12 Y/R (Andrews at 8.62 is on the LOW end of typical)

### Derrick Henry Rushing Efficiency

**Actual 2025 Performance**:
```
Total Carries: 166
Total Yards: 807
Actual YPC: 4.86
Median YPC: 3.00
Std Dev: 8.39 (high variance due to big runs)
```

**Weekly YPC Breakdown**:
```
Week 1:  9.39 (169 yards on 18 carries) ← Huge game
Week 2:  2.09 (23 yards on 11 carries)
Week 3:  4.17 (50 yards on 12 carries)
Week 4:  5.25 (42 yards on 8 carries)
Week 5:  2.20 (33 yards on 15 carries)
Week 6:  5.08 (122 yards on 24 carries)
Week 8:  3.38 (71 yards on 21 carries)
Week 9:  6.26 (119 yards on 19 carries)
Week 10: 3.75 (75 yards on 20 carries)
Week 11: 5.72 (103 yards on 18 carries)

Overall: 4.86 YPC
Recent 4 weeks (8-11): 4.72 YPC
EWMA (span=4): 4.97 YPC
```

**Model Projection**:
- Model Projected YPC: **2.61**
- Actual YPC: **4.86**
- **Error: -46.3%** (model underestimates by 2.25 yards/carry)

**Career Baseline**: Derrick Henry career YPC ≈ 4.5

### Opponent Defense Analysis (Jets Week 12)

**Jets Run Defense EPA**:
```
Jets Run Defense EPA: +0.0520
League Average Run EPA: -0.0001
Difference: +0.0520 (POSITIVE = WEAK DEFENSE)

Interpretation: Jets allow MORE yards than league average
→ Should INCREASE Henry projection, not decrease
```

**Jets Pass Defense EPA**:
```
Jets Pass Defense EPA: +0.1459
League Average Pass EPA: +0.0272
Difference: +0.1187 (WEAK DEFENSE)

Interpretation: Jets allow significantly more passing yards
→ Should INCREASE Andrews projection, not decrease
```

**Conclusion**: Opponent defense is FAVORABLE for both players. Models should be predicting HIGHER efficiency, not lower.

---

## Root Cause Analysis

### Efficiency Predictor Model Structure

**File**: `data/models/efficiency_predictor_v2_defense.joblib`

**Model Details**:
- Version: v2_with_defense
- Trained: 2025-11-15 17:12:19
- Training Samples: 5,365

**RB Efficiency Model Features**:
```
1. week
2. trailing_yards_per_carry  ← Input feature (historical YPC)
3. trailing_yards_per_target
4. trailing_td_rate_rush
5. trailing_td_rate_pass
6. opp_rush_def_epa  ← Opponent defense strength
7. opp_rush_def_rank
8. opp_pass_def_epa
9. trailing_opp_rush_def_epa
10. team_pace
```

### The Critical Bug

**For Derrick Henry**:
```
INPUT FEATURES:
- trailing_yards_per_carry: ~4.72 (calculated from weeks 8-11)
- opp_rush_def_epa: +0.0520 (Jets WEAK vs run)

MODEL OUTPUT:
- predicted_yards_per_carry: 2.61

PROBLEM: Model predicts 2.61 YPC despite:
1. Input trailing YPC of 4.72 (healthy)
2. Opponent allowing +0.0520 EPA (weak defense)
3. No injuries, weather, or unusual game script

→ Model is applying -2.11 YPC NEGATIVE adjustment (-44.7%)
→ This suggests severe overfitting or regression to mean issue
```

**For Mark Andrews**:
```
INPUT FEATURES:
- Historical Y/R: ~8.62 (actual performance)
- opp_pass_def_epa: +0.1459 (Jets VERY WEAK vs pass)

MODEL OUTPUT:
- predicted_yards_per_reception: 5.38

PROBLEM: Model predicts 5.38 Y/R despite:
1. Actual Y/R of 8.62 (typical TE performance)
2. Opponent allowing +0.1459 EPA (very weak pass defense)

→ Model is applying -3.25 Y/R NEGATIVE adjustment (-37.7%)
```

### Hypotheses

**Hypothesis #1: Model Overfitting to Training Data**
- Model trained on 2022-2024 data (older seasons)
- 2025 season may have different offensive/defensive dynamics
- Model applying outdated regression patterns

**Hypothesis #2: Negative Opponent DEF EPA Interpretation**
- Model may be interpreting +0.0520 EPA INCORRECTLY
- Expected: Positive EPA = Weak defense = MORE yards allowed
- Actual behavior: Positive EPA = Applying NEGATIVE adjustment?
- **Sign flip bug in opponent defense feature?**

**Hypothesis #3: Extreme Regression to Mean**
- Model may be over-regressing to league average
- League average RB YPC ≈ 4.0
- League average TE Y/R ≈ 9.5
- But model regressing TOO AGGRESSIVELY (below league average)

**Hypothesis #4: Feature Engineering Bug**
- `trailing_yards_per_carry` calculation may be wrong
- Possible issues:
  - Using wrong trailing window (all-season vs 4-week)
  - Calculating per-game average instead of per-carry
  - Mixing rushing and receiving stats

---

## Validation Steps Needed

### Step 1: Verify Trailing Feature Calculations

**Action**: Trace `trailing_yards_per_carry` calculation for Derrick Henry

Expected flow:
```python
# In generate_model_predictions.py
player_df = weekly_stats[(player == 'Derrick Henry') & (week < 12)]
trailing_carries = player_df.tail(4)['carries'].sum()
trailing_rush_yards = player_df.tail(4)['rushing_yards'].sum()
trailing_ypc = trailing_rush_yards / trailing_carries
# Expected: ~4.72 YPC
```

**Test**: Add logging to print actual `trailing_yards_per_carry` value being passed to efficiency model

### Step 2: Inspect Model Predictions with Debug Logging

**Action**: Run efficiency model with test inputs

```python
import joblib
eff_model = joblib.load('data/models/efficiency_predictor_v2_defense.joblib')

# Test with Derrick Henry features
test_input = {
    'week': 12,
    'trailing_yards_per_carry': 4.72,
    'trailing_yards_per_target': 10.9,
    'trailing_td_rate_rush': 0.08,
    'trailing_td_rate_pass': 0.025,
    'opp_rush_def_epa': 0.0520,
    'opp_rush_def_rank': 25,  # Estimate
    'opp_pass_def_epa': 0.1459,
    'trailing_opp_rush_def_epa': 0.0,
    'team_pace': 60  # Estimate
}

prediction = eff_model.predict([test_input])
print(f'Predicted YPC: {prediction}')  # Should be ~4.5-5.5, NOT 2.61
```

### Step 3: Check for Sign Flip in Opponent DEF EPA

**File**: `nfl_quant/utils/epa_utils.py`

**Verify**:
```python
# Expected: Positive EPA = Defense allows MORE yards (weak)
# Expected model behavior: +0.0520 → INCREASE YPC projection

# Possible bug:
# If model trained with INVERTED EPA (negative = weak)
# Then +0.0520 would decrease projection (WRONG)
```

**Test**: Compare EPA signs in training data vs production data

### Step 4: Review Model Training Script

**File**: `scripts/train/train_efficiency_predictor_v2_with_defense.py`

**Check**:
1. Is `yards_per_carry` target calculated correctly? (should be `rushing_yards / carries`)
2. Are opponent DEF EPA features correctly signed?
3. Is regression to mean applied AFTER model prediction? (double regression bug)
4. Are training data filters excluding extreme performances? (causing low-mean bias)

---

## Immediate Recommendations

### 🔥 TIER 1 - Critical (Do This Week)

**1. Add Debug Logging to Efficiency Predictor**

**File**: `nfl_quant/models/efficiency_predictor.py`

Add logging in `predict()` method:
```python
def predict_efficiency(self, features: Dict) -> Dict:
    logger.info(f"EFFICIENCY DEBUG: Input features: {features}")

    # Make prediction
    ypc_prediction = self.models['yards_per_carry'].predict([features])[0]

    logger.info(f"EFFICIENCY DEBUG: YPC prediction: {ypc_prediction:.2f}")
    logger.info(f"  Input trailing_ypc: {features.get('trailing_yards_per_carry', 'N/A')}")
    logger.info(f"  Adjustment: {ypc_prediction - features.get('trailing_yards_per_carry', 0):.2f}")

    return ypc_prediction
```

**Run** for Derrick Henry Week 12 and capture logs

**2. Validate Trailing Feature Calculation**

**File**: `scripts/predict/generate_model_predictions.py`

Add logging around line 330-336:
```python
if position == 'RB' and 'carries' in player_df.columns:
    total_qb_carries = player_df['carries'].sum()
    if total_qb_carries > 0:
        if 'rushing_yards' in player_df.columns:
            trailing_yards_per_carry = player_df['rushing_yards'].sum() / total_qb_carries

            # DEBUG LOG
            logger.info(f"TRAILING YPC DEBUG for {player_name}:")
            logger.info(f"  Total carries: {total_qb_carries}")
            logger.info(f"  Total yards: {player_df['rushing_yards'].sum()}")
            logger.info(f"  Calculated YPC: {trailing_yards_per_carry:.2f}")
```

**3. Quick Fix: Apply Manual Efficiency Adjustment**

**Short-term workaround** while investigating root cause:

**File**: `scripts/predict/generate_model_predictions.py`

Add after efficiency prediction (around line 900):
```python
# TEMPORARY FIX: Models severely underestimate efficiency
# Apply 1.5x multiplier to YPC/Y/R predictions
if position == 'RB':
    predicted_ypc = predicted_ypc * 1.5  # Correct -33% underprediction
elif position in ['WR', 'TE']:
    predicted_yards_per_target = predicted_yards_per_target * 1.3  # Correct -23% underprediction

logger.warning(f"Applied efficiency correction multiplier for {player_name}")
```

**Note**: This is a HACK and should be removed once root cause is fixed

### ⭐ TIER 2 - Medium-Term (Next Week)

**4. Retrain Efficiency Models with 2025 Data**

**Issue**: Current models trained on 2022-2024 data (11/15/2025)

**Action**:
```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
.venv/bin/python scripts/train/train_efficiency_predictor_v2_with_defense.py --seasons 2023,2024,2025 --validate
```

**Expected Impact**: Models will learn 2025 offensive/defensive patterns

**5. Verify Opponent DEF EPA Sign Convention**

**File**: `nfl_quant/utils/epa_utils.py`

Add assertion tests:
```python
def test_epa_sign_convention():
    # Jets are weak vs run → should have POSITIVE rush_def_epa
    jets_epa = calculate_defensive_epa('NYJ', 'run', week=12, season=2025)
    assert jets_epa > 0, f"Expected positive EPA for weak Jets run D, got {jets_epa}"

    # Bills are strong vs run → should have NEGATIVE rush_def_epa
    bills_epa = calculate_defensive_epa('BUF', 'run', week=12, season=2025)
    assert bills_epa < 0, f"Expected negative EPA for strong Bills run D, got {bills_epa}"
```

**6. Add Model Prediction Bounds**

**File**: `nfl_quant/models/efficiency_predictor.py`

Add sanity checks:
```python
def predict_efficiency(self, features: Dict) -> float:
    prediction = self.model.predict([features])[0]

    # Sanity bounds: Prediction should not be below 60% of trailing average
    trailing_avg = features.get('trailing_yards_per_carry', 4.0)
    min_allowed = trailing_avg * 0.6
    max_allowed = trailing_avg * 1.4

    if prediction < min_allowed:
        logger.warning(f"Prediction {prediction:.2f} below min {min_allowed:.2f}, capping")
        prediction = min_allowed
    elif prediction > max_allowed:
        logger.warning(f"Prediction {prediction:.2f} above max {max_allowed:.2f}, capping")
        prediction = max_allowed

    return prediction
```

### 🚀 TIER 3 - Long-Term (Off-Season)

**7. Implement Ensemble Efficiency Models**

Combine multiple approaches:
- XGBoost (current)
- Linear regression (baseline)
- Simple regression to mean (fallback)

**8. Add Uncertainty Quantification**

Return prediction intervals, not just point estimates:
```python
predicted_ypc = 4.5  # Point estimate
ypc_std = 1.2  # Standard deviation
ypc_80pct_interval = (3.3, 5.7)
```

---

## Impact on Current Recommendations

### Mark Andrews Under 36.5 Receiving Yards

**Current Recommendation**:
- Edge: 22.3% (ELITE)
- Projected Y/R: 5.38
- Projected Receptions: 3.0
- Projected Total: 16.14 yards
- Line: 36.5 yards

**If Efficiency Corrected to 8.62 Y/R**:
- Projected Total: 3.0 × 8.62 = **25.86 yards**
- Still well UNDER 36.5 line
- Edge would INCREASE (even more confident UNDER)
- **Recommendation STILL VALID**

### Derrick Henry Under 86.5 Rushing Yards

**Current Recommendation**:
- Edge: 15.0% (HIGH)
- Projected YPC: 2.61
- Projected Attempts: 18.81
- Projected Total: 49.06 yards
- Line: 86.5 yards

**If Efficiency Corrected to 4.86 YPC**:
- Projected Total: 18.81 × 4.86 = **91.42 yards**
- NOW OVER 86.5 line (was UNDER)
- Pick flips from UNDER to OVER
- **Recommendation COMPLETELY REVERSED** 🚨

**This is a CRITICAL error** - the model bug caused a wrong recommendation

---

## Conclusion

### Summary of Findings

1. ✅ **EWMA Snap Share**: Working correctly (58.7% for Henry is mathematically valid)

2. 🚨 **Efficiency Projections**: SEVERELY BROKEN
   - Mark Andrews: -37.7% error (5.38 vs 8.62 Y/R)
   - Derrick Henry: -46.3% error (2.61 vs 4.86 YPC)
   - **Root cause**: Models predicting below their own input features
   - **Impact**: Derrick Henry recommendation is WRONG (should be OVER, not UNDER)

### Urgency Level

**CRITICAL** 🚨🚨🚨

This bug affects ALL rushing yards and receiving yards recommendations. The system cannot be trusted for Week 12 production use without fixing or applying the temporary multiplier workaround.

### Recommended Actions (Priority Order)

1. **IMMEDIATE** (Tonight): Add debug logging and validate trailing feature calculations
2. **URGENT** (This Weekend): Apply 1.5x efficiency multiplier as temporary fix OR remove all rushing/receiving yards picks
3. **THIS WEEK**: Retrain efficiency models with 2025 data
4. **NEXT WEEK**: Verify opponent DEF EPA signs and add prediction bounds
5. **OFF-SEASON**: Implement ensemble methods and uncertainty quantification

---

**Analysis Completed**: November 23, 2025
**Analyst**: Senior Sports Betting Analytics Engineer
**Status**: 🚨 CRITICAL BUG IDENTIFIED - System not production-ready for efficiency-based markets
