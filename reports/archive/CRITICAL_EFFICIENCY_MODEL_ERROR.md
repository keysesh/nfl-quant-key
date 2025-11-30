# CRITICAL: Efficiency Model Making Opposite Adjustments

**Date**: November 23, 2025
**Status**: 🚨 **CRITICAL BUG DISCOVERED**
**Priority**: BLOCKING - Must fix before using predictions

---

## 🚨 The Problem

The efficiency predictor is making adjustments in the **OPPOSITE direction** of what's expected:

### George Kittle Example (Week 12 vs CAR)

**Input Features**:
- Historical Yards/Target: **9.417** (4-week trailing average)
- Opponent DEF EPA: **+0.0686** (POSITIVE = WEAK defense)
- Expected: Kittle should be MORE efficient vs weak defense

**Model Output**:
- Predicted Yards/Target: **6.623**
- Adjustment: **-2.794** yards/target (**-30% reduction**)
- 🚨 **WRONG DIRECTION**: Model decreased efficiency when facing WEAK defense!

**Debug Log**:
```
🔍 EFFICIENCY DEBUG [George Kittle] Input Features:
   📊 trailing_yards_per_target: 9.417 (input historical Y/Tgt)
   🛡️  opp_pass_def_epa: +0.0686 (POSITIVE = WEAK defense)
   🎯 PREDICTED yards_per_target: 6.623 (input: 9.417, adjustment: -2.794)
   🚨 WARNING: Large NEGATIVE adjustment (-2.794) for yards_per_target!
```

---

## Root Cause Analysis

### Hypothesis 1: EPA Sign Convention Error ⚠️ **MOST LIKELY**

**Problem**: The efficiency model may have been trained with **reversed EPA signs**.

**EPA Convention (Correct)**:
- **Positive EPA** = Weak defense (allows MORE yards) → Should INCREASE efficiency
- **Negative EPA** = Strong defense (prevents yards) → Should DECREASE efficiency

**What the model is doing**:
- Positive EPA (+0.0686) → DECREASES efficiency by 30%
- This suggests model thinks positive EPA = STRONG defense (WRONG!)

**How this happened**:
1. Model was trained on historical data with EPA features
2. EPA sign convention may have been flipped during training
3. Model learned: "higher EPA = decrease yards" (backwards!)

### Hypothesis 2: Feature Scaling Issue

**Problem**: EPA values not properly scaled, causing extreme adjustments.

**Evidence**:
- -2.794 yards/target adjustment is HUGE (30% reduction)
- Model may be treating EPA as raw value instead of normalized

### Hypothesis 3: Model Trained with Wrong Field

**Problem**: Model trained with `trailing_yards_per_opportunity` but now receiving `trailing_yards_per_target`.

**Evidence**:
- We just fixed Bug #3 to use `trailing_yards_per_target`
- But efficiency model was trained BEFORE the fix
- Model expects ~7.0 (generic) but receives 9.417 (position-specific)
- Model sees 9.417 as "too high" and reduces it

---

## Impact Assessment

### Financial Impact: **CRITICAL**

If the model is systematically making opposite adjustments:
- All "good matchups" (weak defenses) get UNDER-projected
- All "bad matchups" (strong defenses) get OVER-projected
- **Complete inversion of betting edge**

**Estimated Loss**:
- Previous bugs: $27,285/season
- This bug: **UNKNOWN - potentially worse** (affects ALL predictions)

### Affected Markets: **ALL RECEIVING/RUSHING YARDS**

- ❌ Player Receiving Yards (WR/TE)
- ❌ Player Rushing Yards (RB)
- ❌ Player Receptions (indirectly)
- ❌ Player TDs (uses efficiency model)

---

## Validation Tests

### Test 1: Check Other Players

Let's verify if this pattern holds across multiple players:

**Expected Pattern (if bug exists)**:
- Players vs WEAK defenses (positive EPA) → Under-projected
- Players vs STRONG defenses (negative EPA) → Over-projected

### Test 2: Check Historical Training Data

Look at the efficiency model training script to see how EPA was encoded.

**Files to check**:
- `scripts/train/train_efficiency_predictor_v2_with_defense.py`
- Look for EPA sign convention
- Check if EPA was multiplied by -1 anywhere

### Test 3: Test Extreme Cases

Create test cases with:
- Very weak defense (EPA = +0.20) → Should boost efficiency
- Very strong defense (EPA = -0.20) → Should reduce efficiency

---

## Immediate Actions Required

### 1. Verify EPA Sign Convention ⚠️ **DO FIRST**

**Check efficiency model training code**:
```bash
grep -n "def_epa" scripts/train/train_efficiency_predictor_v2_with_defense.py
```

Look for:
- How EPA features are created
- Any sign flips (multiply by -1)
- Feature normalization

### 2. Test Multiple Players

Run efficiency tests on:
- Player vs weak defense (positive EPA)
- Player vs strong defense (negative EPA)
- Verify adjustments go in correct direction

### 3. Check Model Training Data

**Load the training data and check EPA values**:
```python
import pandas as pd
training_data = pd.read_parquet('data/training/efficiency_training.parquet')
print(training_data[['opp_pass_def_epa', 'yards_per_target']].describe())
```

Look for correlation:
- Should be POSITIVE (higher EPA = more yards)
- If NEGATIVE, EPA signs were flipped

---

## Potential Fixes

### Fix Option 1: Flip EPA Sign in Predictions (Quick)

If model was trained with reversed EPA:
```python
# In generate_model_predictions.py, before passing to model:
opponent_def_epa_vs_position = -1 * original_epa  # Flip sign
```

**Pros**: Quick fix, no retraining needed
**Cons**: Hacky, doesn't fix root cause

### Fix Option 2: Retrain Efficiency Model (Correct)

Retrain model with correct EPA sign convention:
1. Verify EPA signs in training data
2. Fix any sign flips in training script
3. Retrain model from scratch
4. Validate on historical data

**Pros**: Fixes root cause
**Cons**: Takes time, requires validation

### Fix Option 3: Adjust for New Field (If Hypothesis 3)

If model was trained with `trailing_yards_per_opportunity`:
1. Temporarily revert to using generic field for efficiency model
2. Retrain model with position-specific fields
3. Re-deploy with new model

---

## Files to Investigate

1. **Model Training Script**:
   - `scripts/train/train_efficiency_predictor_v2_with_defense.py`
   - Check EPA feature creation

2. **Efficiency Predictor**:
   - `nfl_quant/models/efficiency_predictor.py`
   - Check prediction logic

3. **EPA Utilities**:
   - `nfl_quant/utils/epa_utils.py`
   - Verify EPA calculation

4. **Training Data**:
   - `data/training/*.parquet`
   - Check EPA value distributions

---

## Next Steps

**IMMEDIATE (Block all predictions until fixed)**:
1. ✅ Document the issue (this file)
2. ❌ **CRITICAL**: Investigate EPA sign convention in training code
3. ❌ **CRITICAL**: Test multiple players to confirm pattern
4. ❌ **CRITICAL**: Determine root cause (EPA flip vs field mismatch)
5. ❌ **CRITICAL**: Apply appropriate fix
6. ❌ **CRITICAL**: Regenerate ALL Week 12 predictions

**DO NOT USE CURRENT PREDICTIONS** until this is resolved!

---

## Status

**Current Predictions**: ⚠️ **UNRELIABLE**
**Bug Fixes (1-5)**: ✅ Validated (fields used correctly)
**Efficiency Model**: 🚨 **MAKING OPPOSITE ADJUSTMENTS**

**Confidence**: **BLOCKED** - Cannot proceed until efficiency model fixed

---

**Prepared By**: NFL QUANT Development Team
**Date**: November 23, 2025
**Urgency**: 🚨 CRITICAL - Blocking issue
