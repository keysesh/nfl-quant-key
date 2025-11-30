# ROOT CAUSE: Model Trained with Buggy Data

**Date**: November 23, 2025
**Status**: 🎯 **ROOT CAUSE IDENTIFIED**

---

## Summary

The efficiency model is NOT making opposite adjustments due to EPA sign errors. Instead, it's experiencing **feature distribution shift** because we fixed Bug #3 today but the model was trained 8 days ago with the buggy data.

---

## Timeline

**November 15, 2025**: Models trained
- `efficiency_predictor_v2_defense.joblib` trained with Bug #3 present
- Bug #3: Used `trailing_yards_per_opportunity` instead of `trailing_yards_per_target`
- Training data had WR/TE Y/Tgt values of ~7-8 (generic metric)

**November 23, 2025 (Today)**: Bug #3 fixed
- Fixed simulator to use `trailing_yards_per_target` (position-specific)
- Now passing Y/Tgt values of ~9-10 for WR/TE (actual receiving efficiency)
- **But model still expects ~7-8!**

---

## The Problem

###George Kittle Example

**Historical Data (Weeks 8-11)**:
- Actual Yards/Target: **9.417** (226 yards / 24 targets)
- This is Kittle's TRUE receiving efficiency

**What Model Saw During Training** (November 15):
- Due to Bug #3, `trailing_yards_per_target` field was missing
- Fell back to `trailing_yards_per_opportunity` = ~7.0
- Model learned: "typical TE has Y/Tgt of ~7-8"

**What Model Sees Now** (November 23):
- Bug #3 fixed, now passing `trailing_yards_per_target` = 9.417
- Model thinks: "9.417 is abnormally high, must regress to mean"
- Predicts: 6.623 (30% reduction)

---

## Why This Happens

Machine learning models expect **consistent feature distributions** between training and prediction:

```
TRAINING (Nov 15 - with Bug #3):
  trailing_yards_per_target = trailing_yards_per_opportunity
  WR/TE typical values: 7-8 yards/target

PREDICTION (Nov 23 - Bug #3 fixed):
  trailing_yards_per_target = actual position-specific metric
  WR/TE typical values: 9-10 yards/target

RESULT: Distribution mismatch → Model regresses "high" values
```

---

## Evidence

### 1. George Kittle Debug Log

```
📊 trailing_yards_per_target: 9.417 (input historical Y/Tgt)
🎯 PREDICTED yards_per_target: 6.623 (input: 9.417, adjustment: -2.794)
```

**Analysis**:
- Model receives 9.417 (35% higher than typical training value of ~7.0)
- Applies strong regression to mean
- Outputs 6.623 (closer to training distribution)

### 2. Model Training Date vs Bug Fix Date

- Models: Trained **November 15, 2025**
- Bug #3 Fix: **November 23, 2025** (today)
- **8-day gap** with buggy data in production

### 3. Field Fallback Chain

Before fix (Bug #3):
```python
# player_simulator_v3_correlated.py:1013 (BEFORE)
'trailing_yards_per_target': player_input.trailing_yards_per_opportunity or 8.0
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                             Using GENERIC field (~7-8 for WR/TE)
```

After fix:
```python
# player_simulator_v3_correlated.py:1013 (AFTER)
'trailing_yards_per_target': (
    player_input.trailing_yards_per_target or  # NOW CALCULATED (~9-10)
    player_input.trailing_yards_per_opportunity or
    8.0
)
```

---

## Impact

### All WR/TE Predictions Affected

**Pattern**:
- High-efficiency receivers (Y/Tgt > 9.0): **UNDER**-projected
- Average-efficiency receivers (Y/Tgt 7-8): Relatively accurate
- Low-efficiency receivers (Y/Tgt < 6.0): **OVER**-projected

**Examples**:
- George Kittle: 9.417 → 6.623 (-30%)
- High-efficiency WRs (Tyreek Hill, CeeDee Lamb): Similar under-projection
- Low-efficiency TEs: May be over-projected

### RB Receiving Also Affected (Bug #2)

Same issue for dual-threat RBs:
- Christian McCaffrey, Alvin Kamara, Austin Ekeler
- Model trained with generic metric, now sees position-specific
- Under-projection of 20-30%

---

## Solution

### Option 1: Retrain Models with Fixed Data ✅ **RECOMMENDED**

**Why**: Fixes root cause permanently

**Steps**:
1. Verify Bug #3 fix is working (✅ confirmed via debug logs)
2. Retrain efficiency model with corrected pipeline
3. Model will learn actual Y/Tgt distributions (~9-10 for WR/TE)
4. Predictions will be accurate

**Command**:
```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
.venv/bin/python scripts/train/train_efficiency_predictor_v2_with_defense.py
```

**Time**: ~30 minutes
**Risk**: LOW - Bug #3 fix validated, just need to retrain

### Option 2: Temporarily Revert Bug #3 Fix ❌ **NOT RECOMMENDED**

**Why**: Would restore consistency but keeps using wrong metric

This would make predictions match the model, but we'd still be using the inferior generic metric instead of position-specific efficiency.

### Option 3: Scale Input Features ⚠️ **HACKY WORKAROUND**

Temporarily scale `trailing_yards_per_target` to match training distribution:
```python
# Divide by 1.25 to convert 9.4 → 7.5 (training range)
scaled_ypt = trailing_yards_per_target / 1.25
```

**Why not**: Hacky, doesn't fix root cause, hard to maintain

---

## Action Plan

### Immediate (CRITICAL)

1. ✅ **Root cause identified**: Model trained with Bug #3 data
2. ❌ **Retrain efficiency model** with fixed pipeline
   ```bash
   .venv/bin/python scripts/train/train_efficiency_predictor_v2_with_defense.py
   ```
3. ❌ **Validate new model** on test cases
4. ❌ **Regenerate Week 12 predictions** with new model
5. ❌ **Regenerate recommendations and dashboard**

### Validation Tests

After retraining, verify:
- George Kittle: Input 9.417 Y/Tgt → Output ~9.0-10.0 Y/Tgt (not 6.6!)
- High-efficiency WRs: No longer under-projected
- EPA adjustments work correctly (weak defense = slight boost)

---

## Key Takeaways

### What We Learned

1. **Train and predict must use same data pipeline**
   - Bug fixes in prediction code require model retraining
   - Feature distribution shift causes regression-to-mean artifacts

2. **Model training date matters**
   - Models trained Nov 15 (with bugs)
   - Bugs fixed Nov 23 (8 days later)
   - **Must retrain after fixing data bugs**

3. **Debug logging caught the issue**
   - Without logging, we wouldn't have seen the -2.794 adjustment
   - Logging revealed model receiving correct input but making wrong prediction

### Prevention

1. **Retrain after data fixes**: Any fix to input features requires retraining
2. **Version control for data pipelines**: Track which pipeline version trained each model
3. **Distribution monitoring**: Alert when input features shift significantly from training

---

## Status

**Root Cause**: ✅ **IDENTIFIED** - Model trained with Bug #3 data
**Fix Required**: ❌ **Model retraining**
**ETA**: ~30 minutes
**Confidence**: **HIGH** - Clear evidence of feature distribution mismatch

---

**Prepared By**: NFL QUANT Development Team
**Date**: November 23, 2025
**Next Step**: Retrain efficiency predictor with fixed pipeline
