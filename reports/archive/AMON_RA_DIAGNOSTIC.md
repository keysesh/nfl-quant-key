# Amon-Ra St. Brown - Projection Diagnostic

**Date**: November 23, 2025
**Week**: 12
**Status**: ⚠️ **MODEL PROJECTIONS SUSPICIOUSLY LOW**

---

## Summary

You were absolutely right to question Amon-Ra St. Brown's projections. While the **math is 100% correct**, the **model inputs are wrong** - the system is projecting only **3.0 targets** for him, leading to absurdly low receptions/yards projections.

---

## Model Projections

| Stat | Model Projection | Season Avg | % of Season Avg |
|------|-----------------|------------|-----------------|
| **Targets** | 3.0 | ~9-10 | **30-33%** ❌ |
| **Receptions** | 1.95 | ~6 | **33%** ❌ |
| **Receiving Yards** | 31.5 | ~61 | **52%** ❌ |

**Conclusion**: Model is projecting **30-50% of Amon-Ra's normal usage**.

---

## Mathematical Validation

### Edge Calculations: ✅ CORRECT

**Receiving Yards (UNDER 80.5)**:
```
Model Mean: 31.47 yards
Model Std: 29.34 yards
Z-score: (80.5 - 31.47) / 29.34 = 1.67

P(Yards < 80.5) = 95.27% (raw)
Calibrated (30% shrinkage): 81.69%
Market probability: 50.21%
Edge: 31.48% ✅

Verification: (0.8169 - 0.5021) × 100 = 31.48% ✅
```

**Receptions (UNDER 7.5)**:
```
Model Mean: 1.95 receptions
Model Std: 1.64 receptions
Z-score: (7.5 - 1.95) / 1.64 = 3.39

P(Recs < 7.5) = 99.91% (raw)
Calibrated: 84.93%
Market probability: 54.67%
Edge: 30.26% ✅

Verification: (0.8493 - 0.5467) × 100 = 30.26% ✅
```

**Math Verdict**: ✅ All calculations are **mathematically perfect**. The probabilities, calibration, and edge calculations are correct given the model projections.

---

## Root Cause Analysis

### Problem: Model Projects Only 3.0 Targets

**Expected**: ~9-10 targets/game (Amon-Ra is an elite WR1)
**Actual Projection**: 3.0 targets

**Impact Chain**:
1. Model projects 3.0 targets
2. Receptions = binomial(3.0, 0.65 catch_rate) = 1.95 receptions
3. Yards = 1.95 rec × ~16 yards/rec = 31.5 yards
4. This is 30-50% of his normal production

### Possible Causes

#### 1. **Data Quality Issue** (Most Likely)
- Trailing stats not loaded correctly
- Player name mismatch in data pipeline
- Missing games in lookback window
- Incorrect team or position filter

#### 2. **Target Share Calculation Error**
- `trailing_target_share` may be calculated incorrectly
- Team pass attempts may be projected too low
- Usage model may have an issue

#### 3. **Model Issue**
- Usage predictor (targets model) predicting incorrectly
- Model not seeing Amon-Ra's historical elite usage
- Defensive adjustment too severe

#### 4. **Missing Context** (Less Likely)
- Injury we're not aware of
- Role change expected (unlikely for WR1)
- Extreme defensive matchup (NYG defense is not elite)

---

## Other Players with Same Issue

Checking other WRs with exactly **1.95 receptions**:

```
A.J. Brown: 1.95 receptions (should be ~6-7)
Amon-Ra St. Brown: 1.95 receptions (should be ~6)
Alec Pierce: 1.95 receptions (normal for WR2/3)
Alex Bachman: 1.95 receptions (backup - reasonable)
Allen Lazard: 2.6 receptions (WR2 - reasonable)
```

**Pattern**: Many WRs have **identical 1.95 receptions**, suggesting a **position-average fallback** is being used when player-specific data is missing.

---

## Diagnostic Steps Needed

### Immediate Checks

1. **Verify trailing stats loaded**:
   ```bash
   # Check if Amon-Ra has trailing data
   grep "amon-ra st. brown" data/nflverse/weekly_stats.parquet
   ```

2. **Check target share calculation**:
   ```python
   # What is Amon-Ra's calculated trailing_target_share?
   # Should be ~0.25-0.30 (25-30% of team targets)
   ```

3. **Check team pass attempts**:
   ```python
   # What are Lions projected for?
   # Should be ~30-35 attempts
   # 0.25 × 32 attempts = 8 targets (reasonable)
   # But model shows 3.0 targets (problem!)
   ```

4. **Check usage model input**:
   ```python
   # What features are being passed to usage_predictor?
   # Are trailing stats present?
   ```

### Recommended Fix

**Option 1: Regenerate predictions with corrected data pipeline**
```bash
# Re-run prediction script with verbose logging
python scripts/predict/generate_model_predictions.py 12 --debug
```

**Option 2: Manual override for known elite players**
- Not recommended (violates no-hardcoding rule)
- But could add sanity checks:
  ```python
  # If player is top-10 WR and projected < 5 targets, flag for review
  if is_elite_wr(player) and projected_targets < 5.0:
      logger.warning(f"Elite WR {player} projected unusually low targets: {projected_targets}")
  ```

**Option 3: Investigate usage_predictor model**
- Check what features it's using
- Verify it has access to trailing stats
- Check if there's a data mismatch

---

## Recommendation Impact

### For Betting

**Current Recommendations**:
- Amon-Ra UNDER 80.5 yards (31.5% edge)
- Amon-Ra UNDER 7.5 receptions (30.3% edge)

**Should we trust these?** ⚠️ **NO**

**Reasoning**:
1. The projections are based on **faulty inputs** (only 3.0 targets)
2. Even though the math is correct, **garbage in = garbage out**
3. Real-world expectation: Amon-Ra will likely get 8-10 targets, 5-7 receptions, 60-80 yards
4. These lines (U80.5, U7.5) would likely **LOSE** in reality

**Action**: **Remove Amon-Ra recommendations from bet card** until data issue is resolved.

---

## Broader Implications

### How Many Players Are Affected?

**Quick check**: Count players with identical 1.95 receptions:
```bash
grep "WR" model_predictions_week12.csv | awk -F',' '{print $10}' | grep "1.9500" | wc -l
```

If many WRs have 1.95 receptions, this is a **systematic data pipeline issue**, not just Amon-Ra.

### System-Wide Audit Needed

1. ✅ Math/calibration: CORRECT
2. ❌ Data pipeline: **NEEDS INVESTIGATION**
3. ❌ Trailing stats loading: **VERIFY**
4. ❌ Usage model inputs: **VERIFY**

---

## Conclusion

**What's Correct**:
- ✅ Edge calculation logic
- ✅ Calibration (30% shrinkage)
- ✅ Probability calculations
- ✅ No hardcoded CV values

**What's Wrong**:
- ❌ Model is projecting only 3.0 targets for Amon-Ra (should be ~9-10)
- ❌ This causes absurdly low receptions (1.95) and yards (31.5) projections
- ❌ Multiple WRs show identical 1.95 receptions (position-average fallback)
- ❌ Recommendations based on faulty data would likely lose

**Next Steps**:
1. Investigate data pipeline for trailing stats
2. Verify target share calculations
3. Check if usage_predictor is receiving correct inputs
4. Re-run predictions after fixing data issue
5. **DO NOT bet Amon-Ra recommendations** until verified

---

**Diagnostic By**: NFL QUANT Framework Validation
**Date**: November 23, 2025
**Status**: 🔴 **DATA PIPELINE ISSUE - DO NOT USE CURRENT AMON-RA PROJECTIONS**
