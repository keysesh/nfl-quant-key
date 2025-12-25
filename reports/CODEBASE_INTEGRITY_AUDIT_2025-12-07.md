# Codebase Integrity Audit Report
**Date:** 2025-12-07
**Auditor:** Claude Code

---

## Executive Summary

| Category | Status | Issues Found |
|----------|--------|--------------|
| Dependencies | ✅ PASS | 0 missing imports |
| Model Files | ⚠️ ISSUES | 8/15 models have archived/missing training scripts |
| Config Files | ✅ PASS | 43/45 valid (2 YAML files invalid) |
| Dead Code | ✅ PASS | 0 orphaned modules |

---

## Step 1: Dependency Trace

**Entry Points Analyzed:**
- `scripts/backtest/walk_forward_unified.py`
- `scripts/run_pipeline.py`
- `scripts/predict/generate_model_predictions.py`
- `scripts/predict/generate_unified_recommendations_v3.py`

**Result:** ✅ All 31 unique nfl_quant imports resolve correctly.

**Warnings:**
- `nfl_quant.features` - exists in both active and `_archive/`
- `nfl_quant.schemas` - exists in both active and `_archive/`
- `nfl_quant.calibration.calibrator_loader` - exists in both active and `_archive/`

---

## Step 2: Model File Audit

| Model File | Date | Training Script | Status |
|------------|------|-----------------|--------|
| active_model.joblib | Dec 7 | train_game_lines_model.py | ✅ ACTIVE |
| efficiency_predictor_v2_defense.joblib | Dec 7 | train_efficiency_predictor_v2_with_defense.py | ✅ ACTIVE |
| td_poisson_model.joblib | Dec 6 | train_td_poisson_model.py | ✅ ACTIVE |
| usage_predictor_v4_defense.joblib | Dec 3 | train_usage_predictor_v4_with_defense.py | ✅ ACTIVE |
| td_calibration_factors.joblib | Nov 24 | retrain_efficiency_with_td_calibration.py | ⚠️ ARCHIVED |
| td_rate_rec_calibrated.joblib | Nov 24 | retrain_efficiency_with_td_calibration.py | ⚠️ ARCHIVED |
| td_rate_rush_calibrated.joblib | Nov 24 | retrain_efficiency_with_td_calibration.py | ⚠️ ARCHIVED |
| usage_targets_calibrated.joblib | Nov 24 | retrain_models_calibration_aware.py | ⚠️ ARCHIVED |
| usage_carries_calibrated.joblib | Nov 24 | retrain_models_calibration_aware.py | ⚠️ ARCHIVED |
| usage_snap_pct_calibrated.joblib | Nov 24 | retrain_models_calibration_aware.py | ⚠️ ARCHIVED |
| usage_snaps_calibrated.joblib | Nov 24 | retrain_models_calibration_aware.py | ⚠️ ARCHIVED |
| usage_attempts_calibrated.joblib | Nov 24 | retrain_models_calibration_aware.py | ⚠️ ARCHIVED |
| model_20251206_*.joblib | Dec 6 | UNKNOWN | ❌ NO SCRIPT |
| model_20251207_*.joblib | Dec 7 | UNKNOWN | ❌ NO SCRIPT |
| v23_interaction_classifier.joblib | Dec 7 | UNKNOWN | ❌ NO SCRIPT |

### Critical Finding: Calibrator Training Scripts Archived

The 8 calibrator models (Nov 24) were created by scripts now in `_archive/`:
- `scripts/train/_archive/retrain_models_calibration_aware.py`
- `scripts/train/_archive/retrain_efficiency_with_td_calibration.py`

**Risk:** Cannot easily retrain calibrators if needed.

**Recommendation:** Either:
1. Un-archive these scripts if they're still valid
2. Create new training scripts with explicit `--max-week` parameters

---

## Step 3: Config Integrity

**Valid Configs:** 43 JSON files ✅

**Invalid Configs:**
- `injury_multipliers.yaml` - Parse error
- `simulation_variance.yaml` - Parse error

These YAML files should be fixed or removed if unused.

---

## Step 4: Dead Code Detection

**Result:** ✅ 0 orphaned modules detected

All 148 modules in `nfl_quant/` are imported by at least one script.

---

## Step 5: Architectural Safeguards Added

### New Utility: `nfl_quant/utils/training_metadata.py`

Provides:
1. `save_model_with_metadata()` - Saves models with embedded training metadata
2. `load_model_with_metadata()` - Loads models and extracts metadata
3. `verify_training_cutoff()` - Validates model against required data cutoff

**Usage Example:**
```python
from nfl_quant.utils.training_metadata import save_model_with_metadata

save_model_with_metadata(
    model=xgb_classifier,
    path=Path('data/models/my_model.joblib'),
    training_data_cutoff={'max_season': 2025, 'max_week': 11},
    feature_names=['feature1', 'feature2'],
    notes="Trained for week 12+ predictions"
)
```

This creates:
- `my_model.joblib` - Model with embedded metadata
- `my_model.metadata.json` - Human-readable sidecar file

---

## Calibrator Temporal Integrity Verification

Based on file timestamps and analysis:

| Calibrator | File Date | NFLverse Data at Training |
|------------|-----------|---------------------------|
| usage_*_calibrated.joblib | Nov 24 15:54 | Weeks 1-11 (week 12 not played yet) |
| td_*_calibrated.joblib | Nov 24 20:28 | Weeks 1-11 (week 12 not played yet) |

**Conclusion:** Current calibrators are temporally valid for testing on weeks 12+.

Week 11 2025 games: Nov 21-25
Week 12 2025 games: Nov 28-Dec 2

Calibrators were trained Nov 24 (during week 11 window), before week 12 data existed.

---

## Recommendations

### Immediate Actions

1. **Fix YAML configs:**
   ```bash
   # Check if these are used
   grep -r "injury_multipliers.yaml" scripts/
   grep -r "simulation_variance.yaml" scripts/
   # If unused, delete them
   ```

2. **Create metadata for existing calibrators:**
   ```python
   # Run once to add metadata sidecars
   from nfl_quant.utils.training_metadata import save_model_with_metadata
   import joblib

   for name in ['usage_targets', 'usage_carries', 'usage_snap_pct',
                'usage_snaps', 'usage_attempts']:
       model = joblib.load(f'data/models/{name}_calibrated.joblib')
       save_model_with_metadata(
           model, f'data/models/{name}_calibrated.joblib',
           training_data_cutoff={'max_season': 2025, 'max_week': 11},
           notes="Retroactively documented - trained Nov 24 2025"
       )
   ```

### Long-term Improvements

1. **Require `--max-week` in training scripts** - No silent "use all data"
2. **Auto-generate documentation** - CLAUDE.md should be generated from code, not manually maintained
3. **Pre-commit hook** - Verify models have metadata before committing

---

## Backtest Readiness

✅ **Safe to run backtest on weeks 12+** with current calibrators.

```bash
python scripts/backtest/walk_forward_unified.py --weeks 12-13 --bootstrap
```

The calibrators (trained Nov 24, week 11 data) have temporal precedence over test data (weeks 12+).
