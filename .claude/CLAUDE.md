# NFL Quant - Claude Code Guide

## Project Overview

NFL player props prediction system using XGBoost classifiers with 39 features.
Predicts P(UNDER) for receptions, yards, and attempts markets.

**Current Model**: V23 (trained December 7, 2025)
**Active Model**: `data/models/active_model.joblib`

## Quick Start

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate
export PYTHONPATH="/Users/keyonnesession/Desktop/NFL QUANT:$PYTHONPATH"
```

## Key Commands

### Train/Retrain Model
```bash
python scripts/train/train_model.py
```

### Generate Predictions
```bash
python scripts/predict/generate_unified_recommendations_v3.py --week 14 --season 2025
```

### Run Walk-Forward Backtest
```bash
python scripts/backtest/walk_forward_unified.py --weeks 5-13
```

### Run Leakage-Free Backtest
```bash
python scripts/backtest/walk_forward_no_leakage.py --start-week 5 --end-week 13
```

---

## IMPORTANT: Pipeline Execution

When running `run_pipeline.py`:
1. ALWAYS use `timeout: 10m` with background execution
2. Do NOT read full output - use `tail -30` to check status
3. Redirect output to log file: `> logs/pipeline_$(date +%Y%m%d).log 2>&1`
4. Only report final summary, not intermediate progress

---

## Pipeline Architecture

```
TRAINING PIPELINE:
==================
combined_odds_actuals_ENRICHED.csv  +  weekly_stats.parquet
(odds, vegas, opponent, target_share)   (player stats, opponent_team)
              |                                    |
              v                                    v
       prepare_data_with_trailing() ─────────────────>
              |
              v
       extract_features_batch() ← nfl_quant/features/batch_extractor.py
              |                   (extracts 39 features per row)
              |                   (calls _add_opponent_features_vectorized)
              v
       XGBoost walk-forward training
              |
              v
       active_model.joblib (5 market classifiers)
```

---

## Critical Data Files

| File | Purpose | Required Columns |
|------|---------|------------------|
| `data/backtest/combined_odds_actuals_ENRICHED.csv` | Training data | opponent, vegas_total, vegas_spread, target_share_stats |
| `data/nflverse/weekly_stats.parquet` | Player stats + defense calc | opponent_team, passing_yards, rushing_yards, receptions |
| `data/models/active_model.joblib` | Trained model | N/A |

### Data Coverage Requirements

| Column | Minimum Coverage | Current |
|--------|------------------|---------|
| opponent | >90% | 92.6% |
| vegas_total | >70% | 76.0% |
| vegas_spread | >70% | 76.0% |
| target_share_stats | >80% | 90.0% |
| weekly_stats seasons | 2023, 2024, 2025 | All present |

---

## Feature Extraction (CRITICAL)

The model uses **39 features**. Key ones that MUST be populated:

| Feature | Source | If Broken |
|---------|--------|-----------|
| `target_share` | ENRICHED.csv → target_share_stats | Defaults to 0, loses #1 signal |
| `has_opponent_context` | weekly_stats.parquet → opponent defense | 0% coverage, loses defense info |
| `vegas_total` | ENRICHED.csv | Defaults to 44 |
| `vegas_spread` | ENRICHED.csv | Defaults to 0 |
| `opp_pass_yds_def_vs_avg` | calculated from weekly_stats | NaN if opponent column missing |

### Verifying Features Work

```python
import joblib
model = joblib.load('data/models/active_model.joblib')
importances = dict(zip(
    model['models']['player_receptions'].feature_names_in_,
    model['models']['player_receptions'].feature_importances_
))
# EXPECTED:
# target_share > 15%
# has_opponent_context > 5%
# vegas_spread > 1%
# vegas_total > 1%

for feat in ['target_share', 'has_opponent_context', 'vegas_spread', 'vegas_total']:
    print(f"{feat}: {importances.get(feat, 0):.1%}")
```

---

## Current Model Performance (V23)

| Market | Hit Rate (70% threshold) | ROI |
|--------|--------------------------|-----|
| player_receptions | 79.4% | +51.5% |
| player_reception_yds | 70.6% | +34.8% |
| player_rush_yds | 61.9% | +18.2% |
| player_pass_yds | 54.0% | +3.1% |
| player_rush_attempts | 53.3% | +1.7% |

---

## December 7, 2025 Fixes (DO NOT REVERT)

### Fix 1: Enriched Odds Data
**Problem**: vegas_total, vegas_spread, opponent, target_share were not in training data
**Solution**: Created `combined_odds_actuals_ENRICHED.csv` with these fields
**File**: `scripts/train/train_model.py` (load_data function uses enriched file)
**Impact**: target_share went from 0% to 17.4% feature importance

### Fix 2: Opponent Feature Merge Collision
**Problem**: `V23_OPPONENT_FEATURES` pre-initialized columns as NaN before merge, causing pandas `_x`/`_y` suffix collision. Result: 0% opponent coverage.
**Solution**: Removed pre-initialization, added column drop before merge
**File**: `nfl_quant/features/batch_extractor.py` (`_add_opponent_features_vectorized` function)
**Impact**: `has_opponent_context` went from 0% to 9.7% feature importance

### Fix 3: Column Name Mismatch
**Problem**: `V23_OPPONENT_FEATURES` expected `opp_pass_def_vs_avg` but cache produced `opp_pass_yds_def_vs_avg`
**Solution**: Updated `V23_OPPONENT_FEATURES` list to match actual column names
**File**: `nfl_quant/features/opponent_features.py`

### Fix 4: Missing 2023 Weekly Stats
**Problem**: `weekly_stats.parquet` only had 2024-2025, so 2023 training data had no opponent context
**Solution**: Downloaded 2023 data via `nfl_data_py` and merged into weekly_stats.parquet
**Impact**: Opponent coverage for all training years (2023, 2024, 2025)

### Fix 5: QB Passing Yards Drastically Underestimated (Dec 13, 2025)
**Problem**: QB passing yards predicted ~120 yards when actual averages are ~230 yards
**Root Cause**: `avg_pass_att` (passing attempts per game) was never calculated or passed to simulator
**Impact**: Simulator defaulted to 20 attempts (line 762 in player_simulator_v4.py), yielding ~104 yards
**Solution**:
- Added `avg_pass_att` EWMA calculation in `generate_model_predictions.py:492-498`
- Added `avg_pass_att` to trailing_stats dictionary (line 649-650)
- Added `avg_pass_att` to `PlayerPropInput` schema (schemas.py:277)
- Passed `avg_pass_att` to PlayerPropInput creation (line 1164)
**Files Modified**:
- `scripts/predict/generate_model_predictions.py`
- `nfl_quant/schemas.py`
**Verification**: C.J. Stroud now projects 30.4 attempts → ~213 yards (vs line 227.5)

### Fix 6: WR/TE/RB Usage Stats Key Mismatch & Missing Fields (Dec 13, 2025)
**Problem**: WR/TE/RB targets and RB carries using fallback estimation instead of actual EWMA data
**Root Cause**: Multiple issues:
1. `avg_rec_tgt` key mismatch: stored as `avg_targets_per_game` but extracted as `avg_rec_tgt`
2. `trailing_targets` never calculated or passed to simulator (WR/TE/RB)
3. `trailing_carries` never calculated or passed to simulator (RB)
4. `trailing_catch_rate` never calculated (falls back to position defaults)
**Impact**: Simulator fell back to less accurate estimation paths:
- WR/TE: Estimated targets from `target_share × team_pass_attempts` instead of actual EWMA
- RB: Estimated carries from `avg_rush_yd / ypc` instead of actual EWMA
- All: Used position defaults (WR: 65%, TE: 70%, RB: 77%) instead of player catch rates
**Solution**:
- Added `avg_rec_tgt` key alias in trailing_stats (line 652)
- Added `trailing_targets`, `trailing_carries` to trailing_stats (lines 653-654)
- Added `trailing_catch_rate` calculation (line 655)
- Added extraction of new fields (lines 905-907)
- Passed all fields to PlayerPropInput (lines 1176-1178)
**Files Modified**:
- `scripts/predict/generate_model_predictions.py`
**Verification**: Now uses direct EWMA values instead of derived estimates

### Fix 7: Comprehensive Simulator Input Audit (Dec 13, 2025)
**Problem**: Multiple fields expected by simulator were using defaults instead of actual data
**Root Cause**: Fields expected via `getattr()` were never calculated/passed to PlayerPropInput
**Fields Fixed**:
1. `opp_pass_def_epa` - Now calculates opponent pass defense EPA
2. `opp_pass_def_rank` - Now calculates pass defense rank (1-32)
3. `opp_rush_def_epa` - Now calculates opponent rush defense EPA
4. `opp_rush_def_rank` - Now calculates rush defense rank (1-32)
5. `adot` - Now calculates average depth of target (receiving_air_yards / targets)
6. `ypt_variance` - Now calculates yards per target variance (volatility measure)
**Solution**:
- Added adot/ypt_variance calculations in trailing stats (lines 500-518)
- Added fields to trailing_stats dictionary (lines 677-678)
- Added opp_pass/rush_def_epa and rank calculations (lines 1169-1182)
- Added new fields to PlayerPropInput schema (schemas.py lines 288-293)
- Extracted and passed all fields to PlayerPropInput (lines 932-934, 1223-1230)
**Files Modified**:
- `scripts/predict/generate_model_predictions.py`
- `nfl_quant/schemas.py`
**Verification**: All simulator getattr() calls now receive actual data instead of defaults

### Fix 8: Schedule Strength & Team Pace (Dec 13, 2025)
**Problem**: Simulator lacked schedule strength context and team-specific pace
**Root Cause**: `trailing_opp_pass_def_epa`, `trailing_opp_rush_def_epa`, and `team_pace` were defaulting
**Data Sources Used**:
1. `weekly_stats.parquet` has `opponent_team` for each player game → can calculate trailing opponent defense
2. `team_pace.parquet` has `plays_per_game` for each team → converted to seconds per play
**Solution**:
- Added trailing opponent defense EPA calculation (lines 520-559)
  - Loops through player's recent games, gets each opponent's defense EPA
  - Applies EWMA weighting (recent games weighted more)
- Added team_pace lookup from team_pace.parquet (lines 520-530)
  - Converts plays_per_game to seconds_per_play (3600 / plays_per_game)
- Added all fields to trailing_stats dictionary (lines 720-724)
- Added fields to PlayerPropInput schema (schemas.py lines 295-300)
- Extracted and passed to PlayerPropInput (lines 982-987, 1284-1288)
**Files Modified**:
- `scripts/predict/generate_model_predictions.py`
- `nfl_quant/schemas.py`

### Fix 9: slot_snap_pct from PBP Data (Dec 13, 2025)
**Problem**: `slot_snap_pct` (percentage of snaps from slot alignment) was using position defaults
**Root Cause**: Field was never calculated - simulator used defaults like WR=65%, TE=15%, RB=10%
**Data Source**: `pbp_{season}.parquet` has `pass_location` column (left/middle/right)
- `middle` = slot alignment (receiver lined up in slot)
- `left`/`right` = outside alignment (receiver on outside)
**Solution**:
- Pre-calculate slot_snap_pct cache before player loop from PBP data (lines 454-479)
  - Filter to pass plays with receiver before current week
  - Group by receiver, count middle vs total pass targets
  - Minimum 5 targets required for reliable estimate
- Add lookup from cache in player loop (line 690): `slot_snap_pct = slot_snap_pct_cache.get(player_name)`
- Add to trailing_stats dictionary (line 756)
- Extract from hist (line 1022)
- Pass to PlayerPropInput (line 1325)
**Files Modified**:
- `scripts/predict/generate_model_predictions.py`
**Verification**: WR/TE slot receivers now use actual slot percentages instead of position defaults

### Fix 10: player_pass_yds Re-enabled (Dec 14, 2025)
**Problem**: player_pass_yds was disabled with 43.2% hit rate in walk-forward validation
**Root Cause**: Walk-forward validation used simple 3-feature GradientBoostingClassifier, NOT the production XGBoost model
- Simple model: 3 features (line_vs_trailing, line, trailing_stat)
- Production model: 44 features (full V24 feature set)
**Discovery**: Production model actually achieves 78.3% hit rate on confident picks
**Solution**:
1. Re-enabled player_pass_yds in `configs/model_config.py`
   - Changed `enabled=False` to `enabled=True`
   - Upgraded tier from LOW to MEDIUM
   - Lowered confidence_threshold from 0.75 to 0.60
2. Updated walk-forward validation to use production model
   - Added `use_production_model=True` flag (default)
   - Added `load_production_model()` function
   - Added `prepare_odds_with_trailing()` for proper feature extraction
   - Added `get_production_model_prediction()` using batch_extractor
**Files Modified**:
- `configs/model_config.py` (lines 433-443)
- `scripts/backtest/walk_forward_unified.py` (multiple functions added)
**Verification**:
```python
# Production model hit rates for player_pass_yds:
# Confident UNDER (prob > 0.6): 78.3% hit rate (314 picks)
# Confident OVER (prob < 0.4): 65.7% hit rate (536 picks)
# Correlation: 0.407
```

### Fix 11: Test Suite Updates (Dec 14, 2025)
**Problem**: 3 tests failing due to outdated expectations
**Root Cause**: Tests expected difference-based line_vs_trailing, but code now uses percentage method
**Solution**:
- Updated `test_feature_engine.py`: Changed expected value from 1.5 to 0.3 (percentage method)
- Updated `test_v2_features.py`: Changed expected value to use percentage calculation
- Updated `test_dashboard_data_shape.py`: Updated REQUIRED_COLUMNS to match actual output
**Files Modified**:
- `tests/test_feature_engine.py`
- `tests/test_v2_features.py`
- `tests/test_dashboard_data_shape.py`
**Verification**: 94 tests passing, 3 skipped, 0 failures

### Fix 12: 13 Zero-Importance Features Fixed (Dec 14, 2025)
**Problem**: 13 features had 0% importance in all markets because they defaulted to constants
**Root Cause**: Features were configured in model_config.py but never actually calculated from source data
- Default functions in batch_extractor.py set constants (e.g., adot=8.5, pressure_rate=0.25)
- Source data (weekly_stats, participation, team_pace) was never being used
- Interaction terms (lvt_x_defense, lvt_x_rest) ran before their dependencies existed

**Features Fixed**:
1. `adot` - Now calculated from weekly_stats (receiving_air_yards / targets)
2. `trailing_catch_rate` - Now calculated from weekly_stats (receptions / targets)
3. `game_pace` - Now loaded from team_pace.parquet
4. `pressure_rate` - Now calculated from participation.was_pressure
5. `opp_pressure_rate` - Now calculated from participation (defense perspective)
6. `slot_snap_pct` - Now calculated from participation route data
7. `opp_wr1_receptions_allowed` - Now calculated from weekly_stats (WR1 vs opponent)
8. `opp_man_coverage_rate_trailing` - Now calculated from participation.defense_man_zone_type
9. `man_coverage_adjustment` - Now derived from coverage rates
10. `slot_funnel_score` - Now calculated from slot % and coverage
11. `lvt_x_defense` - Now runs AFTER opponent features are calculated
12. `lvt_x_rest` - Now has proper calculation (requires rest_days data)
13. `oline_health_score` - Now works for player_rush_yds market

**Solution**:
1. Created `nfl_quant/features/broken_feature_fixes.py` with actual calculations
2. Modified `nfl_quant/features/batch_extractor.py`:
   - V25 fix runs at Step 3 (BEFORE defaults)
   - Added `_calculate_interaction_terms()` at Step 7.5 (AFTER opponent features)
3. Reordered pipeline: fix → defaults → opponent features → interaction terms

**Files Modified**:
- `nfl_quant/features/broken_feature_fixes.py` (NEW)
- `nfl_quant/features/batch_extractor.py`

**Impact**:
- player_receptions: 11/13 features now have importance > 0.1%
  - `game_pace`: 0% → 3.80% (top 10!)
  - `slot_snap_pct`: 0% → 3.04% (top 10!)
  - `adot`: 0% → 2.52%
  - `lvt_x_defense`: 0% → 1.57%
- player_rush_yds: 7/13 features (appropriate for rush market)
  - `oline_health_score`: 0% → 3.08% (now working!)
  - `opp_pressure_rate`: 0% → 3.56% (top 10!)
  - `pressure_rate`: 0% → 3.18% (top 10!)

**Verification**:
```python
import joblib
m = joblib.load('data/models/active_model.joblib')
imp = dict(zip(m['models']['player_receptions'].feature_names_in_,
               m['models']['player_receptions'].feature_importances_))
for f in ['adot', 'game_pace', 'slot_snap_pct', 'lvt_x_defense']:
    print(f'{f}: {imp.get(f,0):.2%}')  # All should be > 0%
```

---

## Common Issues & Solutions

### Issue: "Opponent context coverage: 0.0%"
**Cause**: Merge collision in batch_extractor.py or missing opponent column
**Diagnosis**:
```python
import pandas as pd
enriched = pd.read_csv('data/backtest/combined_odds_actuals_ENRICHED.csv')
print(f"opponent coverage: {enriched['opponent'].notna().mean():.1%}")
```
**Fix**: Verify ENRICHED.csv has `opponent` column; check batch_extractor.py doesn't pre-initialize opponent columns

### Issue: Feature importance is 0% for target_share/vegas features
**Cause**: Features defaulting to constants during training
**Diagnosis**:
```python
import pandas as pd
enriched = pd.read_csv('data/backtest/combined_odds_actuals_ENRICHED.csv')
for col in ['target_share_stats', 'vegas_total', 'vegas_spread']:
    print(f"{col}: {enriched[col].nunique()} unique values")
```
**Fix**: Regenerate ENRICHED.csv with proper data

### Issue: 2023 data has no opponent context
**Cause**: weekly_stats.parquet missing 2023 data
**Fix**:
```python
import nfl_data_py as nfl
import pandas as pd
stats_2023 = nfl.import_weekly_data([2023])
existing = pd.read_parquet('data/nflverse/weekly_stats.parquet')
combined = pd.concat([stats_2023, existing]).drop_duplicates(subset=['player_id', 'season', 'week'])
combined.to_parquet('data/nflverse/weekly_stats.parquet', index=False)
```

---

## Model Retraining Checklist

### Before Retraining
- [ ] ENRICHED.csv has `opponent` column (>90% coverage)
- [ ] ENRICHED.csv has `vegas_total` (>70% non-null)
- [ ] ENRICHED.csv has `vegas_spread` (>70% non-null)
- [ ] ENRICHED.csv has `target_share_stats` (>80% non-null)
- [ ] weekly_stats.parquet has seasons 2023, 2024, 2025
- [ ] Clear caches before training:
  ```python
  from nfl_quant.features.batch_extractor import clear_caches
  clear_caches()
  ```

### After Retraining
- [ ] `target_share` importance > 10%
- [ ] `has_opponent_context` importance > 5%
- [ ] `vegas_spread` / `vegas_total` importance > 1%
- [ ] Walk-forward validation shows > 55% hit rate at 70% threshold
- [ ] No data leakage (model correlation < line correlation)

---

## Anti-Leakage Mechanisms

The pipeline uses three mechanisms to prevent data leakage:

1. **`shift(1)` in trailing stats** - Only uses prior weeks' player data
2. **`shift(1)` in opponent defense** - Only uses prior games' defense stats
3. **Walk-forward validation** - Trains on `week < test_week - 1`, tests on `week == test_week`

### Leakage Validation
```python
import pandas as pd
import numpy as np

results = pd.read_csv('data/backtest/unified_validation_results.csv')
valid = results.dropna(subset=['clf_prob_under', 'actual_stat', 'line'])

# Model should NOT beat line correlation
line_corr = np.corrcoef(valid['line'], valid['actual_stat'])[0, 1]
print(f"Line vs Actual: {line_corr:.3f}")  # Should be ~0.8+
# If model correlation is suspiciously high (>0.5), check for leakage
```

---

## Directory Structure

```
NFL QUANT/
├── configs/
│   └── model_config.py         # Model version, features, params
├── data/
│   ├── backtest/
│   │   └── combined_odds_actuals_ENRICHED.csv  # Training data
│   ├── models/
│   │   └── active_model.joblib                 # Production model
│   ├── nflverse/
│   │   └── weekly_stats.parquet               # Player stats (2023-2025)
│   ├── picks/                                  # Generated picks
│   └── recommendations/                        # Full recommendations
├── nfl_quant/
│   └── features/
│       ├── batch_extractor.py    # Feature extraction (39 features)
│       └── opponent_features.py  # Opponent defense calculations
├── scripts/
│   ├── train/
│   │   └── train_model.py        # Model training script
│   ├── predict/
│   │   └── generate_unified_recommendations_v3.py
│   └── backtest/
│       ├── walk_forward_unified.py
│       └── walk_forward_no_leakage.py
└── .claude/
    └── CLAUDE.md                 # This file
```

---

## Key Files Modified (December 2025)

These files contain critical fixes. Review carefully before modifying:

| File | Fix Applied |
|------|-------------|
| `scripts/train/train_model.py` | Uses ENRICHED.csv, handles 2023 team column |
| `nfl_quant/features/batch_extractor.py` | Fixed opponent merge collision |
| `nfl_quant/features/opponent_features.py` | Fixed V23_OPPONENT_FEATURES column names |
| `data/nflverse/weekly_stats.parquet` | Added 2023 data |

---

## Quick Debugging

### Check Model Health
```bash
python -c "
import joblib
m = joblib.load('data/models/active_model.joblib')
print(f'Version: {m[\"version\"]}')
print(f'Trained: {m[\"trained_date\"]}')
imp = dict(zip(m['models']['player_receptions'].feature_names_in_,
               m['models']['player_receptions'].feature_importances_))
for f in ['target_share', 'has_opponent_context']: print(f'{f}: {imp.get(f,0):.1%}')
"
```

### Check Data Health
```bash
python -c "
import pandas as pd
e = pd.read_csv('data/backtest/combined_odds_actuals_ENRICHED.csv')
w = pd.read_parquet('data/nflverse/weekly_stats.parquet')
print(f'Enriched rows: {len(e)}, opponent: {e[\"opponent\"].notna().mean():.1%}')
print(f'Weekly stats: {len(w)}, seasons: {sorted(w[\"season\"].unique())}')
"
```

---

## Architecture Documentation

### Key Architecture Files
- `ARCHITECTURE.md` - System overview and design decisions
- `docs/dependency-map.md` - Module dependency graph
- `docs/component-diagram.md` - ASCII component diagrams
- `docs/CODEBASE_SNAPSHOT.md` - Full codebase for AI context (generated)

### Generating Codebase Snapshot
```bash
# Generate helicopter view for architecture discussions
python scripts/generate_codebase_snapshot.py

# Output: docs/CODEBASE_SNAPSHOT.md (~100-200KB markdown file)
```

### Architecture Review Process
1. Before major changes, review `ARCHITECTURE.md`
2. Run snapshot generator for AI-assisted refactoring
3. Have AI analyze dependencies before adding new modules

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V24 | Dec 11, 2025 | Position-specific defensive matchup features |
| V23 | Dec 7, 2025 | Fixed opponent features, target_share, vegas context |
| V22 | Nov 2025 | EWMA integration, ensemble infrastructure |

**Last Updated**: December 13, 2025
