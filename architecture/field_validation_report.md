# NFLverse Field Validation Report

**Generated**: 2025-12-14
**Project**: NFL QUANT
**Auditor**: NFL Data Architect (automated)

---

## Summary

| Metric | Value |
|--------|-------|
| Files scanned | 85+ Python files |
| Critical issues found | 3 (FIXED) |
| Warnings | 8 |
| Field references validated | 700+ (via data dictionary) |

---

## Critical Issues (FIXED)

All critical issues have been resolved during this audit.

### Issue 1: Stale PBP Data Path - contextual_integration.py
- **File**: `nfl_quant/utils/contextual_integration.py:58`
- **Previous**: Hardcoded `data/processed/pbp_{season}.parquet`
- **Problem**: Same bug as EPA issue - processed file may be stale
- **Fix Applied**: Cascading path lookup (nflverse → nflverse_season → processed)
- **Status**: FIXED

### Issue 2: Stale PBP Data Path - contextual_factors_integration.py
- **File**: `nfl_quant/utils/contextual_factors_integration.py:54`
- **Previous**: Hardcoded `data/processed/pbp_{season}.parquet`
- **Fix Applied**: Cascading path lookup
- **Status**: FIXED

### Issue 3: Stale PBP Data Path - unified_integration.py
- **File**: `nfl_quant/utils/unified_integration.py:1152`
- **Previous**: Hardcoded `data/processed/pbp_{season}.parquet`
- **Fix Applied**: Cascading path lookup
- **Status**: FIXED

---

## Warnings

These are potential issues that should be monitored but don't require immediate action.

### W1: fillna(0) Usage in Model Features
- **Location**: Multiple files in `nfl_quant/features/`
- **Count**: 35+ instances
- **Risk**: May mask data quality issues or introduce bias
- **Mitigation**: Most are for specific stat columns (yards, TDs) where 0 is semantically correct. Model features should use `feature_defaults.py` defaults.
- **Recommendation**: Review `nfl_quant/features/feature_defaults.py` for proper semantic defaults

### W2: player_id vs gsis_id Ambiguity (RESOLVED)
- **Location**: 50+ files
- **Issue**: `player_id` is used inconsistently - sometimes NFLverse player_id, sometimes custom
- **Mitigation**: NFLverse weekly_stats uses `player_id` which IS the gsis_id
- **Resolution**: Documented in CLAUDE.md under "NFLverse ID Conventions" section
- **Status**: RESOLVED

### W3: Test Files Using Stale Data Paths (RESOLVED)
- **Files**:
  - `tests/test_integrated_td_features.py`
  - `tests/test_td_predictor.py`
  - `tests/test_game_script.py`
  - `tests/test_snap_share.py`
- **Issue**: Tests hardcoded `data/processed/pbp_2025.parquet`
- **Resolution**: Added `get_pbp_path()` helper function with cascading lookup
- **Status**: RESOLVED

### W4: Archive Scripts With Stale Paths
- **Location**: `scripts/_archive/` directory
- **Count**: 10+ files
- **Risk**: VERY LOW - Archive files not in production use
- **Recommendation**: No action needed (archived code)

### W5: Fields Requiring Participation Data
- **Fields**: `was_pressure`, `route`, `time_to_throw`
- **Location**: `nfl_quant/features/broken_feature_fixes.py`
- **Status**: Correctly implemented - loads from `participation.parquet`
- **Recommendation**: No action needed

### W6: Potential Type Mismatch - game_id
- **Pattern searched**: `int(.*game_id`
- **Instances found**: 0 in core code
- **Note**: `game_id` is correctly used as string throughout codebase
- **Status**: OK

### W7: Week Filtering Edge Cases
- **Pattern searched**: `week.*<=.*current_week\s*-\s*\d`
- **Instances found**: 0
- **Note**: Codebase correctly uses `between()` or explicit range for week filtering
- **Status**: OK

### W8: Fallback Default Values
- **Location**: `nfl_quant/features/feature_defaults.py`
- **Note**: Contains comprehensive semantic defaults for all 46 features
- **Status**: OK - properly documented

---

## Field Usage Analysis

### Correctly Implemented Fields

| Field | Dataset | Usage Location | Status |
|-------|---------|----------------|--------|
| `epa` | PBP | defensive_stats_integration.py | OK + regression to mean |
| `play_type` | PBP | Multiple files | OK |
| `defteam` | PBP | defensive_stats_integration.py | OK |
| `posteam` | PBP | Multiple files | OK |
| `game_id` | All | Multiple files | OK (string) |
| `week` | All | Multiple files | OK (integer) |
| `season` | All | Multiple files | OK |
| `player_id` | Weekly Stats | core.py, batch_extractor.py | OK |
| `position` | Rosters | Multiple files | OK |
| `was_pressure` | Participation | broken_feature_fixes.py | OK |
| `route` | Participation | broken_feature_fixes.py | OK |

### Calculated Fields Status

| Field | Source | Calculation | Regression Applied |
|-------|--------|-------------|-------------------|
| `def_epa_allowed` | PBP | `epa.mean()` by defteam | YES |
| `pass_def_epa` | PBP | `epa.mean()` where play_type='pass' | YES |
| `rush_def_epa` | PBP | `epa.mean()` where play_type='run' | YES |
| `success` | PBP | `epa > 0` | N/A (binary) |
| `slot_snap_pct` | PBP | `pass_location='middle'` / total | N/A |
| `adot` | Weekly Stats | `receiving_air_yards / targets` | N/A |
| `catch_rate` | Weekly Stats | `receptions / targets` | N/A |
| `target_share` | Weekly Stats | `targets / team_targets` | N/A |

---

## Data Path Priority (Standard)

All PBP data access should use this cascading lookup:

```python
# CORRECT PATTERN
pbp_path = Path('data/nflverse/pbp.parquet')           # Fresh (updated daily)
if not pbp_path.exists():
    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')  # Season-specific
if not pbp_path.exists():
    pbp_path = Path(f'data/processed/pbp_{season}.parquet') # Processed (may be stale)
```

**Files using correct pattern**:
- `nfl_quant/utils/defensive_stats_integration.py` (after fix)
- `nfl_quant/utils/contextual_integration.py` (after fix)
- `nfl_quant/utils/contextual_factors_integration.py` (after fix)
- `nfl_quant/utils/unified_integration.py` (after fix)

---

## EPA Regression to Mean Implementation

The codebase correctly implements EPA regression to account for sample size variance:

**Central utility**: `nfl_quant/utils/epa_utils.py`

```python
def regress_epa_to_mean(raw_epa, sample_size, league_mean=0.0, regression_factor=0.5):
    base_samples = 10
    regression_weight = regression_factor * (base_samples / max(sample_size, 1))
    regression_weight = min(regression_weight, 0.75)  # Cap at 75%
    return raw_epa * (1 - regression_weight) + league_mean * regression_weight
```

**Usage locations**:
- `defensive_stats_integration.py` - Defensive EPA calculations
- `defensive_metrics.py` - Team defense metrics
- `epa_utils.py` - All team EPA calculations

---

## Recommendations

### High Priority
1. **Monitor for zeros** - Run diagnostic check before each prediction run:
   ```python
   from architecture.claude import diagnose_nfl_data
   diagnose_nfl_data(predictions_df, 'opponent_def_epa_vs_position')
   ```

### Medium Priority
2. **Document player_id convention** - Add to CLAUDE.md that NFLverse `player_id` = `gsis_id`
3. **Update test files** - Migrate tests to use cascading path lookup

### Low Priority
4. **Clean archive** - Consider removing or updating `scripts/_archive/` stale references

---

## Verification Commands

### Check Data Freshness
```bash
python -c "
import pandas as pd
from pathlib import Path

for p in ['data/nflverse/pbp.parquet', 'data/processed/pbp_2025.parquet']:
    path = Path(p)
    if path.exists():
        df = pd.read_parquet(path)
        print(f'{p}: weeks {sorted(df[\"week\"].unique())}')
    else:
        print(f'{p}: NOT FOUND')
"
```

### Check EPA Coverage
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/nflverse/pbp.parquet')
print(f'Total plays: {len(df)}')
print(f'EPA coverage: {df[\"epa\"].notna().mean():.1%}')
print(f'Weeks: {sorted(df[\"week\"].unique())}')
"
```

---

## Changelog

| Date | Action | Files Modified |
|------|--------|----------------|
| 2025-12-14 | Fixed stale PBP path | contextual_integration.py |
| 2025-12-14 | Fixed stale PBP path | contextual_factors_integration.py |
| 2025-12-14 | Fixed stale PBP path | unified_integration.py |
| 2025-12-14 | Created validation report | This file |
| 2025-12-14 | Created architecture guidelines | architecture/claude.md |
| 2025-12-14 | Updated test files with cascading path | test_integrated_td_features.py, test_td_predictor.py, test_game_script.py, test_snap_share.py |
| 2025-12-14 | Documented player_id = gsis_id | CLAUDE.md |

---

**Report generated by NFL Data Architect audit**
