# NFL QUANT Codebase Modernization Report

**Date**: 2025-12-04
**Analyst**: Principal Software Architect Assessment
**Codebase Version**: V17 (2.3.0)

---

## Executive Summary

The NFL QUANT codebase has significant technical debt resulting from rapid iteration without consolidation. Over **55% of files are archived** (252 of 459 Python files), and active code contains **substantial duplication** across 6 key areas. This report identifies all fragmentation points and provides a root-cause modernization plan.

---

## Section 1: Quantified Fragmentation

### 1.1 File Statistics

| Metric | Count |
|--------|-------|
| Total Python files | 459 |
| Archived files | 252 (55%) |
| Active files | ~207 |
| Archive directories | 6 |

### 1.2 Duplication Hotspots

| Pattern | Occurrences | Files Affected |
|---------|-------------|----------------|
| Kelly fraction implementations | 40+ | 15+ files |
| Player name normalization | 25+ | 20+ files |
| FeatureEngine class | 2 | core.py, engine.py |
| Edge calculation functions | 10+ | 8+ files |
| Threshold management | 8+ | 6+ files |
| Market definitions | 6+ | 5+ files |
| Model loading functions | 5+ | 4+ files |

---

## Section 2: Critical Duplication Details

### 2.1 Kelly Fraction Implementations (CRITICAL)

**Files with distinct implementations:**

1. `nfl_quant/betting/kelly_criterion.py:30` - `calculate_kelly_fraction()`
2. `nfl_quant/betting/bet_sizing.py:77` - `BetSizer.calculate_kelly_bet()`
3. `nfl_quant/betting/edge_detection.py:184` - `EdgeDetector.kelly_fraction()`
4. `nfl_quant/core/unified_betting.py:107` - `calculate_kelly_fraction()`
5. `nfl_quant/integration/nflverse_integration.py:331` - `calculate_kelly_fraction()`
6. `nfl_quant/utils/odds.py:73` - `OddsEngine.kelly_fraction()`
7. `nfl_quant/config_loader.py:113` - `get_kelly_fraction()`
8. Multiple archive scripts with inline implementations

**Risk**: Behavioral drift, different default fractions (0.25 vs 0.5), inconsistent limits

### 2.2 Player Name Normalization (HIGH)

**Files with `normalize_player_name` or `normalize_name`:**

1. `nfl_quant/utils/player_names.py:5` - **CANONICAL** (but only 1 function)
2. `nfl_quant/utils/unified_integration.py:1171` & `:1285` - TWO inline copies!
3. `nfl_quant/utils/pick_narrative_generator.py:244` - inline
4. `nfl_quant/features/route_metrics.py:199` - imports from utils (correct)
5. `nfl_quant/data/adapters/base_adapter.py:58` - class method
6. `scripts/tracking/track_bet_results.py:118` - inline copy
7. 15+ more in archive scripts

**Risk**: Player matching failures, missed joins, silent data loss

### 2.3 FeatureEngine Duplication (CRITICAL)

| File | Purpose | Lines |
|------|---------|-------|
| `nfl_quant/features/core.py:142` | **CANONICAL** - Full V17 feature set | ~2200 |
| `nfl_quant/features/engine.py:15` | Team-level features only | ~200 |

**Issues:**
- `engine.py` uses old `nfl_quant/config.py`
- `core.py` uses new `configs/model_config.py`
- Both export a class named `FeatureEngine`
- Import confusion: which one to use?

### 2.4 Edge/Threshold Calculation Sprawl (HIGH)

**Edge calculation implementations:**
1. `nfl_quant/core/unified_betting.py:61` - `calculate_edge()`
2. `nfl_quant/core/unified_betting.py:77` - `calculate_edge_percentage()`
3. `nfl_quant/betting/edge_detection.py:134` - `EdgeDetector.calculate_edge()`
4. Multiple archive scripts with variations

**Threshold management locations:**
1. `nfl_quant/config_loader.py:90` - `get_edge_threshold()`
2. `nfl_quant/calibration/position_market_calibrator.py:286` - `get_edge_threshold()`
3. `nfl_quant/integration/nflverse_integration.py:286` - `get_edge_threshold()`
4. `nfl_quant/betting/edge_detection.py:63` - `min_edge_thresholds` dict
5. `nfl_quant/betting/bet_sizing.py:149` - `get_edge_threshold_for_market()`

### 2.5 Market Definitions Sprawl (MEDIUM)

**Different market lists:**
1. `configs/model_config.py:173` - `SUPPORTED_MARKETS` (canonical)
2. `nfl_quant/data/draftkings_client.py:32` - `CORE_MARKETS`
3. `scripts/dashboard/generate_pro_dashboard.py:38` - `V16_MARKETS`
4. `scripts/dashboard/generate_pro_dashboard.py:1451` - `VALIDATED_MARKETS`
5. Archive scripts: `BACKTEST_MARKETS`, `PROP_MARKETS`, `NFL_PLAYER_PROP_MARKETS`

### 2.6 Configuration Sprawl (HIGH)

**Config files in project:**

| File | Purpose | Issues |
|------|---------|--------|
| `configs/model_config.py` | V17 model config (NEW) | Canonical for model |
| `nfl_quant/config.py` | Pydantic settings | Season, paths, API |
| `nfl_quant/config_enhanced.py` | Enhanced config | Feature engineering |
| `nfl_quant/config_loader.py` | Config loader | Betting, thresholds |
| `nfl_quant/utils/season_config.py` | Season config | Overlaps with config.py |

**Issues:**
- `CURRENT_WEEK` defined in `config.py` AND environment variables
- `SEASON` hardcoded in multiple places
- Thresholds split across `config_loader.py` and `position_market_calibrator.py`

---

## Section 3: Model Loading Fragmentation

**Functions for loading models:**

1. `scripts/predict/generate_unified_recommendations_v3.py:591` - `load_active_model()`
2. `scripts/predict/generate_unified_recommendations_v3.py:812` - `load_v16_model()` (alias)
3. `scripts/predict/generate_unified_recommendations_v3.py:1406` - `load_model_predictions()`
4. `nfl_quant/models/production_loader.py:20` - `load_production_model()`
5. `scripts/test/calibration_check.py:26` - `load_model_and_data()`

**Risk**: Different loading logic, inconsistent error handling, model version confusion

---

## Section 4: Archive Analysis

### 4.1 Archive Directories

```
./nfl_quant/simulation/_archive
./scripts/backtest/_archive
./scripts/fetch/_archive
./scripts/train/_archive
./scripts/_archive
./data/models/_archive
```

### 4.2 Archive Statistics

- **252 Python files** in archive directories
- Many contain **inline duplicates** of core functions
- Some archive scripts are **still being imported** by active code
- Version suffixes: v2, v3, v4, v12, v14, v15, v16 scattered throughout

---

## Section 5: Root Cause Analysis

### 5.1 Why This Happened

1. **Rapid iteration** without consolidation phases
2. **Copy-paste development** to "not break working code"
3. **No import governance** - easy to create inline copies
4. **Archive-but-don't-delete** policy created sprawl
5. **Version suffixes** instead of git history for versioning

### 5.2 Impact

| Impact | Severity | Description |
|--------|----------|-------------|
| Maintenance burden | HIGH | Changes require updating multiple files |
| Behavioral drift | HIGH | Same function behaves differently across files |
| Onboarding friction | MEDIUM | Confusing which module to use |
| Test coverage gaps | MEDIUM | Duplicate code not tested consistently |
| Import errors | LOW | Occasional circular import issues |

---

## Section 6: Modernization Plan

### Phase 1: Consolidate Core Utilities (Priority: CRITICAL)

**1.1 Player Name Normalization**
- Keep: `nfl_quant/utils/player_names.py`
- Action: Delete all inline copies, update imports
- Files to modify: ~15

**1.2 Kelly Criterion**
- Keep: `nfl_quant/betting/kelly_criterion.py`
- Action: Create single `calculate_kelly()` function
- Consolidate: 6 implementations → 1

**1.3 Edge Calculations**
- Keep: `nfl_quant/core/unified_betting.py`
- Action: Make `calculate_edge_percentage()` the only implementation

### Phase 2: Unify FeatureEngine (Priority: CRITICAL)

**Action:**
1. Delete `nfl_quant/features/engine.py` (team-level features → merge into core.py)
2. Update any imports
3. Ensure `core.py` is the single entry point

### Phase 3: Consolidate Configuration (Priority: HIGH)

**Target Architecture:**
```
configs/
├── model_config.py      # Model-specific (V17 features, thresholds)
├── betting_config.py    # Kelly, edge thresholds (NEW - merged)
└── api_config.py        # API keys, endpoints (NEW - from .env)

nfl_quant/
├── config.py            # Pydantic settings (paths, season only)
└── (delete config_loader.py, config_enhanced.py)
```

### Phase 4: Archive Cleanup (Priority: MEDIUM)

**Actions:**
1. Move archive directories to `/archive_YYYYMMDD/` (single location)
2. Remove from PYTHONPATH
3. Add to `.gitignore` (or commit as historical reference)
4. Update any remaining imports from archive

### Phase 5: Model Loading Unification (Priority: MEDIUM)

**Target:**
- Single `nfl_quant/models/loader.py` with:
  - `load_active_model()` - production model
  - `load_model_for_version(version)` - historical models
  - `load_predictions(week)` - predictions CSV

### Phase 6: Market Definitions (Priority: LOW)

**Action:**
- Single `SUPPORTED_MARKETS` in `configs/model_config.py`
- All other files import from there
- Remove: V16_MARKETS, CORE_MARKETS, etc.

---

## Section 7: Implementation Order

| Phase | Effort | Risk | Dependencies |
|-------|--------|------|--------------|
| 1.1 Player Names | 2h | LOW | None |
| 1.2 Kelly | 4h | MEDIUM | Tests |
| 1.3 Edge | 2h | LOW | None |
| 2 FeatureEngine | 4h | MEDIUM | Phase 1 |
| 3 Config | 6h | HIGH | Phase 2 |
| 4 Archive | 2h | LOW | None |
| 5 Model Loading | 3h | MEDIUM | Phase 3 |
| 6 Markets | 1h | LOW | Phase 3 |

**Total Estimated Effort**: 24h (3 days)

---

## Section 8: Verification Checklist

After modernization, verify:

- [ ] `from nfl_quant.utils.player_names import normalize_player_name` works everywhere
- [ ] `from nfl_quant.betting.kelly_criterion import calculate_kelly` is single source
- [ ] `from nfl_quant.features.core import FeatureEngine` is only FeatureEngine
- [ ] All tests pass
- [ ] Walk-forward backtest produces same results
- [ ] Pipeline runs end-to-end
- [ ] No imports from archive directories

---

## Appendix A: Files to Delete

```
nfl_quant/features/engine.py          # Duplicate FeatureEngine
nfl_quant/config_enhanced.py          # Merge into config.py
nfl_quant/config_loader.py            # Merge into configs/
nfl_quant/utils/season_config.py      # Duplicate of config.py
```

## Appendix B: Import Governance Rules

1. **Player name normalization**: Only `from nfl_quant.utils.player_names import normalize_player_name`
2. **Kelly calculations**: Only `from nfl_quant.betting.kelly_criterion import calculate_kelly_fraction`
3. **Edge calculations**: Only `from nfl_quant.core.unified_betting import calculate_edge_percentage`
4. **FeatureEngine**: Only `from nfl_quant.features.core import FeatureEngine`
5. **Market lists**: Only `from configs.model_config import SUPPORTED_MARKETS`

---

*Generated by codebase modernization analysis*
