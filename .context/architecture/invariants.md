# Architectural Invariants

**Version**: 1.0 | **Last Updated**: 2025-12-26

---

## Purpose

This document defines **invariants** - rules that must never be violated. Breaking an invariant causes incorrect predictions, data leakage, or system failures.

Each invariant includes:
- **Rule**: What must always be true
- **Rationale**: Why it matters
- **Violation Example**: What NOT to do
- **Correct Pattern**: What TO do
- **Detection**: How to verify compliance

---

## INV-001: No Fallback Data Files

**Status**: ACTIVE | **Added**: 2025-12-26

### Rule
Use only generic NFLverse files. If a file doesn't exist, fail explicitly - never fall back to stale season-specific files.

### Rationale
R script `fetch_nflverse_data.R` creates fresh generic files daily (`pbp.parquet`). Season-specific files (`pbp_2025.parquet`) may be weeks old. Falling back silently causes predictions based on stale data.

### Violation Example
```python
# WRONG - falls back to stale data silently
pbp_path = Path('data/nflverse/pbp.parquet')
if not pbp_path.exists():
    pbp_path = Path('data/nflverse/pbp_2025.parquet')  # May be weeks old!
```

### Correct Pattern
```python
# CORRECT - fail if fresh data missing
pbp_path = Path('data/nflverse/pbp.parquet')
if not pbp_path.exists():
    raise FileNotFoundError(
        f"PBP file not found: {pbp_path}. "
        "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
    )
```

### Detection
```bash
# Check for fallback patterns in code
grep -r "if not.*exists.*pbp_2025" nfl_quant/ scripts/
```

---

## INV-002: Shift Before Expand for Rolling Stats

**Status**: ACTIVE | **Added**: 2025-12-15

### Rule
When computing rolling/expanding statistics, always `shift(1)` BEFORE `expanding()`.

### Rationale
`expanding().mean().shift(1)` computes the mean including the current row, then shifts the result. This leaks the current row's data into the feature.

### Violation Example
```python
# WRONG - leaks future data
df['market_under_rate'] = df['under_hit'].expanding().mean().shift(1)
```

### Correct Pattern
```python
# CORRECT - excludes current row before computing
df['market_under_rate'] = df['under_hit'].shift(1).expanding().mean()
```

### Detection
```bash
# Find potential violations
grep -r "expanding().*shift(1)" nfl_quant/ scripts/
```

---

## INV-003: NFLverse Column Naming Convention

**Status**: ACTIVE | **Added**: 2025-12-26

### Rule
Use NFLverse native column names when working with NFLverse data:
- `carries` (not `rushing_attempts`)
- `attempts` (not `passing_attempts`)
- `completions` (not `passing_completions`)

### Rationale
NFLverse data uses specific column names. Using incorrect names causes silent failures (column not found) or incorrect joins.

### Violation Example
```python
# WRONG - NFLverse doesn't have these columns
df['rushing_attempts']  # Should be df['carries']
df['passing_attempts']  # Should be df['attempts']
```

### Correct Pattern
```python
# CORRECT - use NFLverse column names
df['carries']       # Rush attempts
df['attempts']      # Pass attempts
df['completions']   # Completions
```

### Detection
Check `nfl_quant/data/stats_schema.py` for canonical column definitions.

---

## INV-004: Walk-Forward Validation Gap

**Status**: ACTIVE | **Added**: 2025-12-15

### Rule
When training for week N, use only data from weeks < N-1. Never use week N-1 data.

### Rationale
Week N-1 features may contain information correlated with week N outcomes (e.g., injury reports filed on game day). The one-week gap prevents temporal leakage.

### Violation Example
```python
# WRONG - uses data too close to test week
train_data = df[df['week'] < test_week]  # Includes week N-1
```

### Correct Pattern
```python
# CORRECT - one week gap
train_data = df[df['week'] < test_week - 1]
```

### Detection
Search for `< test_week` without `- 1` in training code.

---

## INV-005: Calibrator Train/Calibration Split

**Status**: ACTIVE | **Added**: 2025-12-15

### Rule
Probability calibrators must be fit on held-out data (80/20 split), not training data.

### Rationale
Fitting calibrators on training data causes overfitting. The calibrator learns to "fix" training errors that won't generalize.

### Violation Example
```python
# WRONG - calibrator fitted on same data as model
model.fit(X_train, y_train)
calibrator.fit(model.predict_proba(X_train), y_train)  # Same data!
```

### Correct Pattern
```python
# CORRECT - separate calibration set
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
calibrator.fit(model.predict_proba(X_calib), y_calib)  # Held-out data
```

### Detection
Check `train_model.py` for proper split.

---

## INV-006: Player ID is GSIS ID

**Status**: ACTIVE | **Added**: 2025-12-20

### Rule
In NFLverse data, `player_id` IS the canonical `gsis_id`. They are the same field.

### Rationale
NFLverse uses `player_id` as the column name, but it contains GSIS IDs. Creating separate `gsis_id` columns causes confusion and merge failures.

### Violation Example
```python
# WRONG - creates redundant column
df['gsis_id'] = df['player_id']  # They're already the same!
```

### Correct Pattern
```python
# CORRECT - use player_id directly
player_id = df['player_id']  # This IS the GSIS ID
```

---

## INV-007: Game ID is STRING, Not INT

**Status**: ACTIVE | **Added**: 2025-12-20

### Rule
`game_id` in NFLverse data is a STRING in format `{season}_{week}_{away}_{home}`.

### Rationale
Casting to int causes data corruption and join failures.

### Violation Example
```python
# WRONG - corrupts game_id
df['game_id'] = df['game_id'].astype(int)
```

### Correct Pattern
```python
# CORRECT - keep as string
df['game_id'] = df['game_id'].astype(str)
```

---

## INV-008: XGBoost NaN Handling

**Status**: ACTIVE | **Added**: 2025-12-14

### Rule
Leave missing opponent features as NaN. Never fill with 0 or mean.

### Rationale
XGBoost handles NaN natively by learning optimal splits for missing values. Filling with 0 creates false signal (e.g., "opponent allows 0 yards" is very different from "unknown").

### Violation Example
```python
# WRONG - creates false signal
df['opp_def_epa'] = df['opp_def_epa'].fillna(0)
```

### Correct Pattern
```python
# CORRECT - let XGBoost handle NaN
df['opp_def_epa'] = df['opp_def_epa']  # Keep NaN as-is
```

---

## Invariant Violation Reporting

If you discover an invariant violation:

1. **Stop work** - Don't propagate the error
2. **Document the violation** - Which invariant, where, impact
3. **Fix immediately** - These are P0 bugs
4. **Add detection** - Update this doc with grep pattern

---

## Change Log

| Date | Invariant | Change |
|------|-----------|--------|
| 2025-12-26 | INV-001 | Added no-fallback invariant |
| 2025-12-26 | INV-003 | Added NFLverse naming convention |
| 2025-12-15 | INV-002 | Added shift/expand order |
| 2025-12-15 | INV-004 | Added walk-forward gap |
| 2025-12-15 | INV-005 | Added calibrator split |
