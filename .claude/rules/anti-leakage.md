# Rule: Anti-Leakage

**Scope**: All feature engineering and model training code

---

## Rules

### 1. Shift Before Expand

When computing rolling/expanding statistics, always `shift(1)` BEFORE `expanding()`.

```python
# CORRECT - excludes current row
df['feature'] = df['col'].shift(1).expanding().mean()

# WRONG - leaks current row
df['feature'] = df['col'].expanding().mean().shift(1)
```

### 2. Walk-Forward Gap

When training for week N, use only data from weeks < N-1.

```python
# CORRECT - one week gap
train_data = df[df['week'] < test_week - 1]

# WRONG - no gap
train_data = df[df['week'] < test_week]
```

### 3. Calibrator Split

Probability calibrators must be fit on held-out data, not training data.

```python
# CORRECT - 80/20 split
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
calibrator.fit(model.predict_proba(X_calib), y_calib)

# WRONG - same data
model.fit(X_train, y_train)
calibrator.fit(model.predict_proba(X_train), y_train)  # NO!
```

### 4. Opponent Defense Shift

Opponent defense features must use prior games, not current game.

```python
# CORRECT - prior games only
df['opp_def_epa'] = df.groupby('opponent')['epa'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)
```

## Why

Data leakage inflates backtest performance but fails in production:
- Backtests show 70% accuracy, production shows 50%
- Model "memorizes" rather than generalizes
- Impossible to reproduce results on new data

## Detection

```bash
# Check for incorrect expanding pattern
grep -r "expanding().*shift(1)" nfl_quant/ scripts/

# Check for missing walk-forward gap
grep -r "< test_week[^-]" scripts/train/
```

## Reference

See [.context/architecture/invariants.md](../../.context/architecture/invariants.md) INV-002, INV-004, INV-005
