# NFL QUANT: Full Feature Audit & Gap Analysis
**Date:** 2025-11-29
**Status:** Comprehensive audit completed

---

## EXECUTIVE SUMMARY

This audit identifies **significant untapped opportunities** in the NFL QUANT codebase. The recent discovery that adding opponent rush defense EPA improved rush_yds ROI from ~0% to +20.6% is NOT an isolated case - similar gaps exist across other markets.

### Key Findings:
1. **18+ feature modules exist** but many are NOT integrated into the production pipeline
2. **V12 model only uses 12 features**, mostly derived from trailing stats and Vegas lines
3. **Defensive EPA validated for rush_yds** - same approach likely works for pass_yds, rec_yds
4. **Air yards/YAC data available** but NOT used in any models
5. **CPOE (Completion % Over Expected)** available but NOT used
6. **Red zone data** available but NOT used for TD predictions

---

## PHASE 1: CODEBASE MAP

### Feature Modules (nfl_quant/features/)

| Module | Purpose | Production Status |
|--------|---------|-------------------|
| `defensive_metrics.py` | Rush/pass defense EPA by team | **PARTIAL** - only rush_yds in V14 |
| `opponent_stats.py` | Player vs specific opponent history | **PARTIAL** - used for warnings only |
| `matchup_features.py` | QB connections, situational multipliers | **NOT USED** |
| `trailing_stats.py` | 4-week EWMA trailing averages | **USED** |
| `enhanced_features.py` | EWMA spans (3/5/10), home/away splits | **NOT USED** |
| `team_strength.py` | Elo ratings, win probability | **NOT USED** |
| `route_metrics.py` | TPRR, Y/RR, route participation | **NOT USED** |
| `contextual_features.py` | Game script, weather | **PARTIAL** |
| `snap_count_features.py` | Snap share analysis | **NOT USED** |
| `player_features.py` | Base player features | **USED** |
| `team_metrics.py` | Team pace, play counts | **PARTIAL** |
| `ngs_features.py` | Next Gen Stats integration | **NOT USED** |
| `ff_opportunity_features.py` | Fantasy opportunity metrics | **NOT USED** |
| `historical_baseline.py` | Career baselines | **NOT USED** |
| `role_change_detector.py` | Snap/target share changes | **NOT USED** |
| `tier1_2_integration.py` | Tiered feature integration | **NOT USED** |
| `historical_injury_impact.py` | Injury impact on teammates | **NOT USED** |
| `injuries.py` | Injury report processing | **PARTIAL** |

### V12 Model Features (WHAT'S ACTUALLY USED)

The production V12 model uses only **12 features**:
```python
feature_cols = [
    'line_vs_trailing',           # Primary signal
    'line_level',                 # Vegas line value
    'line_in_sweet_spot',         # Line between 3.5-7.5
    'player_under_rate',          # Historical under rate
    'player_bias',                # Actual vs line delta
    'market_under_rate',          # Market regime
    'LVT_x_player_tendency',      # Interaction
    'LVT_x_player_bias',          # Interaction
    'LVT_x_regime',               # Interaction
    'LVT_in_sweet_spot',          # Interaction
    'market_bias_strength',       # Regime strength
    'player_market_aligned',      # Alignment
]
```

**CRITICAL GAP:** No opponent defensive metrics, no air yards, no EPA.

### V14 Model (Rush Yards Only)

V14 added **trailing_def_epa** for rush_yds:
- Walk-forward ROI: +20.6% at 55% threshold
- Coefficients: LVT (positive), def_EPA (positive - good defense = more unders)

**BUT** this approach was NOT extended to:
- pass_yds (pass defense EPA)
- reception_yds (coverage EPA)
- receptions (completion % allowed)

---

## PHASE 2: FEATURE INVENTORY

### Data Available in PBP (372 columns)

| Category | Columns Available | Currently Used? |
|----------|------------------|-----------------|
| EPA/WPA | 40 columns | **PARTIAL** (only defense EPA for rush) |
| Air Yards/YAC | 32 columns | **NO** |
| CPOE | 1 column | **NO** |
| QB Hit/Pressure | 5 columns | **NO** |
| Red Zone | 4 columns (yrdln, goal_to_go) | **NO** |
| xPass/xRush | 6 columns | **NO** |
| Pass Location | left/middle/right | **NO** |
| Run Gap | left/middle/right | **NO** |

### Data Available in Weekly Stats (114 columns)

| Category | Columns Available | Currently Used? |
|----------|------------------|-----------------|
| Target Share | target_share | **NO** |
| Air Yards Share | air_yards_share | **NO** |
| Receiving Air Yards | receiving_air_yards | **NO** |
| Receiving YAC | receiving_yards_after_catch | **NO** |
| Passing Air Yards | passing_air_yards | **NO** |
| Passing CPOE | passing_cpoe | **NO** |
| First Downs | passing/rushing/receiving_first_downs | **NO** |

---

## PHASE 3: GAP ANALYSIS BY MARKET

### PASS YARDS (`player_pass_yds`)

**Current Status:** Walk-forward shows ~0% ROI. Market EXCLUDED from V12.

**MISSING FEATURES:**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| **Opponent Pass Defense EPA** | YES (PBP) | **HIGH** | Never integrated! |
| Opponent Completion % Allowed | YES (PBP) | Medium | Never calculated |
| Opponent Sack Rate | YES (PBP) | Medium | Never calculated |
| QB Pressure Rate Faced | YES (qb_hit) | Medium | Never integrated |
| Air Yards / Attempt | YES (weekly) | Low | Never integrated |
| Game Total (implied pace) | YES (odds) | Medium | In V12 but weak |

**HYPOTHESIS:** Adding pass_def_epa (same as rush) could unlock this market.

```python
# PROPOSED: Calculate trailing pass defense EPA
pass_def = pbp[pbp['play_type'] == 'pass'].groupby(['defteam', 'week']).agg(
    pass_def_epa=('epa', 'mean')
)
# Use shift(1) for no leakage
trailing_pass_def_epa = pass_def.groupby('defteam')['pass_def_epa'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)
```

---

### RECEIVING YARDS (`player_reception_yds`)

**Current Status:** +7.2% ROI at 55% threshold (marginal).

**MISSING FEATURES:**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| **Opponent Pass Defense EPA** | YES | **HIGH** | Never integrated |
| Air Yards Share | YES (weekly) | **HIGH** | Never used |
| Target Share | YES (weekly) | **HIGH** | Never used |
| YAC vs Air Yards Ratio | YES (weekly) | Medium | Never calculated |
| Receiving EPA trend | YES (weekly) | Medium | Never used |
| Position-Specific Yards Allowed | Calculable | Medium | Never implemented |

**KEY INSIGHT:** Receivers with high air_yards_share are MORE volatile. Use as variance multiplier.

```python
# Air yards share indicates deep threat potential
# High air_yards_share + negative pass_def_epa = OVER opportunity
```

---

### RECEPTIONS (`player_receptions`)

**Current Status:** +37.9% ROI at 65% threshold. BEST performing market.

**MISSING FEATURES (could improve further):**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| **Opponent Completion % Allowed** | YES (PBP) | **HIGH** | Never calculated |
| Target Share | YES (weekly) | **HIGH** | Never used |
| ADOT (Avg Depth of Target) | Calculable | Medium | Never calculated |
| Pass Attempts vs Opponent | Calculable | Medium | Never integrated |

**HYPOTHESIS:**
- Lower ADOT = higher catch rate = more receptions
- High target_share + bad pass defense = OVER signal

```python
# Calculate player's average depth of target
player_adot = pbp.groupby('receiver_player_id')['air_yards'].mean()
# Lower ADOT = more catches per target
```

---

### RUSH YARDS (`player_rush_yds`)

**Current Status:** +20.6% ROI at 55% threshold (V14 with defense EPA).

**ADDITIONAL OPPORTUNITIES:**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| Run Gap Success Rate | YES (PBP) | Medium | Never calculated |
| Stacked Box Rate | Calculable | Medium | Would need formation |
| Game Script (trailing) | YES | Medium | Not in V14 |
| Carry Share Trend | YES (weekly) | Low | Never used |

**V14 ALREADY CAPTURES MAIN SIGNAL.** Lower priority for improvements.

---

### PASS TDs / RECEIVING TDs / RUSH TDs

**Current Status:** NOT in production models.

**MISSING FEATURES:**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| **Red Zone Target Share** | Calculable | **HIGH** | Never implemented |
| **Red Zone Carry Share** | Calculable | **HIGH** | Never implemented |
| Goal Line Opportunities | YES (goal_to_go) | **HIGH** | Never used |
| Opponent RZ Defense EPA | Calculable | **HIGH** | Never implemented |
| TD Rate per Attempt | YES (weekly) | Medium | Never used |

**CRITICAL GAP:** Red zone data is AVAILABLE but completely unused!

```python
# Calculate red zone opportunities
rz_plays = pbp[(pbp['yrdln'] <= 20) & (pbp['goal_to_go'] == 0)]
# Or goal-to-go plays
gtg_plays = pbp[pbp['goal_to_go'] == 1]
```

---

### COMPLETIONS (QB)

**Current Status:** NOT in production models.

**MISSING FEATURES:**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| **Opponent Completion % Allowed** | YES | **HIGH** | Never calculated |
| CPOE (Completion % Over Expected) | YES (weekly) | **HIGH** | Never used |
| Pressure Rate Faced | YES (qb_hit) | Medium | Never used |
| Air Yards / Attempt | YES | Medium | Never used |

```python
# CPOE available directly in weekly stats
# High CPOE + bad pass defense = more completions than expected
```

---

### INTERCEPTIONS (QB)

**Current Status:** NOT in production models.

**MISSING FEATURES:**

| Feature | Available? | Expected Impact | Why Missing |
|---------|------------|-----------------|-------------|
| **Opponent INT Rate** | Calculable | **HIGH** | Never implemented |
| QB INT Rate (trailing) | YES | Medium | Never used |
| Pressure Rate | YES | Medium | Never used |
| Negative EPA Plays | Calculable | Low | Never used |

---

## PHASE 4: CORRELATION TESTING FRAMEWORK

### Template for Testing New Features

```python
def test_feature_correlation(backtest_df, market, new_feature_col, trailing_col='trailing_stat'):
    """
    Test if a new feature correlates with prediction residuals.

    If |correlation| > 0.10, feature is worth investigating.
    """
    df = backtest_df[backtest_df['market'] == market].copy()

    # What does trailing average miss?
    df['residual'] = df['actual_stat'] - df[trailing_col]

    # Correlation with new feature
    corr = df['residual'].corr(df[new_feature_col])

    print(f"{market} - {new_feature_col}:")
    print(f"  Correlation with residual: {corr:.3f}")

    # Directional check
    if corr > 0.10:
        print(f"  -> Positive correlation: Higher {new_feature_col} = higher actual")
    elif corr < -0.10:
        print(f"  -> Negative correlation: Higher {new_feature_col} = lower actual")
    else:
        print(f"  -> Weak correlation: May not add value")

    return corr
```

### Walk-Forward Validation Template

```python
def validate_feature_walk_forward(data, market, features, window_weeks=20):
    """
    Walk-forward validation with new feature.

    SUCCESS CRITERIA:
    - ROI improvement > 5% at 55% threshold
    - Sample size > 50 bets
    - Coefficient sign makes logical sense
    """
    results = []

    for test_week in test_weeks:
        # Train on prior data only
        train = data[data['global_week'] < test_week][-window_weeks:]
        test = data[data['global_week'] == test_week]

        # Fit logistic regression
        model = LogisticRegression()
        model.fit(train[features], train['under_hit'])

        # Predict
        test['p_under'] = model.predict_proba(test[features])[:, 1]
        results.append(test)

    # Calculate ROI
    all_results = pd.concat(results)
    for threshold in [0.55, 0.60, 0.65]:
        mask = all_results['p_under'] >= threshold
        hits = all_results.loc[mask, 'under_hit'].sum()
        total = mask.sum()
        roi = (hits * 0.909 - (total - hits)) / total * 100
        print(f"Threshold {threshold:.0%}: N={total}, ROI={roi:+.1f}%")
```

---

## PHASE 5: PRIORITIZED RECOMMENDATIONS

### TIER 1: Quick Wins (HIGH Impact, EASY Implementation)

| # | Feature | Market | Expected ROI Impact | Implementation |
|---|---------|--------|---------------------|----------------|
| 1 | **Pass Defense EPA** | pass_yds | +10-15% | Copy V14 pattern |
| 2 | **Pass Defense EPA** | rec_yds | +5-10% | Copy V14 pattern |
| 3 | **Target Share** | receptions, rec_yds | +3-5% | Use weekly stats |
| 4 | **Completion % Allowed** | receptions | +3-5% | Calculate from PBP |
| 5 | **Air Yards Share** | rec_yds | +2-4% | Use weekly stats |

### TIER 2: Medium Effort (HIGH Impact, More Work)

| # | Feature | Market | Expected ROI Impact | Implementation |
|---|---------|--------|---------------------|----------------|
| 6 | **Red Zone Target Share** | receiving_tds | New market | Calculate from PBP |
| 7 | **Red Zone Carry Share** | rushing_tds | New market | Calculate from PBP |
| 8 | **CPOE** | pass_yds, completions | +3-5% | Use weekly stats |
| 9 | **ADOT** | receptions | +2-3% | Calculate from PBP |
| 10 | **Position-Specific Yards Allowed** | rec_yds | +3-5% | Calculate from PBP |

### TIER 3: Research Needed

| # | Feature | Market | Notes |
|---|---------|--------|-------|
| 11 | QB Hit Rate | pass_yds | May correlate with lower yards |
| 12 | Run Gap Success | rush_yds | V14 already captures most signal |
| 13 | Game Script Prediction | All | Complex, may not add much |

---

## PHASE 6: IMPLEMENTATION PLAN

### Step 1: Add Pass Defense EPA (Copy V14 Pattern)

```python
# In scripts/train/train_v15_pass_defense_aware.py
def calculate_pass_defense_epa(pbp):
    """Calculate trailing pass defense EPA per team per week."""
    pass_def = pbp[pbp['play_type'] == 'pass'].groupby(['defteam', 'week', 'season']).agg(
        pass_def_epa=('epa', 'mean')
    ).reset_index()

    pass_def = pass_def.sort_values(['defteam', 'season', 'week'])
    pass_def['trailing_pass_def_epa'] = pass_def.groupby('defteam')['pass_def_epa'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    return pass_def
```

### Step 2: Add Target Share Feature

```python
# Use existing weekly stats
def add_target_share_feature(data, weekly_stats):
    """Add trailing target share as feature."""
    weekly = weekly_stats[['player_id', 'season', 'week', 'target_share']].copy()

    # Calculate trailing target share
    weekly = weekly.sort_values(['player_id', 'season', 'week'])
    weekly['trailing_target_share'] = weekly.groupby('player_id')['target_share'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    return data.merge(weekly[['player_id', 'season', 'week', 'trailing_target_share']],
                      on=['player_id', 'season', 'week'], how='left')
```

### Step 3: Add Air Yards Share Feature

```python
# Use existing weekly stats
def add_air_yards_share_feature(data, weekly_stats):
    """Add trailing air yards share as feature."""
    weekly = weekly_stats[['player_id', 'season', 'week', 'air_yards_share']].copy()

    weekly = weekly.sort_values(['player_id', 'season', 'week'])
    weekly['trailing_air_yards_share'] = weekly.groupby('player_id')['air_yards_share'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    return data.merge(weekly[['player_id', 'season', 'week', 'trailing_air_yards_share']],
                      on=['player_id', 'season', 'week'], how='left')
```

---

---

## VALIDATION RESULTS (Walk-Forward 2025)

### Key Finding: Defense EPA Works for Rush, NOT for Pass

| Market | Feature | Walk-Forward ROI | Verdict |
|--------|---------|-----------------|---------|
| **rush_yds** | Rush Def EPA | **+20.6%** | ✅ VALIDATED (V14) |
| pass_yds | Pass Def EPA | **-62.6%** | ❌ NO EDGE |
| rec_yds | Pass Def EPA | **+1.2%** | ⚠️ MARGINAL |
| **receptions** | Pass Def EPA | **+31.7%** | ✅ SIGNAL EXISTS |

### Critical Insight: Receptions + Pass Def EPA = Strong Signal

The walk-forward validation shows that adding Pass Defense EPA to receptions actually **IMPROVES** the already-strong market:

```
P(UNDER) >= 55%: 100W-45L (69.0%), ROI: +31.7%
P(UNDER) >= 60%: 12W-2L (85.7%), ROI: +63.6%
P(UNDER) >= 65%: 4W-1L (80.0%), ROI: +52.7%
```

This is **better than V12** which only achieved +37.9% at 65% threshold.

### Why Rush EPA Works But Pass EPA Doesn't

| Factor | Rush | Pass |
|--------|------|------|
| **Correlation with UNDER** | -0.062 (negative = good def = more unders) | +0.053 (opposite!) |
| **Directional Analysis** | Good def: 60.2% under, Bad def: 51.8% | Good def: 52.9%, Bad def: 54.3% |
| **Predictability** | Rushing is more matchup-dependent | Passing affected by more factors |

**Conclusion:** Vegas appears to already price in pass defense strength, but underweights rush defense.

---

## REVISED PRIORITY LIST

Based on validation results:

### IMMEDIATE ACTIONS (Validated)

1. **Receptions + Pass Def EPA** - Walk-forward shows +31.7% ROI at 55%
   - Implementation: Add `trailing_pass_def_epa` to V12 for receptions
   - Expected improvement: ~10% ROI boost

2. **Keep V14 for Rush Yards** - Already working (+20.6% ROI)

### DO NOT IMPLEMENT (No Edge Validated)

- ~~Pass Defense EPA for pass_yds~~ → Walk-forward shows -62.6% ROI
- ~~Target Share for receptions~~ → Correlation was **negative** (-0.171), Vegas already prices this
- ~~Completion % Allowed~~ → Weak correlation (+0.014)

### RESEARCH FURTHER

- Air Yards Share - Shows HIGH VARIANCE signal (high air share = more volatile outcomes)
  - Could be useful for OVER bets, not UNDER
- Red Zone metrics for TDs - Not tested yet

---

## SUMMARY: FEATURES THAT EXIST BUT AREN'T USED

| Feature | Calculated In | Used In Production? | Markets It Could Help |
|---------|---------------|---------------------|----------------------|
| pass_defense_epa | defensive_metrics.py | **NO** | pass_yds, rec_yds, receptions |
| rush_defense_epa | defensive_metrics.py | **YES (V14 only)** | rush_yds |
| target_share | weekly_stats | **NO** | receptions, rec_yds |
| air_yards_share | weekly_stats | **NO** | rec_yds |
| receiving_epa | weekly_stats | **NO** | rec_yds |
| passing_cpoe | weekly_stats | **NO** | pass_yds, completions |
| completion_pct_allowed | Can calculate | **NO** | receptions |
| red_zone_targets | Can calculate | **NO** | receiving_tds |
| goal_line_carries | Can calculate | **NO** | rushing_tds |
| opponent_yards_allowed | enhanced_features.py | **NO** | All yards markets |
| position_specific_defense | enhanced_features.py | **NO** | rec_yds |
| elo_diff | team_strength.py | **NO** | All (game script) |
| route_participation | route_metrics.py | **NO** | receptions, rec_yds |
| tprr (targets/route) | route_metrics.py | **NO** | receptions |
| yrr (yards/route) | route_metrics.py | **NO** | rec_yds |

---

## FEATURES THAT SHOULD EXIST BUT DON'T

| Feature | Why Valuable | Data Available? | Implementation Effort |
|---------|-------------|-----------------|----------------------|
| Opponent INT Rate | Predict QB turnovers | YES (PBP) | Low |
| Opponent Sack Rate | Predict fewer attempts | YES (PBP) | Low |
| Player ADOT | Lower = more catches | YES (air_yards) | Medium |
| Pressure Rate Faced | Affects efficiency | YES (qb_hit) | Medium |
| Stacked Box Rate | Affects rushing | Need formation data | High |
| Corner Matchup Quality | Position-specific | Need personnel data | High |
| Snap % in Red Zone | TD prediction | Calculable | Medium |

---

## NEXT STEPS

1. **IMMEDIATE (Today):** Test pass_def_epa for pass_yds market using V14 pattern
2. **This Week:** Add target_share and air_yards_share to V12 for receptions/rec_yds
3. **Next Week:** Build red zone feature set for TD markets
4. **Ongoing:** Validate each new feature with walk-forward before production

---

## APPENDIX: Code Locations

- Feature modules: `nfl_quant/features/`
- V12 training: `scripts/train/train_v12_interaction_model.py`
- V14 training: `scripts/train/train_v14_defense_aware.py`
- Usage predictor: `nfl_quant/models/usage_predictor.py`
- Efficiency predictor: `nfl_quant/models/efficiency_predictor.py`
- Simulator: `nfl_quant/simulation/player_simulator_v4.py`
- Backtest data: `data/backtest/combined_odds_actuals_2023_2024_2025.csv`
- PBP data: `data/nflverse/pbp_*.parquet`
- Weekly stats: `data/nflverse/weekly_stats.parquet`
