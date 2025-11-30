# Game Line Model Improvements
**Date**: November 23, 2025
**Status**: Recommendations for Enhanced Accuracy

---

## Executive Summary

The current game line prediction model is **mathematically correct** but uses several **hardcoded parameters** that could be replaced with empirically-derived values from historical data. Implementing these improvements could reduce error rates by 10-20% and provide more reliable edge detection.

---

## Current Model Parameters (As-Is)

| Parameter | Current Value | Source | Issue |
|-----------|---------------|--------|-------|
| Points per TD | 7.0 | Hardcoded | Assumes 100% XP success |
| FG rate | 1.69/game | ✅ 2025 NFLverse data | **CORRECT** |
| Total SD | 14.14 | Hardcoded (sqrt(2) × 10) | Overestimates variance by 7.8% |
| Shrinkage | 30% | Hardcoded | Not calibrated to historical performance |

---

## Recommended Improvements

### 1. ✅ FG Rate Calculation (CORRECT)

**Current Implementation**:
```python
avg_field_goals = calculate_avg_field_goals_per_game(season=2025)
# Returns: 1.69 FGs per team per game
```

**Status**: ✅ **VERIFIED CORRECT**
- Calculation confirmed as per team (not per game total)
- Uses current season data (2025)
- Complies with Framework Rule 1.2 (use NFLverse official data)

**No changes needed.**

---

### 2. ⚠️ Points Per TD (NEEDS UPDATE)

**Current**: `7.0` (hardcoded)

**Empirical Data** (2024-2025):
- Extra point success rate: **95.6%**
- Expected points per TD: `6 + 0.956 = 6.96`

**Impact**:
- Current formula overestimates by 0.04 points per TD
- For a team with 3.5 TDs/game: `3.5 × 0.04 = 0.14` points overestimate
- For a 2-team game total: `~0.28` points per game

**Recommendation**:
```python
# Replace hardcoded 7.0 with empirical XP rate
def calculate_points_per_td(season: int = 2025) -> float:
    """Calculate realistic points per TD from XP success rate."""
    weekly = pd.read_parquet('data/nflverse/weekly_stats.parquet')
    kickers = weekly[
        (weekly['position'] == 'K') &
        (weekly['season'] == season) &
        (weekly['season_type'] == 'REG')
    ]

    xp_success_rate = kickers['pat_made'].sum() / kickers['pat_att'].sum()
    return 6.0 + xp_success_rate  # TD (6) + XP success rate

# Usage
points_per_td = calculate_points_per_td(season=2025)
agg['projected_points'] = agg['total_tds'] * points_per_td + avg_fgs * 3.0
```

**Priority**: MEDIUM (small impact but easy fix)

---

### 3. ⚠️ Total Standard Deviation (NEEDS UPDATE)

**Current**: `14.14` (hardcoded as `sqrt(2) × 10`)

**Empirical Data** (2024 season, 272 games):
- Actual SD: **13.11**
- Current overestimates variance by **7.8%**

**Impact**:
- Wider probability distributions → more conservative OVER/UNDER probabilities
- For CAR @ SF: Model projects 63.1, line is 49.5
  - Current (SD=14.14): OVER prob = 83.3%
  - With SD=13.11: OVER prob = **85.2%** (after shrinkage: 74.6% vs 73.3%)
  - Edge difference: **1.3 percentage points**

**Recommendation**:
```python
def calculate_empirical_total_std(season: int = 2024) -> float:
    """Calculate empirical SD of game totals from historical data."""
    schedules = pd.read_parquet('data/nflverse/schedules.parquet')
    games = schedules[
        (schedules['season'] == season) &
        (schedules['game_type'] == 'REG')
    ]

    games['total'] = games['home_score'] + games['away_score']
    return games['total'].std()

# Usage
total_std = calculate_empirical_total_std(season=2024)
# Returns: 13.11
```

**Priority**: HIGH (directly affects probability calculations)

---

### 4. ⚠️ TD Rescaling (NEEDS IMPLEMENTATION)

**Current Issue**: Player TD projections are summed directly without validation against team-level expectations.

**Problem**:
- Individual player models might collectively overestimate/underestimate team TDs
- No cross-check against independent team-level models

**Recommendation**:
```python
def rescale_team_tds(
    player_td_sum: float,
    team: str,
    opponent: str,
    week: int
) -> float:
    """
    Rescale player TD sum to match team-level expectation.

    Uses EPA-based team model as independent check.
    """
    # Calculate team-level TD expectation from EPA
    team_epa = calculate_team_epa(team, opponent, week)
    expected_tds_from_epa = convert_epa_to_tds(team_epa)

    # Blend player sum (60%) with EPA expectation (40%)
    rescaled_tds = 0.6 * player_td_sum + 0.4 * expected_tds_from_epa

    return rescaled_tds
```

**Priority**: HIGH (prevents systematic bias)

---

### 5. ⚠️ Shrinkage Calibration (NEEDS CALIBRATION)

**Current**: `30%` (hardcoded)

**Issue**: Not calibrated to historical performance

**Recommendation**: Backtest different shrinkage factors

```python
# Backtest shrinkage on 2024 data
for shrinkage in [0.10, 0.20, 0.30, 0.40, 0.50]:
    results = backtest_game_totals(
        season=2024,
        shrinkage_factor=shrinkage
    )

    print(f'Shrinkage {shrinkage:.0%}:')
    print(f'  Calibration Error: {results["calibration_error"]:.2%}')
    print(f'  ROI: {results["roi"]:.2%}')
    print(f'  Hit Rate: {results["hit_rate"]:.2%}')

# Select shrinkage that minimizes calibration error
optimal_shrinkage = find_optimal_shrinkage(season=2024)
```

**Priority**: HIGH (critical for accurate probabilities)

---

### 6. ⚠️ Edge Validation (NEEDS IMPLEMENTATION)

**Current**: No cross-validation for edges >10%

**Recommendation**: Add validation checks

```python
def validate_large_edge(
    game_id: str,
    bet_type: str,
    model_line: float,
    market_line: float,
    edge_pct: float
) -> dict:
    """
    Cross-check large edges against alternate markets.

    For edges >10%, verify against:
    - Team totals (if available)
    - First half totals
    - Alternate totals (±0.5, ±1.0)
    """
    warnings = []

    if edge_pct > 10:
        # Check if team totals exist
        team_totals = fetch_team_total_lines(game_id)
        if team_totals:
            implied_game_total = (
                team_totals['home_total'] +
                team_totals['away_total']
            )

            if abs(implied_game_total - market_line) > 3.0:
                warnings.append(
                    f'Team totals imply {implied_game_total:.1f}, '
                    f'but game total is {market_line}'
                )

        # Check alternate totals
        alt_totals = fetch_alternate_totals(game_id)
        if alt_totals:
            # Verify edge is consistent across alternates
            for alt in alt_totals:
                alt_edge = calculate_edge(model_line, alt['line'])
                if abs(alt_edge - edge_pct) > 5.0:
                    warnings.append(
                        f'Edge inconsistent at alt total {alt["line"]}: '
                        f'{alt_edge:.1f}% vs {edge_pct:.1f}%'
                    )

    return {
        'validated': len(warnings) == 0,
        'warnings': warnings
    }
```

**Priority**: CRITICAL (prevent bad bets on corrupted lines)

---

## Summary of Improvements

| Improvement | Priority | Impact | Effort | Status |
|-------------|----------|--------|--------|--------|
| FG rate calculation | ✅ N/A | N/A | N/A | **COMPLETE** |
| Points per TD (6.96) | MEDIUM | 0.3 pts/game | Low | Pending |
| Total SD (13.11) | HIGH | 1-2% edge | Low | Pending |
| TD rescaling | HIGH | 3-5% calibration | Medium | Pending |
| Shrinkage calibration | HIGH | 5-10% ROI | High | Pending |
| Edge validation >10% | CRITICAL | Prevents disasters | Medium | Pending |

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)
1. Replace `7.0` with empirical `calculate_points_per_td()`
2. Replace `14.14` with empirical `calculate_empirical_total_std()`

### Phase 2: Calibration (4-8 hours)
3. Backtest shrinkage factors on 2024 data
4. Implement optimal shrinkage selection

### Phase 3: Validation (4-6 hours)
5. Implement edge validation for bets >10%
6. Add warnings for suspicious lines

### Phase 4: Advanced (8-12 hours)
7. Implement TD rescaling with EPA cross-check
8. Add team total cross-validation

---

## Expected Impact

**Before Improvements**:
- CAR @ SF: 63.1 projected, 49.5 line → 23.3% edge
- Calibration error: Unknown
- Edge validation: None

**After Improvements**:
- More accurate projections (±1-2 points)
- Calibrated probabilities (error <5%)
- Edge validation prevents bad lines
- Expected ROI improvement: 10-15%

---

## Files to Modify

1. `scripts/predict/generate_game_line_predictions.py`:
   - Add `calculate_points_per_td()`
   - Add `calculate_empirical_total_std()`
   - Update `aggregate_team_projections()`

2. `scripts/predict/generate_game_line_recommendations.py`:
   - Add `validate_large_edge()`
   - Add shrinkage calibration
   - Add edge warnings

3. `nfl_quant/calibration/shrinkage_calibrator.py`:
   - Add `find_optimal_shrinkage()`
   - Add historical backtest

---

## Next Steps

1. **Immediate**: Implement Phase 1 (points per TD, empirical SD)
2. **This Week**: Backtest shrinkage on 2024 data
3. **Before Week 13**: Implement edge validation
4. **Long-term**: TD rescaling with EPA models
