# NFL QUANT Feature Engineering Gap Analysis

**Generated**: 2025-12-05
**Analyst**: Senior Sports Analytics Engineer
**Trigger**: Rashee Rice prediction discrepancy (75.8% model confidence vs 25% WR1 hit rate vs Houston)

---

## Executive Summary

The NFL QUANT system has **significant feature engineering gaps** that explain prediction anomalies like the Rashee Rice case. The V17 classifier uses only **14 betting tendency features**, while rich player skill and matchup features are **calculated but dropped** before reaching the model.

### Critical Gaps Identified:
1. **Static catch rate** (V4 simulator uses hardcoded 0.65 for WR, ignores coverage quality)
2. **No depth-chart-specific defense** (WR1/WR2/WR3 treated identically)
3. **NGS features orphaned** (separation, cushion calculated but not in model)
4. **14-feature bottleneck** (classifier only sees betting biases, not player skills)

---

## Gap Inventory Table

| # | Gap | Current State | Ideal State | Impact | Effort | File(s) |
|---|-----|---------------|-------------|:------:|:------:|---------|
| **DEFENSIVE GAPS** |||||
| 1 | WR1-specific defense | All WRs grouped together | Separate WR1/WR2/WR3/slot stats | 5 | 4 | `defensive_metrics.py:119-221` |
| 2 | Coverage scheme | Not tracked | Man/Zone tendencies by down | 4 | 4 | N/A (needs PFF data) |
| 3 | CB matchup | Not tracked | Specific CB assignments | 3 | 5 | N/A (needs PFF data) |
| 4 | Reception count distribution | Not tracked | % of WR1s hitting 5+/6+/7+ | 5 | 2 | `defensive_metrics.py` |
| **DYNAMIC PLAYER GAPS** |||||
| 5 | Static catch rate | Hardcoded 0.65/0.70/0.77 | Dynamic based on coverage + player | 5 | 3 | `player_simulator_v4.py:807-810` |
| 6 | NGS separation not in model | Calculated, not used | In V17_FEATURE_COLS | 4 | 1 | `model_config.py:164-167` |
| 7 | NGS cushion not in model | Calculated, not used | In V17_FEATURE_COLS | 3 | 1 | `model_config.py:164-167` |
| 8 | YAC above expectation | Calculated, not used | In V17_FEATURE_COLS | 3 | 1 | `ngs_features.py` |
| **FEATURE INTEGRATION GAPS** |||||
| 9 | 14-feature bottleneck | Only betting biases used | Add 8-10 skill features | 5 | 3 | `model_config.py:148-170` |
| 10 | Catch rate from NFLverse | V3 had it, V4 removed | Restore dynamic lookup | 4 | 2 | `player_simulator_v4.py` |
| 11 | Snap share in model | Extracted, not in classifier | Add to V17_FEATURE_COLS | 3 | 1 | `model_config.py` |
| 12 | Target share in model | Extracted, not in classifier | Add to V17_FEATURE_COLS | 3 | 1 | `model_config.py` |
| **DATA UTILIZATION GAPS** |||||
| 13 | Depth charts underused | Loaded, not for defense | Use for WR1/WR2/WR3 mapping | 4 | 3 | `core.py:2416-2429` |
| 14 | NGS receiving underused | 2.3k records | Use separation in model | 4 | 2 | `ngs_features.py` |
| 15 | Participation data unused | 45.9k records | Route participation features | 3 | 3 | `route_metrics.py` |
| 16 | FF opportunity unused | 10k records | Expected vs actual regression | 3 | 3 | `ff_opportunity_features.py` |
| **MARKET-SPECIFIC GAPS** |||||
| 17 | Receptions: no catch rate × coverage | Static multiplication | catch_rate × coverage_quality | 5 | 3 | `player_simulator_v4.py` |
| 18 | Rush yards: no box count | Generic rush defense | box_defenders × scheme | 3 | 3 | `player_simulator_v4.py` |
| 19 | Pass yards: no protection | Generic pass offense | pressure_rate × time_to_throw | 3 | 4 | `player_simulator_v4.py` |

**Impact**: 1-5 (5 = highest impact on prediction accuracy)
**Effort**: 1-5 (5 = highest effort to implement)

---

## Priority Matrix

```
                    HIGH IMPACT
                         │
        ┌────────────────┼────────────────┐
        │ QUICK WINS     │ STRATEGIC      │
        │                │                │
        │ #6 NGS sep     │ #1 WR1 defense │
        │ #7 NGS cushion │ #5 Dynamic CR  │
        │ #9 Feature     │ #17 CR×Cov     │
        │    bottleneck  │ #4 Rec distrib │
LOW ────┼────────────────┼────────────────┼──── HIGH
EFFORT  │                │                │    EFFORT
        │ FILL-INS       │ DEFER          │
        │                │                │
        │ #11 Snap share │ #2 Coverage    │
        │ #12 Target     │ #3 CB matchup  │
        │ #8 YAC above   │ #19 Protection │
        │                │                │
        └────────────────┼────────────────┘
                         │
                    LOW IMPACT
```

---

## Detailed Analysis

### GAP #5: Static Catch Rate (CRITICAL)

**Location**: `nfl_quant/simulation/player_simulator_v4.py:807-810`

**Current Code**:
```python
def _get_catch_rate(self, player_input: PlayerPropInput) -> float:
    """Get position-specific catch rate."""
    defaults = {'RB': 0.77, 'WR': 0.65, 'TE': 0.70}  # STATIC!
    return defaults.get(player_input.position, 0.65)
```

**Problem**: ALL WRs get 0.65 catch rate regardless of:
- Opponent coverage quality (Houston allows only 25% of WR1s to hit 7+ receptions)
- Player's historical catch rate
- Game script (trailing = more short passes = higher catch rate)

**Impact on Rashee Rice**: Model assumed 65% catch rate when Houston's WR1 coverage suppresses it to ~50%.

**Fix**: Restore V3's `_get_catch_rate_from_data()` and add coverage adjustment:
```python
def _get_catch_rate(self, player_input: PlayerPropInput) -> float:
    # Base catch rate from player's trailing data
    base_rate = player_input.trailing_catch_rate or 0.65

    # Adjust for opponent coverage quality
    coverage_adj = 1.0 - (player_input.opp_position_def_epa * 0.1)

    return min(0.85, max(0.45, base_rate * coverage_adj))
```

---

### GAP #1: No WR1-Specific Defense (CRITICAL)

**Location**: `nfl_quant/features/defensive_metrics.py:119-221`

**Current Code**:
```python
elif position in ['WR', 'TE']:
    # Pass defense (coverage) - ALL WRs grouped together
    plays = plays[plays['target_player_id'].isin(
        pos_mapping[pos_mapping['position'] == position]['player_id']
    )]
```

**Problem**: A team's WR1 suppression ability is hidden in aggregate WR stats.
- Houston allows only 4.8 receptions/game to WR1s
- But their overall WR defense is average
- Model sees "average defense" → high confidence on OVER

**Fix**: Use depth charts to differentiate:
```python
# In defensive_metrics.py, add:
def get_defense_vs_wr_depth(
    self,
    defense_team: str,
    depth_rank: int,  # 1 = WR1, 2 = WR2, 3 = WR3
    current_week: int
) -> Dict[str, float]:
    """Defense performance against WR by depth chart position."""
    # Load depth charts for opponent history
    depth = self._load_depth_charts()

    # Get WR1s faced in last 4 weeks
    wr1s_faced = depth[
        (depth['position'] == 'WR') &
        (depth['depth_team'] == defense_team) &
        (depth['depth_chart_position'] == 1)
    ]

    # Calculate receptions allowed to WR1s specifically
    ...
```

---

### GAP #9: 14-Feature Bottleneck (CRITICAL)

**Location**: `configs/model_config.py:148-170`

**Current V17 Features** (only 14):
```python
V17_FEATURE_COLS = [
    'line_vs_trailing',         # Betting: line gap
    'line_level',               # Betting: line magnitude
    'line_in_sweet_spot',       # Betting: line range
    'player_under_rate',        # Betting: player tendency
    'player_bias',              # Betting: player bias
    'market_under_rate',        # Betting: market tendency
    'LVT_x_player_tendency',    # Betting: interaction
    'LVT_x_player_bias',        # Betting: interaction
    'LVT_x_regime',             # Betting: interaction
    'LVT_in_sweet_spot',        # Betting: interaction
    'market_bias_strength',     # Betting: market strength
    'player_market_aligned',    # Betting: alignment
    'lvt_x_defense',            # NEW: defense interaction
    'lvt_x_rest',               # NEW: rest interaction
]
```

**Problem**: ALL 14 features are betting tendency features! No player skill features:
- No `avg_separation` (calculated in NGS, not used)
- No `snap_share` (calculated, not used)
- No `trailing_catch_rate` (calculated, not used)
- No `target_share` (calculated, not used)

**Fix**: Add skill features to V17:
```python
V17_SKILL_FEATURES = [
    'avg_separation',           # NGS: receiver skill
    'trailing_catch_rate',      # Player: catch consistency
    'snap_share',               # Usage: role importance
    'target_share',             # Usage: target volume
    'opp_position_def_rank',    # Defense: matchup difficulty
]

V17_FEATURE_COLS = V12_FEATURE_COLS + V17_NEW_FEATURES + V17_SKILL_FEATURES
```

---

## Quick Wins (Effort ≤ 2, Impact ≥ 3)

### 1. Add NGS Separation to Model
**File**: `configs/model_config.py:164-167`
**Change**: Add `'avg_separation'` to `V17_NEW_FEATURES`
**Impact**: WRs with high separation (skill) differentiated from low separation

### 2. Add NGS Cushion to Model
**File**: `configs/model_config.py:164-167`
**Change**: Add `'avg_cushion'` to `V17_NEW_FEATURES`
**Impact**: Captures defensive alignment (tight coverage = harder catches)

### 3. Add Snap Share to Model
**File**: `configs/model_config.py:164-167`
**Change**: Add `'snap_share'` to `V17_NEW_FEATURES`
**Impact**: Distinguishes starters from backups

### 4. Add Target Share to Model
**File**: `configs/model_config.py:164-167`
**Change**: Add `'target_share'` to `V17_NEW_FEATURES`
**Impact**: Captures target concentration

### 5. Restore Dynamic Catch Rate
**File**: `nfl_quant/simulation/player_simulator_v4.py`
**Change**: Copy `_get_catch_rate_from_data()` from V3 archived version
**Impact**: Uses actual player catch rate instead of static 0.65

---

## Structural Changes Required

### 1. Depth-Chart-Aware Defense (Medium Effort)
**Files**:
- `nfl_quant/features/defensive_metrics.py` - Add WR depth chart integration
- `nfl_quant/features/core.py` - Add `get_defense_vs_wr_depth()` method

**Data Source**: `data/nflverse/depth_charts.parquet` (335k records, already loaded)

**New Features**:
- `def_vs_wr1_receptions_allowed`
- `def_vs_wr1_over_rate_7plus`
- `def_vs_wr2_receptions_allowed`

### 2. Coverage-Adjusted Catch Rate (Medium Effort)
**Files**:
- `nfl_quant/simulation/player_simulator_v4.py` - Modify `_get_catch_rate()`
- `nfl_quant/features/core.py` - Add coverage quality feature

**New Features**:
- `coverage_adjusted_catch_rate`
- `opp_wr1_catch_rate_allowed`

### 3. Model Retraining (Required after feature additions)
**Command**:
```bash
python scripts/train/train_v12_interaction_model_v2.py
```

**Note**: The training script already uses `V17_FEATURE_COLS` from config, so adding features to config will automatically include them in training.

---

## Validation: Would This Have Caught Rashee Rice?

If implemented, these features would have shown:

| Feature | Current Value | With Fix |
|---------|---------------|----------|
| `def_vs_wr1_receptions_allowed` | N/A | 4.8 |
| `def_vs_wr1_over_7_rate` | N/A | 0.25 (25%) |
| `coverage_adjusted_catch_rate` | 0.65 (static) | 0.52 (coverage-adjusted) |
| `avg_separation` | Not in model | In model (would show skill) |

**Result**: Model would have seen:
- Houston allows only 25% of WR1s to hit 7+ receptions
- Catch rate adjustment: 0.65 → 0.52 (Houston's WR1 suppression)
- Monte Carlo simulation would produce lower receptions projection
- Confidence on OVER 6.5 would drop from 75.8% to ~40%

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. Add NGS features to V17_FEATURE_COLS
2. Add snap_share and target_share to model
3. Retrain model with new features

### Phase 2: Dynamic Catch Rate (2-3 days)
1. Restore V3's `_get_catch_rate_from_data()`
2. Add coverage quality adjustment
3. Test with historical WR1 matchups

### Phase 3: Depth-Chart Defense (3-5 days)
1. Add WR1/WR2/WR3 differentiation to defensive_metrics.py
2. Create new features: `def_vs_wr1_receptions_allowed`, `def_vs_wr1_over_rate`
3. Integrate into prediction pipeline
4. Retrain model

### Phase 4: Full Integration (1 week)
1. Add all new features to model
2. Run walk-forward validation
3. Compare accuracy: old vs new features
4. Deploy if improvement confirmed

---

## Appendix: Feature Flow Diagram

```
DATA SOURCES                    FEATURE EXTRACTION              MODEL USAGE
─────────────                   ──────────────────              ───────────

weekly_stats.parquet ───────────┐
                                ├──► trailing_stats.py ────────► USED in classifier
snap_counts.parquet ────────────┤
                                │
ngs_receiving.parquet ──────────┼──► ngs_features.py ──────────► CALCULATED but NOT USED
                                │    (separation, cushion)
depth_charts.parquet ───────────┼──► defensive_metrics.py ─────► PARTIALLY USED
                                │    (position-level only)       (team-level EPA only)
                                │
participation.parquet ──────────┼──► route_metrics.py ─────────► NOT USED
                                │    (route participation)
ff_opportunity.parquet ─────────┴──► ff_opportunity_features ──► NOT USED
                                     (expected fantasy)
```

**The Gap**: Rich data flows through feature extraction → but only 14 betting tendency features reach the V17 classifier.

---

## Files Modified During Analysis

None (read-only analysis)

## Recommended Next Steps

1. **Immediate**: Add `avg_separation`, `snap_share` to V17_FEATURE_COLS and retrain
2. **This Week**: Restore dynamic catch rate from V3
3. **Next Week**: Implement WR1-specific defensive features
4. **Validate**: Run walk-forward on Weeks 5-13 to confirm improvement

---

*Generated by NFL QUANT Feature Engineering Gap Analysis*
