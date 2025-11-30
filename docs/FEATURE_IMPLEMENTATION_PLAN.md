# NFL QUANT: Systematic Feature Implementation Plan

**Date:** 2025-11-29
**Expert Role:** Quantitative Sports Analytics Engineer / ML Systems Architect
**Goal:** Systematically integrate all 18+ feature modules into a unified, validated pipeline

---

## EXECUTIVE SUMMARY

### Current State Analysis

Your codebase has a **significant architectural gap**:

| Component | Status | Issue |
|-----------|--------|-------|
| Feature Modules (18+) | ✅ Built | Rich features calculated but disconnected |
| V12 Classifier | ✅ Working | Uses only 12 inline features, ignores modules |
| V14 Classifier | ✅ Working | Only uses LVT + rush_def_epa |
| FeatureAggregator | ✅ Built | Only used for simulation multipliers, not training |
| Unified Pipeline | ❌ Missing | No connection between modules → classifiers |

### The Core Problem

```
CURRENT FLOW (Broken):
┌─────────────────┐    ┌─────────────────┐
│ Feature Modules │    │ V12/V14 Train   │
│ (18+ modules)   │    │ (inline calcs)  │
└────────┬────────┘    └────────┬────────┘
         │                      │
         │ NOT CONNECTED        │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ FeatureAggr.    │    │ Classifier      │
│ (simulations)   │    │ Models          │
└─────────────────┘    └─────────────────┘

PROPOSED FLOW (Unified):
┌─────────────────┐
│ Unified Feature │
│ Engineering Hub │
├─────────────────┤
│ • All 18+ mods  │
│ • Validation    │
│ • No leakage    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Classifier      │◄───│ Walk-Forward    │
│ Training        │    │ Validation      │
└────────┬────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Production      │
│ Predictions     │
└─────────────────┘
```

---

## PHASE 1: UNIFIED FEATURE ENGINEERING HUB

### 1.1 Create Central Feature Registry

**File:** `nfl_quant/features/feature_registry.py`

```python
"""
Central registry for all feature extractors.
Controls which features are enabled per market.
"""

FEATURE_REGISTRY = {
    # TIER 1: VALIDATED FEATURES (currently working)
    'line_vs_trailing': {
        'module': 'inline',
        'markets': ['all'],
        'validated': True,
        'correlation': 0.49,
        'importance': 'PRIMARY',
    },
    'trailing_def_epa_rush': {
        'module': 'defensive_metrics',
        'markets': ['player_rush_yds'],
        'validated': True,
        'walk_forward_roi': 20.6,
    },
    'trailing_def_epa_pass': {
        'module': 'defensive_metrics',
        'markets': ['player_receptions'],
        'validated': True,
        'walk_forward_roi': 31.7,
    },

    # TIER 2: AVAILABLE BUT NOT VALIDATED
    'target_share': {
        'module': 'weekly_stats',
        'markets': ['player_receptions', 'player_reception_yds'],
        'validated': False,
        'expected_impact': 'MEDIUM',
    },
    'air_yards_share': {
        'module': 'weekly_stats',
        'markets': ['player_reception_yds'],
        'validated': False,
        'expected_impact': 'MEDIUM',
    },
    'route_participation': {
        'module': 'route_metrics',
        'markets': ['player_receptions', 'player_reception_yds'],
        'validated': False,
        'expected_impact': 'HIGH',
    },
    'tprr': {
        'module': 'route_metrics',
        'markets': ['player_receptions'],
        'validated': False,
        'expected_impact': 'HIGH',
    },
    'completion_pct_allowed': {
        'module': 'defensive_metrics',
        'markets': ['player_receptions'],
        'validated': False,
        'expected_impact': 'MEDIUM',
    },

    # TIER 3: AVAILABLE, NEEDS RESEARCH
    'red_zone_target_share': {
        'module': 'needs_implementation',
        'markets': ['player_receiving_tds'],
        'validated': False,
        'expected_impact': 'HIGH',
    },
    'cpoe': {
        'module': 'weekly_stats',
        'markets': ['player_pass_yds', 'player_completions'],
        'validated': False,
        'expected_impact': 'MEDIUM',
    },
    # ... etc for all 18+ features
}
```

### 1.2 Create Unified Feature Extractor

**File:** `nfl_quant/features/unified_extractor.py`

This module will:
1. Load all feature modules
2. Extract features for each player-week with NO data leakage
3. Merge features into a single DataFrame
4. Handle missing data gracefully

```python
class UnifiedFeatureExtractor:
    """
    Central hub for extracting ALL features for training and prediction.

    Key Design Principles:
    1. NO DATA LEAKAGE - all features use shift(1) or global_week < current
    2. MODULAR - features can be enabled/disabled per market
    3. CACHED - heavy calculations cached to avoid redundant work
    4. VALIDATED - only use features that pass walk-forward validation
    """

    def extract_all_features(
        self,
        backtest_df: pd.DataFrame,
        market: str,
        feature_config: dict = None
    ) -> pd.DataFrame:
        """
        Extract all enabled features for a market.

        Returns DataFrame with columns:
        - All original backtest columns
        - All enabled features for this market
        - global_week for temporal ordering
        """
        pass
```

---

## PHASE 2: FEATURE VALIDATION PIPELINE

### 2.1 Correlation Testing Framework

Before using any feature, test its correlation with prediction residuals:

```python
def test_feature_correlation(
    backtest_df: pd.DataFrame,
    market: str,
    feature_col: str,
    min_correlation: float = 0.05
) -> dict:
    """
    Test if a feature correlates with what trailing stats miss.

    Returns:
        {
            'feature': feature_col,
            'correlation': float,
            'significant': bool,
            'direction': 'positive' | 'negative' | 'neutral',
            'recommendation': str
        }
    """
```

### 2.2 Walk-Forward Validation

**CRITICAL**: Every feature must pass walk-forward validation before production.

```python
def walk_forward_validate_feature(
    backtest_df: pd.DataFrame,
    market: str,
    feature_col: str,
    base_features: list,
    test_weeks: int = 12,
    success_criteria: dict = None
) -> dict:
    """
    Walk-forward validation comparing model with vs without feature.

    SUCCESS CRITERIA (default):
    - ROI improvement > 3% OR
    - Sample size maintained with similar ROI
    - Coefficient sign makes logical sense
    - No significant degradation in other metrics

    Returns:
        {
            'feature': feature_col,
            'base_roi': float,
            'with_feature_roi': float,
            'improvement': float,
            'coefficient': float,
            'coefficient_sign_correct': bool,
            'validated': bool,
            'recommendation': str
        }
    """
```

### 2.3 Feature Validation Status Tracking

**File:** `data/feature_validation_status.json`

```json
{
    "last_updated": "2025-11-29",
    "features": {
        "trailing_def_epa_pass": {
            "validated_date": "2025-11-29",
            "markets": ["player_receptions"],
            "walk_forward_roi": 31.7,
            "sample_size": 145,
            "status": "VALIDATED"
        },
        "target_share": {
            "tested_date": "2025-11-29",
            "markets": ["player_receptions"],
            "correlation": -0.171,
            "status": "REJECTED",
            "reason": "Negative correlation - Vegas already prices this"
        }
    }
}
```

---

## PHASE 3: IMPLEMENTATION PRIORITY

### Priority 1: Immediate (Already Validated)

| Feature | Market | Status | Implementation |
|---------|--------|--------|----------------|
| `trailing_pass_def_epa` | receptions | Walk-forward: +31.7% ROI | Add to V12 for receptions |
| `trailing_rush_def_epa` | rush_yds | Walk-forward: +20.6% ROI | Already in V14 |

**Action:** Create V15 that combines V12 + pass_def_epa for receptions

### Priority 2: High Potential (Need Validation)

| Feature | Market | Source | Expected Impact |
|---------|--------|--------|-----------------|
| `route_participation` | receptions, rec_yds | route_metrics.py | HIGH |
| `tprr` | receptions | route_metrics.py | HIGH |
| `air_yards_share` | rec_yds | weekly_stats | MEDIUM-HIGH |
| `completion_pct_allowed` | receptions | defensive_metrics.py | MEDIUM |

### Priority 3: New Markets (Need Implementation + Validation)

| Feature | Market | Source | Implementation |
|---------|--------|--------|----------------|
| `red_zone_target_share` | receiving_tds | PBP calculation | New calculator needed |
| `goal_line_carry_share` | rushing_tds | PBP calculation | New calculator needed |
| `opponent_int_rate` | interceptions | PBP calculation | New calculator needed |

### Priority 4: Research Required

| Feature | Market | Notes |
|---------|--------|-------|
| `elo_diff` | All | May correlate with game script |
| `qb_pressure_rate` | pass_yds | Need to verify data availability |
| `snap_pct_in_red_zone` | TDs | Complex calculation |

---

## PHASE 4: IMPLEMENTATION STEPS

### Step 1: Create Unified Feature Extractor (Day 1)

```bash
# File: nfl_quant/features/unified_extractor.py
# Creates central hub that calls all feature modules
```

**Tasks:**
1. Create `UnifiedFeatureExtractor` class
2. Implement `extract_defensive_features()` using existing `defensive_metrics.py`
3. Implement `extract_route_features()` using existing `route_metrics.py`
4. Implement `extract_weekly_stats_features()` (target_share, air_yards_share)
5. Add temporal guards (NO LEAKAGE)

### Step 2: Create Feature Validation Script (Day 2)

```bash
# File: scripts/validate/validate_feature.py
# Usage: python scripts/validate/validate_feature.py --feature target_share --market player_receptions
```

**Tasks:**
1. Implement correlation testing
2. Implement walk-forward validation
3. Create validation report output
4. Update `feature_validation_status.json`

### Step 3: Validate Priority 2 Features (Day 3-4)

Run validation on each Priority 2 feature:

```bash
# Test each feature
python scripts/validate/validate_feature.py --feature route_participation --market player_receptions
python scripts/validate/validate_feature.py --feature tprr --market player_receptions
python scripts/validate/validate_feature.py --feature air_yards_share --market player_reception_yds
python scripts/validate/validate_feature.py --feature completion_pct_allowed --market player_receptions
```

### Step 4: Build V15 Unified Classifier (Day 5)

```bash
# File: scripts/train/train_v15_unified.py
# Combines V12 interaction model with validated defensive features
```

**Tasks:**
1. Integrate `UnifiedFeatureExtractor` into training
2. Use only VALIDATED features from registry
3. Walk-forward validation on all 2025 weeks
4. Compare to V12 and V14 baselines

### Step 5: Implement Red Zone Features (Day 6-7)

```bash
# File: nfl_quant/features/red_zone_features.py
# Calculates red zone usage metrics for TD predictions
```

**Tasks:**
1. Calculate `red_zone_target_share` from PBP
2. Calculate `goal_line_carry_share` from PBP
3. Calculate `opponent_rz_td_rate_allowed`
4. Validate for TD markets

### Step 6: Production Integration (Day 8)

**Tasks:**
1. Update `generate_model_predictions.py` to use V15
2. Update `generate_unified_recommendations_v3.py`
3. A/B test V15 vs V12 on Week 14
4. Full deployment if validated

---

## PHASE 5: DATA FLOW ARCHITECTURE

### Training Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES                                                          │
├──────────────────────────────────────────────────────────────────────┤
│ • combined_odds_actuals_2023_2024_2025.csv (backtest data)           │
│ • weekly_stats.parquet (player stats)                                 │
│ • pbp_*.parquet (play-by-play)                                        │
│ • schedules (game context)                                            │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ UNIFIED FEATURE EXTRACTOR                                             │
├──────────────────────────────────────────────────────────────────────┤
│ extract_all_features(backtest_df, market='player_receptions')        │
│                                                                       │
│ For each row:                                                         │
│   1. Get global_week                                                  │
│   2. Extract trailing stats (shift=1)                                 │
│   3. Extract defense EPA (week < current)                             │
│   4. Extract route metrics (week < current)                           │
│   5. Extract weekly stats features (week < current)                   │
│   6. Merge all features                                               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FEATURE MATRIX                                                        │
├──────────────────────────────────────────────────────────────────────┤
│ Columns:                                                              │
│ • player_norm, season, week, global_week                              │
│ • line, actual_stat, under_hit (target)                               │
│ • line_vs_trailing (PRIMARY)                                          │
│ • trailing_def_epa (VALIDATED)                                        │
│ • route_participation (if validated)                                  │
│ • tprr (if validated)                                                 │
│ • ... all enabled features                                            │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ V15 CLASSIFIER TRAINING                                               │
├──────────────────────────────────────────────────────────────────────┤
│ Walk-forward validation:                                              │
│   for test_week in 2025_weeks:                                        │
│     train = feature_matrix[global_week < test_week][-20 weeks]        │
│     test = feature_matrix[global_week == test_week]                   │
│     model.fit(train[features], train['under_hit'])                    │
│     predictions.append(model.predict_proba(test))                     │
│                                                                       │
│ Output: v15_unified_classifier.joblib                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Prediction Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ CURRENT WEEK ODDS (odds_player_props_*.csv)                           │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ UNIFIED FEATURE EXTRACTOR (same as training)                          │
│ extract_all_features(current_odds, market, week=13)                   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ V15 CLASSIFIER PREDICTION                                             │
│ p_under = model.predict_proba(features)                               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ RECOMMENDATIONS                                                        │
│ If p_under >= threshold: BET UNDER                                    │
│ Output: CURRENT_WEEK_RECOMMENDATIONS.csv                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 6: TESTING AND MONITORING

### 6.1 Feature Drift Detection

Monitor feature distributions to detect when features become stale:

```python
def detect_feature_drift(
    feature: str,
    historical_mean: float,
    historical_std: float,
    current_mean: float,
    threshold_z: float = 2.0
) -> bool:
    """Returns True if feature has drifted significantly."""
    z_score = abs(current_mean - historical_mean) / historical_std
    return z_score > threshold_z
```

### 6.2 ROI Tracking by Feature

Track which features contribute to ROI:

```python
{
    "week": 13,
    "market": "player_receptions",
    "total_roi": 35.2,
    "feature_contributions": {
        "line_vs_trailing": 0.45,
        "trailing_def_epa": 0.30,
        "route_participation": 0.15,
        "other": 0.10
    }
}
```

### 6.3 A/B Testing Framework

When adding new features, run parallel predictions:

```bash
# Week 14 predictions
V12 model: 15 bets, 10 wins (66.7%), ROI: +28%
V15 model: 18 bets, 13 wins (72.2%), ROI: +35%

# If V15 > V12 for 3 consecutive weeks → Full deployment
```

---

## SUCCESS METRICS

| Metric | Target | Measurement |
|--------|--------|-------------|
| Unified Extractor Coverage | 100% of feature modules | All 18+ modules callable |
| Feature Validation Rate | >50% of features tested | validation_status.json |
| Walk-Forward ROI (receptions) | >35% | Week 13+ performance |
| Walk-Forward ROI (rush_yds) | >15% | Week 13+ performance |
| New Market Coverage | 2+ TD markets | receiving_tds, rushing_tds |
| No Data Leakage | 0 leakage incidents | Temporal guard tests |

---

## APPENDIX: FILES TO CREATE

1. `nfl_quant/features/unified_extractor.py` - Central feature extraction hub
2. `nfl_quant/features/feature_registry.py` - Feature configuration and status
3. `scripts/validate/validate_feature.py` - Feature validation script
4. `scripts/train/train_v15_unified.py` - Unified classifier training
5. `nfl_quant/features/red_zone_features.py` - Red zone metric calculations
6. `data/feature_validation_status.json` - Validation tracking
7. `configs/feature_config.yaml` - Feature enablement by market

---

## APPENDIX: EXISTING MODULES TO INTEGRATE

| Module | Features | Integration Status |
|--------|----------|-------------------|
| `defensive_metrics.py` | pass/rush EPA, completion % allowed | PARTIAL (only rush) |
| `route_metrics.py` | TPRR, Y/RR, route participation | NOT USED |
| `enhanced_features.py` | EWMA spans, home/away, trends | NOT USED |
| `team_strength.py` | Elo ratings, win probability | NOT USED |
| `contextual_features.py` | Game script, weather | PARTIAL |
| `snap_count_features.py` | Snap share analysis | NOT USED |
| `ngs_features.py` | Next Gen Stats | NOT USED |
| `opponent_stats.py` | Player vs opponent history | WARNINGS ONLY |
| `trailing_stats.py` | 4-week EWMA trailing | USED |
| `matchup_features.py` | QB connections | NOT USED |
| `ff_opportunity_features.py` | Fantasy opportunity | NOT USED |
| `historical_baseline.py` | Career baselines | NOT USED |
| `role_change_detector.py` | Snap/target changes | NOT USED |
| `tier1_2_integration.py` | Tiered feature integration | NOT USED |
| `historical_injury_impact.py` | Injury impact on teammates | NOT USED |
| `injuries.py` | Injury report processing | PARTIAL |

---

## NEXT IMMEDIATE ACTION

**START HERE:**

```bash
# Step 1: Create the unified feature extractor
# This is the foundation for everything else

touch nfl_quant/features/unified_extractor.py
touch nfl_quant/features/feature_registry.py
```

Then implement `UnifiedFeatureExtractor.extract_defensive_features()` first since we already know defense EPA works for receptions (+31.7% ROI).
