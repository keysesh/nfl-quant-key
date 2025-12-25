# NFL QUANT Pipeline Architecture

**Generated**: 2025-12-20
**Purpose**: Map all scripts, models, and data flows to identify what's used vs orphaned

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline.py                                      │
│                    (Main Entry Point)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  GROUP 1      │         │  GROUP 2        │         │  GROUP 3        │
│  Data Fetch   │         │  Predictions    │         │  Recommendations│
│  (PARALLEL)   │────────▶│  (SEQUENTIAL)   │────────▶│  (PARALLEL)     │
└───────────────┘         └─────────────────┘         └─────────────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────┐
                                                      │  GROUP 4        │
                                                      │  Dashboard      │
                                                      │  (SEQUENTIAL)   │
                                                      └─────────────────┘
```

---

## SCRIPTS CALLED BY PIPELINE

### Group 1: Data Fetching (PARALLEL)
| Script | Status | Output |
|--------|--------|--------|
| `scripts/fetch/fetch_nflverse_data.R` | ✅ USED | `data/nflverse/*.parquet` |
| `scripts/fetch/fetch_injuries_sleeper.py` | ✅ USED | `data/injuries/*.json` |
| `scripts/fetch/fetch_live_odds.py` | ✅ USED | `data/odds/live_odds_*.csv` |
| `scripts/fetch/fetch_nfl_player_props.py` | ✅ USED | `data/odds/odds_player_props_*.csv` |

### Group 2: Predictions (SEQUENTIAL)
| Script | Status | Output |
|--------|--------|--------|
| `scripts/fetch/check_data_freshness.py` | ✅ USED | Console warnings |
| `scripts/predict/generate_model_predictions.py` | ✅ USED | `reports/model_predictions_*.csv` |

### Group 3: Recommendations (PARALLEL)
| Script | Status | Output | Notes |
|--------|--------|--------|-------|
| `scripts/predict/generate_edge_recommendations.py` | ✅ USED (--edge-mode) | `reports/edge_recommendations_*.csv` | Default mode |
| `scripts/predict/generate_unified_recommendations_v3.py` | ⚠️ ALT | `reports/recommendations_*.csv` | Non-edge mode |
| `scripts/predict/generate_game_line_recommendations.py` | ✅ USED | `reports/game_line_recommendations.csv` | |
| `scripts/predict/generate_parlay_recommendations.py` | ✅ USED | `reports/parlay_recommendations.json` | Optional |

### Group 4: Dashboard (SEQUENTIAL)
| Script | Status | Output |
|--------|--------|--------|
| `scripts/dashboard/generate_pro_dashboard.py` | ✅ USED | `reports/pro_dashboard.html` |

---

## SCRIPTS NOT CALLED BY PIPELINE

### Fetch Scripts (Orphaned/Manual)
| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/fetch/capture_closing_lines.py` | ❌ ORPHANED | Capture lines at game time |
| `scripts/fetch/fetch_historical_props.py` | ❌ ORPHANED | Backfill historical odds |
| `scripts/fetch/fetch_nfl_weather.py` | ❌ ORPHANED | Weather data |

### Training Scripts (Manual - Run Before Pipeline)
| Script | Model Output | Status |
|--------|--------------|--------|
| `scripts/train/train_ensemble.py` | `lvt_edge_model.joblib`, `player_bias_edge_model.joblib` | ⚠️ MANUAL |
| `scripts/train/train_model.py` | `active_model.joblib` | ⚠️ MANUAL |
| `scripts/train/train_td_poisson_edge.py` | `td_poisson_edge.joblib` | ⚠️ MANUAL |
| `scripts/train/train_lvt_edge.py` | (part of ensemble) | ⚠️ MANUAL |
| `scripts/train/train_player_bias_edge.py` | (part of ensemble) | ⚠️ MANUAL |
| `scripts/train/train_qb_model.py` | `qb_passing_model.joblib` | ❓ UNKNOWN |
| `scripts/train/train_attd_ensemble.py` | `attd_ensemble.joblib` | ❓ UNKNOWN |
| `scripts/train/train_game_lines_model.py` | ? | ❓ UNKNOWN |
| `scripts/train/train_game_line_ml_models.py` | ? | ❓ UNKNOWN |
| `scripts/train/train_td_poisson_model.py` | `td_poisson_model.joblib` | ❓ UNKNOWN |
| `scripts/train/train_td_props_model.py` | ? | ❓ UNKNOWN |
| `scripts/train/train_efficiency_predictor_v2_with_defense.py` | `efficiency_predictor_v2_defense.joblib` | ❓ UNKNOWN |
| `scripts/train/train_usage_predictor_v4_with_defense.py` | `usage_predictor_v4_defense.joblib` | ❓ UNKNOWN |
| `scripts/train/retrain_calibrators.py` | `usage_*_calibrated.joblib` | ❓ UNKNOWN |
| `scripts/train/retrain_td_calibrators.py` | `td_rate_*_calibrated.joblib` | ❓ UNKNOWN |

### Prediction Scripts (Orphaned/Alt)
| Script | Status | Notes |
|--------|--------|-------|
| `scripts/predict/generate_recommendations.py` | ❌ ORPHANED | Old version |
| `scripts/predict/generate_hybrid_recommendations.py` | ❌ ORPHANED | Old version |
| `scripts/predict/generate_contrarian_picks.py` | ❌ ORPHANED | Experimental |
| `scripts/predict/generate_game_line_predictions.py` | ❓ UNKNOWN | vs recommendations? |
| `scripts/predict/predict_game_lines.py` | ❓ UNKNOWN | Duplicate? |

### Validation/Backtest Scripts (Manual)
| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/backtest/walk_forward_unified.py` | ⚠️ MANUAL | Walk-forward validation |
| `scripts/backtest/walk_forward_no_leakage.py` | ⚠️ MANUAL | Anti-leakage validation |
| `scripts/backtest/walk_forward_recalibrated.py` | ⚠️ MANUAL | Calibration testing |
| `scripts/validation/validate_pipeline_integration.py` | ⚠️ MANUAL | Integration tests |
| `scripts/validation/backtest_game_lines.py` | ⚠️ MANUAL | Game line backtest |

---

## MODEL DEPENDENCIES

### Core Models Used by Pipeline
```
generate_edge_recommendations.py
    ├── lvt_edge_model.joblib         ← train_ensemble.py
    ├── player_bias_edge_model.joblib ← train_ensemble.py
    └── td_poisson_edge.joblib        ← train_td_poisson_edge.py (if --include-td)

generate_model_predictions.py
    └── active_model.joblib           ← train_model.py
```

### Models Used by generate_model_predictions.py
```
✅ usage_predictor_v4_defense.joblib   ← train_usage_predictor_v4_with_defense.py
✅ efficiency_predictor_v2_defense.joblib ← train_efficiency_predictor_v2_with_defense.py
✅ td_rate_*_calibrated.joblib         ← retrain_td_calibrators.py
✅ usage_*_calibrated.joblib           ← retrain_calibrators.py
```

### Models Used by generate_edge_recommendations.py (with flags)
```
✅ attd_ensemble.joblib               ← train_attd_ensemble.py (--include-attd flag)
```

### Models of Unknown Status
```
❓ qb_passing_model.joblib            ← train_qb_model.py (may be unused)
❓ td_poisson_model.joblib            ← train_td_poisson_model.py (vs td_poisson_edge?)
❓ td_calibration_factors.joblib
❓ calibrated_epa_factor.joblib       ← calibrate_epa_factor.py
```

---

## TRAINING ORDER (When Retraining Everything)

```bash
# ═══════════════════════════════════════════════════════════════════
# FULL RETRAIN SEQUENCE (Run in this order)
# ═══════════════════════════════════════════════════════════════════

# Step 1: Train predictors used by Monte Carlo simulation
python scripts/train/train_usage_predictor_v4_with_defense.py      # → usage_predictor_v4_defense.joblib
python scripts/train/train_efficiency_predictor_v2_with_defense.py # → efficiency_predictor_v2_defense.joblib

# Step 2: Train calibrators
python scripts/train/retrain_calibrators.py            # → usage_*_calibrated.joblib (5 files)
python scripts/train/retrain_td_calibrators.py         # → td_rate_*_calibrated.joblib (2 files)

# Step 3: Train core XGBoost classifier
python scripts/train/train_model.py                    # → active_model.joblib

# Step 4: Train edge ensemble (LVT + Player Bias) - MOST IMPORTANT
python scripts/train/train_ensemble.py                 # → lvt_edge_model.joblib
                                                       # → player_bias_edge_model.joblib

# Step 5: Train TD models (optional, for --include-td flag)
python scripts/train/train_td_poisson_edge.py          # → td_poisson_edge.joblib

# Step 6: Train ATTD ensemble (optional, for --include-attd flag)
python scripts/train/train_attd_ensemble.py            # → attd_ensemble.joblib

# ═══════════════════════════════════════════════════════════════════
# AFTER TRAINING: Run pipeline
# ═══════════════════════════════════════════════════════════════════
python scripts/run_pipeline.py <WEEK> --edge-mode
```

### Quick Retrain (Edge Models Only)
```bash
# If you only changed edge thresholds in edge_config.py:
python scripts/train/train_ensemble.py
python scripts/predict/generate_edge_recommendations.py --week <WEEK>
python scripts/dashboard/generate_pro_dashboard.py --week <WEEK>
```

---

## DATA FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  NFLverse (R)          Sleeper API         Odds API                         │
│  ├── pbp.parquet       └── injuries.json   ├── live_odds_*.csv              │
│  ├── weekly_stats      └── depth_charts    └── odds_player_props_*.csv      │
│  ├── snap_counts                                                            │
│  └── rosters                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING DATA                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  data/backtest/combined_odds_actuals_ENRICHED.csv                           │
│  (37,461 historical bets with actuals)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODELS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  CORE (Used):                        UNKNOWN STATUS:                         │
│  ├── active_model.joblib             ├── qb_passing_model.joblib             │
│  ├── lvt_edge_model.joblib           ├── attd_ensemble.joblib                │
│  ├── player_bias_edge_model.joblib   ├── efficiency_predictor_*.joblib       │
│  └── td_poisson_edge.joblib          └── usage_*_calibrated.joblib           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUTS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  reports/                                                                    │
│  ├── edge_recommendations_week*.csv   (Player prop picks)                   │
│  ├── game_line_recommendations.csv    (Spread/total picks)                  │
│  ├── parlay_recommendations.json      (Parlay suggestions)                  │
│  └── pro_dashboard.html               (Visual dashboard)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ISSUES IDENTIFIED

### 1. Orphaned Scripts (Can Be Archived)
- `scripts/fetch/capture_closing_lines.py`
- `scripts/fetch/fetch_historical_props.py`
- `scripts/fetch/fetch_nfl_weather.py`
- `scripts/predict/generate_recommendations.py`
- `scripts/predict/generate_hybrid_recommendations.py`
- `scripts/predict/generate_contrarian_picks.py`

### 2. Unknown Model Usage
These models exist but it's unclear if they're used:
- `qb_passing_model.joblib`
- `attd_ensemble.joblib`
- `efficiency_predictor_v2_defense.joblib`
- `usage_predictor_v4_defense.joblib`
- All `*_calibrated.joblib` files

### 3. Duplicate/Conflicting Scripts
- `predict_game_lines.py` vs `generate_game_line_predictions.py` vs `generate_game_line_recommendations.py`
- `train_td_poisson_model.py` vs `train_td_poisson_edge.py`
- `train_game_lines_model.py` vs `train_game_line_ml_models.py`

### 4. Training Not Automated
All training scripts must be run manually before pipeline. Consider:
- Adding `--retrain` flag to pipeline
- Creating `scripts/train/train_all.py` master script

---

## RECOMMENDED CLEANUP

### Archive These (Move to `scripts/_archive/`)
```bash
mv scripts/fetch/capture_closing_lines.py scripts/_archive/fetch/
mv scripts/fetch/fetch_historical_props.py scripts/_archive/fetch/
mv scripts/fetch/fetch_nfl_weather.py scripts/_archive/fetch/
mv scripts/predict/generate_recommendations.py scripts/_archive/predict/
mv scripts/predict/generate_hybrid_recommendations.py scripts/_archive/predict/
mv scripts/predict/generate_contrarian_picks.py scripts/_archive/predict/
```

### Investigate These
```bash
# Check if these are used anywhere
grep -r "qb_passing_model" --include="*.py" .
grep -r "attd_ensemble" --include="*.py" .
grep -r "efficiency_predictor" --include="*.py" .
grep -r "usage_predictor" --include="*.py" .
```

---

## QUICK REFERENCE

### Full Retrain + Pipeline
```bash
# 1. Retrain all models
python scripts/train/train_model.py
python scripts/train/train_ensemble.py
python scripts/train/train_td_poisson_edge.py

# 2. Run pipeline
python scripts/run_pipeline.py <WEEK> --edge-mode
```

### Just Regenerate (No Retrain)
```bash
python scripts/predict/generate_edge_recommendations.py --week <WEEK>
python scripts/dashboard/generate_pro_dashboard.py --week <WEEK>
```

### Backtest New Thresholds
```bash
python scripts/backtest/walk_forward_unified.py
```
