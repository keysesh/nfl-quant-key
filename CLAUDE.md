# NFL QUANT Framework

**Status**: Production | **Updated**: 2025-12-11
**Current Model**: V24 | **Features**: 46

---

## Quick Start

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# RECOMMENDED: Run full pipeline
python scripts/run_pipeline.py <WEEK>
```

### Manual Steps (if needed)
```bash
# 1. Refresh NFLverse data
Rscript scripts/fetch/fetch_nflverse_data.R

# 2. Fetch injuries (Sleeper API)
python scripts/fetch/fetch_injuries_sleeper.py

# 3. Check data freshness
python scripts/fetch/check_data_freshness.py

# 4. Fetch odds
python scripts/fetch/fetch_live_odds.py
python scripts/fetch/fetch_nfl_player_props.py

# 5. Generate predictions
python scripts/predict/generate_model_predictions.py <WEEK>

# 6. Generate recommendations
python scripts/predict/generate_unified_recommendations_v3.py --week <WEEK>

# 7. Generate dashboard
python scripts/dashboard/generate_pro_dashboard.py
```

---

## Centralized Configuration

**All model configuration lives in `configs/model_config.py`:**

| Setting | Description |
|---------|-------------|
| `MODEL_VERSION` | Current version (24) |
| `FEATURES` | 46 feature columns |
| `CLASSIFIER_MARKETS` | Markets for training |
| `MODEL_PARAMS` | XGBoost hyperparameters |
| `MARKET_SNR_CONFIG` | Signal-to-noise thresholds |

**To upgrade versions:**
1. Change `MODEL_VERSION` in `configs/model_config.py`
2. Add features to `FEATURES` list
3. Run `python scripts/train/train_model.py`

---

## Environment Variables

Required in `.env`:
```bash
ODDS_API_KEY=your_key_here  # From the-odds-api.com
```

Optional:
```bash
ENABLE_REGIME_DETECTION=1   # Enable regime-based adjustments
CURRENT_WEEK=14             # Override auto-detected week
CURRENT_SEASON=2025         # Override auto-detected season
```

---

## Key Files

| Component | File |
|-----------|------|
| **Model Config** | `configs/model_config.py` |
| **Feature Extraction** | `nfl_quant/features/batch_extractor.py` |
| **Feature Defaults** | `nfl_quant/features/feature_defaults.py` |
| **Opponent Features** | `nfl_quant/features/opponent_features.py` |
| **Active Model** | `data/models/active_model.joblib` |
| **Training Data** | `data/backtest/combined_odds_actuals_ENRICHED.csv` |
| **Weekly Stats** | `data/nflverse/weekly_stats.parquet` |
| **Pipeline Runner** | `scripts/run_pipeline.py` |
| **Model Training** | `scripts/train/train_model.py` |
| **Recommendations** | `scripts/predict/generate_unified_recommendations_v3.py` |

---

## Data Freshness Requirements

| Data | Max Age | Source |
|------|---------|--------|
| weekly_stats.parquet | 12h | NFLverse (R) |
| snap_counts.parquet | 12h | NFLverse (R) |
| depth_charts.parquet | 12h | NFLverse (R) |
| rosters.parquet | 24h | NFLverse (R) |
| injuries.parquet | 6h | Sleeper API |
| odds_player_props_*.csv | 4h | Odds API |

**Refresh command:**
```bash
Rscript scripts/fetch/fetch_nflverse_data.R
```

---

## Feature Architecture

### Current Features (V24 - 46 total)

**Core V12 (12):** line_vs_trailing, line_level, line_in_sweet_spot, player_under_rate, player_bias, market_under_rate, LVT_x_*, market_bias_strength, player_market_aligned

**V17 Skill (8):** lvt_x_defense, lvt_x_rest, avg_separation, avg_cushion, trailing_catch_rate, snap_share, target_share, opp_wr1_receptions_allowed

**V18 Context (7):** game_pace, vegas_total, vegas_spread, implied_team_total, adot, pressure_rate, opp_pressure_rate

**V19 Rush/Rec (4):** oline_health_score, box_count_expected, slot_snap_pct, target_share_trailing

**V23 Opponent (4):** opp_pass_def_vs_avg, opp_rush_def_vs_avg, opp_def_epa, has_opponent_context

**V24 Position Matchup (11):** pos_rank, is_starter, is_slot_receiver, slot_alignment_pct, opp_position_yards_allowed_trailing, opp_position_volume_allowed_trailing, opp_man_coverage_rate_trailing, slot_funnel_score, man_coverage_adjustment, position_role_x_opp_yards, has_position_context

### Feature Defaults

All features have semantic defaults in `nfl_quant/features/feature_defaults.py`:
- **NEVER use `.fillna(0)`** for model features
- Use `safe_fillna(df, FEATURE_DEFAULTS)` instead
- XGBoost handles NaN natively for opponent features

---

## Supported Markets

**Enabled for betting:**
- `player_receptions` (HIGH SNR)
- `player_rush_yds` (HIGH SNR)
- `player_reception_yds` (MEDIUM SNR)
- `player_rush_attempts` (HIGH SNR)

**Disabled (see `MARKET_SNR_CONFIG`):**
- `player_pass_yds` - Low correlation with actuals
- `player_pass_completions` - ~0 predictive power
- `player_pass_attempts` - Can't distinguish starters
- `player_pass_tds` - Binary distribution wrong for XGBoost

---

## Protected Fixes (DO NOT REVERT)

### Fix 1: Enriched Training Data (Dec 7, 2025)
- **File**: `scripts/train/train_model.py:62-69`
- **What**: Uses `combined_odds_actuals_ENRICHED.csv` with vegas_total, vegas_spread, opponent, target_share
- **Impact**: target_share importance went 0% → 17%

### Fix 2: Opponent Merge Collision (Dec 7, 2025)
- **File**: `nfl_quant/features/batch_extractor.py:776-849`
- **What**: Don't pre-initialize opponent columns before merge
- **Impact**: has_opponent_context importance went 0% → 9.7%

### Fix 3: V23_OPPONENT_FEATURES Column Names (Dec 7, 2025)
- **File**: `nfl_quant/features/opponent_features.py:284-292`
- **What**: Column names match calculate_opponent_trailing_defense output

### Fix 9: slot_snap_pct from PBP Data (Dec 13, 2025)
- **File**: `scripts/predict/generate_model_predictions.py:454-479, 690, 756, 1022, 1325`
- **What**: Calculates slot alignment % from PBP `pass_location` (middle=slot, left/right=outside)
- **Impact**: Slot receivers (e.g., Amon-Ra St. Brown) now use actual slot% instead of position defaults

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No predictions generated | Check data freshness, verify odds file exists |
| Model not found | Run `python scripts/train/train_model.py` |
| Stale data | Run `Rscript scripts/fetch/fetch_nflverse_data.R` |
| ODDS_API_KEY error | Add key to `.env` file |
| Opponent context 0% | Check batch_extractor.py merge collision fix |
| target_share 0% importance | Verify ENRICHED.csv has target_share_stats |

### Diagnostic Commands

```bash
# Check model health
python -c "
import joblib
m = joblib.load('data/models/active_model.joblib')
print(f'Version: {m[\"version\"]}')
print(f'Trained: {m[\"trained_date\"]}')
imp = dict(zip(m['models']['player_receptions'].feature_names_in_,
               m['models']['player_receptions'].feature_importances_))
for f in ['target_share', 'has_opponent_context']:
    print(f'{f}: {imp.get(f,0):.1%}')
"

# Check data health
python -c "
import pandas as pd
e = pd.read_csv('data/backtest/combined_odds_actuals_ENRICHED.csv')
w = pd.read_parquet('data/nflverse/weekly_stats.parquet')
print(f'Enriched: {len(e)} rows, opponent: {e[\"opponent\"].notna().mean():.1%}')
print(f'Weekly stats: {len(w)} rows, seasons: {sorted(w[\"season\"].unique())}')
"
```

---

## Directory Structure

```
NFL QUANT/
├── configs/
│   └── model_config.py        # Single source of truth for model config
├── data/
│   ├── backtest/              # Training data
│   │   └── combined_odds_actuals_ENRICHED.csv
│   ├── models/                # Trained models
│   │   └── active_model.joblib
│   └── nflverse/              # Source data (parquet files)
├── nfl_quant/                 # Core package
│   ├── features/
│   │   ├── batch_extractor.py  # Vectorized feature extraction
│   │   ├── feature_defaults.py # Safe fillna defaults
│   │   └── opponent_features.py # V23 opponent context
│   └── ...
├── scripts/
│   ├── fetch/                 # Data fetching
│   ├── predict/               # Prediction pipeline
│   ├── train/                 # Model training
│   ├── backtest/              # Walk-forward validation
│   ├── dashboard/             # Dashboard generation
│   └── tracking/              # Results tracking
└── reports/                   # Output files
```

---

## Betting Parameters

- **Unit size**: $5 base
- **Max bet**: $10 (2 units)
- Kelly sizing in dashboard is reference only - use fixed units

---

## Model Retraining

```bash
# Clear caches first
python -c "from nfl_quant.features.batch_extractor import clear_caches; clear_caches()"

# Train model (~30 seconds)
python scripts/train/train_model.py

# Verify
python scripts/predict/generate_unified_recommendations_v3.py --week <WEEK>
```

---

## Anti-Leakage Mechanisms

1. **`shift(1)` in trailing stats** - Only uses prior weeks
2. **`shift(1)` in opponent defense** - Only uses prior games
3. **Walk-forward validation** - Trains on `week < test_week - 1`
4. **Global week ordering** - Prevents temporal leakage across seasons

---

**Last Updated**: 2025-12-11
