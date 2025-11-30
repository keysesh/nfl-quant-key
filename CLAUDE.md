# NFL QUANT Framework

**Status**: V12 Production | Updated: 2025-11-30

---

## Quick Start

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# Step 0: Check data freshness (refresh if >12h stale)
python scripts/fetch/check_data_freshness.py

# Step 1: Generate predictions
python scripts/predict/generate_model_predictions.py <WEEK>

# Step 2: Generate recommendations  
python scripts/predict/generate_unified_recommendations_v3.py --week <WEEK>

# Step 3: After games - track results
python scripts/tracking/track_bet_results.py --week <WEEK>
```

---

## What to Bet (Validated Edge)

| Market | Threshold | Expected ROI | Action |
|--------|-----------|--------------|--------|
| **player_receptions** | 65% | +38% | ✅ BET |
| **player_reception_yds** | 55% | +7% | ✅ BET (marginal) |
| player_rush_yds | - | -3% | ❌ SKIP |
| player_pass_yds | - | ~0% | ❌ SKIP |

**Regime Warning**: Edge depends on UNDER-favoring market. Monitor weekly hit rates.

---

## Pipeline

```
Data Freshness Check
        ↓
NFLverse Data → FeatureEngine → Monte Carlo (10k) → V12 Model → Recommendations
                     ↑
              SINGLE SOURCE
               OF TRUTH
```

---

## Key Files

| Component | File |
|-----------|------|
| **FeatureEngine** | `nfl_quant/features/core.py` |
| Predictions | `scripts/predict/generate_model_predictions.py` |
| Recommendations | `scripts/predict/generate_unified_recommendations_v3.py` |
| Data Refresh | `scripts/fetch/check_data_freshness.py` |
| V12 Model | `data/models/v12_interaction_classifier.joblib` |
| Results Tracker | `scripts/tracking/track_bet_results.py` |

---

## Data Freshness

**Check before predictions** - stale data = bad predictions.

```bash
python scripts/fetch/check_data_freshness.py
```

| Data | Max Age | Why |
|------|---------|-----|
| weekly_stats.parquet | 12h | Player performance |
| snap_counts.parquet | 12h | Usage patterns |
| rosters_2025.csv | 24h | Team assignments |
| injuries_2025.csv | 6h | Game-day status |
| odds_player_props_*.csv | 4h | Current lines |

**Force refresh:**
```bash
python scripts/fetch/fetch_nflverse_extended.py
```

---

## Using FeatureEngine

All features MUST go through `core.py`. No inline calculations.

```python
from nfl_quant.features import get_feature_engine, calculate_trailing_stat

engine = get_feature_engine()

# Trailing stats (shift(1) automatic - no leakage)
df['trailing_rec'] = calculate_trailing_stat(df, 'receptions')

# Defense EPA
rush_def = engine.get_rush_defense_epa('KC', 2025, 13)

# Full feature extraction
features = engine.extract_features_for_bet(
    player_name="CeeDee Lamb",
    player_id="00-0036900",
    team="DAL", opponent="NYG",
    position="WR", market="player_receptions",
    line=6.5, season=2025, week=13,
    trailing_stat=5.5
)
```

---

## Retraining (If Needed)

```bash
# V12 interaction model
python scripts/train/train_v12_interaction_model_v2.py
```

---

## File Structure

```
NFL QUANT/
├── scripts/
│   ├── fetch/       # Data fetching
│   ├── predict/     # Prediction pipeline
│   ├── train/       # Model training
│   └── tracking/    # Results tracking
├── data/
│   ├── nflverse/    # Source data
│   ├── models/      # Trained models
│   └── backtest/    # Historical data
├── nfl_quant/
│   └── features/
│       └── core.py  # FeatureEngine
├── reports/         # Output
└── CLAUDE.md        # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No predictions | Check data freshness, verify odds file exists |
| Model not found | `python scripts/train/train_v12_interaction_model_v2.py` |
| Stale data | `python scripts/fetch/fetch_nflverse_extended.py` |

---

## Deprecated (Do Not Use)

- `train_v5/v6/v7/v8/v9/v10*.py` - Use V12
- Any script with inline EWMA/trailing calculations - Use FeatureEngine
- Old markdown docs in root - This file is source of truth