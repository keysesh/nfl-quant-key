# NFL QUANT - Quantitative NFL Betting Framework

A production-grade quantitative analytics pipeline for NFL player prop betting, featuring Monte Carlo simulation, V12 interaction models, and validated edge detection.

## Status

**Version**: V12 Production
**Season**: 2025
**Python**: 3.10 - 3.12

## Validated Edge

| Market | Threshold | Expected ROI | Recommendation |
|--------|-----------|--------------|----------------|
| **player_receptions** | 65% UNDER | +38% | BET |
| **player_reception_yds** | 55% UNDER | +7% | BET (marginal) |
| player_rush_yds | - | -3% | SKIP |
| player_pass_yds | - | ~0% | SKIP |

## Quick Start

```bash
# Setup
cd "/Users/keyonnesession/Desktop/NFL QUANT"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Step 0: Check data freshness (refresh if >12h stale)
python scripts/fetch/check_data_freshness.py

# Step 1: Generate predictions
python scripts/predict/generate_model_predictions.py <WEEK>

# Step 2: Generate recommendations
python scripts/predict/generate_unified_recommendations_v3.py --week <WEEK>

# Step 3: After games - track results
python scripts/tracking/track_bet_results.py --week <WEEK>
```

## Pipeline

```
Data Freshness Check
        ↓
NFLverse Data → FeatureEngine → Monte Carlo (10k) → V12 Model → Recommendations
                     ↑
              SINGLE SOURCE
               OF TRUTH
```

## Project Structure

```
NFL QUANT/
├── nfl_quant/              # Core package
│   ├── features/
│   │   └── core.py         # FeatureEngine (SINGLE SOURCE OF TRUTH)
│   ├── calibration/        # Probability calibration
│   ├── models/             # ML models
│   ├── simulation/         # Monte Carlo simulation
│   └── validation/         # Data validation
├── scripts/
│   ├── fetch/              # Data fetching (NFLverse via R)
│   ├── predict/            # Prediction pipeline
│   ├── train/              # Model training
│   └── tracking/           # Results tracking
├── data/
│   ├── nflverse/           # Source data (parquet/csv)
│   ├── models/             # Trained models (.joblib)
│   └── backtest/           # Historical validation
├── configs/                # Configuration files
├── reports/                # Output reports
└── tests/                  # Test suite
```

## Key Files

| Component | File |
|-----------|------|
| **FeatureEngine** | `nfl_quant/features/core.py` |
| Predictions | `scripts/predict/generate_model_predictions.py` |
| Recommendations | `scripts/predict/generate_unified_recommendations_v3.py` |
| Data Refresh | `scripts/fetch/check_data_freshness.py` |
| V12 Model | `data/models/v12_interaction_classifier.joblib` |
| Results Tracker | `scripts/tracking/track_bet_results.py` |

## Data Freshness

| Data | Max Age | Purpose |
|------|---------|---------|
| weekly_stats.parquet | 12h | Player performance |
| snap_counts.parquet | 12h | Usage patterns |
| rosters_2025.csv | 24h | Team assignments |
| injuries_2025.csv | 6h | Game-day status |
| odds_player_props_*.csv | 4h | Current lines |

## CLI Commands

```bash
# Generate predictions and recommendations
quant props --week 13

# Train V12 model
quant train-models

# Fetch data
quant fetch --season 2025 --data-type all
```

## Retraining

```bash
# V12 interaction model
python scripts/train/train_v12_interaction_model_v2.py
```

## Requirements

- Python 3.10 - 3.12
- R with nflreadr (for data fetching)
- See `requirements.txt` for Python dependencies

## License

MIT
