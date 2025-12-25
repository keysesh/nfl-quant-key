# NFL QUANT Dependency Map

**Generated**: December 2025 | **Model Version**: V24

This document provides a detailed view of all dependencies in the NFL QUANT system.

---

## 1. Internal Module Dependencies

### Core Package Structure

```
nfl_quant/
├── __init__.py
│   └── exports: DataFetcher, FeatureEngine, MonteCarloSimulator
│
├── config_paths.py (FOUNDATION - no internal deps)
│   └── exports: PROJECT_ROOT, DATA_DIR, MODELS_DIR, get_*_file()
│
├── config.py
│   └── imports: config_paths
│
├── constants.py (no deps)
│   └── exports: TEAM_ABBREVIATIONS, POSITION_GROUPS
│
└── schemas.py
    └── imports: pydantic
```

### Feature Engineering Module

```
nfl_quant/features/
│
├── batch_extractor.py (MAIN ORCHESTRATOR)
│   ├── imports: opponent_features
│   ├── imports: position_role
│   ├── imports: defense_vs_position
│   ├── imports: coverage_tendencies
│   ├── imports: feature_defaults
│   └── imports: configs.model_config (smooth_sweet_spot, FEATURE_FLAGS)
│
├── opponent_features.py
│   ├── imports: nfl_quant.config_paths
│   └── exports: calculate_opponent_trailing_defense, V23_OPPONENT_FEATURES
│
├── position_role.py (V24)
│   ├── imports: nfl_quant.config_paths
│   └── exports: load_depth_charts, get_all_position_roles_vectorized
│
├── defense_vs_position.py (V24)
│   ├── imports: nfl_quant.config_paths
│   └── exports: calculate_defense_vs_position_stats
│
├── coverage_tendencies.py (V24)
│   ├── imports: nfl_quant.config_paths
│   └── exports: calculate_team_coverage_tendencies
│
├── feature_defaults.py
│   └── exports: FEATURE_DEFAULTS, safe_fillna()
│
├── trailing_stats.py
│   └── exports: calculate_trailing_averages()
│
├── enhanced_features.py
│   ├── imports: trailing_stats
│   └── exports: extract_enhanced_features()
│
└── matchup_features.py
    ├── imports: nfl_quant.data.matchup_extractor
    └── exports: calculate_matchup_features()
```

### Models Module

```
nfl_quant/models/
│
├── classifier_registry.py
│   ├── imports: nfl_quant.config_paths
│   └── exports: register_model(), load_model()
│
├── bayesian_shrinkage.py
│   └── exports: BayesianShrinkageEstimator
│
└── td_predictor.py
    ├── imports: classifier_registry
    └── exports: TDPredictor
```

### Betting Module

```
nfl_quant/betting/
│
├── kelly_criterion.py
│   └── exports: calculate_kelly_fraction(), half_kelly()
│
├── bet_sizing.py
│   ├── imports: kelly_criterion
│   └── exports: size_bet()
│
└── draftkings_prop_rules.py
    └── exports: VALID_MARKETS, validate_prop()
```

### Data Module

```
nfl_quant/data/
│
├── fetcher.py
│   ├── imports: nfl_quant.config_paths
│   └── exports: DataFetcher
│
├── stats_loader.py
│   ├── imports: nfl_quant.config_paths
│   └── exports: load_weekly_stats(), load_snap_counts()
│
├── game_lines_loader.py
│   ├── imports: nfl_quant.config_paths
│   └── exports: load_game_lines()
│
└── matchup_extractor.py
    └── exports: extract_matchup_data()
```

### Utils Module

```
nfl_quant/utils/
│
├── player_names.py
│   └── exports: normalize_player_name(), fuzzy_match_player()
│
├── team_names.py
│   └── exports: normalize_team_name(), TEAM_MAPPING
│
├── season_utils.py
│   └── exports: get_current_week(), get_current_season()
│
├── nflverse_loader.py
│   ├── imports: nfl_quant.config_paths
│   └── exports: load_parquet(), load_nflverse_data()
│
└── epa_utils.py
    └── exports: calculate_epa()
```

---

## 2. Scripts Dependencies

### Pipeline Scripts

```
scripts/run_pipeline.py
├── imports: nfl_quant.config_paths (PROJECT_ROOT)
├── calls: Rscript scripts/fetch/fetch_nflverse_data.R
├── calls: python scripts/fetch/fetch_injuries_sleeper.py
├── calls: python scripts/fetch/check_data_freshness.py
├── calls: python scripts/fetch/fetch_live_odds.py
├── calls: python scripts/fetch/fetch_nfl_player_props.py
├── calls: python scripts/predict/generate_model_predictions.py
├── calls: python scripts/predict/generate_unified_recommendations_v3.py
├── calls: python scripts/predict/generate_game_line_recommendations.py
└── calls: python scripts/dashboard/generate_pro_dashboard.py
```

### Training Scripts

```
scripts/train/train_model.py
├── imports: nfl_quant.config_paths
├── imports: configs.model_config (MODEL_VERSION, FEATURES, CLASSIFIER_MARKETS, etc.)
├── imports: nfl_quant.features.batch_extractor
├── imports: nfl_quant.features.feature_defaults
├── imports: nfl_quant.utils.player_names
├── imports: nfl_quant.models.classifier_registry
├── reads: data/backtest/combined_odds_actuals_ENRICHED.csv
├── reads: data/nflverse/player_stats_2024_2025.csv
├── reads: data/nflverse/player_stats_2023.csv
├── reads: data/nflverse/weekly_stats.parquet
└── writes: data/models/active_model.joblib
```

### Prediction Scripts

```
scripts/predict/generate_model_predictions.py
├── imports: nfl_quant.config_paths
├── imports: configs.model_config
├── imports: nfl_quant.features.batch_extractor
├── reads: data/models/active_model.joblib
├── reads: data/nfl_player_props_draftkings.csv
├── reads: data/nflverse/weekly_stats.parquet
└── writes: data/model_predictions_week{N}.csv

scripts/predict/generate_unified_recommendations_v3.py
├── imports: nfl_quant.config_paths
├── imports: configs.model_config (MARKET_SNR_CONFIG, is_market_enabled)
├── imports: nfl_quant.features.batch_extractor
├── imports: nfl_quant.betting.kelly_criterion
├── reads: data/model_predictions_week{N}.csv
├── reads: data/nfl_player_props_draftkings.csv
├── writes: reports/recommendations_week{N}_*.csv
└── writes: data/picks/*_filtered.csv
```

### Fetch Scripts

```
scripts/fetch/fetch_nflverse_data.R
├── requires: R >= 4.0
├── requires: nflreadr package
├── writes: data/nflverse/weekly_stats.parquet
├── writes: data/nflverse/snap_counts.parquet
├── writes: data/nflverse/depth_charts.parquet
├── writes: data/nflverse/rosters.parquet
└── writes: data/nflverse/schedules.parquet

scripts/fetch/fetch_injuries_sleeper.py
├── imports: requests
├── calls: api.sleeper.app/v1/players/nfl
└── writes: data/nflverse/injuries.parquet

scripts/fetch/fetch_live_odds.py
├── imports: requests
├── requires: ODDS_API_KEY env var
├── calls: api.the-odds-api.com
└── writes: data/odds_week{N}.csv

scripts/fetch/fetch_nfl_player_props.py
├── imports: requests
├── requires: ODDS_API_KEY env var
├── calls: api.the-odds-api.com
└── writes: data/nfl_player_props_draftkings.csv
```

### Backtest Scripts

```
scripts/backtest/walk_forward_unified.py
├── imports: nfl_quant.features.batch_extractor
├── imports: configs.model_config
├── reads: data/backtest/combined_odds_actuals_ENRICHED.csv
└── writes: data/backtest/unified_validation_results.csv

scripts/backtest/walk_forward_no_leakage.py
├── imports: nfl_quant.features.batch_extractor
├── imports: configs.model_config
├── reads: data/backtest/combined_odds_actuals_ENRICHED.csv
└── writes: data/backtest/no_leakage_validation_results.csv
```

---

## 3. External Package Dependencies

### Core Dependencies (requirements.txt)

```
# Data Processing
pandas>=1.5.0,<2.0          # DataFrame operations
  └── used by: all feature extraction, all scripts
numpy>=1.24.0,<2.0          # Numerical operations
  └── used by: all feature extraction, model training
pyarrow>=12.0.0             # Parquet file I/O
  └── used by: nfl_quant.utils.nflverse_loader, all parquet reads
scipy>=1.10.0               # Statistical functions
  └── used by: nfl_quant.distributions, calibration

# Machine Learning
xgboost>=2.0.0              # Primary model framework
  └── used by: scripts/train/train_model.py, prediction scripts
scikit-learn>=1.3.0         # ML utilities, preprocessing
  └── used by: calibration, model evaluation
joblib>=1.3.0               # Model serialization
  └── used by: model save/load operations

# CLI & Configuration
typer>=0.9.0                # CLI framework
  └── used by: nfl_quant.cli
pydantic>=2.0.0             # Data validation
  └── used by: nfl_quant.schemas
pydantic-settings>=2.0.0    # Settings management
  └── used by: config loading
pyyaml>=6.0                 # YAML config parsing
  └── used by: config_loader.py

# API & Web
requests>=2.31.0            # HTTP client
  └── used by: all fetch scripts
tqdm>=4.65.0                # Progress bars
  └── used by: long-running scripts

# Optional
polars>=0.19.0              # Alternative DataFrame (faster for some ops)
  └── used by: some feature extraction paths
lightgbm>=4.0.0             # Alternative ML framework
  └── used by: experimental models (not in production)
```

### R Dependencies

```
R (>= 4.0)
├── nflreadr                # NFLverse data access
├── arrow                   # Parquet file writing
└── tidyverse (optional)    # Data manipulation
```

---

## 4. Data File Dependencies

### Training Data Dependencies

```
Training Pipeline requires:
│
├── data/backtest/combined_odds_actuals_ENRICHED.csv
│   ├── columns: player, season, week, market, line, actual_stat, outcome
│   ├── columns: opponent, vegas_total, vegas_spread, target_share_stats
│   └── source: manually enriched from raw odds + schedule data
│
├── data/nflverse/weekly_stats.parquet
│   ├── columns: player_id, season, week, receptions, receiving_yards, etc.
│   └── source: fetch_nflverse_data.R
│
├── data/nflverse/player_stats_2024_2025.csv
│   └── source: fetch_nflverse_data.R
│
└── data/nflverse/player_stats_2023.csv
    └── source: fetch_nflverse_data.R
```

### Prediction Data Dependencies

```
Prediction Pipeline requires:
│
├── data/models/active_model.joblib
│   └── source: scripts/train/train_model.py
│
├── data/nfl_player_props_draftkings.csv
│   └── source: fetch_nfl_player_props.py
│
├── data/nflverse/weekly_stats.parquet
│   └── source: fetch_nflverse_data.R
│
├── data/nflverse/snap_counts.parquet
│   └── source: fetch_nflverse_data.R
│
├── data/nflverse/injuries.parquet
│   └── source: fetch_injuries_sleeper.py
│
├── data/nflverse/depth_charts.parquet (V24)
│   └── source: fetch_nflverse_data.R
│
└── data/odds_week{N}.csv
    └── source: fetch_live_odds.py
```

---

## 5. External API Dependencies

### The Odds API

```
Endpoint: https://api.the-odds-api.com/v4/
Authentication: API key via ?apiKey= parameter
Rate Limit: Varies by plan (check remaining quota in response headers)

Used by:
├── scripts/fetch/fetch_live_odds.py
│   └── GET /sports/americanfootball_nfl/odds
│
└── scripts/fetch/fetch_nfl_player_props.py
    └── GET /sports/americanfootball_nfl/events/{eventId}/odds
        └── markets=player_receptions,player_reception_yds,player_rush_yds,...

Required env var: ODDS_API_KEY
```

### Sleeper API

```
Endpoint: https://api.sleeper.app/v1/
Authentication: None required
Rate Limit: Reasonable use (no documented limit)

Used by:
└── scripts/fetch/fetch_injuries_sleeper.py
    └── GET /players/nfl
```

### NFLverse (via R)

```
Endpoint: Various GitHub releases, S3 buckets
Authentication: None required

Used by:
└── scripts/fetch/fetch_nflverse_data.R
    ├── nflreadr::load_player_stats()
    ├── nflreadr::load_snap_counts()
    ├── nflreadr::load_rosters()
    ├── nflreadr::load_depth_charts()
    └── nflreadr::load_schedules()
```

---

## 6. Environment Dependencies

### Required Environment Variables

```bash
# .env file
ODDS_API_KEY=your_api_key_here    # Required for odds fetching

# Optional overrides
ENABLE_REGIME_DETECTION=1         # Enable regime-based adjustments
CURRENT_WEEK=15                   # Override auto-detected week
CURRENT_SEASON=2025               # Override auto-detected season
```

### System Requirements

```
Python: 3.10 - 3.12 (3.13 not yet supported)
R: >= 4.0 (for NFLverse data fetching)
OS: macOS, Linux (Windows untested)
Memory: >= 8GB recommended for training
Disk: ~2GB for data files
```

---

## 7. Dependency Graph (Visual)

```
                              ┌─────────────────────┐
                              │   External APIs     │
                              │  ┌───────────────┐  │
                              │  │ Odds API      │  │
                              │  │ Sleeper API   │  │
                              │  │ NFLverse (R)  │  │
                              │  └───────────────┘  │
                              └─────────┬───────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           DATA FILES                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │ nflverse/*.pqt  │  │ *_props.csv     │  │ injuries.pqt    │           │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘           │
└───────────┼─────────────────────┼────────────────────┼────────────────────┘
            │                     │                    │
            └─────────────────────┼────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       configs/model_config.py                              │
│                   (FEATURES, MODEL_PARAMS, MARKETS)                        │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │ batch_      │ │ train_      │ │ generate_   │
          │ extractor   │ │ model.py    │ │ recommend.  │
          └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  active_model.joblib   │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  reports/*.csv         │
                    │  data/picks/*.csv      │
                    └────────────────────────┘
```

---

**Last Updated**: December 2025
