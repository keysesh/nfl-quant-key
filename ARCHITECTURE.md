# NFL QUANT Architecture

**Version**: 1.2 | **Last Updated**: December 15, 2025 | **Model Version**: V27

---

## 1. System Overview

NFL QUANT is a quantitative sports analytics pipeline that predicts NFL player prop outcomes using multiple edge strategies:

1. **Edge Ensemble (LVT + Player Bias)**: XGBoost classifiers for continuous stats (yards, receptions, attempts)
2. **TD Poisson Edge**: Poisson regression for touchdown props (count data)

The system ingests data from multiple sources (NFLverse, Odds API, Sleeper), engineers 46+ features capturing player performance, game context, and opponent matchups, then generates probability-weighted betting recommendations.

**Core Problem Solved**: Converting raw NFL statistics into actionable betting recommendations with positive expected value.

**Key Inputs**:
- Player weekly statistics (NFLverse via R)
- Live odds and player props (The Odds API)
- Injury reports (Sleeper API)
- Historical betting outcomes (backtest data)

**Key Outputs**:
- Model predictions with confidence scores
- Ranked betting recommendations with Kelly-sized units
- Pro dashboard summarizing top picks

### Edge System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EDGE SYSTEM                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            CONTINUOUS STATS (Yards, Rec, Att)       │    │
│  │  ┌───────────────┐       ┌───────────────────┐      │    │
│  │  │   LVT Edge    │       │  Player Bias Edge │      │    │
│  │  │ (Line vs      │       │  (Historical      │      │    │
│  │  │  Trailing)    │       │   Tendencies)     │      │    │
│  │  │ 65-70% @ low  │       │  55-60% @ high    │      │    │
│  │  │ volume        │       │  volume           │      │    │
│  │  └───────┬───────┘       └─────────┬─────────┘      │    │
│  │          └──────────┬──────────────┘                │    │
│  │                     ▼                               │    │
│  │              EdgeEnsemble                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            TOUCHDOWN PROPS (Count Data)             │    │
│  │  ┌───────────────────────────────────────────┐      │    │
│  │  │           TD Poisson Edge                 │      │    │
│  │  │  - Poisson regression for TD counts       │      │    │
│  │  │  - P(over)/P(under) via Poisson CDF       │      │    │
│  │  │  - Red zone features                      │      │    │
│  │  │  - 62% hit rate @ 58%+ confidence         │      │    │
│  │  └───────────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA INGESTION LAYER                             │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   NFLverse (R)  │   Odds API      │   Sleeper API   │   Manual Overrides    │
│   ┌───────────┐ │   ┌───────────┐ │   ┌───────────┐ │   ┌───────────────┐   │
│   │weekly_    │ │   │player     │ │   │injuries   │ │   │snap_overrides │   │
│   │stats.pqt  │ │   │props.csv  │ │   │.parquet   │ │   │.json          │   │
│   │snap_counts│ │   │game_lines │ │   └───────────┘ │   └───────────────┘   │
│   │rosters    │ │   │.csv       │ │                 │                       │
│   │depth_charts││   └───────────┘ │                 │                       │
│   └───────────┘ │                 │                 │                       │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         └─────────────────┴─────────────────┴─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE ENGINEERING LAYER                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    batch_extractor.py                                │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │   │
│  │  │ Core V12     │ │ V17 Skill    │ │ V18 Context  │ │ V23 Opponent│ │   │
│  │  │ Features     │ │ Features     │ │ Features     │ │ Features    │ │   │
│  │  │ (12 cols)    │ │ (8 cols)     │ │ (7 cols)     │ │ (4 cols)    │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ │   │
│  │  ┌──────────────┐ ┌──────────────┐                                   │   │
│  │  │ V19 Rush/Rec │ │ V24 Position │                                   │   │
│  │  │ (4 cols)     │ │ Matchup (11) │                                   │   │
│  │  └──────────────┘ └──────────────┘                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                              46 Features                                    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    active_model.joblib                               │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           │   │
│  │  │ XGBClassifier  │ │ XGBClassifier  │ │ XGBClassifier  │  ...      │   │
│  │  │ player_        │ │ player_rush_   │ │ player_        │           │   │
│  │  │ receptions     │ │ yds            │ │ reception_yds  │           │   │
│  │  └────────────────┘ └────────────────┘ └────────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                            P(UNDER) scores                                  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              generate_unified_recommendations_v3.py                  │   │
│  │                                                                      │   │
│  │  1. Calculate edge = model_prob - implied_prob                       │   │
│  │  2. Apply SNR filters (market-specific thresholds)                   │   │
│  │  3. Kelly criterion sizing (capped at 2 units)                       │   │
│  │  4. Rank by confidence × edge                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                         Ranked Recommendations                              │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ reports/     │  │ data/picks/  │  │ Dashboard    │                      │
│  │ recommendations│  │ *_filtered   │  │ (HTML)       │                      │
│  │ _week*.csv   │  │ .csv         │  │              │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
NFL QUANT/
├── configs/                    # Configuration files
│   └── model_config.py         # SINGLE SOURCE OF TRUTH for model version, features, params
│
├── data/                       # All data files (gitignored except schemas)
│   ├── backtest/               # Historical odds + actuals for training
│   │   └── combined_odds_actuals_ENRICHED.csv  # Primary training data
│   ├── models/                 # Trained model artifacts
│   │   └── active_model.joblib # Production model
│   ├── nflverse/               # NFLverse parquet/CSV files
│   │   ├── weekly_stats.parquet
│   │   ├── snap_counts.parquet
│   │   ├── depth_charts.parquet
│   │   └── rosters.parquet
│   ├── injuries/               # Injury reports by week
│   ├── picks/                  # Filtered betting picks
│   └── recommendations/        # Full recommendation outputs
│
├── nfl_quant/                  # Core Python package
│   ├── __init__.py             # Package exports
│   ├── config_paths.py         # Centralized path configuration
│   ├── features/               # Feature engineering modules
│   │   ├── batch_extractor.py  # Main vectorized feature extraction
│   │   ├── opponent_features.py # V23 opponent defense features
│   │   ├── position_role.py    # V24 position matchup features
│   │   ├── red_zone_features.py # Red zone snap allocation (for TD props)
│   │   ├── trailing_stats.py   # EWMA trailing statistics
│   │   └── feature_defaults.py # Safe NaN handling
│   ├── edges/                  # Edge detection strategies
│   │   ├── lvt_edge.py         # Line vs Trailing edge
│   │   ├── player_bias_edge.py # Player historical tendency edge
│   │   ├── ensemble.py         # EdgeEnsemble combining edges
│   │   └── calibration.py      # Probability calibration
│   ├── models/                 # Model utilities
│   │   ├── classifier_registry.py
│   │   └── td_poisson_edge.py  # Poisson regression for TD props
│   ├── betting/                # Bet sizing and Kelly criterion
│   ├── calibration/            # Probability calibration
│   ├── simulation/             # Monte Carlo simulation (legacy)
│   └── utils/                  # Shared utilities
│
├── scripts/                    # Executable scripts
│   ├── run_pipeline.py         # MAIN ENTRY POINT - runs full pipeline
│   ├── fetch/                  # Data fetching scripts
│   │   ├── fetch_nflverse_data.R    # R script for NFLverse
│   │   ├── fetch_injuries_sleeper.py
│   │   ├── fetch_live_odds.py
│   │   └── fetch_nfl_player_props.py
│   ├── train/                  # Model training
│   │   ├── train_model.py      # XGBoost classifier training
│   │   ├── train_ensemble.py   # Train LVT + Player Bias edges
│   │   ├── train_lvt_edge.py   # Train LVT edge only
│   │   ├── train_player_bias_edge.py # Train Player Bias edge only
│   │   └── train_td_poisson_edge.py  # Train TD Poisson edge
│   ├── predict/                # Prediction generation
│   │   ├── generate_model_predictions.py
│   │   ├── generate_unified_recommendations_v3.py
│   │   └── generate_edge_recommendations.py  # Edge-based picks (incl. TD)
│   ├── backtest/               # Walk-forward validation
│   │   ├── walk_forward_unified.py
│   │   └── walk_forward_no_leakage.py
│   └── dashboard/              # Dashboard generation
│       └── generate_pro_dashboard.py
│
├── reports/                    # Generated outputs (gitignored)
│   └── recommendations_week*.csv
│
├── tests/                      # Test suite
│
├── .claude/                    # Claude Code configuration
│   └── CLAUDE.md               # Project-specific AI instructions
│
├── CLAUDE.md                   # Public project documentation
├── ARCHITECTURE.md             # This file
└── pyproject.toml              # Package configuration
```

---

## 4. Core Components

### 4.1 Configuration (`configs/model_config.py`)

**Purpose**: Single source of truth for all model configuration.

**Key Contents**:
- `MODEL_VERSION = "27"` - Current model version
- `FEATURES` - List of 46 feature column names
- `CLASSIFIER_MARKETS` - Markets enabled for training
- `MARKET_SNR_CONFIG` - Signal-to-noise thresholds per market
- `MARKET_FILTERS` - V27 game context filters (spread, position, bye week)
- `MODEL_PARAMS` - XGBoost hyperparameters

**Dependencies**: numpy, pathlib

**Why This Design**: Centralizing configuration eliminates version number drift and ensures all scripts use consistent feature lists.

---

### 4.2 Feature Engineering (`nfl_quant/features/`)

**Purpose**: Transform raw data into model-ready features.

**Key Files**:
| File | Features | Description |
|------|----------|-------------|
| `batch_extractor.py` | All 46 | Vectorized extraction orchestrator |
| `opponent_features.py` | V23 (4) | Opponent defensive strength |
| `position_role.py` | V24 (11) | Position-specific matchups |
| `feature_defaults.py` | N/A | Safe NaN handling (no `.fillna(0)`) |
| `trailing_stats.py` | Core (12) | EWMA player trailing averages |

**Dependencies**: pandas, numpy, configs.model_config

**Outputs**: DataFrame with 46 feature columns per row

**Critical Design Decision**: XGBoost handles NaN natively, so opponent features are left as NaN when unavailable rather than filled with potentially misleading defaults.

---

### 4.3 Model Training (`scripts/train/train_model.py`)

**Purpose**: Train market-specific XGBoost classifiers.

**Key Steps**:
1. Load enriched training data (odds + actuals)
2. Prepare trailing stats from NFLverse
3. Extract 46 features per row
4. Train separate classifier per market
5. Save to `active_model.joblib`

**Dependencies**:
- `nfl_quant.features.batch_extractor`
- `configs.model_config`
- xgboost, pandas, joblib

**Outputs**: `data/models/active_model.joblib` containing:
```python
{
    'version': 'V24',
    'trained_date': '2025-12-07',
    'features': [...46 feature names...],
    'models': {
        'player_receptions': XGBClassifier,
        'player_rush_yds': XGBClassifier,
        ...
    }
}
```

---

### 4.4 Prediction Pipeline (`scripts/predict/`)

**Purpose**: Generate predictions for current week.

**Key Files**:
| File | Purpose |
|------|---------|
| `generate_model_predictions.py` | Raw P(UNDER) scores |
| `generate_unified_recommendations_v3.py` | Filtered, ranked picks |

**Flow**:
```
Live Odds → Feature Extraction → Model Inference → Edge Calculation → SNR Filtering → Kelly Sizing → Ranked Output
```

**Dependencies**:
- `data/models/active_model.joblib`
- `data/nfl_player_props_draftkings.csv`
- `data/nflverse/*.parquet`

**Outputs**:
- `data/model_predictions_week{N}.csv`
- `reports/recommendations_week{N}_*.csv`

---

### 4.5 Data Fetching (`scripts/fetch/`)

**Purpose**: Refresh data from external sources.

**Key Scripts**:
| Script | Source | Output | Frequency |
|--------|--------|--------|-----------|
| `fetch_nflverse_data.R` | NFLverse (R) | `data/nflverse/*.parquet` | Weekly |
| `fetch_injuries_sleeper.py` | Sleeper API | `data/injuries/*.parquet` | Daily |
| `fetch_live_odds.py` | Odds API | `data/odds_week{N}.csv` | Pre-game |
| `fetch_nfl_player_props.py` | Odds API | `data/nfl_player_props_*.csv` | Pre-game |

**External Dependencies**:
- R + nflreadr package
- `ODDS_API_KEY` environment variable

---

### 4.6 Pipeline Orchestrator (`scripts/run_pipeline.py`)

**Purpose**: Execute full pipeline with parallel optimization.

**Execution Strategy** (parallel by default):
```
PARALLEL GROUP 1 (Data Fetching):
├── NFLverse R script     ─┐
├── Injuries (Sleeper)    ─┼─ Run simultaneously
├── Live Odds             ─┤
└── Player Props          ─┘

SEQUENTIAL: Freshness Check
SEQUENTIAL: Model Predictions (longest step)

PARALLEL GROUP 2 (Recommendations):
├── Player Prop Recs      ─┬─ Run simultaneously
└── Game Line Recs        ─┘

SEQUENTIAL: Dashboard
```

**Performance**: ~25-30 min (parallel) vs ~40 min (sequential)

**Usage**:
```bash
python scripts/run_pipeline.py 15              # Parallel (default)
python scripts/run_pipeline.py 15 --sequential # Force sequential
```

---

### 4.7 Edge System (`nfl_quant/edges/`)

**Purpose**: Specialized edge detection strategies for different market types.

#### 4.7.1 LVT Edge (Line vs Trailing)

**File**: `nfl_quant/edges/lvt_edge.py`

**Concept**: Captures statistical reversion when Vegas lines diverge significantly from trailing performance.

**Key Features**:
- `line_vs_trailing`: Core signal = (line - trailing) / trailing * 100
- `line_in_sweet_spot`: Gaussian decay based on optimal line ranges
- `market_under_rate`: Market regime indicator

**Target**: 65-70% hit rate at low volume (highly selective)

#### 4.7.2 Player Bias Edge

**File**: `nfl_quant/edges/player_bias_edge.py`

**Concept**: Captures persistent player tendencies to go over or under their lines.

**Key Features**:
- `player_under_rate`: Historical % of times player goes under
- `player_bias`: Average (actual - line) over history
- `LVT_x_player_tendency`: Alignment between LVT and player history

**Target**: 55-60% hit rate at higher volume

#### 4.7.3 Edge Ensemble

**File**: `nfl_quant/edges/ensemble.py`

**Concept**: Combines LVT and Player Bias edges for continuous stats markets.

**Decision Logic**:
1. If only LVT triggers → Use LVT confidence
2. If only Player Bias triggers → Use Player Bias confidence
3. If both agree → Boost confidence
4. If both disagree → No bet

---

### 4.8 TD Poisson Edge (`nfl_quant/models/td_poisson_edge.py`)

**Purpose**: Predict touchdown props using Poisson regression (appropriate for count data).

**Why Poisson, Not XGBoost**:
- TDs are count data (0, 1, 2, 3...) following a Poisson distribution
- Distribution: 34% zero, 29% one, 23% two
- XGBoost assumes normal distribution → systematic errors on TD props

**Key Features**:
- `trailing_passing_tds`: EWMA of passing TDs
- `trailing_passing_yards`: Volume indicator
- `vegas_total`: Game environment
- `vegas_spread`: Game script indicator
- `opponent_pass_td_allowed`: Defense metric

**Probability Calculation**:
```python
# Expected TDs from Poisson regression
expected_tds = model.predict(features)

# P(TDs > line) using Poisson CDF
from scipy.stats import poisson
p_over = 1 - poisson.cdf(floor(line), expected_tds)
p_under = poisson.cdf(floor(line), expected_tds)
```

**Validation Results**:
| Direction | Confidence | Hit Rate | ROI |
|-----------|------------|----------|-----|
| OVER | 58%+ | 62.3% | +18.9% |
| UNDER | 58%+ | 56.2% | +7.2% |

**Training**:
```bash
python scripts/train/train_td_poisson_edge.py
```

**Usage in Recommendations**:
```bash
python scripts/predict/generate_edge_recommendations.py --week 15 --include-td
```

---

## 5. Data Flow

### 5.1 Training Data Flow

```
NFLverse (R)                    Odds API (Historical)
    │                                  │
    ▼                                  ▼
weekly_stats.parquet ───┐    combined_odds_actuals.csv
snap_counts.parquet    │              │
depth_charts.parquet   │              │
    │                  │              │
    └──────────────────┼──────────────┘
                       │
                       ▼
            prepare_data_with_trailing()
                       │
                       ▼
            extract_features_batch()
                       │
                       ▼
                 46 Features + Labels
                       │
                       ▼
               XGBoost Training (walk-forward)
                       │
                       ▼
              active_model.joblib
```

### 5.2 Prediction Data Flow

```
Live Player Props              NFLverse (Current Week)
    │                                  │
    ▼                                  ▼
odds_player_props_*.csv ───┐    weekly_stats.parquet
                          │    injuries.parquet
                          │           │
                          └─────┬─────┘
                                │
                                ▼
                    extract_features_batch()
                                │
                                ▼
                          46 Features
                                │
                                ▼
                    model.predict_proba()
                                │
                                ▼
                         P(UNDER) scores
                                │
                                ▼
                    Edge = P(UNDER) - implied_prob
                                │
                                ▼
                    SNR Filtering (market-specific)
                                │
                                ▼
                    Kelly Sizing (max 2 units)
                                │
                                ▼
                   Ranked Recommendations
```

---

## 6. Key Design Decisions

### 6.1 XGBoost Classification over Regression
**Decision**: Use XGBoost classifiers predicting P(UNDER) instead of regressing actual stats.
**Rationale**: Betting edges come from directional correctness, not point prediction accuracy. A classifier trained on binary outcomes (stat < line) directly optimizes for betting success.

### 6.2 Vectorized Feature Extraction
**Decision**: Use pandas vectorized operations instead of iterrows().
**Rationale**: 100x performance improvement (60s vs 10+ minutes for full training set).

### 6.3 NaN Handling Strategy
**Decision**: Leave opponent features as NaN when unavailable, let XGBoost handle natively.
**Rationale**: Filling with 0 or mean creates false signal. XGBoost learns optimal splits for missing values.

### 6.4 Walk-Forward Validation
**Decision**: Train on data from weeks < test_week - 1 only.
**Rationale**: Prevents temporal data leakage that would inflate backtested performance.

### 6.5 R for NFLverse, Python for Everything Else
**Decision**: Use R's nflreadr for data fetching, Python for ML.
**Rationale**: nflreadr is faster and more reliable than Python alternatives. The parquet bridge is clean.

### 6.6 Signal-to-Noise Ratio Filtering
**Decision**: Market-specific confidence thresholds based on outcome variance.
**Rationale**: Receptions (discrete, low variance) are more predictable than passing yards (high variance). Different markets need different edge thresholds.

### 6.7 Poisson for TD Props, XGBoost for Continuous Stats
**Decision**: Use Poisson regression for touchdown props, XGBoost for yards/receptions/attempts.
**Rationale**: TDs are count data (0, 1, 2, 3...) with a Poisson distribution. XGBoost assumes normal distribution and systematically fails on TD props (-18.2% ROI in testing). Poisson regression achieves +18.9% ROI on OVER bets.

### 6.8 Calibrator Train/Test Split
**Decision**: Fit probability calibrators on held-out data (80/20 split), not training data.
**Rationale**: Fitting calibrators on the same data used to train the model causes overfitting. The calibrator learns to "fix" training errors that won't generalize.

### 6.9 Feature Leakage Prevention
**Decision**: Always `shift(1)` before `expanding()` for rolling statistics.
**Rationale**: `expanding().mean().shift(1)` computes the mean on ALL data first, then shifts - leaking future information. `shift(1).expanding().mean()` correctly excludes the current row before computing the rolling statistic.

### 6.10 Centralized Configuration
**Decision**: All version numbers, feature lists, and parameters in `configs/model_config.py`.
**Rationale**: Eliminates version drift bugs where training uses different features than prediction.

### 6.11 MarketFilter for Game Context (V27)
**Decision**: Use `MarketFilter` dataclass to apply game-context-based filtering before generating recommendations.
**Rationale**: Game context significantly affects prop betting outcomes:
- **Spread filter**: Blowout games (|spread| > 7) have high variance for receiving props
- **Position exclusion**: TEs have 50% win rate in receptions market (no edge)
- **Bye week filter**: Prevents betting on inactive players
- **Game start time filter**: Prevents betting after game has started
- **Snap share filter**: Only established players (40%+ snaps) have predictable usage

**Implementation**:
```python
@dataclass
class MarketFilter:
    max_spread: Optional[float] = None
    min_snap_share: Optional[float] = None
    exclude_positions: Optional[List[str]] = None

MARKET_FILTERS = {
    'player_receptions': MarketFilter(
        max_spread=7.0,
        min_snap_share=0.40,
        exclude_positions=['TE'],
    ),
}
```

---

## 7. Entry Points

### Primary Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/run_pipeline.py` | Full weekly pipeline | `python scripts/run_pipeline.py 15` |
| `scripts/train/train_model.py` | Retrain XGBoost model | `python scripts/train/train_model.py` |
| `scripts/predict/generate_unified_recommendations_v3.py` | Generate picks | `python scripts/predict/generate_unified_recommendations_v3.py --week 15` |

### Edge System Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/train/train_ensemble.py` | Train LVT + Player Bias edges | `python scripts/train/train_ensemble.py` |
| `scripts/train/train_td_poisson_edge.py` | Train TD Poisson edge | `python scripts/train/train_td_poisson_edge.py` |
| `scripts/predict/generate_edge_recommendations.py` | Edge-based picks | `python scripts/predict/generate_edge_recommendations.py --week 15` |
| `scripts/predict/generate_edge_recommendations.py` | Edge + TD picks | `python scripts/predict/generate_edge_recommendations.py --week 15 --include-td` |

### CLI Entry Point
```bash
quant  # Defined in pyproject.toml, points to nfl_quant.cli:app
```

### Configuration Files
| File | Purpose |
|------|---------|
| `configs/model_config.py` | Model version, features, parameters |
| `configs/edge_config.py` | Edge thresholds, TD Poisson config |
| `.env` | API keys (ODDS_API_KEY) |
| `data/manual_snap_overrides.json` | Manual snap count corrections |

---

## 8. Dependencies Map

### Internal Module Dependencies

```
configs/model_config.py
  └── depends on: nfl_quant.config_paths

nfl_quant/features/batch_extractor.py
  ├── depends on: configs.model_config
  ├── depends on: nfl_quant.features.opponent_features
  ├── depends on: nfl_quant.features.position_role
  ├── depends on: nfl_quant.features.coverage_tendencies
  └── depends on: nfl_quant.features.feature_defaults

nfl_quant/features/opponent_features.py
  └── depends on: nfl_quant.config_paths

scripts/train/train_model.py
  ├── depends on: configs.model_config
  ├── depends on: nfl_quant.features.batch_extractor
  ├── depends on: nfl_quant.features.feature_defaults
  └── depends on: nfl_quant.utils.player_names

scripts/predict/generate_unified_recommendations_v3.py
  ├── depends on: configs.model_config
  ├── depends on: nfl_quant.features.batch_extractor
  └── depends on: nfl_quant.config_paths

scripts/run_pipeline.py
  └── depends on: nfl_quant.config_paths
```

### External Package Dependencies

```
Core:
  ├── pandas>=1.5.0,<2.0
  ├── numpy>=1.24.0,<2.0
  ├── pyarrow>=12.0.0
  └── scipy>=1.10.0

ML:
  ├── xgboost>=2.0.0
  ├── scikit-learn>=1.3.0
  └── joblib>=1.3.0

CLI:
  ├── typer>=0.9.0
  └── pydantic>=2.0.0

Data:
  └── requests>=2.31.0 (for API calls)
```

### External API Dependencies

```
Odds API (the-odds-api.com)
  ├── Used by: fetch_live_odds.py, fetch_nfl_player_props.py
  ├── Requires: ODDS_API_KEY in .env
  └── Rate limit: Check API plan

Sleeper API (api.sleeper.app)
  ├── Used by: fetch_injuries_sleeper.py
  └── No authentication required

NFLverse (via R)
  ├── Used by: fetch_nflverse_data.R
  └── Requires: R + nflreadr package
```

---

## 9. Architecture Issues & Recommendations

### Current Issues

1. **Large Monolithic Scripts**: `generate_unified_recommendations_v3.py` (185KB) and `generate_model_predictions.py` (141KB) should be refactored into smaller modules.

2. **Archive Clutter**: `_archive/` directories contain significant dead code that increases cognitive load.

3. **Test Coverage**: Limited automated tests for feature extraction logic.

4. **Hardcoded Paths in Some Scripts**: Some older scripts don't use `config_paths.py`.

### Recommended Improvements

1. **Split Large Scripts**: Extract edge calculation, SNR filtering, and output formatting into separate modules.

2. **Add Type Hints**: Feature extraction functions lack comprehensive type annotations.

3. **Implement Feature Versioning**: Track which features were used for each model version in the joblib artifact.

4. **Add Integration Tests**: Automated tests that verify feature extraction matches expected outputs.

---

## 10. Quick Reference

### Health Check Commands

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

### Common Operations

```bash
# Full pipeline
python scripts/run_pipeline.py 15

# Train model only
python scripts/train/train_model.py

# Predictions only
python scripts/predict/generate_unified_recommendations_v3.py --week 15

# Refresh NFLverse data only
Rscript scripts/fetch/fetch_nflverse_data.R
```

---

**Document Owner**: NFL Quant Team
**Last Updated**: December 2025
**Review Cycle**: After each model version update
