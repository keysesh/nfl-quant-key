# NFL QUANT - Quantitative NFL Betting Framework

A production-grade quantitative analytics pipeline for NFL player prop betting, featuring XGBoost classifiers, edge-based betting signals, and walk-forward validation.

## Status

**Version**: V27 Production
**Season**: 2025
**Python**: 3.10 - 3.12
**Features**: 46

## Validated Edge (Holdout Weeks 12-14 2025)

| Market | Direction | Confidence | Hit Rate | ROI |
|--------|-----------|------------|----------|-----|
| **player_receptions** | UNDER | 58%+ | 58.2% | +10.4% |
| **player_reception_yds** | UNDER | 55%+ | 56.7% | +6.1% |
| **player_rush_yds** | UNDER | 54%+ | 52.3% | +0.4% |
| **player_rush_attempts** | UNDER | 52%+ | 54.8% | +4.2% |
| player_pass_tds (Poisson) | OVER | 58%+ | 62.3% | +18.9% |

## Quick Start

```bash
# Setup
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# Run full pipeline (parallel mode - ~25 min)
python scripts/run_pipeline.py <WEEK>

# Edge-based recommendations (XGBoost + Player Bias)
python scripts/predict/generate_edge_recommendations.py --week <WEEK>

# Include TD prop predictions (Poisson model)
python scripts/predict/generate_edge_recommendations.py --week <WEEK> --include-td

# After games - track results
python scripts/tracking/track_bet_results.py --week <WEEK>
```

## Edge System Architecture

```
                    EDGE SYSTEM
┌──────────────────────────────────────────────────┐
│                                                  │
│  CONTINUOUS STATS (Yards, Rec, Attempts)         │
│  ┌──────────────┐     ┌───────────────────┐     │
│  │  LVT Edge    │     │  Player Bias Edge │     │
│  │ (Line vs     │     │  (Historical      │     │
│  │  Trailing)   │     │   Tendencies)     │     │
│  │ 65-70% @     │     │  55-60% @ high    │     │
│  │ low volume   │     │  volume           │     │
│  └──────┬───────┘     └─────────┬─────────┘     │
│         └─────────┬─────────────┘               │
│                   ▼                              │
│            EdgeEnsemble                          │
│                                                  │
│  TD PROPS (Count Data)                           │
│  ┌──────────────────────────────────┐           │
│  │        TD Poisson Edge           │           │
│  │  - Poisson regression for TDs    │           │
│  │  - P(over)/P(under) via CDF      │           │
│  │  - 62% hit rate @ 58%+ conf      │           │
│  └──────────────────────────────────┘           │
└──────────────────────────────────────────────────┘
```

## V27 Features

**New in V27:**
- **MarketFilter class**: Game context filtering (spread, position, bye week)
- **Spread filter**: Skip blowouts (|spread| > 7 for receptions)
- **TE exclusion**: TEs have 50% win rate in receptions (no edge)
- **Bye week filter**: Skip teams on bye
- **Game start time filter**: 10 min minimum before kickoff
- **Snap share filter**: 40% minimum for established players
- **Recalibrated thresholds**: Receptions 52% → 58%, sweet spot 4.5 → 5.5

## Pipeline

```
Data Ingestion (Parallel)
├── NFLverse (R)         → weekly_stats.parquet
├── Sleeper API          → injuries.parquet
├── Odds API             → player_props.csv
└── Odds API             → game_lines.csv
        │
        ▼
Feature Engineering (46 Features)
├── Core V12 (12)        → LVT, player tendencies
├── V17 Skill (8)        → NGS separation, target share
├── V18 Context (7)      → Vegas total/spread, pace
├── V19 Rush/Rec (4)     → O-line health, box count
├── V23 Opponent (4)     → Defense strength
└── V24 Position (11)    → Matchup analysis
        │
        ▼
Model Inference
├── XGBoost classifiers  → P(UNDER) per market
├── Edge calculation     → model_prob - implied_prob
└── Poisson regression   → TD prop predictions
        │
        ▼
Filtering & Ranking
├── SNR thresholds       → Market-specific confidence
├── MarketFilter         → Game context filters
├── Direction constraint → UNDER_ONLY (except TDs)
└── Kelly sizing         → Max 2 units
        │
        ▼
Output
├── reports/recommendations_week*.csv
├── Dashboard (HTML)
└── Picks summary
```

## Project Structure

```
NFL QUANT/
├── nfl_quant/              # Core package
│   ├── features/           # Feature engineering (46 features)
│   │   ├── batch_extractor.py   # Main extractor
│   │   ├── opponent_features.py # V23 opponent features
│   │   └── position_role.py     # V24 position matchups
│   ├── edges/              # Edge detection strategies
│   │   ├── lvt_edge.py          # Line vs Trailing
│   │   ├── player_bias_edge.py  # Historical tendencies
│   │   └── ensemble.py          # Edge combination
│   ├── models/             # ML models
│   │   └── td_poisson_edge.py   # Poisson for TDs
│   └── calibration/        # Probability calibration
├── scripts/
│   ├── run_pipeline.py     # Main entry point
│   ├── fetch/              # Data fetching
│   ├── train/              # Model training
│   │   ├── train_model.py       # XGBoost training
│   │   ├── train_ensemble.py    # Edge training
│   │   └── train_td_poisson_edge.py
│   └── predict/            # Prediction generation
│       ├── generate_edge_recommendations.py
│       └── generate_unified_recommendations_v3.py
├── configs/
│   ├── model_config.py     # SINGLE SOURCE OF TRUTH
│   ├── edge_config.py      # Edge thresholds
│   └── ensemble_config.py  # Ensemble settings
├── data/
│   ├── models/             # Trained models
│   │   ├── active_model.joblib
│   │   ├── lvt_edge_model.joblib
│   │   └── td_poisson_edge.joblib
│   ├── nflverse/           # NFLverse data
│   └── backtest/           # Training data
└── reports/                # Output reports
```

## Key Files

| Component | File |
|-----------|------|
| **Model Config** | `configs/model_config.py` |
| **Edge Config** | `configs/edge_config.py` |
| **Feature Extraction** | `nfl_quant/features/batch_extractor.py` |
| **Active Model** | `data/models/active_model.joblib` |
| **LVT Edge** | `data/models/lvt_edge_model.joblib` |
| **Player Bias Edge** | `data/models/player_bias_edge_model.joblib` |
| **TD Poisson Edge** | `data/models/td_poisson_edge.joblib` |
| **Training Data** | `data/backtest/combined_odds_actuals_ENRICHED.csv` |
| **Pipeline Runner** | `scripts/run_pipeline.py` |

## Data Freshness

| Data | Max Age | Purpose |
|------|---------|---------|
| weekly_stats.parquet | 12h | Player performance |
| snap_counts.parquet | 12h | Usage patterns |
| injuries.parquet | 6h | Game-day status |
| odds_player_props_*.csv | 4h | Current lines |

```bash
# Refresh NFLverse data
Rscript scripts/fetch/fetch_nflverse_data.R

# Refresh player props
python scripts/fetch/fetch_nfl_player_props.py --week <WEEK>
```

## Training

```bash
# XGBoost classifiers (main model)
python scripts/train/train_model.py

# LVT + Player Bias edges
python scripts/train/train_ensemble.py

# TD Poisson edge
python scripts/train/train_td_poisson_edge.py
```

## Requirements

- Python 3.10 - 3.12
- R with nflreadr (for data fetching)
- See `requirements.txt` for Python dependencies

## Environment Variables

Required in `.env`:
```bash
ODDS_API_KEY=your_key_here  # From the-odds-api.com
```

## Documentation

- `CLAUDE.md` - Quick reference guide
- `ARCHITECTURE.md` - Detailed system architecture
- `CHANGELOG.md` - Version history

## License

MIT
