# NFL QUANT Context Substrate

**Version**: 1.0 | **Last Updated**: 2025-12-26 | **Model Version**: V28

---

## Navigation

This is the entry point for understanding the NFL QUANT system. Use this document to navigate to domain-specific documentation.

### Quick Links

| Domain | Document | Purpose |
|--------|----------|---------|
| **Architecture** | [architecture/overview.md](architecture/overview.md) | System design, bounded contexts |
| **Architecture** | [architecture/invariants.md](architecture/invariants.md) | Critical rules that must never break |
| **Data** | [data/contracts.md](data/contracts.md) | Data file schemas, ownership, freshness |
| **Data** | [data/nflverse.md](data/nflverse.md) | NFLverse conventions and patterns |
| **Features** | [features/extraction.md](features/extraction.md) | Feature engineering patterns |
| **Models** | [models/edge-system.md](models/edge-system.md) | Edge detection strategies |

---

## System Identity

### What NFL QUANT Is

- A **quantitative sports analytics pipeline** for NFL player prop betting
- A **probability estimation system** that calculates P(UNDER) for betting lines
- A **multi-edge strategy** combining statistical and ML approaches
- A **production system** generating weekly betting recommendations

### What NFL QUANT Is Not

- Not a general-purpose sports betting platform
- Not a real-time live betting system (batch predictions only)
- Not a portfolio management or bankroll tracking system
- Not a fantasy football optimizer

---

## Core Invariants (DO NOT VIOLATE)

These are system-wide rules that must never be broken. See [architecture/invariants.md](architecture/invariants.md) for full details.

| Invariant | Rule | Consequence of Violation |
|-----------|------|--------------------------|
| **No Fallback Files** | Use generic files only (e.g., `pbp.parquet`), fail if missing | Silent use of stale data |
| **Shift Before Expand** | `shift(1).expanding()` not `expanding().shift(1)` | Future data leakage |
| **NFLverse Naming** | Use `carries`, `attempts`, `completions` | Column mismatch errors |
| **Walk-Forward Gap** | Train on `week < test_week - 1` | Temporal leakage |
| **Calibrator Split** | 80/20 train/calibration split | Calibrator overfitting |

---

## Tech Stack

```
Language:      Python 3.12 + R (for NFLverse)
ML Framework:  XGBoost, scikit-learn, scipy
Data:          pandas, pyarrow (parquet format)
Web:           Next.js 16, React 19, Tailwind CSS 4 (dashboard)
External APIs: The Odds API, Sleeper API
Data Source:   NFLverse (via nflreadr R package)
```

---

## Directory Map

```
NFL QUANT/
├── .claude/              # Claude Code configuration
│   ├── CLAUDE.md         # AI behavior instructions
│   ├── commands/         # Slash commands
│   └── rules/            # Modular AI rules (new)
│
├── .context/             # Project context (this directory)
│   ├── substrate.md      # This file - navigation hub
│   ├── architecture/     # System design docs
│   ├── data/             # Data contracts and patterns
│   ├── features/         # Feature engineering docs
│   └── models/           # ML model documentation
│
├── configs/              # Configuration files
│   ├── model_config.py   # SINGLE SOURCE OF TRUTH
│   └── edge_config.py    # Edge thresholds
│
├── data/                 # All data files (gitignored)
│   ├── nflverse/         # Fresh NFLverse parquet files
│   ├── models/           # Trained model artifacts
│   ├── backtest/         # Historical training data
│   └── picks/            # Generated recommendations
│
├── nfl_quant/            # Core Python package
│   ├── features/         # Feature extraction modules
│   ├── edges/            # Edge detection strategies
│   ├── calibration/      # Probability calibration
│   ├── betting/          # Bet sizing and Kelly
│   └── integration/      # Pipeline integration
│
├── scripts/              # Executable scripts
│   ├── fetch/            # Data fetching
│   ├── train/            # Model training
│   ├── predict/          # Prediction generation
│   └── dashboard/        # Dashboard generation
│
├── deploy/web/           # Next.js dashboard
│
├── ARCHITECTURE.md       # Detailed system architecture
├── CLAUDE.md             # Quick reference (public)
└── CHANGELOG.md          # Version history
```

---

## Workflow Reference

### Daily Operations

```bash
# Activate environment
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# Full pipeline (parallel mode, ~25 min)
python scripts/run_pipeline.py <WEEK>

# Edge-based recommendations
python scripts/predict/generate_edge_recommendations.py --week <WEEK> --include-td

# Regenerate dashboard only
python scripts/dashboard/generate_pro_dashboard.py --week <WEEK>
```

### Model Retraining

```bash
# Clear feature caches first
python -c "from nfl_quant.features.batch_extractor import clear_caches; clear_caches()"

# Train XGBoost model
python scripts/train/train_model.py

# Train edge models
python scripts/train/train_ensemble.py
python scripts/train/train_td_poisson_edge.py
```

### Data Refresh

```bash
# NFLverse data (creates fresh generic files)
Rscript scripts/fetch/fetch_nflverse_data.R

# Injuries
python scripts/fetch/fetch_injuries_sleeper.py

# Live odds (requires ODDS_API_KEY)
python scripts/fetch/fetch_live_odds.py
python scripts/fetch/fetch_nfl_player_props.py
```

---

## Document Ownership

| Document | Owner | Review Cycle |
|----------|-------|--------------|
| `substrate.md` | System Architect | Monthly |
| `architecture/*.md` | System Architect | Per major version |
| `data/contracts.md` | Data Engineer | Per schema change |
| `features/*.md` | ML Engineer | Per feature addition |
| `models/*.md` | ML Engineer | Per model version |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-26 | Initial creation | Claude |
| 2025-12-26 | Added no-fallback invariant | Claude |

---

**Next**: Read [architecture/overview.md](architecture/overview.md) for system design details.
