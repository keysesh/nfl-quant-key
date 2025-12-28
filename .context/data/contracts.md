# Data Contracts

**Version**: 1.0 | **Last Updated**: 2025-12-26

---

## Purpose

This document defines the **contracts** for all data files in the NFL QUANT system. Each contract specifies:
- **Owner**: Who creates/maintains this file
- **Location**: Where the file lives
- **Refresh**: How often it's updated
- **Schema**: Required columns and types
- **Consumers**: What code reads this file
- **Freshness SLA**: Maximum acceptable age

---

## NFLverse Data Files

All NFLverse files are created by `scripts/fetch/fetch_nflverse_data.R`.

### data/nflverse/pbp.parquet

**Play-by-Play Data**

| Attribute | Value |
|-----------|-------|
| Owner | `fetch_nflverse_data.R` |
| Location | `data/nflverse/pbp.parquet` |
| Refresh | Daily during season |
| Freshness SLA | < 24 hours during game weeks |

**Schema** (key columns):
```
game_id: string         # {season}_{week}_{away}_{home}
play_id: int            # Unique play identifier
posteam: string         # Possession team
defteam: string         # Defensive team
play_type: string       # run, pass, etc.
epa: float              # Expected Points Added
week: int               # Week number
season: int             # Season year
```

**Consumers**:
- `nfl_quant/integration/feature_aggregator.py`
- `scripts/predict/generate_unified_recommendations_v3.py`
- `scripts/dashboard/generate_pro_dashboard.py`

**Invariant**: INV-001 (no fallback to `pbp_2025.parquet`)

---

### data/nflverse/depth_charts.parquet

**Team Depth Charts**

| Attribute | Value |
|-----------|-------|
| Owner | `fetch_nflverse_data.R` |
| Location | `data/nflverse/depth_charts.parquet` |
| Refresh | Daily during season |
| Freshness SLA | < 24 hours |

**Schema** (ESPN format, 2025+):
```
dt: timestamp           # Depth chart timestamp (for freshness)
team: string            # Team abbreviation
pos_abb: string         # Position abbreviation (QB, RB, WR, TE)
pos_rank: int           # Depth rank (1=starter, 2=backup)
player_name: string     # Full player name
gsis_id: string         # Player ID (same as player_id)
```

**Consumers**:
- `scripts/dashboard/generate_pro_dashboard.py`
- `scripts/predict/generate_game_line_predictions.py`
- `nfl_quant/features/qb_starter.py`
- `nfl_quant/utils/contextual_integration.py`

**Note**: Use `dt` column for freshness filtering. Sort by `dt` descending to get most recent.

---

### data/nflverse/player_stats.parquet

**Player Statistics**

| Attribute | Value |
|-----------|-------|
| Owner | `fetch_nflverse_data.R` |
| Location | `data/nflverse/player_stats.parquet` |
| Refresh | Daily during season |
| Freshness SLA | < 24 hours |

**Schema** (key columns - uses NFLverse naming):
```
player_id: string       # GSIS ID (canonical player identifier)
player_name: string     # Full player name
position: string        # Position (QB, RB, WR, TE)
team: string            # Team abbreviation
season: int             # Season year
week: int               # Week number

# Passing (NFLverse naming)
attempts: int           # Pass attempts (NOT passing_attempts)
completions: int        # Completions (NOT passing_completions)
passing_yards: float    # Passing yards
passing_tds: int        # Passing touchdowns
interceptions: int      # Interceptions

# Rushing (NFLverse naming)
carries: int            # Rush attempts (NOT rushing_attempts)
rushing_yards: float    # Rushing yards
rushing_tds: int        # Rushing touchdowns

# Receiving
receptions: int         # Receptions
receiving_yards: float  # Receiving yards
receiving_tds: int      # Receiving touchdowns
targets: int            # Targets
```

**Consumers**:
- `nfl_quant/data/dynamic_parameters.py`
- `nfl_quant/features/trailing_stats.py`
- Training scripts

**Invariant**: INV-003 (use NFLverse column names)

---

### data/nflverse/weekly_stats.parquet

**Weekly Player Stats (alias for player_stats)**

| Attribute | Value |
|-----------|-------|
| Owner | `fetch_nflverse_data.R` |
| Location | `data/nflverse/weekly_stats.parquet` |
| Refresh | Daily during season |
| Freshness SLA | < 24 hours |

Same schema as `player_stats.parquet`.

---

### data/nflverse/rosters.parquet

**Team Rosters**

| Attribute | Value |
|-----------|-------|
| Owner | `fetch_nflverse_data.R` |
| Location | `data/nflverse/rosters.parquet` |
| Refresh | Daily during season |
| Freshness SLA | < 24 hours |

**Schema**:
```
player_id: string       # GSIS ID
player_name: string     # Full player name
position: string        # Position
team: string            # Team abbreviation
headshot_url: string    # Player headshot URL
```

**Consumers**:
- `scripts/dashboard/generate_pro_dashboard.py`

---

### data/nflverse/snap_counts.parquet

**Snap Count Data**

| Attribute | Value |
|-----------|-------|
| Owner | `fetch_nflverse_data.R` |
| Location | `data/nflverse/snap_counts.parquet` |
| Refresh | Daily during season |
| Freshness SLA | < 24 hours |

**Schema**:
```
player_id: string       # GSIS ID
player_name: string     # Player name
team: string            # Team abbreviation
week: int               # Week number
season: int             # Season year
offense_snaps: int      # Offensive snaps played
offense_pct: float      # Offensive snap percentage (0-1)
```

**Consumers**:
- `nfl_quant/features/snap_count_features.py`
- `nfl_quant/features/batch_extractor.py`

---

## Model Artifacts

### data/models/active_model.joblib

**Production XGBoost Model**

| Attribute | Value |
|-----------|-------|
| Owner | `scripts/train/train_model.py` |
| Location | `data/models/active_model.joblib` |
| Refresh | Per model version |

**Schema** (Python dict):
```python
{
    'version': str,           # e.g., "V28"
    'trained_date': str,      # ISO date
    'features': list[str],    # 46 feature names
    'models': {
        'player_receptions': XGBClassifier,
        'player_rush_yds': XGBClassifier,
        'player_reception_yds': XGBClassifier,
        # ... per-market classifiers
    }
}
```

**Consumers**:
- `scripts/predict/generate_model_predictions.py`
- `scripts/predict/generate_unified_recommendations_v3.py`

---

### data/models/td_poisson_edge.joblib

**TD Poisson Edge Model**

| Attribute | Value |
|-----------|-------|
| Owner | `scripts/train/train_td_poisson_edge.py` |
| Location | `data/models/td_poisson_edge.joblib` |
| Refresh | Per model version |

**Schema**:
```python
{
    'models': {
        'player_pass_tds': PoissonRegressor,
        # ... per-market regressors
    },
    'features': list[str],
    'trained_date': str
}
```

---

## Training Data

### data/backtest/combined_odds_actuals_ENRICHED.csv

**Historical Betting Data with Outcomes**

| Attribute | Value |
|-----------|-------|
| Owner | Manual curation |
| Location | `data/backtest/combined_odds_actuals_ENRICHED.csv` |
| Refresh | Weekly (add new weeks) |

**Schema**:
```
player_name: string     # Player name
market: string          # Prop market (player_receptions, etc.)
line: float             # Betting line
actual: float           # Actual result
under_hit: int          # 1 if under, 0 if over
season: int             # Season year
week: int               # Week number
opponent: string        # Opponent team (>90% populated)
vegas_total: float      # Game total (>70% populated)
vegas_spread: float     # Point spread
```

**Critical**: Must have `opponent` >90% populated for opponent features to work.

---

## Freshness Validation

Run this check before pipeline execution:

```python
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def check_data_freshness():
    """Verify critical data files are fresh."""
    files = {
        'pbp.parquet': 24,  # hours
        'depth_charts.parquet': 24,
        'player_stats.parquet': 24,
        'weekly_stats.parquet': 24,
    }

    for filename, max_age_hours in files.items():
        path = Path(f'data/nflverse/{filename}')
        if not path.exists():
            raise FileNotFoundError(f"{filename} missing")

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        if age > timedelta(hours=max_age_hours):
            print(f"WARNING: {filename} is {age.total_seconds()/3600:.1f}h old")
```

---

## Change Log

| Date | File | Change |
|------|------|--------|
| 2025-12-26 | All | Initial creation |
| 2025-12-26 | depth_charts.parquet | Document dt column for freshness |
| 2025-12-26 | All | Remove fallback file patterns |
