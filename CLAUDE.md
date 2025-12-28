# NFL QUANT Framework

**Status**: Production | **Updated**: 2025-12-26
**Current Model**: V29 | **Features**: 57

> **Documentation**: See [.context/substrate.md](.context/substrate.md) for navigation hub

---

## Quick Start

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# Run full pipeline (parallel mode by default - ~25-30 min)
python scripts/run_pipeline.py <WEEK>

# Edge-based recommendations (LVT + Player Bias)
python scripts/predict/generate_edge_recommendations.py --week <WEEK>

# Include TD prop predictions (Poisson model)
python scripts/predict/generate_edge_recommendations.py --week <WEEK> --include-td
```

---

## Centralized Configuration

**All model configuration lives in `configs/model_config.py`:**

| Setting | Description |
|---------|-------------|
| `MODEL_VERSION` | Current version (29) |
| `FEATURES` | 57 feature columns |
| `CLASSIFIER_MARKETS` | Markets for training |
| `MODEL_PARAMS` | XGBoost hyperparameters |
| `MARKET_SNR_CONFIG` | Signal-to-noise thresholds |
| `MARKET_FILTERS` | V27 game context filters |

---

## V27 Changes

**New in V27 (Dec 15, 2025):**

| Change | Details |
|--------|---------|
| MarketFilter class | Game context filtering (spread, position, bye week) |
| Spread filter | Skip blowouts (|spread| > 7 for receptions) |
| TE exclusion | TEs have 50% win rate (no edge) - exclude from receptions |
| Bye week filter | Skip players on bye (team name normalization fix) |
| Game start time | 10 min minimum before kickoff |
| Snap share filter | 40% minimum for established players |
| Receptions threshold | Raised 52% → 58% for better ROI |
| Sweet spot center | Recalibrated 4.5 → 5.5 receptions |

---

## Environment Variables

Required in `.env`:
```bash
ODDS_API_KEY=your_key_here  # From the-odds-api.com
```

---

## Key Files

| Component | File |
|-----------|------|
| **Model Config** | `configs/model_config.py` |
| **Edge Config** | `configs/edge_config.py` |
| **Feature Extraction** | `nfl_quant/features/batch_extractor.py` |
| **Active Model** | `data/models/active_model.joblib` |
| **LVT Edge Model** | `data/models/lvt_edge_model.joblib` |
| **Player Bias Edge** | `data/models/player_bias_edge_model.joblib` |
| **TD Poisson Edge** | `data/models/td_poisson_edge.joblib` |
| **Training Data** | `data/backtest/combined_odds_actuals_ENRICHED.csv` |
| **Pipeline Runner** | `scripts/run_pipeline.py` |
| **Architecture Docs** | `.context/architecture/overview.md` |

---

## Edge System

The system uses **three specialized edges** for different market types:

| Edge | Markets | Model | Target |
|------|---------|-------|--------|
| **LVT Edge** | Yards, Rec, Att | XGBoost | 65-70% @ low volume |
| **Player Bias Edge** | Yards, Rec, Att | XGBoost | 55-60% @ high volume |
| **TD Poisson Edge** | Pass TDs, Rush TDs | Poisson | 62% @ 58%+ conf |

**Training commands:**
```bash
# Train LVT + Player Bias edges
python scripts/train/train_ensemble.py

# Train TD Poisson edge
python scripts/train/train_td_poisson_edge.py
```

---

## Supported Markets

### Continuous Stats (XGBoost)
- `player_receptions` (HIGH SNR) - 58% threshold
- `player_rush_yds` (HIGH SNR)
- `player_reception_yds` (MEDIUM SNR)
- `player_rush_attempts` (HIGH SNR)
- `player_pass_yds` (MEDIUM SNR) - **CLOSE GAMES ONLY** (|spread| <= 3)

### TD Props (Poisson)
- `player_pass_tds` - **Poisson model** (62% hit rate on OVER @ 58%+ conf)

**Disabled:**
- `player_pass_yds` - -15.8% ROI in holdout, failing both directions (Dec 2025)

**Re-enabled (Dec 2025):**
- `player_pass_completions` - Added back to CLASSIFIER_MARKETS with filters
- `player_pass_attempts` - Added back to CLASSIFIER_MARKETS with filters

### Market Direction Constraints (V27)

**UNDER_ONLY** for all continuous stats markets based on holdout analysis:
- UNDER picks: +4.0% ROI (2998 bets, 54.5% win rate)
- OVER picks: -14.8% ROI (1766 bets, 44.6% win rate)

---

## Data Freshness Requirements

| Data | Max Age | Source |
|------|---------|--------|
| weekly_stats.parquet | 12h | NFLverse (R) |
| snap_counts.parquet | 12h | NFLverse (R) |
| injuries.parquet | 6h | Sleeper API |
| odds_player_props_*.csv | 4h | Odds API |

**Refresh command:**
```bash
Rscript scripts/fetch/fetch_nflverse_data.R
```

---

## NFLverse ID Conventions

**IMPORTANT**: In NFLverse data, `player_id` IS the canonical `gsis_id`.

| ID Field | Usage |
|----------|-------|
| `player_id` / `gsis_id` | Primary player identifier (same field) |
| `game_id` | Format: `{season}_{week}_{away}_{home}` (STRING, not int!) |

**Data files - NO FALLBACK, fail if missing**:

| Data Type | Required File |
|-----------|---------------|
| Play-by-play | `data/nflverse/pbp.parquet` |
| Depth charts | `data/nflverse/depth_charts.parquet` |
| Player stats | `data/nflverse/player_stats.parquet` |
| Weekly stats | `data/nflverse/weekly_stats.parquet` |
| Rosters | `data/nflverse/rosters.parquet` |

```python
# NO FALLBACK - fail explicitly if file is missing
pbp_path = Path('data/nflverse/pbp.parquet')
if not pbp_path.exists():
    raise FileNotFoundError(
        f"PBP file not found: {pbp_path}. "
        "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
    )
```

**WHY**: R script creates these generic files fresh daily. Season-specific files (`*_2025.parquet`) are stale and should NOT be used as fallback.

---

## NFLverse Column Naming Convention

**CRITICAL**: Use NFLverse native column names when reading raw data:

| Correct (NFLverse) | Wrong (Legacy) | Description |
|--------------------|----------------|-------------|
| `carries` | `rushing_attempts` | Rush attempts |
| `attempts` | `passing_attempts` | Pass attempts |
| `completions` | `passing_completions` | Completions |

**Stats columns** (all use NFLverse naming):
- `passing_yards`, `passing_tds`, `interceptions`
- `rushing_yards`, `rushing_tds`
- `receiving_yards`, `receiving_tds`, `receptions`, `targets`

**Depth charts** (2025+ format uses `dt` timestamp):
```python
# Sort by dt (timestamp) to get most recent depth chart
depth_df = depth_df.sort_values('dt', ascending=False)
# Key columns: player_name, team, pos_name, pos_rank
```

---

## Pipeline Optimization

The pipeline uses **parallel execution by default**, reducing runtime by ~30-40%:

```
PARALLEL GROUP 1 (Data Fetching):
├── NFLverse R script     ─┐
├── Injuries (Sleeper)    ─┼─ Run simultaneously (~5 min total)
├── Live Odds             ─┤
└── Player Props          ─┘

SEQUENTIAL: Model Predictions (~15-25 min)

PARALLEL GROUP 2 (Recommendations):
├── Player Prop Recs      ─┬─ Run simultaneously (~3 min total)
└── Game Line Recs        ─┘

SEQUENTIAL: Dashboard
```

---

## Protected Fixes (DO NOT REVERT)

| Fix | Date | Impact |
|-----|------|--------|
| Enriched Training Data | Dec 7 | target_share 0% → 17% importance |
| Opponent Merge Collision | Dec 7 | has_opponent_context 0% → 9.7% |
| Stale PBP Path Fix | Dec 14 | EPA now uses fresh data |
| 13 Zero-Importance Features | Dec 14 | game_pace, slot_snap_pct now working |
| QB Model Spread Filter | Dec 14 | player_pass_yds: -14.8% → +21.9% ROI (close games only) |
| Calibrator Train/Calib Split | Dec 15 | 80/20 split prevents calibrator overfitting |
| market_under_rate Leakage | Dec 15 | shift() before expanding() prevents future leak |
| Walk-forward Validation Gap | Dec 15 | Same historical cutoff for train and test |
| TD Poisson Edge | Dec 15 | Proper model for count data (62% hit rate) |
| V27 MarketFilter | Dec 15 | Game context filtering (spread, TE, bye) |
| No-Fallback Data Files | Dec 26 | Use generic files only, fail if missing (no stale fallbacks) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No predictions generated | Check data freshness, verify odds file exists |
| Model not found | Run `python scripts/train/train_model.py` |
| Stale data | Run `Rscript scripts/fetch/fetch_nflverse_data.R` |
| ODDS_API_KEY error | Add key to `.env` file |
| >50% zeros in EPA | Check pbp.parquet exists (run R fetch script) |
| All picks filtered | Check bye week filter team name format |

### Quick Health Check

```bash
python -c "
import joblib
m = joblib.load('data/models/active_model.joblib')
print(f'Version: {m[\"version\"]}')
print(f'Features: {len(m[\"models\"][\"player_receptions\"].feature_names_in_)}')
"
```

---

## Model Retraining

```bash
# Clear caches first
python -c "from nfl_quant.features.batch_extractor import clear_caches; clear_caches()"

# Train model (~30 seconds)
python scripts/train/train_model.py
```

---

## Anti-Leakage Mechanisms

1. **`shift(1)` in trailing stats** - Only uses prior weeks
2. **`shift(1)` in opponent defense** - Only uses prior games
3. **Walk-forward validation** - Trains on `week < test_week - 1`
4. **`shift(1).expanding()` order** - CRITICAL: shift BEFORE expanding (not after)
5. **Calibrator train/calib split** - 80% train, 20% held-out for calibration

**WRONG (leaks future):**
```python
df['market_under_rate'] = df['under_hit'].expanding().mean().shift(1)  # BAD!
```

**CORRECT:**
```python
df['market_under_rate'] = df['under_hit'].shift(1).expanding().mean()  # GOOD
```

---

## Related Documentation

### Primary Documentation (.context pattern)

| Document | Purpose |
|----------|---------|
| [.context/substrate.md](.context/substrate.md) | Navigation hub - start here |
| [.context/architecture/overview.md](.context/architecture/overview.md) | System design, bounded contexts |
| [.context/architecture/invariants.md](.context/architecture/invariants.md) | Rules that must never break |
| [.context/data/contracts.md](.context/data/contracts.md) | Data file schemas |

### Modular Rules (.claude/rules/)

| Rule | Scope |
|------|-------|
| [.claude/rules/data-freshness.md](.claude/rules/data-freshness.md) | No fallbacks, use generic files |
| [.claude/rules/nflverse-naming.md](.claude/rules/nflverse-naming.md) | Column naming conventions |
| [.claude/rules/anti-leakage.md](.claude/rules/anti-leakage.md) | Feature engineering patterns |

### Legacy Documentation

- `ARCHITECTURE.md` - Detailed system architecture
- `CHANGELOG.md` - Version history

---

## Betting Parameters

- **Unit size**: $5 base
- **Max bet**: $10 (2 units)
- Kelly sizing in dashboard is reference only

---

**Last Updated**: 2025-12-26
