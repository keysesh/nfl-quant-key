# NFL Quant - Claude Code Guide

## Output Management

When running commands that may produce verbose output (training scripts, data processing, validation, tests):

1. **Always redirect to log file**: `command > logs/name_$(date +%Y%m%d_%H%M%S).log 2>&1`
2. **Then tail the result**: `&& tail -30 logs/name_*.log | tail -30`
3. **Never stream 500+ lines directly** - this will overflow context and break /compact

### Examples:
```bash
# Training
python scripts/train/train_model.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 && tail -30 logs/train_*.log | tail -30

# Validation
python scripts/validation/validate_pipeline_integration.py > logs/validate_$(date +%Y%m%d_%H%M%S).log 2>&1 && tail -50 logs/validate_*.log | tail -50

# Any verbose command
<command> > logs/output.log 2>&1 && echo "Done. Last 30 lines:" && tail -30 logs/output.log
```

---

## Project Overview

NFL player props prediction system using multiple edge strategies:

1. **Edge Ensemble (LVT + Player Bias)**: XGBoost for continuous stats
2. **TD Poisson Edge**: Poisson regression for touchdown props

**Current Model**: V27 | **Features**: 46
**Active Model**: `data/models/active_model.joblib`

---

## Quick Start

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# Run full pipeline (parallel mode - ~25 min)
python scripts/run_pipeline.py <WEEK>

# Edge-based recommendations (LVT + Player Bias)
python scripts/predict/generate_edge_recommendations.py --week <WEEK>

# Include TD props (Poisson model)
python scripts/predict/generate_edge_recommendations.py --week <WEEK> --include-td
```

---

## Key Commands

| Command | Purpose |
|---------|---------|
| `python scripts/run_pipeline.py 15` | Full pipeline for week 15 |
| `python scripts/train/train_model.py` | Retrain XGBoost model |
| `python scripts/train/train_ensemble.py` | Train LVT + Player Bias edges |
| `python scripts/train/train_td_poisson_edge.py` | Train TD Poisson edge |
| `python scripts/predict/generate_edge_recommendations.py --week 15 --include-td` | Edge + TD picks |
| `Rscript scripts/fetch/fetch_nflverse_data.R` | Refresh NFLverse data |

---

## V27 Changes (Dec 15, 2025)

| Change | Details |
|--------|---------|
| MarketFilter class | Game context filtering (spread, position, bye week) |
| Spread filter | Skip blowouts (|spread| > 7 for receptions) |
| TE exclusion | TEs have 50% win rate - excluded from receptions |
| Bye week filter | Skip players on bye (team name normalization fix) |
| Game start time | 10 min minimum before kickoff |
| Snap share filter | 40% minimum for established players |
| Receptions threshold | Raised 52% → 58% for better ROI |
| Sweet spot center | Recalibrated 4.5 → 5.5 receptions |

---

## Critical Data Files

| File | Purpose |
|------|---------|
| `data/backtest/combined_odds_actuals_ENRICHED.csv` | Training data (must have opponent, vegas_total, vegas_spread) |
| `data/nflverse/weekly_stats.parquet` | Player stats (seasons 2023-2025) |
| `data/models/active_model.joblib` | XGBoost model |
| `data/models/lvt_edge_model.joblib` | LVT edge model |
| `data/models/player_bias_edge_model.joblib` | Player Bias edge model |
| `data/models/td_poisson_edge.joblib` | TD Poisson edge model |

---

## Feature Extraction

The model uses **46 features** extracted by `nfl_quant/features/batch_extractor.py`.

**Key features that MUST be populated:**
- `target_share` - #1 signal (~17% importance)
- `has_opponent_context` - Defense info (~10% importance)
- `vegas_total`, `vegas_spread` - Game context

**Verify features work:**
```python
import joblib
m = joblib.load('data/models/active_model.joblib')
imp = dict(zip(m['models']['player_receptions'].feature_names_in_,
               m['models']['player_receptions'].feature_importances_))
print(f"target_share: {imp.get('target_share', 0):.1%}")  # Should be >10%
```

---

## Protected Fixes (DO NOT REVERT)

These fixes are critical. Reverting them breaks the model.

| Fix | What It Does |
|-----|--------------|
| **Enriched Training Data** | Uses ENRICHED.csv with vegas/opponent data |
| **Opponent Merge Collision** | Don't pre-initialize opponent columns before merge |
| **Stale PBP Path** | Use cascading path lookup (nflverse → processed) |
| **Zero-Importance Features** | `broken_feature_fixes.py` calculates actual values |
| **Calibrator Split** | 80/20 split prevents calibrator overfitting |
| **market_under_rate Order** | `shift(1).expanding()` not `expanding().shift(1)` |
| **Walk-forward Gap** | Same historical cutoff for train and test features |
| **TD Poisson Edge** | Poisson regression for TDs (not XGBoost) |
| **V27 MarketFilter** | Game context filtering (spread, TE, bye) |

---

## NFLverse Data Conventions

**IMPORTANT**: `player_id` IS `gsis_id` (same field)

**Data path priority** (always use cascading lookup):
```python
pbp_path = Path('data/nflverse/pbp.parquet')           # Try first
if not pbp_path.exists():
    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
if not pbp_path.exists():
    pbp_path = Path(f'data/processed/pbp_{season}.parquet')  # Last resort
```

**Common mistakes:**
- Using `data/processed/` directly without checking `data/nflverse/` first
- Casting `game_id` to int (it's a STRING)

---

## Anti-Leakage Mechanisms

1. **`shift(1)` in trailing stats** - Only uses prior weeks
2. **`shift(1)` in opponent defense** - Only uses prior games
3. **Walk-forward validation** - Trains on `week < test_week - 1`
4. **`shift(1).expanding()` order** - shift BEFORE expanding
5. **Calibrator 80/20 split** - Held-out data for calibration

**CRITICAL - Correct order for rolling stats:**
```python
# WRONG (leaks future):
df['feature'] = df['col'].expanding().mean().shift(1)

# CORRECT:
df['feature'] = df['col'].shift(1).expanding().mean()
```

---

## Model Retraining Checklist

**Before:**
- [ ] ENRICHED.csv has `opponent` (>90%)
- [ ] ENRICHED.csv has `vegas_total` (>70%)
- [ ] weekly_stats.parquet has 2023, 2024, 2025
- [ ] Clear caches: `from nfl_quant.features.batch_extractor import clear_caches; clear_caches()`

**After:**
- [ ] `target_share` importance > 10%
- [ ] `has_opponent_context` importance > 5%

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Opponent context 0% | Check batch_extractor.py merge collision fix |
| target_share 0% | Verify ENRICHED.csv has target_share_stats |
| >50% zeros in EPA field | PBP path using stale data - fix cascading lookup |
| All picks filtered | Check bye week filter team name format |

---

## Related Documentation

- `CLAUDE.md` (root) - Quick reference
- `ARCHITECTURE.md` - System design, data flows
- `architecture/claude.md` - NFLverse data validation rules
- `CHANGELOG.md` - Version history

---

**Last Updated**: 2025-12-15
