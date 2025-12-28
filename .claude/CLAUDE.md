# NFL QUANT - Claude Code Guide

**Version**: 2.0 | **Last Updated**: 2025-12-26 | **Model Version**: V28

---

## Context & Documentation

This project uses the `.context` pattern for comprehensive documentation:

| Document | Purpose |
|----------|---------|
| [.context/substrate.md](../.context/substrate.md) | Navigation hub - start here |
| [.context/architecture/overview.md](../.context/architecture/overview.md) | System design |
| [.context/architecture/invariants.md](../.context/architecture/invariants.md) | Rules that must never break |
| [.context/data/contracts.md](../.context/data/contracts.md) | Data file schemas |

## Modular Rules

Topic-specific rules are in `.claude/rules/`:

| Rule | Scope |
|------|-------|
| [rules/data-freshness.md](rules/data-freshness.md) | Use generic files, no fallbacks |
| [rules/nflverse-naming.md](rules/nflverse-naming.md) | Column naming conventions |
| [rules/anti-leakage.md](rules/anti-leakage.md) | Feature engineering patterns |

---

## Output Management

When running commands that may produce verbose output:

```bash
# Always redirect to log file, then tail
python scripts/train/train_model.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 && tail -30 logs/train_*.log | tail -30

# Any verbose command
<command> > logs/output.log 2>&1 && echo "Done." && tail -30 logs/output.log
```

**Never stream 500+ lines directly** - this overflows context.

---

## Quick Start

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
source .venv/bin/activate

# Full pipeline (~25 min)
python scripts/run_pipeline.py <WEEK>

# Edge + TD recommendations
python scripts/predict/generate_edge_recommendations.py --week <WEEK> --include-td

# Regenerate dashboard only
python scripts/dashboard/generate_pro_dashboard.py --week <WEEK>
```

---

## Key Commands

| Command | Purpose |
|---------|---------|
| `python scripts/run_pipeline.py 17` | Full pipeline for week 17 |
| `python scripts/train/train_model.py` | Retrain XGBoost model |
| `python scripts/train/train_ensemble.py` | Train LVT + Player Bias edges |
| `python scripts/train/train_td_poisson_edge.py` | Train TD Poisson edge |
| `Rscript scripts/fetch/fetch_nflverse_data.R` | Refresh NFLverse data |

---

## Critical Invariants (DO NOT VIOLATE)

| Invariant | Rule | Reference |
|-----------|------|-----------|
| No Fallback Files | Use `pbp.parquet`, fail if missing | [rules/data-freshness.md](rules/data-freshness.md) |
| Shift Before Expand | `shift(1).expanding()` not `expanding().shift(1)` | [rules/anti-leakage.md](rules/anti-leakage.md) |
| NFLverse Naming | `carries`, `attempts`, `completions` | [rules/nflverse-naming.md](rules/nflverse-naming.md) |
| Walk-Forward Gap | Train on `week < test_week - 1` | [rules/anti-leakage.md](rules/anti-leakage.md) |
| Calibrator Split | 80/20 train/calibration split | [rules/anti-leakage.md](rules/anti-leakage.md) |

Full list: [.context/architecture/invariants.md](../.context/architecture/invariants.md)

---

## Data Files

**NO FALLBACK - fail if missing**:

| Data Type | Required File |
|-----------|---------------|
| Play-by-play | `data/nflverse/pbp.parquet` |
| Depth charts | `data/nflverse/depth_charts.parquet` |
| Player stats | `data/nflverse/player_stats.parquet` |
| Weekly stats | `data/nflverse/weekly_stats.parquet` |

Full schemas: [.context/data/contracts.md](../.context/data/contracts.md)

---

## Feature Extraction

The model uses **46 features** from `nfl_quant/features/batch_extractor.py`.

**Key features that MUST be populated:**
- `target_share` - #1 signal (~17% importance)
- `has_opponent_context` - Defense info (~10% importance)
- `vegas_total`, `vegas_spread` - Game context

**Verify:**
```python
import joblib
m = joblib.load('data/models/active_model.joblib')
imp = dict(zip(m['models']['player_receptions'].feature_names_in_,
               m['models']['player_receptions'].feature_importances_))
print(f"target_share: {imp.get('target_share', 0):.1%}")  # Should be >10%
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| File not found error | Run `Rscript scripts/fetch/fetch_nflverse_data.R` |
| Opponent context 0% | Check batch_extractor.py merge collision fix |
| target_share 0% | Verify ENRICHED.csv has target_share_stats |
| Stale depth chart | Verify using `depth_charts.parquet` not `depth_charts_2025.parquet` |

---

## When Uncertain

1. **Check invariants first** - [.context/architecture/invariants.md](../.context/architecture/invariants.md)
2. **Check data contracts** - [.context/data/contracts.md](../.context/data/contracts.md)
3. **Ask the user** - Don't guess at intent
4. **Prefer explicit failure** - Better to crash than silently produce wrong results

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` (root) | Public quick reference |
| `ARCHITECTURE.md` | Detailed system architecture |
| `CHANGELOG.md` | Version history |

---

**Last Updated**: 2025-12-26
