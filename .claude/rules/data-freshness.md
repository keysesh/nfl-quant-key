# Rule: Data Freshness

**Scope**: All code that reads NFLverse data files

---

## Rule

Always use fresh generic data files. Never fall back to stale season-specific files.

## Required Files

| Data Type | Use This File | NOT This |
|-----------|---------------|----------|
| Play-by-play | `pbp.parquet` | `pbp_2025.parquet` |
| Depth charts | `depth_charts.parquet` | `depth_charts_2025.parquet` |
| Player stats | `player_stats.parquet` | `player_stats_2024_2025.parquet` |
| Weekly stats | `weekly_stats.parquet` | `weekly_2024.parquet` |
| Rosters | `rosters.parquet` | `rosters_2025.parquet` |

## Pattern

```python
# CORRECT
path = Path('data/nflverse/pbp.parquet')
if not path.exists():
    raise FileNotFoundError(
        f"File not found: {path}. "
        "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
    )

# WRONG - never write fallback patterns
path = Path('data/nflverse/pbp.parquet')
if not path.exists():
    path = Path('data/nflverse/pbp_2025.parquet')  # NO!
```

## Why

- R script `fetch_nflverse_data.R` creates fresh generic files daily
- Season-specific files (`*_2025.parquet`) may be weeks old
- Silent fallback causes predictions based on stale data

## Reference

See [.context/architecture/invariants.md](../../.context/architecture/invariants.md) INV-001
