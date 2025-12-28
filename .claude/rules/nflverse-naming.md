# Rule: NFLverse Column Naming

**Scope**: All code that reads or references NFLverse data

---

## Rule

Use NFLverse native column names, not legacy/intuitive names.

## Column Mapping

| Correct (NFLverse) | Wrong (Legacy) | Description |
|--------------------|----------------|-------------|
| `carries` | `rushing_attempts` | Rush attempts |
| `attempts` | `passing_attempts` | Pass attempts |
| `completions` | `passing_completions` | Completions |
| `player_id` | `gsis_id` | Player identifier (they are the same) |

## Stats Columns (All NFLverse)

```python
# Passing
'attempts'         # Pass attempts
'completions'      # Completions
'passing_yards'    # Passing yards
'passing_tds'      # Passing TDs
'interceptions'    # Interceptions

# Rushing
'carries'          # Rush attempts (NOT rushing_attempts)
'rushing_yards'    # Rushing yards
'rushing_tds'      # Rushing TDs

# Receiving
'receptions'       # Receptions
'receiving_yards'  # Receiving yards
'receiving_tds'    # Receiving TDs
'targets'          # Targets
```

## Why

NFLverse data uses specific column names. Using incorrect names causes:
- `KeyError` when column doesn't exist
- Silent bugs if column exists with wrong semantics
- Join failures on mismatched column names

## Reference

- Canonical schema: `nfl_quant/data/stats_schema.py`
- NFLverse dictionary: https://nflreadr.nflverse.com/articles/dictionary_player_stats.html
- See [.context/architecture/invariants.md](../../.context/architecture/invariants.md) INV-003
