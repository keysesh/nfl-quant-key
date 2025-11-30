# Historical Data Directory

This directory contains archived historical data for backtesting and analysis.

## Master Files

### Game Lines Master (⭐ PRIMARY SOURCE)
**File**: `game_lines_master_2025.csv`

Consolidated game line odds (spread, total, moneyline) for the entire 2025 season.

**Coverage**:
- 1,614 odds records
- 272 unique games (100% of season)
- All 18 weeks
- 3 markets: spread, total, moneyline

**Schema**:
```
season, week, gameday, game_id, home_team, away_team,
market, side, point, price, sportsbook, data_source, collected_at
```

**Usage**:
```python
from nfl_quant.data.game_lines_loader import load_game_lines
odds = load_game_lines(season=2025)
```

**Built By**: `scripts/data/build_master_game_lines.py`

### Coverage Report
**File**: `coverage_report_2025.json`

Week-by-week breakdown of data coverage:
```json
{
  "1": {
    "total_games": 16,
    "games_with_odds": 16,
    "coverage_pct": 100.0,
    "spread_count": 16,
    "total_count": 16,
    "moneyline_count": 16
  },
  ...
}
```

## Subdirectories

### game_lines/
Individual week files (legacy format, kept for compatibility):
- `game_lines_2025_week01.csv` through `game_lines_2025_week18.csv`

**Note**: Master file is preferred. These are maintained as backups and for incremental updates.

### historical_odds_*.csv
Player prop odds (NOT game lines):
- `historical_odds_2024_week*.csv` - 2024 season player props
- `historical_odds_2025_week*.csv` - 2025 season player props

**Contains**: Anytime TD, passing yards, rushing yards, reception yards, etc.

**Does NOT contain**: Spread, total, or moneyline odds

## Data Freshness

| File Type | Last Updated | Update Frequency |
|-----------|--------------|------------------|
| Master file | As needed | After new data collected |
| Weekly files | Per week | Before games start |
| Coverage report | With master | Same as master |
| Player props | Per week | Real-time API fetch |

## Rebuilding Master File

To consolidate all sources and rebuild the master file:

```bash
python scripts/data/build_master_game_lines.py
```

This scans:
- `data/odds_week*_draftkings.csv`
- `data/odds_week*_comprehensive.csv`
- `data/historical/game_lines/game_lines_*.csv`

And produces:
- `data/historical/game_lines_master_2025.csv`
- `data/historical/coverage_report_2025.json`

## Verification

Run integration tests to verify data integrity:

```bash
python scripts/utils/test_game_lines_system.py
```

Expected output:
```
✅ PASS     Master File Exists
✅ PASS     Master File Format
✅ PASS     Loader Functions
✅ PASS     Coverage Completeness
✅ PASS     Backtest Integration
======================================================================
OVERALL: 5/5 tests passed
🎉 All systems operational!
```

## See Also

- [Game Lines Workflow Documentation](../../docs/GAME_LINES_WORKFLOW.md)
- [Data Collection Notes](../../DATA_COLLECTION_NOTES.md)
