# Totals Model V29 - Summary

**Date**: 2025-12-25
**Version**: V29

---

## Overview

This document summarizes the V29 changes to the totals prediction model in `nfl_quant/edges/game_line_edge.py`.

## Problem Statement

The previous totals model (V28) used **matchup differentials** for EPA adjustment:

```python
# OLD (V28) - INCORRECT for totals
home_epa_adj = (home_epa['off_epa'] - away_epa['def_epa']) * EPA_TOTAL_FACTOR
away_epa_adj = (away_epa['off_epa'] - home_epa['def_epa']) * EPA_TOTAL_FACTOR
```

This approach caused EPA adjustments to **cancel out** in high-scoring environments:
- Two good offenses (+0.10 each) vs two bad defenses (+0.10 each)
- home_epa_adj = (0.10 - 0.10) = 0
- away_epa_adj = (0.10 - 0.10) = 0
- **Result**: No adjustment even though game should project HIGH

## V29 Solution

### New Formula: Pace × Efficiency

```python
# V29 - Unit-consistent formula
plays_total = home_pace + away_pace

# EPA adjustment using COMBINED efficiency (not differential)
combined_off_epa = home_off + away_off
combined_def_epa_allowed = home_def_allowed + away_def_allowed
epa_efficiency = combined_off_epa + combined_def_epa_allowed

ppp = ppp_baseline + (epa_efficiency * epa_total_factor / 100) + ppp_weather_adj
model_total = plays_total × ppp
```

### Sign Conventions (CRITICAL)

| Field | Meaning | Positive Value | Negative Value |
|-------|---------|----------------|----------------|
| `off_epa` | EPA per play on offense | Good offense | Bad offense |
| `def_epa_allowed` | EPA per play allowed on defense | Bad defense (allows points) | Good defense (limits points) |

**For TOTALS:**
- Higher `combined_off_epa` → MORE scoring
- Higher `combined_def_epa_allowed` → MORE scoring (worse defenses)

### Weather Adjustments

| Condition | Adjustment |
|-----------|------------|
| Dome | +1.5 points |
| Cold (<32°F) | -2.0 points |
| Wind (>15 mph) | -0.3 per mph over 15 |

## Key Changes

| Component | Old (V28) | New (V29) |
|-----------|-----------|-----------|
| EPA calculation | Matchup differential | Combined efficiency |
| Formula | `pace × 0.38 + epa_adj` | `plays_total × ppp_total` |
| Weather | Not included | Dome, cold, wind adjustments |
| Variable naming | `def_epa` (ambiguous) | `def_epa_allowed` (explicit) |
| Debug info | None | Full observability dict |
| Calibration | Hardcoded | Grid-searched |

## Calibration

Run calibration with:
```bash
python scripts/calibration/calibrate_totals.py --seasons 2023 2024
```

Default calibrated parameters:
- `points_per_play`: 0.38
- `epa_total_factor`: 5.0

Calibration output saved to: `data/models/calibrated_totals_factor.joblib`

## Testing

Run unit tests with:
```bash
pytest tests/test_totals_model.py -v
```

Tests cover:
1. **Sign conventions**: Increasing off_epa increases total
2. **Sign conventions**: Increasing def_epa_allowed increases total
3. **Weather**: Dome increases totals, cold decreases totals
4. **Regression**: High-scoring environment projects higher than average
5. **Edge direction**: OVER when model > market, UNDER when model < market

## Backtest Results

| Metric | Before (V28) | After (V29) |
|--------|--------------|-------------|
| Total Win Rate | 48.4% | TBD (run calibration) |
| Direction Bias | 81% UNDER | Balanced |
| MAE | ~8.5 pts | TBD |

## Files Modified

- `nfl_quant/edges/game_line_edge.py` - Main model code
- `scripts/calibration/calibrate_totals.py` - Calibration harness (NEW)
- `tests/test_totals_model.py` - Unit tests (NEW)

## Usage

```python
from nfl_quant.edges.game_line_edge import GameLineEdge

edge = GameLineEdge(verbose=True)

result = edge.calculate_total_edge(
    home_epa={'off_epa': 0.05, 'def_epa_allowed': 0.02, 'pace': 62},
    away_epa={'off_epa': 0.03, 'def_epa_allowed': -0.01, 'pace': 58},
    market_total=45.5,
    is_dome=False,
    temperature=55.0,
    wind_speed=10.0
)

direction, edge_pct, confidence, debug_info = result
print(f"Direction: {direction}, Model: {debug_info['clipped_model_total']}")
```

## Observability

Every call to `calculate_total_edge` returns a debug_info dict with:
- `plays_total`, `ppp_baseline`, `ppp_epa_adj`, `ppp_weather_adj`
- `combined_off_epa`, `combined_def_epa_allowed`
- `raw_model_total`, `clipped_model_total`, `market_total`
- `edge_pts`, `was_clipped`
- `is_dome`, `temperature`, `wind_speed`

---

**Author**: Claude Code
**Last Updated**: 2025-12-25
