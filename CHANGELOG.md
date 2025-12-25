# Changelog

All notable changes to NFL QUANT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [V27] - 2025-12-15

### Added

#### MarketFilter Class (Game Context Filtering)
- **`MarketFilter` dataclass**: New class in `configs/model_config.py` for game-context-based filtering
  - `max_spread`: Skip blowout games (|spread| > 7 for receptions)
  - `min_snap_share`: Only established players (40%+ snaps)
  - `exclude_positions`: Exclude positions with no edge (TEs in receptions market)

- **Bye week filter**: `get_active_teams()` function validates players are on active teams
  - Includes team name normalization (abbreviations + full names)

- **Game start time filter**: 10-minute minimum before kickoff
  - Prevents betting on games already in progress

#### V27 Filter Implementation
- `generate_edge_recommendations.py` updated with:
  - Spread-based filtering
  - Position exclusion
  - Bye week validation with team name mapping
  - Game start time check
  - Snap share minimum

### Changed

- **Receptions confidence threshold**: 52% → 58% (improved ROI)
- **Sweet spot center**: 4.5 → 5.5 receptions (recalibrated to actual data)
- **Documentation**: Updated README.md, CLAUDE.md, ARCHITECTURE.md to V27

### Fixed

- **Bye week filter team name mismatch**: Team names in props data (abbreviations like "PIT") didn't match schedule data (full names like "Pittsburgh Steelers")
  - Added bidirectional team name lookup in `get_active_teams()`

### Removed

- **V25 synergy features** (V26 cleanup): 8 features with 0% importance removed in prior version

---

## [V26] - 2025-12-15

### Removed

- **8 V25 synergy features**: All had 0% importance across all markets
  - `team_synergy_multiplier`
  - `oline_health_score_v25`
  - `wr_corps_health`
  - `has_synergy_bonus`
  - `cascade_efficiency_boost`
  - `wr_coverage_reduction`
  - `returning_player_count`
  - `has_synergy_context`

### Changed

- Feature count: 54 → 46

---

## [V25] - 2025-12-15

### Added

#### TD Poisson Edge (New Market!)
- **`nfl_quant/models/td_poisson_edge.py`**: Poisson regression for touchdown props
  - TDs are count data (0, 1, 2, 3...) following a Poisson distribution
  - XGBoost assumes normal distribution → systematic errors on TD props
  - Uses `scipy.stats.poisson.cdf()` to convert expected TDs → P(over)/P(under)
  - Features: `trailing_passing_tds`, `trailing_passing_yards`, `vegas_total`, `vegas_spread`, `opponent_pass_td_allowed`
  - **Validation Results**: OVER @ 58% confidence → 62.3% hit rate, +18.9% ROI

- **`nfl_quant/features/red_zone_features.py`**: Red zone snap allocation from PBP
  - Computes `rz_rush_share`, `rz_target_share` for TD prediction
  - 85% of rushing TDs come from red zone - critical signal

- **`scripts/train/train_td_poisson_edge.py`**: TD model training pipeline

- **`configs/edge_config.py`**: Added `TD_POISSON_MARKETS` and `TD_POISSON_THRESHOLDS`

- **`--include-td` flag**: `generate_edge_recommendations.py` now supports TD props

### Fixed

#### Data Leakage Fixes (CRITICAL)
All four leakage issues identified and fixed:

1. **Calibrator Train/Calib Split** (`lvt_edge.py`, `player_bias_edge.py`)
   - **Before**: Calibrator fit on same data used for training → overfitting
   - **After**: 80/20 split - train model on 80%, fit calibrator on held-out 20%

2. **market_under_rate Leakage** (`train_lvt_edge.py`, `train_player_bias_edge.py`)
   - **Before**: `expanding().mean().shift(1)` - computes on ALL data first
   - **After**: `shift(1).expanding().mean()` - shift BEFORE expanding

3. **Walk-forward Validation Gap** (`train_model.py`)
   - **Before**: Test features used `global_week < test_week` (extra week of data)
   - **After**: Same cutoff for train and test: `global_week < test_week - 1`

4. **player_bet_count** - Verified as correct (cumcount gives prior count)

#### Results After Leakage Fixes
- **Before**: 4,578 bets, 71.2% hit rate, +35.9% ROI
- **After**: 2,041 bets, 71.4% hit rate, +36.3% ROI (more selective, stronger signals)

### Changed

- Updated model version to V25
- Edge models updated to v3 (`lvt_edge_v3`, `player_bias_edge_v3`)
- Documentation updated: `ARCHITECTURE.md`, `CLAUDE.md`, `.claude/CLAUDE.md`

---

## [V17] - 2025-12-04

### Added

#### Centralized Version Management
- **`configs/model_config.py`**: New single source of truth for all model configuration
  - `MODEL_VERSION = "17"` - No more version string sprawl
  - `SWEET_SPOT_PARAMS` - Market-specific Gaussian decay parameters
  - `FEATURE_FLAGS` - Toggle new features on/off safely
  - `V17_FEATURE_COLS` - Centralized feature column definitions
  - `get_monotonic_constraints()` - XGBoost constraint generation
  - `get_interaction_constraints()` - XGBoost interaction constraint generation
  - `get_model_path()` - Centralized path management

#### Gaussian Sweet Spot (Replaces Binary)
- **Problem Solved**: Binary sweet spot (`1.0 if 3.5 <= line <= 7.5 else 0.0`) excluded 51-69% of data
- **Solution**: Gaussian decay function preserves ~95% of data while still weighting center lines higher
- **Formula**: `exp(-((line - center)^2) / (2 * width^2))`
- **Market-specific parameters**:
  - `player_receptions`: center=4.5, width=2.0
  - `player_rush_yds`: center=55, width=25
  - `player_reception_yds`: center=55, width=25
  - `player_pass_yds`: center=250, width=50
- **Function**: `smooth_sweet_spot(line, market=None, center=None, width=None)`

#### New Feature Interactions
- **`lvt_x_defense`**: LVT × opponent position defense EPA
  - Hypothesis: Strong defenses (negative EPA) amplify the LVT signal for unders
  - Monotonic constraint: -1 (higher defense interaction → lower P(under))
- **`lvt_x_rest`**: LVT × normalized rest days
  - Formula: `lvt * (days_rest - 7) / 7`
  - Hypothesis: Rest deviations may modulate LVT predictive power
  - Monotonic constraint: 0 (no strong prior)

### Changed

#### FeatureEngine (`nfl_quant/features/core.py`)
- `calculate_v12_features()` now accepts optional `market`, `opp_position_def_epa`, `days_rest` parameters
- Uses Gaussian sweet spot when `FEATURE_FLAGS.use_smooth_sweet_spot=True`
- Adds V17 interaction features when enabled in `FEATURE_FLAGS`

#### Feature Defaults (`nfl_quant/features/feature_defaults.py`)
- Added `lvt_x_defense: 0.0` default
- Added `lvt_x_rest: 0.0` default
- Updated comments from "V16" to "V17"

#### Input Validation (`nfl_quant/validation/input_validation.py`)
- Added bounds for `lvt_x_defense`: (-50.0, 50.0)
- Added bounds for `lvt_x_rest`: (-100.0, 100.0)
- Updated comments from "V16" to "V17"

#### Training Script (`scripts/train/train_v12_interaction_model_v2.py`)
- Imports from `configs.model_config` instead of hardcoded values
- Uses `FEATURE_COLS` from config (V17 or V12 based on flags)
- Saves model with feature flags metadata
- Maintains backward compatibility by also saving to `v12_interaction_classifier.joblib`

### Backward Compatibility

- All changes are backward compatible with V16 models
- `FEATURE_FLAGS` can disable all V17 features to produce V12-equivalent models
- Legacy binary sweet spot available when `use_smooth_sweet_spot=False`
- Old models can still be loaded and used

### Migration Guide

1. **No action required** - V17 features are enabled by default but backward compatible
2. **To disable V17 features**: Edit `configs/model_config.py`:
   ```python
   FEATURE_FLAGS = FeatureFlags(
       use_lvt_x_defense=False,
       use_lvt_x_rest=False,
       use_smooth_sweet_spot=False,  # Revert to binary
   )
   ```
3. **To retrain with V17**: Run `python scripts/train/train_v12_interaction_model_v2.py`

---

## [V16] - 2025-11-30

### Added
- Position-specific defense EPA features (`def_vs_wr_epa`, `def_vs_rb_epa`, etc.)
- Game context features (`team_implied_total`, `spread`, `game_total`)
- Injury-related features (`backup_qb_flag`, `teammate_injury_boost`, `games_since_return`)

### Changed
- Centralized all feature calculations in FeatureEngine
- Added `safe_fillna()` with semantic defaults (replaces dangerous `fillna(0)`)

---

## [V12] - 2025-10-30

### Added
- LVT (Line vs Trailing) as hub feature
- Interaction features: `LVT_x_player_tendency`, `LVT_x_player_bias`, `LVT_x_regime`, `LVT_in_sweet_spot`
- XGBoost with interaction constraints and monotonic constraints
- Walk-forward validation framework

### Initial Features
- `line_vs_trailing`
- `line_level`
- `line_in_sweet_spot` (binary)
- `player_under_rate`
- `player_bias`
- `market_under_rate`
- `LVT_x_player_tendency`
- `LVT_x_player_bias`
- `LVT_x_regime`
- `LVT_in_sweet_spot`
- `market_bias_strength`
- `player_market_aligned`
