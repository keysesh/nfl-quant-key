# NFL QUANT Feature Flow Diagram

**Generated**: 2025-12-05
**Purpose**: Document how features flow through the prediction pipeline

---

## 1. Feature Categories

### Betting Tendency Features (14 features)
| Feature | Source | Description |
|---------|--------|-------------|
| `line_vs_trailing` | `calculate_v12_features()` | (line - trailing_stat) / trailing_stat |
| `line_level` | `calculate_v12_features()` | Raw line value |
| `line_in_sweet_spot` | `calculate_v12_features()` | Gaussian decay from center |
| `player_under_rate` | Historical odds data | Player's historical under hit rate |
| `player_bias` | Historical odds data | Player's actual - line bias |
| `market_under_rate` | Historical odds data | Market's historical under rate |
| `LVT_x_player_tendency` | `calculate_v12_features()` | LVT × player_under_rate |
| `LVT_x_player_bias` | `calculate_v12_features()` | LVT × player_bias |
| `LVT_x_regime` | `calculate_v12_features()` | LVT × market_under_rate |
| `LVT_in_sweet_spot` | `calculate_v12_features()` | LVT × sweet_spot |
| `market_bias_strength` | `calculate_v12_features()` | |market_under_rate - 0.5| |
| `player_market_aligned` | `calculate_v12_features()` | Sign alignment check |
| `lvt_x_defense` | `calculate_v12_features()` | LVT × opp_def_epa |
| `lvt_x_rest` | `calculate_v12_features()` | LVT × rest_deviation |

### Skill Features (6 features) - NEW V17
| Feature | Source | Description | Required Data |
|---------|--------|-------------|---------------|
| `avg_separation` | `get_avg_separation()` | NGS receiver separation | `player_id` |
| `avg_cushion` | `get_avg_cushion()` | NGS defensive cushion | `player_id` |
| `trailing_catch_rate` | `get_trailing_catch_rate()` | Receptions / targets | `player_id` |
| `snap_share` | `get_snap_share()` | Player snap % | `player_name` |
| `target_share` | `get_target_share()` | Player target % | `player_id` |
| `opp_wr1_receptions_allowed` | `get_wr1_defense_stat()` | Defense vs WR1 | `opponent`, `position` |

---

## 2. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  combined_odds_actuals.csv    weekly_stats.parquet    ngs_receiving.parquet │
│  ─────────────────────────    ───────────────────    ─────────────────────  │
│  • player                     • player_id             • player_id            │
│  • season, week               • position              • avg_separation       │
│  • market, line               • opponent_team         • avg_cushion          │
│  • actual_stat                • receptions            └─────────────────────  │
│  • under_hit                  • targets                                      │
│  └───────┬───────────────────┴───────┬───────────────────────────────────┘   │
│          │                           │                                       │
│          ▼                           ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    MERGE (prepare_data_with_trailing)                  │  │
│  │                                                                        │  │
│  │  odds + stats → odds_merged                                            │  │
│  │    • player_norm                                                       │  │
│  │    • player_id        ← CRITICAL for skill features                   │  │
│  │    • position         ← CRITICAL for WR1 defense                      │  │
│  │    • opponent         ← CRITICAL for defense features                 │  │
│  │    • trailing_* stats                                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  extract_v12_features_for_week()                                             │
│  ─────────────────────────────                                               │
│                                                                              │
│  For each row in odds_merged:                                                │
│                                                                              │
│  1. BETTING FEATURES (calculate_v12_features)                                │
│     ├─ line_vs_trailing = (line - trailing) / trailing                       │
│     ├─ line_level = line                                                     │
│     ├─ line_in_sweet_spot = gaussian_decay(line, market)                     │
│     ├─ player_under_rate = lookup(historical_odds)                           │
│     ├─ player_bias = lookup(historical_odds)                                 │
│     ├─ market_under_rate = lookup(historical_odds)                           │
│     ├─ LVT_x_player_tendency = LVT × player_under_rate                       │
│     ├─ LVT_x_player_bias = LVT × player_bias                                 │
│     ├─ LVT_x_regime = LVT × market_under_rate                                │
│     ├─ LVT_in_sweet_spot = LVT × sweet_spot                                  │
│     ├─ market_bias_strength = |market_under_rate - 0.5|                      │
│     ├─ player_market_aligned = sign check                                    │
│     ├─ lvt_x_defense = LVT × opp_def_epa                                     │
│     └─ lvt_x_rest = LVT × rest_deviation                                     │
│                                                                              │
│  2. SKILL FEATURES (if player_id available)                                  │
│     ├─ avg_separation = get_avg_separation(player_id)                        │
│     ├─ avg_cushion = get_avg_cushion(player_id)                              │
│     ├─ trailing_catch_rate = get_trailing_catch_rate(player_id)              │
│     └─ target_share = get_target_share(player_id)                            │
│                                                                              │
│  3. USAGE FEATURES (from player_name)                                        │
│     └─ snap_share = get_snap_share(player_name)                              │
│                                                                              │
│  4. DEFENSE FEATURES (if opponent + position available)                      │
│     └─ opp_wr1_receptions_allowed = get_wr1_defense_stat(opponent)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL TRAINING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  XGBoost Classifier per market:                                              │
│                                                                              │
│  player_receptions model:                                                    │
│    Features: [line_vs_trailing, ..., snap_share, avg_separation, ...]       │
│    Target: under_hit (0 or 1)                                                │
│    Output: P(under)                                                          │
│                                                                              │
│  Walk-forward validation:                                                    │
│    Train on weeks 1-N, test on week N+1                                      │
│    Find optimal threshold per market                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  generate_unified_recommendations_v3.py                                      │
│  ─────────────────────────────────────                                       │
│                                                                              │
│  For each prop bet:                                                          │
│                                                                              │
│  1. Load model_predictions (from Monte Carlo)                                │
│     └─ Has: model_projection, model_std, player_id, position, opponent       │
│                                                                              │
│  2. Extract betting features (calculate_v12_features)                        │
│     └─ 14 betting tendency features                                          │
│                                                                              │
│  3. Extract skill features (if player_id available)                          │
│     └─ 6 skill features                                                      │
│                                                                              │
│  4. Build feature vector [20 features]                                       │
│     └─ Fill missing with FEATURE_DEFAULTS                                    │
│                                                                              │
│  5. Model prediction                                                         │
│     └─ p_under = model.predict(features)                                     │
│                                                                              │
│  6. Apply threshold                                                          │
│     └─ if p_under > threshold → recommend UNDER                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        MONTE CARLO SIMULATOR                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  player_simulator_v4.py                                                      │
│  ─────────────────────                                                       │
│                                                                              │
│  Uses skill features for simulation (NOT model prediction):                  │
│                                                                              │
│  1. Dynamic Catch Rate:                                                      │
│     base_rate = player_input.trailing_catch_rate or position_default         │
│     coverage_adj = 1.0 - (opp_def_epa × 0.15)                                │
│     catch_rate = base_rate × coverage_adj  [bounded 0.45-0.85]               │
│                                                                              │
│  2. Reception Simulation:                                                    │
│     targets ~ NegBin(μ, σ²)                                                  │
│     receptions = targets × catch_rate                                        │
│                                                                              │
│  Note: The simulator uses features DIFFERENTLY than the classifier!          │
│        - Classifier: predicts P(under) from all 20 features                  │
│        - Simulator: uses features to parameterize distributions              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Current Gap Status

### Before Fix (Model trained with 15 features)
```
Trained features: [14 betting + snap_share]
Missing: avg_separation, avg_cushion, trailing_catch_rate, target_share, opp_wr1_receptions_allowed

Why missing: Training data lacked player_id, position, opponent columns
```

### After Fix (To be retrained with 20 features)
```
Training data will have: player_id, position, opponent_team (merged from stats)
Model will train on: [14 betting + 6 skill] = 20 features
```

---

## 4. Feature Usage by Component

| Component | Betting Features | Skill Features | Notes |
|-----------|-----------------|----------------|-------|
| **V17 Classifier** | ✅ All 14 | ✅ All 6 | Predicts P(under) |
| **Monte Carlo Sim** | ❌ Not used | ✅ catch_rate, opp_def | Parameterizes distributions |
| **Recommendations** | ✅ All 14 | ✅ All 6 | Uses classifier output |

---

## 5. Files Modified

| File | Changes |
|------|---------|
| `configs/model_config.py` | Added V17_SKILL_FEATURES |
| `nfl_quant/schemas.py` | Added trailing_catch_rate field |
| `nfl_quant/features/core.py` | Added skill feature extractors |
| `nfl_quant/features/defensive_metrics.py` | Added WR1-specific defense |
| `nfl_quant/simulation/player_simulator_v4.py` | Dynamic catch rate |
| `scripts/train/train_v12_interaction_model_v2.py` | Merge player context |
| `scripts/predict/generate_unified_recommendations_v3.py` | Extract skill features |
