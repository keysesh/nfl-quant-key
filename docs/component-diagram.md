# NFL QUANT Component Diagrams

**Generated**: December 2025 | **Model Version**: V24

This document contains ASCII diagrams showing the system architecture from various perspectives.

---

## 1. System Context Diagram

Shows NFL QUANT in relation to external systems.

```
                                    ┌─────────────────────────────┐
                                    │         USER                │
                                    │   (Sports Bettor/Analyst)   │
                                    └─────────────┬───────────────┘
                                                  │
                                                  │ runs pipeline
                                                  │ views recommendations
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                            NFL QUANT SYSTEM                                 │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                     Python Application                          │     │
│    │                                                                 │     │
│    │   Data Fetching → Feature Engineering → ML Models → Outputs    │     │
│    │                                                                 │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   The Odds API  │  │  Sleeper API    │  │   NFLverse      │
│                 │  │                 │  │   (via R)       │
│ • Player props  │  │ • Injury data   │  │ • Player stats  │
│ • Game lines    │  │                 │  │ • Snap counts   │
│ • Live odds     │  │                 │  │ • Rosters       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 2. Container Diagram

Shows the high-level containers within NFL QUANT.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              NFL QUANT SYSTEM                                 │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        SCRIPTS LAYER                                    │  │
│  │                                                                         │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │   │ run_pipeline │  │ train_model  │  │ generate_    │                 │  │
│  │   │     .py      │  │    .py       │  │ recomm.py    │                 │  │
│  │   │              │  │              │  │              │                 │  │
│  │   │ Orchestrates │  │ Trains       │  │ Creates      │                 │  │
│  │   │ full flow    │  │ XGBoost      │  │ ranked picks │                 │  │
│  │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │  │
│  │          │                 │                 │                          │  │
│  └──────────┼─────────────────┼─────────────────┼──────────────────────────┘  │
│             │                 │                 │                             │
│             └─────────────────┼─────────────────┘                             │
│                               │                                               │
│                               ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        NFL_QUANT PACKAGE                                │  │
│  │                                                                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │  │
│  │  │  features/ │  │  models/   │  │  betting/  │  │   utils/   │       │  │
│  │  │            │  │            │  │            │  │            │       │  │
│  │  │ • batch_   │  │ • registry │  │ • kelly    │  │ • player_  │       │  │
│  │  │   extractor│  │ • shrinkage│  │ • sizing   │  │   names    │       │  │
│  │  │ • opponent │  │ • td_pred  │  │            │  │ • team_    │       │  │
│  │  │ • position │  │            │  │            │  │   names    │       │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘       │  │
│  │                                                                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                        │  │
│  │  │   data/    │  │calibration/│  │ simulation/│                        │  │
│  │  │            │  │            │  │            │                        │  │
│  │  │ • fetcher  │  │ • isotonic │  │ • monte_   │                        │  │
│  │  │ • loaders  │  │ • position │  │   carlo    │                        │  │
│  │  │            │  │            │  │ (legacy)   │                        │  │
│  │  └────────────┘  └────────────┘  └────────────┘                        │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                               │                                               │
│                               ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        CONFIGURATION                                    │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────┐  ┌──────────────────────────┐            │  │
│  │  │ configs/model_config.py  │  │ nfl_quant/config_paths.py│            │  │
│  │  │                          │  │                          │            │  │
│  │  │ • MODEL_VERSION          │  │ • PROJECT_ROOT           │            │  │
│  │  │ • FEATURES (46 cols)     │  │ • DATA_DIR               │            │  │
│  │  │ • MARKET_SNR_CONFIG      │  │ • MODELS_DIR             │            │  │
│  │  │ • MODEL_PARAMS           │  │ • get_*_file()           │            │  │
│  │  └──────────────────────────┘  └──────────────────────────┘            │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                               │                                               │
│                               ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          DATA LAYER                                     │  │
│  │                                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │  │
│  │  │ data/nflverse│  │ data/backtest│  │ data/models/ │                  │  │
│  │  │              │  │              │  │              │                  │  │
│  │  │ • weekly_    │  │ • combined_  │  │ • active_    │                  │  │
│  │  │   stats.pqt  │  │   odds_      │  │   model.     │                  │  │
│  │  │ • snap_      │  │   actuals_   │  │   joblib     │                  │  │
│  │  │   counts.pqt │  │   ENRICHED   │  │              │                  │  │
│  │  │ • rosters    │  │   .csv       │  │              │                  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Engineering Component Diagram

Detailed view of the feature engineering subsystem.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING SUBSYSTEM                          │
│                                                                             │
│  INPUT: odds_with_trailing DataFrame (player, line, trailing stats)        │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                   batch_extractor.py                               │     │
│  │                   (Main Orchestrator)                              │     │
│  │                                                                    │     │
│  │   extract_features_batch(odds_with_trailing, historical, market)  │     │
│  │                                                                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│        ┌─────────────────────┼─────────────────────┐                       │
│        │                     │                     │                        │
│        ▼                     ▼                     ▼                        │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │
│  │ Core V12      │    │ V17 Skill     │    │ V18 Context   │               │
│  │ Features      │    │ Features      │    │ Features      │               │
│  │               │    │               │    │               │               │
│  │• line_vs_     │    │• snap_share   │    │• vegas_total  │               │
│  │  trailing     │    │• target_share │    │• vegas_spread │               │
│  │• line_level   │    │• avg_         │    │• game_pace    │               │
│  │• sweet_spot   │    │  separation   │    │• implied_     │               │
│  │• player_      │    │• catch_rate   │    │  team_total   │               │
│  │  under_rate   │    │• opp_wr1_     │    │• pressure_    │               │
│  │• LVT_x_*      │    │  allowed      │    │  rate         │               │
│  └───────────────┘    └───────────────┘    └───────────────┘               │
│                              │                                              │
│        ┌─────────────────────┼─────────────────────┐                       │
│        │                     │                     │                        │
│        ▼                     ▼                     ▼                        │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │
│  │ V19 Rush/Rec  │    │ V23 Opponent  │    │ V24 Position  │               │
│  │ Features      │    │ Features      │    │ Matchup       │               │
│  │               │    │               │    │               │               │
│  │• oline_health │    │• opp_pass_    │    │• pos_rank     │               │
│  │• box_count    │    │  def_vs_avg   │    │• is_starter   │               │
│  │• slot_snap_%  │    │• opp_rush_    │    │• slot_align%  │               │
│  │• target_share │    │  def_vs_avg   │    │• opp_yards_   │               │
│  │  _trailing    │    │• opp_def_epa  │    │  allowed      │               │
│  │               │    │• has_opponent │    │• coverage_adj │               │
│  │               │    │  _context     │    │• has_position │               │
│  └───────────────┘    └───────────────┘    └───────────────┘               │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    feature_defaults.py                             │     │
│  │                                                                    │     │
│  │   safe_fillna(df, FEATURE_DEFAULTS)                               │     │
│  │   • XGBoost handles NaN natively for opponent features            │     │
│  │   • Only fills where semantic default exists                       │     │
│  │                                                                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│  OUTPUT: DataFrame with 46 feature columns                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Training Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA LOADING                                  │   │
│  │                                                                      │   │
│  │  ┌──────────────────────┐    ┌──────────────────────┐               │   │
│  │  │ combined_odds_       │    │ player_stats_        │               │   │
│  │  │ actuals_ENRICHED.csv │    │ 2023/2024/2025.csv   │               │   │
│  │  │                      │    │                      │               │   │
│  │  │ • player, market     │    │ • receptions         │               │   │
│  │  │ • line, actual_stat  │    │ • receiving_yards    │               │   │
│  │  │ • opponent           │    │ • rushing_yards      │               │   │
│  │  │ • vegas_total/spread │    │ • snap_counts        │               │   │
│  │  │ • target_share_stats │    │                      │               │   │
│  │  └──────────┬───────────┘    └──────────┬───────────┘               │   │
│  │             │                           │                            │   │
│  │             └─────────────┬─────────────┘                            │   │
│  │                           ▼                                          │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TRAILING STATS MERGE                              │   │
│  │                                                                      │   │
│  │   prepare_data_with_trailing(odds, stats)                           │   │
│  │                                                                      │   │
│  │   • Join odds with player stats by (player, season, week)           │   │
│  │   • Calculate EWMA trailing averages (shift(1) to avoid leakage)    │   │
│  │   • Add vegas context from enriched odds                             │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FEATURE EXTRACTION                                │   │
│  │                                                                      │   │
│  │   for market in CLASSIFIER_MARKETS:                                 │   │
│  │       features = extract_features_batch(                             │   │
│  │           odds_with_trailing[market],                                │   │
│  │           historical_odds[global_week < train_week],                 │   │
│  │           market                                                     │   │
│  │       )                                                              │   │
│  │                                                                      │   │
│  │   Output: DataFrame with 46 features + 'label' column               │   │
│  │           label = 1 if actual_stat < line (UNDER hit)               │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WALK-FORWARD TRAINING                             │   │
│  │                                                                      │   │
│  │   for test_week in range(5, max_week):                              │   │
│  │       train_data = data[global_week < test_week - 1]                │   │
│  │       test_data = data[global_week == test_week]                     │   │
│  │                                                                      │   │
│  │       model = XGBClassifier(**MODEL_PARAMS)                         │   │
│  │       model.fit(train_data[FEATURES], train_data['label'])          │   │
│  │       predictions = model.predict_proba(test_data[FEATURES])        │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MODEL SERIALIZATION                               │   │
│  │                                                                      │   │
│  │   model_artifact = {                                                 │   │
│  │       'version': 'V24',                                             │   │
│  │       'trained_date': datetime.now(),                                │   │
│  │       'features': FEATURES,                                          │   │
│  │       'models': {                                                    │   │
│  │           'player_receptions': xgb_classifier,                       │   │
│  │           'player_rush_yds': xgb_classifier,                         │   │
│  │           'player_reception_yds': xgb_classifier,                    │   │
│  │           ...                                                        │   │
│  │       }                                                              │   │
│  │   }                                                                  │   │
│  │                                                                      │   │
│  │   joblib.dump(model_artifact, 'data/models/active_model.joblib')    │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Prediction Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREDICTION PIPELINE                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA LOADING                                    │   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │   │
│  │  │ active_model   │  │ nfl_player_    │  │ weekly_stats   │         │   │
│  │  │ .joblib        │  │ props_dk.csv   │  │ .parquet       │         │   │
│  │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘         │   │
│  │          │                   │                   │                   │   │
│  └──────────┼───────────────────┼───────────────────┼───────────────────┘   │
│             │                   │                   │                       │
│             ▼                   ▼                   ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FEATURE EXTRACTION                                │   │
│  │                                                                      │   │
│  │   for each player-prop in live_odds:                                │   │
│  │       features = extract_features_batch(                             │   │
│  │           player_odds_with_trailing,                                 │   │
│  │           historical_odds,                                           │   │
│  │           market                                                     │   │
│  │       )                                                              │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MODEL INFERENCE                                   │   │
│  │                                                                      │   │
│  │   for market in ['player_receptions', 'player_rush_yds', ...]:      │   │
│  │       model = model_artifact['models'][market]                       │   │
│  │       probs = model.predict_proba(features)[:, 1]  # P(UNDER)       │   │
│  │       df['clf_prob_under'] = probs                                   │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EDGE CALCULATION                                  │   │
│  │                                                                      │   │
│  │   implied_prob = 1 / american_to_decimal(odds)                       │   │
│  │   model_prob = clf_prob_under  # or 1 - clf_prob_under for OVER      │   │
│  │                                                                      │   │
│  │   edge = model_prob - implied_prob                                   │   │
│  │   direction = 'UNDER' if clf_prob_under > 0.5 else 'OVER'           │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SNR FILTERING                                     │   │
│  │                                                                      │   │
│  │   from configs.model_config import MARKET_SNR_CONFIG                │   │
│  │                                                                      │   │
│  │   for each bet:                                                      │   │
│  │       config = MARKET_SNR_CONFIG[market]                             │   │
│  │                                                                      │   │
│  │       if not config.enabled:                                         │   │
│  │           FILTER: market disabled                                    │   │
│  │                                                                      │   │
│  │       if model_confidence < config.confidence_threshold:             │   │
│  │           FILTER: confidence too low                                 │   │
│  │                                                                      │   │
│  │       if abs(line_vs_trailing) < config.min_line_deviation:          │   │
│  │           FILTER: line deviation too small                           │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    KELLY SIZING                                      │   │
│  │                                                                      │   │
│  │   kelly_fraction = (model_prob * (odds + 1) - 1) / odds             │   │
│  │   half_kelly = kelly_fraction / 2                                    │   │
│  │                                                                      │   │
│  │   units = min(half_kelly * bankroll / unit_size, 2)  # max 2 units  │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    OUTPUT GENERATION                                 │   │
│  │                                                                      │   │
│  │   Sort by: confidence * edge (descending)                           │   │
│  │                                                                      │   │
│  │   Output files:                                                      │   │
│  │   • reports/recommendations_week{N}_{timestamp}.csv                 │   │
│  │   • data/picks/{market}_filtered.csv                                 │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Freshness Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA FRESHNESS REQUIREMENTS                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PRE-GAME (Day of Games)                           │   │
│  │                                                                      │   │
│  │   1. NFLverse Data (max age: 12h)                                   │   │
│  │      ┌──────────────────────────────────────────────────────┐       │   │
│  │      │ Rscript scripts/fetch/fetch_nflverse_data.R          │       │   │
│  │      │                                                       │       │   │
│  │      │ → weekly_stats.parquet    (player game logs)         │       │   │
│  │      │ → snap_counts.parquet     (participation %)          │       │   │
│  │      │ → depth_charts.parquet    (position rankings)        │       │   │
│  │      │ → rosters.parquet         (team assignments)         │       │   │
│  │      └──────────────────────────────────────────────────────┘       │   │
│  │                           │                                          │   │
│  │                           ▼                                          │   │
│  │   2. Injuries (max age: 6h)                                         │   │
│  │      ┌──────────────────────────────────────────────────────┐       │   │
│  │      │ python scripts/fetch/fetch_injuries_sleeper.py       │       │   │
│  │      │                                                       │       │   │
│  │      │ → injuries.parquet        (injury status, practice)  │       │   │
│  │      └──────────────────────────────────────────────────────┘       │   │
│  │                           │                                          │   │
│  │                           ▼                                          │   │
│  │   3. Live Odds (max age: 4h, refresh 1h before kickoff)             │   │
│  │      ┌──────────────────────────────────────────────────────┐       │   │
│  │      │ python scripts/fetch/fetch_live_odds.py              │       │   │
│  │      │ python scripts/fetch/fetch_nfl_player_props.py       │       │   │
│  │      │                                                       │       │   │
│  │      │ → odds_week{N}.csv        (game totals/spreads)      │       │   │
│  │      │ → nfl_player_props_dk.csv (player prop lines)        │       │   │
│  │      └──────────────────────────────────────────────────────┘       │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FRESHNESS CHECK                                   │   │
│  │                                                                      │   │
│  │   python scripts/fetch/check_data_freshness.py                      │   │
│  │                                                                      │   │
│  │   Checks:                                                            │   │
│  │   ┌────────────────────┬──────────────┬──────────────────────┐      │   │
│  │   │ File               │ Max Age      │ Status               │      │   │
│  │   ├────────────────────┼──────────────┼──────────────────────┤      │   │
│  │   │ weekly_stats.pqt   │ 12 hours     │ [OK] / [STALE]       │      │   │
│  │   │ snap_counts.pqt    │ 12 hours     │ [OK] / [STALE]       │      │   │
│  │   │ injuries.pqt       │ 6 hours      │ [OK] / [STALE]       │      │   │
│  │   │ odds_player_props  │ 4 hours      │ [OK] / [STALE]       │      │   │
│  │   └────────────────────┴──────────────┴──────────────────────┘      │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Anti-Leakage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ANTI-LEAKAGE MECHANISMS                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. TRAILING STATS (shift(1))                      │   │
│  │                                                                      │   │
│  │   Week 10 Prediction:                                                │   │
│  │                                                                      │   │
│  │   Week:     1    2    3    4    5    6    7    8    9   [10]        │   │
│  │   Stats:   [x]  [x]  [x]  [x]  [x]  [x]  [x]  [x]  [x]  [?]         │   │
│  │   Used:    ────────────────────────────────────────────▶             │   │
│  │            Trailing average uses weeks 1-9 only                      │   │
│  │            Week 10 actual is NEVER used in features                  │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. OPPONENT DEFENSE (shift(1))                    │   │
│  │                                                                      │   │
│  │   Calculating opponent strength vs position:                         │   │
│  │                                                                      │   │
│  │   Defense Week: 1    2    3    4    5    6    7    8    9   [10]    │   │
│  │   Games Played: [x]  [x]  [x]  [x]  [x]  [x]  [x]  [x]  [x]  [?]    │   │
│  │   Used:         ────────────────────────────────────────▶            │   │
│  │                 Defense stats use opponent's prior 9 games only      │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. WALK-FORWARD VALIDATION                        │   │
│  │                                                                      │   │
│  │   Testing Week 10:                                                   │   │
│  │                                                                      │   │
│  │   Week:     1    2    3    4    5    6    7    8   [9]  [10]        │   │
│  │   Role:    ─────────TRAIN──────────────────────▶  gap   TEST        │   │
│  │                                                                      │   │
│  │   Training data: weeks 1-8 (global_week < test_week - 1)            │   │
│  │   Gap week: 9 (excluded to prevent any overlap)                      │   │
│  │   Test data: week 10 only                                            │   │
│  │                                                                      │   │
│  │   This prevents:                                                     │   │
│  │   • Using week 10 data in training                                   │   │
│  │   • Using week 9 data that might contain week 10 signal              │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    4. GLOBAL WEEK ORDERING                           │   │
│  │                                                                      │   │
│  │   Cross-season ordering to prevent 2024 week 1 being "after" 2023:  │   │
│  │                                                                      │   │
│  │   global_week = (season - 2023) * 18 + week                         │   │
│  │                                                                      │   │
│  │   2023 Week 1  → global_week = 1                                    │   │
│  │   2023 Week 18 → global_week = 18                                   │   │
│  │   2024 Week 1  → global_week = 19                                   │   │
│  │   2024 Week 18 → global_week = 36                                   │   │
│  │   2025 Week 1  → global_week = 37                                   │   │
│  │                                                                      │   │
│  │   Ensures temporal ordering is maintained across season boundaries   │   │
│  │                                                                      │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Model Artifact Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    active_model.joblib STRUCTURE                            │
│                                                                             │
│  {                                                                          │
│      'version': 'V24',                                                     │
│      'trained_date': '2025-12-07T14:30:00',                                │
│      'feature_count': 46,                                                   │
│      'features': [                                                          │
│          'line_vs_trailing',                                                │
│          'line_level',                                                      │
│          'line_in_sweet_spot',                                              │
│          ...                         # 46 feature names                     │
│          'has_position_context'                                             │
│      ],                                                                     │
│      'models': {                                                            │
│          'player_receptions': XGBClassifier(                                │
│              n_estimators=150,                                              │
│              max_depth=4,                                                   │
│              learning_rate=0.1,                                             │
│              ...                                                            │
│          ),                                                                 │
│          'player_rush_yds': XGBClassifier(...),                             │
│          'player_reception_yds': XGBClassifier(...),                        │
│          'player_pass_yds': XGBClassifier(...),                             │
│          'player_rush_attempts': XGBClassifier(...)                         │
│      },                                                                     │
│      'training_metrics': {                                                  │
│          'player_receptions': {                                             │
│              'auc': 0.72,                                                   │
│              'hit_rate_70': 0.794,                                          │
│              'samples': 12500                                               │
│          },                                                                 │
│          ...                                                                │
│      },                                                                     │
│      'feature_importances': {                                               │
│          'player_receptions': {                                             │
│              'target_share': 0.174,                                         │
│              'line_vs_trailing': 0.156,                                     │
│              'has_opponent_context': 0.097,                                 │
│              ...                                                            │
│          },                                                                 │
│          ...                                                                │
│      }                                                                      │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: December 2025
