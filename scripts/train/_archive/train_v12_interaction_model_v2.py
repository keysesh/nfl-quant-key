#!/usr/bin/env python3
"""
V17 Interaction Model (NFL QUANT)

This version uses the centralized FeatureEngine for ALL feature calculations.
No more inline feature calculations - single source of truth.

V17 UPGRADES:
- Gaussian decay sweet spot (replaces binary 0/1)
- LVT × defense EPA interaction
- LVT × rest days interaction
- Centralized version management via configs/model_config.py

FEATURES FROM FeatureEngine:
- calculate_trailing_stat() with no_leakage=True
- calculate_player_under_rate() using only prior weeks
- calculate_player_bias() using only prior outcomes
- calculate_market_under_rate() using only prior weeks
- calculate_v12_features() for all features (V12 base + V17 extensions)

Walk-Forward Results (V16):
- player_receptions: +10.1% ROI at 70% threshold (VALIDATED)
- player_reception_yds: +2% ROI at 60% threshold (MARGINAL)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import joblib
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CENTRALIZED IMPORTS - Single source of truth
from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.features.feature_defaults import safe_fillna, FEATURE_DEFAULTS
from nfl_quant.utils.player_names import normalize_player_name

# V18: Import from centralized config
from configs.model_config import (
    MODEL_VERSION,
    MODEL_VERSION_FULL,
    V12_FEATURE_COLS,
    V17_FEATURE_COLS,
    V18_FEATURE_COLS,
    CURRENT_FEATURE_COLS,
    FEATURE_FLAGS,
    get_monotonic_constraints,
    get_interaction_constraints,
    get_model_path,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Feature columns - from centralized config
# V18 includes: game_pace, vegas_total, vegas_spread, implied_team_total, adot, pressure_rate, opp_pressure_rate
FEATURE_COLS = CURRENT_FEATURE_COLS  # V18 with all Phase 1 features


def load_data():
    """Load all historical data."""
    logger.info("Loading data...")

    # Odds with actuals
    odds_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    odds = pd.read_csv(odds_path)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    # Deduplicate (keep primary line)
    odds['group_key'] = odds['player_norm'] + '_' + odds['season'].astype(str) + '_' + odds['week'].astype(str) + '_' + odds['market']
    market_medians = odds.groupby('market')['line'].median()

    def get_primary_line(group):
        if len(group) == 1:
            return group
        market = group['market'].iloc[0]
        median = market_medians.get(market, group['line'].median())
        group['dist_from_median'] = abs(group['line'] - median)
        return group.nsmallest(1, 'dist_from_median')

    odds = odds.groupby('group_key', group_keys=False).apply(get_primary_line)
    odds = odds.drop(columns=['dist_from_median', 'group_key'], errors='ignore')

    # Add global week for temporal ordering
    odds['global_week'] = (odds['season'] - 2023) * 18 + odds['week']

    # Player stats
    stats_2024 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)
    stats_2023 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2023.csv', low_memory=False)
    stats = pd.concat([stats_2024, stats_2023], ignore_index=True)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

    logger.info(f"  Odds: {len(odds)} rows")
    logger.info(f"  Stats: {len(stats)} rows")

    return odds, stats


def prepare_data_with_trailing(odds: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data with trailing stats using centralized FeatureEngine.

    NO INLINE CALCULATIONS - all trailing stats from FeatureEngine.
    """
    logger.info("Calculating trailing stats using FeatureEngine...")

    engine = get_feature_engine()

    # Sort stats for proper calculation
    stats = stats.sort_values(['player_norm', 'season', 'week'])
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    # CENTRALIZED: Calculate all trailing stats
    stat_cols = ['receptions', 'receiving_yards', 'rushing_yards', 'passing_yards']
    for col in stat_cols:
        if col in stats.columns:
            stats[f'trailing_{col}'] = calculate_trailing_stat(
                df=stats,
                stat_col=col,
                player_col='player_norm',
                span=4,
                min_periods=1,
                no_leakage=True  # CRITICAL: shift(1) applied
            )
            logger.info(f"  ✓ Calculated trailing_{col}")

    # Merge trailing stats AND player context to odds
    # V18 Enhancement: Include player_id, position, team, opponent_team for all features
    trailing_cols = [col for col in stats.columns if 'trailing_' in col]
    context_cols = ['player_id', 'position', 'team', 'opponent_team']  # For V18 feature extraction
    available_context = [c for c in context_cols if c in stats.columns]
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols + available_context
    stats_dedup = stats[merge_cols].drop_duplicates(subset=['player_norm', 'season', 'week'])
    odds_merged = odds.merge(stats_dedup, on=['player_norm', 'season', 'week'], how='left')

    # Rename for consistency
    if 'opponent_team' in odds_merged.columns:
        odds_merged['opponent'] = odds_merged['opponent_team']

    logger.info(f"  Added context columns: {available_context}")

    logger.info(f"  Merged data: {len(odds_merged)} rows")
    return odds_merged


def train_v12_refactored(odds_merged: pd.DataFrame, market: str, window_weeks: int = 20) -> dict:
    """
    Train V12 model using centralized FeatureEngine.

    ALL features calculated via FeatureEngine - NO inline calculations.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training V12 (Refactored): {market}")
    logger.info(f"Using centralized FeatureEngine")
    logger.info(f"{'='*60}")

    engine = get_feature_engine()

    # Map market to stat column
    stat_col_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
    }
    trailing_col = f"trailing_{stat_col_map.get(market)}"

    # Filter to market and drop rows without trailing
    market_odds = odds_merged[odds_merged['market'] == market].copy()
    market_odds = market_odds.dropna(subset=[trailing_col])

    if len(market_odds) < 200:
        logger.warning(f"Not enough data for {market}")
        return None

    # Walk-forward validation on 2025
    test_start_week = (2025 - 2023) * 18 + 1
    test_weeks = sorted(market_odds[market_odds['global_week'] >= test_start_week]['global_week'].unique())

    all_preds = []
    all_actuals = []
    all_lvt = []

    logger.info(f"  Walk-forward validation on {len(test_weeks)} weeks...")

    for test_week in test_weeks:
        # Get training weeks (before test week)
        train_weeks = sorted([w for w in market_odds['global_week'].unique() if w < test_week])[-window_weeks:]

        if len(train_weeks) < 5:
            continue

        # CENTRALIZED: Extract features for training weeks
        train_data_list = []
        for train_week in train_weeks:
            week_features = engine.extract_v12_features_for_week(
                odds_with_trailing=odds_merged,
                all_historical_odds=odds_merged,
                target_global_week=train_week,
                market=market
            )
            if len(week_features) > 0:
                train_data_list.append(week_features)

        if len(train_data_list) == 0:
            continue

        train_data = pd.concat(train_data_list, ignore_index=True)

        # CENTRALIZED: Extract features for test week
        test_data = engine.extract_v12_features_for_week(
            odds_with_trailing=odds_merged,
            all_historical_odds=odds_merged,
            target_global_week=test_week,
            market=market
        )

        if len(test_data) == 0:
            continue

        # Get available features (use centralized FEATURE_COLS)
        available_features = [f for f in FEATURE_COLS if f in train_data.columns and f in test_data.columns]

        # Clean data - apply safe_fillna with semantic defaults before dropna
        train_subset = train_data[available_features + ['under_hit']].copy()
        test_subset = test_data[available_features + ['under_hit']].copy()
        train_subset[available_features] = safe_fillna(train_subset[available_features])
        test_subset[available_features] = safe_fillna(test_subset[available_features])
        train_clean = train_subset.dropna(subset=['under_hit'])
        test_clean = test_subset.dropna(subset=['under_hit'])

        if len(train_clean) < 50 or len(test_clean) == 0:
            continue

        X_train = train_clean[available_features]
        y_train = train_clean['under_hit']
        X_test = test_clean[available_features]
        y_test = test_clean['under_hit']

        # Build XGBoost constraints
        feature_indices = {f: i for i, f in enumerate(available_features)}
        lvt_idx = feature_indices.get('line_vs_trailing', 0)

        interaction_constraints = [[lvt_idx] + list(range(len(available_features)))]
        for i in range(len(available_features)):
            if i != lvt_idx:
                interaction_constraints.append([lvt_idx, i])

        monotonic = []
        for f in available_features:
            if f == 'line_vs_trailing':
                monotonic.append(1)
            elif f == 'player_under_rate':
                monotonic.append(1)
            elif f == 'market_under_rate':
                monotonic.append(1)
            elif f == 'player_bias':
                monotonic.append(-1)
            else:
                monotonic.append(0)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'interaction_constraints': str(interaction_constraints),
            'monotone_constraints': '(' + ','.join(map(str, monotonic)) + ')',
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params, dtrain, num_boost_round=200,
            evals=[(dtest, 'test')], early_stopping_rounds=30,
            verbose_eval=False
        )

        preds = model.predict(dtest)
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)
        all_lvt.extend(X_test['line_vs_trailing'].values)

    if len(all_preds) == 0:
        return None

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Evaluate
    logger.info("\n  === VALIDATION RESULTS (FeatureEngine) ===")
    results = {}

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = all_preds >= threshold
        n_bets = mask.sum()

        if n_bets < 10:
            continue

        hit_rate = all_actuals[mask].mean()
        profits = [0.909 if a == 1 else -1.0 for a in all_actuals[mask]]
        roi = np.mean(profits) * 100 if profits else 0

        results[threshold] = {'n_bets': n_bets, 'hit_rate': hit_rate, 'roi': roi}
        logger.info(f"    Threshold {threshold:.0%}: N={n_bets}, Hit={hit_rate:.1%}, ROI={roi:+.1f}%")

    # Train final model
    logger.info("\n  Training final production model...")

    all_train_data = []
    all_weeks = sorted(market_odds['global_week'].unique())

    for week in all_weeks:
        week_features = engine.extract_v12_features_for_week(
            odds_with_trailing=odds_merged,
            all_historical_odds=odds_merged,
            target_global_week=week,
            market=market
        )
        if len(week_features) > 0:
            all_train_data.append(week_features)

    final_train = pd.concat(all_train_data, ignore_index=True)
    available_features = [f for f in FEATURE_COLS if f in final_train.columns]

    # Apply safe_fillna with semantic defaults before dropna
    final_subset = final_train[available_features + ['under_hit']].copy()
    final_subset[available_features] = safe_fillna(final_subset[available_features])
    final_clean = final_subset.dropna(subset=['under_hit'])

    X_final = final_clean[available_features]
    y_final = final_clean['under_hit']

    dtrain_final = xgb.DMatrix(X_final, label=y_final)
    final_model = xgb.train(params, dtrain_final, num_boost_round=200, verbose_eval=False)

    # Feature importance
    importance = final_model.get_score(importance_type='gain')
    logger.info(f"\n  Feature Importance (Gain):")
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6]:
        pct = score / sum(importance.values()) * 100
        logger.info(f"    {feat}: {pct:.1f}%")

    # Best threshold
    best_threshold = 0.55
    best_roi = -999
    for thresh, res in results.items():
        if res['roi'] > best_roi and res['n_bets'] >= 20:
            best_roi = res['roi']
            best_threshold = thresh

    return {
        'model': final_model,
        'feature_cols': available_features,
        'results': results,
        'best_threshold': best_threshold,
        'best_roi': best_roi,
        'feature_importance': importance,
    }


def main():
    print("="*80)
    print(f"{MODEL_VERSION_FULL} INTERACTION MODEL (NFL QUANT)")
    print("Using centralized FeatureEngine - Single Source of Truth")
    print("="*80)
    print()
    print(f"MODEL VERSION: {MODEL_VERSION_FULL}")
    print(f"FEATURE FLAGS:")
    print(f"  - Gaussian Sweet Spot: {FEATURE_FLAGS.use_smooth_sweet_spot}")
    print(f"  - LVT x Defense: {FEATURE_FLAGS.use_lvt_x_defense}")
    print(f"  - LVT x Rest: {FEATURE_FLAGS.use_lvt_x_rest}")
    print()
    print("FEATURES FROM FeatureEngine:")
    print("  - calculate_trailing_stat() with no_leakage=True")
    print("  - calculate_player_under_rate() using only prior weeks")
    print("  - calculate_player_bias() using only prior outcomes")
    print("  - calculate_market_under_rate() using only prior weeks")
    print(f"  - calculate_v12_features() for all {len(FEATURE_COLS)} features")
    print()

    # Load data
    odds, stats = load_data()

    # Prepare data with trailing stats (using FeatureEngine)
    odds_merged = prepare_data_with_trailing(odds, stats)

    # Train on all 4 markets (rush_yds and pass_yds added)
    markets = ['player_receptions', 'player_reception_yds', 'player_rush_yds', 'player_pass_yds']

    all_models = {}
    all_metrics = {}

    for market in markets:
        result = train_v12_refactored(odds_merged, market)
        if result:
            all_models[market] = result['model']
            all_metrics[market] = result

    # Summary
    print()
    print("="*80)
    print(f"{MODEL_VERSION_FULL} MODEL SUMMARY")
    print("="*80)
    print()
    print(f"{'Market':<25} {'Thresh':>8} {'N Bets':>8} {'Hit%':>8} {'ROI':>10}")
    print("-"*65)

    for market, m in all_metrics.items():
        best = m['results'].get(m['best_threshold'], {})
        print(f"{market:<25} {m['best_threshold']:>7.0%} {best.get('n_bets', 0):>8} {best.get('hit_rate', 0):>7.1%} {best.get('roi', 0):>+9.1f}%")

    # Save using centralized path
    output_path = get_model_path()
    bundle = {
        'models': all_models,
        'metrics': all_metrics,
        'version': f'v{MODEL_VERSION}',
        'trained_date': datetime.now().isoformat(),
        'feature_source': 'nfl_quant.features.core.FeatureEngine',
        'description': f'{MODEL_VERSION_FULL} with Gaussian sweet spot and new interactions',
        'validated_markets': list(all_models.keys()),
        'feature_cols': FEATURE_COLS,
        'feature_flags': {
            'smooth_sweet_spot': FEATURE_FLAGS.use_smooth_sweet_spot,
            'lvt_x_defense': FEATURE_FLAGS.use_lvt_x_defense,
            'lvt_x_rest': FEATURE_FLAGS.use_lvt_x_rest,
        },
    }
    joblib.dump(bundle, output_path)
    print(f"\nSaved to: {output_path}")

    # Also save as v12 for backward compatibility
    v12_path = PROJECT_ROOT / 'data' / 'models' / 'v12_interaction_classifier.joblib'
    joblib.dump(bundle, v12_path)
    print(f"Also saved to: {v12_path} (backward compat)")

    print()
    print("="*80)
    print("FEATURE SOURCE (Single Source of Truth)")
    print("="*80)
    print(f"  All features from: nfl_quant.features.core.FeatureEngine")
    print(f"  Config from: configs/model_config.py")
    print(f"  Data leakage prevention: All features use shift(1)")
    print()
    print(f"{MODEL_VERSION_FULL} Model trained and saved!")


if __name__ == '__main__':
    main()
