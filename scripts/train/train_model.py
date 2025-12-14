#!/usr/bin/env python3
"""
Model Training Script - Version Agnostic

This script trains the model using configuration from configs/model_config.py.
No hardcoded version numbers exist in this file.

To upgrade to a new version:
1. Update MODEL_VERSION in configs/model_config.py
2. Add new features to FEATURES list in configs/model_config.py
3. Run this script

Usage:
    python scripts/train/train_model.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import joblib
import xgboost as xgb
import warnings
import time
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use centralized path configuration
from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, BACKTEST_DIR, MODELS_DIR

# Import from centralized config - NO version numbers imported
from configs.model_config import (
    MODEL_VERSION,
    MODEL_VERSION_FULL,
    FEATURES,
    FEATURE_COUNT,
    FEATURE_FLAGS,
    CLASSIFIER_MARKETS,  # Only train volume markets (no TD props)
    MODEL_PARAMS,
    get_active_model_path,
    get_versioned_model_path,
)

from nfl_quant.features.batch_extractor import extract_features_batch, clear_caches
from nfl_quant.features.feature_defaults import safe_fillna
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.models.classifier_registry import register_model

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load all historical data."""
    logger.info("Loading data...")
    start = time.time()

    # Odds with actuals - use enriched version if available (has vegas_total, vegas_spread, opponent)
    enriched_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    original_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'

    if enriched_path.exists():
        odds_path = enriched_path
        logger.info("  Using ENRICHED odds data (with vegas_total, vegas_spread, opponent, target_share)")
    else:
        odds_path = original_path
        logger.warning("  Using original odds data (enriched not found)")

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

    # Normalize column names (2023 uses 'recent_team', 2024 uses 'team')
    if 'recent_team' in stats_2023.columns and 'team' not in stats_2023.columns:
        stats_2023['team'] = stats_2023['recent_team']

    stats = pd.concat([stats_2024, stats_2023], ignore_index=True)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

    logger.info(f"  Loaded {len(odds):,} odds, {len(stats):,} stats in {time.time()-start:.1f}s")

    return odds, stats


def prepare_data_with_trailing(odds: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with trailing stats."""
    logger.info("Calculating trailing stats...")
    start = time.time()

    # Sort stats for proper calculation
    stats = stats.sort_values(['player_norm', 'season', 'week'])
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    # Calculate all trailing stats using vectorized groupby
    # Expanded to support additional markets (rush attempts, pass completions, TDs, etc.)
    stat_cols = [
        # Core stats (original 4 markets)
        'receptions', 'receiving_yards', 'rushing_yards', 'passing_yards',
        # Additional markets
        'carries',            # rush_attempts
        'completions',        # pass_completions
        'attempts',           # pass_attempts
        'passing_tds',        # pass_tds
        'rushing_tds',        # rush_tds
        'receiving_tds',      # receiving_tds
    ]
    for col in stat_cols:
        if col in stats.columns:
            stats[f'trailing_{col}'] = stats.groupby('player_norm')[col].transform(
                lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
            )

    # Merge trailing stats AND player context to odds
    trailing_cols = [col for col in stats.columns if 'trailing_' in col]
    context_cols = ['player_id', 'position', 'team', 'opponent_team']
    available_context = [c for c in context_cols if c in stats.columns]
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols + available_context
    stats_dedup = stats[merge_cols].drop_duplicates(subset=['player_norm', 'season', 'week'])
    odds_merged = odds.merge(stats_dedup, on=['player_norm', 'season', 'week'], how='left')

    # Fix team column collision (odds has team, stats has team -> team_x, team_y)
    # Priority: odds team (team_x) > stats team (team_y) > player_team
    if 'team_x' in odds_merged.columns and 'team_y' in odds_merged.columns:
        odds_merged['team'] = odds_merged['team_x'].fillna(odds_merged['team_y'])
        odds_merged = odds_merged.drop(columns=['team_x', 'team_y'], errors='ignore')
    elif 'team_x' in odds_merged.columns:
        odds_merged['team'] = odds_merged['team_x']
        odds_merged = odds_merged.drop(columns=['team_x'], errors='ignore')

    # Fallback: use player_team if team is still missing
    if 'team' not in odds_merged.columns or odds_merged['team'].isna().all():
        if 'player_team' in odds_merged.columns:
            odds_merged['team'] = odds_merged['player_team']
    else:
        odds_merged['team'] = odds_merged['team'].fillna(odds_merged.get('player_team'))

    # Rename for consistency
    if 'opponent_team' in odds_merged.columns:
        odds_merged['opponent'] = odds_merged['opponent_team']

    logger.info(f"  Prepared {len(odds_merged):,} rows in {time.time()-start:.1f}s")

    return odds_merged


def train_market(
    odds_merged: pd.DataFrame,
    market: str,
) -> dict:
    """Train model for a market using vectorized feature extraction."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {MODEL_VERSION_FULL}: {market}")
    logger.info(f"{'='*60}")
    start = time.time()

    # Map market to stat column (expanded for additional markets)
    stat_col_map = {
        # Core markets (original 4)
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        # Additional markets
        'player_rush_attempts': 'carries',
        'player_pass_completions': 'completions',
        'player_pass_attempts': 'attempts',
        'player_pass_tds': 'passing_tds',
        'player_rush_tds': 'rushing_tds',
        'player_receiving_tds': 'receiving_tds',
    }
    stat_col = stat_col_map.get(market)
    if stat_col is None:
        logger.warning(f"  No stat column mapping for {market}, skipping")
        return None
    trailing_col = f'trailing_{stat_col}'

    if trailing_col not in odds_merged.columns:
        logger.warning(f"  Missing {trailing_col}, skipping")
        return None

    # Filter to market
    market_data = odds_merged[odds_merged['market'] == market].copy()
    if len(market_data) == 0:
        logger.warning(f"  No data for {market}")
        return None

    # Get available weeks
    weeks = sorted(market_data['global_week'].unique())
    if len(weeks) < 2:
        logger.warning(f"  Not enough weeks for {market}")
        return None

    # Walk-forward validation
    logger.info(f"  Walk-forward validation on {min(10, len(weeks)-1)} weeks...")

    val_start = max(weeks) - 10
    val_weeks = [w for w in weeks if w >= val_start]

    all_preds = []

    for test_week in val_weeks:
        train_data = market_data[market_data['global_week'] < test_week - 1].copy()
        test_data = market_data[market_data['global_week'] == test_week].copy()

        if len(train_data) < 50 or len(test_data) == 0:
            continue

        # Vectorized feature extraction
        train_features = extract_features_batch(
            train_data,
            market_data[market_data['global_week'] < test_week - 1],
            market
        )
        test_features = extract_features_batch(
            test_data,
            market_data[market_data['global_week'] < test_week],
            market
        )

        if len(train_features) == 0 or len(test_features) == 0:
            continue

        # Get available features from config
        available_features = [f for f in FEATURES if f in train_features.columns and f in test_features.columns]

        if len(available_features) < 5:
            continue

        # Prepare X, y
        X_train = safe_fillna(train_features[available_features])
        y_train = train_features['under_hit']
        X_test = safe_fillna(test_features[available_features])
        y_test = test_features['under_hit']

        # Train model using params from config
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=MODEL_PARAMS.max_depth,
            learning_rate=MODEL_PARAMS.learning_rate,
            subsample=MODEL_PARAMS.subsample,
            colsample_bytree=MODEL_PARAMS.colsample_bytree,
            random_state=MODEL_PARAMS.random_state,
            verbosity=MODEL_PARAMS.verbosity
        )
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict_proba(X_test)[:, 1]

        # Store results
        for i, (_, row) in enumerate(test_features.iterrows()):
            all_preds.append({
                'global_week': test_week,
                'player_norm': row.get('player_norm', ''),
                'line': row.get('line', 0),
                'under_hit': row.get('under_hit', 0),
                'p_under': preds[i]
            })

    if len(all_preds) == 0:
        logger.warning("  No predictions generated")
        return None

    preds_df = pd.DataFrame(all_preds)

    # Calculate validation metrics
    logger.info(f"\n  === VALIDATION RESULTS ===")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        high_conf = preds_df[preds_df['p_under'] >= thresh]
        if len(high_conf) > 0:
            hits = high_conf['under_hit'].sum()
            total = len(high_conf)
            hit_rate = hits / total
            roi = (hit_rate * 0.909) - (1 - hit_rate)
            logger.info(f"    Threshold {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

    # Train final production model on ALL data
    logger.info(f"\n  Training final production model...")

    final_features = extract_features_batch(market_data, market_data, market)
    available_features = [f for f in FEATURES if f in final_features.columns]
    X_final = safe_fillna(final_features[available_features])
    y_final = final_features['under_hit']

    final_model = xgb.XGBClassifier(
        n_estimators=MODEL_PARAMS.n_estimators,
        max_depth=MODEL_PARAMS.max_depth,
        learning_rate=MODEL_PARAMS.learning_rate,
        subsample=MODEL_PARAMS.subsample,
        colsample_bytree=MODEL_PARAMS.colsample_bytree,
        random_state=MODEL_PARAMS.random_state,
        verbosity=MODEL_PARAMS.verbosity
    )
    final_model.fit(X_final, y_final)

    # Feature importance
    logger.info(f"\n  Feature Importance (Top 8):")
    importances = dict(zip(available_features, final_model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:8]:
        logger.info(f"    {feat}: {imp:.1%}")

    elapsed = time.time() - start
    logger.info(f"\n  Completed {market} in {elapsed:.1f}s")

    return {
        'model': final_model,
        'features': available_features,
        'validation': preds_df,
        'elapsed': elapsed
    }


def main():
    """Main training function."""
    total_start = time.time()

    print("="*80)
    print(f"MODEL TRAINING - {MODEL_VERSION_FULL}")
    print("="*80)
    print(f"\nVersion: {MODEL_VERSION_FULL} (from configs/model_config.py)")
    print(f"Features: {FEATURE_COUNT} columns")
    print(f"Markets: {', '.join(CLASSIFIER_MARKETS)}")

    # Load data
    odds, stats = load_data()

    # Prepare with trailing stats
    odds_merged = prepare_data_with_trailing(odds, stats)

    # Train each market
    results = {}
    for market in CLASSIFIER_MARKETS:
        result = train_market(odds_merged, market)
        if result:
            results[market] = result

    # Clear caches
    clear_caches()

    # Summary
    print("\n" + "="*80)
    print(f"{MODEL_VERSION_FULL} MODEL SUMMARY")
    print("="*80)
    print(f"\n{'Market':<30} {'Thresh':<8} {'N Bets':<10} {'Hit%':<10} {'ROI':<10}")
    print("-"*70)

    for market, res in results.items():
        val = res['validation']
        best_thresh = 0.60
        best_roi = -1
        for thresh in [0.55, 0.60, 0.65, 0.70]:
            high = val[val['p_under'] >= thresh]
            if len(high) >= 10:
                hit_rate = high['under_hit'].mean()
                roi = (hit_rate * 0.909) - (1 - hit_rate)
                if roi > best_roi:
                    best_roi = roi
                    best_thresh = thresh

        high = val[val['p_under'] >= best_thresh]
        hit_rate = high['under_hit'].mean() if len(high) > 0 else 0
        print(f"{market:<30} {best_thresh:.0%}      {len(high):<10} {hit_rate:.1%}      {best_roi:+.1%}")

    # Save model
    model_data = {
        'models': {m: r['model'] for m, r in results.items()},
        'metrics': {m: {
            'n_bets': len(r['validation']),
            'elapsed': r['elapsed']
        } for m, r in results.items()},
        'version': MODEL_VERSION_FULL,
        'trained_date': datetime.now().isoformat(),
        'feature_source': 'batch_extractor.py (vectorized)',
        'description': f'{MODEL_VERSION_FULL} model trained with {FEATURE_COUNT} features',
        'validated_markets': list(results.keys()),
        'feature_cols': FEATURES,
        'feature_count': FEATURE_COUNT,
        'feature_flags': {
            'use_smooth_sweet_spot': FEATURE_FLAGS.use_smooth_sweet_spot,
            'use_lvt_x_defense': FEATURE_FLAGS.use_lvt_x_defense,
            'use_lvt_x_rest': FEATURE_FLAGS.use_lvt_x_rest,
        },
    }

    # Save to versioned path and active model
    versioned_path = get_versioned_model_path()
    active_path = get_active_model_path()

    joblib.dump(model_data, versioned_path)
    logger.info(f"\nSaved to: {versioned_path}")

    joblib.dump(model_data, active_path)
    logger.info(f"Also saved to: {active_path}")

    # Register in version-free registry
    model_id = register_model(
        model_data=model_data,
        features=FEATURES,
        markets=list(results.keys()),
        metrics=model_data['metrics'],
        set_active=True,
        notes=f"Trained {MODEL_VERSION_FULL}"
    )
    logger.info(f"Registered in registry: {model_id}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"TOTAL TRAINING TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
