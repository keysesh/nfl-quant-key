#!/usr/bin/env python3
"""
V18 Fast Training Script

Uses vectorized batch feature extraction for 10-100x speedup over iterrows().

Target: Complete training in <5 minutes (vs 30+ minutes with old approach)

Usage:
    python scripts/train/train_v18_fast.py
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
import time
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports
from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.features.batch_extractor import extract_features_batch, clear_caches
from nfl_quant.features.feature_defaults import safe_fillna, FEATURE_DEFAULTS
from nfl_quant.utils.player_names import normalize_player_name
from configs.model_config import (
    MODEL_VERSION,
    MODEL_VERSION_FULL,
    V18_FEATURE_COLS,
    CURRENT_FEATURE_COLS,
    FEATURE_FLAGS,
    get_model_path,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# V18 Feature columns
FEATURE_COLS = V18_FEATURE_COLS

# Markets to train
MARKETS = ['player_receptions', 'player_reception_yds', 'player_rush_yds', 'player_pass_yds']


def load_data():
    """Load all historical data."""
    logger.info("Loading data...")
    start = time.time()

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
    stat_cols = ['receptions', 'receiving_yards', 'rushing_yards', 'passing_yards']
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

    # Rename for consistency
    if 'opponent_team' in odds_merged.columns:
        odds_merged['opponent'] = odds_merged['opponent_team']

    logger.info(f"  Prepared {len(odds_merged):,} rows in {time.time()-start:.1f}s")

    return odds_merged


def train_market_fast(
    odds_merged: pd.DataFrame,
    market: str,
    window_weeks: int = 20
) -> dict:
    """
    Train model for a market using vectorized feature extraction.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training V18 (Fast): {market}")
    logger.info(f"{'='*60}")
    start = time.time()

    # Map market to stat column
    stat_col_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
    }
    stat_col = stat_col_map.get(market)
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

    # =========================================================================
    # Walk-forward validation using VECTORIZED extraction
    # =========================================================================
    logger.info(f"  Walk-forward validation on {min(10, len(weeks)-1)} weeks...")

    val_start = max(weeks) - 10
    val_weeks = [w for w in weeks if w >= val_start]

    all_preds = []

    for test_week in val_weeks:
        # Training data: all weeks before test_week with buffer
        train_data = market_data[market_data['global_week'] < test_week - 1].copy()
        test_data = market_data[market_data['global_week'] == test_week].copy()

        if len(train_data) < 50 or len(test_data) == 0:
            continue

        # VECTORIZED feature extraction for training set
        train_features = extract_features_batch(
            train_data,
            market_data[market_data['global_week'] < test_week - 1],
            market
        )

        # VECTORIZED feature extraction for test set
        test_features = extract_features_batch(
            test_data,
            market_data[market_data['global_week'] < test_week],
            market
        )

        if len(train_features) == 0 or len(test_features) == 0:
            continue

        # Get available features
        available_features = [f for f in FEATURE_COLS if f in train_features.columns and f in test_features.columns]

        if len(available_features) < 5:
            continue

        # Prepare X, y
        X_train = safe_fillna(train_features[available_features])
        y_train = train_features['under_hit']
        X_test = safe_fillna(test_features[available_features])
        y_test = test_features['under_hit']

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
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
            roi = (hit_rate * 0.909) - (1 - hit_rate)  # -110 odds
            logger.info(f"    Threshold {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

    # =========================================================================
    # Train final production model on ALL data
    # =========================================================================
    logger.info(f"\n  Training final production model...")

    final_features = extract_features_batch(
        market_data,
        market_data,
        market
    )

    available_features = [f for f in FEATURE_COLS if f in final_features.columns]
    X_final = safe_fillna(final_features[available_features])
    y_final = final_features['under_hit']

    final_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    final_model.fit(X_final, y_final)

    # Feature importance
    logger.info(f"\n  Feature Importance (Top 6):")
    importances = dict(zip(available_features, final_model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:6]:
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
    print("V18 FAST TRAINING (Vectorized Feature Extraction)")
    print("="*80)
    print(f"\nMODEL VERSION: V{MODEL_VERSION}")
    print(f"FEATURES: {len(FEATURE_COLS)} columns")
    print(f"FEATURE FLAGS:")
    print(f"  - Gaussian Sweet Spot: {FEATURE_FLAGS.use_smooth_sweet_spot}")
    print(f"  - LVT x Defense: {FEATURE_FLAGS.use_lvt_x_defense}")
    print(f"  - LVT x Rest: {FEATURE_FLAGS.use_lvt_x_rest}")

    # Load data
    odds, stats = load_data()

    # Prepare with trailing stats
    odds_merged = prepare_data_with_trailing(odds, stats)

    # Train each market
    results = {}
    for market in MARKETS:
        result = train_market_fast(odds_merged, market)
        if result:
            results[market] = result

    # Clear caches
    clear_caches()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("V18 MODEL SUMMARY")
    print("="*80)
    print(f"\n{'Market':<30} {'Thresh':<8} {'N Bets':<10} {'Hit%':<10} {'ROI':<10}")
    print("-"*70)

    for market, res in results.items():
        val = res['validation']
        # Find best threshold
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

    # =========================================================================
    # Save model
    # =========================================================================
    model_data = {
        'models': {m: r['model'] for m, r in results.items()},
        'metrics': {m: {
            'n_bets': len(r['validation']),
            'elapsed': r['elapsed']
        } for m, r in results.items()},
        'version': f'V{MODEL_VERSION}',
        'trained_date': datetime.now().isoformat(),
        'feature_source': 'batch_extractor.py (vectorized)',
        'description': f'V{MODEL_VERSION} model with vectorized training',
        'validated_markets': list(results.keys()),
        'feature_cols': FEATURE_COLS,
        'feature_flags': {
            'use_smooth_sweet_spot': FEATURE_FLAGS.use_smooth_sweet_spot,
            'use_lvt_x_defense': FEATURE_FLAGS.use_lvt_x_defense,
            'use_lvt_x_rest': FEATURE_FLAGS.use_lvt_x_rest,
        }
    }

    # Save paths
    v18_path = PROJECT_ROOT / 'data' / 'models' / f'v{MODEL_VERSION.lower()}_interaction_classifier.joblib'
    joblib.dump(model_data, v18_path)
    logger.info(f"\nSaved to: {v18_path}")

    # Also save as active model
    active_path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'
    joblib.dump(model_data, active_path)
    logger.info(f"Also saved to: {active_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"TOTAL TRAINING TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
