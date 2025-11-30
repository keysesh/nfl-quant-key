#!/usr/bin/env python3
"""
V14 Defense-Aware Classifier for Rush Yards (REFACTORED)

This version uses the centralized FeatureEngine for all feature calculations.
No more inline feature calculations - single source of truth.

CHANGES FROM ORIGINAL:
- Uses FeatureEngine.calculate_defense_epa() instead of inline calculation
- Uses FeatureEngine.calculate_trailing_stat() instead of inline calculation
- Uses FeatureEngine.calculate_line_vs_trailing() instead of inline calculation
- Removes duplicate normalize_name() (uses nfl_quant.utils.player_names)
- Removes duplicate load_pbp_all_seasons() (uses FeatureEngine cache)

Walk-Forward Results (unchanged):
- P(UNDER) > 55%: +20.6% ROI (96W-56L)
- P(UNDER) > 60%: +60.8% ROI (16W-3L)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# CENTRALIZED IMPORTS - Single source of truth
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def build_training_data():
    """
    Build training data using centralized FeatureEngine.

    NO INLINE FEATURE CALCULATIONS - all features come from FeatureEngine.
    """
    logger.info("Building training data for V14 (using FeatureEngine)...")

    # Get the centralized feature engine
    engine = get_feature_engine()

    # Load weekly stats
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    stats = pd.read_parquet(stats_path)

    # Filter to RBs with rushing data
    rb_stats = stats[(stats['position'] == 'RB') & (stats['carries'] > 0)].copy()
    logger.info(f"RB records: {len(rb_stats)}")

    # Load PBP for all seasons (using engine's loader with caching)
    pbp_dfs = []
    for year in [2023, 2024, 2025]:
        try:
            pbp = engine._load_pbp(year)
            pbp['season'] = year
            pbp_dfs.append(pbp)
            logger.info(f"Loaded {len(pbp):,} plays from {year}")
        except FileNotFoundError:
            logger.warning(f"PBP file not found for {year}")

    pbp_all = pd.concat(pbp_dfs, ignore_index=True)

    # CENTRALIZED: Calculate defense EPA using FeatureEngine
    # This is the SINGLE SOURCE OF TRUTH for defense EPA calculation
    rush_def = engine.calculate_defense_epa(
        pbp=pbp_all,
        play_type='run',
        trailing_weeks=4,
        no_leakage=True  # CRITICAL: prevents data leakage
    )
    logger.info(f"Defense EPA calculated: {len(rush_def)} team-weeks")

    # Merge opponent defense to RB stats
    rb_stats = rb_stats.merge(
        rush_def[['defteam', 'week', 'season', 'trailing_def_epa']],
        left_on=['opponent_team', 'week', 'season'],
        right_on=['defteam', 'week', 'season'],
        how='left'
    )

    # CENTRALIZED: Calculate trailing stats using FeatureEngine
    # This ensures consistent EWMA calculation with shift(1)
    rb_stats = rb_stats.sort_values(['player_id', 'season', 'week'])
    rb_stats['trailing_rush_yds'] = calculate_trailing_stat(
        df=rb_stats,
        stat_col='rushing_yards',
        player_col='player_id',
        span=4,
        min_periods=1,
        no_leakage=True  # CRITICAL: shift(1) applied
    )
    logger.info("Trailing stats calculated using FeatureEngine")

    # Load backtest data with lines
    backtest_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    backtest = pd.read_csv(backtest_path)
    rush_bt = backtest[backtest['market'] == 'player_rush_yds'].copy()

    # Normalize names using centralized function
    rb_stats['player_norm'] = rb_stats['player_display_name'].apply(normalize_player_name)
    rush_bt['player_norm'] = rush_bt['player'].apply(normalize_player_name)

    # Merge to get lines
    merged = rb_stats.merge(
        rush_bt[['player_norm', 'season', 'week', 'line', 'under_hit']],
        on=['player_norm', 'season', 'week'],
        how='inner'
    )

    # Drop rows without trailing data
    merged = merged[merged['trailing_rush_yds'].notna() & merged['trailing_def_epa'].notna()]

    # CENTRALIZED: Calculate LVT using FeatureEngine
    merged['line_vs_trailing'] = merged.apply(
        lambda row: engine.calculate_line_vs_trailing(
            line=row['line'],
            trailing_stat=row['trailing_rush_yds'],
            method='ratio'
        ),
        axis=1
    )

    # Create global week for temporal ordering
    merged['global_week'] = (merged['season'] - 2023) * 17 + merged['week']
    merged = merged.sort_values('global_week')

    logger.info(f"Training data: {len(merged)} records")
    return merged


def walk_forward_validation(data):
    """Run walk-forward validation."""
    logger.info("Running walk-forward validation...")

    results = []
    features = ['line_vs_trailing', 'trailing_def_epa']

    for test_gw in range(10, data['global_week'].max() + 1):
        train = data[data['global_week'] < test_gw]
        test = data[data['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 100:
            continue

        # Train
        X_train = train[features].fillna(0)
        y_train = train['under_hit']

        X_test = test[features].fillna(0)
        y_test = test['under_hit']

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict
        test = test.copy()
        test['p_under'] = model.predict_proba(X_test)[:, 1]

        for _, row in test.iterrows():
            results.append({
                'global_week': row['global_week'],
                'season': row['season'],
                'week': row['week'],
                'player': row['player_norm'],
                'line': row['line'],
                'actual': row['rushing_yards'],
                'under_hit': row['under_hit'],
                'p_under': row['p_under'],
            })

    return pd.DataFrame(results)


def calculate_roi(results_df, threshold):
    """Calculate ROI at given probability threshold."""
    mask = results_df['p_under'] > threshold
    if mask.sum() == 0:
        return None, 0, 0

    hits = results_df.loc[mask, 'under_hit'].sum()
    total = mask.sum()

    profit = hits * 0.909 - (total - hits) * 1.0
    roi = profit / total * 100

    return roi, hits, total


def train_and_save_model(data):
    """Train final model and save."""
    logger.info("Training final V14 model...")

    features = ['line_vs_trailing', 'trailing_def_epa']
    X = data[features].fillna(0)
    y = data['under_hit']

    model = LogisticRegression()
    model.fit(X, y)

    # Create model bundle with metadata
    bundle = {
        'model': model,
        'features': features,
        'version': 'v14_refactored',
        'market': 'player_rush_yds',
        'trained_date': datetime.now().isoformat(),
        'feature_source': 'nfl_quant.features.core.FeatureEngine',
        'coefficients': {
            'line_vs_trailing': model.coef_[0][0],
            'trailing_def_epa': model.coef_[0][1],
            'intercept': model.intercept_[0]
        },
        'thresholds': {
            'conservative': 0.60,
            'balanced': 0.55,
        },
        'interpretation': {
            'line_vs_trailing': 'Higher LVT (line > trailing) → MORE likely UNDER',
            'trailing_def_epa': 'Negative EPA (good defense) → MORE likely UNDER',
        },
        'data_leakage_prevention': {
            'trailing_stat': 'shift(1) applied via FeatureEngine',
            'defense_epa': 'shift(1) applied via FeatureEngine',
        }
    }

    # Save
    model_path = PROJECT_ROOT / 'data' / 'models' / 'v14_defense_aware_classifier.joblib'
    joblib.dump(bundle, model_path)
    logger.info(f"Saved model to {model_path}")

    return bundle


def main():
    """Train V14 Defense-Aware Classifier using centralized FeatureEngine."""
    print("="*80)
    print("V14 DEFENSE-AWARE CLASSIFIER - RUSH YARDS")
    print("Using centralized FeatureEngine (single source of truth)")
    print("="*80)

    # Build training data (uses FeatureEngine internally)
    data = build_training_data()

    # Walk-forward validation
    results = walk_forward_validation(data)

    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*60)

    for threshold in [0.55, 0.60, 0.65]:
        roi, hits, total = calculate_roi(results, threshold)
        if roi is not None:
            hit_rate = hits / total * 100
            print(f"P(UNDER) > {threshold:.0%}: {int(hits)}W-{int(total-hits)}L ({hit_rate:.1f}%), ROI: {roi:+.1f}%")

    # Compare to LVT-only baseline
    print("\nLVT-Only Baseline:")
    for lvt_thresh in [1.3, 1.4, 1.5]:
        lvt_mask = data['line_vs_trailing'] > lvt_thresh
        if lvt_mask.sum() > 0:
            hits = data.loc[lvt_mask, 'under_hit'].sum()
            total = lvt_mask.sum()
            hit_rate = hits / total * 100
            profit = hits * 0.909 - (total - hits) * 1.0
            roi = profit / total * 100
            print(f"LVT > {lvt_thresh}: {int(hits)}W-{int(total-hits)}L ({hit_rate:.1f}%), ROI: {roi:+.1f}%")

    # Train and save final model
    bundle = train_and_save_model(data)

    print("\n" + "="*60)
    print("MODEL COEFFICIENTS")
    print("="*60)
    for feat, coef in bundle['coefficients'].items():
        print(f"  {feat}: {coef:.4f}")

    print("\n" + "="*60)
    print("FEATURE SOURCE (Single Source of Truth)")
    print("="*60)
    print(f"  All features from: {bundle['feature_source']}")
    print(f"  Data leakage prevention: ✅ shift(1) applied")

    print("\n✅ V14 Defense-Aware Classifier trained and saved!")


if __name__ == '__main__':
    main()
