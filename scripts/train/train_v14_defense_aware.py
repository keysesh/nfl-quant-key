#!/usr/bin/env python3
"""
V14 Defense-Aware Classifier for Rush Yards

Walk-forward validated approach that uses:
1. LVT (Line vs Trailing) - primary signal
2. Opponent Rush Defense EPA - secondary signal

Key Finding (2025-11-29):
- Adding defensive EPA to LVT creates significant edge for rush_yds
- Negative def EPA (good defense) → MORE unders hit
- Vegas may overvalue "smash spots" vs bad defenses

Walk-Forward Results:
- P(UNDER) > 55%: +20.6% ROI (96W-56L)
- P(UNDER) > 60%: +60.8% ROI (16W-3L)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace('.', '').replace("'", "")


def load_pbp_all_seasons():
    """Load play-by-play data for all available seasons."""
    pbp_dfs = []
    for year in [2023, 2024, 2025]:
        path = PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{year}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['season'] = year
            pbp_dfs.append(df)
            logger.info(f"Loaded {len(df):,} plays from {year}")
    return pd.concat(pbp_dfs, ignore_index=True)


def calculate_rush_defense_epa(pbp):
    """Calculate trailing rush defense EPA per team per week."""
    # Rush defense EPA per game
    rush_def = pbp[pbp['play_type'] == 'run'].groupby(['defteam', 'week', 'season']).agg(
        rush_def_epa=('epa', 'mean')
    ).reset_index()

    # Calculate trailing (no leakage - shift by 1)
    rush_def = rush_def.sort_values(['defteam', 'season', 'week'])
    rush_def['trailing_def_epa'] = rush_def.groupby('defteam')['rush_def_epa'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    return rush_def


def build_training_data():
    """Build training data with all features."""
    logger.info("Building training data for V14...")

    # Load weekly stats
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    stats = pd.read_parquet(stats_path)

    # Filter to RBs with rushing data
    rb_stats = stats[(stats['position'] == 'RB') & (stats['carries'] > 0)].copy()
    logger.info(f"RB records: {len(rb_stats)}")

    # Load PBP and calculate defense EPA
    pbp = load_pbp_all_seasons()
    rush_def = calculate_rush_defense_epa(pbp)

    # Merge opponent defense to RB stats
    rb_stats = rb_stats.merge(
        rush_def[['defteam', 'week', 'season', 'trailing_def_epa']],
        left_on=['opponent_team', 'week', 'season'],
        right_on=['defteam', 'week', 'season'],
        how='left'
    )

    # Calculate trailing stats (strict no-leakage)
    rb_stats = rb_stats.sort_values(['player_id', 'season', 'week'])
    rb_stats['trailing_rush_yds'] = rb_stats.groupby('player_id')['rushing_yards'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    # Load backtest data with lines
    backtest_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    backtest = pd.read_csv(backtest_path)
    rush_bt = backtest[backtest['market'] == 'player_rush_yds'].copy()

    # Normalize names for matching
    rb_stats['player_norm'] = rb_stats['player_display_name'].apply(normalize_name)
    rush_bt['player_norm'] = rush_bt['player'].apply(normalize_name)

    # Merge to get lines
    merged = rb_stats.merge(
        rush_bt[['player_norm', 'season', 'week', 'line', 'under_hit']],
        on=['player_norm', 'season', 'week'],
        how='inner'
    )

    # Drop rows without trailing data
    merged = merged[merged['trailing_rush_yds'].notna() & merged['trailing_def_epa'].notna()]

    # Calculate LVT
    merged['line_vs_trailing'] = merged['line'] / merged['trailing_rush_yds'].replace(0, np.nan)

    # Create global week
    merged['global_week'] = (merged['season'] - 2023) * 17 + merged['week']
    merged = merged.sort_values('global_week')

    logger.info(f"Training data: {len(merged)} records")
    return merged


def walk_forward_validation(data):
    """Run walk-forward validation."""
    logger.info("Running walk-forward validation...")

    results = []

    for test_gw in range(10, data['global_week'].max() + 1):
        train = data[data['global_week'] < test_gw]
        test = data[data['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 100:
            continue

        # Train
        features = ['line_vs_trailing', 'trailing_def_epa']
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
    hit_rate = hits / total

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

    # Create model bundle
    bundle = {
        'model': model,
        'features': features,
        'version': 'v14',
        'market': 'player_rush_yds',
        'coefficients': {
            'line_vs_trailing': model.coef_[0][0],
            'trailing_def_epa': model.coef_[0][1],
            'intercept': model.intercept_[0]
        },
        'thresholds': {
            'conservative': 0.60,  # Higher confidence, fewer bets
            'balanced': 0.55,      # Good balance
        },
        'interpretation': {
            'line_vs_trailing': 'Higher LVT (line > trailing) → MORE likely UNDER',
            'trailing_def_epa': 'Negative EPA (good defense) → MORE likely UNDER',
        }
    }

    # Save
    model_path = PROJECT_ROOT / 'data' / 'models' / 'v14_defense_aware_classifier.joblib'
    joblib.dump(bundle, model_path)
    logger.info(f"Saved model to {model_path}")

    return bundle


def main():
    """Train V14 Defense-Aware Classifier."""
    print("="*80)
    print("V14 DEFENSE-AWARE CLASSIFIER - RUSH YARDS")
    print("="*80)

    # Build training data
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
        mask = results['p_under'].notna()  # We need the merged data
        # Use the original data for LVT comparison
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
    print("RECOMMENDED THRESHOLDS")
    print("="*60)
    print(f"  Conservative (P > 60%): Fewer bets, higher confidence")
    print(f"  Balanced (P > 55%): More bets, still profitable")

    print("\n✅ V14 Defense-Aware Classifier trained and saved!")


if __name__ == '__main__':
    main()
