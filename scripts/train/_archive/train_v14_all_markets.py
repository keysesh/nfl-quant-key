#!/usr/bin/env python3
"""
V14 Defense-Aware Classifier - ALL MARKETS

Trains V14 models for all supported prop markets:
- player_rush_yds (RB) - uses rush defense EPA
- player_pass_yds (QB) - uses pass defense EPA
- player_receptions (WR/TE/RB) - uses pass defense EPA
- player_reception_yds (WR/TE/RB) - uses pass defense EPA
- player_pass_completions (QB) - uses pass defense EPA
- player_pass_attempts (QB) - uses pass defense EPA
- player_rush_attempts (RB) - uses rush defense EPA

Features: LVT (line vs trailing) + opponent defense EPA
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Market configurations
MARKET_CONFIGS = {
    'player_rush_yds': {
        'positions': ['RB'],
        'stat_col': 'rushing_yards',
        'defense_type': 'run',
        'min_stat': 'carries',
        'min_value': 0,
    },
    'player_pass_yds': {
        'positions': ['QB'],
        'stat_col': 'passing_yards',
        'defense_type': 'pass',
        'min_stat': 'attempts',
        'min_value': 5,
    },
    'player_receptions': {
        'positions': ['WR', 'TE', 'RB'],
        'stat_col': 'receptions',
        'defense_type': 'pass',
        'min_stat': 'targets',
        'min_value': 0,
    },
    'player_reception_yds': {
        'positions': ['WR', 'TE', 'RB'],
        'stat_col': 'receiving_yards',
        'defense_type': 'pass',
        'min_stat': 'targets',
        'min_value': 0,
    },
    'player_pass_completions': {
        'positions': ['QB'],
        'stat_col': 'completions',
        'defense_type': 'pass',
        'min_stat': 'attempts',
        'min_value': 5,
    },
    'player_pass_attempts': {
        'positions': ['QB'],
        'stat_col': 'attempts',
        'defense_type': 'pass',
        'min_stat': 'attempts',
        'min_value': 5,
    },
    'player_rush_attempts': {
        'positions': ['RB'],
        'stat_col': 'carries',
        'defense_type': 'run',
        'min_stat': 'carries',
        'min_value': 0,
    },
}


def build_training_data_for_market(market: str, config: dict):
    """Build training data for a specific market."""
    logger.info(f"Building training data for {market}...")

    engine = get_feature_engine()

    # Load weekly stats
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    stats = pd.read_parquet(stats_path)

    # Filter by position and minimum activity
    positions = config['positions']
    min_stat = config['min_stat']
    min_value = config['min_value']

    filtered = stats[
        (stats['position'].isin(positions)) &
        (stats[min_stat] > min_value)
    ].copy()
    logger.info(f"  Filtered to {len(filtered)} records for positions {positions}")

    # Load PBP for defense EPA
    pbp_dfs = []
    for year in [2023, 2024, 2025]:
        try:
            pbp = engine._load_pbp(year)
            pbp['season'] = year
            pbp_dfs.append(pbp)
        except FileNotFoundError:
            pass

    if not pbp_dfs:
        logger.error("No PBP data found")
        return None

    pbp_all = pd.concat(pbp_dfs, ignore_index=True)

    # Calculate defense EPA (rush or pass based on market)
    defense_type = config['defense_type']
    def_epa = engine.calculate_defense_epa(
        pbp=pbp_all,
        play_type=defense_type,
        trailing_weeks=4,
        no_leakage=True
    )
    logger.info(f"  Defense EPA ({defense_type}): {len(def_epa)} team-weeks")

    # Merge opponent defense
    filtered = filtered.merge(
        def_epa[['defteam', 'week', 'season', 'trailing_def_epa']],
        left_on=['opponent_team', 'week', 'season'],
        right_on=['defteam', 'week', 'season'],
        how='left'
    )

    # Calculate trailing stat
    stat_col = config['stat_col']
    filtered = filtered.sort_values(['player_id', 'season', 'week'])
    filtered['trailing_stat'] = calculate_trailing_stat(
        df=filtered,
        stat_col=stat_col,
        player_col='player_id',
        span=4,
        min_periods=1,
        no_leakage=True
    )

    # Load backtest data
    backtest_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    if not backtest_path.exists():
        logger.error(f"Backtest file not found: {backtest_path}")
        return None

    backtest = pd.read_csv(backtest_path)
    market_bt = backtest[backtest['market'] == market].copy()

    if len(market_bt) == 0:
        logger.warning(f"  No backtest data for {market}")
        return None

    # Normalize names
    filtered['player_norm'] = filtered['player_display_name'].apply(normalize_player_name)
    market_bt['player_norm'] = market_bt['player'].apply(normalize_player_name)

    # Merge
    merged = filtered.merge(
        market_bt[['player_norm', 'season', 'week', 'line', 'under_hit']],
        on=['player_norm', 'season', 'week'],
        how='inner'
    )

    # Drop missing
    merged = merged[merged['trailing_stat'].notna() & merged['trailing_def_epa'].notna()]

    # Calculate LVT
    merged['line_vs_trailing'] = (merged['line'] - merged['trailing_stat']) / (merged['trailing_stat'] + 0.1)

    # Global week for ordering
    merged['global_week'] = (merged['season'] - 2023) * 17 + merged['week']
    merged = merged.sort_values('global_week')

    logger.info(f"  Training data: {len(merged)} records")
    return merged


def walk_forward_validation(data, market):
    """Run walk-forward validation for a market."""
    if data is None or len(data) < 100:
        return None

    results = []
    features = ['line_vs_trailing', 'trailing_def_epa']

    for test_gw in range(10, data['global_week'].max() + 1):
        train = data[data['global_week'] < test_gw]
        test = data[data['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 50:
            continue

        X_train = train[features].fillna(0)
        y_train = train['under_hit']
        X_test = test[features].fillna(0)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        test = test.copy()
        test['p_under'] = model.predict_proba(X_test)[:, 1]

        for _, row in test.iterrows():
            results.append({
                'global_week': row['global_week'],
                'p_under': row['p_under'],
                'under_hit': row['under_hit'],
            })

    return pd.DataFrame(results) if results else None


def calculate_roi(results_df, threshold):
    """Calculate ROI at threshold."""
    if results_df is None:
        return None, 0, 0

    mask = results_df['p_under'] > threshold
    if mask.sum() == 0:
        return None, 0, 0

    hits = results_df.loc[mask, 'under_hit'].sum()
    total = mask.sum()
    profit = hits * 0.909 - (total - hits) * 1.0
    roi = profit / total * 100

    return roi, int(hits), int(total)


def train_model_for_market(data, market, config):
    """Train and save model for a market."""
    if data is None or len(data) < 50:
        return None

    features = ['line_vs_trailing', 'trailing_def_epa']
    X = data[features].fillna(0)
    y = data['under_hit']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return {
        'model': model,
        'features': features,
        'market': market,
        'defense_type': config['defense_type'],
        'coefficients': {
            'line_vs_trailing': float(model.coef_[0][0]),
            'trailing_def_epa': float(model.coef_[0][1]),
            'intercept': float(model.intercept_[0])
        },
        'n_samples': len(data),
    }


def main():
    """Train V14 models for all markets."""
    print("=" * 80)
    print("V14 DEFENSE-AWARE CLASSIFIER - ALL MARKETS")
    print("=" * 80)

    all_models = {}
    all_thresholds = {}
    all_coefficients = {}
    validation_results = {}

    for market, config in MARKET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"MARKET: {market}")
        print(f"{'='*60}")

        # Build training data
        data = build_training_data_for_market(market, config)

        if data is None or len(data) < 100:
            logger.warning(f"Insufficient data for {market}, skipping")
            all_thresholds[market] = {'excluded': True, 'reason': 'insufficient_data'}
            continue

        # Walk-forward validation
        wf_results = walk_forward_validation(data, market)

        if wf_results is None:
            logger.warning(f"Walk-forward failed for {market}")
            all_thresholds[market] = {'excluded': True, 'reason': 'validation_failed'}
            continue

        # Find optimal threshold
        best_roi = -100
        best_threshold = 0.55
        best_record = (0, 0)

        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            roi, wins, total = calculate_roi(wf_results, thresh)
            if roi is not None and total >= 20:
                print(f"  Threshold {thresh:.0%}: ROI={roi:+.1f}% ({wins}W-{total-wins}L)")
                if roi > best_roi:
                    best_roi = roi
                    best_threshold = thresh
                    best_record = (wins, total)

        # Only include if positive ROI with enough samples
        if best_roi > 0 and best_record[1] >= 30:
            print(f"  ✓ VALIDATED: {best_threshold:.0%} threshold, {best_roi:+.1f}% ROI")

            # Train final model
            model_bundle = train_model_for_market(data, market, config)
            if model_bundle:
                all_models[market] = model_bundle['model']
                all_coefficients[market] = model_bundle['coefficients']
                all_thresholds[market] = {
                    'threshold': best_threshold,
                    'roi': best_roi,
                    'wins': best_record[0],
                    'total': best_record[1],
                    'defense_type': config['defense_type'],
                }
                validation_results[market] = {
                    'roi': best_roi,
                    'threshold': best_threshold,
                    'record': f"{best_record[0]}W-{best_record[1]-best_record[0]}L"
                }
        else:
            print(f"  ✗ EXCLUDED: Best ROI={best_roi:+.1f}% insufficient")
            all_thresholds[market] = {
                'excluded': True,
                'reason': f'negative_roi_{best_roi:.1f}',
                'best_threshold': best_threshold,
            }

    # Save combined model bundle
    bundle = {
        'models': all_models,
        'coefficients': all_coefficients,
        'thresholds': all_thresholds,
        'version': 'v14_all_markets',
        'trained_date': datetime.now().isoformat(),
        'validation_results': validation_results,
        'features': ['line_vs_trailing', 'trailing_def_epa'],
    }

    model_path = PROJECT_ROOT / 'data' / 'models' / 'v14_all_markets.joblib'
    joblib.dump(bundle, model_path)
    print(f"\n✓ Saved to {model_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    validated = [m for m, t in all_thresholds.items() if not t.get('excluded', False)]
    excluded = [m for m, t in all_thresholds.items() if t.get('excluded', False)]

    print(f"\nValidated markets ({len(validated)}):")
    for m in validated:
        t = all_thresholds[m]
        print(f"  ✓ {m}: {t['threshold']:.0%} threshold, {t['roi']:+.1f}% ROI")

    print(f"\nExcluded markets ({len(excluded)}):")
    for m in excluded:
        print(f"  ✗ {m}: {all_thresholds[m].get('reason', 'unknown')}")

    return bundle


if __name__ == "__main__":
    main()
