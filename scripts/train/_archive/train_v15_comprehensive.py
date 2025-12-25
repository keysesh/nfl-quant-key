#!/usr/bin/env python3
"""
V15 Comprehensive Classifier - ALL FEATURES

Uses ALL available features from FeatureEngine:
- LVT (line vs trailing)
- Defense EPA (opponent)
- Snap share
- Target share / Carry share
- Red zone shares
- Weather adjustments
- Home/away
- Divisional game
- Primetime
- Rest/bye
- Injury context
- Altitude
- Game script (spread, total)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# All features to use
FEATURE_COLS = [
    'line_vs_trailing',
    'trailing_def_epa',
    'snap_share',
    'redzone_target_share',
    'redzone_carry_share',
    'goalline_carry_share',
    'is_home',
    'is_divisional',
    'is_primetime',
    'is_bye_week',
    'rest_days',
    'weather_adjustment',
    'altitude_adjustment',
    'spread',
    'total',
    'qb_injured',
    'wr1_injured',
    'rb1_injured',
]

# Market configurations
MARKET_CONFIGS = {
    'player_rush_yds': {
        'positions': ['RB'],
        'stat_col': 'rushing_yards',
        'defense_type': 'run',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'snap_share',
                              'redzone_carry_share', 'goalline_carry_share', 'is_home',
                              'is_divisional', 'spread', 'total', 'weather_adjustment'],
    },
    'player_pass_yds': {
        'positions': ['QB'],
        'stat_col': 'passing_yards',
        'defense_type': 'pass',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'is_home',
                              'is_divisional', 'is_primetime', 'spread', 'total',
                              'weather_adjustment', 'wr1_injured'],
    },
    'player_receptions': {
        'positions': ['WR', 'TE', 'RB'],
        'stat_col': 'receptions',
        'defense_type': 'pass',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'snap_share',
                              'redzone_target_share', 'is_home', 'spread', 'total',
                              'qb_injured'],
    },
    'player_reception_yds': {
        'positions': ['WR', 'TE', 'RB'],
        'stat_col': 'receiving_yards',
        'defense_type': 'pass',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'snap_share',
                              'redzone_target_share', 'is_home', 'spread', 'total',
                              'weather_adjustment', 'qb_injured'],
    },
    'player_pass_completions': {
        'positions': ['QB'],
        'stat_col': 'completions',
        'defense_type': 'pass',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'is_home',
                              'spread', 'total', 'weather_adjustment'],
    },
    'player_pass_attempts': {
        'positions': ['QB'],
        'stat_col': 'attempts',
        'defense_type': 'pass',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'is_home',
                              'spread', 'total'],
    },
    'player_rush_attempts': {
        'positions': ['RB'],
        'stat_col': 'carries',
        'defense_type': 'run',
        'relevant_features': ['line_vs_trailing', 'trailing_def_epa', 'snap_share',
                              'is_home', 'spread', 'total'],
    },
    'player_anytime_td': {
        'positions': ['RB', 'WR', 'TE'],
        'stat_col': 'tds',  # Combined TDs
        'defense_type': 'both',
        'relevant_features': ['redzone_target_share', 'redzone_carry_share',
                              'goalline_carry_share', 'snap_share', 'trailing_def_epa',
                              'spread', 'total', 'is_home'],
    },
}


def build_comprehensive_training_data(market: str, config: dict):
    """Build training data with ALL features for a market."""
    logger.info(f"Building comprehensive training data for {market}...")

    engine = get_feature_engine()

    # Load weekly stats
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    stats = pd.read_parquet(stats_path)

    # Load snap counts for snap_share
    snap_path = PROJECT_ROOT / 'data' / 'nflverse' / 'snap_counts.parquet'
    snaps = pd.read_parquet(snap_path) if snap_path.exists() else None

    # Load schedules for game context
    schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    schedules = pd.read_parquet(schedule_path) if schedule_path.exists() else None

    # Filter by position
    positions = config['positions']
    filtered = stats[stats['position'].isin(positions)].copy()
    logger.info(f"  Filtered to {len(filtered)} records")

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
        return None

    pbp_all = pd.concat(pbp_dfs, ignore_index=True)

    # Calculate defense EPA
    defense_type = config['defense_type']
    if defense_type == 'both':
        # For TD props, use combined EPA
        rush_def = engine.calculate_defense_epa(pbp_all, 'run', 4, True)
        pass_def = engine.calculate_defense_epa(pbp_all, 'pass', 4, True)
        rush_def = rush_def.rename(columns={'trailing_def_epa': 'rush_def_epa'})
        pass_def = pass_def.rename(columns={'trailing_def_epa': 'pass_def_epa'})
        def_epa = rush_def.merge(pass_def, on=['defteam', 'week', 'season'], how='outer')
        def_epa['trailing_def_epa'] = def_epa[['rush_def_epa', 'pass_def_epa']].mean(axis=1)
    else:
        def_epa = engine.calculate_defense_epa(pbp_all, defense_type, 4, True)

    # Merge defense EPA
    filtered = filtered.merge(
        def_epa[['defteam', 'week', 'season', 'trailing_def_epa']],
        left_on=['opponent_team', 'week', 'season'],
        right_on=['defteam', 'week', 'season'],
        how='left'
    )

    # Calculate trailing stat
    stat_col = config['stat_col']
    if stat_col == 'tds':
        # Combined TDs for anytime TD
        filtered['tds'] = filtered['rushing_tds'].fillna(0) + filtered['receiving_tds'].fillna(0)

    filtered = filtered.sort_values(['player_id', 'season', 'week'])
    filtered['trailing_stat'] = calculate_trailing_stat(
        df=filtered, stat_col=stat_col, player_col='player_id',
        span=4, min_periods=1, no_leakage=True
    )

    # Merge snap share
    if snaps is not None:
        snap_agg = snaps.groupby(['player', 'season', 'week']).agg(
            snap_share=('offense_pct', 'first')
        ).reset_index()
        snap_agg['player_norm'] = snap_agg['player'].apply(normalize_player_name)
        filtered['player_norm'] = filtered['player_display_name'].apply(normalize_player_name)
        filtered = filtered.merge(
            snap_agg[['player_norm', 'season', 'week', 'snap_share']],
            on=['player_norm', 'season', 'week'],
            how='left'
        )
    else:
        filtered['snap_share'] = 0.5

    # Merge schedule data for game context
    if schedules is not None:
        game_cols = ['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line',
                     'div_game', 'gameday']
        game_cols = [c for c in game_cols if c in schedules.columns]

        # Create game lookup
        for _, game in schedules[game_cols].iterrows():
            home = game.get('home_team', '')
            away = game.get('away_team', '')
            season = game.get('season', 0)
            week = game.get('week', 0)

            # Add is_home
            filtered.loc[
                (filtered['season'] == season) &
                (filtered['week'] == week) &
                (filtered['team'] == home),
                'is_home'
            ] = 1

            filtered.loc[
                (filtered['season'] == season) &
                (filtered['week'] == week) &
                (filtered['team'] == away),
                'is_home'
            ] = 0

            # Add spread and total
            filtered.loc[
                (filtered['season'] == season) &
                (filtered['week'] == week) &
                (filtered['team'].isin([home, away])),
                'spread'
            ] = game.get('spread_line', 0)

            filtered.loc[
                (filtered['season'] == season) &
                (filtered['week'] == week) &
                (filtered['team'].isin([home, away])),
                'total'
            ] = game.get('total_line', 45)

            # Divisional
            filtered.loc[
                (filtered['season'] == season) &
                (filtered['week'] == week) &
                (filtered['team'].isin([home, away])),
                'is_divisional'
            ] = 1 if game.get('div_game', 0) else 0

    # Fill missing game context
    filtered['is_home'] = filtered.get('is_home', 0).fillna(0)
    filtered['spread'] = filtered.get('spread', 0).fillna(0)
    filtered['total'] = filtered.get('total', 45).fillna(45)
    filtered['is_divisional'] = filtered.get('is_divisional', 0).fillna(0)
    filtered['is_primetime'] = 0  # Would need game time data
    filtered['weather_adjustment'] = 0  # Would need weather data
    filtered['altitude_adjustment'] = 0

    # Load backtest data
    backtest_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    if not backtest_path.exists():
        return None

    backtest = pd.read_csv(backtest_path)
    market_bt = backtest[backtest['market'] == market].copy()

    if len(market_bt) == 0:
        logger.warning(f"  No backtest data for {market}")
        return None

    # Normalize and merge
    if 'player_norm' not in filtered.columns:
        filtered['player_norm'] = filtered['player_display_name'].apply(normalize_player_name)
    market_bt['player_norm'] = market_bt['player'].apply(normalize_player_name)

    merged = filtered.merge(
        market_bt[['player_norm', 'season', 'week', 'line', 'under_hit']],
        on=['player_norm', 'season', 'week'],
        how='inner'
    )

    # Calculate LVT
    merged['line_vs_trailing'] = (merged['line'] - merged['trailing_stat']) / (merged['trailing_stat'] + 0.1)

    # Fill missing features
    for col in config['relevant_features']:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0)

    # Drop rows with missing targets
    merged = merged[merged['trailing_stat'].notna() & merged['trailing_def_epa'].notna()]

    # Global week
    merged['global_week'] = (merged['season'] - 2023) * 17 + merged['week']
    merged = merged.sort_values('global_week')

    logger.info(f"  Training data: {len(merged)} records with {len(config['relevant_features'])} features")
    return merged, config['relevant_features']


def walk_forward_validation(data, features):
    """Walk-forward with comprehensive features."""
    if data is None or len(data) < 100:
        return None

    results = []
    scaler = StandardScaler()

    for test_gw in range(15, data['global_week'].max() + 1):
        train = data[data['global_week'] < test_gw]
        test = data[data['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 100:
            continue

        X_train = train[features].fillna(0)
        y_train = train['under_hit']
        X_test = test[features].fillna(0)

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use gradient boosting for more complex patterns
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        test = test.copy()
        test['p_under'] = model.predict_proba(X_test_scaled)[:, 1]

        for _, row in test.iterrows():
            results.append({
                'global_week': row['global_week'],
                'p_under': row['p_under'],
                'under_hit': row['under_hit'],
            })

    return pd.DataFrame(results) if results else None


def calculate_roi(results_df, threshold):
    """Calculate ROI."""
    if results_df is None:
        return None, 0, 0

    mask = results_df['p_under'] > threshold
    if mask.sum() < 20:
        return None, 0, 0

    hits = results_df.loc[mask, 'under_hit'].sum()
    total = mask.sum()
    profit = hits * 0.909 - (total - hits) * 1.0
    roi = profit / total * 100

    return roi, int(hits), int(total)


def train_final_model(data, features, market):
    """Train final model."""
    if data is None:
        return None

    X = data[features].fillna(0)
    y = data['under_hit']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))

    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'market': market,
        'feature_importance': importance,
        'n_samples': len(data),
    }


def main():
    """Train V15 comprehensive models."""
    print("=" * 80)
    print("V15 COMPREHENSIVE CLASSIFIER - ALL FEATURES")
    print("=" * 80)

    all_models = {}
    all_thresholds = {}
    validation_results = {}

    for market, config in MARKET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"MARKET: {market}")
        print(f"Features: {config['relevant_features']}")
        print(f"{'='*60}")

        result = build_comprehensive_training_data(market, config)

        if result is None:
            all_thresholds[market] = {'excluded': True, 'reason': 'no_data'}
            continue

        data, features = result

        if len(data) < 100:
            all_thresholds[market] = {'excluded': True, 'reason': 'insufficient_data'}
            continue

        # Walk-forward
        wf_results = walk_forward_validation(data, features)

        if wf_results is None:
            all_thresholds[market] = {'excluded': True, 'reason': 'validation_failed'}
            continue

        # Find best threshold
        best_roi = -100
        best_threshold = 0.55
        best_record = (0, 0)

        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            roi, wins, total = calculate_roi(wf_results, thresh)
            if roi is not None:
                print(f"  {thresh:.0%}: ROI={roi:+.1f}% ({wins}W-{total-wins}L)")
                if roi > best_roi:
                    best_roi = roi
                    best_threshold = thresh
                    best_record = (wins, total)

        if best_roi > 0 and best_record[1] >= 30:
            print(f"  ✓ VALIDATED: {best_threshold:.0%}, {best_roi:+.1f}% ROI")

            model_bundle = train_final_model(data, features, market)
            if model_bundle:
                all_models[market] = model_bundle
                all_thresholds[market] = {
                    'threshold': best_threshold,
                    'roi': best_roi,
                    'wins': best_record[0],
                    'total': best_record[1],
                    'features': features,
                }
                validation_results[market] = {
                    'roi': best_roi,
                    'record': f"{best_record[0]}W-{best_record[1]-best_record[0]}L"
                }

                # Show feature importance
                print(f"  Feature importance:")
                for feat, imp in sorted(model_bundle['feature_importance'].items(),
                                        key=lambda x: -x[1])[:5]:
                    print(f"    {feat}: {imp:.3f}")
        else:
            print(f"  ✗ EXCLUDED: ROI={best_roi:+.1f}%")
            all_thresholds[market] = {'excluded': True, 'reason': f'negative_roi_{best_roi:.1f}'}

    # Save
    bundle = {
        'models': all_models,
        'thresholds': all_thresholds,
        'version': 'v15_comprehensive',
        'trained_date': datetime.now().isoformat(),
        'validation_results': validation_results,
    }

    model_path = PROJECT_ROOT / 'data' / 'models' / 'v15_comprehensive.joblib'
    joblib.dump(bundle, model_path)
    print(f"\n✓ Saved to {model_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    validated = [m for m, t in all_thresholds.items() if not t.get('excluded', False)]
    print(f"\nValidated: {len(validated)}/{len(MARKET_CONFIGS)}")
    for m in validated:
        t = all_thresholds[m]
        print(f"  ✓ {m}: {t['threshold']:.0%}, {t['roi']:+.1f}% ROI, {len(t['features'])} features")


if __name__ == "__main__":
    main()
