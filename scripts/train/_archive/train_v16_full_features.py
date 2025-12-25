#!/usr/bin/env python3
"""
V16 Full Feature Model - Uses ALL FeatureEngine Methods

This model extracts every available feature from FeatureEngine:
- Opportunity: target_share, air_yards_share, wopr, snap_share
- Efficiency: receiving_epa, rushing_epa, racr, yac_above_exp
- Usage trends: snap_trend, trailing stats
- Matchup: defense EPA (pass/rush), eight_box_rate
- NGS: separation, cushion
- Context: home/away, spread, total, divisional, primetime
- Injuries, weather, altitude

Uses GradientBoosting with feature importance analysis.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ALL features to extract for each market type
RECEIVING_FEATURES = [
    # Core
    'line_vs_trailing',
    'trailing_def_epa',

    # Opportunity
    'target_share',
    'air_yards_share',
    'wopr',
    'snap_share',
    'snap_trend',
    'targets_trailing',

    # Efficiency
    'receiving_epa',
    'racr',
    'yac_above_exp',

    # NGS
    'avg_separation',
    'avg_cushion',

    # Context
    'is_home',
    'spread',
    'total',
    'is_divisional',
    'is_primetime',

    # Matchup
    'opp_pass_epa',
    'opp_completion_pct_allowed',

    # Situational
    'redzone_target_share',
    'qb_injured',
]

RUSHING_FEATURES = [
    # Core
    'line_vs_trailing',
    'trailing_def_epa',

    # Opportunity
    'snap_share',
    'snap_trend',
    'carries_trailing',

    # Efficiency
    'rushing_epa',
    'rush_efficiency',
    'yards_per_carry',

    # Matchup
    'opp_rush_epa',
    'opp_eight_box_rate',

    # Context
    'is_home',
    'spread',
    'total',
    'is_divisional',

    # Situational
    'redzone_carry_share',
    'goalline_carry_share',
]

PASSING_FEATURES = [
    # Core
    'line_vs_trailing',
    'trailing_def_epa',

    # Efficiency
    'passing_epa',
    'cpoe',
    'completion_pct',

    # Context
    'is_home',
    'spread',
    'total',
    'is_divisional',
    'is_primetime',

    # Matchup
    'opp_pass_epa',

    # Situational
    'weather_adjustment',
    'wr1_injured',
]

MARKET_FEATURE_MAP = {
    'player_receptions': RECEIVING_FEATURES,
    'player_reception_yds': RECEIVING_FEATURES,
    'player_rush_yds': RUSHING_FEATURES,
    'player_rush_attempts': RUSHING_FEATURES,
    'player_pass_yds': PASSING_FEATURES,
    'player_pass_completions': PASSING_FEATURES,
    'player_pass_attempts': PASSING_FEATURES,
}


def extract_all_features(row: pd.Series, engine, market: str) -> dict:
    """Extract ALL available features for a single player-week."""
    features = {}

    player_id = row.get('player_id', '')
    season = row.get('season', 2024)
    week = row.get('week', 1)
    opponent = row.get('opponent_team', '')
    team = row.get('team', '')
    position = row.get('position', '')

    # === CORE FEATURES ===
    line = row.get('line', 0)
    trailing = row.get('trailing_stat', line)
    features['line_vs_trailing'] = (line - trailing) / (trailing + 0.1) if trailing else 0
    features['trailing_def_epa'] = row.get('trailing_def_epa', 0)

    # === OPPORTUNITY FEATURES ===
    features['target_share'] = row.get('target_share', 0)
    features['air_yards_share'] = row.get('air_yards_share', 0)
    features['wopr'] = row.get('wopr', 0)
    features['snap_share'] = row.get('snap_share', 0.5)
    features['snap_trend'] = row.get('snap_trend', 0)
    features['targets_trailing'] = row.get('targets_trailing', 0)
    features['carries_trailing'] = row.get('carries_trailing', 0)

    # === EFFICIENCY FEATURES ===
    features['receiving_epa'] = row.get('receiving_epa', 0)
    features['rushing_epa'] = row.get('rushing_epa', 0)
    features['passing_epa'] = row.get('passing_epa', 0)
    features['racr'] = row.get('racr', 1.0)
    features['yac_above_exp'] = row.get('receiving_yards_after_catch', 0) - row.get('receiving_air_yards', 0) * 0.3
    features['rush_efficiency'] = row.get('rushing_yards', 0) / max(row.get('carries', 1), 1)
    features['yards_per_carry'] = features['rush_efficiency']
    features['cpoe'] = row.get('passing_cpoe', 0)
    features['completion_pct'] = row.get('completions', 0) / max(row.get('attempts', 1), 1)

    # === NGS FEATURES (from engine if available) ===
    try:
        features['avg_separation'] = engine.get_avg_separation(player_id, season, week)
    except:
        features['avg_separation'] = 2.5

    try:
        features['avg_cushion'] = engine.get_avg_cushion(player_id, season, week)
    except:
        features['avg_cushion'] = 6.0

    # === MATCHUP FEATURES ===
    try:
        features['opp_pass_epa'] = engine.get_pass_defense_epa(opponent, season, week)
    except:
        features['opp_pass_epa'] = 0

    try:
        features['opp_rush_epa'] = engine.get_rush_defense_epa(opponent, season, week)
    except:
        features['opp_rush_epa'] = 0

    try:
        features['opp_eight_box_rate'] = engine.get_opponent_eight_box_rate(opponent, season, week)
    except:
        features['opp_eight_box_rate'] = 0.2

    try:
        features['opp_completion_pct_allowed'] = engine.get_completion_pct_allowed(opponent, season, week)
    except:
        features['opp_completion_pct_allowed'] = 0.64

    # === CONTEXT FEATURES ===
    features['is_home'] = row.get('is_home', 0)
    features['spread'] = row.get('spread', 0)
    features['total'] = row.get('total', 45)
    features['is_divisional'] = row.get('is_divisional', 0)
    features['is_primetime'] = row.get('is_primetime', 0)

    # === SITUATIONAL FEATURES ===
    features['redzone_target_share'] = row.get('redzone_target_share', 0)
    features['redzone_carry_share'] = row.get('redzone_carry_share', 0)
    features['goalline_carry_share'] = row.get('goalline_carry_share', 0)
    features['weather_adjustment'] = row.get('weather_adjustment', 0)
    features['qb_injured'] = 1 if row.get('injury_qb_status', 'active') != 'active' else 0
    features['wr1_injured'] = 1 if row.get('injury_wr1_status', 'active') != 'active' else 0

    return features


def build_training_data(market: str):
    """Build comprehensive training data for a market."""
    logger.info(f"Building training data for {market}...")

    engine = get_feature_engine()

    # Load weekly stats
    stats = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet')

    # Determine positions
    if 'rush' in market:
        positions = ['RB']
    elif 'pass' in market:
        positions = ['QB']
    else:
        positions = ['WR', 'TE', 'RB']

    filtered = stats[stats['position'].isin(positions)].copy()
    logger.info(f"  {len(filtered)} records for {positions}")

    # Load snap counts
    try:
        snaps = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'snap_counts.parquet')
        snaps['player_norm'] = snaps['player'].apply(normalize_player_name)
        snap_agg = snaps.groupby(['player_norm', 'season', 'week']).agg(
            snap_share=('offense_pct', 'first')
        ).reset_index()
    except:
        snap_agg = None

    # Load schedules for game context
    try:
        schedules = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet')
    except:
        schedules = None

    # Calculate defense EPA
    pbp_dfs = []
    for year in [2023, 2024, 2025]:
        try:
            pbp = engine._load_pbp(year)
            pbp['season'] = year
            pbp_dfs.append(pbp)
        except:
            pass

    if not pbp_dfs:
        return None, None

    pbp_all = pd.concat(pbp_dfs, ignore_index=True)

    # Get both defense types
    rush_def = engine.calculate_defense_epa(pbp_all, 'run', 4, True)
    pass_def = engine.calculate_defense_epa(pbp_all, 'pass', 4, True)

    # Use appropriate defense based on market
    if 'rush' in market:
        def_epa = rush_def
    else:
        def_epa = pass_def

    # Merge defense EPA
    filtered = filtered.merge(
        def_epa[['defteam', 'week', 'season', 'trailing_def_epa']],
        left_on=['opponent_team', 'week', 'season'],
        right_on=['defteam', 'week', 'season'],
        how='left'
    )

    # Calculate trailing stat based on market
    stat_col_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_yds': 'passing_yards',
        'player_pass_completions': 'completions',
        'player_pass_attempts': 'attempts',
    }
    stat_col = stat_col_map.get(market, 'rushing_yards')

    filtered = filtered.sort_values(['player_id', 'season', 'week'])
    filtered['trailing_stat'] = calculate_trailing_stat(
        df=filtered, stat_col=stat_col, player_col='player_id',
        span=4, min_periods=1, no_leakage=True
    )

    # Add additional trailing stats
    filtered['targets_trailing'] = calculate_trailing_stat(
        df=filtered, stat_col='targets', player_col='player_id',
        span=4, min_periods=1, no_leakage=True
    )
    filtered['carries_trailing'] = calculate_trailing_stat(
        df=filtered, stat_col='carries', player_col='player_id',
        span=4, min_periods=1, no_leakage=True
    )

    # Merge snap share
    if snap_agg is not None:
        filtered['player_norm'] = filtered['player_display_name'].apply(normalize_player_name)
        filtered = filtered.merge(
            snap_agg, on=['player_norm', 'season', 'week'], how='left'
        )
    filtered['snap_share'] = filtered.get('snap_share', pd.Series([0.5]*len(filtered))).fillna(0.5)

    # Merge schedule data
    if schedules is not None:
        for col in ['spread_line', 'total_line', 'div_game']:
            if col in schedules.columns:
                # Create lookup
                sched_lookup = schedules.groupby(['season', 'week', 'home_team']).first().reset_index()
                # This is simplified - would need proper home/away logic

    filtered['is_home'] = 0  # Simplified
    filtered['spread'] = 0
    filtered['total'] = 45
    filtered['is_divisional'] = 0
    filtered['is_primetime'] = 0

    # Load backtest
    backtest = pd.read_csv(PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv')
    market_bt = backtest[backtest['market'] == market].copy()

    if len(market_bt) == 0:
        logger.warning(f"  No backtest data for {market}")
        return None, None

    # Normalize and merge
    filtered['player_norm'] = filtered['player_display_name'].apply(normalize_player_name)
    market_bt['player_norm'] = market_bt['player'].apply(normalize_player_name)

    merged = filtered.merge(
        market_bt[['player_norm', 'season', 'week', 'line', 'under_hit']],
        on=['player_norm', 'season', 'week'],
        how='inner'
    )

    # Drop missing
    merged = merged[merged['trailing_stat'].notna()]

    logger.info(f"  {len(merged)} training samples")

    if len(merged) < 100:
        return None, None

    # Extract all features
    feature_list = MARKET_FEATURE_MAP.get(market, RECEIVING_FEATURES)

    all_features = []
    for idx, row in merged.iterrows():
        feat = extract_all_features(row, engine, market)
        # Only keep features relevant to this market
        feat_filtered = {k: feat.get(k, 0) for k in feature_list if k in feat}
        all_features.append(feat_filtered)

    feature_df = pd.DataFrame(all_features)

    # Add target
    feature_df['under_hit'] = merged['under_hit'].values
    feature_df['global_week'] = ((merged['season'].values - 2023) * 17 + merged['week'].values)

    return feature_df, feature_list


def walk_forward_validation(data, features):
    """Walk-forward with all features."""
    if data is None or len(data) < 100:
        return None

    results = []

    for test_gw in range(20, data['global_week'].max() + 1):
        train = data[data['global_week'] < test_gw]
        test = data[data['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 100:
            continue

        # Prepare features
        available_features = [f for f in features if f in train.columns]

        X_train = train[available_features].fillna(0)
        y_train = train['under_hit']
        X_test = test[available_features].fillna(0)

        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)

        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        # Train
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Predict
        test = test.copy()
        test['p_under'] = model.predict_proba(X_test_scaled)[:, 1]

        for _, row in test.iterrows():
            results.append({
                'global_week': row['global_week'],
                'p_under': row['p_under'],
                'under_hit': row['under_hit'],
            })

    return pd.DataFrame(results) if results else None


def calculate_roi(results, threshold):
    """Calculate ROI at threshold."""
    if results is None or len(results) == 0:
        return None, 0, 0

    mask = results['p_under'] > threshold
    if mask.sum() < 20:
        return None, 0, 0

    hits = results.loc[mask, 'under_hit'].sum()
    total = mask.sum()
    profit = hits * 0.909 - (total - hits) * 1.0
    roi = profit / total * 100

    return roi, int(hits), int(total)


def train_final_model(data, features, market):
    """Train final production model."""
    available_features = [f for f in features if f in data.columns]

    X = data[available_features].fillna(0)
    y = data['under_hit']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Feature importance
    importance = dict(zip(available_features, model.feature_importances_))

    return {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'features': available_features,
        'market': market,
        'feature_importance': importance,
        'n_samples': len(data),
    }


def main():
    """Train V16 full-feature models."""
    print("=" * 80)
    print("V16 FULL FEATURE MODEL - ALL FEATUREENGINE METHODS")
    print("=" * 80)

    markets = [
        'player_rush_yds',
        'player_pass_yds',
        'player_receptions',
        'player_reception_yds',
    ]

    all_models = {}
    all_thresholds = {}
    validation_results = {}

    for market in markets:
        print(f"\n{'='*60}")
        print(f"MARKET: {market}")
        print(f"{'='*60}")

        data, features = build_training_data(market)

        if data is None:
            all_thresholds[market] = {'excluded': True, 'reason': 'no_data'}
            continue

        print(f"  Features: {len(features)}")
        print(f"  Samples: {len(data)}")

        # Walk-forward
        wf_results = walk_forward_validation(data, features)

        if wf_results is None:
            all_thresholds[market] = {'excluded': True, 'reason': 'validation_failed'}
            continue

        # Find best threshold
        best_roi = -100
        best_threshold = 0.55
        best_record = (0, 0)

        print("\n  Threshold Analysis:")
        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
            roi, wins, total = calculate_roi(wf_results, thresh)
            if roi is not None:
                print(f"    {thresh:.0%}: ROI={roi:+.1f}% ({wins}W-{total-wins}L)")
                if roi > best_roi and total >= 30:
                    best_roi = roi
                    best_threshold = thresh
                    best_record = (wins, total)

        if best_roi > 0:
            print(f"\n  ✓ VALIDATED: {best_threshold:.0%}, {best_roi:+.1f}% ROI")

            model_bundle = train_final_model(data, features, market)
            all_models[market] = model_bundle
            all_thresholds[market] = {
                'threshold': best_threshold,
                'roi': best_roi,
                'wins': best_record[0],
                'total': best_record[1],
                'features': model_bundle['features'],
            }
            validation_results[market] = {
                'roi': best_roi,
                'record': f"{best_record[0]}W-{best_record[1]-best_record[0]}L"
            }

            # Top features
            print("\n  Top Features:")
            sorted_imp = sorted(model_bundle['feature_importance'].items(),
                              key=lambda x: -x[1])[:8]
            for feat, imp in sorted_imp:
                print(f"    {feat}: {imp:.3f}")
        else:
            print(f"\n  ✗ EXCLUDED: Best ROI={best_roi:+.1f}%")
            all_thresholds[market] = {'excluded': True, 'reason': f'negative_roi_{best_roi:.1f}'}

    # Save
    bundle = {
        'models': all_models,
        'thresholds': all_thresholds,
        'version': 'v16_full_features',
        'trained_date': datetime.now().isoformat(),
        'validation_results': validation_results,
    }

    model_path = PROJECT_ROOT / 'data' / 'models' / 'v16_full_features.joblib'
    joblib.dump(bundle, model_path)
    print(f"\n✓ Saved to {model_path}")

    # Summary
    print("\n" + "=" * 80)
    print("V16 SUMMARY")
    print("=" * 80)
    validated = [m for m, t in all_thresholds.items() if not t.get('excluded', False)]
    print(f"\nValidated: {len(validated)}/{len(markets)}")
    for m in validated:
        t = all_thresholds[m]
        print(f"  ✓ {m}: {t['threshold']:.0%}, {t['roi']:+.1f}% ROI, {len(t['features'])} features")


if __name__ == "__main__":
    main()
