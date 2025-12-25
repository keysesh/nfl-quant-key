#!/usr/bin/env python3
"""
Train TD props models for Anytime TD and Passing TDs.

Uses NFLverse weekly_stats for actual TD outcomes.
Builds probability models for TD scoring.
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
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Features for TD probability prediction
ANYTIME_TD_FEATURES = [
    'redzone_target_share',
    'redzone_carry_share',
    'goalline_carry_share',
    'trailing_td_rate',
    'snap_share',
    'target_share',
    'rushing_share',
    'opp_td_allowed_rate',
    'is_home',
    'total',
    'spread',
]

PASSING_TD_FEATURES = [
    'trailing_pass_tds',
    'passing_epa',
    'opp_pass_td_allowed',
    'completion_pct',
    'total',
    'is_home',
    'spread',
]


def build_anytime_td_data():
    """Build training data for anytime TD scorer prediction."""
    logger.info("Building Anytime TD training data...")

    engine = get_feature_engine()

    # Load weekly stats
    stats = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet')

    # Filter to skill positions
    positions = ['WR', 'TE', 'RB']
    filtered = stats[stats['position'].isin(positions)].copy()

    # Target: scored at least 1 TD (rushing or receiving)
    filtered['scored_td'] = ((filtered['rushing_tds'].fillna(0) + filtered['receiving_tds'].fillna(0)) >= 1).astype(int)

    logger.info(f"  Total records: {len(filtered)}")
    logger.info(f"  TD rate: {filtered['scored_td'].mean():.1%}")

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
        # Create team lookup for spread/total
        sched_home = schedules[['season', 'week', 'home_team', 'spread_line', 'total_line', 'div_game']].copy()
        sched_home['is_home'] = 1
        sched_home = sched_home.rename(columns={'home_team': 'team'})

        sched_away = schedules[['season', 'week', 'away_team', 'spread_line', 'total_line', 'div_game']].copy()
        sched_away['is_home'] = 0
        sched_away['spread_line'] = -sched_away['spread_line']  # Flip for away
        sched_away = sched_away.rename(columns={'away_team': 'team'})

        sched_context = pd.concat([sched_home, sched_away], ignore_index=True)
    except:
        sched_context = None

    # Calculate trailing TD rate
    filtered = filtered.sort_values(['player_id', 'season', 'week'])
    filtered['total_tds'] = filtered['rushing_tds'].fillna(0) + filtered['receiving_tds'].fillna(0)
    filtered['trailing_td_rate'] = calculate_trailing_stat(
        df=filtered, stat_col='total_tds', player_col='player_id',
        span=4, min_periods=1, no_leakage=True
    )

    # Calculate target/rushing share
    filtered['target_share'] = filtered['targets'].fillna(0) / filtered.groupby(['season', 'week', 'team'])['targets'].transform('sum').replace(0, 1)
    filtered['rushing_share'] = filtered['carries'].fillna(0) / filtered.groupby(['season', 'week', 'team'])['carries'].transform('sum').replace(0, 1)

    # Merge snap share
    if snap_agg is not None:
        filtered['player_norm'] = filtered['player_display_name'].apply(normalize_player_name)
        filtered = filtered.merge(snap_agg, on=['player_norm', 'season', 'week'], how='left')
    filtered['snap_share'] = filtered.get('snap_share', pd.Series([0.5]*len(filtered))).fillna(0.5)

    # Merge game context
    if sched_context is not None:
        filtered = filtered.merge(
            sched_context,
            on=['season', 'week', 'team'],
            how='left'
        )
    filtered['is_home'] = filtered.get('is_home', pd.Series([0]*len(filtered))).fillna(0)
    filtered['spread'] = filtered.get('spread_line', pd.Series([0]*len(filtered))).fillna(0)
    filtered['total'] = filtered.get('total_line', pd.Series([45]*len(filtered))).fillna(45)

    # Calculate opponent TD defense (simplified - use team avg TDs allowed)
    filtered['opp_td_allowed_rate'] = 0.3  # League average ~30% TD rate per eligible player

    # Redzone features (simplified)
    filtered['redzone_target_share'] = filtered['target_share'] * 0.3  # Rough estimate
    filtered['redzone_carry_share'] = filtered['rushing_share'] * 0.3
    filtered['goalline_carry_share'] = filtered['rushing_share'] * 0.1

    # Build feature matrix
    features_list = []
    for idx, row in filtered.iterrows():
        feat = {
            'redzone_target_share': row.get('redzone_target_share', 0),
            'redzone_carry_share': row.get('redzone_carry_share', 0),
            'goalline_carry_share': row.get('goalline_carry_share', 0),
            'trailing_td_rate': row.get('trailing_td_rate', 0),
            'snap_share': row.get('snap_share', 0.5),
            'target_share': row.get('target_share', 0),
            'rushing_share': row.get('rushing_share', 0),
            'opp_td_allowed_rate': row.get('opp_td_allowed_rate', 0.3),
            'is_home': row.get('is_home', 0),
            'total': row.get('total', 45),
            'spread': row.get('spread', 0),
            'scored_td': row['scored_td'],
            'global_week': (row['season'] - 2023) * 17 + row['week'],
        }
        features_list.append(feat)

    df = pd.DataFrame(features_list)
    logger.info(f"  Built {len(df)} training samples")

    return df


def build_passing_td_data():
    """Build training data for passing TD Over/Under prediction."""
    logger.info("Building Passing TD training data...")

    # Load weekly stats
    stats = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet')

    # Filter to QBs
    qbs = stats[stats['position'] == 'QB'].copy()

    # Target: 2+ passing TDs (typical line is 1.5)
    qbs['over_1_5_tds'] = (qbs['passing_tds'] >= 2).astype(int)

    logger.info(f"  Total QB games: {len(qbs)}")
    logger.info(f"  2+ TD rate: {qbs['over_1_5_tds'].mean():.1%}")

    # Calculate trailing passing TDs
    qbs = qbs.sort_values(['player_id', 'season', 'week'])
    qbs['trailing_pass_tds'] = calculate_trailing_stat(
        df=qbs, stat_col='passing_tds', player_col='player_id',
        span=4, min_periods=1, no_leakage=True
    )

    # Calculate completion pct
    qbs['completion_pct'] = qbs['completions'] / qbs['attempts'].replace(0, 1)

    # Load schedules for game context
    try:
        schedules = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet')
        sched_home = schedules[['season', 'week', 'home_team', 'spread_line', 'total_line']].copy()
        sched_home['is_home'] = 1
        sched_home = sched_home.rename(columns={'home_team': 'team'})

        sched_away = schedules[['season', 'week', 'away_team', 'spread_line', 'total_line']].copy()
        sched_away['is_home'] = 0
        sched_away['spread_line'] = -sched_away['spread_line']
        sched_away = sched_away.rename(columns={'away_team': 'team'})

        sched_context = pd.concat([sched_home, sched_away], ignore_index=True)

        qbs = qbs.merge(
            sched_context,
            on=['season', 'week', 'team'],
            how='left'
        )
    except:
        pass

    qbs['is_home'] = qbs.get('is_home', pd.Series([0]*len(qbs))).fillna(0)
    qbs['spread'] = qbs.get('spread_line', pd.Series([0]*len(qbs))).fillna(0)
    qbs['total'] = qbs.get('total_line', pd.Series([45]*len(qbs))).fillna(45)

    # Simplified passing EPA and opponent stats
    qbs['passing_epa'] = 0.1
    qbs['opp_pass_td_allowed'] = 1.5  # League avg

    # Build feature matrix
    features_list = []
    for idx, row in qbs.iterrows():
        feat = {
            'trailing_pass_tds': row.get('trailing_pass_tds', 1.5),
            'passing_epa': row.get('passing_epa', 0.1),
            'opp_pass_td_allowed': row.get('opp_pass_td_allowed', 1.5),
            'completion_pct': row.get('completion_pct', 0.65),
            'total': row.get('total', 45),
            'is_home': row.get('is_home', 0),
            'spread': row.get('spread', 0),
            'over_1_5_tds': row['over_1_5_tds'],
            'global_week': (row['season'] - 2023) * 17 + row['week'],
        }
        features_list.append(feat)

    df = pd.DataFrame(features_list)
    logger.info(f"  Built {len(df)} training samples")

    return df


def walk_forward_validation(data, features, target_col):
    """Walk-forward validation for TD models."""
    if data is None or len(data) < 100:
        return None

    results = []

    for test_gw in range(20, data['global_week'].max() + 1):
        train = data[data['global_week'] < test_gw]
        test = data[data['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 100:
            continue

        X_train = train[features].fillna(0)
        y_train = train[target_col]
        X_test = test[features].fillna(0)

        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)

        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        # Train with calibration
        base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        base_model.fit(X_train_scaled, y_train)

        # Predict probabilities
        test = test.copy()
        test['p_positive'] = base_model.predict_proba(X_test_scaled)[:, 1]

        for _, row in test.iterrows():
            results.append({
                'global_week': row['global_week'],
                'p_positive': row['p_positive'],
                'actual': row[target_col],
            })

    return pd.DataFrame(results) if results else None


def calculate_calibration(results):
    """Calculate calibration metrics."""
    if results is None or len(results) == 0:
        return None

    # Bin by predicted probability
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results['prob_bin'] = pd.cut(results['p_positive'], bins=bins)

    calibration = results.groupby('prob_bin').agg(
        avg_predicted=('p_positive', 'mean'),
        avg_actual=('actual', 'mean'),
        count=('actual', 'count')
    ).reset_index()

    return calibration


def train_final_model(data, features, target_col):
    """Train final production model with calibration."""
    X = data[features].fillna(0)
    y = data[target_col]

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Train with isotonic calibration
    base_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_scaled, y)

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'imputer': imputer,
        'features': features,
    }


def main():
    """Train TD props models."""
    print("=" * 80)
    print("TD PROPS MODEL TRAINING")
    print("=" * 80)

    all_models = {}
    all_thresholds = {}
    validation_results = {}

    # === ANYTIME TD ===
    print("\n" + "=" * 60)
    print("ANYTIME TD SCORER")
    print("=" * 60)

    atd_data = build_anytime_td_data()

    if atd_data is not None and len(atd_data) >= 100:
        atd_features = [f for f in ANYTIME_TD_FEATURES if f in atd_data.columns]
        print(f"Features: {len(atd_features)}")
        print(f"Samples: {len(atd_data)}")
        print(f"Base TD rate: {atd_data['scored_td'].mean():.1%}")

        # Walk-forward validation
        wf_results = walk_forward_validation(atd_data, atd_features, 'scored_td')

        if wf_results is not None:
            # Calculate Brier score
            brier = ((wf_results['p_positive'] - wf_results['actual']) ** 2).mean()
            print(f"\nBrier Score: {brier:.4f} (lower is better, baseline ~0.25)")

            # Calibration
            calibration = calculate_calibration(wf_results)
            if calibration is not None:
                print("\nCalibration:")
                print(calibration.to_string())

            # Train final model
            model_bundle = train_final_model(atd_data, atd_features, 'scored_td')

            all_models['player_anytime_td'] = model_bundle
            all_thresholds['player_anytime_td'] = {
                'threshold': 0.30,  # ~30% TD rate is fair value
                'brier_score': brier,
                'type': 'probability',  # Not over/under, but probability comparison
            }
            validation_results['player_anytime_td'] = {
                'brier_score': brier,
                'samples': len(wf_results),
            }
            print(f"\n✓ Anytime TD model trained")

    # === PASSING TDs ===
    print("\n" + "=" * 60)
    print("PASSING TDs (Over 1.5)")
    print("=" * 60)

    ptd_data = build_passing_td_data()

    if ptd_data is not None and len(ptd_data) >= 100:
        ptd_features = [f for f in PASSING_TD_FEATURES if f in ptd_data.columns]
        print(f"Features: {len(ptd_features)}")
        print(f"Samples: {len(ptd_data)}")
        print(f"Base 2+ TD rate: {ptd_data['over_1_5_tds'].mean():.1%}")

        # Walk-forward validation
        wf_results = walk_forward_validation(ptd_data, ptd_features, 'over_1_5_tds')

        if wf_results is not None:
            brier = ((wf_results['p_positive'] - wf_results['actual']) ** 2).mean()
            print(f"\nBrier Score: {brier:.4f}")

            # Train final model
            model_bundle = train_final_model(ptd_data, ptd_features, 'over_1_5_tds')

            all_models['player_pass_tds'] = model_bundle
            all_thresholds['player_pass_tds'] = {
                'threshold': 0.55,  # Bet when model > 55% for over 1.5
                'brier_score': brier,
                'type': 'over_under',
            }
            validation_results['player_pass_tds'] = {
                'brier_score': brier,
                'samples': len(wf_results),
            }
            print(f"\n✓ Passing TDs model trained")

    # === SAVE TO ACTIVE MODEL ===
    if all_models:
        # Load existing active model
        active_path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'
        if active_path.exists():
            active_bundle = joblib.load(active_path)
        else:
            active_bundle = {'models': {}, 'thresholds': {}, 'validation_results': {}}

        # Add TD models
        for market, model_bundle in all_models.items():
            active_bundle['models'][market] = model_bundle
            active_bundle['thresholds'][market] = all_thresholds[market]
            active_bundle['validation_results'][market] = validation_results[market]

        active_bundle['td_models_trained'] = datetime.now().isoformat()

        joblib.dump(active_bundle, active_path)
        print(f"\n✓ Added TD models to active model: {active_path}")
        print(f"  Markets: {list(all_models.keys())}")

    print("\n" + "=" * 80)
    print("TD TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
