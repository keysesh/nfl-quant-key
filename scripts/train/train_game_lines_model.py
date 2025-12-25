#!/usr/bin/env python3
"""
Train models for game lines: Totals (Over/Under), Spreads, Moneyline.

Uses historical game data from NFLverse schedules with betting lines and outcomes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.features import get_feature_engine


def load_game_data():
    """Load historical games with betting lines and outcomes."""
    sched_path = project_root / 'data' / 'nflverse' / 'schedules.parquet'
    sched = pd.read_parquet(sched_path)

    # Filter to completed regular season games with betting data
    games = sched[
        (sched['home_score'].notna()) &
        (sched['game_type'] == 'REG') &
        (sched['total_line'].notna()) &
        (sched['spread_line'].notna())
    ].copy()

    # Calculate outcomes
    games['total_points'] = games['home_score'] + games['away_score']
    games['home_margin'] = games['home_score'] - games['away_score']

    # Over/Under outcome (1 = OVER hit)
    games['over_hit'] = (games['total_points'] > games['total_line']).astype(int)

    # Spread outcome (1 = HOME covered)
    games['home_covered'] = (games['home_margin'] > games['spread_line']).astype(int)

    # Moneyline outcome (1 = HOME won)
    games['home_won'] = (games['home_margin'] > 0).astype(int)

    print(f"Loaded {len(games)} games with betting data")
    print(f"Seasons: {sorted(games['season'].unique())}")

    return games


def build_game_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build features for each game based on team stats."""
    engine = get_feature_engine()
    features_list = []

    for idx, row in games.iterrows():
        season = row['season']
        week = row['week']
        home_team = row['home_team']
        away_team = row['away_team']

        try:
            # Get team-level features from FeatureEngine
            # Home team offense vs Away team defense
            home_pass_epa = engine.get_pass_defense_epa(home_team, season, week) if hasattr(engine, 'get_pass_defense_epa') else 0
            away_pass_def = engine.get_pass_defense_epa(away_team, season, week) if hasattr(engine, 'get_pass_defense_epa') else 0
            home_rush_epa = engine.get_rush_defense_epa(home_team, season, week) if hasattr(engine, 'get_rush_defense_epa') else 0
            away_rush_def = engine.get_rush_defense_epa(away_team, season, week) if hasattr(engine, 'get_rush_defense_epa') else 0

            features = {
                'game_id': row['game_id'],
                'season': season,
                'week': week,

                # Betting lines
                'total_line': row['total_line'],
                'spread_line': row['spread_line'],
                'home_moneyline': row['home_moneyline'],
                'away_moneyline': row['away_moneyline'],

                # Line-derived features
                'implied_total': row['total_line'],
                'implied_home_score': (row['total_line'] - row['spread_line']) / 2,
                'implied_away_score': (row['total_line'] + row['spread_line']) / 2,

                # Spread magnitude
                'spread_magnitude': abs(row['spread_line']),
                'is_pick_em': 1 if abs(row['spread_line']) <= 1.5 else 0,
                'is_big_favorite': 1 if abs(row['spread_line']) >= 7 else 0,

                # Total line buckets
                'is_high_total': 1 if row['total_line'] >= 48 else 0,
                'is_low_total': 1 if row['total_line'] <= 40 else 0,

                # Defense EPA matchup
                'home_pass_def_epa': home_pass_epa,
                'away_pass_def_epa': away_pass_def,
                'home_rush_def_epa': home_rush_epa,
                'away_rush_def_epa': away_rush_def,

                # Combined matchup
                'total_def_epa': home_pass_epa + away_pass_def + home_rush_epa + away_rush_def,

                # Outcomes (targets)
                'over_hit': row['over_hit'],
                'home_covered': row['home_covered'],
                'home_won': row['home_won'],
                'total_points': row['total_points'],
                'home_margin': row['home_margin'],
            }

            features_list.append(features)

        except Exception as e:
            print(f"Error processing {row['game_id']}: {e}")
            continue

    df = pd.DataFrame(features_list)
    print(f"Built features for {len(df)} games")
    return df


def calculate_implied_probability(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(american_odds):
        return 0.5
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def train_market_model(df: pd.DataFrame, market: str, target_col: str, features: list):
    """Train a model for a specific market."""
    print(f"\n=== Training {market} model ===")

    # Prepare data
    X = df[features].copy()
    y = df[target_col].copy()

    # Remove rows with missing target
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    print(f"Training samples: {len(X)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Impute and scale
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    val_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Validate
        probs = model.predict_proba(X_val)[:, 1]

        # Calculate ROI at different thresholds
        for threshold in [0.50, 0.55, 0.60, 0.65]:
            picks = probs >= threshold
            if picks.sum() > 0:
                wins = (y_val[picks] == 1).sum()
                total = picks.sum()
                # Assume -110 odds
                roi = (wins * 0.909 - (total - wins)) / total * 100 if total > 0 else 0
                val_results.append({
                    'fold': fold,
                    'threshold': threshold,
                    'picks': total,
                    'wins': wins,
                    'win_pct': wins / total if total > 0 else 0,
                    'roi': roi
                })

    val_df = pd.DataFrame(val_results)

    # Find best threshold
    best_results = val_df.groupby('threshold').agg({
        'picks': 'sum',
        'wins': 'sum',
        'roi': 'mean'
    }).reset_index()
    best_results['win_pct'] = best_results['wins'] / best_results['picks']

    print("\nValidation Results by Threshold:")
    print(best_results.to_string())

    # Select threshold with best ROI (minimum 20 picks)
    valid = best_results[best_results['picks'] >= 20]
    if len(valid) > 0:
        best_row = valid.loc[valid['roi'].idxmax()]
        best_threshold = best_row['threshold']
        best_roi = best_row['roi']
    else:
        best_threshold = 0.55
        best_roi = 0

    # Train final model on all data
    final_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    final_model.fit(X_scaled, y)

    return {
        'model': final_model,
        'imputer': imputer,
        'scaler': scaler,
        'features': features,
        'threshold': best_threshold,
        'roi': best_roi,
        'validation': best_results.to_dict('records')
    }


def main():
    print("=" * 60)
    print("GAME LINES MODEL TRAINING")
    print("=" * 60)

    # Load data
    games = load_game_data()

    # Build features
    df = build_game_features(games)

    # Sort by season/week for time series validation
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    # Define feature sets
    totals_features = [
        'total_line', 'is_high_total', 'is_low_total',
        'spread_magnitude', 'total_def_epa',
        'home_pass_def_epa', 'away_pass_def_epa',
        'home_rush_def_epa', 'away_rush_def_epa',
    ]

    spread_features = [
        'spread_line', 'spread_magnitude', 'is_pick_em', 'is_big_favorite',
        'total_line', 'total_def_epa',
        'home_pass_def_epa', 'away_pass_def_epa',
    ]

    # Train models
    models = {}

    # Totals (Over/Under)
    result = train_market_model(df, 'totals', 'over_hit', totals_features)
    if result['roi'] > 0:
        models['game_totals'] = result
        print(f"\n✅ Totals: {result['roi']:+.1f}% ROI at {result['threshold']:.0%}")
    else:
        print(f"\n❌ Totals: No positive ROI found")

    # Spreads
    result = train_market_model(df, 'spreads', 'home_covered', spread_features)
    if result['roi'] > 0:
        models['game_spreads'] = result
        print(f"\n✅ Spreads: {result['roi']:+.1f}% ROI at {result['threshold']:.0%}")
    else:
        print(f"\n❌ Spreads: No positive ROI found")

    # Save to active model
    if models:
        # Load existing active model
        active_path = project_root / 'data' / 'models' / 'active_model.joblib'
        if active_path.exists():
            active_bundle = joblib.load(active_path)
        else:
            active_bundle = {'models': {}, 'thresholds': {}, 'validation_results': {}}

        # Add game line models
        for market, result in models.items():
            active_bundle['models'][market] = {
                'model': result['model'],
                'imputer': result['imputer'],
                'scaler': result['scaler'],
                'features': result['features']
            }
            active_bundle['thresholds'][market] = {
                'threshold': result['threshold'],
                'roi': result['roi'],
                'excluded': result['roi'] <= 0
            }
            active_bundle['validation_results'][market] = {
                'roi': result['roi'],
                'threshold': result['threshold']
            }

        active_bundle['game_lines_trained'] = datetime.now().isoformat()

        joblib.dump(active_bundle, active_path)
        print(f"\n✅ Saved game lines models to {active_path}")
        print(f"   Markets: {list(models.keys())}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
