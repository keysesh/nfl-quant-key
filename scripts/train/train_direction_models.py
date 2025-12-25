#!/usr/bin/env python3
"""
Train Direction-Specific Edge Models

Instead of one model that predicts P(UNDER), train TWO models:
1. UNDER model: Learns when UNDER is the right pick
2. OVER model: Learns when OVER is the right pick

This fixes the problem where the unified model is overconfident on OVER picks
because OVER requires different signals than UNDER.

Usage:
    python scripts/train/train_direction_models.py
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import MODELS_DIR, DATA_DIR
from nfl_quant.utils.player_names import normalize_player_name
from configs.edge_config import EDGE_MARKETS

# Features for UNDER model (regression to mean signals)
UNDER_FEATURES = [
    'line_vs_trailing',      # Positive = line above trailing (UNDER signal)
    'line_vs_trailing_pct',  # Percentage difference
    'line_level',            # Raw line value
    'trailing_cv',           # Player consistency (high CV = volatile)
    'recent_trend',          # Is player trending down?
    'games_played',          # Sample size
    'market_under_rate',     # Historical UNDER rate for this market
    'vegas_spread',          # Game context
    'implied_team_total',    # Scoring environment
]

# Features for OVER model (breakout signals)
OVER_FEATURES = [
    'line_vs_trailing',      # Negative = line below trailing (OVER signal)
    'line_vs_trailing_pct',  # Percentage difference
    'line_level',            # Raw line value
    'recent_trend',          # Is player trending up?
    'trend_strength',        # How strong is the trend?
    'games_above_line',      # Recent games above this line level
    'target_share_trend',    # Increasing usage?
    'snap_share',            # Playing time
    'favorable_matchup',     # Opponent defense ranking
]


def load_training_data() -> pd.DataFrame:
    """Load historical odds/actuals data for training."""
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'

    if enriched_path.exists():
        print(f"Loading enriched data: {enriched_path}")
        df = pd.read_csv(enriched_path, low_memory=False)
    else:
        combined_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
        print(f"Loading combined data: {combined_path}")
        df = pd.read_csv(combined_path, low_memory=False)

    print(f"Loaded {len(df)} rows")
    return df


def compute_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for direction-specific models from historical actuals."""
    df = df.copy()

    # Sort by player, market, then time for proper rolling calculations
    df = df.sort_values(['player', 'market', 'season', 'week']).reset_index(drop=True)

    # Normalize player names
    if 'player_norm' not in df.columns and 'player' in df.columns:
        df['player_norm'] = df['player'].apply(normalize_player_name)

    # Create global week for sorting
    df['global_week'] = (df['season'] - 2023) * 18 + df['week']

    # Compute trailing average from actual stats (shifted to prevent leakage)
    # Group by player + market, then compute rolling mean of prior games
    df['trailing_avg'] = (
        df.groupby(['player', 'market'])['actual']
        .transform(lambda x: x.shift(1).rolling(6, min_periods=3).mean())
    )

    # Fill NaN trailing with line value (no historical data yet)
    df['trailing_avg'] = df['trailing_avg'].fillna(df['line'])

    # Core LVT feature: line - trailing_avg
    # Positive = line is higher than player's average (UNDER signal)
    # Negative = line is lower than player's average (OVER signal)
    df['line_vs_trailing'] = df['line'] - df['trailing_avg']
    df['line_vs_trailing_pct'] = np.where(
        df['trailing_avg'] > 0,
        (df['line'] - df['trailing_avg']) / df['trailing_avg'] * 100,
        0
    )

    # Line level
    df['line_level'] = df['line']

    # Trailing CV (consistency measure) - std of last 6 games
    df['trailing_std'] = (
        df.groupby(['player', 'market'])['actual']
        .transform(lambda x: x.shift(1).rolling(6, min_periods=3).std())
    ).fillna(0)

    df['trailing_cv'] = np.where(
        df['trailing_avg'] > 0,
        df['trailing_std'] / df['trailing_avg'],
        0.3
    )

    # Recent trend: last 3 games avg vs last 6 games avg
    df['recent_3'] = (
        df.groupby(['player', 'market'])['actual']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    )
    df['recent_6'] = df['trailing_avg']  # Already computed
    recent_trend = np.where(
        df['recent_6'] > 0,
        (df['recent_3'] - df['recent_6']) / df['recent_6'],
        0
    )
    df['recent_trend'] = pd.Series(recent_trend, index=df.index).fillna(0)
    df['trend_strength'] = df['recent_trend'].abs()

    # Games played (count of prior games)
    df['games_played'] = (
        df.groupby(['player', 'market'])['actual']
        .transform(lambda x: x.shift(1).expanding().count())
    ).fillna(0)

    # Market UNDER rate (shifted to prevent leakage)
    df['market_under_rate'] = (
        df.groupby('market')['under_hit']
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0.5)

    # Vegas context
    df['vegas_spread'] = df['vegas_spread'].fillna(0)

    # Implied team total
    if 'vegas_total' in df.columns and 'vegas_spread' in df.columns:
        df['implied_team_total'] = df['vegas_total'] / 2 - df['vegas_spread'] / 2
    else:
        df['implied_team_total'] = 24.0

    # Games above this line level (for OVER model)
    df['above_line'] = (df['actual'] > df['line']).astype(int)
    df['games_above_line'] = (
        df.groupby(['player', 'market'])['above_line']
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).sum())
    ).fillna(0)

    # Clean up temp columns
    df = df.drop(columns=['recent_3', 'above_line'], errors='ignore')

    return df


def train_under_model(df: pd.DataFrame, market: str) -> dict:
    """Train model specifically for UNDER predictions."""
    print(f"\n  Training UNDER model for {market}...")

    # Filter to this market
    mdf = df[df['market'] == market].copy()

    # Only use cases where we have a strong UNDER signal (positive LVT)
    # This teaches the model what UNDER conditions look like
    mdf = mdf[mdf['line_vs_trailing'] > 0].copy()

    if len(mdf) < 50:
        print(f"    Insufficient UNDER samples: {len(mdf)}")
        return None

    # Target: did UNDER actually hit?
    y = mdf['under_hit']

    # Features
    features = ['line_vs_trailing', 'line_vs_trailing_pct', 'line_level',
                'trailing_cv', 'games_played', 'market_under_rate',
                'vegas_spread', 'implied_team_total']

    available_features = [f for f in features if f in mdf.columns]
    X = mdf[available_features]

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Temporal split (80% train, 20% test)
    if 'global_week' not in mdf.columns:
        mdf['global_week'] = (mdf['season'] - 2023) * 18 + mdf['week']

    sorted_idx = mdf.sort_values('global_week').index
    n_train = int(len(sorted_idx) * 0.8)

    train_pos = [mdf.index.get_loc(idx) for idx in sorted_idx[:n_train]]
    test_pos = [mdf.index.get_loc(idx) for idx in sorted_idx[n_train:]]

    X_train, X_test = X_scaled[train_pos], X_scaled[test_pos]
    y_train, y_test = y.iloc[train_pos].values, y.iloc[test_pos].values

    # Train XGBoost
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    train_acc = ((train_preds > 0.5) == y_train).mean()
    test_acc = ((test_preds > 0.5) == y_test).mean()

    print(f"    Samples: {len(mdf)}, Train: {train_acc:.1%}, Test: {test_acc:.1%}")

    return {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'features': available_features,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_samples': len(mdf),
    }


def train_over_model(df: pd.DataFrame, market: str) -> dict:
    """Train model specifically for OVER predictions."""
    print(f"\n  Training OVER model for {market}...")

    # Filter to this market
    mdf = df[df['market'] == market].copy()

    # Only use cases where we have a potential OVER signal (negative LVT or low line)
    # This teaches the model what OVER conditions look like
    mdf = mdf[mdf['line_vs_trailing'] < 0].copy()

    if len(mdf) < 50:
        print(f"    Insufficient OVER samples: {len(mdf)}")
        return None

    # Target: did OVER actually hit? (inverse of under_hit)
    y = 1 - mdf['under_hit']

    # Features - different emphasis for OVER
    features = ['line_vs_trailing', 'line_vs_trailing_pct', 'line_level',
                'trailing_cv', 'games_played', 'market_under_rate',
                'vegas_spread', 'implied_team_total', 'snap_share']

    available_features = [f for f in features if f in mdf.columns]
    X = mdf[available_features]

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Temporal split (80% train, 20% test)
    if 'global_week' not in mdf.columns:
        mdf['global_week'] = (mdf['season'] - 2023) * 18 + mdf['week']

    sorted_idx = mdf.sort_values('global_week').index
    n_train = int(len(sorted_idx) * 0.8)

    train_pos = [mdf.index.get_loc(idx) for idx in sorted_idx[:n_train]]
    test_pos = [mdf.index.get_loc(idx) for idx in sorted_idx[n_train:]]

    X_train, X_test = X_scaled[train_pos], X_scaled[test_pos]
    y_train, y_test = y.iloc[train_pos].values, y.iloc[test_pos].values

    # Train XGBoost
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    train_acc = ((train_preds > 0.5) == y_train).mean()
    test_acc = ((test_preds > 0.5) == y_test).mean()

    print(f"    Samples: {len(mdf)}, Train: {train_acc:.1%}, Test: {test_acc:.1%}")

    return {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'features': available_features,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_samples': len(mdf),
    }


def main():
    print("=" * 70)
    print("TRAINING DIRECTION-SPECIFIC EDGE MODELS")
    print("=" * 70)

    # Load data
    df = load_training_data()

    # Compute features (once for all markets)
    print("\nComputing direction features...")
    df = compute_direction_features(df)

    # Show LVT distribution
    print(f"\nLVT distribution:")
    print(f"  Positive (UNDER signals): {(df['line_vs_trailing'] > 0).sum():,}")
    print(f"  Negative (OVER signals):  {(df['line_vs_trailing'] < 0).sum():,}")
    print(f"  Zero (no signal):         {(df['line_vs_trailing'] == 0).sum():,}")

    # Train models for each market
    under_models = {}
    over_models = {}

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    for market in EDGE_MARKETS:
        print(f"\n{market}")
        print("-" * 50)

        # Train UNDER model
        under_result = train_under_model(df, market)
        if under_result:
            under_models[market] = under_result

        # Train OVER model
        over_result = train_over_model(df, market)
        if over_result:
            over_models[market] = over_result

    # Save models
    bundle = {
        'under_models': under_models,
        'over_models': over_models,
        'version': 'direction_v1',
        'trained_date': datetime.now().isoformat(),
    }

    save_path = MODELS_DIR / 'direction_edge_models.joblib'
    joblib.dump(bundle, save_path)
    print(f"\n\nSaved to: {save_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Market':<25} {'UNDER Test':<12} {'OVER Test':<12}")
    print("-" * 50)

    for market in EDGE_MARKETS:
        under_acc = under_models.get(market, {}).get('test_acc', 0)
        over_acc = over_models.get(market, {}).get('test_acc', 0)
        print(f"{market:<25} {under_acc:>10.1%} {over_acc:>10.1%}")


if __name__ == '__main__':
    main()
