#!/usr/bin/env python3
"""
Train ML Models for Game Line Predictions (Spreads & Totals)

Uses LightGBM to enhance Monte Carlo simulation predictions with ML-based features.
This combines simulation outputs with team EPA features for better accuracy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.utils.season_utils import get_current_season


def load_game_features():
    """Load game data with features for ML training."""
    # Load calibration data (has simulation outputs + actuals)
    cal_path = PROJECT_ROOT / 'reports' / 'game_line_calibration_data.csv'
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration data not found at {cal_path}")

    df = pd.read_csv(cal_path)
    print(f"Loaded {len(df)} games from calibration data")

    # Load schedules for additional features (Vegas lines, weather, etc.)
    schedules_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if schedules_path.exists():
        schedules = pd.read_parquet(schedules_path)

        # Merge with calibration data
        schedules['merge_key'] = schedules.apply(
            lambda r: f"{r['season']}_{r['week']:02d}_{r['away_team']}_{r['home_team']}", axis=1
        )
        df = df.merge(
            schedules[['merge_key', 'spread_line', 'total_line', 'home_rest', 'away_rest',
                      'div_game', 'temp', 'wind', 'roof']],
            left_on='game_id',
            right_on='merge_key',
            how='left'
        )
        print(f"  Merged with schedule features")

    return df


def create_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features for game line prediction."""
    features = pd.DataFrame()

    # Simulation features (from Monte Carlo)
    features['sim_home_win_prob'] = df['home_win_prob']
    features['sim_fair_spread'] = df['fair_spread']
    features['sim_fair_total'] = df['fair_total']

    # Calculate simulation confidence (spread of predictions)
    # Higher spread confidence = more certain prediction
    features['sim_spread_confidence'] = abs(df['fair_spread']) / 10.0  # Normalize

    # Vegas market lines (if available)
    if 'spread_line' in df.columns:
        features['vegas_spread'] = df['spread_line'].fillna(0)
        features['vegas_total'] = df['total_line'].fillna(45)
        features['has_vegas_line'] = df['spread_line'].notna().astype(int)

        # Deviation from Vegas
        features['sim_vs_vegas_spread'] = features['sim_fair_spread'] - features['vegas_spread']
        features['sim_vs_vegas_total'] = features['sim_fair_total'] - features['vegas_total']
    else:
        features['vegas_spread'] = 0
        features['vegas_total'] = 45
        features['has_vegas_line'] = 0
        features['sim_vs_vegas_spread'] = 0
        features['sim_vs_vegas_total'] = 0

    # Rest advantage
    if 'home_rest' in df.columns:
        features['home_rest'] = df['home_rest'].fillna(7)
        features['away_rest'] = df['away_rest'].fillna(7)
        features['rest_advantage'] = features['home_rest'] - features['away_rest']
    else:
        features['home_rest'] = 7
        features['away_rest'] = 7
        features['rest_advantage'] = 0

    # Divisional game
    if 'div_game' in df.columns:
        features['is_divisional'] = df['div_game'].fillna(0).astype(int)
    elif 'is_divisional' in df.columns:
        features['is_divisional'] = df['is_divisional'].astype(int)
    else:
        features['is_divisional'] = 0

    # Weather features
    if 'temp' in df.columns:
        features['temperature'] = df['temp'].fillna(70)
        features['cold_game'] = (features['temperature'] < 40).astype(int)
    else:
        features['temperature'] = 70
        features['cold_game'] = 0

    if 'wind' in df.columns:
        features['wind_speed'] = df['wind'].fillna(0)
        features['high_wind'] = (features['wind_speed'] > 15).astype(int)
    else:
        features['wind_speed'] = 0
        features['high_wind'] = 0

    # Dome indicator
    if 'roof' in df.columns:
        features['is_dome'] = df['roof'].isin(['dome', 'closed']).astype(int)
    elif 'is_dome' in df.columns:
        features['is_dome'] = df['is_dome'].astype(int)
    else:
        features['is_dome'] = 0

    # Week of season (early vs late season)
    features['week'] = df['week']
    features['early_season'] = (df['week'] <= 4).astype(int)
    features['late_season'] = (df['week'] >= 15).astype(int)

    return features


def train_spread_model(df: pd.DataFrame, features: pd.DataFrame):
    """Train LightGBM model for spread prediction."""
    print("\n" + "=" * 70)
    print("TRAINING SPREAD PREDICTION MODEL")
    print("=" * 70)

    # Target: actual spread (home - away)
    y = df['actual_spread']
    X = features

    print(f"Training samples: {len(X)}")
    print(f"Features: {list(X.columns)}")

    # Time series split (preserve temporal order)
    tscv = TimeSeriesSplit(n_splits=3)

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200,
        'early_stopping_rounds': 20,
    }

    # Cross-validation scores
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        cv_scores.append({'mae': mae, 'rmse': rmse})
        models.append(model)

        print(f"  Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Train final model on all data
    final_model = lgb.LGBMRegressor(**{k: v for k, v in params.items() if k != 'early_stopping_rounds'})
    final_model.fit(X, y)

    # Compare to simulation baseline
    sim_mae = mean_absolute_error(y, df['fair_spread'])
    ml_mae = np.mean([s['mae'] for s in cv_scores])

    print(f"\n=== SPREAD MODEL RESULTS ===")
    print(f"Simulation MAE: {sim_mae:.2f} points")
    print(f"ML Model MAE: {ml_mae:.2f} points")
    print(f"Improvement: {(sim_mae - ml_mae) / sim_mae * 100:.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

    return final_model, cv_scores, importance


def train_total_model(df: pd.DataFrame, features: pd.DataFrame):
    """Train LightGBM model for total points prediction."""
    print("\n" + "=" * 70)
    print("TRAINING TOTAL POINTS PREDICTION MODEL")
    print("=" * 70)

    # Target: actual total
    y = df['actual_total']
    X = features

    print(f"Training samples: {len(X)}")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200,
        'early_stopping_rounds': 20,
    }

    # Cross-validation
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        cv_scores.append({'mae': mae, 'rmse': rmse})
        print(f"  Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Train final model
    final_model = lgb.LGBMRegressor(**{k: v for k, v in params.items() if k != 'early_stopping_rounds'})
    final_model.fit(X, y)

    # Compare to simulation baseline
    sim_mae = mean_absolute_error(y, df['fair_total'])
    ml_mae = np.mean([s['mae'] for s in cv_scores])

    print(f"\n=== TOTAL MODEL RESULTS ===")
    print(f"Simulation MAE: {sim_mae:.2f} points")
    print(f"ML Model MAE: {ml_mae:.2f} points")
    print(f"Improvement: {(sim_mae - ml_mae) / sim_mae * 100:.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

    return final_model, cv_scores, importance


def train_win_prob_model(df: pd.DataFrame, features: pd.DataFrame):
    """Train LightGBM model for win probability prediction."""
    print("\n" + "=" * 70)
    print("TRAINING WIN PROBABILITY MODEL")
    print("=" * 70)

    # Target: binary outcome (home win)
    y = df['home_actual_win']
    X = features

    print(f"Training samples: {len(X)}")
    print(f"Home win rate: {y.mean():.3f}")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)

    # LightGBM parameters for classification
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200,
        'early_stopping_rounds': 20,
    }

    # Cross-validation
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        val_pred_prob = model.predict_proba(X_val)[:, 1]

        # Calculate Brier score
        from sklearn.metrics import brier_score_loss, log_loss
        brier = brier_score_loss(y_val, val_pred_prob)
        logloss = log_loss(y_val, val_pred_prob)

        cv_scores.append({'brier': brier, 'logloss': logloss})
        print(f"  Fold {fold + 1}: Brier={brier:.4f}, LogLoss={logloss:.4f}")

    # Train final model
    final_model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'early_stopping_rounds'})
    final_model.fit(X, y)

    # Compare to simulation baseline
    from sklearn.metrics import brier_score_loss
    sim_brier = brier_score_loss(y, df['home_win_prob'])
    ml_brier = np.mean([s['brier'] for s in cv_scores])

    print(f"\n=== WIN PROBABILITY MODEL RESULTS ===")
    print(f"Simulation Brier: {sim_brier:.4f}")
    print(f"ML Model Brier: {ml_brier:.4f}")
    print(f"Improvement: {(sim_brier - ml_brier) / sim_brier * 100:.1f}%")

    # Train calibrator for ML predictions
    final_probs = final_model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip', increasing=True)
    calibrator.fit(final_probs, y)

    calibrated_probs = calibrator.predict(final_probs)
    calibrated_brier = brier_score_loss(y, calibrated_probs)

    print(f"Calibrated Brier: {calibrated_brier:.4f}")

    return final_model, calibrator, cv_scores


def save_models(spread_model, total_model, win_prob_model, win_prob_calibrator, metadata):
    """Save trained models and metadata."""
    model_dir = PROJECT_ROOT / 'data' / 'models' / 'game_lines'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(spread_model, model_dir / 'spread_predictor.joblib')
    joblib.dump(total_model, model_dir / 'total_predictor.joblib')
    joblib.dump(win_prob_model, model_dir / 'win_prob_predictor.joblib')
    joblib.dump(win_prob_calibrator, model_dir / 'win_prob_calibrator.joblib')

    # Save metadata
    with open(model_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Models saved to {model_dir}")


def main():
    print("=" * 70)
    print("TRAINING ML MODELS FOR GAME LINE PREDICTIONS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    print("\n1. Loading game data with features...")
    df = load_game_features()

    if len(df) < 50:
        print(f"ERROR: Only {len(df)} games. Need at least 50 for training.")
        return

    # Create features
    print("\n2. Creating ML features...")
    features = create_game_features(df)
    print(f"  Created {len(features.columns)} features")

    # Train models
    print("\n3. Training models...")

    spread_model, spread_scores, spread_importance = train_spread_model(df, features)
    total_model, total_scores, total_importance = train_total_model(df, features)
    win_prob_model, win_prob_calibrator, win_prob_scores = train_win_prob_model(df, features)

    # Save models
    print("\n4. Saving models...")
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df),
        'seasons': sorted(df['season'].unique().tolist()),
        'features': list(features.columns),
        'spread_mae': np.mean([s['mae'] for s in spread_scores]),
        'total_mae': np.mean([s['mae'] for s in total_scores]),
        'win_prob_brier': np.mean([s['brier'] for s in win_prob_scores]),
        'spread_top_features': spread_importance.head(5).to_dict('records'),
        'total_top_features': total_importance.head(5).to_dict('records'),
    }

    save_models(spread_model, total_model, win_prob_model, win_prob_calibrator, metadata)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Spread MAE: {metadata['spread_mae']:.2f} points")
    print(f"Total MAE: {metadata['total_mae']:.2f} points")
    print(f"Win Prob Brier: {metadata['win_prob_brier']:.4f}")
    print(f"\nModels saved to: data/models/game_lines/")
    print(f"  - spread_predictor.joblib")
    print(f"  - total_predictor.joblib")
    print(f"  - win_prob_predictor.joblib")
    print(f"  - win_prob_calibrator.joblib")


if __name__ == '__main__':
    main()
