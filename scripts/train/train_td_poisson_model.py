#!/usr/bin/env python3
"""
TD Poisson Model Training Script

Trains Poisson regression models for TD props:
- player_pass_tds: QB passing touchdowns (count: 0, 1, 2, 3, 4+)
- player_anytime_td: Binary (0 or 1) - uses logistic regression

Key Features:
- Uses historical odds with actuals for training
- Walk-forward validation (no leakage)
- Red zone TD rate from PBP data
- Game environment features (total, spread)

Usage:
    python scripts/train/train_td_poisson_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_historical_data() -> pd.DataFrame:
    """Load historical odds and match with actuals from weekly_stats."""
    logger.info("Loading historical TD prop data...")

    # Load historical odds
    hist_path = PROJECT_ROOT / 'data' / 'historical' / 'historical_odds_2024_2025_complete.csv'
    if not hist_path.exists():
        hist_path = PROJECT_ROOT / 'data' / 'historical' / 'historical_odds_2024_complete.csv'

    if not hist_path.exists():
        logger.error(f"Historical data not found at {hist_path}")
        return pd.DataFrame()

    odds = pd.read_csv(hist_path, low_memory=False)

    # Filter to TD markets
    td_markets = ['player_pass_tds', 'player_anytime_td']
    odds = odds[odds['market'].isin(td_markets)].copy()
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    logger.info(f"  Loaded {len(odds):,} TD props from odds")
    logger.info(f"  Markets: {odds['market'].value_counts().to_dict()}")

    # Load weekly stats to get actuals
    stats = load_player_stats()

    # Create actuals lookup: for player_pass_tds, use passing_tds
    # For player_anytime_td, compute (passing_tds + rushing_tds + receiving_tds)
    stats_summary = stats.groupby(['player_norm', 'season', 'week']).agg({
        'passing_tds': 'sum',
        'rushing_tds': 'sum',
        'receiving_tds': 'sum',
        'position': 'first'
    }).reset_index()

    stats_summary['total_tds'] = (
        stats_summary['passing_tds'].fillna(0) +
        stats_summary['rushing_tds'].fillna(0) +
        stats_summary['receiving_tds'].fillna(0)
    )

    # Merge actuals into odds
    odds = odds.merge(
        stats_summary[['player_norm', 'season', 'week', 'passing_tds', 'total_tds', 'position']],
        on=['player_norm', 'season', 'week'],
        how='left'
    )

    # Set actual based on market type
    odds['actual'] = np.where(
        odds['market'] == 'player_pass_tds',
        odds['passing_tds'],
        odds['total_tds']  # For anytime TD
    )

    # Filter to rows with actuals
    before = len(odds)
    odds = odds.dropna(subset=['actual'])
    after = len(odds)

    logger.info(f"  Matched {after:,} of {before:,} props with actuals")

    # Add global week for ordering
    odds['global_week'] = (odds['season'] - 2023) * 18 + odds['week']

    return odds


def load_player_stats() -> pd.DataFrame:
    """Load player stats for feature extraction."""
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    if stats_path.exists():
        stats = pd.read_parquet(stats_path)
    else:
        # Fallback to CSV
        stats = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv')

    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    return stats


def compute_qb_rz_td_rates(pbp_path: Path, season: int) -> Dict[str, float]:
    """
    Compute red zone TD rate for each QB from PBP data.

    Returns dict: {player_name: rz_td_rate}
    """
    if not pbp_path.exists():
        return {}

    pbp = pd.read_parquet(pbp_path)

    # Filter to red zone passes
    rz_passes = pbp[
        (pbp['yardline_100'] <= 20) &
        (pbp['play_type'] == 'pass') &
        (pbp['season'] == season)
    ].copy()

    if len(rz_passes) == 0:
        return {}

    # Calculate QB RZ TD rate
    qb_rz = rz_passes.groupby('passer_player_name').agg({
        'pass_touchdown': 'sum',
        'play_id': 'count'
    }).rename(columns={'play_id': 'rz_attempts'})

    qb_rz['rz_td_rate'] = qb_rz['pass_touchdown'] / qb_rz['rz_attempts'].clip(lower=1)

    return qb_rz['rz_td_rate'].to_dict()


def compute_trailing_tds(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing TD stats for QBs."""
    qb_stats = stats[stats['position'] == 'QB'].copy()
    qb_stats = qb_stats.sort_values(['player_norm', 'global_week'])

    # EWMA of passing TDs
    qb_stats['trailing_pass_tds'] = qb_stats.groupby('player_norm')['passing_tds'].transform(
        lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
    )

    # Trailing pass attempts
    qb_stats['trailing_pass_attempts'] = qb_stats.groupby('player_norm')['attempts'].transform(
        lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
    )

    # TD rate
    qb_stats['trailing_td_rate'] = qb_stats['trailing_pass_tds'] / qb_stats['trailing_pass_attempts'].clip(lower=1)

    return qb_stats[['player_norm', 'season', 'week', 'trailing_pass_tds', 'trailing_pass_attempts', 'trailing_td_rate']]


def prepare_training_data(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    market: str
) -> pd.DataFrame:
    """Prepare training data with features for a specific market."""

    # Filter to market
    df = odds[odds['market'] == market].copy()

    if market == 'player_pass_tds':
        # Merge with trailing stats
        trailing = compute_trailing_tds(stats)
        df = df.merge(
            trailing,
            on=['player_norm', 'season', 'week'],
            how='left'
        )

        # Fill NaN with position averages
        df['trailing_pass_tds'] = df['trailing_pass_tds'].fillna(1.5)
        df['trailing_td_rate'] = df['trailing_td_rate'].fillna(0.05)

    # Add line features
    df['line'] = df['line'].fillna(1.5)

    # Compute target variable
    if 'actual' in df.columns:
        df['actual'] = df['actual'].fillna(0)
        df['under_hit'] = (df['actual'] < df['line']).astype(int)
    else:
        df['under_hit'] = np.nan

    # Drop rows without actuals
    df = df.dropna(subset=['under_hit'])

    return df


def train_pass_tds_model(df: pd.DataFrame) -> Dict:
    """
    Train Poisson regression for player_pass_tds.

    Returns model data dict.
    """
    logger.info("\n" + "="*60)
    logger.info("Training PLAYER_PASS_TDS Poisson Model")
    logger.info("="*60)

    # Feature columns
    feature_cols = ['trailing_pass_tds', 'trailing_td_rate', 'line']
    available_cols = [c for c in feature_cols if c in df.columns]

    if len(available_cols) < 2:
        logger.warning("Not enough features available")
        return None

    # Get weeks for walk-forward
    weeks = sorted(df['global_week'].unique())

    if len(weeks) < 5:
        logger.warning("Not enough weeks for validation")
        return None

    # Walk-forward validation
    logger.info(f"Walk-forward validation on {len(weeks)} weeks...")

    all_preds = []
    val_weeks = weeks[-10:]  # Last 10 weeks for validation

    for test_week in val_weeks:
        train = df[df['global_week'] < test_week - 1].copy()
        test = df[df['global_week'] == test_week].copy()

        if len(train) < 30 or len(test) == 0:
            continue

        # Prepare X, y
        X_train = train[available_cols].fillna(0)
        y_train = train['actual'].astype(int).clip(0, 6)  # Cap at 6 TDs
        X_test = test[available_cols].fillna(0)
        y_test = test['actual'].astype(int)

        # Train Poisson model
        model = PoissonRegressor(alpha=0.1, max_iter=1000)
        model.fit(X_train, y_train)

        # Predict lambda (expected TDs)
        predicted_lambda = model.predict(X_test)

        # Calculate P(UNDER line) using Poisson CDF
        for i, (idx, row) in enumerate(test.iterrows()):
            line = row['line']
            lamb = predicted_lambda[i]

            # P(X < line) = P(X <= floor(line - 0.5))
            # For line=1.5: P(X <= 1) = P(0) + P(1)
            p_under = scipy_stats.poisson.cdf(int(line - 0.5), lamb)

            all_preds.append({
                'global_week': test_week,
                'player_norm': row['player_norm'],
                'line': line,
                'actual': row['actual'],
                'under_hit': row['under_hit'],
                'predicted_lambda': lamb,
                'p_under': p_under
            })

    if len(all_preds) == 0:
        logger.warning("No predictions generated")
        return None

    preds_df = pd.DataFrame(all_preds)

    # Calculate validation metrics
    logger.info(f"\n=== VALIDATION RESULTS ===")
    logger.info(f"Total bets: {len(preds_df)}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        high_conf = preds_df[preds_df['p_under'] >= thresh]
        if len(high_conf) >= 5:
            hits = high_conf['under_hit'].sum()
            total = len(high_conf)
            hit_rate = hits / total
            roi = (hit_rate * 0.909) - (1 - hit_rate)
            logger.info(f"  UNDER @ {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

        # Also check OVER
        low_conf = preds_df[preds_df['p_under'] <= (1 - thresh)]
        if len(low_conf) >= 5:
            # OVER hits when actual >= line
            over_hits = (low_conf['actual'] >= low_conf['line']).sum()
            total = len(low_conf)
            hit_rate = over_hits / total
            roi = (hit_rate * 0.909) - (1 - hit_rate)
            logger.info(f"  OVER  @ {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

    # Train final model on all data
    logger.info("\nTraining final production model...")

    X_final = df[available_cols].fillna(0)
    y_final = df['actual'].astype(int).clip(0, 6)

    final_model = PoissonRegressor(alpha=0.1, max_iter=1000)
    final_model.fit(X_final, y_final)

    # Feature importance (coefficients)
    logger.info("\nFeature Coefficients:")
    for feat, coef in zip(available_cols, final_model.coef_):
        logger.info(f"  {feat}: {coef:.4f}")
    logger.info(f"  intercept: {final_model.intercept_:.4f}")

    return {
        'model': final_model,
        'features': available_cols,
        'scaler': None,  # Poisson doesn't need scaling
        'validation': preds_df,
        'model_type': 'poisson',
        'market': 'player_pass_tds',
    }


def train_anytime_td_model(df: pd.DataFrame) -> Dict:
    """
    Train Logistic Regression for player_anytime_td (binary).

    Returns model data dict.
    """
    logger.info("\n" + "="*60)
    logger.info("Training PLAYER_ANYTIME_TD Logistic Model")
    logger.info("="*60)

    # For anytime TD, the outcome is binary: 0 or 1+
    # The "line" is typically 0.5 (so actual >= 1 = OVER hits)
    df = df.copy()
    df['scored_td'] = (df['actual'] >= 1).astype(int)

    # Features: need position, usage, etc.
    # For now, use simple features available
    feature_cols = ['line']
    available_cols = [c for c in feature_cols if c in df.columns]

    if len(available_cols) < 1:
        logger.warning("Not enough features for anytime TD model")
        return None

    # Get weeks for walk-forward
    weeks = sorted(df['global_week'].unique())

    if len(weeks) < 5:
        logger.warning("Not enough weeks for validation")
        return None

    # Walk-forward validation
    logger.info(f"Walk-forward validation on {len(weeks)} weeks...")

    all_preds = []
    val_weeks = weeks[-10:]

    for test_week in val_weeks:
        train = df[df['global_week'] < test_week - 1].copy()
        test = df[df['global_week'] == test_week].copy()

        if len(train) < 30 or len(test) == 0:
            continue

        X_train = train[available_cols].fillna(0)
        y_train = train['scored_td']
        X_test = test[available_cols].fillna(0)

        # Train logistic model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Predict P(scores TD)
        probs = model.predict_proba(X_test)[:, 1]

        for i, (idx, row) in enumerate(test.iterrows()):
            all_preds.append({
                'global_week': test_week,
                'player_norm': row['player_norm'],
                'line': row['line'],
                'actual': row['actual'],
                'scored_td': row['scored_td'],
                'p_over': probs[i],  # P(scores at least 1 TD)
            })

    if len(all_preds) == 0:
        return None

    preds_df = pd.DataFrame(all_preds)

    # Validation metrics
    logger.info(f"\n=== VALIDATION RESULTS ===")
    logger.info(f"Total bets: {len(preds_df)}")

    for thresh in [0.50, 0.55, 0.60, 0.65]:
        high_conf = preds_df[preds_df['p_over'] >= thresh]
        if len(high_conf) >= 5:
            hits = high_conf['scored_td'].sum()
            total = len(high_conf)
            hit_rate = hits / total
            roi = (hit_rate * 0.909) - (1 - hit_rate)
            logger.info(f"  OVER  @ {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

    # Train final model
    X_final = df[available_cols].fillna(0)
    y_final = df['scored_td']

    final_model = LogisticRegression(random_state=42, max_iter=1000)
    final_model.fit(X_final, y_final)

    return {
        'model': final_model,
        'features': available_cols,
        'scaler': None,
        'validation': preds_df,
        'model_type': 'logistic',
        'market': 'player_anytime_td',
    }


def main():
    """Main training function."""
    print("="*80)
    print("TD POISSON MODEL TRAINING")
    print("="*80)

    # Load data
    odds = load_historical_data()
    if len(odds) == 0:
        logger.error("No historical data found")
        return

    stats = load_player_stats()

    # Prepare data for each market
    results = {}

    # Train player_pass_tds model
    pass_tds_df = prepare_training_data(odds, stats, 'player_pass_tds')
    if len(pass_tds_df) > 0:
        logger.info(f"\npass_tds data: {len(pass_tds_df)} rows")
        result = train_pass_tds_model(pass_tds_df)
        if result:
            results['player_pass_tds'] = result

    # Train player_anytime_td model
    anytime_df = prepare_training_data(odds, stats, 'player_anytime_td')
    if len(anytime_df) > 0:
        logger.info(f"\nanytime_td data: {len(anytime_df)} rows")
        result = train_anytime_td_model(anytime_df)
        if result:
            results['player_anytime_td'] = result

    if not results:
        logger.error("No models trained successfully")
        return

    # Save combined model file
    model_data = {
        'models': {m: r['model'] for m, r in results.items()},
        'features': {m: r['features'] for m, r in results.items()},
        'model_types': {m: r['model_type'] for m, r in results.items()},
        'version': 'td_poisson_v1',
        'trained_date': datetime.now().isoformat(),
        'markets': list(results.keys()),
    }

    # Save to models directory
    model_path = PROJECT_ROOT / 'data' / 'models' / 'td_poisson_model.joblib'
    joblib.dump(model_data, model_path)
    logger.info(f"\nSaved to: {model_path}")

    # Summary
    print("\n" + "="*80)
    print("TD MODEL TRAINING COMPLETE")
    print("="*80)
    for market, result in results.items():
        val = result['validation']
        logger.info(f"\n{market}:")
        logger.info(f"  Training samples: {len(val)}")
        if 'under_hit' in val.columns:
            logger.info(f"  Base rate (UNDER): {val['under_hit'].mean():.1%}")


if __name__ == '__main__':
    main()
