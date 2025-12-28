#!/usr/bin/env python3
"""
TD Poisson Model - Walk-Forward Validation

Validates the Poisson regression model for QB pass TDs.
Unlike TD Enhanced (binary: did player score TD?), this predicts:
- Expected TD count (lambda parameter of Poisson distribution)
- P(over 1.5) = P(TDs >= 2) for betting

All features use shift(1) or historical data to prevent leakage.

Usage:
    python scripts/backtest/walk_forward_td_poisson.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE COLUMNS FOR QB PASS TDS
# ============================================================================

FEATURE_COLS = [
    'trailing_passing_tds',     # Historical TD rate
    'trailing_passing_yards',   # Volume indicator
    'trailing_passing_attempts',  # Attempts per game
    'trailing_completion_pct',  # QB efficiency
    'vegas_total',              # Game environment
    'vegas_spread',             # Game script
    'opp_pass_tds_allowed',     # Opponent RZ pass TD rate allowed
]


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    """Load all required data sources."""
    logger.info("Loading data...")

    # Weekly stats
    stats = pd.read_parquet(DATA_DIR / 'nflverse' / 'weekly_stats.parquet')
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    # Filter to QBs only
    stats = stats[stats['position'] == 'QB'].copy()

    # Ensure we have pass TD column
    if 'passing_tds' not in stats.columns:
        logger.error("No passing_tds column in stats")
        return None, None

    logger.info(f"  QB Stats: {len(stats):,} rows")

    # PBP for opponent defense
    pbp = pd.read_parquet(DATA_DIR / 'nflverse' / 'pbp.parquet')
    pbp['global_week'] = (pbp['season'] - 2023) * 18 + pbp['week']

    logger.info(f"  PBP: {len(pbp):,} rows")

    return stats, pbp


def compute_opponent_pass_td_defense(pbp: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """
    Compute opponent defensive features for pass TDs using only historical data.
    """
    pbp = pbp[pbp['global_week'] < max_global_week].copy()

    if len(pbp) == 0:
        return pd.DataFrame()

    # Games per defense
    def_games = pbp.groupby('defteam')['game_id'].nunique().reset_index()
    def_games.columns = ['team', 'games']

    # Pass TDs allowed
    tds_allowed = pbp.groupby('defteam').agg({
        'pass_touchdown': 'sum',
    }).reset_index()
    tds_allowed.columns = ['team', 'pass_tds_allowed_total']

    # Merge
    defense = def_games.merge(tds_allowed, on='team', how='left')
    defense['opp_pass_tds_allowed'] = defense['pass_tds_allowed_total'] / defense['games']

    return defense[['team', 'opp_pass_tds_allowed']]


def compute_trailing_stats(stats: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """
    Compute trailing QB stats using only data before max_global_week.
    """
    hist = stats[stats['global_week'] < max_global_week].copy()
    hist = hist.sort_values(['player_norm', 'season', 'week'])

    # EWMA with shift(1) to prevent leakage
    ewma_span = 4

    # Trailing pass TDs
    hist['trailing_passing_tds'] = hist.groupby('player_norm')['passing_tds'].transform(
        lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean()
    )

    # Trailing pass yards
    hist['trailing_passing_yards'] = hist.groupby('player_norm')['passing_yards'].transform(
        lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean()
    )

    # Trailing attempts (use 'attempts' column from NFLverse)
    if 'attempts' in hist.columns:
        hist['trailing_passing_attempts'] = hist.groupby('player_norm')['attempts'].transform(
            lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean()
        )
    else:
        hist['trailing_passing_attempts'] = 30.0  # Default

    # Trailing completion percentage
    if 'completions' in hist.columns and 'attempts' in hist.columns:
        hist['completion_pct'] = hist['completions'] / hist['attempts'].clip(lower=1)
        hist['trailing_completion_pct'] = hist.groupby('player_norm')['completion_pct'].transform(
            lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean()
        )
    else:
        hist['trailing_completion_pct'] = 0.65  # Default

    # Get latest trailing value per player
    latest = hist.sort_values('global_week').groupby('player_norm').last().reset_index()

    return latest[['player_norm', 'trailing_passing_tds', 'trailing_passing_yards',
                   'trailing_passing_attempts', 'trailing_completion_pct']]


def prepare_features(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
    test_global_week: int
) -> pd.DataFrame:
    """
    Prepare all features for test week using only historical data.
    """
    # Get test week data
    test_data = stats[stats['global_week'] == test_global_week].copy()

    if len(test_data) == 0:
        return pd.DataFrame()

    # 1. Trailing stats
    trailing = compute_trailing_stats(stats, test_global_week)
    test_data = test_data.merge(trailing, on='player_norm', how='left')

    # 2. Opponent defense
    opp_defense = compute_opponent_pass_td_defense(pbp, test_global_week)
    if len(opp_defense) > 0 and 'opponent_team' in test_data.columns:
        test_data = test_data.merge(
            opp_defense,
            left_on='opponent_team',
            right_on='team',
            how='left'
        )

    # 3. Vegas features (if available)
    if 'vegas_total' not in test_data.columns:
        test_data['vegas_total'] = 45.0
    if 'vegas_spread' not in test_data.columns:
        test_data['vegas_spread'] = 0.0

    # Fill NaN with sensible defaults
    fill_values = {
        'trailing_passing_tds': 1.8,    # League avg ~1.8 pass TDs/game
        'trailing_passing_yards': 230.0,
        'trailing_passing_attempts': 32.0,
        'trailing_completion_pct': 0.65,
        'vegas_total': 45.0,
        'vegas_spread': 0.0,
        'opp_pass_tds_allowed': 1.5,
    }

    for col, val in fill_values.items():
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(val)
        else:
            test_data[col] = val

    return test_data


def prepare_training_data(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
    max_global_week: int
) -> pd.DataFrame:
    """
    Prepare training data with all features.
    Uses data from global_week < max_global_week - 1 (1 week gap).
    """
    # Training weeks
    train_weeks = sorted(stats[stats['global_week'] < max_global_week - 1]['global_week'].unique())

    if len(train_weeks) < 3:
        return pd.DataFrame()

    all_train = []

    for week in train_weeks[-20:]:  # Use last 20 weeks for efficiency
        week_features = prepare_features(stats, pbp, week)
        if len(week_features) > 0:
            all_train.append(week_features)

    if not all_train:
        return pd.DataFrame()

    train_df = pd.concat(all_train, ignore_index=True)

    return train_df


# ============================================================================
# MODEL TRAINING AND PREDICTION
# ============================================================================

def train_poisson_model(train_df: pd.DataFrame):
    """
    Train Poisson regression model to predict expected pass TDs.
    Returns model and scaler.
    """
    # Prepare features
    available_features = [f for f in FEATURE_COLS if f in train_df.columns]

    if len(available_features) < 3:
        logger.warning(f"Only {len(available_features)} features available")
        return None, None, available_features

    X = train_df[available_features].fillna(0)
    y = train_df['passing_tds'].fillna(0).astype(int).clip(lower=0)

    if len(X) < 50:
        return None, None, available_features

    # Scale features for Poisson regression stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Poisson regression
    model = PoissonRegressor(alpha=0.1, max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler, available_features


def predict_td_probabilities(
    model,
    scaler,
    features: list,
    test_df: pd.DataFrame,
    line: float = 1.5
) -> dict:
    """
    Predict expected TDs and P(over line) using Poisson distribution.

    For line=1.5, P(over) = P(TDs >= 2) = 1 - P(TDs <= 1)
    """
    X = test_df[features].fillna(0)
    X_scaled = scaler.transform(X)

    # Predict expected TD count (lambda)
    expected_tds = model.predict(X_scaled)

    # P(over 1.5) = P(TDs >= 2) = 1 - P(TDs <= 1)
    threshold = int(np.floor(line))
    p_over = 1 - poisson.cdf(threshold, expected_tds)
    p_under = poisson.cdf(threshold, expected_tds)

    return {
        'expected_tds': expected_tds,
        'p_over': p_over,
        'p_under': p_under,
    }


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward():
    """Run walk-forward validation for TD Poisson model."""
    logger.info("=" * 70)
    logger.info("TD POISSON MODEL - WALK-FORWARD VALIDATION (QB PASS TDs)")
    logger.info("=" * 70)

    # Load data
    stats, pbp = load_data()

    if stats is None:
        return

    # Get test weeks (start from week 24 = 2024 week 6, need history)
    all_weeks = sorted(stats['global_week'].unique())
    test_weeks = [w for w in all_weeks if w >= 24]  # 2024 week 6+

    logger.info(f"Testing {len(test_weeks)} weeks")

    all_results = []

    for test_week in test_weeks:
        # Prepare training data (weeks < test_week - 1)
        train_df = prepare_training_data(stats, pbp, test_week)

        if len(train_df) < 50:
            continue

        # Train model
        model, scaler, features = train_poisson_model(train_df)

        if model is None:
            continue

        # Prepare test data
        test_df = prepare_features(stats, pbp, test_week)

        if len(test_df) == 0:
            continue

        # Predict
        preds = predict_td_probabilities(model, scaler, features, test_df, line=1.5)
        test_df['expected_tds'] = preds['expected_tds']
        test_df['p_over'] = preds['p_over']
        test_df['p_under'] = preds['p_under']

        # Record results
        season = 2024 if test_week <= 36 else 2025
        week_num = test_week - 18 if test_week <= 36 else test_week - 36

        for idx, row in test_df.iterrows():
            actual_tds = row['passing_tds']
            over_hit = 1 if actual_tds >= 2 else 0  # Over 1.5 means >= 2
            under_hit = 1 if actual_tds <= 1 else 0  # Under 1.5 means <= 1

            all_results.append({
                'season': season,
                'week': week_num,
                'global_week': test_week,
                'player': row.get('player_display_name', row['player_norm']),
                'team': row.get('recent_team', ''),
                'opponent': row.get('opponent_team', ''),
                'line': 1.5,
                'trailing_tds': row.get('trailing_passing_tds', 0),
                'expected_tds': row['expected_tds'],
                'p_over': row['p_over'],
                'p_under': row['p_under'],
                'actual_tds': actual_tds,
                'over_hit': over_hit,
                'under_hit': under_hit,
            })

    if not all_results:
        logger.error("No results generated")
        return

    results_df = pd.DataFrame(all_results)

    # Save results
    output_path = DATA_DIR / 'backtest' / 'walk_forward_td_poisson.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(results_df):,} predictions to {output_path}")

    # ========================================================================
    # ANALYSIS - OVER BETS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("OVER 1.5 PASS TDs - BY CONFIDENCE THRESHOLD")
    logger.info("=" * 70)
    logger.info(f"{'Threshold':>10} {'N':>8} {'Hits':>8} {'WR':>10} {'ROI':>10}")
    logger.info("-" * 70)

    # Base rate
    base_rate = results_df['over_hit'].mean()
    logger.info(f"{'Base rate':>10} {len(results_df):>8} {results_df['over_hit'].sum():>8} {base_rate*100:>9.1f}% {'N/A':>10}")

    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65]:
        subset = results_df[results_df['p_over'] >= thresh]
        if len(subset) >= 10:
            hits = subset['over_hit'].sum()
            n = len(subset)
            wr = hits / n
            # ROI at -110 odds (0.909 payout)
            roi = (wr * 0.909 - (1 - wr)) * 100
            logger.info(f"{thresh*100:>9.0f}% {n:>8} {hits:>8} {wr*100:>9.1f}% {roi:>+9.1f}%")

    # ========================================================================
    # ANALYSIS - UNDER BETS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("UNDER 1.5 PASS TDs - BY CONFIDENCE THRESHOLD")
    logger.info("=" * 70)
    logger.info(f"{'Threshold':>10} {'N':>8} {'Hits':>8} {'WR':>10} {'ROI':>10}")
    logger.info("-" * 70)

    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65]:
        subset = results_df[results_df['p_under'] >= thresh]
        if len(subset) >= 10:
            hits = subset['under_hit'].sum()
            n = len(subset)
            wr = hits / n
            roi = (wr * 0.909 - (1 - wr)) * 100
            logger.info(f"{thresh*100:>9.0f}% {n:>8} {hits:>8} {wr*100:>9.1f}% {roi:>+9.1f}%")

    # ========================================================================
    # CALIBRATION CHECK
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION CHECK (Predicted vs Actual)")
    logger.info("=" * 70)

    # Bin by predicted probability
    bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for low, high in bins:
        subset = results_df[(results_df['p_over'] >= low) & (results_df['p_over'] < high)]
        if len(subset) >= 10:
            predicted = subset['p_over'].mean()
            actual = subset['over_hit'].mean()
            diff = actual - predicted
            logger.info(f"  P(over) {low:.0%}-{high:.0%}: N={len(subset):>4}, "
                       f"Pred={predicted:.1%}, Actual={actual:.1%}, Diff={diff:+.1%}")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    if model is not None:
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE COEFFICIENTS (Poisson Regression)")
        logger.info("=" * 70)

        coefs = dict(zip(features, model.coef_))
        for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            logger.info(f"  {feat:<30}: {coef:+.4f}")


if __name__ == '__main__':
    run_walk_forward()
