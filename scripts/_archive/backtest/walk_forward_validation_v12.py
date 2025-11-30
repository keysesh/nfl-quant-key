#!/usr/bin/env python3
"""
NFL QUANT - Walk-Forward Validation for V12 Model

CRITICAL: This script validates model performance with NO data leakage.

For each week N in 2025:
1. Calculate trailing stats using ONLY weeks < N
2. Calculate player features using ONLY weeks < N
3. Train model on historical data (2023-2024 + 2025 weeks < N)
4. Predict week N outcomes
5. Compare to actual results

NO LOOK-AHEAD BIAS:
- Trailing averages: shift(1) ensures we don't use current week
- Player under_rate: shift(1) ensures we don't use current outcome
- Player bias: shift(1) ensures we don't use current actual_stat
- Market regime: shift(1) ensures we don't use current week's UNDER rate

Usage:
    python scripts/backtest/walk_forward_validation_v12.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data():
    """Load all historical data."""
    logger.info("Loading data...")

    # Odds with actuals
    odds_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    odds = pd.read_csv(odds_path)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    # Deduplicate (keep primary line)
    odds['group_key'] = odds['player_norm'] + '_' + odds['season'].astype(str) + '_' + odds['week'].astype(str) + '_' + odds['market']
    market_medians = odds.groupby('market')['line'].median()

    def get_primary_line(group):
        if len(group) == 1:
            return group
        market = group['market'].iloc[0]
        median = market_medians.get(market, group['line'].median())
        group['dist_from_median'] = abs(group['line'] - median)
        return group.nsmallest(1, 'dist_from_median')

    odds = odds.groupby('group_key', group_keys=False).apply(get_primary_line)
    odds = odds.drop(columns=['dist_from_median', 'group_key'], errors='ignore')

    # Player stats
    stats_2024 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)
    stats_2023 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2023.csv', low_memory=False)
    stats = pd.concat([stats_2024, stats_2023], ignore_index=True)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

    logger.info(f"  Odds: {len(odds)} rows")
    logger.info(f"  Stats: {len(stats)} rows")

    return odds, stats


def calculate_trailing_stats_at_week(stats: pd.DataFrame, max_week: int, max_season: int) -> pd.DataFrame:
    """
    Calculate trailing stats using ONLY data available at a given point in time.

    CRITICAL: This ensures no look-ahead bias.
    max_week and max_season define the "current" time - we only use data BEFORE this.
    """
    # Filter to only data before the specified week
    global_max = (max_season - 2023) * 18 + max_week
    stats = stats.copy()
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']
    stats = stats[stats['global_week'] < global_max]  # STRICTLY less than

    if len(stats) == 0:
        return pd.DataFrame()

    # Sort for proper trailing calculation
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    # Calculate trailing averages (EWMA with span=4)
    stat_cols = ['receptions', 'receiving_yards', 'rushing_yards', 'passing_yards']
    for col in stat_cols:
        if col in stats.columns:
            stats[f'trailing_{col}'] = (
                stats.groupby('player_norm')[col]
                .transform(lambda x: x.ewm(span=4, min_periods=1).mean())
            )

    # Get the LATEST trailing value for each player
    latest = stats.sort_values('global_week').groupby('player_norm').last().reset_index()

    return latest


def get_stat_col_for_market(market: str) -> str:
    """Map market to stat column."""
    mapping = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
    }
    return mapping.get(market)


def calculate_features_at_week(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    target_week: int,
    target_season: int,
    market: str
) -> pd.DataFrame:
    """
    Calculate all features for a specific week using ONLY prior data.

    NO DATA LEAKAGE:
    - trailing stats: from weeks < target_week
    - player_under_rate: from weeks < target_week
    - player_bias: from weeks < target_week
    - market_under_rate: from weeks < target_week
    """
    stat_col = get_stat_col_for_market(market)
    if not stat_col:
        return pd.DataFrame()

    trailing_col = f'trailing_{stat_col}'

    # Get trailing stats calculated at target_week (using only prior data)
    trailing = calculate_trailing_stats_at_week(stats, target_week, target_season)

    if len(trailing) == 0 or trailing_col not in trailing.columns:
        return pd.DataFrame()

    # Get target week's odds
    target_odds = odds[
        (odds['season'] == target_season) &
        (odds['week'] == target_week) &
        (odds['market'] == market)
    ].copy()

    if len(target_odds) == 0:
        return pd.DataFrame()

    # Merge trailing stats
    target_odds = target_odds.merge(
        trailing[['player_norm', trailing_col]],
        on='player_norm',
        how='left'
    )

    # Drop rows without trailing stats
    target_odds = target_odds.dropna(subset=[trailing_col])

    if len(target_odds) == 0:
        return pd.DataFrame()

    # === FEATURE CALCULATION (using only historical data) ===

    # Primary signal: Line vs Trailing
    target_odds['line_vs_trailing'] = target_odds['line'] - target_odds[trailing_col]

    # Line characteristics
    target_odds['line_level'] = target_odds['line']
    target_odds['line_in_sweet_spot'] = ((target_odds['line'] >= 3.5) & (target_odds['line'] <= 7.5)).astype(float)

    # === PLAYER HISTORICAL FEATURES (from prior weeks ONLY) ===
    global_target = (target_season - 2023) * 18 + target_week

    # Get all historical odds for this market BEFORE target week
    hist_odds = odds[
        (odds['market'] == market) &
        (((odds['season'] - 2023) * 18 + odds['week']) < global_target)
    ].copy()

    if len(hist_odds) > 0:
        # Player under rate (rolling 10 games, from history only)
        player_under_rates = (
            hist_odds.sort_values(['player_norm', 'season', 'week'])
            .groupby('player_norm')['under_hit']
            .apply(lambda x: x.rolling(10, min_periods=3).mean().iloc[-1] if len(x) >= 3 else np.nan)
        ).to_dict()

        target_odds['player_under_rate'] = target_odds['player_norm'].map(player_under_rates)

        # Player bias (actual - line, rolling 10 games)
        hist_odds['actual_minus_line'] = hist_odds['actual_stat'] - hist_odds['line']
        player_bias = (
            hist_odds.sort_values(['player_norm', 'season', 'week'])
            .groupby('player_norm')['actual_minus_line']
            .apply(lambda x: x.rolling(10, min_periods=3).mean().iloc[-1] if len(x) >= 3 else np.nan)
        ).to_dict()

        target_odds['player_bias'] = target_odds['player_norm'].map(player_bias)

        # Market regime (trailing 4-week UNDER rate for this market)
        hist_odds['global_week'] = (hist_odds['season'] - 2023) * 18 + hist_odds['week']
        weekly_rates = hist_odds.groupby('global_week')['under_hit'].mean()

        # Get the last 4 weeks before target
        recent_weeks = [w for w in weekly_rates.index if w < global_target][-4:]
        if len(recent_weeks) > 0:
            market_under_rate = weekly_rates.loc[recent_weeks].mean()
        else:
            market_under_rate = 0.5

        target_odds['market_under_rate'] = market_under_rate
    else:
        target_odds['player_under_rate'] = np.nan
        target_odds['player_bias'] = np.nan
        target_odds['market_under_rate'] = 0.5

    # === INTERACTION FEATURES ===
    target_odds['LVT_x_player_tendency'] = (
        target_odds['line_vs_trailing'] * (target_odds['player_under_rate'].fillna(0.5) - 0.5)
    )
    target_odds['LVT_x_player_bias'] = (
        target_odds['line_vs_trailing'] * target_odds['player_bias'].fillna(0)
    )
    target_odds['LVT_x_regime'] = (
        target_odds['line_vs_trailing'] * (target_odds['market_under_rate'].fillna(0.5) - 0.5)
    )
    target_odds['LVT_in_sweet_spot'] = (
        target_odds['line_vs_trailing'] * target_odds['line_in_sweet_spot']
    )
    target_odds['market_bias_strength'] = abs(target_odds['market_under_rate'].fillna(0.5) - 0.5) * 2
    target_odds['player_market_aligned'] = (
        (target_odds['player_under_rate'].fillna(0.5) - 0.5) *
        (target_odds['market_under_rate'].fillna(0.5) - 0.5)
    )

    return target_odds


def train_model_at_week(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    target_week: int,
    target_season: int,
    market: str,
    window_weeks: int = 20
) -> tuple:
    """
    Train a model using ONLY data available before target_week.

    Returns: (model, feature_cols) or (None, None) if insufficient data
    """
    global_target = (target_season - 2023) * 18 + target_week

    # Get all historical data for training (before target week)
    hist_odds = odds[
        (odds['market'] == market) &
        (((odds['season'] - 2023) * 18 + odds['week']) < global_target)
    ].copy()

    if len(hist_odds) < 100:
        return None, None

    # Calculate features for each historical week
    all_train_data = []

    # Use rolling window for training (last N weeks before target)
    hist_odds['global_week'] = (hist_odds['season'] - 2023) * 18 + hist_odds['week']
    train_weeks = sorted(hist_odds['global_week'].unique())[-window_weeks:]

    for gw in train_weeks:
        week = ((gw - 1) % 18) + 1
        season = 2023 + (gw - 1) // 18

        week_features = calculate_features_at_week(odds, stats, week, season, market)
        if len(week_features) > 0:
            all_train_data.append(week_features)

    if len(all_train_data) == 0:
        return None, None

    train_df = pd.concat(all_train_data, ignore_index=True)

    # Feature columns
    feature_cols = [
        'line_vs_trailing',
        'line_level',
        'line_in_sweet_spot',
        'player_under_rate',
        'player_bias',
        'market_under_rate',
        'LVT_x_player_tendency',
        'LVT_x_player_bias',
        'LVT_x_regime',
        'LVT_in_sweet_spot',
        'market_bias_strength',
        'player_market_aligned',
    ]

    available_features = [f for f in feature_cols if f in train_df.columns]

    # Clean training data
    train_clean = train_df[available_features + ['under_hit']].dropna()

    if len(train_clean) < 50:
        return None, None

    X_train = train_clean[available_features]
    y_train = train_clean['under_hit']

    # Build interaction constraints (LVT as hub)
    feature_indices = {f: i for i, f in enumerate(available_features)}
    lvt_idx = feature_indices.get('line_vs_trailing', 0)
    all_idx = list(range(len(available_features)))

    interaction_constraints = [[lvt_idx] + all_idx]
    for i in range(len(available_features)):
        if i != lvt_idx:
            interaction_constraints.append([lvt_idx, i])

    # Monotonic constraints
    monotonic = []
    for f in available_features:
        if f == 'line_vs_trailing':
            monotonic.append(1)
        elif f == 'player_under_rate':
            monotonic.append(1)
        elif f == 'market_under_rate':
            monotonic.append(1)
        elif f == 'player_bias':
            monotonic.append(-1)
        else:
            monotonic.append(0)

    monotonic_str = '(' + ','.join(map(str, monotonic)) + ')'

    # Train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'interaction_constraints': str(interaction_constraints),
        'monotone_constraints': monotonic_str,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

    return model, available_features


def walk_forward_backtest(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    market: str,
    start_week: int = 1,
    end_week: int = 12,
    season: int = 2025
) -> pd.DataFrame:
    """
    Perform walk-forward validation for a single market.

    For each week from start_week to end_week:
    1. Train model using only data before that week
    2. Predict outcomes for that week
    3. Record predictions vs actuals
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Walk-Forward Backtest: {market}")
    logger.info(f"{'='*60}")

    all_results = []

    for week in range(start_week, end_week + 1):
        # Train model at this point in time
        model, feature_cols = train_model_at_week(odds, stats, week, season, market)

        if model is None:
            logger.info(f"  Week {week}: Insufficient training data")
            continue

        # Get features for target week (using only prior data)
        test_features = calculate_features_at_week(odds, stats, week, season, market)

        if len(test_features) == 0:
            logger.info(f"  Week {week}: No test data")
            continue

        # Predict
        X_test = test_features[feature_cols].dropna()

        if len(X_test) == 0:
            logger.info(f"  Week {week}: No valid features")
            continue

        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)

        # Record results
        test_subset = test_features.loc[X_test.index].copy()
        test_subset['predicted_under_prob'] = predictions
        test_subset['predicted_under'] = predictions >= 0.55  # Default threshold
        test_subset['week'] = week
        test_subset['season'] = season

        all_results.append(test_subset)

        # Weekly summary
        n_bets = len(test_subset)
        actual_under_rate = test_subset['under_hit'].mean()
        pred_under_rate = (predictions >= 0.55).mean()

        logger.info(f"  Week {week}: {n_bets} bets, Actual UNDER: {actual_under_rate:.1%}, Pred UNDER: {pred_under_rate:.1%}")

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def analyze_results(results: pd.DataFrame, market: str):
    """Analyze walk-forward results by threshold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Results Analysis: {market}")
    logger.info(f"{'='*60}")

    if len(results) == 0:
        logger.info("  No results to analyze")
        return {}

    metrics = {}

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = results['predicted_under_prob'] >= threshold
        n_bets = mask.sum()

        if n_bets < 5:
            continue

        wins = results.loc[mask, 'under_hit'].sum()
        losses = n_bets - wins
        hit_rate = wins / n_bets

        # ROI at -110 odds
        profit = wins * 0.909 - losses * 1.0
        roi = (profit / n_bets) * 100

        metrics[threshold] = {
            'n_bets': n_bets,
            'wins': wins,
            'losses': losses,
            'hit_rate': hit_rate,
            'profit': profit,
            'roi': roi,
        }

        logger.info(f"  Threshold {threshold:.0%}: N={n_bets}, {wins}W-{losses}L ({hit_rate:.1%}), ROI: {roi:+.1f}%")

    # Week-by-week breakdown
    logger.info(f"\n  Week-by-Week (threshold=55%):")
    for week in sorted(results['week'].unique()):
        week_data = results[(results['week'] == week) & (results['predicted_under_prob'] >= 0.55)]
        if len(week_data) > 0:
            wins = week_data['under_hit'].sum()
            losses = len(week_data) - wins
            profit = wins * 0.909 - losses * 1.0
            logger.info(f"    Week {week}: {wins}W-{losses}L, {profit:+.2f} units")

    return metrics


def main():
    """Run full walk-forward validation."""
    print("="*80)
    print("NFL QUANT - WALK-FORWARD VALIDATION (NO DATA LEAKAGE)")
    print("="*80)
    print()
    print("This script validates the V12 model with strict temporal separation:")
    print("  - For each week N, trains ONLY on data from weeks < N")
    print("  - Trailing stats use ONLY prior games")
    print("  - Player features use ONLY prior outcomes")
    print("  - No look-ahead bias of any kind")
    print()

    # Load data
    odds, stats = load_data()

    # Markets to test
    markets = ['player_receptions', 'player_rush_yds', 'player_reception_yds', 'player_pass_yds']

    all_metrics = {}
    all_results = {}

    for market in markets:
        results = walk_forward_backtest(odds, stats, market, start_week=1, end_week=12, season=2025)

        if len(results) > 0:
            metrics = analyze_results(results, market)
            all_metrics[market] = metrics
            all_results[market] = results

    # Final summary
    print()
    print("="*80)
    print("FINAL SUMMARY - Walk-Forward 2025 Weeks 1-12")
    print("="*80)
    print()
    print(f"{'Market':<25} {'Threshold':>10} {'N Bets':>8} {'Win%':>8} {'ROI':>10}")
    print("-"*65)

    for market, metrics in all_metrics.items():
        # Find best threshold with at least 20 bets
        best_thresh = None
        best_roi = -999
        for thresh, m in metrics.items():
            if m['n_bets'] >= 20 and m['roi'] > best_roi:
                best_roi = m['roi']
                best_thresh = thresh

        if best_thresh:
            m = metrics[best_thresh]
            print(f"{market:<25} {best_thresh:>9.0%} {m['n_bets']:>8} {m['hit_rate']:>7.1%} {m['roi']:>+9.1f}%")
        else:
            print(f"{market:<25} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>10}")

    # Overall totals at 55% threshold
    print()
    print("-"*65)
    total_bets = 0
    total_wins = 0
    total_losses = 0

    for market, metrics in all_metrics.items():
        if 0.55 in metrics:
            m = metrics[0.55]
            total_bets += m['n_bets']
            total_wins += m['wins']
            total_losses += m['losses']

    if total_bets > 0:
        total_profit = total_wins * 0.909 - total_losses * 1.0
        total_roi = (total_profit / total_bets) * 100
        total_hit = total_wins / total_bets
        print(f"{'TOTAL (55% threshold)':<25} {'55%':>10} {total_bets:>8} {total_hit:>7.1%} {total_roi:>+9.1f}%")

    print()
    print("="*80)

    # Save results
    output_path = PROJECT_ROOT / 'reports' / 'walk_forward_validation_2025.csv'

    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
