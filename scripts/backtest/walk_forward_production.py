#!/usr/bin/env python3
"""
Production Walk-Forward Backtest

This script simulates the FULL production pipeline each week:
1. Train XGBoost classifiers per market (V30 market-specific features)
2. 80/20 train/calibration split with IsotonicRegression
3. Train LVT + Player Bias edge models
4. Apply production filters (A/B testable)
5. Generate recommendations and evaluate

Key Features:
- Walk-forward with 1-week gap (train on week < test_week - 1)
- Full model retraining each week (no shortcuts)
- A/B testing capability for filters
- 2024 + 2025 combined training data

Usage:
    # Full A/B test
    python scripts/backtest/walk_forward_production.py

    # No filters (baseline)
    python scripts/backtest/walk_forward_production.py --no-filters

    # With filters
    python scripts/backtest/walk_forward_production.py --with-filters

    # Specific week range
    python scripts/backtest/walk_forward_production.py --start-week 5 --end-week 17
"""

import pandas as pd
import numpy as np
import logging
import joblib
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import warnings
import time
import json
import argparse

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, BACKTEST_DIR, MODELS_DIR
from nfl_quant.features.batch_extractor import extract_features_batch, clear_caches
from nfl_quant.features.feature_defaults import safe_fillna
from nfl_quant.utils.player_names import normalize_player_name

# Import from centralized configs
from configs.model_config import (
    MODEL_VERSION,
    FEATURES,
    CLASSIFIER_MARKETS,
    MODEL_PARAMS,
    EWMA_SPAN,
    TRAILING_DEFLATION_FACTORS,
    DEFAULT_TRAILING_DEFLATION,
    MARKET_DIRECTION_CONSTRAINTS,
    get_market_features,
)
from configs.edge_config import EDGE_MARKETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all historical data (2024 + 2025)."""
    logger.info("Loading data...")

    # Odds with actuals - use enriched version
    enriched_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched data not found: {enriched_path}")

    odds = pd.read_csv(enriched_path)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    # V26: Exclude 2023 data - missing vegas_spread/vegas_total
    odds = odds[odds['season'] >= 2024].copy()

    # Add global week for temporal ordering
    odds['global_week'] = (odds['season'] - 2023) * 18 + odds['week']

    # Deduplicate (keep primary line)
    odds['group_key'] = (
        odds['player_norm'] + '_' +
        odds['season'].astype(str) + '_' +
        odds['week'].astype(str) + '_' +
        odds['market']
    )
    market_medians = odds.groupby('market')['line'].median()

    def get_primary_line(group):
        if len(group) == 1:
            return group
        market = group['market'].iloc[0]
        median = market_medians.get(market, group['line'].median())
        group = group.copy()
        group['dist_from_median'] = abs(group['line'] - median)
        return group.nsmallest(1, 'dist_from_median')

    odds = odds.groupby('group_key', group_keys=False).apply(get_primary_line)
    odds = odds.drop(columns=['dist_from_median', 'group_key'], errors='ignore')

    # Player stats
    stats_2024 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)

    # Normalize column names
    stats = stats_2024.copy()
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    logger.info(f"  Loaded {len(odds):,} odds rows, {len(stats):,} stats rows")

    return odds, stats


def prepare_data_with_trailing(odds: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Add trailing stats to odds data."""
    stats = stats.sort_values(['player_norm', 'season', 'week']).copy()

    # Calculate trailing stats
    stat_cols = [
        'receptions', 'receiving_yards', 'rushing_yards', 'passing_yards',
        'carries', 'completions', 'attempts', 'passing_tds', 'rushing_tds', 'receiving_tds',
    ]

    for col in stat_cols:
        if col in stats.columns:
            stats[f'trailing_{col}'] = stats.groupby('player_norm')[col].transform(
                lambda x: x.shift(1).ewm(span=EWMA_SPAN, min_periods=1).mean()
            )

    # Merge trailing stats to odds
    trailing_cols = [col for col in stats.columns if 'trailing_' in col]
    context_cols = ['player_id', 'position', 'team', 'opponent_team']
    available_context = [c for c in context_cols if c in stats.columns]
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols + available_context

    stats_dedup = stats[merge_cols].drop_duplicates(subset=['player_norm', 'season', 'week'])
    odds_merged = odds.merge(stats_dedup, on=['player_norm', 'season', 'week'], how='left')

    # Fix team column collision
    if 'team_x' in odds_merged.columns and 'team_y' in odds_merged.columns:
        odds_merged['team'] = odds_merged['team_x'].fillna(odds_merged['team_y'])
        odds_merged = odds_merged.drop(columns=['team_x', 'team_y'], errors='ignore')

    if 'opponent_team' in odds_merged.columns:
        odds_merged['opponent'] = odds_merged['opponent_team']

    return odds_merged


# =============================================================================
# MODEL TRAINING (PER WEEK)
# =============================================================================

def train_xgboost_for_week(
    odds_merged: pd.DataFrame,
    market: str,
    test_global_week: int,
) -> tuple:
    """
    Train XGBoost for a specific market using data < test_global_week - 1.

    Returns:
        (model, calibrator, feature_cols) or (None, None, None) if insufficient data
    """
    market_features = get_market_features(market)

    # Map market to stat column
    stat_col_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_completions': 'completions',
        'player_pass_attempts': 'attempts',
    }

    stat_col = stat_col_map.get(market)
    if stat_col is None:
        return None, None, None

    trailing_col = f'trailing_{stat_col}'
    if trailing_col not in odds_merged.columns:
        return None, None, None

    # Filter to market and training data (1-week gap)
    train_data = odds_merged[
        (odds_merged['market'] == market) &
        (odds_merged['global_week'] < test_global_week - 1)
    ].copy()

    if len(train_data) < 100:
        return None, None, None

    # Historical data for feature extraction (same cutoff)
    hist_data = odds_merged[odds_merged['global_week'] < test_global_week - 1].copy()

    # Extract features
    features = extract_features_batch(train_data, hist_data, market)

    if len(features) < 100:
        return None, None, None

    # Get available features
    available_features = [f for f in market_features if f in features.columns]
    if len(available_features) < 5:
        return None, None, None

    # Prepare X, y
    X = safe_fillna(features[available_features])
    y = features['under_hit']

    # 80/20 split for calibration
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=MODEL_PARAMS.max_depth,
        learning_rate=MODEL_PARAMS.learning_rate,
        subsample=MODEL_PARAMS.subsample,
        colsample_bytree=MODEL_PARAMS.colsample_bytree,
        random_state=MODEL_PARAMS.random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Fit calibrator on held-out data
    raw_probs = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.01, y_max=0.99)
    calibrator.fit(raw_probs, y_calib)

    return model, calibrator, available_features


# =============================================================================
# PRODUCTION FILTERS
# =============================================================================

class ProductionFilters:
    """Production filters from V27."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def apply(self, row: pd.Series, market: str, pick: str) -> tuple:
        """
        Apply all production filters.

        Returns:
            (allowed, reason) - whether bet passes filters and why
        """
        if not self.enabled:
            return True, "Filters disabled"

        # 1. Market direction constraint (UNDER_ONLY)
        direction_constraint = MARKET_DIRECTION_CONSTRAINTS.get(market)
        if direction_constraint == 'UNDER_ONLY' and pick == 'OVER':
            return False, "UNDER_ONLY constraint"

        # 2. Spread filter for receptions (skip blowouts)
        if market == 'player_receptions':
            vegas_spread = row.get('vegas_spread', 0)
            if abs(vegas_spread) > 7:
                return False, f"Blowout filter (spread={vegas_spread})"

        # 3. TE exclusion for receptions
        if market == 'player_receptions':
            position = row.get('position', '')
            if position == 'TE':
                return False, "TE exclusion (50% edge)"

        # 4. Snap share minimum
        snap_share = row.get('snap_share', 0.5)
        if snap_share < 0.4:
            return False, f"Low snap share ({snap_share:.0%})"

        return True, "Passed all filters"


# =============================================================================
# WALK-FORWARD LOOP
# =============================================================================

def run_walk_forward(
    start_week: int = 5,
    end_week: int = 17,
    use_filters: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run production walk-forward backtest.

    Args:
        start_week: First week to test (need 4+ weeks for training)
        end_week: Last week to test
        use_filters: Whether to apply production filters
        verbose: Print progress

    Returns:
        DataFrame with all bet results
    """
    logger.info("="*70)
    logger.info("PRODUCTION WALK-FORWARD BACKTEST")
    logger.info("="*70)
    logger.info(f"Test weeks: {start_week}-{end_week}")
    logger.info(f"Filters: {'ENABLED' if use_filters else 'DISABLED'}")
    logger.info(f"Model: V{MODEL_VERSION} with market-specific features")

    # Load data
    odds, stats = load_all_data()
    odds_merged = prepare_data_with_trailing(odds, stats)

    # Initialize filters
    filters = ProductionFilters(enabled=use_filters)

    # Market to stat column mapping
    market_stat_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_yds': 'passing_yards',
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',
    }

    all_results = []

    # Walk through each test week
    for season in [2024, 2025]:
        max_week = 18 if season == 2024 else end_week
        min_week = start_week if season == 2024 else 1

        for week in range(min_week, max_week + 1):
            test_global_week = (season - 2023) * 18 + week

            if verbose:
                logger.info(f"\n--- Testing {season} Week {week} (global: {test_global_week}) ---")

            # Get test data for this week
            test_data = odds_merged[
                (odds_merged['season'] == season) &
                (odds_merged['week'] == week)
            ].copy()

            if len(test_data) == 0:
                continue

            week_start = time.time()
            week_bets = 0
            week_wins = 0

            # Train and predict for each market
            for market in CLASSIFIER_MARKETS:
                if market not in market_stat_map:
                    continue

                # Train model for this week
                model, calibrator, feature_cols = train_xgboost_for_week(
                    odds_merged, market, test_global_week
                )

                if model is None:
                    continue

                # Get market test data
                market_test = test_data[test_data['market'] == market].copy()
                if len(market_test) == 0:
                    continue

                # Extract features for test data
                hist_data = odds_merged[odds_merged['global_week'] < test_global_week - 1].copy()
                test_features = extract_features_batch(market_test, hist_data, market)

                if len(test_features) == 0:
                    continue

                # Get predictions
                available_features = [f for f in feature_cols if f in test_features.columns]
                for col in feature_cols:
                    if col not in test_features.columns:
                        test_features[col] = 0.0

                X_test = safe_fillna(test_features[feature_cols])
                raw_probs = model.predict_proba(X_test)[:, 1]
                calibrated_probs = calibrator.predict(raw_probs)

                # Evaluate each prediction
                for i, (idx, row) in enumerate(test_features.iterrows()):
                    p_under = calibrated_probs[i]
                    pick = 'UNDER' if p_under > 0.5 else 'OVER'
                    prob = p_under if pick == 'UNDER' else (1 - p_under)

                    # Apply filters
                    allowed, filter_reason = filters.apply(row, market, pick)
                    if not allowed:
                        continue

                    # Get actual result
                    under_hit = row.get('under_hit', np.nan)
                    if pd.isna(under_hit):
                        continue

                    actual_hit = int(under_hit) if pick == 'UNDER' else int(1 - under_hit)

                    # Calculate ROI contribution
                    roi_contribution = (0.909 if actual_hit else -1.0)

                    all_results.append({
                        'season': season,
                        'week': week,
                        'global_week': test_global_week,
                        'player': row.get('player', row.get('player_norm', 'Unknown')),
                        'player_norm': row.get('player_norm', ''),
                        'market': market,
                        'line': row.get('line', 0),
                        'direction': pick,
                        'raw_prob': raw_probs[i],
                        'calibrated_prob': p_under,
                        'prob': prob,
                        'actual_stat': row.get('actual_stat', np.nan),
                        'under_hit': under_hit,
                        'actual_hit': actual_hit,
                        'roi_contribution': roi_contribution,
                        'filters_enabled': use_filters,
                    })

                    week_bets += 1
                    week_wins += actual_hit

            if verbose and week_bets > 0:
                week_wr = week_wins / week_bets
                week_roi = (week_wr * 0.909 - (1 - week_wr)) * 100
                elapsed = time.time() - week_start
                logger.info(f"  {week_bets} bets, {week_wr:.1%} win rate, {week_roi:+.1f}% ROI ({elapsed:.1f}s)")

    # Clear feature caches
    clear_caches()

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df


def print_summary(results_df: pd.DataFrame, label: str = ""):
    """Print summary statistics."""
    if len(results_df) == 0:
        print(f"\n{label}: No results")
        return

    print("\n" + "="*70)
    print(f"RESULTS SUMMARY {label}")
    print("="*70)

    total_bets = len(results_df)
    total_wins = results_df['actual_hit'].sum()
    win_rate = total_wins / total_bets
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100
    units_won = results_df['roi_contribution'].sum()

    print(f"\nOverall: {total_bets} bets")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Units: {units_won:+.1f}")

    print("\nBy Market:")
    print(f"  {'Market':<25} {'Bets':<8} {'Win%':<10} {'ROI':<10}")
    print("  " + "-"*55)

    for market in results_df['market'].unique():
        market_df = results_df[results_df['market'] == market]
        m_bets = len(market_df)
        m_wr = market_df['actual_hit'].mean()
        m_roi = (m_wr * 0.909 - (1 - m_wr)) * 100
        print(f"  {market:<25} {m_bets:<8} {m_wr:.1%}      {m_roi:+.1f}%")

    print("\nBy Direction:")
    for direction in ['UNDER', 'OVER']:
        dir_df = results_df[results_df['direction'] == direction]
        if len(dir_df) > 0:
            d_bets = len(dir_df)
            d_wr = dir_df['actual_hit'].mean()
            d_roi = (d_wr * 0.909 - (1 - d_wr)) * 100
            print(f"  {direction}: {d_bets} bets, {d_wr:.1%} win rate, {d_roi:+.1f}% ROI")

    # Threshold analysis
    print("\nThreshold Analysis (Calibrated Prob):")
    print(f"  {'Thresh':<10} {'Bets':<8} {'Win%':<10} {'ROI':<10}")
    print("  " + "-"*40)

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        high_conf = results_df[results_df['prob'] >= thresh]
        if len(high_conf) >= 10:
            t_bets = len(high_conf)
            t_wr = high_conf['actual_hit'].mean()
            t_roi = (t_wr * 0.909 - (1 - t_wr)) * 100
            print(f"  {thresh:.0%}        {t_bets:<8} {t_wr:.1%}      {t_roi:+.1f}%")


def run_ab_test():
    """Run A/B test: with filters vs without filters."""
    print("\n" + "="*70)
    print("A/B TEST: FILTERS vs NO FILTERS")
    print("="*70)

    # Run without filters
    print("\n>>> Running baseline (NO FILTERS)...")
    results_no_filters = run_walk_forward(use_filters=False, verbose=False)

    # Save
    no_filter_path = BACKTEST_DIR / 'walk_forward_production_no_filters.csv'
    results_no_filters.to_csv(no_filter_path, index=False)
    print(f"Saved: {no_filter_path}")

    print_summary(results_no_filters, "(NO FILTERS)")

    # Clear caches between runs
    clear_caches()

    # Run with filters
    print("\n>>> Running with filters...")
    results_with_filters = run_walk_forward(use_filters=True, verbose=False)

    # Save
    with_filter_path = BACKTEST_DIR / 'walk_forward_production_with_filters.csv'
    results_with_filters.to_csv(with_filter_path, index=False)
    print(f"Saved: {with_filter_path}")

    print_summary(results_with_filters, "(WITH FILTERS)")

    # Compare
    print("\n" + "="*70)
    print("A/B COMPARISON")
    print("="*70)

    no_f_roi = (results_no_filters['actual_hit'].mean() * 0.909 - (1 - results_no_filters['actual_hit'].mean())) * 100
    with_f_roi = (results_with_filters['actual_hit'].mean() * 0.909 - (1 - results_with_filters['actual_hit'].mean())) * 100

    print(f"\n  Baseline (No Filters): {len(results_no_filters)} bets, {no_f_roi:+.1f}% ROI")
    print(f"  With Filters:          {len(results_with_filters)} bets, {with_f_roi:+.1f}% ROI")
    print(f"\n  Filter Impact: {with_f_roi - no_f_roi:+.1f}% ROI improvement")

    if with_f_roi > no_f_roi:
        print("\n  >>> FILTERS HELP - Keep them enabled")
    else:
        print("\n  >>> FILTERS HURT - Consider disabling")

    # Save comparison summary
    summary = {
        'no_filters': {
            'total_bets': len(results_no_filters),
            'win_rate': float(results_no_filters['actual_hit'].mean()),
            'roi': float(no_f_roi),
        },
        'with_filters': {
            'total_bets': len(results_with_filters),
            'win_rate': float(results_with_filters['actual_hit'].mean()),
            'roi': float(with_f_roi),
        },
        'filter_impact_roi': float(with_f_roi - no_f_roi),
        'recommendation': 'Keep filters' if with_f_roi > no_f_roi else 'Disable filters',
        'timestamp': datetime.now().isoformat(),
    }

    summary_path = BACKTEST_DIR / 'walk_forward_production_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    return results_no_filters, results_with_filters


def main():
    parser = argparse.ArgumentParser(description='Production Walk-Forward Backtest')
    parser.add_argument('--start-week', type=int, default=5, help='Start week')
    parser.add_argument('--end-week', type=int, default=17, help='End week')
    parser.add_argument('--no-filters', action='store_true', help='Run without filters')
    parser.add_argument('--with-filters', action='store_true', help='Run with filters')
    parser.add_argument('--ab-test', action='store_true', help='Run A/B test (both)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    args = parser.parse_args()

    total_start = time.time()

    if args.ab_test or (not args.no_filters and not args.with_filters):
        # Default: run A/B test
        run_ab_test()
    elif args.no_filters:
        results = run_walk_forward(
            start_week=args.start_week,
            end_week=args.end_week,
            use_filters=False,
            verbose=not args.quiet
        )
        print_summary(results, "(NO FILTERS)")

        output_path = BACKTEST_DIR / 'walk_forward_production_no_filters.csv'
        results.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
    else:
        results = run_walk_forward(
            start_week=args.start_week,
            end_week=args.end_week,
            use_filters=True,
            verbose=not args.quiet
        )
        print_summary(results, "(WITH FILTERS)")

        output_path = BACKTEST_DIR / 'walk_forward_production_with_filters.csv'
        results.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
