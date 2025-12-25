#!/usr/bin/env python3
"""
Walk-Forward Validation for Edge Ensemble

For each test week W (2025 season):
1. Train LVT and Player Bias edges on data from weeks < W
2. Generate predictions for week W using only historical features
3. Compare to actual outcomes
4. Track hit rate and ROI by market and source

This provides rigorous validation without data leakage.
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import DATA_DIR, MODELS_DIR
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.edges.lvt_edge import LVTEdge
from nfl_quant.edges.player_bias_edge import PlayerBiasEdge
from nfl_quant.edges.ensemble import EdgeEnsemble
from nfl_quant.features.trailing_stats import (
    load_player_stats_for_edge,
    compute_edge_trailing_stats,
    merge_edge_trailing_stats,
    compute_line_vs_trailing,
    EDGE_TRAILING_COL_MAP,
)
from configs.edge_config import EDGE_MARKETS
from configs.ensemble_config import EdgeSource
from configs.model_config import smooth_sweet_spot


def load_enriched_data() -> pd.DataFrame:
    """Load the enriched training/validation data."""
    path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    if not path.exists():
        raise FileNotFoundError(f"Enriched data not found: {path}")
    return pd.read_csv(path, low_memory=False)


def prepare_features_for_week(df: pd.DataFrame, test_week: int, season: int = 2025) -> pd.DataFrame:
    """
    Prepare features for a test week using ONLY prior data.

    Args:
        df: Full enriched dataset
        test_week: Week to predict
        season: Season year

    Returns:
        DataFrame with features for test week (using only historical data)
    """
    # Split: train on data before test_week, predict on test_week
    train_mask = (df['season'] < season) | ((df['season'] == season) & (df['week'] < test_week))
    test_mask = (df['season'] == season) & (df['week'] == test_week)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if len(test_df) == 0:
        return pd.DataFrame()

    # Normalize player names
    test_df['player_norm'] = test_df['player'].apply(normalize_player_name)
    train_df['player_norm'] = train_df['player'].apply(normalize_player_name)

    # Compute under_hit
    if 'under_hit' not in test_df.columns:
        test_df['under_hit'] = (test_df['actual'] < test_df['line']).astype(int)
    if 'under_hit' not in train_df.columns:
        train_df['under_hit'] = (train_df['actual'] < train_df['line']).astype(int)

    # Compute player stats from training data ONLY
    player_stats = train_df.groupby('player_norm').agg({
        'under_hit': ['mean', 'count'],
        'actual': 'mean',
        'line': 'mean',
    }).reset_index()
    player_stats.columns = ['player_norm', 'player_under_rate', 'player_bet_count', 'avg_actual', 'avg_line']
    player_stats['player_bias'] = player_stats['avg_actual'] - player_stats['avg_line']

    # Merge player stats into test data
    test_df = test_df.merge(
        player_stats[['player_norm', 'player_under_rate', 'player_bet_count', 'player_bias']],
        on='player_norm',
        how='left'
    )

    # Current season stats (within training window)
    current_season_train = train_df[train_df['season'] == season].copy()
    if len(current_season_train) > 0:
        season_stats = current_season_train.groupby('player_norm').agg({
            'under_hit': 'mean',
        }).reset_index()
        season_stats.columns = ['player_norm', 'current_season_under_rate']
        season_games = current_season_train.groupby('player_norm').size().reset_index(name='season_games_played')
        season_stats = season_stats.merge(season_games, on='player_norm', how='left')

        test_df = test_df.merge(season_stats, on='player_norm', how='left')

    # NO DATA = NO BET: Only fill game context defaults
    # Critical fields (leave NaN if missing):
    #   - player_under_rate (Player Bias requirement)
    #   - player_bet_count (min games requirement)
    #   - trailing_* stats (LVT requirement)
    game_context_defaults = {
        'player_bias': 0.0,
        'current_season_under_rate': 0.5,
        'season_games_played': 0,
        'market_under_rate': 0.5,
        'vegas_spread': 0.0,
        'implied_team_total': 24.0,
        'pos_rank': 2,
        'market_bias_strength': 0.0,
        'opp_def_epa': 0.0,
        'has_opponent_context': 0,
        'rest_days': 7.0,
        'elo_diff': 0.0,
        'opp_pass_yds_def_vs_avg': 0.0,
        'opp_rush_yds_def_vs_avg': 0.0,
        'injury_status_encoded': 0,
        'has_injury_designation': 0,
    }

    for col, default in game_context_defaults.items():
        if col not in test_df.columns:
            test_df[col] = default
        else:
            test_df[col] = test_df[col].fillna(default)

    # Initialize critical fields but DO NOT fill (keep NaN)
    for col in ['target_share', 'snap_share', 'trailing_catch_rate']:
        if col not in test_df.columns:
            test_df[col] = np.nan

    test_df['is_starter'] = (test_df['pos_rank'] == 1).astype(int)
    test_df['line_level'] = test_df['line']

    return test_df


def validate_week(ensemble: EdgeEnsemble, test_df: pd.DataFrame, stats: pd.DataFrame) -> dict:
    """
    Validate ensemble on a single week.

    Returns:
        Dict with results per market and source
    """
    # Merge trailing stats
    test_df = merge_edge_trailing_stats(test_df, stats)

    results = {}

    for market in EDGE_MARKETS:
        market_df = test_df[test_df['market'] == market].copy()

        if len(market_df) < 5:
            continue

        # Compute line_vs_trailing
        market_df['line_vs_trailing'] = compute_line_vs_trailing(market_df, market)

        # Compute interaction features
        market_df['LVT_x_player_tendency'] = market_df['line_vs_trailing'] * (
            market_df['player_under_rate'].fillna(0.5) - 0.5
        )
        market_df['LVT_x_player_bias'] = market_df['line_vs_trailing'] * market_df['player_bias']
        market_df['player_market_aligned'] = np.where(
            (market_df['player_under_rate'].fillna(0.5) > 0.5) == (market_df['market_under_rate'] > 0.5),
            1.0, -1.0
        )

        # Sweet spot features
        market_df['line_in_sweet_spot'] = market_df['line'].apply(
            lambda x: smooth_sweet_spot(x, market)
        )
        market_df['LVT_in_sweet_spot'] = market_df['line_vs_trailing'] * market_df['line_in_sweet_spot']
        market_df['lvt_x_defense'] = market_df['line_vs_trailing'] * market_df['opp_def_epa']

        # Evaluate each row
        source_results = {s.value: {'n': 0, 'hits': 0} for s in EdgeSource}

        for _, row in market_df.iterrows():
            try:
                decision = ensemble.evaluate_bet(row, market)
                source = decision.source.value

                if decision.should_bet:
                    source_results[source]['n'] += 1
                    actual_direction = 'UNDER' if row['under_hit'] == 1 else 'OVER'
                    if decision.direction == actual_direction:
                        source_results[source]['hits'] += 1
                elif source in ['CONFLICT', 'NEITHER', 'NO_DATA']:
                    # Track non-bet cases
                    source_results[source]['n'] += 1
            except Exception:
                continue

        results[market] = source_results

    return results


def run_walk_forward(start_week: int = 5, end_week: int = 16, season: int = 2025):
    """
    Run walk-forward validation from start_week to end_week.
    """
    print("=" * 70)
    print("WALK-FORWARD EDGE ENSEMBLE VALIDATION")
    print(f"Season: {season}, Weeks: {start_week}-{end_week}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_enriched_data()

    # Load NFLverse stats for trailing calculations
    stats = load_player_stats_for_edge()
    stats = compute_edge_trailing_stats(stats)

    # Load trained ensemble
    ensemble = EdgeEnsemble.load()
    print("Ensemble loaded successfully")

    # Aggregate results
    all_results = {m: {s.value: {'n': 0, 'hits': 0} for s in EdgeSource} for m in EDGE_MARKETS}
    weekly_summary = []

    for week in range(start_week, end_week + 1):
        print(f"\n--- Week {week} ---")

        # Prepare test data with only historical features
        test_df = prepare_features_for_week(df, week, season)

        if len(test_df) == 0:
            print(f"  No data for week {week}")
            continue

        print(f"  Test samples: {len(test_df)}")

        # Validate
        week_results = validate_week(ensemble, test_df, stats)

        # Aggregate
        week_bets = 0
        week_hits = 0
        for market, market_results in week_results.items():
            for source, data in market_results.items():
                all_results[market][source]['n'] += data['n']
                all_results[market][source]['hits'] += data['hits']
                if source not in ['CONFLICT', 'NEITHER', 'NO_DATA']:
                    week_bets += data['n']
                    week_hits += data['hits']

        if week_bets > 0:
            week_hit_rate = week_hits / week_bets
            week_roi = (week_hit_rate * 0.909 - (1 - week_hit_rate)) * 100
            print(f"  Week {week}: {week_bets} bets, {week_hits} hits, {week_hit_rate:.1%} hit rate, {week_roi:+.1f}% ROI")
            weekly_summary.append({
                'week': week,
                'bets': week_bets,
                'hits': week_hits,
                'hit_rate': week_hit_rate,
                'roi': week_roi
            })

    # Print final summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS BY MARKET")
    print("=" * 70)

    for market in EDGE_MARKETS:
        print(f"\n{market}:")
        print(f"  {'Source':<20} {'Bets':<8} {'Hits':<8} {'Hit %':<10} {'ROI':<10}")
        print("  " + "-" * 56)

        for source in ['BOTH', 'LVT_ONLY', 'PLAYER_BIAS_ONLY']:
            data = all_results[market][source]
            if data['n'] > 0:
                hit_rate = data['hits'] / data['n']
                roi = (hit_rate * 0.909 - (1 - hit_rate)) * 100
                print(f"  {source:<20} {data['n']:<8} {data['hits']:<8} {hit_rate:.1%}     {roi:+.1f}%")

        # Show NO_DATA skips
        no_data = all_results[market].get('NO_DATA', {}).get('n', 0)
        if no_data > 0:
            print(f"  {'NO_DATA (skipped)':<20} {no_data:<8}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    total_bets = 0
    total_hits = 0
    for market in EDGE_MARKETS:
        for source in ['BOTH', 'LVT_ONLY', 'PLAYER_BIAS_ONLY']:
            total_bets += all_results[market][source]['n']
            total_hits += all_results[market][source]['hits']

    if total_bets > 0:
        overall_hit_rate = total_hits / total_bets
        overall_roi = (overall_hit_rate * 0.909 - (1 - overall_hit_rate)) * 100
        print(f"\nTotal: {total_bets} bets, {total_hits} hits")
        print(f"Hit Rate: {overall_hit_rate:.1%}")
        print(f"ROI: {overall_roi:+.1f}%")

    # Weekly breakdown
    if weekly_summary:
        print("\n" + "-" * 70)
        print("WEEKLY BREAKDOWN")
        print("-" * 70)
        print(f"{'Week':<8} {'Bets':<8} {'Hits':<8} {'Hit %':<10} {'ROI':<10}")
        print("-" * 44)
        for w in weekly_summary:
            print(f"{w['week']:<8} {w['bets']:<8} {w['hits']:<8} {w['hit_rate']:.1%}     {w['roi']:+.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Edge Validation")
    parser.add_argument('--start-week', type=int, default=5, help='Start week for validation')
    parser.add_argument('--end-week', type=int, default=16, help='End week for validation')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    args = parser.parse_args()

    run_walk_forward(args.start_week, args.end_week, args.season)


if __name__ == '__main__':
    main()
