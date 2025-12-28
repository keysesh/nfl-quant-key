#!/usr/bin/env python3
"""
Train Edge Ensemble

Trains both LVT and Player Bias edges and validates the ensemble.

Usage:
    python scripts/train/train_ensemble.py
    python scripts/train/train_ensemble.py --validate-only
"""
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.edges.lvt_edge import LVTEdge
from nfl_quant.edges.player_bias_edge import PlayerBiasEdge
from nfl_quant.edges.ensemble import EdgeEnsemble
from nfl_quant.config_paths import MODELS_DIR, DATA_DIR
from nfl_quant.features.trailing_stats import (
    load_player_stats_for_edge,
    compute_edge_trailing_stats,
    merge_edge_trailing_stats,
    compute_line_vs_trailing,
    EDGE_TRAILING_COL_MAP,
)
from configs.edge_config import EDGE_MARKETS
from configs.ensemble_config import EdgeSource


def train_lvt_edge():
    """Train LVT edge by running training script."""
    print("\n" + "="*60)
    print("TRAINING LVT EDGE")
    print("="*60)

    script_path = PROJECT_ROOT / 'scripts' / 'train' / 'train_lvt_edge.py'
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"LVT training failed:\n{result.stderr}")
        return False

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return True


def train_player_bias_edge():
    """Train Player Bias edge by running training script."""
    print("\n" + "="*60)
    print("TRAINING PLAYER BIAS EDGE")
    print("="*60)

    script_path = PROJECT_ROOT / 'scripts' / 'train' / 'train_player_bias_edge.py'
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Player Bias training failed:\n{result.stderr}")
        return False

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return True


def validate_ensemble():
    """Validate ensemble performance on holdout data."""
    print("\n" + "="*60)
    print("VALIDATING ENSEMBLE")
    print("="*60)

    # Load trained edges
    try:
        ensemble = EdgeEnsemble.load()
        print("Loaded ensemble successfully")
    except Exception as e:
        print(f"Failed to load ensemble: {e}")
        return None

    # Load holdout data (2025 season) - prefer MARKET_ENRICHED for RB features
    market_enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_MARKET_ENRICHED.csv'
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'

    if market_enriched_path.exists():
        print(f"Loading market-enriched data: {market_enriched_path}")
        df = pd.read_csv(market_enriched_path, low_memory=False)
    elif enriched_path.exists():
        print(f"Loading enriched data: {enriched_path}")
        df = pd.read_csv(enriched_path, low_memory=False)
    else:
        print("No enriched data for validation")
        return None
    holdout = df[df['season'] == 2025].copy()

    if len(holdout) < 100:
        print(f"Insufficient holdout data: {len(holdout)} rows")
        return None

    print(f"Holdout data: {len(holdout)} rows (2025 season)")

    # Prepare features
    from nfl_quant.utils.player_names import normalize_player_name

    holdout['player_norm'] = holdout['player'].apply(normalize_player_name)
    holdout = holdout.sort_values(['player_norm', 'season', 'week'])

    # Compute under_hit if missing
    if 'under_hit' not in holdout.columns:
        holdout['under_hit'] = (holdout['actual'] < holdout['line']).astype(int)

    # Compute player stats
    holdout['player_under_rate'] = (
        holdout.groupby('player_norm')['under_hit']
        .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
    )

    # Compute player bet count (cumulative count per player)
    holdout['player_bet_count'] = holdout.groupby('player_norm').cumcount()

    # Compute current season bias (2025 only)
    current_season = holdout['season'].max()
    season_mask = holdout['season'] == current_season
    holdout['current_season_under_rate'] = 0.5  # Default
    if season_mask.any():
        holdout.loc[season_mask, 'current_season_under_rate'] = (
            holdout[season_mask].groupby('player_norm')['under_hit']
            .transform(lambda x: x.rolling(8, min_periods=2).mean().shift(1))
        ).fillna(0.5)

    # Season games played
    holdout['season_games_played'] = 0
    if season_mask.any():
        holdout.loc[season_mask, 'season_games_played'] = holdout[season_mask].groupby('player_norm').cumcount()

    # =========================================================================
    # CRITICAL FIX: Compute trailing stats and LVT from NFLverse data
    # =========================================================================
    print("\nComputing trailing stats for LVT...")
    try:
        # Load and compute trailing stats from NFLverse
        stats = load_player_stats_for_edge()
        stats = compute_edge_trailing_stats(stats)

        # Merge trailing stats into holdout
        holdout = merge_edge_trailing_stats(holdout, stats)

        # Report trailing stats coverage
        for market in EDGE_MARKETS:
            trailing_col = EDGE_TRAILING_COL_MAP.get(market)
            if trailing_col and trailing_col in holdout.columns:
                coverage = holdout[trailing_col].notna().mean()
                print(f"  {market}: {trailing_col} coverage = {coverage:.1%}")
    except Exception as e:
        print(f"  Warning: Could not load trailing stats: {e}")
        print("  LVT edge will not trigger without trailing stats")

    # NO DATA = NO BET: Only fill game context defaults (reasonable neutral values)
    # Critical fields (trailing stats, player history) are NOT filled - missing = skip bet
    for col, default in [
        # Game context (OK to have neutral defaults)
        ('market_under_rate', 0.5),
        ('vegas_spread', 0.0),
        ('implied_team_total', 24.0),
        ('pos_rank', 2),
        ('market_bias_strength', 0.0),
        ('opp_def_epa', 0.0),
        ('has_opponent_context', 0),
        ('rest_days', 7.0),
        ('elo_diff', 0.0),
        ('opp_pass_yds_def_vs_avg', 0.0),
        ('opp_rush_yds_def_vs_avg', 0.0),
        ('injury_status_encoded', 0),
        ('has_injury_designation', 0),
        # NOTE: These are NOT filled - missing = no bet:
        # - trailing_* (LVT edge requirement)
        # - player_under_rate (Player Bias requirement)
        # - player_bet_count (min games requirement)
        # - target_share, snap_share, trailing_catch_rate (left as NaN)
    ]:
        if col not in holdout.columns:
            holdout[col] = default

    # Initialize critical fields if missing (but keep NaN - don't fill with defaults)
    for col in ['player_bias', 'target_share', 'snap_share', 'trailing_catch_rate']:
        if col not in holdout.columns:
            holdout[col] = np.nan

    holdout['is_starter'] = (holdout['pos_rank'] == 1).astype(int)
    holdout['line_level'] = holdout['line']

    # Import smooth_sweet_spot
    from configs.model_config import smooth_sweet_spot

    # Validate per market
    results = {}
    for market in EDGE_MARKETS:
        market_df = holdout[holdout['market'] == market].dropna(subset=['player_under_rate'])

        if len(market_df) < 20:
            continue

        # Compute line_vs_trailing for this market using trailing stats
        market_df = market_df.copy()
        market_df['line_vs_trailing'] = compute_line_vs_trailing(market_df, market)

        # Compute interaction features now that we have LVT
        market_df['LVT_x_player_tendency'] = market_df['line_vs_trailing'] * (
            market_df['player_under_rate'].fillna(0.5) - 0.5
        )
        market_df['LVT_x_player_bias'] = market_df['line_vs_trailing'] * market_df['player_bias']
        market_df['player_market_aligned'] = np.where(
            (market_df['player_under_rate'].fillna(0.5) > 0.5) == (market_df['market_under_rate'] > 0.5),
            1.0, -1.0
        )

        # Add sweet spot features
        market_df['line_in_sweet_spot'] = market_df['line'].apply(
            lambda x: smooth_sweet_spot(x, market)
        )
        market_df['LVT_in_sweet_spot'] = market_df['line_vs_trailing'] * market_df['line_in_sweet_spot']

        # Add lvt_x_defense interaction (V28 feature)
        market_df['lvt_x_defense'] = market_df['line_vs_trailing'] * market_df['opp_def_epa']

        # Add RB-specific features for rush_attempts market
        if market == 'player_rush_attempts':
            # These should already be in the data from merge_edge_trailing_stats
            # Fill missing with 0 to avoid feature mismatch
            for col in ['trailing_carries', 'trailing_ypc', 'trailing_cv_carries', 'trailing_rb_snap_share']:
                if col not in market_df.columns:
                    market_df[col] = 0.0
                else:
                    market_df[col] = market_df[col].fillna(0.0)

        print(f"\n{market}:")
        print(f"  Total samples: {len(market_df)}")

        # Report LVT stats
        lvt_nonzero = (market_df['line_vs_trailing'] != 0).sum()
        lvt_high = (market_df['line_vs_trailing'].abs() > 5).sum()
        print(f"  LVT non-zero: {lvt_nonzero} ({lvt_nonzero/len(market_df):.1%})")
        print(f"  LVT |value| > 5%: {lvt_high} ({lvt_high/len(market_df):.1%})")

        # Evaluate each row
        source_results = {s.value: {'n': 0, 'hits': 0} for s in EdgeSource}

        for _, row in market_df.iterrows():
            decision = ensemble.evaluate_bet(row, market)
            source = decision.source.value

            if decision.should_bet:
                source_results[source]['n'] += 1
                # Check if bet won
                actual_direction = 'UNDER' if row['under_hit'] == 1 else 'OVER'
                if decision.direction == actual_direction:
                    source_results[source]['hits'] += 1
            elif source in ['CONFLICT', 'NEITHER', 'NO_DATA']:
                source_results[source]['n'] += 1

        # Print results
        print(f"\n  {'Source':<20} {'Bets':<8} {'Hits':<8} {'Hit %':<10} {'ROI':<10}")
        print("  " + "-"*56)

        market_results = {}
        for source, data in source_results.items():
            if data['n'] > 0:
                if source in ['CONFLICT', 'NEITHER', 'NO_DATA']:
                    print(f"  {source:<20} {data['n']:<8} {'N/A':<8} {'N/A':<10} {'N/A':<10}")
                else:
                    hit_rate = data['hits'] / data['n']
                    roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
                    print(f"  {source:<20} {data['n']:<8} {data['hits']:<8} {hit_rate:.1%}     {roi:+.1f}%")
                    market_results[source] = {
                        'n': data['n'],
                        'hits': data['hits'],
                        'hit_rate': hit_rate,
                        'roi': roi
                    }

        results[market] = market_results

    return results


def save_ensemble_config(results: dict):
    """Save ensemble validation results."""
    config_path = MODELS_DIR / 'edge_ensemble_config.json'

    config = {
        'version': 'edge_v1',
        'trained_date': datetime.now().isoformat(),
        'validation_results': results,
        'markets': EDGE_MARKETS,
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nEnsemble config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Edge Ensemble")
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing models')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation after training')
    args = parser.parse_args()

    print("="*60)
    print("EDGE ENSEMBLE TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not args.validate_only:
        # Train LVT edge
        if not train_lvt_edge():
            print("LVT training failed, aborting")
            return

        # Train Player Bias edge
        if not train_player_bias_edge():
            print("Player Bias training failed, aborting")
            return

    if not args.skip_validation:
        # Validate ensemble
        results = validate_ensemble()

        if results:
            save_ensemble_config(results)

            # Print summary
            print("\n" + "="*60)
            print("ENSEMBLE VALIDATION SUMMARY")
            print("="*60)

            total_bets = 0
            total_hits = 0
            for market, market_results in results.items():
                for source, data in market_results.items():
                    total_bets += data['n']
                    total_hits += data['hits']

            if total_bets > 0:
                overall_hit_rate = total_hits / total_bets
                overall_roi = (overall_hit_rate * 0.909 + (1 - overall_hit_rate) * -1.0) * 100
                print(f"\nOverall: {total_bets} bets, {total_hits} hits, "
                      f"{overall_hit_rate:.1%} hit rate, {overall_roi:+.1f}% ROI")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
