#!/usr/bin/env python3
"""
Walk-Forward Validation for V4 Simulator

Validates the fixed V4 simulator pipeline:
1. Uses NegBin + Lognormal + Copula distributions
2. Uses fixed trailing_stats lookup (most recent week)
3. Compares predictions to actual NFLverse outcomes

Usage:
    python scripts/backtest/walk_forward_v4_validation.py
    python scripts/backtest/walk_forward_v4_validation.py --weeks 5-13
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_weekly_stats(season: int = 2025) -> pd.DataFrame:
    """Load NFLverse weekly stats."""
    path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    df = pd.read_parquet(path)
    df = df[df['season'] == season].copy()
    logger.info(f"Loaded {len(df):,} player-week records ({season})")
    return df


def calculate_trailing_averages(player_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate trailing averages from historical data."""
    if len(player_data) == 0:
        return {}

    return {
        'avg_targets': player_data['targets'].mean() if 'targets' in player_data else 0,
        'avg_receptions': player_data['receptions'].mean() if 'receptions' in player_data else 0,
        'avg_receiving_yards': player_data['receiving_yards'].mean() if 'receiving_yards' in player_data else 0,
        'avg_rushing_yards': player_data['rushing_yards'].mean() if 'rushing_yards' in player_data else 0,
        'avg_passing_yards': player_data['passing_yards'].mean() if 'passing_yards' in player_data else 0,
        'games_played': len(player_data),
    }


def run_validation(
    weeks: List[int],
    season: int = 2025,
    positions: List[str] = ['WR', 'TE', 'RB'],
    min_games: int = 3,
    trials: int = 5000,
) -> pd.DataFrame:
    """
    Run walk-forward validation for specified weeks.

    For each week N:
    1. Calculate trailing stats using weeks < N
    2. Generate predictions using V4 simulator
    3. Compare to actual week N outcomes
    """
    logger.info("=" * 60)
    logger.info("V4 SIMULATOR WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Weeks: {min(weeks)}-{max(weeks)}")
    logger.info(f"Positions: {positions}")
    logger.info(f"Min games required: {min_games}")
    logger.info(f"Monte Carlo trials: {trials:,}")
    logger.info("")

    # Load data
    weekly = load_weekly_stats(season)

    # Load simulator
    logger.info("Loading V4 simulator...")
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(usage_predictor, efficiency_predictor, trials=trials)
    logger.info("  Simulator loaded (NegBin + Lognormal + Copula)")
    logger.info("")

    all_results = []

    for week in weeks:
        logger.info(f"--- Week {week} ---")

        # Get trailing data (weeks before current)
        trailing_data = weekly[weekly['week'] < week].copy()

        # Get actual outcomes for target week
        target_data = weekly[weekly['week'] == week].copy()

        if len(target_data) == 0:
            logger.warning(f"  No data for week {week}")
            continue

        week_results = []

        for _, row in target_data.iterrows():
            player_name = row['player_display_name']
            position = row['position']
            team = row['team']

            if position not in positions:
                continue

            # Get player's historical data
            player_history = trailing_data[
                trailing_data['player_display_name'] == player_name
            ]

            if len(player_history) < min_games:
                continue

            # Calculate trailing averages
            trailing = calculate_trailing_averages(player_history)

            try:
                # Create player input
                player_input = PlayerPropInput(
                    player_name=player_name,
                    player_id=f"val_{player_name}_{week}",
                    position=position,
                    team=team,
                    opponent=row['opponent_team'] if 'opponent_team' in row.index else 'UNK',
                    week=week,
                    season=season,
                    projected_team_total=24.0,
                    projected_opponent_total=24.0,
                    projected_game_script=0.0,
                    projected_pace=60.0,
                    trailing_snap_share=0.85,
                    trailing_target_share=trailing['avg_targets'] / 35.0,
                    trailing_carry_share=0.0,
                    trailing_yards_per_opportunity=trailing['avg_receiving_yards'] / trailing['avg_targets'] if trailing['avg_targets'] > 0 else 8.0,
                    trailing_td_rate=0.05,
                    opponent_def_epa_vs_position=0.0,
                    avg_rec_tgt=trailing['avg_targets'],  # KEY: Pass actual targets
                    avg_rec_yd=trailing['avg_receiving_yards'],
                )

                # Run simulation
                sim_output = simulator.simulate_player(player_input)

                # Extract receiving yards samples
                rec_yards_samples = sim_output.get('receiving_yards', np.array([0]))
                pred_mean = np.mean(rec_yards_samples)
                pred_std = np.std(rec_yards_samples)
                pred_cv = pred_std / pred_mean if pred_mean > 0 else 0

                # Actual outcome
                actual = row['receiving_yards'] if pd.notna(row.get('receiving_yards')) else 0

                # Calculate error
                error = actual - pred_mean
                abs_error = abs(error)
                pct_error = abs_error / actual if actual > 0 else 0

                week_results.append({
                    'week': week,
                    'player': player_name,
                    'position': position,
                    'team': team,
                    'trailing_avg': trailing['avg_receiving_yards'],
                    'pred_mean': pred_mean,
                    'pred_std': pred_std,
                    'pred_cv': pred_cv,
                    'actual': actual,
                    'error': error,
                    'abs_error': abs_error,
                    'pct_error': pct_error,
                })

            except Exception as e:
                logger.debug(f"  Error for {player_name}: {e}")
                continue

        if week_results:
            week_df = pd.DataFrame(week_results)
            mae = week_df['abs_error'].mean()
            avg_cv = week_df['pred_cv'].mean()
            logger.info(f"  Players: {len(week_results)}, MAE: {mae:.1f}, Avg CV: {avg_cv:.3f}")
            all_results.extend(week_results)

    results_df = pd.DataFrame(all_results)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    if len(results_df) > 0:
        logger.info(f"Total predictions: {len(results_df)}")
        logger.info(f"Mean Absolute Error: {results_df['abs_error'].mean():.1f} yards")
        logger.info(f"Median Absolute Error: {results_df['abs_error'].median():.1f} yards")
        logger.info(f"Mean CV: {results_df['pred_cv'].mean():.3f} (target: 0.4-0.6)")
        logger.info(f"CV Range: {results_df['pred_cv'].min():.3f} - {results_df['pred_cv'].max():.3f}")

        # Correlation
        corr = results_df['pred_mean'].corr(results_df['actual'])
        logger.info(f"Prediction-Actual Correlation: {corr:.3f}")

        # Check if predictions are reasonable
        good_cv = results_df[(results_df['pred_cv'] >= 0.3) & (results_df['pred_cv'] <= 0.8)]
        logger.info(f"Predictions with valid CV (0.3-0.8): {len(good_cv)}/{len(results_df)} ({100*len(good_cv)/len(results_df):.1f}%)")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='V4 Simulator Walk-Forward Validation')
    parser.add_argument('--weeks', type=str, default='5-13', help='Week range (e.g., 5-13)')
    parser.add_argument('--trials', type=int, default=5000, help='Monte Carlo trials')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    # Parse weeks
    if '-' in args.weeks:
        start, end = map(int, args.weeks.split('-'))
        weeks = list(range(start, end + 1))
    else:
        weeks = [int(args.weeks)]

    # Run validation
    results = run_validation(weeks=weeks, trials=args.trials)

    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to: {args.output}")
    else:
        output_path = PROJECT_ROOT / 'reports' / 'walk_forward_v4_validation.csv'
        results.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
