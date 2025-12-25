#!/usr/bin/env python3
"""
Train Game Line Calibrator with Historical Data
================================================

This script:
1. Loads historical schedules data from nflverse
2. Runs simulations for completed games
3. Compares model predictions to actual outcomes
4. Trains an isotonic regression calibrator
5. Saves the calibrator for use in recommendations

Requires: Completed games from previous seasons/weeks
Output: configs/game_line_calibrator.json

Usage:
    python scripts/train/train_game_line_calibrator.py
    python scripts/train/train_game_line_calibrator.py --seasons 2024 2023 --min-games 50
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput
from nfl_quant.utils.season_utils import get_current_season, get_current_week
from nfl_quant.utils.epa_utils import regress_epa_to_mean

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_schedules(seasons: list) -> pd.DataFrame:
    """Load schedule data from nflverse."""
    schedules_path = PROJECT_ROOT / "data" / "nflverse" / "schedules.parquet"

    if not schedules_path.exists():
        raise FileNotFoundError(f"Schedules file not found: {schedules_path}")

    df = pd.read_parquet(schedules_path)
    df = df[df['season'].isin(seasons)]

    # Filter to completed games only
    df = df[
        (df['home_score'].notna()) &
        (df['away_score'].notna()) &
        (df['game_type'] == 'REG')  # Regular season only
    ]

    return df


def calculate_team_epa_for_game(pbp_df: pd.DataFrame, team: str, season: int, week: int, weeks: int = 6) -> dict:
    """
    Calculate team EPA metrics using data available BEFORE the specified week.

    CRITICAL: Only uses weeks < current week to prevent leakage.
    """
    available_weeks = list(range(max(1, week - weeks), week))

    if len(available_weeks) == 0:
        return {
            'offensive_epa': 0.0,
            'defensive_epa': 0.0,
            'pace': 65.0,
            'sample_games': 0
        }

    # Filter to relevant season and weeks
    team_pbp = pbp_df[
        (pbp_df['season'] == season) &
        (pbp_df['week'].isin(available_weeks))
    ]

    # Offensive EPA
    off_plays = team_pbp[
        (team_pbp['posteam'] == team) &
        (team_pbp['play_type'].isin(['pass', 'run']))
    ]

    # Defensive EPA
    def_plays = team_pbp[
        (team_pbp['defteam'] == team) &
        (team_pbp['play_type'].isin(['pass', 'run']))
    ]

    if len(off_plays) == 0 or len(def_plays) == 0:
        return {
            'offensive_epa': 0.0,
            'defensive_epa': 0.0,
            'pace': 65.0,
            'sample_games': 0
        }

    sample_games = min(
        len(off_plays['game_id'].unique()),
        len(def_plays['game_id'].unique())
    )

    raw_off_epa = off_plays['epa'].mean()
    raw_def_epa = def_plays['epa'].mean()

    off_epa = regress_epa_to_mean(raw_off_epa, sample_games)
    def_epa = regress_epa_to_mean(raw_def_epa, sample_games)

    off_games = len(off_plays['game_id'].unique())
    pace = len(off_plays) / off_games if off_games > 0 else 65.0

    return {
        'offensive_epa': float(off_epa),
        'defensive_epa': float(def_epa),
        'pace': float(pace),
        'sample_games': sample_games
    }


def run_simulation_for_game(
    game_row: pd.Series,
    pbp_df: pd.DataFrame,
    simulator: MonteCarloSimulator
) -> dict:
    """Run simulation for a single completed game and return prediction vs actual."""

    season = int(game_row['season'])
    week = int(game_row['week'])
    home_team = game_row['home_team']
    away_team = game_row['away_team']
    home_score = float(game_row['home_score'])
    away_score = float(game_row['away_score'])

    # Calculate EPA using only prior data
    home_epa = calculate_team_epa_for_game(pbp_df, home_team, season, week)
    away_epa = calculate_team_epa_for_game(pbp_df, away_team, season, week)

    # Skip if insufficient data
    if home_epa['sample_games'] < 2 or away_epa['sample_games'] < 2:
        return None

    # Create simulation input
    sim_input = SimulationInput(
        game_id=game_row['game_id'],
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
        home_offensive_epa=home_epa['offensive_epa'],
        home_defensive_epa=home_epa['defensive_epa'],
        away_offensive_epa=away_epa['offensive_epa'],
        away_defensive_epa=away_epa['defensive_epa'],
        home_pace=home_epa['pace'],
        away_pace=away_epa['pace'],
        home_injury_offensive_adjustment=0.0,
        home_injury_defensive_adjustment=0.0,
        away_injury_offensive_adjustment=0.0,
        away_injury_defensive_adjustment=0.0,
        is_divisional=False,
        game_type='Regular',
        is_dome=False,
        temperature=None,
        wind_speed=None,
        precipitation=None
    )

    # Run simulation
    sim_output = simulator.simulate_game(sim_input, trials=10000)

    # Actual outcomes
    home_won = 1 if home_score > away_score else 0
    actual_spread = home_score - away_score
    actual_total = home_score + away_score

    return {
        'game_id': game_row['game_id'],
        'season': season,
        'week': week,
        'home_team': home_team,
        'away_team': away_team,
        # Predictions
        'pred_home_win_prob': sim_output.home_win_prob,
        'pred_spread': sim_output.fair_spread,
        'pred_total': sim_output.fair_total,
        # Actuals
        'actual_home_won': home_won,
        'actual_spread': actual_spread,
        'actual_total': actual_total,
        # For debugging
        'home_off_epa': home_epa['offensive_epa'],
        'home_def_epa': home_epa['defensive_epa'],
        'away_off_epa': away_epa['offensive_epa'],
        'away_def_epa': away_epa['defensive_epa'],
    }


def train_calibrator(predictions_df: pd.DataFrame) -> dict:
    """
    Train isotonic regression calibrator for win probabilities.

    Returns dict with X_thresholds, y_thresholds for use in apply_calibration().
    """
    # Filter to valid predictions
    valid = predictions_df[
        (predictions_df['pred_home_win_prob'].notna()) &
        (predictions_df['actual_home_won'].notna())
    ].copy()

    if len(valid) < 20:
        raise ValueError(f"Need at least 20 samples, got {len(valid)}")

    X = valid['pred_home_win_prob'].values
    y = valid['actual_home_won'].values

    # Train isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(X, y)

    # Extract thresholds for serialization
    # Isotonic regression creates a piecewise linear function
    X_thresholds = ir.X_thresholds_.tolist()
    y_thresholds = ir.y_thresholds_.tolist()

    # Calculate calibration metrics
    calibrated_probs = ir.predict(X)
    brier_before = np.mean((X - y) ** 2)
    brier_after = np.mean((calibrated_probs - y) ** 2)

    print(f"\nCalibration Results:")
    print(f"  Samples: {len(valid)}")
    print(f"  Brier Score (before): {brier_before:.4f}")
    print(f"  Brier Score (after): {brier_after:.4f}")
    print(f"  Improvement: {(brier_before - brier_after) / brier_before * 100:.1f}%")

    return {
        'X_thresholds': X_thresholds,
        'y_thresholds': y_thresholds,
        'X_min': float(min(X)),
        'X_max': float(max(X)),
        'trained_at': datetime.now().isoformat(),
        'n_samples': len(valid),
        'brier_before': float(brier_before),
        'brier_after': float(brier_after),
    }


def main():
    parser = argparse.ArgumentParser(description='Train Game Line Calibrator')
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help='Seasons to use (default: current and previous)')
    parser.add_argument('--min-games', type=int, default=50,
                        help='Minimum games required (default: 50)')
    parser.add_argument('--max-games', type=int, default=500,
                        help='Maximum games to process (default: 500)')
    args = parser.parse_args()

    current_season = get_current_season()

    if args.seasons is None:
        # Use current season + previous season
        args.seasons = [current_season, current_season - 1]

    print("=" * 70)
    print("TRAINING GAME LINE CALIBRATOR")
    print("=" * 70)
    print(f"Seasons: {args.seasons}")
    print(f"Min games: {args.min_games}")
    print()

    # Load schedules
    print("Loading schedules...")
    schedules = load_schedules(args.seasons)
    print(f"Found {len(schedules)} completed regular season games")

    # Load PBP data
    print("\nLoading play-by-play data...")
    pbp_dfs = []
    for season in args.seasons:
        pbp_path = PROJECT_ROOT / "data" / "nflverse" / f"pbp_{season}.parquet"
        if pbp_path.exists():
            df = pd.read_parquet(pbp_path)
            pbp_dfs.append(df)
            print(f"  Loaded {len(df):,} plays from {season}")
        else:
            print(f"  Warning: PBP file not found for {season}")

    if not pbp_dfs:
        raise FileNotFoundError("No PBP data found")

    pbp_df = pd.concat(pbp_dfs, ignore_index=True)
    print(f"Total: {len(pbp_df):,} plays")

    # Sample games if too many (for speed)
    if len(schedules) > args.max_games:
        print(f"\nSampling {args.max_games} games from {len(schedules)} available...")
        schedules = schedules.sample(n=args.max_games, random_state=42)

    # Run simulations
    print(f"\nRunning simulations for {len(schedules)} games...")
    simulator = MonteCarloSimulator(seed=42)

    results = []
    for idx, (_, game_row) in enumerate(schedules.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(schedules)} games...")

        result = run_simulation_for_game(game_row, pbp_df, simulator)
        if result:
            results.append(result)

    print(f"\nSuccessfully simulated {len(results)} games")

    if len(results) < args.min_games:
        raise ValueError(f"Need at least {args.min_games} games, only got {len(results)}")

    # Create DataFrame
    predictions_df = pd.DataFrame(results)

    # Train calibrator
    print("\nTraining calibrator...")
    calibrator = train_calibrator(predictions_df)

    # Save calibrator
    output_path = PROJECT_ROOT / "configs" / "game_line_calibrator.json"

    # Backup old calibrator
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        output_path.rename(backup_path)
        print(f"\nBacked up old calibrator to: {backup_path}")

    with open(output_path, 'w') as f:
        json.dump(calibrator, f, indent=2)

    print(f"\nSaved new calibrator to: {output_path}")
    print(f"  Samples: {calibrator['n_samples']}")
    print(f"  X range: [{calibrator['X_min']:.3f}, {calibrator['X_max']:.3f}]")

    # Also save predictions for analysis
    predictions_path = PROJECT_ROOT / "reports" / "game_line_calibration_data.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nSaved calibration data to: {predictions_path}")


if __name__ == '__main__':
    main()
