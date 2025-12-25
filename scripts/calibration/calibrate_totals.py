#!/usr/bin/env python3
"""
Totals Model Calibration Script

Calibrates the totals model parameters by grid-searching:
- points_per_play: Base points per play
- epa_total_factor: How much EPA affects totals

Evaluates on historical data and saves optimal parameters.

Usage:
    python scripts/calibration/calibrate_totals.py
    python scripts/calibration/calibrate_totals.py --seasons 2023 2024
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from itertools import product
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import DATA_DIR, MODELS_DIR


def load_data(seasons: list = None) -> tuple:
    """Load PBP and schedule data for calibration."""
    if seasons is None:
        seasons = [2023, 2024]

    # Load PBP
    pbp_path = DATA_DIR / 'nflverse' / 'pbp.parquet'
    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP not found at {pbp_path}")

    pbp_df = pd.read_parquet(pbp_path)
    pbp_df = pbp_df[pbp_df['season'].isin(seasons)]
    print(f"Loaded {len(pbp_df):,} plays from seasons {seasons}")

    # Load schedule with actual scores
    schedule_path = DATA_DIR / 'nflverse' / 'schedules.parquet'
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found at {schedule_path}")

    schedule_df = pd.read_parquet(schedule_path)
    schedule_df = schedule_df[
        (schedule_df['season'].isin(seasons)) &
        (schedule_df['home_score'].notna()) &
        (schedule_df['away_score'].notna())
    ].copy()

    schedule_df['actual_total'] = schedule_df['home_score'] + schedule_df['away_score']
    print(f"Loaded {len(schedule_df):,} completed games")

    return pbp_df, schedule_df


def calculate_team_epa(
    pbp_df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    lookback_weeks: int = 6
) -> dict:
    """Calculate team EPA using only data BEFORE the game week."""
    min_week = max(1, week - lookback_weeks)

    recent_pbp = pbp_df[
        (pbp_df['season'] == season) &
        (pbp_df['week'] >= min_week) &
        (pbp_df['week'] < week) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    off_plays = recent_pbp[recent_pbp['posteam'] == team]
    def_plays = recent_pbp[recent_pbp['defteam'] == team]

    if len(off_plays) == 0 or len(def_plays) == 0:
        return {'off_epa': 0.0, 'def_epa_allowed': 0.0, 'pace': 60.0, 'games': 0}

    off_epa = off_plays['epa'].mean()
    def_epa_allowed = def_plays['epa'].mean()

    games = len(off_plays['game_id'].unique())
    pace = len(off_plays) / games if games > 0 else 60.0

    regression_factor = min(1.0, games / 6)
    off_epa *= regression_factor
    def_epa_allowed *= regression_factor

    return {
        'off_epa': off_epa,
        'def_epa_allowed': def_epa_allowed,
        'pace': pace,
        'games': games
    }


def project_total(
    home_epa: dict,
    away_epa: dict,
    points_per_play: float,
    epa_total_factor: float,
    is_dome: bool = False,
    temperature: float = None,
) -> tuple:
    """Project game total using the V29 formula."""
    home_pace = home_epa.get('pace', 60.0)
    away_pace = away_epa.get('pace', 60.0)

    plays_total = home_pace + away_pace
    ppp_baseline = points_per_play

    # EPA adjustment
    combined_off_epa = home_epa['off_epa'] + away_epa['off_epa']
    combined_def_epa_allowed = home_epa['def_epa_allowed'] + away_epa['def_epa_allowed']
    epa_efficiency = combined_off_epa + combined_def_epa_allowed
    ppp_epa_adj = epa_efficiency * (epa_total_factor / 100)

    # Weather adjustment
    ppp_weather_adj = 0.0
    if is_dome:
        ppp_weather_adj += 1.5 / plays_total
    elif temperature is not None and temperature < 32:
        ppp_weather_adj += -2.0 / plays_total

    ppp = ppp_baseline + ppp_epa_adj + ppp_weather_adj
    ppp = max(0.30, min(0.50, ppp))

    raw_total = plays_total * ppp
    clipped_total = max(30.0, min(70.0, raw_total))
    was_clipped = (clipped_total != raw_total)

    return clipped_total, raw_total, was_clipped


def evaluate_parameters(
    pbp_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    points_per_play: float,
    epa_total_factor: float,
    min_week: int = 5,
) -> dict:
    """Evaluate a set of parameters on historical data."""
    results = []
    clip_count = 0

    for _, game in schedule_df.iterrows():
        season = game['season']
        week = game['week']

        if week < min_week:
            continue

        home_team = game['home_team']
        away_team = game['away_team']
        actual_total = game['actual_total']

        # Get EPA using only prior data
        home_epa = calculate_team_epa(pbp_df, home_team, season, week)
        away_epa = calculate_team_epa(pbp_df, away_team, season, week)

        if home_epa['games'] < 3 or away_epa['games'] < 3:
            continue

        # Weather context
        roof = str(game.get('roof', '')).lower()
        is_dome = roof in ['dome', 'closed', 'retractable roof closed']
        temp = game.get('temp')
        temperature = float(temp) if pd.notna(temp) else (72.0 if is_dome else None)

        # Project total
        model_total, raw_total, was_clipped = project_total(
            home_epa, away_epa, points_per_play, epa_total_factor,
            is_dome=is_dome, temperature=temperature
        )

        if was_clipped:
            clip_count += 1

        results.append({
            'game_id': game.get('game_id', f"{away_team}@{home_team}"),
            'actual_total': actual_total,
            'model_total': model_total,
            'raw_total': raw_total,
            'error': model_total - actual_total,
            'abs_error': abs(model_total - actual_total),
        })

    if len(results) == 0:
        return {'mae': float('inf'), 'bias': 0, 'rmse': float('inf'), 'clip_rate': 0, 'n_games': 0}

    df = pd.DataFrame(results)

    mae = df['abs_error'].mean()
    bias = df['error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    clip_rate = clip_count / len(df) * 100

    return {
        'mae': mae,
        'bias': bias,
        'rmse': rmse,
        'clip_rate': clip_rate,
        'n_games': len(df),
    }


def grid_search_parameters(
    pbp_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> dict:
    """Grid search to find optimal parameters."""
    print("\n" + "="*60)
    print("GRID SEARCH: Totals Calibration")
    print("="*60)

    # Parameter grid
    ppp_values = [0.36, 0.37, 0.38, 0.39, 0.40]
    epa_factor_values = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    best_mae = float('inf')
    best_params = None
    all_results = []

    total_combos = len(ppp_values) * len(epa_factor_values)
    print(f"Testing {total_combos} parameter combinations...")

    for i, (ppp, epa_factor) in enumerate(product(ppp_values, epa_factor_values)):
        metrics = evaluate_parameters(pbp_df, schedule_df, ppp, epa_factor)

        all_results.append({
            'points_per_play': ppp,
            'epa_total_factor': epa_factor,
            **metrics
        })

        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_params = {
                'points_per_play': ppp,
                'epa_total_factor': epa_factor,
                **metrics
            }

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total_combos}")

    return best_params, all_results


def main():
    parser = argparse.ArgumentParser(description="Calibrate Totals Model")
    parser.add_argument('--seasons', type=int, nargs='+', default=[2023, 2024],
                        help='Seasons to use for calibration')
    parser.add_argument('--output', type=str, help='Output path for calibration file')
    args = parser.parse_args()

    print("="*60)
    print("TOTALS MODEL CALIBRATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Load data
    pbp_df, schedule_df = load_data(args.seasons)

    # Grid search
    best_params, all_results = grid_search_parameters(pbp_df, schedule_df)

    # Print results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    print(f"\nOptimal Parameters:")
    print(f"  points_per_play: {best_params['points_per_play']:.3f}")
    print(f"  epa_total_factor: {best_params['epa_total_factor']:.1f}")

    print(f"\nModel Performance:")
    print(f"  MAE: {best_params['mae']:.2f} points")
    print(f"  Bias: {best_params['bias']:+.2f} points")
    print(f"  RMSE: {best_params['rmse']:.2f} points")
    print(f"  Clip Rate: {best_params['clip_rate']:.1f}%")
    print(f"  Games Evaluated: {best_params['n_games']}")

    # Compare to baseline (no EPA adjustment)
    baseline_metrics = evaluate_parameters(pbp_df, schedule_df, 0.38, 0.0)
    print(f"\nBaseline (no EPA adjustment):")
    print(f"  MAE: {baseline_metrics['mae']:.2f} points")
    print(f"  Improvement: {baseline_metrics['mae'] - best_params['mae']:.2f} points")

    # Save calibration
    calibration = {
        'points_per_play': best_params['points_per_play'],
        'epa_total_factor': best_params['epa_total_factor'],
        'mae': best_params['mae'],
        'bias': best_params['bias'],
        'rmse': best_params['rmse'],
        'clip_rate': best_params['clip_rate'],
        'n_games': best_params['n_games'],
        'seasons': args.seasons,
        'calibrated_date': datetime.now().isoformat(),
    }

    output_path = args.output if args.output else MODELS_DIR / 'calibrated_totals_factor.joblib'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibration, output_path)
    print(f"\nSaved calibration to: {output_path}")

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_path = DATA_DIR / 'calibration' / 'totals_grid_search.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved grid search results to: {results_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return calibration


if __name__ == '__main__':
    main()
