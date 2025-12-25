#!/usr/bin/env python3
"""
EPA to Points Calibration Script

Derives the optimal EPA_TO_POINTS_FACTOR from historical game data.
Replaces the hardcoded 3.5 factor in game_line_edge.py.

The factor converts EPA differential to expected point differential:
    predicted_margin = net_EPA * EPA_TO_POINTS_FACTOR + home_field_advantage

This script:
1. Loads historical game results (2021-2024)
2. Calculates team EPA for each game using prior data only
3. Fits linear regression: actual_margin ~ net_EPA
4. Extracts the optimal conversion factor
5. Saves to data/models/calibrated_epa_factor.joblib

Usage:
    python scripts/calibration/calibrate_epa_factor.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import PROJECT_ROOT, MODELS_DIR


def load_historical_games(seasons: list = None) -> pd.DataFrame:
    """Load historical game results from schedules."""
    if seasons is None:
        # Use 2021-2024 for calibration (2025 is test data)
        seasons = [2021, 2022, 2023, 2024]

    schedules_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if not schedules_path.exists():
        raise FileNotFoundError(f"Schedules not found at {schedules_path}")

    schedules = pd.read_parquet(schedules_path)

    # Filter to completed games in target seasons
    games = schedules[
        (schedules['season'].isin(seasons)) &
        (schedules['home_score'].notna()) &
        (schedules['away_score'].notna())
    ].copy()

    # Calculate actual margin (positive = home win)
    games['actual_margin'] = games['home_score'] - games['away_score']

    print(f"Loaded {len(games):,} completed games from seasons {seasons}")
    return games


def calculate_team_epa_for_game(
    pbp_df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    lookback_weeks: int = 6
) -> dict:
    """
    Calculate team EPA using only data BEFORE the game week.

    Args:
        pbp_df: Play-by-play DataFrame
        team: Team abbreviation
        season: Season year
        week: Game week (use data before this)
        lookback_weeks: Number of weeks to look back

    Returns:
        Dict with 'off_epa', 'def_epa', 'games'
    """
    # Filter to plays BEFORE the game week
    min_week = max(1, week - lookback_weeks)

    # Get plays from same season, before game week
    season_pbp = pbp_df[
        (pbp_df['season'] == season) &
        (pbp_df['week'] >= min_week) &
        (pbp_df['week'] < week) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    # Offensive plays
    off_plays = season_pbp[season_pbp['posteam'] == team]

    # Defensive plays
    def_plays = season_pbp[season_pbp['defteam'] == team]

    if len(off_plays) == 0 or len(def_plays) == 0:
        return {'off_epa': 0.0, 'def_epa': 0.0, 'games': 0}

    # Calculate EPA per play
    off_epa = off_plays['epa'].mean()
    def_epa = def_plays['epa'].mean()
    games = len(off_plays['game_id'].unique())

    # Regress toward 0 for small samples
    regression_factor = min(1.0, games / 6)
    off_epa *= regression_factor
    def_epa *= regression_factor

    return {'off_epa': off_epa, 'def_epa': def_epa, 'games': games}


def build_calibration_dataset(games: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset with EPA differentials and actual margins.

    For each game, calculates:
    - Home team EPA (off/def) before the game
    - Away team EPA (off/def) before the game
    - Net EPA differential
    - Actual margin
    """
    records = []

    total_games = len(games)
    print(f"Processing {total_games} games...")

    for idx, game in games.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total_games}")

        home_team = game['home_team']
        away_team = game['away_team']
        season = game['season']
        week = game['week']

        # Skip early weeks (not enough data)
        if week <= 3:
            continue

        # Calculate EPA for each team BEFORE this game
        home_epa = calculate_team_epa_for_game(pbp_df, home_team, season, week)
        away_epa = calculate_team_epa_for_game(pbp_df, away_team, season, week)

        # Skip if insufficient data
        if home_epa['games'] < 2 or away_epa['games'] < 2:
            continue

        # Net EPA differential
        # Home power = home_off - away_def (how home performs vs away defense)
        # Away power = away_off - home_def (how away performs vs home defense)
        home_power = home_epa['off_epa'] - away_epa['def_epa']
        away_power = away_epa['off_epa'] - home_epa['def_epa']
        net_epa = home_power - away_power

        records.append({
            'game_id': game['game_id'],
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'home_off_epa': home_epa['off_epa'],
            'home_def_epa': home_epa['def_epa'],
            'away_off_epa': away_epa['off_epa'],
            'away_def_epa': away_epa['def_epa'],
            'net_epa': net_epa,
            'actual_margin': game['actual_margin'],
            'home_rest': game.get('home_rest', 7),
            'away_rest': game.get('away_rest', 7),
            'div_game': game.get('div_game', False),
        })

    df = pd.DataFrame(records)
    print(f"Built calibration dataset with {len(df)} games")
    return df


def calibrate_epa_factor(df: pd.DataFrame) -> dict:
    """
    Fit linear regression to find optimal EPA factor.

    Model: actual_margin = β0 + β1 * net_EPA

    The coefficient β1 is the optimal EPA_TO_POINTS_FACTOR.
    The intercept β0 should be close to home_field_advantage.
    """
    X = df[['net_epa']].values
    y = df['actual_margin'].values

    # Fit simple linear regression
    model = LinearRegression()
    model.fit(X, y)

    epa_factor = model.coef_[0]
    intercept = model.intercept_

    # Calculate metrics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # R-squared
    r_squared = model.score(X, y)

    print(f"\n{'='*60}")
    print("EPA CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOptimal EPA_TO_POINTS_FACTOR: {epa_factor:.3f}")
    print(f"Intercept (home field advantage): {intercept:.2f}")
    print(f"\nModel Fit:")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  MAE (training): {mae:.2f} points")
    print(f"  MAE (5-fold CV): {cv_mae:.2f} points")
    print(f"\nComparison to hardcoded factor (3.5):")
    print(f"  Old factor: 3.5")
    print(f"  New factor: {epa_factor:.3f}")
    print(f"  Change: {(epa_factor - 3.5):.3f} ({((epa_factor - 3.5) / 3.5) * 100:.1f}%)")

    return {
        'epa_to_points_factor': epa_factor,
        'home_field_advantage': intercept,
        'r_squared': r_squared,
        'mae_training': mae,
        'mae_cv': cv_mae,
        'n_games': len(df),
        'seasons': list(df['season'].unique()),
        'calibrated_date': datetime.now().isoformat(),
    }


def calibrate_rest_adjustments(df: pd.DataFrame) -> dict:
    """
    Calibrate rest day adjustments empirically.

    Analyzes how rest differential affects game margin.
    """
    # Create rest differential (positive = home has more rest)
    df = df.copy()
    df['rest_diff'] = df['home_rest'] - df['away_rest']

    # Group by rest differential and calculate average margin
    rest_impact = df.groupby('rest_diff').agg({
        'actual_margin': ['mean', 'count']
    }).reset_index()
    rest_impact.columns = ['rest_diff', 'avg_margin', 'n_games']

    # Filter to meaningful sample sizes
    rest_impact = rest_impact[rest_impact['n_games'] >= 20]

    print(f"\n{'='*60}")
    print("REST ADJUSTMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"\nRest Differential Impact on Home Margin:")
    for _, row in rest_impact.sort_values('rest_diff').iterrows():
        print(f"  Rest diff {int(row['rest_diff']):+d}d: {row['avg_margin']:+.1f} pts (n={int(row['n_games'])})")

    # Calculate per-day rest value
    if len(rest_impact) > 2:
        X = rest_impact[['rest_diff']].values
        y = rest_impact['avg_margin'].values
        model = LinearRegression()
        model.fit(X, y)
        per_day_value = model.coef_[0]
        print(f"\n  Per-day rest value: {per_day_value:.2f} points")
    else:
        per_day_value = 0.3  # Default

    return {
        'rest_diff_impact': rest_impact.to_dict('records'),
        'per_day_rest_value': per_day_value,
    }


def calibrate_divisional_shrinkage(df: pd.DataFrame) -> dict:
    """
    Calibrate divisional game shrinkage factor.

    Divisional games tend to be closer than model predicts.
    """
    df = df.copy()

    # Compare divisional vs non-divisional
    div_games = df[df['div_game'] == True]
    non_div_games = df[df['div_game'] == False]

    if len(div_games) < 50:
        print("\nInsufficient divisional games for analysis")
        return {'divisional_shrinkage': 0.7}  # Keep default

    # Calculate variance in margin for each group
    div_margin_std = div_games['actual_margin'].std()
    non_div_margin_std = non_div_games['actual_margin'].std()

    # Shrinkage factor = ratio of stds
    shrinkage = div_margin_std / non_div_margin_std

    print(f"\n{'='*60}")
    print("DIVISIONAL GAME ANALYSIS")
    print(f"{'='*60}")
    print(f"\nDivisional games (n={len(div_games)}):")
    print(f"  Avg margin (abs): {div_games['actual_margin'].abs().mean():.1f}")
    print(f"  Margin std: {div_margin_std:.1f}")
    print(f"\nNon-divisional games (n={len(non_div_games)}):")
    print(f"  Avg margin (abs): {non_div_games['actual_margin'].abs().mean():.1f}")
    print(f"  Margin std: {non_div_margin_std:.1f}")
    print(f"\nOptimal divisional shrinkage: {shrinkage:.3f}")
    print(f"  (Current hardcoded: 0.7)")

    return {'divisional_shrinkage': shrinkage}


def main():
    """Run EPA calibration pipeline."""
    print("="*60)
    print("EPA TO POINTS FACTOR CALIBRATION")
    print("="*60)

    # Load PBP data
    pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
    if not pbp_path.exists():
        # Try season-specific files
        pbp_dfs = []
        for season in [2021, 2022, 2023, 2024]:
            season_path = PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{season}.parquet'
            if season_path.exists():
                pbp_dfs.append(pd.read_parquet(season_path))
        if pbp_dfs:
            pbp_df = pd.concat(pbp_dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No PBP data found")
    else:
        pbp_df = pd.read_parquet(pbp_path)

    print(f"\nLoaded PBP data: {len(pbp_df):,} plays")

    # Load historical games
    games = load_historical_games([2021, 2022, 2023, 2024])

    # Build calibration dataset
    cal_df = build_calibration_dataset(games, pbp_df)

    # Calibrate EPA factor
    epa_results = calibrate_epa_factor(cal_df)

    # Calibrate rest adjustments
    rest_results = calibrate_rest_adjustments(cal_df)

    # Calibrate divisional shrinkage
    div_results = calibrate_divisional_shrinkage(cal_df)

    # Combine all results
    calibration_results = {
        **epa_results,
        **rest_results,
        **div_results,
    }

    # Save results
    output_path = MODELS_DIR / 'calibrated_epa_factor.joblib'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibration_results, output_path)
    print(f"\n{'='*60}")
    print(f"Saved calibration results to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    print(f"\nFINAL CALIBRATED VALUES:")
    print(f"  EPA_TO_POINTS_FACTOR: {calibration_results['epa_to_points_factor']:.3f}")
    print(f"  HOME_FIELD_ADVANTAGE: {calibration_results['home_field_advantage']:.2f}")
    print(f"  PER_DAY_REST_VALUE: {calibration_results['per_day_rest_value']:.2f}")
    print(f"  DIVISIONAL_SHRINKAGE: {calibration_results['divisional_shrinkage']:.3f}")

    return calibration_results


if __name__ == '__main__':
    main()
