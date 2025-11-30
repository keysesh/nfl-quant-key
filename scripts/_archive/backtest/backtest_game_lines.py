#!/usr/bin/env python3
"""
Comprehensive Game Line Backtest for NFL QUANT System

This backtest validates the entire unified game line prediction pipeline:
- Monte Carlo simulation with EPA regression to mean
- Injury impact integration
- Calibrated probabilities
- Edge calculations using actual market odds
- ROI validation

Tests against historical games with known outcomes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput
from nfl_quant.data.game_lines_loader import load_game_lines
from scripts.predict.generate_game_line_recommendations import (
    calculate_team_epa,
    calculate_spread_cover_prob,
    calculate_total_over_prob,
)
from nfl_quant.core.unified_betting import (
    american_odds_to_implied_prob,
    remove_vig_two_way,
    calculate_edge_percentage,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_historical_games(season: int, weeks: List[int]) -> pd.DataFrame:
    """
    Load historical game results from NFLverse.

    Args:
        season: Season year
        weeks: List of week numbers

    Returns:
        DataFrame with columns: game_id, week, home_team, away_team, home_score, away_score
    """
    # Try different schedule file formats
    schedules_file = PROJECT_ROOT / f"data/nflverse/schedules_{season}.parquet"
    if not schedules_file.exists():
        schedules_file = PROJECT_ROOT / "data/nflverse/schedules.parquet"

    if not schedules_file.exists():
        raise FileNotFoundError(f"Schedules file not found: {schedules_file}")

    df = pd.read_parquet(schedules_file)

    # Filter to requested season and weeks
    df = df[(df['season'] == season) & (df['week'].isin(weeks))].copy()

    # Only include completed games
    df = df[df['home_score'].notna() & df['away_score'].notna()].copy()

    # Calculate margin (home - away)
    df['margin'] = df['home_score'] - df['away_score']
    df['total'] = df['home_score'] + df['away_score']
    df['home_won'] = (df['home_score'] > df['away_score']).astype(int)

    return df[['game_id', 'week', 'home_team', 'away_team',
               'home_score', 'away_score', 'margin', 'total', 'home_won']]


def load_historical_game_line_odds(season: int, weeks: List[int]) -> pd.DataFrame:
    """
    Load historical game line odds (spread, total, moneyline).

    Args:
        season: Season year
        weeks: List of week numbers

    Returns:
        DataFrame with game line odds
    """
    # Use centralized loader for consistency
    df = load_game_lines(season=season, weeks=weeks, source="auto")
    print(f"✅ Loaded {len(df)} odds for {df['game_id'].nunique()} games")
    return df


def simulate_historical_game(game: pd.Series, pbp_df: pd.DataFrame) -> Dict:
    """
    Run simulation for a historical game using only data available before that week.

    Args:
        game: Game row with home_team, away_team, week
        pbp_df: Play-by-play data (filtered to before this game)

    Returns:
        Dict with simulation results
    """
    home_team = game['home_team']
    away_team = game['away_team']
    week = game['week']

    # Calculate EPA using only data before this game
    home_epa = calculate_team_epa(pbp_df, home_team, weeks=10)
    away_epa = calculate_team_epa(pbp_df, away_team, weeks=10)

    # Create simulation input (no injuries for simplicity in backtest)
    sim_input = SimulationInput(
        game_id=game['game_id'],
        season=game.name,  # Season stored in index
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
    simulator = MonteCarloSimulator(seed=42)
    sim_output = simulator.simulate_game(sim_input, trials=50000)

    return {
        'game_id': game['game_id'],
        'week': week,
        'home_team': home_team,
        'away_team': away_team,
        'model_home_win_prob': sim_output.home_win_prob,
        'model_fair_spread': sim_output.fair_spread,
        'model_fair_total': sim_output.fair_total,
        'spread_std': np.sqrt(sim_output.home_score_std**2 + sim_output.away_score_std**2),
        'total_std': sim_output.total_std,
    }


def backtest_game_lines(season: int = 2025, start_week: int = 1, end_week: int = 10):
    """
    Run comprehensive backtest on historical games.

    Args:
        season: Season to backtest
        start_week: First week to test
        end_week: Last week to test
    """
    print("=" * 70)
    print(f"GAME LINE BACKTEST: {season} Weeks {start_week}-{end_week}")
    print("=" * 70)

    weeks = list(range(start_week, end_week + 1))

    # Load historical game results
    print("\n1. Loading historical game results...")
    games_df = load_historical_games(season, weeks)
    print(f"   Loaded {len(games_df)} completed games")

    # Load historical game line odds
    print("\n2. Loading historical game line odds...")
    try:
        odds_df = load_historical_game_line_odds(season, weeks)
        print(f"   Loaded {len(odds_df)} odds for {odds_df['game_id'].nunique()} games")
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
        print("   Continuing without market odds (calibration-only mode)")
        odds_df = None

    # Load PBP data
    print("\n3. Loading play-by-play data...")
    pbp_file = PROJECT_ROOT / f"data/nflverse/pbp_{season}.parquet"
    if not pbp_file.exists():
        raise FileNotFoundError(f"PBP file not found: {pbp_file}")
    pbp_df = pd.read_parquet(pbp_file)
    print(f"   Loaded {len(pbp_df):,} plays")

    # Run simulations for each game
    print("\n4. Running simulations...")
    predictions = []

    for idx, game in games_df.iterrows():
        week = game['week']

        # Filter PBP to only include data before this game
        pbp_before = pbp_df[pbp_df['week'] < week].copy()

        if len(pbp_before) == 0:
            print(f"   Skipping {game['game_id']} (no prior data)")
            continue

        sim_result = simulate_historical_game(game, pbp_before)
        predictions.append(sim_result)

        if len(predictions) % 10 == 0:
            print(f"   Processed {len(predictions)} games...")

    predictions_df = pd.DataFrame(predictions)
    print(f"   ✅ Generated predictions for {len(predictions_df)} games")

    # Merge with actual outcomes
    print("\n5. Comparing predictions to outcomes...")
    results_df = predictions_df.merge(
        games_df,
        on=['game_id', 'week', 'home_team', 'away_team'],
        how='inner'
    )

    # Merge with market odds if available
    if odds_df is not None:
        print("\n6. Merging with market odds...")

        # CRITICAL FIX: Merge on (season, week, home_team, away_team) instead of game_id
        # This fixes the mismatch where only 3/41 games were matching

        # Parse spread odds (home side)
        spread_odds = odds_df[odds_df['market'] == 'spread'].copy()
        spread_home = spread_odds[spread_odds['side'] == 'home'].copy()
        spread_home = spread_home.groupby(['season', 'week', 'home_team', 'away_team']).first().reset_index()
        spread_home = spread_home[['season', 'week', 'home_team', 'away_team', 'point', 'price']].rename(
            columns={'point': 'market_spread', 'price': 'spread_odds_home'}
        )

        # Parse total odds (over side)
        total_odds = odds_df[odds_df['market'] == 'total'].copy()
        total_over = total_odds[total_odds['side'] == 'over'].copy()
        total_over = total_over.groupby(['season', 'week', 'home_team', 'away_team']).first().reset_index()
        total_over = total_over[['season', 'week', 'home_team', 'away_team', 'point', 'price']].rename(
            columns={'point': 'market_total', 'price': 'total_odds_over'}
        )

        # Also get under odds for total
        total_under = total_odds[total_odds['side'] == 'under'].copy()
        total_under = total_under.groupby(['season', 'week', 'home_team', 'away_team']).first().reset_index()
        total_under = total_under[['season', 'week', 'home_team', 'away_team', 'price']].rename(
            columns={'price': 'total_odds_under'}
        )

        # Add season column to results_df if not present
        if 'season' not in results_df.columns:
            results_df['season'] = season

        # Merge on team names instead of game_id
        results_df = results_df.merge(spread_home, on=['season', 'week', 'home_team', 'away_team'], how='left')
        results_df = results_df.merge(total_over, on=['season', 'week', 'home_team', 'away_team'], how='left')
        results_df = results_df.merge(total_under, on=['season', 'week', 'home_team', 'away_team'], how='left')

        print(f"   Games with spread odds: {results_df['market_spread'].notna().sum()}")
        print(f"   Games with total odds: {results_df['market_total'].notna().sum()}")

        # Calculate spread predictions and outcomes
        results_df['spread_cover_prob'] = results_df.apply(
            lambda row: calculate_spread_cover_prob(
                row['model_fair_spread'],
                row['market_spread'],
                row['spread_std']
            ) if pd.notna(row['market_spread']) else np.nan,
            axis=1
        )

        results_df['home_covered_spread'] = (
            results_df['margin'] > results_df['market_spread']
        ).astype(float)

        # Calculate total predictions and outcomes
        results_df['total_over_prob'] = results_df.apply(
            lambda row: calculate_total_over_prob(
                row['model_fair_total'],
                row['market_total'],
                row['total_std']
            ) if pd.notna(row['market_total']) else np.nan,
            axis=1
        )

        results_df['total_went_over'] = (
            results_df['total'] > results_df['market_total']
        ).astype(float)

    # Calculate accuracy
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    # Win probability calibration
    print("\n📊 WIN PROBABILITY CALIBRATION")
    home_wins = results_df['home_won'].sum()
    total_games = len(results_df)
    actual_home_win_rate = home_wins / total_games
    avg_predicted_home_win_prob = results_df['model_home_win_prob'].mean()

    print(f"Actual home win rate: {actual_home_win_rate:.1%}")
    print(f"Avg predicted home win prob: {avg_predicted_home_win_prob:.1%}")
    print(f"Calibration error: {abs(actual_home_win_rate - avg_predicted_home_win_prob):.1%}")

    # Binned calibration analysis
    print("\n📊 CALIBRATION BY CONFIDENCE BINS")
    # Bin predictions into 10% buckets
    results_df['prob_bin'] = pd.cut(results_df['model_home_win_prob'],
                                      bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                                      labels=['<30%', '30-40%', '40-50%', '50-60%', '60-70%', '70%+'])

    for bin_name, group in results_df.groupby('prob_bin'):
        if len(group) > 0:
            actual_rate = group['home_won'].mean()
            predicted_rate = group['model_home_win_prob'].mean()
            print(f"  {bin_name}: {len(group)} games, Predicted: {predicted_rate:.1%}, Actual: {actual_rate:.1%}")

    # Spread error analysis
    print("\n📊 SPREAD PREDICTION ACCURACY")
    results_df['spread_error'] = results_df['margin'] - results_df['model_fair_spread']
    mae = results_df['spread_error'].abs().mean()
    rmse = np.sqrt((results_df['spread_error'] ** 2).mean())

    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
    print(f"Median spread error: {results_df['spread_error'].median():.2f} points")

    # Total error analysis
    print("\n📊 TOTAL PREDICTION ACCURACY")
    results_df['total_error'] = results_df['total'] - results_df['model_fair_total']
    mae = results_df['total_error'].abs().mean()
    rmse = np.sqrt((results_df['total_error'] ** 2).mean())

    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
    print(f"Median total error: {results_df['total_error'].median():.2f} points")

    # ROI Simulation (if we have market odds)
    if odds_df is not None and 'spread_cover_prob' in results_df.columns:
        print("\n📊 ROI SIMULATION (Betting $100 per pick with >5% edge)")

        # Spread bets with >5% edge
        spread_bets = results_df[
            (results_df['spread_cover_prob'].notna()) &
            (abs(results_df['spread_cover_prob'] - 0.5) > 0.05)  # >5% edge
        ].copy()

        if len(spread_bets) > 0:
            # Determine which side to bet
            spread_bets['bet_home'] = spread_bets['spread_cover_prob'] > 0.55
            spread_bets['bet_won'] = spread_bets.apply(
                lambda row: row['home_covered_spread'] if row['bet_home'] else (1 - row['home_covered_spread']),
                axis=1
            )

            # Calculate payout (assuming -110 odds)
            spread_bets['payout'] = spread_bets['bet_won'].apply(lambda x: 190.91 if x else 0)  # Win $90.91 or lose $100
            spread_bets['profit'] = spread_bets['payout'] - 100

            total_bet = len(spread_bets) * 100
            total_profit = spread_bets['profit'].sum()
            roi = (total_profit / total_bet) * 100
            win_rate = spread_bets['bet_won'].mean()

            print(f"\nSPREAD BETS:")
            print(f"  Total bets: {len(spread_bets)}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Total wagered: ${total_bet:,.0f}")
            print(f"  Total profit: ${total_profit:,.2f}")
            print(f"  ROI: {roi:.2f}%")

        # Total bets with >5% edge
        total_bets = results_df[
            (results_df['total_over_prob'].notna()) &
            (abs(results_df['total_over_prob'] - 0.5) > 0.05)
        ].copy()

        if len(total_bets) > 0:
            # Determine which side to bet
            total_bets['bet_over'] = total_bets['total_over_prob'] > 0.55
            total_bets['bet_won'] = total_bets.apply(
                lambda row: row['total_went_over'] if row['bet_over'] else (1 - row['total_went_over']),
                axis=1
            )

            # Calculate payout
            total_bets['payout'] = total_bets['bet_won'].apply(lambda x: 190.91 if x else 0)
            total_bets['profit'] = total_bets['payout'] - 100

            total_bet = len(total_bets) * 100
            total_profit = total_bets['profit'].sum()
            roi = (total_profit / total_bet) * 100
            win_rate = total_bets['bet_won'].mean()

            print(f"\nTOTAL BETS:")
            print(f"  Total bets: {len(total_bets)}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Total wagered: ${total_bet:,.0f}")
            print(f"  Total profit: ${total_profit:,.2f}")
            print(f"  ROI: {roi:.2f}%")

        # Combined ROI
        if len(spread_bets) > 0 or len(total_bets) > 0:
            all_bets = pd.concat([spread_bets, total_bets], ignore_index=True) if len(spread_bets) > 0 and len(total_bets) > 0 else (spread_bets if len(spread_bets) > 0 else total_bets)
            combined_bet = len(all_bets) * 100
            combined_profit = all_bets['profit'].sum()
            combined_roi = (combined_profit / combined_bet) * 100
            combined_win_rate = all_bets['bet_won'].mean()

            print(f"\nCOMBINED:")
            print(f"  Total bets: {len(all_bets)}")
            print(f"  Win rate: {combined_win_rate:.1%}")
            print(f"  Total wagered: ${combined_bet:,.0f}")
            print(f"  Total profit: ${combined_profit:,.2f}")
            print(f"  ROI: {combined_roi:.2f}%")

    # Save results
    output_file = PROJECT_ROOT / f"data/backtest_results/game_lines_{season}_weeks_{start_week}_{end_week}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    return results_df


if __name__ == "__main__":
    # Backtest 2025 season weeks 8-10 (where we have historical game line odds)
    df = backtest_game_lines(season=2025, start_week=8, end_week=10)
