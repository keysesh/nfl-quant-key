#!/usr/bin/env python3
"""
Regenerate game simulation JSON files with complete schema.

This fixes missing fields like home_win_prob, fair_spread, fair_total, etc.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_team_epa(pbp_df: pd.DataFrame, team: str) -> dict:
    """Calculate team EPA metrics from PBP data."""
    # Offensive EPA
    off_plays = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    # Defensive EPA
    def_plays = pbp_df[
        (pbp_df['defteam'] == team) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    if len(off_plays) == 0 or len(def_plays) == 0:
        return {
            'offensive_epa': 0.0,
            'defensive_epa': 0.0,
            'pace': 65.0
        }

    # Calculate EPA
    offensive_epa = off_plays['epa'].mean()
    defensive_epa = def_plays['epa'].mean()

    # Regress to mean (0.0)
    sample_games = len(off_plays) / 60  # ~60 plays per game
    regression_weight = min(1.0, sample_games / 8.0)

    offensive_epa = offensive_epa * regression_weight
    defensive_epa = defensive_epa * regression_weight

    # Calculate pace (seconds per play)
    pace = 65.0
    if 'game_seconds_remaining' in pbp_df.columns:
        team_drives = pbp_df[pbp_df['posteam'] == team]
        if len(team_drives) > 0:
            pace = team_drives['game_seconds_remaining'].diff().abs().median()
            pace = max(50.0, min(80.0, pace))

    return {
        'offensive_epa': float(offensive_epa),
        'defensive_epa': float(defensive_epa),
        'pace': float(pace)
    }


def load_team_epa(season: int = 2025, week: int = 12) -> pd.DataFrame:
    """Load team EPA data from PBP."""
    logger.info(f"Calculating team EPA through week {week-1}")

    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP data not found: {pbp_path}")

    pbp = pd.read_parquet(pbp_path)
    pbp = pbp[pbp['week'] < week].copy()  # Only use data before current week

    # Calculate EPA for all teams
    teams = pd.concat([pbp['home_team'], pbp['away_team']]).unique()
    team_epa_list = []

    for team in teams:
        epa = calculate_team_epa(pbp, team)
        team_epa_list.append({
            'team': team,
            **epa
        })

    return pd.DataFrame(team_epa_list)


def get_games_for_week(season: int, week: int) -> list:
    """Get list of games for the week from odds data."""
    odds_path = Path(f'data/odds_week{week}.csv')
    if not odds_path.exists():
        raise FileNotFoundError(f"Odds data not found: {odds_path}")

    odds_df = pd.read_csv(odds_path)

    # Get unique games
    games = []
    seen_game_ids = set()

    for _, row in odds_df.iterrows():
        game_id = row['game_id']
        if game_id in seen_game_ids:
            continue

        seen_game_ids.add(game_id)
        games.append({
            'game_id': game_id,
            'home_team': row['home_team'],
            'away_team': row['away_team']
        })

    logger.info(f"Found {len(games)} games for Week {week}")
    return games


def regenerate_simulations(season: int = 2025, week: int = 12, trials: int = 50000):
    """Regenerate all game simulations for the week."""

    # Load team EPA
    team_epa = load_team_epa(season, week)

    # Get games
    games = get_games_for_week(season, week)

    # Initialize simulator
    simulator = MonteCarloSimulator(seed=42)

    # Process each game
    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']

        logger.info(f"Simulating {game_id}: {away_team} @ {home_team}")

        # Get EPA for both teams
        try:
            home_epa = team_epa[team_epa['team'] == home_team].iloc[0]
            away_epa = team_epa[team_epa['team'] == away_team].iloc[0]
        except IndexError:
            logger.warning(f"  EPA data missing for {home_team} or {away_team}, skipping")
            continue

        # Create simulation input (simplified - no injuries for now)
        sim_input = SimulationInput(
            game_id=game_id,
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
        sim_output = simulator.simulate_game(sim_input, trials=trials)

        # Save to JSON
        output_path = Path(f'reports/sim_{season}_{week}_{away_team}_{home_team}.json')
        with open(output_path, 'w') as f:
            json.dump(sim_output.model_dump(), f, indent=2)

        logger.info(f"  ✅ Saved: {output_path}")
        logger.info(f"     Home Win: {sim_output.home_win_prob:.1%}, "
                   f"Fair Spread: {sim_output.fair_spread:+.1f}, "
                   f"Fair Total: {sim_output.fair_total:.1f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Regenerate game simulation JSON files')
    parser.add_argument('--week', type=int, default=12, help='Week number')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--trials', type=int, default=50000, help='Number of Monte Carlo trials')

    args = parser.parse_args()

    logger.info(f"Regenerating game simulations for {args.season} Week {args.week}")
    logger.info(f"Monte Carlo trials: {args.trials:,}")

    regenerate_simulations(args.season, args.week, args.trials)

    logger.info("✅ All simulations regenerated successfully")
