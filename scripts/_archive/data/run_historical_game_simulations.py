#!/usr/bin/env python3
"""
Run Historical Game Simulations
================================

Generate game simulations for 2024 season weeks 1-8 to create
calibration training data.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput
from nfl_quant.utils.season_utils import get_current_season


def load_2024_games(week: int) -> pd.DataFrame:
    """Load game schedule for a specific week in 2024."""
    try:
        from nfl_quant.utils.nflverse_loader import load_schedules
        schedules = load_schedules(seasons=2024)

        week_games = schedules[
            (schedules['week'] == week) &
            (schedules['game_type'] == 'REG')
        ].copy()

        return week_games
    except Exception as e:
        print(f"❌ Error loading 2024 schedule: {e}")
        return pd.DataFrame()


def load_historical_team_stats(week: int) -> dict:
    """
    Load historical team EPA stats and pace for a given week.
    Uses nflverse play-by-play data aggregated up to that point.
    """
    try:
        from nfl_quant.utils.nflverse_loader import load_pbp

        # Load play-by-play for 2024 season
        pbp = load_pbp(seasons=2024)

        # Filter to games before this week
        pbp_historical = pbp[pbp['week'] < week].copy()

        # Aggregate EPA by team (offensive)
        offense_stats = pbp_historical.groupby('posteam').agg({
            'epa': 'mean',
            'play_id': 'count',  # Total plays for pace calculation
        }).reset_index()

        offense_stats.columns = ['team', 'epa_per_play', 'total_plays']

        # Calculate pace (plays per game)
        games_played = pbp_historical.groupby('posteam')['game_id'].nunique().reset_index()
        games_played.columns = ['team', 'games']

        offense_stats = offense_stats.merge(games_played, on='team')
        offense_stats['pace'] = offense_stats['total_plays'] / offense_stats['games']

        # Aggregate EPA by team (defensive)
        defense_stats = pbp_historical.groupby('defteam').agg({
            'epa': 'mean',
        }).reset_index()

        defense_stats.columns = ['team', 'def_epa_per_play']

        # Combine
        team_stats = {}
        for _, row in offense_stats.iterrows():
            team = row['team']
            team_stats[team] = {
                'offensive_epa': row['epa_per_play'],
                'pace': row['pace'] if not pd.isna(row['pace']) else 65.0,  # Fallback to NFL average
                'success_rate': 0.0,  # Placeholder
            }

        for _, row in defense_stats.iterrows():
            team = row['team']
            if team in team_stats:
                team_stats[team]['defensive_epa'] = row['def_epa_per_play']

        return team_stats

    except Exception as e:
        print(f"⚠️  Error loading historical stats: {e}")
        import traceback
        traceback.print_exc()
        return {}


def simulate_game(game_row: pd.Series, team_stats: dict, week: int, season: int = None) -> dict:
    """Simulate a single game and return results."""
    if season is None:
        season = get_current_season()

    home_team = game_row['home_team']
    away_team = game_row['away_team']

    # Get team stats
    home_stats = team_stats.get(home_team, {})
    away_stats = team_stats.get(away_team, {})

    # Create simulation input
    sim_input = SimulationInput(
        game_id=f"{season}_{week:02d}_{away_team}_{home_team}",
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,

        # Offensive EPA
        home_offensive_epa=home_stats.get('offensive_epa', 0.0),
        away_offensive_epa=away_stats.get('offensive_epa', 0.0),

        # Defensive EPA (opponent's offensive EPA against them)
        home_defensive_epa=home_stats.get('defensive_epa', 0.0),
        away_defensive_epa=away_stats.get('defensive_epa', 0.0),

        # Pace (calculated from historical data, or use NFL average)
        home_pace=home_stats.get('pace', 65.0),
        away_pace=away_stats.get('pace', 65.0),

        # Additional context
        is_divisional=(game_row.get('div_game', 0) == 1),
        is_dome=False,  # Simplified - would need stadium lookup
        game_type='REG',

        # Injury impacts (simplified - set to 0)
        home_injury_impact=0.0,
        away_injury_impact=0.0,
    )

    # Run simulation
    simulator = MonteCarloSimulator(seed=42)
    result = simulator.simulate_game(sim_input)

    # Convert to dict for JSON serialization
    result_dict = {
        'game_id': sim_input.game_id,
        'season': sim_input.season,
        'week': sim_input.week,
        'home_team': sim_input.home_team,
        'away_team': sim_input.away_team,

        # Simulation results (using median as proxy for mean)
        'home_score_mean': float(result.home_score_median),
        'away_score_mean': float(result.away_score_median),
        'home_score_std': float(result.home_score_std),
        'away_score_std': float(result.away_score_std),
        'total_mean': float(result.total_median),
        'total_std': float(result.total_std),
        'home_win_prob': float(result.home_win_prob),

        # Derived values
        'projected_total': float(result.total_median),
        'projected_spread': float(result.fair_spread),
        'fair_total': float(result.fair_total),
        'fair_spread': float(result.fair_spread),
    }

    return result_dict


def run_simulations_for_week(week: int, season: int = None):
    """Run simulations for all games in a week."""
    if season is None:
        season = get_current_season()

    print(f"\n{'='*80}")
    print(f"SIMULATING WEEK {week} - {season} SEASON")
    print(f"{'='*80}")

    # Load games
    games = load_2024_games(week)

    if games.empty:
        print(f"❌ No games found for week {week}")
        return

    print(f"✅ Found {len(games)} games")

    # Load historical team stats
    print(f"📊 Loading team stats (up to week {week})...")
    team_stats = load_historical_team_stats(week)

    if not team_stats:
        print(f"⚠️  No team stats available, using defaults")
    else:
        print(f"✅ Loaded stats for {len(team_stats)} teams")

    # Run simulations
    print(f"🎲 Running simulations...")

    results = []
    for idx, game in games.iterrows():
        try:
            result = simulate_game(game, team_stats, week, season)
            results.append(result)

            # Save individual file
            filename = f"reports/sim_{season}_{week:02d}_{result['away_team']}_{result['home_team']}_42.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"   ✅ {result['away_team']} @ {result['home_team']}: {result['projected_total']:.1f} total, {result['projected_spread']:+.1f} spread")

        except Exception as e:
            print(f"   ❌ Error simulating {game['away_team']} @ {game['home_team']}: {e}")
            continue

    print(f"\n✅ Simulated {len(results)} games for Week {week}")
    return results


def main(auto_confirm=False, season=None):
    """Run simulations for weeks 1-8."""
    if season is None:
        season = get_current_season()

    print("="*80)
    print(f"HISTORICAL GAME SIMULATIONS - {season} SEASON")
    print("="*80)
    print()
    print("Generating game simulations for weeks 1-8 to train calibrator")
    print()

    # Confirm
    if not auto_confirm:
        response = input("This will generate ~128 simulation files. Continue? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
    else:
        print("Auto-confirmed via command line argument")

    # Run simulations for each week
    all_results = []
    for week in range(1, 9):  # Weeks 1-8
        results = run_simulations_for_week(week, season)
        if results:
            all_results.extend(results)
        print(f"   Week {week} complete: {len(results)} games simulated")
        print()

    print(f"\n{'='*80}")
    print(f"✅ SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total simulations: {len(all_results)}")
    print(f"\n💡 Next steps:")
    print(f"   1. Collect calibration data:")
    print(f"      python scripts/data/collect_game_line_calibration_data.py")
    print(f"   2. Train calibrator:")
    print(f"      python scripts/train/train_game_line_calibrator.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run historical game simulations')
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm')
    parser.add_argument('--season', type=int, default=None, help='Season to simulate (default: current season)')
    args = parser.parse_args()

    main(auto_confirm=args.yes, season=args.season)
