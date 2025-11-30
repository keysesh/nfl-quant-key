#!/usr/bin/env python3
"""
Simulate missing games for a specific week.

This script simulates games that are in the odds data but not in the nflverse schedule.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput
from nfl_quant.features.team_metrics import TeamMetricsExtractor
from nfl_quant.data.fetcher import DataFetcher
from nfl_quant.utils.season_utils import get_current_season, get_current_week
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_missing_game(away_team: str, home_team: str, week: int, season: int = None):
    """Simulate a single game."""
    if season is None:
        season = get_current_season()
    logger.info(f"Simulating {away_team} @ {home_team} (Week {week})...")

    # Load data
    fetcher = DataFetcher()
    pbp = fetcher.load_pbp_parquet(season)
    pbp_to_date = pbp[pbp["week"] < week].copy()

    # Calculate EPAs
    home_off_epa = pbp_to_date[pbp_to_date["posteam"] == home_team]["epa"].mean() or 0.0
    home_def_epa = -(pbp_to_date[pbp_to_date["defteam"] == home_team]["epa"].mean() or 0.0)
    away_off_epa = pbp_to_date[pbp_to_date["posteam"] == away_team]["epa"].mean() or 0.0
    away_def_epa = -(pbp_to_date[pbp_to_date["defteam"] == away_team]["epa"].mean() or 0.0)

    # Get pace
    extractor = TeamMetricsExtractor()
    home_expected_plays = extractor.get_combined_pace(home_team, away_team, week)
    away_expected_plays = extractor.get_combined_pace(away_team, home_team, week)
    total_game_plays = home_expected_plays + away_expected_plays

    if total_game_plays > 0:
        avg_pace = 3600 / total_game_plays
        home_play_ratio = home_expected_plays / total_game_plays
        away_play_ratio = away_expected_plays / total_game_plays
        home_pace = avg_pace * (1.0 - (home_play_ratio - 0.5) * 0.2)
        away_pace = avg_pace * (1.0 - (away_play_ratio - 0.5) * 0.2)
    else:
        logger.warning(f"No pace data for {home_team} vs {away_team}, using default 25.0")
        home_pace = 25.0
        away_pace = 25.0

    # Create simulation input
    game_id = f"{season}_{week:02d}_{away_team}_{home_team}"
    sim_input = SimulationInput(
        game_id=game_id,
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
        home_offensive_epa=float(home_off_epa),
        away_offensive_epa=float(away_off_epa),
        home_defensive_epa=float(home_def_epa),
        away_defensive_epa=float(away_def_epa),
        home_pace=home_pace,
        away_pace=away_pace,
    )

    # Run simulation
    simulator = MonteCarloSimulator(seed=42)
    sim_output = simulator.simulate_game(sim_input, trials=50000)

    # Save result
    output_file = project_root / f"reports/sim_{season}_{week:02d}_{away_team}_{home_team}_42.json"
    result = {
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'home_score_median': float(sim_output.home_score_median),
        'away_score_median': float(sim_output.away_score_median),
        'home_score_std': float(sim_output.home_score_std),
        'away_score_std': float(sim_output.away_score_std),
        'home_win_prob': float(sim_output.home_win_prob),
        'away_win_prob': float(sim_output.away_win_prob),
        'tie_prob': float(sim_output.tie_prob),
        'fair_spread': float(sim_output.fair_spread),
        'fair_total': float(sim_output.fair_total),
        'trial_count': 50000,
        'seed': 42,
        # Add pace information for player projections
        'pace': float(sim_output.pace) if sim_output.pace else (home_pace + away_pace) / 2.0,
        'home_pace': float(home_pace),
        'away_pace': float(away_pace),
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"✅ Saved: {output_file}")
    return result


def main():
    """Simulate missing games for current week."""
    week = get_current_week()
    season = get_current_season()

    # Week 11 games from odds data
    missing_games = [
        ('BAL', 'CLE'),
        ('CAR', 'ATL'),
        ('CHI', 'MIN'),
        ('CIN', 'PIT'),
        ('DAL', 'LV'),
        ('DET', 'PHI'),
        ('GB', 'NYG'),
        ('HOU', 'TEN'),
        ('KC', 'DEN'),
        ('LAC', 'JAX'),
        ('NE', 'NYG'),
        ('NYJ', 'IND'),
        ('SEA', 'LAR'),
        ('SF', 'ARI'),
        ('TB', 'BUF'),
        ('WAS', 'MIA'),
    ]

    logger.info(f"Simulating {len(missing_games)} missing games for Week {week}...")

    results = []
    for away, home in missing_games:
        try:
            result = simulate_missing_game(away, home, week, season)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Error simulating {away} @ {home}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\n✅ Simulated {len(results)} games")
    return results


if __name__ == '__main__':
    main()
