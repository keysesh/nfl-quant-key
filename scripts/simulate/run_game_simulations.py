#!/usr/bin/env python3
"""
Generate game simulation files for player predictions.

Creates JSON files with projected team stats (pass attempts, rush attempts, targets)
based on league averages and matchup data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.data.dynamic_parameters import get_parameter_provider


def get_current_season():
    now = datetime.now()
    return now.year if now.month >= 8 else now.year - 1


# Map full team names to abbreviations (NFLverse standard)
TEAM_ABBREVS = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA',  # NFLverse uses 'LA' not 'LAR'
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
}


def get_week_games(week: int, season: int = None):
    """Get games for the week from NFLverse schedules (decoupled from odds/props)."""
    if season is None:
        season = get_current_season()

    sched_file = project_root / 'data' / 'nflverse' / 'schedules.parquet'
    if not sched_file.exists():
        raise FileNotFoundError(f"NFLverse schedules not found: {sched_file}.")

    sched = pd.read_parquet(sched_file)

    # Filter to specified week and season, regular season games only
    week_games = sched[
        (sched['season'] == season) &
        (sched['week'] == week) &
        (sched['game_type'] == 'REG')
    ]

    if week_games.empty:
        raise ValueError(f"No games found for {season} season week {week}")

    game_list = []
    for _, row in week_games.iterrows():
        game_list.append({
            'home_team': row['home_team'],
            'away_team': row['away_team'],
        })

    return game_list


def get_team_scoring_average(team: str, season: int = None):
    """Get actual team scoring average from NFLverse schedules data."""
    if season is None:
        season = get_current_season()

    # Convert full team name to abbreviation if needed
    team_abbrev = TEAM_ABBREVS.get(team, team)

    sched_path = project_root / 'data' / 'nflverse' / 'schedules.parquet'
    if not sched_path.exists():
        raise FileNotFoundError(f"NFLverse schedules not found: {sched_path}.")

    sched = pd.read_parquet(sched_path)
    season_games = sched[(sched['season'] == season) & (sched['home_score'].notna())]

    if season_games.empty:
        # Fallback to league average
        return 23.0

    # Get team's average points (both home and away games)
    home_games = season_games[season_games['home_team'] == team_abbrev]['home_score']
    away_games = season_games[season_games['away_team'] == team_abbrev]['away_score']

    all_scores = pd.concat([home_games, away_games])

    if all_scores.empty:
        return 23.0  # Fallback

    return float(all_scores.mean())


def get_team_pace(team: str, season: int = None):
    """Get actual team plays per game from NFLverse PBP data."""
    if season is None:
        season = get_current_season()

    # Convert full team name to abbreviation if needed
    team_abbrev = TEAM_ABBREVS.get(team, team)

    pbp_path = project_root / 'data' / 'nflverse' / f'pbp_{season}.parquet'
    if not pbp_path.exists():
        return 65.0  # Fallback to league average

    pbp = pd.read_parquet(pbp_path)

    # Count actual plays per game for this team (possession team)
    team_plays = pbp[pbp['posteam'] == team_abbrev].groupby('game_id').size()

    if team_plays.empty:
        return 65.0  # Fallback

    return float(team_plays.mean())


def generate_team_projections(home_team: str, away_team: str, season: int = None):
    """Generate team stat projections based on actual NFLverse data."""
    param_provider = get_parameter_provider()

    if season is None:
        season = get_current_season()

    # Get actual team-specific pass and rush attempts
    home_pass_attempts = param_provider.get_team_pass_attempts(home_team)
    home_rush_attempts = param_provider.get_team_rush_attempts(home_team)
    away_pass_attempts = param_provider.get_team_pass_attempts(away_team)
    away_rush_attempts = param_provider.get_team_rush_attempts(away_team)

    # Targets approximately equal pass attempts
    home_targets = home_pass_attempts
    away_targets = away_pass_attempts

    # Use actual team scoring averages
    home_projected_points = get_team_scoring_average(home_team, season)
    away_projected_points = get_team_scoring_average(away_team, season)

    # Get actual team pace (plays per game)
    home_pace = get_team_pace(home_team, season)
    away_pace = get_team_pace(away_team, season)

    return {
        'home': {
            'pass_attempts': home_pass_attempts,
            'rush_attempts': home_rush_attempts,
            'targets': home_targets,
            'projected_points': home_projected_points,
            'pace': home_pace,
        },
        'away': {
            'pass_attempts': away_pass_attempts,
            'rush_attempts': away_rush_attempts,
            'targets': away_targets,
            'projected_points': away_projected_points,
            'pace': away_pace,
        }
    }


def run_simulations(week: int, season: int = None):
    """Run game simulations and save results."""
    if season is None:
        season = get_current_season()

    # Create reports directory if needed
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    # Get games for this week
    games = get_week_games(week, season)
    print(f"Found {len(games)} games for week {week}")

    # Generate simulation files
    for game in games:
        home = game['home_team']
        away = game['away_team']

        # Generate projections - pass season to use actual data
        projections = generate_team_projections(home, away, season)

        # Get team abbreviations (already abbreviations from schedule)
        home_abbrev = TEAM_ABBREVS.get(home, home)
        away_abbrev = TEAM_ABBREVS.get(away, away)

        # Calculate actual plays per game from both teams' pace
        actual_plays_per_game = projections['home']['pace'] + projections['away']['pace']

        # Create simulation result
        sim_result = {
            'week': week,
            'season': season,
            'home_team': home,  # Use abbreviation for matching
            'away_team': away,  # Use abbreviation for matching
            'home_team_full': home,
            'away_team_full': away,
            'home_score_median': projections['home']['projected_points'],
            'away_score_median': projections['away']['projected_points'],
            'plays_per_game': actual_plays_per_game,
            'home_pace': projections['home']['pace'],
            'away_pace': projections['away']['pace'],
            'home_pass_attempts': projections['home']['pass_attempts'],
            'home_rush_attempts': projections['home']['rush_attempts'],
            'home_targets': projections['home']['targets'],
            'away_pass_attempts': projections['away']['pass_attempts'],
            'away_rush_attempts': projections['away']['rush_attempts'],
            'away_targets': projections['away']['targets'],
            'simulation_timestamp': datetime.now().isoformat(),
        }

        # Save to file - format: sim_{season}_{week}_{AWAY}_{HOME}.json
        filename = f"sim_{season}_{week:02d}_{away}_{home}.json"
        filepath = reports_dir / filename

        with open(filepath, 'w') as f:
            json.dump(sim_result, f, indent=2)

        print(f"  Created {filename}")

    print(f"\nGenerated {len(games)} game simulation files")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run game simulations')
    parser.add_argument('--week', type=int, required=True, help='NFL week')
    parser.add_argument('--season', type=int, default=None, help='NFL season')
    args = parser.parse_args()

    run_simulations(args.week, args.season)


if __name__ == '__main__':
    main()
