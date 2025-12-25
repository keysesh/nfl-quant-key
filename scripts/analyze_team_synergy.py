#!/usr/bin/env python3
"""
Team Health Synergy Analysis Script

Analyzes team health synergy effects for a specific game or week.
Generates detailed reports showing how returning players create compound effects.

Usage:
    python scripts/analyze_team_synergy.py --team TB --week 15
    python scripts/analyze_team_synergy.py --home TB --away LAC --week 15
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, REPORTS_DIR
from nfl_quant.features.team_synergy import (
    calculate_team_synergy_adjustment,
    apply_synergy_adjustments,
    generate_synergy_report,
    load_player_statuses_from_injuries,
    detect_returning_players_from_stats,
    calculate_team_total_adjustment,
    get_synergy_betting_implications,
    PlayerStatus,
    UnitHealthScore,
)
from nfl_quant.features.ir_return_detector import detect_returning_players_snap_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data(season: int = 2025):
    """Load required data files."""
    data = {}

    # Weekly stats
    weekly_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    if weekly_path.exists():
        data['weekly_stats'] = pd.read_parquet(weekly_path)
        logger.info(f"Loaded weekly_stats: {len(data['weekly_stats'])} rows")
    else:
        logger.error(f"Weekly stats not found: {weekly_path}")
        return None

    # Rosters
    roster_path = DATA_DIR / 'nflverse' / 'rosters.parquet'
    if roster_path.exists():
        data['rosters'] = pd.read_parquet(roster_path)
        logger.info(f"Loaded rosters: {len(data['rosters'])} rows")
    else:
        # Try CSV
        roster_path = DATA_DIR / 'nflverse' / f'rosters_{season}.csv'
        if roster_path.exists():
            data['rosters'] = pd.read_csv(roster_path)
            logger.info(f"Loaded rosters (CSV): {len(data['rosters'])} rows")

    # Injuries
    injury_path = DATA_DIR / 'nflverse' / 'injuries.parquet'
    if injury_path.exists():
        data['injuries'] = pd.read_parquet(injury_path)
        logger.info(f"Loaded injuries: {len(data['injuries'])} rows")
    else:
        injury_path = DATA_DIR / 'nflverse' / f'injuries_{season}.csv'
        if injury_path.exists():
            data['injuries'] = pd.read_csv(injury_path)
            logger.info(f"Loaded injuries (CSV): {len(data['injuries'])} rows")
        else:
            data['injuries'] = pd.DataFrame()
            logger.warning("No injury data found")

    # Depth charts
    depth_path = DATA_DIR / 'nflverse' / 'depth_charts.parquet'
    if depth_path.exists():
        data['depth_charts'] = pd.read_parquet(depth_path)
        logger.info(f"Loaded depth_charts: {len(data['depth_charts'])} rows")
    else:
        data['depth_charts'] = None

    # Snap counts
    snap_path = DATA_DIR / 'nflverse' / 'snap_counts.parquet'
    if snap_path.exists():
        data['snap_counts'] = pd.read_parquet(snap_path)
        logger.info(f"Loaded snap_counts: {len(data['snap_counts'])} rows")
    else:
        data['snap_counts'] = None

    return data


def create_player_statuses_manual(team: str, data: dict, week: int, season: int) -> list:
    """
    Create player statuses from available data.

    This is a manual approach when automatic detection needs supplementation.
    """
    statuses = []

    rosters = data.get('rosters')
    injuries = data.get('injuries')
    depth_charts = data.get('depth_charts')
    weekly_stats = data.get('weekly_stats')

    if rosters is None:
        logger.warning("No roster data available")
        return statuses

    # Filter to team
    team_col = 'team' if 'team' in rosters.columns else 'recent_team'
    team_roster = rosters[rosters[team_col] == team].copy()

    for _, row in team_roster.iterrows():
        player_name = row.get('player_name', row.get('full_name', ''))
        player_id = str(row.get('player_id', row.get('gsis_id', '')))
        position = row.get('position', '')

        if not player_name or not position:
            continue

        # Default values
        game_status = 'Active'
        snap_expectation = 1.0
        position_rank = 1
        weeks_missed = 0
        is_returning = False

        # Check injuries
        if injuries is not None and len(injuries) > 0:
            if 'player_name' in injuries.columns:
                injury_mask = injuries['player_name'].str.lower() == player_name.lower()
                if injury_mask.any():
                    inj_row = injuries[injury_mask].iloc[-1]  # Most recent
                    status = inj_row.get('status', inj_row.get('injury_status', 'Active'))

                    if status in ['Out', 'IR', 'Reserve/Injured', 'Injured Reserve']:
                        game_status = 'Out'
                        snap_expectation = 0.0
                    elif status in ['Doubtful']:
                        game_status = 'Doubtful'
                        snap_expectation = 0.1
                    elif status in ['Questionable']:
                        game_status = 'Questionable'
                        snap_expectation = 0.70
                    elif status in ['Probable', 'Day-to-day']:
                        game_status = 'Probable'
                        snap_expectation = 0.90

        # Check depth chart for position rank
        if depth_charts is not None and len(depth_charts) > 0:
            if 'full_name' in depth_charts.columns:
                dc_mask = depth_charts['full_name'].str.lower() == player_name.lower()
                if dc_mask.any():
                    dc_row = depth_charts[dc_mask].iloc[0]
                    pos_rank = dc_row.get('depth_chart_position', dc_row.get('depth_team', 1))
                    try:
                        position_rank = int(pos_rank) if pos_rank else 1
                    except (ValueError, TypeError):
                        position_rank = 1

        # Check for returning players from weekly stats
        if weekly_stats is not None:
            season_stats = weekly_stats[weekly_stats['season'] == season]
            player_stats = season_stats[
                season_stats['player_display_name'].str.lower() == player_name.lower()
            ]

            if len(player_stats) > 0:
                weeks_played = sorted(player_stats['week'].unique())
                # Check for gaps indicating injury
                if len(weeks_played) >= 2 and max(weeks_played) < week:
                    gap = week - max(weeks_played)
                    if gap >= 2:
                        # Player returning from missed weeks
                        is_returning = True
                        weeks_missed = gap
                        # Estimate snap expectation based on games missed
                        if weeks_missed >= 4:
                            snap_expectation = min(snap_expectation, 0.50)
                        elif weeks_missed >= 2:
                            snap_expectation = min(snap_expectation, 0.70)

        statuses.append(PlayerStatus(
            player_name=player_name,
            player_id=player_id,
            position=position,
            position_rank=position_rank,
            team=team,
            game_status=game_status,
            snap_expectation=snap_expectation,
            weeks_missed=weeks_missed,
            games_since_return=0,
            is_returning=is_returning
        ))

    return statuses


def analyze_team(team: str, data: dict, week: int, season: int, vegas_total: float = None):
    """Analyze synergy for a single team."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ANALYZING TEAM: {team}")
    logger.info(f"{'='*60}")

    # Create player statuses
    player_statuses = create_player_statuses_manual(team, data, week, season)
    logger.info(f"Found {len(player_statuses)} players on roster")

    # Detect returning players
    weekly_stats = data.get('weekly_stats')
    if weekly_stats is not None:
        returning = detect_returning_players_snap_status(weekly_stats, week, season)
        returning_team = [r for r in returning if r['team'] == team]
        logger.info(f"Auto-detected {len(returning_team)} returning players")

        # Update statuses
        for r in returning_team:
            for status in player_statuses:
                if status.player_name.lower() == r['player_name'].lower():
                    status.is_returning = True
                    status.snap_expectation = r['expected_snap_pct']
                    status.weeks_missed = r['weeks_missed']
                    status.games_since_return = r['games_since_return']

    returning_players = [s for s in player_statuses if s.is_returning]

    # Calculate synergy
    synergy = calculate_team_synergy_adjustment(
        player_statuses, team, returning_players
    )

    # Generate report
    report = generate_synergy_report(synergy)
    print(report)

    # Betting implications
    if vegas_total:
        total_adj = calculate_team_total_adjustment(synergy, vegas_total)
        print(f"\nTeam Total Adjustment:")
        print(f"  Raw: {total_adj['raw_total']:.1f}")
        print(f"  Adjusted: {total_adj['adjusted_total']:.1f}")
        print(f"  Delta: {total_adj['delta']:+.1f} points")

    implications = get_synergy_betting_implications(synergy, vegas_total)
    if implications:
        print("\nBetting Implications:")
        for imp in implications:
            print(f"  • {imp}")

    return synergy


def analyze_game(home_team: str, away_team: str, data: dict, week: int, season: int,
                 home_total: float = None, away_total: float = None):
    """Analyze synergy for both teams in a game."""
    print(f"\n{'#'*70}")
    print(f"#  GAME SYNERGY ANALYSIS: {away_team} @ {home_team}")
    print(f"#  Week {week}, {season} Season")
    print(f"{'#'*70}")

    home_synergy = analyze_team(home_team, data, week, season, home_total)
    away_synergy = analyze_team(away_team, data, week, season, away_total)

    # Comparison
    print(f"\n{'='*60}")
    print("SYNERGY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Team':<10} {'Offense':<12} {'Defense':<12} {'Overall':<12}")
    print("-" * 46)
    print(f"{home_team:<10} {home_synergy.offense_multiplier:.3f}x      {home_synergy.defense_multiplier:.3f}x      {home_synergy.team_multiplier:.3f}x")
    print(f"{away_team:<10} {away_synergy.offense_multiplier:.3f}x      {away_synergy.defense_multiplier:.3f}x      {away_synergy.team_multiplier:.3f}x")

    # Edge calculation
    home_edge = home_synergy.offense_multiplier / away_synergy.defense_multiplier
    away_edge = away_synergy.offense_multiplier / home_synergy.defense_multiplier

    print(f"\nOffense vs Defense Edges:")
    print(f"  {home_team} offense vs {away_team} defense: {home_edge:.3f}x")
    print(f"  {away_team} offense vs {home_team} defense: {away_edge:.3f}x")

    return home_synergy, away_synergy


def save_report(synergy, team: str, week: int, season: int):
    """Save synergy report to file."""
    report_dir = REPORTS_DIR / 'synergy'
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'synergy_{team}_week{week}_{timestamp}.txt'

    report = generate_synergy_report(synergy)

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Saved report to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Analyze team health synergy effects')
    parser.add_argument('--team', '-t', help='Single team to analyze (e.g., TB)')
    parser.add_argument('--home', '-H', help='Home team for game analysis')
    parser.add_argument('--away', '-A', help='Away team for game analysis')
    parser.add_argument('--week', '-w', type=int, default=15, help='NFL week')
    parser.add_argument('--season', '-s', type=int, default=2025, help='NFL season')
    parser.add_argument('--vegas-total', '-v', type=float, help='Vegas implied team total')
    parser.add_argument('--home-total', type=float, help='Home team implied total')
    parser.add_argument('--away-total', type=float, help='Away team implied total')
    parser.add_argument('--save', action='store_true', help='Save report to file')

    args = parser.parse_args()

    # Validate args
    if not args.team and not (args.home and args.away):
        parser.error("Must specify either --team or both --home and --away")

    # Load data
    logger.info("Loading data...")
    data = load_data(args.season)

    if data is None:
        logger.error("Failed to load required data")
        return 1

    # Run analysis
    if args.team:
        synergy = analyze_team(
            args.team, data, args.week, args.season, args.vegas_total
        )
        if args.save:
            save_report(synergy, args.team, args.week, args.season)
    else:
        home_synergy, away_synergy = analyze_game(
            args.home, args.away, data, args.week, args.season,
            args.home_total, args.away_total
        )
        if args.save:
            save_report(home_synergy, args.home, args.week, args.season)
            save_report(away_synergy, args.away, args.week, args.season)

    return 0


if __name__ == '__main__':
    sys.exit(main())
