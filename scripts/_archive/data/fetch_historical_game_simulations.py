#!/usr/bin/env python3
"""
Fetch Historical Game Simulations
==================================

Generate game simulations for weeks 1-8 of 2024 season using our framework,
then match against actual outcomes to create calibration dataset.

This gives us real predictions vs outcomes for:
- Game totals (over/under)
- Spreads
- Moneylines
"""

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season


def check_what_we_have(season: int = None):
    """Check what simulation data and actual outcomes we have."""
    if season is None:
        season = get_current_season()

    print("="*80)
    print("CHECKING AVAILABLE DATA")
    print("="*80)
    print()

    # Check for simulation results
    sim_files_season = list(Path('reports').glob(f'sim_{season}_*.json'))

    print(f"📊 Simulation Results:")
    print(f"   {season} Season: {len(sim_files_season)} files")

    if len(sim_files_season) > 0:
        print(f"\n   Available simulations:")
        for f in sorted(sim_files_season)[:5]:
            print(f"     - {f.name}")

    # Check for actual game outcomes
    try:
        from nfl_quant.utils.nflverse_loader import load_schedules
        schedules_season = load_schedules(seasons=season)
        completed_season = schedules_season[
            (schedules_season['game_type'] == 'REG') &
            (schedules_season['home_score'].notna())
        ]
        print(f"\n📅 Actual Game Outcomes ({season}):")
        print(f"   Total games: {len(completed_season)}")
        print(f"   Weeks available: {sorted(completed_season['week'].unique())}")
    except Exception as e:
        print(f"\n❌ Could not load {season} outcomes: {e}")
        completed_season = pd.DataFrame()

    return {
        'sim_files': sim_files_season,
        'actual_outcomes': completed_season
    }


def run_simulations_for_week(week: int, season: int = None):
    """
    Run game simulations for a specific week.

    This uses the same simulation engine as our current predictions,
    but runs it on historical weeks to create training data.
    """
    if season is None:
        season = get_current_season()

    print(f"\n{'='*80}")
    print(f"SIMULATING WEEK {week} - {season} SEASON")
    print(f"{'='*80}")

    # Check if we have nflverse data for this week
    try:
        from nfl_quant.utils.nflverse_loader import load_schedules
        schedules = load_schedules(seasons=season)
        week_games = schedules[
            (schedules['week'] == week) &
            (schedules['game_type'] == 'REG')
        ]

        if len(week_games) == 0:
            print(f"⚠️  No games found for Week {week}")
            return []

        print(f"✅ Found {len(week_games)} games for Week {week}")

    except Exception as e:
        print(f"❌ Error loading schedule: {e}")
        return []

    # Import the simulation command
    try:
        # Note: simulate is a click command in cli.py, not directly importable
        # Use subprocess to call the CLI properly
        import subprocess

        print(f"🎲 Running simulations via CLI...")

        # Call the nfl-quant CLI simulate command
        result = subprocess.run(
            ['.venv/bin/python', '-m', 'nfl_quant.cli', 'simulate',
             '--week', str(week), '--season', str(season),
             '--trials', '50000', '--seed', '42'],
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            raise Exception(f"Simulation failed: {result.stderr}")

        print(result.stdout)

        print(f"✅ Simulations completed for Week {week}")

        # Return list of created files
        sim_files = list(Path('reports').glob(f'sim_{season}_{week:02d}_*.json'))
        return sim_files

    except ImportError as e:
        print(f"❌ Simulation module not available: {e}")
        print(f"   Trying alternative approach...")

        # Alternative: Use the CLI command directly
        import subprocess
        result = subprocess.run(
            ['nfl-quant', 'simulate', '--week', str(week), '--season', str(season)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ Simulations completed via CLI")
            sim_files = list(Path('reports').glob(f'sim_{season}_{week:02d}_*.json'))
            return sim_files
        else:
            print(f"❌ CLI simulation failed: {result.stderr}")
            return []


def main(season: int = None):
    """Main data collection pipeline."""
    if season is None:
        season = get_current_season()

    print("="*80)
    print("HISTORICAL GAME SIMULATION DATA COLLECTION")
    print("="*80)
    print()
    print("Purpose: Generate game line calibration training data")
    print("Method: Run simulations on historical weeks, match to actual outcomes")
    print()

    # Step 1: Check what we have
    data_status = check_what_we_have(season)

    # Step 2: Determine what we need
    target_weeks = list(range(1, 9))  # Weeks 1-8
    target_season = season

    print(f"\n{'='*80}")
    print(f"TARGET: Simulations for {target_season} Weeks {min(target_weeks)}-{max(target_weeks)}")
    print(f"{'='*80}")

    # Check if simulations already exist
    existing_weeks = set()
    for sim_file in data_status['sim_files']:
        # Parse: sim_2024_05_GB_LAR_42.json
        parts = sim_file.stem.split('_')
        if len(parts) >= 3:
            try:
                week_num = int(parts[2])
                existing_weeks.add(week_num)
            except:
                pass

    missing_weeks = [w for w in target_weeks if w not in existing_weeks]

    if len(missing_weeks) == 0:
        print(f"\n✅ All simulations already exist!")
        print(f"   Existing weeks: {sorted(existing_weeks)}")
        print(f"\n💡 Proceeding to calibration data collection...")
    else:
        print(f"\n⚠️  Missing simulations for weeks: {missing_weeks}")
        print(f"   Need to generate {len(missing_weeks)} weeks")

        # Ask user confirmation
        response = input(f"\nGenerate simulations for {len(missing_weeks)} weeks? (y/n): ")

        if response.lower() != 'y':
            print("❌ Cancelled by user")
            return

        # Step 3: Run simulations for missing weeks
        print(f"\n{'='*80}")
        print(f"GENERATING SIMULATIONS")
        print(f"{'='*80}")

        all_sim_files = []
        for week in missing_weeks:
            sim_files = run_simulations_for_week(week, target_season)
            all_sim_files.extend(sim_files)
            print(f"   Week {week}: {len(sim_files)} simulations created")

        print(f"\n✅ Total simulations created: {len(all_sim_files)}")

    # Step 4: Now run the calibration data collection
    print(f"\n{'='*80}")
    print(f"COLLECTING CALIBRATION DATA")
    print(f"{'='*80}")

    import subprocess
    result = subprocess.run(
        ['python', 'scripts/data/collect_game_line_calibration_data.py'],
        cwd=Path.cwd()
    )

    if result.returncode == 0:
        print(f"\n✅ Calibration data collection completed")
        print(f"\n💡 Next step: Train calibrator")
        print(f"   python scripts/train/train_game_line_calibrator.py")
    else:
        print(f"\n⚠️  Calibration data collection had issues")
        print(f"   Check the output above for details")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fetch historical game simulations')
    parser.add_argument('--season', type=int, default=None, help='Season to fetch (default: current season)')
    args = parser.parse_args()

    main(season=args.season)
