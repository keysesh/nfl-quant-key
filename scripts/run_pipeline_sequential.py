#!/usr/bin/env python3
"""
NFL QUANT Full Pipeline Runner

Runs all data fetching and prediction steps in the correct order:
1. Refresh NFLverse data (R script)
2. Fetch Sleeper injuries (critical for predictions)
3. Fetch live odds (game lines)
4. Fetch player props
5. Generate model predictions
6. Generate unified recommendations

Usage:
    python scripts/run_pipeline.py <WEEK>
    python scripts/run_pipeline.py 13
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Use centralized path configuration
from nfl_quant.config_paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

def run_step(description: str, command: list, required: bool = True) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(command)}")
    print()

    try:
        result = subprocess.run(command, check=True)
        print(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        if required:
            print(f"\n❌ {description} - FAILED (exit code {e.returncode})")
            return False
        else:
            print(f"\n⚠️  {description} - FAILED (optional, continuing)")
            return True
    except FileNotFoundError as e:
        print(f"\n❌ {description} - FAILED (command not found: {e})")
        return required == False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pipeline.py <WEEK>")
        print("Example: python scripts/run_pipeline.py 13")
        sys.exit(1)

    week = sys.argv[1]

    try:
        week_num = int(week)
        if not 1 <= week_num <= 18:
            raise ValueError("Week must be 1-18")
    except ValueError as e:
        print(f"Error: Invalid week '{week}' - {e}")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         NFL QUANT FULL PIPELINE                              ║
║                              Week {week_num:2d} - {datetime.now().strftime('%Y-%m-%d %H:%M')}                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    steps = [
        # Step 1: NFLverse data refresh (R script)
        (
            "Refresh NFLverse Data (stats, rosters, depth charts)",
            ["Rscript", "scripts/fetch/fetch_nflverse_data.R"],
            True
        ),
        # Step 2: Sleeper injuries (CRITICAL - was missing!)
        (
            "Fetch Injury Reports (Sleeper API)",
            [str(venv_python), "scripts/fetch/fetch_injuries_sleeper.py"],
            True
        ),
        # Step 3: Data freshness check
        (
            "Check Data Freshness",
            [str(venv_python), "scripts/fetch/check_data_freshness.py"],
            False  # Warning only - don't fail pipeline
        ),
        # Step 3: Game odds
        (
            "Fetch Live Odds (game lines)",
            [str(venv_python), "scripts/fetch/fetch_live_odds.py", week],
            True
        ),
        # Step 4: Player props (saves to data/nfl_player_props_draftkings.csv)
        (
            "Fetch Player Props (Odds API)",
            [str(venv_python), "scripts/fetch/fetch_nfl_player_props.py"],
            True  # Required - uses ODDS_API_KEY from .env
        ),
        # Step 5: Generate predictions
        (
            f"Generate Model Predictions (Week {week_num})",
            [str(venv_python), "scripts/predict/generate_model_predictions.py", week],
            True
        ),
        # Step 6: Generate player prop recommendations
        (
            f"Generate Player Prop Recommendations (Week {week_num})",
            [str(venv_python), "scripts/predict/generate_unified_recommendations_v3.py", "--week", week],
            True
        ),
        # Step 7: Generate game line recommendations
        (
            f"Generate Game Line Recommendations (Week {week_num})",
            [str(venv_python), "scripts/predict/generate_game_line_recommendations.py"],
            True
        ),
        # Step 8: Generate dashboard
        (
            "Generate Pro Dashboard",
            [str(venv_python), "scripts/dashboard/generate_pro_dashboard.py"],
            True
        ),
    ]

    results = []
    for description, command, required in steps:
        success = run_step(description, command, required)
        results.append((description, success))

        if not success and required:
            print(f"\n❌ Pipeline failed at: {description}")
            print("Stopping execution.")
            sys.exit(1)

    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    for desc, success in results:
        status = "✅" if success else "⚠️ "
        print(f"  {status} {desc}")

    print(f"\n📊 Check reports/ directory for output files")
    print(f"📁 Latest recommendations: reports/recommendations_week{week_num}_*.csv")


if __name__ == "__main__":
    main()
