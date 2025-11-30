#!/usr/bin/env python3
"""
Automated Weekly Backtest Workflow

Fetches results, validates picks, calculates ROI, updates calibrator.
Run this script every Monday after games complete.

Usage:
    python scripts/workflows/weekly_backtest.py --week 10
    python scripts/workflows/weekly_backtest.py --week 10 --season 2024
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.season_config import CURRENT_NFL_SEASON, infer_season_from_week

def run_command(cmd, description):
    """Run a shell command and return success status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Automated weekly backtest workflow')
    parser.add_argument('--week', type=int, required=True, help='NFL week number')
    parser.add_argument('--season', type=int, default=None,
                       help='NFL season year (default: auto-infer from week)')
    parser.add_argument('--skip-calibrator', action='store_true', help='Skip calibrator retraining')
    parser.add_argument('--skip-report', action='store_true', help='Skip generating report')
    args = parser.parse_args()

    week = args.week

    # Smart season inference
    if args.season is None:
        season = infer_season_from_week(week)
        print(f"ℹ️  No season specified, inferred season {season} for week {week}")
    else:
        season = args.season

    print("="*80)
    print(f"AUTOMATED WEEKLY BACKTEST: Week {week}, {season} Season")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Fetch player stats for the week
    step1 = run_command(
        f'PYTHONPATH="/Users/keyonnesession/Desktop/NFL QUANT:$PYTHONPATH" '
        f'.venv/bin/python scripts/backtest/fetch_week_player_stats.py {week} {season}',
        f"Fetch Week {week} player stats"
    )

    if not step1:
        print("\n❌ CRITICAL ERROR: Could not fetch player stats. Aborting.")
        return 1

    # Check if player stats file exists
    stats_file = Path(f'data/results/{season}/week{week}_player_stats.csv')
    if not stats_file.exists():
        print(f"\n❌ ERROR: Expected file not found: {stats_file}")
        return 1

    # Step 2: Run validation on picks
    picks_file = Path(f'reports/all_picks_ranked_week{week}.csv')
    if not picks_file.exists():
        print(f"\n⚠️  WARNING: Picks file not found: {picks_file}")
        step2 = False
    else:
        # Use generic validation script (works for any week)
        validation_script = Path('scripts/backtest/validate_week_generic.py')
        if validation_script.exists():
            step2 = run_command(
                f'.venv/bin/python {validation_script} --week {week} --season {season}',
                f"Validate Week {week} picks"
            )
        else:
            print(f"\n⚠️  WARNING: Generic validation script not found: {validation_script}")
            print("Please ensure scripts/backtest/validate_week_generic.py exists")
            step2 = False

    # Step 3: Optionally retrain calibrator
    if not args.skip_calibrator:
        calibrator_script = Path('scripts/train/retrain_calibrator_nflverse.py')
        if calibrator_script.exists():
            step3 = run_command(
                f'PYTHONPATH="/Users/keyonnesession/Desktop/NFL QUANT:$PYTHONPATH" '
                f'.venv/bin/python {calibrator_script}',
                f"Retrain calibrator with Week {week} data"
            )
        else:
            print(f"\n⚠️  WARNING: Calibrator retraining script not found: {calibrator_script}")
            step3 = False
    else:
        print("\n⏭️  SKIPPED: Calibrator retraining")
        step3 = True

    # Step 4: Generate summary report
    if not args.skip_report:
        print(f"\n{'='*80}")
        print(f"GENERATING SUMMARY REPORT")
        print(f"{'='*80}")

        # Read backtest results
        results_file = Path(f'reports/WEEK{week}_BACKTEST_COMPLETE.csv')
        if results_file.exists():
            df = pd.read_csv(results_file)

            print(f"\n📊 Week {week} Performance Summary:")
            print(f"   Total Evaluated: {len(df)} bets")
            print(f"   Wins: {df['won'].sum()}")
            print(f"   Losses: {(~df['won']).sum()}")
            print(f"   Win Rate: {df['won'].mean()*100:.1f}%")
            print(f"   Total Wagered: ${df['wager'].sum():,.2f}")
            print(f"   Total Profit: ${df['profit'].sum():+,.2f}")
            if df['wager'].sum() > 0:
                print(f"   ROI: {(df['profit'].sum()/df['wager'].sum())*100:+.1f}%")

            # Best pick
            best_pick = df.nlargest(1, 'profit').iloc[0]
            print(f"\n🏆 Best Pick:")
            print(f"   {best_pick['player']}: {best_pick['pick']}")
            print(f"   Profit: ${best_pick['profit']:+.2f}")

            # Worst pick
            worst_pick = df.nsmallest(1, 'profit').iloc[0]
            print(f"\n💸 Worst Pick:")
            print(f"   {worst_pick['player']}: {worst_pick['pick']}")
            print(f"   Loss: ${worst_pick['profit']:+.2f}")

        else:
            print(f"\n⚠️  WARNING: Results file not found: {results_file}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"WORKFLOW COMPLETE")
    print(f"{'='*80}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_success = step1 and step2 and step3
    if all_success:
        print("\n✅ All steps completed successfully!")
        return 0
    else:
        print("\n⚠️  Some steps had warnings or failures. Review output above.")
        return 0  # Don't fail hard, just warn

if __name__ == "__main__":
    sys.exit(main())
