#!/usr/bin/env python3
"""
Backfill historical validation for multiple weeks.
Fetches player stats and validates picks for weeks 1-N.

Usage:
    python scripts/workflows/backfill_historical_validation.py --start-week 1 --end-week 9
    python scripts/workflows/backfill_historical_validation.py --weeks 1 2 3 4 5
    python scripts/workflows/backfill_historical_validation.py --start-week 1 --end-week 10 --season 2024
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.season_config import CURRENT_NFL_SEASON

def run_command(cmd, description, allow_failure=False):
    """Run a shell command and return success status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0 and not allow_failure:
        print(f"❌ FAILED: {description}")
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Backfill historical validation for multiple weeks')
    parser.add_argument('--start-week', type=int, help='Starting week number')
    parser.add_argument('--end-week', type=int, help='Ending week number')
    parser.add_argument('--weeks', nargs='+', type=int, help='Specific week numbers to process')
    parser.add_argument('--season', type=int, default=None,
                       help=f'NFL season year (default: {CURRENT_NFL_SEASON}, use --season 2024 for historical)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip weeks with existing results')
    args = parser.parse_args()

    # Default to current season if not specified
    if args.season is None:
        season = CURRENT_NFL_SEASON
        print(f"ℹ️  No season specified, using current season {season}")
        print(f"   For historical validation, use --season 2024")
    else:
        season = args.season

    # Determine which weeks to process
    if args.weeks:
        weeks = args.weeks
    elif args.start_week and args.end_week:
        weeks = list(range(args.start_week, args.end_week + 1))
    else:
        print("❌ ERROR: Must specify either --weeks or --start-week/--end-week")
        return 1

    print("="*80)
    print(f"HISTORICAL VALIDATION BACKFILL: Weeks {min(weeks)}-{max(weeks)}, {season} Season")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing {len(weeks)} weeks: {weeks}")

    results_summary = []

    for week in weeks:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING WEEK {week}")
        print(f"{'#'*80}")

        # Check if results already exist
        results_file = Path(f'reports/WEEK{week}_BACKTEST_COMPLETE.csv')
        if args.skip_existing and results_file.exists():
            print(f"\n⏭️  SKIPPING: Results already exist for Week {week}")
            # Load existing results for summary
            try:
                df = pd.read_csv(results_file)
                results_summary.append({
                    'week': week,
                    'status': 'SKIPPED',
                    'evaluated': len(df),
                    'wins': df['won'].sum(),
                    'losses': (~df['won']).sum(),
                    'win_rate': df['won'].mean() * 100 if len(df) > 0 else 0,
                    'total_wagered': df['wager'].sum(),
                    'total_profit': df['profit'].sum(),
                    'roi': (df['profit'].sum() / df['wager'].sum() * 100) if df['wager'].sum() > 0 else 0
                })
            except:
                pass
            continue

        # Step 1: Fetch player stats
        stats_file = Path(f'data/results/{season}/week{week}_player_stats.csv')
        if not stats_file.exists():
            step1 = run_command(
                f'.venv/bin/python scripts/backtest/fetch_week_player_stats.py {week} {season}',
                f"Fetch Week {week} player stats",
                allow_failure=True
            )
            if not step1:
                print(f"\n⚠️  WARNING: Could not fetch stats for Week {week}. Skipping.")
                results_summary.append({
                    'week': week,
                    'status': 'FAILED (No Stats)',
                    'evaluated': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_wagered': 0,
                    'total_profit': 0,
                    'roi': 0
                })
                continue
        else:
            print(f"\n✅ Player stats already exist for Week {week}")

        # Step 2: Run validation
        picks_file = Path(f'reports/all_picks_ranked_week{week}.csv')
        if not picks_file.exists():
            print(f"\n⚠️  WARNING: No picks file found for Week {week}. Skipping.")
            results_summary.append({
                'week': week,
                'status': 'FAILED (No Picks)',
                'evaluated': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_wagered': 0,
                'total_profit': 0,
                'roi': 0
            })
            continue

        step2 = run_command(
            f'.venv/bin/python scripts/backtest/validate_week_generic.py --week {week} --season {season}',
            f"Validate Week {week} picks",
            allow_failure=True
        )

        # Load results for summary
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                results_summary.append({
                    'week': week,
                    'status': 'SUCCESS',
                    'evaluated': len(df),
                    'wins': df['won'].sum(),
                    'losses': (~df['won']).sum(),
                    'win_rate': df['won'].mean() * 100 if len(df) > 0 else 0,
                    'total_wagered': df['wager'].sum(),
                    'total_profit': df['profit'].sum(),
                    'roi': (df['profit'].sum() / df['wager'].sum() * 100) if df['wager'].sum() > 0 else 0
                })
            except Exception as e:
                print(f"⚠️  Warning: Could not load results for Week {week}: {e}")
                results_summary.append({
                    'week': week,
                    'status': 'PARTIAL',
                    'evaluated': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_wagered': 0,
                    'total_profit': 0,
                    'roi': 0
                })
        else:
            results_summary.append({
                'week': week,
                'status': 'FAILED',
                'evaluated': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_wagered': 0,
                'total_profit': 0,
                'roi': 0
            })

    # Generate cumulative summary
    print(f"\n\n{'='*80}")
    print(f"BACKFILL SUMMARY")
    print(f"{'='*80}")

    if results_summary:
        summary_df = pd.DataFrame(results_summary)

        print(f"\n📊 Week-by-Week Results:")
        print(summary_df.to_string(index=False))

        # Calculate totals
        successful_weeks = summary_df[summary_df['status'].str.contains('SUCCESS|SKIPPED')]
        if len(successful_weeks) > 0:
            total_evaluated = successful_weeks['evaluated'].sum()
            total_wins = successful_weeks['wins'].sum()
            total_losses = successful_weeks['losses'].sum()
            total_wagered = successful_weeks['total_wagered'].sum()
            total_profit = successful_weeks['total_profit'].sum()

            print(f"\n💰 Cumulative Performance:")
            print(f"   Weeks Processed: {len(successful_weeks)}")
            print(f"   Total Bets: {total_evaluated}")
            print(f"   Total Wins: {total_wins}")
            print(f"   Total Losses: {total_losses}")
            print(f"   Overall Win Rate: {(total_wins/total_evaluated*100) if total_evaluated > 0 else 0:.1f}%")
            print(f"   Total Wagered: ${total_wagered:,.2f}")
            print(f"   Total Profit: ${total_profit:+,.2f}")
            print(f"   Cumulative ROI: {(total_profit/total_wagered*100) if total_wagered > 0 else 0:+.1f}%")

        # Save summary
        summary_file = Path(f'reports/HISTORICAL_BACKTEST_SUMMARY_WEEKS_{min(weeks)}_{max(weeks)}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ Summary saved to: {summary_file}")

    print(f"\n{'='*80}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
