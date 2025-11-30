#!/usr/bin/env python3
"""
Diagnostic Script: Validate Pre-Game Odds Filtering
=====================================================

This script validates the pre-game odds filtering system on historical data
to ensure it correctly identifies and filters out in-game/stale odds.

Usage:
    python scripts/utils/validate_odds_filtering.py --week 8
    python scripts/utils/validate_odds_filtering.py --week 8 --show-rejections
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.odds import (
    load_game_status_map,
    filter_pregame_odds,
    is_valid_pregame_odds
)


def load_historical_odds(week: int) -> pd.DataFrame:
    """Load historical odds data for validation.

    Args:
        week: NFL week number

    Returns:
        DataFrame with odds data
    """
    # Try multiple file locations
    possible_files = [
        Path(f'data/nfl_player_props_draftkings.csv'),
        Path(f'data/odds_week{week}_player_props.csv'),
        Path(f'data/odds_week{week}_comprehensive.csv'),
    ]

    for odds_file in possible_files:
        if odds_file.exists():
            print(f"📂 Loading odds from: {odds_file}")
            return pd.read_csv(odds_file)

    print(f"❌ No odds file found for Week {week}")
    print(f"   Tried: {[str(f) for f in possible_files]}")
    return pd.DataFrame()


def analyze_rejection_reasons(odds_df: pd.DataFrame, game_status_map: dict,
                               current_time: datetime) -> dict:
    """Analyze rejection reasons for each odds record.

    Args:
        odds_df: DataFrame with odds
        game_status_map: Game status mapping
        current_time: Current time for validation

    Returns:
        Dictionary with analysis results
    """
    rejection_summary = {
        'total': len(odds_df),
        'valid': 0,
        'rejected': 0,
        'reasons': {}
    }

    for idx, row in odds_df.iterrows():
        is_valid, reason = is_valid_pregame_odds(
            row,
            game_status_map,
            current_time,
            min_minutes_before_kickoff=5,
            max_hours_stale=24
        )

        if is_valid:
            rejection_summary['valid'] += 1
        else:
            rejection_summary['rejected'] += 1
            rejection_summary['reasons'][reason] = \
                rejection_summary['reasons'].get(reason, 0) + 1

    return rejection_summary


def print_diagnostic_report(summary: dict, show_details: bool = False):
    """Print formatted diagnostic report.

    Args:
        summary: Analysis summary dictionary
        show_details: Whether to show detailed breakdown
    """
    print("\n" + "=" * 80)
    print("ODDS FILTERING DIAGNOSTIC REPORT")
    print("=" * 80)

    total = summary['total']
    valid = summary['valid']
    rejected = summary['rejected']

    print(f"\n📊 Overall Statistics:")
    print(f"   Total odds records:     {total:,}")
    print(f"   Valid pre-game odds:    {valid:,} ({valid/total*100:.1f}%)")
    print(f"   Rejected odds:          {rejected:,} ({rejected/total*100:.1f}%)")

    if rejected > 0:
        print(f"\n🚫 Rejection Reason Breakdown:")
        reasons = sorted(summary['reasons'].items(),
                         key=lambda x: x[1], reverse=True)

        for reason, count in reasons:
            pct = count / rejected * 100
            print(f"   {reason:30s} {count:6,} ({pct:5.1f}%)")

        # Analysis and recommendations
        print(f"\n💡 Analysis:")

        # Check for high in-game percentage
        in_game_count = sum(count for reason, count in summary['reasons'].items()
                            if 'in_progress' in reason or 'complete' in reason)

        if in_game_count > total * 0.3:
            print(f"   ⚠️  High percentage of in-game/completed odds "
                  f"({in_game_count/total*100:.1f}%)")
            print(f"      → Games may have already started")
            print(f"      → Consider fetching odds earlier before kickoff")

        # Check for stale odds
        stale_count = sum(count for reason, count in summary['reasons'].items()
                          if 'stale' in reason)

        if stale_count > total * 0.1:
            print(f"   ⚠️  Significant stale odds detected "
                  f"({stale_count/total*100:.1f}%)")
            print(f"      → Odds may not have been refreshed recently")
            print(f"      → Re-fetch odds before making predictions")

        # Check for close-to-kickoff
        close_count = sum(count for reason, count in summary['reasons'].items()
                          if 'too_close_to_kickoff' in reason)

        if close_count > total * 0.2:
            print(f"   ⚠️  Many odds too close to kickoff "
                  f"({close_count/total*100:.1f}%)")
            print(f"      → Fetch odds >5 minutes before game starts")
            print(f"      → Increase min_minutes_before_kickoff if needed")

    print("\n" + "=" * 80)


def validate_filtering_accuracy(week: int, show_details: bool = False):
    """Validate filtering accuracy for a given week.

    Args:
        week: NFL week number
        show_details: Show detailed breakdown
    """
    print(f"\n🔍 Validating odds filtering for Week {week}")
    print("-" * 80)

    # Load historical odds
    odds_df = load_historical_odds(week)

    if odds_df.empty:
        print("❌ No odds data available for validation")
        return

    print(f"✅ Loaded {len(odds_df)} odds records")

    # Load game status map
    print(f"\n📋 Loading game status map...")
    game_status_map = load_game_status_map(week)

    if not game_status_map:
        print(f"⚠️  No game status data for Week {week}")
        print(f"   Validation will rely only on temporal checks")
    else:
        print(f"✅ Loaded status for {len(game_status_map)} games")

        # Show game status breakdown
        status_counts = {}
        for game_id, status in game_status_map.items():
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"\n   Game Status Breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"      {status:15s} {count:3d} games")

    # Simulate current time (use current time for validation)
    current_time = datetime.now(timezone.utc)

    print(f"\n⏰ Validation Time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Analyze rejection reasons
    print(f"\n🔬 Analyzing odds validation...")
    summary = analyze_rejection_reasons(odds_df, game_status_map, current_time)

    # Print diagnostic report
    print_diagnostic_report(summary, show_details)

    # Show sample rejections if requested
    if show_details and summary['rejected'] > 0:
        print("\n📋 Sample Rejected Odds (first 10):")
        print("-" * 80)

        rejection_count = 0
        for idx, row in odds_df.iterrows():
            is_valid, reason = is_valid_pregame_odds(
                row,
                game_status_map,
                current_time
            )

            if not is_valid:
                game_id = row.get('game_id', 'N/A')
                player = row.get('player_name', row.get('player', 'N/A'))
                market = row.get('market', 'N/A')
                commence = row.get('commence_time', 'N/A')

                print(f"\n   Game: {game_id}")
                print(f"   Player: {player}")
                print(f"   Market: {market}")
                print(f"   Commence: {commence}")
                print(f"   ❌ Rejected: {reason}")

                rejection_count += 1
                if rejection_count >= 10:
                    break


def compare_edge_rates(week: int):
    """Compare edge rates before and after filtering.

    This simulates the expected improvement in edge detection accuracy.

    Args:
        week: NFL week number
    """
    print(f"\n📈 Edge Rate Comparison (Simulated)")
    print("-" * 80)

    # Load odds
    odds_df = load_historical_odds(week)

    if odds_df.empty:
        return

    # Apply filtering
    filtered_df = filter_pregame_odds(odds_df, week=week)

    total_before = len(odds_df)
    total_after = len(filtered_df)
    filtered_out = total_before - total_after

    print(f"\n   Odds before filtering:  {total_before:,}")
    print(f"   Odds after filtering:   {total_after:,}")
    print(f"   Filtered out:           {filtered_out:,} "
          f"({filtered_out/total_before*100:.1f}%)")

    print(f"\n💡 Expected Impact:")
    print(f"   • Reduced false positive edges (in-game line movements)")
    print(f"   • More reliable pre-game predictions")
    print(f"   • Fewer 'stale line' bets that are no longer available")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate pre-game odds filtering system'
    )
    parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='NFL week number to validate'
    )
    parser.add_argument(
        '--show-rejections',
        action='store_true',
        help='Show detailed rejection examples'
    )
    parser.add_argument(
        '--compare-edges',
        action='store_true',
        help='Compare edge rates before/after filtering'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PRE-GAME ODDS FILTERING VALIDATION")
    print("=" * 80)

    # Run validation
    validate_filtering_accuracy(
        week=args.week,
        show_details=args.show_rejections
    )

    # Compare edge rates if requested
    if args.compare_edges:
        compare_edge_rates(week=args.week)

    print("\n✅ Validation complete!")
    print("\n💡 Next Steps:")
    print("   1. If rejection rate is high (>50%), re-fetch odds")
    print("   2. If stale odds detected, increase fetch frequency")
    print("   3. Review rejection reasons and adjust thresholds if needed")
    print()


if __name__ == '__main__':
    main()
