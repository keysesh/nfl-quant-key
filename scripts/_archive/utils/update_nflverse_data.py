#!/usr/bin/env python3
"""
Update NFLverse Data Cache
==========================

Manually refresh NFLverse player statistics cache.

This script:
1. Checks current cache status
2. Fetches latest data from NFLverse API
3. Updates local cache files
4. Displays data summary and catch rates

Usage:
    python scripts/utils/update_nflverse_data.py                    # Update current season
    python scripts/utils/update_nflverse_data.py --season 2024      # Update specific season
    python scripts/utils/update_nflverse_data.py --force            # Force refresh even if cache fresh
    python scripts/utils/update_nflverse_data.py --all              # Update both 2024 and 2025

Run this script:
- Before generating predictions for a new week
- When you notice catch rate errors
- To verify data freshness

Author: NFL Quant Analytics
Last Updated: 2025-01-10
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.nflverse_data_loader import NFLverseDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_summary(loader: NFLverseDataLoader, season: int):
    """Print data summary for a season."""
    summary = loader.get_data_summary(season)

    if 'error' in summary:
        logger.error(f"\n❌ {season} Data: {summary['error']}")
        return

    print(f"\n📊 {season} Season Summary:")
    print(f"   Total Records: {summary['total_rows']:,}")
    print(f"   Weeks Available: {summary['weeks']}")
    print(f"   Week Count: {summary['week_count']}")
    print(f"   Positions: {', '.join(summary['positions'])}")

    print(f"\n🎯 Catch Rates by Position:")
    for pos in ['RB', 'WR', 'TE']:
        if pos in summary['catch_rates']:
            data = summary['catch_rates'][pos]
            if 'error' in data:
                print(f"   {pos}: ❌ {data['error']}")
            else:
                rate = data['rate']
                rec = data['receptions']
                tgt = data['targets']
                print(f"   {pos}: {rate:.3f} ({rec:,} receptions / {tgt:,} targets)")


def check_cache_status(loader: NFLverseDataLoader, season: int):
    """Check and display cache status."""
    status = loader.check_for_updates(season)

    print(f"\n📦 Cache Status for {season}:")
    if status['has_cache']:
        age_hours = status['cache_age_hours']
        print(f"   ✅ Cache exists (age: {age_hours:.1f} hours)")
        print(f"   Weeks in cache: {status['cache_weeks']}")

        if status['needs_update']:
            print(f"   ⚠️  Cache is stale (>24 hours old) - recommend refresh")
        else:
            print(f"   ✅ Cache is fresh (<24 hours old)")
    else:
        print(f"   ❌ No cache found - will fetch from NFLverse API")

    return status


def update_season(
    loader: NFLverseDataLoader,
    season: int,
    force: bool = False
):
    """Update data for a specific season."""
    print_header(f"Updating {season} Season Data")

    # Check current status
    status = check_cache_status(loader, season)

    # Determine if update needed
    if not force and status['has_cache'] and not status['needs_update']:
        logger.info(f"\n✅ {season} cache is fresh - skipping update")
        logger.info(f"   Use --force to refresh anyway")
        print_summary(loader, season)
        return

    # Fetch data
    logger.info(f"\n🔄 Fetching {season} data from NFLverse API...")
    try:
        df = loader.get_weekly_stats([season], force_refresh=True)
        logger.info(f"   ✅ Fetched {len(df):,} records")

        # Display summary
        print_summary(loader, season)

        logger.info(f"\n✅ Successfully updated {season} cache!")

    except Exception as e:
        logger.error(f"\n❌ Failed to update {season}: {e}")
        raise


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Update NFLverse player statistics cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/utils/update_nflverse_data.py                # Update current season (2025)
  python scripts/utils/update_nflverse_data.py --season 2024  # Update 2024 season
  python scripts/utils/update_nflverse_data.py --all          # Update both 2024 and 2025
  python scripts/utils/update_nflverse_data.py --force        # Force refresh even if fresh
  python scripts/utils/update_nflverse_data.py --clear        # Clear cache and redownload
        """
    )

    parser.add_argument(
        '--season',
        type=int,
        help='Specific season to update (default: current season 2025)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Update both 2024 and 2025 seasons'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if cache is fresh'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear cache before updating'
    )

    args = parser.parse_args()

    # Initialize loader
    loader = NFLverseDataLoader()

    print_header("NFLverse Data Cache Updater")

    # Determine which seasons to update
    if args.all:
        seasons = [2024, 2025]
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [2025]  # Default to current season

    # Clear cache if requested
    if args.clear:
        logger.info("\n🗑️  Clearing cache...")
        for season in seasons:
            loader.clear_cache(season)
        logger.info("   ✅ Cache cleared")

    # Update each season
    success_count = 0
    for season in seasons:
        try:
            update_season(loader, season, force=args.force)
            success_count += 1
        except Exception as e:
            logger.error(f"\n❌ Failed to update {season}: {e}")
            continue

    # Final summary
    print_header("Update Summary")
    if success_count == len(seasons):
        logger.info(f"\n✅ Successfully updated all {len(seasons)} season(s)!")
    else:
        logger.warning(
            f"\n⚠️  Updated {success_count}/{len(seasons)} season(s) "
            f"({len(seasons) - success_count} failed)"
        )

    # Next steps
    print("\n💡 Next Steps:")
    print("   1. Run predictions: python scripts/predict/generate_model_predictions.py 10")
    print("   2. Check data: python -c 'from nfl_quant.utils.nflverse_data_loader import NFLverseDataLoader; loader = NFLverseDataLoader(); print(loader.get_data_summary(2025))'")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
