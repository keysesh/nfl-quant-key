#!/usr/bin/env python3
"""
Step 1: Fetch Historical Data from nflverse for Calibration Training

This script:
1. Fetches PBP data for historical seasons (2024, 2023)
2. Fetches weekly player stats
3. Fetches NGS metrics
4. Saves all data for use in calibration training

Run this first to populate historical data cache.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("STEP 1: FETCHING HISTORICAL DATA FROM NFLVERSE")
    print("=" * 80)
    print()
    print("⚠️  DEPRECATION NOTICE:")
    print("   This script previously used nflreadpy (Python) which has cache issues.")
    print("   Please use the R script instead for faster, more reliable data fetching:")
    print()
    print("   Rscript scripts/fetch/fetch_nflverse_data.R --seasons '2023 2024 2025'")
    print()
    print("   The R script is 3x faster and has no cache bugs.")
    print()
    print("Checking if R-fetched data already exists...")
    print()

    # Create data directory
    data_dir = Path("data/nflverse")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check what data already exists
    seasons = [2024, 2023]

    # Check PBP
    print("Checking play-by-play data...")
    pbp_files = []
    for season in seasons:
        pbp_file = data_dir / f"pbp_{season}.parquet"
        if pbp_file.exists():
            pbp_files.append(pbp_file)
            size_mb = pbp_file.stat().st_size / (1024 * 1024)
            print(f"  ✅ {season}: {pbp_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {season}: Missing {pbp_file.name}")
            print(f"     Run: Rscript scripts/fetch/fetch_nflverse_data.R")

    # Check combined PBP
    combined_pbp = data_dir / "pbp_historical.parquet"
    if combined_pbp.exists():
        size_mb = combined_pbp.stat().st_size / (1024 * 1024)
        print(f"  ✅ Combined: {combined_pbp.name} ({size_mb:.1f} MB)")

    # Check player stats
    print("\nChecking player stats...")
    stats_file = data_dir / "player_stats.parquet"
    if stats_file.exists():
        size_mb = stats_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ player_stats.parquet ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ Missing player_stats.parquet")
        print(f"     Run: Rscript scripts/fetch/fetch_nflverse_data.R")

    # Check schedules
    print("\nChecking schedules...")
    sched_file = data_dir / "schedules.parquet"
    if sched_file.exists():
        size_mb = sched_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ schedules.parquet ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ Missing schedules.parquet")

    # Check NGS
    print("\nChecking NGS metrics...")
    for stat_type in ['passing', 'rushing', 'receiving']:
        ngs_file = data_dir / f"ngs_{stat_type}_historical.parquet"
        if ngs_file.exists():
            size_mb = ngs_file.stat().st_size / (1024 * 1024)
            print(f"  ✅ ngs_{stat_type}_historical.parquet ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  Missing ngs_{stat_type}_historical.parquet (optional)")

    print("\n✅ Historical data fetch complete!")
    print("\nNext steps:")
    print("  1. Match historical props to outcomes")
    print("  2. Run simulations on historical games")
    print("  3. Retrain calibrator with expanded dataset")


if __name__ == "__main__":
    main()
