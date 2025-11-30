#!/usr/bin/env python3
"""
Monitor Historical Props Fetching Progress

Check progress of historical props fetching for 2023-2024 seasons.
"""

import sys
from pathlib import Path
import pandas as pd
import glob
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_props_progress():
    """Check what props have been fetched."""
    print("="*80)
    print("HISTORICAL PROPS FETCHING PROGRESS")
    print("="*80)
    print()

    backfill_dir = Path("data/historical/backfill")
    archive_file = Path("data/historical/live_archive/player_props_archive.csv")

    # Check individual files
    prop_files = sorted(glob.glob(str(backfill_dir / "player_props_history_*.csv")))

    print(f"📁 Backfill directory: {backfill_dir}")
    print(f"   Files found: {len(prop_files)}")
    print()

    if prop_files:
        total_props = 0
        for prop_file in prop_files[:10]:  # Show first 10
            try:
                df = pd.read_csv(prop_file)
                total_props += len(df)
                file_time = Path(prop_file).stat().st_mtime
                time_str = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
                print(f"   ✅ {Path(prop_file).name}: {len(df):,} props ({time_str})")
            except Exception as e:
                print(f"   ⚠️  {Path(prop_file).name}: Error reading ({e})")

        if len(prop_files) > 10:
            print(f"   ... and {len(prop_files) - 10} more files")

        print()
        print(f"   Total props in backfill files: {total_props:,}")

    # Check archive
    if archive_file.exists():
        try:
            archive_df = pd.read_csv(archive_file)
            print(f"📦 Archive file: {archive_file}")
            print(f"   Total props: {len(archive_df):,}")

            # Check by season if we have that column
            if 'season' in archive_df.columns:
                for season in [2023, 2024]:
                    season_count = len(archive_df[archive_df['season'] == season])
                    print(f"   Season {season}: {season_count:,} props")
        except Exception as e:
            print(f"   ⚠️  Error reading archive: {e}")
    else:
        print(f"📦 Archive file: Not found yet")

    print()
    print("="*80)
    print("FETCH STATUS")
    print("="*80)
    print()
    print("If fetching is still in progress, check:")
    print("  ps aux | grep backfill_historical_player_props")
    print()
    print("To check logs, look for output in:")
    print("  data/historical/backfill/")
    print()


if __name__ == "__main__":
    check_props_progress()
