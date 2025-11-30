#!/usr/bin/env python3
"""
Summary: Historical Data Collection Status

Shows what we've collected and what remains to be fetched.
"""

import sys
from pathlib import Path
import pandas as pd
import glob
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("="*80)
    print("HISTORICAL DATA COLLECTION SUMMARY")
    print("="*80)
    print()

    # Check archive
    archive = Path('data/historical/live_archive/player_props_archive.csv')

    if archive.exists():
        df = pd.read_csv(archive)
        print(f"✅ Archive: {len(df):,} props collected")

        # Extract date info
        if 'commence_time' in df.columns:
            df['commence_dt'] = pd.to_datetime(df['commence_time'], errors='coerce')
            df['year'] = df['commence_dt'].dt.year
            df['month'] = df['commence_dt'].dt.month

            print(f"\nBy Season:")
            for year in sorted(df['year'].dropna().unique()):
                year_data = df[df['year'] == year]
                print(f"  {int(year)}: {len(year_data):,} props")
                print(f"    Events: {year_data['event_id'].nunique()}")

        print(f"\nMarkets:")
        market_counts = df['market'].value_counts()
        for market, count in market_counts.head(10).items():
            print(f"  {market}: {count:,}")

    # Check backfill files
    backfill_files = sorted(glob.glob('data/historical/backfill/player_props_history_*.csv'))
    print(f"\n📁 Backfill files: {len(backfill_files)}")

    # Check what we have vs what we need
    print(f"\n{'='*80}")
    print("DATA STATUS")
    print(f"{'='*80}\n")

    print("✅ Have:")
    print("  • PBP data: 2023 + 2024")
    print("  • Weekly stats: 2023 + 2024")
    print("  • Schedules: 2023 + 2024 (extracted from PBP)")
    print(f"  • Historical props: {len(df):,} props" if archive.exists() else "  • Historical props: Collecting...")

    print("\n⚠️  Status:")
    print("  • Some 401 API errors encountered (quota/rate limit)")
    print("  • Fetching continues for available dates")
    print("  • Scripts updated to handle errors gracefully")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    print("1. Continue fetching (already running in background)")
    print("2. Match props to outcomes:")
    print("   python scripts/data/match_historical_props.py")
    print("3. Run simulations:")
    print("   python scripts/data/simulate_historical_props.py")
    print("4. Retrain calibrator:")
    print("   python scripts/train/retrain_calibrator_nflverse.py")
    print()


if __name__ == "__main__":
    main()
