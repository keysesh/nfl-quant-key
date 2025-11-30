#!/usr/bin/env python3
"""
Inventory Available Historical Data

Check what historical data we already have available for training.
"""

import sys
from pathlib import Path
import pandas as pd
import glob
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("="*80)
    print("HISTORICAL DATA INVENTORY")
    print("="*80)
    print()

    # Check archive
    archive = Path("data/historical/live_archive/player_props_archive.csv")
    if archive.exists():
        df = pd.read_csv(archive)
        print(f"✅ Player Props Archive: {len(df):,} props")
        print(f"   File: {archive}")
        if 'season' in df.columns:
            print(f"   Seasons: {sorted(df['season'].unique())}")
        print()

    # Check nflverse data
    nflverse_dir = Path("data/nflverse")
    if nflverse_dir.exists():
        print("✅ NFLverse Data:")
        pbp = nflverse_dir / "pbp_historical.parquet"
        weekly = nflverse_dir / "weekly_historical.parquet"
        if pbp.exists():
            df = pd.read_parquet(pbp)
            print(f"   PBP: {len(df):,} plays ({pbp})")
        if weekly.exists():
            df = pd.read_parquet(weekly)
            print(f"   Weekly Stats: {len(df):,} player-weeks ({weekly})")
            if 'season' in df.columns:
                print(f"   Seasons: {sorted(df['season'].unique())}")
        print()

    # Check matched props
    matched = Path("data/calibration/historical_props_matched.parquet")
    if matched.exists():
        df = pd.read_parquet(matched)
        print(f"✅ Matched Props: {len(df):,} props")
        print(f"   File: {matched}")
        print(f"   Props with outcomes: {df['bet_won'].notna().sum():,}")
        print()

    # Check simulated props
    simulated = Path("data/calibration/historical_props_simulated.parquet")
    if simulated.exists():
        df = pd.read_parquet(simulated)
        print(f"✅ Simulated Props: {len(df):,} props")
        print(f"   File: {simulated}")
        print(f"   Props with model_prob_raw: {df['model_prob_raw'].notna().sum():,}")
        print()

    # Check backfill files
    backfill_files = sorted(glob.glob("data/historical/backfill/player_props_history_*.csv"))
    if backfill_files:
        print(f"✅ Backfill Files: {len(backfill_files)} files")
        total_props = sum(len(pd.read_csv(f)) for f in backfill_files[:10])
        print(f"   Sample total: {total_props:,} props (first 10 files)")
        print(f"   Date range: {Path(backfill_files[0]).stem} to {Path(backfill_files[-1]).stem}")
        print()

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("We have:")
    print("  • Historical props from API (archive + backfill files)")
    print("  • NFLverse PBP and weekly stats (2023-2024)")
    print("  • Matched props to outcomes")
    print("  • Some simulated props (may need fixing)")
    print()


if __name__ == "__main__":
    main()































