#!/usr/bin/env python3
"""
Match Historical Props to Outcomes for 2023-2024

This script:
1. Loads existing player stats from nflverse (2023-2024)
2. Checks for historical props files
3. Provides instructions for fetching missing props
4. Creates a matching script to connect props to outcomes
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict
import glob

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_files(seasons: List[int]) -> Dict:
    """Check what data files we have."""
    results = {}
    
    for season in seasons:
        results[season] = {
            'pbp': [],
            'weekly_stats': [],
            'props': [],
            'schedules': [],
        }
        
        # Check PBP
        pbp_files = glob.glob(f"data/**/*pbp*{season}*.parquet", recursive=True)
        results[season]['pbp'] = pbp_files
        
        # Check weekly stats
        weekly_files = glob.glob(f"data/**/*weekly*{season}*.parquet", recursive=True)
        weekly_files += glob.glob(f"data/**/*player_stats*{season}*.parquet", recursive=True)
        results[season]['weekly_stats'] = weekly_files
        
        # Check props
        props_files = glob.glob(f"data/**/*prop*{season}*.csv", recursive=True)
        props_files += glob.glob(f"data/**/*odds*{season}*.csv", recursive=True)
        results[season]['props'] = props_files
        
        # Check schedules
        schedule_files = glob.glob(f"data/**/*schedule*{season}*.json", recursive=True)
        schedule_files += glob.glob(f"data/**/*schedule*{season}*.csv", recursive=True)
        results[season]['schedules'] = schedule_files
    
    return results


def summarize_data(results: Dict) -> None:
    """Print summary of available data."""
    print("\n" + "="*80)
    print("DATA AVAILABILITY SUMMARY")
    print("="*80)
    print()
    
    for season in sorted(results.keys()):
        print(f"📅 Season {season}:")
        
        # PBP
        if results[season]['pbp']:
            for f in results[season]['pbp']:
                try:
                    df = pd.read_parquet(f)
                    print(f"   ✅ PBP: {len(df):,} plays ({f})")
                except:
                    print(f"   ⚠️  PBP: File exists but can't read ({f})")
        else:
            print(f"   ❌ PBP: Not found")
        
        # Weekly stats
        if results[season]['weekly_stats']:
            for f in results[season]['weekly_stats']:
                try:
                    df = pd.read_parquet(f)
                    print(f"   ✅ Weekly Stats: {len(df):,} player-weeks ({f})")
                except:
                    print(f"   ⚠️  Weekly Stats: File exists but can't read ({f})")
        else:
            print(f"   ❌ Weekly Stats: Not found")
        
        # Props
        if results[season]['props']:
            total_props = 0
            for f in results[season]['props']:
                try:
                    df = pd.read_csv(f)
                    total_props += len(df)
                    print(f"   ✅ Props: {len(df):,} props ({f})")
                except:
                    print(f"   ⚠️  Props: File exists but can't read ({f})")
            if total_props > 0:
                print(f"   📊 Total Props: {total_props:,}")
        else:
            print(f"   ❌ Props: Not found")
        
        # Schedules
        if results[season]['schedules']:
            for f in results[season]['schedules']:
                print(f"   ✅ Schedule: {f}")
        else:
            print(f"   ❌ Schedule: Not found")
        
        print()


def print_next_steps(results: Dict, seasons: List[int]) -> None:
    """Print next steps for fetching missing data."""
    print("\n" + "="*80)
    print("NEXT STEPS TO FETCH MISSING DATA")
    print("="*80)
    print()
    
    needs_props = [s for s in seasons if not results[s]['props']]
    needs_schedules = [s for s in seasons if not results[s]['schedules']]
    
    if needs_props:
        print("📥 To fetch historical player props:")
        print()
        print("   1. Ensure ODDS_API_KEY is set in environment")
        print("   2. For each season, you'll need a schedule file")
        print("   3. Run the backfill script:")
        print()
        for season in needs_props:
            print(f"   Season {season}:")
            print(f"   python scripts/fetch/backfill_historical_player_props.py \\")
            print(f"     --season {season} \\")
            print(f"     --weeks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \\")
            print(f"     --schedule-path data/raw/sleeper_schedule_{season}.json")
            print()
    
    if needs_schedules:
        print("📅 To fetch schedules:")
        print()
        print("   Schedules can be fetched from Sleeper API or nflverse.")
        print("   You can also manually create schedule files if needed.")
        print()
    
    print("🔗 To match props to outcomes:")
    print()
    print("   1. Use scripts/data/match_historical_props.py")
    print("   2. This will match props to actual outcomes from weekly stats")
    print("   3. Then run simulations on matched props")
    print()


def main():
    print("="*80)
    print("HISTORICAL DATA CHECKER (2023-2024)")
    print("="*80)
    
    seasons = [2023, 2024]
    
    # Check what we have
    results = check_data_files(seasons)
    
    # Summarize
    summarize_data(results)
    
    # Print next steps
    print_next_steps(results, seasons)
    
    print("\n" + "="*80)
    print("✅ CHECK COMPLETE")
    print("="*80)
    print()


if __name__ == "__main__":
    main()

