#!/usr/bin/env python3
"""
Check Simulation Progress

Quick script to check if historical props simulation is making progress.
"""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("="*80)
    print("SIMULATION PROGRESS CHECK")
    print("="*80)
    print()
    
    # Check trailing stats
    stats_file = Path("data/nflverse/historical_trailing_stats.json")
    if stats_file.exists():
        stats = json.load(open(stats_file))
        print(f"✅ Trailing stats: {len(stats):,} player-weeks")
        if stats:
            sample_key = list(stats.keys())[0]
            print(f"   Sample keys: {list(stats[sample_key].keys())}")
            # Check for required fields
            required = ['trailing_yards_per_opportunity', 'trailing_td_rate', 
                       'projected_team_total', 'projected_opponent_total', 'projected_game_script']
            missing = [k for k in required if k not in stats[sample_key]]
            if missing:
                print(f"   ⚠️  Missing fields: {missing}")
            else:
                print(f"   ✅ All required fields present")
    else:
        print("❌ Trailing stats file not found")
    
    print()
    
    # Check input props
    input_file = Path("data/calibration/historical_props_matched.parquet")
    if input_file.exists():
        df = pd.read_parquet(input_file)
        print(f"✅ Input props: {len(df):,}")
        print(f"   Props with model_prob_raw: {(~df['model_prob_raw'].isna()).sum():,}")
        print(f"   Props without model_prob_raw: {df['model_prob_raw'].isna().sum():,}")
    else:
        print("❌ Input props file not found")
    
    print()
    
    # Check output file
    output_file = Path("data/calibration/historical_props_simulated.parquet")
    if output_file.exists():
        df = pd.read_parquet(output_file)
        print(f"✅ Simulated props: {len(df):,}")
        print(f"   Props with model_prob_raw: {(~df['model_prob_raw'].isna()).sum():,}")
        print(f"   Win rate: {df['bet_won'].mean():.2%}")
    else:
        print("⏳ Output file not created yet (simulation in progress)")
    
    print()
    print("="*80)
    print("STATUS")
    print("="*80)
    print()
    print("The simulation script is running in the background.")
    print("It will:")
    print("  1. Generate trailing stats for all player-weeks")
    print("  2. Run simulations on historical props")
    print("  3. Save results to data/calibration/historical_props_simulated.parquet")
    print()
    print("To check progress, run this script again or check:")
    print("  ps aux | grep simulate_historical_props")
    print()


if __name__ == "__main__":
    main()
































