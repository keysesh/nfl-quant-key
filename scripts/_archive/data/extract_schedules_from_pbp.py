#!/usr/bin/env python3
"""
Extract Schedules from Existing PBP Data and Prepare for Props Fetching

This script:
1. Extracts schedule information from existing PBP data (2023-2024)
2. Creates schedule JSON files in format expected by backfill script
3. Provides ready-to-run commands for fetching historical props
"""

import sys
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_schedule_from_pbp(pbp_file: Path, season: int) -> List[Dict]:
    """Extract schedule from PBP data."""
    logger.info(f"Loading PBP data from {pbp_file}...")
    pbp = pd.read_parquet(pbp_file)
    
    # Get unique games
    games = pbp.groupby(['game_id', 'game_date', 'home_team', 'away_team', 'week']).first().reset_index()
    
    schedule = []
    for _, row in games.iterrows():
        # Parse game date
        game_date = pd.to_datetime(row['game_date']).date()
        
        schedule.append({
            'game_id': str(row['game_id']),
            'week': int(row['week']),
            'date': game_date.strftime('%Y-%m-%d'),
            'home_team': str(row['home_team']),
            'away_team': str(row['away_team']),
        })
    
    # Sort by week and date
    schedule.sort(key=lambda x: (x['week'], x['date']))
    
    logger.info(f"Extracted {len(schedule)} games for season {season}")
    return schedule


def save_schedule_json(schedule: List[Dict], output_path: Path):
    """Save schedule as JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    logger.info(f"✅ Saved schedule to {output_path}")


def main():
    print("="*80)
    print("EXTRACT SCHEDULES FROM EXISTING DATA")
    print("="*80)
    print()
    
    seasons = [2023, 2024]
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    schedules_created = []
    
    for season in seasons:
        # Find PBP file
        pbp_file = Path(f"data/nflverse/pbp_{season}.parquet")
        
        if not pbp_file.exists():
            logger.warning(f"PBP file not found: {pbp_file}")
            continue
        
        # Extract schedule
        schedule = extract_schedule_from_pbp(pbp_file, season)
        
        # Save schedule JSON
        schedule_file = raw_dir / f"sleeper_schedule_{season}.json"
        save_schedule_json(schedule, schedule_file)
        schedules_created.append((season, schedule_file, len(schedule)))
    
    # Print summary
    print("\n" + "="*80)
    print("SCHEDULE EXTRACTION COMPLETE")
    print("="*80)
    print()
    
    for season, schedule_file, game_count in schedules_created:
        print(f"✅ Season {season}: {game_count} games → {schedule_file}")
    
    # Print instructions for fetching props
    print("\n" + "="*80)
    print("NEXT STEPS: FETCH HISTORICAL PROPS")
    print("="*80)
    print()
    print("Now you can fetch historical player props using:")
    print()
    
    for season in seasons:
        print(f"# Season {season}")
        print(f"python scripts/fetch/backfill_historical_player_props.py \\")
        print(f"  --season {season} \\")
        print(f"  --weeks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \\")
        print(f"  --schedule-path data/raw/sleeper_schedule_{season}.json \\")
        print(f"  --markets player_pass_yds player_rush_yds player_reception_yds player_receptions player_anytime_td")
        print()
    
    print("Note: This requires ODDS_API_KEY environment variable to be set.")
    print()


if __name__ == "__main__":
    main()

