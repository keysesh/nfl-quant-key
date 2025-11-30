#!/usr/bin/env python3
"""
NFL QUANT - Line Movement Tracker

Fetches and stores prop line snapshots over time to track line movement.
Line movement is a key signal - when lines move toward UNDER, sharp money
is betting UNDER, which we should follow.

Usage:
    # Fetch current snapshot (run every 4-6 hours Thursday-Sunday)
    python scripts/fetch/fetch_line_movement.py --snapshot

    # Calculate line movement for current week
    python scripts/fetch/fetch_line_movement.py --calculate --week 13

    # Set up automated fetching (shows cron command)
    python scripts/fetch/fetch_line_movement.py --setup-cron

API: Uses the-odds-api.com (same as fetch_player_props_live.py)
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.draftkings_client import DKClient, CORE_MARKETS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = PROJECT_ROOT / 'data' / 'line_movement' / 'snapshots'
MOVEMENT_DIR = PROJECT_ROOT / 'data' / 'line_movement'


def fetch_and_store_snapshot() -> Optional[Path]:
    """
    Fetch current prop lines and store as timestamped snapshot.

    Returns path to saved snapshot file.
    """
    logger.info("Fetching current prop lines...")

    client = DKClient()
    props_df = client.get_all_props(markets=CORE_MARKETS)

    if props_df.empty:
        logger.warning("No props returned from API")
        return None

    logger.info(f"Fetched {len(props_df)} prop lines")

    # Add fetch timestamp
    timestamp = datetime.now()
    props_df['snapshot_timestamp'] = timestamp.isoformat()
    props_df['snapshot_id'] = timestamp.strftime('%Y%m%d_%H%M%S')

    # Determine week from game dates
    if 'commence_time' in props_df.columns:
        try:
            from nfl_quant.utils.season_utils import get_current_week
            week = get_current_week()
        except:
            week = 13  # Default
    else:
        week = 13

    props_df['week'] = week

    # Save snapshot
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}_week{week}.csv"
    filepath = SNAPSHOTS_DIR / filename

    props_df.to_csv(filepath, index=False)
    logger.info(f"Saved snapshot to {filepath}")
    logger.info(f"API quota remaining: {client.remaining}")

    return filepath


def load_week_snapshots(week: int) -> pd.DataFrame:
    """Load all snapshots for a given week."""
    snapshots = []

    for f in SNAPSHOTS_DIR.glob(f"snapshot_*_week{week}.csv"):
        try:
            df = pd.read_csv(f)
            snapshots.append(df)
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if not snapshots:
        return pd.DataFrame()

    combined = pd.concat(snapshots, ignore_index=True)
    logger.info(f"Loaded {len(snapshots)} snapshots with {len(combined)} total lines for Week {week}")

    return combined


def calculate_line_movement(week: int) -> pd.DataFrame:
    """
    Calculate line movement for each player/market.

    Returns DataFrame with:
    - player, market, current_line, opening_line
    - line_movement (current - opening, positive = line moved up)
    - movement_direction ('up', 'down', 'stable')
    - n_snapshots (how many snapshots we have)
    """
    snapshots = load_week_snapshots(week)

    if snapshots.empty:
        logger.warning(f"No snapshots found for Week {week}")
        return pd.DataFrame()

    # Parse timestamps
    snapshots['snapshot_dt'] = pd.to_datetime(snapshots['snapshot_timestamp'])

    # Group by player + market
    movement_data = []

    for (player, market), group in snapshots.groupby(['player_name', 'market']):
        # Sort by time
        group = group.sort_values('snapshot_dt')

        if len(group) < 1:
            continue

        opening_line = group.iloc[0]['line']
        current_line = group.iloc[-1]['line']
        opening_time = group.iloc[0]['snapshot_dt']
        current_time = group.iloc[-1]['snapshot_dt']

        # Calculate movement
        line_movement = current_line - opening_line
        movement_pct = (line_movement / opening_line * 100) if opening_line > 0 else 0

        # Direction
        if line_movement > 0.5:
            direction = 'up'  # Line increased = sharp money on OVER
        elif line_movement < -0.5:
            direction = 'down'  # Line decreased = sharp money on UNDER
        else:
            direction = 'stable'

        # Also track odds movement
        opening_over_odds = group.iloc[0].get('over_price', -110)
        current_over_odds = group.iloc[-1].get('over_price', -110)
        odds_movement = current_over_odds - opening_over_odds

        movement_data.append({
            'player': player,
            'market': market,
            'opening_line': opening_line,
            'current_line': current_line,
            'line_movement': line_movement,
            'line_movement_pct': movement_pct,
            'movement_direction': direction,
            'opening_over_odds': opening_over_odds,
            'current_over_odds': current_over_odds,
            'odds_movement': odds_movement,
            'n_snapshots': len(group),
            'opening_time': opening_time,
            'current_time': current_time,
            'hours_tracked': (current_time - opening_time).total_seconds() / 3600,
            'week': week,
        })

    movement_df = pd.DataFrame(movement_data)
    logger.info(f"Calculated movement for {len(movement_df)} player/market combinations")

    # Save movement data
    MOVEMENT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MOVEMENT_DIR / f'line_movement_week{week}.csv'
    movement_df.to_csv(output_path, index=False)
    logger.info(f"Saved line movement to {output_path}")

    return movement_df


def display_significant_movement(movement_df: pd.DataFrame):
    """Display lines with significant movement."""
    if movement_df.empty:
        print("No movement data available")
        return

    print("\n" + "=" * 80)
    print("LINE MOVEMENT ANALYSIS")
    print("=" * 80)

    # Lines that moved DOWN (sharp money on UNDER - follow this!)
    down = movement_df[movement_df['movement_direction'] == 'down'].sort_values('line_movement')
    if not down.empty:
        print("\n🔻 LINES THAT MOVED DOWN (Sharp money on UNDER):")
        print("-" * 60)
        for _, row in down.head(15).iterrows():
            market_name = row['market'].replace('player_', '').replace('_', ' ')
            print(f"  {row['player']:<25} {market_name:<15} {row['opening_line']:>6.1f} → {row['current_line']:>6.1f} ({row['line_movement']:+.1f})")

    # Lines that moved UP
    up = movement_df[movement_df['movement_direction'] == 'up'].sort_values('line_movement', ascending=False)
    if not up.empty:
        print("\n🔺 LINES THAT MOVED UP (Sharp money on OVER):")
        print("-" * 60)
        for _, row in up.head(10).iterrows():
            market_name = row['market'].replace('player_', '').replace('_', ' ')
            print(f"  {row['player']:<25} {market_name:<15} {row['opening_line']:>6.1f} → {row['current_line']:>6.1f} ({row['line_movement']:+.1f})")

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print(f"  Total player/markets tracked: {len(movement_df)}")
    print(f"  Lines moved DOWN: {len(down)} ({len(down)/len(movement_df)*100:.1f}%)")
    print(f"  Lines moved UP: {len(up)} ({len(up)/len(movement_df)*100:.1f}%)")
    print(f"  Lines stable: {len(movement_df) - len(down) - len(up)}")
    print(f"  Avg snapshots per line: {movement_df['n_snapshots'].mean():.1f}")
    print(f"  Hours tracked: {movement_df['hours_tracked'].max():.1f}")
    print("=" * 80)


def show_cron_setup():
    """Display cron setup instructions."""
    script_path = Path(__file__).resolve()

    print("\n" + "=" * 80)
    print("CRON SETUP FOR AUTOMATED LINE FETCHING")
    print("=" * 80)
    print()
    print("Add these lines to your crontab (run 'crontab -e'):")
    print()
    print("# NFL QUANT - Fetch line snapshots every 4 hours Thu-Sun")
    print(f"0 8,12,16,20 * * 4,5,6,0 cd {PROJECT_ROOT} && .venv/bin/python {script_path} --snapshot >> logs/line_fetch.log 2>&1")
    print()
    print("# Capture final closing lines 30 mins before Sunday games")
    print(f"30 12 * * 0 cd {PROJECT_ROOT} && .venv/bin/python {script_path} --snapshot >> logs/line_fetch.log 2>&1")
    print()
    print("This will:")
    print("  - Fetch at 8am, 12pm, 4pm, 8pm on Thu/Fri/Sat/Sun")
    print("  - Give you ~4+ snapshots per week to track movement")
    print()
    print("Alternative: Use launchd on macOS (see Apple docs)")
    print("=" * 80)


def get_movement_for_predictions(week: int) -> Dict[str, Dict]:
    """
    Get line movement data formatted for integration with V5 classifier.

    Returns dict keyed by (player, market) with movement features.
    """
    movement_path = MOVEMENT_DIR / f'line_movement_week{week}.csv'

    if not movement_path.exists():
        logger.warning(f"No movement data for Week {week}")
        return {}

    movement_df = pd.read_csv(movement_path)

    result = {}
    for _, row in movement_df.iterrows():
        key = (row['player'].lower().strip(), row['market'])
        result[key] = {
            'line_movement': row['line_movement'],
            'line_movement_pct': row['line_movement_pct'],
            'movement_direction': row['movement_direction'],
            'sharp_under': row['movement_direction'] == 'down',  # Key signal!
            'sharp_over': row['movement_direction'] == 'up',
            'n_snapshots': row['n_snapshots'],
        }

    return result


def import_snapshot_from_csv(csv_path: str, week: int = None) -> Optional[Path]:
    """
    Import a snapshot from an existing CSV file.
    Use this if you fetch props manually or from another source.
    """
    input_path = Path(csv_path)
    if not input_path.exists():
        logger.error(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} lines from {csv_path}")

    # Add timestamp if missing
    timestamp = datetime.now()
    if 'snapshot_timestamp' not in df.columns:
        df['snapshot_timestamp'] = timestamp.isoformat()
    if 'snapshot_id' not in df.columns:
        df['snapshot_id'] = timestamp.strftime('%Y%m%d_%H%M%S')

    # Determine week
    if week is None:
        try:
            from nfl_quant.utils.season_utils import get_current_week
            week = get_current_week()
        except:
            week = 13
    df['week'] = week

    # Normalize column names if needed
    col_mapping = {
        'player': 'player_name',
        'line': 'line',
        'over_odds': 'over_price',
        'under_odds': 'under_price',
    }
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Save as snapshot
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}_week{week}.csv"
    filepath = SNAPSHOTS_DIR / filename

    df.to_csv(filepath, index=False)
    logger.info(f"Imported snapshot to {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description='Track NFL prop line movement')
    parser.add_argument('--snapshot', action='store_true', help='Fetch and store current snapshot')
    parser.add_argument('--import-csv', type=str, help='Import snapshot from existing CSV file')
    parser.add_argument('--calculate', action='store_true', help='Calculate line movement')
    parser.add_argument('--week', type=int, help='Week number for calculations')
    parser.add_argument('--setup-cron', action='store_true', help='Show cron setup instructions')
    parser.add_argument('--list-snapshots', action='store_true', help='List available snapshots')

    args = parser.parse_args()

    if args.setup_cron:
        show_cron_setup()
        return

    if args.list_snapshots:
        snapshots = list(SNAPSHOTS_DIR.glob("snapshot_*.csv"))
        print(f"\nFound {len(snapshots)} snapshots:")
        for s in sorted(snapshots)[-20:]:
            print(f"  {s.name}")
        return

    if args.import_csv:
        filepath = import_snapshot_from_csv(args.import_csv, args.week)
        if filepath:
            print(f"\n✅ Snapshot imported: {filepath}")
        return

    if args.snapshot:
        filepath = fetch_and_store_snapshot()
        if filepath:
            print(f"\n✅ Snapshot saved: {filepath}")
        else:
            print("\n❌ Failed to fetch snapshot. API key may be deactivated.")
            print("   Options:")
            print("   1. Renew API subscription at https://the-odds-api.com")
            print("   2. Import CSV manually: --import-csv <path> --week <N>")
        return

    if args.calculate:
        if not args.week:
            try:
                from nfl_quant.utils.season_utils import get_current_week
                args.week = get_current_week()
            except:
                args.week = 13

        movement_df = calculate_line_movement(args.week)
        display_significant_movement(movement_df)
        return

    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
