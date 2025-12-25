#!/usr/bin/env python3
"""
Consolidate Historical Injury Data from Multiple Sources

This script:
1. Consolidates Sleeper API snapshots into a historical injury dataset
2. Merges with NFLverse injury data
3. Creates a unified injury lookup for backtesting

Usage:
    python scripts/data/consolidate_injury_history.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_nfl_week_from_date(date: datetime, season: int = None) -> tuple:
    """
    Convert a date to NFL season and week.

    NFL season typically starts first Thursday after Labor Day.
    Week 1 is around Sept 7-13.
    """
    if season is None:
        # Determine season from date
        if date.month >= 9:
            season = date.year
        else:
            season = date.year - 1 if date.month <= 2 else date.year

    # Approximate week 1 start (first Thursday in September)
    # 2024: Sept 5, 2025: Sept 4, 2023: Sept 7
    week1_starts = {
        2023: datetime(2023, 9, 7),
        2024: datetime(2024, 9, 5),
        2025: datetime(2025, 9, 4),
    }

    week1_start = week1_starts.get(season, datetime(season, 9, 7))

    # Calculate week number
    days_since_week1 = (date - week1_start).days
    week = max(1, min(18, (days_since_week1 // 7) + 1))

    return season, week


def parse_sleeper_timestamp(filename: str) -> datetime:
    """Extract datetime from Sleeper snapshot filename."""
    # Format: injuries_sleeper_20251113_182911.csv
    match = re.search(r'injuries_sleeper_(\d{8})_(\d{6})\.csv', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    return None


def consolidate_sleeper_snapshots(injuries_dir: Path) -> pd.DataFrame:
    """
    Consolidate all Sleeper injury snapshots into a single dataset.

    For each snapshot:
    - Extract the date from filename
    - Map to NFL season/week
    - Keep the most recent snapshot per week for deduplication
    """
    logger.info("Consolidating Sleeper injury snapshots...")

    snapshots = []
    snapshot_files = sorted(injuries_dir.glob("injuries_sleeper_*.csv"))

    logger.info(f"Found {len(snapshot_files)} Sleeper snapshot files")

    for filepath in snapshot_files:
        timestamp = parse_sleeper_timestamp(filepath.name)
        if timestamp is None:
            continue

        season, week = get_nfl_week_from_date(timestamp)

        try:
            df = pd.read_csv(filepath)
            df['snapshot_date'] = timestamp
            df['season'] = season
            df['week'] = week
            df['source'] = 'sleeper'
            snapshots.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")

    if not snapshots:
        logger.warning("No Sleeper snapshots found")
        return pd.DataFrame()

    # Combine all snapshots
    combined = pd.concat(snapshots, ignore_index=True)
    logger.info(f"Combined {len(combined)} injury records from {len(snapshots)} snapshots")

    # Deduplicate: keep the latest snapshot per player/season/week
    combined = combined.sort_values('snapshot_date', ascending=False)
    combined = combined.drop_duplicates(
        subset=['player_id', 'season', 'week'],
        keep='first'
    )

    logger.info(f"After deduplication: {len(combined)} unique player/week records")

    return combined


def load_nflverse_injuries(nflverse_dir: Path) -> pd.DataFrame:
    """Load NFLverse injury data."""
    injuries_path = nflverse_dir / 'injuries.parquet'

    if not injuries_path.exists():
        logger.warning(f"NFLverse injuries not found at {injuries_path}")
        return pd.DataFrame()

    df = pd.read_parquet(injuries_path)
    df['source'] = 'nflverse'

    logger.info(f"Loaded {len(df)} NFLverse injury records")
    logger.info(f"NFLverse seasons: {sorted(df['season'].unique())}")

    return df


def normalize_injury_status(status: str) -> int:
    """
    Normalize injury status to numeric encoding.

    Returns:
        0: None/Probable/Full
        1: Questionable
        2: Doubtful
        3: Out/IR/PUP
    """
    if pd.isna(status):
        return 0

    status = str(status).lower().strip()

    if status in ['out', 'ir', 'pup', 'cov', 'injured reserve']:
        return 3
    elif status in ['doubtful', 'd']:
        return 2
    elif status in ['questionable', 'q']:
        return 1
    else:  # probable, none, etc.
        return 0


def normalize_practice_status(status: str) -> int:
    """
    Normalize practice status to numeric encoding.

    Returns:
        0: Full participation
        1: Limited participation
        2: Did not participate (DNP)
    """
    if pd.isna(status):
        return 0

    status = str(status).lower().strip()

    if 'did not' in status or 'dnp' in status:
        return 2
    elif 'limited' in status:
        return 1
    else:  # full participation
        return 0


def create_unified_injury_lookup(
    sleeper_df: pd.DataFrame,
    nflverse_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a unified injury lookup combining Sleeper and NFLverse data.

    Priority: NFLverse for historical (more complete), Sleeper for current season
    """
    logger.info("Creating unified injury lookup...")

    records = []

    # Process NFLverse data
    if len(nflverse_df) > 0:
        for _, row in nflverse_df.iterrows():
            records.append({
                'player_id': row.get('gsis_id', ''),
                'player_name': row.get('full_name', ''),
                'team': row.get('team', ''),
                'position': row.get('position', ''),
                'season': row.get('season'),
                'week': row.get('week'),
                'injury_status': row.get('report_status', ''),
                'injury_status_encoded': normalize_injury_status(row.get('report_status')),
                'practice_status': row.get('practice_status', ''),
                'practice_status_encoded': normalize_practice_status(row.get('practice_status')),
                'injury_body_part': row.get('report_primary_injury', ''),
                'source': 'nflverse',
            })

    # Process Sleeper data
    if len(sleeper_df) > 0:
        for _, row in sleeper_df.iterrows():
            records.append({
                'player_id': str(row.get('player_id', '')),
                'player_name': row.get('player_name', ''),
                'team': row.get('team', ''),
                'position': row.get('position', ''),
                'season': row.get('season'),
                'week': row.get('week'),
                'injury_status': row.get('injury_status', ''),
                'injury_status_encoded': normalize_injury_status(row.get('injury_status')),
                'practice_status': '',  # Sleeper doesn't have practice status
                'practice_status_encoded': 0,
                'injury_body_part': row.get('injury_body_part', ''),
                'source': 'sleeper',
            })

    unified = pd.DataFrame(records)

    if len(unified) == 0:
        logger.warning("No injury records to unify")
        return pd.DataFrame()

    # Deduplicate: prefer NFLverse over Sleeper for same player/week
    unified['source_priority'] = unified['source'].map({'nflverse': 0, 'sleeper': 1})
    unified = unified.sort_values(['source_priority', 'injury_status_encoded'], ascending=[True, False])
    unified = unified.drop_duplicates(
        subset=['player_name', 'team', 'season', 'week'],
        keep='first'
    )
    unified = unified.drop(columns=['source_priority'])

    logger.info(f"Unified injury lookup: {len(unified)} records")
    logger.info(f"By source: {unified['source'].value_counts().to_dict()}")
    logger.info(f"By season: {unified.groupby('season').size().to_dict()}")

    return unified


def create_player_name_mapping(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create player name to ID mapping for joining with backtest data.

    Backtest data uses player names, while injury data may use IDs.
    """
    if len(unified_df) == 0:
        return pd.DataFrame()

    # Create normalized name for matching
    mapping = unified_df[['player_id', 'player_name', 'team', 'position']].copy()
    mapping = mapping.drop_duplicates()

    # Normalize player names (lowercase, remove punctuation)
    mapping['player_name_norm'] = (
        mapping['player_name']
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.strip()
    )

    return mapping


def main():
    """Main execution."""

    injuries_dir = PROJECT_ROOT / 'data' / 'injuries'
    nflverse_dir = PROJECT_ROOT / 'data' / 'nflverse'
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Consolidate Sleeper snapshots
    sleeper_df = consolidate_sleeper_snapshots(injuries_dir)

    # 2. Load NFLverse injuries
    nflverse_df = load_nflverse_injuries(nflverse_dir)

    # 3. Create unified lookup
    unified_df = create_unified_injury_lookup(sleeper_df, nflverse_df)

    if len(unified_df) == 0:
        logger.error("No injury data to save")
        return

    # 4. Save outputs
    # Full unified dataset
    unified_path = output_dir / 'unified_injury_history.parquet'
    unified_df.to_parquet(unified_path, index=False)
    logger.info(f"Saved unified injury history: {unified_path}")

    # CSV version for inspection
    unified_csv_path = output_dir / 'unified_injury_history.csv'
    unified_df.to_csv(unified_csv_path, index=False)
    logger.info(f"Saved CSV version: {unified_csv_path}")

    # 5. Create player name mapping
    mapping_df = create_player_name_mapping(unified_df)
    if len(mapping_df) > 0:
        mapping_path = output_dir / 'injury_player_mapping.parquet'
        mapping_df.to_parquet(mapping_path, index=False)
        logger.info(f"Saved player mapping: {mapping_path}")

    # 6. Summary stats
    print("\n" + "="*60)
    print("UNIFIED INJURY HISTORY SUMMARY")
    print("="*60)
    print(f"\nTotal records: {len(unified_df):,}")
    print(f"\nBy source:")
    print(unified_df['source'].value_counts().to_string())
    print(f"\nBy season:")
    print(unified_df.groupby('season').size().to_string())
    print(f"\nInjury status distribution:")
    print(unified_df['injury_status_encoded'].value_counts().sort_index().to_string())
    print(f"\nOutput saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
