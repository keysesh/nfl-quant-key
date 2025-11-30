#!/usr/bin/env python3
"""
Convert NFLverse historical data to canonical format.

This script:
1. Loads NFLverse data for 2023-2024 seasons
2. Transforms to canonical format using NFLVerseAdapter
3. Saves as Sleeper-compatible CSV files for backtest compatibility
4. Creates canonical parquet files for efficient loading

This enables the backtest system to use rich historical NFLverse data
for comprehensive calibrator training.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.adapters import NFLVerseAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("CONVERTING NFLVERSE DATA TO CANONICAL FORMAT")
    print("=" * 80)
    print()

    adapter = NFLVerseAdapter()

    # Seasons to convert
    seasons = [2023, 2024]

    # Output directories
    sleeper_dir = Path("data/sleeper_stats")
    canonical_dir = Path("data/stats_canonical")
    sleeper_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    total_weeks = 0
    total_players = 0

    for season in seasons:
        print(f"Processing season {season}...")
        print()

        # Check if NFLverse data exists for this season
        if not adapter.is_available(week=1, season=season):
            print(f"  ⚠️  NFLverse data not available for season {season}")
            print()
            continue

        # Get all available weeks for this season
        season_data = adapter._load_season_data(season)
        if season_data is None:
            print(f"  ⚠️  Could not load season data for {season}")
            print()
            continue

        weeks = sorted(season_data['week'].unique())
        print(f"  Found {len(weeks)} weeks: {weeks}")
        print()

        season_canonical_data = []

        for week in weeks:
            try:
                # Load in canonical format
                canonical = adapter.load_weekly_stats(week, season)

                # Save as Sleeper-compatible CSV
                sleeper_csv = sleeper_dir / f"stats_week{week}_{season}.csv"
                sleeper_format = _convert_to_sleeper_format(canonical)
                sleeper_format.to_csv(sleeper_csv, index=False)

                # Collect for season parquet
                season_canonical_data.append(canonical)

                total_weeks += 1
                total_players += len(canonical)

                print(f"    Week {week:2d}: {len(canonical):4d} players → {sleeper_csv.name}")

            except Exception as e:
                print(f"    Week {week:2d}: ❌ Failed - {e}")

        # Save season parquet
        if season_canonical_data:
            season_df = pd.concat(season_canonical_data, ignore_index=True)
            season_parquet = canonical_dir / f"weekly_{season}.parquet"
            season_df.to_parquet(season_parquet, index=False)
            print()
            print(f"  ✅ Saved canonical parquet: {season_parquet.name}")
            print(f"     Total: {len(season_df)} player-weeks")
        print()

    print("=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  Seasons processed: {seasons}")
    print(f"  Weeks converted: {total_weeks}")
    print(f"  Player-weeks: {total_players}")
    print()
    print(f"Output:")
    print(f"  Sleeper CSVs: {sleeper_dir}/ (for backtest compatibility)")
    print(f"  Canonical parquets: {canonical_dir}/ (for efficient loading)")
    print()
    print("Next steps:")
    print("  1. ✅ Re-run calibrator training with historical data")
    print("  2. ✅ Backtest system will now find 2023-2024 stats automatically")
    print()


def _convert_to_sleeper_format(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert canonical format back to Sleeper CSV format.

    This ensures backward compatibility with existing backtest code
    that expects Sleeper format.

    Args:
        canonical_df: DataFrame in canonical format

    Returns:
        DataFrame in Sleeper format
    """
    sleeper = pd.DataFrame()

    # Sleeper columns (exact order matters for compatibility)
    sleeper["player_id"] = canonical_df["player_id"]
    sleeper["player_name"] = canonical_df["player_name"]
    sleeper["position"] = canonical_df["position"]
    sleeper["team"] = canonical_df["team"]
    sleeper["week"] = canonical_df["week"]

    # Passing (canonical → sleeper mapping)
    sleeper["pass_yd"] = canonical_df["passing_yards"]
    sleeper["pass_att"] = canonical_df["passing_attempts"]
    sleeper["pass_cmp"] = canonical_df["passing_completions"]
    sleeper["pass_td"] = canonical_df["passing_tds"]
    sleeper["pass_int"] = canonical_df["interceptions"]

    # Rushing
    sleeper["rush_yd"] = canonical_df["rushing_yards"]
    sleeper["rush_att"] = canonical_df["rushing_attempts"]
    sleeper["rush_td"] = canonical_df["rushing_tds"]

    # Receiving
    sleeper["rec"] = canonical_df["receptions"]
    sleeper["rec_yd"] = canonical_df["receiving_yards"]
    sleeper["rec_td"] = canonical_df["receiving_tds"]
    sleeper["rec_tgt"] = canonical_df["targets"]

    return sleeper


if __name__ == "__main__":
    main()
