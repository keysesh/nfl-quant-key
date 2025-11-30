#!/usr/bin/env python3
"""
Fetch NFLverse data for both historical (2024) and current (2025) seasons.

NFLverse provides comprehensive NFL data including:
- Weekly player stats (passing, rushing, receiving)
- Play-by-play data
- Game schedules and results
- Historical data for all seasons

This script:
1. Fetches 2024 data (historical - for training/backtesting)
2. Fetches 2025 data (current season - for live predictions)
3. Saves both to separate files for clarity
4. Creates a combined historical file for easy access
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season, get_training_seasons

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# NFLverse data URLs
NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

URLS = {
    'weekly': f"{NFLVERSE_BASE}/player_stats/player_stats.parquet",
    'pbp': f"{NFLVERSE_BASE}/pbp/play_by_play_{{season}}.parquet",
    'games': f"{NFLVERSE_BASE}/games/games.parquet",
}


def fetch_weekly_stats(seasons=[2024, 2025]):
    """
    Fetch weekly player stats for specified seasons.

    NFLverse provides a single file with ALL seasons.
    We'll filter and save 2024 and 2025 separately.
    """
    logger.info("=" * 80)
    logger.info("FETCHING WEEKLY PLAYER STATS")
    logger.info("=" * 80)
    logger.info("")

    try:
        logger.info(f"📥 Downloading from: {URLS['weekly']}")

        # NFLverse weekly stats contains ALL seasons
        df_all = pd.read_parquet(URLS['weekly'])

        logger.info(f"✅ Downloaded {len(df_all):,} player-week records")
        logger.info(f"   Seasons available: {sorted(df_all['season'].unique())}")
        logger.info("")

        output_dir = Path("data/nflverse")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each requested season separately
        for season in seasons:
            df_season = df_all[df_all['season'] == season].copy()

            if len(df_season) == 0:
                logger.warning(f"⚠️  No data found for {season} season")
                continue

            # Save season-specific file
            output_file = output_dir / f"weekly_{season}.parquet"
            df_season.to_parquet(output_file, index=False)

            weeks = sorted(df_season['week'].unique())
            logger.info(f"✅ {season} Season:")
            logger.info(f"   Records: {len(df_season):,}")
            logger.info(f"   Weeks: {weeks[0]}-{weeks[-1]}")
            logger.info(f"   Saved: {output_file}")
            logger.info("")

        # Also save combined historical file (all seasons)
        historical_file = output_dir / "weekly_historical.parquet"
        df_historical = df_all[df_all['season'].isin(seasons)].copy()
        df_historical.to_parquet(historical_file, index=False)

        logger.info(f"✅ Combined Historical File:")
        logger.info(f"   Records: {len(df_historical):,}")
        logger.info(f"   Seasons: {sorted(df_historical['season'].unique())}")
        logger.info(f"   Saved: {historical_file}")
        logger.info("")

        return df_historical

    except Exception as e:
        logger.error(f"❌ Failed to fetch weekly stats: {e}")
        return None


def fetch_pbp_data(seasons=[2024, 2025]):
    """
    Fetch play-by-play data for specified seasons.

    Each season has its own PBP file.
    """
    logger.info("=" * 80)
    logger.info("FETCHING PLAY-BY-PLAY DATA")
    logger.info("=" * 80)
    logger.info("")

    output_dir = Path("data/nflverse")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pbp = []

    for season in seasons:
        try:
            url = URLS['pbp'].format(season=season)
            logger.info(f"📥 Downloading {season} PBP data...")
            logger.info(f"   URL: {url}")

            df_pbp = pd.read_parquet(url)

            # Save season-specific file
            output_file = output_dir / f"pbp_{season}.parquet"
            df_pbp.to_parquet(output_file, index=False)

            weeks = sorted(df_pbp['week'].unique())
            games = df_pbp['game_id'].nunique()

            logger.info(f"✅ {season} Season:")
            logger.info(f"   Plays: {len(df_pbp):,}")
            logger.info(f"   Games: {games}")
            logger.info(f"   Weeks: {weeks[0]}-{weeks[-1]}")
            logger.info(f"   Saved: {output_file}")
            logger.info("")

            all_pbp.append(df_pbp)

        except Exception as e:
            logger.error(f"❌ Failed to fetch {season} PBP data: {e}")
            logger.info("")
            continue

    # Save combined historical PBP
    if all_pbp:
        historical_file = output_dir / "pbp_historical.parquet"
        df_combined = pd.concat(all_pbp, ignore_index=True)
        df_combined.to_parquet(historical_file, index=False)

        logger.info(f"✅ Combined PBP File:")
        logger.info(f"   Plays: {len(df_combined):,}")
        logger.info(f"   Seasons: {sorted(df_combined['season'].unique())}")
        logger.info(f"   Saved: {historical_file}")
        logger.info("")

        return df_combined

    return None


def fetch_games_schedule():
    """
    Fetch game schedule and results for all seasons.

    NFLverse provides a single file with all games.
    """
    logger.info("=" * 80)
    logger.info("FETCHING GAME SCHEDULE & RESULTS")
    logger.info("=" * 80)
    logger.info("")

    try:
        logger.info(f"📥 Downloading from: {URLS['games']}")

        df_games = pd.read_parquet(URLS['games'])

        logger.info(f"✅ Downloaded {len(df_games):,} games")
        logger.info(f"   Seasons: {sorted(df_games['season'].unique())}")
        logger.info("")

        # Save full games file
        output_dir = Path("data/nflverse")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "games.parquet"
        df_games.to_parquet(output_file, index=False)

        logger.info(f"   Saved: {output_file}")
        logger.info("")

        # Show recent seasons
        for season in [2024, 2025]:
            season_games = df_games[df_games['season'] == season]
            if len(season_games) > 0:
                completed = season_games[season_games['game_type'] == 'REG'].dropna(subset=['home_score'])
                logger.info(f"   {season}: {len(season_games)} games ({len(completed)} completed)")

        logger.info("")
        return df_games

    except Exception as e:
        logger.error(f"❌ Failed to fetch games: {e}")
        return None


def verify_data_quality():
    """
    Verify that downloaded data is complete and valid.
    """
    logger.info("=" * 80)
    logger.info("DATA QUALITY VERIFICATION")
    logger.info("=" * 80)
    logger.info("")

    checks_passed = 0
    checks_total = 0

    nflverse_dir = Path("data/nflverse")

    # Check 1: Weekly stats files exist
    for season in [2024, 2025]:
        checks_total += 1
        weekly_file = nflverse_dir / f"weekly_{season}.parquet"

        if weekly_file.exists():
            df = pd.read_parquet(weekly_file)
            if len(df) > 0 and 'week' in df.columns:
                weeks = sorted(df['week'].unique())
                logger.info(f"✅ CHECK: {season} weekly stats - {len(df):,} records, weeks {weeks[0]}-{weeks[-1]}")
                checks_passed += 1
            else:
                logger.error(f"❌ CHECK: {season} weekly stats - Empty or malformed")
        else:
            logger.error(f"❌ CHECK: {season} weekly stats - File not found")

    # Check 2: PBP files exist
    for season in [2024, 2025]:
        checks_total += 1
        pbp_file = nflverse_dir / f"pbp_{season}.parquet"

        if pbp_file.exists():
            df = pd.read_parquet(pbp_file)
            if len(df) > 0 and 'play_id' in df.columns:
                logger.info(f"✅ CHECK: {season} PBP data - {len(df):,} plays")
                checks_passed += 1
            else:
                logger.error(f"❌ CHECK: {season} PBP data - Empty or malformed")
        else:
            logger.warning(f"⚠️  CHECK: {season} PBP data - File not found (non-critical)")

    # Check 3: Games file exists
    checks_total += 1
    games_file = nflverse_dir / "games.parquet"

    if games_file.exists():
        df = pd.read_parquet(games_file)
        if len(df) > 0 and 'game_id' in df.columns:
            logger.info(f"✅ CHECK: Games schedule - {len(df):,} games")
            checks_passed += 1
        else:
            logger.error(f"❌ CHECK: Games schedule - Empty or malformed")
    else:
        logger.error(f"❌ CHECK: Games schedule - File not found")

    logger.info("")
    logger.info(f"Quality Checks: {checks_passed}/{checks_total} passed")
    logger.info("")

    if checks_passed == checks_total:
        logger.info("🎉 ALL CHECKS PASSED - Data is ready!")
    elif checks_passed >= 2:
        logger.info("⚠️  PARTIAL SUCCESS - Core data available, some optional files missing")
    else:
        logger.error("❌ CRITICAL FAILURES - Data may be incomplete")

    logger.info("")
    return checks_passed >= 2


def generate_summary():
    """
    Generate summary of fetched data.
    """
    logger.info("=" * 80)
    logger.info("DATA FETCH SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    logger.info("📁 Data Location: data/nflverse/")
    logger.info("")
    logger.info("Files Created:")

    nflverse_dir = Path("data/nflverse")

    files_to_check = [
        ("weekly_2024.parquet", "2024 Weekly Stats"),
        ("weekly_2025.parquet", "2025 Weekly Stats"),
        ("weekly_historical.parquet", "Combined Weekly Stats"),
        ("pbp_2024.parquet", "2024 Play-by-Play"),
        ("pbp_2025.parquet", "2025 Play-by-Play"),
        ("pbp_historical.parquet", "Combined Play-by-Play"),
        ("games.parquet", "Game Schedule"),
    ]

    for filename, description in files_to_check:
        filepath = nflverse_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {description:30s} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ❌ {description:30s} (missing)")

    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. ✅ NFLverse data fetched for 2024 and 2025")
    logger.info("2. 🎯 You can now:")
    logger.info("   - Run backtests on 2024 data")
    logger.info("   - Generate predictions for 2025 week 9")
    logger.info("   - Use enhanced NFLverse stats in simulations")
    logger.info("")
    logger.info("3. 🔄 Keep data fresh:")
    logger.info("   - Re-run this script weekly to get latest 2025 data")
    logger.info("   - NFLverse updates stats ~24 hours after games")
    logger.info("")


def main():
    """
    Main execution: Fetch NFLverse data for training seasons (previous + current).
    """
    # Get training seasons (e.g., [2024, 2025])
    training_seasons = get_training_seasons()
    current_season = get_current_season()

    logger.info("")
    logger.info("=" * 80)
    logger.info("🏈 NFL QUANT - NFLverse Data Fetch")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Current Season: {current_season}")
    logger.info(f"Training Seasons: {training_seasons}")
    logger.info("")
    logger.info("This script fetches comprehensive NFL data from NFLverse:")
    logger.info(f"  • {training_seasons[0]} season (historical - for training/backtesting)")
    logger.info(f"  • {training_seasons[-1]} season (current - for live predictions)")
    logger.info("")

    # Fetch all data
    weekly_data = fetch_weekly_stats(seasons=training_seasons)
    pbp_data = fetch_pbp_data(seasons=training_seasons)
    games_data = fetch_games_schedule()

    # Verify quality
    quality_ok = verify_data_quality()

    # Summary
    generate_summary()

    if quality_ok:
        logger.info("✅ NFLverse data fetch completed successfully!")
        return 0
    else:
        logger.error("❌ NFLverse data fetch completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
