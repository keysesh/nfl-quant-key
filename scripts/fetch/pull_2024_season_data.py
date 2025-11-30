#!/usr/bin/env python3
"""
Pull complete 2024 NFL season data for calibrator training.

Downloads:
- Play-by-play data (nflfastR)
- Player props and outcomes (nflverse + historical odds)
- Game results
- Historical lines (if available from Odds API)

This gives us ~16,000 props + 285 games to improve calibrator training.
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pull_2024_pbp_data():
    """Pull 2024 play-by-play data from nflfastR."""
    logger.info("📥 Pulling 2024 PBP data from nflfastR...")

    url = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet'

    try:
        pbp_2024 = pd.read_parquet(url)

        # Save locally
        output_path = Path('data/processed/pbp_2024.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pbp_2024.to_parquet(output_path)

        logger.info(f"✅ Downloaded {len(pbp_2024):,} plays from 2024 season")
        logger.info(f"   Saved to: {output_path}")

        # Summary stats
        games = pbp_2024['game_id'].nunique()
        weeks = pbp_2024['week'].nunique()
        logger.info(f"   Games: {games}, Weeks: {weeks}")

        return pbp_2024

    except Exception as e:
        logger.error(f"❌ Failed to download 2024 PBP data: {e}")
        return None


def pull_2024_player_stats():
    """Pull 2024 weekly player stats from nflverse."""
    logger.info("📥 Pulling 2024 player stats from nflverse...")

    try:
        # Pull weekly stats from nflverse
        url = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2024.parquet'
        stats_2024 = pd.read_parquet(url)

        # Save to file
        output_path = Path('data/historical/player_stats_2024.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats_2024.to_csv(output_path, index=False)

        logger.info(f"✅ Downloaded {len(stats_2024):,} player-week records from 2024")
        logger.info(f"   Saved to: {output_path}")

        # Summary stats
        weeks = stats_2024['week'].nunique() if 'week' in stats_2024.columns else 0
        players = stats_2024['player_id'].nunique() if 'player_id' in stats_2024.columns else len(stats_2024)
        logger.info(f"   Players: {players}, Weeks: {weeks}")

        return stats_2024

    except Exception as e:
        logger.error(f"❌ Failed to download 2024 player stats: {e}")
        return None


def pull_2024_games():
    """Pull 2024 game results and schedules."""
    logger.info("📥 Pulling 2024 game schedule and results...")

    url = 'https://github.com/nflverse/nflverse-data/releases/download/games/games.parquet'

    try:
        all_games = pd.read_parquet(url)
        games_2024 = all_games[all_games['season'] == 2024].copy()

        # Save
        output_path = Path('data/historical/games_2024.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        games_2024.to_csv(output_path, index=False)

        logger.info(f"✅ Downloaded {len(games_2024)} games from 2024 season")
        logger.info(f"   Saved to: {output_path}")

        # Summary
        completed = games_2024[games_2024['game_type'] == 'REG'].dropna(subset=['home_score'])
        logger.info(f"   Completed: {len(completed)} games")

        return games_2024

    except Exception as e:
        logger.error(f"❌ Failed to download 2024 games: {e}")
        return None


def check_odds_api_availability():
    """Check if we have Odds API key for historical lines."""
    config_path = Path('configs/odds_api_config.json')

    if not config_path.exists():
        logger.warning("⚠️  No Odds API config found")
        logger.info("   To get 2024 historical lines:")
        logger.info("   1. Sign up at https://the-odds-api.com/")
        logger.info("   2. Create configs/odds_api_config.json:")
        logger.info('      {"api_key": "your_key_here"}')
        logger.info("   3. Historical data costs ~$100-500 for full season")
        return None

    with open(config_path) as f:
        config = json.load(f)
        return config.get('api_key')


def pull_2024_historical_lines():
    """Pull 2024 historical betting lines (if API available)."""
    api_key = check_odds_api_availability()

    if not api_key:
        logger.info("⚠️  Skipping historical lines (no API key)")
        logger.info("   Can still train calibrators without this data")
        return None

    logger.info("📥 Pulling 2024 historical lines from Odds API...")

    # TODO: Implement historical line pulls
    # This requires:
    # 1. Odds API historical endpoint access
    # 2. Date ranges for each week
    # 3. Proper rate limiting

    logger.info("   Historical line pulls require manual implementation")
    logger.info("   See audit section 2 for CLV tracking infrastructure")

    return None


def create_2024_prop_training_dataset():
    """
    Create 2024 player prop training dataset by matching:
    - Player stats (actual outcomes)
    - Historical prop lines (if available)
    - Player predictions (need to generate)
    """
    logger.info("📊 Creating 2024 prop training dataset...")

    # Load what we just pulled
    stats_path = Path('data/historical/player_stats_2024.csv')

    if not stats_path.exists():
        logger.error("❌ Player stats not found - run pull_2024_player_stats() first")
        return None

    stats_2024 = pd.read_csv(stats_path)

    # We need to:
    # 1. Convert stats to prop outcomes (did player go over/under X yards?)
    # 2. Match with historical lines (if available)
    # 3. Or use typical lines based on player season averages

    logger.info("   Creating prop outcomes from actual stats...")

    props = []

    # For each player-week, create prop outcomes
    for _, row in stats_2024.iterrows():
        player_id = row['player_id']
        week = row['week']

        # Passing yards
        if pd.notna(row.get('pass_yd', np.nan)):
            props.append({
                'player_id': player_id,
                'week': week,
                'season': 2024,
                'market': 'player_pass_yds',
                'actual_value': row['pass_yd'],
                'position': 'QB'
            })

        # Rushing yards
        if pd.notna(row.get('rush_yd', np.nan)):
            props.append({
                'player_id': player_id,
                'week': week,
                'season': 2024,
                'market': 'player_rush_yds',
                'actual_value': row['rush_yd'],
                'position': 'RB'  # Simplified
            })

        # Receiving yards
        if pd.notna(row.get('rec_yd', np.nan)):
            props.append({
                'player_id': player_id,
                'week': week,
                'season': 2024,
                'market': 'player_rec_yds',
                'actual_value': row['rec_yd'],
                'position': 'WR'  # Simplified
            })

        # Receptions
        if pd.notna(row.get('rec', np.nan)):
            props.append({
                'player_id': player_id,
                'week': week,
                'season': 2024,
                'market': 'player_receptions',
                'actual_value': row['rec'],
                'position': 'WR'  # Simplified
            })

    props_df = pd.DataFrame(props)

    # Save raw prop outcomes
    output_path = Path('data/historical/prop_outcomes_2024.csv')
    props_df.to_csv(output_path, index=False)

    logger.info(f"✅ Created {len(props_df):,} prop outcomes from 2024 stats")
    logger.info(f"   Saved to: {output_path}")

    # Show breakdown
    logger.info("\n   Breakdown by market:")
    for market, count in props_df['market'].value_counts().items():
        logger.info(f"     {market}: {count:,}")

    return props_df


def generate_summary_report():
    """Generate summary of downloaded 2024 data."""
    logger.info("\n" + "="*80)
    logger.info("📋 2024 DATA PULL SUMMARY")
    logger.info("="*80)

    # Check what we have
    files = {
        'PBP Data': 'data/processed/pbp_2024.parquet',
        'Player Stats': 'data/historical/player_stats_2024.csv',
        'Games': 'data/historical/games_2024.csv',
        'Prop Outcomes': 'data/historical/prop_outcomes_2024.csv',
    }

    for name, path in files.items():
        if Path(path).exists():
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            logger.info(f"✅ {name}: {len(df):,} records")
        else:
            logger.info(f"❌ {name}: Not found")

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("="*80)
    logger.info("1. ✅ 2024 data downloaded")
    logger.info("2. 🔄 Run: python retrain_calibrator_with_2024.py")
    logger.info("3. 🔄 Compare old vs new calibrator performance")
    logger.info("4. 🔄 Deploy improved calibrator for Week 9+")
    logger.info("="*80)


def main():
    """Execute full 2024 data pull."""
    logger.info("🚀 Starting 2024 NFL Season Data Pull")
    logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Step 1: PBP Data
    pbp_2024 = pull_2024_pbp_data()

    # Step 2: Player Stats
    stats_2024 = pull_2024_player_stats()

    # Step 3: Game Results
    games_2024 = pull_2024_games()

    # Step 4: Historical Lines (optional)
    lines_2024 = pull_2024_historical_lines()

    # Step 5: Create prop training dataset
    if stats_2024 is not None:
        props_2024 = create_2024_prop_training_dataset()

    # Step 6: Summary
    generate_summary_report()

    logger.info("\n✅ 2024 data pull complete!")


if __name__ == "__main__":
    main()
