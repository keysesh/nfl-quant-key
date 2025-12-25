#!/usr/bin/env python3
"""
Match Edge Predictions to Actuals

Loads predictions from logs and matches them to actual game results.
Updates prediction logs with hit/miss status for ROI tracking.

Usage:
    python scripts/monitoring/match_edge_actuals.py
    python scripts/monitoring/match_edge_actuals.py --date 2024-12-15
"""
import argparse
from pathlib import Path
from datetime import datetime, date
import sys

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import DATA_DIR
from nfl_quant.monitoring.edge_logger import EdgePredictionLogger
from nfl_quant.utils.player_names import normalize_player_name


def load_actuals_from_nflverse(season: int, week: int) -> pd.DataFrame:
    """
    Load actual player stats from NFLverse data.

    Args:
        season: NFL season
        week: NFL week

    Returns:
        DataFrame with player, market, and actual values
    """
    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'

    if not stats_path.exists():
        print(f"Warning: {stats_path} not found")
        return pd.DataFrame()

    stats = pd.read_parquet(stats_path)

    # Filter to season/week
    stats = stats[(stats['season'] == season) & (stats['week'] == week)].copy()

    if stats.empty:
        print(f"No stats found for {season} Week {week}")
        return pd.DataFrame()

    # Normalize player names
    stats['player'] = stats['player_display_name'].apply(normalize_player_name)

    # Map to betting markets
    market_mapping = {
        'player_receptions': 'receptions',
        'player_rush_yds': 'rushing_yards',
        'player_reception_yds': 'receiving_yards',
        'player_rush_attempts': 'carries',
        'player_pass_yds': 'passing_yards',
    }

    # Create long-form actuals
    actuals_list = []
    for market, stat_col in market_mapping.items():
        if stat_col in stats.columns:
            market_df = stats[['player', stat_col]].copy()
            market_df['market'] = market
            market_df['actual'] = market_df[stat_col]
            market_df = market_df[['player', 'market', 'actual']]
            actuals_list.append(market_df)

    if not actuals_list:
        return pd.DataFrame()

    return pd.concat(actuals_list, ignore_index=True)


def match_predictions(
    prediction_date: date,
    season: int = None,
    week: int = None,
) -> pd.DataFrame:
    """
    Match predictions to actuals.

    Args:
        prediction_date: Date of predictions
        season: NFL season (auto-detected if None)
        week: NFL week (auto-detected if None)

    Returns:
        DataFrame with matched predictions
    """
    logger = EdgePredictionLogger()

    # Load predictions
    predictions = logger.load_predictions(prediction_date=prediction_date)

    if predictions.empty:
        print(f"No predictions found for {prediction_date}")
        return pd.DataFrame()

    print(f"Found {len(predictions)} predictions for {prediction_date}")

    # Auto-detect season/week from predictions if not provided
    if season is None:
        season = predictions.get('season', pd.Series()).mode()
        season = int(season.iloc[0]) if len(season) > 0 else datetime.now().year
    if week is None:
        week = predictions.get('week', pd.Series()).mode()
        week = int(week.iloc[0]) if len(week) > 0 else 15

    print(f"Loading actuals for {season} Week {week}...")

    # Load actuals
    actuals = load_actuals_from_nflverse(season, week)

    if actuals.empty:
        print("No actuals found - cannot match predictions")
        return predictions

    print(f"Found {len(actuals)} actual stat lines")

    # Match
    matched = logger.match_actuals(actuals, prediction_date=prediction_date)

    # Calculate results
    valid = matched[matched['hit'].notna()]
    if len(valid) > 0:
        hit_rate = valid['hit'].mean()
        roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
        print(f"\nResults: {len(valid)} matched, {valid['hit'].sum()} hits, "
              f"{hit_rate:.1%} hit rate, {roi:+.1f}% ROI")

    return matched


def main():
    parser = argparse.ArgumentParser(description="Match Edge Predictions to Actuals")
    parser.add_argument('--date', type=str, help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('--season', type=int, help='NFL season')
    parser.add_argument('--week', type=int, help='NFL week')
    parser.add_argument('--save', action='store_true', help='Save matched results')
    args = parser.parse_args()

    # Parse date
    if args.date:
        prediction_date = date.fromisoformat(args.date)
    else:
        # Default to yesterday (games would have finished)
        prediction_date = date.today()

    print(f"Matching predictions for {prediction_date}")

    # Match
    matched = match_predictions(
        prediction_date=prediction_date,
        season=args.season,
        week=args.week,
    )

    if matched.empty:
        return

    # Save if requested
    if args.save:
        logger = EdgePredictionLogger()
        filepath = logger.save_matched_results(matched, prediction_date=prediction_date)
        print(f"\nResults saved to: {filepath}")

    # Print summary by source
    if 'hit' in matched.columns:
        valid = matched[matched['hit'].notna()]
        if len(valid) > 0:
            print("\nBy Source:")
            for source in valid['source'].unique():
                source_df = valid[valid['source'] == source]
                hit_rate = source_df['hit'].mean()
                roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
                print(f"  {source}: {len(source_df)} bets, {hit_rate:.1%} hit rate, {roi:+.1f}% ROI")


if __name__ == '__main__':
    main()
