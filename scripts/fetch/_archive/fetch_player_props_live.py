#!/usr/bin/env python3
"""
Fetch Live DraftKings Player Props and Find Edges Against Model Predictions

This script uses the-odds-api.com to fetch live player props from DraftKings
and compares them against NFL QUANT model predictions to find betting edges.

Usage:
    python scripts/fetch/fetch_player_props_live.py --week 12
    python scripts/fetch/fetch_player_props_live.py --week 12 --save-odds
    python scripts/fetch/fetch_player_props_live.py --capture-closing  # Run 30 mins before games

API Key: 73ec9367021badb173a0b68c35af818f
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.draftkings_client import (
    DKClient, find_edges, save_current_odds, CORE_MARKETS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_model_predictions(week: int) -> pd.DataFrame:
    """
    Load model predictions and convert to format needed by find_edges().

    Returns DataFrame with: player_name, stat_type, predicted_value, predicted_std
    """
    pred_file = PROJECT_ROOT / f'data/model_predictions_week{week}.csv'

    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")

    preds = pd.read_csv(pred_file)
    logger.info(f"Loaded {len(preds)} player predictions from week {week}")

    # Convert to format needed by find_edges
    rows = []

    # Map of stat types to column prefixes
    stat_mappings = [
        ('pass_yds', 'passing_yards'),
        ('rush_yds', 'rushing_yards'),
        ('rec_yds', 'receiving_yards'),
        ('receptions', 'receptions'),
        ('rush_attempts', 'rushing_attempts'),
        ('targets', 'targets'),
    ]

    for _, player in preds.iterrows():
        player_name = player['player_name']

        for stat_type, col_prefix in stat_mappings:
            mean_col = f'{col_prefix}_mean'
            std_col = f'{col_prefix}_std'

            if mean_col in player.index and std_col in player.index:
                mean_val = player[mean_col]
                std_val = player[std_col]

                # Skip if no meaningful value
                if pd.isna(mean_val) or mean_val <= 0:
                    continue
                if pd.isna(std_val) or std_val <= 0:
                    std_val = mean_val * 0.5  # Default 50% CV if missing

                rows.append({
                    'player_name': player_name,
                    'stat_type': stat_type,
                    'predicted_value': mean_val,
                    'predicted_std': std_val,
                    'team': player.get('team', ''),
                    'position': player.get('position', ''),
                })

    df = pd.DataFrame(rows)
    logger.info(f"Converted to {len(df)} stat predictions")

    return df


def display_edges(edges: pd.DataFrame):
    """Pretty print the edges found."""
    if edges.empty:
        print("\n⚠️  No edges found meeting criteria")
        return

    print("\n" + "=" * 80)
    print("🎯 EDGES FOUND AGAINST DRAFTKINGS PLAYER PROPS")
    print("=" * 80)

    # Group by confidence
    elite = edges[edges['edge'] >= 0.10]
    high = edges[(edges['edge'] >= 0.05) & (edges['edge'] < 0.10)]
    standard = edges[edges['edge'] < 0.05]

    if not elite.empty:
        print("\n🔥 ELITE EDGES (10%+):")
        for _, row in elite.head(10).iterrows():
            market_name = row['market'].replace('player_', '').replace('_', ' ').title()
            print(f"  {row['player']} {row['direction']} {row['line']} {market_name}")
            print(f"    Edge: {row['edge']*100:.1f}% | EV: {row['ev']*100:.1f}% | Kelly: {row['kelly']*100:.2f}%")
            print(f"    Model: {row['model_prob']*100:.1f}% vs DK: {row['dk_prob']*100:.1f}%")
            print(f"    {row['matchup']}")
            print()

    if not high.empty:
        print("\n✅ HIGH EDGES (5-10%):")
        for _, row in high.head(10).iterrows():
            market_name = row['market'].replace('player_', '').replace('_', ' ').title()
            print(f"  {row['player']} {row['direction']} {row['line']} {market_name}")
            print(f"    Edge: {row['edge']*100:.1f}% | EV: {row['ev']*100:.1f}% | Kelly: {row['kelly']*100:.2f}%")
            print()

    if not standard.empty:
        print(f"\n📊 STANDARD EDGES (3-5%): {len(standard)} found")
        for _, row in standard.head(5).iterrows():
            market_name = row['market'].replace('player_', '').replace('_', ' ').title()
            print(f"  {row['player']} {row['direction']} {row['line']} {market_name} (edge: {row['edge']*100:.1f}%)")

    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print(f"  Total edges: {len(edges)}")
    print(f"  Elite (10%+): {len(elite)}")
    print(f"  High (5-10%): {len(high)}")
    print(f"  Standard (3-5%): {len(standard)}")
    if not edges.empty:
        print(f"  Best EV: {edges['ev'].max()*100:.1f}%")
        print(f"  Total Kelly allocation: {edges['kelly'].sum()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Fetch DraftKings player props and find edges')
    parser.add_argument('--week', type=int, help='NFL week number')
    parser.add_argument('--min-edge', type=float, default=0.03, help='Minimum edge threshold (default: 0.03)')
    parser.add_argument('--min-ev', type=float, default=0.02, help='Minimum EV threshold (default: 0.02)')
    parser.add_argument('--save-odds', action='store_true', help='Save fetched odds to CSV')
    parser.add_argument('--capture-closing', action='store_true', help='Capture closing lines only')
    parser.add_argument('--output', type=str, help='Output CSV file for edges')
    parser.add_argument('--test', action='store_true', help='Test API connection only')

    args = parser.parse_args()

    # Test mode - just check API connectivity
    if args.test:
        print("🧪 Testing DraftKings API connection...")
        client = DKClient()
        events = client.get_events()
        print(f"✅ Found {len(events)} upcoming NFL games")
        for e in events[:5]:
            print(f"  {e['away_team']} @ {e['home_team']} - {e['commence_time']}")
        print(f"\n📊 API quota remaining: {client.remaining}")
        return

    # Capture closing lines mode
    if args.capture_closing:
        print("📸 Capturing closing lines...")
        output_dir = PROJECT_ROOT / 'data' / 'closing_lines'
        filepath = save_current_odds(output_dir)
        if filepath:
            print(f"✅ Saved to {filepath}")
        return

    # Need week for edge finding
    if not args.week:
        # Auto-detect from current date
        try:
            from nfl_quant.utils.season_utils import get_current_week
            args.week = get_current_week()
        except:
            args.week = 12  # Default
        logger.info(f"Auto-detected week: {args.week}")

    print(f"\n🏈 NFL QUANT - Live Player Props Edge Finder")
    print(f"   Week {args.week} | Min Edge: {args.min_edge*100:.0f}% | Min EV: {args.min_ev*100:.0f}%")
    print()

    # Load model predictions
    try:
        model_preds = load_model_predictions(args.week)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Run: python scripts/predict/generate_model_predictions.py {args.week}")
        return

    # Optionally save raw odds first
    if args.save_odds:
        print("💾 Saving current odds...")
        save_current_odds()

    # Find edges
    print("🔍 Finding edges against DraftKings player props...")
    edges = find_edges(
        model_preds,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        variance_inflation=1.25  # Fix for variance underestimation
    )

    # Display results
    display_edges(edges)

    # Save to file if requested
    if args.output and not edges.empty:
        output_path = Path(args.output)
        edges.to_csv(output_path, index=False)
        print(f"\n💾 Saved {len(edges)} edges to {output_path}")

    # Also save to standard location
    if not edges.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        standard_output = PROJECT_ROOT / 'reports' / f'live_prop_edges_week{args.week}_{timestamp}.csv'
        edges.to_csv(standard_output, index=False)
        print(f"💾 Auto-saved to {standard_output}")


if __name__ == '__main__':
    main()
