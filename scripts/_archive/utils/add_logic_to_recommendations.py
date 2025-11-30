#!/usr/bin/env python3
"""
Add logic/narrative column to existing recommendations CSV.

This script adds a human-readable explanation for each pick based on
the model's calculations, historical data, and opponent analysis.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.utils.pick_narrative_generator import (
    generate_narrative_with_historical_data,
    generate_short_logic
)


def add_logic_column(
    recommendations_path: str = "reports/CURRENT_WEEK_RECOMMENDATIONS.csv",
    output_path: str = None,
    narrative_style: str = "full"
):
    """
    Add logic column to recommendations CSV.

    Args:
        recommendations_path: Path to recommendations CSV
        output_path: Where to save (None = overwrite original)
        narrative_style: 'full' (detailed) or 'short' (concise)
    """

    print(f"Loading recommendations from {recommendations_path}...")
    recommendations = pd.read_csv(recommendations_path)

    print(f"Found {len(recommendations)} recommendations")

    # Load historical data for context
    try:
        weekly_stats = pd.read_parquet('data/nflverse/weekly_stats.parquet')
        print(f"Loaded {len(weekly_stats)} weekly stat records")
        use_historical = True
    except FileNotFoundError:
        print("⚠️  weekly_stats.parquet not found, using simplified narratives")
        weekly_stats = None
        use_historical = False

    # Generate narratives
    print(f"\nGenerating {narrative_style} narratives for all picks...")

    narratives = []

    for idx, row in recommendations.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(recommendations)}...")

        if narrative_style == 'short':
            narrative = generate_short_logic(row)
        else:
            if use_historical:
                narrative = generate_narrative_with_historical_data(
                    row,
                    weekly_stats
                )
            else:
                # Use basic narrative without historical data
                from nfl_quant.utils.pick_narrative_generator import generate_pick_narrative
                narrative = generate_pick_narrative(row)

        narratives.append(narrative)

    # Add to DataFrame
    recommendations['logic'] = narratives

    # Save
    if output_path is None:
        output_path = recommendations_path

    recommendations.to_csv(output_path, index=False)

    print(f"\n✅ Successfully added logic column to {len(recommendations)} picks")
    print(f"   Saved to: {output_path}")

    # Show examples
    print(f"\n{'=' * 100}")
    print("SAMPLE NARRATIVES (Top 5 picks by edge)")
    print('=' * 100)

    top_5 = recommendations.nlargest(5, 'edge_pct')

    for idx, (_, pick) in enumerate(top_5.iterrows(), 1):
        print(f"\n[{idx}] {pick['player']} - {pick['pick']} {pick['market']} {pick['line']}")
        print(f"    Edge: {pick['edge_pct']:.1f}% | Conf: {pick['confidence']}")
        print(f"    Logic: {pick['logic']}")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add logic narratives to recommendations")
    parser.add_argument(
        '--input',
        default='reports/CURRENT_WEEK_RECOMMENDATIONS.csv',
        help='Path to recommendations CSV'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output path (default: overwrite input)'
    )
    parser.add_argument(
        '--style',
        choices=['full', 'short'],
        default='full',
        help='Narrative style: full (detailed) or short (concise)'
    )

    args = parser.parse_args()

    add_logic_column(
        recommendations_path=args.input,
        output_path=args.output,
        narrative_style=args.style
    )
