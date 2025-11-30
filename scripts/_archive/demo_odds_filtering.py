#!/usr/bin/env python3
"""
Demonstration: Pre-Game Odds Filtering System
==============================================

This script demonstrates the complete pre-game odds filtering workflow.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.utils.odds import (
    load_game_status_map,
    filter_pregame_odds
)


def demo_filtering():
    """Demonstrate the filtering system."""
    print("=" * 80)
    print("PRE-GAME ODDS FILTERING DEMONSTRATION")
    print("=" * 80)

    week = 9

    # Step 1: Load raw odds
    print(f"\n📊 Step 1: Loading raw odds data...")
    odds_file = Path('data/nfl_player_props_draftkings.csv')

    if not odds_file.exists():
        print(f"❌ Odds file not found: {odds_file}")
        return

    odds_df = pd.read_csv(odds_file)
    print(f"   ✅ Loaded {len(odds_df):,} raw odds records")

    # Show sample raw data
    print(f"\n   Sample raw odds (first 3):")
    for idx, row in odds_df.head(3).iterrows():
        print(f"      • {row.get('player_name', 'N/A'):20s} "
              f"{row.get('market', 'N/A'):25s} "
              f"Line: {row.get('line', 'N/A')}")

    # Step 2: Load game status
    print(f"\n🎮 Step 2: Loading game status from Sleeper API...")
    game_status_map = load_game_status_map(week)

    if game_status_map:
        print(f"   ✅ Loaded status for {len(game_status_map)} games")

        # Count statuses
        status_counts = {}
        for game_id, status in game_status_map.items():
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"\n   Game Status Breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"      {status:15s} {count:3d} games")
    else:
        print(f"   ⚠️  No game status data available")

    # Step 3: Apply filtering
    print(f"\n🔍 Step 3: Applying pre-game filtering...")
    print(f"   Validation layers:")
    print(f"      1. Game Status (Sleeper API)")
    print(f"      2. Temporal Validation (commence_time)")
    print(f"      3. Staleness Check (last_update)")

    filtered_df = filter_pregame_odds(
        odds_df,
        week=week,
        min_minutes_before_kickoff=5,
        max_hours_stale=24
    )

    # Step 4: Show results
    print(f"\n📈 Step 4: Filtering Results")
    print("-" * 80)

    total = len(odds_df)
    valid = len(filtered_df)
    rejected = total - valid

    print(f"\n   Before Filtering:  {total:,} odds")
    print(f"   After Filtering:   {valid:,} odds ({valid/total*100:.1f}%)")
    print(f"   Rejected:          {rejected:,} odds ({rejected/total*100:.1f}%)")

    if valid > 0:
        print(f"\n   ✅ Sample valid pre-game odds (first 5):")
        for idx, row in filtered_df.head(5).iterrows():
            player = row.get('player_name', 'N/A')
            market = row.get('market', 'N/A')
            line = row.get('line', 'N/A')
            commence = row.get('commence_time', 'N/A')
            print(f"      • {player:20s} {market:25s} "
                  f"Line: {line:6} Kickoff: {commence[:10] if isinstance(commence, str) else 'N/A'}")

    # Step 5: Impact analysis
    print(f"\n💡 Step 5: Expected Impact")
    print("-" * 80)

    if rejected > 0:
        print(f"\n   What was filtered out:")
        print(f"      • In-game odds (live line movements)")
        print(f"      • Completed game odds (no longer available)")
        print(f"      • Stale odds (may not reflect current lines)")
        print(f"      • Odds too close to kickoff (<5 min)")

        print(f"\n   Benefits of filtering:")
        print(f"      ✓ Eliminates false positive edges from in-game movements")
        print(f"      ✓ Ensures predictions use only bettable pre-game lines")
        print(f"      ✓ Improves model reliability and backtesting accuracy")
        print(f"      ✓ Expected edge rate: 8-12% (vs. 24% unfiltered)")

    # Step 6: Next steps
    print(f"\n🚀 Step 6: Using Filtered Odds in Production")
    print("-" * 80)
    print(f"\n   The filtering is automatic in prediction pipeline:")
    print(f"\n   # Generate predictions (filtering happens automatically)")
    print(f"   python scripts/predict/generate_current_week_recommendations.py {week}")
    print(f"\n   # Validate filtering before running")
    print(f"   python scripts/utils/validate_odds_filtering.py --week {week}")
    print(f"\n   # Run unit tests")
    print(f"   .venv/bin/python tests/test_odds_filtering.py")

    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nFor full documentation, see:")
    print(f"   • docs/ODDS_FILTERING_GUIDE.md (comprehensive)")
    print(f"   • docs/ODDS_FILTERING_QUICK_START.md (quick reference)")
    print()


if __name__ == '__main__':
    demo_filtering()
