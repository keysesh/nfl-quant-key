#!/usr/bin/env python3
"""
Build matchup and QB connection databases from play-by-play data.

This script extracts and caches contextual data for use in predictions:
- Individual matchups (WR vs CB, RB vs LB)
- QB-WR connections (target shares with specific QBs)
- Team-level matchup history

Usage:
    python scripts/data/build_matchup_databases.py [season] [week]

Example:
    python scripts/data/build_matchup_databases.py 2025 9
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.matchup_extractor import MatchupExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Build matchup and connection databases."""
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    current_week = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    print("=" * 80)
    print(f"BUILDING MATCHUP & QB CONNECTION DATABASES")
    print("=" * 80)
    print(f"Season: {season}")
    print(f"Current Week: {current_week}")

    # Load PBP data
    pbp_path = Path(f'data/processed/pbp_{season}.parquet')
    if not pbp_path.exists():
        print(f"\n❌ PBP file not found: {pbp_path}")
        print("   Run data collection first to generate PBP data")
        return

    print(f"\n📊 Loading play-by-play data...")
    pbp_df = pd.read_parquet(pbp_path)
    print(f"   ✅ Loaded {len(pbp_df):,} plays")

    # Create extractor
    extractor = MatchupExtractor(pbp_df)

    # Extract QB-WR connections
    print(f"\n🔗 Extracting QB-WR connections...")
    qb_connections_df = extractor.extract_qb_wr_connections(season)

    if len(qb_connections_df) > 0:
        output_path = Path(f'data/connections/qb_wr_connections_{season}.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qb_connections_df.to_parquet(output_path, index=False)
        print(f"   ✅ Saved {len(qb_connections_df):,} QB-WR connection records to {output_path}")
    else:
        print(f"   ⚠️  No QB-WR connections extracted")

    # Extract team matchups
    print(f"\n🏈 Extracting team matchups...")
    matchup_df = extractor.extract_team_matchups(season)

    if len(matchup_df) > 0:
        output_path = Path(f'data/matchups/team_matchups_{season}.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matchup_df.to_parquet(output_path, index=False)
        print(f"   ✅ Saved {len(matchup_df):,} matchup records to {output_path}")
    else:
        print(f"   ⚠️  No matchup records extracted")

    # Summary
    print("\n" + "=" * 80)
    print("✅ DATABASE BUILD COMPLETE")
    print("=" * 80)
    print(f"   QB-WR Connections: {len(qb_connections_df):,} records")
    print(f"   Team Matchups: {len(matchup_df):,} records")
    print("\n💡 These databases are now available for use in predictions!")


if __name__ == '__main__':
    main()































