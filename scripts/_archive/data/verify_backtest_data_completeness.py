#!/usr/bin/env python3
"""
Verify we have complete data for backtesting weeks 1-8 of 2024

Checks:
1. Do we have sleeper stats for all weeks?
2. Do we have prop lines for all weeks?
3. Can we match props to stats?
4. What percentage of props have matching stats?
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season

def normalize_player_name(name: str) -> str:
    """Normalize player name for matching"""
    return name.lower().replace(".", "").replace("'", "").replace("-", "").replace(" ", "")

def main():
    print("=" * 80)
    print("BACKTEST DATA COMPLETENESS CHECK")
    print("=" * 80)
    print()

    season = get_current_season()
    weeks = range(1, 9)  # Weeks 1-8

    # Check sleeper stats
    print("1. SLEEPER STATS FILES")
    print("=" * 80)
    missing_stats = []
    for week in weeks:
        stats_path = Path(f"data/sleeper_stats/stats_week{week}_{season}.csv")
        if stats_path.exists():
            df = pd.read_csv(stats_path)
            print(f"✅ Week {week}: {len(df)} players")
        else:
            print(f"❌ Week {week}: MISSING")
            missing_stats.append(week)

    print()
    if missing_stats:
        print(f"⚠️  Missing stats for weeks: {missing_stats}")
    else:
        print("✅ All stats files present")
    print()

    # Check prop files
    print("2. PROP FILES")
    print("=" * 80)

    prop_files = sorted(Path("data/historical/backfill").glob("player_props_history_*.csv"))
    print(f"Found {len(prop_files)} prop files:")
    for f in prop_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print()

    # Load all props
    print("3. LOADING PROPS")
    print("=" * 80)

    frames = []
    for f in prop_files:
        df = pd.read_csv(f)
        frames.append(df)

    props = pd.concat(frames, ignore_index=True)
    print(f"Total props loaded: {len(props):,}")
    print()

    # Load schedule to map games to weeks
    import json
    schedule_path = Path("data/raw/sleeper_games.json")
    if not schedule_path.exists():
        print("❌ Missing sleeper_games.json - cannot map props to weeks")
        return

    with open(schedule_path) as f:
        schedule = json.load(f)

    schedule_map = {}
    for game_id, game_info in schedule.items():
        if game_info.get("season") == season:
            schedule_map[game_id] = game_info.get("week")

    props["week"] = props["game_id"].map(schedule_map)
    props = props[props["week"].between(1, 8)]

    print(f"Props for weeks 1-8: {len(props):,}")
    print()

    # Check prop coverage by week
    print("4. PROPS BY WEEK")
    print("=" * 80)
    for week in weeks:
        week_props = props[props["week"] == week]
        print(f"Week {week}: {len(week_props):,} props")
    print()

    # Check matching between props and stats
    print("5. PROP-TO-STATS MATCHING")
    print("=" * 80)

    total_matched = 0
    total_unmatched = 0

    for week in weeks:
        stats_path = Path(f"data/sleeper_stats/stats_week{week}_{season}.csv")
        if not stats_path.exists():
            continue

        stats_df = pd.read_csv(stats_path)
        stats_df["player_key"] = stats_df["player_name"].apply(normalize_player_name)
        stats_df["team"] = stats_df["team"].str.upper()
        stats_players = set(zip(stats_df["player_key"], stats_df["team"]))

        week_props = props[props["week"] == week].copy()
        week_props["player_key"] = week_props["description"].apply(normalize_player_name)

        # Try to extract team from prop data
        # Props should have a team column or we need to infer it
        if "team" not in week_props.columns:
            # Try to get team from game_id or other fields
            week_props["team"] = None  # We'll need to fix this

        matched = 0
        unmatched = 0
        unmatched_players = []

        for _, prop in week_props.iterrows():
            player_key = prop["player_key"]

            # Try to find player in stats (any team)
            player_in_stats = any(player_key == pk for pk, _ in stats_players)

            if player_in_stats:
                matched += 1
            else:
                unmatched += 1
                if len(unmatched_players) < 10:
                    unmatched_players.append(prop["description"])

        total_matched += matched
        total_unmatched += unmatched

        match_rate = (matched / (matched + unmatched) * 100) if (matched + unmatched) > 0 else 0
        print(f"Week {week}: {matched}/{matched + unmatched} matched ({match_rate:.1f}%)")

        if unmatched_players:
            print(f"  Sample unmatched: {', '.join(unmatched_players[:3])}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = total_matched + total_unmatched
    match_rate = (total_matched / total * 100) if total > 0 else 0
    print(f"Total props evaluated: {total:,}")
    print(f"Matched to stats: {total_matched:,} ({match_rate:.1f}%)")
    print(f"Unmatched: {total_unmatched:,} ({100-match_rate:.1f}%)")
    print()

    if match_rate < 90:
        print("⚠️  LOW MATCH RATE!")
        print()
        print("Possible issues:")
        print("  1. Player name mismatches (e.g. 'A.J. Brown' vs 'AJ Brown')")
        print("  2. Props for players who didn't play that week")
        print("  3. Props from unsupported markets")
        print("  4. Team information missing from props")
        print()
        print("Recommendation: Check prop format and improve player name normalization")
    else:
        print("✅ Good match rate - data should be sufficient for backtesting")

if __name__ == "__main__":
    main()
