"""
Analyze historical prop coverage to identify missing weeks
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict
import glob

def analyze_coverage():
    """Analyze what weeks we have historical props for"""

    backfill_dir = Path("data/historical/backfill")

    # Find all historical prop files
    prop_files = list(backfill_dir.glob("player_props_history_*.csv"))

    if not prop_files:
        print("❌ No historical prop files found")
        return

    print(f"Found {len(prop_files)} historical prop files\n")

    # Collect all props
    all_props = []
    for file in prop_files:
        try:
            df = pd.read_csv(file)
            all_props.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file.name}: {e}")

    if not all_props:
        print("❌ No props could be loaded")
        return

    # Combine all props
    props_df = pd.concat(all_props, ignore_index=True)

    print(f"Total props loaded: {len(props_df)}\n")

    # Check for required columns
    if 'season' not in props_df.columns or 'week' not in props_df.columns:
        # Try to extract from game_id or other fields
        if 'game_id' in props_df.columns:
            # game_id format might be like '2023_01_...'
            props_df['season'] = props_df['game_id'].str[:4].astype(int)
            props_df['week'] = props_df['game_id'].str[5:7].astype(int)
        elif 'commence_time' in props_df.columns:
            # Extract year from commence_time
            props_df['commence_time'] = pd.to_datetime(props_df['commence_time'])
            props_df['season'] = props_df['commence_time'].dt.year
            # Estimate week from date (NFL season starts ~Sept 7)
            props_df['week'] = ((props_df['commence_time'].dt.dayofyear - 250) // 7 + 1).clip(1, 18)

    # Group by season and week
    coverage = props_df.groupby(['season', 'week']).size().reset_index(name='num_props')
    coverage = coverage.sort_values(['season', 'week'])

    print("=" * 60)
    print("COVERAGE BY SEASON AND WEEK")
    print("=" * 60)

    for season in sorted(coverage['season'].unique()):
        season_data = coverage[coverage['season'] == season]
        print(f"\n{season} Season:")
        print(f"  Weeks covered: {sorted(season_data['week'].tolist())}")
        print(f"  Total props: {season_data['num_props'].sum()}")
        print(f"  Props per week: {season_data['num_props'].mean():.0f} (avg)")

        # Show week by week
        for _, row in season_data.iterrows():
            print(f"    Week {row['week']:2d}: {row['num_props']:4d} props")

    print("\n" + "=" * 60)
    print("IDENTIFYING MISSING WEEKS")
    print("=" * 60)

    # Now check what we should have based on NFLverse data
    nflverse_path = Path("data/nflverse/weekly_historical.parquet")
    if nflverse_path.exists():
        nflverse = pd.read_parquet(nflverse_path)
        available_weeks = nflverse.groupby(['season', 'week']).size().reset_index(name='num_players')

        print("\nNFLverse weeks available:")
        for season in sorted(available_weeks['season'].unique()):
            season_weeks = available_weeks[available_weeks['season'] == season]['week'].tolist()
            print(f"  {season}: weeks {min(season_weeks)}-{max(season_weeks)} ({len(season_weeks)} weeks)")

        # Find missing weeks
        print("\n" + "=" * 60)
        print("MISSING WEEKS (we have NFLverse but no props)")
        print("=" * 60)

        missing_weeks = []
        for _, row in available_weeks.iterrows():
            season, week = int(row['season']), int(row['week'])
            has_props = ((coverage['season'] == season) & (coverage['week'] == week)).any()
            if not has_props:
                missing_weeks.append((season, week))
                print(f"  {season} Week {week}")

        if not missing_weeks:
            print("  ✅ No missing weeks! We have props for all NFLverse data.")
        else:
            print(f"\n  Total missing: {len(missing_weeks)} weeks")

            # Group by season
            missing_by_season = defaultdict(list)
            for season, week in missing_weeks:
                missing_by_season[season].append(week)

            print("\n  Missing weeks by season:")
            for season in sorted(missing_by_season.keys()):
                weeks = sorted(missing_by_season[season])
                print(f"    {season}: {weeks}")
    else:
        print("\n⚠️ NFLverse data not found, cannot identify missing weeks")

    # Market coverage
    if 'market' in props_df.columns:
        print("\n" + "=" * 60)
        print("MARKET COVERAGE")
        print("=" * 60)
        market_counts = props_df['market'].value_counts()
        for market, count in market_counts.items():
            print(f"  {market}: {count:,} props")

if __name__ == "__main__":
    analyze_coverage()
