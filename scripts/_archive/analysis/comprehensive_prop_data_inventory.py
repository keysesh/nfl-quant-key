"""
Comprehensive inventory of ALL historical prop data across the project
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_all_prop_sources():
    """Analyze all sources of historical prop data"""

    print("=" * 80)
    print("COMPREHENSIVE HISTORICAL PROP DATA INVENTORY")
    print("=" * 80)

    all_props = []
    sources = []

    # Source 1: Backfill directory
    backfill_dir = Path("data/historical/backfill")
    if backfill_dir.exists():
        backfill_files = list(backfill_dir.glob("player_props_history_*.csv"))
        print(f"\n📁 Source 1: Backfill Directory ({len(backfill_files)} files)")

        for file in backfill_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = file.name
                df['source_type'] = 'backfill'
                all_props.append(df)
                sources.append(('backfill', file.name, len(df)))
            except Exception as e:
                print(f"  ⚠️ Error reading {file.name}: {e}")

        if all_props:
            print(f"  ✅ Loaded {sum(s[2] for s in sources if s[0] == 'backfill'):,} props from {len(sources)} files")

    # Source 2: Live archive
    archive_path = Path("data/historical/live_archive/player_props_archive.csv")
    if archive_path.exists():
        print(f"\n📁 Source 2: Live Archive")
        try:
            archive_df = pd.read_csv(archive_path)
            archive_df['source_type'] = 'live_archive'
            all_props.append(archive_df)
            sources.append(('live_archive', 'player_props_archive.csv', len(archive_df)))
            print(f"  ✅ Loaded {len(archive_df):,} props")
        except Exception as e:
            print(f"  ⚠️ Error reading archive: {e}")

    # Source 3: Any other historical directories
    hist_dir = Path("data/historical")
    if hist_dir.exists():
        other_csvs = []
        for csv_file in hist_dir.rglob("*.csv"):
            if 'backfill' not in str(csv_file) and 'archive' not in str(csv_file):
                other_csvs.append(csv_file)

        if other_csvs:
            print(f"\n📁 Source 3: Other Historical Files ({len(other_csvs)} files)")
            for csv_file in other_csvs:
                try:
                    df = pd.read_csv(csv_file)
                    if 'player' in df.columns or 'player_name' in df.columns:
                        df['source_file'] = str(csv_file.relative_to(Path.cwd()))
                        df['source_type'] = 'other'
                        all_props.append(df)
                        sources.append(('other', str(csv_file.relative_to(Path.cwd())), len(df)))
                        print(f"  ✅ {csv_file.name}: {len(df):,} props")
                except Exception as e:
                    pass

    # Source 4: Calibration directory
    cal_dir = Path("data/calibration")
    if cal_dir.exists():
        cal_files = list(cal_dir.glob("*.csv")) + list(cal_dir.glob("*.parquet"))
        if cal_files:
            print(f"\n📁 Source 4: Calibration Files ({len(cal_files)} files)")
            for cal_file in cal_files:
                try:
                    if cal_file.suffix == '.parquet':
                        df = pd.read_parquet(cal_file)
                    else:
                        df = pd.read_csv(cal_file)

                    if 'player' in df.columns or 'player_name' in df.columns:
                        df['source_file'] = cal_file.name
                        df['source_type'] = 'calibration'
                        all_props.append(df)
                        sources.append(('calibration', cal_file.name, len(df)))
                        print(f"  ✅ {cal_file.name}: {len(df):,} props")
                except Exception as e:
                    pass

    if not all_props:
        print("\n❌ No prop data found across any sources!")
        return

    # Combine all data
    print("\n" + "=" * 80)
    print("COMBINING ALL SOURCES")
    print("=" * 80)

    combined = pd.concat(all_props, ignore_index=True)
    print(f"\nTotal props before deduplication: {len(combined):,}")

    # Standardize column names
    if 'player' not in combined.columns and 'player_name' in combined.columns:
        combined['player'] = combined['player_name']

    # Try to extract season/week info
    print("\n" + "=" * 80)
    print("EXTRACTING TEMPORAL INFO")
    print("=" * 80)

    if 'season' not in combined.columns or 'week' not in combined.columns:
        if 'commence_time' in combined.columns:
            print("  Extracting season/week from commence_time...")
            combined['commence_time'] = pd.to_datetime(combined['commence_time'])
            combined['season'] = combined['commence_time'].dt.year

            # NFL season logic: Sept-Dec = current year, Jan-Feb = previous year
            combined['season'] = combined.apply(
                lambda row: row['season'] if row['commence_time'].month >= 9
                else row['season'] - 1,
                axis=1
            )

            # Calculate week number (approx)
            # Week 1 typically starts first Thu after Labor Day (~Sept 5-11)
            def estimate_week(dt):
                year = dt.year if dt.month >= 9 else dt.year
                # Estimate: Sept 7 = Week 1
                season_start = datetime(year, 9, 7)
                days_diff = (dt - season_start).days
                week = (days_diff // 7) + 1
                return max(1, min(22, week))  # Clamp to 1-22

            combined['week'] = combined['commence_time'].apply(estimate_week)
            print(f"  ✅ Extracted season/week for {len(combined)} props")
        elif 'source_file' in combined.columns:
            # Try to extract from filename (e.g., player_props_history_20231010T000000Z.csv)
            print("  Extracting season/week from filenames...")

            def extract_date_from_filename(filename):
                import re
                match = re.search(r'(\d{8})', str(filename))
                if match:
                    date_str = match.group(1)
                    try:
                        dt = datetime.strptime(date_str, '%Y%m%d')
                        return dt
                    except:
                        pass
                return None

            combined['file_date'] = combined['source_file'].apply(extract_date_from_filename)

            def estimate_week_from_date(dt):
                if pd.isna(dt):
                    return None
                year = dt.year if dt.month >= 9 else dt.year
                season_start = datetime(year, 9, 7)
                days_diff = (dt - season_start).days
                week = (days_diff // 7) + 1
                return max(1, min(22, week))

            combined['season'] = combined['file_date'].apply(
                lambda x: x.year if pd.notna(x) and x.month >= 9
                else (x.year - 1 if pd.notna(x) else None)
            )
            combined['week'] = combined['file_date'].apply(estimate_week_from_date)

            non_null = combined[['season', 'week']].notna().all(axis=1).sum()
            print(f"  ✅ Extracted season/week for {non_null} props")

    # Analyze coverage
    print("\n" + "=" * 80)
    print("TEMPORAL COVERAGE ANALYSIS")
    print("=" * 80)

    if 'season' in combined.columns and 'week' in combined.columns:
        coverage = combined[combined['season'].notna() & combined['week'].notna()].copy()
        coverage_stats = coverage.groupby(['season', 'week']).agg({
            'player': 'count',
            'market': lambda x: x.nunique() if 'market' in coverage.columns else 0
        }).reset_index()
        coverage_stats.columns = ['season', 'week', 'num_props', 'num_markets']
        coverage_stats = coverage_stats.sort_values(['season', 'week'])

        for season in sorted(coverage_stats['season'].unique()):
            season_data = coverage_stats[coverage_stats['season'] == season]
            print(f"\n{int(season)} Season:")
            print(f"  Weeks covered: {sorted(season_data['week'].astype(int).tolist())}")
            print(f"  Total props: {season_data['num_props'].sum():,}")

            # Show detail
            for _, row in season_data.head(10).iterrows():
                print(f"    Week {int(row['week']):2d}: {int(row['num_props']):5,} props, {int(row['num_markets'])} markets")

            if len(season_data) > 10:
                print(f"    ... and {len(season_data) - 10} more weeks")

    # Check against NFLverse coverage
    print("\n" + "=" * 80)
    print("COMPARING TO NFLVERSE COVERAGE")
    print("=" * 80)

    nflverse_path = Path("data/nflverse/weekly_historical.parquet")
    if nflverse_path.exists():
        nflverse = pd.read_parquet(nflverse_path)
        nflverse_weeks = nflverse.groupby(['season', 'week']).size().reset_index(name='num_players')

        print("\nNFLverse weeks available:")
        for season in sorted(nflverse_weeks['season'].unique()):
            season_weeks = sorted(nflverse_weeks[nflverse_weeks['season'] == season]['week'].tolist())
            print(f"  {int(season)}: weeks {min(season_weeks)}-{max(season_weeks)} ({len(season_weeks)} weeks)")

        # Find gaps
        if 'season' in combined.columns and 'week' in combined.columns:
            prop_weeks = set(
                (int(row['season']), int(row['week']))
                for _, row in coverage_stats.iterrows()
            )
            nflverse_weeks_set = set(
                (int(row['season']), int(row['week']))
                for _, row in nflverse_weeks.iterrows()
            )

            missing = sorted(nflverse_weeks_set - prop_weeks)

            print(f"\n🔍 Missing prop coverage ({len(missing)} weeks):")

            if missing:
                missing_by_season = defaultdict(list)
                for season, week in missing:
                    missing_by_season[season].append(week)

                for season in sorted(missing_by_season.keys()):
                    weeks = sorted(missing_by_season[season])
                    print(f"  {season}: {weeks}")
            else:
                print("  ✅ Complete coverage! All NFLverse weeks have props.")

    # Market breakdown
    if 'market' in combined.columns:
        print("\n" + "=" * 80)
        print("MARKET COVERAGE")
        print("=" * 80)

        market_counts = combined['market'].value_counts()
        for market, count in market_counts.items():
            print(f"  {market}: {count:,} props")

    # Deduplication analysis
    print("\n" + "=" * 80)
    print("DEDUPLICATION ANALYSIS")
    print("=" * 80)

    # Try to identify duplicates
    dup_cols = ['player', 'market', 'line']
    if 'commence_time' in combined.columns:
        dup_cols.append('commence_time')

    available_dup_cols = [col for col in dup_cols if col in combined.columns]

    if available_dup_cols:
        before = len(combined)
        combined_dedup = combined.drop_duplicates(subset=available_dup_cols)
        after = len(combined_dedup)
        duplicates = before - after

        print(f"  Total props: {before:,}")
        print(f"  Unique props: {after:,}")
        print(f"  Duplicates removed: {duplicates:,} ({100*duplicates/before:.1f}%)")

        return combined_dedup
    else:
        print("  ⚠️ Cannot deduplicate: missing key columns")
        return combined

if __name__ == "__main__":
    final_df = analyze_all_prop_sources()

    if final_df is not None and len(final_df) > 0:
        output_path = Path("data/analysis/comprehensive_prop_inventory.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved comprehensive inventory to {output_path}")
        print(f"   Total unique props: {len(final_df):,}")
