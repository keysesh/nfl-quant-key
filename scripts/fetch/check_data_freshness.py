#!/usr/bin/env python3
"""
Check data freshness and refresh if stale.

Run this BEFORE generating predictions to ensure data is current.

Usage:
    python scripts/fetch/check_data_freshness.py
    python scripts/fetch/check_data_freshness.py --week 13

    # Force refresh regardless of staleness
    python scripts/fetch/check_data_freshness.py --force
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import subprocess
import glob as glob_module

# Project root (adjust if needed)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.config_paths import (
    NFLVERSE_DIR, DATA_DIR, INJURIES_DIR,
    WEEKLY_STATS_FILE, SNAP_COUNTS_FILE, DEPTH_CHARTS_FILE,
    ROSTERS_FILE, INJURIES_FILE
)

# Dynamic current season
CURRENT_SEASON = get_current_season()

# NFL season start dates by year
NFL_SEASON_STARTS = {
    2024: datetime(2024, 9, 5, tzinfo=timezone.utc),  # Week 1 Thursday
    2025: datetime(2025, 9, 4, tzinfo=timezone.utc),  # Week 1 Thursday
    2026: datetime(2026, 9, 10, tzinfo=timezone.utc), # Estimated
}


def get_current_week():
    """Estimate current NFL week based on date with timezone awareness."""
    season = CURRENT_SEASON
    season_start = NFL_SEASON_STARTS.get(season, datetime(season, 9, 1, tzinfo=timezone.utc))

    # Use timezone-aware datetime
    today = datetime.now(timezone.utc)

    if today < season_start:
        return 1
    days_since_start = (today - season_start).days
    return min(18, max(1, (days_since_start // 7) + 1))


def get_staleness_thresholds(week: int = None):
    """
    Get staleness thresholds using centralized config paths.

    Uses absolute paths from config_paths.py to ensure consistency
    with what loaders actually use.
    """
    if week is None:
        week = get_current_week()

    # Use centralized paths from config_paths.py
    thresholds = {
        # NFLverse core data - refresh if >12 hours old
        str(WEEKLY_STATS_FILE): 12,
        str(SNAP_COUNTS_FILE): 12,
        str(NFLVERSE_DIR / 'ngs_receiving.parquet'): 12,
        str(NFLVERSE_DIR / 'ngs_rushing.parquet'): 12,

        # PBP - the R script creates pbp.parquet (combined)
        str(NFLVERSE_DIR / 'pbp.parquet'): 12,

        # Depth charts - critical for features
        str(DEPTH_CHARTS_FILE): 24,

        # Rosters change less frequently
        str(ROSTERS_FILE): 24,

        # Injuries - check main parquet location
        str(INJURIES_FILE): 6,

        # Current week's odds (CRITICAL for predictions)
        str(DATA_DIR / f'odds_player_props_week{week}.csv'): 4,
        str(DATA_DIR / f'odds_week{week}.csv'): 4,
    }
    return thresholds


# Optional files - warn but don't fail (use absolute paths)
OPTIONAL_FILES = [
    str(INJURIES_FILE),
    str(NFLVERSE_DIR / 'ngs_receiving.parquet'),
    str(NFLVERSE_DIR / 'ngs_rushing.parquet'),
    str(ROSTERS_FILE),
]


def get_file_age_hours(filepath: Path) -> float:
    """Get file age in hours (timezone-aware)."""
    if not filepath.exists():
        return float('inf')

    mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    return age.total_seconds() / 3600


def format_age(hours: float) -> str:
    """Format age for display."""
    if hours == float('inf'):
        return "MISSING"
    elif hours < 1:
        return f"{hours * 60:.0f}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        return f"{hours / 24:.1f}d"


def check_freshness(verbose: bool = True, week: int = None) -> tuple[bool, list[str], list[str]]:
    """
    Check freshness of all data files.

    Args:
        verbose: Print detailed output
        week: NFL week to check odds for (default: auto-detect)

    Returns:
        (all_fresh, stale_files, missing_required) tuple
    """
    stale_files = []
    missing_required = []
    now = datetime.now(timezone.utc)

    if week is None:
        week = get_current_week()

    staleness_thresholds = get_staleness_thresholds(week)

    if verbose:
        print("=" * 70)
        print(f"DATA FRESHNESS CHECK - {now.strftime('%Y-%m-%d %H:%M')} (Week {week})")
        print("=" * 70)
        print(f"{'File':<50} {'Age':>8} {'Max':>6} {'Status':>8}")
        print("-" * 70)

    for file_path_str, max_hours in staleness_thresholds.items():
        # Paths are now absolute from config_paths
        full_path = Path(file_path_str)

        # Get display path (relative to project root for readability)
        try:
            display_path = str(full_path.relative_to(PROJECT_ROOT))
        except ValueError:
            display_path = str(full_path)

        # Check if this is an optional file
        is_optional = file_path_str in OPTIONAL_FILES

        # For odds files, also check alternative naming patterns
        if 'odds_' in file_path_str and not full_path.exists():
            # Try to find any matching odds file for this week
            pattern = f"data/odds*week{week}*.csv"
            matches = list(PROJECT_ROOT.glob(pattern))
            if matches:
                full_path = matches[0]
                display_path = str(full_path.relative_to(PROJECT_ROOT))

        age_hours = get_file_age_hours(full_path)
        age_str = format_age(age_hours)

        if age_hours == float('inf'):
            status = "⚠️ MISSING" if is_optional else "❌ MISSING"
            if not is_optional:
                missing_required.append(display_path)
                stale_files.append(display_path)
        elif age_hours > max_hours:
            status = "⚠️ STALE"
            stale_files.append(display_path)
        else:
            status = "✅ FRESH"

        if verbose:
            # Truncate path for display
            short_path = display_path if len(display_path) <= 48 else "..." + display_path[-45:]
            print(f"{short_path:<50} {age_str:>8} {max_hours:>5}h {status:>8}")

    if verbose:
        print("=" * 70)

    all_fresh = len(stale_files) == 0

    return all_fresh, stale_files, missing_required


def refresh_nflverse_data():
    """Refresh NFLverse data using the extended fetch script."""
    fetch_script = PROJECT_ROOT / 'scripts' / 'fetch' / 'fetch_nflverse_extended.py'
    
    if not fetch_script.exists():
        # Fallback: try to find any NFLverse fetch script
        fetch_scripts = list((PROJECT_ROOT / 'scripts' / 'fetch').glob('*nflverse*.py'))
        if fetch_scripts:
            fetch_script = fetch_scripts[0]
        else:
            print("❌ No NFLverse fetch script found!")
            print("   Expected: scripts/fetch/fetch_nflverse_extended.py")
            return False
    
    print(f"\n🔄 Running: python {fetch_script.relative_to(PROJECT_ROOT)}")
    result = subprocess.run(
        [sys.executable, str(fetch_script)],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    return result.returncode == 0


def refresh_rosters():
    """Refresh roster data from NFLverse GitHub."""
    print("\n🔄 Refreshing rosters...")
    
    try:
        import pandas as pd
        url = 'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2025.csv'
        df = pd.read_csv(url)
        
        output_path = PROJECT_ROOT / 'data' / 'nflverse' / 'rosters_2025.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"   ✅ Saved {len(df)} roster entries")
        return True
    except Exception as e:
        print(f"   ❌ Failed to refresh rosters: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Check and refresh data freshness')
    parser.add_argument('--force', action='store_true', help='Force refresh regardless of staleness')
    parser.add_argument('--no-refresh', action='store_true', help='Check only, do not refresh')
    parser.add_argument('--auto-refresh', action='store_true',
                        help='Auto-refresh stale data without prompting (for CI/automation)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail with exit code 1 if any required data is missing/stale (for betting mode)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument('--week', '-w', type=int, help='NFL week to check (default: auto-detect)')
    args = parser.parse_args()

    verbose = not args.quiet
    week = args.week

    if args.force:
        print("🔄 Force refresh requested...")
        refresh_nflverse_data()
        refresh_rosters()
        print("\n✅ Refresh complete. Re-checking freshness...")
        all_fresh, stale_files, missing = check_freshness(verbose=verbose, week=week)
        return 0

    all_fresh, stale_files, missing_required = check_freshness(verbose=verbose, week=week)
    
    if all_fresh:
        if verbose:
            print("\n✅ All data is fresh. Ready for predictions.")
        return 0
    
    # Report issues
    if verbose:
        print(f"\n⚠️  {len(stale_files)} file(s) are stale or missing:")
        for f in stale_files[:5]:
            print(f"   - {f}")
        if len(stale_files) > 5:
            print(f"   ... and {len(stale_files) - 5} more")
    
    if missing_required:
        print(f"\n❌ CRITICAL: {len(missing_required)} required file(s) missing!")
        for f in missing_required:
            print(f"   - {f}")

    # STRICT MODE: Fail immediately if any required data missing/stale
    if args.strict:
        if missing_required or stale_files:
            print("\n❌ STRICT MODE: Data is not fresh. Failing.")
            print("   Run with --auto-refresh to fix, or manually refresh data.")
            return 1
        return 0

    # NO-REFRESH MODE: Just report, don't refresh
    if args.no_refresh:
        print("\n⚠️  Skipping refresh (--no-refresh specified)")
        return 1 if missing_required else 0

    # Determine if we should refresh (auto or interactive)
    should_refresh = False

    if args.auto_refresh:
        # AUTO-REFRESH MODE: Refresh without prompting
        should_refresh = True
        print("\n🔄 Auto-refreshing stale data...")
    elif sys.stdin.isatty():
        # INTERACTIVE MODE: Prompt user (only if TTY available)
        try:
            response = input("\n🔄 Refresh stale data now? [Y/n]: ").strip().lower()
            should_refresh = (response != 'n')
        except (EOFError, KeyboardInterrupt):
            should_refresh = False
            print()
    else:
        # NON-INTERACTIVE (pipe/cron/CI): Default to no-refresh
        print("\n⚠️  Non-interactive mode: Skipping refresh (use --auto-refresh to enable)")
        return 1 if missing_required else 0

    if should_refresh:
        print()

        # Determine what to refresh
        needs_nflverse = any('nflverse' in f or 'weekly' in f or 'snap' in f or 'ngs' in f or 'pbp' in f
                            for f in stale_files)
        needs_rosters = any('roster' in f for f in stale_files)

        if needs_nflverse:
            refresh_nflverse_data()

        if needs_rosters:
            refresh_rosters()

        # Re-check
        print("\n📋 Re-checking freshness after refresh...")
        all_fresh, stale_files, missing = check_freshness(verbose=verbose, week=week)

        if all_fresh:
            print("\n✅ All data is now fresh. Ready for predictions.")
            return 0
        else:
            print(f"\n⚠️  Some files still stale. May need manual intervention.")
            # Check if odds are missing - provide instructions
            odds_missing = [f for f in stale_files if 'odds' in f]
            if odds_missing:
                print(f"\n📋 ODDS FILES MISSING/STALE:")
                print(f"   Run: python scripts/fetch/fetch_comprehensive_odds.py")
                print(f"   Or manually download from sportsbook")
            return 1
    else:
        print("\n⚠️  Skipping refresh. Data may be stale.")
        return 1 if missing_required else 0


if __name__ == '__main__':
    sys.exit(main())
