#!/usr/bin/env python3
"""
Phase 0: Fetch Run Inputs

Collects all network-dependent inputs and saves them to a pinned snapshot
directory for reproducible pipeline runs.

This script MUST be run before the main pipeline to ensure:
1. All network calls happen in Phase 0 only
2. Pipeline runs are deterministic (use snapshots, not live data)
3. Run artifacts are preserved for debugging

Usage:
    python scripts/fetch/fetch_run_inputs.py --week 17 --run-id my_run_123

    # Or let it auto-generate run ID
    python scripts/fetch/fetch_run_inputs.py --week 17

Output:
    runs/<run_id>/inputs/
        # API data (live fetch)
        players.parquet      # Sleeper player data
        injuries.parquet     # Current injury data

        # NFLverse data (copied from global)
        pbp.parquet          # Play-by-play
        weekly_stats.parquet # Weekly player stats
        snap_counts.parquet  # Snap count data
        depth_charts.parquet # Depth chart data
        rosters.parquet      # Roster data
        schedules.parquet    # Schedule data

        # Odds data (if available)
        odds_player_props.csv  # Player props
        odds_game.csv          # Game lines

        manifest.json        # Run manifest with all metadata

The manifest.json contains:
    - run_id, timestamp, week, season
    - snapshot paths and checksums
    - resolver availability status
    - any warnings or errors encountered
"""

import argparse
import json
import hashlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import (
    NFLVERSE_DIR, DATA_DIR,
    WEEKLY_STATS_FILE, SNAP_COUNTS_FILE, SCHEDULES_FILE,
    DEPTH_CHARTS_FILE, ROSTERS_FILE, INJURIES_FILE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()[:16]  # First 16 chars for brevity


def fetch_player_data(output_dir: Path) -> Dict[str, Any]:
    """Fetch and snapshot Sleeper player data."""
    from nfl_quant.data.player_resolver import PlayerResolver, ResolverMode

    logger.info("Fetching player data from Sleeper API...")

    try:
        # Use ONLINE mode to fetch fresh data
        resolver = PlayerResolver(mode=ResolverMode.ONLINE)

        if not resolver.is_available:
            return {
                'success': False,
                'error': f"Resolver unavailable: {resolver.unavailable_reason}",
                'path': None
            }

        # Save snapshot
        snapshot_path = output_dir / 'players.parquet'
        metadata = resolver.save_snapshot(snapshot_path)

        # Compute hash
        file_hash = compute_file_hash(snapshot_path)

        return {
            'success': True,
            'path': str(snapshot_path),
            'hash': file_hash,
            'record_count': metadata.get('record_count', 0),
            'gsis_coverage': metadata.get('gsis_coverage', 0),
            'saved_at': metadata.get('saved_at'),
        }

    except Exception as e:
        logger.error(f"Failed to fetch player data: {e}")
        return {
            'success': False,
            'error': str(e),
            'path': None
        }


def fetch_injury_data(output_dir: Path, season: int, week: int) -> Dict[str, Any]:
    """Fetch and snapshot Sleeper injury data."""
    from nfl_quant.data.injury_loader import get_injuries, InjuryDataError

    logger.info("Fetching injury data from Sleeper API...")

    try:
        # Force refresh to get latest data
        injuries_df = get_injuries(season=season, week=week, refresh=True)

        # Save snapshot
        snapshot_path = output_dir / 'injuries.parquet'
        injuries_df.to_parquet(snapshot_path, index=False)

        # Save metadata
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'season': season,
            'week': week,
            'record_count': len(injuries_df),
            'teams': injuries_df['team'].nunique() if not injuries_df.empty else 0,
            'statuses': injuries_df['status'].value_counts().to_dict() if not injuries_df.empty else {},
        }

        metadata_path = output_dir / 'injuries.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Compute hash
        file_hash = compute_file_hash(snapshot_path)

        return {
            'success': True,
            'path': str(snapshot_path),
            'hash': file_hash,
            'record_count': metadata['record_count'],
            'saved_at': metadata['saved_at'],
        }

    except InjuryDataError as e:
        logger.error(f"Failed to fetch injury data: {e}")
        return {
            'success': False,
            'error': str(e),
            'path': None
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching injury data: {e}")
        return {
            'success': False,
            'error': str(e),
            'path': None
        }


def snapshot_nflverse_files(output_dir: Path, week: int, season: int) -> Dict[str, Any]:
    """
    Snapshot all required NFLverse files from global data directory.

    This copies the current state of NFLverse data to the run's inputs directory,
    ensuring the pipeline uses pinned data regardless of future refreshes.
    """
    logger.info("Snapshotting NFLverse data files...")

    # Files to snapshot (source -> destination name)
    nflverse_files = {
        'pbp.parquet': NFLVERSE_DIR / 'pbp.parquet',
        'weekly_stats.parquet': WEEKLY_STATS_FILE,
        'snap_counts.parquet': SNAP_COUNTS_FILE,
        'depth_charts.parquet': DEPTH_CHARTS_FILE,
        'rosters.parquet': ROSTERS_FILE,
        'schedules.parquet': SCHEDULES_FILE,
    }

    # Also try season-specific PBP if generic doesn't exist
    if not (NFLVERSE_DIR / 'pbp.parquet').exists():
        nflverse_files['pbp.parquet'] = NFLVERSE_DIR / f'pbp_{season}.parquet'

    results = {}
    success_count = 0
    missing_required = []

    # Required files - fail if missing
    required_files = ['pbp.parquet', 'weekly_stats.parquet', 'snap_counts.parquet']

    for dest_name, src_path in nflverse_files.items():
        if not src_path.exists():
            is_required = dest_name in required_files
            if is_required:
                missing_required.append(dest_name)
            results[dest_name] = {
                'success': False,
                'error': f"Source file not found: {src_path}",
                'required': is_required
            }
            logger.warning(f"  MISSING: {dest_name} ({src_path})")
            continue

        try:
            dest_path = output_dir / dest_name
            shutil.copy2(src_path, dest_path)

            # Compute hash
            file_hash = compute_file_hash(dest_path)

            results[dest_name] = {
                'success': True,
                'path': str(dest_path),
                'hash': file_hash,
                'size_mb': dest_path.stat().st_size / (1024 * 1024),
                'source': str(src_path),
            }
            success_count += 1
            logger.info(f"  OK: {dest_name} ({results[dest_name]['size_mb']:.1f} MB)")

        except Exception as e:
            results[dest_name] = {
                'success': False,
                'error': str(e),
            }
            logger.error(f"  FAILED: {dest_name} - {e}")

    return {
        'success': len(missing_required) == 0,
        'files': results,
        'count': success_count,
        'total': len(nflverse_files),
        'missing_required': missing_required,
    }


def snapshot_odds_files(output_dir: Path, week: int) -> Dict[str, Any]:
    """Snapshot odds files for the current week if available."""
    logger.info(f"Snapshotting odds files for week {week}...")

    odds_files = {
        'odds_player_props.csv': DATA_DIR / f'odds_player_props_week{week}.csv',
        'odds_game.csv': DATA_DIR / f'odds_week{week}.csv',
    }

    results = {}
    success_count = 0

    for dest_name, src_path in odds_files.items():
        if not src_path.exists():
            results[dest_name] = {
                'success': False,
                'error': f"Source file not found: {src_path}",
                'required': False  # Odds are fetched live in Phase 1
            }
            logger.warning(f"  SKIPPED: {dest_name} (not yet fetched)")
            continue

        try:
            dest_path = output_dir / dest_name
            shutil.copy2(src_path, dest_path)

            file_hash = compute_file_hash(dest_path)

            results[dest_name] = {
                'success': True,
                'path': str(dest_path),
                'hash': file_hash,
                'source': str(src_path),
            }
            success_count += 1
            logger.info(f"  OK: {dest_name}")

        except Exception as e:
            results[dest_name] = {
                'success': False,
                'error': str(e),
            }
            logger.error(f"  FAILED: {dest_name} - {e}")

    return {
        'success': True,  # Odds are optional in Phase 0
        'files': results,
        'count': success_count,
        'total': len(odds_files),
    }


def create_manifest(
    run_id: str,
    season: int,
    week: int,
    output_dir: Path,
    snapshots: Dict[str, Any],
    warnings: List[str]
) -> Dict[str, Any]:
    """Create the run manifest with all metadata."""

    manifest = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(),
        'season': season,
        'week': week,
        'phase': 'inputs',
        'snapshots': snapshots,
        'warnings': warnings,
        'resolver_available': snapshots.get('players', {}).get('success', False),
        'injuries_available': snapshots.get('injuries', {}).get('success', False),
        'nflverse_available': snapshots.get('nflverse', {}).get('success', False),
        'odds_available': snapshots.get('odds', {}).get('count', 0) > 0,
    }

    # Determine overall status - nflverse is required, others are optional
    nflverse_ok = snapshots.get('nflverse', {}).get('success', False)
    all_success = nflverse_ok and all(
        s.get('success', False) for k, s in snapshots.items()
        if k not in ['odds']  # Odds are optional
    )
    manifest['status'] = 'complete' if all_success else 'partial'

    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved manifest to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Phase 0: Fetch and snapshot all network-dependent inputs'
    )
    parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='NFL week number'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help='NFL season (default: 2025)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run identifier (auto-generated if not provided)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail if any snapshot fails (default: continue with warnings)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory (default: runs/<run_id>/inputs)'
    )

    args = parser.parse_args()

    # Generate run ID if not provided
    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"week{args.week}_{timestamp}"

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'runs' / run_id / 'inputs'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Phase 0: Fetch Run Inputs ===")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Season: {args.season}, Week: {args.week}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    snapshots = {}
    warnings = []

    # Fetch player data
    logger.info("--- Fetching Player Data ---")
    player_result = fetch_player_data(output_dir)
    snapshots['players'] = player_result
    if player_result['success']:
        logger.info(f"  OK: {player_result['record_count']} players")
    else:
        msg = f"Player data failed: {player_result.get('error')}"
        logger.warning(f"  WARN: {msg}")
        warnings.append(msg)
        if args.strict:
            logger.error("Strict mode: failing due to player data error")
            sys.exit(1)

    # Fetch injury data
    logger.info("--- Fetching Injury Data ---")
    injury_result = fetch_injury_data(output_dir, args.season, args.week)
    snapshots['injuries'] = injury_result
    if injury_result['success']:
        logger.info(f"  OK: {injury_result['record_count']} injuries")
    else:
        msg = f"Injury data failed: {injury_result.get('error')}"
        logger.warning(f"  WARN: {msg}")
        warnings.append(msg)
        if args.strict:
            logger.error("Strict mode: failing due to injury data error")
            sys.exit(1)

    # Snapshot NFLverse data files
    logger.info("--- Snapshotting NFLverse Data ---")
    nflverse_result = snapshot_nflverse_files(output_dir, args.week, args.season)
    snapshots['nflverse'] = nflverse_result
    if nflverse_result['success']:
        logger.info(f"  OK: {nflverse_result['count']}/{nflverse_result['total']} files")
    else:
        missing = nflverse_result.get('missing_required', [])
        msg = f"NFLverse snapshot failed: missing required files: {missing}"
        logger.error(f"  ERROR: {msg}")
        warnings.append(msg)
        if args.strict:
            logger.error("Strict mode: failing due to missing NFLverse data")
            sys.exit(1)

    # Snapshot odds files (optional - may not exist yet)
    logger.info("--- Snapshotting Odds Data ---")
    odds_result = snapshot_odds_files(output_dir, args.week)
    snapshots['odds'] = odds_result
    logger.info(f"  {odds_result['count']}/{odds_result['total']} files snapshotted")

    # Create manifest
    logger.info("--- Creating Manifest ---")
    manifest = create_manifest(
        run_id=run_id,
        season=args.season,
        week=args.week,
        output_dir=output_dir,
        snapshots=snapshots,
        warnings=warnings
    )

    # Summary
    logger.info("")
    logger.info("=== Phase 0 Complete ===")
    logger.info(f"Status: {manifest['status']}")
    logger.info(f"NFLverse data: {'✓' if manifest['nflverse_available'] else '✗'}")
    logger.info(f"Player resolver: {'✓' if manifest['resolver_available'] else '✗'}")
    logger.info(f"Injuries: {'✓' if manifest['injuries_available'] else '✗'}")
    logger.info(f"Odds: {'✓' if manifest['odds_available'] else '(not yet fetched)'}")
    if warnings:
        logger.info(f"Warnings: {len(warnings)}")
        for w in warnings:
            logger.info(f"  - {w}")
    logger.info("")
    logger.info(f"Inputs saved to: {output_dir}")
    logger.info(f"Use with pipeline: python scripts/run_pipeline.py --run-id {run_id}")

    # Return success if at least partial completion
    if manifest['status'] == 'complete':
        sys.exit(0)
    else:
        sys.exit(0 if not args.strict else 1)


if __name__ == '__main__':
    main()
