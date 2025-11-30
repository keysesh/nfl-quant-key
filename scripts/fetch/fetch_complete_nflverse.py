#!/usr/bin/env python3
"""
Complete NFLverse Data Fetch Orchestrator

This script orchestrates fetching ALL nflverse data sources by:
1. Running the R fetch script (primary method)
2. Validating all expected files exist
3. Falling back to Python nflreadpy if needed

Usage:
    python scripts/fetch/fetch_complete_nflverse.py
    python scripts/fetch/fetch_complete_nflverse.py --season 2025
"""

import subprocess
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_current_season() -> int:
    """Determine current NFL season."""
    now = datetime.now()
    if now.month >= 8:  # Aug-Dec = current year
        return now.year
    else:  # Jan-Jul = previous year
        return now.year - 1


def run_r_fetch(seasons: list, output_dir: str = 'data/nflverse') -> bool:
    """
    Run the R fetch script to get all nflverse data.

    Args:
        seasons: List of seasons to fetch
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info("RUNNING R FETCH SCRIPT")
    logger.info("=" * 70)

    r_script = Path('scripts/fetch/fetch_nflverse_data.R')
    if not r_script.exists():
        logger.error(f"R fetch script not found: {r_script}")
        return False

    seasons_str = ' '.join(map(str, seasons))
    cmd = [
        'Rscript',
        str(r_script),
        '--seasons', seasons_str,
        '--output-dir', output_dir,
        '--format', 'parquet'
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info("R fetch completed successfully")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"R fetch failed with code {result.returncode}")
            logger.error(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("R fetch timed out after 10 minutes")
        return False
    except FileNotFoundError:
        logger.error("Rscript not found. Is R installed?")
        return False
    except Exception as e:
        logger.error(f"Error running R fetch: {e}")
        return False


def validate_data_files(output_dir: str = 'data/nflverse') -> dict:
    """
    Validate that all expected data files exist.

    Args:
        output_dir: Directory to check

    Returns:
        Dict with file status
    """
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATING DATA FILES")
    logger.info("=" * 70)

    expected_files = [
        # Core files (required)
        ('pbp.parquet', True),
        ('player_stats.parquet', True),
        ('schedules.parquet', True),
        ('rosters.parquet', True),
        ('weekly_stats.parquet', True),

        # Advanced files (optional but highly valuable)
        ('snap_counts.parquet', False),
        ('ngs_passing.parquet', False),
        ('ngs_receiving.parquet', False),
        ('ngs_rushing.parquet', False),
        ('ff_opportunity.parquet', False),
        ('injuries.parquet', False),
        ('participation.parquet', False),
        ('depth_charts.parquet', False),
        ('combine.parquet', False),
    ]

    status = {
        'all_required': True,
        'missing_required': [],
        'missing_optional': [],
        'present': [],
    }

    output_path = Path(output_dir)

    for filename, required in expected_files:
        filepath = output_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            status['present'].append(filename)
            logger.info(f"  ✅ {filename} ({size_mb:.1f} MB)")
        else:
            if required:
                status['missing_required'].append(filename)
                status['all_required'] = False
                logger.error(f"  ❌ {filename} (REQUIRED - MISSING)")
            else:
                status['missing_optional'].append(filename)
                logger.warning(f"  ⚠️  {filename} (optional - missing)")

    logger.info(f"\nPresent: {len(status['present'])}/{len(expected_files)}")
    logger.info(f"Missing Required: {len(status['missing_required'])}")
    logger.info(f"Missing Optional: {len(status['missing_optional'])}")

    return status


def fetch_with_python_fallback(seasons: list, output_dir: str = 'data/nflverse'):
    """
    Fallback to Python nflreadpy for any missing files.

    Args:
        seasons: Seasons to fetch
        output_dir: Output directory
    """
    logger.info("\n" + "=" * 70)
    logger.info("PYTHON FALLBACK - FETCHING MISSING DATA")
    logger.info("=" * 70)

    try:
        import nflreadpy as nfl
        import pandas as pd

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Try to fetch missing critical files
        missing_fetchers = {
            'snap_counts.parquet': lambda: nfl.load_snap_counts(seasons),
            'ngs_passing.parquet': lambda: nfl.load_nextgen_stats(seasons, stat_type='passing'),
            'ngs_receiving.parquet': lambda: nfl.load_nextgen_stats(seasons, stat_type='receiving'),
            'ngs_rushing.parquet': lambda: nfl.load_nextgen_stats(seasons, stat_type='rushing'),
            'ff_opportunity.parquet': lambda: nfl.load_ff_opportunity(seasons),
            'injuries.parquet': lambda: nfl.load_injuries(seasons),
            'participation.parquet': lambda: nfl.load_participation(seasons),
        }

        for filename, fetcher in missing_fetchers.items():
            filepath = output_path / filename
            if not filepath.exists():
                try:
                    logger.info(f"  Fetching {filename} via Python...")
                    data = fetcher()

                    # Convert polars to pandas if needed
                    if hasattr(data, 'to_pandas'):
                        data = data.to_pandas()

                    data.to_parquet(filepath)
                    logger.info(f"  ✅ Saved {filename}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not fetch {filename}: {e}")

    except ImportError:
        logger.warning("nflreadpy not available for Python fallback")
    except Exception as e:
        logger.error(f"Error in Python fallback: {e}")


def main():
    """Main orchestrator."""
    parser = argparse.ArgumentParser(description='Fetch complete NFLverse data')
    parser.add_argument('--season', type=int, default=None,
                        help='Season to fetch (default: current + previous)')
    parser.add_argument('--output-dir', type=str, default='data/nflverse',
                        help='Output directory')
    parser.add_argument('--skip-r', action='store_true',
                        help='Skip R fetch, use Python only')
    args = parser.parse_args()

    # Determine seasons
    if args.season:
        seasons = [args.season]
    else:
        current = get_current_season()
        seasons = [current - 1, current]

    logger.info("=" * 70)
    logger.info("NFL QUANT - COMPLETE NFLVERSE DATA FETCH")
    logger.info("=" * 70)
    logger.info(f"Seasons: {seasons}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Run R fetch (primary method)
    if not args.skip_r:
        r_success = run_r_fetch(seasons, args.output_dir)
    else:
        r_success = False
        logger.info("Skipping R fetch as requested")

    # Step 2: Validate files
    validation = validate_data_files(args.output_dir)

    # Step 3: Python fallback for missing optional files
    if validation['missing_optional']:
        fetch_with_python_fallback(seasons, args.output_dir)

    # Step 4: Final validation
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION")
    logger.info("=" * 70)
    final_validation = validate_data_files(args.output_dir)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 70)

    if final_validation['all_required']:
        logger.info("✅ All required files present")
    else:
        logger.error("❌ Missing required files:")
        for f in final_validation['missing_required']:
            logger.error(f"   - {f}")

    if final_validation['missing_optional']:
        logger.warning("⚠️  Missing optional files (advanced features may be limited):")
        for f in final_validation['missing_optional']:
            logger.warning(f"   - {f}")

    # Feature availability
    logger.info("\nFeature Availability:")
    if (Path(args.output_dir) / 'ngs_passing.parquet').exists():
        logger.info("  ✅ Next Gen Stats (QB skill metrics)")
    else:
        logger.warning("  ❌ Next Gen Stats - NGS features will use defaults")

    if (Path(args.output_dir) / 'ngs_receiving.parquet').exists():
        logger.info("  ✅ Receiver Separation Metrics")
    else:
        logger.warning("  ❌ Receiver Separation - will estimate from PBP")

    if (Path(args.output_dir) / 'snap_counts.parquet').exists():
        logger.info("  ✅ Snap Counts (usage prediction)")
    else:
        logger.warning("  ❌ Snap Counts - will estimate from PBP")

    if (Path(args.output_dir) / 'ff_opportunity.parquet').exists():
        logger.info("  ✅ Expected Fantasy Points (regression analysis)")
    else:
        logger.warning("  ❌ Expected FP - regression analysis unavailable")

    logger.info("\n" + "=" * 70)
    logger.info("Next Steps:")
    logger.info("1. Run calibrator training: python scripts/train/train_position_market_calibrators.py")
    logger.info("2. Generate predictions: python scripts/predict/generate_enhanced_recommendations.py")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
