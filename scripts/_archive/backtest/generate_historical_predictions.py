#!/usr/bin/env python3
"""
Generate Historical Predictions for Backtesting
================================================

Generates predictions for Weeks 1-11 using TIER 1 & 2 models.

This script simulates walk-forward validation by training on past data
and predicting future weeks.

Usage:
    python scripts/backtest/generate_historical_predictions.py --weeks 1-11
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def generate_predictions_for_week(week: int) -> bool:
    """
    Generate predictions for a specific week using TIER 1&2 models.

    Args:
        week: Week number to generate predictions for

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING PREDICTIONS FOR WEEK {week}")
    logger.info(f"{'='*80}\n")

    # Path to prediction script
    pred_script = project_root / "scripts/predict/generate_model_predictions.py"

    if not pred_script.exists():
        logger.error(f"Prediction script not found: {pred_script}")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(pred_script),
        str(week),
        # Add flag to use TIER 1&2 models if available
        # "--use-tier12-models"  # Uncomment when flag is implemented
    ]

    # Set environment
    env = {
        "PYTHONPATH": f"{project_root}:$PYTHONPATH"
    }

    try:
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout
            env=env
        )

        if result.returncode == 0:
            logger.info(f"✅ Week {week} predictions generated successfully")

            # Check if output file was created
            output_file = project_root / f"data/model_predictions_week{week}.csv"
            if output_file.exists():
                logger.info(f"  📄 Output: {output_file}")
                return True
            else:
                logger.warning(f"  ⚠️  Expected output file not found: {output_file}")
                return False
        else:
            logger.error(f"❌ Week {week} prediction failed with exit code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Week {week} prediction timed out after 20 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Week {week} prediction failed: {e}")
        return False


def generate_historical_predictions(start_week: int = 1, end_week: int = 11) -> Dict[int, bool]:
    """
    Generate predictions for all historical weeks.

    Args:
        start_week: First week to generate
        end_week: Last week to generate

    Returns:
        Dictionary mapping week number to success status
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING HISTORICAL PREDICTIONS: WEEKS {start_week}-{end_week}")
    logger.info(f"{'='*80}\n")

    results = {}

    for week in range(start_week, end_week + 1):
        success = generate_predictions_for_week(week)
        results[week] = success

        if not success:
            logger.warning(f"⚠️  Week {week} failed, continuing to next week...")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATION SUMMARY")
    logger.info(f"{'='*80}\n")

    successful = sum(1 for s in results.values() if s)
    total = len(results)

    logger.info(f"Successful: {successful}/{total} weeks")
    logger.info(f"\nResults by week:")
    for week, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  Week {week:2d}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical predictions for backtesting"
    )
    parser.add_argument(
        '--weeks',
        type=str,
        default="1-11",
        help="Week range (e.g., '1-11' or '5-8')"
    )
    parser.add_argument(
        '--week',
        type=int,
        help="Generate predictions for single week only"
    )

    args = parser.parse_args()

    if args.week:
        # Single week
        success = generate_predictions_for_week(args.week)
        sys.exit(0 if success else 1)
    else:
        # Parse week range
        if '-' in args.weeks:
            start, end = map(int, args.weeks.split('-'))
        else:
            start = end = int(args.weeks)

        results = generate_historical_predictions(start, end)

        # Exit with error if any week failed
        if not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
