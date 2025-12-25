#!/usr/bin/env python3
"""
NFL QUANT Parallel Pipeline Runner

Optimized version of run_pipeline.py that runs independent steps in parallel.
Safely parallelizes data fetching and recommendation generation.

Parallelization Strategy:
- Group 1 (PARALLEL): NFLverse, Injuries, Live Odds, Player Props
- Group 2 (SEQUENTIAL): Freshness Check, Model Predictions
- Group 3 (PARALLEL): Player Prop Recommendations, Game Line Recommendations
- Group 4 (SEQUENTIAL): Dashboard

Expected speedup: ~30-40% reduction in total runtime

Usage:
    python scripts/run_pipeline_parallel.py <WEEK>
    python scripts/run_pipeline_parallel.py 15

    # Force sequential mode (same as original)
    python scripts/run_pipeline_parallel.py 15 --sequential
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import argparse

# Use centralized path configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from nfl_quant.config_paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)


def run_command(args: Tuple[str, List[str], bool]) -> Tuple[str, bool, float, str]:
    """
    Run a single command and return results.

    Args:
        args: Tuple of (description, command, required)

    Returns:
        Tuple of (description, success, duration_seconds, error_message)
    """
    description, command, required = args
    start_time = time.time()
    error_msg = ""

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per step
        )
        success = True
    except subprocess.CalledProcessError as e:
        success = not required
        error_msg = f"Exit code {e.returncode}: {e.stderr[:500] if e.stderr else 'No stderr'}"
    except subprocess.TimeoutExpired:
        success = False
        error_msg = "Timeout (30 minutes)"
    except FileNotFoundError as e:
        success = not required
        error_msg = f"Command not found: {e}"
    except Exception as e:
        success = False
        error_msg = str(e)

    duration = time.time() - start_time
    return (description, success, duration, error_msg)


def run_parallel_group(
    steps: List[Tuple[str, List[str], bool]],
    group_name: str,
    max_workers: int = 4
) -> Tuple[bool, List[Tuple[str, bool, float, str]]]:
    """
    Run a group of steps in parallel.

    Returns:
        Tuple of (all_required_passed, results_list)
    """
    print(f"\n{'='*80}")
    print(f"PARALLEL: {group_name}")
    print(f"{'='*80}")
    print(f"Running {len(steps)} tasks in parallel...")

    results = []
    all_required_passed = True

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_step = {
            executor.submit(run_command, step): step
            for step in steps
        }

        for future in as_completed(future_to_step):
            desc, success, duration, error = future.result()
            results.append((desc, success, duration, error))

            status = "✅" if success else "❌"
            print(f"  {status} {desc} ({duration:.1f}s)")

            if not success and error:
                print(f"      Error: {error[:200]}")

            # Check if this was a required step that failed
            original_step = future_to_step[future]
            if not success and original_step[2]:  # required=True
                all_required_passed = False

    return all_required_passed, results


def run_sequential_step(
    description: str,
    command: List[str],
    required: bool = True,
    timeout: int = 1800  # Default 30 minute timeout
) -> Tuple[bool, float, str]:
    """
    Run a single step sequentially with live output.

    Returns:
        Tuple of (success, duration_seconds, error_message)
    """
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(command)}")
    timeout_min = timeout // 60
    print(f"Timeout: {timeout_min} minutes")
    print()

    start_time = time.time()
    error_msg = ""

    try:
        # Run with live output (not captured)
        result = subprocess.run(
            command,
            check=True,
            timeout=timeout
        )
        success = True
        print(f"\n✅ {description} - SUCCESS ({time.time() - start_time:.1f}s)")
    except subprocess.CalledProcessError as e:
        success = not required
        error_msg = f"Exit code {e.returncode}"
        status = "⚠️ " if not required else "❌"
        print(f"\n{status} {description} - FAILED ({error_msg})")
    except subprocess.TimeoutExpired:
        success = False
        error_msg = f"Timeout ({timeout // 60} minutes)"
        print(f"\n❌ {description} - TIMEOUT ({timeout // 60} min)")
    except FileNotFoundError as e:
        success = not required
        error_msg = f"Command not found: {e}"
        print(f"\n❌ {description} - {error_msg}")

    duration = time.time() - start_time
    return (success, duration, error_msg)


def main():
    parser = argparse.ArgumentParser(description='NFL QUANT Parallel Pipeline')
    parser.add_argument('week', type=int, help='NFL week number (1-18)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run in sequential mode (no parallelization)')
    parser.add_argument('--edge-mode', action='store_true',
                        help='Use edge-based ensemble (LVT + Player Bias) instead of unified model')
    args = parser.parse_args()

    week = args.week
    if not 1 <= week <= 18:
        print(f"Error: Week must be 1-18, got {week}")
        sys.exit(1)

    mode = "SEQUENTIAL" if args.sequential else "PARALLEL"
    model_mode = "EDGE ENSEMBLE" if args.edge_mode else "UNIFIED MODEL"

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NFL QUANT PIPELINE ({mode} MODE)                       ║
║                         Week {week:2d} - {datetime.now().strftime('%Y-%m-%d %H:%M')}                              ║
║                         Model: {model_mode:<18}                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    pipeline_start = time.time()
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    all_results = []

    # =========================================================================
    # GROUP 1: Data Fetching (PARALLEL)
    # These are independent API calls that write to different files
    # =========================================================================
    data_fetch_steps = [
        (
            "Refresh NFLverse Data",
            ["Rscript", "scripts/fetch/fetch_nflverse_data.R"],
            True
        ),
        (
            "Fetch Injury Reports (Sleeper)",
            [venv_python, "scripts/fetch/fetch_injuries_sleeper.py"],
            True
        ),
        (
            "Fetch Live Odds",
            [venv_python, "scripts/fetch/fetch_live_odds.py", str(week)],
            True
        ),
        (
            "Fetch Player Props",
            [venv_python, "scripts/fetch/fetch_nfl_player_props.py"],
            True
        ),
    ]

    if args.sequential:
        # Sequential mode
        for desc, cmd, req in data_fetch_steps:
            success, duration, error = run_sequential_step(desc, cmd, req)
            all_results.append((desc, success, duration, error))
            if not success and req:
                print(f"\n❌ Pipeline failed at: {desc}")
                sys.exit(1)
    else:
        # Parallel mode
        success, results = run_parallel_group(
            data_fetch_steps,
            "Data Fetching (NFLverse + Injuries + Odds + Props)",
            max_workers=4
        )
        all_results.extend(results)
        if not success:
            print(f"\n❌ Pipeline failed during data fetching")
            sys.exit(1)

    # =========================================================================
    # GROUP 2: Validation & Predictions (SEQUENTIAL)
    # These depend on Group 1 completing
    # =========================================================================

    # Freshness check (optional - warning only)
    success, duration, error = run_sequential_step(
        "Check Data Freshness",
        [venv_python, "scripts/fetch/check_data_freshness.py"],
        required=False
    )
    all_results.append(("Check Data Freshness", success, duration, error))

    # Model predictions (CRITICAL - longest step, needs 60 min timeout)
    success, duration, error = run_sequential_step(
        f"Generate Model Predictions (Week {week})",
        [venv_python, "scripts/predict/generate_model_predictions.py", str(week)],
        required=True,
        timeout=3600  # 60 minutes for 500+ players x 30k simulations
    )
    all_results.append((f"Generate Model Predictions (Week {week})", success, duration, error))
    if not success:
        print(f"\n❌ Pipeline failed at: Model Predictions")
        sys.exit(1)

    # =========================================================================
    # GROUP 3: Recommendations (PARALLEL)
    # Player props and game lines are independent
    # =========================================================================

    # Choose player prop recommendations script based on mode
    if args.edge_mode:
        player_prop_step = (
            f"Generate Edge Recommendations (Week {week})",
            [venv_python, "scripts/predict/generate_edge_recommendations.py", "--week", str(week)],
            True
        )
    else:
        player_prop_step = (
            f"Generate Player Prop Recommendations (Week {week})",
            [venv_python, "scripts/predict/generate_unified_recommendations_v3.py", "--week", str(week)],
            True
        )

    recommendation_steps = [
        player_prop_step,
        (
            "Generate Game Line Recommendations",
            [venv_python, "scripts/predict/generate_game_line_recommendations.py"],
            True
        ),
    ]

    if args.sequential:
        for desc, cmd, req in recommendation_steps:
            success, duration, error = run_sequential_step(desc, cmd, req)
            all_results.append((desc, success, duration, error))
            if not success and req:
                print(f"\n❌ Pipeline failed at: {desc}")
                sys.exit(1)
    else:
        success, results = run_parallel_group(
            recommendation_steps,
            "Recommendations (Player Props + Game Lines)",
            max_workers=2
        )
        all_results.extend(results)
        if not success:
            print(f"\n❌ Pipeline failed during recommendation generation")
            sys.exit(1)

    # =========================================================================
    # GROUP 3.5: Parlay Recommendations (SEQUENTIAL - needs v3 or edge recommendations)
    # Uses v3 unified recommendations by default (auto-fallback to edge if available)
    # =========================================================================
    success, duration, error = run_sequential_step(
        f"Generate Parlay Recommendations (Week {week})",
        [venv_python, "scripts/predict/generate_parlay_recommendations.py", "--week", str(week)],
        required=False  # Not required - parlay generation is optional
    )
    all_results.append((f"Generate Parlay Recommendations (Week {week})", success, duration, error))
    if not success:
        print(f"  ⚠️  Parlay recommendations failed (non-critical)")

    # =========================================================================
    # GROUP 4: Dashboard (SEQUENTIAL)
    # Needs both recommendation outputs
    # =========================================================================
    success, duration, error = run_sequential_step(
        "Generate Pro Dashboard",
        [venv_python, "scripts/dashboard/generate_pro_dashboard.py"],
        required=True
    )
    all_results.append(("Generate Pro Dashboard", success, duration, error))
    if not success:
        print(f"\n❌ Pipeline failed at: Dashboard")
        sys.exit(1)

    # =========================================================================
    # GROUP 5: Deploy to Vercel (SEQUENTIAL)
    # Pushes dashboard to GitHub, triggers Vercel auto-deploy
    # =========================================================================
    success, duration, error = run_sequential_step(
        "Deploy Dashboard to Vercel",
        ["bash", "scripts/deploy_dashboard.sh"],
        required=False  # Deploy failure shouldn't fail pipeline
    )
    all_results.append(("Deploy Dashboard to Vercel", success, duration, error))
    if not success:
        print(f"  ⚠️  Dashboard deploy failed (non-critical) - run manually with: scripts/deploy_dashboard.sh")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_duration = time.time() - pipeline_start

    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE - {mode} MODE")
    print(f"{'='*80}")
    print(f"\nStep Timings:")
    print(f"{'-'*60}")

    for desc, success, duration, error in all_results:
        status = "✅" if success else "⚠️ "
        print(f"  {status} {desc:<45} {duration:>6.1f}s")

    print(f"{'-'*60}")
    print(f"  {'TOTAL':<45} {total_duration:>6.1f}s")
    print(f"  {'(minutes)':<45} {total_duration/60:>6.1f}m")

    # Calculate theoretical sequential time for comparison
    if not args.sequential:
        sequential_estimate = sum(d for _, _, d, _ in all_results)
        savings = sequential_estimate - total_duration
        if savings > 0:
            print(f"\n⚡ Parallel mode saved approximately {savings:.0f}s ({savings/60:.1f}m)")

    print(f"\n📊 Check reports/ directory for output files")
    print(f"📁 Latest recommendations: reports/recommendations_week{week}_*.csv")
    print(f"📱 Mobile dashboard: https://nfl-quant.vercel.app")


if __name__ == "__main__":
    main()
