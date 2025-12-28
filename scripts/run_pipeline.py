#!/usr/bin/env python3
"""
NFL QUANT Pipeline Runner

Runs independent steps in parallel for faster execution (~25-30 min).

Parallelization Strategy:
- Group 1 (PARALLEL): NFLverse, Injuries, Live Odds, Player Props
- Group 2 (SEQUENTIAL): Freshness Check, Model Predictions
- Group 3 (PARALLEL): Player Prop Recommendations, Game Line Recommendations
- Group 4 (SEQUENTIAL): Dashboard, Deploy

Usage:
    python scripts/run_pipeline.py <WEEK>
    python scripts/run_pipeline.py 17

    # Edge mode (LVT + Player Bias + TD Poisson)
    python scripts/run_pipeline.py 17 --edge-mode
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
from nfl_quant.config_paths import PROJECT_ROOT, set_run_context, clear_run_context

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


def run_phase0(week: int, season: int, run_id: str) -> Tuple[bool, str]:
    """
    Run Phase 0: Fetch all network inputs and create snapshots.

    Returns:
        Tuple of (success, run_id)
    """
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")

    print(f"\n{'='*80}")
    print("PHASE 0: Fetching Network Inputs")
    print(f"{'='*80}")

    cmd = [
        venv_python,
        "scripts/fetch/fetch_run_inputs.py",
        "--week", str(week),
        "--season", str(season),
        "--run-id", run_id
    ]

    try:
        result = subprocess.run(cmd, check=True, timeout=300)
        print(f"\n✅ Phase 0 complete - inputs saved to runs/{run_id}/inputs/")
        return (True, run_id)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Phase 0 failed (exit code {e.returncode})")
        return (False, run_id)
    except subprocess.TimeoutExpired:
        print(f"\n❌ Phase 0 timeout (5 min)")
        return (False, run_id)


def setup_run_resolver(run_id: str) -> bool:
    """
    Set up the run-specific resolver from Phase 0 snapshots.

    Returns True if resolver is available, False otherwise.
    """
    from nfl_quant.data.player_resolver import (
        PlayerResolver, ResolverMode, set_run_resolver, ResolverNotAvailableError
    )

    inputs_dir = PROJECT_ROOT / "runs" / run_id / "inputs"
    snapshot_path = inputs_dir / "players.parquet"

    if not snapshot_path.exists():
        print(f"  ⚠️  No player snapshot found at {snapshot_path}")
        return False

    try:
        resolver = PlayerResolver.from_snapshot(snapshot_path, strict=False)
        set_run_resolver(resolver)
        print(f"  ✅ Run resolver loaded from {snapshot_path}")
        return resolver.is_available
    except ResolverNotAvailableError as e:
        print(f"  ⚠️  Resolver not available: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='NFL QUANT Pipeline')
    parser.add_argument('week', type=int, help='NFL week number (1-18)')
    parser.add_argument('--edge-mode', action='store_true',
                        help='Use edge-based ensemble (LVT + Player Bias + TD Poisson)')
    parser.add_argument('--unified-mode', action='store_true', default=True,
                        help='Use unified XGBoost model - DEFAULT')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID for Phase 0 snapshots (auto-generated if not provided)')
    parser.add_argument('--skip-phase0', action='store_true',
                        help='Skip Phase 0 input collection (use existing snapshots with --run-id)')
    parser.add_argument('--season', type=int, default=2025,
                        help='NFL season (default: 2025)')
    args = parser.parse_args()

    week = args.week
    if not 1 <= week <= 18:
        print(f"Error: Week must be 1-18, got {week}")
        sys.exit(1)

    # Generate run ID if not provided
    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"week{week}_{timestamp}"

    # XGBoost unified mode is default unless --edge-mode is explicitly set
    use_edge_mode = args.edge_mode
    model_mode = "EDGE ENSEMBLE + TD POISSON" if use_edge_mode else "XGBOOST CLASSIFIER"

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           NFL QUANT PIPELINE                                 ║
║                         Week {week:2d} - {datetime.now().strftime('%Y-%m-%d %H:%M')}                              ║
║                         Model: {model_mode:<18}                              ║
║                         Run ID: {run_id:<35}               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    pipeline_start = time.time()
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    all_results = []

    # =========================================================================
    # PHASE 0: Network Input Collection (Optional)
    # Fetches all network-dependent data and saves to runs/<run_id>/inputs/
    # =========================================================================
    snapshot_mode_active = False

    if not args.skip_phase0:
        phase0_success, run_id = run_phase0(week, args.season, run_id)
        if phase0_success:
            # Enable snapshot mode - all path functions will use run-specific paths
            set_run_context(run_id)
            snapshot_mode_active = True
            print(f"  ✅ Snapshot isolation enabled: runs/{run_id}/inputs/")

            # Set up run-specific resolver from snapshots
            resolver_available = setup_run_resolver(run_id)
            if not resolver_available:
                print("  ⚠️  Resolver unavailable - player matching may be degraded")
        else:
            print("  ⚠️  Phase 0 failed - continuing with legacy data fetching")
    else:
        print(f"\n  Skipping Phase 0 - using existing snapshots for run {run_id}")
        # Enable snapshot mode if using existing snapshots
        inputs_dir = PROJECT_ROOT / "runs" / run_id / "inputs"
        if inputs_dir.exists():
            set_run_context(run_id)
            snapshot_mode_active = True
            print(f"  ✅ Snapshot isolation enabled: runs/{run_id}/inputs/")
        else:
            print(f"  ⚠️  Snapshot directory not found: {inputs_dir}")

        resolver_available = setup_run_resolver(run_id)
        if not resolver_available:
            print("  ⚠️  No existing snapshots found - data freshness not guaranteed")

    # =========================================================================
    # GROUP 1: Data Fetching (PARALLEL)
    # These are independent API calls that write to different files
    # Note: With Phase 0, these may overlap with snapshots but provide latest data
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

    # Run data fetching in parallel
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
    # Freshness check - use --no-refresh to prevent interactive prompts
    # Data was already fetched in Group 1, this is just validation
    success, duration, error = run_sequential_step(
        "Check Data Freshness",
        [venv_python, "scripts/fetch/check_data_freshness.py", "--no-refresh"],
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

    # V32 HYBRID MODEL ROUTING (Dec 28, 2025)
    # Walk-forward validated: use best model for each market
    # - XGBoost: player_receptions (74% WR), player_reception_yds (68% WR)
    # - Edge: player_pass_attempts (55% WR), player_rush_yds (55% WR)
    # - TD Enhanced: anytime TD (RB @ 60% = 58% WR)
    # - DISABLED: player_pass_completions, player_rush_attempts (negative ROI)

    if use_edge_mode:
        # Legacy edge-only mode (not recommended)
        player_prop_step = (
            f"Generate Edge Recommendations + TD Props (Week {week})",
            [venv_python, "scripts/predict/generate_edge_recommendations.py", "--week", str(week), "--include-td"],
            True
        )
        edge_step = None
        td_enhanced_step = None
    else:
        # V32 HYBRID MODE (default) - Run both XGBoost and Edge for their respective markets
        # 1. XGBoost for receptions/reception_yds (outputs to CURRENT_WEEK_RECOMMENDATIONS.csv)
        player_prop_step = (
            f"Generate XGBoost Recommendations (Week {week})",
            [venv_python, "scripts/predict/generate_unified_recommendations_v3.py", "--week", str(week)],
            True
        )
        # 2. Edge for pass_attempts/rush_yds (outputs to edge_recommendations_weekX.csv)
        edge_step = (
            f"Generate Edge Recommendations (Week {week})",
            [venv_python, "scripts/predict/generate_edge_recommendations.py", "--week", str(week)],
            True
        )
        # 3. TD Enhanced for anytime TD (appends to edge_recommendations_weekX.csv)
        td_enhanced_step = (
            f"Generate TD Enhanced Recommendations (Week {week})",
            [venv_python, "scripts/predict/generate_edge_recommendations.py", "--week", str(week), "--include-td-enhanced"],
            False  # Not required - TD Enhanced is optional
        )

    recommendation_steps = [
        player_prop_step,
        (
            "Generate Game Line Recommendations",
            [venv_python, "scripts/predict/generate_game_line_recommendations.py"],
            True
        ),
    ]

    # Add Edge step for hybrid mode
    if edge_step:
        recommendation_steps.append(edge_step)

    # Add TD Enhanced step
    if td_enhanced_step:
        recommendation_steps.append(td_enhanced_step)

    # Run recommendations in parallel
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
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nStep Timings:")
    print(f"{'-'*60}")

    for desc, success, duration, error in all_results:
        status = "✅" if success else "⚠️ "
        print(f"  {status} {desc:<45} {duration:>6.1f}s")

    print(f"{'-'*60}")
    print(f"  {'TOTAL':<45} {total_duration:>6.1f}s")
    print(f"  {'(minutes)':<45} {total_duration/60:>6.1f}m")

    print(f"\n📊 Check reports/ directory for output files")
    print(f"📁 Latest recommendations: reports/recommendations_week{week}_*.csv")
    print(f"📱 Mobile dashboard: https://nfl-quant.vercel.app")


if __name__ == "__main__":
    main()
