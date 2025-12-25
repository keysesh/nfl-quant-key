#!/usr/bin/env python3
"""
Generate Model Predictions for All Players
Creates data/model_predictions_week{week}.csv with all player projections

This script:
1. Loads usage and efficiency predictors
2. Gets all players from odds data
3. Simulates each player to get projections
4. Outputs predictions in CSV format expected by generate_current_week_recommendations.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple progress display
    def tqdm(iterable, desc="", total=None):
        if total:
            print(f"{desc}... ({total} items)")
            for i, item in enumerate(iterable):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Progress: {i + 1}/{total} ({100 * (i + 1) / total:.1f}%)")
                yield item
        else:
            return iterable

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.simulation.player_simulator_v4 import PlayerSimulatorV4
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.v4_output import PlayerPropOutputV4
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.models.td_predictor import TouchdownPredictor, estimate_usage_factors
from nfl_quant.utils.unified_integration import integrate_all_factors
from nfl_quant.utils.season_utils import get_current_season, get_current_week
from nfl_quant.data.dynamic_parameters import get_parameter_provider
from nfl_quant.features.role_change_detector import load_role_overrides
from nfl_quant.utils.micro_metrics import MicroMetricsCalculator
from functools import lru_cache
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE OPTIMIZATION: Memoized name normalization and lookup caches
# ============================================================================

# Memoized normalize_player_name to avoid repeated regex operations
@lru_cache(maxsize=10000)
def _normalize_player_name_cached(name: str) -> str:
    """Cached version of normalize_player_name for O(1) repeated lookups."""
    from nfl_quant.utils.player_names import normalize_player_name
    return normalize_player_name(name)


def build_trailing_stats_lookup(trailing_stats: Dict) -> Dict[str, Dict]:
    """
    Pre-build a normalized_name -> stats dict for O(1) lookups.

    Converts O(n*m) nested loop lookups to O(1) dict access.
    Called ONCE at start, used throughout prediction generation.

    Returns:
        Dict mapping normalized player names to their stats
    """
    if not trailing_stats:
        return {}

    lookup = {}
    for key, stats in trailing_stats.items():
        # Extract name from key (format: "Player Name_week{week}")
        key_name = key.split('_week')[0]
        normalized = _normalize_player_name_cached(key_name)
        if normalized:
            lookup[normalized] = stats

    return lookup

# NFLverse parquet data directory (populated by R/nflreadr)
NFLVERSE_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'nflverse'

# Global cache for injury data (to avoid reloading for each player)
_injury_data_cache = {}
_injury_data_week = None

# Global cache for micro metrics (team-level features)
_micro_metrics_cache = None
_micro_metrics_week = None

# Parallel execution settings
# Using ProcessPoolExecutor for true parallelism (bypasses GIL)
PARALLEL_WORKERS = min(multiprocessing.cpu_count(), 8)
ENABLE_PARALLEL_SIMULATION = True  # Using ProcessPoolExecutor for ~8x speedup

# Process-local storage for worker simulators (each process gets its own)
_process_simulator = None
_process_worker_id = None


def _init_process_worker(seed_base: int):
    """
    Initialize process-local simulator when worker process starts.
    Called once per worker process via ProcessPoolExecutor initializer.

    Each process loads its own predictors and creates its own simulator
    with a unique seed to ensure reproducibility and avoid conflicts.
    """
    global _process_simulator, _process_worker_id

    # Get unique worker ID from PID
    _process_worker_id = os.getpid()
    worker_seed = seed_base + (_process_worker_id % 1000)  # Unique seed per worker

    # Load predictors in this process (each process needs its own)
    from nfl_quant.simulation.player_simulator import load_predictors
    from nfl_quant.simulation.player_simulator_v4 import PlayerSimulatorV4

    usage_pred, efficiency_pred = load_predictors()

    # PERF FIX (Dec 14, 2025): Use environment variable or config default (30k)
    # Reduced from 50k to 30k for 40% speedup with <0.5% accuracy loss
    n_simulations = int(os.environ.get('NFL_QUANT_SIMULATIONS', 30000))

    # Create simulator for this process
    _process_simulator = PlayerSimulatorV4(
        usage_predictor=usage_pred,
        efficiency_predictor=efficiency_pred,
        trials=n_simulations,
        seed=worker_seed
    )

    # Log initialization (will appear once per worker)
    import logging
    logging.getLogger(__name__).info(
        f"   Worker {_process_worker_id} initialized with seed {worker_seed}"
    )


def _simulate_player_for_process(args: tuple) -> tuple:
    """
    Process worker function for parallel simulation.
    Must be at module level for pickling.

    Args:
        args: Tuple of (player_input, team_game_context, player_name, team, position)

    Returns:
        Tuple of (result_dict, player_input, player_name, team, position, error_str)
    """
    global _process_simulator

    player_input, team_game_context, player_name, team, position = args

    try:
        if _process_simulator is None:
            return (None, player_input, player_name, team, position, "Simulator not initialized")

        result = _process_simulator.simulate_player(player_input, game_context=team_game_context)
        return (result, player_input, player_name, team, position, None)
    except Exception as e:
        return (None, player_input, player_name, team, position, str(e))


def _simulate_single_player_worker(args):
    """
    Worker function for parallel player simulation.

    Args:
        args: Tuple of (row_dict, simulator, trailing_stats, game_context, week, season, normalize_func, stats_lookup)
              stats_lookup is optional - if None, will use O(n) fallback

    Returns:
        Tuple of (prediction_dict, skip_reason) - prediction_dict is None if skipped
    """
    # Unpack args - stats_lookup is optional (for backwards compatibility)
    if len(args) == 8:
        row_dict, simulator, trailing_stats, game_context, week, season, normalize_func, stats_lookup = args
    else:
        row_dict, simulator, trailing_stats, game_context, week, season, normalize_func = args
        stats_lookup = None

    player_name = row_dict['player_name']
    team = row_dict['team']
    position = row_dict['position']

    # Basic validation
    if pd.isna(player_name) or pd.isna(position):
        return None, "Missing basic fields"

    # Check team validity
    if team and not pd.isna(team):
        team_str = str(team).upper()
        if team_str in ('UNK', 'NAN', 'NONE'):
            team = None

    # PERF FIX: Use O(1) lookup if stats_lookup provided, else O(n) fallback
    normalized_name = _normalize_player_name_cached(str(player_name))
    if stats_lookup:
        stats = stats_lookup.get(normalized_name, {})
    else:
        # Fallback to O(n) search if no lookup provided
        stats = None
        for key, value in trailing_stats.items():
            key_name = key.split('_week')[0]
            if _normalize_player_name_cached(key_name) == normalized_name:
                stats = value
                break
        if not stats:
            stats = {}

    # Check for activity
    lookback_weeks_played = stats.get('lookback_weeks_played', 0)
    games_played_in_lookback = stats.get('games_played_in_lookback', 0)
    avg_rec_yd = stats.get('avg_rec_yd_per_game', stats.get('avg_rec_yd', 0) or 0)
    avg_rush_yd = stats.get('avg_rush_yd_per_game', stats.get('avg_rush_yd', 0) or 0)
    avg_pass_yd = stats.get('avg_pass_yd_per_game', stats.get('avg_pass_yd', 0) or 0)

    has_activity = avg_rec_yd > 0 or avg_rush_yd > 0 or avg_pass_yd > 0
    if not has_activity and games_played_in_lookback < 2:
        return None, f"Insufficient data ({games_played_in_lookback} games in lookback)"

    # WR/TE specific check
    if position in ['WR', 'TE']:
        target_share = stats.get('trailing_target_share')
        if target_share is None and avg_rec_yd == 0:
            return None, "No target share data and no receiving yards"
        if target_share == 0.0 and avg_rec_yd == 0:
            return None, "trailing_target_share=0.0 with no receiving data"

    try:
        # Import here to avoid circular imports in worker
        from nfl_quant.schemas import PlayerPropInput

        # Get team game context FIRST to get correct opponent
        team_game_context = game_context.get(team, {}) if game_context and team else {}

        # Get opponent from game_context (correct) NOT from trailing_stats (stale/wrong)
        opponent = team_game_context.get('opponent', 'UNK') if team_game_context else stats.get('opponent', 'UNK')

        # Create player input (simplified - main function does more complex creation)
        player_input = PlayerPropInput(
            player_name=player_name,
            team=team or 'UNK',
            position=position,
            opponent=opponent,
            season=season,
            week=week,
            trailing_stats=stats
        )

        # Run simulation
        result = simulator.simulate_player(player_input, game_context=team_game_context)

        # Extract predictions
        pred = {
            'player_name': player_name,
            'player_dk': player_name,
            'player_pbp': player_name,
            'team': team,
            'position': position,
            'week': week,
            'opponent': player_input.opponent,
        }

        # Extract means for each stat type
        stat_mappings = [
            ('passing_yards', 'passing_yards_mean', 'passing_yards_std'),
            ('passing_completions', 'passing_completions_mean', 'passing_completions_std'),
            ('passing_attempts', 'passing_attempts_mean', 'passing_attempts_std'),
            ('passing_tds', 'passing_tds_mean', 'passing_tds_std'),
            ('rushing_yards', 'rushing_yards_mean', 'rushing_yards_std'),
            ('rushing_tds', 'rushing_tds_mean', 'rushing_tds_std'),
            ('receiving_yards', 'receiving_yards_mean', 'receiving_yards_std'),
            ('receptions', 'receptions_mean', 'receptions_std'),
            ('targets', 'targets_mean', 'targets_std'),
            ('receiving_tds', 'receiving_tds_mean', 'receiving_tds_std'),
        ]

        for src, mean_col, std_col in stat_mappings:
            if src in result:
                pred[mean_col] = float(np.mean(result[src]))
                pred[std_col] = float(np.std(result[src]))

        # Handle carries -> rushing_attempts
        if 'carries' in result:
            pred['rushing_attempts_mean'] = float(np.mean(result['carries']))
            pred['rushing_attempts_std'] = float(np.std(result['carries']))

        if 'anytime_td' in result:
            pred['anytime_td_prob'] = float(np.mean(result['anytime_td'] > 0))

        return pred, None

    except Exception as e:
        return None, f"Simulation error: {str(e)[:50]}"


def load_players_from_odds(odds_file: Path, trailing_stats: Dict = None, stats_lookup: Dict = None) -> pd.DataFrame:
    """Extract unique players from odds data.

    PERF OPTIMIZATION (Dec 14, 2025): Uses pre-built stats_lookup for O(1) lookups
    instead of O(n*m) nested loop. Speedup: ~100x for large player lists.
    """
    if not odds_file.exists():
        logger.warning(f"Odds file not found: {odds_file}")
        return pd.DataFrame()

    lookup_start = time.time()
    df = pd.read_csv(odds_file)

    # Get unique players - need to infer team from home_team/away_team
    players_list = []

    # Load trailing stats once if not provided (for team/position lookup)
    if trailing_stats is None:
        trailing_stats = load_trailing_stats()

    # PERF FIX: Build or use pre-built lookup dict for O(1) access
    if stats_lookup is None:
        stats_lookup = build_trailing_stats_lookup(trailing_stats)

    for _, row in df.iterrows():
        player_name = row.get('player_name')
        if pd.isna(player_name):
            continue

        # PERF FIX: Use cached normalize function
        normalized_name = _normalize_player_name_cached(str(player_name))
        if not normalized_name:  # Skip special cases (D/ST, etc.)
            continue

        # PERF FIX: O(1) dict lookup instead of O(n) loop through trailing_stats
        stats = stats_lookup.get(normalized_name)
        team = stats.get('team') if stats else None
        position = stats.get('position') if stats else None

        # Include player even if team is missing (they might have stats)
        if position:  # Only require position
            players_list.append({
                'player_name': player_name,
                'team': team if team and not pd.isna(team) else None,
                'position': position
            })

    logger.debug(f"  load_players_from_odds lookup: {time.time() - lookup_start:.2f}s")

    if not players_list:
        logger.warning("   ⚠️  Could not extract players with position info")
        # Fallback: just get unique player names
        unique_players = df['player_name'].dropna().unique()
        return pd.DataFrame({'player_name': unique_players})

    players_df = pd.DataFrame(players_list).drop_duplicates(subset=['player_name'])

    return players_df


def load_active_players_from_nflverse(week: int, season: int, trailing_stats: Dict = None) -> pd.DataFrame:
    """
    Load ALL active players from NFLverse data (not just those with betting odds).

    This is the CORRECT architecture: Generate predictions for ALL active players,
    then filter to players with odds during recommendation generation.

    Args:
        week: Current week number
        season: Current season year
        trailing_stats: Pre-loaded trailing stats dict (avoids reloading)

    Returns:
        DataFrame with columns: player_name, team, position
    """
    # Load trailing stats if not provided
    if trailing_stats is None:
        trailing_stats = load_trailing_stats(week, season)

    # Extract all unique players from trailing stats
    players_list = []

    for key, stats in trailing_stats.items():
        # Key format: "Player Name_week{week}"
        player_name = key.split('_week')[0]
        team = stats.get('team')
        position = stats.get('position')
        weeks_played = stats.get('weeks_played', 0)

        # Only include players who have played this season
        # (weeks_played > 0 means they have recent game data)
        if position and weeks_played > 0:
            players_list.append({
                'player_name': player_name,
                'team': team,
                'position': position
            })

    if not players_list:
        logger.warning("   ⚠️  No active players found in NFLverse trailing stats")
        return pd.DataFrame()

    # Convert to DataFrame and deduplicate
    players_df = pd.DataFrame(players_list).drop_duplicates(subset=['player_name'])

    # Filter to skill positions only (QB, RB, WR, TE)
    # Exclude special teams, defense, and inactive players
    skill_positions = ['QB', 'RB', 'WR', 'TE']
    players_df = players_df[players_df['position'].isin(skill_positions)].copy()

    return players_df


def load_trailing_stats(current_week: int = None, current_season: int = None, force_refresh: bool = False):
    """
    Load trailing stats from R-generated NFLverse parquet files.

    Reads parquet files in data/nflverse/ that were created by R/nflreadr.
    No legacy JSON files are used - all data comes from the canonical R/nflverse source.

    Args:
        current_week: Current week number (for trailing window calculation)
        current_season: Current season year (default: 2025)
        force_refresh: If True, bypass cache and recompute stats

    Returns:
        Dict mapping "Player Name_week{week}" -> stats dict
    """
    if current_season is None:
        current_season = get_current_season()
    if current_week is None:
        current_week = get_current_week()  # Auto-detect current week

    # PERF FIX (Dec 14, 2025): Add trailing stats cache for instant subsequent runs
    cache_dir = Path('data/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'trailing_stats_{current_season}_week{current_week}.parquet'

    if not force_refresh and cache_file.exists():
        try:
            import pickle
            # Check if cache is fresh (less than 6 hours old)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 6 * 3600:  # 6 hours
                logger.info(f"   💾 Loading cached trailing stats from {cache_file} (age: {cache_age/3600:.1f}h)")
                cache_df = pd.read_parquet(cache_file)
                # Convert DataFrame back to dict
                trailing_stats = {}
                for _, row in cache_df.iterrows():
                    key = row['_cache_key']
                    trailing_stats[key] = row.drop('_cache_key').to_dict()
                logger.info(f"   ✅ Loaded {len(trailing_stats)} cached player stats")
                return trailing_stats
            else:
                logger.info(f"   💾 Cache expired (age: {cache_age/3600:.1f}h > 6h), refreshing...")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load cache: {e}, recomputing...")

    trailing_stats = {}

    try:
        logger.info(f"   Loading NFLverse parquet data for {current_season} season...")

        # Load player stats from R-generated parquet files
        # Primary: weekly_stats.parquet (contains all seasons)
        weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_stats.parquet'

        if not weekly_stats_file.exists():
            # Fallback to season-specific file
            weekly_stats_file = NFLVERSE_DATA_DIR / f'weekly_{current_season}.parquet'

        if not weekly_stats_file.exists():
            raise FileNotFoundError(
                f"NFLverse parquet data not found. Expected:\n"
                f"  {NFLVERSE_DATA_DIR / 'weekly_stats.parquet'} or\n"
                f"  {NFLVERSE_DATA_DIR / f'weekly_{current_season}.parquet'}\n"
                f"Run your R script to fetch nflverse data first."
            )

        stats_df = pd.read_parquet(weekly_stats_file)
        logger.info(f"   Loaded {len(stats_df):,} records from {weekly_stats_file.name}")

        # Filter to weeks before current week for trailing stats
        current_season_stats = stats_df[
            (stats_df['season'] == current_season) &
            (stats_df['week'] < current_week) &
            (stats_df['week'] > 0)  # Exclude preseason
        ].copy()

        if len(current_season_stats) == 0:
            logger.warning(f"   ⚠️  No NFLverse data found for {current_season} weeks 1-{current_week-1}")
            # Try loading from local parquet cache
            project_root = Path(__file__).parent.parent.parent
            weekly_file = project_root / f'data/nflverse/weekly_{current_season}.parquet'
            if weekly_file.exists():
                logger.info(f"   Using cached NFLverse data: {weekly_file}")
                current_season_stats = pd.read_parquet(weekly_file)
                current_season_stats = current_season_stats[
                    (current_season_stats['week'] < current_week) &
                    (current_season_stats['week'] > 0)
                ].copy()

        if len(current_season_stats) == 0:
            raise ValueError(f"No NFLverse data available for {current_season}")

        # FIX: Pre-calculate slot_snap_pct for all players from PBP pass_location data
        # slot = middle location, outside = left/right location
        slot_snap_pct_cache = {}
        try:
            pbp_file = NFLVERSE_DATA_DIR / f'pbp_{current_season}.parquet'
            if pbp_file.exists():
                pbp_df = pd.read_parquet(pbp_file)
                # Filter to pass plays with a receiver, before current week
                pass_plays = pbp_df[
                    (pbp_df['play_type'] == 'pass') &
                    (pbp_df['receiver_player_name'].notna()) &
                    (pbp_df['week'] < current_week)
                ]
                if len(pass_plays) > 0:
                    # Calculate slot percentage per receiver
                    # middle = slot alignment, left/right = outside alignment
                    player_locations = pass_plays.groupby('receiver_player_name').agg({
                        'pass_location': lambda x: (x == 'middle').sum(),  # middle targets
                        'play_id': 'count'  # total targets
                    })
                    for player_name, row in player_locations.iterrows():
                        if row['play_id'] >= 5:  # Minimum 5 targets for reliable estimate
                            slot_snap_pct_cache[player_name] = row['pass_location'] / row['play_id']
                    logger.info(f"   ✅ Calculated slot_snap_pct for {len(slot_snap_pct_cache)} receivers from PBP data")
        except Exception as e:
            logger.debug(f"Could not calculate slot_snap_pct from PBP: {e}")

        # Group by player and calculate trailing averages
        # BUG FIX #7 (Nov 23, 2025): Don't group by team - use most recent team instead
        # Issue: Players who changed teams mid-season (e.g., John Metchie III PHI->NYJ)
        #        created multiple groups, and system used OLD team with MORE data points
        # Solution: Group by player only, then get most recent team from their latest week
        player_groups = current_season_stats.groupby(['player_display_name', 'position'])

        # PERF FIX (Dec 14, 2025): Load team_pace.parquet ONCE before loop (was reading 500+ times!)
        # BUG FIX (Dec 24, 2025): Use plays_per_game directly, NOT 3600/plays_per_game!
        # The usage predictor was trained with team_pace = plays_per_game (~63),
        # but prediction was converting to seconds_per_play (~41). This caused
        # QB pass attempts to be severely underestimated (Dak: 25.6 vs actual 35.8)
        team_pace_lookup = {}
        try:
            team_pace_df = pd.read_parquet('data/nflverse/team_pace.parquet')
            for _, row in team_pace_df.iterrows():
                plays_per_game = row.get('plays_per_game', 0)
                if plays_per_game > 0:
                    # FIX: Use plays_per_game directly to match training data
                    team_pace_lookup[row['team']] = plays_per_game
            logger.info(f"   ✅ Loaded team pace for {len(team_pace_lookup)} teams")
        except Exception as e:
            logger.debug(f"Could not load team pace: {e}")

        # PERF FIX (Dec 14, 2025): Pre-compute ALL defensive EPA values ONCE
        # This replaces 14,000+ PBP parquet reads with 1 read + O(1) lookups
        defensive_epa_lookup = {}
        try:
            import time as _time
            _epa_start = _time.time()
            pbp_file = NFLVERSE_DATA_DIR / f'pbp_{current_season}.parquet'
            if pbp_file.exists():
                pbp_df = pd.read_parquet(pbp_file)
                # Filter to plays with EPA before current week
                valid_plays = pbp_df[
                    (pbp_df['week'] < current_week) &
                    (pbp_df['epa'].notna())
                ].copy()

                if len(valid_plays) > 0:
                    # Calculate defensive EPA by team and week (defense = defteam)
                    # Pass defense: plays where pass == 1
                    # Rush defense: plays where rush == 1
                    for def_team in valid_plays['defteam'].dropna().unique():
                        team_def_plays = valid_plays[valid_plays['defteam'] == def_team]

                        for game_week in range(1, current_week):
                            week_plays = team_def_plays[team_def_plays['week'] <= game_week]

                            # Pass defense EPA (negative = good defense)
                            pass_plays = week_plays[week_plays['pass'] == 1]
                            if len(pass_plays) >= 10:
                                pass_def_epa = pass_plays['epa'].mean()
                                # Store for WR/TE/QB positions
                                defensive_epa_lookup[(def_team, 'WR', game_week)] = pass_def_epa
                                defensive_epa_lookup[(def_team, 'TE', game_week)] = pass_def_epa
                                defensive_epa_lookup[(def_team, 'QB', game_week)] = pass_def_epa

                            # Rush defense EPA (negative = good defense)
                            rush_plays = week_plays[week_plays['rush'] == 1]
                            if len(rush_plays) >= 10:
                                rush_def_epa = rush_plays['epa'].mean()
                                defensive_epa_lookup[(def_team, 'RB', game_week)] = rush_def_epa

                    logger.info(f"   ✅ Built defensive EPA lookup ({len(defensive_epa_lookup)} entries) in {_time.time() - _epa_start:.1f}s")
            else:
                logger.warning(f"   ⚠️  PBP file not found: {pbp_file}")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not build defensive EPA lookup: {e}")

        # PERF FIX (Dec 14, 2025): Add progress logging with ETA
        import time as _time
        total_players = len(player_groups)
        player_count = 0
        loop_start_time = _time.time()

        for (player_name, position), player_df in player_groups:
            player_count += 1

            # Log every 50 players with ETA (improved from every 100)
            if player_count % 50 == 0 or player_count == total_players:
                elapsed = _time.time() - loop_start_time
                rate = player_count / elapsed if elapsed > 0 else 0
                remaining = (total_players - player_count) / rate if rate > 0 else 0
                logger.info(f"   📊 Trailing stats: {player_count}/{total_players} players ({rate:.1f}/sec, ETA: {remaining:.0f}s)")

            weeks_played = len(player_df)

            if weeks_played == 0:
                continue

            # Get most recent team (from latest week played)
            team = player_df.sort_values('week', ascending=False).iloc[0]['team']

            # VALIDATION (Bug #7): Detect mid-season team changes
            unique_teams = player_df['team'].unique()
            if len(unique_teams) > 1:
                old_teams = [t for t in unique_teams if t != team]
                logger.info(f"   📋 {player_name} changed teams this season: {', '.join(old_teams)} → {team}")
                logger.info(f"      Using most recent team ({team}) for Week {current_week} projection")

            # Calculate trailing stats
            # FIX CRITICAL-2: Use EWMA (Exponential Weighted Moving Average) to match training
            # Training uses: .ewm(span=4, min_periods=1).mean()
            # This gives more weight to recent weeks (N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%)

            # Sort player_df by week to ensure proper EWMA calculation
            player_df_sorted = player_df.sort_values('week')

            # EWMA trailing averages (matching training computation)
            avg_targets = player_df_sorted['targets'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'targets' in player_df.columns else 0
            avg_receptions = player_df_sorted['receptions'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'receptions' in player_df.columns else 0
            avg_rec_yards = player_df_sorted['receiving_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'receiving_yards' in player_df.columns else 0
            avg_carries = player_df_sorted['carries'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'carries' in player_df.columns else 0
            avg_rush_yards = player_df_sorted['rushing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'rushing_yards' in player_df.columns else 0
            avg_pass_yards = player_df_sorted['passing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'passing_yards' in player_df.columns else 0
            # FIX: Add avg_pass_att for QBs (was missing, causing 20 attempt default)
            avg_pass_att = 0
            if position == 'QB':
                if 'attempts' in player_df.columns:
                    avg_pass_att = player_df_sorted['attempts'].ewm(span=4, min_periods=1).mean().iloc[-1]
                elif 'passing_attempts' in player_df.columns:
                    avg_pass_att = player_df_sorted['passing_attempts'].ewm(span=4, min_periods=1).mean().iloc[-1]

            # FIX: Add archetype features for dynamic correlation (WR/TE)
            # adot = Average Depth of Target (from receiving_air_yards / targets)
            adot = None
            if position in ['WR', 'TE', 'RB'] and 'receiving_air_yards' in player_df.columns and 'targets' in player_df.columns:
                total_air_yards = player_df['receiving_air_yards'].sum()
                total_targets_for_adot = player_df['targets'].sum()
                if total_targets_for_adot > 0:
                    adot = total_air_yards / total_targets_for_adot

            # ypt_variance = Variance of yards per target (volatility measure)
            ypt_variance = None
            if position in ['WR', 'TE', 'RB'] and 'receiving_yards' in player_df.columns and 'targets' in player_df.columns:
                # Calculate Y/T for each week
                weekly_ypt = player_df.apply(
                    lambda row: row['receiving_yards'] / row['targets'] if row['targets'] > 0 else None,
                    axis=1
                ).dropna()
                if len(weekly_ypt) >= 2:
                    ypt_variance = weekly_ypt.var()

            # FIX: Get team pace from pre-loaded lookup (PERF: was reading parquet in loop!)
            team_pace = team_pace_lookup.get(team)

            # PERF FIX (Dec 14, 2025): RE-ENABLED with O(1) lookups from pre-computed cache
            # This was calling get_defensive_epa_for_player() in nested loops, reading PBP file
            # thousands of times. Now uses defensive_epa_lookup built ONCE before the loop.
            trailing_opp_pass_def_epa = None
            trailing_opp_rush_def_epa = None

            # PERF FIX: Vectorized defense EPA calculation (replaces iterrows)
            if 'opponent_team' in player_df.columns and defensive_epa_lookup:
                # Extract opponent teams and weeks as arrays (vectorized access)
                opps = player_df_sorted['opponent_team'].values
                weeks = player_df_sorted['week'].values.astype(int)

                # Vectorized lookup using list comprehension (faster than iterrows)
                opp_pass_epas = [
                    defensive_epa_lookup.get((opp, 'WR', wk))
                    for opp, wk in zip(opps, weeks)
                    if pd.notna(opp) and (opp, 'WR', wk) in defensive_epa_lookup
                ]
                opp_rush_epas = [
                    defensive_epa_lookup.get((opp, 'RB', wk))
                    for opp, wk in zip(opps, weeks)
                    if pd.notna(opp) and (opp, 'RB', wk) in defensive_epa_lookup
                ]

                # Calculate EWMA of opponent defense EPAs (more recent opponents weighted more)
                if opp_pass_epas:
                    trailing_opp_pass_def_epa = pd.Series(opp_pass_epas).ewm(span=4, min_periods=1).mean().iloc[-1]
                if opp_rush_epas:
                    trailing_opp_rush_def_epa = pd.Series(opp_rush_epas).ewm(span=4, min_periods=1).mean().iloc[-1]

            # Calculate TD rate
            total_tds = 0
            if 'rushing_tds' in player_df.columns:
                total_tds += player_df['rushing_tds'].sum()
            if 'receiving_tds' in player_df.columns:
                total_tds += player_df['receiving_tds'].sum()
            if 'passing_tds' in player_df.columns:
                total_tds += player_df['passing_tds'].sum()

            total_opportunities = 0
            if 'carries' in player_df.columns:
                total_opportunities += player_df['carries'].sum()
            if 'targets' in player_df.columns:
                total_opportunities += player_df['targets'].sum()
            # FIX: R parquet uses 'attempts' not 'passing_attempts' for QB pass attempts
            if position == 'QB':
                if 'attempts' in player_df.columns:
                    total_opportunities += player_df['attempts'].sum()
                elif 'passing_attempts' in player_df.columns:
                    total_opportunities += player_df['passing_attempts'].sum()

            td_rate = total_tds / total_opportunities if total_opportunities > 0 else 0.0

            # Calculate yards per opportunity
            total_yards = 0
            if 'rushing_yards' in player_df.columns:
                total_yards += player_df['rushing_yards'].sum()
            if 'receiving_yards' in player_df.columns:
                total_yards += player_df['receiving_yards'].sum()

            yards_per_opp = total_yards / total_opportunities if total_opportunities > 0 else 0.0

            # QB-specific efficiency metrics (for trained model)
            trailing_comp_pct = None
            trailing_yards_per_completion = None
            trailing_td_rate_pass = None
            trailing_yards_per_carry = None
            trailing_td_rate_rush = None

            if position == 'QB':
                # QB passing efficiency
                if 'attempts' in player_df.columns and 'completions' in player_df.columns:
                    total_attempts = player_df['attempts'].sum()
                    total_completions = player_df['completions'].sum()
                    trailing_comp_pct = total_completions / total_attempts if total_attempts > 0 else None

                    if 'passing_yards' in player_df.columns and total_completions > 0:
                        trailing_yards_per_completion = player_df['passing_yards'].sum() / total_completions

                    if 'passing_tds' in player_df.columns and total_attempts > 0:
                        trailing_td_rate_pass = player_df['passing_tds'].sum() / total_attempts

                # QB rushing efficiency
                if 'carries' in player_df.columns:
                    total_qb_carries = player_df['carries'].sum()
                    if total_qb_carries > 0:
                        if 'rushing_yards' in player_df.columns:
                            trailing_yards_per_carry = player_df['rushing_yards'].sum() / total_qb_carries
                        if 'rushing_tds' in player_df.columns:
                            trailing_td_rate_rush = player_df['rushing_tds'].sum() / total_qb_carries

            # POSITION-SPECIFIC receiving efficiency for RB/WR/TE (FIX for Bug #2 and #3)
            trailing_yards_per_target = None
            if position in ['RB', 'WR', 'TE']:
                # Calculate yards per target (receiving efficiency)
                if 'targets' in player_df.columns and 'receiving_yards' in player_df.columns:
                    total_targets = player_df['targets'].sum()
                    total_rec_yards = player_df['receiving_yards'].sum()
                    trailing_yards_per_target = total_rec_yards / total_targets if total_targets > 0 else None

                # Also calculate position-specific TD rates for receiving
                if trailing_td_rate_pass is None:  # Not already set by QB logic
                    if 'receiving_tds' in player_df.columns and 'targets' in player_df.columns:
                        total_targets = player_df['targets'].sum()
                        trailing_td_rate_pass = player_df['receiving_tds'].sum() / total_targets if total_targets > 0 else None

            # POSITION-SPECIFIC rushing efficiency for RB (if not already set by QB logic)
            if position == 'RB' and trailing_yards_per_carry is None:
                if 'carries' in player_df.columns and 'rushing_yards' in player_df.columns:
                    total_rb_carries = player_df['carries'].sum()
                    total_rb_rush_yards = player_df['rushing_yards'].sum()
                    if total_rb_carries > 0:
                        trailing_yards_per_carry = total_rb_rush_yards / total_rb_carries
                        # FIX (Dec 14): Guard division to prevent RuntimeWarning
                        if 'rushing_tds' in player_df.columns:
                            trailing_td_rate_rush = player_df['rushing_tds'].sum() / total_rb_carries

            # FIX: Use actual per-game averages as the share (since simulator multiplies by team attempts)
            # CRITICAL: Use dynamic parameters from NFLverse data, not hardcoded values
            param_provider = get_parameter_provider()

            # Get actual team-specific pass/rush attempts from NFLverse data
            team_avg_pass_attempts = param_provider.get_team_pass_attempts(team, up_to_week=current_week)
            team_avg_rush_attempts = param_provider.get_team_rush_attempts(team, up_to_week=current_week)

            # CORRECTED: target_share = avg_targets / team_avg_pass_attempts
            # So when simulator does: mean_targets = target_share * team_pass_attempts
            # And team_pass_attempts ~= team_avg_pass_attempts, we get ~avg_targets
            # FIX (Dec 14): Guard division to prevent RuntimeWarning
            target_share = avg_targets / team_avg_pass_attempts if position in ['WR', 'TE', 'RB'] and team_avg_pass_attempts > 0 else 0.0
            # FIX: QBs also need carry_share for QB rushing predictions (Lamar, Allen, etc.)
            carry_share = avg_carries / team_avg_rush_attempts if position in ['RB', 'QB'] and team_avg_rush_attempts > 0 else 0.0

            # FIX: Look up slot_snap_pct from pre-calculated cache (root problem fix)
            slot_snap_pct = slot_snap_pct_cache.get(player_name)

            # === EXTRACT LAST 4 GAMES FOR CHART VISUALIZATION ===
            # Get the most recent 4 games (sorted by week descending)
            last_4_games = player_df_sorted.tail(4).sort_values('week', ascending=False)

            # Build game history arrays for each stat type
            game_history = {
                'weeks': last_4_games['week'].tolist(),
                'opponents': last_4_games['opponent_team'].tolist() if 'opponent_team' in last_4_games.columns else [],
                'receiving_yards': last_4_games['receiving_yards'].fillna(0).tolist() if 'receiving_yards' in last_4_games.columns else [],
                'receptions': last_4_games['receptions'].fillna(0).tolist() if 'receptions' in last_4_games.columns else [],
                'rushing_yards': last_4_games['rushing_yards'].fillna(0).tolist() if 'rushing_yards' in last_4_games.columns else [],
                'carries': last_4_games['carries'].fillna(0).tolist() if 'carries' in last_4_games.columns else [],
                'passing_yards': last_4_games['passing_yards'].fillna(0).tolist() if 'passing_yards' in last_4_games.columns else [],
                'pass_attempts': last_4_games['attempts'].fillna(0).tolist() if 'attempts' in last_4_games.columns else [],
                'targets': last_4_games['targets'].fillna(0).tolist() if 'targets' in last_4_games.columns else [],
            }

            # Get actual snap share from PBP data (season-long opportunity share, not per-game %)
            # This uses the same calculation as integrate_all_factors to ensure consistency
            from nfl_quant.utils.unified_integration import calculate_snap_share_from_data
            snap_share = calculate_snap_share_from_data(
                player_name=player_name,
                position=position,
                team=team,
                week=current_week,
                season=current_season,
                lookback_weeks=None  # Use all available historical data
            )
            # If PBP calculation returns None, use 0.0 instead of fallback estimation
            if snap_share is None:
                snap_share = 0.0

            key = f"{player_name}_week{current_week}"
            trailing_stats[key] = {
                'trailing_snap_share': snap_share,
                'trailing_target_share': target_share,
                'trailing_carry_share': carry_share,
                'trailing_yards_per_opportunity': yards_per_opp,
                'trailing_td_rate': td_rate,
                # QB-specific efficiency metrics for trained model
                'trailing_comp_pct': trailing_comp_pct,
                'trailing_yards_per_completion': trailing_yards_per_completion,
                'trailing_td_rate_pass': trailing_td_rate_pass,
                'trailing_yards_per_carry': trailing_yards_per_carry,
                'trailing_td_rate_rush': trailing_td_rate_rush,
                'trailing_yards_per_target': trailing_yards_per_target,  # FIX: Bug #4 - Add position-specific receiving efficiency
                'team': team,
                'position': position,
                'weeks_played': weeks_played,
                'lookback_weeks_played': weeks_played,
                'games_played_in_lookback': weeks_played,
                'lookback_weeks_used': list(player_df['week'].unique()),
                'lookback_window_complete': weeks_played >= 4,
                'lookback_window_complete_games': weeks_played >= 4,
                'team_changed_in_lookback': False,
                'position_changed_in_lookback': False,
                'avg_rec_yd': avg_rec_yards,
                'avg_rush_yd': avg_rush_yards,
                'avg_pass_yd': avg_pass_yards,
                'avg_rec_yd_per_game': avg_rec_yards,
                'avg_rush_yd_per_game': avg_rush_yards,
                'avg_pass_yd_per_game': avg_pass_yards,
                'avg_targets_per_game': avg_targets,
                'avg_receptions_per_game': avg_receptions,
                'avg_carries_per_game': avg_carries,
                'avg_pass_att': avg_pass_att,  # FIX: Add QB pass attempts
                'avg_pass_att_per_game': avg_pass_att,
                # FIX: Add missing fields expected by simulator (same pattern as avg_pass_att fix)
                'avg_rec_tgt': avg_targets,  # FIX: Key mismatch - simulator expects avg_rec_tgt not avg_targets_per_game
                'trailing_targets': avg_targets,  # FIX: Direct EWMA targets for simulator (WR/TE/RB)
                'trailing_carries': avg_carries,  # FIX: Direct EWMA carries for simulator (RB)
                'trailing_catch_rate': avg_receptions / avg_targets if avg_targets > 0 else None,  # FIX: Player catch rate
                # FIX: Add archetype features for dynamic correlation
                'adot': adot,  # Average depth of target (WR/TE/RB)
                'ypt_variance': ypt_variance,  # Yards per target variance (volatility)
                # FIX: Add trailing opponent defense EPA (schedule strength)
                'trailing_opp_pass_def_epa': trailing_opp_pass_def_epa,
                'trailing_opp_rush_def_epa': trailing_opp_rush_def_epa,
                # FIX: Add team pace (plays per game, ~55-70)
                'team_pace': team_pace,
                # FIX: Add slot_snap_pct from PBP data (root problem fix)
                'slot_snap_pct': slot_snap_pct,
                # Game history for chart visualization (last 4 games)
                'game_history': game_history,
                'seasons': [current_season],
                'data_sources': ['nflverse'],
            }

        logger.info(f"   ✅ Loaded {len(trailing_stats)} players from NFLverse ({current_season} season)")
        logger.info(f"      Weeks included: 1-{current_week-1}")
        logger.info(f"      Data source: nflverse (canonical)")

    except Exception as e:
        logger.error(f"   ❌ Failed to load NFLverse parquet data: {e}")
        logger.error(f"   Ensure R/nflreadr has populated data/nflverse/ directory")
        logger.error(f"   Run your R script to fetch nflverse data first")
        raise

    if not trailing_stats:
        raise ValueError(
            f"No player stats found in NFLverse for {current_season} season. "
            f"Run scripts/fetch/fetch_nflverse_data.R to populate data/nflverse/"
        )

    # PERF FIX (Dec 14, 2025): Save trailing stats to cache for instant subsequent runs
    try:
        # Convert dict to DataFrame for efficient parquet storage
        cache_records = []
        for key, stats in trailing_stats.items():
            record = {'_cache_key': key}
            record.update(stats)
            cache_records.append(record)
        cache_df = pd.DataFrame(cache_records)
        cache_df.to_parquet(cache_file, index=False)
        logger.info(f"   💾 Cached {len(trailing_stats)} player stats to {cache_file}")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not save cache: {e}")

    return trailing_stats


def create_player_input_from_odds(
    player_name: str,
    team: str,
    position: str,
    week: int,
    trailing_stats: Dict,
    game_context: Dict = None,
    season: int = None
) -> PlayerPropInput:
    """Create PlayerPropInput from odds data and trailing stats."""
    if season is None:
        season = get_current_season()

    # Import name normalization
    from nfl_quant.utils.player_names import normalize_player_name

    # Normalize player name for matching
    normalized_name = normalize_player_name(str(player_name))

    # Get trailing stats for this player - try normalized matching
    hist = {}

    # Try exact match first (with week suffix)
    key = f"{player_name}_week{week}"
    hist = trailing_stats.get(key, {})

    # If not found, try normalized matching
    if not hist:
        for key, stats in trailing_stats.items():
            # Extract name from key (format: "Player Name_week{week}")
            key_name = key.split('_week')[0]
            key_week = None
            if '_week' in key:
                try:
                    key_week = int(key.split('_week')[1])
                except:
                    pass

            # Match by normalized name
            if normalize_player_name(key_name) == normalized_name:
                # Prefer current week, but accept any week
                if key_week == week:
                    hist = stats
                    break
                elif not hist:  # Use first match if no exact week match
                    hist = stats

    # If exact week not found, try most recent week
    if not hist:
        available_weeks = []
        for k in trailing_stats.keys():
            key_name = k.split('_week')[0]
            if normalize_player_name(key_name) == normalized_name:
                try:
                    week_num = int(k.split('_week')[1])
                    available_weeks.append((week_num, k))
                except:
                    pass

        if available_weeks:
            most_recent_week, most_recent_key = max(available_weeks, key=lambda x: x[0])
            hist = trailing_stats.get(most_recent_key, {})
            logger.debug(f"   Using week {most_recent_week} stats for {player_name} (requested week {week})")

    # No legacy JSON fallback - if player not in NFLverse data, they shouldn't be predicted
    if not hist:
        logger.warning(f"   ⚠️  No NFLverse stats found for {player_name}")

    # Get game context - handle case where team might be None
    # If team is None, try to infer from stats
    if not team or pd.isna(team):
        # Try to get team from stats
        if hist and hist.get('team') and not pd.isna(hist.get('team')):
            team = hist.get('team')
            logger.debug(f"   Inferred team {team} from stats for {player_name}")
        else:
            # No team available - can't get game context
            raise ValueError(
                f"Team not available for {player_name} and cannot infer from stats. "
                f"Game context requires team assignment."
            )

    if not game_context or team not in game_context:
        raise ValueError(
            f"Game context not available for team {team} in week {week}. "
            f"Run game simulations BEFORE generating player predictions. "
            f"Command: python scripts/simulate/run_game_simulations.py --week {week}"
        )

    ctx = game_context[team]

    # FAIL EXPLICITLY if required game context values missing
    projected_team_total = ctx.get('team_total') or ctx.get('projected_total')
    projected_opponent_total = ctx.get('opponent_total')
    pace = ctx.get('pace')
    opponent = ctx.get('opponent')
    projected_team_pass_attempts = ctx.get('projected_team_pass_attempts')
    projected_team_rush_attempts = ctx.get('projected_team_rush_attempts')
    projected_team_targets = ctx.get('projected_team_targets')

    if projected_team_total is None:
        raise ValueError(f"Game context missing 'team_total' for {team} in week {week}")
    if projected_opponent_total is None:
        raise ValueError(f"Game context missing 'opponent_total' for {team} in week {week}")
    if pace is None:
        raise ValueError(f"Game context missing 'pace' for {team} in week {week}")
    if not opponent or opponent == 'UNK':
        raise ValueError(f"Game context missing 'opponent' for {team} in week {week}")
    if projected_team_pass_attempts is None:
        raise ValueError(f"Game context missing 'projected_team_pass_attempts' for {player_name} ({team}) in week {week}. Run game simulations to populate team usage data.")
    if projected_team_rush_attempts is None:
        raise ValueError(f"Game context missing 'projected_team_rush_attempts' for {player_name} ({team}) in week {week}. Run game simulations to populate team usage data.")
    if projected_team_targets is None:
        raise ValueError(f"Game context missing 'projected_team_targets' for {player_name} ({team}) in week {week}. Run game simulations to populate team usage data.")

    projected_game_script = ctx.get('game_script', projected_team_total - projected_opponent_total)

    # Try to load from simulation files if game_context not provided (should not happen)
    # This is a safety check - game_context should always be provided
    if not game_context or team not in game_context:
        import glob
        import json
        from pathlib import Path

        sim_files = sorted(glob.glob(f'reports/sim_{season}_{week:02d}_*.json'))
        found_context = False

        for sim_file in sim_files:
            try:
                with open(sim_file) as f:
                    sim_data = json.load(f)

                # Check if this simulation is for our team
                if sim_data.get('home_team') == team or sim_data.get('away_team') == team:
                    home_team = sim_data.get('home_team')
                    away_team = sim_data.get('away_team')

                    if team == home_team:
                        projected_team_total = sim_data.get('home_score_median')
                        projected_opponent_total = sim_data.get('away_score_median')
                        opponent = away_team
                    else:
                        projected_team_total = sim_data.get('away_score_median')
                        projected_opponent_total = sim_data.get('home_score_median')
                        opponent = home_team

                    # FAIL EXPLICITLY if required scores missing
                    if projected_team_total is None or projected_opponent_total is None:
                        raise ValueError(
                            f"Simulation file {sim_file} missing required scores for {team}. "
                            f"Expected 'home_score_median' and 'away_score_median'. "
                            f"Run game simulations to populate this data."
                        )

                    projected_game_script = projected_team_total - projected_opponent_total

                    # Try to get pace from simulation - FAIL if not available
                    pace = sim_data.get('pace')
                    if pace is None:
                        # Try to estimate from plays per game if available
                        if 'plays_per_game' in sim_data:
                            pace = sim_data['plays_per_game'] / 2.0  # Rough estimate: ~29 seconds per play
                        else:
                            # Calculate from team pace metrics if available
                            try:
                                from nfl_quant.features.team_metrics import TeamMetricsExtractor
                                extractor = TeamMetricsExtractor()
                                home_pace = extractor.get_team_pace(home_team, week)
                                away_pace = extractor.get_team_pace(away_team, week)
                                if home_pace.get('neutral_plays_per_game') and away_pace.get('neutral_plays_per_game'):
                                    avg_plays = (home_pace['neutral_plays_per_game'] + away_pace['neutral_plays_per_game']) / 2.0
                                    if avg_plays > 0:
                                        pace = (60 * 60) / avg_plays
                                    else:
                                        raise ValueError(f"Invalid plays_per_game for {home_team} vs {away_team}")
                                else:
                                    raise ValueError(
                                        f"Pace data not available for {home_team} vs {away_team}. "
                                        f"Expected 'pace' in simulation or team pace metrics. "
                                        f"Run game simulations or team metrics extraction."
                                    )
                            except Exception as e:
                                raise ValueError(
                                    f"Could not calculate pace for {home_team} vs {away_team}: {e}. "
                                    f"Run game simulations to populate pace data."
                                )
                    found_context = True
                    break
            except Exception as e:
                logger.debug(f"Error loading simulation file: {e}")
                continue

        if not found_context:
            raise ValueError(
                f"Game context not found for team {team} in week {week}. "
                f"Run game simulations BEFORE generating player predictions. "
                f"Command: python scripts/simulate/run_game_simulations.py --week {week}"
            )

    # Extract trailing stats from calculated historical data (no hardcoded defaults)
    # If stats don't exist, they remain None or 0.0 - this is intentional
    trailing_snap_share = hist.get("trailing_snap_share")  # May be None
    trailing_target_share = hist.get("trailing_target_share")  # May be None or 0.0
    trailing_carry_share = hist.get("trailing_carry_share")  # May be None or 0.0
    trailing_yards_per_opportunity = hist.get("trailing_yards_per_opportunity")  # May be None or 0.0
    trailing_td_rate = hist.get("trailing_td_rate")  # May be None or 0.0

    # QB-specific efficiency metrics (for trained model)
    trailing_comp_pct = hist.get("trailing_comp_pct")  # QB only
    trailing_yards_per_completion = hist.get("trailing_yards_per_completion")  # QB only
    trailing_td_rate_pass = hist.get("trailing_td_rate_pass")  # QB/WR/TE/RB
    trailing_yards_per_carry = hist.get("trailing_yards_per_carry")  # QB/RB
    trailing_td_rate_rush = hist.get("trailing_td_rate_rush")  # QB/RB
    trailing_yards_per_target = hist.get("trailing_yards_per_target")  # FIX: Bug #4 - WR/TE/RB receiving efficiency

    # DEBUG: Log QB rushing stats to diagnose 0.0 rushing TD issue
    if position == 'QB' and trailing_td_rate_rush is not None:
        logger.info(f"   QB RUSHING DEBUG {player_name}: ypc={trailing_yards_per_carry:.4f}, td_rate={trailing_td_rate_rush:.4f}")

    # Extract actual historical averages (for use when trailing_target_share is 0.0)
    avg_rec_yd = hist.get("avg_rec_yd")  # Actual average receiving yards per game
    avg_rec_tgt = hist.get("avg_rec_tgt")  # Actual average targets per game (if available)
    avg_rush_yd = hist.get("avg_rush_yd")  # Actual average rushing yards per game
    avg_pass_att = hist.get("avg_pass_att")  # FIX: QB pass attempts per game

    # FIX: Extract direct EWMA usage stats for simulator (same pattern as avg_pass_att fix)
    trailing_targets = hist.get("trailing_targets")  # EWMA targets per game (WR/TE/RB)
    trailing_carries = hist.get("trailing_carries")  # EWMA carries per game (RB)
    trailing_catch_rate = hist.get("trailing_catch_rate")  # Player catch rate (receptions/targets)

    # FIX: Extract archetype features for dynamic correlation
    adot = hist.get("adot")  # Average depth of target (WR/TE/RB)
    ypt_variance = hist.get("ypt_variance")  # Yards per target variance (volatility)

    # FIX: Extract trailing opponent defense EPA (schedule strength)
    trailing_opp_pass_def_epa = hist.get("trailing_opp_pass_def_epa")
    trailing_opp_rush_def_epa = hist.get("trailing_opp_rush_def_epa")

    # FIX: Extract team pace
    team_pace = hist.get("team_pace")

    # FIX: Extract slot_snap_pct from PBP data (root problem fix)
    slot_snap_pct = hist.get("slot_snap_pct")

    # Note: We allow None/0.0 values when no historical data exists
    # The simulator should handle these cases appropriately

    # === APPLY INJURY ADJUSTMENTS ===
    # Load injury data and apply target/carry redistribution based on teammate injuries
    try:
        from nfl_quant.data.matchup_extractor import ContextBuilder
        # pd is already imported at module level, don't reimport

        # Use cached injury data if available (loaded in generate_model_predictions)
        global _injury_data_cache, _injury_data_week
        if '_injury_data_cache' in globals() and _injury_data_week == week:
            injury_data_by_team = _injury_data_cache
        else:
            # Fallback: load injury data if not cached
            from nfl_quant.utils.contextual_integration import load_injury_data
            injury_data_by_team = load_injury_data(week)

        team_injury_data = injury_data_by_team.get(team, {})

        if team_injury_data:
            # Create a context builder for injury adjustments
            # We only need the build_situational_adjustments method
            context_builder = ContextBuilder()  # No data needed for situational adjustments

            # Get injury-based multipliers for this player
            injury_multipliers = context_builder.build_situational_adjustments(
                team=team,
                week=week,
                player_name=player_name,
                position=position,
                injury_data=team_injury_data
            )

            # BUG FIX (Nov 23, 2025): Check if THIS player is OUT before doing ANY calculations
            # If target_boost=0.0 or carry_boost=0.0, the player is marked OUT - skip them entirely
            target_boost = injury_multipliers.get('target_share_multiplier', 1.0)
            carry_boost = injury_multipliers.get('carry_share_multiplier', 1.0)

            if target_boost == 0.0 or carry_boost == 0.0:
                logger.warning(f"   ⚠️  {player_name} is OUT due to injury - skipping predictions")
                raise ValueError(f"{player_name} is OUT due to injury (multiplier=0.0)")

            # Try to use HISTORICAL data instead of generic multipliers for target adjustments
            try:
                from nfl_quant.features.historical_injury_impact import get_injury_adjusted_projection

                # Get historical multiplier if player teammate is out
                if trailing_target_share is not None and trailing_target_share > 0:
                    historical_target_share, confidence = get_injury_adjusted_projection(
                        player_name=player_name,
                        team=team,
                        position=position,
                        baseline_projection=trailing_target_share,
                        injury_data=team_injury_data,
                        seasons=[get_current_season()],
                        stat_type='targets'
                    )

                    # Use historical multiplier if confidence is medium or high
                    if confidence in ['medium', 'high']:
                        historical_boost = historical_target_share / trailing_target_share
                        logger.info(f"   HISTORICAL injury adjustment for {player_name}: "
                                   f"target_share {trailing_target_share:.3f} -> {historical_target_share:.3f} "
                                   f"(x{historical_boost:.2f}, conf={confidence})")
                        trailing_target_share = historical_target_share
                        target_boost = historical_boost  # Update for logging
                    elif target_boost != 1.0:
                        # Fall back to generic multiplier if no historical data
                        original_target_share = trailing_target_share
                        trailing_target_share = trailing_target_share * target_boost
                        logger.debug(f"   Generic injury adjustment for {player_name}: "
                                    f"target_share {original_target_share:.3f} -> {trailing_target_share:.3f} (x{target_boost:.2f})")
                elif target_boost != 1.0 and trailing_target_share is not None:
                    original_target_share = trailing_target_share
                    trailing_target_share = trailing_target_share * target_boost
                    logger.debug(f"   Injury adjustment for {player_name}: target_share {original_target_share:.3f} -> {trailing_target_share:.3f} (x{target_boost:.2f})")
            except ImportError:
                # Historical module not available, use generic multipliers
                if target_boost != 1.0 and trailing_target_share is not None:
                    original_target_share = trailing_target_share
                    trailing_target_share = trailing_target_share * target_boost
                    logger.debug(f"   Injury adjustment for {player_name}: target_share {original_target_share:.3f} -> {trailing_target_share:.3f} (x{target_boost:.2f})")
            except Exception as hist_err:
                logger.debug(f"   Historical injury lookup failed for {player_name}: {hist_err}")
                if target_boost != 1.0 and trailing_target_share is not None:
                    original_target_share = trailing_target_share
                    trailing_target_share = trailing_target_share * target_boost
                    logger.debug(f"   Injury adjustment for {player_name}: target_share {original_target_share:.3f} -> {trailing_target_share:.3f} (x{target_boost:.2f})")

            # Apply carry share multiplier for RBs (also try historical data)
            carry_boost = injury_multipliers.get('carry_share_multiplier', 1.0)
            if position == 'RB' and trailing_carry_share is not None and trailing_carry_share > 0:
                try:
                    from nfl_quant.features.historical_injury_impact import get_injury_adjusted_projection

                    historical_carry_share, confidence = get_injury_adjusted_projection(
                        player_name=player_name,
                        team=team,
                        position=position,
                        baseline_projection=trailing_carry_share,
                        injury_data=team_injury_data,
                        seasons=[get_current_season()],
                        stat_type='carries'
                    )

                    if confidence in ['medium', 'high']:
                        historical_boost = historical_carry_share / trailing_carry_share
                        logger.info(f"   HISTORICAL injury adjustment for {player_name}: "
                                   f"carry_share {trailing_carry_share:.3f} -> {historical_carry_share:.3f} "
                                   f"(x{historical_boost:.2f}, conf={confidence})")
                        trailing_carry_share = historical_carry_share
                        carry_boost = historical_boost
                    elif carry_boost != 1.0:
                        original_carry_share = trailing_carry_share
                        trailing_carry_share = trailing_carry_share * carry_boost
                        logger.debug(f"   Generic injury adjustment for {player_name}: "
                                    f"carry_share {original_carry_share:.3f} -> {trailing_carry_share:.3f} (x{carry_boost:.2f})")
                except Exception as hist_err:
                    logger.debug(f"   Historical carry lookup failed for {player_name}: {hist_err}")
                    if carry_boost != 1.0:
                        original_carry_share = trailing_carry_share
                        trailing_carry_share = trailing_carry_share * carry_boost
                        logger.debug(f"   Injury adjustment for {player_name}: carry_share {original_carry_share:.3f} -> {trailing_carry_share:.3f} (x{carry_boost:.2f})")
            elif carry_boost != 1.0 and trailing_carry_share is not None:
                original_carry_share = trailing_carry_share
                trailing_carry_share = trailing_carry_share * carry_boost
                logger.debug(f"   Injury adjustment for {player_name}: carry_share {original_carry_share:.3f} -> {trailing_carry_share:.3f} (x{carry_boost:.2f})")

            # Note: OUT player check moved earlier (line 714) to skip calculations before they happen
    except ImportError:
        # If injury modules not available, continue without adjustment
        logger.debug(f"   Injury modules not available for {player_name}")
    except Exception as e:
        # Log but don't fail - injury adjustments are enhancements
        if "OUT due to injury" in str(e):
            raise  # Re-raise if player is actually out
        logger.debug(f"   Could not apply injury adjustments for {player_name}: {e}")

    # === SNAP SHARE VALIDATION (ENHANCED NOV 23, 2025) ===
    # Cap injury-adjusted shares based on actual snap share to prevent unrealistic projections
    # ENHANCEMENTS:
    # - Use EWMA snap share (weights recent weeks more)
    # - Dynamic threshold based on trend (emerging/declining roles)
    # - Role stability check (4+ weeks establishes role)

    # Get enhanced snap share metrics
    from nfl_quant.utils.unified_integration import calculate_snap_share_metrics

    snap_metrics = calculate_snap_share_metrics(
        player_name=player_name,
        position=position,
        team=team,
        week=week,
        season=season  # Use 'season' parameter from main() function
    )

    ewma_snap_share = snap_metrics['ewma_snap_share']
    snap_trend = snap_metrics['snap_trend']
    is_emerging = snap_metrics['is_emerging']
    is_declining = snap_metrics['is_declining']
    is_established = snap_metrics['is_established']

    # ENHANCEMENT #2: Dynamic backup threshold based on trend
    if is_emerging:
        # Emerging role (+10% snap increase) - relax threshold
        backup_threshold = 0.15
        logger.debug(f"   {player_name}: Emerging role detected (trend={snap_trend:+.1%}), using relaxed threshold (15%)")
    elif is_declining:
        # Declining role (-10% snap decrease) - tighten threshold
        backup_threshold = 0.25
        logger.debug(f"   {player_name}: Declining role detected (trend={snap_trend:+.1%}), using strict threshold (25%)")
    else:
        # Stable role - standard threshold
        backup_threshold = 0.20

    # ENHANCEMENT #3: Established roles bypass backup cap
    if is_established:
        logger.debug(f"   {player_name}: Established role ({snap_metrics['weeks_with_role']} weeks >15% snaps), no backup cap applied")
        is_backup = False
    elif ewma_snap_share < backup_threshold:
        is_backup = True
        logger.debug(
            f"   {player_name}: Backup detected (EWMA snap={ewma_snap_share:.1%} < {backup_threshold:.0%}, "
            f"trend={snap_trend:+.1%})"
        )
    else:
        is_backup = False
        logger.debug(f"   {player_name}: Not a backup (EWMA snap={ewma_snap_share:.1%} >= {backup_threshold:.0%})")

    # Apply backup caps if player is classified as backup
    if is_backup and trailing_snap_share is not None:
        # Player is a backup - validate adjusted shares don't exceed realistic limits

        # Cap target_share based on snap_share (backup can't get more targets than snaps played)
        if trailing_target_share is not None and position in ['WR', 'TE', 'RB']:
            # Theoretical max: snap_share × pass_attempts × catch_opportunity_rate (0.25 for backups)
            # This prevents backups from getting 40% target share when they only play 10% of snaps
            max_target_share = trailing_snap_share * 1.5  # Conservative: 1.5x snap share

            if trailing_target_share > max_target_share:
                logger.info(
                    f"   SNAP SHARE CAP: {player_name} target_share capped {trailing_target_share:.3f} → {max_target_share:.3f} "
                    f"(snap_share={trailing_snap_share:.1%}, backup player)"
                )
                trailing_target_share = max_target_share

        # Cap carry_share based on snap_share (backup RB can't get more carries than snaps played)
        if trailing_carry_share is not None and position == 'RB':
            # Conservative cap: backup RB can get at most 1x their snap share as carry share
            max_carry_share = trailing_snap_share * 1.0

            if trailing_carry_share > max_carry_share:
                logger.info(
                    f"   SNAP SHARE CAP: {player_name} carry_share capped {trailing_carry_share:.3f} → {max_carry_share:.3f} "
                    f"(snap_share={trailing_snap_share:.1%}, backup player)"
                )
                trailing_carry_share = max_carry_share

    # Calculate opponent defensive EPA - FAIL if not available
    if opponent == 'UNK':
        raise ValueError(f"Opponent not available for {player_name} ({team}) in week {week}")

    try:
        from nfl_quant.utils.defensive_stats_integration import get_defensive_epa_for_player
        # Get position-specific EPA (used as primary value)
        opponent_def_epa = get_defensive_epa_for_player(opponent, position, week)
        if opponent_def_epa is None:
            raise ValueError(
                f"Defensive EPA not available for {opponent} vs {position} in week {week}. "
                f"Run defensive stats extraction to populate this data."
            )

        # FIX: Get BOTH pass and rush defense EPA for complete context
        # Pass defense affects QB/WR/TE AND receiving RBs
        # Rush defense affects RBs
        opp_pass_def_epa = get_defensive_epa_for_player(opponent, 'WR', week) or 0.0
        opp_rush_def_epa = get_defensive_epa_for_player(opponent, 'RB', week) or 0.0

        # FIX: Calculate defense ranks (1-32, 1 = best defense = most negative EPA)
        # Note: This is simplified - ideally would be calculated once per week for all teams
        # For now, use a heuristic based on EPA value (can be improved with caching)
        # More negative EPA = better defense = lower rank
        # Typical NFL range: -0.2 (elite) to +0.2 (worst)
        # Map to 1-32: rank = 16 - (epa * 75)  (clamped to 1-32)
        opp_pass_def_rank = max(1, min(32, int(16 - opp_pass_def_epa * 75)))
        opp_rush_def_rank = max(1, min(32, int(16 - opp_rush_def_epa * 75)))

    except Exception as e:
        raise ValueError(
            f"Could not calculate defensive EPA for {opponent} vs {position}: {e}. "
            f"Run defensive stats extraction to populate this data."
        )

    return PlayerPropInput(
        player_id=player_name.lower().replace(' ', '_'),
        player_name=player_name,
        team=team,
        position=position,
        week=week,
        opponent=opponent,
        projected_team_total=projected_team_total,
        projected_opponent_total=projected_opponent_total,
        projected_game_script=projected_game_script,
        projected_pace=pace,
        trailing_snap_share=trailing_snap_share,
        trailing_target_share=trailing_target_share,
        trailing_carry_share=trailing_carry_share,
        trailing_yards_per_opportunity=trailing_yards_per_opportunity,
        trailing_td_rate=trailing_td_rate,
        # QB-specific efficiency metrics (for trained model)
        trailing_comp_pct=trailing_comp_pct,
        trailing_yards_per_completion=trailing_yards_per_completion,
        trailing_td_rate_pass=trailing_td_rate_pass,
        trailing_yards_per_carry=trailing_yards_per_carry,
        trailing_td_rate_rush=trailing_td_rate_rush,
        trailing_yards_per_target=trailing_yards_per_target,  # FIX: Bug #4
        opponent_def_epa_vs_position=opponent_def_epa,
        # Add actual historical averages for use when trailing_target_share is 0.0
        avg_rec_yd=avg_rec_yd,
        avg_rec_tgt=avg_rec_tgt,
        avg_rush_yd=avg_rush_yd,
        avg_pass_att=avg_pass_att,  # FIX: QB pass attempts (was missing)
        # FIX: Add direct EWMA usage stats for simulator (same pattern as avg_pass_att fix)
        trailing_targets=trailing_targets,  # EWMA targets per game (WR/TE/RB)
        trailing_carries=trailing_carries,  # EWMA carries per game (RB)
        trailing_catch_rate=trailing_catch_rate,  # Player catch rate (receptions/targets)
        # FIX: Add granular opponent defense stats (Dec 13, 2025)
        opp_pass_def_epa=opp_pass_def_epa,  # Pass defense EPA
        opp_pass_def_rank=opp_pass_def_rank,  # Pass defense rank 1-32
        opp_rush_def_epa=opp_rush_def_epa,  # Rush defense EPA
        opp_rush_def_rank=opp_rush_def_rank,  # Rush defense rank 1-32
        # FIX: Add archetype features for dynamic correlation
        adot=adot,  # Average depth of target (WR/TE/RB)
        ypt_variance=ypt_variance,  # Yards per target variance
        # FIX: Add trailing opponent defense EPA (schedule strength)
        trailing_opp_pass_def_epa=trailing_opp_pass_def_epa,
        trailing_opp_rush_def_epa=trailing_opp_rush_def_epa,
        # FIX: Add team pace
        team_pace=team_pace,
        # FIX: Add slot_snap_pct from PBP data (root problem fix)
        slot_snap_pct=slot_snap_pct,
        # Add team usage projections from game simulations
        projected_team_pass_attempts=projected_team_pass_attempts,
        projected_team_rush_attempts=projected_team_rush_attempts,
        projected_team_targets=projected_team_targets,
    )


def load_game_context(week: int, season: int = None) -> Dict[str, Dict]:
    """Load game context from simulation results or odds data."""
    import glob
    import json

    if season is None:
        season = get_current_season()

    game_context = {}

    # Look for simulation files first
    sim_files = sorted(glob.glob(f'reports/sim_{season}_{week:02d}_*.json'))
    if not sim_files:
        # Try alternative pattern
        sim_files = sorted(glob.glob(f'reports/sim_*_week{week}_*.json'))

    logger.info(f"   Found {len(sim_files)} simulation files for week {week}")
    if sim_files:
        logger.info(f"   Sample files: {sim_files[:3]}")

    for sim_file in sim_files:
        try:
            with open(sim_file) as f:
                sim_data = json.load(f)

            # Extract team names from filename if not in JSON
            home_team = sim_data.get('home_team', '')
            away_team = sim_data.get('away_team', '')

            # If teams not in JSON, extract from filename: sim_{season}_{week}_AWAY_HOME_42.json
            if not home_team or not away_team:
                filename_parts = Path(sim_file).stem.split('_')
                if len(filename_parts) >= 5:
                    # Format: sim_{season}_{week}_AWAY_HOME_42
                    away_team = filename_parts[3]  # 4th element (0-indexed: 3)
                    home_team = filename_parts[4]  # 5th element (0-indexed: 4)

            if not home_team or not away_team:
                logger.warning(f"   ⚠️  Could not extract team names from {sim_file}, skipping")
                continue

            # FAIL EXPLICITLY if required simulation data missing
            home_score = sim_data.get('home_score_median')
            away_score = sim_data.get('away_score_median')

            if home_score is None or away_score is None:
                raise ValueError(
                    f"Simulation file {sim_file} missing required scores. "
                    f"Expected 'home_score_median' and 'away_score_median'. "
                    f"Run game simulations to populate this data."
                )

            # Get pace from simulation input - FAIL if not available
            pace = sim_data.get('pace')
            if pace is None:
                # Try to estimate from plays per game if available
                if 'plays_per_game' in sim_data:
                    pace = sim_data['plays_per_game'] / 2.0  # Rough estimate: ~29 seconds per play
                else:
                    # Calculate from team pace metrics if available
                    try:
                        from nfl_quant.features.team_metrics import TeamMetricsExtractor
                        extractor = TeamMetricsExtractor()
                        home_pace = extractor.get_team_pace(home_team, week)
                        away_pace = extractor.get_team_pace(away_team, week)
                        if home_pace.get('neutral_plays_per_game') and away_pace.get('neutral_plays_per_game'):
                            avg_plays = (home_pace['neutral_plays_per_game'] + away_pace['neutral_plays_per_game']) / 2.0
                            if avg_plays > 0:
                                pace = (60 * 60) / avg_plays
                            else:
                                raise ValueError(f"Invalid plays_per_game for {home_team} vs {away_team}")
                        else:
                            raise ValueError(
                                f"Pace data not available for {home_team} vs {away_team}. "
                                f"Expected 'pace' in simulation or team pace metrics. "
                                f"Run game simulations or team metrics extraction."
                            )
                    except Exception as e:
                        raise ValueError(
                            f"Could not calculate pace for {home_team} vs {away_team}: {e}. "
                            f"Run game simulations to populate pace data."
                        )

            # Calculate team usage projections from pace and team totals
            # Pace can be in different formats:
            # - If pace > 50: It's plays per team per game (e.g., 64.6)
            # - If pace < 50: It's seconds per play (e.g., 30.0)
            # We need to get team_plays (plays per team per game)
            if pace > 50:
                # Pace is already plays per team per game
                team_plays = pace
            elif pace > 0:
                # Pace is seconds per play, convert to plays per game
                total_plays_per_game = 3600 / pace
                team_plays = total_plays_per_game / 2.0  # Each team gets ~half
            else:
                team_plays = 65.0  # Default to league average if invalid

            # League averages: ~60% pass, ~40% rush
            # Adjust based on game script (leading teams run more, trailing teams pass more)
            game_script_home = home_score - away_score
            base_pass_rate = 0.60

            # Game script adjustments
            if game_script_home > 7:  # Leading by more than 7
                pass_rate_home = base_pass_rate - 0.10  # Run more
            elif game_script_home < -7:  # Trailing by more than 7
                pass_rate_home = base_pass_rate + 0.10  # Pass more
            else:
                pass_rate_home = base_pass_rate

            game_script_away = away_score - home_score
            if game_script_away > 7:
                pass_rate_away = base_pass_rate - 0.10
            elif game_script_away < -7:
                pass_rate_away = base_pass_rate + 0.10
            else:
                pass_rate_away = base_pass_rate

            # Calculate team usage
            # CRITICAL FIX: Use actual team historical pass/rush attempts from NFLverse
            # NOT generic league averages, to match the target_share calculation
            try:
                from nfl_quant.data.dynamic_parameters import get_parameter_provider
                param_provider = get_parameter_provider()

                # Get actual team pass/rush attempts from NFLverse data
                home_pass_attempts = param_provider.get_team_pass_attempts(home_team, up_to_week=week)
                home_rush_attempts = param_provider.get_team_rush_attempts(home_team, up_to_week=week)
                away_pass_attempts = param_provider.get_team_pass_attempts(away_team, up_to_week=week)
                away_rush_attempts = param_provider.get_team_rush_attempts(away_team, up_to_week=week)

                # Apply game script adjustments (±10% based on lead/deficit)
                if game_script_home > 7:
                    home_pass_attempts *= 0.9  # Run more when leading
                    home_rush_attempts *= 1.1
                elif game_script_home < -7:
                    home_pass_attempts *= 1.1  # Pass more when trailing
                    home_rush_attempts *= 0.9

                if game_script_away > 7:
                    away_pass_attempts *= 0.9
                    away_rush_attempts *= 1.1
                elif game_script_away < -7:
                    away_pass_attempts *= 1.1
                    away_rush_attempts *= 0.9

                logger.debug(f"   Using NFLverse team stats: {home_team}={home_pass_attempts:.1f} pass, {away_team}={away_pass_attempts:.1f} pass")
            except Exception as e:
                # Fallback to generic league averages if NFLverse data not available
                logger.warning(f"   Could not get NFLverse team stats, using generic: {e}")
                home_pass_attempts = max(20.0, team_plays * pass_rate_home)
                home_rush_attempts = max(15.0, team_plays * (1 - pass_rate_home))
                away_pass_attempts = max(20.0, team_plays * pass_rate_away)
                away_rush_attempts = max(15.0, team_plays * (1 - pass_rate_away))

            home_targets = home_pass_attempts * 1.0  # ~1 target per pass attempt
            away_targets = away_pass_attempts * 1.0

            game_context[home_team] = {
                'team_total': home_score,
                'opponent_total': away_score,
                'game_script': game_script_home,
                'pace': pace,
                'opponent': away_team,
                'is_home': True,
                'home_field_advantage_points': 1.5,  # Standard NFL home field advantage
                # Provide BOTH key formats for compatibility
                'projected_team_pass_attempts': home_pass_attempts,
                'projected_team_rush_attempts': home_rush_attempts,
                'projected_team_targets': home_targets,
                'team_pass_attempts': home_pass_attempts,  # For simulator line 246
                'team_rush_attempts': home_rush_attempts,  # For simulator line 247
                'team_targets': home_targets,
            }

            game_context[away_team] = {
                'team_total': away_score,
                'opponent_total': home_score,
                'game_script': game_script_away,
                'pace': pace,
                'opponent': home_team,
                'is_home': False,
                'home_field_advantage_points': -1.5,  # Away team disadvantage
                # Provide BOTH key formats for compatibility
                'projected_team_pass_attempts': away_pass_attempts,
                'projected_team_rush_attempts': away_rush_attempts,
                'projected_team_targets': away_targets,
                'team_pass_attempts': away_pass_attempts,  # For simulator line 246
                'team_rush_attempts': away_rush_attempts,  # For simulator line 247
                'team_targets': away_targets,
            }
        except Exception as e:
            logger.debug(f"Error loading {sim_file}: {e}")
            continue

    # FAIL EXPLICITLY if no simulation files found
    if not game_context:
        raise FileNotFoundError(
            f"No game simulation files found for week {week}. "
            f"Expected files matching pattern: reports/sim_2025_{week:02d}_*.json or reports/sim_*_week{week}_*.json "
            f"Run game simulations BEFORE generating player predictions. "
            f"Command: python scripts/simulate/run_game_simulations.py --week {week}"
        )

    return game_context


def normalize_tds_to_game_totals(df: pd.DataFrame, week: int, season: int = None) -> pd.DataFrame:
    """
    Normalize player TD predictions to match game simulation totals.

    This ensures consistency between:
    - Game-level predictions (team total points/TDs)
    - Player-level predictions (individual TD probabilities)

    Uses top-down approach: game totals constrain player TDs while
    preserving relative TD shares between players.

    Args:
        df: DataFrame with player predictions
        week: Week number
        season: NFL season year (None for auto-detect)

    Returns:
        DataFrame with normalized TD predictions
    """
    import glob
    import json
    from pathlib import Path
    import os

    if season is None:
        season = get_current_season()

    # Load game simulations - try both relative and absolute paths
    # First try relative (works if run from project root)
    sim_files = glob.glob(f'reports/sim_{season}_{week:02d}_*.json')

    # If not found, try absolute path from script location
    if not sim_files:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        reports_dir = project_root / 'reports'
        sim_files = list(reports_dir.glob(f'sim_{season}_{week:02d}_*.json'))
        sim_files = [str(f) for f in sim_files]

    if not sim_files:
        logger.warning(f"   ⚠️  No game simulation files found for week {week}")
        logger.warning(f"   Searched in: reports/sim_{season}_{week:02d}_*.json")
        logger.warning("   Skipping TD normalization (player TDs may be inconsistent with game totals)")
        return df

    logger.info(f"   Found {len(sim_files)} game simulation files")

    # Build team total TDs from game sims
    team_game_tds = {}
    files_processed = 0
    files_failed = 0

    for sim_file in sim_files:
        try:
            with open(sim_file) as f:
                sim_data = json.load(f)

            # Extract team names from game_id (format: "2025_09_AWAY_HOME")
            game_id = sim_data.get('game_id', '')
            if game_id:
                parts = game_id.split('_')
                if len(parts) >= 4:
                    away_team = parts[2]  # Third part is away team
                    home_team = parts[3]  # Fourth part is home team
                else:
                    # Fallback: try home_team/away_team keys
                    home_team = sim_data.get('home_team')
                    away_team = sim_data.get('away_team')
            else:
                home_team = sim_data.get('home_team')
                away_team = sim_data.get('away_team')

            # Estimate TDs from median score
            # Based on analysis: game sims average 2.62 TDs/team for ~19pts = 7.25 pts/TD
            # NFL actual averages 2.40 TDs/team
            # Use 7.5 points per TD (accounts for TDs + PATs, some FGs)
            # REQUIRE actual scores - NO hardcoded 24.0 defaults
            if 'home_score_median' not in sim_data:
                raise ValueError(
                    f"Game simulation {sim_file} missing 'home_score_median'. "
                    f"NO HARDCODED 24.0 - regenerate game sims with actual data."
                )
            if 'away_score_median' not in sim_data:
                raise ValueError(
                    f"Game simulation {sim_file} missing 'away_score_median'. "
                    f"NO HARDCODED 24.0 - regenerate game sims with actual data."
                )
            home_score = sim_data['home_score_median']
            away_score = sim_data['away_score_median']

            # More accurate estimate: 1 TD per 7.5 points
            home_tds = home_score / 7.5
            away_tds = away_score / 7.5

            if home_team:
                team_game_tds[home_team] = home_tds
            if away_team:
                team_game_tds[away_team] = away_tds

            files_processed += 1

        except Exception as e:
            files_failed += 1
            logger.debug(f"   Error loading {sim_file}: {e}")
            continue

    logger.info(f"   Processed {files_processed} files, {files_failed} failed")

    if not team_game_tds:
        logger.warning("   ⚠️  Could not load any game totals")
        logger.warning("   Skipping TD normalization")
        return df

    logger.info(f"   Loaded game totals for {len(team_game_tds)} teams")

    # Normalize TDs for each team
    teams_normalized = 0
    teams_skipped = 0

    for team in df['team'].unique():
        if pd.isna(team) or team not in team_game_tds:
            teams_skipped += 1
            continue

        team_players = df['team'] == team

        # Calculate current sum of TDs for this team (only if columns exist)
        # CRITICAL: Do NOT count passing_tds because they are the SAME as receiving_tds
        # (QB throws TD = WR/RB/TE catches TD - same touchdown counted twice)
        # Team total TDs = rushing TDs + receiving TDs (passing TDs are already in receiving TDs)
        current_rush_tds = 0
        current_rec_tds = 0

        if 'rushing_tds_mean' in df.columns:
            current_rush_tds = df.loc[team_players, 'rushing_tds_mean'].fillna(0).sum()
        if 'receiving_tds_mean' in df.columns:
            current_rec_tds = df.loc[team_players, 'receiving_tds_mean'].fillna(0).sum()

        current_total_tds = current_rush_tds + current_rec_tds

        # Get target TDs from game simulation
        target_tds = team_game_tds[team]

        if current_total_tds < 0.01:
            # No TDs predicted, skip normalization
            teams_skipped += 1
            continue

        # Calculate scaling factor
        scaling_factor = target_tds / current_total_tds

        # Apply scaling to rushing and receiving TDs for this team (only if columns exist)
        # CRITICAL: Do NOT scale passing_tds - they are derived from receiving_tds
        # and QB passing TDs should reflect actual TD rate, not game total scaling
        if 'rushing_tds_mean' in df.columns:
            df.loc[team_players, 'rushing_tds_mean'] *= scaling_factor
        if 'receiving_tds_mean' in df.columns:
            df.loc[team_players, 'receiving_tds_mean'] *= scaling_factor
        # NOTE: passing_tds_mean is NOT scaled - it should stay as simulated
        # because it represents QB's actual TD rate, and scales naturally with receiving_tds

        teams_normalized += 1

        # Log for verification
        logger.debug(f"   {team}: {current_total_tds:.2f} TDs → {target_tds:.2f} TDs "
                    f"(scaling: {scaling_factor:.3f})")

    logger.info(f"   ✅ Normalized {teams_normalized} teams to match game totals")
    if teams_skipped > 0:
        logger.info(f"   ⚠️  Skipped {teams_skipped} teams (no game data or no TDs)")

    # Log sample after normalization
    logger.info("\n   Sample normalized TD predictions:")
    sample_teams = df['team'].unique()[:2]
    for team in sample_teams:
        if pd.notna(team):
            team_df = df[df['team'] == team].head(3)
            rush_sum = team_df['rushing_tds_mean'].fillna(0).sum() if 'rushing_tds_mean' in team_df.columns else 0
            rec_sum = team_df['receiving_tds_mean'].fillna(0).sum() if 'receiving_tds_mean' in team_df.columns else 0
            pass_sum = team_df['passing_tds_mean'].fillna(0).sum() if 'passing_tds_mean' in team_df.columns else 0
            total_tds = rush_sum + rec_sum + pass_sum
            logger.info(f"     {team} top 3 players: {total_tds:.2f} TDs")
            for _, row in team_df.iterrows():
                rush = row.get('rushing_tds_mean', 0) or 0
                rec = row.get('receiving_tds_mean', 0) or 0
                pass_td = row.get('passing_tds_mean', 0) or 0
                logger.info(f"       {row['player_name']}: {rush:.3f} rush, {rec:.3f} rec, {pass_td:.3f} pass")

    return df


def calibrate_td_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply position-specific TD calibrators to TD predictions.

    Uses position-specific calibrators (QB, RB, WR, TE) for improved accuracy.
    Falls back to unified calibrator if position-specific not available.
    """
    from pathlib import Path
    from nfl_quant.calibration.td_calibrator_loader import get_td_calibrator_loader

    # Try position-specific TD calibrators first
    td_loader = get_td_calibrator_loader()

    # Fallback to old unified calibrators if position-specific not available
    fallback_calibrator = None
    if not td_loader.is_available():
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent

        calibrator_paths = [
            project_root / 'data/models/td_calibrator_v2_improved.joblib',
            project_root / 'data/models/td_calibrator_v1.joblib',
        ]

        for calibrator_path in calibrator_paths:
            if calibrator_path.exists():
                fallback_calibrator = joblib.load(calibrator_path)
                logger.info(f"   ✅ Loaded fallback TD calibrator from {calibrator_path.name}")
                break

    if not td_loader.is_available() and fallback_calibrator is None:
        logger.warning("   ⚠️  No TD calibrators found - predictions will not be calibrated")
        return df

    # Log which calibrator system is being used
    if td_loader.is_available():
        loaded_positions = [pos for pos in ['QB', 'RB', 'WR', 'TE'] if td_loader.get_calibrator(pos)]
        logger.info(f"   ✅ Using position-specific TD calibrators: {', '.join(loaded_positions)}")
    else:
        logger.info(f"   ⚠️  Using fallback unified TD calibrator (position-specific not available)")

    # Calculate raw TD probability for each player
    total_tds = pd.Series(0, index=df.index)
    if 'rushing_tds_mean' in df.columns:
        total_tds += df['rushing_tds_mean'].fillna(0)
    if 'receiving_tds_mean' in df.columns:
        total_tds += df['receiving_tds_mean'].fillna(0)
    if 'passing_tds_mean' in df.columns:
        total_tds += df['passing_tds_mean'].fillna(0)

    df['raw_td_prob'] = np.clip(total_tds, 0.0, 0.95)

    # Apply position-specific calibration
    calibrated_probs = []
    position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'UNK': 0, 'FALLBACK': 0}

    for idx in df.index:
        raw_prob = df.loc[idx, 'raw_td_prob']
        position = df.loc[idx].get('position', 'UNK')

        if pd.isna(raw_prob) or raw_prob <= 0:
            calibrated_probs.append(0.0)
            continue

        # Try position-specific calibrator first
        if td_loader.is_available() and position in ['QB', 'RB', 'WR', 'TE']:
            cal_prob = td_loader.calibrate_td_probability(raw_prob, position)
            calibrated_probs.append(cal_prob)
            position_counts[position] += 1
        elif fallback_calibrator is not None:
            # Use fallback unified calibrator
            cal_prob = float(fallback_calibrator.predict([raw_prob])[0])
            calibrated_probs.append(cal_prob)
            position_counts['FALLBACK'] += 1
        else:
            # No calibrator available - use simple shrinkage
            cal_prob = 0.5 + (raw_prob - 0.5) * 0.75
            calibrated_probs.append(cal_prob)
            position_counts['UNK'] += 1

    df['calibrated_td_prob'] = calibrated_probs

    # Back out calibrated TD means by scaling proportionally
    # This preserves the distribution across TD types (rush/rec/pass)
    for idx in df.index:
        raw_prob = df.loc[idx, 'raw_td_prob']
        cal_prob = df.loc[idx, 'calibrated_td_prob']

        if raw_prob > 0.001:
            scaling = cal_prob / raw_prob
            if 'rushing_tds_mean' in df.columns:
                df.loc[idx, 'rushing_tds_mean'] *= scaling
            if 'receiving_tds_mean' in df.columns:
                df.loc[idx, 'receiving_tds_mean'] *= scaling
            if 'passing_tds_mean' in df.columns:
                df.loc[idx, 'passing_tds_mean'] *= scaling

    # Log calibration statistics
    logger.info(f"   Raw TD prob (mean): {df['raw_td_prob'].mean():.1%}")
    logger.info(f"   Calibrated TD prob (mean): {df['calibrated_td_prob'].mean():.1%}")
    if td_loader.is_available():
        logger.info(f"   Position breakdown: QB={position_counts['QB']}, RB={position_counts['RB']}, "
                   f"WR={position_counts['WR']}, TE={position_counts['TE']}, "
                   f"Fallback={position_counts['FALLBACK']}, Unknown={position_counts['UNK']}")

    return df


def enhance_td_predictions(df: pd.DataFrame, week: int, game_context: Dict, season: int = None) -> pd.DataFrame:
    """
    Replace placeholder TD predictions with statistical TD model.

    Args:
        df: DataFrame with player predictions
        week: Current week number
        game_context: Dict with team projected totals
        season: NFL season year (None for auto-detect)

    Returns:
        DataFrame with enhanced TD predictions
    """
    if season is None:
        season = get_current_season()

    logger.info("   Initializing TD predictor...")

    # Load historical stats for TD rates
    hist_stats_path = Path(f'data/nflverse_cache/stats_player_week_{season}.csv')
    # Use nflverse data as single source of truth (includes most recent week)
    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
    td_predictor = TouchdownPredictor(
        historical_stats_path=hist_stats_path if hist_stats_path.exists() else None,
        pbp_path=pbp_path if pbp_path.exists() else None
    )

    logger.info(f"   Processing {len(df)} players...")

    enhanced_count = 0

    for idx, row in df.iterrows():
        player_name = row['player_name']
        position = row.get('position', 'UNK')

        if position == 'UNK':
            continue

        # Get usage projections
        # Use rushing_attempts_mean if available (from CSV), otherwise estimate from yards
        projected_carries = row.get('rushing_attempts_mean', 0.0)
        # Use targets_mean if available (now extracted from simulator), otherwise estimate from receptions
        projected_targets = row.get('targets_mean', 0.0)
        if projected_targets == 0.0 and position in ['RB', 'WR', 'TE']:
            # Fallback: Estimate targets from receptions if targets_mean not available
            projected_targets = row.get('receptions_mean', 0.0) * 1.5 if position in ['WR', 'TE'] else row.get('receptions_mean', 0.0) * 1.3
        projected_pass_attempts = row.get('passing_completions_mean', 0.0) * 1.7 if position == 'QB' else 0.0  # Estimate from completions

        # Fallback: Estimate usage from yards if attempts are missing/zero
        # This handles cases where the model predicts yards but not attempts
        if projected_carries == 0.0 and position in ['RB', 'QB']:
            rushing_yards = row.get('rushing_yards_mean', 0.0)
            if rushing_yards > 0:
                # Estimate carries from yards (typical: 4-5 yards per carry)
                projected_carries = max(1.0, rushing_yards / 4.5)

        if projected_targets == 0.0 and position in ['RB', 'WR', 'TE']:
            receiving_yards = row.get('receiving_yards_mean', 0.0)
            if receiving_yards > 0:
                # Estimate targets from receiving yards (typical: 8-10 yards per target)
                # Receptions = targets * catch_rate (typically 0.65-0.75)
                estimated_receptions = receiving_yards / 9.0  # Rough estimate
                if position in ['WR', 'TE']:
                    projected_targets = estimated_receptions * 1.5
                else:  # RB
                    projected_targets = estimated_receptions * 1.3

        if projected_pass_attempts == 0.0 and position == 'QB':
            passing_yards = row.get('passing_yards_mean', 0.0)
            if passing_yards > 0:
                # Estimate pass attempts from yards (typical: 7-8 yards per attempt)
                projected_pass_attempts = max(1.0, passing_yards / 7.5)

        # Get team context - REQUIRE actual data, NO hardcoded defaults
        team = row.get('team', '')
        opponent = row.get('opponent', 'UNK')

        if team not in game_context:
            raise ValueError(
                f"Team {team} not found in game_context. "
                f"NO HARDCODED DEFAULTS - ensure game simulations are generated for all teams."
            )

        # REQUIRE actual team totals from game simulations
        if 'team_total' not in game_context[team]:
            raise ValueError(
                f"Team {team} missing 'team_total' in game_context. "
                f"NO HARDCODED 24.0 - regenerate game simulations with actual data."
            )
        if 'opponent_total' not in game_context[team]:
            raise ValueError(
                f"Team {team} missing 'opponent_total' in game_context. "
                f"NO HARDCODED 24.0 - regenerate game simulations with actual data."
            )

        team_total = game_context[team]['team_total']
        opponent_total = game_context[team]['opponent_total']
        projected_game_script = game_context[team].get('game_script', team_total - opponent_total)
        if opponent == 'UNK':
            opponent = game_context[team].get('opponent', 'UNK')

        # Get snap share (important for starter vs backup discrimination)
        # FIRST: Try to get actual snap share from NFLverse data
        param_provider = get_parameter_provider()
        actual_snap_share = param_provider.get_player_snap_share(player_name, team, n_weeks=4)

        if actual_snap_share > 0.1:
            # Use actual snap share from NFLverse data
            snap_share = actual_snap_share
        else:
            # FALLBACK: Estimate from usage if no snap data available
            snap_share = 1.0  # Default to full starter
            if position == 'QB':
                snap_share = 0.98  # QBs play almost every snap
            elif position == 'RB':
                # Estimate from carries
                if projected_carries > 15:
                    snap_share = 0.85  # Workhorse back
                elif projected_carries > 10:
                    snap_share = 0.60  # Committee back
                elif projected_carries > 5:
                    snap_share = 0.40  # Backup
                else:
                    snap_share = 0.20  # Deep backup
            elif position in ['WR', 'TE']:
                # Estimate from targets
                targets = projected_targets
                if targets > 8:
                    snap_share = 0.90  # WR1/TE1
                elif targets > 5:
                    snap_share = 0.75  # WR2/TE2
                elif targets > 3:
                    snap_share = 0.55  # WR3
                else:
                    snap_share = 0.35  # Depth player

        # Estimate usage factors
        usage_factors = estimate_usage_factors(row, position)

        try:
            # Predict TD probability with defensive matchup, game script, and snap share
            td_pred = td_predictor.predict_touchdown_probability(
                player_name=player_name,
                position=position,
                projected_carries=projected_carries,
                projected_targets=projected_targets,
                projected_pass_attempts=projected_pass_attempts,
                red_zone_share=usage_factors['red_zone_share'],
                goal_line_role=usage_factors['goal_line_role'],
                team_projected_total=team_total,
                opponent_team=opponent if opponent != 'UNK' else None,
                current_week=week,
                projected_point_differential=projected_game_script,
                projected_snap_share=snap_share,
            )

            # Debug logging for first few players
            if enhanced_count < 3:
                logger.info(f"\n   DEBUG - {player_name} ({position}):")
                logger.info(f"     Carries: {projected_carries:.1f}, Targets: {projected_targets:.1f}")
                logger.info(f"     Usage: RZ={usage_factors['red_zone_share']:.2f}, GL={usage_factors['goal_line_role']:.2f}")
                logger.info(f"     td_pred keys: {list(td_pred.keys())}")
                logger.info(f"     td_pred values: {td_pred}")

            # Update TD predictions
            if position == 'QB':
                # CRITICAL FIX: Do NOT overwrite QB TDs from TouchdownPredictor
                # The simulator already calculates correct QB TDs using actual trailing stats
                # TouchdownPredictor uses hardcoded defaults (0.045 TD/attempt) which under-predict
                # Keep the simulator's values which use the player's real trailing_td_rate_pass
                if enhanced_count < 3:
                    logger.info(f"     QB {player_name}: Keeping simulator TDs (TouchdownPredictor would overwrite with defaults)")
                    logger.info(f"     Current passing_tds_mean: {df.at[idx, 'passing_tds_mean']:.3f}")
                    logger.info(f"     Current rushing_tds_mean: {df.at[idx, 'rushing_tds_mean']:.3f}")
                # Intentionally NOT updating QB TD values - simulator values are correct

            elif position == 'RB':
                if 'rushing_tds_mean' in td_pred:
                    old_val = df.at[idx, 'rushing_tds_mean']
                    df.at[idx, 'rushing_tds_mean'] = td_pred['rushing_tds_mean']
                    if enhanced_count < 3:
                        logger.info(f"     Updated rushing_tds_mean: {old_val:.3f} -> {td_pred['rushing_tds_mean']:.3f}")
                if 'receiving_tds_mean' in td_pred:
                    old_val = df.at[idx, 'receiving_tds_mean']
                    df.at[idx, 'receiving_tds_mean'] = td_pred['receiving_tds_mean']
                    if enhanced_count < 3:
                        logger.info(f"     Updated receiving_tds_mean: {old_val:.3f} -> {td_pred['receiving_tds_mean']:.3f}")

            elif position in ['WR', 'TE']:
                if 'receiving_tds_mean' in td_pred:
                    old_val = df.at[idx, 'receiving_tds_mean']
                    df.at[idx, 'receiving_tds_mean'] = td_pred['receiving_tds_mean']
                    if enhanced_count < 3:
                        logger.info(f"     Updated receiving_tds_mean: {old_val:.3f} -> {td_pred['receiving_tds_mean']:.3f}")

            enhanced_count += 1

        except Exception as e:
            logger.debug(f"   Error enhancing TD for {player_name}: {e}")
            continue

    logger.info(f"   ✅ Enhanced TD predictions for {enhanced_count} players")

    # Log sample of enhanced predictions
    logger.info("\n   Sample TD predictions:")
    sample = df[df['position'].isin(['QB', 'RB', 'WR'])].head(5)
    for _, row in sample.iterrows():
        p_name = row['player_name']
        pos = row['position']
        if pos == 'QB':
            pass_td = row.get('passing_tds_mean', 0)
            rush_td = row.get('rushing_tds_mean', 0)
            logger.info(f"     {p_name} ({pos}): {pass_td:.2f} pass TDs, {rush_td:.2f} rush TDs")
        elif pos == 'RB':
            rush_td = row.get('rushing_tds_mean', 0)
            rec_td = row.get('receiving_tds_mean', 0)
            logger.info(f"     {p_name} ({pos}): {rush_td:.2f} rush TDs, {rec_td:.2f} rec TDs")
        elif pos in ['WR', 'TE']:
            rec_td = row.get('receiving_tds_mean', 0)
            logger.info(f"     {p_name} ({pos}): {rec_td:.2f} rec TDs")

    return df


def apply_role_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply role change overrides for backup-to-starter transitions.

    Critical for cases like RJ Harvey becoming starter when JK Dobbins goes to IR.
    Without this, the model defaults to backup historical averages (e.g., 5 carries)
    instead of starter workload (e.g., 13+ carries).

    Args:
        df: DataFrame with player predictions

    Returns:
        DataFrame with role overrides applied
    """
    try:
        role_overrides = load_role_overrides()
    except Exception as e:
        logger.warning(f"   Could not load role overrides: {e}")
        return df

    if not role_overrides:
        logger.info("   No role overrides configured")
        return df

    logger.info(f"   Found {len(role_overrides)} role override(s)")

    overrides_applied = 0

    for override in role_overrides:
        player_name = override.get('player', '').lower()
        if not player_name:
            continue

        # Find player in dataframe (case-insensitive)
        mask = df['player_name'].str.lower() == player_name
        if not mask.any():
            logger.warning(f"   Player {override.get('player')} not found in predictions")
            continue

        player_idx = df[mask].index[0]
        original_row = df.loc[player_idx].copy()

        change_type = override.get('change_type', '')

        if change_type == 'BACKUP_TO_STARTER':
            # Apply inherited workload calculation
            if 'inherited_volume' in override:
                inherited_carries = override['inherited_volume']
                volume_absorption = override.get('volume_absorption_rate', 0.85)
                player_ypc = override.get('player_ypc', 4.5)

                # Calculate expected statistics
                expected_carries = inherited_carries * volume_absorption
                expected_rush_yards = expected_carries * player_ypc

                # Also estimate rush attempts (carries = rush attempts)
                expected_rush_attempts = expected_carries

                # Update rushing predictions
                original_yards = df.loc[player_idx, 'rushing_yards_mean']
                original_attempts = df.loc[player_idx, 'rushing_attempts_mean']

                df.loc[player_idx, 'rushing_yards_mean'] = expected_rush_yards
                df.loc[player_idx, 'rushing_attempts_mean'] = expected_rush_attempts

                # Also update carries if column exists
                if 'carries_mean' in df.columns:
                    df.loc[player_idx, 'carries_mean'] = expected_carries

                # Mark that override was applied
                if 'role_override_applied' not in df.columns:
                    df['role_override_applied'] = False
                df.loc[player_idx, 'role_override_applied'] = True

                logger.info(f"   ✅ {override.get('player')} ({override.get('team')}):")
                logger.info(f"      Change: {change_type}")
                logger.info(f"      Injured starter: {override.get('injured_player', 'N/A')}")
                logger.info(f"      Rush attempts: {original_attempts:.1f} → {expected_rush_attempts:.1f}")
                logger.info(f"      Rush yards: {original_yards:.1f} → {expected_rush_yards:.1f}")
                logger.info(f"      Calculation: {inherited_carries:.1f} carries × {volume_absorption:.0%} absorption × {player_ypc:.2f} YPC")

                overrides_applied += 1

        elif change_type == 'RECEIVER_PROMOTION':
            # Handle WR/TE workload increases (e.g., WR2 becomes WR1)
            if 'inherited_targets' in override:
                inherited_targets = override['inherited_targets']
                absorption_rate = override.get('volume_absorption_rate', 0.7)
                yards_per_target = override.get('player_yards_per_target', 8.0)

                expected_targets = inherited_targets * absorption_rate
                expected_rec_yards = expected_targets * yards_per_target

                original_yards = df.loc[player_idx, 'receiving_yards_mean']
                original_targets = df.loc[player_idx, 'targets_mean'] if 'targets_mean' in df.columns else 0

                df.loc[player_idx, 'receiving_yards_mean'] = expected_rec_yards
                if 'targets_mean' in df.columns:
                    df.loc[player_idx, 'targets_mean'] = expected_targets

                if 'role_override_applied' not in df.columns:
                    df['role_override_applied'] = False
                df.loc[player_idx, 'role_override_applied'] = True

                logger.info(f"   ✅ {override.get('player')} ({override.get('team')}):")
                logger.info(f"      Change: {change_type}")
                logger.info(f"      Rec yards: {original_yards:.1f} → {expected_rec_yards:.1f}")

                overrides_applied += 1

    if overrides_applied > 0:
        logger.info(f"   Applied {overrides_applied} role override(s)")
    else:
        logger.warning("   No overrides applied (players not found or invalid override data)")

    return df


def generate_model_predictions(week: int, season: int = None, simulator_version: str = 'v3') -> pd.DataFrame:
    """
    Generate model predictions for all players.

    Args:
        week: Week number
        season: NFL season year
        simulator_version: 'v3' (legacy Normal) or 'v4' (NegBin+Lognormal+Copula)

    Returns:
        DataFrame with predictions
    """
    if season is None:
        season = get_current_season()

    logger.info("="*80)
    logger.info(f"GENERATING MODEL PREDICTIONS - WEEK {week} ({season} SEASON)")
    logger.info(f"SIMULATOR: {simulator_version.upper()}")
    logger.info("="*80)
    logger.info("")

    # PERF FIX (Dec 14, 2025): Add step timing for performance monitoring
    import time as _time
    pipeline_start = _time.time()

    # Load predictors
    step_start = _time.time()
    logger.info("1. Loading trained models...")
    try:
        usage_predictor, efficiency_predictor = load_predictors()
        logger.info(f"   ✅ Loaded usage and efficiency predictors ({_time.time() - step_start:.1f}s)")
    except Exception as e:
        logger.error(f"   ❌ Failed to load predictors: {e}")
        logger.error("   Cannot generate predictions without models")
        return pd.DataFrame()

    # Load calibrator
    calibrator = None
    calibrator_path = Path('configs/calibrator.json')
    if calibrator_path.exists():
        try:
            calibrator = NFLProbabilityCalibrator()
            calibrator.load(str(calibrator_path))
            logger.info("   ✅ Loaded calibrator")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load calibrator: {e}")

    # PERF FIX (Dec 14, 2025): Use environment variable or config default (30k)
    # Reduced from 50k to 30k for 40% speedup with <0.5% accuracy loss
    n_simulations = int(os.environ.get('NFL_QUANT_SIMULATIONS', 30000))

    # Create simulator (V3 or V4)
    step_start = _time.time()
    logger.info("\n2. Creating simulator...")
    logger.info(f"   Monte Carlo trials: {n_simulations:,}")
    if simulator_version == 'v4':
        # V4: Systematic probabilistic distributions
        simulator = PlayerSimulatorV4(
            usage_predictor=usage_predictor,
            efficiency_predictor=efficiency_predictor,
            trials=n_simulations,
            seed=42
        )
        logger.info(f"   ✅ V4 Simulator created (NegBin + Lognormal + Copula) ({_time.time() - step_start:.1f}s)")
    else:
        # V3: Legacy Normal distributions
        simulator = PlayerSimulator(
            usage_predictor=usage_predictor,
            efficiency_predictor=efficiency_predictor,
            trials=n_simulations,
            seed=42,
            calibrator=calibrator,
        )
        logger.info(f"   ✅ V3 Simulator created (Normal distributions) ({_time.time() - step_start:.1f}s)")

    # Load trailing stats first (needed for player loading) - Phase 5.2: Pass week context
    step_start = _time.time()
    logger.info("\n3. Loading trailing stats...")
    trailing_stats = load_trailing_stats(current_week=week, current_season=season)
    logger.info(f"   ✅ Loaded stats for {len(trailing_stats)} player-week combinations ({_time.time() - step_start:.1f}s)")

    # Debug: Check sample keys
    if trailing_stats:
        sample_keys = list(trailing_stats.keys())[:5]
        logger.info(f"   Sample keys: {sample_keys}")
        sample_player = list(trailing_stats.keys())[0]
        sample_stats = trailing_stats[sample_player]
        logger.info(f"   Sample stats for '{sample_player}': weeks_played={sample_stats.get('weeks_played')}, "
                   f"lookback_weeks_played={sample_stats.get('lookback_weeks_played', 0)}, "
                   f"avg_rec_yd={sample_stats.get('avg_rec_yd')}, avg_rush_yd={sample_stats.get('avg_rush_yd')}, "
                   f"avg_pass_yd={sample_stats.get('avg_pass_yd')}")

    # Load ALL active players from NFLverse data (not just those with betting odds)
    # This ensures complete prediction coverage for all players
    step_start = _time.time()
    logger.info("\n4. Loading active players from NFLverse roster data...")
    players_df = load_active_players_from_nflverse(week, season, trailing_stats=trailing_stats)

    if players_df.empty:
        logger.error("   ❌ No active players found in NFLverse data")
        logger.error(f"   Make sure NFLverse data exists for week {week}, season {season}")
        return pd.DataFrame()

    logger.info(f"   ✅ Found {len(players_df)} unique active players ({_time.time() - step_start:.1f}s)")

    # Load game context
    step_start = _time.time()
    logger.info("\n5. Loading game context...")
    game_context = load_game_context(week, season=season)
    logger.info(f"   ✅ Loaded context for {len(game_context)} teams ({_time.time() - step_start:.1f}s)")

    # Load injury data ONCE for all players (cached in module global)
    logger.info("\n5b. Loading injury data for target redistribution...")
    try:
        from nfl_quant.utils.contextual_integration import load_injury_data
        injury_data_cache = load_injury_data(week)
        if injury_data_cache:
            # Log key injuries for verification
            teams_with_key_injuries = []
            for team, data in injury_data_cache.items():
                wr1_out = data.get('top_wr_1_status', 'active').lower() in ['out', 'pup', 'ir', 'doubtful']
                rb1_out = data.get('top_rb_status', 'active').lower() in ['out', 'pup', 'ir', 'doubtful']
                if wr1_out:
                    teams_with_key_injuries.append(f"{team} WR1 OUT ({data.get('top_wr_1', 'N/A')})")
                if rb1_out:
                    teams_with_key_injuries.append(f"{team} RB1 OUT ({data.get('top_rb', 'N/A')})")

            logger.info(f"   ✅ Loaded injury data for {len(injury_data_cache)} teams")
            if teams_with_key_injuries:
                logger.info(f"   Key injuries: {', '.join(teams_with_key_injuries[:10])}")
                if len(teams_with_key_injuries) > 10:
                    logger.info(f"   ... and {len(teams_with_key_injuries) - 10} more")
        else:
            logger.warning("   ⚠️  No injury data loaded")
            injury_data_cache = {}
    except Exception as e:
        logger.warning(f"   ⚠️  Could not load injury data: {e}")
        injury_data_cache = {}

    # Store in module-level cache for create_player_input_from_odds to use
    global _injury_data_cache, _injury_data_week
    _injury_data_cache = injury_data_cache
    _injury_data_week = week

    # 5b. Load Micro Metrics (team-level advanced features)
    logger.info("\n5b. Loading micro metrics (pressure, explosive plays, turnover luck)...")
    global _micro_metrics_cache, _micro_metrics_week
    try:
        micro_calc = MicroMetricsCalculator(season)
        _micro_metrics_cache = micro_calc.get_all_team_micro_metrics(through_week=week - 1)
        _micro_metrics_week = week
        if _micro_metrics_cache is not None and not _micro_metrics_cache.empty:
            logger.info(f"   ✅ Loaded {len(_micro_metrics_cache.columns)} micro metric features for {len(_micro_metrics_cache)} teams")
            # Show sample of features
            feature_cols = [c for c in _micro_metrics_cache.columns if c != 'team']
            logger.info(f"   Features: {feature_cols[:8]}..." if len(feature_cols) > 8 else f"   Features: {feature_cols}")
        else:
            logger.warning("   ⚠️  No micro metrics generated")
            _micro_metrics_cache = pd.DataFrame()
    except Exception as e:
        logger.warning(f"   ⚠️  Could not load micro metrics: {e}")
        _micro_metrics_cache = pd.DataFrame()

    # Generate predictions for each player
    step_start = _time.time()
    logger.info("\n6. Generating predictions...")
    logger.info(f"   Processing {len(players_df)} players (this may take a few minutes)...")
    predictions = []

    # Suppress pandas FutureWarnings for cleaner progress bar
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Import name normalization for parallel workers
    from nfl_quant.utils.player_names import normalize_player_name

    # Track filtering stats
    total_players = len(players_df)
    skipped_count = 0
    skip_reasons = {}

    # PERF OPTIMIZATION: Build O(1) lookup dict ONCE before processing all players
    lookup_build_start = time.time()
    stats_lookup = build_trailing_stats_lookup(trailing_stats)
    logger.info(f"   Built stats lookup ({len(stats_lookup)} entries) in {time.time() - lookup_build_start:.2f}s")

    # PARALLEL EXECUTION PATH (Option A: Pre-create inputs, parallelize simulation)
    if ENABLE_PARALLEL_SIMULATION:
        logger.info(f"   Using PARALLEL execution with {PARALLEL_WORKERS} workers...")

        # STEP 1: Pre-create all PlayerPropInput objects sequentially (fast)
        logger.info("   Step 1: Creating player inputs...")
        player_inputs = []  # List of (player_input, team_game_context, player_name, team, position)

        for idx, row in tqdm(players_df.iterrows(), total=total_players,
                            desc="Creating inputs", ncols=100,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            player_name = row['player_name']
            team = row['team']
            position = row['position']

            # Basic validation
            if pd.isna(player_name) or pd.isna(position):
                skipped_count += 1
                skip_reasons["Missing basic fields"] = skip_reasons.get("Missing basic fields", 0) + 1
                continue

            # Check team validity
            if team and not pd.isna(team):
                team_str = str(team).upper()
                if team_str in ('UNK', 'NAN', 'NONE'):
                    team = None

            # PERF FIX: O(1) lookup instead of O(n) loop
            normalized_name = _normalize_player_name_cached(str(player_name))
            stats = stats_lookup.get(normalized_name, {})

            # Check activity
            games_played_in_lookback = stats.get('games_played_in_lookback', 0)
            avg_rec_yd = stats.get('avg_rec_yd_per_game', stats.get('avg_rec_yd', 0) or 0)
            avg_rush_yd = stats.get('avg_rush_yd_per_game', stats.get('avg_rush_yd', 0) or 0)
            avg_pass_yd = stats.get('avg_pass_yd_per_game', stats.get('avg_pass_yd', 0) or 0)
            has_activity = avg_rec_yd > 0 or avg_rush_yd > 0 or avg_pass_yd > 0

            if not has_activity and games_played_in_lookback < 2:
                skipped_count += 1
                skip_reasons[f"Insufficient data ({games_played_in_lookback} games)"] = skip_reasons.get(f"Insufficient data ({games_played_in_lookback} games)", 0) + 1
                continue

            # WR/TE specific check
            if position in ['WR', 'TE']:
                target_share = stats.get('trailing_target_share')
                if target_share is None and avg_rec_yd == 0:
                    skipped_count += 1
                    skip_reasons["No target share/receiving data"] = skip_reasons.get("No target share/receiving data", 0) + 1
                    continue
                if target_share == 0.0 and avg_rec_yd == 0:
                    skipped_count += 1
                    skip_reasons["target_share=0, no receiving data"] = skip_reasons.get("target_share=0, no receiving data", 0) + 1
                    continue

            try:
                player_input = create_player_input_from_odds(
                    player_name, team, position, week, trailing_stats, game_context, season=season
                )
                team_game_context = game_context.get(team, {}) if game_context and team else {}
                player_inputs.append((player_input, team_game_context, player_name, team, position))
            except Exception as e:
                skipped_count += 1
                skip_reasons[f"Input creation error"] = skip_reasons.get(f"Input creation error", 0) + 1
                continue

        logger.info(f"   Created {len(player_inputs)} valid inputs, {skipped_count} skipped in validation")

        # STEP 2: Parallelize simulation using ProcessPoolExecutor (the expensive part)
        # ProcessPoolExecutor bypasses Python's GIL for true parallelism (~8x speedup)
        logger.info("   Step 2: Running parallel simulations with ProcessPoolExecutor...")
        logger.info(f"   Spawning {PARALLEL_WORKERS} worker processes...")

        # Args for process workers - just the picklable data (no predictors, no callbacks)
        # Each process loads its own predictors via the initializer function
        parallel_args = [
            (pi, tgc, pn, t, pos)
            for pi, tgc, pn, t, pos in player_inputs
        ]

        simulation_results = []
        with ProcessPoolExecutor(
            max_workers=PARALLEL_WORKERS,
            initializer=_init_process_worker,
            initargs=(42,)  # Base seed for reproducibility
        ) as executor:
            # Use map for ordered results
            futures = list(executor.map(_simulate_player_for_process, parallel_args))
            for result_tuple in tqdm(futures, desc="Simulating (parallel)", ncols=100,
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                simulation_results.append(result_tuple)

        # STEP 3: Extract predictions sequentially
        logger.info("   Step 3: Extracting predictions...")
        for result, player_input, player_name, team, position in tqdm(
                [(r[0], r[1], r[2], r[3], r[4]) for r in simulation_results if r[0] is not None],
                desc="Extracting", ncols=100):
            # Get game context for this team
            team_ctx = game_context.get(team, {}) if game_context else {}

            pred = {
                'player_name': player_name,
                'player_dk': player_name,
                'player_pbp': player_name,
                'team': team,
                'position': position,
                'week': week,
                'opponent': player_input.opponent,
                # Game context fields
                'game_script_dynamic': team_ctx.get('game_script', 0),
                'is_home': team_ctx.get('is_home', False),
                'home_field_advantage_points': team_ctx.get('home_field_advantage_points', 0),
            }

            # Extract means for each stat type
            stat_mappings = [
                ('passing_yards', 'passing_yards_mean', 'passing_yards_std'),
                ('passing_completions', 'passing_completions_mean', 'passing_completions_std'),
                ('passing_attempts', 'passing_attempts_mean', 'passing_attempts_std'),
                ('passing_tds', 'passing_tds_mean', 'passing_tds_std'),
                ('rushing_yards', 'rushing_yards_mean', 'rushing_yards_std'),
                ('rushing_tds', 'rushing_tds_mean', 'rushing_tds_std'),
                ('receiving_yards', 'receiving_yards_mean', 'receiving_yards_std'),
                ('receptions', 'receptions_mean', 'receptions_std'),
                ('targets', 'targets_mean', 'targets_std'),
                ('receiving_tds', 'receiving_tds_mean', 'receiving_tds_std'),
            ]
            for src, mean_col, std_col in stat_mappings:
                if src in result:
                    pred[mean_col] = float(np.mean(result[src]))
                    pred[std_col] = float(np.std(result[src]))

            if 'carries' in result:
                pred['rushing_attempts_mean'] = float(np.mean(result['carries']))
                pred['rushing_attempts_std'] = float(np.std(result['carries']))

            if 'anytime_td' in result:
                pred['anytime_td_prob'] = float(np.mean(result['anytime_td'] > 0))

            predictions.append(pred)

        # Count simulation errors
        sim_errors = sum(1 for r in simulation_results if r[5] is not None)
        if sim_errors > 0:
            skipped_count += sim_errors
            skip_reasons["Simulation error"] = skip_reasons.get("Simulation error", 0) + sim_errors

        logger.info(f"   Parallel execution complete: {len(predictions)} predictions, {skipped_count} skipped")

    # SEQUENTIAL EXECUTION PATH (fallback)
    else:
        # Use tqdm for progress bar
        player_iterator = tqdm(
            players_df.iterrows(),
            total=total_players,
            desc="Generating predictions",
            leave=True,
            ncols=100,  # Fixed width for cleaner output
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for idx, row in player_iterator:
            player_name = row['player_name']
            team = row['team']
            position = row['position']

            # Basic validation - allow team=None (will try to infer from stats)
            if pd.isna(player_name) or pd.isna(position):
                skipped_count += 1
                reason = "Missing basic fields"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            # Check team validity - normalize invalid teams to None
            # We'll try to infer team from stats later if needed
            if team and not pd.isna(team):
                team_str = str(team).upper()
                if team_str == 'UNK' or team_str == 'NAN' or team_str == 'NONE':
                    team = None  # Set to None, will try to infer from stats

            # PERF FIX: O(1) lookup using pre-built stats_lookup dict
            normalized_name = _normalize_player_name_cached(str(player_name))
            stats = stats_lookup.get(normalized_name, {})

            weeks_played = stats.get('weeks_played', 0)
            # Phase 5.1: Use lookback_weeks_played for filtering (not total career weeks)
            lookback_weeks_played = stats.get('lookback_weeks_played', 0)
            games_played_in_lookback = stats.get('games_played_in_lookback', 0)

            # Phase 2: Use per-game averages for activity checks (not weighted averages)
            avg_rec_yd = stats.get('avg_rec_yd_per_game', stats.get('avg_rec_yd', 0) or 0)
            avg_rush_yd = stats.get('avg_rush_yd_per_game', stats.get('avg_rush_yd', 0) or 0)
            avg_pass_yd = stats.get('avg_pass_yd_per_game', stats.get('avg_pass_yd', 0) or 0)

            # Debug: Log first few players to understand matching
            if skipped_count < 5:
                logger.info(f"   DEBUG {player_name}: normalized={normalized_name}, "
                           f"weeks_played={weeks_played}, lookback_weeks={lookback_weeks_played}, "
                           f"games_in_lookback={games_played_in_lookback}, "
                           f"rec_yd={avg_rec_yd:.1f}, rush_yd={avg_rush_yd:.1f}, pass_yd={avg_pass_yd:.1f}, "
                           f"stats_found={stats is not None and len(stats) > 0}, "
                           f"has_activity={avg_rec_yd > 0 or avg_rush_yd > 0 or avg_pass_yd > 0}")

            # Phase 5.1: Skip if insufficient lookback window data
            # Use games_played_in_lookback (excludes BYE/injury weeks) for minimum threshold
            has_activity = avg_rec_yd > 0 or avg_rush_yd > 0 or avg_pass_yd > 0
            if not has_activity and games_played_in_lookback < 2:
                skipped_count += 1
                reason = f"Insufficient data ({games_played_in_lookback} games in lookback, all stats zero)"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            # For WR/TE: require target share data OR receiving activity
            if position in ['WR', 'TE']:
                target_share = stats.get('trailing_target_share')
                # Phase 2: Use per-game average for activity check
                avg_rec_yd = stats.get('avg_rec_yd_per_game', stats.get('avg_rec_yd', 0) or 0)

                # Skip if no target share AND no receiving data
                if target_share is None and avg_rec_yd == 0:
                    skipped_count += 1
                    reason = "No target share data and no receiving yards"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

                # Skip if trailing_target_share=0.0 AND no receiving data (can't estimate)
                if target_share == 0.0 and avg_rec_yd == 0:
                    skipped_count += 1
                    reason = "trailing_target_share=0.0 with no receiving data (cannot simulate)"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

            try:
                # Create player input
                player_input = create_player_input_from_odds(
                    player_name, team, position, week, trailing_stats, game_context, season=season
                )

                # Simulate player - CRITICAL FIX: Pass game_context to get team pass/rush attempts
                team_game_context = game_context.get(team, {}) if game_context else {}
                result = simulator.simulate_player(player_input, game_context=team_game_context)

                # Extract predictions (medians)
                pred = {
                    'player_name': player_name,
                    'player_dk': player_name,  # DraftKings name (same)
                    'player_pbp': player_name,  # PBP name (same)
                    'team': team,
                    'position': position,
                    'week': week,
                    'opponent': player_input.opponent,
                    # Game context fields
                    'game_script_dynamic': team_game_context.get('game_script', 0),
                    'is_home': team_game_context.get('is_home', False),
                    'home_field_advantage_points': team_game_context.get('home_field_advantage_points', 0),
                }

                # Extract means for each stat type (NOT medians - distributions are skewed)
                if 'passing_yards' in result:
                    # FIX: Use MEAN not MEDIAN - gamma distribution is right-skewed
                    pred['passing_yards_mean'] = float(np.mean(result['passing_yards']))
                    pred['passing_yards_std'] = float(np.std(result['passing_yards']))

                if 'passing_completions' in result:
                    # FIX: Use MEAN not MEDIAN for discrete count data
                    pred['passing_completions_mean'] = float(np.mean(result['passing_completions']))
                    pred['passing_completions_std'] = float(np.std(result['passing_completions']))

                if 'passing_attempts' in result:
                    # FIX: Use MEAN not MEDIAN for discrete count data
                    pred['passing_attempts_mean'] = float(np.mean(result['passing_attempts']))
                    pred['passing_attempts_std'] = float(np.std(result['passing_attempts']))

                if 'passing_tds' in result:
                    # FIX: Use MEAN not MEDIAN for TDs - median causes 0.0 for low-frequency events
                    pred['passing_tds_mean'] = float(np.mean(result['passing_tds']))
                    pred['passing_tds_std'] = float(np.std(result['passing_tds']))

                if 'rushing_yards' in result:
                    # FIX: Use MEAN not MEDIAN - gamma distribution is right-skewed
                    pred['rushing_yards_mean'] = float(np.mean(result['rushing_yards']))
                    pred['rushing_yards_std'] = float(np.std(result['rushing_yards']))

                # Simulator returns 'carries' not 'rushing_attempts'
                if 'carries' in result:
                    # FIX: Use MEAN not MEDIAN for discrete count data
                    pred['rushing_attempts_mean'] = float(np.mean(result['carries']))
                    pred['rushing_attempts_std'] = float(np.std(result['carries']))
                elif 'rushing_attempts' in result:
                    # FIX: Use MEAN not MEDIAN for discrete count data
                    pred['rushing_attempts_mean'] = float(np.mean(result['rushing_attempts']))
                    pred['rushing_attempts_std'] = float(np.std(result['rushing_attempts']))

                if 'rushing_tds' in result:
                    # FIX: Use MEAN not MEDIAN for TDs - median causes 0.0 for low-frequency events
                    pred['rushing_tds_mean'] = float(np.mean(result['rushing_tds']))
                    pred['rushing_tds_std'] = float(np.std(result['rushing_tds']))

                if 'receiving_yards' in result:
                    # FIX: Use MEAN not MEDIAN - gamma distribution is right-skewed, median << mean
                    pred['receiving_yards_mean'] = float(np.mean(result['receiving_yards']))
                    pred['receiving_yards_std'] = float(np.std(result['receiving_yards']))

                if 'receptions' in result:
                    # FIX: Use MEAN not MEDIAN for discrete count data (Binomial)
                    pred['receptions_mean'] = float(np.mean(result['receptions']))
                    pred['receptions_std'] = float(np.std(result['receptions']))

                if 'targets' in result:
                    # FIX: Use MEAN not MEDIAN for discrete count data (Poisson)
                    pred['targets_mean'] = float(np.mean(result['targets']))
                    pred['targets_std'] = float(np.std(result['targets']))

                if 'receiving_tds' in result:
                    # FIX: Use MEAN not MEDIAN for TDs - median causes 0.0 for low-frequency events
                    pred['receiving_tds_mean'] = float(np.mean(result['receiving_tds']))
                    pred['receiving_tds_std'] = float(np.std(result['receiving_tds']))

                if 'anytime_td' in result:
                    pred['anytime_td_prob'] = float(np.mean(result['anytime_td'] > 0))

                # STATISTICAL BOUNDS VALIDATION (Framework Rule 8.3)
                # Validate projections against historical performance (3σ threshold)
                try:
                    from nfl_quant.validation import validate_player_projections

                    # Build projections dict from pred
                    projections_to_validate = {}
                    if 'receptions_mean' in pred:
                        projections_to_validate['receptions'] = pred['receptions_mean']
                    if 'receiving_yards_mean' in pred:
                        projections_to_validate['receiving_yards'] = pred['receiving_yards_mean']
                    if 'receiving_tds_mean' in pred:
                        projections_to_validate['receiving_tds'] = pred['receiving_tds_mean']
                    if 'rushing_attempts_mean' in pred:
                        projections_to_validate['rushing_attempts'] = pred['rushing_attempts_mean']
                    if 'rushing_yards_mean' in pred:
                        projections_to_validate['rushing_yards'] = pred['rushing_yards_mean']
                    if 'rushing_tds_mean' in pred:
                        projections_to_validate['rushing_tds'] = pred['rushing_tds_mean']
                    if 'pass_attempts_mean' in pred:
                        projections_to_validate['pass_attempts'] = pred['pass_attempts_mean']
                    if 'pass_completions_mean' in pred:
                        projections_to_validate['pass_completions'] = pred['pass_completions_mean']
                    if 'pass_yards_mean' in pred:
                        projections_to_validate['pass_yards'] = pred['pass_yards_mean']
                    if 'pass_tds_mean' in pred:
                        projections_to_validate['pass_tds'] = pred['pass_tds_mean']
                    if 'interceptions_mean' in pred:
                        projections_to_validate['interceptions'] = pred['interceptions_mean']

                    # Validate projections (apply_caps=True will automatically cap extreme values)
                    if projections_to_validate:
                        capped_projections, validation_results = validate_player_projections(
                            player_name, position, projections_to_validate, apply_caps=True
                        )

                        # Apply capped values back to pred
                        if 'receptions' in capped_projections:
                            pred['receptions_mean'] = capped_projections['receptions']
                        if 'receiving_yards' in capped_projections:
                            pred['receiving_yards_mean'] = capped_projections['receiving_yards']
                        if 'receiving_tds' in capped_projections:
                            pred['receiving_tds_mean'] = capped_projections['receiving_tds']
                        if 'rushing_attempts' in capped_projections:
                            pred['rushing_attempts_mean'] = capped_projections['rushing_attempts']
                        if 'rushing_yards' in capped_projections:
                            pred['rushing_yards_mean'] = capped_projections['rushing_yards']
                        if 'rushing_tds' in capped_projections:
                            pred['rushing_tds_mean'] = capped_projections['rushing_tds']
                        if 'pass_attempts' in capped_projections:
                            pred['pass_attempts_mean'] = capped_projections['pass_attempts']
                        if 'pass_completions' in capped_projections:
                            pred['pass_completions_mean'] = capped_projections['pass_completions']
                        if 'pass_yards' in capped_projections:
                            pred['pass_yards_mean'] = capped_projections['pass_yards']
                        if 'pass_tds' in capped_projections:
                            pred['pass_tds_mean'] = capped_projections['pass_tds']
                        if 'interceptions' in capped_projections:
                            pred['interceptions_mean'] = capped_projections['interceptions']

                        # Log any warnings for projections that were flagged
                        for stat, result in validation_results.items():
                            if not result['is_valid']:
                                logger.warning(f"   ⚠️  {player_name} ({position}): {result['message']}")

                except Exception as e:
                    # Don't fail if validation module unavailable
                    logger.debug(f"   Statistical bounds validation skipped for {player_name}: {e}")

                predictions.append(pred)

            except Exception as e:
                logger.warning(f"   ⚠️  Error simulating {player_name} ({team}): {e}")
                logger.debug(f"   Full traceback:", exc_info=True)
                skipped_count += 1
                reason = f"Simulation error: {str(e)[:50]}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

    # Log filtering summary
    simulated_count = len(predictions)
    logger.info(f"\n📊 Player Filtering Summary:")
    logger.info(f"   Total players in odds: {len(players_df)}")
    logger.info(f"   Skipped (incomplete data/errors): {skipped_count}")
    logger.info(f"   Successfully simulated: {simulated_count}")
    if skip_reasons:
        logger.info(f"\n   Top reasons for skipping:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"      {reason}: {count}")

    logger.info(f"   ✅ Generated predictions for {len(predictions)} players ({_time.time() - step_start:.1f}s)")

    # Create DataFrame
    df = pd.DataFrame(predictions)

    if df.empty:
        logger.error("   ❌ No predictions generated")
        return pd.DataFrame()

    # ENHANCE TD PREDICTIONS with statistical model
    logger.info("\n7. Enhancing TD predictions with statistical model...")
    df = enhance_td_predictions(df, week, game_context, season=season)

    # NORMALIZE TD PREDICTIONS to match game totals
    logger.info("\n8. Normalizing TD predictions to match game totals...")
    df = normalize_tds_to_game_totals(df, week, season=season)

    # CALIBRATE TD PREDICTIONS
    logger.info("\n9. Calibrating TD predictions...")
    df = calibrate_td_predictions(df)

    # INTEGRATE ALL FACTORS (EPA, Weather, Divisional, Contextual, Injuries, etc.)
    logger.info("\n10. Integrating ALL factors via unified integration...")
    try:
        # Load odds data for integration (needed for opponent info)
        odds_file = Path('data/nfl_player_props_draftkings.csv')
        if odds_file.exists():
            odds_df = pd.read_csv(odds_file)
            # Filter to current week if week column exists
            if 'week' in odds_df.columns:
                odds_df = odds_df[odds_df['week'] == week]
        else:
            odds_df = pd.DataFrame()
            logger.warning("   ⚠️  Odds file not found - some factors may not integrate correctly")

        # Integrate all factors
        columns_before = set(df.columns)
        df = integrate_all_factors(
            week=week,
            season=season,
            players_df=df,
            odds_df=odds_df,
            fail_on_missing=False  # Don't fail if some factors unavailable
        )
        columns_after = set(df.columns)
        new_columns = columns_after - columns_before

        if new_columns:
            logger.info(f"   ✅ All factors integrated - Added {len(new_columns)} columns")
            logger.info(f"   New columns: {sorted(new_columns)[:10]}..." if len(new_columns) > 10 else f"   New columns: {sorted(new_columns)}")
        else:
            logger.warning("   ⚠️  Integration completed but no new columns added")

        # Verify critical columns were added
        critical_columns = [
            'opponent_def_epa_vs_position',
            'weather_total_adjustment',
            'is_divisional_game',
            'rest_epa_adjustment',
            'travel_epa_adjustment'
        ]
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            logger.warning(f"   ⚠️  Missing critical columns: {missing_critical}")
        else:
            logger.info("   ✅ All critical columns verified")
    except Exception as e:
        logger.error(f"   ❌ Unified integration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("   Continuing with predictions (some factors may be missing)")

    # MERGE MICRO METRICS (team-level advanced features)
    logger.info("\n10b. Merging micro metrics into predictions...")
    try:
        if _micro_metrics_cache is not None and not _micro_metrics_cache.empty and 'team' in df.columns:
            # Merge on player's team
            micro_cols_before = len(df.columns)
            df = df.merge(
                _micro_metrics_cache,
                on='team',
                how='left'
            )
            micro_cols_added = len(df.columns) - micro_cols_before
            logger.info(f"   ✅ Merged {micro_cols_added} micro metric columns")

            # Also merge opponent metrics if opponent column exists
            if 'opponent' in df.columns:
                opp_metrics = _micro_metrics_cache.copy()
                opp_metrics.columns = ['opp_' + c if c != 'team' else 'opponent' for c in opp_metrics.columns]
                df = df.merge(opp_metrics, on='opponent', how='left')
                logger.info(f"   ✅ Merged opponent micro metrics")
        else:
            if _micro_metrics_cache is None or _micro_metrics_cache.empty:
                logger.warning("   ⚠️  No micro metrics available to merge")
            elif 'team' not in df.columns:
                logger.warning("   ⚠️  'team' column not in predictions, cannot merge micro metrics")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not merge micro metrics: {e}")

    # APPLY ROLE CHANGE OVERRIDES - Critical for backup-to-starter transitions
    logger.info("\n11. Applying role change overrides...")
    try:
        df = apply_role_overrides(df)
    except Exception as e:
        logger.error(f"   ❌ Role override application failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("   Continuing without role overrides")

    # ADD OPPONENT-AWARE FEATURES (V13 support)
    logger.info("\n12. Adding opponent-aware features for V13 model...")
    try:
        from nfl_quant.features.opponent_stats import (
            calculate_vs_opponent_stats_time_aware,
            calculate_opponent_divergence
        )

        # Load historical player stats with opponent info (include 2023 for more opponent history)
        project_root = Path(__file__).parent.parent.parent
        stats_path_2024_2025 = project_root / 'data' / 'nflverse' / 'player_stats_2024_2025.csv'
        stats_path_2023 = project_root / 'data' / 'nflverse' / 'player_stats_2023.csv'

        historical_dfs = []
        if stats_path_2024_2025.exists():
            historical_dfs.append(pd.read_csv(stats_path_2024_2025, low_memory=False))
            logger.info(f"   Loaded {len(historical_dfs[-1])} rows from 2024-2025 stats")
        if stats_path_2023.exists():
            historical_dfs.append(pd.read_csv(stats_path_2023, low_memory=False))
            logger.info(f"   Loaded {len(historical_dfs[-1])} rows from 2023 stats")

        if historical_dfs:
            historical_stats = pd.concat(historical_dfs, ignore_index=True)
            logger.info(f"   Combined {len(historical_stats)} total historical records (2023-2025)")

            # Normalize player names for matching
            from nfl_quant.utils.player_names import normalize_player_name
            historical_stats['player_norm'] = historical_stats['player_display_name'].apply(normalize_player_name)

            # Calculate time-aware opponent stats
            stats_with_opp = calculate_vs_opponent_stats_time_aware(historical_stats)

            # For each prediction, lookup opponent history
            opp_features_added = 0
            for idx, row in df.iterrows():
                player_name = row.get('player_name', '')
                opponent = row.get('opponent', '')

                if not opponent or opponent == 'UNK':
                    continue

                # Normalize player name for lookup
                player_norm = normalize_player_name(player_name)

                # Find historical stats vs this opponent
                player_opp_stats = stats_with_opp[
                    (stats_with_opp['player_norm'] == player_norm) &
                    (stats_with_opp['opponent_team'] == opponent)
                ]

                if len(player_opp_stats) == 0:
                    continue

                # Get most recent (latest global_week)
                latest = player_opp_stats.sort_values('global_week', ascending=False).iloc[0]

                # Add vs_opponent features for each stat type
                for stat in ['receptions', 'receiving_yards', 'rushing_yards', 'passing_yards', 'completions']:
                    vs_avg_col = f'vs_opp_avg_{stat}'
                    vs_games_col = f'vs_opp_games_{stat}'

                    if vs_avg_col in latest.index and pd.notna(latest[vs_avg_col]):
                        df.at[idx, vs_avg_col] = latest[vs_avg_col]
                        df.at[idx, vs_games_col] = latest.get(vs_games_col, 0)
                        opp_features_added += 1

            logger.info(f"   ✅ Added {opp_features_added} opponent-specific features")

            # Calculate divergence for key markets
            # This compares vs_opponent avg to overall trailing avg
            for market, trailing_col in [
                ('player_receptions', 'receptions_mean'),
                ('player_reception_yds', 'receiving_yards_mean'),
                ('player_rush_yds', 'rushing_yards_mean'),
                ('player_pass_yds', 'passing_yards_mean'),
            ]:
                stat_map = {
                    'player_receptions': 'receptions',
                    'player_reception_yds': 'receiving_yards',
                    'player_rush_yds': 'rushing_yards',
                    'player_pass_yds': 'passing_yards',
                }
                stat = stat_map.get(market)
                vs_avg_col = f'vs_opp_avg_{stat}'
                vs_games_col = f'vs_opp_games_{stat}'

                if vs_avg_col in df.columns and trailing_col in df.columns:
                    # Calculate divergence: (vs_opp - trailing) / trailing
                    df[f'{market}_vs_opp_divergence'] = (
                        (df[vs_avg_col] - df[trailing_col]) / df[trailing_col].clip(lower=0.1)
                    ).clip(lower=-1, upper=1).fillna(0)

                    # Confidence based on games played
                    if vs_games_col in df.columns:
                        df[f'{market}_vs_opp_confidence'] = (
                            (df[vs_games_col] - 1).clip(lower=0) / 3
                        ).clip(upper=1).fillna(0)
                    else:
                        df[f'{market}_vs_opp_confidence'] = 0

            logger.info("   ✅ Calculated opponent divergence features")

            # CRITICAL: Apply opponent adjustment to Monte Carlo means
            # This makes opponent history PART OF the prediction, not just a warning
            logger.info("\n   Applying opponent adjustments to projections...")

            adjustments_applied = 0
            adjustment_details = []

            # Map markets to their mean columns
            market_mean_map = {
                'player_receptions': 'receptions_mean',
                'player_reception_yds': 'receiving_yards_mean',
                'player_rush_yds': 'rushing_yards_mean',
                'player_pass_yds': 'passing_yards_mean',
            }

            for market, mean_col in market_mean_map.items():
                div_col = f'{market}_vs_opp_divergence'
                conf_col = f'{market}_vs_opp_confidence'
                std_col = mean_col.replace('_mean', '_std')

                if div_col not in df.columns or mean_col not in df.columns:
                    continue

                # Create adjustment columns
                df[f'{market}_opponent_adjustment'] = 1.0  # Default: no adjustment
                df[f'{market}_original_mean'] = df[mean_col].copy()  # Store original

                # Apply adjustment where confidence is sufficient (2+ prior games)
                mask = (df[conf_col] >= 0.33) & (df[div_col].abs() > 0.05)  # 5% min divergence

                if mask.any():
                    # Calculate adjustment factor: 1 + divergence
                    # Divergence of +0.45 means player avg 45% MORE vs opponent
                    # So we multiply mean by 1.45
                    adjustment_factor = 1 + df.loc[mask, div_col]

                    # Store adjustment factor FOR DISPLAY ONLY (do not apply to projections)
                    # Real results showed opponent adjustments add noise, not signal
                    # LVT remains the primary signal - opponent data is informational only
                    df.loc[mask, f'{market}_opponent_adjustment'] = adjustment_factor

                    # DO NOT apply to mean - keep original LVT-based projection
                    # df.loc[mask, mean_col] = df.loc[mask, mean_col] * adjustment_factor

                    # DO NOT apply to std
                    # if std_col in df.columns:
                    #     df.loc[mask, std_col] = df.loc[mask, std_col] * adjustment_factor.abs()

                    adjustments_applied += mask.sum()

                    # Log notable adjustments
                    notable = df[mask & (df[div_col].abs() > 0.20)]
                    for _, row in notable.head(3).iterrows():
                        player = row.get('player_name', 'Unknown')
                        orig = row.get(f'{market}_original_mean', 0)
                        adj = row.get(mean_col, 0)
                        div = row.get(div_col, 0)
                        opp = row.get('opponent', 'UNK')
                        adjustment_details.append(
                            f"      {player} vs {opp}: {orig:.1f} → {adj:.1f} ({div:+.0%} adjustment)"
                        )

            logger.info(f"   ✅ Added opponent history for {adjustments_applied} player-markets (INFO ONLY - no projection changes)")
            if adjustment_details:
                logger.info("   Notable adjustments:")
                for detail in adjustment_details[:10]:  # Limit to 10
                    logger.info(detail)

        else:
            logger.warning(f"   ⚠️  Historical stats not found")

    except ImportError as e:
        logger.warning(f"   ⚠️  Opponent stats module not available: {e}")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not add opponent features: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Save predictions
    output_file = Path(f'data/model_predictions_week{week}.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    # PERF FIX (Dec 14, 2025): Log total pipeline time
    total_time = _time.time() - pipeline_start
    logger.info(f"\n✅ Saved {len(df)} predictions to {output_file}")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   ⏱️  Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return df


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate model predictions for all players"
    )
    parser.add_argument(
        'week',
        type=int,
        nargs='?',
        default=9,
        help='Week number to generate predictions for'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=None,
        help='NFL season year (default: auto-detect from current date)'
    )
    parser.add_argument(
        '--enable-regime',
        action='store_true',
        help='Enable regime-aware trailing stats (experimental)'
    )
    parser.add_argument(
        '--simulator',
        type=str,
        choices=['v3', 'v4'],
        default='v3',
        help='Simulator version: v3 (legacy Normal distributions) or v4 (NegBin+Lognormal+Copula)'
    )

    args = parser.parse_args()

    # Set environment variable for regime detection
    if args.enable_regime:
        os.environ['ENABLE_REGIME_DETECTION'] = '1'
        logger.info("=" * 80)
        logger.info("REGIME DETECTION ENABLED")
        logger.info("Using regime-aware dynamic windows instead of fixed 4-week windows")
        logger.info("=" * 80)
    else:
        os.environ['ENABLE_REGIME_DETECTION'] = '0'

    # Log simulator version
    if args.simulator == 'v4':
        logger.info("=" * 80)
        logger.info("V4 SIMULATOR ENABLED")
        logger.info("Using Negative Binomial + Lognormal + Gaussian Copula")
        logger.info("Output includes full percentile distributions (p5, p25, p50, p75, p95)")
        logger.info("=" * 80)

    predictions = generate_model_predictions(args.week, season=args.season, simulator_version=args.simulator)

    if predictions.empty:
        logger.error("\n❌ Failed to generate predictions")
        sys.exit(1)

    logger.info("\n✅ Model predictions generation complete!")
    logger.info(f"   File: data/model_predictions_week{args.week}.csv")
    logger.info(f"   Players: {len(predictions)}")
    logger.info("\n💡 Next step: Run generate_current_week_recommendations.py")


if __name__ == '__main__':
    main()
