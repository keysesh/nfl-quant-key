#!/usr/bin/env python3
"""
Backtest NFL player prop bets using The Odds API historical lines.

This script evaluates how the NFL Quant simulator would have performed versus
actual outcomes for a range of weeks. It expects:
  1. NFLverse weekly stats (data/nflverse_cache/stats_player_week_{season}.csv)
  2. Week-specific trailing stats (data/week_specific_trailing_stats.json)
  3. Historical prop odds exported via fetch_historical_player_props.py
     (saved under data/historical/)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.constants import FALLBACK_VALUES, RECEIVING_POSITIONS, RUSHING_POSITIONS
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.utils.season_utils import get_current_season

# Try to load defensive stats integration and unified integration
try:
    from nfl_quant.utils.defensive_stats_integration import get_defensive_epa_for_player
    DEFENSIVE_STATS_AVAILABLE = True
except ImportError:
    DEFENSIVE_STATS_AVAILABLE = False
    logger.warning("Defensive stats integration not available")

try:
    from nfl_quant.utils.unified_integration import integrate_all_factors
    UNIFIED_INTEGRATION_AVAILABLE = True
except ImportError:
    UNIFIED_INTEGRATION_AVAILABLE = False
    logger.warning("Unified integration not available")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("player_prop_backtest")
logging.getLogger("nfl_quant").setLevel(logging.WARNING)


MARKET_CONFIG = {
    "player_pass_yds": {
        "sim_key": "passing_yards",
        "actual_column": "pass_yd",
        "positions": {"QB"},
    },
    "player_pass_completions": {
        "sim_key": "passing_completions",
        "actual_column": "pass_cmp",
        "positions": {"QB"},
    },
    "player_rush_yds": {
        "sim_key": "rushing_yards",
        "actual_column": "rush_yd",
        "positions": {"RB", "QB"},
    },
    "player_reception_yds": {
        "sim_key": "receiving_yards",
        "actual_column": "rec_yd",
        "positions": set(RECEIVING_POSITIONS),
    },
    "player_receptions": {
        "sim_key": "receptions",
        "actual_column": "rec",
        "positions": set(RECEIVING_POSITIONS),
    },
    "player_anytime_td": {
        "sim_key": "anytime_td",
        "actual_column": "td_scorer",  # Will calculate from TD columns
        "positions": {"QB", "RB", "WR", "TE"},
    },
    "player_1st_td": {
        "sim_key": "first_td",
        "actual_column": "td_scorer",  # Will calculate from TD columns
        "positions": {"QB", "RB", "WR", "TE"},
    },
    "player_pass_tds": {
        "sim_key": "passing_tds",
        "actual_column": "pass_td",
        "positions": {"QB"},
    },
}


def normalize_player_name(name: str) -> str:
    """Normalize player name by removing suffixes and standardizing format."""
    import re
    name = str(name).strip().lower()
    # Remove suffixes like jr., sr., ii, iii, iv
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', name, flags=re.IGNORECASE)
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def american_to_profit(odds: float) -> float:
    """Return profit for a 1 unit stake at American odds."""
    if odds is None or pd.isna(odds):
        return 0.0
    odds = float(odds)
    if odds >= 0:
        return odds / 100.0
    return 100.0 / -odds


def load_schedule_map(schedule_path: Path) -> Dict[str, int]:
    """Map Odds API game_id (YYYYMMDD_AWAY_HOME) to NFL week number."""
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule file not found at {schedule_path}")

    with open(schedule_path) as f:
        games = json.load(f)

    mapping = {}
    for game in games:
        date = game["date"].replace("-", "")
        game_id = f"{date}_{game['away']}_{game['home']}"
        mapping[game_id] = game["week"]
    return mapping


def load_weather_data(week: int) -> pd.DataFrame:
    """Load weather data for the week."""
    weather_dir = Path('data/weather')
    if not weather_dir.exists():
        return pd.DataFrame()

    weather_files = list(weather_dir.glob(f'weather_week{week}_*.csv'))
    if len(weather_files) == 0:
        return pd.DataFrame()

    return pd.read_csv(sorted(weather_files)[-1])


def apply_weather_adjustment(model_projection: float, market: str, weather_data: dict, position: str) -> float:
    """Apply weather adjustment to model projection."""
    if not weather_data or weather_data.get('total_adjustment', 0.0) == 0.0:
        return model_projection

    total_adj = weather_data.get('total_adjustment', 0.0)
    passing_adj = weather_data.get('passing_adjustment', 0.0)

    if market in ['player_pass_yds', 'player_pass_completions', 'player_pass_attempts']:
        multiplier = 1.0 + passing_adj if abs(passing_adj) > abs(total_adj) else 1.0 + total_adj
    elif market in ['player_reception_yds', 'player_receptions']:
        if position in ['QB', 'WR', 'TE']:
            multiplier = 1.0 + passing_adj if abs(passing_adj) > abs(total_adj) else 1.0 + total_adj
        else:
            multiplier = 1.0 + total_adj
    elif market in ['player_rush_yds', 'player_rush_attempts']:
        multiplier = 1.0 + total_adj
    else:
        multiplier = 1.0 + total_adj

    multiplier = max(0.7, min(1.1, multiplier))
    return model_projection * multiplier


def load_game_script_data(week: int) -> dict:
    """Load game script data from simulation results."""
    import glob
    game_context = {}
    sim_files = sorted(glob.glob(f'reports/sim_*_week{week}_*.json'))

    for sim_file in sim_files:
        try:
            with open(sim_file) as f:
                sim_data = json.load(f)

            home_team = sim_data.get('home_team', '')
            away_team = sim_data.get('away_team', '')

            if not home_team or not away_team:
                continue

            home_score = sim_data.get('home_score_median', 24.0)
            away_score = sim_data.get('away_score_median', 21.0)
            game_script_home = home_score - away_score
            pace = sim_data.get('pace', 29.0)

            game_context[home_team] = {
                'team_total': home_score,
                'opponent_total': away_score,
                'game_script': game_script_home,
                'pace': pace
            }

            game_context[away_team] = {
                'team_total': away_score,
                'opponent_total': home_score,
                'game_script': -game_script_home,
                'pace': pace
            }
        except Exception:
            continue

    return game_context


def load_week_specific_stats(path: Path) -> Dict[str, dict]:
    if not path.exists():
        logger.warning("⚠️  week_specific_trailing_stats.json not found - using fallback values only.")
        return {}
    with open(path) as f:
        return json.load(f)


def create_player_input(
    player_name: str,
    team: str,
    position: str,
    week: int,
    trailing_stats: Dict[str, dict],
    game_context: dict = None,
    season: int = None
) -> PlayerPropInput:
    if season is None:
        season = get_current_season()
    # First try exact week, then fall back to most recent week's data
    key = f"{player_name}_week{week}"
    hist = trailing_stats.get(key, {})

    # If exact week not found, use most recent week available
    if not hist and trailing_stats:
        available_weeks = [int(k.split('_week')[1]) for k in trailing_stats.keys()
                          if k.startswith(f"{player_name}_week") and k.split('_week')[1].isdigit()]
        if available_weeks:
            most_recent_week = max(available_weeks)
            key_fallback = f"{player_name}_week{most_recent_week}"
            hist = trailing_stats.get(key_fallback, {})

    trailing_snap_share = hist.get("trailing_snap_share", FALLBACK_VALUES["snap_share"])

    trailing_target_share = hist.get("trailing_target_share")
    if trailing_target_share is None and position in RECEIVING_POSITIONS:
        trailing_target_share = FALLBACK_VALUES["target_share"]

    trailing_carry_share = hist.get("trailing_carry_share")
    if trailing_carry_share is None and position in RUSHING_POSITIONS:
        trailing_carry_share = FALLBACK_VALUES["carry_share"] if position == "RB" else 0.0

    trailing_yards_per_opportunity = hist.get(
        "trailing_yards_per_opportunity", FALLBACK_VALUES["yards_per_opportunity"]
    )
    trailing_td_rate = hist.get("trailing_td_rate", FALLBACK_VALUES["td_rate"])

    opponent = hist.get("opponent", "UNK")

    # Use game context from simulations if available, otherwise use defaults
    if game_context and team in game_context:
        ctx = game_context[team]
        projected_tot = ctx.get('team_total', 25.0)
        opponent_tot = ctx.get('opponent_total', 22.0)
        game_script = ctx.get('game_script', projected_tot - opponent_tot)
        pace = ctx.get('pace', 28.0)
    else:
        projected_tot = hist.get("projected_team_total", 25.0)
        opponent_tot = hist.get("projected_opponent_total", 22.0)
        game_script = hist.get("projected_game_script", projected_tot - opponent_tot)
        pace = hist.get("projected_pace", 28.0)

    # Get defensive EPA - use unified integration if available, otherwise calculate individually
    opponent_def_epa = hist.get("opponent_def_epa_vs_position", None)

    # If not in stats, try to calculate from PBP
    if opponent_def_epa is None or opponent_def_epa == 0.0:
        if DEFENSIVE_STATS_AVAILABLE and opponent != "UNK":
            try:
                opponent_def_epa = get_defensive_epa_for_player(opponent, position, week)
            except Exception:
                opponent_def_epa = 0.0  # Explicit 0.0 if calculation fails (not a default)
        else:
            opponent_def_epa = 0.0  # Explicit 0.0 if defensive stats unavailable

    # Note: For full integration of ALL factors, use unified_integration.integrate_all_factors()
    # on a batch of players before creating PlayerPropInput. This function only handles EPA
    # to avoid performance issues from per-player integration calls.

    # Create base player input
    player_input = PlayerPropInput(
        player_id=player_name,
        player_name=player_name,
        team=team,
        position=position,
        week=week,
        opponent=opponent or "UNK",
        projected_team_total=float(projected_tot),
        projected_opponent_total=float(opponent_tot),
        projected_game_script=float(game_script),
        projected_pace=float(pace),
        trailing_snap_share=float(trailing_snap_share),
        trailing_target_share=trailing_target_share,
        trailing_carry_share=trailing_carry_share,
        trailing_yards_per_opportunity=float(trailing_yards_per_opportunity),
        trailing_td_rate=float(trailing_td_rate),
        opponent_def_epa_vs_position=float(opponent_def_epa),
    )

    # Enhance with contextual factors if available
    try:
        from nfl_quant.utils.contextual_integration import enhance_player_input_with_context
        # Try to load PBP data for contextual factors
        pbp_path = Path(f'data/processed/pbp_{season}.parquet')
        if pbp_path.exists():
            pbp_df = pd.read_parquet(pbp_path)
            # Load injury data if available
            from nfl_quant.utils.contextual_integration import load_injury_data
            injury_data = load_injury_data(week)
            team_injury_data = injury_data.get(team, {})

            player_input = enhance_player_input_with_context(
                player_input,
                pbp_df=pbp_df,
                injury_data=team_injury_data if team_injury_data else None
            )
    except Exception as e:
        logger.debug(f"Could not enhance with contextual factors: {e}")
        # Continue with base input

    return player_input


def select_latest_snapshots(props: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest line per (event, market, player, prop_type)."""
    props = props.copy()
    props["snapshot_timestamp"] = pd.to_datetime(props["snapshot_timestamp"], errors="coerce")
    props.sort_values("snapshot_timestamp", inplace=True)
    dedup_cols = ["event_id", "market", "player", "prop_type"]
    props = props.drop_duplicates(subset=dedup_cols, keep="last")
    return props


def load_historical_props(directory: Path) -> pd.DataFrame:
    """Load all historical prop files from directory (searches recursively)."""
    # Search recursively for player_props_history_*.csv files
    files = sorted(directory.rglob("player_props_history_*.csv"))
    if not files:
        raise FileNotFoundError(f"No historical prop files found in {directory} (searched recursively)")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["commence_time"] = pd.to_datetime(combined.get("commence_time"))
    return combined


def build_simulator() -> PlayerSimulator:
    usage_predictor, efficiency_predictor = load_predictors()

    # Load main calibrator (for yards/receptions)
    calibrator = NFLProbabilityCalibrator()
    calibrator_path = Path("configs/calibrator.json")
    if calibrator_path.exists():
        calibrator.load(str(calibrator_path))

    # Load position-specific TD calibrators
    from nfl_quant.calibration.td_calibrator_loader import get_td_calibrator_loader

    td_loader = get_td_calibrator_loader()

    # Fallback to old unified calibrator if position-specific not available
    td_calibrator = None
    if not td_loader.is_available():
        td_calibrator_path = Path("data/models/td_calibrator_v1.joblib")
        if td_calibrator_path.exists():
            import joblib
            td_calibrator = joblib.load(td_calibrator_path)
            logger.info(f"✅ Loaded fallback TD calibrator from {td_calibrator_path}")
        else:
            logger.warning(f"⚠️  No TD calibrators found")
    else:
        loaded_positions = [pos for pos in ['QB', 'RB', 'WR', 'TE'] if td_loader.get_calibrator(pos)]
        logger.info(f"✅ Loaded position-specific TD calibrators: {', '.join(loaded_positions)}")
        # Store loader for use in calibration (will be accessed via position)
        td_calibrator = td_loader

    return PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=5000,
        seed=42,
        calibrator=calibrator,
        td_calibrator=td_calibrator,  # Pass TD calibrator loader or fallback calibrator
    )


def run_backtest(
    start_week: int,
    end_week: int,
    season: int,
    prop_files_dir: Path,
    min_edge: float,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    schedule_map = load_schedule_map(Path("data/raw/sleeper_games.json"))
    trailing_stats = load_week_specific_stats(Path("data/week_specific_trailing_stats.json"))
    props = load_historical_props(prop_files_dir)

    props["week"] = props["game_id"].map(schedule_map)
    props = props[props["week"].between(start_week, end_week)]
    props = props[props["market"].isin(MARKET_CONFIG.keys())]
    props = props[props["prop_type"].isin(["over", "under"])]
    props = select_latest_snapshots(props)

    if props.empty:
        raise ValueError("No historical props found for the requested weeks/markets.")

    simulator = build_simulator()

    results: List[dict] = []
    all_evaluations: List[dict] = []  # Track ALL props for audit
    simulation_cache: Dict[Tuple[int, str, str], Dict[str, np.ndarray]] = {}

    # Debug counters
    debug_stats = {
        "total_props": 0,
        "no_actual_row": 0,
        "wrong_position": 0,
        "no_actual_value": 0,
        "no_sim_result": 0,
        "no_american_price": 0,
        "no_implied_prob": 0,
        "edge_too_low": 0,
        "confidence_too_low": 0,
        "passed": 0
    }

    # Load all stats once from NFLverse cache
    stats_path = Path(f"data/nflverse_cache/stats_player_week_{season}.csv")
    if not stats_path.exists():
        logger.error(f"⚠️  Missing NFLverse stats file: {stats_path}")
        raise FileNotFoundError(f"NFLverse stats not found: {stats_path}")

    all_stats_df = pd.read_csv(stats_path)
    all_stats_df["player_key"] = all_stats_df["player_name"].apply(normalize_player_name)
    all_stats_df["team"] = all_stats_df["team"].str.upper()

    for week in range(start_week, end_week + 1):
        # Filter to specific week
        stats_df = all_stats_df[all_stats_df['week'] == week].copy()

        if stats_df.empty:
            logger.warning(f"⚠️  No stats for week {week}")
            continue

        stats_lookup = stats_df.set_index(["player_key", "team"]).sort_index()

        # Load weather and game context for this week
        weather_df = load_weather_data(week)
        game_context = load_game_script_data(week)

        if not weather_df.empty:
            logger.info(f"   🌤️  Loaded weather data for {len(weather_df)} teams")
        if game_context:
            logger.info(f"   📊 Loaded game context for {len(game_context)} teams")

        week_props = props[props["week"] == week]
        logger.info(f"\n📅 Week {week}: evaluating {len(week_props)} prop lines")

        debug_stats["total_props"] += len(week_props)

        for _, row in week_props.iterrows():
            player_name = str(row["player"]).strip()
            player_key = normalize_player_name(player_name)
            market = row["market"]
            config = MARKET_CONFIG[market]

            # Determine player team from lines + stats
            candidate_teams = [row["home_team"], row["away_team"]]
            actual_row = None
            matched_team = None
            for team in candidate_teams:
                key = (player_key, str(team).upper())
                if key in stats_lookup.index:
                    actual_row = stats_lookup.loc[key]
                    if isinstance(actual_row, pd.DataFrame):
                        actual_row = actual_row.iloc[0]
                    matched_team = str(team).upper()
                    break

            if actual_row is None:
                debug_stats["no_actual_row"] += 1
                continue

            position = actual_row["position"]
            if position not in config["positions"]:
                debug_stats["wrong_position"] += 1
                continue

            actual_value = actual_row[config["actual_column"]]
            if pd.isna(actual_value):
                debug_stats["no_actual_value"] += 1
                continue

            cache_key = (week, player_key, matched_team)
            if cache_key not in simulation_cache:
                player_input = create_player_input(
                    player_name=actual_row["player_name"],
                    team=matched_team,
                    position=position,
                    week=week,
                    trailing_stats=trailing_stats,
                    game_context=game_context,
                    season=season,
                )
                try:
                    sim_result = simulator.simulate_player(player_input)
                    simulation_cache[cache_key] = sim_result
                    if len(sim_result) == 0:
                        logger.warning(f"Empty simulation result for {player_name} ({position})")
                    else:
                        logger.info(f"Simulated {player_name} ({position}): keys={list(sim_result.keys())[:5]}")
                except Exception as exc:
                    logger.warning(f"Simulation failed for {player_name} ({position}): {exc}")
                    simulation_cache[cache_key] = {}

            sim_result = simulation_cache.get(cache_key, {})

            # Handle TD markets separately
            if market in ["player_anytime_td", "player_1st_td"]:
                # Calculate TD probability from distributions
                prob_any_td = 0.0
                if position == "QB":
                    pass_tds = sim_result.get("passing_tds", np.array([0]))
                    rush_tds = sim_result.get("rushing_tds", np.array([0]))
                    prob_any_td = float(np.mean((pass_tds > 0) | (rush_tds > 0)))
                elif position == "RB":
                    rush_tds = sim_result.get("rushing_tds", np.array([0]))
                    rec_tds = sim_result.get("receiving_tds", np.array([0]))
                    prob_any_td = float(np.mean((rush_tds > 0) | (rec_tds > 0)))
                elif position in ["WR", "TE"]:
                    rec_tds = sim_result.get("receiving_tds", np.array([0]))
                    prob_any_td = float(np.mean(rec_tds > 0))

                # For 1st TD, reduce probability (very low probability event)
                if market == "player_1st_td":
                    prob_any_td = prob_any_td * 0.15  # Rough estimate: 15% of TD scorers get first TD

                # Apply weather adjustment
                if not weather_df.empty and matched_team in weather_df["team"].values:
                    team_weather = weather_df[weather_df["team"] == matched_team].iloc[0].to_dict()
                    weather_multiplier = max(0.8, min(1.0, 1.0 + team_weather.get('total_adjustment', 0.0)))
                    prob_any_td = prob_any_td * weather_multiplier

                # Calculate actual TD scorer (1 if any TD, 0 otherwise)
                actual_value = 1.0 if (actual_row.get("rush_td", 0) > 0 or
                                      actual_row.get("rec_td", 0) > 0 or
                                      actual_row.get("pass_td", 0) > 0) else 0.0

                line = 0.5  # TD bets are yes/no
                model_prob = prob_any_td if row["prop_type"] == "yes" or row["prop_type"] == "over" else (1 - prob_any_td)
                bet_outcome = float(actual_value > 0.5) if row["prop_type"] == "yes" or row["prop_type"] == "over" else float(actual_value < 0.5)
                model_prob_raw = model_prob
            else:
                stat_key = config["sim_key"]
                if stat_key not in sim_result:
                    debug_stats["no_sim_result"] += 1
                    continue

                distribution = sim_result[stat_key]
                line = row.get("line")
                if pd.isna(line):
                    continue

                # Apply weather adjustment if available
                if not weather_df.empty and matched_team in weather_df["team"].values:
                    team_weather = weather_df[weather_df["team"] == matched_team].iloc[0].to_dict()
                    # Adjust distribution mean for weather
                    original_mean = np.mean(distribution)
                    adjusted_mean = apply_weather_adjustment(original_mean, market, team_weather, position)
                    if abs(adjusted_mean - original_mean) > 0.01:
                        # Shift distribution by the difference
                        adjustment = adjusted_mean - original_mean
                        distribution = distribution + adjustment

                actual_value = actual_row[config["actual_column"]]
                if pd.isna(actual_value):
                    continue

                if row["prop_type"] == "over":
                    model_prob_raw = float(np.mean(distribution > line))
                    bet_outcome = float(actual_value > line)
                else:
                    model_prob_raw = float(np.mean(distribution < line))
                    bet_outcome = float(actual_value < line)

                # Apply calibration - use TD calibrator for TD markets, main calibrator for others
                is_td_market = 'pass_tds' in market or 'rush_tds' in market or 'rec_tds' in market

                if is_td_market:
                    # For TD props: Use position-specific TD calibrator if available
                    from nfl_quant.calibration.td_calibrator_loader import PositionSpecificTDCalibratorLoader

                    if isinstance(simulator.td_calibrator, PositionSpecificTDCalibratorLoader):
                        # Use position-specific TD calibration
                        model_prob = simulator.td_calibrator.calibrate_td_probability(model_prob_raw, position)
                    elif simulator.td_calibrator is not None:
                        # Use fallback unified TD calibrator (sklearn format)
                        try:
                            if hasattr(simulator.td_calibrator, 'predict'):
                                model_prob = float(simulator.td_calibrator.predict([model_prob_raw])[0])
                            elif hasattr(simulator.td_calibrator, 'transform'):
                                model_prob = float(simulator.td_calibrator.transform([model_prob_raw])[0])
                            else:
                                model_prob = model_prob_raw
                        except Exception as e:
                            logger.warning(f"TD calibration failed for {player_name}, using raw prob: {e}")
                            model_prob = model_prob_raw
                    else:
                        # No TD calibrator available - use raw probabilities
                        model_prob = model_prob_raw
                elif simulator.calibrator and simulator.calibrator.is_fitted:
                    # Use main calibrator for yards/receptions
                    try:
                        model_prob = float(simulator.calibrator.transform(np.array([model_prob_raw]))[0])
                    except Exception as e:
                        logger.warning(f"Calibration failed for {player_name}, using raw prob: {e}")
                        model_prob = model_prob_raw
                else:
                    model_prob = model_prob_raw

            american_price = row.get("american_price")
            if pd.isna(american_price):
                american_price = row.get("price")

            if pd.isna(american_price):
                debug_stats["no_american_price"] += 1
                continue

            implied_prob = american_to_implied_prob(american_price)
            if pd.isna(implied_prob):
                debug_stats["no_implied_prob"] += 1
                continue

            edge = model_prob - implied_prob

            # Save ALL evaluations for audit (regardless of edge)
            all_evaluations.append({
                "week": week,
                "player": actual_row["player_name"],
                "team": matched_team,
                "position": position,
                "market": market,
                "prop_type": row["prop_type"],
                "line": line,
                "american_price": american_price,
                "model_prob": model_prob,
                "model_prob_raw": model_prob_raw if 'model_prob_raw' in locals() else model_prob,
                "implied_prob": implied_prob,
                "edge": edge,
                "filter_status": "positive_edge" if edge >= min_edge else "negative_edge",
            })

            if edge < min_edge:
                debug_stats["edge_too_low"] += 1
                continue

            # Filter by confidence: model_prob must be >= min_confidence or <= (1 - min_confidence)
            # This means we're confident either way (e.g., 95% confident over OR 95% confident under)
            if min_confidence > 0:
                if model_prob < min_confidence and model_prob > (1 - min_confidence):
                    debug_stats["confidence_too_low"] += 1
                    continue

            debug_stats["passed"] += 1

            # Determine bet outcome
            if market == "player_anytime_td":
                bet_outcome = float(actual_value > 0.5) if row["prop_type"] == "over" else float(actual_value < 0.5)
            else:
                bet_outcome = float(actual_value > line) if row["prop_type"] == "over" else float(actual_value < line)

            profit = american_to_profit(american_price)
            unit_return = profit if bet_outcome == 1.0 else (-1.0 if bet_outcome == 0.0 else 0.0)

            results.append(
                {
                    "week": week,
                    "player": actual_row["player_name"],
                    "team": matched_team,
                    "position": position,
                    "market": market,
                    "prop_type": row["prop_type"],
                    "line": line,
                    "american_price": american_price,
                    "model_prob": model_prob,
                    "model_prob_raw": model_prob_raw if 'model_prob_raw' in locals() else model_prob,
                    "implied_prob": implied_prob,
                    "edge": edge,
                    "actual_value": actual_value,
                    "bet_won": bet_outcome == 1.0,
                    "bet_push": actual_value == line,
                    "unit_return": unit_return if actual_value != line else 0.0,
                    "source_file": row.get("source_file"),
                }
            )

    # Log debug stats
    logger.info("\n🔍 Debug Statistics:")
    logger.info(f"  Total props evaluated: {debug_stats['total_props']}")
    logger.info(f"  No actual_row found: {debug_stats['no_actual_row']}")
    logger.info(f"  Wrong position: {debug_stats['wrong_position']}")
    logger.info(f"  No actual value: {debug_stats['no_actual_value']}")
    logger.info(f"  No sim result: {debug_stats['no_sim_result']}")
    logger.info(f"  No american price: {debug_stats['no_american_price']}")
    logger.info(f"  No implied prob: {debug_stats['no_implied_prob']}")
    logger.info(f"  Edge too low (<{min_edge:.2%}): {debug_stats['edge_too_low']}")
    if min_confidence > 0:
        logger.info(f"  Confidence too low (<{min_confidence:.0%} or >{1-min_confidence:.0%}): {debug_stats['confidence_too_low']}")
    logger.info(f"  ✅ Passed all filters: {debug_stats['passed']}")

    # Save audit file with ALL evaluations for manual review
    if all_evaluations:
        audit_df = pd.DataFrame(all_evaluations)
        audit_path = Path(f"reports/backtest_edge_audit_weeks{start_week}-{end_week}.csv")
        audit_df.to_csv(audit_path, index=False)
        logger.info(f"\n📊 Edge audit saved to: {audit_path}")
        logger.info(f"   - Total evaluations: {len(audit_df)}")
        logger.info(f"   - Positive edge: {len(audit_df[audit_df['filter_status']=='positive_edge'])}")
        logger.info(f"   - Negative edge: {len(audit_df[audit_df['filter_status']=='negative_edge'])}")

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> None:
    if results.empty:
        logger.info("⚠️  No bets passed the edge filter.")
        return

    total_bets = len(results)
    pushes = int(results["bet_push"].sum())
    settled = total_bets - pushes
    wins = int(results["bet_won"].sum())
    losses = settled - wins
    roi = results["unit_return"].sum() / max(settled, 1)

    logger.info("\n" + "=" * 80)
    logger.info("PLAYER PROP BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total bets: {total_bets}")
    logger.info(f"  Wins:   {wins}")
    logger.info(f"  Losses: {losses}")
    logger.info(f"  Pushes: {pushes}")
    logger.info(f"Hit rate: {wins / max(settled, 1):.1%}")
    logger.info(f"ROI: {roi:.2%}")

    market_summary = (
        results.groupby("market")
        .agg(
            bets=("market", "size"),
            wins=("bet_won", "sum"),
            pushes=("bet_push", "sum"),
            avg_edge=("edge", "mean"),
            roi=("unit_return", "mean"),
        )
        .sort_values("bets", ascending=False)
    )

    logger.info("\nMarket breakdown:")
    for market, row in market_summary.iterrows():
        effective_bets = row["bets"] - row["pushes"]
        hit_rate = row["wins"] / max(effective_bets, 1)
        logger.info(
            f"  • {market}: {int(row['bets'])} bets | hit {hit_rate:.1%} | "
            f"avg edge {row['avg_edge']:.2%} | ROI {row['roi']:.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest player props using historical odds.")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: auto-detect from current date)")
    parser.add_argument("--start-week", type=int, default=1, help="First week to backtest (inclusive).")
    parser.add_argument("--end-week", type=int, default=7, help="Last week to backtest (inclusive).")
    parser.add_argument(
        "--historical-dir",
        type=Path,
        default=Path("data/historical"),
        help="Directory containing player_props_history_*.csv files.",
    )
    parser.add_argument("--min-edge", type=float, default=0.03, help="Minimum edge threshold to bet.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold (0.0-1.0). Uses model_prob >= min_confidence or <= (1-min_confidence).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write detailed bet-level CSV results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_week > args.end_week:
        raise ValueError("start_week must be <= end_week")

    season = args.season if args.season is not None else get_current_season()
    logger.info(f"Running backtest for {season} season, weeks {args.start_week}-{args.end_week}")

    results = run_backtest(
        start_week=args.start_week,
        end_week=args.end_week,
        season=season,
        prop_files_dir=args.historical_dir,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence,
    )

    summarize(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        logger.info(f"\n💾 Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
