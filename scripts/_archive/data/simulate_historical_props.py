#!/usr/bin/env python3
"""
Step 3: Run Simulations on Historical Props

This script:
1. Loads matched historical props from Step 2
2. Loads SAME PlayerSimulator with usage/efficiency predictors
3. For each prop, creates PlayerPropInput with historical context
4. Runs simulation to get model_prob_raw
5. Saves expanded dataset with both model_prob_raw and bet_won

CRITICAL: Uses SAME pipeline as current predictions to preserve context.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc="", total=None):
        return iterable


def load_historical_props() -> pd.DataFrame:
    """Load matched historical props from Step 2."""
    props_file = Path("data/calibration/historical_props_matched.parquet")

    if not props_file.exists():
        raise FileNotFoundError(
            f"Historical props not found at {props_file}. "
            "Run match_historical_props.py first."
        )

    df = pd.read_parquet(props_file)
    logger.info(f"✅ Loaded {len(df):,} historical props")
    return df


def load_historical_trailing_stats() -> Dict:
    """
    Load or generate historical trailing stats for each player-week.

    For historical simulations, we need to compute trailing stats
    based on what was available at that point in time.
    """
    # Check if we have cached historical trailing stats
    cache_file = Path("data/nflverse/historical_trailing_stats.json")

    if cache_file.exists():
        logger.info("Loading cached historical trailing stats...")
        with open(cache_file) as f:
            return json.load(f)

    # Generate trailing stats from weekly data
    logger.info("Generating historical trailing stats...")
    weekly_df = pd.read_parquet("data/nflverse/weekly_historical.parquet")

    trailing_stats = {}

    # Group by player and calculate rolling stats
    for player_name in tqdm(weekly_df['player_display_name'].unique(),
                           desc="Processing players"):
        player_df = weekly_df[
            weekly_df['player_display_name'] == player_name
        ].sort_values(['season', 'week'])

        for idx, row in player_df.iterrows():
            season = row['season']
            week = row['week']
            position = row['position']
            team = row.get('recent_team', '')

            # Get previous games (trailing window)
            prev_games = player_df[
                ((player_df['season'] == season) & (player_df['week'] < week)) |
                (player_df['season'] < season)
            ].tail(4)  # Last 4 games

            if len(prev_games) == 0:
                continue

            # Calculate trailing averages (convert to native Python types for JSON)
            key = f"{player_name}_week{week}_season{season}"

            # Calculate efficiency metrics
            prev_targets = prev_games.get('targets', pd.Series([0]))
            prev_receptions = prev_games.get('receptions', pd.Series([0]))
            prev_receiving_yds = prev_games.get('receiving_yards', pd.Series([0]))
            prev_carries = prev_games.get('carries', pd.Series([0]))
            prev_rushing_yds = prev_games.get('rushing_yards', pd.Series([0]))
            prev_tds = prev_games.get('rushing_tds', pd.Series([0])) + prev_games.get('receiving_tds', pd.Series([0]))

            # Calculate yards per opportunity
            total_opportunities = prev_targets.sum() + prev_carries.sum()
            total_yards = prev_receiving_yds.sum() + prev_rushing_yds.sum()
            yards_per_opportunity = total_yards / total_opportunities if total_opportunities > 0 else 5.0

            # Calculate TD rate (TDs per game)
            games = len(prev_games)
            td_rate = prev_tds.sum() / games if games > 0 else 0.05

            trailing_stats[key] = {
                'player_name': player_name,
                'position': position,
                'team': team,
                'season': int(season),
                'week': int(week),
                # Snap share (normalized)
                'trailing_snap_share': float(prev_games.get('fantasy_points_ppr', pd.Series([0])).mean() / 20.0),  # Rough proxy
                # Target share
                'trailing_target_share': float(prev_targets.mean() / 10.0),
                # Carry share
                'trailing_carry_share': float(prev_carries.mean() / 20.0),
                # Efficiency metrics (REQUIRED FIELDS)
                'trailing_yards_per_opportunity': float(yards_per_opportunity),
                'trailing_td_rate': float(td_rate),
                # Game context defaults
                'projected_team_total': 25.0,
                'projected_opponent_total': 22.0,
                'projected_game_script': 3.0,
                'projected_pace': 28.0,
                # Additional metrics
                'avg_passing_yards': float(prev_games.get('passing_yards', pd.Series([0])).mean()),
                'avg_rushing_yards': float(prev_games.get('rushing_yards', pd.Series([0])).mean()),
                'avg_receiving_yards': float(prev_games.get('receiving_yards', pd.Series([0])).mean()),
                'avg_receptions': float(prev_games.get('receptions', pd.Series([0])).mean()),
                # Recent form
                'games_played': int(len(prev_games))
            }

    # Cache for future use
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(trailing_stats, f)

    logger.info(f"✅ Generated trailing stats for {len(trailing_stats):,} player-weeks")
    return trailing_stats


def create_historical_player_input(
    row: pd.Series,
    trailing_stats: Dict
) -> PlayerPropInput:
    """
    Create PlayerPropInput for a historical prop.

    Uses historical context available at that time.
    """
    from nfl_quant.constants import FALLBACK_VALUES

    player_name = row['player_name']
    player_id = row.get('player_id', player_name)  # Use player_name as ID if not available
    season = row['season']
    week = row['week']
    position = row['position']
    team = row['team']
    opponent = row.get('opponent', 'UNK')

    # Get trailing stats - try multiple key formats
    key = f"{player_name}_week{week}_season{season}"
    hist = trailing_stats.get(key, {})

    # If not found, try without season
    if not hist:
        key = f"{player_name}_week{week}"
        hist = trailing_stats.get(key, {})

    # If still not found, use fallback values
    if not hist:
        # Fallback to defaults based on position
        hist = {
            'trailing_snap_share': FALLBACK_VALUES.get('snap_share', 0.5),
            'trailing_target_share': FALLBACK_VALUES.get('target_share', 0.1) if position in ['WR', 'TE', 'RB'] else None,
            'trailing_carry_share': FALLBACK_VALUES.get('carry_share', 0.15) if position in ['RB', 'FB'] else None,
            'trailing_yards_per_opportunity': FALLBACK_VALUES.get('yards_per_opportunity', 5.0),
            'trailing_td_rate': FALLBACK_VALUES.get('td_rate', 0.05),
            'projected_team_total': 25.0,
            'projected_opponent_total': 22.0,
            'projected_game_script': 3.0,
            'projected_pace': 28.0,
        }

    # Extract values with fallbacks
    trailing_snap_share = hist.get('trailing_snap_share', FALLBACK_VALUES.get('snap_share', 0.5))
    trailing_target_share = hist.get('trailing_target_share')
    if trailing_target_share is None and position in ['WR', 'TE', 'RB']:
        trailing_target_share = FALLBACK_VALUES.get('target_share', 0.1)

    trailing_carry_share = hist.get('trailing_carry_share')
    if trailing_carry_share is None and position in ['RB', 'FB']:
        trailing_carry_share = FALLBACK_VALUES.get('carry_share', 0.15) if position == 'RB' else 0.0

    trailing_yards_per_opportunity = hist.get('trailing_yards_per_opportunity', FALLBACK_VALUES.get('yards_per_opportunity', 5.0))
    trailing_td_rate = hist.get('trailing_td_rate', FALLBACK_VALUES.get('td_rate', 0.05))

    # Game context (use defaults if not available)
    projected_team_total = hist.get('projected_team_total', 25.0)
    projected_opponent_total = hist.get('projected_opponent_total', 22.0)
    projected_game_script = hist.get('projected_game_script', projected_team_total - projected_opponent_total)
    projected_pace = hist.get('projected_pace', 28.0)

    # Create PlayerPropInput with all required fields
    player_input = PlayerPropInput(
        player_id=str(player_id),
        player_name=player_name,
        position=position,
        team=team,
        opponent=opponent,
        week=week,
        # Game context
        projected_team_total=float(projected_team_total),
        projected_opponent_total=float(projected_opponent_total),
        projected_game_script=float(projected_game_script),
        projected_pace=float(projected_pace),
        # Usage features from trailing stats
        trailing_snap_share=float(trailing_snap_share),
        trailing_target_share=float(trailing_target_share) if trailing_target_share is not None else None,
        trailing_carry_share=float(trailing_carry_share) if trailing_carry_share is not None else None,
        trailing_yards_per_opportunity=float(trailing_yards_per_opportunity),
        trailing_td_rate=float(trailing_td_rate),
        # Defensive matchup (neutral if not available)
        opponent_def_epa_vs_position=hist.get('opponent_def_epa_vs_position', 0.0),
    )

    return player_input


def simulate_historical_props_batch(
    props_df: pd.DataFrame,
    simulator: PlayerSimulator,
    trailing_stats: Dict,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Run simulations on historical props in batches.

    Returns props_df with populated model_prob_raw, model_projection, model_std.
    """
    logger.info(f"Starting simulations for {len(props_df):,} props...")

    results = []
    total_batches = (len(props_df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(props_df))
        batch = props_df.iloc[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                   f"({len(batch)} props)...")

        for idx, row in batch.iterrows():
            try:
                # Create player input
                player_input = create_historical_player_input(row, trailing_stats)

                # Run simulation
                sim_results = simulator.simulate_player(player_input)
                
                # Validate simulation results
                if not sim_results or len(sim_results) == 0:
                    logger.warning(f"Empty simulation result for {row['player_name']} "
                                 f"week {row['week']} ({row['position']})")
                    result_row = row.copy()
                    results.append(result_row)
                    continue

                # Extract relevant stat based on market
                market = row['market']
                position = row['position']
                
                # Map markets to simulator keys
                stat_values = None
                if market == 'player_pass_yds':
                    stat_values = sim_results.get('passing_yards')
                elif market == 'player_rush_yds':
                    stat_values = sim_results.get('rushing_yards')
                elif market == 'player_reception_yds':
                    stat_values = sim_results.get('receiving_yards')
                elif market == 'player_receptions':
                    stat_values = sim_results.get('receptions')
                else:
                    logger.warning(f"Unknown market: {market} for {row['player_name']}")
                    result_row = row.copy()
                    results.append(result_row)
                    continue

                # Validate stat values
                if stat_values is None or len(stat_values) == 0:
                    logger.warning(f"No stat values for {market} - {row['player_name']} "
                                 f"week {row['week']} (available keys: {list(sim_results.keys())})")
                    result_row = row.copy()
                    results.append(result_row)
                    continue

                # Ensure it's a numpy array
                if not isinstance(stat_values, np.ndarray):
                    stat_values = np.array(stat_values)

                # Calculate projection and std
                projection = float(np.mean(stat_values))
                std = float(np.std(stat_values))

                # Calculate raw probability
                line = float(row['line'])
                pick_type = str(row['pick_type'])

                if std > 0:
                    z_score = (line - projection) / std
                    prob_over = 1.0 - norm.cdf(z_score)
                    prob_under = norm.cdf(z_score)
                else:
                    # No variance - deterministic
                    prob_over = 1.0 if projection > line else 0.0
                    prob_under = 1.0 if projection < line else 0.0

                # Clamp to reasonable range (don't clamp too aggressively)
                prob_over = np.clip(prob_over, 0.01, 0.99)
                prob_under = np.clip(prob_under, 0.01, 0.99)

                model_prob_raw = prob_over if pick_type == 'Over' else prob_under

                # Store result
                result_row = row.copy()
                result_row['model_prob_raw'] = float(model_prob_raw)
                result_row['model_projection'] = projection
                result_row['model_std'] = std
                results.append(result_row)

            except Exception as e:
                import traceback
                logger.error(f"Failed to simulate {row.get('player_name', 'unknown')} "
                           f"week {row.get('week', 'unknown')}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                # Keep row with NaN values
                result_row = row.copy()
                results.append(result_row)

        # Log batch completion
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Completed {end_idx:,}/{len(props_df):,} props "
                       f"({100 * end_idx / len(props_df):.1f}%)")

    results_df = pd.DataFrame(results)
    logger.info(f"✅ Completed simulations for {len(results_df):,} props")

    # Report statistics
    valid_sims = results_df['model_prob_raw'].notna()
    logger.info(f"  Valid simulations: {valid_sims.sum():,} "
               f"({100 * valid_sims.mean():.1f}%)")

    return results_df


def main():
    print("=" * 80)
    print("STEP 3: RUNNING SIMULATIONS ON HISTORICAL PROPS")
    print("=" * 80)
    print()

    # Load historical props
    print("Loading historical props...")
    props_df = load_historical_props()
    print(f"  Total props: {len(props_df):,}")
    print()

    # Use FULL dataset for comprehensive calibration
    print(f"  📊 Using full dataset: {len(props_df):,} props")
    print()

    # Load trailing stats
    print("Loading historical trailing stats...")
    trailing_stats = load_historical_trailing_stats()
    print(f"  Loaded stats for {len(trailing_stats):,} player-weeks")
    print()

    # Load simulator with SAME predictors
    print("Loading PlayerSimulator with current predictors...")
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,  # Use 10k trials for speed (can increase to 50k for accuracy)
        seed=42
    )
    print("  ✅ Simulator loaded")
    print()

    # Run simulations
    print("Running simulations on historical props...")
    print(f"  This may take 1-2 hours for {len(props_df):,} props...")
    print()

    results_df = simulate_historical_props_batch(
        props_df=props_df,
        simulator=simulator,
        trailing_stats=trailing_stats,
        batch_size=1000
    )

    # Save results
    output_file = Path("data/calibration/historical_props_simulated.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_file, index=False)

    print()
    print(f"✅ Saved simulated props to: {output_file}")
    print()

    # Report statistics
    print("📊 Simulation Summary:")
    print(f"  Total props: {len(results_df):,}")
    valid = results_df['model_prob_raw'].notna()
    print(f"  Valid simulations: {valid.sum():,} ({100 * valid.mean():.1f}%)")
    print()

    if valid.sum() > 0:
        print(f"  Model prob range: "
              f"{results_df.loc[valid, 'model_prob_raw'].min():.3f} - "
              f"{results_df.loc[valid, 'model_prob_raw'].max():.3f}")
        print(f"  Model prob mean: "
              f"{results_df.loc[valid, 'model_prob_raw'].mean():.3f}")
        print()

    print("Next steps:")
    print("  1. Retrain calibrator with this expanded dataset (Step 4)")
    print("  2. Test improved calibration on current week (Step 5)")
    print()


if __name__ == "__main__":
    main()
