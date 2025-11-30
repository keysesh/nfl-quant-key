#!/usr/bin/env python3
"""
Re-simulate ONLY the reception yards props with the fixed simulator

This script:
1. Loads historical_props_simulated.parquet
2. Filters to reception yards props only
3. Re-runs simulations with FIXED reception yards logic
4. Updates the parquet file with corrected data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput
from scipy.stats import norm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("RE-SIMULATING RECEPTION YARDS PROPS WITH FIXED SIMULATOR")
    print("=" * 80)
    print()

    # Load existing simulated props
    props_file = Path("data/calibration/historical_props_simulated.parquet")
    df = pd.read_parquet(props_file)

    print(f"Total props loaded: {len(df):,}")
    print()

    # Filter to reception yards props only
    rec_yds_props = df[df['market'] == 'player_reception_yds'].copy()
    other_props = df[df['market'] != 'player_reception_yds'].copy()

    print(f"Reception yards props: {len(rec_yds_props):,}")
    print(f"Other props (keep as-is): {len(other_props):,}")
    print()

    if len(rec_yds_props) == 0:
        print("⚠️ No reception yards props found to re-simulate")
        return

    # Load simulator with FIXED reception yards logic
    print("Loading simulator with fixed reception yards logic...")
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,
        seed=42
    )
    print("✅ Simulator loaded")
    print()

    # Load trailing stats
    import json
    trailing_stats_file = Path("data/nflverse/historical_trailing_stats.json")
    if trailing_stats_file.exists():
        with open(trailing_stats_file) as f:
            trailing_stats = json.load(f)
        print(f"✅ Loaded trailing stats for {len(trailing_stats):,} player-weeks")
    else:
        print("⚠️ No trailing stats file found, using defaults")
        trailing_stats = {}
    print()

    # Re-simulate reception yards props
    print(f"Re-simulating {len(rec_yds_props):,} reception yards props...")
    print()

    from nfl_quant.constants import FALLBACK_VALUES

    results = []
    for idx, row in rec_yds_props.iterrows():
        try:
            # Create player input
            player_name = row['player_name']
            season = row['season']
            week = row['week']
            position = row['position']

            # Get trailing stats
            key = f"{player_name}_week{week}_season{season}"
            hist = trailing_stats.get(key, {})

            if not hist:
                # Use fallback values based on position
                if position == 'WR':
                    hist = {
                        'trailing_snap_share': 0.70,
                        'trailing_target_share': 0.18,
                        'trailing_carry_share': None,
                        'trailing_yards_per_opportunity': 8.0,
                        'trailing_td_rate': 0.05,
                        'projected_team_total': 25.0,
                        'projected_opponent_total': 22.0,
                        'projected_game_script': 0.0,
                        'projected_pace': 30.0,
                    }
                elif position == 'TE':
                    hist = {
                        'trailing_snap_share': 0.65,
                        'trailing_target_share': 0.12,
                        'trailing_carry_share': None,
                        'trailing_yards_per_opportunity': 7.0,
                        'trailing_td_rate': 0.04,
                        'projected_team_total': 25.0,
                        'projected_opponent_total': 22.0,
                        'projected_game_script': 0.0,
                        'projected_pace': 30.0,
                    }
                else:  # RB
                    hist = {
                        'trailing_snap_share': 0.55,
                        'trailing_target_share': 0.08,
                        'trailing_carry_share': 0.45,
                        'trailing_yards_per_opportunity': 6.5,
                        'trailing_td_rate': 0.04,
                        'projected_team_total': 25.0,
                        'projected_opponent_total': 22.0,
                        'projected_game_script': 0.0,
                        'projected_pace': 30.0,
                    }

            player_input = PlayerPropInput(
                player_id=str(row.get('player_id', player_name)),
                player_name=player_name,
                position=position,
                team=row['team'],
                opponent=row.get('opponent', 'UNK'),
                week=week,
                projected_team_total=float(hist.get('projected_team_total', 25.0)),
                projected_opponent_total=float(hist.get('projected_opponent_total', 22.0)),
                projected_game_script=float(hist.get('projected_game_script', 0.0)),
                projected_pace=float(hist.get('projected_pace', 30.0)),
                trailing_snap_share=float(hist.get('trailing_snap_share', 0.5)),
                trailing_target_share=hist.get('trailing_target_share', 0.1),
                trailing_carry_share=hist.get('trailing_carry_share'),
                trailing_yards_per_opportunity=float(hist.get('trailing_yards_per_opportunity', 7.0)),
                trailing_td_rate=float(hist.get('trailing_td_rate', 0.04)),
                opponent_def_epa_vs_position=hist.get('opponent_def_epa_vs_position', 0.0),
            )

            # Run simulation
            sim_results = simulator.simulate_player(player_input)

            if not sim_results or 'receiving_yards' not in sim_results:
                logger.warning(f"No receiving_yards for {player_name} week {week}")
                result_row = row.copy()
                results.append(result_row)
                continue

            stat_values = sim_results['receiving_yards']
            if len(stat_values) == 0:
                logger.warning(f"Empty receiving_yards for {player_name} week {week}")
                result_row = row.copy()
                results.append(result_row)
                continue

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
                prob_over = 1.0 if projection > line else 0.0
                prob_under = 1.0 if projection < line else 0.0

            prob_over = np.clip(prob_over, 0.01, 0.99)
            prob_under = np.clip(prob_under, 0.01, 0.99)

            model_prob_raw = prob_over if pick_type == 'Over' else prob_under

            # Store result
            result_row = row.copy()
            result_row['model_prob_raw'] = float(model_prob_raw)
            result_row['model_projection'] = projection
            result_row['model_std'] = std
            results.append(result_row)

            if len(results) % 500 == 0:
                print(f"  Progress: {len(results)}/{len(rec_yds_props)} ({100*len(results)/len(rec_yds_props):.1f}%)")

        except Exception as e:
            logger.error(f"Failed to simulate {row.get('player_name', 'unknown')}: {e}")
            result_row = row.copy()
            results.append(result_row)

    # Create updated dataframe
    rec_yds_props_updated = pd.DataFrame(results)

    print()
    print(f"✅ Re-simulated {len(rec_yds_props_updated):,} reception yards props")
    print()

    # Report statistics
    valid = rec_yds_props_updated['model_prob_raw'].notna()
    print(f"Valid simulations: {valid.sum():,} ({100*valid.mean():.1f}%)")

    if valid.sum() > 0:
        print(f"  Projection mean: {rec_yds_props_updated.loc[valid, 'model_projection'].mean():.1f} yards")
        print(f"  Projection std dev: {rec_yds_props_updated.loc[valid, 'model_std'].mean():.1f} yards")
        print()

        # Compare old vs new projections
        old_mean = rec_yds_props['model_projection'].mean()
        new_mean = rec_yds_props_updated.loc[valid, 'model_projection'].mean()
        change_pct = ((new_mean - old_mean) / old_mean) * 100

        print(f"PROJECTION CHANGE:")
        print(f"  Old mean projection: {old_mean:.1f} yards")
        print(f"  New mean projection: {new_mean:.1f} yards")
        print(f"  Change: {change_pct:+.1f}%")
    print()

    # Combine with other props
    print("Combining with other props...")
    combined = pd.concat([other_props, rec_yds_props_updated], ignore_index=True)
    print(f"Total props: {len(combined):,}")
    print()

    # Save updated file
    output_file = Path("data/calibration/historical_props_simulated.parquet")
    combined.to_parquet(output_file, index=False)

    print(f"✅ Saved updated props to {output_file}")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Retrain calibrator with corrected reception yards data")
    print("2. Run backtest to validate reception yards performance")
    print()

if __name__ == "__main__":
    main()
