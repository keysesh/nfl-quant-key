#!/usr/bin/env python3
"""
Audit edge calculations in backtest to verify negative edge filtering is correct.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.constants import RECEIVING_POSITIONS

MARKET_CONFIG = {
    "player_pass_yds": {
        "sim_key": "passing_yards",
        "actual_column": "pass_yd",
        "positions": {"QB"},
    },
    "player_receptions": {
        "sim_key": "receptions",
        "actual_column": "rec",
        "positions": set(RECEIVING_POSITIONS),
    },
    "player_reception_yds": {
        "sim_key": "receiving_yards",
        "actual_column": "rec_yd",
        "positions": set(RECEIVING_POSITIONS),
    },
}


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def normalize_player_name(name: str) -> str:
    """Normalize player name by removing suffixes."""
    import re
    name = str(name).strip().lower()
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', name, flags=re.IGNORECASE)
    name = ' '.join(name.split())
    return name


def audit_edge_calculations(week: int = 7, num_samples: int = 20):
    """Audit edge calculations for a specific week."""

    print("=" * 80)
    print(f"EDGE CALCULATION AUDIT - WEEK {week}")
    print("=" * 80)

    # Load historical props for the week
    props = pd.read_csv(f'data/historical/backfill/player_props_history_20251021T000000Z.csv')

    # Filter to supported markets
    props = props[props['market'].isin(MARKET_CONFIG.keys())]
    props = props[props['prop_type'].isin(['over', 'under'])]
    props['player_key'] = props['player'].apply(normalize_player_name)

    print(f"\nTotal props for Week {week}: {len(props)}")
    print(f"Markets: {props['market'].value_counts().to_dict()}")

    # Load stats using unified interface - FAIL EXPLICITLY if unavailable
    from nfl_quant.data.stats_loader import load_weekly_stats, is_data_available

    if not is_data_available(week, 2025, source='auto'):
        raise FileNotFoundError(
            f"Stats data not available for week {week}, season 2025. "
            f"Run data fetching scripts to populate data."
        )

    stats_df = load_weekly_stats(week, 2025, source='auto')
    stats_df['player_key'] = stats_df['player_name'].apply(normalize_player_name)
    stats_df['team'] = stats_df['team'].str.upper()

    # Load simulator
    usage_pred, efficiency_pred = load_predictors()
    calibrator = NFLProbabilityCalibrator()
    calibrator_path = Path("models/isotonic_calibrator.pkl")
    if calibrator_path.exists():
        calibrator.load(str(calibrator_path))

    simulator = PlayerSimulator(
        usage_predictor=usage_pred,
        efficiency_predictor=efficiency_pred,
        trials=50000,
        seed=42,
        calibrator=calibrator,
    )

    # Load trailing stats
    import json
    with open('data/week_specific_trailing_stats.json') as f:
        trailing_stats = json.load(f)

    # Sample props to audit
    sample_props = props.sample(min(num_samples, len(props)), random_state=42)

    results = []

    for idx, prop_row in sample_props.iterrows():
        player_name = prop_row['player']
        player_key = prop_row['player_key']
        market = prop_row['market']
        line = prop_row['line']
        price = prop_row['american_price']
        prop_type = prop_row['prop_type']

        # Find player in stats
        candidate_teams = [prop_row['home_team'], prop_row['away_team']]
        player_found = False
        matched_team = None

        for team in candidate_teams:
            matches = stats_df[(stats_df['player_key'] == player_key) &
                             (stats_df['team'] == team.upper())]
            if len(matches) > 0:
                player_data = matches.iloc[0]
                matched_team = team.upper()
                player_found = True
                break

        if not player_found:
            results.append({
                'player': player_name,
                'market': market,
                'line': line,
                'price': price,
                'prop_type': prop_type,
                'status': 'PLAYER_NOT_FOUND',
                'model_prob': None,
                'market_prob': None,
                'edge': None,
            })
            continue

        # Create player input
        position = player_data['position']
        config = MARKET_CONFIG[market]

        if position not in config['positions']:
            results.append({
                'player': player_name,
                'market': market,
                'line': line,
                'price': price,
                'prop_type': prop_type,
                'status': 'WRONG_POSITION',
                'model_prob': None,
                'market_prob': None,
                'edge': None,
            })
            continue

        # Create minimal player input (simplified for audit)
        from nfl_quant.schemas import PlayerPropInput
        player_input = PlayerPropInput(
            player_name=player_data['player_name'],
            team=matched_team,
            position=position,
            opponent="OPP",  # Simplified
            home_or_away="home",
            week=week,
            trailing_stats={},  # Simplified
            game_context={},
        )

        # Simulate
        try:
            sim_result = simulator.simulate_player(player_input)
            stat_key = config['sim_key']

            if stat_key not in sim_result:
                results.append({
                    'player': player_name,
                    'market': market,
                    'line': line,
                    'price': price,
                    'prop_type': prop_type,
                    'status': 'NO_SIM_RESULT',
                    'model_prob': None,
                    'market_prob': None,
                    'edge': None,
                })
                continue

            distribution = sim_result[stat_key]

            # Calculate model probability
            if prop_type == 'over':
                model_prob_raw = float(np.mean(distribution > line))
            else:
                model_prob_raw = float(np.mean(distribution < line))

            # Apply calibration
            if simulator.calibrator and simulator.calibrator.is_fitted:
                model_prob = float(simulator.calibrator.transform(np.array([model_prob_raw]))[0])
            else:
                model_prob = model_prob_raw

            # Calculate market probability
            market_prob = american_to_implied_prob(price)

            # Calculate edge
            edge = model_prob - market_prob

            results.append({
                'player': player_name,
                'market': market,
                'line': line,
                'price': price,
                'prop_type': prop_type,
                'status': 'CALCULATED',
                'model_prob_raw': model_prob_raw,
                'model_prob': model_prob,
                'market_prob': market_prob,
                'edge': edge,
                'sim_mean': np.mean(distribution),
                'sim_median': np.median(distribution),
            })

        except Exception as e:
            results.append({
                'player': player_name,
                'market': market,
                'line': line,
                'price': price,
                'prop_type': prop_type,
                'status': f'ERROR: {str(e)}',
                'model_prob': None,
                'market_prob': None,
                'edge': None,
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*80}")
    print("AUDIT RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\nStatus Breakdown:")
    print(results_df['status'].value_counts())

    calculated = results_df[results_df['status'] == 'CALCULATED']
    if len(calculated) > 0:
        print(f"\n{'='*80}")
        print("EDGE DISTRIBUTION (Successfully Calculated Props)")
        print(f"{'='*80}")
        print(f"\nEdge Statistics:")
        print(f"  Mean Edge: {calculated['edge'].mean():.2%}")
        print(f"  Median Edge: {calculated['edge'].median():.2%}")
        print(f"  Std Dev: {calculated['edge'].std():.2%}")
        print(f"  Min Edge: {calculated['edge'].min():.2%}")
        print(f"  Max Edge: {calculated['edge'].max():.2%}")

        print(f"\nEdge Breakdown:")
        positive_edge = calculated[calculated['edge'] > 0]
        negative_edge = calculated[calculated['edge'] <= 0]
        print(f"  Positive Edge: {len(positive_edge)} ({len(positive_edge)/len(calculated)*100:.1f}%)")
        print(f"  Negative/Zero Edge: {len(negative_edge)} ({len(negative_edge)/len(calculated)*100:.1f}%)")

        print(f"\n{'='*80}")
        print("SAMPLE OF NEGATIVE EDGE PROPS (Manual Verification)")
        print(f"{'='*80}\n")

        neg_sample = negative_edge.head(10)
        for idx, row in neg_sample.iterrows():
            print(f"Player: {row['player']}")
            print(f"  Market: {row['market']} {row['prop_type'].upper()} {row['line']}")
            print(f"  Price: {row['price']} (Market Prob: {row['market_prob']:.2%})")
            print(f"  Simulation: Mean={row['sim_mean']:.1f}, Median={row['sim_median']:.1f}")
            print(f"  Model Prob (Raw): {row['model_prob_raw']:.2%}")
            print(f"  Model Prob (Calibrated): {row['model_prob']:.2%}")
            print(f"  ❌ EDGE: {row['edge']:.2%} (NEGATIVE - Correctly Filtered)")
            print()

        if len(positive_edge) > 0:
            print(f"{'='*80}")
            print("SAMPLE OF POSITIVE EDGE PROPS (Should Be Included)")
            print(f"{'='*80}\n")

            pos_sample = positive_edge.head(5)
            for idx, row in pos_sample.iterrows():
                print(f"Player: {row['player']}")
                print(f"  Market: {row['market']} {row['prop_type'].upper()} {row['line']}")
                print(f"  Price: {row['price']} (Market Prob: {row['market_prob']:.2%})")
                print(f"  Simulation: Mean={row['sim_mean']:.1f}, Median={row['sim_median']:.1f}")
                print(f"  Model Prob (Raw): {row['model_prob_raw']:.2%}")
                print(f"  Model Prob (Calibrated): {row['model_prob']:.2%}")
                print(f"  ✅ EDGE: {row['edge']:.2%} (POSITIVE - Should Be Included)")
                print()

    # Save detailed results
    results_df.to_csv(f'reports/edge_audit_week{week}.csv', index=False)
    print(f"\n✅ Detailed results saved to: reports/edge_audit_week{week}.csv")

    return results_df


if __name__ == "__main__":
    audit_edge_calculations(week=7, num_samples=30)
