"""
Analyze TD prediction distribution before and after normalization.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.models.td_predictor import TouchdownPredictor, estimate_usage_factors
from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_week(week: int):
    """Analyze TD predictions for one week before/after normalization."""

    # Load actual outcomes from NFLverse
    actuals_file = 'data/nflverse_cache/stats_player_week_2025.csv'
    actuals_df = pd.read_csv(actuals_file)
    actuals_df = actuals_df[actuals_df['week'] == week].copy()  # Filter to specific week
    skill_positions = ['QB', 'RB', 'WR', 'TE']
    actuals_df = actuals_df[actuals_df['position'].isin(skill_positions)]

    # Load models
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(usage_predictor, efficiency_predictor, trials=5000)

    # Initialize TD predictor
    hist_stats_path = Path('data/nflverse_cache/stats_player_week_2025.csv')
    td_predictor = TouchdownPredictor(hist_stats_path)

    # Load nflverse for trailing stats
    nflverse_df = pd.read_csv(hist_stats_path)

    results = []

    for _, row in actuals_df.iterrows():
        player_name = row['player_name']
        position = row['position']
        team = row['team']

        # Calculate trailing stats (simple version)
        player_hist = nflverse_df[
            (nflverse_df['player_display_name'] == player_name) &
            (nflverse_df['week'] < week) &
            (nflverse_df['week'] >= week - 4)
        ]

        if player_hist.empty:
            continue

        # Get predicted carries/targets from trailing average
        projected_carries = player_hist['carries'].mean() if position in ['QB', 'RB'] else 0
        projected_targets = player_hist['targets'].mean() if position in ['WR', 'TE', 'RB'] else 0
        projected_pass_att = player_hist['attempts'].mean() if position == 'QB' else 0

        # Create mock player data for usage factors
        player_data = pd.Series({
            'rushing_attempts_mean': projected_carries,
            'receptions_mean': projected_targets / 1.5 if position in ['WR', 'TE'] else projected_targets / 1.3,
        })

        usage_factors = estimate_usage_factors(player_data, position)

        # Predict TDs
        td_pred = td_predictor.predict_touchdown_probability(
            player_name=player_name,
            position=position,
            projected_carries=projected_carries,
            projected_targets=projected_targets,
            projected_pass_attempts=projected_pass_att,
            red_zone_share=usage_factors['red_zone_share'],
            goal_line_role=usage_factors['goal_line_role'],
            team_projected_total=24.0,
            opponent_td_rate_allowed=1.0,
        )

        # Calculate prob(any TD) - BEFORE normalization
        rushing_td_mean = td_pred.get('rushing_tds_mean', 0)
        receiving_td_mean = td_pred.get('receiving_tds_mean', 0)
        passing_td_mean = td_pred.get('passing_tds_mean', 0)

        # P(any TD) = 1 - P(no TD) = 1 - exp(-λ_total)
        lambda_total = rushing_td_mean + receiving_td_mean + passing_td_mean
        prob_any_td = 1 - np.exp(-lambda_total)

        # Actual outcome
        actual_rush_td = row.get('rush_td', 0)
        actual_rec_td = row.get('rec_td', 0)
        actual_pass_td = row.get('pass_td', 0)
        scored_any_td = 1 if (actual_rush_td + actual_rec_td + actual_pass_td) > 0 else 0

        results.append({
            'player_name': player_name,
            'position': position,
            'team': team,
            'projected_carries': projected_carries,
            'projected_targets': projected_targets,
            'lambda_total': lambda_total,
            'prob_any_td_pre_norm': prob_any_td,
            'scored_any_td': scored_any_td,
            'red_zone_share': usage_factors['red_zone_share'],
            'goal_line_role': usage_factors['goal_line_role'],
        })

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == '__main__':
    logger.info("Analyzing TD predictions BEFORE normalization...")

    # Analyze week 2 (high actual TD rate: 34.6%)
    results = analyze_week(2)

    print(f"\n{'='*80}")
    print(f"WEEK 2 ANALYSIS - BEFORE NORMALIZATION")
    print(f"{'='*80}\n")

    print(f"Total players: {len(results)}")
    print(f"Actual TD rate: {results['scored_any_td'].mean():.1%}")
    print(f"Predicted TD prob (avg): {results['prob_any_td_pre_norm'].mean():.1%}")

    # Bin analysis
    print(f"\nCALIBRATION BY PREDICTED PROBABILITY BIN (PRE-NORMALIZATION):")
    print(f"{'='*80}")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results['prob_bin'] = pd.cut(results['prob_any_td_pre_norm'], bins=bins)

    calibration = results.groupby('prob_bin').agg({
        'prob_any_td_pre_norm': 'mean',
        'scored_any_td': ['mean', 'count']
    })

    print(calibration)

    # Show top underestimated players
    print(f"\n\nTOP 10 UNDERESTIMATED PLAYERS (scored TD but low predicted prob):")
    print(f"{'='*80}")
    underest = results[results['scored_any_td'] == 1].sort_values('prob_any_td_pre_norm').head(10)
    print(underest[['player_name', 'position', 'prob_any_td_pre_norm', 'projected_carries', 'projected_targets', 'red_zone_share']])
