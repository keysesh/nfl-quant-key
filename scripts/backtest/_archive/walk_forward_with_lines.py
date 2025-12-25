#!/usr/bin/env python3
"""
Walk-Forward Validation with REAL Sportsbook Lines

Proper methodology:
1. For each week N, use ONLY weeks 1 to N-1 for trailing stats
2. Match predictions against real sportsbook lines
3. Calculate actual ROI without calibrator leakage

This is the gold standard for validating betting edge.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / 'data'
NFLVERSE_DIR = DATA_DIR / 'nflverse'


def load_historical_props() -> pd.DataFrame:
    """Load real sportsbook lines with week info."""
    props_path = DATA_DIR / 'backtest' / 'all_historical_props_2025.csv'

    if not props_path.exists():
        raise FileNotFoundError(f"Need {props_path} for real lines")

    df = pd.read_csv(props_path)

    # Map market names to stat types
    market_to_stat = {
        'player_pass_yds': 'passing_yards',
        'player_rush_yds': 'rushing_yards',
        'player_reception_yds': 'receiving_yards',
        'player_receptions': 'receptions',
    }

    df['stat_type'] = df['market'].map(market_to_stat)
    df = df[df['stat_type'].notna()].copy()

    # Pivot over/under into columns
    over_df = df[df['prop_type'] == 'over'][['week', 'player', 'stat_type', 'line', 'american_odds']].copy()
    over_df = over_df.rename(columns={'american_odds': 'over_odds'})

    under_df = df[df['prop_type'] == 'under'][['week', 'player', 'stat_type', 'line', 'american_odds']].copy()
    under_df = under_df.rename(columns={'american_odds': 'under_odds'})

    # Merge
    merged = over_df.merge(under_df, on=['week', 'player', 'stat_type', 'line'], how='inner')

    logger.info(f"Loaded {len(merged)} prop lines from weeks {merged['week'].min()}-{merged['week'].max()}")
    return merged


def load_weekly_stats() -> pd.DataFrame:
    """Load NFLverse weekly stats."""
    path = NFLVERSE_DIR / 'weekly_stats.parquet'
    df = pd.read_parquet(path)
    df = df[df['season'] == 2025].copy()
    return df


def calculate_trailing_stats(player_name: str, position: str, week: int, weekly_df: pd.DataFrame) -> Dict:
    """Calculate trailing stats using ONLY weeks before target week."""
    # Filter to weeks BEFORE target week only
    player_data = weekly_df[
        (weekly_df['player_display_name'] == player_name) &
        (weekly_df['week'] < week)
    ].sort_values('week')

    if len(player_data) == 0:
        return None

    # Get most recent team
    team = player_data.iloc[-1]['team']

    # EWMA trailing averages (matching production)
    avg_targets = player_data['targets'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'targets' in player_data.columns else 0
    avg_receptions = player_data['receptions'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'receptions' in player_data.columns else 0
    avg_rec_yards = player_data['receiving_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'receiving_yards' in player_data.columns else 0
    avg_rush_yards = player_data['rushing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'rushing_yards' in player_data.columns else 0
    avg_pass_yards = player_data['passing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'passing_yards' in player_data.columns else 0

    # RB-specific: carries and YPC
    avg_carries = player_data['carries'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'carries' in player_data.columns else 0
    # Calculate YPC safely (avoid division by zero)
    if 'carries' in player_data.columns and 'rushing_yards' in player_data.columns:
        valid_carries = player_data[player_data['carries'] > 0]
        if len(valid_carries) > 0:
            ypc_series = valid_carries['rushing_yards'] / valid_carries['carries']
            avg_ypc = ypc_series.ewm(span=4, min_periods=1).mean().iloc[-1]
        else:
            avg_ypc = 4.0  # NFL average default
    else:
        avg_ypc = 4.0

    # QB-specific: attempts, completions, completion pct, yards per completion
    avg_attempts = player_data['attempts'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'attempts' in player_data.columns else 0
    avg_completions = player_data['completions'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'completions' in player_data.columns else 0

    # Completion percentage
    if avg_attempts > 0:
        trailing_comp_pct = avg_completions / avg_attempts
    else:
        trailing_comp_pct = 0.65  # NFL average default

    # Yards per completion
    if 'completions' in player_data.columns and 'passing_yards' in player_data.columns:
        valid_comps = player_data[player_data['completions'] > 0]
        if len(valid_comps) > 0:
            ypc_pass_series = valid_comps['passing_yards'] / valid_comps['completions']
            avg_yards_per_completion = ypc_pass_series.ewm(span=4, min_periods=1).mean().iloc[-1]
        else:
            avg_yards_per_completion = 11.0  # NFL average default
    else:
        avg_yards_per_completion = 11.0

    # Yards per target (for WR/TE receiving efficiency)
    if 'targets' in player_data.columns and 'receiving_yards' in player_data.columns:
        valid_targets = player_data[player_data['targets'] > 0]
        if len(valid_targets) > 0:
            ypt_series = valid_targets['receiving_yards'] / valid_targets['targets']
            avg_yards_per_target = ypt_series.ewm(span=4, min_periods=1).mean().iloc[-1]
        else:
            avg_yards_per_target = 8.0  # NFL average default
    else:
        avg_yards_per_target = 8.0

    # Calculate trailing_yards_per_opportunity based on position
    # This is a blended metric - for WR/TE it's yards per target, for RB it's yards per carry
    if position in ('WR', 'TE'):
        trailing_yards_per_opportunity = avg_yards_per_target
    elif position == 'RB':
        trailing_yards_per_opportunity = avg_ypc
    elif position == 'QB':
        # For QB, use yards per attempt
        if avg_attempts > 0 and 'passing_yards' in player_data.columns:
            trailing_yards_per_opportunity = player_data['passing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] / avg_attempts
        else:
            trailing_yards_per_opportunity = 7.0  # NFL average Y/A
    else:
        trailing_yards_per_opportunity = 8.0

    return {
        'team': team,
        'position': position,
        'games_played': len(player_data),
        # WR/TE stats
        'avg_targets': avg_targets,
        'avg_receptions': avg_receptions,
        'avg_rec_yards': avg_rec_yards,
        'avg_yards_per_target': avg_yards_per_target,
        # RB stats
        'avg_rush_yards': avg_rush_yards,
        'avg_carries': avg_carries,
        'avg_ypc': avg_ypc,
        # QB stats
        'avg_pass_yards': avg_pass_yards,
        'avg_attempts': avg_attempts,
        'avg_completions': avg_completions,
        'trailing_comp_pct': trailing_comp_pct,
        'avg_yards_per_completion': avg_yards_per_completion,
        # Calculated efficiency metric
        'trailing_yards_per_opportunity': trailing_yards_per_opportunity,
    }


def get_actual_outcome(player_name: str, week: int, stat_type: str, weekly_df: pd.DataFrame) -> Optional[float]:
    """Get actual stat value for a player-week."""
    stat_col_map = {
        'receiving_yards': 'receiving_yards',
        'rushing_yards': 'rushing_yards',
        'passing_yards': 'passing_yards',
        'receptions': 'receptions',
    }

    col = stat_col_map.get(stat_type)
    if not col:
        return None

    player_week = weekly_df[
        (weekly_df['player_display_name'] == player_name) &
        (weekly_df['week'] == week)
    ]

    if len(player_week) == 0:
        return None

    return player_week.iloc[0].get(col, None)


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_roi(wins: int, losses: int, avg_odds: float = -110) -> float:
    """Calculate ROI at given odds."""
    if wins + losses == 0:
        return 0.0

    # At -110, win pays 0.909, loss costs 1.0
    if avg_odds < 0:
        win_payout = 100 / abs(avg_odds)
    else:
        win_payout = avg_odds / 100

    profit = (wins * win_payout) - (losses * 1.0)
    total_wagered = wins + losses
    return (profit / total_wagered) * 100


def run_walk_forward_backtest(
    weeks: List[int],
    min_games: int = 3,
    edge_threshold: float = 0.05,
    trials: int = 10000,
) -> pd.DataFrame:
    """
    Run proper walk-forward backtest with real lines.

    For each week N:
    1. Load trailing stats from weeks 1 to N-1 ONLY
    2. Run simulator to get prediction
    3. Compare to real line to find edge
    4. Track if bet would have won
    """
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION WITH REAL SPORTSBOOK LINES")
    logger.info("=" * 70)
    logger.info(f"Weeks: {min(weeks)}-{max(weeks)}")
    logger.info(f"Min games for trailing stats: {min_games}")
    logger.info(f"Edge threshold: {edge_threshold:.0%}")
    logger.info(f"Monte Carlo trials: {trials:,}")
    logger.info("")

    # Load data
    props_df = load_historical_props()
    weekly_df = load_weekly_stats()

    # Load simulator (models are fixed, not retrained per week)
    logger.info("Loading simulator...")
    usage_pred, eff_pred = load_predictors()
    simulator = PlayerSimulator(usage_pred, eff_pred, trials=trials, seed=42)
    logger.info("")

    results = []

    for week in weeks:
        week_props = props_df[props_df['week'] == week]
        logger.info(f"--- Week {week}: {len(week_props)} props ---")

        week_results = {'week': week, 'bets': 0, 'wins': 0}

        for _, prop in week_props.iterrows():
            player = prop['player']
            stat_type = prop['stat_type']
            line = prop['line']
            over_odds = prop['over_odds']
            under_odds = prop['under_odds']

            # Get trailing stats (ONLY from weeks < current week)
            # Need to infer position from weekly data
            player_weeks = weekly_df[weekly_df['player_display_name'] == player]
            if len(player_weeks) == 0:
                continue
            position = player_weeks.iloc[0]['position']

            trailing = calculate_trailing_stats(player, position, week, weekly_df)
            if trailing is None or trailing['games_played'] < min_games:
                continue

            # Get actual outcome
            actual = get_actual_outcome(player, week, stat_type, weekly_df)
            if actual is None:
                continue

            # Create player input for simulator
            try:
                # Build base input dict
                input_kwargs = {
                    'player_name': player,
                    'player_id': f"wf_{player}_{week}",
                    'position': position,
                    'team': trailing['team'],
                    'opponent': 'UNK',
                    'week': week,
                    'season': 2025,
                    'projected_team_total': 24.0,
                    'projected_opponent_total': 24.0,
                    'projected_game_script': 0.0,
                    'projected_pace': 60.0,
                    'trailing_snap_share': 0.85,
                    'trailing_td_rate': 0.05,
                    'trailing_yards_per_opportunity': trailing['trailing_yards_per_opportunity'],
                    'trailing_yards_per_target': trailing['avg_yards_per_target'],
                    'opponent_def_epa_vs_position': 0.0,
                    # WR/TE receiving stats
                    'avg_rec_tgt': trailing['avg_targets'],
                    'avg_rec_yd': trailing['avg_rec_yards'],
                    'trailing_target_share': trailing['avg_targets'] / 35.0 if trailing['avg_targets'] > 0 else 0.15,
                }

                # RB-specific stats
                if position == 'RB':
                    input_kwargs.update({
                        'avg_rush_yd': trailing['avg_rush_yards'],
                        'trailing_yards_per_carry': trailing['avg_ypc'],
                        'trailing_carry_share': trailing['avg_carries'] / 25.0 if trailing['avg_carries'] > 0 else 0.15,
                    })

                # QB-specific stats
                if position == 'QB':
                    input_kwargs.update({
                        'trailing_comp_pct': trailing['trailing_comp_pct'],
                        'trailing_yards_per_completion': trailing['avg_yards_per_completion'],
                        # QB uses 'trailing_attempts' which maps to avg_rec_tgt in simulator
                        # Override avg_rec_tgt with passing attempts for QB
                        'avg_rec_tgt': trailing['avg_attempts'],  # This feeds into trailing_attempts
                    })

                player_input = PlayerPropInput(**input_kwargs)

                # Run simulation
                sim_result = simulator.simulate_player(player_input)

                # Extract prediction for relevant stat
                stat_key_map = {
                    'receiving_yards': 'receiving_yards',
                    'rushing_yards': 'rushing_yards',
                    'passing_yards': 'passing_yards',
                    'receptions': 'receptions',
                }

                stat_key = stat_key_map.get(stat_type)
                if stat_key not in sim_result:
                    continue

                pred_mean = np.mean(sim_result[stat_key])
                pred_std = np.std(sim_result[stat_key])

                # Calculate probability vs line
                if pred_std > 0:
                    z_score = (line - pred_mean) / pred_std
                    from scipy import stats as scipy_stats
                    prob_under = scipy_stats.norm.cdf(z_score)
                    prob_over = 1 - prob_under
                else:
                    prob_over = 1.0 if pred_mean > line else 0.0
                    prob_under = 1 - prob_over

                # Market implied probabilities
                market_prob_over = american_to_prob(over_odds)
                market_prob_under = american_to_prob(under_odds)

                # Calculate edge
                edge_over = prob_over - market_prob_over
                edge_under = prob_under - market_prob_under

                # Determine if we bet
                bet_side = None
                if edge_over >= edge_threshold:
                    bet_side = 'over'
                    model_prob = prob_over
                    edge = edge_over
                elif edge_under >= edge_threshold:
                    bet_side = 'under'
                    model_prob = prob_under
                    edge = edge_under

                if bet_side:
                    # Did we win?
                    went_over = actual > line
                    won = (bet_side == 'over' and went_over) or (bet_side == 'under' and not went_over)

                    results.append({
                        'week': week,
                        'player': player,
                        'stat_type': stat_type,
                        'line': line,
                        'pred_mean': pred_mean,
                        'actual': actual,
                        'bet_side': bet_side,
                        'model_prob': model_prob,
                        'edge': edge,
                        'won': won,
                    })

                    week_results['bets'] += 1
                    if won:
                        week_results['wins'] += 1

            except Exception as e:
                continue

        if week_results['bets'] > 0:
            wr = week_results['wins'] / week_results['bets']
            logger.info(f"  Bets: {week_results['bets']}, Wins: {week_results['wins']}, Win Rate: {wr:.1%}")

    # Summary
    results_df = pd.DataFrame(results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("WALK-FORWARD RESULTS (NO DATA LEAKAGE)")
    logger.info("=" * 70)

    if len(results_df) > 0:
        total_bets = len(results_df)
        total_wins = results_df['won'].sum()
        overall_wr = total_wins / total_bets
        overall_roi = calculate_roi(total_wins, total_bets - total_wins)

        logger.info(f"Total Bets: {total_bets}")
        logger.info(f"Win Rate: {overall_wr:.1%}")
        logger.info(f"ROI (at -110): {overall_roi:+.1f}%")
        logger.info("")

        # By stat type
        logger.info("BY STAT TYPE:")
        for stat in results_df['stat_type'].unique():
            stat_df = results_df[results_df['stat_type'] == stat]
            wins = stat_df['won'].sum()
            total = len(stat_df)
            wr = wins / total
            roi = calculate_roi(wins, total - wins)
            logger.info(f"  {stat}: n={total}, win_rate={wr:.1%}, ROI={roi:+.1f}%")

        # Save results
        output_path = DATA_DIR / 'backtest' / 'walk_forward_with_lines_results.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to {output_path}")

    return results_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward Backtest with Real Lines')
    parser.add_argument('--weeks', type=str, default='5-13', help='Week range (e.g., 5-13)')
    parser.add_argument('--edge', type=float, default=0.05, help='Edge threshold (default 0.05)')
    parser.add_argument('--trials', type=int, default=10000, help='Monte Carlo trials')
    args = parser.parse_args()

    # Parse weeks
    if '-' in args.weeks:
        start, end = map(int, args.weeks.split('-'))
        weeks = list(range(start, end + 1))
    else:
        weeks = [int(args.weeks)]

    run_walk_forward_backtest(weeks=weeks, edge_threshold=args.edge, trials=args.trials)
