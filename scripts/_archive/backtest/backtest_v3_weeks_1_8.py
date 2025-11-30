#!/usr/bin/env python3
"""
V3 Backtest: Weeks 1-8 Validation

Generates V3 predictions for weeks 1-8, compares with Sleeper actual stats,
and calculates comprehensive performance metrics (CRPS, MAE, coverage).

Usage:
    python scripts/backtest/backtest_v3_weeks_1_8.py --weeks 1-8
    python scripts/backtest/backtest_v3_weeks_1_8.py --week 1
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.validation.calibration_metrics import CalibrationMetrics
from nfl_quant.schemas import PlayerPropInput


def load_sleeper_actuals(week: int) -> pd.DataFrame:
    """Load actual player stats from NFLverse for a given week."""
    file_path = Path("data/nflverse_cache/stats_player_week_2025.csv")

    if not file_path.exists():
        print(f"⚠️  No NFLverse stats found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df = df[df['week'] == week].copy()  # Filter to specific week

    # Standardize column names (NFLverse uses different names than Sleeper)
    # NFLverse columns: carries, attempts, completions, targets
    df = df.rename(columns={
        'carries': 'rushing_attempts',  # NFLverse calls it 'carries'
        'attempts': 'passing_attempts',  # NFLverse 'attempts' = passing attempts
        'completions': 'passing_completions',  # NFLverse uses 'completions'
        # These are already correct in NFLverse:
        # 'rushing_yards', 'passing_yards', 'receiving_yards'
        # 'rushing_tds', 'passing_tds', 'receiving_tds'
        # 'receptions', 'targets'
    })

    # Fill NaN values with 0 for stats that don't apply to all positions
    stat_columns = [
        'rushing_yards', 'rushing_attempts', 'rushing_tds',
        'receiving_yards', 'receptions', 'targets', 'receiving_tds',
        'passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds'
    ]
    for col in stat_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def calculate_trailing_stats(player_name: str, position: str, team: str, current_week: int,
                            all_weeks_stats: pd.DataFrame, lookback: int = 4) -> Dict:
    """Calculate trailing stats for a player from previous weeks."""
    # Get player's historical stats (weeks before current week)
    player_history = all_weeks_stats[
        (all_weeks_stats['player_name'] == player_name) &
        (all_weeks_stats['team'] == team) &
        (all_weeks_stats['week'] < current_week) &
        (all_weeks_stats['week'] >= max(1, current_week - lookback))
    ]

    if len(player_history) == 0:
        # Return league averages by position if no history
        defaults = {
            'QB': {'snap_share': 1.0, 'target_share': None, 'carry_share': None,
                   'yards_per_opp': 7.5, 'td_rate': 0.04},
            'RB': {'snap_share': 0.50, 'target_share': 0.08, 'carry_share': 0.35,
                   'yards_per_opp': 4.5, 'td_rate': 0.06},
            'WR': {'snap_share': 0.70, 'target_share': 0.18, 'carry_share': None,
                   'yards_per_opp': 7.0, 'td_rate': 0.05},
            'TE': {'snap_share': 0.65, 'target_share': 0.15, 'carry_share': None,
                   'yards_per_opp': 8.0, 'td_rate': 0.06},
        }
        return defaults.get(position, defaults['WR'])

    # Calculate trailing stats
    stats = {}

    # Snap share (assume full for now - would need snap count data)
    stats['snap_share'] = {'QB': 1.0, 'RB': 0.55, 'WR': 0.70, 'TE': 0.65}.get(position, 0.70)

    # Target share (receiving positions)
    if position in ['WR', 'TE', 'RB']:
        total_team_targets = 0
        player_targets = player_history['targets'].sum()

        # Get team total targets in same weeks
        for week in player_history['week'].values:
            week_team_targets = all_weeks_stats[
                (all_weeks_stats['team'] == team) &
                (all_weeks_stats['week'] == week) &
                (all_weeks_stats['position'].isin(['WR', 'TE', 'RB', 'QB']))
            ]['targets'].sum()
            total_team_targets += week_team_targets

        stats['target_share'] = player_targets / total_team_targets if total_team_targets > 0 else 0.15
    else:
        stats['target_share'] = None

    # Carry share (RBs)
    if position == 'RB':
        total_team_carries = 0
        player_carries = player_history['rushing_attempts'].sum()

        for week in player_history['week'].values:
            week_team_carries = all_weeks_stats[
                (all_weeks_stats['team'] == team) &
                (all_weeks_stats['week'] == week) &
                (all_weeks_stats['position'].isin(['RB', 'QB']))
            ]['rushing_attempts'].sum()
            total_team_carries += week_team_carries

        stats['carry_share'] = player_carries / total_team_carries if total_team_carries > 0 else 0.30
    else:
        stats['carry_share'] = None

    # Yards per opportunity
    total_opportunities = player_history['targets'].sum() + player_history['rushing_attempts'].sum()
    total_yards = player_history['receiving_yards'].sum() + player_history['rushing_yards'].sum()
    stats['yards_per_opp'] = total_yards / total_opportunities if total_opportunities > 0 else 6.0

    # TD rate
    total_tds = player_history['receiving_tds'].sum() + player_history['rushing_tds'].sum()
    stats['td_rate'] = total_tds / total_opportunities if total_opportunities > 0 else 0.05

    return stats


def calculate_defensive_epa(opponent: str, position: str, week: int, pbp_df: pd.DataFrame) -> float:
    """Calculate opponent's defensive EPA vs position from PBP data."""
    try:
        # Get opponent's defensive plays from previous weeks
        opponent_def_plays = pbp_df[
            (pbp_df['defteam'] == opponent) &
            (pbp_df['week'] < week) &
            (pbp_df['week'] >= max(1, week - 4))  # Last 4 weeks
        ].copy()

        if len(opponent_def_plays) == 0:
            return 0.0

        # Filter by play type based on position
        if position == 'QB':
            # Passing plays against this defense
            relevant_plays = opponent_def_plays[opponent_def_plays['play_type'] == 'pass']
        elif position in ['RB', 'FB']:
            # Rushing plays against this defense
            relevant_plays = opponent_def_plays[opponent_def_plays['play_type'] == 'run']
        elif position in ['WR', 'TE']:
            # Passing plays against this defense (for receivers)
            relevant_plays = opponent_def_plays[opponent_def_plays['play_type'] == 'pass']
        else:
            return 0.0

        if len(relevant_plays) == 0:
            return 0.0

        # Calculate average EPA allowed (negative = good defense)
        avg_epa = relevant_plays['epa'].mean()
        return float(avg_epa) if pd.notna(avg_epa) else 0.0

    except Exception as e:
        # Return neutral if calculation fails
        return 0.0


def load_game_context(week: int, team: str, position: str, pbp_df: pd.DataFrame) -> Dict:
    """Load game context from nflverse pbp data."""
    # Get team's games in current week
    team_games = pbp_df[
        (pbp_df['week'] == week) &
        ((pbp_df['home_team'] == team) | (pbp_df['away_team'] == team))
    ]

    if len(team_games) == 0:
        # Return neutral defaults with team usage estimates
        return {
            'opponent': 'UNK',
            'projected_team_total': 22.5,
            'projected_opponent_total': 22.5,
            'projected_game_script': 0.0,
            'projected_pace': 29.0,
            'opponent_def_epa_vs_position': 0.0,
            'projected_team_pass_attempts': 35.0,  # NFL average
            'projected_team_rush_attempts': 28.0,  # NFL average
            'projected_team_targets': 35.0,  # NFL average
        }

    # Get opponent
    game_row = team_games.iloc[0]
    opponent = game_row['away_team'] if game_row['home_team'] == team else game_row['home_team']
    is_home = game_row['home_team'] == team

    # Calculate actual game totals from this game (for backtest, we use actuals)
    home_score = game_row.get('home_score', 22.5) if 'home_score' in game_row else 22.5
    away_score = game_row.get('away_score', 22.5) if 'away_score' in game_row else 22.5

    team_score = home_score if is_home else away_score
    opp_score = away_score if is_home else home_score

    # Calculate pace (seconds per play)
    team_plays = team_games[team_games['posteam'] == team]
    if len(team_plays) > 0:
        avg_pace = 29.0  # Default NFL average
    else:
        avg_pace = 29.0

    # Calculate defensive EPA for opponent
    def_epa = calculate_defensive_epa(opponent, position, week, pbp_df)

    # Calculate team usage from actual game (backtest uses actuals)
    team_pass_plays = team_games[
        (team_games['posteam'] == team) &
        (team_games['play_type'] == 'pass')
    ]
    team_rush_plays = team_games[
        (team_games['posteam'] == team) &
        (team_games['play_type'] == 'run')
    ]

    pass_attempts = len(team_pass_plays) if len(team_pass_plays) > 0 else 35.0
    rush_attempts = len(team_rush_plays) if len(team_rush_plays) > 0 else 28.0
    targets = len(team_pass_plays) if len(team_pass_plays) > 0 else 35.0  # Approximate targets as pass plays

    return {
        'opponent': opponent,
        'projected_team_total': team_score if team_score > 0 else 22.5,
        'projected_opponent_total': opp_score if opp_score > 0 else 22.5,
        'projected_game_script': team_score - opp_score if team_score > 0 else 0.0,
        'projected_pace': avg_pace,
        'opponent_def_epa_vs_position': def_epa,
        'projected_team_pass_attempts': pass_attempts,
        'projected_team_rush_attempts': rush_attempts,
        'projected_team_targets': targets,
    }


def generate_predictions_for_week(week: int, simulator: PlayerSimulator, all_weeks_stats: pd.DataFrame,
                                 pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Generate V3 predictions for all players in a given week."""
    print(f"\n📊 Generating V3 predictions for Week {week}...")

    # Load actual stats to know which players to predict
    actuals = load_sleeper_actuals(week)

    if actuals.empty:
        print(f"❌ No actuals for Week {week}, cannot generate predictions")
        return pd.DataFrame()

    # Filter to relevant players (those with actual stats)
    relevant_players = actuals[
        (actuals['receiving_yards'] > 0) |
        (actuals['rushing_yards'] > 0) |
        (actuals['passing_yards'] > 0)
    ].copy()

    print(f"   Found {len(relevant_players)} players with stats in Week {week}")

    predictions = []
    successful = 0
    failed = 0

    for idx, row in relevant_players.iterrows():
        player_name = row['player_name']
        team = row['team']
        position = row['position']

        # Skip non-skill positions
        if position not in ['QB', 'RB', 'WR', 'TE']:
            continue

        try:
            # Calculate trailing stats from historical data
            trailing = calculate_trailing_stats(
                player_name=player_name,
                position=position,
                team=team,
                current_week=week,
                all_weeks_stats=all_weeks_stats,
                lookback=4
            )

            # Load game context (now includes defensive EPA)
            game_ctx = load_game_context(week, team, position, pbp_df)

            # Create player input with proper parameters
            player_input = PlayerPropInput(
                player_id=f"{player_name.lower().replace(' ', '_')}_{week}",
                player_name=player_name,
                team=team,
                position=position,
                week=week,
                opponent=game_ctx['opponent'],
                projected_team_total=game_ctx['projected_team_total'],
                projected_opponent_total=game_ctx['projected_opponent_total'],
                projected_game_script=game_ctx['projected_game_script'],
                projected_pace=game_ctx['projected_pace'],
                trailing_snap_share=trailing['snap_share'],
                trailing_target_share=trailing['target_share'],
                trailing_carry_share=trailing['carry_share'],
                trailing_yards_per_opportunity=trailing['yards_per_opp'],
                trailing_td_rate=trailing['td_rate'],
                opponent_def_epa_vs_position=game_ctx['opponent_def_epa_vs_position'],
                projected_team_pass_attempts=game_ctx['projected_team_pass_attempts'],
                projected_team_rush_attempts=game_ctx['projected_team_rush_attempts'],
                projected_team_targets=game_ctx['projected_team_targets'],
            )

            # Simulate
            result = simulator.simulate_player(player_input)

            if result:
                pred = {
                    'week': week,
                    'player_name': player_name,
                    'team': team,
                    'position': position,
                }

                # Extract predictions
                for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards',
                                  'passing_tds', 'rushing_tds', 'receiving_tds',
                                  'receptions', 'targets', 'rushing_attempts']:
                    if stat_type in result:
                        pred[f'{stat_type}_samples'] = result[stat_type]
                        pred[f'{stat_type}_mean'] = np.mean(result[stat_type])
                        pred[f'{stat_type}_median'] = np.median(result[stat_type])
                        pred[f'{stat_type}_std'] = np.std(result[stat_type])

                predictions.append(pred)
                successful += 1

        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first 5 errors
                print(f"   ⚠️  Error predicting {player_name} ({position}): {e}")
            continue

    print(f"   ✅ Generated {successful} predictions ({failed} failed)")

    return pd.DataFrame(predictions)


def calculate_metrics(predictions: pd.DataFrame, actuals: pd.DataFrame, week: int = None) -> Dict:
    """Calculate comprehensive performance metrics."""
    metrics = {
        'n_players': 0,
        'by_stat': {}
    }

    # Position-specific stat types to avoid mixing (e.g., RB trick plays counted as passing yards)
    position_stats = [
        ('QB', 'passing_yards'),
        ('RB', 'rushing_yards'),
        ('WR', 'receiving_yards'),
        ('TE', 'receiving_yards'),
    ]

    for position, stat_type in position_stats:
        pred_col = f'{stat_type}_samples'
        actual_col = stat_type

        if pred_col not in predictions.columns or actual_col not in actuals.columns:
            continue
        if 'position' not in predictions.columns or 'position' not in actuals.columns:
            continue

        # Filter to specific position to avoid cross-contamination
        position_preds = predictions[predictions['position'] == position].copy()
        position_actuals = actuals[actuals['position'] == position].copy()

        if len(position_preds) == 0 or len(position_actuals) == 0:
            continue

        # Merge predictions with actuals
        merged = position_preds.merge(
            position_actuals[['player_name', 'team', 'week', actual_col]],
            on=['player_name', 'team', 'week'],
            how='inner',
            suffixes=('', '_actual')
        )

        # Use position-specific key for metrics
        metric_key = f'{position}_{stat_type}'

        if len(merged) == 0:
            continue

        crps_scores = []
        mae_scores = []
        coverage_50 = []
        coverage_80 = []
        coverage_90 = []

        for _, row in merged.iterrows():
            if pd.notna(row[actual_col]) and row[pred_col] is not None:
                samples = row[pred_col]
                actual = row[actual_col]

                # CRPS
                try:
                    crps = CalibrationMetrics.crps(samples, actual)
                    crps_scores.append(crps)
                except:
                    pass

                # MAE
                mean_col = f'{stat_type}_mean'
                if mean_col in row and pd.notna(row[mean_col]):
                    predicted_mean = row[mean_col]
                    mae = abs(predicted_mean - actual)
                    mae_scores.append(mae)

                # Coverage
                p5, p25, p75, p95 = np.percentile(samples, [5, 25, 75, 95])
                coverage_50.append(p25 <= actual <= p75)
                coverage_80.append(p5 <= actual <= p95)

                p5_90, p95_90 = np.percentile(samples, [5, 95])
                in_90 = p5_90 <= actual <= p95_90
                coverage_90.append(in_90)

                # DEBUG: Log first few QB passing yards to understand the issue
                if stat_type == 'passing_yards' and len(coverage_90) <= 10:
                    pred_mean = np.mean(samples)
                    print(f"[DEBUG] {row.get('player_name', 'Unknown')} (Week {week}):")
                    print(f"  Predicted: mean={pred_mean:.1f}, std={np.std(samples):.1f}")
                    print(f"  90% interval: [{p5_90:.1f}, {p95_90:.1f}] (width={p95_90-p5_90:.1f})")
                    print(f"  Actual: {actual}")
                    print(f"  In interval: {in_90}, Error: {actual - pred_mean:.1f} yards")

        if crps_scores:
            if stat_type == 'passing_yards' and position == 'QB':
                print(f"[DEBUG] {position} {stat_type}: {len(coverage_90)} total, {sum(coverage_90)} in interval, coverage={100*sum(coverage_90)/len(coverage_90):.1f}%")
            metrics['by_stat'][metric_key] = {
                'n': len(crps_scores),
                'crps_mean': np.mean(crps_scores),
                'crps_median': np.median(crps_scores),
                'mae_mean': np.mean(mae_scores),
                'mae_median': np.median(mae_scores),
                'coverage_50': np.mean(coverage_50) * 100,
                'coverage_80': np.mean(coverage_80) * 100,
                'coverage_90': np.mean(coverage_90) * 100,
            }

    metrics['n_players'] = len(predictions)

    return metrics


def run_backtest(weeks: List[int]) -> None:
    """Run comprehensive backtest for specified weeks."""
    print(f"\n{'='*80}")
    print(f"V3 BACKTEST: WEEKS {min(weeks)}-{max(weeks)}")
    print(f"{'='*80}\n")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weeks: {weeks}")
    print()

    # Load all historical stats (weeks 1-8)
    print("📥 Loading historical NFLverse stats...")
    file_path = Path("data/nflverse_cache/stats_player_week_2025.csv")
    if file_path.exists():
        all_weeks_stats = pd.read_csv(file_path)
        all_weeks_stats = all_weeks_stats[all_weeks_stats['week'].between(1, 8)].copy()

        # Standardize column names (same as load_sleeper_actuals)
        all_weeks_stats = all_weeks_stats.rename(columns={
            'carries': 'rushing_attempts',
            'attempts': 'passing_attempts',
            'completions': 'passing_completions',
        })

        # Fill NaN values with 0
        stat_columns = [
            'rushing_yards', 'rushing_attempts', 'rushing_tds',
            'receiving_yards', 'receptions', 'targets', 'receiving_tds',
            'passing_yards', 'passing_attempts', 'passing_completions', 'passing_tds'
        ]
        for col in stat_columns:
            if col in all_weeks_stats.columns:
                all_weeks_stats[col] = all_weeks_stats[col].fillna(0)

        print(f"   ✅ Loaded {len(all_weeks_stats)} player-week records\n")
    else:
        print(f"   ❌ NFLverse stats not found: {file_path}")
        all_weeks_stats = pd.DataFrame()

    # Load nflverse play-by-play data
    print("📥 Loading nflverse play-by-play data...")
    pbp_df = pd.read_parquet("data/nflverse/pbp_2025.parquet")
    print(f"   ✅ Loaded {len(pbp_df)} plays from {pbp_df.game_id.nunique()} games\n")

    # Load predictors
    print("📥 Loading V3 predictors...")
    usage_predictor, efficiency_predictor = load_predictors()

    # Create V3 simulator
    simulator = PlayerSimulator(
        usage_predictor,
        efficiency_predictor,
        trials=5000,  # Sufficient for CRPS
        seed=42
    )
    print("✅ V3 simulator ready\n")

    all_results = []
    all_predictions = []  # Collect all predictions for CSV export

    for week in weeks:
        print(f"\n{'='*80}")
        print(f"WEEK {week}")
        print(f"{'='*80}")

        # Generate predictions
        predictions = generate_predictions_for_week(week, simulator, all_weeks_stats, pbp_df)

        # Collect predictions for later export
        if not predictions.empty:
            all_predictions.append(predictions)

        if predictions.empty:
            print(f"❌ No predictions for Week {week}")
            continue

        # Load actuals
        actuals = load_sleeper_actuals(week)

        if actuals.empty:
            print(f"❌ No actuals for Week {week}")
            continue

        # Calculate metrics
        metrics = calculate_metrics(predictions, actuals, week=week)
        metrics['week'] = week
        all_results.append(metrics)

        # Print week summary
        print(f"\n📊 Week {week} Results:")
        print(f"   Players predicted: {metrics['n_players']}")

        for stat_type, stat_metrics in metrics.get('by_stat', {}).items():
            print(f"\n   {stat_type}:")
            print(f"      CRPS (mean):  {stat_metrics['crps_mean']:.2f}")
            print(f"      MAE (mean):   {stat_metrics['mae_mean']:.2f}")
            print(f"      Coverage 50%: {stat_metrics['coverage_50']:.1f}%")
            print(f"      Coverage 80%: {stat_metrics['coverage_80']:.1f}%")
            print(f"      Coverage 90%: {stat_metrics['coverage_90']:.1f}%")

    # Generate summary report
    print(f"\n\n{'='*80}")
    print("AGGREGATE RESULTS (All Weeks)")
    print(f"{'='*80}\n")

    # Aggregate metrics across all weeks
    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        crps_values = []
        mae_values = []
        cov_90_values = []

        for result in all_results:
            if stat_type in result.get('by_stat', {}):
                crps_values.append(result['by_stat'][stat_type]['crps_mean'])
                mae_values.append(result['by_stat'][stat_type]['mae_mean'])
                cov_90_values.append(result['by_stat'][stat_type]['coverage_90'])

        if crps_values:
            print(f"{stat_type}:")
            print(f"   CRPS (avg): {np.mean(crps_values):.2f}")
            print(f"   MAE (avg):  {np.mean(mae_values):.2f}")
            print(f"   Coverage 90% (avg): {np.mean(cov_90_values):.1f}%")
            print()

    # Save results
    output_file = Path(f"reports/v3_backtest_weeks_{min(weeks)}_{max(weeks)}.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")

    # Save all predictions to CSV for betting analysis
    if all_predictions:
        all_preds_df = pd.concat(all_predictions, ignore_index=True)

        # Create a flattened version for betting analysis
        betting_rows = []
        for _, row in all_preds_df.iterrows():
            for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
                samples_col = f'{stat_type}_samples'
                if samples_col in row and row[samples_col] is not None:
                    # Check if samples is an array (not NaN or a scalar)
                    samples = row[samples_col]
                    if hasattr(samples, '__iter__') and not isinstance(samples, str):
                        try:
                            betting_rows.append({
                                'week': row['week'],
                                'player_name': row['player_name'],
                                'team': row['team'],
                                'position': row['position'],
                                'stat_type': stat_type,
                                'mean': row[f'{stat_type}_mean'],
                                'median': row[f'{stat_type}_median'],
                                'std': row[f'{stat_type}_std'],
                                'samples': ','.join(map(str, samples[:100]))  # Save first 100 samples
                            })
                        except (TypeError, KeyError):
                            continue

        betting_df = pd.DataFrame(betting_rows)
        predictions_csv = Path(f"reports/v3_backtest_predictions.csv")
        betting_df.to_csv(predictions_csv, index=False)
        print(f"✅ Predictions saved to: {predictions_csv} ({len(betting_df)} player-stat combinations)")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="V3 Backtest for weeks 1-8")
    parser.add_argument(
        '--weeks',
        type=str,
        default='1-8',
        help='Week range (e.g., "1-8") or single week (e.g., "1")'
    )

    args = parser.parse_args()

    # Parse weeks
    if '-' in args.weeks:
        start, end = map(int, args.weeks.split('-'))
        weeks = list(range(start, end + 1))
    else:
        weeks = [int(args.weeks)]

    run_backtest(weeks)


if __name__ == '__main__':
    main()
