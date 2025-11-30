#!/usr/bin/env python3
"""
Generate Game Line Predictions (Totals, Spreads, Moneylines)

Uses team-level aggregations from player projections to predict:
- Game totals (over/under total points)
- Spreads (point differential)
- Moneylines (win probability)

Usage:
    python scripts/predict/generate_game_line_predictions.py --week 12
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple
import joblib
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.features.team_strength import EnhancedEloCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_avg_field_goals_per_game(season: int = None) -> float:
    """
    Calculate average field goals made per team per game from NFLverse data.

    Args:
        season: NFL season (default: None - auto-detects current season)

    Returns:
        Average FGs made per team per game
    """
    if season is None:
        season = get_current_season()
    weekly_stats_path = Path('data/nflverse/weekly_stats.parquet')
    if not weekly_stats_path.exists():
        logger.warning(f"NFLverse data not found, using fallback: 1.72 FGs/game (2024 average)")
        return 1.72

    weekly = pd.read_parquet(weekly_stats_path)

    # Get kickers from current season
    kickers = weekly[
        (weekly['position'] == 'K') &
        (weekly['season'] == season) &
        (weekly['season_type'] == 'REG')
    ]

    if len(kickers) == 0:
        logger.warning(f"No kicker data for {season}, using fallback: 1.72 FGs/game (2024 average)")
        return 1.72

    # Calculate average FGs made per team per game
    avg_fgs = kickers.groupby(['team', 'week'])['fg_made'].sum().mean()

    logger.info(f"Calculated average FGs per game from {season} data: {avg_fgs:.2f}")
    return avg_fgs


def calculate_points_per_td(season: int = None) -> float:
    """
    Calculate realistic points per TD from XP success rate.

    Uses actual NFLverse data instead of hardcoded 7.0.

    Args:
        season: NFL season (default: None - auto-detects current season)

    Returns:
        Expected points per TD (6.0 + XP success rate)
    """
    if season is None:
        season = get_current_season()
    weekly_stats_path = Path('data/nflverse/weekly_stats.parquet')
    if not weekly_stats_path.exists():
        logger.warning(f"NFLverse data not found, using fallback: 6.96 pts/TD (95.6% XP rate)")
        return 6.96

    weekly = pd.read_parquet(weekly_stats_path)

    # Get kickers from current season
    kickers = weekly[
        (weekly['position'] == 'K') &
        (weekly['season'] == season) &
        (weekly['season_type'] == 'REG')
    ]

    if len(kickers) == 0:
        logger.warning(f"No kicker data for {season}, using fallback: 6.96 pts/TD (95.6% XP rate)")
        return 6.96

    # Calculate XP success rate
    total_xp_made = kickers['pat_made'].sum()
    total_xp_att = kickers['pat_att'].sum()

    if total_xp_att == 0:
        logger.warning(f"No XP attempts found for {season}, using fallback: 6.96 pts/TD")
        return 6.96

    xp_success_rate = total_xp_made / total_xp_att
    points_per_td = 6.0 + xp_success_rate

    logger.info(f"Calculated points per TD from {season} data: {points_per_td:.3f} (XP rate: {xp_success_rate:.3f})")
    return points_per_td


def calculate_empirical_total_std(season: int = None) -> float:
    """
    Calculate empirical standard deviation of game totals.

    Uses actual game data instead of hardcoded 14.14.

    Args:
        season: NFL season to analyze (default: None - auto-detects current season)

    Returns:
        Standard deviation of game totals
    """
    if season is None:
        season = get_current_season()
    schedules_path = Path('data/nflverse/schedules.parquet')
    if not schedules_path.exists():
        logger.warning(f"Schedules data not found, using fallback: 13.11 (2024 empirical)")
        return 13.11

    schedules = pd.read_parquet(schedules_path)

    # Get completed regular season games
    games = schedules[
        (schedules['season'] == season) &
        (schedules['game_type'] == 'REG') &
        (schedules['home_score'].notna()) &
        (schedules['away_score'].notna())
    ]

    if len(games) == 0:
        logger.warning(f"No completed games found for {season}, using fallback: 13.11")
        return 13.11

    # Calculate game totals
    game_totals = games['home_score'] + games['away_score']
    total_std = game_totals.std()

    logger.info(f"Calculated total SD from {season} data ({len(games)} games): {total_std:.2f}")
    return total_std


def load_player_predictions(week: int) -> pd.DataFrame:
    """Load player predictions for the week."""
    pred_path = Path(f'data/model_predictions_week{week}.csv')
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Player predictions not found: {pred_path}\n"
            f"Run: python scripts/predict/generate_model_predictions.py {week}"
        )

    df = pd.read_csv(pred_path)
    logger.info(f"Loaded {len(df)} player predictions")
    return df


def load_game_simulations(week: int) -> Dict[str, Dict]:
    """Load game simulation files."""
    import json

    sim_dir = Path('reports')
    sim_files = list(sim_dir.glob(f'sim_2025_{week}_*.json'))

    games = {}
    for sim_file in sim_files:
        with open(sim_file) as f:
            data = json.load(f)
            game_key = f"{data['away_team']} @ {data['home_team']}"
            games[game_key] = data

    logger.info(f"Loaded {len(games)} game simulations")
    return games


def load_td_calibration_factors() -> Dict[str, float]:
    """Load TD calibration factors from trained model."""
    calibration_path = Path('data/models/td_calibration_factors.joblib')
    if calibration_path.exists():
        factors = joblib.load(calibration_path)
        logger.info(f"Loaded TD calibration: rush={factors['rush_td_factor']:.3f}, rec={factors['rec_td_factor']:.3f}")
        return factors
    else:
        logger.warning("TD calibration factors not found, using defaults")
        return {
            'rush_td_factor': 0.808,  # Historical calibration
            'rec_td_factor': 0.607,
            'historical_rush_tds': 0.96,
            'historical_rec_tds': 1.48
        }


def aggregate_team_projections(
    predictions: pd.DataFrame,
    team: str,
    avg_field_goals: float,
    points_per_td: float,
    td_calibration: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Aggregate player projections to team level with TD calibration.

    TD calibration ensures team totals match historical averages:
    - Historical rushing TDs per team: ~0.96
    - Historical receiving TDs per team: ~1.48
    - Historical total TDs per team: ~2.44

    NOTE (Nov 27, 2025): Game lines are kept SEPARATE from opponent-aware player props.
    Player props use opponent adjustments (V13), but game lines use base projections.
    This prevents double-counting and keeps the two systems independent.
    """
    team_players = predictions[predictions['team'] == team].copy()

    # Load calibration if not provided
    if td_calibration is None:
        td_calibration = load_td_calibration_factors()

    # Raw TD sums
    raw_rushing_tds = team_players['rushing_tds_mean'].sum()
    raw_receiving_tds = team_players['receiving_tds_mean'].sum()

    # Apply TD calibration factors
    # CRITICAL FIX (Nov 24, 2025): Scale TDs to match historical rates
    calibrated_rushing_tds = raw_rushing_tds * td_calibration['rush_td_factor']
    calibrated_receiving_tds = raw_receiving_tds * td_calibration['rec_td_factor']

    # Use ORIGINAL (non-opponent-adjusted) projections for game lines
    # Check if original_mean columns exist, otherwise use the standard mean
    passing_col = 'player_pass_yds_original_mean'
    if passing_col in team_players.columns:
        # Use original (pre-adjustment) passing yards for game lines
        passing_yards = team_players[passing_col].fillna(team_players['passing_yards_mean']).sum()
    else:
        passing_yards = team_players['passing_yards_mean'].sum()

    # Sum up team totals
    agg = {
        'passing_yards': passing_yards,
        'passing_tds': team_players['passing_tds_mean'].sum(),
        'rushing_yards': team_players['rushing_yards_mean'].sum(),
        'rushing_tds': calibrated_rushing_tds,  # Calibrated
        'receiving_tds': calibrated_receiving_tds,  # Calibrated
        'raw_rushing_tds': raw_rushing_tds,  # For debugging
        'raw_receiving_tds': raw_receiving_tds,  # For debugging
        'total_tds': calibrated_rushing_tds + calibrated_receiving_tds,
        'total_yards': (
            passing_yards +
            team_players['rushing_yards_mean'].sum()
        ),
    }

    # Estimate points from TDs + field goals
    # CRITICAL FIX (Nov 23, 2025): Use actual NFL data (Framework Rule 1.2)
    # Dynamically calculated from current season NFLverse data
    agg['projected_points'] = (
        agg['total_tds'] * points_per_td +  # Points per TD from actual XP rate
        avg_field_goals * 3.0  # Field goals from actual data
    )

    return agg


def predict_game_total(
    home_proj: Dict[str, float],
    away_proj: Dict[str, float],
    total_std: float
) -> Dict[str, float]:
    """Predict game total points."""
    home_points = home_proj['projected_points']
    away_points = away_proj['projected_points']

    total = home_points + away_points

    return {
        'total_mean': total,
        'total_std': total_std,
        'home_points': home_points,
        'away_points': away_points,
    }


def predict_spread(
    home_proj: Dict[str, float],
    away_proj: Dict[str, float],
    elo_features: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Predict point spread (positive = home favored).

    Uses Elo-based spread as anchor, then adjusts based on player projections.
    This prevents the model from making unrealistic predictions (e.g., NYG over DET).

    Args:
        home_proj: Home team player projection aggregates
        away_proj: Away team player projection aggregates
        elo_features: Elo-based features from EnhancedEloCalculator (optional)
    """
    home_points = home_proj['projected_points']
    away_points = away_proj['projected_points']

    # Player-based spread (raw)
    player_spread = home_points - away_points

    if elo_features is not None:
        # Elo-based spread includes home field advantage and bye week effects
        elo_spread = elo_features.get('elo_spread', 0.0)

        # CRITICAL: Blend Elo (60%) with player projections (40%)
        # Elo is more reliable for team strength, player projections capture injuries/context
        # This prevents wild predictions like NYG beating DET
        spread = 0.6 * elo_spread + 0.4 * player_spread

        # Use Elo win probability as anchor (already accounts for home field)
        elo_win_prob = elo_features.get('home_win_prob', 0.5)
    else:
        # Fallback: use fixed home field advantage
        spread = player_spread + 2.5
        elo_win_prob = None

    # Spread variance (historical std ~13.5 points)
    spread_std = 13.5

    result = {
        'spread_mean': spread,
        'spread_std': spread_std,
        'player_spread': player_spread,
        'home_win_prob': calculate_win_probability(spread, spread_std),
    }

    if elo_features is not None:
        result['elo_spread'] = elo_features.get('elo_spread', 0.0)
        result['elo_win_prob'] = elo_win_prob
        result['home_elo'] = elo_features.get('home_elo', 1505)
        result['away_elo'] = elo_features.get('away_elo', 1505)
        result['elo_diff'] = elo_features.get('elo_diff', 0)

    return result


def calculate_win_probability(spread: float, std: float) -> float:
    """Calculate win probability from spread."""
    from scipy.stats import norm

    # Home team wins if they cover spread of 0
    # P(home_points - away_points > 0) = P(spread > 0)
    win_prob = norm.cdf(spread / std)

    return win_prob


def simulate_game_outcomes(
    total_mean: float,
    total_std: float,
    spread_mean: float,
    spread_std: float,
    n_trials: int = 10000
) -> Dict[str, np.ndarray]:
    """Monte Carlo simulation of game outcomes."""
    # Sample totals and spreads
    totals = np.random.normal(total_mean, total_std, n_trials)
    spreads = np.random.normal(spread_mean, spread_std, n_trials)

    # Calculate individual team scores
    # total = home + away
    # spread = home - away
    # Solving: home = (total + spread) / 2
    #         away = (total - spread) / 2
    home_scores = (totals + spreads) / 2
    away_scores = (totals - spreads) / 2

    return {
        'totals': totals,
        'spreads': spreads,
        'home_scores': home_scores,
        'away_scores': away_scores,
    }


def get_games_from_predictions(predictions: pd.DataFrame, week: int, season: int = None) -> list:
    """
    Extract unique games from player predictions using schedules for correct home/away.

    Player predictions have both teams in 'team' column, we need schedules to determine
    which team is home vs away.
    """
    if season is None:
        season = get_current_season()

    # Get unique team-opponent pairs (both directions)
    games_df = predictions[['team', 'opponent']].drop_duplicates()

    # Load schedules to get correct home/away designation
    schedules_path = Path('data/nflverse/schedules.parquet')
    if not schedules_path.exists():
        raise FileNotFoundError(f"Schedules file not found: {schedules_path}")

    schedules = pd.read_parquet(schedules_path)

    week_games = schedules[
        (schedules['season'] == season) &
        (schedules['week'] == week) &
        (schedules['game_type'] == 'REG')
    ]

    game_list = []
    for _, game in week_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']

        # Check if we have predictions for this game
        has_home = ((predictions['team'] == home_team) & (predictions['opponent'] == away_team)).any()
        has_away = ((predictions['team'] == away_team) & (predictions['opponent'] == home_team)).any()

        if has_home or has_away:
            game_list.append({
                'game': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team
            })

    return game_list


def generate_game_predictions(week: int, season: int = None) -> pd.DataFrame:
    """
    Generate all game line predictions for the week.

    Integrates:
    - Player projections (totals, injuries, context)
    - Elo-based team strength (prevents bad teams being favored)
    - Vegas sanity checks (flags predictions > 7 points from market)
    """
    if season is None:
        season = get_current_season()

    # Load data
    predictions = load_player_predictions(week)

    # Calculate parameters from actual NFLverse data (Framework Rule 1.2)
    avg_field_goals = calculate_avg_field_goals_per_game()
    points_per_td = calculate_points_per_td()
    total_std = calculate_empirical_total_std()

    # Load TD calibration factors
    td_calibration = load_td_calibration_factors()

    # Initialize Elo calculator for team strength
    elo_calc = EnhancedEloCalculator()
    logger.info("Initialized Elo calculator for team strength")

    # Load Vegas lines for sanity checks
    schedules = pd.read_parquet('data/nflverse/schedules.parquet')
    week_schedule = schedules[
        (schedules['season'] == season) &
        (schedules['week'] == week)
    ]

    # Get games
    games = get_games_from_predictions(predictions, week=week)
    logger.info(f"Found {len(games)} games")

    results = []

    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        logger.info(f"Processing {game['game']}")

        # Get Elo features for this matchup (from home team perspective)
        elo_raw = elo_calc.get_team_features(home_team, away_team, season, week, is_home=True)

        # Map to expected format
        elo_features = {
            'home_elo': elo_raw['team_elo'],
            'away_elo': elo_raw['opp_elo'],
            'elo_diff': elo_raw['elo_diff'],
            'home_win_prob': elo_raw['win_probability'],
            'elo_spread': elo_raw['expected_spread'],  # Positive = home favored
            'vegas_spread': None,  # Will get from schedule
            'vegas_total': None,
        }

        # Get Vegas lines from schedule
        game_row = week_schedule[
            (week_schedule['home_team'] == home_team) &
            (week_schedule['away_team'] == away_team)
        ]
        if len(game_row) > 0:
            elo_features['vegas_spread'] = game_row.iloc[0].get('spread_line')
            elo_features['vegas_total'] = game_row.iloc[0].get('total_line')

        # Aggregate team projections (with TD calibration)
        home_proj = aggregate_team_projections(
            predictions, home_team, avg_field_goals, points_per_td, td_calibration
        )
        away_proj = aggregate_team_projections(
            predictions, away_team, avg_field_goals, points_per_td, td_calibration
        )

        # Predict total
        total_pred = predict_game_total(home_proj, away_proj, total_std)

        # Predict spread WITH ELO (critical fix)
        spread_pred = predict_spread(home_proj, away_proj, elo_features)

        # Monte Carlo simulation
        sim = simulate_game_outcomes(
            total_pred['total_mean'],
            total_pred['total_std'],
            spread_pred['spread_mean'],
            spread_pred['spread_std']
        )

        # Get Vegas lines for sanity check
        vegas_spread = elo_features.get('vegas_spread')
        vegas_total = elo_features.get('vegas_total')

        # Sanity checks
        spread_diff = None
        total_diff = None
        spread_sane = True
        total_sane = True

        if vegas_spread is not None and not pd.isna(vegas_spread):
            spread_diff = abs(spread_pred['spread_mean'] - vegas_spread)
            spread_sane = spread_diff <= 7.0
            if not spread_sane:
                logger.warning(
                    f"⚠️ {game['game']}: Spread differs from Vegas by {spread_diff:.1f} pts "
                    f"(Model: {spread_pred['spread_mean']:+.1f}, Vegas: {vegas_spread:+.1f})"
                )

        if vegas_total is not None and not pd.isna(vegas_total):
            total_diff = abs(total_pred['total_mean'] - vegas_total)
            total_sane = total_diff <= 7.0
            if not total_sane:
                logger.warning(
                    f"⚠️ {game['game']}: Total differs from Vegas by {total_diff:.1f} pts "
                    f"(Model: {total_pred['total_mean']:.1f}, Vegas: {vegas_total:.1f})"
                )

        # Store results
        result = {
            'game': game['game'],
            'home_team': home_team,
            'away_team': away_team,
            'week': week,

            # Total predictions
            'projected_total': total_pred['total_mean'],
            'total_std': total_pred['total_std'],
            'home_projected_points': total_pred['home_points'],
            'away_projected_points': total_pred['away_points'],

            # Spread predictions
            'projected_spread': spread_pred['spread_mean'],
            'spread_std': spread_pred['spread_std'],
            'home_win_prob': spread_pred['home_win_prob'],
            'away_win_prob': 1 - spread_pred['home_win_prob'],

            # Elo features
            'home_elo': spread_pred.get('home_elo', 1505),
            'away_elo': spread_pred.get('away_elo', 1505),
            'elo_spread': spread_pred.get('elo_spread', 0),
            'player_spread': spread_pred.get('player_spread', 0),

            # Vegas comparison
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
            'spread_diff_vs_vegas': spread_diff,
            'total_diff_vs_vegas': total_diff,
            'spread_sane': spread_sane,
            'total_sane': total_sane,

            # Simulation percentiles
            'total_p25': np.percentile(sim['totals'], 25),
            'total_p50': np.percentile(sim['totals'], 50),
            'total_p75': np.percentile(sim['totals'], 75),
            'spread_p25': np.percentile(sim['spreads'], 25),
            'spread_p50': np.percentile(sim['spreads'], 50),
            'spread_p75': np.percentile(sim['spreads'], 75),
        }

        results.append(result)

    df = pd.DataFrame(results)

    # Summary stats
    if len(df) > 0:
        sane_spreads = df['spread_sane'].sum() if 'spread_sane' in df.columns else len(df)
        sane_totals = df['total_sane'].sum() if 'total_sane' in df.columns else len(df)
        logger.info(f"Sanity check: {sane_spreads}/{len(df)} spreads, {sane_totals}/{len(df)} totals within 7 pts of Vegas")

    return df


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Generate game line predictions'
    )
    parser.add_argument(
        '--week',
        type=int,
        default=12,
        help='Week number (default: 12)'
    )
    args = parser.parse_args()

    logger.info(f"Generating game line predictions for Week {args.week}")

    # Generate predictions
    predictions = generate_game_predictions(args.week)

    # Save
    output_path = Path(f'data/game_line_predictions_week{args.week}.csv')
    predictions.to_csv(output_path, index=False)

    logger.info(f"✅ Saved {len(predictions)} game predictions to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("GAME LINE PREDICTIONS SUMMARY (with Elo)")
    print("="*80)
    print(f"\nWeek {args.week} - {len(predictions)} games\n")

    for _, row in predictions.iterrows():
        # Sanity check emoji
        spread_ok = "✅" if row.get('spread_sane', True) else "⚠️"
        total_ok = "✅" if row.get('total_sane', True) else "⚠️"

        print(f"{row['game']}")
        print(f"  Elo: {row['home_team']} {row.get('home_elo', 1505):.0f} vs {row['away_team']} {row.get('away_elo', 1505):.0f}")
        print(f"  Projected Total: {row['projected_total']:.1f} {total_ok}")
        if row['projected_spread'] > 0:
            print(f"  Projected Spread: {row['projected_spread']:+.1f} ({row['home_team']} favored) {spread_ok}")
        else:
            print(f"  Projected Spread: {row['projected_spread']:+.1f} ({row['away_team']} favored) {spread_ok}")
        print(f"  Home Win Prob: {row['home_win_prob']:.1%}")

        # Show Vegas comparison if available
        if pd.notna(row.get('vegas_spread')):
            print(f"  Vegas Spread: {row['vegas_spread']:+.1f} | Diff: {row.get('spread_diff_vs_vegas', 0):.1f} pts")
        if pd.notna(row.get('vegas_total')):
            print(f"  Vegas Total: {row['vegas_total']:.1f} | Diff: {row.get('total_diff_vs_vegas', 0):.1f} pts")
        print()


if __name__ == '__main__':
    main()
