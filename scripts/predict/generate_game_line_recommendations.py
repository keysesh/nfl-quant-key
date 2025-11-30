#!/usr/bin/env python3
"""
NFL QUANT - Game Line Recommendation Generator

Generates spread, total, and moneyline recommendations by:
1. Running Monte Carlo simulations with team EPA data
2. Comparing model predictions to market odds
3. Applying calibration to probabilities
4. Calculating edge and Kelly optimal bet sizing

Uses the same rigorous framework as player props pipeline.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput, SimulationOutput
from nfl_quant.utils.season_utils import get_current_season, get_current_week
from nfl_quant.core.unified_betting import (
    american_odds_to_implied_prob,
    remove_vig_two_way,
    calculate_edge_percentage,
    calculate_kelly_fraction,
    assign_confidence_tier,
    calculate_expected_roi,
    select_best_side,
)
from nfl_quant.features.injuries import InjuryImpactModel
from nfl_quant.utils.epa_utils import regress_epa_to_mean  # Centralized EPA utility

PROJECT_ROOT = Path(__file__).parent.parent.parent


def apply_probability_shrinkage(prob: float, shrinkage_factor: float = 0.30) -> float:
    """
    Shrink probability toward 50% to reduce model overconfidence.

    NFL markets are highly efficient, so extreme probabilities (70%+) are often
    overconfident. This applies a conservative shrinkage toward 50% (no edge).

    Args:
        prob: Raw model probability (0-1)
        shrinkage_factor: How much to shrink toward 50% (0.30 = 30% shrinkage)

    Returns:
        Shrunken probability

    Example:
        75% → 50% + 0.70 * (75% - 50%) = 67.5%
        60% → 50% + 0.70 * (60% - 50%) = 57.0%
    """
    return 0.5 + (1.0 - shrinkage_factor) * (prob - 0.5)

# Auto-detect current week and season
AUTO_WEEK = get_current_week()
AUTO_SEASON = get_current_season()

CURRENT_WEEK = int(os.getenv('CURRENT_WEEK', AUTO_WEEK))
CURRENT_SEASON = int(os.getenv('CURRENT_SEASON', AUTO_SEASON))

# File paths
ODDS_FILE = PROJECT_ROOT / "data" / f"odds_week{CURRENT_WEEK}.csv"
SCHEDULES_FILE = PROJECT_ROOT / "data" / "nflverse" / "schedules.parquet"
PBP_FILE = PROJECT_ROOT / "data" / "nflverse" / f"pbp_{CURRENT_SEASON}.parquet"
OUTPUT_FILE = PROJECT_ROOT / "reports" / f"WEEK{CURRENT_WEEK}_GAME_LINE_RECOMMENDATIONS.csv"
CALIBRATOR_FILE = PROJECT_ROOT / "configs" / "game_line_calibrator.json"


# Legacy functions kept for backward compatibility
# New code should use unified_betting module

def american_odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    return american_odds_to_implied_prob(odds)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def calculate_team_epa(pbp_df: pd.DataFrame, team: str, weeks: int = 10) -> dict:
    """
    Calculate team EPA metrics from play-by-play data with regression to mean.

    Returns:
        dict with offensive_epa, defensive_epa, pass_epa, rush_epa, pace

    IMPORTANT: EPA values are REGRESSED toward league mean (0.0) to prevent
    extreme predictions from small sample sizes. This is standard practice
    in sports analytics.

    defensive_epa represents what opponents score AGAINST this team:
    - Positive = opponents score well (bad defense)
    - Negative = opponents score poorly (good defense)
    """
    # Filter to relevant weeks (use available data before current week)
    max_week = pbp_df['week'].max()
    recent_weeks = range(max(1, max_week - weeks + 1), max_week + 1)

    # Offensive EPA (when team has possession)
    off_plays = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(recent_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    # Defensive EPA (when team is defending)
    def_plays = pbp_df[
        (pbp_df['defteam'] == team) &
        (pbp_df['week'].isin(recent_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    if len(off_plays) == 0 or len(def_plays) == 0:
        return {
            'offensive_epa': 0.0,
            'defensive_epa': 0.0,
            'pass_epa': 0.0,
            'rush_epa': 0.0,
            'pace': 65.0,
            'sample_games': 0
        }

    # Count sample size for regression
    off_games = len(off_plays['game_id'].unique())
    def_games = len(def_plays['game_id'].unique())
    sample_games = min(off_games, def_games)

    # Calculate RAW EPA per play
    raw_offensive_epa = off_plays['epa'].mean()
    raw_defensive_epa = def_plays['epa'].mean()

    # CRITICAL: Regress EPA values toward league mean (0.0)
    # This prevents extreme predictions from small samples
    # The market is generally efficient, so extreme deviations are often noise
    offensive_epa = regress_epa_to_mean(raw_offensive_epa, sample_games)
    defensive_epa = regress_epa_to_mean(raw_defensive_epa, sample_games)

    # Split by play type (also regressed)
    pass_plays = off_plays[off_plays['play_type'] == 'pass']
    rush_plays = off_plays[off_plays['play_type'] == 'run']

    raw_pass_epa = pass_plays['epa'].mean() if len(pass_plays) > 0 else 0.0
    raw_rush_epa = rush_plays['epa'].mean() if len(rush_plays) > 0 else 0.0

    pass_epa = regress_epa_to_mean(raw_pass_epa, sample_games)
    rush_epa = regress_epa_to_mean(raw_rush_epa, sample_games)

    # Calculate pace (plays per game)
    pace = len(off_plays) / off_games if off_games > 0 else 65.0

    return {
        'offensive_epa': float(offensive_epa),
        'defensive_epa': float(defensive_epa),  # Regressed opponent EPA
        'pass_epa': float(pass_epa),
        'rush_epa': float(rush_epa),
        'pace': float(pace),
        'sample_games': sample_games
    }


def load_game_line_calibrator():
    """Load isotonic regression calibrator for win probabilities."""
    if not CALIBRATOR_FILE.exists():
        print(f"Warning: Calibrator not found at {CALIBRATOR_FILE}")
        return None

    with open(CALIBRATOR_FILE, 'r') as f:
        data = json.load(f)

    return data


def apply_calibration(prob: float, calibrator: dict) -> float:
    """
    Apply isotonic calibration to a probability.

    IMPORTANT FIX (Nov 23, 2025):
    - Previously clipped probabilities below X_min to y[0] = 0.0
    - This caused extreme underdogs (TEN vs SEA) to get 0% win prob
    - After shrinkage, 0% → 15%, but this was still incorrect
    - NEW: Use linear extrapolation for out-of-range probabilities
    - This preserves relative probability relationships
    """
    if calibrator is None:
        return prob

    X = np.array(calibrator['X_thresholds'])
    y = np.array(calibrator['y_thresholds'])

    idx = np.searchsorted(X, prob)
    if idx == 0:
        # OUT OF RANGE (below minimum)
        # CRITICAL FIX: For extreme underdogs, calibrator is unreliable
        # Return raw probability instead of clipping to 0.0
        # This preserves the relative probability (e.g., 1.74% for TEN vs SEA)
        # Shrinkage will later adjust this toward 50% conservatively
        return prob
    elif idx >= len(X):
        # OUT OF RANGE (above maximum)
        # CRITICAL FIX: For extreme favorites, calibrator is unreliable
        # Return raw probability instead of clipping to 1.0
        # This preserves the relative probability (e.g., 98.26% for SEA vs TEN)
        # Shrinkage will later adjust this toward 50% conservatively
        return prob
    else:
        # WITHIN RANGE - interpolate normally
        x0, x1 = X[idx-1], X[idx]
        y0, y1 = y[idx-1], y[idx]
        return y0 + (prob - x0) * (y1 - y0) / (x1 - x0)


def calculate_spread_cover_prob(model_fair_spread: float, market_spread: float,
                                 spread_std: float) -> float:
    """
    Calculate probability of home team covering the spread.

    Args:
        model_fair_spread: Model's predicted fair spread (home_score - away_score)
            Positive = home wins by that amount
            Negative = home loses by that amount
        market_spread: Market spread from odds file (home team's line)
            Negative = home is favored (e.g., -3.0 means home -3)
            Positive = home is underdog (e.g., +7.5 means home +7.5)
        spread_std: Standard deviation of spread from simulations

    Returns:
        Probability of home team covering the spread

    Convention:
        fair_spread = home_score - away_score
        market_spread = points the home team must beat (in spread betting)

        Home covers spread if: home_score - market_spread > away_score
        i.e., (home_score - away_score) > market_spread
        i.e., actual_spread > market_spread

    Example 1: CHI @ MIN
        - Model fair spread = -7 (MIN loses by 7, CHI wins by 7)
        - Market spread = -3 (MIN -3, MIN favored by 3)
        - MIN covers -3 if actual_spread > -3
        - Model says actual_spread ~ N(-7, 10)
        - P(actual_spread > -3) = P(Z > (-3 - (-7))/10) = P(Z > 0.4) = 34.5%

    Example 2: HOU @ TEN
        - Model fair spread = -22 (TEN loses by 22, HOU wins by 22)
        - Market spread = +7.5 (TEN +7.5, TEN is underdog by 7.5)
        - TEN covers +7.5 if actual_spread > -7.5 (TEN loses by less than 7.5)
        - Model says actual_spread ~ N(-22, 10)
        - P(actual_spread > -7.5) = P(Z > (-7.5 - (-22))/10) = P(Z > 1.45) = 7.4%
    """
    if spread_std <= 0:
        spread_std = 10.0  # Default if not available

    # CRITICAL: Spread betting convention
    # market_spread is the HOME team's line (e.g., +7.5 means home getting 7.5 points)
    # Home covers if: home_score + market_spread > away_score
    # Rearranging: (home_score - away_score) > -market_spread
    # So: actual_spread > -market_spread
    #
    # Therefore, we need to NEGATE the market_spread for the calculation
    z_score = (-market_spread - model_fair_spread) / spread_std
    cover_prob = 1.0 - stats.norm.cdf(z_score)

    return float(cover_prob)


def calculate_total_over_prob(model_fair_total: float, market_total: float,
                               total_std: float) -> float:
    """
    Calculate probability of game going over the total.

    Args:
        model_fair_total: Model's predicted total points
        market_total: Market total line
        total_std: Standard deviation of total from simulations

    Returns:
        Probability of going over
    """
    if total_std <= 0:
        total_std = 10.0  # Default

    z_score = (market_total - model_fair_total) / total_std
    over_prob = 1 - stats.norm.cdf(z_score)

    return float(over_prob)


# Kelly and confidence tier now use unified module
# These wrapper functions ensure backward compatibility

def calculate_kelly_fraction_legacy(win_prob: float, odds: int, fractional: float = 0.25) -> float:
    """
    Calculate Kelly optimal bet sizing using unified module.

    Args:
        win_prob: Probability of winning
        odds: American odds
        fractional: Kelly fraction (0.25 = quarter Kelly for safety)

    Returns:
        Fraction of bankroll to bet
    """
    return calculate_kelly_fraction(win_prob, odds, fractional)


def assign_confidence_tier_legacy(edge_pct: float, model_prob: float, bet_type: str) -> str:
    """Assign confidence tier using unified system."""
    tier = assign_confidence_tier(edge_pct, model_prob, bet_type)
    return tier.value


def load_game_line_predictions() -> pd.DataFrame:
    """
    Load pre-computed game line predictions from file.

    This provides a single source of truth for game predictions,
    ensuring consistency between predictions and recommendations.

    Returns:
        DataFrame with columns: game, home_team, away_team, week,
                               home_win_prob, away_win_prob, fair_spread, fair_total, etc.

    Raises:
        FileNotFoundError: If predictions file doesn't exist
    """
    predictions_file = PROJECT_ROOT / "data" / f"game_line_predictions_week{CURRENT_WEEK}.csv"

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Game line predictions not found: {predictions_file}\n"
            f"Please run first: python scripts/predict/generate_game_line_predictions.py --week {CURRENT_WEEK}"
        )

    predictions_df = pd.read_csv(predictions_file)
    print(f"Loaded game line predictions for {len(predictions_df)} games")

    return predictions_df


def generate_game_line_recommendations():
    """Main function to generate game line recommendations."""
    print("=" * 70)
    print(f"GENERATING WEEK {CURRENT_WEEK} GAME LINE RECOMMENDATIONS")
    print("=" * 70)
    print(f"Season: {CURRENT_SEASON}, Week: {CURRENT_WEEK}")
    print()

    # CRITICAL FIX: Load pre-computed predictions instead of running fresh simulations
    # This ensures consistency with the predictions file (single source of truth)
    try:
        predictions_df = load_game_line_predictions()
        use_predictions_file = True
        print("✅ Using pre-computed predictions (consistent with predictions file)")
    except FileNotFoundError as e:
        print(f"⚠️  WARNING: {e}")
        print("⚠️  Falling back to fresh simulations (may differ from predictions file)")
        predictions_df = None
        use_predictions_file = False

    # Load odds data
    if not ODDS_FILE.exists():
        raise FileNotFoundError(f"Odds file not found: {ODDS_FILE}")

    odds_df = pd.read_csv(ODDS_FILE)
    print(f"Loaded odds for {len(odds_df)} lines")

    # Load schedule to get game info
    if not SCHEDULES_FILE.exists():
        raise FileNotFoundError(f"Schedule file not found: {SCHEDULES_FILE}")

    schedules = pd.read_parquet(SCHEDULES_FILE)
    week_games = schedules[
        (schedules['season'] == CURRENT_SEASON) &
        (schedules['week'] == CURRENT_WEEK)
    ]
    print(f"Found {len(week_games)} games for Week {CURRENT_WEEK}")

    # Load play-by-play for EPA calculation
    if not PBP_FILE.exists():
        print(f"Warning: PBP file not found: {PBP_FILE}")
        pbp_df = None
    else:
        pbp_df = pd.read_parquet(PBP_FILE)
        print(f"Loaded {len(pbp_df)} plays for EPA calculation")

    # Load calibrator
    calibrator = load_game_line_calibrator()
    if calibrator:
        print("Loaded game line calibrator")
    else:
        print("Warning: No calibrator loaded, using raw probabilities")

    # Initialize injury model
    injury_config_path = PROJECT_ROOT / "configs" / "injury_multipliers.yaml"
    injury_model = InjuryImpactModel(str(injury_config_path))
    print("Loaded injury impact model")

    # Initialize simulator
    simulator = MonteCarloSimulator(seed=42)

    # Process each game
    results = []

    for _, game in week_games.iterrows():
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']

        # Skip completed games (have scores)
        if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
            print(f"  Skipping {away_team}@{home_team} (completed)")
            continue

        print(f"\nProcessing {away_team} @ {home_team}...")

        # CRITICAL FIX: Use predictions file if available (single source of truth)
        if use_predictions_file:
            # Load pre-computed simulation results
            game_key = f"{away_team} @ {home_team}"
            game_pred = predictions_df[predictions_df['game'] == game_key]

            if len(game_pred) == 0:
                print(f"  ⚠️  WARNING: No prediction found for {game_key}, skipping")
                continue

            game_pred = game_pred.iloc[0]

            # Extract simulation output from predictions file
            raw_home_win_prob = float(game_pred['home_win_prob'])
            fair_spread = float(game_pred['projected_spread'])
            fair_total = float(game_pred['projected_total'])
            total_std = float(game_pred['total_std'])

            # Get EPA for context (still needed for some calculations)
            if pbp_df is not None:
                home_epa = calculate_team_epa(pbp_df, home_team)
                away_epa = calculate_team_epa(pbp_df, away_team)
            else:
                home_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pace': 65.0}
                away_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pace': 65.0}

            print(f"  Using predictions file: Home Win {raw_home_win_prob:.1%}, "
                  f"Fair Spread {fair_spread:+.1f}, Fair Total {fair_total:.1f}")

        else:
            # Fallback: Run fresh simulation (old behavior)
            # Get team EPA metrics
            if pbp_df is not None:
                home_epa = calculate_team_epa(pbp_df, home_team)
                away_epa = calculate_team_epa(pbp_df, away_team)
            else:
                # Use defaults if no PBP data
                home_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pace': 65.0}
                away_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pace': 65.0}

        # Calculate injury impacts for both teams
        # These adjustments are applied to EPA in the simulation
        if not use_predictions_file:
            try:
                home_injury_impact = injury_model.compute_injury_impact(CURRENT_SEASON, CURRENT_WEEK, home_team)
                away_injury_impact = injury_model.compute_injury_impact(CURRENT_SEASON, CURRENT_WEEK, away_team)

                # Get raw injury impacts (already scaled in injury model)
                home_injury_off_adj = home_injury_impact.total_impact_offensive_epa
                home_injury_def_adj = home_injury_impact.total_impact_defensive_epa
                away_injury_off_adj = away_injury_impact.total_impact_offensive_epa
                away_injury_def_adj = away_injury_impact.total_impact_defensive_epa

                # IMPORTANT: Integrate injuries with EPA proportionally
                # If a team has bad EPA already (-0.10), injuries hurt less (already struggling)
                # If a team has good EPA (+0.10), injuries hurt more (losing key performers)
                # This creates a more realistic injury impact model

                # Apply injury adjustments to actual EPA values (not as separate adjustment)
                # This integrates injuries directly into the EPA calculation
                if home_epa['offensive_epa'] > 0:
                    # Good offense - injuries hurt more
                    effective_home_off_adj = home_injury_off_adj * (1.0 + abs(home_epa['offensive_epa']) * 5)
                else:
                    # Bad offense - injuries hurt less (already struggling)
                    effective_home_off_adj = home_injury_off_adj * (1.0 - abs(home_epa['offensive_epa']) * 2)

                if away_epa['offensive_epa'] > 0:
                    effective_away_off_adj = away_injury_off_adj * (1.0 + abs(away_epa['offensive_epa']) * 5)
                else:
                    effective_away_off_adj = away_injury_off_adj * (1.0 - abs(away_epa['offensive_epa']) * 2)

                # For defense, positive EPA = bad defense (opponents score well)
                # So injuries to good defense (negative EPA) hurt more
                if home_epa['defensive_epa'] < 0:
                    # Good defense - injuries hurt more
                    effective_home_def_adj = home_injury_def_adj * (1.0 + abs(home_epa['defensive_epa']) * 5)
                else:
                    # Bad defense - injuries hurt less
                    effective_home_def_adj = home_injury_def_adj * (1.0 - abs(home_epa['defensive_epa']) * 2)

                if away_epa['defensive_epa'] < 0:
                    effective_away_def_adj = away_injury_def_adj * (1.0 + abs(away_epa['defensive_epa']) * 5)
                else:
                    effective_away_def_adj = away_injury_def_adj * (1.0 - abs(away_epa['defensive_epa']) * 2)

                # Cap adjustments to reasonable bounds
                effective_home_off_adj = max(-0.20, min(0.0, effective_home_off_adj))
                effective_home_def_adj = max(-0.15, min(0.0, effective_home_def_adj))
                effective_away_off_adj = max(-0.20, min(0.0, effective_away_off_adj))
                effective_away_def_adj = max(-0.15, min(0.0, effective_away_def_adj))

                # Log significant injuries
                if abs(effective_home_off_adj) > 0.02 or abs(effective_home_def_adj) > 0.02:
                    print(f"  {home_team} injuries: Off {effective_home_off_adj:+.3f}, Def {effective_home_def_adj:+.3f} "
                          f"(Raw: Off {home_injury_off_adj:+.3f}, Def {home_injury_def_adj:+.3f})")
                if abs(effective_away_off_adj) > 0.02 or abs(effective_away_def_adj) > 0.02:
                    print(f"  {away_team} injuries: Off {effective_away_off_adj:+.3f}, Def {effective_away_def_adj:+.3f} "
                          f"(Raw: Off {away_injury_off_adj:+.3f}, Def {away_injury_def_adj:+.3f})")

                # Use effective adjustments
                home_injury_off_adj = effective_home_off_adj
                home_injury_def_adj = effective_home_def_adj
                away_injury_off_adj = effective_away_off_adj
                away_injury_def_adj = effective_away_def_adj

            except Exception as e:
                print(f"  Warning: Could not compute injuries: {e}")
                home_injury_off_adj = 0.0
                home_injury_def_adj = 0.0
                away_injury_off_adj = 0.0
                away_injury_def_adj = 0.0

            # Create simulation input
            sim_input = SimulationInput(
                game_id=game_id,
                season=CURRENT_SEASON,
                week=CURRENT_WEEK,
                home_team=home_team,
                away_team=away_team,
                home_offensive_epa=home_epa['offensive_epa'],
                home_defensive_epa=home_epa['defensive_epa'],
                away_offensive_epa=away_epa['offensive_epa'],
                away_defensive_epa=away_epa['defensive_epa'],
                home_pace=home_epa['pace'],
                away_pace=away_epa['pace'],
                home_injury_offensive_adjustment=home_injury_off_adj,
                home_injury_defensive_adjustment=home_injury_def_adj,
                away_injury_offensive_adjustment=away_injury_off_adj,
                away_injury_defensive_adjustment=away_injury_def_adj,
                is_divisional=False,  # Could enhance with division lookup
                game_type='Regular',
                is_dome=False,  # Could enhance with stadium data
                temperature=None,
                wind_speed=None,
                precipitation=None
            )

            # Run Monte Carlo simulation
            sim_output = simulator.simulate_game(sim_input, trials=50000)

            # Extract values from simulation
            raw_home_win_prob = sim_output.home_win_prob
            fair_spread = sim_output.fair_spread
            fair_total = sim_output.fair_total
            total_std = sim_output.total_std

            print(f"  Model: Home Win {raw_home_win_prob:.1%}, "
                  f"Fair Spread {fair_spread:+.1f}, "
                  f"Fair Total {fair_total:.1f}")

        # Get market odds for this game
        game_odds = odds_df[odds_df['game_id'] == game_id]

        if len(game_odds) == 0:
            print(f"  Warning: No odds found for {game_id}")
            continue

        # Apply calibration to win probability
        # raw_home_win_prob was set above in either the predictions file or simulation branch
        calibrated_home_win_prob_raw = apply_calibration(raw_home_win_prob, calibrator)

        # Apply 30% probability shrinkage to reduce overconfidence
        calibrated_home_win_prob = apply_probability_shrinkage(calibrated_home_win_prob_raw, shrinkage_factor=0.30)

        # --- SPREAD RECOMMENDATIONS ---
        # Spreads are 'home'/'away' or 'home_spread'/'away_spread' sides with non-null point values
        spread_odds = game_odds[
            (game_odds['side'].isin(['home', 'away', 'home_spread', 'away_spread'])) &
            (game_odds['point'].notna())
        ]
        home_spread_row = spread_odds[spread_odds['side'].isin(['home', 'home_spread'])].iloc[0] if len(spread_odds[spread_odds['side'].isin(['home', 'home_spread'])]) > 0 else None
        away_spread_row = spread_odds[spread_odds['side'].isin(['away', 'away_spread'])].iloc[0] if len(spread_odds[spread_odds['side'].isin(['away', 'away_spread'])]) > 0 else None

        if home_spread_row is not None:
            market_spread = float(home_spread_row['point'])
            spread_odds = int(home_spread_row['american_odds'])

            # Calculate cover probability using total_std as proxy for spread_std
            # Spread std is typically about 10-11 points
            # total_std and fair_spread were set above in either branch
            spread_std = total_std / 1.5 if total_std else 10.0

            home_cover_prob = calculate_spread_cover_prob(
                fair_spread, market_spread, spread_std
            )

            # Apply 30% probability shrinkage to reduce overconfidence
            home_cover_prob_shrunken = apply_probability_shrinkage(home_cover_prob, shrinkage_factor=0.30)
            away_cover_prob_shrunken = 1.0 - home_cover_prob_shrunken

            # Get away spread odds
            away_spread_odds = int(away_spread_row['american_odds']) if away_spread_row is not None else -110

            # CRITICAL FIX: Calculate market implied probabilities from ACTUAL ODDS
            # Not assuming 0.5! The market prices in the vig.
            market_prob_home_cover = american_odds_to_implied_prob(spread_odds)
            market_prob_away_cover = american_odds_to_implied_prob(away_spread_odds)

            # Remove vig to get fair market probabilities
            fair_market_home, fair_market_away = remove_vig_two_way(
                market_prob_home_cover, market_prob_away_cover
            )

            # Use unified side selection
            spread_selection = select_best_side(
                prob_side1=home_cover_prob_shrunken,
                prob_side2=away_cover_prob_shrunken,
                odds_side1=spread_odds,
                odds_side2=away_spread_odds,
                name_side1=f"{home_team} {market_spread:+.1f}",
                name_side2=f"{away_team} {-market_spread:+.1f}"
            )

            spread_pick = spread_selection['pick']
            spread_prob = spread_selection['model_prob']
            spread_market_prob = spread_selection['market_prob']
            spread_edge = spread_selection['edge_pct']
            spread_odds_used = spread_selection['american_odds']

            # Calculate Kelly and confidence using unified module
            spread_kelly = calculate_kelly_fraction(spread_prob, spread_odds_used)
            spread_conf = assign_confidence_tier_legacy(spread_edge, spread_prob, 'spread')
            spread_expected_roi = calculate_expected_roi(spread_edge, spread_odds_used)

            results.append({
                'game_id': game_id,
                'game': f"{away_team} @ {home_team}",
                'bet_type': 'spread',
                'pick': spread_pick,
                'market_line': market_spread,
                'model_fair_line': fair_spread,
                'model_prob': spread_prob,
                'market_prob': spread_market_prob,  # Now uses actual market prob, not 0.5
                'edge_pct': spread_edge,
                'expected_roi': spread_expected_roi,
                'kelly_fraction': spread_kelly,
                'recommended_units': round(spread_kelly * 100, 1),
                'confidence_tier': spread_conf,
                'home_win_prob': calibrated_home_win_prob,
                'home_epa': home_epa['offensive_epa'],
                'away_epa': away_epa['offensive_epa'],
            })

        # --- TOTAL RECOMMENDATIONS ---
        over_row = game_odds[game_odds['side'] == 'over'].iloc[0] if len(game_odds[game_odds['side'] == 'over']) > 0 else None

        if over_row is not None:
            market_total = float(over_row['point'])
            over_odds = int(over_row['american_odds'])
            under_odds = int(game_odds[game_odds['side'] == 'under'].iloc[0]['american_odds']) if len(game_odds[game_odds['side'] == 'under']) > 0 else -110

            over_prob = calculate_total_over_prob(
                fair_total, market_total, total_std
            )

            # Apply 30% probability shrinkage to reduce overconfidence
            over_prob_shrunken = apply_probability_shrinkage(over_prob, shrinkage_factor=0.30)
            under_prob_shrunken = 1.0 - over_prob_shrunken

            # CRITICAL FIX: Use actual market odds, not 0.5
            total_selection = select_best_side(
                prob_side1=over_prob_shrunken,
                prob_side2=under_prob_shrunken,
                odds_side1=over_odds,
                odds_side2=under_odds,
                name_side1=f"OVER {market_total}",
                name_side2=f"UNDER {market_total}"
            )

            total_pick = total_selection['pick']
            total_prob = total_selection['model_prob']
            total_market_prob = total_selection['market_prob']
            total_edge = total_selection['edge_pct']
            total_odds_used = total_selection['american_odds']

            total_kelly = calculate_kelly_fraction(total_prob, total_odds_used)
            total_conf = assign_confidence_tier_legacy(total_edge, total_prob, 'total')
            total_expected_roi = calculate_expected_roi(total_edge, total_odds_used)

            # Get percentiles from either predictions file or sim_output
            if use_predictions_file:
                total_p25 = float(game_pred.get('total_p25', None) or 0)
                total_p50 = float(game_pred.get('total_p50', None) or 0)
                total_p75 = float(game_pred.get('total_p75', None) or 0)
                total_p5 = None  # Not in predictions file
                total_p95 = None  # Not in predictions file
            else:
                total_p5 = sim_output.total_p5 if hasattr(sim_output, 'total_p5') else None
                total_p25 = sim_output.total_p25 if hasattr(sim_output, 'total_p25') else None
                total_p50 = sim_output.total_p50 if hasattr(sim_output, 'total_p50') else None
                total_p75 = sim_output.total_p75 if hasattr(sim_output, 'total_p75') else None
                total_p95 = sim_output.total_p95 if hasattr(sim_output, 'total_p95') else None

            results.append({
                'game_id': game_id,
                'game': f"{away_team} @ {home_team}",
                'bet_type': 'total',
                'pick': total_pick,
                'market_line': market_total,
                'model_fair_line': fair_total,
                'model_prob': total_prob,
                'market_prob': total_market_prob,  # Now uses actual market prob
                'edge_pct': total_edge,
                'expected_roi': total_expected_roi,
                'kelly_fraction': total_kelly,
                'recommended_units': round(total_kelly * 100, 1),
                'confidence_tier': total_conf,
                'home_win_prob': calibrated_home_win_prob,
                'home_epa': home_epa['offensive_epa'],
                'away_epa': away_epa['offensive_epa'],
                # Distribution percentiles
                'total_p5': total_p5,
                'total_p25': total_p25,
                'total_p50': total_p50,
                'total_p75': total_p75,
                'total_p95': total_p95,
                'total_std': total_std,
                # Context data
                'home_pace': home_epa['pace'],
                'away_pace': away_epa['pace'],
                'home_pass_epa': home_epa['pass_epa'],
                'away_pass_epa': away_epa['pass_epa'],
                'home_rush_epa': home_epa['rush_epa'],
                'away_rush_epa': away_epa['rush_epa'],
                'home_def_epa': home_epa['defensive_epa'],
                'away_def_epa': away_epa['defensive_epa'],
            })

        # --- MONEYLINE RECOMMENDATIONS ---
        # Moneylines are 'home'/'away' with null point, or 'home_ml'/'away_ml'
        ml_odds = game_odds[
            (game_odds['side'].isin(['home', 'away', 'home_ml', 'away_ml'])) &
            ((game_odds['point'].isna()) | (game_odds['point'] == 0) | (game_odds['side'].str.contains('_ml')))
        ]
        home_ml_row = ml_odds[ml_odds['side'].isin(['home', 'home_ml'])].iloc[0] if len(ml_odds[ml_odds['side'].isin(['home', 'home_ml'])]) > 0 else None
        away_ml_row = ml_odds[ml_odds['side'].isin(['away', 'away_ml'])].iloc[0] if len(ml_odds[ml_odds['side'].isin(['away', 'away_ml'])]) > 0 else None

        # Use moneyline odds if available
        if home_ml_row is not None and away_ml_row is not None:
            home_ml_odds = int(home_ml_row['american_odds'])
            away_ml_odds = int(away_ml_row['american_odds'])

            # CRITICAL FIX: Use actual market odds for moneyline
            ml_selection = select_best_side(
                prob_side1=calibrated_home_win_prob,
                prob_side2=1 - calibrated_home_win_prob,
                odds_side1=home_ml_odds,
                odds_side2=away_ml_odds,
                name_side1=f"{home_team} ML",
                name_side2=f"{away_team} ML"
            )

            ml_pick = ml_selection['pick']
            ml_prob = ml_selection['model_prob']
            ml_market_prob = ml_selection['market_prob']
            ml_edge = ml_selection['edge_pct']
            ml_odds_used = ml_selection['american_odds']

            # Only recommend if edge is meaningful
            if ml_edge > 2.0:
                ml_kelly = calculate_kelly_fraction(ml_prob, ml_odds_used)
                ml_conf = assign_confidence_tier_legacy(ml_edge, ml_prob, 'moneyline')
                ml_expected_roi = calculate_expected_roi(ml_edge, ml_odds_used)

                results.append({
                    'game_id': game_id,
                    'game': f"{away_team} @ {home_team}",
                    'bet_type': 'moneyline',
                    'pick': ml_pick,
                    'market_line': None,
                    'model_fair_line': calibrated_home_win_prob,
                    'model_prob': ml_prob,
                    'market_prob': ml_market_prob,  # Now uses actual market prob
                    'edge_pct': ml_edge,
                    'expected_roi': ml_expected_roi,
                    'kelly_fraction': ml_kelly,
                    'recommended_units': round(ml_kelly * 100, 1),
                    'confidence_tier': ml_conf,
                    'home_win_prob': calibrated_home_win_prob,
                    'home_epa': home_epa['offensive_epa'],
                    'away_epa': away_epa['offensive_epa'],
                })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\nNo recommendations generated")
        return results_df

    # Sort by edge
    results_df = results_df.sort_values('edge_pct', ascending=False)

    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("TOP GAME LINE RECOMMENDATIONS")
    print("=" * 70)

    # Filter to minimum edge threshold
    min_edge = 2.0
    strong_picks = results_df[results_df['edge_pct'] >= min_edge]

    print(f"\nPicks with ≥{min_edge}% edge: {len(strong_picks)}")

    for i, (_, row) in enumerate(strong_picks.head(15).iterrows(), 1):
        emoji = "🔥" if row['edge_pct'] > 5 else "⭐" if row['edge_pct'] > 3 else "📊"
        print(f"\n{i}. {emoji} {row['pick']} ({row['bet_type'].upper()})")
        print(f"   Game: {row['game']}")
        print(f"   Model Fair: {row['model_fair_line']:.1f} | Market: {row['market_line']}")
        print(f"   Win Prob: {row['model_prob']:.1%} | Edge: {row['edge_pct']:+.1f}%")
        print(f"   Kelly: {row['recommended_units']:.1f} units | Confidence: {row['confidence_tier']}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total recommendations: {len(results_df)}")
    print(f"  Spreads: {len(results_df[results_df['bet_type'] == 'spread'])}")
    print(f"  Totals: {len(results_df[results_df['bet_type'] == 'total'])}")
    print(f"  Moneylines: {len(results_df[results_df['bet_type'] == 'moneyline'])}")
    print(f"\nHIGH confidence: {len(results_df[results_df['confidence_tier'] == 'HIGH'])}")
    print(f"MEDIUM confidence: {len(results_df[results_df['confidence_tier'] == 'MEDIUM'])}")
    print(f"LOW confidence: {len(results_df[results_df['confidence_tier'] == 'LOW'])}")

    if len(strong_picks) > 0:
        print(f"\nAverage edge (≥{min_edge}%): {strong_picks['edge_pct'].mean():.2f}%")
        print(f"Expected ROI: {strong_picks['expected_roi'].mean():.2f}%")

    return results_df


if __name__ == "__main__":
    df = generate_game_line_recommendations()
