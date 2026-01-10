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
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput, SimulationOutput
from nfl_quant.utils.season_utils import get_current_season, get_current_week
from nfl_quant.config_enhanced import config
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
from nfl_quant.utils.ats_tracker import get_team_ats_context
from nfl_quant.utils.defensive_rankings import get_team_defensive_ranks
from nfl_quant.features.team_power_ratings import TeamPowerRatings
from nfl_quant.features.team_strength import EnhancedEloCalculator
from nfl_quant.utils.team_names import normalize_team_name

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load game line betting config from centralized config
_gl_config = config.simulation.game_line_betting
_shrinkage_config = _gl_config.get('shrinkage', {})
_blending_config = _gl_config.get('blending_weights', {})
_spread_config = _gl_config.get('spread_calculation', {})
_hfa_config = _gl_config.get('home_field_advantage', {})
_record_config = _gl_config.get('record_adjustment', {})
_edge_thresholds = _gl_config.get('minimum_edge_thresholds', {})
_confidence_tiers = _gl_config.get('confidence_tiers_post_shrinkage', {})

# Shrinkage factors (from config, with defaults)
WIN_PROB_SHRINKAGE = _shrinkage_config.get('win_probability', 0.30)
SPREAD_COVER_SHRINKAGE = _shrinkage_config.get('spread_cover_probability', 0.40)
TOTAL_OVER_SHRINKAGE = _shrinkage_config.get('total_over_probability', 0.40)

# Blending weights
SIMULATION_WEIGHT = _blending_config.get('simulation_weight', 0.40)
ELO_WEIGHT = _blending_config.get('elo_weight', 0.40)
RECORD_WEIGHT = _blending_config.get('record_weight', 0.20)

# Spread calculation
SPREAD_STD_DIVISOR = _spread_config.get('spread_std_divisor', 1.414)

# Record adjustment
RECORD_MAX_ADJUSTMENT = _record_config.get('max_adjustment', 0.05)

# Minimum edge thresholds
MIN_EDGE_SPREAD = _edge_thresholds.get('spread', 0.0)
MIN_EDGE_TOTAL = _edge_thresholds.get('total', 0.0)
MIN_EDGE_MONEYLINE = _edge_thresholds.get('moneyline', 2.0)

# Confidence tier thresholds (post-shrinkage adjusted)
ELITE_THRESHOLD = _confidence_tiers.get('elite_threshold', 0.62)
HIGH_THRESHOLD = _confidence_tiers.get('high_threshold', 0.57)
STANDARD_THRESHOLD = _confidence_tiers.get('standard_threshold', 0.53)

print(f"Loaded game line config: shrinkage={WIN_PROB_SHRINKAGE:.0%}/{SPREAD_COVER_SHRINKAGE:.0%}/{TOTAL_OVER_SHRINKAGE:.0%}")


def normalize_game_id(game_id: str) -> str:
    """
    Normalize team codes in a game_id string.

    Converts game_ids like '2025_17_LAR_ATL' to use consistent team codes.
    The schedule uses 'LA' but odds API uses 'LAR' for the Rams.

    Args:
        game_id: Game ID in format 'YYYY_WW_AWAY_HOME'

    Returns:
        Normalized game ID with consistent team codes
    """
    if not isinstance(game_id, str):
        return game_id

    parts = game_id.split('_')
    if len(parts) != 4:
        return game_id

    season, week, away, home = parts
    # Normalize team codes (LAR -> LA for schedule compatibility)
    away_normalized = 'LA' if away == 'LAR' else away
    home_normalized = 'LA' if home == 'LAR' else home

    return f"{season}_{week}_{away_normalized}_{home_normalized}"


# Initialize power ratings system (combines Elo + EPA + SOS)
_power_ratings: TeamPowerRatings = None

def get_power_ratings() -> TeamPowerRatings:
    """Get or initialize the team power ratings system."""
    global _power_ratings
    if _power_ratings is None:
        _power_ratings = TeamPowerRatings()
    return _power_ratings


def get_comprehensive_team_features(
    home_team: str,
    away_team: str,
    season: int,
    week: int
) -> dict:
    """
    Get all available team features for a matchup.

    Combines:
    - Power ratings (60% Elo + 30% EPA + 10% SOS)
    - Win/loss records and point differentials
    - Momentum (last 6 games ATS)
    - Strength of schedule
    - Rest days
    - Red zone efficiency (from EPA data)

    Returns:
        Dict with home_ and away_ prefixed features
    """
    pr = get_power_ratings()
    elo_calc = pr.elo_calculator

    # Get power ratings for both teams
    try:
        home_power = pr.calculate_team_power(home_team, season, week)
        away_power = pr.calculate_team_power(away_team, season, week)
    except Exception as e:
        print(f"  Warning: Could not calculate power ratings: {e}")
        home_power = {'power_rating': 50, 'elo': 1505, 'net_epa': 0, 'sos': 1505}
        away_power = {'power_rating': 50, 'elo': 1505, 'net_epa': 0, 'sos': 1505}

    # Get team records
    try:
        home_record = elo_calc.get_team_record(home_team, season, week)
        away_record = elo_calc.get_team_record(away_team, season, week)
    except Exception as e:
        print(f"  Warning: Could not get team records: {e}")
        home_record = {'win_pct': 0.5, 'point_diff_per_game': 0, 'wins': 0, 'losses': 0}
        away_record = {'win_pct': 0.5, 'point_diff_per_game': 0, 'wins': 0, 'losses': 0}

    # Get Elo-based features (includes rest days and HFA)
    try:
        elo_features = elo_calc.get_team_features(home_team, away_team, season, week, is_home=True)
    except Exception as e:
        print(f"  Warning: Could not get Elo features: {e}")
        elo_features = {
            'team_elo': 1505, 'opp_elo': 1505,
            'elo_diff_adjusted': 0, 'win_probability': 0.5,
            'expected_spread': 0, 'team_rest': 7, 'opp_rest': 7
        }

    # Calculate momentum from recent performance
    # Use Elo change over last few weeks as momentum indicator
    home_momentum = home_power.get('net_epa', 0) * 10  # Scale EPA to momentum score
    away_momentum = away_power.get('net_epa', 0) * 10

    # Combine all features
    features = {
        # Power ratings
        'home_power_rating': home_power.get('power_rating', 50),
        'away_power_rating': away_power.get('power_rating', 50),
        'power_diff': home_power.get('power_rating', 50) - away_power.get('power_rating', 50),

        # Elo ratings
        'home_elo': home_power.get('elo', 1505),
        'away_elo': away_power.get('elo', 1505),
        'elo_diff': home_power.get('elo', 1505) - away_power.get('elo', 1505),
        'elo_win_prob': elo_features.get('win_probability', 0.5),
        'elo_expected_spread': elo_features.get('expected_spread', 0),  # Negative = away favored

        # Records
        'home_win_pct': home_record.get('win_pct', 0.5),
        'away_win_pct': away_record.get('win_pct', 0.5),
        'home_wins': home_record.get('wins', 0),
        'home_losses': home_record.get('losses', 0),
        'away_wins': away_record.get('wins', 0),
        'away_losses': away_record.get('losses', 0),
        'home_point_diff_pg': home_record.get('point_diff_per_game', 0),
        'away_point_diff_pg': away_record.get('point_diff_per_game', 0),

        # EPA breakdown
        'home_off_epa': home_power.get('off_epa', 0),
        'home_def_epa': home_power.get('def_epa', 0),
        'away_off_epa': away_power.get('off_epa', 0),
        'away_def_epa': away_power.get('def_epa', 0),
        'home_net_epa': home_power.get('net_epa', 0),
        'away_net_epa': away_power.get('net_epa', 0),

        # Strength of schedule
        'home_sos': home_power.get('sos', 1505),
        'away_sos': away_power.get('sos', 1505),

        # Rest days
        'home_rest': elo_features.get('team_rest', 7),
        'away_rest': elo_features.get('opp_rest', 7),

        # Momentum (scaled net EPA)
        'home_momentum': home_momentum,
        'away_momentum': away_momentum,

        # Sample size for confidence
        'home_sample_games': home_power.get('sample_games', 0),
        'away_sample_games': away_power.get('sample_games', 0),
    }

    return features


def blend_predictions(
    simulation_home_win_prob: float,
    elo_home_win_prob: float,
    home_record_edge: float,
    simulation_weight: float = None,
    elo_weight: float = None,
    record_weight: float = None
) -> float:
    """
    Blend simulation probability with Elo and record-based adjustments.

    This combines:
    - Monte Carlo simulation (EPA-based) - default 40%
    - Elo model prediction - default 40%
    - Record-based edge - default 20%

    Weights are loaded from config (simulation_config.json -> game_line_betting).
    This reduces overconfidence from any single model.
    """
    # Use config defaults if not specified
    if simulation_weight is None:
        simulation_weight = SIMULATION_WEIGHT
    if elo_weight is None:
        elo_weight = ELO_WEIGHT
    if record_weight is None:
        record_weight = RECORD_WEIGHT

    # Convert record edge to probability adjustment
    # If home team is much better by record, add small probability boost
    record_adj = home_record_edge * RECORD_MAX_ADJUSTMENT  # From config (default 5%)

    blended = (
        simulation_weight * simulation_home_win_prob +
        elo_weight * elo_home_win_prob +
        record_weight * (0.5 + record_adj)  # Center at 50% + adjustment
    )

    # Clamp to valid probability range
    return max(0.05, min(0.95, blended))


def apply_probability_shrinkage(prob: float, shrinkage_factor: float = 0.50) -> float:
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


def enrich_with_team_context(df: pd.DataFrame, week: int, season: int) -> pd.DataFrame:
    """
    Enrich game line recommendations with ATS and defensive ranking data.

    Adds columns for dashboard display:
    - home_ats_record, home_last6_ats, home_ats_pct
    - away_ats_record, away_last6_ats, away_ats_pct
    - home_ou_record, away_ou_record
    - home_pass_def_rank, home_rush_def_rank, home_total_def_rank
    - away_pass_def_rank, away_rush_def_rank, away_total_def_rank
    """
    if len(df) == 0:
        return df

    df = df.copy()

    # Extract home and away teams from game string (format: "AWAY @ HOME")
    def extract_teams(game_str):
        if pd.isna(game_str) or '@' not in str(game_str):
            return None, None
        parts = str(game_str).split(' @ ')
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None, None

    # Get unique games
    games = df['game'].unique()
    team_context_cache = {}

    for game in games:
        away_team, home_team = extract_teams(game)
        if not home_team or not away_team:
            continue

        # Get ATS context for both teams
        if home_team not in team_context_cache:
            home_ats = get_team_ats_context(home_team, season, week)
            home_def = get_team_defensive_ranks(home_team, season, week)
            team_context_cache[home_team] = {**home_ats, **home_def}

        if away_team not in team_context_cache:
            away_ats = get_team_ats_context(away_team, season, week)
            away_def = get_team_defensive_ranks(away_team, season, week)
            team_context_cache[away_team] = {**away_ats, **away_def}

    # Add columns to DataFrame
    for col_prefix, team_type in [('home', 'home'), ('away', 'away')]:
        df[f'{col_prefix}_ats_record'] = ''
        df[f'{col_prefix}_last6_ats'] = ''
        df[f'{col_prefix}_ats_pct'] = 0.5
        df[f'{col_prefix}_ou_record'] = ''
        df[f'{col_prefix}_ou_over_pct'] = 0.5
        df[f'{col_prefix}_pass_def_rank'] = 16
        df[f'{col_prefix}_rush_def_rank'] = 16
        df[f'{col_prefix}_total_def_rank'] = 16

    for idx, row in df.iterrows():
        away_team, home_team = extract_teams(row['game'])
        if not home_team or not away_team:
            continue

        # Home team context
        if home_team in team_context_cache:
            ctx = team_context_cache[home_team]
            df.at[idx, 'home_ats_record'] = ctx.get('season_ats', '')
            df.at[idx, 'home_last6_ats'] = ctx.get('last_6_ats', '')
            df.at[idx, 'home_ats_pct'] = ctx.get('season_ats_pct', 0.5)
            df.at[idx, 'home_ou_record'] = ctx.get('ou_record', '')
            df.at[idx, 'home_ou_over_pct'] = ctx.get('ou_over_pct', 0.5)
            df.at[idx, 'home_pass_def_rank'] = ctx.get('pass_def_rank', 16)
            df.at[idx, 'home_rush_def_rank'] = ctx.get('rush_def_rank', 16)
            df.at[idx, 'home_total_def_rank'] = ctx.get('total_def_rank', 16)

        # Away team context
        if away_team in team_context_cache:
            ctx = team_context_cache[away_team]
            df.at[idx, 'away_ats_record'] = ctx.get('season_ats', '')
            df.at[idx, 'away_last6_ats'] = ctx.get('last_6_ats', '')
            df.at[idx, 'away_ats_pct'] = ctx.get('season_ats_pct', 0.5)
            df.at[idx, 'away_ou_record'] = ctx.get('ou_record', '')
            df.at[idx, 'away_ou_over_pct'] = ctx.get('ou_over_pct', 0.5)
            df.at[idx, 'away_pass_def_rank'] = ctx.get('pass_def_rank', 16)
            df.at[idx, 'away_rush_def_rank'] = ctx.get('rush_def_rank', 16)
            df.at[idx, 'away_total_def_rank'] = ctx.get('total_def_rank', 16)

    print(f"  Enriched {len(df)} recommendations with team context")
    return df


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


# Import canonical odds utilities - no more duplicate functions
from nfl_quant.utils.odds import american_to_prob, american_to_decimal

# Alias for backward compatibility
american_odds_to_prob = american_to_prob


def calculate_team_epa(pbp_df: pd.DataFrame, team: str, weeks: int = 10, decay_factor: float = 0.85) -> dict:
    """
    Calculate team EPA metrics from play-by-play data with EXPONENTIAL DECAY weighting.

    Returns:
        dict with offensive_epa, defensive_epa, pass_epa, rush_epa, pace

    IMPORTANT:
    1. Uses EXPONENTIAL DECAY to weight recent weeks more heavily
       - decay_factor=0.85 means each week back is worth 85% of the previous
       - Week 0 (most recent): weight=1.0
       - Week 1 ago: weight=0.85
       - Week 2 ago: weight=0.72
       - Week 5 ago: weight=0.44
       This makes the model responsive to recent form changes.

    2. EPA values are REGRESSED toward league mean (0.0) to prevent
       extreme predictions from small sample sizes.

    defensive_epa represents what opponents score AGAINST this team:
    - Positive = opponents score well (bad defense)
    - Negative = opponents score poorly (good defense)
    """
    # Filter to relevant weeks (use available data before current week)
    max_week = pbp_df['week'].max()
    recent_weeks = list(range(max(1, max_week - weeks + 1), max_week + 1))

    # Offensive EPA (when team has possession)
    off_plays = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(recent_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ].copy()

    # Defensive EPA (when team is defending)
    def_plays = pbp_df[
        (pbp_df['defteam'] == team) &
        (pbp_df['week'].isin(recent_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ].copy()

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

    # EXPONENTIAL DECAY WEIGHTING
    # More recent weeks get higher weight
    # weight = decay_factor^(weeks_ago) where weeks_ago = max_week - play_week
    off_plays['weeks_ago'] = max_week - off_plays['week']
    off_plays['weight'] = decay_factor ** off_plays['weeks_ago']

    def_plays['weeks_ago'] = max_week - def_plays['week']
    def_plays['weight'] = decay_factor ** def_plays['weeks_ago']

    # Calculate WEIGHTED EPA per play
    raw_offensive_epa = np.average(off_plays['epa'].fillna(0), weights=off_plays['weight'])
    raw_defensive_epa = np.average(def_plays['epa'].fillna(0), weights=def_plays['weight'])

    # CRITICAL: Regress EPA values toward league mean (0.0)
    # This prevents extreme predictions from small samples
    # The market is generally efficient, so extreme deviations are often noise
    offensive_epa = regress_epa_to_mean(raw_offensive_epa, sample_games)
    defensive_epa = regress_epa_to_mean(raw_defensive_epa, sample_games)

    # Split by play type (also weighted and regressed)
    pass_plays = off_plays[off_plays['play_type'] == 'pass']
    rush_plays = off_plays[off_plays['play_type'] == 'run']

    if len(pass_plays) > 0:
        raw_pass_epa = np.average(pass_plays['epa'].fillna(0), weights=pass_plays['weight'])
    else:
        raw_pass_epa = 0.0

    if len(rush_plays) > 0:
        raw_rush_epa = np.average(rush_plays['epa'].fillna(0), weights=rush_plays['weight'])
    else:
        raw_rush_epa = 0.0

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


def assign_game_line_confidence_tier(model_prob: float) -> str:
    """
    Assign confidence tier for game line bets using post-shrinkage adjusted thresholds.

    Game line probabilities are heavily shrunk toward 50%, so we use lower thresholds
    than player props. Thresholds are loaded from config.

    Args:
        model_prob: Model probability AFTER shrinkage (0.0 to 1.0)

    Returns:
        Tier string: 'ELITE', 'HIGH', 'STANDARD', or 'LOW'
    """
    # Use config thresholds (adjusted for post-shrinkage probabilities)
    if model_prob >= ELITE_THRESHOLD:
        return 'ELITE'
    if model_prob >= HIGH_THRESHOLD:
        return 'HIGH'
    if model_prob >= STANDARD_THRESHOLD:
        return 'STANDARD'
    return 'LOW'


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


def process_single_game(game_row, context):
    """
    Process a single game and return recommendations.

    Args:
        game_row: Row from week_games DataFrame
        context: Dict with shared resources (predictions_df, odds_df, pbp_df, etc.)

    Returns:
        List of recommendation dicts for this game
    """
    # Unpack context
    predictions_df = context['predictions_df']
    use_predictions_file = context['use_predictions_file']
    odds_df = context['odds_df']
    pbp_df = context['pbp_df']
    calibrator = context['calibrator']
    injury_model = context['injury_model']
    simulator_seed = context['simulator_seed']
    current_season = context['current_season']
    current_week = context['current_week']

    # THREAD SAFETY: Create a new simulator for each game with unique seed
    # Using game_id hash ensures reproducibility across runs
    game_seed = simulator_seed + hash(game_row['game_id']) % 10000
    simulator = MonteCarloSimulator(seed=game_seed)

    results = []

    game_id = game_row['game_id']
    home_team = game_row['home_team']
    away_team = game_row['away_team']

    # Skip completed games (have scores)
    if pd.notna(game_row.get('home_score')) and pd.notna(game_row.get('away_score')):
        return results

    game_key = f"{away_team} @ {home_team}"

    # CRITICAL FIX: Use predictions file if available (single source of truth)
    if use_predictions_file:
        # Load pre-computed simulation results
        game_pred = predictions_df[predictions_df['game'] == game_key]

        if len(game_pred) == 0:
            return results

        game_pred = game_pred.iloc[0]

        # Extract simulation output from predictions file
        raw_home_win_prob = float(game_pred['home_win_prob'])
        fair_spread = float(game_pred['projected_spread'])
        fair_total = float(game_pred['projected_total'])
        total_std = float(game_pred['total_std'])

        # OPTIMIZATION: Extract EPA from predictions file instead of recalculating
        # Predictions file already has EPA data computed during generation
        home_epa = {
            'offensive_epa': float(game_pred.get('home_off_epa', 0.0) or 0.0),
            'defensive_epa': float(game_pred.get('home_def_epa', 0.0) or 0.0),
            'pace': float(game_pred.get('home_pace', 65.0) or 65.0),
            'pass_epa': float(game_pred.get('home_pass_epa', 0.0) or 0.0),
            'rush_epa': float(game_pred.get('home_rush_epa', 0.0) or 0.0),
        }
        away_epa = {
            'offensive_epa': float(game_pred.get('away_off_epa', 0.0) or 0.0),
            'defensive_epa': float(game_pred.get('away_def_epa', 0.0) or 0.0),
            'pace': float(game_pred.get('away_pace', 65.0) or 65.0),
            'pass_epa': float(game_pred.get('away_pass_epa', 0.0) or 0.0),
            'rush_epa': float(game_pred.get('away_rush_epa', 0.0) or 0.0),
        }

    else:
        # Fallback: Run fresh simulation (old behavior)
        # Get team EPA metrics with 6-week window for recent performance focus
        if pbp_df is not None:
            home_epa = calculate_team_epa(pbp_df, home_team, weeks=6)
            away_epa = calculate_team_epa(pbp_df, away_team, weeks=6)
        else:
            # Use defaults if no PBP data
            home_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pace': 65.0, 'pass_epa': 0.0, 'rush_epa': 0.0}
            away_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pace': 65.0, 'pass_epa': 0.0, 'rush_epa': 0.0}

    # Get comprehensive team features (Elo, records, momentum, SOS)
    try:
        team_features = get_comprehensive_team_features(home_team, away_team, current_season, current_week)
    except Exception as e:
        print(f"  Warning: Could not get comprehensive features: {e}")
        team_features = {
            'elo_win_prob': 0.5, 'home_win_pct': 0.5, 'away_win_pct': 0.5,
            'home_elo': 1505, 'away_elo': 1505, 'power_diff': 0,
            'home_rest': 7, 'away_rest': 7
        }

    # Calculate injury impacts for both teams (when not using predictions file)
    if not use_predictions_file:
        try:
            home_injury_impact = injury_model.compute_injury_impact(current_season, current_week, home_team)
            away_injury_impact = injury_model.compute_injury_impact(current_season, current_week, away_team)

            # Get raw injury impacts
            home_injury_off_adj = home_injury_impact.total_impact_offensive_epa
            home_injury_def_adj = home_injury_impact.total_impact_defensive_epa
            away_injury_off_adj = away_injury_impact.total_impact_offensive_epa
            away_injury_def_adj = away_injury_impact.total_impact_defensive_epa

            # Apply proportional injury adjustments
            if home_epa['offensive_epa'] > 0:
                effective_home_off_adj = home_injury_off_adj * (1.0 + abs(home_epa['offensive_epa']) * 5)
            else:
                effective_home_off_adj = home_injury_off_adj * (1.0 - abs(home_epa['offensive_epa']) * 2)

            if away_epa['offensive_epa'] > 0:
                effective_away_off_adj = away_injury_off_adj * (1.0 + abs(away_epa['offensive_epa']) * 5)
            else:
                effective_away_off_adj = away_injury_off_adj * (1.0 - abs(away_epa['offensive_epa']) * 2)

            if home_epa['defensive_epa'] < 0:
                effective_home_def_adj = home_injury_def_adj * (1.0 + abs(home_epa['defensive_epa']) * 5)
            else:
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

            home_injury_off_adj = effective_home_off_adj
            home_injury_def_adj = effective_home_def_adj
            away_injury_off_adj = effective_away_off_adj
            away_injury_def_adj = effective_away_def_adj

        except Exception:
            home_injury_off_adj = 0.0
            home_injury_def_adj = 0.0
            away_injury_off_adj = 0.0
            away_injury_def_adj = 0.0

        # Extract game context from schedule data (not hardcoded!)
        is_divisional = bool(game_row.get('div_game', 0))
        game_type = game_row.get('game_type', 'REG')
        roof = str(game_row.get('roof', '')).lower()
        is_dome = roof in ['dome', 'closed', 'retractable roof closed']

        # Weather data (may be NaN for future games)
        temp_val = game_row.get('temp')
        wind_val = game_row.get('wind')
        temperature = float(temp_val) if pd.notna(temp_val) else (72.0 if is_dome else None)
        wind_speed = float(wind_val) if pd.notna(wind_val) else None

        # Create simulation input
        sim_input = SimulationInput(
            game_id=game_id,
            season=current_season,
            week=current_week,
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
            is_divisional=is_divisional,
            game_type=game_type,
            is_dome=is_dome,
            temperature=temperature,
            wind_speed=wind_speed,
            precipitation=None  # Not typically in schedule data
        )

        # Run Monte Carlo simulation
        sim_output = simulator.simulate_game(sim_input, trials=50000)

        # Extract values from simulation
        raw_home_win_prob = sim_output.home_win_prob
        fair_spread = sim_output.fair_spread
        fair_total = sim_output.fair_total
        total_std = sim_output.total_std

    # Get market odds for this game
    game_odds = odds_df[odds_df['game_id'] == game_id]

    if len(game_odds) == 0:
        return results

    # Apply calibration to win probability
    calibrated_home_win_prob_raw = apply_calibration(raw_home_win_prob, calibrator)

    # BLEND simulation with Elo and record-based predictions
    # This reduces overconfidence from any single model
    elo_win_prob = team_features.get('elo_win_prob', 0.5)
    home_record_edge = team_features.get('home_win_pct', 0.5) - team_features.get('away_win_pct', 0.5)

    # Blend: 40% simulation + 40% Elo + 20% record-based
    blended_home_win_prob = blend_predictions(
        simulation_home_win_prob=calibrated_home_win_prob_raw,
        elo_home_win_prob=elo_win_prob,
        home_record_edge=home_record_edge,
        simulation_weight=0.40,
        elo_weight=0.40,
        record_weight=0.20
    )

    # Apply additional shrinkage to the blended result for conservatism
    calibrated_home_win_prob = apply_probability_shrinkage(blended_home_win_prob, shrinkage_factor=WIN_PROB_SHRINKAGE)

    # BLEND the fair spread with Elo expected spread
    # This prevents the simulation from dominating spread predictions
    elo_expected_spread = team_features.get('elo_expected_spread', 0)
    blended_fair_spread = (
        0.40 * fair_spread +           # Simulation spread
        0.40 * elo_expected_spread +   # Elo spread
        0.20 * 0                        # Neutral (market is efficient)
    )

    # --- SPREAD RECOMMENDATIONS ---
    spread_odds_df = game_odds[
        (game_odds['side'].isin(['home', 'away', 'home_spread', 'away_spread'])) &
        (game_odds['point'].notna())
    ]
    home_spread_row = spread_odds_df[spread_odds_df['side'].isin(['home', 'home_spread'])].iloc[0] if len(spread_odds_df[spread_odds_df['side'].isin(['home', 'home_spread'])]) > 0 else None
    away_spread_row = spread_odds_df[spread_odds_df['side'].isin(['away', 'away_spread'])].iloc[0] if len(spread_odds_df[spread_odds_df['side'].isin(['away', 'away_spread'])]) > 0 else None

    if home_spread_row is not None:
        market_spread = float(home_spread_row['point'])
        spread_odds_val = int(home_spread_row['american_odds'])

        spread_std = total_std / SPREAD_STD_DIVISOR if total_std else 10.0

        # Use BLENDED fair spread instead of raw simulation spread
        home_cover_prob = calculate_spread_cover_prob(
            blended_fair_spread, market_spread, spread_std
        )

        home_cover_prob_shrunken = apply_probability_shrinkage(home_cover_prob, shrinkage_factor=SPREAD_COVER_SHRINKAGE)
        away_cover_prob_shrunken = 1.0 - home_cover_prob_shrunken

        away_spread_odds = int(away_spread_row['american_odds']) if away_spread_row is not None else -110

        market_prob_home_cover = american_odds_to_implied_prob(spread_odds_val)
        market_prob_away_cover = american_odds_to_implied_prob(away_spread_odds)

        fair_market_home, fair_market_away = remove_vig_two_way(
            market_prob_home_cover, market_prob_away_cover
        )

        spread_selection = select_best_side(
            prob_side1=home_cover_prob_shrunken,
            prob_side2=away_cover_prob_shrunken,
            odds_side1=spread_odds_val,
            odds_side2=away_spread_odds,
            name_side1=f"{home_team} {market_spread:+.1f}",
            name_side2=f"{away_team} {-market_spread:+.1f}"
        )

        spread_pick = spread_selection['pick']
        spread_prob = spread_selection['model_prob']
        spread_market_prob = spread_selection['market_prob']
        spread_edge = spread_selection['edge_pct']
        spread_odds_used = spread_selection['american_odds']

        spread_kelly = calculate_kelly_fraction(spread_prob, spread_odds_used)
        spread_conf = assign_game_line_confidence_tier(spread_prob)
        spread_expected_roi = calculate_expected_roi(spread_edge, spread_odds_used)

        # Get weather/depth chart from predictions if available
        weather_data = {}
        depth_chart_data = {}
        if use_predictions_file:
            weather_data = {
                'wind_bucket': game_pred.get('wind_bucket', 'calm'),
                'temp_bucket': game_pred.get('temp_bucket', 'comfortable'),
                'weather_pass_mult': game_pred.get('weather_pass_mult', 1.0),
                'weather_rush_boost': game_pred.get('weather_rush_boost', 0.0),
                'weather_off_mult': game_pred.get('weather_off_mult', 1.0),
                'temperature': game_pred.get('temperature'),
                'wind_speed': game_pred.get('wind_speed'),
                'precip_chance': game_pred.get('precip_chance', 0),
                'precip_type': game_pred.get('precip_type'),
                'is_dome': game_pred.get('is_dome', False),
                'conditions': game_pred.get('conditions', ''),
            }
            depth_chart_data = {
                'dc_home_qb': game_pred.get('dc_home_qb'),
                'dc_away_qb': game_pred.get('dc_away_qb'),
                'home_qb_backup_active': game_pred.get('home_qb_backup_active', False),
                'away_qb_backup_active': game_pred.get('away_qb_backup_active', False),
                'home_rb_backup_active': game_pred.get('home_rb_backup_active', False),
                'away_rb_backup_active': game_pred.get('away_rb_backup_active', False),
                'home_qb': game_pred.get('home_qb'),
                'away_qb': game_pred.get('away_qb'),
                'home_qb_epa': game_pred.get('home_qb_epa'),
                'away_qb_epa': game_pred.get('away_qb_epa'),
            }

        results.append({
            'game_id': game_id,
            'game': game_key,
            'bet_type': 'spread',
            'pick': spread_pick,
            'market_line': market_spread,
            'model_fair_line': blended_fair_spread,  # Use blended spread
            'model_prob': spread_prob,
            'market_prob': spread_market_prob,
            'edge_pct': spread_edge,
            'expected_roi': spread_expected_roi,
            'kelly_fraction': spread_kelly,
            'recommended_units': round(spread_kelly * 100, 1),
            'confidence_tier': spread_conf,
            'home_win_prob': calibrated_home_win_prob,
            'home_epa': home_epa['offensive_epa'],
            'away_epa': away_epa['offensive_epa'],
            # New comprehensive features
            'home_elo': team_features.get('home_elo', 1505),
            'away_elo': team_features.get('away_elo', 1505),
            'elo_diff': team_features.get('elo_diff', 0),
            'home_power': team_features.get('home_power_rating', 50),
            'away_power': team_features.get('away_power_rating', 50),
            'power_diff': team_features.get('power_diff', 0),
            'home_record': f"{team_features.get('home_wins', 0)}-{team_features.get('home_losses', 0)}",
            'away_record': f"{team_features.get('away_wins', 0)}-{team_features.get('away_losses', 0)}",
            'home_win_pct': team_features.get('home_win_pct', 0.5),
            'away_win_pct': team_features.get('away_win_pct', 0.5),
            'home_sos': team_features.get('home_sos', 1505),
            'away_sos': team_features.get('away_sos', 1505),
            'home_rest_days': team_features.get('home_rest', 7),
            'away_rest_days': team_features.get('away_rest', 7),
            **weather_data,
            **depth_chart_data,
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

        over_prob_shrunken = apply_probability_shrinkage(over_prob, shrinkage_factor=TOTAL_OVER_SHRINKAGE)
        under_prob_shrunken = 1.0 - over_prob_shrunken

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
        total_conf = assign_game_line_confidence_tier(total_prob)
        total_expected_roi = calculate_expected_roi(total_edge, total_odds_used)

        # Get percentiles from predictions file if available
        if use_predictions_file:
            total_p25 = float(game_pred.get('total_p25', None) or 0)
            total_p50 = float(game_pred.get('total_p50', None) or 0)
            total_p75 = float(game_pred.get('total_p75', None) or 0)
            total_p5 = None
            total_p95 = None
        else:
            total_p5 = sim_output.total_p5 if hasattr(sim_output, 'total_p5') else None
            total_p25 = sim_output.total_p25 if hasattr(sim_output, 'total_p25') else None
            total_p50 = sim_output.total_p50 if hasattr(sim_output, 'total_p50') else None
            total_p75 = sim_output.total_p75 if hasattr(sim_output, 'total_p75') else None
            total_p95 = sim_output.total_p95 if hasattr(sim_output, 'total_p95') else None

        results.append({
            'game_id': game_id,
            'game': game_key,
            'bet_type': 'total',
            'pick': total_pick,
            'market_line': market_total,
            'model_fair_line': fair_total,
            'model_prob': total_prob,
            'market_prob': total_market_prob,
            'edge_pct': total_edge,
            'expected_roi': total_expected_roi,
            'kelly_fraction': total_kelly,
            'recommended_units': round(total_kelly * 100, 1),
            'confidence_tier': total_conf,
            'home_win_prob': calibrated_home_win_prob,
            'home_epa': home_epa['offensive_epa'],
            'away_epa': away_epa['offensive_epa'],
            'total_p5': total_p5,
            'total_p25': total_p25,
            'total_p50': total_p50,
            'total_p75': total_p75,
            'total_p95': total_p95,
            'total_std': total_std,
            'home_pace': home_epa['pace'],
            'away_pace': away_epa['pace'],
            'home_pass_epa': home_epa.get('pass_epa', 0.0),
            'away_pass_epa': away_epa.get('pass_epa', 0.0),
            'home_rush_epa': home_epa.get('rush_epa', 0.0),
            'away_rush_epa': away_epa.get('rush_epa', 0.0),
            'home_def_epa': home_epa['defensive_epa'],
            'away_def_epa': away_epa['defensive_epa'],
            # New comprehensive features
            'home_elo': team_features.get('home_elo', 1505),
            'away_elo': team_features.get('away_elo', 1505),
            'elo_diff': team_features.get('elo_diff', 0),
            'home_power': team_features.get('home_power_rating', 50),
            'away_power': team_features.get('away_power_rating', 50),
            'power_diff': team_features.get('power_diff', 0),
            'home_record': f"{team_features.get('home_wins', 0)}-{team_features.get('home_losses', 0)}",
            'away_record': f"{team_features.get('away_wins', 0)}-{team_features.get('away_losses', 0)}",
            'home_win_pct': team_features.get('home_win_pct', 0.5),
            'away_win_pct': team_features.get('away_win_pct', 0.5),
            'home_sos': team_features.get('home_sos', 1505),
            'away_sos': team_features.get('away_sos', 1505),
            'home_rest_days': team_features.get('home_rest', 7),
            'away_rest_days': team_features.get('away_rest', 7),
            **weather_data,
            **depth_chart_data,
        })

    # --- MONEYLINE RECOMMENDATIONS ---
    ml_odds = game_odds[
        (game_odds['side'].isin(['home', 'away', 'home_ml', 'away_ml'])) &
        ((game_odds['point'].isna()) | (game_odds['point'] == 0) | (game_odds['side'].str.contains('_ml')))
    ]
    home_ml_row = ml_odds[ml_odds['side'].isin(['home', 'home_ml'])].iloc[0] if len(ml_odds[ml_odds['side'].isin(['home', 'home_ml'])]) > 0 else None
    away_ml_row = ml_odds[ml_odds['side'].isin(['away', 'away_ml'])].iloc[0] if len(ml_odds[ml_odds['side'].isin(['away', 'away_ml'])]) > 0 else None

    if home_ml_row is not None and away_ml_row is not None:
        home_ml_odds = int(home_ml_row['american_odds'])
        away_ml_odds = int(away_ml_row['american_odds'])

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

        # Only recommend if edge is meaningful (from config)
        if ml_edge > MIN_EDGE_MONEYLINE:
            ml_kelly = calculate_kelly_fraction(ml_prob, ml_odds_used)
            ml_conf = assign_game_line_confidence_tier(ml_prob)
            ml_expected_roi = calculate_expected_roi(ml_edge, ml_odds_used)

            results.append({
                'game_id': game_id,
                'game': game_key,
                'bet_type': 'moneyline',
                'pick': ml_pick,
                'market_line': None,
                'model_fair_line': calibrated_home_win_prob,
                'model_prob': ml_prob,
                'market_prob': ml_market_prob,
                'edge_pct': ml_edge,
                'expected_roi': ml_expected_roi,
                'kelly_fraction': ml_kelly,
                'recommended_units': round(ml_kelly * 100, 1),
                'confidence_tier': ml_conf,
                'home_win_prob': calibrated_home_win_prob,
                'home_epa': home_epa['offensive_epa'],
                'away_epa': away_epa['offensive_epa'],
            })

    return results


def generate_game_line_recommendations():
    """Main function to generate game line recommendations."""
    print("=" * 70)
    print(f"GENERATING WEEK {CURRENT_WEEK} GAME LINE RECOMMENDATIONS")
    print("=" * 70)
    print(f"Season: {CURRENT_SEASON}, Week: {CURRENT_WEEK}")
    print()

    # CRITICAL FIX: Load pre-computed predictions instead of running fresh simulations
    # This ensures consistency with the predictions file (single source of truth)
    # BUT: Skip stale predictions if odds file is newer (user fetched fresh odds)
    predictions_file = PROJECT_ROOT / "data" / f"game_line_predictions_week{CURRENT_WEEK}.csv"

    # Freshness check: If odds file is newer than predictions, use fresh odds
    use_predictions_file = False
    predictions_df = None

    if predictions_file.exists() and ODDS_FILE.exists():
        odds_mtime = ODDS_FILE.stat().st_mtime
        predictions_mtime = predictions_file.stat().st_mtime

        if odds_mtime > predictions_mtime:
            # Odds file is NEWER than predictions - odds were updated after predictions were generated
            from datetime import datetime
            odds_time = datetime.fromtimestamp(odds_mtime).strftime('%Y-%m-%d %H:%M:%S')
            pred_time = datetime.fromtimestamp(predictions_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"⚠️  STALE PREDICTIONS DETECTED:")
            print(f"   Odds file updated:        {odds_time}")
            print(f"   Predictions file created: {pred_time}")
            print(f"   → Using FRESH odds (predictions file is outdated)")
            use_predictions_file = False
        else:
            # Predictions are newer or same age - use them
            try:
                predictions_df = load_game_line_predictions()
                use_predictions_file = True
                print("✅ Using pre-computed predictions (consistent with predictions file)")
            except FileNotFoundError as e:
                print(f"⚠️  WARNING: {e}")
                use_predictions_file = False
    elif predictions_file.exists():
        # Predictions exist but no odds file yet (unusual case)
        try:
            predictions_df = load_game_line_predictions()
            use_predictions_file = True
            print("✅ Using pre-computed predictions (no odds file to compare)")
        except FileNotFoundError as e:
            print(f"⚠️  WARNING: {e}")
            use_predictions_file = False
    else:
        print("⚠️  No predictions file found, will use fresh simulations")

    # Load odds data
    if not ODDS_FILE.exists():
        raise FileNotFoundError(f"Odds file not found: {ODDS_FILE}")

    odds_df = pd.read_csv(ODDS_FILE)
    # Normalize game_ids to match schedule format (LAR -> LA)
    if 'game_id' in odds_df.columns:
        odds_df['game_id'] = odds_df['game_id'].apply(normalize_game_id)
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

    # NOTE: Do NOT create a shared simulator here - it has mutable state (_game_counter)
    # that causes race conditions in ThreadPoolExecutor. Each thread will create its own.

    # Build context for parallel processing
    # THREAD SAFETY: Each thread creates its own MonteCarloSimulator instance
    context = {
        'predictions_df': predictions_df,
        'use_predictions_file': use_predictions_file,
        'odds_df': odds_df,
        'pbp_df': pbp_df,
        'calibrator': calibrator,
        'injury_model': injury_model,
        'simulator_seed': 42,  # Pass seed instead of simulator instance
        'current_season': CURRENT_SEASON,
        'current_week': CURRENT_WEEK,
    }

    # Filter to upcoming games only
    upcoming_games = []
    for _, game in week_games.iterrows():
        if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
            print(f"  Skipping {game['away_team']}@{game['home_team']} (completed)")
            continue
        upcoming_games.append(game)

    print(f"\nProcessing {len(upcoming_games)} games in parallel...")

    # Process games in parallel using ThreadPoolExecutor
    all_results = []

    # Handle case when no upcoming games
    if len(upcoming_games) == 0:
        print("  No upcoming games to process")
        all_results = []
    else:
        num_workers = min(len(upcoming_games), 4)  # Cap at 4 workers

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all games for processing
            future_to_game = {
                executor.submit(process_single_game, game, context): game
                for game in upcoming_games
            }

            # Collect results as they complete
            for future in as_completed(future_to_game):
                game = future_to_game[future]
                game_key = f"{game['away_team']} @ {game['home_team']}"
                try:
                    game_results = future.result()
                    all_results.extend(game_results)
                    if game_results:
                        print(f"  ✅ {game_key}: {len(game_results)} recommendations")
                    else:
                        print(f"  ⚠️  {game_key}: No odds or predictions found")
                except Exception as e:
                    print(f"  ❌ {game_key}: Error - {e}")

    results = all_results

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\nNo recommendations generated")
        return results_df

    # Sort by edge
    results_df = results_df.sort_values('edge_pct', ascending=False)

    # Enrich with ATS and defensive ranking data for dashboard
    print("\nEnriching with ATS and defensive rankings...")
    results_df = enrich_with_team_context(results_df, CURRENT_WEEK, CURRENT_SEASON)

    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("TOP GAME LINE RECOMMENDATIONS")
    print("=" * 70)

    # Filter to minimum edge thresholds (from config)
    # Apply different thresholds for spreads vs totals
    spread_picks = results_df[
        (results_df['bet_type'] == 'spread') &
        (results_df['edge_pct'] >= MIN_EDGE_SPREAD)
    ]
    total_picks = results_df[
        (results_df['bet_type'] == 'total') &
        (results_df['edge_pct'] >= MIN_EDGE_TOTAL)
    ]
    ml_picks = results_df[
        (results_df['bet_type'] == 'moneyline') &
        (results_df['edge_pct'] >= MIN_EDGE_MONEYLINE)
    ]
    strong_picks = pd.concat([spread_picks, total_picks, ml_picks])
    strong_picks = strong_picks.sort_values('edge_pct', ascending=False)

    print(f"\nFiltered picks (spread≥{MIN_EDGE_SPREAD}%, total≥{MIN_EDGE_TOTAL}%, ML≥{MIN_EDGE_MONEYLINE}%): {len(strong_picks)}")

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
        print(f"\nAverage edge (filtered): {strong_picks['edge_pct'].mean():.2f}%")
        print(f"Expected ROI: {strong_picks['expected_roi'].mean():.2f}%")

    return results_df


if __name__ == "__main__":
    df = generate_game_line_recommendations()
