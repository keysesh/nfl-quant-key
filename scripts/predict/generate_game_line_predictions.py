#!/usr/bin/env python3
"""
Generate Game Line Predictions (Totals, Spreads, Moneylines)

ENHANCED VERSION: Uses full MonteCarloSimulator with all features:
- Team EPA (offensive, defensive, pass, rush)
- Team pace
- Injury adjustments
- Weather conditions
- Game context (primetime, divisional)
- Elo ratings as anchor

Usage:
    python scripts/predict/generate_game_line_predictions.py --week 12
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import joblib

from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.features.team_strength import EnhancedEloCalculator
from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.schemas import SimulationInput, SimulationOutput
from nfl_quant.features.injuries import InjuryImpactModel
from nfl_quant.utils.epa_utils import regress_epa_to_mean
from nfl_quant.features.weather_features_v2 import WeatherAdjusterV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root for file paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


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


def load_depth_charts(season: int = 2025) -> pd.DataFrame:
    """
    Load official depth charts for starter/backup identification.

    Args:
        season: NFL season (default 2025)

    Returns:
        DataFrame with columns: team, position, depth_rank, player_name, gsis_id
    """
    # Use canonical depth chart loader
    from nfl_quant.data.depth_chart_loader import get_depth_charts
    dc = get_depth_charts(season=season)

    if dc.empty:
        raise ValueError("Empty depth chart data returned from canonical loader")

    # Filter to skill positions
    skill_pos = ['QB', 'RB', 'WR', 'TE', 'FB']
    pos_col = 'pos_abb' if 'pos_abb' in dc.columns else 'position'
    dc = dc[dc[pos_col].isin(skill_pos)]

    # Normalize to common format
    required_cols = ['team', 'player_name']
    for col in required_cols:
        if col not in dc.columns:
            raise ValueError(f"Missing required column: {col}")

    # Build result with available columns
    result_cols = {
        'team': 'team',
        'position': pos_col,
        'depth_rank': 'pos_rank' if 'pos_rank' in dc.columns else 'depth_team',
        'player_name': 'player_name',
        'gsis_id': 'gsis_id' if 'gsis_id' in dc.columns else None
    }

    result_data = {}
    for new_col, old_col in result_cols.items():
        if old_col and old_col in dc.columns:
            result_data[new_col] = dc[old_col]
        else:
            result_data[new_col] = None

    result = pd.DataFrame(result_data)
    result = result.dropna(subset=['team', 'player_name'])

    logger.info(f"Loaded {len(result)} depth chart entries from canonical loader")
    return result


def get_team_starters(depth_charts: pd.DataFrame, team: str) -> Dict[str, dict]:
    """
    Get projected starters and backups for a team from depth charts.

    Args:
        depth_charts: Loaded depth chart DataFrame
        team: Team abbreviation

    Returns:
        Dict with position -> {starter: name, backup: name, starter_id: gsis_id, backup_id: gsis_id}
    """
    team_dc = depth_charts[depth_charts['team'] == team]

    if len(team_dc) == 0:
        return {}

    result = {}
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_dc = team_dc[team_dc['position'] == pos].sort_values('depth_rank')

        if len(pos_dc) == 0:
            continue

        # Get starter (depth_rank = 1)
        starters = pos_dc[pos_dc['depth_rank'] == 1]
        backups = pos_dc[pos_dc['depth_rank'] == 2]

        if pos == 'WR':
            # WR can have multiple starters (WR1, WR2, WR3)
            starter_list = starters['player_name'].tolist()[:3]
            backup_list = backups['player_name'].tolist()[:3]
            result[pos] = {
                'starters': starter_list,
                'backups': backup_list,
                'starter_ids': starters['gsis_id'].tolist()[:3],
                'backup_ids': backups['gsis_id'].tolist()[:3]
            }
        else:
            # Single starter positions
            starter = starters.iloc[0] if len(starters) > 0 else None
            backup = backups.iloc[0] if len(backups) > 0 else None

            result[pos] = {
                'starter': starter['player_name'] if starter is not None else None,
                'backup': backup['player_name'] if backup is not None else None,
                'starter_id': starter['gsis_id'] if starter is not None else None,
                'backup_id': backup['gsis_id'] if backup is not None else None
            }

    return result


def get_active_starter(
    depth_chart_info: dict,
    position: str,
    injury_df: pd.DataFrame,
    team: str
) -> Tuple[str, str, bool]:
    """
    Determine the active starter accounting for injuries.

    Args:
        depth_chart_info: Output from get_team_starters()
        position: Position to check (QB, RB, TE)
        injury_df: Injury DataFrame
        team: Team abbreviation

    Returns:
        Tuple of (active_player_name, status, is_backup)
        status: 'starter', 'backup_due_to_injury', 'unknown'
    """
    if position not in depth_chart_info:
        return None, 'unknown', False

    pos_info = depth_chart_info[position]
    starter = pos_info.get('starter')
    backup = pos_info.get('backup')

    if starter is None:
        return backup, 'unknown', True

    # Check if starter is injured (OUT or Doubtful)
    if len(injury_df) > 0:
        # Try to find starter in injury list
        starter_injuries = injury_df[
            (injury_df['team'] == team) &
            (injury_df['full_name'].str.contains(starter.split()[-1], case=False, na=False))
        ]

        if len(starter_injuries) > 0:
            injury_status = starter_injuries.iloc[0].get('report_status', '')
            if injury_status in ['Out', 'Doubtful', 'IR', 'PUP', 'Suspended']:
                logger.info(f"  📋 Depth chart: {team} {position} {starter} is {injury_status}, backup {backup} projected")
                return backup, 'backup_due_to_injury', True

    return starter, 'starter', False


def filter_garbage_time(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out garbage time plays where EPA is artificially inflated.

    Garbage time definition:
    - Win probability < 10% or > 90% in 4th quarter
    - Score differential > 21 points after halftime
    """
    if len(pbp_df) == 0:
        return pbp_df

    # Create mask for non-garbage time plays
    is_4th_qtr = pbp_df['qtr'] == 4
    has_wp = pbp_df['wp'].notna()

    # Win prob extreme in 4th quarter
    wp_garbage = is_4th_qtr & has_wp & ((pbp_df['wp'] < 0.10) | (pbp_df['wp'] > 0.90))

    # Score differential > 21 after halftime
    is_second_half = pbp_df['qtr'].isin([3, 4])
    score_diff = (pbp_df['posteam_score'] - pbp_df['defteam_score']).abs()
    blowout = is_second_half & (score_diff > 21)

    # Combine garbage time conditions
    is_garbage_time = wp_garbage | blowout

    # Return non-garbage time plays
    return pbp_df[~is_garbage_time]


def calculate_team_epa(pbp_df: pd.DataFrame, team: str, current_week: int, weeks: int = 10) -> dict:
    """
    Calculate team EPA metrics from play-by-play data with regression to mean.

    ENHANCED: Now includes garbage time filtering, red zone splits, and home/away splits.

    CRITICAL: Only uses data from BEFORE current_week to prevent leakage.

    Returns:
        dict with offensive_epa, defensive_epa, pass_epa, rush_epa, pace,
              red_zone_epa, home_epa, away_epa
    """
    # Filter to weeks BEFORE current week (prevent leakage)
    available_weeks = [w for w in range(max(1, current_week - weeks), current_week)]

    if len(available_weeks) == 0:
        logger.warning(f"No prior weeks available for {team}, using league average EPA")
        return {
            'offensive_epa': 0.0,
            'defensive_epa': 0.0,
            'pass_epa': 0.0,
            'rush_epa': 0.0,
            'pace': 65.0,
            'sample_games': 0,
            'red_zone_epa': 0.0,
            'home_off_epa': 0.0,
            'away_off_epa': 0.0,
            'garbage_time_filtered': True
        }

    # Offensive EPA (when team has possession)
    off_plays = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(available_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    # Defensive EPA (when team is defending)
    def_plays = pbp_df[
        (pbp_df['defteam'] == team) &
        (pbp_df['week'].isin(available_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    if len(off_plays) == 0 or len(def_plays) == 0:
        logger.warning(f"No PBP data for {team} in weeks {available_weeks}, using defaults")
        return {
            'offensive_epa': 0.0,
            'defensive_epa': 0.0,
            'pass_epa': 0.0,
            'rush_epa': 0.0,
            'pace': 65.0,
            'sample_games': 0,
            'red_zone_epa': 0.0,
            'home_off_epa': 0.0,
            'away_off_epa': 0.0,
            'garbage_time_filtered': True
        }

    # GARBAGE TIME FILTERING - Remove inflated EPA from blowouts
    off_plays_clean = filter_garbage_time(off_plays)
    def_plays_clean = filter_garbage_time(def_plays)

    plays_filtered = len(off_plays) - len(off_plays_clean)
    if plays_filtered > 0:
        logger.debug(f"  {team}: Filtered {plays_filtered} garbage time plays ({plays_filtered/len(off_plays):.1%})")

    # Use filtered plays for EPA calculation
    off_plays = off_plays_clean
    def_plays = def_plays_clean

    # Count sample size for regression
    off_games = len(off_plays['game_id'].unique())
    def_games = len(def_plays['game_id'].unique())
    sample_games = min(off_games, def_games)

    # Calculate RAW EPA per play
    raw_offensive_epa = off_plays['epa'].mean() if len(off_plays) > 0 else 0.0
    raw_defensive_epa = def_plays['epa'].mean() if len(def_plays) > 0 else 0.0

    # CRITICAL: Regress EPA values toward league mean (0.0)
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

    # RED ZONE EPA (inside the 20)
    red_zone_plays = off_plays[off_plays['yardline_100'] <= 20] if 'yardline_100' in off_plays.columns else pd.DataFrame()
    red_zone_epa = red_zone_plays['epa'].mean() if len(red_zone_plays) > 10 else offensive_epa

    # HOME/AWAY SPLITS
    home_plays = off_plays[off_plays['home_team'] == team] if 'home_team' in off_plays.columns else pd.DataFrame()
    away_plays = off_plays[off_plays['away_team'] == team] if 'away_team' in off_plays.columns else pd.DataFrame()

    home_off_epa = home_plays['epa'].mean() if len(home_plays) > 20 else offensive_epa
    away_off_epa = away_plays['epa'].mean() if len(away_plays) > 20 else offensive_epa

    return {
        'offensive_epa': float(offensive_epa),
        'defensive_epa': float(defensive_epa),
        'pass_epa': float(pass_epa),
        'rush_epa': float(rush_epa),
        'pace': float(pace),
        'sample_games': sample_games,
        'red_zone_epa': float(red_zone_epa),
        'home_off_epa': float(home_off_epa),
        'away_off_epa': float(away_off_epa),
        'garbage_time_filtered': True
    }


def calculate_qb_specific_epa(
    pbp_df: pd.DataFrame,
    team: str,
    current_week: int,
    injury_df: pd.DataFrame = None
) -> dict:
    """
    Calculate QB-specific EPA and detect QB changes.

    Returns:
        dict with:
        - current_qb: Name of projected starter
        - current_qb_epa: EPA of current starter
        - current_qb_games: Games started by current QB
        - previous_qb: Name of previous starter (if changed)
        - previous_qb_epa: EPA of previous starter
        - qb_change_detected: True if QB change detected
        - pass_epa_adjustment: Adjustment to apply to team pass EPA
    """
    # Get pass plays for this team before current week
    available_weeks = [w for w in range(1, current_week)]

    team_passes = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(available_weeks)) &
        (pbp_df['play_type'] == 'pass') &
        (pbp_df['passer_player_name'].notna())
    ]

    if len(team_passes) == 0:
        return {
            'current_qb': 'Unknown',
            'current_qb_epa': 0.0,
            'current_qb_games': 0,
            'previous_qb': None,
            'previous_qb_epa': 0.0,
            'qb_change_detected': False,
            'pass_epa_adjustment': 0.0
        }

    # Calculate EPA by QB
    qb_stats = team_passes.groupby('passer_player_name').agg({
        'epa': ['count', 'mean', 'sum'],
        'week': ['min', 'max', 'nunique']
    })
    qb_stats.columns = ['attempts', 'epa_per_play', 'total_epa', 'first_week', 'last_week', 'games']
    qb_stats = qb_stats.sort_values('attempts', ascending=False)

    # Identify primary QB (most attempts overall)
    primary_qb = qb_stats.index[0]
    primary_qb_epa = qb_stats.loc[primary_qb, 'epa_per_play']
    primary_qb_games = qb_stats.loc[primary_qb, 'games']

    # Check most recent 2 weeks for QB change
    recent_weeks = [w for w in available_weeks if w >= current_week - 2]
    recent_passes = team_passes[team_passes['week'].isin(recent_weeks)]

    if len(recent_passes) > 0:
        recent_qb_stats = recent_passes.groupby('passer_player_name').size()
        recent_primary_qb = recent_qb_stats.idxmax()
    else:
        recent_primary_qb = primary_qb

    # Detect QB change
    qb_change_detected = False
    previous_qb = None
    previous_qb_epa = 0.0
    current_qb = recent_primary_qb
    current_qb_epa = primary_qb_epa  # Default to primary

    if recent_primary_qb != primary_qb:
        qb_change_detected = True
        previous_qb = primary_qb
        previous_qb_epa = primary_qb_epa

        # Get new QB's EPA (if they have enough data)
        if recent_primary_qb in qb_stats.index:
            new_qb_data = qb_stats.loc[recent_primary_qb]
            if new_qb_data['attempts'] >= 20:  # Minimum sample
                current_qb_epa = new_qb_data['epa_per_play']
            else:
                # Limited sample - regress heavily toward league average (0.0)
                raw_epa = new_qb_data['epa_per_play']
                sample_weight = min(new_qb_data['attempts'] / 100, 0.5)  # Max 50% weight
                current_qb_epa = raw_epa * sample_weight + 0.0 * (1 - sample_weight)
                logger.info(f"  {team} backup QB {recent_primary_qb}: limited sample ({new_qb_data['attempts']} att), regressed EPA: {current_qb_epa:+.3f}")
        else:
            # New QB with no data - use league average with slight penalty
            current_qb_epa = -0.05  # Backup QB penalty
            logger.info(f"  {team} backup QB {recent_primary_qb}: no data, using backup penalty EPA: {current_qb_epa:+.3f}")

    # Also check injury data for QB injuries
    if injury_df is not None and len(injury_df) > 0:
        team_injuries = injury_df[
            (injury_df['team'] == team) &
            (injury_df['position'].isin(['QB']))
        ]

        for _, inj in team_injuries.iterrows():
            injured_player = inj.get('player', inj.get('player_name', ''))
            injury_status = inj.get('injury_status', inj.get('status', ''))

            # Check if primary QB is injured
            if primary_qb in str(injured_player) or str(injured_player) in primary_qb:
                if injury_status in ['Out', 'IR', 'Doubtful', 'PUP']:
                    if not qb_change_detected:
                        qb_change_detected = True
                        previous_qb = primary_qb
                        previous_qb_epa = primary_qb_epa
                        # Find backup
                        if len(qb_stats) > 1:
                            backup_qb = qb_stats.index[1]
                            backup_data = qb_stats.loc[backup_qb]
                            current_qb = backup_qb
                            if backup_data['attempts'] >= 20:
                                current_qb_epa = backup_data['epa_per_play']
                            else:
                                current_qb_epa = backup_data['epa_per_play'] * 0.3 - 0.03  # Heavy regression + penalty
                        else:
                            current_qb = 'Unknown Backup'
                            current_qb_epa = -0.08  # Significant backup penalty
                        logger.info(f"  {team} QB {primary_qb} OUT ({injury_status}), backup {current_qb} EPA: {current_qb_epa:+.3f}")

    # Calculate adjustment to apply
    pass_epa_adjustment = current_qb_epa - primary_qb_epa if qb_change_detected else 0.0

    return {
        'current_qb': current_qb,
        'current_qb_epa': float(current_qb_epa),
        'current_qb_games': int(qb_stats.loc[current_qb, 'games']) if current_qb in qb_stats.index else 0,
        'previous_qb': previous_qb,
        'previous_qb_epa': float(previous_qb_epa) if previous_qb else 0.0,
        'qb_change_detected': qb_change_detected,
        'pass_epa_adjustment': float(pass_epa_adjustment)
    }


def calculate_rb_specific_epa(
    pbp_df: pd.DataFrame,
    team: str,
    current_week: int,
    injury_df: pd.DataFrame = None
) -> dict:
    """
    Calculate RB-specific rushing EPA and detect RB changes.

    Tracks the primary ball carrier and detects when a key RB is out.

    Returns:
        dict with:
        - primary_rb: Name of primary RB (most carries)
        - primary_rb_epa: EPA of primary RB
        - primary_rb_carries: Carries by primary RB
        - rb_depth: List of RBs with their EPA/carries
        - injured_rbs: List of injured RBs
        - rush_epa_adjustment: Adjustment to apply to team rush EPA
    """
    available_weeks = [w for w in range(1, current_week)]

    team_rushes = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(available_weeks)) &
        (pbp_df['play_type'] == 'run') &
        (pbp_df['rusher_player_name'].notna())
    ]

    if len(team_rushes) == 0:
        return {
            'primary_rb': 'Unknown',
            'primary_rb_epa': 0.0,
            'primary_rb_carries': 0,
            'rb_depth': [],
            'injured_rbs': [],
            'rush_epa_adjustment': 0.0
        }

    # Calculate EPA by rusher
    rb_stats = team_rushes.groupby('rusher_player_name').agg({
        'epa': ['count', 'mean', 'sum'],
        'week': ['min', 'max', 'nunique']
    })
    rb_stats.columns = ['carries', 'epa_per_carry', 'total_epa', 'first_week', 'last_week', 'games']
    rb_stats = rb_stats.sort_values('carries', ascending=False)

    # Filter to likely RBs (minimum 10 carries, exclude QB sneaks)
    rb_stats = rb_stats[rb_stats['carries'] >= 10]

    if len(rb_stats) == 0:
        return {
            'primary_rb': 'Unknown',
            'primary_rb_epa': 0.0,
            'primary_rb_carries': 0,
            'rb_depth': [],
            'injured_rbs': [],
            'rush_epa_adjustment': 0.0
        }

    # Primary RB
    primary_rb = rb_stats.index[0]
    primary_rb_epa = rb_stats.loc[primary_rb, 'epa_per_carry']
    primary_rb_carries = int(rb_stats.loc[primary_rb, 'carries'])

    # Build depth chart
    rb_depth = []
    for rb_name in rb_stats.index[:4]:  # Top 4 RBs
        rb_depth.append({
            'name': rb_name,
            'carries': int(rb_stats.loc[rb_name, 'carries']),
            'epa': float(rb_stats.loc[rb_name, 'epa_per_carry']),
            'games': int(rb_stats.loc[rb_name, 'games'])
        })

    # Check for injured RBs
    injured_rbs = []
    rush_epa_adjustment = 0.0

    if injury_df is not None and len(injury_df) > 0:
        team_injuries = injury_df[
            (injury_df['team'] == team) &
            (injury_df['position'].isin(['RB', 'FB']))
        ]

        for _, inj in team_injuries.iterrows():
            injured_player = inj.get('player', inj.get('player_name', ''))
            injury_status = inj.get('injury_status', inj.get('status', ''))

            # Check if any key RB is injured
            for rb in rb_depth[:2]:  # Check top 2 RBs
                if rb['name'] in str(injured_player) or str(injured_player) in rb['name']:
                    if injury_status in ['Out', 'IR', 'Doubtful', 'PUP']:
                        injured_rbs.append({
                            'name': rb['name'],
                            'status': injury_status,
                            'epa': rb['epa'],
                            'carries': rb['carries']
                        })

                        # Calculate adjustment based on share of carries
                        total_carries = sum(r['carries'] for r in rb_depth)
                        carry_share = rb['carries'] / total_carries if total_carries > 0 else 0

                        # Find replacement EPA (next healthy RB)
                        replacement_epa = 0.0  # League average
                        for backup in rb_depth:
                            if backup['name'] not in [irb['name'] for irb in injured_rbs]:
                                replacement_epa = backup['epa']
                                break

                        # Adjustment = (replacement - injured) * carry share
                        adjustment = (replacement_epa - rb['epa']) * carry_share
                        rush_epa_adjustment += adjustment

                        logger.info(f"  {team} RB {rb['name']} OUT ({injury_status}), "
                                    f"EPA impact: {adjustment:+.3f}")

    return {
        'primary_rb': primary_rb,
        'primary_rb_epa': float(primary_rb_epa),
        'primary_rb_carries': primary_rb_carries,
        'rb_depth': rb_depth,
        'injured_rbs': injured_rbs,
        'rush_epa_adjustment': float(rush_epa_adjustment)
    }


def calculate_receiver_specific_epa(
    pbp_df: pd.DataFrame,
    team: str,
    current_week: int,
    injury_df: pd.DataFrame = None,
    position_filter: str = None  # 'WR', 'TE', or None for all
) -> dict:
    """
    Calculate receiver-specific EPA and detect key receiver injuries.

    Tracks top receivers and their individual receiving EPA.

    Args:
        pbp_df: Play-by-play data
        team: Team abbreviation
        current_week: Current week (to prevent leakage)
        injury_df: Injury data
        position_filter: Filter to specific position ('WR', 'TE') or None for all

    Returns:
        dict with:
        - top_receivers: List of top receivers with EPA
        - injured_receivers: List of injured receivers
        - pass_epa_adjustment: Adjustment to team pass EPA from injuries
        - target_share_by_receiver: Dict of receiver -> target share
    """
    available_weeks = [w for w in range(1, current_week)]

    team_passes = pbp_df[
        (pbp_df['posteam'] == team) &
        (pbp_df['week'].isin(available_weeks)) &
        (pbp_df['play_type'] == 'pass') &
        (pbp_df['receiver_player_name'].notna())
    ]

    if len(team_passes) == 0:
        return {
            'top_receivers': [],
            'injured_receivers': [],
            'pass_epa_adjustment': 0.0,
            'target_share_by_receiver': {}
        }

    # Calculate EPA by receiver
    rec_stats = team_passes.groupby('receiver_player_name').agg({
        'epa': ['count', 'mean', 'sum'],
        'complete_pass': 'sum',
        'week': ['min', 'max', 'nunique']
    })
    rec_stats.columns = ['targets', 'epa_per_target', 'total_epa', 'receptions',
                         'first_week', 'last_week', 'games']
    rec_stats = rec_stats.sort_values('targets', ascending=False)

    # Calculate target share
    total_targets = rec_stats['targets'].sum()
    rec_stats['target_share'] = rec_stats['targets'] / total_targets if total_targets > 0 else 0

    # Filter to significant receivers (minimum 8 targets)
    rec_stats = rec_stats[rec_stats['targets'] >= 8]

    # If position filter, try to identify position from rosters
    # For now, use target volume as proxy (top targets likely WRs)

    # Build receiver list
    top_receivers = []
    for rec_name in rec_stats.index[:6]:  # Top 6 pass catchers
        top_receivers.append({
            'name': rec_name,
            'targets': int(rec_stats.loc[rec_name, 'targets']),
            'epa': float(rec_stats.loc[rec_name, 'epa_per_target']),
            'target_share': float(rec_stats.loc[rec_name, 'target_share']),
            'games': int(rec_stats.loc[rec_name, 'games']),
            'receptions': int(rec_stats.loc[rec_name, 'receptions'])
        })

    target_share_by_receiver = {
        rec_name: float(rec_stats.loc[rec_name, 'target_share'])
        for rec_name in rec_stats.index
    }

    # Check for injured receivers
    injured_receivers = []
    pass_epa_adjustment = 0.0

    if injury_df is not None and len(injury_df) > 0:
        team_injuries = injury_df[
            (injury_df['team'] == team) &
            (injury_df['position'].isin(['WR', 'TE']))
        ]

        for _, inj in team_injuries.iterrows():
            injured_player = inj.get('player', inj.get('player_name', ''))
            injury_status = inj.get('injury_status', inj.get('status', ''))
            injured_position = inj.get('position', '')

            # Apply position filter if specified
            if position_filter and injured_position != position_filter:
                continue

            # Check if any top receiver is injured
            for rec in top_receivers[:4]:  # Check top 4 receivers
                if rec['name'] in str(injured_player) or str(injured_player) in rec['name']:
                    if injury_status in ['Out', 'IR', 'Doubtful', 'PUP']:
                        injured_receivers.append({
                            'name': rec['name'],
                            'position': injured_position,
                            'status': injury_status,
                            'epa': rec['epa'],
                            'target_share': rec['target_share']
                        })

                        # Calculate adjustment based on target share
                        # When WR1/TE1 is out, targets redistribute to lower-EPA receivers

                        # Find average EPA of remaining healthy receivers
                        healthy_receivers = [
                            r for r in top_receivers
                            if r['name'] not in [ir['name'] for ir in injured_receivers]
                        ]

                        if len(healthy_receivers) > 0:
                            avg_healthy_epa = sum(r['epa'] for r in healthy_receivers) / len(healthy_receivers)
                        else:
                            avg_healthy_epa = 0.0  # League average

                        # Adjustment = (redistributed EPA - injured EPA) * target share
                        adjustment = (avg_healthy_epa - rec['epa']) * rec['target_share']
                        pass_epa_adjustment += adjustment

                        logger.info(f"  {team} {injured_position} {rec['name']} OUT ({injury_status}), "
                                    f"target share: {rec['target_share']:.1%}, EPA impact: {adjustment:+.3f}")

    return {
        'top_receivers': top_receivers,
        'injured_receivers': injured_receivers,
        'pass_epa_adjustment': float(pass_epa_adjustment),
        'target_share_by_receiver': target_share_by_receiver
    }


def calculate_defensive_position_epa(
    pbp_df: pd.DataFrame,
    team: str,
    current_week: int,
    injury_df: pd.DataFrame = None
) -> dict:
    """
    Calculate defensive position-group EPA for game line adjustments.

    Tracks:
    - Pass rush EPA (pressure effectiveness)
    - Coverage EPA (against pass)
    - Run defense EPA (against rush)

    Returns:
        dict with:
        - pass_rush_epa: Effectiveness of pass rush
        - coverage_epa: EPA allowed in coverage
        - run_def_epa: EPA allowed against the run
        - pressure_rate: Percentage of dropbacks with pressure
        - injured_defenders: Key defensive injuries
        - def_epa_adjustment: Total defensive adjustment
    """
    available_weeks = [w for w in range(1, current_week)]

    # Get defensive plays for this team
    def_plays = pbp_df[
        (pbp_df['defteam'] == team) &
        (pbp_df['week'].isin(available_weeks)) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    if len(def_plays) == 0:
        return {
            'pass_rush_epa': 0.0,
            'coverage_epa': 0.0,
            'run_def_epa': 0.0,
            'pressure_rate': 0.0,
            'injured_defenders': [],
            'def_epa_adjustment': 0.0
        }

    # Filter garbage time
    def_plays = filter_garbage_time(def_plays)

    # PASS RUSH EPA - EPA on sacks, pressures, QB hits
    # Lower (more negative) = better pass rush
    pass_plays = def_plays[def_plays['play_type'] == 'pass']

    # Sacks have big negative EPA
    sack_plays = pass_plays[pass_plays['sack'] == 1] if 'sack' in pass_plays.columns else pd.DataFrame()
    non_sack_pass = pass_plays[pass_plays.get('sack', 0) != 1]

    # Pass rush effectiveness = sack rate + EPA on sacks
    sack_rate = len(sack_plays) / len(pass_plays) if len(pass_plays) > 0 else 0
    sack_epa = sack_plays['epa'].mean() if len(sack_plays) > 0 else 0.0

    # Pressure rate (if available)
    if 'qb_hit' in pass_plays.columns:
        qb_hits = pass_plays['qb_hit'].sum()
        pressure_rate = (len(sack_plays) + qb_hits) / len(pass_plays) if len(pass_plays) > 0 else 0
    else:
        pressure_rate = sack_rate * 3  # Estimate: pressures ~3x sacks

    # Pass rush EPA = blend of sack EPA and pressure impact
    pass_rush_epa = sack_epa * sack_rate * 10  # Scale for impact

    # COVERAGE EPA - EPA allowed on completed passes (lower = better coverage)
    coverage_epa = non_sack_pass['epa'].mean() if len(non_sack_pass) > 0 else 0.0

    # RUN DEFENSE EPA - EPA allowed on runs (lower = better run D)
    run_plays = def_plays[def_plays['play_type'] == 'run']
    run_def_epa = run_plays['epa'].mean() if len(run_plays) > 0 else 0.0

    # Check for injured key defenders
    injured_defenders = []
    def_epa_adjustment = 0.0

    if injury_df is not None and len(injury_df) > 0:
        # Key defensive positions
        team_injuries = injury_df[
            (injury_df['team'] == team) &
            (injury_df['position'].isin(['DE', 'DT', 'EDGE', 'OLB', 'ILB', 'LB', 'CB', 'S', 'FS', 'SS']))
        ]

        for _, inj in team_injuries.iterrows():
            injured_player = inj.get('player', inj.get('player_name', ''))
            injury_status = inj.get('injury_status', inj.get('status', ''))
            position = inj.get('position', '')

            if injury_status in ['Out', 'IR', 'Doubtful', 'PUP']:
                # Estimate impact based on position
                position_impact = {
                    'DE': 0.03, 'DT': 0.02, 'EDGE': 0.03,  # Pass rushers
                    'CB': 0.04, 'S': 0.02, 'FS': 0.02, 'SS': 0.02,  # Coverage
                    'LB': 0.02, 'OLB': 0.02, 'ILB': 0.02,  # Linebackers
                }.get(position, 0.01)

                injured_defenders.append({
                    'name': injured_player,
                    'position': position,
                    'status': injury_status,
                    'impact': position_impact
                })

                # Losing defenders makes defense worse (positive EPA adjustment = worse)
                def_epa_adjustment += position_impact

    return {
        'pass_rush_epa': float(pass_rush_epa),
        'coverage_epa': float(coverage_epa),
        'run_def_epa': float(run_def_epa),
        'pressure_rate': float(pressure_rate),
        'sack_rate': float(sack_rate),
        'injured_defenders': injured_defenders,
        'def_epa_adjustment': float(def_epa_adjustment)
    }


def calculate_game_script_projection(
    home_epa: dict,
    away_epa: dict,
    elo_features: dict,
    vegas_spread: float = None
) -> dict:
    """
    Project game script to adjust for pace and play-calling tendencies.

    Teams with lead tend to run more; teams trailing pass more.
    This affects expected play volume and EPA application.

    Returns:
        dict with projected pass/rush ratios for each team
    """
    # Use spread to estimate game script
    if vegas_spread is not None:
        spread = vegas_spread
    elif elo_features:
        spread = elo_features.get('elo_spread', 0)
    else:
        spread = 0

    # Home team favored by spread points
    # If favored, expect more rushing (protecting lead)
    # If underdog, expect more passing (catching up)

    # Base pass/run ratio is ~60/40
    base_pass_rate = 0.60
    base_rush_rate = 0.40

    # Adjust based on expected game script
    # Each point of spread = ~0.5% shift in pass rate
    spread_adjustment = spread * 0.005

    home_pass_rate = base_pass_rate - spread_adjustment  # Favorite runs more
    home_rush_rate = base_rush_rate + spread_adjustment

    away_pass_rate = base_pass_rate + spread_adjustment  # Underdog passes more
    away_rush_rate = base_rush_rate - spread_adjustment

    # Clamp to reasonable bounds
    home_pass_rate = max(0.45, min(0.75, home_pass_rate))
    home_rush_rate = 1 - home_pass_rate
    away_pass_rate = max(0.45, min(0.75, away_pass_rate))
    away_rush_rate = 1 - away_pass_rate

    # Calculate script-adjusted EPA
    home_script_epa = (
        home_epa.get('pass_epa', 0) * home_pass_rate +
        home_epa.get('rush_epa', 0) * home_rush_rate
    )

    away_script_epa = (
        away_epa.get('pass_epa', 0) * away_pass_rate +
        away_epa.get('rush_epa', 0) * away_rush_rate
    )

    return {
        'home_pass_rate': float(home_pass_rate),
        'home_rush_rate': float(home_rush_rate),
        'away_pass_rate': float(away_pass_rate),
        'away_rush_rate': float(away_rush_rate),
        'home_script_epa': float(home_script_epa),
        'away_script_epa': float(away_script_epa),
        'spread_used': float(spread)
    }


def calculate_all_skill_position_epa(
    pbp_df: pd.DataFrame,
    team: str,
    current_week: int,
    injury_df: pd.DataFrame = None
) -> dict:
    """
    Calculate EPA adjustments for ALL skill positions and aggregate into team impact.

    ENHANCED: Now includes defensive position groups and starter/backup differential.

    Combines QB, RB, WR, TE, and defensive analysis for comprehensive team adjustment.

    Returns:
        dict with:
        - qb_info: QB-specific analysis
        - rb_info: RB-specific analysis
        - receiver_info: WR/TE combined analysis
        - defense_info: Defensive position analysis
        - total_offensive_epa_adjustment: Combined adjustment to team offensive EPA
        - total_defensive_epa_adjustment: Adjustment to defensive EPA
        - total_pass_epa_adjustment: Adjustment to pass EPA
        - total_rush_epa_adjustment: Adjustment to rush EPA
        - key_injuries: List of all key player injuries
    """
    # Get individual position analyses
    qb_info = calculate_qb_specific_epa(pbp_df, team, current_week, injury_df)
    rb_info = calculate_rb_specific_epa(pbp_df, team, current_week, injury_df)
    receiver_info = calculate_receiver_specific_epa(pbp_df, team, current_week, injury_df)
    defense_info = calculate_defensive_position_epa(pbp_df, team, current_week, injury_df)

    # Calculate pass EPA adjustment (QB + WR/TE)
    # QB change is most impactful (~60-70% of pass EPA)
    # Receiver changes affect remaining (~30-40%)
    qb_weight = 0.65
    receiver_weight = 0.35

    total_pass_epa_adjustment = (
        qb_info['pass_epa_adjustment'] * qb_weight +
        receiver_info['pass_epa_adjustment'] * receiver_weight
    )

    # Rush EPA adjustment comes from RB
    total_rush_epa_adjustment = rb_info['rush_epa_adjustment']

    # Total offensive adjustment (pass ~60% of offense, rush ~40%)
    pass_weight = 0.60
    rush_weight = 0.40

    total_offensive_epa_adjustment = (
        total_pass_epa_adjustment * pass_weight +
        total_rush_epa_adjustment * rush_weight
    )

    # Defensive adjustment from injured defenders
    total_defensive_epa_adjustment = defense_info['def_epa_adjustment']

    # Collect all key injuries
    key_injuries = []

    if qb_info['qb_change_detected']:
        key_injuries.append({
            'position': 'QB',
            'player': qb_info['previous_qb'],
            'replacement': qb_info['current_qb'],
            'epa_impact': qb_info['pass_epa_adjustment'],
            'side': 'offense'
        })

    for rb in rb_info['injured_rbs']:
        key_injuries.append({
            'position': 'RB',
            'player': rb['name'],
            'status': rb['status'],
            'epa_impact': rb['epa'],
            'side': 'offense'
        })

    for rec in receiver_info['injured_receivers']:
        key_injuries.append({
            'position': rec['position'],
            'player': rec['name'],
            'status': rec['status'],
            'epa_impact': rec['target_share'],
            'side': 'offense'
        })

    for defender in defense_info['injured_defenders']:
        key_injuries.append({
            'position': defender['position'],
            'player': defender['name'],
            'status': defender['status'],
            'epa_impact': defender['impact'],
            'side': 'defense'
        })

    return {
        'qb_info': qb_info,
        'rb_info': rb_info,
        'receiver_info': receiver_info,
        'defense_info': defense_info,
        'total_offensive_epa_adjustment': float(total_offensive_epa_adjustment),
        'total_defensive_epa_adjustment': float(total_defensive_epa_adjustment),
        'total_pass_epa_adjustment': float(total_pass_epa_adjustment),
        'total_rush_epa_adjustment': float(total_rush_epa_adjustment),
        'key_injuries': key_injuries
    }


def load_injury_data(week: int) -> pd.DataFrame:
    """Load injury data from available sources."""
    # Try sleeper injuries first
    sleeper_path = PROJECT_ROOT / 'data' / 'injuries' / f'injuries_week{week}.parquet'
    if sleeper_path.exists():
        return pd.read_parquet(sleeper_path)

    # Try nflverse injuries
    nflverse_path = PROJECT_ROOT / 'data' / 'nflverse' / 'injuries.parquet'
    if nflverse_path.exists():
        injuries = pd.read_parquet(nflverse_path)
        return injuries[injuries['week'] >= week - 1]

    return pd.DataFrame()


def load_weather_data(season: int, week: int) -> Dict[str, dict]:
    """
    Load weather data for games using LIVE weather fetcher.

    UPDATED: Uses live weather data from nflweather.com instead of NFLverse schedules.
    Falls back to NFLverse if live weather not available.

    Returns dict mapping game_id -> weather info
    """
    weather_data = {}

    # Try to load from LIVE weather fetcher first
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'fetch'))
        from fetch_nfl_weather import load_current_week_weather, get_week_weather_manual

        # Load live weather
        live_games = load_current_week_weather(week, season)

        if live_games:
            logger.info(f"Using LIVE weather data for {len(live_games)} games")

            # Load schedules to get game_ids
            schedules_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
            if schedules_path.exists():
                schedules = pd.read_parquet(schedules_path)
                week_games = schedules[
                    (schedules['season'] == season) &
                    (schedules['week'] == week)
                ]

                # Map live weather to game_ids
                for _, game in week_games.iterrows():
                    game_id = game['game_id']
                    home_team = game['home_team']
                    away_team = game['away_team']

                    # Find matching live weather
                    for lw in live_games:
                        if lw.home_team == home_team and lw.away_team == away_team:
                            weather_data[game_id] = {
                                'is_dome': lw.is_dome,
                                'temperature': float(lw.temperature),
                                'wind_speed': float(lw.wind_speed),
                                'precipitation': lw.precip_type,
                                'precip_chance': lw.precip_chance,
                                'conditions': lw.conditions,
                                'passing_adjustment': lw.passing_adjustment,
                                'total_adjustment': lw.total_adjustment,
                                'severity': lw.severity,
                            }
                            break
                    else:
                        # No live weather for this game, use schedule fallback
                        roof = game.get('roof', '')
                        is_dome = roof in ['dome', 'closed'] if pd.notna(roof) else False
                        weather_data[game_id] = {
                            'is_dome': is_dome,
                            'temperature': None,
                            'wind_speed': None,
                            'precipitation': None,
                        }

            return weather_data

    except ImportError as e:
        logger.warning(f"Could not import live weather fetcher: {e}")
    except Exception as e:
        logger.warning(f"Error loading live weather: {e}")

    # Fallback to NFLverse schedules
    logger.info("Falling back to NFLverse schedule weather data")
    schedules_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if schedules_path.exists():
        schedules = pd.read_parquet(schedules_path)
        week_games = schedules[
            (schedules['season'] == season) &
            (schedules['week'] == week)
        ]

        for _, game in week_games.iterrows():
            game_id = game['game_id']

            # Determine if dome game
            roof = game.get('roof', '')
            is_dome = roof in ['dome', 'closed'] if pd.notna(roof) else False

            # Get temperature/weather if available
            temp = game.get('temp') if pd.notna(game.get('temp')) else None
            wind = game.get('wind') if pd.notna(game.get('wind')) else None

            weather_data[game_id] = {
                'is_dome': is_dome,
                'temperature': float(temp) if temp is not None else None,
                'wind_speed': float(wind) if wind is not None else None,
                'precipitation': None
            }

    return weather_data


def get_game_context(game_row: pd.Series, schedules_df: pd.DataFrame) -> dict:
    """
    Determine game context (primetime, divisional, etc.)

    Returns dict with is_divisional, game_type, is_primetime
    """
    home_team = game_row['home_team']
    away_team = game_row['away_team']

    # Divisional matchups
    DIVISIONS = {
        'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
        'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
        'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
        'AFC_WEST': ['DEN', 'KC', 'LAC', 'LV'],
        'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
        'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
        'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
        'NFC_WEST': ['ARI', 'LAR', 'SEA', 'SF'],
    }

    is_divisional = False
    for div_teams in DIVISIONS.values():
        if home_team in div_teams and away_team in div_teams:
            is_divisional = True
            break

    # Game type detection from gameday/time
    gameday = game_row.get('gameday', '')
    gametime = game_row.get('gametime', '')

    game_type = 'Regular'
    is_primetime = False

    # Thursday Night Football
    if pd.notna(gameday) and 'Thu' in str(gameday):
        game_type = 'TNF'
        is_primetime = True
    # Sunday Night Football (typically 8:20 PM ET)
    elif pd.notna(gametime) and '20:' in str(gametime):
        game_type = 'SNF'
        is_primetime = True
    # Monday Night Football
    elif pd.notna(gameday) and 'Mon' in str(gameday):
        game_type = 'MNF'
        is_primetime = True

    return {
        'is_divisional': is_divisional,
        'game_type': game_type,
        'is_primetime': is_primetime
    }


def calculate_injury_adjustments(
    injury_model: InjuryImpactModel,
    season: int,
    week: int,
    team: str,
    team_epa: dict
) -> dict:
    """
    Calculate injury-adjusted EPA impacts.

    Injury impact is scaled based on team's baseline EPA:
    - Good teams (high EPA) hurt more from injuries
    - Bad teams (low EPA) hurt less from injuries
    """
    try:
        injury_impact = injury_model.compute_injury_impact(season, week, team)

        raw_off_adj = injury_impact.total_impact_offensive_epa
        raw_def_adj = injury_impact.total_impact_defensive_epa

        # Scale injury impact based on team quality
        # Good offense losing players hurts more
        if team_epa['offensive_epa'] > 0:
            effective_off_adj = raw_off_adj * (1.0 + abs(team_epa['offensive_epa']) * 5)
        else:
            effective_off_adj = raw_off_adj * (1.0 - abs(team_epa['offensive_epa']) * 2)

        # Good defense losing players hurts more (negative EPA = good defense)
        if team_epa['defensive_epa'] < 0:
            effective_def_adj = raw_def_adj * (1.0 + abs(team_epa['defensive_epa']) * 5)
        else:
            effective_def_adj = raw_def_adj * (1.0 - abs(team_epa['defensive_epa']) * 2)

        # Cap to reasonable bounds
        effective_off_adj = max(-0.20, min(0.0, effective_off_adj))
        effective_def_adj = max(-0.15, min(0.0, effective_def_adj))

        return {
            'offensive_adjustment': effective_off_adj,
            'defensive_adjustment': effective_def_adj,
            'raw_offensive': raw_off_adj,
            'raw_defensive': raw_def_adj
        }

    except Exception as e:
        logger.warning(f"Could not compute injuries for {team}: {e}")
        return {
            'offensive_adjustment': 0.0,
            'defensive_adjustment': 0.0,
            'raw_offensive': 0.0,
            'raw_defensive': 0.0
        }


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
    Generate all game line predictions for the week using FULL MonteCarloSimulator.

    ENHANCED VERSION - Uses all available features:
    - Team EPA (offensive, defensive, pass, rush) from PBP data
    - Team pace
    - Injury adjustments (scaled by team quality)
    - Weather conditions (temperature, wind, dome)
    - Game context (primetime, divisional)
    - Elo ratings as anchor (blended with EPA predictions)

    This ensures the narrative in the dashboard reflects actual prediction inputs.
    """
    if season is None:
        season = get_current_season()

    logger.info("=" * 70)
    logger.info(f"GENERATING GAME LINE PREDICTIONS - WEEK {week} (Full Simulator)")
    logger.info("=" * 70)

    # ==========================================================================
    # LOAD ALL DATA SOURCES
    # ==========================================================================

    # Load player predictions (still used for reference)
    predictions = load_player_predictions(week)

    # Load play-by-play data for EPA calculation
    pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{season}.parquet'
    if pbp_path.exists():
        pbp_df = pd.read_parquet(pbp_path)
        logger.info(f"Loaded {len(pbp_df)} plays for EPA calculation")
    else:
        logger.warning(f"PBP data not found: {pbp_path}")
        pbp_df = None

    # Load schedules
    schedules = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet')
    week_schedule = schedules[
        (schedules['season'] == season) &
        (schedules['week'] == week)
    ]

    # Load weather data
    weather_data = load_weather_data(season, week)
    logger.info(f"Loaded weather data for {len(weather_data)} games")

    # Initialize Elo calculator
    elo_calc = EnhancedEloCalculator()
    logger.info("Initialized Elo calculator for team strength anchor")

    # Initialize injury model
    injury_config_path = PROJECT_ROOT / "configs" / "injury_multipliers.yaml"
    try:
        injury_model = InjuryImpactModel(str(injury_config_path))
        logger.info("Loaded injury impact model")
    except Exception as e:
        logger.warning(f"Could not load injury model: {e}")
        injury_model = None

    # Initialize Monte Carlo simulator
    simulator = MonteCarloSimulator(seed=42)
    logger.info("Initialized MonteCarloSimulator")

    # Initialize weather adjuster with research-backed impacts
    weather_adjuster = WeatherAdjusterV2()
    logger.info("Initialized WeatherAdjusterV2 (research-backed wind/temp/precip impacts)")

    # Load injury data for QB-specific adjustments
    injury_df = load_injury_data(week)
    logger.info(f"Loaded {len(injury_df)} injury records for QB analysis")

    # Load official depth charts for starter identification
    depth_charts = load_depth_charts(season)
    if len(depth_charts) > 0:
        logger.info(f"Loaded depth charts for {depth_charts['team'].nunique()} teams")
    else:
        logger.warning("No depth charts loaded - will use PBP-based starter detection")

    # Get games from predictions
    games = get_games_from_predictions(predictions, week=week)
    logger.info(f"Found {len(games)} games to process")

    # ==========================================================================
    # PROCESS EACH GAME
    # ==========================================================================

    results = []

    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        logger.info(f"\nProcessing {game['game']}...")

        # Get game row from schedule
        game_row = week_schedule[
            (week_schedule['home_team'] == home_team) &
            (week_schedule['away_team'] == away_team)
        ]

        if len(game_row) == 0:
            logger.warning(f"  No schedule entry found, skipping")
            continue

        game_row = game_row.iloc[0]
        game_id = game_row['game_id']

        # ------------------------------------------------------------------
        # 1. CALCULATE TEAM EPA FROM PBP DATA
        # ------------------------------------------------------------------
        if pbp_df is not None:
            home_epa = calculate_team_epa(pbp_df, home_team, week)
            away_epa = calculate_team_epa(pbp_df, away_team, week)
            logger.info(f"  {home_team} EPA: Off={home_epa['offensive_epa']:+.3f}, Def={home_epa['defensive_epa']:+.3f}, Pace={home_epa['pace']:.0f}")
            logger.info(f"  {away_team} EPA: Off={away_epa['offensive_epa']:+.3f}, Def={away_epa['defensive_epa']:+.3f}, Pace={away_epa['pace']:.0f}")

            # ------------------------------------------------------------------
            # 1a. DEPTH CHART STARTER IDENTIFICATION
            # ------------------------------------------------------------------
            home_dc_info = get_team_starters(depth_charts, home_team) if len(depth_charts) > 0 else {}
            away_dc_info = get_team_starters(depth_charts, away_team) if len(depth_charts) > 0 else {}

            # Check for injured starters and identify backups
            depth_chart_adjustments = {'home': {}, 'away': {}}
            for team_key, dc_info, team_name in [('home', home_dc_info, home_team), ('away', away_dc_info, away_team)]:
                if dc_info:
                    for pos in ['QB', 'RB', 'TE']:
                        active_player, status, is_backup = get_active_starter(dc_info, pos, injury_df, team_name)
                        if is_backup and status == 'backup_due_to_injury':
                            depth_chart_adjustments[team_key][pos] = {
                                'original_starter': dc_info[pos].get('starter'),
                                'active_player': active_player,
                                'is_backup': True
                            }

            # ------------------------------------------------------------------
            # 1b. ALL SKILL POSITION EPA ADJUSTMENTS (QB, RB, WR, TE)
            # ------------------------------------------------------------------
            home_skill = calculate_all_skill_position_epa(pbp_df, home_team, week, injury_df)
            away_skill = calculate_all_skill_position_epa(pbp_df, away_team, week, injury_df)

            # Extract QB info for backward compatibility
            home_qb = home_skill['qb_info']
            away_qb = away_skill['qb_info']

            # Log key player info
            logger.info(f"  {home_team} QB: {home_qb['current_qb']} (EPA: {home_qb['current_qb_epa']:+.3f})")
            logger.info(f"  {away_team} QB: {away_qb['current_qb']} (EPA: {away_qb['current_qb_epa']:+.3f})")

            # Log RB info
            home_rb = home_skill['rb_info']
            away_rb = away_skill['rb_info']
            logger.info(f"  {home_team} RB1: {home_rb['primary_rb']} (EPA: {home_rb['primary_rb_epa']:+.3f}, {home_rb['primary_rb_carries']} carries)")
            logger.info(f"  {away_team} RB1: {away_rb['primary_rb']} (EPA: {away_rb['primary_rb_epa']:+.3f}, {away_rb['primary_rb_carries']} carries)")

            # Log top receivers
            home_rec = home_skill['receiver_info']
            away_rec = away_skill['receiver_info']
            if len(home_rec['top_receivers']) > 0:
                top_wr = home_rec['top_receivers'][0]
                logger.info(f"  {home_team} WR1: {top_wr['name']} (EPA: {top_wr['epa']:+.3f}, {top_wr['target_share']:.0%} target share)")
            if len(away_rec['top_receivers']) > 0:
                top_wr = away_rec['top_receivers'][0]
                logger.info(f"  {away_team} WR1: {top_wr['name']} (EPA: {top_wr['epa']:+.3f}, {top_wr['target_share']:.0%} target share)")

            # Log depth chart verification / discrepancies
            for team_key, dc_info, team_name, skill_info in [
                ('home', home_dc_info, home_team, home_skill),
                ('away', away_dc_info, away_team, away_skill)
            ]:
                if dc_info and 'QB' in dc_info:
                    dc_qb = dc_info['QB'].get('starter', '')
                    pbp_qb = skill_info['qb_info']['current_qb']
                    # Check if depth chart starter differs from PBP-detected QB
                    if dc_qb and pbp_qb and dc_qb.split()[-1].lower() != pbp_qb.split('.')[-1].lower():
                        logger.info(f"  📋 {team_name} Depth Chart QB: {dc_qb} (PBP shows: {pbp_qb})")

            # Log any depth chart-detected backup situations
            for team_key, team_name in [('home', home_team), ('away', away_team)]:
                if depth_chart_adjustments[team_key]:
                    for pos, adj in depth_chart_adjustments[team_key].items():
                        if adj.get('is_backup'):
                            logger.warning(f"  📋 {team_name} {pos}: Starter {adj['original_starter']} → Backup {adj['active_player']} (per depth chart + injury)")

            # Apply ALL skill position EPA adjustments
            if abs(home_skill['total_offensive_epa_adjustment']) > 0.001:
                logger.warning(f"  ⚠️ {home_team} SKILL POSITION ADJUSTMENTS:")
                if home_qb['qb_change_detected']:
                    logger.warning(f"     QB: {home_qb['previous_qb']} → {home_qb['current_qb']} ({home_qb['pass_epa_adjustment']:+.3f})")
                for rb_inj in home_rb['injured_rbs']:
                    logger.warning(f"     RB: {rb_inj['name']} OUT ({rb_inj['status']})")
                for rec_inj in home_rec['injured_receivers']:
                    logger.warning(f"     {rec_inj['position']}: {rec_inj['name']} OUT ({rec_inj['status']})")
                logger.warning(f"     Total EPA adjustment: {home_skill['total_offensive_epa_adjustment']:+.3f}")

                # Apply adjustments to team EPA
                home_epa['pass_epa'] += home_skill['total_pass_epa_adjustment']
                home_epa['rush_epa'] += home_skill['total_rush_epa_adjustment']
                home_epa['offensive_epa'] += home_skill['total_offensive_epa_adjustment']

            if abs(away_skill['total_offensive_epa_adjustment']) > 0.001:
                logger.warning(f"  ⚠️ {away_team} SKILL POSITION ADJUSTMENTS:")
                if away_qb['qb_change_detected']:
                    logger.warning(f"     QB: {away_qb['previous_qb']} → {away_qb['current_qb']} ({away_qb['pass_epa_adjustment']:+.3f})")
                for rb_inj in away_rb['injured_rbs']:
                    logger.warning(f"     RB: {rb_inj['name']} OUT ({rb_inj['status']})")
                for rec_inj in away_rec['injured_receivers']:
                    logger.warning(f"     {rec_inj['position']}: {rec_inj['name']} OUT ({rec_inj['status']})")
                logger.warning(f"     Total EPA adjustment: {away_skill['total_offensive_epa_adjustment']:+.3f}")

                # Apply adjustments to team EPA
                away_epa['pass_epa'] += away_skill['total_pass_epa_adjustment']
                away_epa['rush_epa'] += away_skill['total_rush_epa_adjustment']
                away_epa['offensive_epa'] += away_skill['total_offensive_epa_adjustment']

            # Log and apply defensive EPA adjustments
            home_def = home_skill.get('defense_info', {})
            away_def = away_skill.get('defense_info', {})

            # Log defensive metrics
            if home_def.get('pressure_rate', 0) > 0 or away_def.get('pressure_rate', 0) > 0:
                logger.info(f"  {home_team} Defense: Pressure={home_def.get('pressure_rate', 0):.1%}, Coverage EPA={home_def.get('coverage_epa', 0):+.3f}")
                logger.info(f"  {away_team} Defense: Pressure={away_def.get('pressure_rate', 0):.1%}, Coverage EPA={away_def.get('coverage_epa', 0):+.3f}")

            # Apply defensive adjustments (injured defenders)
            if abs(home_skill.get('total_defensive_epa_adjustment', 0)) > 0.001:
                logger.warning(f"  ⚠️ {home_team} DEFENSIVE INJURIES:")
                for defender in home_def.get('injured_defenders', []):
                    logger.warning(f"     {defender['position']}: {defender['name']} OUT ({defender['status']})")
                logger.warning(f"     Total DEF EPA adjustment: +{home_skill['total_defensive_epa_adjustment']:+.3f}")
                home_epa['defensive_epa'] += home_skill['total_defensive_epa_adjustment']

            if abs(away_skill.get('total_defensive_epa_adjustment', 0)) > 0.001:
                logger.warning(f"  ⚠️ {away_team} DEFENSIVE INJURIES:")
                for defender in away_def.get('injured_defenders', []):
                    logger.warning(f"     {defender['position']}: {defender['name']} OUT ({defender['status']})")
                logger.warning(f"     Total DEF EPA adjustment: +{away_skill['total_defensive_epa_adjustment']:+.3f}")
                away_epa['defensive_epa'] += away_skill['total_defensive_epa_adjustment']

        else:
            home_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pass_epa': 0.0, 'rush_epa': 0.0, 'pace': 65.0, 'sample_games': 0, 'red_zone_epa': 0.0}
            away_epa = {'offensive_epa': 0.0, 'defensive_epa': 0.0, 'pass_epa': 0.0, 'rush_epa': 0.0, 'pace': 65.0, 'sample_games': 0, 'red_zone_epa': 0.0}
            home_qb = {'current_qb': 'Unknown', 'current_qb_epa': 0.0, 'qb_change_detected': False, 'pass_epa_adjustment': 0.0}
            away_qb = {'current_qb': 'Unknown', 'current_qb_epa': 0.0, 'qb_change_detected': False, 'pass_epa_adjustment': 0.0}
            home_skill = {'rb_info': {'primary_rb': 'Unknown', 'injured_rbs': [], 'rb_depth': []},
                         'receiver_info': {'top_receivers': [], 'injured_receivers': []},
                         'defense_info': {'pressure_rate': 0.0, 'coverage_epa': 0.0, 'run_def_epa': 0.0, 'injured_defenders': []},
                         'key_injuries': [], 'total_defensive_epa_adjustment': 0.0}
            away_skill = {'rb_info': {'primary_rb': 'Unknown', 'injured_rbs': [], 'rb_depth': []},
                         'receiver_info': {'top_receivers': [], 'injured_receivers': []},
                         'defense_info': {'pressure_rate': 0.0, 'coverage_epa': 0.0, 'run_def_epa': 0.0, 'injured_defenders': []},
                         'key_injuries': [], 'total_defensive_epa_adjustment': 0.0}

        # ------------------------------------------------------------------
        # 2. CALCULATE INJURY ADJUSTMENTS
        # ------------------------------------------------------------------
        if injury_model is not None:
            home_injuries = calculate_injury_adjustments(injury_model, season, week, home_team, home_epa)
            away_injuries = calculate_injury_adjustments(injury_model, season, week, away_team, away_epa)

            if abs(home_injuries['offensive_adjustment']) > 0.02 or abs(home_injuries['defensive_adjustment']) > 0.02:
                logger.info(f"  {home_team} injuries: Off={home_injuries['offensive_adjustment']:+.3f}, Def={home_injuries['defensive_adjustment']:+.3f}")
            if abs(away_injuries['offensive_adjustment']) > 0.02 or abs(away_injuries['defensive_adjustment']) > 0.02:
                logger.info(f"  {away_team} injuries: Off={away_injuries['offensive_adjustment']:+.3f}, Def={away_injuries['defensive_adjustment']:+.3f}")
        else:
            home_injuries = {'offensive_adjustment': 0.0, 'defensive_adjustment': 0.0}
            away_injuries = {'offensive_adjustment': 0.0, 'defensive_adjustment': 0.0}

        # ------------------------------------------------------------------
        # 3. GET WEATHER DATA & CALCULATE RESEARCH-BACKED ADJUSTMENTS
        # ------------------------------------------------------------------
        weather = weather_data.get(game_id, {
            'is_dome': False,
            'temperature': None,
            'wind_speed': None,
            'precipitation': None
        })

        # Calculate weather adjustments using research-backed model
        weather_adj = weather_adjuster.calculate_weather_adjustments(
            team=home_team,
            wind_mph=weather.get('wind_speed', 0) or 0,
            temp_f=weather.get('temperature', 65) or 65,
            precip_prob=weather.get('precip_chance', 0) or 0,
            precip_type=weather.get('precip_type'),
            is_dome=weather.get('is_dome', False)
        )

        # Apply weather adjustments to EPA values
        # Passing EPA is affected by wind, temp, and precipitation
        # Defensive EPA adjustment is inverse (good for defense in bad weather)
        passing_mult = weather_adj['passing_epa_multiplier']
        rush_boost = weather_adj['rush_rate_boost']

        # Adjust offensive EPA (passing hurt, rushing helped)
        # Offensive EPA is ~60% passing, 40% rushing weighted
        weather_off_multiplier = (0.6 * passing_mult) + (0.4 * (1.0 + rush_boost))

        # Store original EPA for logging BEFORE applying adjustments
        home_epa_original = home_epa['offensive_epa']
        away_epa_original = away_epa['offensive_epa']

        # Apply weather adjustments to both teams' offensive EPA
        home_epa_weather_adj = home_epa_original * weather_off_multiplier
        away_epa_weather_adj = away_epa_original * weather_off_multiplier

        # Update EPA dict with weather-adjusted values
        home_epa['offensive_epa'] = home_epa_weather_adj
        away_epa['offensive_epa'] = away_epa_weather_adj

        # Log weather info with impact
        if weather.get('temperature') is not None or weather.get('is_dome'):
            if weather.get('is_dome'):
                dome_str = "DOME"
            else:
                dome_str = f"{weather.get('temperature', '?')}°F"

            wind_str = f", {weather.get('wind_speed', 0):.0f} mph wind" if weather.get('wind_speed') else ""
            conditions_str = f" ({weather.get('conditions', '')})" if weather.get('conditions') else ""
            precip_str = f", {weather.get('precip_chance', 0):.0%} precip" if weather.get('precip_chance', 0) > 0 else ""

            # Weather bucket info
            bucket_str = f" [Wind: {weather_adj['wind_bucket']}, Temp: {weather_adj['temp_bucket']}]"

            logger.info(f"  Weather: {dome_str}{wind_str}{precip_str}{conditions_str}{bucket_str}")

            # Log weather impact if significant
            if abs(passing_mult - 1.0) > 0.01 or abs(rush_boost) > 0.01:
                logger.info(f"  Weather Impact: Pass EPA {passing_mult:.0%}, Rush boost {rush_boost:+.0%}, Net EPA mult {weather_off_multiplier:.2f}")
                if home_epa_original != 0:
                    logger.info(f"    {home_team} Off EPA: {home_epa_original:+.3f} → {home_epa_weather_adj:+.3f}")
                if away_epa_original != 0:
                    logger.info(f"    {away_team} Off EPA: {away_epa_original:+.3f} → {away_epa_weather_adj:+.3f}")

        # ------------------------------------------------------------------
        # 4. GET GAME CONTEXT
        # ------------------------------------------------------------------
        context = get_game_context(game_row, schedules)
        if context['is_divisional'] or context['is_primetime']:
            logger.info(f"  Context: {'Divisional ' if context['is_divisional'] else ''}{context['game_type']}")

        # ------------------------------------------------------------------
        # 5. GET ELO FEATURES
        # ------------------------------------------------------------------
        elo_raw = elo_calc.get_team_features(home_team, away_team, season, week, is_home=True)
        elo_features = {
            'home_elo': elo_raw['team_elo'],
            'away_elo': elo_raw['opp_elo'],
            'elo_diff': elo_raw['elo_diff'],
            'home_win_prob': elo_raw['win_probability'],
            'elo_spread': elo_raw['expected_spread'],
        }
        logger.info(f"  Elo: {home_team}={elo_features['home_elo']:.0f}, {away_team}={elo_features['away_elo']:.0f}, Spread={elo_features['elo_spread']:+.1f}")

        # ------------------------------------------------------------------
        # 6. RUN FULL MONTE CARLO SIMULATION
        # ------------------------------------------------------------------
        sim_input = SimulationInput(
            game_id=game_id,
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
            # EPA features
            home_offensive_epa=home_epa['offensive_epa'],
            home_defensive_epa=home_epa['defensive_epa'],
            away_offensive_epa=away_epa['offensive_epa'],
            away_defensive_epa=away_epa['defensive_epa'],
            # Pace
            home_pace=home_epa['pace'],
            away_pace=away_epa['pace'],
            # Injury adjustments
            home_injury_offensive_adjustment=home_injuries['offensive_adjustment'],
            home_injury_defensive_adjustment=home_injuries['defensive_adjustment'],
            away_injury_offensive_adjustment=away_injuries['offensive_adjustment'],
            away_injury_defensive_adjustment=away_injuries['defensive_adjustment'],
            # Game context
            is_divisional=context['is_divisional'],
            game_type=context['game_type'],
            # Weather
            is_dome=weather.get('is_dome', False),
            temperature=weather.get('temperature'),
            wind_speed=weather.get('wind_speed'),
            precipitation=weather.get('precip_chance', 0) if weather.get('precip_chance') else None,
        )

        # Run simulation with 50,000 trials
        sim_output = simulator.simulate_game(sim_input, trials=50000)

        logger.info(f"  Simulation: Total={sim_output.fair_total:.1f}, Spread={sim_output.fair_spread:+.1f}, Home Win={sim_output.home_win_prob:.1%}")

        # ------------------------------------------------------------------
        # 7. BLEND EPA SIMULATION WITH ELO ANCHOR
        # ------------------------------------------------------------------
        # Elo provides stability, EPA provides context
        # 60% Elo, 40% EPA simulation for spread (Elo is more reliable for team strength)
        # 100% EPA simulation for totals (Elo doesn't predict scoring well)

        blended_spread = 0.6 * elo_features['elo_spread'] + 0.4 * sim_output.fair_spread
        blended_win_prob = 0.6 * elo_features['home_win_prob'] + 0.4 * sim_output.home_win_prob

        # Totals come purely from EPA simulation (more responsive to context)
        blended_total = sim_output.fair_total

        logger.info(f"  Blended: Total={blended_total:.1f}, Spread={blended_spread:+.1f}, Win={blended_win_prob:.1%}")

        # ------------------------------------------------------------------
        # 8. VEGAS SANITY CHECKS
        # ------------------------------------------------------------------
        vegas_spread = game_row.get('spread_line')
        vegas_total = game_row.get('total_line')

        spread_diff = None
        total_diff = None
        spread_sane = True
        total_sane = True

        if vegas_spread is not None and not pd.isna(vegas_spread):
            spread_diff = abs(blended_spread - vegas_spread)
            spread_sane = spread_diff <= 7.0
            if not spread_sane:
                logger.warning(f"  ⚠️ Spread differs from Vegas by {spread_diff:.1f} pts (Model: {blended_spread:+.1f}, Vegas: {vegas_spread:+.1f})")

        if vegas_total is not None and not pd.isna(vegas_total):
            total_diff = abs(blended_total - vegas_total)
            total_sane = total_diff <= 7.0
            if not total_sane:
                logger.warning(f"  ⚠️ Total differs from Vegas by {total_diff:.1f} pts (Model: {blended_total:.1f}, Vegas: {vegas_total:.1f})")

        # ------------------------------------------------------------------
        # 9. BUILD RESULT
        # ------------------------------------------------------------------
        result = {
            'game': game['game'],
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'week': week,

            # Blended predictions (what we recommend)
            'projected_total': blended_total,
            'projected_spread': blended_spread,
            'home_win_prob': blended_win_prob,
            'away_win_prob': 1 - blended_win_prob,

            # Raw simulation output
            'sim_total': sim_output.fair_total,
            'sim_spread': sim_output.fair_spread,
            'sim_home_win_prob': sim_output.home_win_prob,

            # Distribution stats
            'total_std': sim_output.total_std,
            'total_p5': sim_output.total_p5,
            'total_p25': sim_output.total_p25,
            'total_p50': sim_output.total_p50,
            'total_p75': sim_output.total_p75,
            'total_p95': sim_output.total_p95,

            # Team projected points
            'home_projected_points': sim_output.home_score_median,
            'away_projected_points': sim_output.away_score_median,

            # Elo features
            'home_elo': elo_features['home_elo'],
            'away_elo': elo_features['away_elo'],
            'elo_spread': elo_features['elo_spread'],
            'elo_win_prob': elo_features['home_win_prob'],

            # EPA features (for dashboard narrative)
            'home_epa': home_epa['offensive_epa'],
            'away_epa': away_epa['offensive_epa'],
            'home_def_epa': home_epa['defensive_epa'],
            'away_def_epa': away_epa['defensive_epa'],
            'home_pass_epa': home_epa['pass_epa'],
            'away_pass_epa': away_epa['pass_epa'],
            'home_rush_epa': home_epa['rush_epa'],
            'away_rush_epa': away_epa['rush_epa'],
            'home_pace': home_epa['pace'],
            'away_pace': away_epa['pace'],

            # Injury adjustments
            'home_injury_off_adj': home_injuries['offensive_adjustment'],
            'home_injury_def_adj': home_injuries['defensive_adjustment'],
            'away_injury_off_adj': away_injuries['offensive_adjustment'],
            'away_injury_def_adj': away_injuries['defensive_adjustment'],

            # Game context
            'is_divisional': context['is_divisional'],
            'game_type': context['game_type'],
            'is_primetime': context['is_primetime'],

            # Weather - raw data
            'is_dome': weather.get('is_dome', False),
            'temperature': weather.get('temperature'),
            'wind_speed': weather.get('wind_speed'),
            'precip_chance': weather.get('precip_chance', 0),
            'precip_type': weather.get('precip_type'),
            'conditions': weather.get('conditions', ''),

            # Weather - model adjustments
            'wind_bucket': weather_adj.get('wind_bucket', 'calm'),
            'temp_bucket': weather_adj.get('temp_bucket', 'comfortable'),
            'weather_pass_mult': weather_adj.get('passing_epa_multiplier', 1.0),
            'weather_rush_boost': weather_adj.get('rush_rate_boost', 0.0),
            'weather_off_mult': weather_off_multiplier,

            # Depth chart - projected starters
            'dc_home_qb': home_dc_info.get('QB', {}).get('starter') if home_dc_info else None,
            'dc_home_rb': home_dc_info.get('RB', {}).get('starter') if home_dc_info else None,
            'dc_home_te': home_dc_info.get('TE', {}).get('starter') if home_dc_info else None,
            'dc_away_qb': away_dc_info.get('QB', {}).get('starter') if away_dc_info else None,
            'dc_away_rb': away_dc_info.get('RB', {}).get('starter') if away_dc_info else None,
            'dc_away_te': away_dc_info.get('TE', {}).get('starter') if away_dc_info else None,

            # Depth chart - backup active flags
            'home_qb_backup_active': depth_chart_adjustments['home'].get('QB', {}).get('is_backup', False),
            'home_rb_backup_active': depth_chart_adjustments['home'].get('RB', {}).get('is_backup', False),
            'home_te_backup_active': depth_chart_adjustments['home'].get('TE', {}).get('is_backup', False),
            'away_qb_backup_active': depth_chart_adjustments['away'].get('QB', {}).get('is_backup', False),
            'away_rb_backup_active': depth_chart_adjustments['away'].get('RB', {}).get('is_backup', False),
            'away_te_backup_active': depth_chart_adjustments['away'].get('TE', {}).get('is_backup', False),

            # QB-specific info
            'home_qb': home_qb['current_qb'],
            'home_qb_epa': home_qb['current_qb_epa'],
            'home_qb_change': home_qb['qb_change_detected'],
            'home_qb_previous': home_qb.get('previous_qb'),
            'away_qb': away_qb['current_qb'],
            'away_qb_epa': away_qb['current_qb_epa'],
            'away_qb_change': away_qb['qb_change_detected'],
            'away_qb_previous': away_qb.get('previous_qb'),

            # RB-specific info
            'home_rb1': home_skill['rb_info'].get('primary_rb', 'Unknown'),
            'home_rb1_epa': home_skill['rb_info'].get('primary_rb_epa', 0.0),
            'home_rb1_carries': home_skill['rb_info'].get('primary_rb_carries', 0),
            'away_rb1': away_skill['rb_info'].get('primary_rb', 'Unknown'),
            'away_rb1_epa': away_skill['rb_info'].get('primary_rb_epa', 0.0),
            'away_rb1_carries': away_skill['rb_info'].get('primary_rb_carries', 0),

            # WR-specific info (top receiver)
            'home_wr1': home_skill['receiver_info']['top_receivers'][0]['name'] if len(home_skill['receiver_info'].get('top_receivers', [])) > 0 else 'Unknown',
            'home_wr1_epa': home_skill['receiver_info']['top_receivers'][0]['epa'] if len(home_skill['receiver_info'].get('top_receivers', [])) > 0 else 0.0,
            'home_wr1_target_share': home_skill['receiver_info']['top_receivers'][0]['target_share'] if len(home_skill['receiver_info'].get('top_receivers', [])) > 0 else 0.0,
            'away_wr1': away_skill['receiver_info']['top_receivers'][0]['name'] if len(away_skill['receiver_info'].get('top_receivers', [])) > 0 else 'Unknown',
            'away_wr1_epa': away_skill['receiver_info']['top_receivers'][0]['epa'] if len(away_skill['receiver_info'].get('top_receivers', [])) > 0 else 0.0,
            'away_wr1_target_share': away_skill['receiver_info']['top_receivers'][0]['target_share'] if len(away_skill['receiver_info'].get('top_receivers', [])) > 0 else 0.0,

            # Defensive position info
            'home_pressure_rate': home_skill.get('defense_info', {}).get('pressure_rate', 0.0),
            'home_coverage_epa': home_skill.get('defense_info', {}).get('coverage_epa', 0.0),
            'home_run_def_epa': home_skill.get('defense_info', {}).get('run_def_epa', 0.0),
            'away_pressure_rate': away_skill.get('defense_info', {}).get('pressure_rate', 0.0),
            'away_coverage_epa': away_skill.get('defense_info', {}).get('coverage_epa', 0.0),
            'away_run_def_epa': away_skill.get('defense_info', {}).get('run_def_epa', 0.0),

            # Red zone efficiency
            'home_red_zone_epa': home_epa.get('red_zone_epa', 0.0),
            'away_red_zone_epa': away_epa.get('red_zone_epa', 0.0),

            # Key injuries summary
            'home_key_injuries': len(home_skill.get('key_injuries', [])),
            'away_key_injuries': len(away_skill.get('key_injuries', [])),
            'home_off_injuries': len([i for i in home_skill.get('key_injuries', []) if i.get('side') == 'offense']),
            'away_off_injuries': len([i for i in away_skill.get('key_injuries', []) if i.get('side') == 'offense']),
            'home_def_injuries': len([i for i in home_skill.get('key_injuries', []) if i.get('side') == 'defense']),
            'away_def_injuries': len([i for i in away_skill.get('key_injuries', []) if i.get('side') == 'defense']),

            # Vegas comparison
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
            'spread_diff_vs_vegas': spread_diff,
            'total_diff_vs_vegas': total_diff,
            'spread_sane': spread_sane,
            'total_sane': total_sane,
        }

        results.append(result)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    df = pd.DataFrame(results)

    if len(df) > 0:
        sane_spreads = df['spread_sane'].sum()
        sane_totals = df['total_sane'].sum()
        qb_changes = df['home_qb_change'].sum() + df['away_qb_change'].sum()
        total_key_injuries = df['home_key_injuries'].sum() + df['away_key_injuries'].sum()
        off_injuries = df['home_off_injuries'].sum() + df['away_off_injuries'].sum()
        def_injuries = df['home_def_injuries'].sum() + df['away_def_injuries'].sum()

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {len(df)} games processed")
        logger.info(f"Sanity check: {sane_spreads}/{len(df)} spreads, {sane_totals}/{len(df)} totals within 7 pts of Vegas")
        logger.info(f"QB changes detected: {qb_changes}")
        logger.info(f"Key injuries tracked: {total_key_injuries} total ({off_injuries} offensive, {def_injuries} defensive)")
        logger.info(f"Features used:")
        logger.info(f"  - Team EPA (offensive, defensive, pass, rush)")
        logger.info(f"  - Position-specific EPA (QB, RB, WR, TE)")
        logger.info(f"  - Defensive metrics (pressure rate, coverage EPA, run defense)")
        logger.info(f"  - Red zone EPA, home/away splits")
        logger.info(f"  - Garbage time filtering")
        logger.info(f"  - Injury adjustments (offense + defense)")
        logger.info(f"  - Weather, Game Context, Elo")
        logger.info(f"{'='*70}")

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
