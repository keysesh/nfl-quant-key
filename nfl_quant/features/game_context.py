"""
Game context features including team implied total, injury returns, etc.

This module provides features derived from game-level context that affect
player performance but aren't captured in individual player stats.

Key Features:
- team_implied_total: Vegas implied points for the team
- games_since_return: Games played since returning from extended absence
- first_game_back: Binary flag for first game after injury
- days_rest: Days since last game played
"""
from typing import Optional, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_team_implied_total(
    game_total: float,
    spread: float
) -> float:
    """
    Calculate team's implied point total from game total and spread.

    Formula:
        team_implied = game_total/2 + abs(spread)/2 (if favorite)
        team_implied = game_total/2 - abs(spread)/2 (if underdog)

    Args:
        game_total: Vegas over/under for the game
        spread: Vegas spread (negative = favorite, positive = underdog)

    Returns:
        Team's implied point total

    Example:
        >>> calculate_team_implied_total(48.5, -3.5)  # Team is 3.5 pt favorite
        26.0
        >>> calculate_team_implied_total(48.5, 3.5)   # Team is 3.5 pt underdog
        22.5
    """
    if pd.isna(game_total) or pd.isna(spread):
        return 22.0  # League average team score

    half_total = game_total / 2
    half_spread = abs(spread) / 2

    if spread < 0:  # Team is favorite
        return half_total + half_spread
    else:  # Team is underdog
        return half_total - half_spread


def calculate_opponent_implied_total(
    game_total: float,
    spread: float
) -> float:
    """
    Calculate opponent's implied point total from game total and spread.

    Args:
        game_total: Vegas over/under for the game
        spread: Vegas spread from team's perspective

    Returns:
        Opponent's implied point total
    """
    team_implied = calculate_team_implied_total(game_total, spread)
    if pd.isna(game_total):
        return 22.0
    return game_total - team_implied


def calculate_games_since_return(
    player_id: str,
    current_week: int,
    current_season: int,
    player_games: pd.DataFrame,
    min_gap: int = 3
) -> int:
    """
    Calculate games played since returning from extended absence.

    An "extended absence" is defined as missing min_gap-1 or more consecutive
    games. This helps identify "rust" effects in first games back from injury.

    Args:
        player_id: Player's unique ID
        current_week: Week being predicted
        current_season: Season year
        player_games: DataFrame with player_id, season, week columns
        min_gap: Minimum gap in weeks to count as absence (default 3 = missed 2+ games)

    Returns:
        0 if no recent absence
        1 for first game back
        2 for second game back
        etc., capped at 4

    Example:
        Player played weeks 1, 2, 3, then missed weeks 4, 5, played week 6:
        - For week 7 prediction: returns 2 (second game back)
    """
    player_weeks = player_games[
        (player_games['player_id'] == player_id) &
        (player_games['season'] == current_season) &
        (player_games['week'] < current_week)
    ].sort_values('week')

    if len(player_weeks) < 2:
        return 0

    weeks_played = player_weeks['week'].values

    # Find most recent gap of min_gap+ weeks
    for i in range(len(weeks_played) - 1, 0, -1):
        gap = weeks_played[i] - weeks_played[i-1]
        if gap >= min_gap:
            # Found a gap - count games played after the gap
            first_game_after_gap = weeks_played[i]
            games_after_return = (weeks_played >= first_game_after_gap).sum()

            # This prediction is for the next game
            return min(games_after_return + 1, 4)

    return 0


def calculate_first_game_back(games_since_return: int) -> int:
    """
    Binary flag for first game back from injury.

    Args:
        games_since_return: Output from calculate_games_since_return()

    Returns:
        1 if first game back, 0 otherwise
    """
    return int(games_since_return == 1)


def calculate_days_rest(
    current_game_date: pd.Timestamp,
    previous_game_date: Optional[pd.Timestamp] = None
) -> int:
    """
    Calculate days of rest since last game.

    Returns 7 as default (standard week), handles Thursday/Monday games.

    Args:
        current_game_date: Date of the game being predicted
        previous_game_date: Date of the player's previous game

    Returns:
        Days between games, clamped to [3, 14]
    """
    if previous_game_date is None:
        return 7

    if pd.isna(previous_game_date) or pd.isna(current_game_date):
        return 7

    days = (current_game_date - previous_game_date).days
    return max(3, min(days, 14))  # Clamp to reasonable range


def calculate_short_rest_flag(days_rest: int, threshold: int = 5) -> int:
    """
    Binary flag for short rest (Thursday games, etc.).

    Args:
        days_rest: Days since last game
        threshold: Days below which is considered short rest (default 5)

    Returns:
        1 if short rest, 0 otherwise
    """
    return int(days_rest < threshold)


def add_game_context_features(
    df: pd.DataFrame,
    odds_df: pd.DataFrame = None,
    schedule_df: pd.DataFrame = None,
    player_games_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Add all game context features to a prediction DataFrame.

    Args:
        df: Main DataFrame with player_id, team, week, season columns
        odds_df: DataFrame with game_total, spread by game (optional)
        schedule_df: DataFrame with game dates (optional)
        player_games_df: DataFrame of games played by each player (optional)

    Returns:
        DataFrame with new context features added:
        - team_implied_total
        - opponent_implied_total
        - games_since_return
        - first_game_back
        - days_rest (if schedule provided)
        - short_rest_flag (if schedule provided)
    """
    df = df.copy()

    # Initialize with defaults
    df['team_implied_total'] = 22.0
    df['opponent_implied_total'] = 22.0
    df['games_since_return'] = 0
    df['first_game_back'] = 0
    df['days_rest'] = 7
    df['short_rest_flag'] = 0

    # Merge odds data if provided
    if odds_df is not None and len(odds_df) > 0:
        if 'game_total' in odds_df.columns and 'spread' in odds_df.columns:
            # Determine merge keys
            merge_keys = []
            if 'game_id' in df.columns and 'game_id' in odds_df.columns:
                merge_keys = ['game_id', 'team']
            elif 'week' in df.columns and 'season' in df.columns:
                if 'week' in odds_df.columns and 'season' in odds_df.columns:
                    if 'team' in odds_df.columns:
                        merge_keys = ['season', 'week', 'team']

            if merge_keys:
                odds_subset = odds_df[merge_keys + ['game_total', 'spread']].drop_duplicates()
                df = df.merge(
                    odds_subset,
                    on=merge_keys,
                    how='left',
                    suffixes=('', '_odds')
                )

                # Calculate implied totals
                df['team_implied_total'] = df.apply(
                    lambda row: calculate_team_implied_total(
                        row.get('game_total', np.nan),
                        row.get('spread', np.nan)
                    ),
                    axis=1
                )

                df['opponent_implied_total'] = df.apply(
                    lambda row: calculate_opponent_implied_total(
                        row.get('game_total', np.nan),
                        row.get('spread', np.nan)
                    ),
                    axis=1
                )

    # Calculate injury return features if player games data provided
    if player_games_df is not None and len(player_games_df) > 0:
        if 'player_id' in df.columns:
            df['games_since_return'] = df.apply(
                lambda row: calculate_games_since_return(
                    row.get('player_id', ''),
                    row.get('week', 1),
                    row.get('season', 2024),
                    player_games_df
                ),
                axis=1
            )

            df['first_game_back'] = df['games_since_return'].apply(
                calculate_first_game_back
            )

    return df


def get_game_script_expectation(
    team_implied_total: float,
    opponent_implied_total: float
) -> Dict[str, float]:
    """
    Calculate expected game script based on implied totals.

    Args:
        team_implied_total: Team's Vegas implied points
        opponent_implied_total: Opponent's Vegas implied points

    Returns:
        Dict with:
        - expected_lead: Positive = expected to lead
        - pass_lean: >0.5 = expected to pass more
        - run_lean: >0.5 = expected to run more
    """
    expected_lead = team_implied_total - opponent_implied_total

    # If trailing, teams pass more (positive correlation)
    # If leading, teams run more
    pass_lean = 0.5 - (expected_lead / 30.0)  # Normalize to ~0.4-0.6 range
    pass_lean = max(0.3, min(0.7, pass_lean))

    return {
        'expected_lead': expected_lead,
        'pass_lean': pass_lean,
        'run_lean': 1 - pass_lean,
    }


def calculate_opponent_target_share_allowed(
    opponent: str,
    position: str,
    season: int,
    max_week: int,
    pbp_df: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Calculate how many targets an opponent allows to a position group.

    This provides signal about whether a defense is exploitable at a position.

    Args:
        opponent: Opponent team abbreviation
        position: Position (WR, TE, RB)
        season: Season year
        max_week: Maximum week to include (for temporal safety)
        pbp_df: Pre-loaded PBP DataFrame (optional)

    Returns:
        Dict with target share metrics
    """
    from pathlib import Path

    if pbp_df is None:
        pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
        if not pbp_path.exists():
            return {'opp_target_share': 0.20, 'opp_targets_per_game': 5.0}
        pbp_df = pd.read_parquet(pbp_path)

    # Filter to games against this opponent and before max_week
    # Opponent was the defense, so we want plays where defteam == opponent
    pass_plays = pbp_df[
        (pbp_df['play_type'] == 'pass') &
        (pbp_df['defteam'] == opponent) &
        (pbp_df['week'] < max_week) &
        (pbp_df['receiver_player_name'].notna())
    ].copy()

    if len(pass_plays) == 0:
        return {'opp_target_share': 0.20, 'opp_targets_per_game': 5.0}

    # Map positions
    if position == 'WR':
        pos_targets = len(pass_plays[pass_plays['receiver_position'].isin(['WR'])])
    elif position == 'TE':
        pos_targets = len(pass_plays[pass_plays['receiver_position'].isin(['TE'])])
    elif position == 'RB':
        pos_targets = len(pass_plays[pass_plays['receiver_position'].isin(['RB', 'FB'])])
    else:
        pos_targets = len(pass_plays)

    total_targets = len(pass_plays)
    n_games = pass_plays['game_id'].nunique()

    target_share = pos_targets / total_targets if total_targets > 0 else 0.20
    targets_per_game = pos_targets / n_games if n_games > 0 else 5.0

    return {
        'opp_target_share': target_share,
        'opp_targets_per_game': targets_per_game,
        'opp_total_targets': total_targets,
        'opp_games_analyzed': n_games,
    }


def categorize_game_environment(game_total: float) -> str:
    """
    Categorize game environment based on total.

    Args:
        game_total: Vegas game total

    Returns:
        Category: 'low_scoring', 'average', 'high_scoring', 'shootout'
    """
    if game_total < 40:
        return 'low_scoring'
    elif game_total < 47:
        return 'average'
    elif game_total < 52:
        return 'high_scoring'
    else:
        return 'shootout'
