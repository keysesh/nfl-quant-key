"""
Matchup Context Module for Dashboard Display.

Aggregates defensive matchup data for player prop cards:
- Defense rank (#1-32) vs position
- Position-specific yards allowed (6-game average)
- Recent trend (improving/declining/stable)
- Coverage tendency (man vs zone %)

Uses existing cached data from defense_vs_position and coverage_tendencies.
All data is trailing (pre-game) to avoid leakage.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from .defense_vs_position import load_defense_vs_position_cache
from .coverage_tendencies import load_coverage_cache

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Module-level caches
_DEFENSE_RANKS_CACHE = None
_MATCHUP_CONTEXT_CACHE = {}


def get_matchup_context(
    opponent: str,
    week: int,
    season: int,
    position: str,
    pos_rank: int = 1,
    market: str = None
) -> Dict:
    """
    Get comprehensive matchup context for a player prop recommendation.

    Args:
        opponent: Opponent team abbreviation (e.g., 'WAS')
        week: Current week number
        season: Season year
        position: Player position (WR, RB, TE, QB)
        pos_rank: Position rank (1, 2, 3)
        market: Market type (player_receptions, player_rush_yds, etc.)

    Returns:
        Dict with:
        - defense_rank: int (1-32, 1=best defense, 32=worst)
        - defense_rank_label: str (e.g., "#5 vs WR1s")
        - yards_allowed_avg: float (last 6 games average)
        - trend: str ('improving', 'declining', 'stable')
        - trend_arrow: str (arrow character)
        - man_coverage_pct: float (0-100)
        - zone_coverage_pct: float (0-100)
        - matchup_quality: str ('favorable', 'neutral', 'tough')
        - has_context: bool
    """
    cache_key = f"{opponent}_{week}_{season}_{position}_{pos_rank}"

    if cache_key in _MATCHUP_CONTEXT_CACHE:
        return _MATCHUP_CONTEXT_CACHE[cache_key]

    result = {
        'defense_rank': None,
        'defense_rank_label': '',
        'yards_allowed_avg': None,
        'trend': 'stable',
        'trend_arrow': '→',
        'man_coverage_pct': None,
        'zone_coverage_pct': None,
        'matchup_quality': 'neutral',
        'has_context': False
    }

    if not opponent or pd.isna(opponent):
        return result

    # Normalize position
    position = str(position).upper() if position else 'WR'
    if position == 'QB':
        position = 'WR'  # QBs don't have position-specific defense, use WR as proxy

    pos_rank = int(pos_rank) if pos_rank and not pd.isna(pos_rank) else 1
    pos_rank = max(1, min(pos_rank, 3))  # Clamp to 1-3

    # Load cached data
    defense_vs_pos = load_defense_vs_position_cache()
    coverage_df, _ = load_coverage_cache()

    # Get defense rank
    rank = calculate_defense_rank(defense_vs_pos, opponent, position, pos_rank, week, season)
    if rank is not None:
        result['defense_rank'] = rank
        result['defense_rank_label'] = f"#{rank} vs {position}{pos_rank}s"
        result['has_context'] = True

    # Get yards allowed average
    yards_avg = get_yards_allowed_avg(defense_vs_pos, opponent, position, pos_rank, week, season)
    if yards_avg is not None:
        result['yards_allowed_avg'] = yards_avg
        result['has_context'] = True

    # Get trend
    trend, arrow = calculate_trend(defense_vs_pos, opponent, position, pos_rank, week, season)
    result['trend'] = trend
    result['trend_arrow'] = arrow

    # Get coverage tendencies
    man_pct, zone_pct = get_coverage_tendencies(coverage_df, opponent, week, season)
    if man_pct is not None:
        result['man_coverage_pct'] = man_pct
        result['zone_coverage_pct'] = zone_pct
        result['has_context'] = True

    # Determine matchup quality
    if result['defense_rank'] is not None:
        if result['defense_rank'] >= 24:  # Bottom 8 defenses
            result['matchup_quality'] = 'favorable'
        elif result['defense_rank'] <= 8:  # Top 8 defenses
            result['matchup_quality'] = 'tough'
        else:
            result['matchup_quality'] = 'neutral'

    # Cache result
    _MATCHUP_CONTEXT_CACHE[cache_key] = result

    return result


def calculate_defense_rank(
    defense_vs_pos: pd.DataFrame,
    opponent: str,
    position: str,
    pos_rank: int,
    week: int,
    season: int
) -> Optional[int]:
    """
    Calculate defense rank (1-32) based on yards allowed.

    Rank 1 = best defense (allows fewest yards)
    Rank 32 = worst defense (allows most yards)

    Uses the z-score columns if available, otherwise calculates from trailing stats.
    """
    global _DEFENSE_RANKS_CACHE

    if defense_vs_pos is None or len(defense_vs_pos) == 0:
        return None

    pos_label = f"{position.lower()}{pos_rank}"

    # Determine the stat column based on position
    if position in ['WR', 'TE']:
        stat_col = f'opp_{pos_label}_receiving_yards_allowed_trailing'
    else:  # RB
        stat_col = f'opp_{pos_label}_rushing_yards_allowed_trailing'

    if stat_col not in defense_vs_pos.columns:
        # Try without position rank suffix
        if position in ['WR', 'TE']:
            stat_col = 'opp_wr1_receiving_yards_allowed_trailing'
        else:
            stat_col = 'opp_rb1_rushing_yards_allowed_trailing'

        if stat_col not in defense_vs_pos.columns:
            return None

    # Get most recent data for each team (week - 1 or closest available)
    cache_key = f"ranks_{season}_{week}_{stat_col}"

    if _DEFENSE_RANKS_CACHE is None:
        _DEFENSE_RANKS_CACHE = {}

    if cache_key not in _DEFENSE_RANKS_CACHE:
        # Filter to current season and available weeks
        season_data = defense_vs_pos[defense_vs_pos['season'] == season].copy()

        if len(season_data) == 0:
            return None

        # Get latest available week for each team (up to current week - 1)
        latest_data = season_data[season_data['week'] < week].copy()

        if len(latest_data) == 0:
            # Fall back to any available data
            latest_data = season_data.copy()

        # Get most recent row per team
        latest_data = latest_data.sort_values('week', ascending=False)
        latest_data = latest_data.groupby('defense_team').first().reset_index()

        # Calculate ranks (higher yards = worse defense = higher rank number)
        latest_data['rank'] = latest_data[stat_col].rank(ascending=False, method='min')

        # Store as dict: team -> rank
        _DEFENSE_RANKS_CACHE[cache_key] = dict(zip(
            latest_data['defense_team'],
            latest_data['rank'].astype(int)
        ))

    ranks = _DEFENSE_RANKS_CACHE.get(cache_key, {})
    return ranks.get(opponent)


def get_yards_allowed_avg(
    defense_vs_pos: pd.DataFrame,
    opponent: str,
    position: str,
    pos_rank: int,
    week: int,
    season: int
) -> Optional[float]:
    """
    Get average yards allowed to this position over trailing period.
    """
    if defense_vs_pos is None or len(defense_vs_pos) == 0:
        return None

    pos_label = f"{position.lower()}{pos_rank}"

    # Determine the stat column
    if position in ['WR', 'TE']:
        stat_col = f'opp_{pos_label}_receiving_yards_allowed_trailing'
    else:
        stat_col = f'opp_{pos_label}_rushing_yards_allowed_trailing'

    if stat_col not in defense_vs_pos.columns:
        return None

    # Get data for this opponent at/before this week
    matchup_data = defense_vs_pos[
        (defense_vs_pos['defense_team'] == opponent) &
        (defense_vs_pos['season'] == season) &
        (defense_vs_pos['week'] <= week)
    ].copy()

    if len(matchup_data) == 0:
        return None

    # Get most recent row
    matchup_data = matchup_data.sort_values('week', ascending=False)
    latest = matchup_data.iloc[0]

    value = latest.get(stat_col)
    if pd.notna(value):
        return round(float(value), 1)

    return None


def calculate_trend(
    defense_vs_pos: pd.DataFrame,
    opponent: str,
    position: str,
    pos_rank: int,
    week: int,
    season: int,
    n_weeks: int = 4
) -> Tuple[str, str]:
    """
    Calculate if defense is improving or declining.

    Compares weeks 1-2 vs weeks 3-4 of the trailing window.

    Returns:
        Tuple of (trend_label, arrow_char)
        - 'improving' + '↓' = defense getting better (allowing fewer yards)
        - 'declining' + '↑' = defense getting worse (allowing more yards)
        - 'stable' + '→' = no significant change
    """
    if defense_vs_pos is None or len(defense_vs_pos) == 0:
        return ('stable', '→')

    pos_label = f"{position.lower()}{pos_rank}"

    # Determine the raw stat column (not trailing)
    if position in ['WR', 'TE']:
        stat_col = 'receiving_yards'
    else:
        stat_col = 'rushing_yards'

    # Need to load non-pivoted data for trend calculation
    # For now, use available trailing data to estimate trend
    if position in ['WR', 'TE']:
        trailing_col = f'opp_{pos_label}_receiving_yards_allowed_trailing'
    else:
        trailing_col = f'opp_{pos_label}_rushing_yards_allowed_trailing'

    if trailing_col not in defense_vs_pos.columns:
        return ('stable', '→')

    # Get recent weeks for this opponent
    recent_data = defense_vs_pos[
        (defense_vs_pos['defense_team'] == opponent) &
        (defense_vs_pos['season'] == season) &
        (defense_vs_pos['week'] < week) &
        (defense_vs_pos['week'] >= week - n_weeks)
    ].copy()

    if len(recent_data) < 2:
        return ('stable', '→')

    recent_data = recent_data.sort_values('week')

    # Split into early and late periods
    mid_point = len(recent_data) // 2
    early_weeks = recent_data.iloc[:mid_point]
    late_weeks = recent_data.iloc[mid_point:]

    early_avg = early_weeks[trailing_col].mean()
    late_avg = late_weeks[trailing_col].mean()

    if pd.isna(early_avg) or pd.isna(late_avg):
        return ('stable', '→')

    # Calculate change percentage
    if early_avg > 0:
        change_pct = (late_avg - early_avg) / early_avg
    else:
        change_pct = 0

    # Threshold for meaningful change (10%)
    if change_pct < -0.10:
        # Allowing fewer yards = defense improving
        return ('improving', '↓')
    elif change_pct > 0.10:
        # Allowing more yards = defense declining
        return ('declining', '↑')
    else:
        return ('stable', '→')


def get_coverage_tendencies(
    coverage_df: pd.DataFrame,
    opponent: str,
    week: int,
    season: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get man/zone coverage percentages for opponent.

    Returns:
        Tuple of (man_coverage_pct, zone_coverage_pct) as 0-100 values
    """
    if coverage_df is None or len(coverage_df) == 0:
        return (None, None)

    # Get data for this opponent
    matchup_data = coverage_df[
        (coverage_df['defense_team'] == opponent) &
        (coverage_df['season'] == season) &
        (coverage_df['week'] < week)
    ].copy()

    if len(matchup_data) == 0:
        return (None, None)

    # Get most recent row
    matchup_data = matchup_data.sort_values('week', ascending=False)
    latest = matchup_data.iloc[0]

    man_rate = latest.get('man_coverage_rate_trailing')
    zone_rate = latest.get('zone_coverage_rate_trailing')

    if pd.notna(man_rate):
        man_pct = round(float(man_rate) * 100, 0)
    else:
        man_pct = None

    if pd.notna(zone_rate):
        zone_pct = round(float(zone_rate) * 100, 0)
    else:
        zone_pct = None

    return (man_pct, zone_pct)


def get_all_defense_rankings(
    season: int,
    week: int,
    position: str = 'WR',
    pos_rank: int = 1
) -> pd.DataFrame:
    """
    Get defense rankings for all 32 teams.

    Returns DataFrame with:
    - team, rank, yards_allowed_avg, trend
    """
    defense_vs_pos = load_defense_vs_position_cache()

    if len(defense_vs_pos) == 0:
        return pd.DataFrame()

    pos_label = f"{position.lower()}{pos_rank}"

    if position in ['WR', 'TE']:
        stat_col = f'opp_{pos_label}_receiving_yards_allowed_trailing'
    else:
        stat_col = f'opp_{pos_label}_rushing_yards_allowed_trailing'

    if stat_col not in defense_vs_pos.columns:
        return pd.DataFrame()

    # Filter to current season
    season_data = defense_vs_pos[defense_vs_pos['season'] == season].copy()

    if len(season_data) == 0:
        return pd.DataFrame()

    # Get latest data per team
    latest_data = season_data[season_data['week'] < week].copy()
    if len(latest_data) == 0:
        latest_data = season_data.copy()

    latest_data = latest_data.sort_values('week', ascending=False)
    latest_data = latest_data.groupby('defense_team').first().reset_index()

    # Calculate ranks
    latest_data['rank'] = latest_data[stat_col].rank(ascending=False, method='min').astype(int)

    # Build result
    result = latest_data[['defense_team', 'rank', stat_col]].copy()
    result = result.rename(columns={
        'defense_team': 'team',
        stat_col: 'yards_allowed_avg'
    })
    result = result.sort_values('rank')

    return result


def get_defense_game_history(
    opponent: str,
    season: int,
    week: int,
    position: str,
    market: str,
    n_games: int = 6,
    pos_rank: int = None
) -> Dict:
    """
    Get defense's game-by-game stats for what they allowed to a specific position.

    Args:
        opponent: Defense team abbreviation
        season: Season year
        week: Current week (get games before this)
        position: Position to check (QB, RB, WR, TE)
        pos_rank: Optional depth chart rank (1=WR1, 2=WR2, etc.) for position-specific filtering
        market: Market type to determine stat (rush_yds, receptions, etc.)
        n_games: Number of games to return

    Returns:
        Dict with:
        - weeks: list of week numbers
        - opponents: list of opponents faced
        - stats: list of stat values allowed
        - stat_type: string describing the stat
        - avg: average of the stats
    """
    result = {
        'weeks': [],
        'opponents': [],
        'stats': [],
        'stat_type': 'yards',
        'avg': 0
    }

    try:
        # Load PBP data
        pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
        if not pbp_path.exists():
            return result

        pbp = pd.read_parquet(pbp_path)
        pbp = pbp[(pbp['season'] == season) & (pbp['week'] < week)]

        if len(pbp) == 0:
            return result

        # Load roster data for position filtering
        roster_path = PROJECT_ROOT / 'data' / 'nflverse' / 'rosters.parquet'
        roster = None
        if roster_path.exists():
            roster = pd.read_parquet(roster_path)
            roster = roster[roster['season'] == season][['gsis_id', 'position']].drop_duplicates()

        # Load position_roles cache for depth chart filtering
        pos_roles = None
        if pos_rank and pos_rank in [1, 2, 3]:
            pos_roles_path = PROJECT_ROOT / 'data' / 'cache' / 'position_roles.parquet'
            if pos_roles_path.exists():
                pos_roles = pd.read_parquet(pos_roles_path)
                # Get most recent week's roles for each player
                pos_roles = pos_roles[pos_roles['season'] == season]
                pos_roles = pos_roles.sort_values('week', ascending=False).drop_duplicates('player_id')

        # Determine stat type and player ID column based on market and position
        player_id_col = None
        pos_filter = position.upper() if position else None

        if 'rush_yds' in market or 'rush_att' in market:
            stat_col = 'rushing_yards' if 'rush_yds' in market else 'rush_attempt'
            play_type = 'run'
            result['stat_type'] = 'rush yds' if 'rush_yds' in market else 'rush att'
            player_id_col = 'rusher_player_id'
            # RBs are the main rushers, but QBs can rush too
            if pos_filter not in ['RB', 'QB']:
                pos_filter = 'RB'  # Default to RB for rush stats
        elif 'reception' in market and 'yds' not in market:
            stat_col = 'complete_pass'
            play_type = 'pass'
            result['stat_type'] = 'rec'
            player_id_col = 'receiver_player_id'
            # For receptions, use the player's actual position
        elif 'reception_yds' in market or 'receiving' in market:
            stat_col = 'receiving_yards'
            play_type = 'pass'
            result['stat_type'] = 'rec yds'
            player_id_col = 'receiver_player_id'
            # For receiving yards, use the player's actual position
        elif 'pass_yds' in market:
            stat_col = 'passing_yards'
            play_type = 'pass'
            result['stat_type'] = 'pass yds'
            player_id_col = 'passer_player_id'
            pos_filter = 'QB'  # Only QBs pass
        elif 'pass_tds' in market or 'passing_tds' in market:
            stat_col = 'pass_touchdown'
            play_type = 'pass'
            result['stat_type'] = 'pass TDs'
            player_id_col = 'passer_player_id'
            pos_filter = 'QB'  # Only QBs pass TDs
        elif 'rush_tds' in market or 'rushing_tds' in market:
            stat_col = 'rush_touchdown'
            play_type = 'run'
            result['stat_type'] = 'rush TDs'
            player_id_col = 'rusher_player_id'
            if pos_filter not in ['RB', 'QB']:
                pos_filter = 'RB'
        elif 'anytime_td' in market or '1st_td' in market or 'last_td' in market:
            # Anytime TD = total TDs (rushing + receiving) - handle specially below
            stat_col = 'touchdown'  # Special marker
            play_type = None  # We'll handle both run and pass
            result['stat_type'] = 'TDs'
            player_id_col = None  # We'll aggregate differently
            if pos_filter not in ['RB', 'WR', 'TE', 'QB']:
                pos_filter = 'RB'  # Default to RB for anytime TD
        elif 'pass_att' in market or 'pass_attempts' in market:
            stat_col = 'pass_attempt'
            play_type = 'pass'
            result['stat_type'] = 'pass att'
            player_id_col = 'passer_player_id'
            pos_filter = 'QB'
        elif 'pass_comp' in market or 'completions' in market:
            stat_col = 'complete_pass'
            play_type = 'pass'
            result['stat_type'] = 'compl'
            player_id_col = 'passer_player_id'
            pos_filter = 'QB'
        else:
            stat_col = 'yards_gained'
            play_type = None
            result['stat_type'] = 'yards'

        # Filter to plays against this defense
        def_plays = pbp[pbp['defteam'] == opponent].copy()

        if play_type:
            def_plays = def_plays[def_plays['play_type'] == play_type]

        if len(def_plays) == 0:
            return result

        # Filter by position if we have roster data and a player ID column
        if roster is not None and player_id_col and pos_filter:
            # Merge with roster to get position
            def_plays = def_plays.merge(
                roster,
                left_on=player_id_col,
                right_on='gsis_id',
                how='left',
                suffixes=('', '_roster')
            )
            # Filter to plays by players of the target position
            def_plays = def_plays[def_plays['position'] == pos_filter]

            # Further filter by depth chart rank if available
            if pos_roles is not None and pos_rank and player_id_col:
                # Get player IDs that match the target depth rank
                depth_players = pos_roles[
                    (pos_roles['position'] == pos_filter) &
                    (pos_roles['pos_rank'] == pos_rank)
                ]['player_id'].tolist()
                if depth_players:
                    def_plays = def_plays[def_plays[player_id_col].isin(depth_players)]
                    # Update stat_type to reflect depth position
                    base_stat = result.get('stat_type', 'yards')
                    result['stat_type'] = f"{base_stat} to {pos_filter}{pos_rank}s"

        if len(def_plays) == 0:
            return result

        # Aggregate stats by week and opponent team
        if stat_col == 'rush_attempt' or stat_col == 'pass_attempt':
            # Count plays
            games = def_plays.groupby(['week', 'posteam']).size().reset_index(name='stat')
        elif stat_col == 'complete_pass':
            # Count completed passes
            games = def_plays[def_plays['complete_pass'] == 1].groupby(['week', 'posteam']).size().reset_index(name='stat')
        elif stat_col in ['pass_touchdown', 'rush_touchdown']:
            # Sum TD flags (0/1 values)
            games = def_plays.groupby(['week', 'posteam']).agg({
                stat_col: 'sum'
            }).reset_index()
            games = games.rename(columns={stat_col: 'stat'})
        elif stat_col == 'touchdown':
            # Anytime TD: Sum both rushing and receiving TDs allowed to position
            # Need to re-query PBP with both play types
            pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
            pbp = pd.read_parquet(pbp_path)
            pbp = pbp[(pbp['season'] == season) & (pbp['week'] < week) & (pbp['defteam'] == opponent)]

            # Get rushing TDs
            rush_tds = pbp[pbp['play_type'] == 'run'].copy()
            if roster is not None and pos_filter:
                rush_tds = rush_tds.merge(roster, left_on='rusher_player_id', right_on='gsis_id', how='left', suffixes=('', '_r'))
                rush_tds = rush_tds[rush_tds['position'] == pos_filter]
            rush_by_game = rush_tds.groupby(['week', 'posteam'])['rush_touchdown'].sum().reset_index()
            rush_by_game = rush_by_game.rename(columns={'rush_touchdown': 'rush_td'})

            # Get receiving TDs
            rec_tds = pbp[pbp['play_type'] == 'pass'].copy()
            if roster is not None and pos_filter:
                rec_tds = rec_tds.merge(roster, left_on='receiver_player_id', right_on='gsis_id', how='left', suffixes=('', '_r'))
                rec_tds = rec_tds[rec_tds['position'] == pos_filter]
            rec_by_game = rec_tds.groupby(['week', 'posteam'])['pass_touchdown'].sum().reset_index()
            rec_by_game = rec_by_game.rename(columns={'pass_touchdown': 'rec_td'})

            # Merge and sum
            games = rush_by_game.merge(rec_by_game, on=['week', 'posteam'], how='outer').fillna(0)
            games['stat'] = games['rush_td'] + games['rec_td']
        else:
            # Sum yards
            games = def_plays.groupby(['week', 'posteam']).agg({
                stat_col: 'sum'
            }).reset_index()
            games = games.rename(columns={stat_col: 'stat'})

        # Sort by week descending and take last n games
        games = games.sort_values('week', ascending=False).head(n_games)

        result['weeks'] = games['week'].tolist()
        result['opponents'] = games['posteam'].tolist()
        result['stats'] = games['stat'].tolist()

        if result['stats']:
            result['avg'] = sum(result['stats']) / len(result['stats'])

    except Exception as e:
        logger.warning(f"Error getting defense game history: {e}")

    return result


def clear_caches():
    """Clear module-level caches."""
    global _DEFENSE_RANKS_CACHE, _MATCHUP_CONTEXT_CACHE
    _DEFENSE_RANKS_CACHE = None
    _MATCHUP_CONTEXT_CACHE = {}
    logger.info("Matchup context caches cleared")
