#!/usr/bin/env python3
"""
NFL QUANT Professional Dashboard Generator
Bloomberg Terminal / DraftKings Pro Aesthetic

Features:
- Dark professional color palette
- Sticky table headers
- Inline filter controls
- Pick counts in tab labels
- Grid layout for pick logic
- Player search/filter
- Monospace fonts for numbers
- Alternating row striping
- Clean sort indicators (no emojis)
"""

import pandas as pd
import numpy as np
import re
import json
import copy
from datetime import datetime
from pathlib import Path
import hashlib
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use centralized path configuration
from nfl_quant.config_paths import PROJECT_ROOT, REPORTS_DIR, DATA_DIR, NFLVERSE_DIR

from nfl_quant.config import settings
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.explanation.causal_narrative import (
    CausalNarrativeGenerator,
    generate_causal_narrative,
    generate_prose_only,
    build_model_narrative as causal_build_narrative,
    CAUSAL_NARRATIVE_CSS
)
from nfl_quant.features.matchup_context import get_matchup_context, get_defense_game_history

# ============================================================================
# CENTRALIZED CONFIDENCE THRESHOLDS
# ============================================================================
# All confidence percentage thresholds should reference these constants
CONF_THRESHOLD_ELITE = 75    # >= 75% = ELITE tier
CONF_THRESHOLD_HIGH = 65     # >= 65% = HIGH tier
CONF_THRESHOLD_STANDARD = 55 # >= 55% = STANDARD tier
# < 55% = LOW tier


# ============================================================================
# MODULE-LEVEL HELPER FUNCTIONS
# ============================================================================
def safe_float(val, default=0):
    """Safely convert value to float, returning default if invalid."""
    return float(val) if pd.notna(val) and val != '' else default


def safe_int(val, default=0):
    """Safely convert value to int, returning default if invalid."""
    return int(val) if pd.notna(val) and val != '' else default


def safe_str(val, default=''):
    """Safely convert value to string, returning default if NaN/None."""
    if pd.isna(val) or val is None:
        return default
    return str(val)


def get_bet_size_suggestion(confidence: float, edge: float) -> tuple:
    """
    Calculate bet size suggestion based on confidence and edge.

    Returns (amount, tier) where:
    - amount: Dollar amount ($2-$5)
    - tier: 'high', 'medium', or 'low' for styling

    Based on betting_config.json confidence_tiers.
    """
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    edge_pct = edge if edge > 0 else 0

    # High confidence + strong edge = max bet
    if conf_pct >= 70 and edge_pct >= 8:
        return (5.00, 'high')
    # Medium confidence or good edge = standard bet
    elif conf_pct >= 60 or edge_pct >= 5:
        return (3.50, 'medium')
    # Low confidence = minimum bet
    else:
        return (2.00, 'low')


def get_projection(row, market: str = '', game_history: dict = None) -> float:
    """
    Get projection for a player prop, with fallbacks.

    Priority:
    1. model_projection (if exists)
    2. expected_tds (for TD markets from Poisson model)
    3. Market-specific mean (receiving_yards_mean, rushing_yards_mean, etc.)
    4. trailing_stat
    5. Game history average (loads from data if not provided)
    6. 0 as last resort
    """
    # Try model_projection first
    proj = safe_float(row.get('model_projection', 0))
    if proj > 0:
        return proj

    # Try market-specific means
    market_lower = str(market).lower() if market else str(row.get('market', '')).lower()

    # Handle TD markets - use expected_tds from Poisson model or p_attd for anytime TD
    if 'anytime_td' in market_lower or ('td' in market_lower and 'yds' not in market_lower):
        # For TD props, use expected_tds (Poisson lambda parameter)
        proj = safe_float(row.get('expected_tds', 0))
        if proj > 0:
            return proj
        # For anytime TD, p_attd is the probability - convert to expected TDs
        # If p_attd = 0.9 (90% chance of TD), expected TDs ≈ -ln(1-p) for geometric dist
        p_attd = safe_float(row.get('p_attd', 0))
        if p_attd > 0:
            # Convert probability to expected TD count (rough approximation)
            # Higher probability = higher expected TDs
            import math
            expected = -math.log(max(0.01, 1 - p_attd)) if p_attd < 0.99 else 2.0
            return round(expected, 2)
        # Try total_tds_mean as fallback for TD markets
        proj = safe_float(row.get('total_tds_mean', 0))
        if proj > 0:
            return proj

    if 'reception_yds' in market_lower or 'receiving' in market_lower:
        proj = safe_float(row.get('receiving_yards_mean', 0))
    elif 'receptions' in market_lower:
        proj = safe_float(row.get('receptions_mean', 0))
    elif 'rush_yds' in market_lower or 'rushing_yards' in market_lower:
        proj = safe_float(row.get('rushing_yards_mean', 0))
    elif 'rush_att' in market_lower or 'rushing_attempts' in market_lower:
        proj = safe_float(row.get('rushing_attempts_mean', 0))
    elif 'pass_yds' in market_lower or 'passing_yards' in market_lower:
        proj = safe_float(row.get('passing_yards_mean', 0))
    elif 'pass_att' in market_lower or 'passing_attempts' in market_lower:
        proj = safe_float(row.get('passing_attempts_mean', 0))
    elif 'completions' in market_lower:
        proj = safe_float(row.get('completions_mean', 0))

    if proj > 0:
        return proj

    # Try trailing_stat
    proj = safe_float(row.get('trailing_stat', 0))
    if proj > 0:
        return proj

    # Try game history average - load from data if not provided
    if game_history is None:
        game_history = row.get('game_history', {})

    # If still no game_history, try to load it directly
    if not game_history or (isinstance(game_history, dict) and not game_history.get('weeks')):
        player = row.get('player', '')
        week = row.get('week', 16)
        if player and market_lower:
            try:
                # Use lazy import to avoid circular reference
                game_history = get_player_game_history(player, market_lower, int(week) if pd.notna(week) else 16)
            except:
                game_history = {}

    if isinstance(game_history, dict) and game_history.get('weeks'):
        if 'reception_yds' in market_lower or 'receiving' in market_lower:
            vals = game_history.get('receiving_yards', [])
        elif 'receptions' in market_lower:
            vals = game_history.get('receptions', [])
        elif 'rush_yds' in market_lower or 'rushing_yards' in market_lower:
            vals = game_history.get('rushing_yards', [])
        elif 'rush_att' in market_lower or 'rushing_attempts' in market_lower:
            vals = game_history.get('rushing_attempts', [])
        elif 'pass_yds' in market_lower or 'passing_yards' in market_lower:
            vals = game_history.get('passing_yards', [])
        elif 'anytime_td' in market_lower or 'td' in market_lower:
            # For TD markets, sum rushing + receiving + passing TDs
            rush_tds = game_history.get('rushing_tds', [])
            rec_tds = game_history.get('receiving_tds', [])
            pass_tds = game_history.get('passing_tds', [])
            # Combine TDs per game
            n_games = len(game_history.get('weeks', []))
            if n_games > 0:
                vals = []
                for i in range(n_games):
                    total_td = 0
                    if rush_tds and i < len(rush_tds): total_td += rush_tds[i] or 0
                    if rec_tds and i < len(rec_tds): total_td += rec_tds[i] or 0
                    if pass_tds and i < len(pass_tds): total_td += pass_tds[i] or 0
                    vals.append(total_td)
            else:
                vals = []
        else:
            vals = []

        if vals and len(vals) > 0:
            valid_vals = [v for v in vals if v is not None and v >= 0]  # TDs can be 0, so allow 0
            if valid_vals:
                return sum(valid_vals) / len(valid_vals)

    return 0


def generate_uuid_short():
    """Generate short unique ID for row expansion."""
    return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]


# Global cache for L6 TD games data
_l6_td_cache = None
_l5_stats_cache = None

def get_l5_hit_rate(player_name: str, market: str, line: float, season: int = None) -> dict:
    """Get the L5 hit rate for a player on a specific market line.

    Args:
        player_name: Player's display name
        market: Market type (e.g., 'player_receptions', 'player_rush_yds')
        line: The betting line to compare against
        season: NFL season year (defaults to current)

    Returns:
        dict with 'hits', 'total_games', 'values' (last 5 game values)
    """
    global _l5_stats_cache

    if season is None:
        season = get_current_season()

    # Map market to stat column
    market_to_stat = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_yds': 'passing_yards',
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',
    }

    stat_col = market_to_stat.get(market)
    if not stat_col:
        return {'hits': 0, 'total_games': 0, 'values': []}

    # Load and cache weekly stats
    if _l5_stats_cache is None:
        try:
            ws_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
            if ws_path.exists():
                _l5_stats_cache = pd.read_parquet(ws_path)
            else:
                return {'hits': 0, 'total_games': 0, 'values': []}
        except Exception as e:
            return {'hits': 0, 'total_games': 0, 'values': []}

    # Get player's last 5 games
    player_data = _l5_stats_cache[
        (_l5_stats_cache['player_display_name'] == player_name) &
        (_l5_stats_cache['season'] == season)
    ].sort_values('week', ascending=False).head(5)

    if len(player_data) == 0 or stat_col not in player_data.columns:
        return {'hits': 0, 'total_games': 0, 'values': []}

    values = player_data[stat_col].fillna(0).tolist()
    hits = sum(1 for v in values if v > line)

    return {
        'hits': hits,
        'total_games': len(values),
        'values': values
    }


def get_l6_td_games(player_name: str, season: int = None) -> dict:
    """Get the number of games with TDs in the last 6 games for a player.

    Args:
        player_name: Player's display name
        season: NFL season year (defaults to current)

    Returns:
        dict with 'games_with_td', 'total_games', 'total_tds'
    """
    global _l6_td_cache

    if season is None:
        season = get_current_season()

    # Load and cache weekly stats
    if _l6_td_cache is None:
        try:
            ws_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
            if ws_path.exists():
                ws = pd.read_parquet(ws_path)
                # Calculate total TDs per game
                ws['total_tds'] = ws['rushing_tds'].fillna(0) + ws['receiving_tds'].fillna(0)
                _l6_td_cache = ws
            else:
                return {'games_with_td': 0, 'total_games': 0, 'total_tds': 0}
        except Exception as e:
            return {'games_with_td': 0, 'total_games': 0, 'total_tds': 0}

    # Get player's last 6 games
    player_data = _l6_td_cache[
        (_l6_td_cache['player_display_name'] == player_name) &
        (_l6_td_cache['season'] == season)
    ].sort_values('week', ascending=False).head(6)

    if len(player_data) == 0:
        return {'games_with_td': 0, 'total_games': 0, 'total_tds': 0}

    games_with_td = (player_data['total_tds'] >= 1).sum()
    total_tds = int(player_data['total_tds'].sum())

    return {
        'games_with_td': int(games_with_td),
        'total_games': len(player_data),
        'total_tds': total_tds
    }


def format_player_name_compact(full_name: str, max_len: int = 14) -> str:
    """Format player name compactly: 'Justin Herbert' -> 'J. Herbert' if too long."""
    if not full_name or len(full_name) <= max_len:
        return full_name
    parts = full_name.split()
    if len(parts) >= 2:
        # Use first initial + last name
        return f"{parts[0][0]}. {parts[-1]}"
    return full_name[:max_len]


# Global cache for game history data
_game_history_cache = None

# Global cache for player headshots
_player_headshots_cache = None

# Global caches for position-specific defense
_defense_vs_position_cache = None
_position_roles_cache = None

# Global caches for QB-specific defense
_defense_vs_qb_cache = None
_qb_style_cache = None


def load_qb_defense_caches():
    """Load QB-specific defense caches for pass attempts/completions/TDs rankings."""
    global _defense_vs_qb_cache, _qb_style_cache

    if _defense_vs_qb_cache is not None:
        return _defense_vs_qb_cache, _qb_style_cache

    try:
        def_vs_qb_path = DATA_DIR / 'cache' / 'defense_vs_qb.parquet'
        qb_style_path = DATA_DIR / 'cache' / 'qb_style.parquet'

        if def_vs_qb_path.exists():
            _defense_vs_qb_cache = pd.read_parquet(def_vs_qb_path)
            print(f"  Loaded defense_vs_qb cache: {len(_defense_vs_qb_cache)} rows")
        else:
            _defense_vs_qb_cache = pd.DataFrame()

        if qb_style_path.exists():
            _qb_style_cache = pd.read_parquet(qb_style_path)
            print(f"  Loaded qb_style cache: {len(_qb_style_cache)} QBs")
        else:
            _qb_style_cache = pd.DataFrame()

        return _defense_vs_qb_cache, _qb_style_cache
    except Exception as e:
        print(f"  Warning: Could not load QB defense caches: {e}")
        _defense_vs_qb_cache = pd.DataFrame()
        _qb_style_cache = pd.DataFrame()
        return _defense_vs_qb_cache, _qb_style_cache


def get_qb_defense_rank(opponent: str, week: int, season: int, market: str) -> tuple:
    """
    Get defense rank vs QBs for a specific market.

    Returns: (rank, rank_type_label) or (None, None) if not available
    """
    def_vs_qb, _ = load_qb_defense_caches()

    if def_vs_qb is None or len(def_vs_qb) == 0:
        return None, None

    # Map market to stat column
    col_map = {
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',
        'player_pass_tds': 'passing_tds',
        'player_pass_yds': 'passing_yards',
    }

    col = col_map.get(market)
    if not col or col not in def_vs_qb.columns:
        return None, None

    # Get latest week where opponent has data
    opp_weeks = def_vs_qb[
        (def_vs_qb['season'] == season) &
        (def_vs_qb['week'] <= week) &
        (def_vs_qb['opponent_team'] == opponent)
    ]
    if len(opp_weeks) == 0:
        return None, None

    latest_week = opp_weeks['week'].max()
    if pd.isna(latest_week):
        return None, None

    # Use this week for all teams to get consistent rankings
    latest = def_vs_qb[
        (def_vs_qb['season'] == season) &
        (def_vs_qb['week'] == latest_week)
    ].copy()

    if len(latest) == 0:
        return None, None

    # Rank teams - lower value = allows fewer = better defense = lower rank number
    latest['rank'] = latest[col].rank()

    team_row = latest[latest['opponent_team'] == opponent]
    if len(team_row) == 0:
        return None, None

    rank = int(team_row['rank'].iloc[0])

    # Create label based on market
    label_map = {
        'player_pass_attempts': 'vs Pass Att',
        'player_pass_completions': 'vs Compl',
        'player_pass_tds': 'vs Pass TDs',
        'player_pass_yds': 'vs Pass Yds',
    }
    label = label_map.get(market, 'vs QBs')

    return rank, label


def load_position_defense_caches():
    """Load position-specific defense caches for WR1/WR2/RB1/etc rankings."""
    global _defense_vs_position_cache, _position_roles_cache

    if _defense_vs_position_cache is not None and _position_roles_cache is not None:
        return _defense_vs_position_cache, _position_roles_cache

    try:
        def_vs_pos_path = DATA_DIR / 'cache' / 'defense_vs_position.parquet'
        roles_path = DATA_DIR / 'cache' / 'position_roles.parquet'

        if def_vs_pos_path.exists():
            _defense_vs_position_cache = pd.read_parquet(def_vs_pos_path)
            print(f"  Loaded defense-vs-position cache: {len(_defense_vs_position_cache)} rows")
        else:
            _defense_vs_position_cache = pd.DataFrame()
            print(f"  Warning: defense_vs_position.parquet not found")

        if roles_path.exists():
            _position_roles_cache = pd.read_parquet(roles_path)
            print(f"  Loaded position_roles cache: {len(_position_roles_cache)} rows")
        else:
            _position_roles_cache = pd.DataFrame()
            print(f"  Warning: position_roles.parquet not found")

        return _defense_vs_position_cache, _position_roles_cache
    except Exception as e:
        print(f"  Warning: Could not load position defense caches: {e}")
        _defense_vs_position_cache = pd.DataFrame()
        _position_roles_cache = pd.DataFrame()
        return _defense_vs_position_cache, _position_roles_cache


def get_player_position_role(player_id: str, team: str, week: int, season: int) -> tuple:
    """
    Get player's position role (WR1, WR2, RB1, etc).

    Returns: (position, pos_rank) or (None, None) if not found
    """
    _, position_roles = load_position_defense_caches()

    if position_roles is None or len(position_roles) == 0:
        return None, None

    # Look up player's role for the most recent available week
    player_roles = position_roles[
        (position_roles['player_id'] == player_id) &
        (position_roles['team'] == team) &
        (position_roles['season'] == season) &
        (position_roles['week'] <= week)
    ].sort_values('week', ascending=False)

    if len(player_roles) == 0:
        return None, None

    row = player_roles.iloc[0]
    return row.get('position'), int(row.get('pos_rank', 1))


def get_position_specific_defense_rank(opponent: str, week: int, season: int, position: str, pos_rank: int, market: str = '') -> tuple:
    """
    Get defense rank vs specific position role (e.g., vs WR1s, vs RB2s).

    Args:
        opponent: Opponent team abbreviation
        week: NFL week number
        season: NFL season year
        position: Player position (WR, RB, TE)
        pos_rank: Position rank (1, 2, 3 for WR1, WR2, WR3)
        market: Market type for stat-specific ranking (player_receptions, player_rush_yds, etc.)

    Returns: (rank, rank_type_label) or (None, None) if not available
    """
    def_vs_pos, _ = load_position_defense_caches()

    if def_vs_pos is None or len(def_vs_pos) == 0:
        return None, None

    # Determine stat type from market
    market_lower = market.lower() if market else ''
    if 'reception' in market_lower and 'yds' not in market_lower:
        stat_type = 'receptions'
    elif 'rush' in market_lower:
        stat_type = 'rushing_yards'
    else:
        stat_type = 'receiving_yards'  # Default for reception_yds and others

    # Map position+rank+stat to column (use vs_avg z-score columns for ranking)
    col_map = {
        # Receiving yards
        ('WR', 1, 'receiving_yards'): 'opp_wr1_receiving_yards_vs_avg',
        ('WR', 2, 'receiving_yards'): 'opp_wr2_receiving_yards_vs_avg',
        ('WR', 3, 'receiving_yards'): 'opp_wr3_receiving_yards_vs_avg',
        ('RB', 1, 'receiving_yards'): 'opp_rb1_receiving_yards_vs_avg',
        ('RB', 2, 'receiving_yards'): 'opp_rb2_receiving_yards_vs_avg',
        ('TE', 1, 'receiving_yards'): 'opp_te1_receiving_yards_vs_avg',
        ('TE', 2, 'receiving_yards'): 'opp_te2_receiving_yards_vs_avg',
        # Receptions
        ('WR', 1, 'receptions'): 'opp_wr1_receptions_vs_avg',
        ('WR', 2, 'receptions'): 'opp_wr2_receptions_vs_avg',
        ('WR', 3, 'receptions'): 'opp_wr3_receptions_vs_avg',
        ('RB', 1, 'receptions'): 'opp_rb1_receptions_vs_avg',
        ('RB', 2, 'receptions'): 'opp_rb2_receptions_vs_avg',
        ('TE', 1, 'receptions'): 'opp_te1_receptions_vs_avg',
        ('TE', 2, 'receptions'): 'opp_te2_receptions_vs_avg',
        # Rushing yards
        ('RB', 1, 'rushing_yards'): 'opp_rb1_rushing_yards_vs_avg',
        ('RB', 2, 'rushing_yards'): 'opp_rb2_rushing_yards_vs_avg',
        ('WR', 1, 'rushing_yards'): 'opp_wr1_rushing_yards_vs_avg',
        ('WR', 2, 'rushing_yards'): 'opp_wr2_rushing_yards_vs_avg',
    }

    col = col_map.get((position, pos_rank, stat_type))
    if not col or col not in def_vs_pos.columns:
        return None, None

    # Get latest week where the OPPONENT team has data (not just max week overall)
    # This handles mid-week games where only some teams have played
    opp_weeks = def_vs_pos[
        (def_vs_pos['season'] == season) &
        (def_vs_pos['week'] <= week) &
        (def_vs_pos['defense_team'] == opponent)
    ]
    if len(opp_weeks) == 0:
        return None, None

    latest_week = opp_weeks['week'].max()
    if pd.isna(latest_week):
        return None, None

    # Use this week for ALL teams to get consistent rankings
    latest = def_vs_pos[
        (def_vs_pos['season'] == season) &
        (def_vs_pos['week'] == latest_week)
    ].copy()

    if len(latest) == 0:
        return None, None

    # Rank teams - higher z-score = allows more yards = worse defense = higher rank number
    latest['rank'] = latest[col].rank(ascending=False, na_option='bottom')

    team_row = latest[latest['defense_team'] == opponent]
    if len(team_row) == 0:
        return None, None

    rank = int(team_row['rank'].iloc[0])
    label = f"vs {position}{pos_rank}s"  # e.g., "vs WR1s", "vs RB2s"

    return rank, label


def generate_matchup_context_html(row: pd.Series, row_id: str, week: int, season: int = 2025, prefix: str = '') -> str:
    """
    Generate expandable matchup context section for a player prop pick.

    Shows:
    - Defense rank vs position (#1-32)
    - Average yards allowed
    - Recent trend (improving/declining)
    - Coverage tendencies (man/zone %)
    """
    # Get opponent from row data (handle NaN values properly)
    opponent_abbr = row.get('opponent_abbr', '')
    opponent_raw = row.get('opponent', '')
    opponent = opponent_abbr if opponent_abbr and not pd.isna(opponent_abbr) else (opponent_raw if not pd.isna(opponent_raw) else '')

    actual_pos = row.get('actual_position', '')
    pos_raw = row.get('position', 'WR')
    position = actual_pos if actual_pos and not pd.isna(actual_pos) else (pos_raw if not pd.isna(pos_raw) else 'WR')

    # Skip if no opponent or if this is a game line
    if not opponent or position == 'GAME':
        return ''

    # Get position rank (default to 1)
    pos_rank = row.get('pos_rank', 1)
    if pd.isna(pos_rank):
        pos_rank = 1
    pos_rank = int(pos_rank)

    market = row.get('market', '')
    player = row.get('player', '')

    # Get market display name and stat key for label
    market_info = {
        'player_rush_yds': ('Rush Yds', 'rushing_yards'),
        'player_rush_attempts': ('Rush Att', 'rushing_attempts'),
        'player_reception_yds': ('Rec Yds', 'receiving_yards'),
        'player_receptions': ('Rec', 'receptions'),
        'player_pass_yds': ('Pass Yds', 'passing_yards'),
        'player_pass_tds': ('Pass TDs', 'passing_tds'),
        'player_rush_tds': ('Rush TDs', 'rushing_tds'),
        'player_rec_tds': ('Rec TDs', 'receiving_tds'),
        'player_pass_attempts': ('Pass Att', 'passing_attempts'),
        'player_pass_completions': ('Comp', 'completions'),
        'player_anytime_td': ('TD', 'total_tds'),
        'player_1st_td': ('1st TD', 'total_tds'),
        'player_last_td': ('Last TD', 'total_tds'),
    }
    market_label, stat_key = market_info.get(market, ('Avg', 'rushing_yards'))

    # Get player's trailing average for this specific market
    # First try trailing_stat, then calculate from game history
    player_avg = row.get('trailing_stat', 0)
    if pd.isna(player_avg) or player_avg == 0:
        # Try to calculate from game history
        game_history = get_player_game_history(player, market, week)
        if game_history and stat_key in game_history:
            stats = game_history.get(stat_key, [])
            if stats:
                # Filter out None/NaN values and calculate average
                valid_stats = [s for s in stats if s is not None and not (isinstance(s, float) and pd.isna(s))]
                if valid_stats:
                    player_avg = sum(valid_stats) / len(valid_stats)

    # Get matchup context
    context = get_matchup_context(
        opponent=opponent,
        week=week,
        season=season,
        position=position,
        pos_rank=pos_rank,
        market=market
    )

    # For QBs, override with row's def_rank (EPA-based) to match modal display
    # The matchup_context module uses WR receiving yards as a proxy for QBs,
    # but the modal uses pass defense EPA which is more accurate for QBs
    if position == 'QB' and row.get('def_rank'):
        row_def_rank = row.get('def_rank')
        if not pd.isna(row_def_rank) and row_def_rank > 0:
            context['defense_rank'] = int(row_def_rank)
            context['defense_rank_label'] = f"#{int(row_def_rank)} vs QBs"
            context['has_context'] = True
            # Update matchup quality based on corrected rank
            if context['defense_rank'] >= 24:
                context['matchup_quality'] = 'favorable'
            elif context['defense_rank'] <= 8:
                context['matchup_quality'] = 'tough'
            else:
                context['matchup_quality'] = 'neutral'

    # Get defense's game-by-game history for this specific stat
    # Pass pos_rank to get depth-specific stats (e.g., rec yds to WR2s)
    def_history = get_defense_game_history(
        opponent=opponent,
        season=season,
        week=week,
        position=position,
        market=market,
        n_games=6,
        pos_rank=pos_rank if position in ['WR', 'RB', 'TE'] else None
    )

    if not context.get('has_context') and not def_history.get('stats'):
        return ''

    # Determine quality class for badge
    quality = context.get('matchup_quality', 'neutral')
    if quality == 'favorable':
        quality_class = 'matchup-favorable'
    elif quality == 'tough':
        quality_class = 'matchup-tough'
    else:
        quality_class = 'matchup-neutral'

    # Build rank display
    rank = context.get('defense_rank')
    rank_label = context.get('defense_rank_label', '')
    rank_display = f"#{rank}" if rank else "N/A"

    # Defense average for specific stat (from game history)
    def_avg = def_history.get('avg', 0)
    def_stat_type = def_history.get('stat_type', 'yards')
    def_avg_display = f"{def_avg:.1f}" if def_avg else "—"

    # Build position label based on market and position
    # If we have a position rank (WR1, RB2, etc.), use it for specificity
    if pos_rank and pos_rank > 0 and position in ['WR', 'RB', 'TE']:
        pos_label = f"vs {position}{pos_rank}s"  # e.g., "vs WR1s", "vs RB2s"
    elif 'rush' in market:
        pos_label = f"vs {position}s" if position else "vs RBs"
    elif 'pass_yds' in market:
        pos_label = "vs QBs"
    elif 'reception' in market or 'receiving' in market:
        pos_label = f"vs {position}s" if position else "vs WRs"
    else:
        pos_label = rank_label.replace(f"#{rank} ", "") if rank else ""

    # Player average display
    player_avg_display = f"{player_avg:.1f}" if player_avg else "—"

    # Build defense trend chart (simple bar chart)
    def_stats = def_history.get('stats', [])
    def_weeks = def_history.get('weeks', [])
    def_opps = def_history.get('opponents', [])

    chart_html = ''
    if def_stats:
        max_stat = max(def_stats) if def_stats else 1
        max_bar_height = 50  # pixels
        bars_html = ''
        for i, (wk, opp, stat) in enumerate(zip(def_weeks, def_opps, def_stats)):
            # Calculate bar height in pixels (more reliable than %)
            height_px = int((stat / max_stat) * max_bar_height) if max_stat > 0 else 4
            height_px = max(4, height_px)  # minimum 4px
            bars_html += f'''
                <div class="def-chart-bar-container" title="Week {wk} vs {opp}: {stat:.0f}">
                    <span class="def-chart-label">{stat:.0f}</span>
                    <div class="def-chart-bar" style="height: {height_px}px;"></div>
                    <span class="def-chart-week">{opp}</span>
                </div>
            '''

        # def_stat_type now includes depth position when available (e.g., "rec yds to WR2s")
        # If it doesn't include "to", add position for backwards compatibility
        if ' to ' not in def_stat_type and position:
            chart_label = f"{def_stat_type} to {position}s"
        else:
            chart_label = def_stat_type
        chart_html = f'''
            <div class="def-chart-section">
                <span class="def-chart-title">{opponent} DEF Last 6 ({chart_label})</span>
                <div class="def-chart-container">
                    {bars_html}
                </div>
            </div>
        '''

    # Create unique ID with prefix to avoid duplicates across views
    unique_id = f"{prefix}{row_id}" if prefix else row_id

    return f'''
        <div class="matchup-context" onclick="event.stopPropagation();">
            <div class="matchup-header" onclick="toggleMatchupContext('{unique_id}'); event.stopPropagation();">
                <span class="matchup-vs">vs</span>
                <span class="matchup-team">{opponent}</span>
                <span class="matchup-rank-badge {quality_class}">{rank_display} {pos_label}</span>
                <span class="matchup-chevron" id="matchup-chevron-{unique_id}">▼</span>
            </div>
            <div class="matchup-details" id="matchup-{unique_id}">
                <div class="matchup-grid">
                    <div class="matchup-item">
                        <span class="matchup-label">Player {market_label} Avg</span>
                        <span class="matchup-value">{player_avg_display}</span>
                    </div>
                    <div class="matchup-item">
                        <span class="matchup-label">{opponent} Allows ({def_stat_type})</span>
                        <span class="matchup-value">{def_avg_display}/game</span>
                    </div>
                    <div class="matchup-item">
                        <span class="matchup-label">DEF Rank</span>
                        <span class="matchup-value">{rank_display} {pos_label}</span>
                    </div>
                </div>
                {chart_html}
            </div>
        </div>
    '''


def load_player_headshots() -> dict:
    """Load player headshots from NFLverse data. Returns dict mapping player name -> headshot URL."""
    global _player_headshots_cache

    if _player_headshots_cache is not None:
        return _player_headshots_cache

    try:
        import nfl_data_py as nfl
        players = nfl.import_players()

        # Create lookup dict: player name -> headshot URL
        # Use display_name as primary key
        headshots = {}
        for _, row in players.iterrows():
            if pd.notna(row.get('headshot')) and row['headshot']:
                name = row.get('display_name', '')
                if name:
                    headshots[name] = row['headshot']

        _player_headshots_cache = headshots
        print(f"Loaded {len(headshots)} player headshots")
        return headshots
    except Exception as e:
        print(f"Warning: Could not load player headshots: {e}")
        _player_headshots_cache = {}
        return {}


def get_player_headshot_html(player_name: str, team: str, size: int = 40) -> str:
    """Get HTML for player headshot image with team logo fallback."""
    headshots = load_player_headshots()
    headshot_url = headshots.get(player_name, '')

    if headshot_url:
        return f'''<img src="{headshot_url}" alt="{player_name}" class="player-headshot"
                    style="width: {size}px; height: {size}px; border-radius: 50%; object-fit: cover; border: 2px solid var(--glass-border); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(59, 130, 246, 0.2); background: var(--surface-elevated);"
                    onerror="this.onerror=null; this.src='https://a.espncdn.com/i/teamlogos/nfl/500/{team.lower()}.png'; this.style.borderRadius='8px';">'''
    else:
        # Fallback to team logo
        return get_team_logo_html(team, size)


def load_game_history_data(week: int, season: int = None) -> pd.DataFrame:
    """Load weekly stats for game history chart visualization."""
    global _game_history_cache

    if _game_history_cache is not None:
        return _game_history_cache

    if season is None:
        season = get_current_season()

    weekly_stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    if not weekly_stats_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(weekly_stats_path)
        # Filter to current season and weeks before the current week
        df = df[(df['season'] == season) & (df['week'] < week)]
        _game_history_cache = df
        return df
    except Exception as e:
        print(f"Warning: Could not load game history: {e}")
        return pd.DataFrame()


def extract_opponent_from_game(game: str, team_abbr: str) -> str:
    """Extract opponent abbreviation from game string like 'MIA @ PIT'."""
    if not game or '@' not in game:
        return ''
    parts = game.replace('@', ' @ ').split()
    # Find team abbreviations (typically 2-4 capital letters)
    abbrs = [p for p in parts if p.isupper() and len(p) <= 4 and p != '@']
    if len(abbrs) >= 2:
        # Handle LA/LAR equivalence (Rams use both abbreviations)
        def matches_team(abbr, team):
            if abbr == team:
                return True
            # LA and LAR are equivalent (Los Angeles Rams)
            if (abbr == 'LA' and team == 'LAR') or (abbr == 'LAR' and team == 'LA'):
                return True
            return False
        return abbrs[1] if matches_team(abbrs[0], team_abbr) else abbrs[0]
    return ''


def get_player_game_history(player_name: str, market: str, week: int, season: int = None) -> dict:
    """Get last 6 games' stats for a player/market combination.

    Returns all stat types so the JavaScript can pick the right one based on market.
    """
    df = load_game_history_data(week, season)

    if df.empty:
        return {'weeks': [], 'opponents': [], 'receiving_yards': [], 'receptions': [],
                'rushing_yards': [], 'rushing_attempts': [], 'passing_yards': [],
                'passing_attempts': [], 'completions': [], 'passing_tds': [],
                'rushing_tds': [], 'receiving_tds': []}

    # Find player (try both display name and normalized name)
    player_df = df[df['player_display_name'] == player_name]
    if player_df.empty:
        player_df = df[df['player_name'] == player_name]
    if player_df.empty:
        return {'weeks': [], 'opponents': [], 'receiving_yards': [], 'receptions': [],
                'rushing_yards': [], 'rushing_attempts': [], 'passing_yards': [],
                'passing_attempts': [], 'completions': [], 'passing_tds': [],
                'rushing_tds': [], 'receiving_tds': []}

    # Sort by week descending and take last 6
    player_df = player_df.sort_values('week', ascending=False).head(6)

    # Helper to safely get column as list
    def safe_get(col, default=0):
        if col in player_df.columns:
            return player_df[col].fillna(default).tolist()
        return []

    # Build result with all stat types
    result = {
        'weeks': player_df['week'].tolist(),
        'opponents': safe_get('opponent_team', ''),
        'receiving_yards': safe_get('receiving_yards'),
        'receptions': safe_get('receptions'),
        'rushing_yards': safe_get('rushing_yards'),
        'rushing_attempts': safe_get('carries'),  # Column name is 'carries' in nflverse
        'passing_yards': safe_get('passing_yards'),
        'passing_attempts': safe_get('attempts'),  # Column name is 'attempts' in nflverse
        'completions': safe_get('completions'),
        'passing_tds': safe_get('passing_tds'),
        'rushing_tds': safe_get('rushing_tds'),
        'receiving_tds': safe_get('receiving_tds'),
    }

    return result


def build_model_narrative(row: pd.Series, pick: str) -> str:
    """Build a causal narrative showing HOW the projection was built.

    Delegates to causal_narrative module for consistent output.
    """
    try:
        return causal_build_narrative(row, pick)
    except Exception as e:
        # Fallback to simple display if causal narrative fails
        market = str(row.get('market', ''))
        projection = get_projection(row, market)
        line = row.get('line', 0)
        prob = row.get('model_prob', row.get('prob_over', 0.5))

        if projection > 0 and line > 0:
            edge_pct = ((projection - line) / line) * 100
            return f'''<div class="model-narrative">
                <div class="narrative-title">Model Analysis</div>
                <div class="narrative-content">
                    Projection: {projection:.1f} vs Line: {line:.1f} ({edge_pct:+.0f}% edge) | Confidence: {prob:.0%}
                </div>
            </div>'''
        return ""


def format_game_time(commence_time_str: str) -> str:
    """
    Format ISO datetime string to readable game time.

    Args:
        commence_time_str: ISO format string like '2025-12-07T18:00:00Z'

    Returns:
        Formatted string like 'Sun 1:00 PM ET'
    """
    if not commence_time_str or pd.isna(commence_time_str):
        return ''

    try:
        # Parse ISO format
        dt = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))

        # Convert UTC to Eastern Time (UTC-5 during EST, UTC-4 during EDT)
        # For December, we're in EST (UTC-5)
        from datetime import timedelta
        et_offset = timedelta(hours=-5)  # EST
        dt_et = dt + et_offset

        # Format: "Sun 1:00 PM ET"
        day_abbr = dt_et.strftime('%a')  # Sun, Mon, etc.
        hour = dt_et.hour
        minute = dt_et.minute

        # Convert to 12-hour format
        am_pm = 'AM' if hour < 12 else 'PM'
        if hour == 0:
            hour = 12
        elif hour > 12:
            hour -= 12

        if minute == 0:
            time_str = f"{hour} {am_pm}"
        else:
            time_str = f"{hour}:{minute:02d} {am_pm}"

        return f"{day_abbr} {time_str} ET"
    except Exception:
        return ''


# Import model version from centralized config
from configs.model_config import MODEL_VERSION_FULL, SUPPORTED_MARKETS

# Market-Specific Thresholds Based on Walk-Forward Backtest Analysis
# Analysis of 1,710 bets from walk_forward_with_lines_results.csv
#
# MARKET ROI ANALYSIS (Weeks 5-13):
# -----------------------------------
# RECEPTIONS:       80%+ = +11.6% ROI, 58.5% WR (248 bets) ✅ PROFITABLE
# RECEIVING_YARDS:  60%+ = +0.3% ROI, 52.5% WR (653 bets) ~ Break-even
# RUSHING_YARDS:    60%+ = -9.3% ROI, 47.5% WR (221 bets) ❌ Negative
# PASSING_YARDS:    70%+ = -7.8% ROI, 48.3% WR (145 bets) ❌ Negative
#
# TWO-TIER THRESHOLD SYSTEM (updated 2025-12-08):
# - FEATURED_THRESHOLDS: High thresholds for "Top Picks" (best ROI per market)
# - ALL_PICKS_THRESHOLD: Relaxed 50% for showing all other picks
FEATURED_THRESHOLDS = {
    'player_receptions': 0.55,       # Lowered to get featured picks (model confidence ~52-60%)
    'player_reception_yds': 0.55,    # Lowered to match receptions
    'player_rush_yds': 0.55,         # Moderate threshold
    'player_rush_attempts': 0.55,    # Moderate threshold
    'player_pass_yds': 0.60,         # Higher bar for pass yards
    'player_pass_attempts': 0.55,    # Moderate threshold
    'player_pass_completions': 0.55, # Moderate threshold
}
DEFAULT_FEATURED_THRESHOLD = 0.60  # Fallback for unknown markets
ALL_PICKS_THRESHOLD = 0.50  # Minimum for showing any pick

# Legacy alias
MARKET_THRESHOLDS = FEATURED_THRESHOLDS
DEFAULT_THRESHOLD = ALL_PICKS_THRESHOLD

# Markets with historically profitable thresholds
PROFITABLE_MARKETS = ['player_receptions']  # Only receptions showed consistent positive ROI
BREAKEVEN_MARKETS = ['player_reception_yds']  # Near break-even
NEGATIVE_ROI_MARKETS = ['player_rush_yds', 'player_pass_yds']  # Negative ROI

# Legacy constants for compatibility
VALIDATED_MARKETS = PROFITABLE_MARKETS + BREAKEVEN_MARKETS
ALL_MODEL_MARKETS = SUPPORTED_MARKETS  # From centralized config
EXCLUDED_MARKETS = ['player_pass_yds']  # -7.8% ROI at best - avoid

# NFL Team Logo URLs (ESPN CDN)
TEAM_LOGOS = {
    'ARI': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
    'ATL': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
    'BAL': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
    'BUF': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
    'CAR': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
    'CHI': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
    'CIN': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
    'CLE': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
    'DAL': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
    'DEN': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
    'DET': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
    'GB': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
    'HOU': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
    'IND': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
    'JAX': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
    'KC': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
    'LAC': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
    'LAR': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
    'LV': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
    'MIA': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
    'MIN': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
    'NE': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
    'NO': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
    'NYG': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
    'NYJ': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
    'PHI': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
    'PIT': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
    'SEA': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
    'SF': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
    'TB': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
    'TEN': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
    'WAS': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
}

# NFL Team Colors (primary, secondary)
TEAM_COLORS = {
    'ARI': ('#97233F', '#000000'),
    'ATL': ('#A71930', '#000000'),
    'BAL': ('#241773', '#9E7C0C'),
    'BUF': ('#00338D', '#C60C30'),
    'CAR': ('#0085CA', '#101820'),
    'CHI': ('#0B162A', '#C83803'),
    'CIN': ('#FB4F14', '#000000'),
    'CLE': ('#311D00', '#FF3C00'),
    'DAL': ('#003594', '#869397'),
    'DEN': ('#FB4F14', '#002244'),
    'DET': ('#0076B6', '#B0B7BC'),
    'GB': ('#203731', '#FFB612'),
    'HOU': ('#03202F', '#A71930'),
    'IND': ('#002C5F', '#A2AAAD'),
    'JAX': ('#006778', '#9F792C'),
    'KC': ('#E31837', '#FFB81C'),
    'LAC': ('#0080C6', '#FFC20E'),
    'LAR': ('#003594', '#FFA300'),
    'LV': ('#000000', '#A5ACAF'),
    'MIA': ('#008E97', '#FC4C02'),
    'MIN': ('#4F2683', '#FFC62F'),
    'NE': ('#002244', '#C60C30'),
    'NO': ('#D3BC8D', '#101820'),
    'NYG': ('#0B2265', '#A71930'),
    'NYJ': ('#125740', '#FFFFFF'),
    'PHI': ('#004C54', '#A5ACAF'),
    'PIT': ('#FFB612', '#101820'),
    'SEA': ('#002244', '#69BE28'),
    'SF': ('#AA0000', '#B3995D'),
    'TB': ('#D50A0A', '#34302B'),
    'TEN': ('#0C2340', '#4B92DB'),
    'WAS': ('#5A1414', '#FFB612'),
}


def get_team_logo_html(team: str, size: int = 20) -> str:
    """Get HTML for team logo image."""
    logo_url = TEAM_LOGOS.get(team.upper(), '')
    if logo_url:
        return f'<img src="{logo_url}" alt="{team}" class="team-logo" style="width: {size}px; height: {size}px;">'
    return f'<span class="team-abbr">{team}</span>'


def parse_game_string(game: str) -> tuple:
    """Parse game string like 'Seattle Seahawks @ Atlanta Falcons' into away/home teams."""
    if not game or pd.isna(game):
        return None, None

    # Handle format: "Team Name @ Team Name"
    if ' @ ' in game:
        parts = game.split(' @ ')
        away = parts[0].strip()
        home = parts[1].strip()
        return away, home
    return None, None


def get_team_abbrev(team_name: str) -> str:
    """Convert full team name to abbreviation."""
    team_map = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
    }
    return team_map.get(team_name, team_name[:3].upper() if team_name else '')


def generate_mini_prop_card(row: pd.Series, format_prop_fn) -> str:
    """Generate a Mini Prop Card for featured picks within game cards."""
    import json
    import urllib.parse

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    market = safe_str(row.get('market', ''), '')
    market_display = row.get('market_display', format_prop_fn(market))
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    week = row.get('week', 15)

    # Get game history (needed for chart AND projection calculation)
    game_history = get_player_game_history(player, market, int(week) if pd.notna(week) else 15)

    # Calculate projection (get_projection now handles game_history fallback internally)
    projection = get_projection(row, market, game_history)

    # Determine pick direction styling
    pick_class = 'pick-over' if 'OVER' in pick else 'pick-under'

    # Confidence level styling (uses centralized thresholds)
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    if conf_pct >= CONF_THRESHOLD_ELITE:
        conf_class = 'conf-elite'
    elif conf_pct >= CONF_THRESHOLD_HIGH:
        conf_class = 'conf-high'
    else:
        conf_class = 'conf-standard'

    # Team logo
    team_logo = get_team_logo_html(team, size=24)

    # Format edge
    edge_val = edge if not pd.isna(edge) else 0
    edge_display = f"+{edge_val:.1f}%" if edge_val > 0 else f"{edge_val:.1f}%"

    # game_history already loaded above for projection calculation

    # Extract opponent from game string
    game = safe_str(row.get('game', ''), '')
    team_abbr = safe_str(row.get('team_abbr', ''), '')
    opponent_abbr = safe_str(row.get('opponent_abbr', ''), '') or (extract_opponent_from_game(game, team_abbr) if game and team_abbr else '')

    # Build pick data for modal (uses module-level safe_float/safe_int)
    pick_data = {
        'player': player,
        'position': position,
        'team': team,
        'opponent': opponent_abbr or safe_str(row.get('opponent', ''), ''),
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': safe_float(line),
        'confidence': safe_float(confidence, 0.5),
        'edge': safe_float(edge_val),
        'projection': safe_float(projection),
        'trailing_avg': safe_float(row.get('trailing_stat', 0)),
        'hist_over_rate': safe_float(row.get('hist_over_rate', 0)),
        'hist_count': safe_int(row.get('hist_count', 0)),
        'def_epa': safe_float(row.get('opponent_def_epa', 0)),
        'snap_share': safe_float(row.get('snap_share', 0)),
        'game': safe_str(row.get('game', ''), ''),
        'lvt': safe_float(row.get('line_vs_trailing', 0)),
        'game_history': game_history,  # Last 6 games for chart
        # Edge model stats
        'source': safe_str(row.get('source', ''), 'EDGE'),
        'lvt_confidence': safe_float(row.get('lvt_confidence', 0)),
        'player_bias_confidence': safe_float(row.get('player_bias_confidence', 0)),
        'player_under_rate': safe_float(row.get('player_under_rate', 0)),
        'player_bet_count': safe_int(row.get('player_bet_count', 0)),
        'current_season_under_rate': safe_float(row.get('current_season_under_rate', 0.5)),
        'season_games_played': safe_int(row.get('season_games_played', 0)),
        # TD Poisson stats
        'expected_tds': safe_float(row.get('expected_tds', 0)),
        'p_over': safe_float(row.get('p_over', 0)),
        'p_under': safe_float(row.get('p_under', 0)),
        # Volume stats (if available)
        'targets_mean': safe_float(row.get('targets_mean', 0)),
        'receptions_mean': safe_float(row.get('receptions_mean', 0)),
        'receiving_yards_mean': safe_float(row.get('receiving_yards_mean', 0)),
        'rushing_attempts_mean': safe_float(row.get('rushing_attempts_mean', 0)),
        'rushing_yards_mean': safe_float(row.get('rushing_yards_mean', 0)),
        'passing_attempts_mean': safe_float(row.get('passing_attempts_mean', 0)),
        'passing_yards_mean': safe_float(row.get('passing_yards_mean', 0)),
        'redzone_target_share': safe_float(row.get('redzone_target_share', 0)),
        'goalline_carry_share': safe_float(row.get('goalline_carry_share', 0)),
        'game_script': safe_float(row.get('game_script_dynamic', 0)),
        # Weather data
        'weather_temp': safe_float(row.get('temperature', 0)),
        'weather_wind': safe_float(row.get('wind_speed', 0)),
        'weather_is_dome': bool(row.get('is_dome', False)),
        'weather_conditions': safe_str(row.get('conditions', ''), ''),
        'weather_severity': safe_str(row.get('severity', 'None'), 'None'),
        'weather_passing_adj': safe_float(row.get('passing_adjustment', 0)),
        # Defensive ranking
        'def_rank': safe_int(row.get('def_rank', 16)),
        'def_rank_type': safe_str(row.get('def_rank_type', 'pass'), 'pass'),
    }
    pick_data_encoded = urllib.parse.quote(json.dumps(pick_data))

    # Show useful stats based on data availability (projection removed - misleading per backtest)
    trailing_stat = row.get('trailing_stat', 0) or 0
    trailing_display = f"Avg: {trailing_stat:.1f}" if trailing_stat > 0 else ""
    stats_line = trailing_display or f"{conf_pct:.0f}% conf"

    return f'''
    <div class="mini-prop-card {pick_class}" onclick="openPickModal('{pick_data_encoded}')">
        <div class="mini-prop-header">
            <div class="mini-prop-player">
                {team_logo}
                <span class="mini-prop-name">{player}</span>
                <span class="mini-prop-pos">{position}</span>
            </div>
            <div class="mini-prop-pick {pick_class}">{pick}</div>
        </div>
        <div class="mini-prop-details">
            <span class="mini-prop-market">{market_display}</span>
            <span class="mini-prop-line">{line}</span>
        </div>
        <div class="mini-prop-stats">
            <span class="mini-prop-conf {conf_class}">{conf_pct:.0f}%</span>
            <span class="mini-prop-edge">{edge_display} edge</span>
            <span class="mini-prop-proj">{stats_line}</span>
        </div>
    </div>
    '''


def generate_game_card(game: str, game_recs: pd.DataFrame, format_prop_fn, game_info: dict = None) -> str:
    """Generate a COMPACT collapsible Game Card for the By Game view.

    NEW: Single-line collapsed header, expands on click to show picks.
    Target: 40px collapsed height.

    Args:
        game: Game string like "Seattle Seahawks @ Atlanta Falcons"
        game_recs: DataFrame of recommendations for this game
        format_prop_fn: Function to format prop type display
        game_info: Optional dict with spread_line, total_line, is_primetime_game, roof, etc.
    """
    away_team, home_team = parse_game_string(game)

    if not away_team or not home_team:
        return ''

    away_abbr = get_team_abbrev(away_team)
    home_abbr = get_team_abbrev(home_team)

    # Get team colors for subtle gradient
    away_color = TEAM_COLORS.get(away_abbr, ('#333', '#666'))[0]
    home_color = TEAM_COLORS.get(home_abbr, ('#333', '#666'))[0]

    # Get compact logos (20px)
    away_logo = get_team_logo_html(away_abbr, size=20)
    home_logo = get_team_logo_html(home_abbr, size=20)

    # Extract game context from first row if available
    first_row = game_recs.iloc[0] if len(game_recs) > 0 else {}
    is_primetime = game_info.get('is_primetime_game') if game_info else first_row.get('is_primetime_game', False)
    commence_time = game_info.get('commence_time') if game_info else first_row.get('commence_time', '')

    # Format game time (compact)
    game_time_display = format_game_time(commence_time) if commence_time else ''

    # Extract weather from first row
    temperature = first_row.get('temperature', None)
    wind_speed = first_row.get('wind_speed', None)
    is_dome = first_row.get('is_dome', False)

    # Build weather badge
    weather_badge = ''
    if is_dome:
        weather_badge = '<span class="weather-badge dome">Dome</span>'
    elif temperature is not None and not pd.isna(temperature):
        temp_display = f"{temperature:.0f}°F"
        wind_display = f" {wind_speed:.0f}mph" if wind_speed and not pd.isna(wind_speed) and wind_speed > 0 else ""
        weather_badge = f'<span class="weather-badge">{temp_display}{wind_display}</span>'

    # Primetime badge (compact)
    primetime_badge = '<span class="primetime-badge">PT</span>' if is_primetime else ''

    # Total picks count
    total_picks = len(game_recs)

    # Generate game card ID
    game_id = f"gc-{away_abbr}-{home_abbr}".lower()

    # Build compact header: [Logo] AWAY @ [Logo] HOME | Time | Picks | Toggle
    return f'''
    <div class="game-card" id="{game_id}">
        <div class="game-card-header" onclick="toggleGameCard('{game_id}')" style="background: linear-gradient(90deg, {away_color}08 0%, {home_color}08 100%);">
            <div class="game-header-left">
                <div class="game-header-teams">
                    {away_logo}
                    <span>{away_abbr}</span>
                    <span class="game-header-at">@</span>
                    {home_logo}
                    <span>{home_abbr}</span>
                </div>
                {primetime_badge}
                {weather_badge}
            </div>
            <div class="game-header-right">
                <span class="game-time">{game_time_display}</span>
                <span class="game-picks-count">{total_picks} picks</span>
                <span class="game-toggle">▼</span>
            </div>
        </div>
        <div class="game-card-body">
            <div class="game-picks-grid">
    '''


def close_game_card() -> str:
    """Close the game card HTML structure."""
    return '''
            </div>
        </div>
    </div>
    '''


def generate_game_pick_card(row: pd.Series, format_prop_fn, view_prefix: str = 'gpc') -> str:
    """Generate a CARD for picks inside the By Game view (matches featured card style)."""
    import json
    import urllib.parse
    import re
    import hashlib

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    # Convert YES/NO to OVER/UNDER for TD props
    if pick == 'YES':
        pick = 'OVER'
    elif pick == 'NO':
        pick = 'UNDER'
    market = safe_str(row.get('market', ''), '')
    market_display = format_prop_fn(market)
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    tier = str(row.get('effective_tier', row.get('quality_tier', ''))).upper()
    week = row.get('week', 0)
    game = safe_str(row.get('game', ''), '')

    # Get kickoff time
    commence_time = row.get('commence_time', '')
    kickoff_display = ''
    if commence_time and not pd.isna(commence_time):
        try:
            from datetime import datetime
            if isinstance(commence_time, str):
                ct = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            else:
                ct = commence_time
            kickoff_display = ct.strftime('%a %I:%M%p').replace(' 0', ' ').lower()
        except:
            kickoff_display = ''

    # Get game history (needed for chart AND projection calculation)
    game_history = get_player_game_history(player, market, int(week) if pd.notna(week) else 15)

    # Calculate projection (get_projection now handles game_history fallback internally)
    projection = get_projection(row, market, game_history)
    odds = row.get('odds', -110)

    # Create stable bet ID
    bet_id = hashlib.md5(f"{player}_{market}_{line}_{pick}".encode()).hexdigest()[:12]

    # Pick styling
    pick_class = 'pick-over' if 'OVER' in pick else 'pick-under'
    arrow = '▲' if 'OVER' in pick else '▼'

    # Confidence styling
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    conf_class = 'conf-elite' if conf_pct >= CONF_THRESHOLD_ELITE else ('conf-high' if conf_pct >= CONF_THRESHOLD_HIGH else '')

    # Edge
    edge_val = edge if not pd.isna(edge) else 0
    edge_class = 'edge-positive' if edge_val > 0 else ''
    edge_display = f"+{edge_val:.1f}%" if edge_val > 0 else f"{edge_val:.1f}%"

    # Bet size suggestion ($2-$5 based on confidence and edge)
    bet_amount, bet_tier = get_bet_size_suggestion(confidence, edge_val)
    bet_class = f'bet-{bet_tier}'  # bet-high, bet-medium, bet-low
    bet_display = f"${bet_amount:.0f}" if bet_amount == int(bet_amount) else f"${bet_amount:.2f}"

    # Tier class
    if tier == 'ELITE':
        tier_class = 'tier-elite'
    elif tier == 'STRONG':
        tier_class = 'tier-strong'
    elif tier == 'MODERATE':
        tier_class = 'tier-moderate'
    elif tier == 'CAUTION':
        tier_class = 'tier-caution'
    else:
        tier_class = ''

    # Get edge model stats for display
    player_under_rate = row.get('player_under_rate', 0)
    player_bet_count = row.get('player_bet_count', 0)

    # Player headshot (with team logo fallback) - 56px for better recognition
    player_headshot = get_player_headshot_html(player, team, size=56)

    # Build model narrative
    model_factors_html = build_model_narrative(row, pick)
    # Use prose-only narrative for modal (excludes redundant stats bar)
    narrative_text = generate_prose_only(row)

    # game_history already loaded above for projection calculation

    # Trailing avg - use row data first, fallback to calculated from game_history
    trailing_avg = row.get('trailing_stat', 0)
    if (not trailing_avg or pd.isna(trailing_avg) or trailing_avg == 0) and game_history:
        # Map market to correct stat array in game_history
        market_to_stat = {
            'player_pass_tds': 'passing_tds',
            'player_rush_tds': 'rushing_tds',
            'player_reception_yds': 'receiving_yards',
            'player_receiving_yards': 'receiving_yards',
            'player_receptions': 'receptions',
            'player_rush_yds': 'rushing_yards',
            'player_rushing_yards': 'rushing_yards',
            'player_rush_attempts': 'rushing_attempts',
            'player_pass_yds': 'passing_yards',
            'player_passing_yards': 'passing_yards',
            'player_pass_attempts': 'passing_attempts',
            'player_pass_completions': 'completions',
        }
        stat_key = market_to_stat.get(market, 'receiving_yards')
        stats = game_history.get(stat_key, [])
        if stats:
            valid_stats = [s for s in stats if s is not None and not (isinstance(s, float) and pd.isna(s))]
            if valid_stats:
                trailing_avg = sum(valid_stats) / len(valid_stats)

    # Trailing avg display
    trailing_display = f"{trailing_avg:.1f}" if trailing_avg and not pd.isna(trailing_avg) else "—"

    # Hit rate display: show the rate that matches the pick direction with color coding
    if player_under_rate and not pd.isna(player_under_rate) and player_bet_count and player_bet_count >= 5:
        if 'UNDER' in pick:
            hit_pct = player_under_rate * 100
            hit_rate_display = f"{hit_pct:.0f}%↓"
        else:
            hit_pct = (1 - player_under_rate) * 100
            hit_rate_display = f"{hit_pct:.0f}%↑"
        # Color code: green for >= 50%, red for < 50%
        hit_rate_class = 'positive' if hit_pct >= 50 else 'negative'
    else:
        hit_rate_display = "—"
        hit_rate_class = ''

    # Extract opponent from game string
    game = safe_str(row.get('game', ''), '')
    team_abbr = safe_str(row.get('team_abbr', ''), '')
    opponent_abbr = safe_str(row.get('opponent_abbr', ''), '') or (extract_opponent_from_game(game, team_abbr) if game and team_abbr else '')

    # Build pick data for modal (includes edge model stats)
    pick_data = {
        'player': player,
        'position': position,
        'team': team,
        'opponent': opponent_abbr or safe_str(row.get('opponent', ''), ''),
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': safe_float(line),
        'confidence': safe_float(confidence, 0.5),
        'edge': safe_float(edge_val),
        'projection': safe_float(projection),
        'trailing_avg': safe_float(trailing_avg),  # Use calculated value, not row
        'hist_over_rate': safe_float(row.get('hist_over_rate', 0)),
        'hist_count': safe_int(row.get('hist_count', 0)),
        'def_epa': safe_float(row.get('opponent_def_epa', 0)),
        'snap_share': safe_float(row.get('snap_share', 0)),
        'game': safe_str(row.get('game', ''), ''),
        'lvt': safe_float(row.get('line_vs_trailing', 0)),
        'game_history': game_history,  # Last 6 games for chart
        # Edge model stats
        'source': safe_str(row.get('source', ''), 'EDGE'),
        'lvt_confidence': safe_float(row.get('lvt_confidence', 0)),
        'player_bias_confidence': safe_float(row.get('player_bias_confidence', 0)),
        'player_under_rate': safe_float(row.get('player_under_rate', 0)),
        'player_bet_count': safe_int(row.get('player_bet_count', 0)),
        'current_season_under_rate': safe_float(row.get('current_season_under_rate', 0.5)),
        'season_games_played': safe_int(row.get('season_games_played', 0)),
        # TD Poisson stats
        'expected_tds': safe_float(row.get('expected_tds', 0)),
        'p_over': safe_float(row.get('p_over', 0)),
        'p_under': safe_float(row.get('p_under', 0)),
        # Volume stats (if available)
        'targets_mean': safe_float(row.get('targets_mean', 0)),
        'receptions_mean': safe_float(row.get('receptions_mean', 0)),
        'receiving_yards_mean': safe_float(row.get('receiving_yards_mean', 0)),
        'rushing_attempts_mean': safe_float(row.get('rushing_attempts_mean', 0)),
        'rushing_yards_mean': safe_float(row.get('rushing_yards_mean', 0)),
        'passing_attempts_mean': safe_float(row.get('passing_attempts_mean', 0)),
        'passing_yards_mean': safe_float(row.get('passing_yards_mean', 0)),
        'redzone_target_share': safe_float(row.get('redzone_target_share', 0)),
        'goalline_carry_share': safe_float(row.get('goalline_carry_share', 0)),
        'game_script': safe_float(row.get('game_script_dynamic', 0)),
        # Weather data
        'weather_temp': safe_float(row.get('temperature', 0)),
        'weather_wind': safe_float(row.get('wind_speed', 0)),
        'weather_is_dome': bool(row.get('is_dome', False)),
        'weather_conditions': safe_str(row.get('conditions', ''), ''),
        'weather_severity': safe_str(row.get('severity', 'None'), 'None'),
        'weather_passing_adj': safe_float(row.get('passing_adjustment', 0)),
        # Defensive ranking
        'def_rank': safe_int(row.get('def_rank', 16)),
        'def_rank_type': safe_str(row.get('def_rank_type', 'pass'), 'pass'),
    }
    pick_data_encoded = urllib.parse.quote(json.dumps(pick_data))

    # Format player name compactly
    display_name = format_player_name_compact(player, max_len=12)

    # Generate matchup context for player props (not game lines)
    matchup_html = ''
    if position != 'GAME':
        matchup_html = generate_matchup_context_html(row, bet_id, int(week) if pd.notna(week) else 15, prefix=f'{view_prefix}-')

    # Bet data for selection
    bet_data = {
        'id': bet_id,
        'player': player,
        'team': team,
        'position': position,
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': float(line) if pd.notna(line) else 0,
        'odds': int(odds) if pd.notna(odds) else -110,
        'confidence': round(conf_pct, 1),
        'edge': round(edge_val, 1),
        'projection': round(float(projection), 1) if pd.notna(projection) else 0,
        'week': int(week) if pd.notna(week) else 0,
        'game': game,
        'tier': tier
    }
    bet_data_json = json.dumps(bet_data).replace("'", "\\'")

    # Tier badge text
    tier_badge = tier if tier else 'MODERATE'
    tier_badge_class = f"tier-badge-{tier.lower()}" if tier else "tier-badge-moderate"

    return f'''
    <div class="game-pick-card {tier_class}" data-bet-id="{bet_id}" onclick="openPickModal('{pick_data_encoded}')">
        <!-- Top Row: Checkbox + Tier Badge -->
        <div class="card-top-bar">
            <label class="bet-checkbox-container" onclick="event.stopPropagation();">
                <input type="checkbox" class="bet-checkbox" id="bet-{bet_id}" name="bet-{bet_id}" data-bet='{bet_data_json}' onchange="toggleBetSelection(this)">
                <span class="bet-checkmark"></span>
            </label>
            <span class="card-tier-badge {tier_badge_class}">{tier_badge}</span>
        </div>

        <!-- HERO: The Pick Decision -->
        <div class="card-hero {pick_class}">
            <span class="card-hero-arrow">{arrow}</span>
            <div class="card-hero-content">
                <span class="card-hero-pick">{pick} {line}</span>
                <span class="card-hero-market">{market_display}</span>
            </div>
        </div>

        <!-- Player Info Row -->
        <div class="card-player-row">
            <div class="card-player-avatar">
                {player_headshot}
            </div>
            <div class="card-player-details">
                <span class="card-player-name" title="{player}">{player}</span>
                <span class="card-player-meta">{team} · {position}{' · ' + kickoff_display if kickoff_display else ''}</span>
            </div>
        </div>

        <!-- Stats Row - Readable -->
        <div class="card-stats-row">
            <div class="card-stat-item">
                <span class="card-stat-value {conf_class}">{conf_pct:.0f}%</span>
                <span class="card-stat-label">Confidence</span>
            </div>
            <div class="card-stat-divider"></div>
            <div class="card-stat-item">
                <span class="card-stat-value {edge_class}">{edge_display}</span>
                <span class="card-stat-label">Edge</span>
            </div>
            <div class="card-stat-divider"></div>
            <div class="card-stat-item">
                <span class="card-stat-value {bet_class}">{bet_display}</span>
                <span class="card-stat-label">Bet Size</span>
            </div>
        </div>
        {matchup_html}
    </div>
    '''


def generate_compact_game_pick_row(row: pd.Series, format_prop_fn) -> str:
    """Generate a COMPACT pick row for inside game cards (28px height)."""
    import json
    import urllib.parse
    import hashlib

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    # Convert YES/NO to OVER/UNDER for TD props
    if pick == 'YES':
        pick = 'OVER'
    elif pick == 'NO':
        pick = 'UNDER'
    market = safe_str(row.get('market', ''), '')
    market_display = format_prop_fn(market)
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    week = row.get('week', 0)
    game = safe_str(row.get('game', ''), '')
    odds = row.get('odds', -110)
    projection = get_projection(row, market)

    # Create stable bet ID
    bet_id = hashlib.md5(f"{player}_{market}_{line}_{pick}".encode()).hexdigest()[:12]

    # Pick styling
    pick_class = 'pick-over' if 'OVER' in pick else 'pick-under'
    arrow = '▲' if 'OVER' in pick else '▼'

    # Confidence styling
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    if conf_pct >= CONF_THRESHOLD_ELITE:
        conf_class = 'conf-elite'
    elif conf_pct >= CONF_THRESHOLD_HIGH:
        conf_class = 'conf-high'
    else:
        conf_class = 'conf-standard'

    # Edge
    edge_val = edge if not pd.isna(edge) else 0
    edge_class = 'edge-pos' if edge_val > 0 else 'edge-neg' if edge_val < 0 else ''
    edge_display = f"+{edge_val:.1f}%" if edge_val > 0 else f"{edge_val:.1f}%"

    # Get game history for chart visualization
    game_history = get_player_game_history(player, market, int(week) if pd.notna(week) else 15)

    # Extract opponent from game string
    game = safe_str(row.get('game', ''), '')
    team_abbr = safe_str(row.get('team_abbr', ''), '')
    opponent_abbr = safe_str(row.get('opponent_abbr', ''), '') or (extract_opponent_from_game(game, team_abbr) if game and team_abbr else '')

    # Build pick data for modal (includes edge model stats)
    pick_data = {
        'player': player,
        'position': position,
        'team': team,
        'opponent': opponent_abbr or safe_str(row.get('opponent', ''), ''),
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': safe_float(line),
        'confidence': safe_float(confidence, 0.5),
        'edge': safe_float(edge_val),
        'projection': get_projection(row, market),
        'trailing_avg': safe_float(row.get('trailing_stat', 0)),
        'hist_over_rate': safe_float(row.get('hist_over_rate', 0)),
        'hist_count': safe_int(row.get('hist_count', 0)),
        'def_epa': safe_float(row.get('opponent_def_epa', 0)),
        'snap_share': safe_float(row.get('snap_share', 0)),
        'game': game,
        'lvt': safe_float(row.get('line_vs_trailing', 0)),
        'game_history': game_history,
        # Edge model stats
        'source': safe_str(row.get('source', ''), 'EDGE'),
        'lvt_confidence': safe_float(row.get('lvt_confidence', 0)),
        'player_bias_confidence': safe_float(row.get('player_bias_confidence', 0)),
        'player_under_rate': safe_float(row.get('player_under_rate', 0)),
        'player_bet_count': safe_int(row.get('player_bet_count', 0)),
        'current_season_under_rate': safe_float(row.get('current_season_under_rate', 0.5)),
        'season_games_played': safe_int(row.get('season_games_played', 0)),
        # TD Poisson stats
        'expected_tds': safe_float(row.get('expected_tds', 0)),
        'p_over': safe_float(row.get('p_over', 0)),
        'p_under': safe_float(row.get('p_under', 0)),
        # Volume stats (if available)
        'targets_mean': safe_float(row.get('targets_mean', 0)),
        'receptions_mean': safe_float(row.get('receptions_mean', 0)),
        'receiving_yards_mean': safe_float(row.get('receiving_yards_mean', 0)),
        'rushing_attempts_mean': safe_float(row.get('rushing_attempts_mean', 0)),
        'rushing_yards_mean': safe_float(row.get('rushing_yards_mean', 0)),
        'passing_attempts_mean': safe_float(row.get('passing_attempts_mean', 0)),
        'passing_yards_mean': safe_float(row.get('passing_yards_mean', 0)),
        'redzone_target_share': safe_float(row.get('redzone_target_share', 0)),
        'goalline_carry_share': safe_float(row.get('goalline_carry_share', 0)),
        'game_script': safe_float(row.get('game_script_dynamic', 0)),
        # Weather data
        'weather_temp': safe_float(row.get('temperature', 0)),
        'weather_wind': safe_float(row.get('wind_speed', 0)),
        'weather_is_dome': bool(row.get('is_dome', False)),
        'weather_conditions': safe_str(row.get('conditions', ''), ''),
        'weather_severity': safe_str(row.get('severity', 'None'), 'None'),
        'weather_passing_adj': safe_float(row.get('passing_adjustment', 0)),
        # Defensive ranking
        'def_rank': safe_int(row.get('def_rank', 16)),
        'def_rank_type': safe_str(row.get('def_rank_type', 'pass'), 'pass'),
    }
    pick_data_encoded = urllib.parse.quote(json.dumps(pick_data))

    # Bet data for selection
    bet_data = {
        'id': bet_id,
        'player': player,
        'team': team,
        'position': position,
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': float(line) if pd.notna(line) else 0,
        'odds': int(odds) if pd.notna(odds) else -110,
        'confidence': round(conf_pct, 1),
        'edge': round(edge_val, 1),
        'projection': round(float(projection), 1) if pd.notna(projection) else 0,
        'week': int(week) if pd.notna(week) else 0,
        'game': game,
        'tier': ''
    }
    bet_data_json = json.dumps(bet_data).replace("'", "\\'")

    return f'''
    <div class="pick-row-wrapper" data-bet-id="{bet_id}">
        <div class="pick-row-left">
            <label class="bet-checkbox-container" onclick="event.stopPropagation();">
                <input type="checkbox" class="bet-checkbox" id="bet-{bet_id}" name="bet-{bet_id}" data-bet='{bet_data_json}' onchange="toggleBetSelection(this)">
                <span class="bet-checkmark"></span>
            </label>
            <span class="pick-row-player" onclick="openPickModal('{pick_data_encoded}')">{player}</span>
            <span class="pick-row-pos">{position}</span>
            <span class="pick-row-divider">|</span>
            <span class="pick-row-market">{market_display}</span>
        </div>
        <div class="pick-row-right" onclick="openPickModal('{pick_data_encoded}')">
            <span class="pick-row-direction {pick_class}">{arrow} {pick}</span>
            <span class="pick-row-line">{line}</span>
            <span class="pick-row-conf {conf_class}">{conf_pct:.0f}%</span>
            <span class="pick-row-edge {edge_class}">{edge_display}</span>
        </div>
    </div>
    '''


def generate_game_pick_row(row: pd.Series, format_prop_fn) -> str:
    """Generate an expandable row for game card details with breakdown info."""
    import json
    import urllib.parse
    import uuid
    import hashlib

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    market = safe_str(row.get('market', ''), '')
    market_display = row.get('market_display', format_prop_fn(market))
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    projection = get_projection(row, market)
    trailing_stat = row.get('trailing_stat', 0)
    hist_over_rate = row.get('hist_over_rate', 0)
    hist_count = row.get('hist_count', 0)
    snap_share = row.get('snap_share', 0)
    tier = str(row.get('effective_tier', row.get('quality_tier', ''))).upper()
    def_epa = row.get('opponent_def_epa', 0)
    market_roi_tier = row.get('market_roi_tier', 'unknown')
    week = row.get('week', 0)
    game = safe_str(row.get('game', ''), '')
    odds = row.get('odds', -110)

    row_id = str(uuid.uuid4())[:8]
    # Create stable bet ID from player+market+line for persistence
    bet_id = hashlib.md5(f"{player}_{market}_{line}_{pick}".encode()).hexdigest()[:12]
    pick_class = 'pick-over' if 'OVER' in pick else 'pick-under'
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    edge_val = edge if not pd.isna(edge) else 0
    trailing_val = trailing_stat if pd.notna(trailing_stat) and trailing_stat != 0 else None
    snap_pct = (snap_share * 100) if pd.notna(snap_share) and snap_share != 0 else None

    # Confidence styling (uses centralized thresholds)
    conf_class = 'conf-elite' if conf_pct >= CONF_THRESHOLD_ELITE else ('conf-high' if conf_pct >= CONF_THRESHOLD_HIGH else '')

    # Edge styling
    edge_class = 'edge-pos' if edge_val > 0 else ''
    edge_display = f"+{edge_val:.1f}" if edge_val > 0 else f"{edge_val:.1f}"

    # Tier badge - show all tiers
    tier_html = ''
    if tier and tier not in ['N/A', 'NAN', '']:
        if tier == 'ELITE':
            tier_class = 'tier-elite'
        elif tier == 'STRONG':
            tier_class = 'tier-strong'
        elif tier == 'CAUTION':
            tier_class = 'tier-caution'
        elif tier == 'MODERATE':
            tier_class = 'tier-moderate'
        else:
            tier_class = 'tier-spec'
        tier_html = f'<span class="row-tier {tier_class}">{tier}</span>'

    team_logo = get_team_logo_html(team, size=18)

    # ROI tier indicator for the row (small dot)
    roi_dot = ''
    if market_roi_tier == 'profitable':
        roi_dot = '<span class="roi-dot roi-profitable" title="Profitable market (+11.6% ROI)">●</span>'
    elif market_roi_tier == 'breakeven':
        roi_dot = '<span class="roi-dot roi-breakeven" title="Break-even market">●</span>'
    elif market_roi_tier == 'negative':
        roi_dot = '<span class="roi-dot roi-negative" title="Negative ROI market">●</span>'

    # Build expanded breakdown content
    breakdown_items = []

    # Model confidence (removed projection - backtest shows it's misleading)
    conf_val = row.get('confidence', row.get('model_p_under', 0.5))
    if isinstance(conf_val, (int, float)) and conf_val > 0:
        conf_pct_val = conf_val * 100 if conf_val <= 1 else conf_val
        conf_class = 'positive' if conf_pct_val >= 55 else 'neutral'
        breakdown_items.append(f'<div class="breakdown-item"><span class="breakdown-label">Model Confidence</span><span class="breakdown-value {conf_class}">{conf_pct_val:.0f}%</span></div>')

    # Trailing average
    if trailing_val is not None:
        diff = trailing_val - line if pd.notna(line) else 0
        diff_class = 'positive' if (diff > 0 and 'OVER' in pick) or (diff < 0 and 'UNDER' in pick) else 'negative'
        breakdown_items.append(f'<div class="breakdown-item"><span class="breakdown-label">4-Game Avg</span><span class="breakdown-value">{trailing_val:.1f} <span class="{diff_class}">({diff:+.1f})</span></span></div>')

    # Historical hit rate
    if pd.notna(hist_over_rate) and pd.notna(hist_count) and hist_count >= 3:
        is_under = 'UNDER' in pick
        hit_rate = (1 - hist_over_rate) if is_under else hist_over_rate
        hit_class = 'positive' if hit_rate >= 0.5 else 'negative'
        breakdown_items.append(f'<div class="breakdown-item"><span class="breakdown-label">Historical</span><span class="breakdown-value {hit_class}">{hit_rate*100:.0f}% hit rate ({int(hist_count)} games)</span></div>')

    # Snap share
    if snap_pct is not None and snap_pct > 0:
        snap_class = 'positive' if snap_pct >= 60 else ('negative' if snap_pct < 40 else '')
        breakdown_items.append(f'<div class="breakdown-item"><span class="breakdown-label">Snap Share</span><span class="breakdown-value {snap_class}">{snap_pct:.0f}%</span></div>')

    # Defense matchup
    if pd.notna(def_epa) and def_epa != 0:
        if def_epa > 0.02:
            def_text = "Weak defense (favorable)"
            def_class = 'positive' if 'OVER' in pick else 'negative'
        elif def_epa < -0.02:
            def_text = "Strong defense (tough)"
            def_class = 'negative' if 'OVER' in pick else 'positive'
        else:
            def_text = "Average defense"
            def_class = ''
        breakdown_items.append(f'<div class="breakdown-item"><span class="breakdown-label">Matchup</span><span class="breakdown-value {def_class}">{def_text}</span></div>')

    # Market historical ROI tier
    if market_roi_tier == 'profitable':
        roi_text = "+11.6% historical ROI"
        roi_class = 'positive'
    elif market_roi_tier == 'breakeven':
        roi_text = "Break-even market"
        roi_class = ''
    elif market_roi_tier == 'negative':
        roi_text = "Negative historical ROI"
        roi_class = 'negative'
    else:
        roi_text = ""
        roi_class = ''

    if roi_text:
        breakdown_items.append(f'<div class="breakdown-item"><span class="breakdown-label">Market ROI</span><span class="breakdown-value {roi_class}">{roi_text}</span></div>')

    breakdown_html = ''.join(breakdown_items) if breakdown_items else '<div class="breakdown-item"><span class="breakdown-label">No additional data</span></div>'

    # Build model narrative
    narrative_html = build_model_narrative(row, pick)

    # Build matchup context section
    matchup_html = generate_matchup_context_html(row, row_id, int(week) if pd.notna(week) else 15, prefix='gpr-')

    # Bet data for selection persistence (JSON encoded)
    bet_data = {
        'id': bet_id,
        'player': player,
        'team': team,
        'position': position,
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': float(line) if pd.notna(line) else 0,
        'odds': int(odds) if pd.notna(odds) else -110,
        'confidence': round(conf_pct, 1),
        'edge': round(edge_val, 1),
        'projection': round(float(projection), 1) if pd.notna(projection) else 0,
        'week': int(week) if pd.notna(week) else 0,
        'game': game,
        'tier': tier
    }
    bet_data_json = json.dumps(bet_data).replace("'", "\\'")

    return f'''
        <div class="pick-row-wrapper" data-bet-id="{bet_id}">
            <div class="pick-row {pick_class}">
                <div class="pick-row-left">
                    <label class="bet-checkbox-container" onclick="event.stopPropagation();">
                        <input type="checkbox" class="bet-checkbox" id="bet-{bet_id}" name="bet-{bet_id}" data-bet='{bet_data_json}' onchange="toggleBetSelection(this)">
                        <span class="bet-checkmark"></span>
                    </label>
                    {team_logo}
                    <span class="pick-row-player" onclick="togglePickRow('{row_id}')">{player}</span>
                    <span class="pick-row-pos">{position}</span>
                    <span class="pick-row-divider">|</span>
                    <span class="pick-row-market">{market_display}</span>
                    {roi_dot}
                </div>
                <div class="pick-row-right" onclick="togglePickRow('{row_id}')">
                    <span class="pick-row-direction {pick_class}">{pick}</span>
                    <span class="pick-row-line">{line}</span>
                    <span class="pick-row-divider">|</span>
                    <span class="pick-row-conf {conf_class}">{conf_pct:.0f}%</span>
                    <span class="pick-row-edge {edge_class}">{edge_display}%</span>
                    <span class="pick-row-divider">|</span>
                    {tier_html}
                    <span class="pick-row-chevron" id="chevron-{row_id}">▼</span>
                </div>
            </div>
            <div class="pick-row-expanded" id="expanded-{row_id}">
                {matchup_html}
                <div class="breakdown-grid">
                    {breakdown_html}
                </div>
                {narrative_html}
            </div>
        </div>
    '''


def generate_featured_pick_card(row: pd.Series, format_prop_fn) -> str:
    """Generate a featured pick card for the Top Picks visual section."""
    import json
    import urllib.parse
    import hashlib

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    # Convert YES/NO to OVER/UNDER for TD props
    if pick == 'YES':
        pick = 'OVER'
    elif pick == 'NO':
        pick = 'UNDER'
    market = safe_str(row.get('market', ''), '')
    market_display = row.get('market_display', format_prop_fn(market))
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    projection = get_projection(row, market)
    tier = str(row.get('effective_tier', row.get('quality_tier', ''))).upper()
    game = safe_str(row.get('game', ''), '')
    week = row.get('week', 0)
    odds = row.get('odds', -110)

    # Get kickoff time
    commence_time = row.get('commence_time', '')
    kickoff_display = ''
    if commence_time and not pd.isna(commence_time):
        try:
            from datetime import datetime
            if isinstance(commence_time, str):
                ct = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            else:
                ct = commence_time
            kickoff_display = ct.strftime('%a %I:%M%p').replace(' 0', ' ').lower()
        except:
            kickoff_display = ''

    # Create stable bet ID for persistence
    bet_id = hashlib.md5(f"{player}_{market}_{line}_{pick}".encode()).hexdigest()[:12]

    # Pick styling
    pick_class = 'pick-over' if 'OVER' in pick else 'pick-under'

    # Confidence styling (uses centralized thresholds)
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    conf_class = 'conf-elite' if conf_pct >= CONF_THRESHOLD_ELITE else ('conf-high' if conf_pct >= CONF_THRESHOLD_HIGH else '')

    # Edge styling
    edge_val = edge if not pd.isna(edge) else 0
    edge_class = 'edge-positive' if edge_val > 0 else ''
    edge_display = f"+{edge_val:.1f}%" if edge_val > 0 else f"{edge_val:.1f}%"

    # Tier styling
    tier_class = 'tier-elite' if tier == 'ELITE' else 'tier-strong'

    # Player headshot (with team logo fallback) - 56px for better recognition
    player_headshot = get_player_headshot_html(player, team, size=56)

    # Build model narrative FIRST so we can include it in pick_data
    model_factors_html = build_model_narrative(row, pick)

    # Use prose-only narrative for modal (excludes redundant stats bar)
    narrative_text = generate_prose_only(row)

    # Get game history for chart visualization
    game_history = get_player_game_history(player, market, int(week) if pd.notna(week) else 15)

    # Calculate trailing_avg from game_history based on market type
    trailing_avg_calculated = row.get('trailing_stat', 0)
    if (not trailing_avg_calculated or pd.isna(trailing_avg_calculated) or trailing_avg_calculated == 0) and game_history:
        market_to_stat = {
            'player_pass_tds': 'passing_tds',
            'player_rush_tds': 'rushing_tds',
            'player_reception_yds': 'receiving_yards',
            'player_receiving_yards': 'receiving_yards',
            'player_receptions': 'receptions',
            'player_rush_yds': 'rushing_yards',
            'player_rushing_yards': 'rushing_yards',
            'player_rush_attempts': 'rushing_attempts',
            'player_pass_yds': 'passing_yards',
            'player_passing_yards': 'passing_yards',
            'player_pass_attempts': 'passing_attempts',
            'player_pass_completions': 'completions',
        }
        stat_key = market_to_stat.get(market, 'receiving_yards')
        stats = game_history.get(stat_key, [])
        if stats:
            valid_stats = [s for s in stats if s is not None and not (isinstance(s, float) and pd.isna(s))]
            if valid_stats:
                trailing_avg_calculated = sum(valid_stats) / len(valid_stats)

    # Extract opponent from game string
    game = safe_str(row.get('game', ''), '')
    team_abbr = safe_str(row.get('team_abbr', ''), '')
    opponent_abbr = safe_str(row.get('opponent_abbr', ''), '') or (extract_opponent_from_game(game, team_abbr) if game and team_abbr else '')

    # Build pick data for modal (uses module-level safe_float/safe_int)
    pick_data = {
        'player': player,
        'position': position,
        'team': team,
        'opponent': opponent_abbr or safe_str(row.get('opponent', ''), ''),
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': safe_float(line),
        'confidence': safe_float(confidence, 0.5),
        'edge': safe_float(edge_val),
        'projection': safe_float(projection),
        'trailing_avg': safe_float(trailing_avg_calculated),
        'hist_over_rate': safe_float(row.get('hist_over_rate', 0)),
        'hist_count': safe_int(row.get('hist_count', 0)),
        'def_epa': safe_float(row.get('opponent_def_epa', 0)),
        'snap_share': safe_float(row.get('snap_share', 0)),
        'game': game,
        'lvt': safe_float(row.get('line_vs_trailing', 0)),
        'game_history': game_history,  # Last 6 games for chart
        # Edge model stats
        'source': safe_str(row.get('source', ''), 'EDGE'),
        'lvt_confidence': safe_float(row.get('lvt_confidence', 0)),
        'player_bias_confidence': safe_float(row.get('player_bias_confidence', 0)),
        'player_under_rate': safe_float(row.get('player_under_rate', 0)),
        'player_bet_count': safe_int(row.get('player_bet_count', 0)),
        'current_season_under_rate': safe_float(row.get('current_season_under_rate', 0.5)),
        'season_games_played': safe_int(row.get('season_games_played', 0)),
        # TD Poisson stats
        'expected_tds': safe_float(row.get('expected_tds', 0)),
        'p_over': safe_float(row.get('p_over', 0)),
        'p_under': safe_float(row.get('p_under', 0)),
        # Volume stats (if available)
        'targets_mean': safe_float(row.get('targets_mean', 0)),
        'receptions_mean': safe_float(row.get('receptions_mean', 0)),
        'receiving_yards_mean': safe_float(row.get('receiving_yards_mean', 0)),
        'rushing_attempts_mean': safe_float(row.get('rushing_attempts_mean', 0)),
        'rushing_yards_mean': safe_float(row.get('rushing_yards_mean', 0)),
        'passing_attempts_mean': safe_float(row.get('passing_attempts_mean', 0)),
        'passing_yards_mean': safe_float(row.get('passing_yards_mean', 0)),
        'redzone_target_share': safe_float(row.get('redzone_target_share', 0)),
        'goalline_carry_share': safe_float(row.get('goalline_carry_share', 0)),
        'game_script': safe_float(row.get('game_script_dynamic', 0)),
        # Weather data
        'weather_temp': safe_float(row.get('temperature', 0)),
        'weather_wind': safe_float(row.get('wind_speed', 0)),
        'weather_is_dome': bool(row.get('is_dome', False)),
        'weather_conditions': safe_str(row.get('conditions', ''), ''),
        'weather_severity': safe_str(row.get('severity', 'None'), 'None'),
        'weather_passing_adj': safe_float(row.get('passing_adjustment', 0)),
        # Defensive ranking
        'def_rank': safe_int(row.get('def_rank', 16)),
        'def_rank_type': safe_str(row.get('def_rank_type', 'pass'), 'pass'),
    }
    pick_data_encoded = urllib.parse.quote(json.dumps(pick_data))

    # Arrow indicator for pick direction
    arrow = '▲' if 'OVER' in pick else '▼'

    # Get edge model stats for display
    player_under_rate = row.get('player_under_rate', 0)
    player_bet_count = row.get('player_bet_count', 0)

    # Trailing avg display (using already calculated trailing_avg_calculated from earlier)
    trailing_display = f"{trailing_avg_calculated:.1f}" if trailing_avg_calculated and not pd.isna(trailing_avg_calculated) else "—"

    # Hit rate display: show the rate that matches the pick direction with color coding
    if player_under_rate and not pd.isna(player_under_rate) and player_bet_count and player_bet_count >= 5:
        if 'UNDER' in pick:
            hit_pct = player_under_rate * 100
            hit_rate_display = f"{hit_pct:.0f}%↓"
        else:
            hit_pct = (1 - player_under_rate) * 100
            hit_rate_display = f"{hit_pct:.0f}%↑"
        # Color code: green for >= 50%, red for < 50%
        hit_rate_class = 'positive' if hit_pct >= 50 else 'negative'
    else:
        hit_rate_display = "—"
        hit_rate_class = ''

    # Format player name compactly
    display_name = format_player_name_compact(player, max_len=14)

    # Generate matchup context for player props (not game lines)
    matchup_html = ''
    if position != 'GAME':
        matchup_html = generate_matchup_context_html(row, bet_id, int(week) if pd.notna(week) else 15, prefix='fpc-')

    # Bet data for selection persistence
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    edge_val = edge if not pd.isna(edge) else 0
    bet_data = {
        'id': bet_id,
        'player': player,
        'team': team,
        'position': position,
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': float(line) if pd.notna(line) else 0,
        'odds': int(odds) if pd.notna(odds) else -110,
        'confidence': round(conf_pct, 1),
        'edge': round(edge_val, 1),
        'projection': round(float(projection), 1) if pd.notna(projection) else 0,
        'week': int(week) if pd.notna(week) else 0,
        'game': game,
        'tier': tier
    }
    bet_data_json = json.dumps(bet_data).replace("'", "\\'")

    # Tier badge text
    tier_badge = tier if tier else 'STRONG'
    tier_badge_class = f"tier-badge-{tier.lower()}" if tier else "tier-badge-strong"
    arrow = '▲' if 'OVER' in pick else '▼'

    return f'''
    <div class="featured-pick-card {tier_class}" data-bet-id="{bet_id}" onclick="openPickModal('{pick_data_encoded}')">
        <!-- Top Row: Checkbox + Tier Badge -->
        <div class="card-top-bar">
            <label class="bet-checkbox-container" onclick="event.stopPropagation();">
                <input type="checkbox" class="bet-checkbox" id="bet-{bet_id}" name="bet-{bet_id}" data-bet='{bet_data_json}' onchange="toggleBetSelection(this)">
                <span class="bet-checkmark"></span>
            </label>
            <span class="card-tier-badge {tier_badge_class}">{tier_badge}</span>
        </div>

        <!-- HERO: The Pick Decision -->
        <div class="card-hero {pick_class}">
            <span class="card-hero-arrow">{arrow}</span>
            <div class="card-hero-content">
                <span class="card-hero-pick">{pick} {line}</span>
                <span class="card-hero-market">{market_display}</span>
            </div>
        </div>

        <!-- Player Info Row -->
        <div class="card-player-row">
            <div class="card-player-avatar">
                {player_headshot}
            </div>
            <div class="card-player-details">
                <span class="card-player-name" title="{player}">{player}</span>
                <span class="card-player-meta">{team} · {position}{' · ' + kickoff_display if kickoff_display else ''}</span>
            </div>
        </div>

        <!-- Stats Row - Readable -->
        <div class="card-stats-row">
            <div class="card-stat-item">
                <span class="card-stat-value {conf_class}">{conf_pct:.0f}%</span>
                <span class="card-stat-label">Confidence</span>
            </div>
            <div class="card-stat-divider"></div>
            <div class="card-stat-item">
                <span class="card-stat-value {edge_class}">{edge_display}</span>
                <span class="card-stat-label">Edge</span>
            </div>
        </div>
        {matchup_html}
    </div>
    '''


def generate_featured_pick_table_row(row: pd.Series, format_prop_fn) -> str:
    """Generate a COMPACT table row for featured picks (replaces cards).

    Target: 28px row height, all essential info visible.
    """
    import json
    import urllib.parse
    import hashlib

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    market = safe_str(row.get('market', ''), '')
    market_display = row.get('market_display', format_prop_fn(market))
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    projection = get_projection(row, market)
    tier = str(row.get('quality_tier', row.get('effective_tier', ''))).upper()
    game = safe_str(row.get('game', ''), '')
    week = row.get('week', 0)
    odds = row.get('odds', -110)

    # Create stable bet ID
    bet_id = hashlib.md5(f"{player}_{market}_{line}_{pick}".encode()).hexdigest()[:12]

    # Pick styling
    pick_class = 'pick-over' if 'OVER' in pick else 'pick-under'
    arrow = '▲' if 'OVER' in pick else '▼'

    # Confidence styling
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    if conf_pct >= CONF_THRESHOLD_ELITE:
        conf_class = 'conf-elite'
    elif conf_pct >= CONF_THRESHOLD_HIGH:
        conf_class = 'conf-high'
    else:
        conf_class = 'conf-standard'

    # Edge styling
    edge_val = edge if not pd.isna(edge) else 0
    edge_class = 'edge-pos' if edge_val > 0 else 'edge-neg' if edge_val < 0 else ''
    edge_display = f"+{edge_val:.1f}%" if edge_val > 0 else f"{edge_val:.1f}%"

    # Tier styling
    tier_class = 'tier-elite' if tier == 'ELITE' else 'tier-strong' if tier == 'STRONG' else 'tier-caution' if tier == 'CAUTION' else 'tier-moderate'

    # Team logo (compact)
    team_logo = get_team_logo_html(team, size=16)

    # Abbreviate game for display
    game_abbrev = abbreviate_game(game) if game else ''

    # Build pick data for modal
    # Get game history for chart (load from weekly stats)
    game_history = get_player_game_history(player, market, int(week) if pd.notna(week) else 15)

    # Extract opponent from game string
    team_abbr = safe_str(row.get('team_abbr', ''), '')
    opponent_abbr = safe_str(row.get('opponent_abbr', ''), '') or (extract_opponent_from_game(game, team_abbr) if game and team_abbr else '')

    pick_data = {
        'player': player,
        'position': position,
        'team': team,
        'opponent': opponent_abbr or safe_str(row.get('opponent', ''), ''),
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': safe_float(line),
        'confidence': safe_float(confidence, 0.5),
        'edge': safe_float(edge_val),
        'projection': safe_float(projection),
        'trailing_avg': safe_float(row.get('trailing_stat', 0)),
        'hist_over_rate': safe_float(row.get('hist_over_rate', 0)),
        'hist_count': safe_int(row.get('hist_count', 0)),
        'def_epa': safe_float(row.get('opponent_def_epa', 0)),
        'snap_share': safe_float(row.get('snap_share', 0)),
        'game': game,
        'lvt': safe_float(row.get('line_vs_trailing', 0)),
        'game_history': game_history,
        # Edge model stats for narrative generation
        'source': safe_str(row.get('source', ''), 'EDGE'),
        'lvt_confidence': safe_float(row.get('lvt_confidence', 0)),
        'player_bias_confidence': safe_float(row.get('player_bias_confidence', 0)),
        'player_under_rate': safe_float(row.get('player_under_rate', 0)),
        'player_bet_count': safe_int(row.get('player_bet_count', 0)),
        'current_season_under_rate': safe_float(row.get('current_season_under_rate', 0.5)),
        'season_games_played': safe_int(row.get('season_games_played', 0)),
        # Volume stats (if available from historical data)
        'targets_mean': safe_float(row.get('targets_mean', 0)),
        'receptions_mean': safe_float(row.get('receptions_mean', 0)),
        'receiving_yards_mean': safe_float(row.get('receiving_yards_mean', 0)),
        'rushing_attempts_mean': safe_float(row.get('rushing_attempts_mean', 0)),
        'rushing_yards_mean': safe_float(row.get('rushing_yards_mean', 0)),
        'passing_attempts_mean': safe_float(row.get('passing_attempts_mean', 0)),
        'passing_yards_mean': safe_float(row.get('passing_yards_mean', 0)),
        'redzone_target_share': safe_float(row.get('redzone_target_share', 0)),
        'goalline_carry_share': safe_float(row.get('goalline_carry_share', 0)),
        'game_script': safe_float(row.get('game_script_dynamic', 0)),
        # Weather data
        'weather_temp': safe_float(row.get('temperature', 0)),
        'weather_wind': safe_float(row.get('wind_speed', 0)),
        'weather_is_dome': bool(row.get('is_dome', False)),
        'weather_conditions': safe_str(row.get('conditions', ''), ''),
        'weather_severity': safe_str(row.get('severity', 'None'), 'None'),
        'weather_passing_adj': safe_float(row.get('passing_adjustment', 0)),
        # Defensive ranking
        'def_rank': safe_int(row.get('def_rank', 16)),
        'def_rank_type': safe_str(row.get('def_rank_type', 'pass'), 'pass'),
    }
    pick_data_encoded = urllib.parse.quote(json.dumps(pick_data))

    # Bet data for selection
    bet_data = {
        'id': bet_id,
        'player': player,
        'team': team,
        'position': position,
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': float(line) if pd.notna(line) else 0,
        'odds': int(odds) if pd.notna(odds) else -110,
        'confidence': round(conf_pct, 1),
        'edge': round(edge_val, 1),
        'projection': round(float(projection), 1) if pd.notna(projection) else 0,
        'week': int(week) if pd.notna(week) else 0,
        'game': game,
        'tier': tier
    }
    bet_data_json = json.dumps(bet_data).replace("'", "\\'")

    return f'''<tr data-bet-id="{bet_id}">
        <td style="width: 24px; padding: 0 4px;">
            <label class="bet-checkbox-container" onclick="event.stopPropagation();" style="margin: 0;">
                <input type="checkbox" class="bet-checkbox" id="bet-{bet_id}" name="bet-{bet_id}" data-bet='{bet_data_json}' onchange="toggleBetSelection(this)">
                <span class="bet-checkmark"></span>
            </label>
        </td>
        <td onclick="openPickModal('{pick_data_encoded}')"><div class="fp-player">{team_logo}<span class="fp-player-name">{player}</span><span class="fp-pos">{position}</span></div></td>
        <td onclick="openPickModal('{pick_data_encoded}')">{market_display}</td>
        <td onclick="openPickModal('{pick_data_encoded}')"><span class="fp-pick {pick_class}"><span class="arrow">{arrow}</span>{pick}</span></td>
        <td class="fp-line" onclick="openPickModal('{pick_data_encoded}')">{line}</td>
        <td onclick="openPickModal('{pick_data_encoded}')"><span class="fp-conf {conf_class}">{conf_pct:.0f}%</span></td>
        <td onclick="openPickModal('{pick_data_encoded}')"><span class="fp-edge {edge_class}">{edge_display}</span></td>
        <td onclick="openPickModal('{pick_data_encoded}')">{projection:.1f}</td>
        <td onclick="openPickModal('{pick_data_encoded}')"><span class="fp-tier {tier_class}">{tier}</span></td>
        <td style="font-size: 10px; color: var(--text-muted);" onclick="openPickModal('{pick_data_encoded}')">{game_abbrev}</td>
    </tr>'''


def generate_featured_prop_card(row: pd.Series, format_prop_fn, headshot_lookup: dict = None) -> str:
    """Generate a premium featured prop card (similar to Game Lines cards).

    Used for top 6-8 picks in the hybrid Cheat Sheet layout.

    Args:
        row: DataFrame row with pick data
        format_prop_fn: Function to format prop display names
        headshot_lookup: Dict mapping player names to headshot URLs
    """
    import hashlib

    headshot_lookup = headshot_lookup or {}

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    # Convert YES/NO to OVER/UNDER for TD props
    if pick == 'YES':
        pick = 'OVER'
    elif pick == 'NO':
        pick = 'UNDER'
    market = safe_str(row.get('market', ''), '')
    market_display = row.get('market_display', format_prop_fn(market))
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    projection = get_projection(row, market)
    game = safe_str(row.get('game', ''), '')

    # Calculate values
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    edge_val = edge if not pd.isna(edge) else 0
    diff = projection - float(line) if line else 0

    # Generate player initials (fallback)
    initials = ''.join(word[0] for word in player.split()[:2]) if player else '??'

    # Get headshot URL
    headshot_url = headshot_lookup.get(player, '')

    # Determine tier based on confidence
    if conf_pct >= 70:
        tier = 'elite'
        tier_text = 'ELITE'
        tier_stars = '★★★★★'
    elif conf_pct >= 60:
        tier = 'high'
        tier_text = 'HIGH'
        tier_stars = '★★★★'
    else:
        tier = ''
        tier_text = ''
        tier_stars = '★★★'

    # Pick styling
    pick_lower = pick.lower()
    pick_class = 'over' if pick_lower == 'over' else 'under'
    pick_text = f"{'O' if pick_lower == 'over' else 'U'} {line}"

    # Edge class
    edge_class = 'positive' if edge_val >= 0 else 'negative'

    # Team abbreviation mapping
    TEAM_ABBREVS = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
        # Common abbreviations
        'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF', 'CAR': 'CAR', 'CHI': 'CHI',
        'CIN': 'CIN', 'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
        'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC', 'LV': 'LV', 'LAC': 'LAC',
        'LAR': 'LAR', 'MIA': 'MIA', 'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO', 'NYG': 'NYG',
        'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT', 'SF': 'SF', 'SEA': 'SEA', 'TB': 'TB',
        'TEN': 'TEN', 'WAS': 'WAS'
    }

    # Get team abbreviation
    team_abbrev = TEAM_ABBREVS.get(team, team[:3].upper() if team else 'NFL')

    # Team logo URL (ESPN CDN)
    team_logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{team_abbrev.lower()}.png"

    # Get game abbreviation
    game_abbrev = game
    try:
        parts = game.split(' @ ')
        if len(parts) == 2:
            away = parts[0].split()[-1][:3].upper()
            home = parts[1].split()[-1][:3].upper()
            game_abbrev = f"{away}@{home}"
    except:
        pass

    # Tier badge
    tier_badge = ''
    if tier == 'elite':
        tier_badge = '<span class="fp-card-tier elite">ELITE</span>'
    elif tier == 'high':
        tier_badge = '<span class="fp-card-tier high">HIGH</span>'

    # Diff styling
    diff_class = 'positive' if diff > 0 else 'negative' if diff < 0 else ''
    diff_text = f"{diff:+.1f}" if diff != 0 else "0.0"

    # Avatar HTML - use headshot image with initials fallback, plus team logo overlay
    if headshot_url:
        avatar_html = f'''<div class="fp-card-avatar-wrapper">
                    <div class="fp-card-avatar has-image">
                        <img src="{headshot_url}" alt="{player}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                        <span class="avatar-fallback">{initials}</span>
                    </div>
                    <div class="fp-card-team-logo">
                        <img src="{team_logo_url}" alt="{team_abbrev}" loading="lazy">
                    </div>
                </div>'''
    else:
        avatar_html = f'''<div class="fp-card-avatar-wrapper">
                    <div class="fp-card-avatar">{initials}</div>
                    <div class="fp-card-team-logo">
                        <img src="{team_logo_url}" alt="{team_abbrev}" loading="lazy">
                    </div>
                </div>'''

    return f'''
    <div class="fp-card {tier}">
        <div class="fp-card-header">
            <div class="fp-card-player">
                {avatar_html}
                <div class="fp-card-info">
                    <span class="fp-card-name">{player}</span>
                    <span class="fp-card-meta">{position} · {team_abbrev}</span>
                </div>
            </div>
            {tier_badge}
        </div>
        <div class="fp-card-body">
            <div class="fp-card-prop">
                <span class="fp-card-market">{market_display}</span>
                <span class="fp-card-game">{game_abbrev}</span>
            </div>
            <div class="fp-card-line">
                <span class="fp-card-line-val">{line}</span>
                <span class="fp-card-proj">Proj: <span class="{diff_class}">{projection:.1f}</span></span>
            </div>
        </div>
        <div class="fp-card-footer">
            <span class="fp-card-pick {pick_class}">{pick_text}</span>
            <div class="fp-card-stats">
                <span class="fp-card-conf">{conf_pct:.0f}%</span>
                <span class="fp-card-edge {edge_class}">{edge_val:+.1f}%</span>
            </div>
        </div>
    </div>
    '''


def generate_cheat_sheet_row(row: pd.Series, format_prop_fn, headshot_lookup: dict = None) -> str:
    """Generate a BettingPros-style cheat sheet table row.

    Features:
    - Player avatar with headshot + team logo
    - Prop line and market type
    - Model projection
    - Projection diff (color-coded)
    - Star rating for confidence
    - Edge percentage
    - Historical hit rate bar
    - Pick badge (OVER/UNDER)
    """
    import json
    import urllib.parse
    import hashlib

    headshot_lookup = headshot_lookup or {}

    player = row.get('player', 'Unknown')
    position = row.get('position', '?')
    team = safe_str(row.get('team', ''), '')
    pick = str(row.get('pick', '')).upper()
    # Convert YES/NO to OVER/UNDER for TD props
    if pick == 'YES':
        pick = 'OVER'
    elif pick == 'NO':
        pick = 'UNDER'
    market = safe_str(row.get('market', ''), '')
    market_display = row.get('market_display', format_prop_fn(market))
    line = row.get('line', 0)
    confidence = row.get('calibrated_prob') or row.get('model_prob') or row.get('combined_confidence') or 0.5
    edge = row.get('edge_pct', 0)
    projection = get_projection(row, market)
    game = safe_str(row.get('game', ''), '')
    week = row.get('week', 0)

    # Get opponent
    opponent = safe_str(row.get('opponent', ''), '')
    if not opponent and game:
        # Extract from game string
        parts = game.split(' @ ')
        if len(parts) == 2:
            home = parts[1].split()[-1] if parts[1] else ''
            away = parts[0].split()[-1] if parts[0] else ''
            opponent = home if team == away or team in parts[0] else away

    # Create stable bet ID
    bet_id = hashlib.md5(f"{player}_{market}_{line}_{pick}".encode()).hexdigest()[:12]

    # Calculate values
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    edge_val = edge if not pd.isna(edge) else 0
    diff = projection - float(line) if line else 0
    hist_over_rate = safe_float(row.get('hist_over_rate', 0.5), 0.5)

    # For UNDER picks, show the under hit rate
    pick_dir = pick.lower()
    if pick_dir == 'under':
        hist_rate = (1 - hist_over_rate) * 100
    else:
        hist_rate = hist_over_rate * 100

    # Generate player initials
    initials = ''.join(word[0] for word in player.split()[:2]) if player else '??'

    # Get headshot URL
    headshot_url = headshot_lookup.get(player, '')

    # Team abbreviation mapping
    TEAM_ABBREVS = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
        'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF', 'CAR': 'CAR', 'CHI': 'CHI',
        'CIN': 'CIN', 'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
        'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC', 'LV': 'LV', 'LAC': 'LAC',
        'LAR': 'LAR', 'MIA': 'MIA', 'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO', 'NYG': 'NYG',
        'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT', 'SF': 'SF', 'SEA': 'SEA', 'TB': 'TB',
        'TEN': 'TEN', 'WAS': 'WAS'
    }

    # Get team abbreviation and logo
    team_abbrev = TEAM_ABBREVS.get(team, team[:3].upper() if team else 'NFL')
    team_logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{team_abbrev.lower()}.png"

    # Generate star rating (1-5 stars based on confidence)
    conf_for_stars = conf_pct if not pd.isna(conf_pct) else 50
    star_count = max(0, min(5, round(conf_for_stars / 20)))
    stars_html = ''.join(f'<span class="star filled">★</span>' for _ in range(star_count))
    stars_html += ''.join(f'<span class="star">★</span>' for _ in range(5 - star_count))

    # Diff class - use standard red/green (negative = red, positive = green)
    diff_class = 'negative' if diff < 0 else 'positive'

    # EV class
    ev_class = 'positive' if edge_val >= 0 else 'negative'

    # Hit rate class
    hit_class = 'high' if hist_rate >= 55 else 'medium' if hist_rate >= 45 else 'low'

    # Check if TD prop - show L6 TD games instead of hist rate
    is_td_prop = 'anytime_td' in market.lower() or '_td' in market.lower()
    l5_hit_html = ''

    if is_td_prop:
        # TD props: show games with TDs
        l6_data = get_l6_td_games(player)
        if l6_data['total_games'] > 0:
            games_with_td = l6_data['games_with_td']
            total_games = l6_data['total_games']
            td_pct = (games_with_td / total_games) * 100 if total_games > 0 else 0
            td_class = 'high' if td_pct >= 67 else 'medium' if td_pct >= 50 else 'low'
            l5_hit_html = f'''
            <div class="l5-hit-indicator">
                <span class="l5-hit-value {td_class}">{games_with_td}/{total_games}</span>
                <span class="l5-hit-label">L{total_games} TDs</span>
            </div>
            '''
    elif market in ['player_receptions', 'player_reception_yds', 'player_rush_yds', 'player_rush_attempts']:
        # Continuous stats: show L5 hit rate vs line
        l5_data = get_l5_hit_rate(player, market, float(line))
        if l5_data['total_games'] > 0:
            hits = l5_data['hits']
            total = l5_data['total_games']
            hit_pct = (hits / total) * 100 if total > 0 else 0
            hit_class = 'high' if hit_pct >= 60 else 'medium' if hit_pct >= 40 else 'low'
            # Market-specific label
            label_map = {
                'player_receptions': 'L5 Rec',
                'player_reception_yds': 'L5 Rec Yds',
                'player_rush_yds': 'L5 Rush',
                'player_rush_attempts': 'L5 Att'
            }
            label = label_map.get(market, 'L5 Hit')
            l5_hit_html = f'''
            <div class="l5-hit-indicator">
                <span class="l5-hit-value {hit_class}">{hits}/{total}</span>
                <span class="l5-hit-label">{label} &gt;{line}</span>
            </div>
            '''

    # Game short format
    game_short = f"{team} vs {opponent}" if opponent else team

    # Get game history for the modal chart
    game_history = get_player_game_history(player, market, week)

    # Build pick_data for modal
    pick_data = {
        'player': player,
        'position': position,
        'team': team,
        'opponent': opponent,
        'pick': pick,
        'market': market,
        'market_display': market_display,
        'line': safe_float(line),
        'confidence': safe_float(confidence, 0.5),
        'edge': safe_float(edge_val),
        'projection': safe_float(projection),
        'trailing_avg': safe_float(row.get('trailing_stat', 0)),
        'hist_over_rate': safe_float(hist_over_rate),
        'hist_count': safe_int(row.get('hist_count', 0)),
        'game': game,
        'game_history': game_history,  # For Last 6 Games chart
    }
    pick_data_json = json.dumps(pick_data)
    pick_data_encoded = urllib.parse.quote(pick_data_json)

    # Tier-based row class for visual hierarchy
    tier = str(row.get('effective_tier', row.get('quality_tier', ''))).upper()
    tier_class = ''
    tier_badge_html = ''
    if tier == 'ELITE':
        tier_class = 'tier-elite'
        tier_badge_html = '<span class="tier-badge elite">ELITE</span>'
    elif tier == 'STRONG':
        tier_class = 'tier-strong'
        tier_badge_html = '<span class="tier-badge strong">STRONG</span>'

    # Escape game for data attribute
    game_escaped = game.replace('"', '&quot;') if game else ''

    # Build avatar HTML with headshot or initials fallback
    if headshot_url:
        avatar_html = f'''<div class="player-avatar has-image">
                        <img src="{headshot_url}" alt="{player}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                        <span class="avatar-fallback">{initials}</span>
                    </div>'''
    else:
        avatar_html = f'<div class="player-avatar">{initials}</div>'

    return f'''
    <tr class="{tier_class}" data-market="{market}" data-direction="{pick_dir}" data-position="{position}" data-confidence="{conf_pct:.1f}" data-tier="{tier.lower()}" data-game="{game_escaped}" data-bet-id="{bet_id}">
        <td>
            <div class="player-cell">
                <div class="player-avatar-wrapper">
                    {avatar_html}
                    <div class="player-team-logo">
                        <img src="{team_logo_url}" alt="{team_abbrev}" loading="lazy">
                    </div>
                </div>
                <div class="player-info">
                    <span class="player-name">{player}</span>
                    <span class="player-meta">{position} · {team_abbrev} vs {opponent if opponent else 'TBD'} {tier_badge_html}</span>
                </div>
            </div>
        </td>
        <td>
            <div class="prop-cell">
                <span class="prop-line">{line}</span>
                <span class="prop-market">{market_display}</span>
            </div>
        </td>
        <td class="mono">{projection:.1f}</td>
        <td class="diff-cell {diff_class}">{diff:+.1f}</td>
        <td>
            <div class="rating-stars">{stars_html}</div>
        </td>
        <td class="ev-{ev_class}">{edge_val:+.1f}%</td>
        <td>
            {l5_hit_html if l5_hit_html else f'''
            <div class="hit-rate">
                <div class="hit-rate-bar">
                    <div class="hit-rate-fill {hit_class}" style="width: {hist_rate:.0f}%"></div>
                </div>
                <span class="hit-rate-value">{hist_rate:.0f}%</span>
            </div>
            '''}
        </td>
        <td>
            <span class="pick-badge {pick_dir} {tier.lower() if tier else ''}">{'O' if pick_dir == 'over' else 'U'} {line}</span>
        </td>
        <td>
            <button class="analyze-btn" onclick="openPickModal('{pick_data_encoded}')" title="View Details">📊</button>
        </td>
    </tr>
    '''


def generate_cheat_sheet_section(recs_df: pd.DataFrame, format_prop_fn, week: int) -> str:
    """Generate the complete BettingPros-style Cheat Sheet section.

    Hybrid layout: Featured cards for top picks + full data table below.

    Args:
        recs_df: DataFrame with all pick recommendations
        format_prop_fn: Function to format prop display names
        week: Current week number

    Returns:
        HTML string for the cheat sheet section
    """
    # Load player headshots from rosters
    headshot_lookup = {}
    try:
        roster_path = PROJECT_ROOT / "data" / "nflverse" / "rosters.parquet"
        if roster_path.exists():
            roster_df = pd.read_parquet(roster_path)
            # Get most recent season's roster
            if 'season' in roster_df.columns:
                roster_df = roster_df[roster_df['season'] == roster_df['season'].max()]
            # Create lookup by full_name
            for _, row in roster_df.iterrows():
                name = row.get('full_name', '')
                url = row.get('headshot_url', '')
                if name and url and pd.notna(url):
                    headshot_lookup[name] = url
            print(f"  Loaded {len(headshot_lookup)} player headshots")
    except Exception as e:
        print(f"  Warning: Could not load headshots: {e}")

    # Sort by confidence (highest first)
    sorted_df = recs_df.sort_values('model_prob', ascending=False)

    # Generate featured cards for top 8 picks
    featured_cards_html = ""
    top_picks = sorted_df.head(8)
    for _, row in top_picks.iterrows():
        featured_cards_html += generate_featured_prop_card(row, format_prop_fn, headshot_lookup)

    # Count featured tiers
    featured_elite = len(top_picks[top_picks['model_prob'] >= 0.70])
    featured_high = len(top_picks[(top_picks['model_prob'] >= 0.60) & (top_picks['model_prob'] < 0.70)])

    # Generate all table rows
    table_rows = ""
    for _, row in sorted_df.iterrows():
        table_rows += generate_cheat_sheet_row(row, format_prop_fn, headshot_lookup)

    # Get unique markets for filter pills
    markets = sorted_df['market'].unique() if 'market' in sorted_df.columns else []

    # Market display mapping
    market_pills = [
        ('all', 'All'),
        ('player_receptions', 'Rec'),
        ('player_reception_yds', 'Rec Yds'),
        ('player_rush_yds', 'Rush Yds'),
        ('player_pass_yds', 'Pass Yds'),
        ('player_pass_tds', 'Pass TDs'),
        ('player_rush_attempts', 'Rush Att'),
        ('player_pass_attempts', 'Pass Att'),
    ]

    # Only show pills for markets that exist in data
    filter_pills_html = ""
    for market_key, market_label in market_pills:
        if market_key == 'all' or market_key in markets:
            active = 'active' if market_key == 'all' else ''
            filter_pills_html += f'<button class="filter-pill {active}" data-filter="{market_key}" onclick="filterCheatSheet(\'{market_key}\', this)">{market_label}</button>\n'

    # NFL team name to abbreviation mapping
    TEAM_ABBREVS = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }

    def get_team_abbrev(team_name):
        """Get NFL abbreviation for a team name."""
        if team_name in TEAM_ABBREVS:
            return TEAM_ABBREVS[team_name]
        # Fallback: try partial match
        for full_name, abbrev in TEAM_ABBREVS.items():
            if team_name in full_name or full_name in team_name:
                return abbrev
        return team_name[:3].upper()

    # Get unique games for filter pills with team logos
    games = sorted_df['game'].dropna().unique() if 'game' in sorted_df.columns else []
    game_pills_html = '<button class="filter-pill small active" data-filter="all" onclick="filterCheatSheetGame(\'all\', this)">All Games</button>\n'
    for game in sorted(games):
        if game:
            game_short = game
            away_abbrev = ''
            home_abbrev = ''
            try:
                parts = game.split(' @ ')
                if len(parts) == 2:
                    away_abbrev = get_team_abbrev(parts[0].strip())
                    home_abbrev = get_team_abbrev(parts[1].strip())
                    game_short = f"{away_abbrev}@{home_abbrev}"
            except:
                pass
            game_escaped = game.replace("'", "\\'")
            # Add team logos to pill
            away_logo = f"https://a.espncdn.com/i/teamlogos/nfl/500/{away_abbrev.lower()}.png" if away_abbrev else ''
            home_logo = f"https://a.espncdn.com/i/teamlogos/nfl/500/{home_abbrev.lower()}.png" if home_abbrev else ''
            if away_logo and home_logo:
                game_pills_html += f'''<button class="filter-pill game-pill" data-filter="{game_escaped}" onclick="filterCheatSheetGame('{game_escaped}', this)">
                    <img src="{away_logo}" alt="{away_abbrev}" class="pill-logo">
                    <span class="pill-at">@</span>
                    <img src="{home_logo}" alt="{home_abbrev}" class="pill-logo">
                </button>\n'''
            else:
                game_pills_html += f'<button class="filter-pill small" data-filter="{game_escaped}" onclick="filterCheatSheetGame(\'{game_escaped}\', this)">{game_short}</button>\n'

    total_picks = len(sorted_df)

    return f'''
        <!-- PICKS VIEW - Hybrid layout: Featured cards + Data table -->
        <div class="view-section active" id="cheat-sheet">
            <!-- Featured Picks Section -->
            <div class="featured-picks-section">
                <div class="featured-picks-header">
                    <h2>🔥 Top Picks This Week</h2>
                    <p class="subtitle">Highest confidence props for Week {week}</p>
                </div>
                <div class="featured-picks-grid">
                    {featured_cards_html}
                </div>
            </div>

            <!-- Section Divider -->
            <div class="section-divider">
                <span class="divider-text">All {total_picks} Picks</span>
            </div>

            <!-- Full Data Table Section -->
            <div class="table-section">
                <div class="table-section-header">
                    <h2>Prop Bet Cheat Sheet</h2>
                    <p class="subtitle">Complete list · sortable · filterable</p>
                </div>

                <!-- Primary Filter Row: Market + Direction + Position -->
                <div class="filter-bar" id="cheat-sheet-filters">
                    {filter_pills_html}
                    <span class="filter-divider">|</span>
                    <button class="filter-pill" data-filter="over" onclick="filterCheatSheetDirection('over', this)">OVER</button>
                    <button class="filter-pill" data-filter="under" onclick="filterCheatSheetDirection('under', this)">UNDER</button>
                    <span class="filter-divider">|</span>
                    <button class="filter-pill active" data-filter="all" onclick="filterCheatSheetPosition('all', this)">All Pos</button>
                    <button class="filter-pill" data-filter="QB" onclick="filterCheatSheetPosition('QB', this)">QB</button>
                    <button class="filter-pill" data-filter="RB" onclick="filterCheatSheetPosition('RB', this)">RB</button>
                    <button class="filter-pill" data-filter="WR" onclick="filterCheatSheetPosition('WR', this)">WR</button>
                    <button class="filter-pill" data-filter="TE" onclick="filterCheatSheetPosition('TE', this)">TE</button>
                </div>

                <!-- Game Filter Row (scrollable pills) -->
                <div class="filter-bar games-row" id="game-filters">
                    <span class="filter-label">Game:</span>
                    <div class="filter-pills-scroll">
                        {game_pills_html}
                    </div>
                    <span class="cs-results-count" id="cs-results-count">{total_picks} picks</span>
                </div>

                <!-- Data Table -->
                <div class="table-container">
                    <table class="data-table" id="cheat-sheet-table">
                        <thead>
                            <tr>
                                <th data-sort="player" onclick="sortCheatSheet('player', this)">Player</th>
                                <th data-sort="line" onclick="sortCheatSheet('line', this)">Prop</th>
                                <th data-sort="projection" onclick="sortCheatSheet('projection', this)">Proj.</th>
                                <th data-sort="diff" onclick="sortCheatSheet('diff', this)">Diff</th>
                                <th data-sort="confidence" class="sorted-desc" onclick="sortCheatSheet('confidence', this)">Rating</th>
                                <th data-sort="edge" onclick="sortCheatSheet('edge', this)">Edge</th>
                                <th data-sort="hist" onclick="sortCheatSheet('hist', this)">Hist %</th>
                                <th class="no-sort">Pick</th>
                                <th class="no-sort">Details</th>
                            </tr>
                        </thead>
                        <tbody id="cheat-sheet-tbody">
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    '''


def generate_game_lines_card(game_data: dict, away_team: str, home_team: str, game_time: str, weather: dict = None) -> str:
    """Generate a compact BettingPros-style game card.

    Args:
        game_data: Dict with spread_row and total_row data
        away_team: Away team name
        home_team: Home team name
        game_time: Formatted game time string
        weather: Dict with temperature, wind_speed, is_dome, conditions

    Returns:
        HTML string for the compact game card
    """
    weather = weather or {}

    # NFL team abbreviations and colors
    NFL_TEAMS = {
        'Arizona Cardinals': {'abbrev': 'ARI', 'primary': '#97233F', 'secondary': '#000000'},
        'Atlanta Falcons': {'abbrev': 'ATL', 'primary': '#A71930', 'secondary': '#000000'},
        'Baltimore Ravens': {'abbrev': 'BAL', 'primary': '#241773', 'secondary': '#9E7C0C'},
        'Buffalo Bills': {'abbrev': 'BUF', 'primary': '#00338D', 'secondary': '#C60C30'},
        'Carolina Panthers': {'abbrev': 'CAR', 'primary': '#0085CA', 'secondary': '#101820'},
        'Chicago Bears': {'abbrev': 'CHI', 'primary': '#0B162A', 'secondary': '#C83803'},
        'Cincinnati Bengals': {'abbrev': 'CIN', 'primary': '#FB4F14', 'secondary': '#000000'},
        'Cleveland Browns': {'abbrev': 'CLE', 'primary': '#311D00', 'secondary': '#FF3C00'},
        'Dallas Cowboys': {'abbrev': 'DAL', 'primary': '#003594', 'secondary': '#869397'},
        'Denver Broncos': {'abbrev': 'DEN', 'primary': '#FB4F14', 'secondary': '#002244'},
        'Detroit Lions': {'abbrev': 'DET', 'primary': '#0076B6', 'secondary': '#B0B7BC'},
        'Green Bay Packers': {'abbrev': 'GB', 'primary': '#203731', 'secondary': '#FFB612'},
        'Houston Texans': {'abbrev': 'HOU', 'primary': '#03202F', 'secondary': '#A71930'},
        'Indianapolis Colts': {'abbrev': 'IND', 'primary': '#002C5F', 'secondary': '#A2AAAD'},
        'Jacksonville Jaguars': {'abbrev': 'JAX', 'primary': '#006778', 'secondary': '#D7A22A'},
        'Kansas City Chiefs': {'abbrev': 'KC', 'primary': '#E31837', 'secondary': '#FFB81C'},
        'Las Vegas Raiders': {'abbrev': 'LV', 'primary': '#000000', 'secondary': '#A5ACAF'},
        'Los Angeles Chargers': {'abbrev': 'LAC', 'primary': '#0080C6', 'secondary': '#FFC20E'},
        'Los Angeles Rams': {'abbrev': 'LAR', 'primary': '#003594', 'secondary': '#FFA300'},
        'Miami Dolphins': {'abbrev': 'MIA', 'primary': '#008E97', 'secondary': '#FC4C02'},
        'Minnesota Vikings': {'abbrev': 'MIN', 'primary': '#4F2683', 'secondary': '#FFC62F'},
        'New England Patriots': {'abbrev': 'NE', 'primary': '#002244', 'secondary': '#C60C30'},
        'New Orleans Saints': {'abbrev': 'NO', 'primary': '#D3BC8D', 'secondary': '#101820'},
        'New York Giants': {'abbrev': 'NYG', 'primary': '#0B2265', 'secondary': '#A71930'},
        'New York Jets': {'abbrev': 'NYJ', 'primary': '#125740', 'secondary': '#000000'},
        'Philadelphia Eagles': {'abbrev': 'PHI', 'primary': '#004C54', 'secondary': '#A5ACAF'},
        'Pittsburgh Steelers': {'abbrev': 'PIT', 'primary': '#FFB612', 'secondary': '#101820'},
        'San Francisco 49ers': {'abbrev': 'SF', 'primary': '#AA0000', 'secondary': '#B3995D'},
        'Seattle Seahawks': {'abbrev': 'SEA', 'primary': '#002244', 'secondary': '#69BE28'},
        'Tampa Bay Buccaneers': {'abbrev': 'TB', 'primary': '#D50A0A', 'secondary': '#FF7900'},
        'Tennessee Titans': {'abbrev': 'TEN', 'primary': '#0C2340', 'secondary': '#4B92DB'},
        'Washington Commanders': {'abbrev': 'WAS', 'primary': '#5A1414', 'secondary': '#FFB612'}
    }

    # Reverse lookup: abbreviation -> team data
    ABBREV_TO_TEAM = {}
    for team_name, data in NFL_TEAMS.items():
        ABBREV_TO_TEAM[data['abbrev']] = {'name': team_name, **data}

    def get_team_info(team):
        """Get team info from full name or abbreviation."""
        if team in NFL_TEAMS:
            return NFL_TEAMS[team]
        if team.upper() in ABBREV_TO_TEAM:
            return ABBREV_TO_TEAM[team.upper()]
        for full_name, data in NFL_TEAMS.items():
            if team in full_name or full_name in team:
                return data
        return {'abbrev': team[:3].upper(), 'primary': '#374151', 'secondary': '#1f2937'}

    def extract_picked_team(pick_str):
        """Extract just the team abbrev from a pick like 'KC +13.5'."""
        if not pick_str:
            return ''
        parts = pick_str.strip().split()
        if parts:
            return parts[0].upper()
        return pick_str.upper()

    # Get team info
    away_info = get_team_info(away_team)
    home_info = get_team_info(home_team)
    away_abbrev = away_info['abbrev']
    home_abbrev = home_info['abbrev']

    # Get team logo URLs (ESPN CDN)
    away_logo_url = TEAM_LOGOS.get(away_abbrev, f'https://a.espncdn.com/i/teamlogos/nfl/500/{away_abbrev.lower()}.png')
    home_logo_url = TEAM_LOGOS.get(home_abbrev, f'https://a.espncdn.com/i/teamlogos/nfl/500/{home_abbrev.lower()}.png')

    spread_row = game_data.get('spread')
    total_row = game_data.get('total')

    # Build weather display (compact) - always show when data exists
    weather_html = ''
    if weather:
        is_dome = weather.get('is_dome', False)
        temp = weather.get('temperature')
        wind = weather.get('wind_speed')
        conditions = weather.get('conditions', '')

        if is_dome:
            weather_html = '<span class="compact-weather">🏟️ Dome</span>'
        elif temp is not None and not pd.isna(temp):
            temp_val = float(temp)
            wind_val = float(wind) if wind and not pd.isna(wind) else 0
            weather_icon = '☀️'
            if conditions:
                cond_lower = conditions.lower()
                if 'rain' in cond_lower or 'shower' in cond_lower:
                    weather_icon = '🌧️'
                elif 'snow' in cond_lower:
                    weather_icon = '🌨️'
                elif 'cloud' in cond_lower or 'overcast' in cond_lower:
                    weather_icon = '☁️'
                elif 'wind' in cond_lower and wind_val > 15:
                    weather_icon = '💨'
            # Show temp and wind if significant
            wind_str = f' {wind_val:.0f}mph' if wind_val >= 10 else ''
            weather_html = f'<span class="compact-weather">{weather_icon} {temp_val:.0f}°{wind_str}</span>'

    # Determine card tier (highest of spread/total)
    spread_tier = str(spread_row.get('confidence_tier', 'STANDARD')).upper() if spread_row else 'STANDARD'
    total_tier = str(total_row.get('confidence_tier', 'STANDARD')).upper() if total_row else 'STANDARD'

    card_tier = ''
    tier_stars = ''
    if spread_tier == 'ELITE' or total_tier == 'ELITE':
        card_tier = 'elite'
        tier_stars = '★★★★'
    elif spread_tier == 'HIGH' or total_tier == 'HIGH':
        card_tier = 'high'
        tier_stars = '★★★'
    else:
        tier_stars = '★★'

    # Game time - include date for clarity
    game_time_short = ''
    if game_time:
        parts = game_time.split()
        if len(parts) >= 3:
            day = parts[0][:3]  # "Wed", "Sun", etc.
            date = parts[1] if len(parts) > 1 else ''  # "12/25"
            time_parts = ' '.join(parts[2:])
            time_parts = time_parts.replace(':00', '').replace(' PM', 'p').replace(' AM', 'a').replace('PM', 'p').replace('AM', 'a')
            # Show as "Thu 12/26 8:15p"
            game_time_short = f"{day} {date} {time_parts}".strip()
        else:
            game_time_short = game_time
    else:
        game_time_short = ''

    # Build spread row HTML
    spread_row_html = ''
    if spread_row is not None:
        spread_line = spread_row.get('market_line', spread_row.get('line', 0))
        spread_pick_raw = str(spread_row.get('pick', '')).upper()
        spread_picked_team = extract_picked_team(spread_pick_raw)
        spread_prob = spread_row.get('model_prob', 0.5)
        spread_conf = spread_prob * 100 if spread_prob <= 1 else spread_prob
        spread_edge = spread_row.get('edge_pct', 0)
        spread_edge_val = spread_edge if not pd.isna(spread_edge) else 0

        away_is_picked = spread_picked_team == away_abbrev or spread_picked_team in away_team.upper()

        if away_is_picked:
            pick_text = f'{away_abbrev} {spread_line:+.1f}'
        else:
            pick_text = f'{home_abbrev} {-spread_line:+.1f}'

        edge_class = 'positive' if spread_edge_val >= 0 else 'negative'
        spread_row_html = f'''
            <div class="compact-bet" data-bet-type="spread">
                <span class="compact-pick spread">{pick_text}</span>
                <div class="compact-stats">
                    <span class="compact-conf">{spread_conf:.0f}%</span>
                    <span class="compact-edge {edge_class}">{spread_edge_val:+.1f}%</span>
                </div>
            </div>
        '''

    # Build total row HTML
    total_row_html = ''
    if total_row is not None:
        total_line = total_row.get('market_line', total_row.get('line', 0))
        total_pick = str(total_row.get('pick', '')).upper()
        total_prob = total_row.get('model_prob', 0.5)
        total_conf = total_prob * 100 if total_prob <= 1 else total_prob
        total_edge = total_row.get('edge_pct', 0)
        total_edge_val = total_edge if not pd.isna(total_edge) else 0

        is_over = 'OVER' in total_pick
        pick_text = f"{'O' if is_over else 'U'} {total_line}"
        pick_class = 'over' if is_over else 'under'

        edge_class = 'positive' if total_edge_val >= 0 else 'negative'
        total_row_html = f'''
            <div class="compact-bet" data-bet-type="total">
                <span class="compact-pick {pick_class}">{pick_text}</span>
                <div class="compact-stats">
                    <span class="compact-conf">{total_conf:.0f}%</span>
                    <span class="compact-edge {edge_class}">{total_edge_val:+.1f}%</span>
                </div>
            </div>
        '''

    # Tier badge text
    tier_badge = ''
    if card_tier == 'elite':
        tier_badge = '<span class="compact-tier elite">ELITE</span>'
    elif card_tier == 'high':
        tier_badge = '<span class="compact-tier high">HIGH</span>'

    # Build compact card
    return f'''
    <div class="compact-card {card_tier}" data-game="{away_abbrev} @ {home_abbrev}">
        <div class="compact-header">
            <div class="compact-team-logo" style="--team-primary: {away_info['primary']};">
                <img src="{away_logo_url}" alt="{away_abbrev}" loading="lazy">
                <span class="logo-fallback">{away_abbrev}</span>
            </div>
            <div class="compact-matchup">
                <span class="compact-teams">{away_abbrev} @ {home_abbrev}</span>
                <span class="compact-meta">{game_time_short}{weather_html}</span>
            </div>
            <div class="compact-team-logo" style="--team-primary: {home_info['primary']};">
                <img src="{home_logo_url}" alt="{home_abbrev}" loading="lazy">
                <span class="logo-fallback">{home_abbrev}</span>
            </div>
            {tier_badge}
        </div>
        <div class="compact-picks">
            {spread_row_html}
            {total_row_html}
        </div>
    </div>
    '''


def generate_game_lines_table_section(game_lines_df: pd.DataFrame, week: int) -> str:
    """Generate the complete BettingPros-style Game Lines section with game cards.

    Args:
        game_lines_df: DataFrame with all game line recommendations
        week: Current week number

    Returns:
        HTML string for the game lines section
    """
    if game_lines_df is None or len(game_lines_df) == 0:
        return '''
        <div class="view-section" id="game-lines">
            <div class="table-section">
                <div class="table-section-header">
                    <h2>NFL Betting Picks: Projected Lines</h2>
                    <p class="subtitle">No game line recommendations available</p>
                </div>
            </div>
        </div>
        '''

    # Load schedule for game times
    schedule_lookup = {}
    try:
        schedule_path = PROJECT_ROOT / "data" / "nflverse" / "schedules.parquet"
        if schedule_path.exists():
            schedule_df = pd.read_parquet(schedule_path)
            week_games = schedule_df[(schedule_df['season'] == 2025) & (schedule_df['week'] == week)]
            for _, sched_row in week_games.iterrows():
                away = sched_row['away_team']
                home = sched_row['home_team']
                game_key = f"{away} @ {home}"
                gameday = sched_row['gameday']
                gametime = sched_row['gametime']
                roof = sched_row.get('roof', '')
                temp = sched_row.get('temp', None)
                wind = sched_row.get('wind', None)
                # Format: "Wed 12/25 1:00 PM"
                try:
                    dt = pd.to_datetime(f"{gameday} {gametime}")
                    formatted = dt.strftime('%a %m/%d %I:%M %p').replace(' 0', ' ')
                    schedule_lookup[game_key] = {
                        'game_time': formatted,
                        'game_datetime': dt,  # Store datetime for sorting
                        'is_dome': roof in ['dome', 'closed'],
                        'temperature': temp,
                        'wind_speed': wind
                    }
                except:
                    pass
    except Exception as e:
        print(f"  Warning: Could not load schedule: {e}")

    # Group bets by game
    games_data = {}
    for _, row in game_lines_df.iterrows():
        game = row.get('game', '')
        if not game:
            continue

        teams = game.split(' @ ')
        if len(teams) != 2:
            continue

        away_team = teams[0].strip()
        home_team = teams[1].strip()

        if game not in games_data:
            # Get game time from schedule lookup
            sched_info = schedule_lookup.get(game, {})
            game_time = sched_info.get('game_time', '')
            game_datetime = sched_info.get('game_datetime', None)

            # Extract weather data (prefer schedule, fallback to row)
            weather_data = {
                'temperature': sched_info.get('temperature') or row.get('temperature', row.get('weather_temp', None)),
                'wind_speed': sched_info.get('wind_speed') or row.get('wind_speed', row.get('weather_wind', None)),
                'is_dome': sched_info.get('is_dome', False) or row.get('is_dome', row.get('weather_is_dome', False)),
                'conditions': row.get('conditions', row.get('weather_conditions', '')),
            }

            games_data[game] = {
                'away_team': away_team,
                'home_team': home_team,
                'game_time': game_time,
                'game_datetime': game_datetime,  # For sorting
                'weather': weather_data,
                'spread': None,
                'total': None
            }

        bet_type = str(row.get('bet_type', '')).lower()
        if 'spread' in bet_type:
            games_data[game]['spread'] = row.to_dict()
        elif 'total' in bet_type:
            games_data[game]['total'] = row.to_dict()

    # Sort games by kickoff time
    sorted_games = sorted(
        games_data.items(),
        key=lambda x: (x[1]['game_datetime'] is None, x[1]['game_datetime'] or pd.Timestamp.max)
    )

    # Generate game cards
    game_cards_html = ""
    for game, data in sorted_games:
        game_cards_html += generate_game_lines_card(
            data,
            data['away_team'],
            data['home_team'],
            data['game_time'],
            data.get('weather', {})
        )

    # Count statistics
    tier_counts = game_lines_df['confidence_tier'].value_counts().to_dict() if 'confidence_tier' in game_lines_df.columns else {}
    elite_count = tier_counts.get('ELITE', 0)
    high_count = tier_counts.get('HIGH', 0)

    spread_count = len(game_lines_df[game_lines_df['bet_type'].str.contains('spread', case=False, na=False)]) if 'bet_type' in game_lines_df.columns else 0
    total_count = len(game_lines_df[game_lines_df['bet_type'].str.contains('total', case=False, na=False)]) if 'bet_type' in game_lines_df.columns else 0

    total_bets = len(game_lines_df)
    num_games = len(games_data)

    return f'''
        <!-- GAME LINES VIEW - BettingPros-style game cards -->
        <div class="view-section" id="game-lines">
            <div class="table-section">
                <div class="table-section-header">
                    <h2>NFL Betting Picks: Projected Lines</h2>
                    <p class="subtitle">Week {week} · Our model creates expert lines for every NFL game to identify potential value</p>
                </div>

                <!-- Filter Pills -->
                <div class="filter-bar" id="game-lines-filters">
                    <button class="filter-pill active" data-filter="all" onclick="filterBPGameLines('all', this)">All<span class="filter-count">{total_bets}</span></button>
                    <button class="filter-pill" data-filter="spread" onclick="filterBPGameLines('spread', this)">Spread<span class="filter-count">{spread_count}</span></button>
                    <button class="filter-pill" data-filter="total" onclick="filterBPGameLines('total', this)">Totals<span class="filter-count">{total_count}</span></button>
                    <span style="margin-left: auto; color: var(--text-muted); font-size: 12px;">|</span>
                    <button class="filter-pill" data-filter="elite" onclick="filterBPGameLinesTier('elite', this)">Elite<span class="filter-count">{elite_count}</span></button>
                    <button class="filter-pill" data-filter="high" onclick="filterBPGameLinesTier('high', this)">High<span class="filter-count">{high_count}</span></button>
                </div>

                <!-- Game Cards -->
                <div class="bp-games-container">
                    {game_cards_html}
                </div>
            </div>
        </div>
    '''


def parse_shap_factors(model_reasoning: str) -> tuple:
    """Parse SHAP factors from model_reasoning into over/under groups.

    Returns:
        tuple: (over_factors, under_factors, p_under) where factors are lists of (name, strength, is_dominant)
    """
    over_factors = []
    under_factors = []
    p_under = None

    if not model_reasoning or pd.isna(model_reasoning):
        return over_factors, under_factors, p_under

    try:
        # Look for SHAP factor pattern in reasoning
        # Format: "Favors OVER:" or "Favors UNDER:" followed by factor lines
        lines = str(model_reasoning).split('\n')
        current_direction = None

        for line in lines:
            line = line.strip()
            if 'Favors OVER' in line:
                current_direction = 'over'
            elif 'Favors UNDER' in line:
                current_direction = 'under'
            elif line.startswith('•') and current_direction:
                # Parse factor: "• feature name (0.25) *"
                is_dominant = '*' in line
                line = line.replace('*', '').strip()
                # Extract name and strength
                if '(' in line and ')' in line:
                    name_part = line.split('(')[0].replace('•', '').strip()
                    strength_part = line.split('(')[1].split(')')[0]
                    try:
                        strength = float(strength_part)
                        if current_direction == 'over':
                            over_factors.append((name_part, strength, is_dominant))
                        else:
                            under_factors.append((name_part, strength, is_dominant))
                    except ValueError:
                        pass
            elif 'P(UNDER)' in line or 'p_under' in line.lower():
                # Try to extract probability
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        p_str = parts[1].strip().replace('%', '')
                        p_under = float(p_str) / 100 if float(p_str) > 1 else float(p_str)
                    except ValueError:
                        pass

    except Exception:
        pass

    return over_factors, under_factors, p_under


def calculate_composite_score(row: pd.Series) -> tuple:
    """
    Calculate composite quality score incorporating multiple reliability signals.

    Returns:
        tuple: (composite_score, model_prob, historical_rate, disagreement)
    """
    weights = {
        'model_prob': 0.30,       # Model confidence
        'historical_rate': 0.25,  # How often this actually hits historically
        'edge_size': 0.25,        # EV edge magnitude
        'projection_confidence': 0.20  # How stable is the projection
    }

    pick = str(row.get('pick', '')).lower()
    is_under = 'under' in pick

    # 1. Model probability (adjusted for direction)
    model_p_under = row.get('model_p_under', None)
    model_p = row.get('model_prob', 0.5)

    if model_p_under is not None and pd.notna(model_p_under) and model_p_under != 0.5:
        model_prob = model_p_under if is_under else (1 - model_p_under)
    else:
        model_prob = model_p

    # 2. Historical alignment score
    # Use hist_over_rate from recommendations (inverted for under picks)
    hist_over_rate = row.get('hist_over_rate', 0.5)
    if pd.isna(hist_over_rate):
        hist_over_rate = 0.5
    historical_rate = (1 - hist_over_rate) if is_under else hist_over_rate

    # 3. Edge score (normalize to 0-1 scale, 5%+ edge = 1.0)
    edge = abs(row.get('edge_pct', 0) or 0)
    edge_score = min(edge / 5.0, 1.0)

    # 4. Projection confidence
    # Compare projection to historical average (hist_avg)
    projection = get_projection(row, market) or 0
    career_avg = row.get('hist_avg', projection) or projection

    if career_avg > 0 and projection > 0:
        deviation = abs(projection - career_avg) / career_avg
        projection_confidence = max(0, 1 - deviation)
    else:
        projection_confidence = 0.5

    # Calculate disagreement between model and history
    disagreement = abs(model_prob - historical_rate)

    # Composite score
    composite = (
        weights['model_prob'] * model_prob +
        weights['historical_rate'] * historical_rate +
        weights['edge_size'] * edge_score +
        weights['projection_confidence'] * projection_confidence
    )

    return composite, model_prob, historical_rate, disagreement


def get_effective_confidence(row: pd.Series) -> float:
    """Get composite confidence measure using multi-factor scoring."""
    composite, model_prob, hist_rate, disagreement = calculate_composite_score(row)
    return composite


def get_adjusted_tier(row: pd.Series) -> str:
    """
    Tier assignment based on RAW model probability.
    CAUTION flag when model and history disagree significantly.
    """
    model_prob = row.get('model_prob', 0.5)

    # Check for model vs history disagreement
    pick = str(row.get('pick', '')).lower()
    is_under = 'under' in pick
    hist_over_rate = row.get('hist_over_rate', 0.5)
    if pd.isna(hist_over_rate):
        hist_over_rate = 0.5
    historical_rate = (1 - hist_over_rate) if is_under else hist_over_rate
    disagreement = abs(model_prob - historical_rate)

    # Red flag: Model and history disagree by >30%
    if disagreement > 0.30:
        return 'CAUTION'

    # Tiers based on raw model probability
    if model_prob >= 0.70:
        return 'ELITE'
    elif model_prob >= 0.60:
        return 'STRONG'
    elif model_prob >= 0.50:
        return 'MODERATE'
    else:
        return 'SPECULATIVE'


def get_confidence_tier(prob: float) -> str:
    """Fallback: Convert probability to tier (used when row not available)."""
    if prob >= 0.75:
        return 'ELITE'
    elif prob >= 0.60:
        return 'STRONG'
    elif prob >= 0.50:
        return 'MODERATE'
    else:
        return 'SPECULATIVE'


def load_schedule(week: int = None) -> pd.DataFrame:
    """Load schedule with game times and venue info from nflverse."""
    # Try parquet first (has roof/surface data)
    parquet_path = PROJECT_ROOT / 'data/nflverse/schedules.parquet'
    csv_path = PROJECT_ROOT / 'data/nflverse/schedules_2024_2025.csv'

    if parquet_path.exists():
        schedule = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        schedule = pd.read_csv(csv_path)
    else:
        return pd.DataFrame()

    schedule = schedule[schedule['season'] == get_current_season()].copy()

    if week is not None:
        schedule = schedule[schedule['week'] == week].copy()

    return schedule


def load_live_edges() -> pd.DataFrame:
    """Load most recent live edges from DraftKings comparison."""
    reports_dir = PROJECT_ROOT / 'reports'
    edge_files = sorted(reports_dir.glob('live_prop_edges_week*.csv'), reverse=True)

    if not edge_files:
        return pd.DataFrame()

    return pd.read_csv(edge_files[0])


def load_edge_recommendations(week: int, season: int = 2025) -> pd.DataFrame:
    """Load edge recommendations and convert to dashboard format.

    This loads from the new edge pipeline (LVT + Player Bias + TD Poisson)
    and maps columns to the format expected by the dashboard.
    """
    reports_dir = PROJECT_ROOT / 'reports'

    # Try to find edge recommendations file
    edge_path = reports_dir / f'edge_recommendations_week{week}_{season}.csv'
    if not edge_path.exists():
        # Try alternate naming
        edge_files = sorted(reports_dir.glob(f'edge_recommendations_week{week}*.csv'), reverse=True)
        if edge_files:
            edge_path = edge_files[0]
        else:
            return pd.DataFrame()

    df = pd.read_csv(edge_path)

    if len(df) == 0:
        return df

    # === ADD PLAYER_ID AND ACTUAL POSITION FROM WEEKLY STATS ===
    # Load weekly stats to create player name -> player_id and player_id -> position mappings
    player_positions = {}  # Will be populated if stats loaded successfully
    try:
        stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if stats_path.exists():
            stats = pd.read_parquet(stats_path, columns=['player_id', 'player_name', 'player_display_name', 'team', 'position'])
            # Create mapping from player_id -> position (use most recent position per player)
            stats_by_id = stats.drop_duplicates(subset=['player_id'], keep='last')
            player_positions = dict(zip(stats_by_id['player_id'], stats_by_id['position']))

            # Create display name -> player_id mapping (full names like "Mike Evans")
            # Deduplicate by display_name to handle name collisions properly
            stats_by_display = stats.drop_duplicates(subset=['player_display_name'], keep='last')
            display_to_id = dict(zip(stats_by_display['player_display_name'], stats_by_display['player_id']))

            # Create abbreviated name -> player_id mapping (like "M.Evans")
            # Include team to disambiguate players with same abbreviated name
            stats_dedup = stats.drop_duplicates(subset=['player_name', 'team'], keep='last')
            name_to_id = dict(zip(stats_dedup['player_name'], stats_dedup['player_id']))
            # Also create team-qualified lookup for disambiguation
            name_team_to_id = {(row['player_name'], row['team']): row['player_id']
                              for _, row in stats_dedup.iterrows()}

            # Add player_id column
            def get_player_id(row):
                player = row.get('player', '')
                team = row.get('team', '')
                if not player or pd.isna(player):
                    return ''

                # Normalize name by removing suffixes (Jr., Sr., II, III, IV, V)
                def normalize_name(name):
                    suffixes = [' Jr.', ' Sr.', ' Jr', ' Sr', ' II', ' III', ' IV', ' V']
                    normalized = name
                    for suffix in suffixes:
                        if normalized.endswith(suffix):
                            normalized = normalized[:-len(suffix)]
                    return normalized

                player_normalized = normalize_name(player)

                # Priority 1: Try display name match (full names like "Mike Evans")
                if player in display_to_id:
                    return display_to_id[player]
                # Also try normalized name
                if player_normalized in display_to_id:
                    return display_to_id[player_normalized]

                # Priority 2: Try abbreviated name with team context (disambiguates M.Evans)
                if player in name_to_id:
                    # Check if team-qualified lookup exists for better match
                    if team and (player, team) in name_team_to_id:
                        return name_team_to_id[(player, team)]
                    return name_to_id[player]

                # Priority 3: Try partial match (first initial + last name)
                parts = player_normalized.split()
                if len(parts) >= 2:
                    # Get last name (skip suffixes that might have slipped through)
                    last_name = parts[-1]
                    short_name = f"{parts[0][0]}.{last_name}"
                    # Try with team context first
                    if team and (short_name, team) in name_team_to_id:
                        return name_team_to_id[(short_name, team)]
                    if short_name in name_to_id:
                        return name_to_id[short_name]
                return ''

            df['player_id'] = df.apply(get_player_id, axis=1)
            matched = (df['player_id'] != '').sum()
            print(f"  Matched {matched}/{len(df)} players to player_id")

            # Add actual position column - use player's real position, not market-inferred
            def get_actual_position(row):
                player_id = row.get('player_id', '')
                source = row.get('source', '')

                # Game lines don't have player positions
                if 'GAME_LINE' in str(source) or row.get('bet_category') == 'GAME_LINE':
                    return 'GAME'

                # Look up actual position from weekly_stats
                if player_id and player_id in player_positions:
                    return player_positions[player_id]

                # Fallback to market-based inference only if player position unknown
                market = safe_str(row.get('market', ''), '')
                if 'pass' in str(market):
                    return 'QB'
                elif 'rush' in str(market):
                    return 'RB'
                elif 'reception' in str(market) or 'rec' in str(market):
                    return 'WR'
                return 'FLEX'

            df['actual_position'] = df.apply(get_actual_position, axis=1)
            pos_lookup = (df['player_id'] != '') & (df['actual_position'] != 'GAME')
            print(f"  Looked up actual position for {pos_lookup.sum()} players")
        else:
            print(f"  Warning: weekly_stats.parquet not found at {stats_path}")
            df['player_id'] = ''
            df['actual_position'] = ''
    except Exception as e:
        print(f"  Warning: Could not map player names to IDs: {e}")
        df['player_id'] = ''
        df['actual_position'] = ''

    # Map edge columns to dashboard columns
    # Edge columns: player, team, market, line, direction, units, source,
    #               lvt_confidence, player_bias_confidence, combined_confidence,
    #               reasoning, player_under_rate, player_bet_count, p_over, p_under, expected_tds

    # Create dashboard-compatible columns
    df['pick'] = df['direction']  # OVER or UNDER
    df['model_prob'] = df['combined_confidence']  # Main confidence metric
    df['model_prob_pct'] = (df['combined_confidence'] * 100).round(1)
    df['confidence'] = df['combined_confidence']
    df['confidence_pct'] = (df['combined_confidence'] * 100).round(1)

    # Kelly/units mapping
    df['kelly_units'] = df['units'] / 100  # Convert to fraction for display
    df['kelly_fraction'] = df['units'] / 100

    # Market display names (consistent with format_prop_display)
    market_display_map = {
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Rec Yards',
        'player_rush_yds': 'Rush Yards',
        'player_rush_attempts': 'Rush Att',
        'player_pass_yds': 'Pass Yards',
        'player_pass_attempts': 'Pass Att',
        'player_pass_completions': 'Completions',
        'player_pass_tds': 'Pass TDs',
        'player_rush_tds': 'Rush TDs',
        'player_rec_tds': 'Rec TDs',
        'player_anytime_td': 'Anytime TD',
    }
    df['market_display'] = df['market'].map(market_display_map).fillna(df['market'])

    # Edge source for display
    df['prediction_source'] = df['source']
    df['strategy'] = df['source']

    # Historical stats from edge
    df['hist_over_rate'] = 1 - df['player_under_rate']
    df['hist_count'] = df['player_bet_count']

    # Tier assignment based on confidence
    def assign_tier(conf):
        if conf >= 0.75:
            return 'ELITE'
        elif conf >= 0.65:
            return 'STRONG'
        elif conf >= 0.55:
            return 'MODERATE'
        else:
            return 'LOW'

    df['effective_tier'] = df['combined_confidence'].apply(assign_tier)
    df['confidence'] = df['effective_tier']

    # Fill in other expected columns with defaults
    df['week'] = week
    df['odds'] = -110  # Default odds
    df['edge_pct'] = ((df['combined_confidence'] - 0.5) * 100).round(1)
    df['roi_pct'] = df['edge_pct'] * 0.9  # Approximate ROI
    df['priority'] = df['combined_confidence'].rank(ascending=False)

    # Position - use actual position looked up earlier (not market-inferred)
    # actual_position was populated from weekly_stats in the player_id mapping section
    if 'actual_position' in df.columns:
        df['position'] = df['actual_position']
    else:
        # Fallback to market-based inference if actual_position not available
        def infer_position_from_market(row):
            market = safe_str(row.get('market', ''), '')
            source = row.get('source', '')

            if 'GAME_LINE' in str(source) or row.get('bet_category') == 'GAME_LINE':
                return 'GAME'
            if 'pass' in str(market):
                return 'QB'
            elif 'rush' in str(market):
                return 'RB'
            elif 'reception' in str(market) or 'rec' in str(market):
                return 'WR'
            return 'FLEX'

        df['position'] = df.apply(infer_position_from_market, axis=1)

    # Model reasoning from edge reasoning
    df['model_reasoning'] = df['reasoning']

    # TD-specific columns
    if 'expected_tds' in df.columns:
        df['model_projection'] = df['expected_tds']

    # LVT signal
    df['line_vs_trailing'] = df.get('lvt_confidence', 0)

    # Ensure is_featured flag
    df['is_featured'] = df['combined_confidence'] >= 0.55

    # Add bet_category column if not present
    if 'bet_category' not in df.columns:
        df['bet_category'] = df['source'].apply(
            lambda x: 'GAME_LINE' if 'GAME_LINE' in str(x) else 'PLAYER_PROP'
        )

    # Handle game line specific display
    # For game lines, the 'player' column is actually the game name
    game_line_mask = df['bet_category'] == 'GAME_LINE'
    if game_line_mask.any():
        # Game lines use 'game' field instead of 'player'
        if 'game' in df.columns:
            df.loc[game_line_mask, 'player'] = df.loc[game_line_mask, 'game']

        # Extract home team from game string (e.g., "GB @ DEN" -> "DEN")
        # This is used for logo display
        def extract_home_team(game_str):
            if pd.isna(game_str) or not isinstance(game_str, str):
                return 'NFL'
            parts = game_str.split(' @ ')
            return parts[1] if len(parts) == 2 else 'NFL'

        df.loc[game_line_mask, 'team'] = df.loc[game_line_mask, 'game'].apply(extract_home_team)

        # Market display for game lines
        df.loc[game_line_mask & (df['market'] == 'spread'), 'market_display'] = 'Spread'
        df.loc[game_line_mask & (df['market'] == 'total'), 'market_display'] = 'Total'

    # Count by category
    player_props = len(df[df['bet_category'] != 'GAME_LINE'])
    game_lines = len(df[df['bet_category'] == 'GAME_LINE'])
    print(f"Loaded {len(df)} edge recommendations from {edge_path.name}")
    print(f"  Player props: {player_props}, Game lines: {game_lines}")

    # === LOAD WEATHER DATA ===
    try:
        from nfl_quant.utils.unified_integration import load_weather_for_week
        from nfl_quant.utils.team_names import normalize_team_name
        from nfl_quant.config_paths import DATA_DIR
        import json

        weather_df = load_weather_for_week(week, season)

        # Load raw weather JSON to build team->opponent mapping
        opponent_map = {}
        weather_cache = DATA_DIR / 'weather' / f'weather_week{week}_{season}.json'
        if weather_cache.exists():
            with open(weather_cache) as f:
                cached_games = json.load(f)
            for game in cached_games:
                home = game.get('home_team', '')
                away = game.get('away_team', '')
                if home and away:
                    opponent_map[home] = away
                    opponent_map[away] = home

        if not weather_df.empty and 'team' in df.columns:
            # Normalize team names in df to match weather_df (which uses abbreviations)
            df['team_abbr'] = df['team'].apply(normalize_team_name)

            # Select columns that exist in weather_df
            weather_cols = ['team']
            for col in ['temperature', 'wind_speed', 'is_dome', 'conditions',
                        'severity', 'passing_adjustment', 'wind_bucket', 'temp_bucket']:
                if col in weather_df.columns:
                    weather_cols.append(col)

            # Rename 'team' to 'team_abbr' in weather_df for merge
            weather_merge = weather_df[weather_cols].drop_duplicates('team').copy()
            weather_merge = weather_merge.rename(columns={'team': 'team_abbr'})

            # Add opponent column from the mapping
            weather_merge['opponent_abbr'] = weather_merge['team_abbr'].map(opponent_map)

            df = df.merge(weather_merge, on='team_abbr', how='left')
            games_with_weather = df['temperature'].notna().sum() if 'temperature' in df.columns else 0
            print(f"  Weather data merged: {games_with_weather} picks with weather")
    except Exception as e:
        print(f"  Warning: Could not load weather data: {e}")

    # === LOAD DEFENSIVE RANKINGS ===
    # First, load position-specific defense caches
    load_position_defense_caches()

    try:
        pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
        if pbp_path.exists():
            pbp = pd.read_parquet(pbp_path)
            pbp_season = pbp[pbp['season'] == season]

            # Pass defense (vs WRs/TEs) - higher EPA = worse defense (fallback)
            pass_plays = pbp_season[pbp_season['play_type'] == 'pass']
            pass_def_epa = pass_plays.groupby('defteam')['epa'].mean()
            pass_def_rank = pass_def_epa.rank(ascending=False)  # 1 = worst (most EPA allowed)

            # Rush defense (vs RBs) - fallback
            rush_plays = pbp_season[pbp_season['play_type'] == 'run']
            rush_def_epa = rush_plays.groupby('defteam')['epa'].mean()
            rush_def_rank = rush_def_epa.rank(ascending=False)  # 1 = worst

            # Add defensive rank - try position-specific first, then fall back to generic
            def get_def_rank_and_type(row):
                # Try multiple opponent column names
                opponent = row.get('opponent_abbr', '') or row.get('opponent', '')
                market = str(row.get('market', '')).lower()
                player_id = row.get('player_id', '')
                team = row.get('team_abbr', '') or row.get('team', '')
                # Use actual position from player lookup (not market-inferred)
                actual_pos = row.get('actual_position', '') or row.get('position', '')

                if not opponent:
                    return 16, 'vs WRs'

                # Use actual player position for defense lookup
                # Only fall back to market inference if actual position unknown or is GAME
                if actual_pos and actual_pos not in ['', 'GAME', 'FLEX']:
                    position = actual_pos  # QB, RB, WR, TE, etc.
                elif 'rush' in market:
                    position = 'RB'
                elif 'reception' in market or 'rec' in market:
                    position = 'WR'
                else:
                    position = 'WR'

                # Try to get player's position role (WR1, RB2, etc.) for position-specific defense
                found_pos_rank = 0
                if player_id and team and position in ['WR', 'RB', 'TE']:
                    pos_actual, pos_rank = get_player_position_role(player_id, team, week, season)
                    if pos_actual and pos_rank:
                        found_pos_rank = pos_rank
                        # Try position-specific defense rank (pass market for stat-specific column)
                        spec_rank, spec_label = get_position_specific_defense_rank(
                            opponent, week, season, pos_actual, pos_rank, market
                        )
                        if spec_rank is not None:
                            return spec_rank, spec_label, found_pos_rank

                # Fall back to generic defense rank based on actual position
                if position == 'RB':
                    return int(rush_def_rank.get(opponent, 16)), 'vs RBs', found_pos_rank
                elif position == 'QB':
                    # Try QB market-specific defense ranking first
                    qb_rank, qb_label = get_qb_defense_rank(opponent, week, season, market)
                    if qb_rank is not None:
                        return qb_rank, qb_label, 0
                    # Fall back to generic pass defense
                    return int(pass_def_rank.get(opponent, 16)), 'vs QBs', 0
                else:
                    return int(pass_def_rank.get(opponent, 16)), 'vs WRs', found_pos_rank

            # Apply and unpack results
            results = df.apply(get_def_rank_and_type, axis=1)
            df['def_rank'] = results.apply(lambda x: x[0])
            df['def_rank_type'] = results.apply(lambda x: x[1])
            df['pos_rank'] = results.apply(lambda x: x[2] if len(x) > 2 else 0)

            # Count position-specific vs generic
            # Match WR1s, RB2s, etc. AND QB-specific like "vs Pass Att", "vs Compl"
            pos_specific = df['def_rank_type'].str.contains(r'vs [A-Z]{2}\ds|vs Pass|vs Compl', regex=True, na=False).sum()
            print(f"  Defensive rankings calculated: {pos_specific} position-specific, {len(df) - pos_specific} generic")
    except Exception as e:
        print(f"  Warning: Could not calculate defensive rankings: {e}")
        df['def_rank'] = 16
        df['def_rank_type'] = 'pass'

    return df


def add_player_ids_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add player_id column to a dataframe by matching player names to weekly stats.

    This enables position-specific defense rankings lookup.
    """
    if len(df) == 0 or 'player_id' in df.columns:
        return df

    try:
        stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if not stats_path.exists():
            print(f"  Warning: weekly_stats.parquet not found for player ID mapping")
            df['player_id'] = ''
            return df

        stats = pd.read_parquet(stats_path, columns=['player_id', 'player_name', 'player_display_name', 'team', 'position'])

        # Create mappings
        stats_by_display = stats.drop_duplicates(subset=['player_display_name'], keep='last')
        display_to_id = dict(zip(stats_by_display['player_display_name'], stats_by_display['player_id']))

        stats_dedup = stats.drop_duplicates(subset=['player_name', 'team'], keep='last')
        name_to_id = dict(zip(stats_dedup['player_name'], stats_dedup['player_id']))
        name_team_to_id = {(row['player_name'], row['team']): row['player_id'] for _, row in stats_dedup.iterrows()}

        # Create position lookup
        stats_by_id = stats.drop_duplicates(subset=['player_id'], keep='last')
        player_positions = dict(zip(stats_by_id['player_id'], stats_by_id['position']))

        def normalize_name(name):
            suffixes = [' Jr.', ' Sr.', ' Jr', ' Sr', ' II', ' III', ' IV', ' V']
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            return name.strip()

        def get_player_id(row):
            player = row.get('player', '')
            team = row.get('team', '')
            if not player or pd.isna(player):
                return ''

            player_norm = normalize_name(player)

            # Try exact match with display name
            if player in display_to_id:
                return display_to_id[player]
            if player_norm in display_to_id:
                return display_to_id[player_norm]

            # Try team-qualified abbreviated name
            if team and (player, team) in name_team_to_id:
                return name_team_to_id[(player, team)]

            # Try abbreviated name alone
            if player in name_to_id:
                return name_to_id[player]

            # Try first initial + last name format
            parts = player.split()
            if len(parts) >= 2:
                abbrev = f"{parts[0][0]}.{parts[-1]}"
                if abbrev in name_to_id:
                    return name_to_id[abbrev]

            return ''

        df['player_id'] = df.apply(get_player_id, axis=1)
        matched = (df['player_id'] != '').sum()
        print(f"  Matched {matched}/{len(df)} players to player_id")

        # Add actual position
        def get_actual_position(row):
            player_id = row.get('player_id', '')
            if player_id and player_id in player_positions:
                return player_positions[player_id]
            return row.get('position', '')

        df['actual_position'] = df.apply(get_actual_position, axis=1)

    except Exception as e:
        print(f"  Warning: Could not map player IDs: {e}")
        df['player_id'] = ''

    return df


def add_defensive_rankings_to_df(df: pd.DataFrame, week: int, season: int = 2025) -> pd.DataFrame:
    """Add defensive rankings to a dataframe (works for both edge and XGBoost predictions).

    This is a standalone function that can be called for any recommendations dataframe.
    """
    if len(df) == 0:
        return df

    # First ensure we have player_id for position-specific lookups
    df = add_player_ids_to_df(df)

    # Load position-specific defense caches
    load_position_defense_caches()

    try:
        pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / 'pbp.parquet'
        if pbp_path.exists():
            pbp = pd.read_parquet(pbp_path)
            pbp_season = pbp[pbp['season'] == season]

            # Pass defense (vs WRs/TEs) - higher EPA = worse defense
            pass_plays = pbp_season[pbp_season['play_type'] == 'pass']
            pass_def_epa = pass_plays.groupby('defteam')['epa'].mean()
            pass_def_rank = pass_def_epa.rank(ascending=False)

            # Rush defense (vs RBs)
            rush_plays = pbp_season[pbp_season['play_type'] == 'run']
            rush_def_epa = rush_plays.groupby('defteam')['epa'].mean()
            rush_def_rank = rush_def_epa.rank(ascending=False)

            def get_def_rank_and_type(row):
                opponent = row.get('opponent_abbr', '') or row.get('opponent', '')
                market = str(row.get('market', '')).lower()
                player_id = row.get('player_id', '')
                team = row.get('team_abbr', '') or row.get('team', '')
                actual_pos = row.get('actual_position', '') or row.get('position', '')

                if not opponent:
                    return 16, 'vs WRs', 0

                if actual_pos and actual_pos not in ['', 'GAME', 'FLEX']:
                    position = actual_pos
                elif 'rush' in market:
                    position = 'RB'
                elif 'reception' in market or 'rec' in market:
                    position = 'WR'
                else:
                    position = 'WR'

                found_pos_rank = 0
                if player_id and team and position in ['WR', 'RB', 'TE']:
                    pos_actual, pos_rank = get_player_position_role(player_id, team, week, season)
                    if pos_actual and pos_rank:
                        found_pos_rank = pos_rank
                        spec_rank, spec_label = get_position_specific_defense_rank(
                            opponent, week, season, pos_actual, pos_rank, market
                        )
                        if spec_rank is not None:
                            return spec_rank, spec_label, found_pos_rank

                if position == 'RB':
                    return int(rush_def_rank.get(opponent, 16)), 'vs RBs', found_pos_rank
                elif position == 'QB':
                    qb_rank, qb_label = get_qb_defense_rank(opponent, week, season, market)
                    if qb_rank is not None:
                        return qb_rank, qb_label, 0
                    return int(pass_def_rank.get(opponent, 16)), 'vs QBs', 0
                else:
                    return int(pass_def_rank.get(opponent, 16)), 'vs WRs', found_pos_rank

            results = df.apply(get_def_rank_and_type, axis=1)
            df['def_rank'] = results.apply(lambda x: x[0])
            df['def_rank_type'] = results.apply(lambda x: x[1])
            df['pos_rank'] = results.apply(lambda x: x[2] if len(x) > 2 else 0)

            pos_specific = df['def_rank_type'].str.contains(r'vs [A-Z]{2}\ds|vs Pass|vs Compl', regex=True, na=False).sum()
            print(f"  Defensive rankings calculated: {pos_specific} position-specific, {len(df) - pos_specific} generic")
    except Exception as e:
        print(f"  Warning: Could not calculate defensive rankings: {e}")
        df['def_rank'] = 16
        df['def_rank_type'] = 'pass'

    return df


def format_bet_display(kelly_units: float, max_units: float = 0.1) -> str:
    """Format bet sizing with visual bar indicator.

    Args:
        kelly_units: Kelly fraction (0.01 = 1%, 0.1 = 10%)
        max_units: Maximum expected units for scaling bar (default 10%)
    """
    if not kelly_units or kelly_units <= 0:
        return '<span class="bet-display bet-none">-</span>'

    amount = kelly_units * 100  # Convert to dollars ($1 = 1%)
    bar_pct = min(100, (kelly_units / max_units) * 100)

    # Determine sizing class
    if kelly_units >= 0.08:
        size_class = 'bet-high'
    elif kelly_units >= 0.05:
        size_class = 'bet-med'
    else:
        size_class = 'bet-low'

    return f'''<div class="bet-display {size_class}">
        <span class="bet-amount">${amount:.0f}</span>
        <div class="bet-bar">
            <div class="bet-bar-fill" style="width: {bar_pct:.0f}%"></div>
        </div>
    </div>'''


def abbreviate_game(game: str) -> str:
    """Abbreviate game name from 'Philadelphia Eagles @ Los Angeles Chargers' to 'PHI @ LAC'."""
    if not game or game == 'N/A':
        return game

    team_abbrevs = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }

    # Handle "Team1 @ Team2" format
    if ' @ ' in game:
        parts = game.split(' @ ')
        away = team_abbrevs.get(parts[0].strip(), parts[0][:3].upper())
        home = team_abbrevs.get(parts[1].strip(), parts[1][:3].upper())
        return f"{away} @ {home}"

    return game


def format_prop_display(market: str) -> str:
    """Convert market name to display format."""
    display_map = {
        'player_pass_yds': 'Pass Yards',
        'player_rush_yds': 'Rush Yards',
        'player_reception_yds': 'Rec Yards',
        'player_receptions': 'Receptions',
        'player_pass_tds': 'Pass TDs',
        'player_rush_tds': 'Rush TDs',
        'player_rec_tds': 'Rec TDs',
        'player_anytime_td': 'Anytime TD',
        'player_1st_td': '1st TD',
        'player_pass_attempts': 'Pass Att',
        'player_pass_completions': 'Completions',
        'player_completions': 'Completions',
        'player_interceptions': 'INTs',
        'player_pass_longest_completion': 'Longest Pass',
        'player_rush_attempts': 'Rush Att',
        'player_rush_longest': 'Longest Rush',
        'player_targets': 'Targets',
        'player_longest_reception': 'Longest Rec',
    }
    return display_map.get(market, market.replace('player_', '').replace('_', ' ').title())


def get_tier_badge_class(tier: str) -> str:
    """Get CSS class for confidence tier badge."""
    tier_upper = str(tier).upper()
    if tier_upper == 'ELITE':
        return 'badge-elite'
    elif tier_upper == 'STRONG' or tier_upper == 'HIGH':
        return 'badge-high'
    elif tier_upper == 'MODERATE' or tier_upper == 'STANDARD':
        return 'badge-standard'
    elif tier_upper == 'CAUTION':
        return 'badge-caution'
    else:
        return 'badge-low'


def get_confidence_badge(prob: float) -> str:
    """Get confidence badge HTML."""
    pct = prob * 100 if prob <= 1 else prob
    if pct >= 65:
        css_class = 'badge-confidence-high'
    elif pct >= 55:
        css_class = 'badge-confidence-med'
    else:
        css_class = 'badge-confidence-low'
    return f'<span class="badge {css_class}">{pct:.1f}%</span>'


def get_edge_badge(edge: float) -> str:
    """Get edge badge HTML."""
    if edge >= 8:
        css_class = 'badge-edge-high'
    else:
        css_class = 'badge-edge-med'
    return f'<span class="badge {css_class}">{edge:.1f}%</span>'


def generate_css() -> str:
    """Generate professional CSS styles - COMPACT single-screen optimized."""
    return '''
        /* Import professional fonts - MUST be first */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            /* ========================================
               NFL QUANT PREMIUM THEME v2.0
               Inspired by Pikkit, Bet Warrior designs
               Deep navy + Electric blue + Glassmorphism
               ======================================== */

            /* === SURFACE ELEVATION SYSTEM === */
            /* Deep navy base with blue undertones */
            --surface-base: #030712;           /* Almost black with blue tint */
            --surface-raised: #0C1222;         /* Cards on background */
            --surface-overlay: #111827;        /* Modals, dropdowns */
            --surface-elevated: #1E293B;       /* Nested cards, tooltips */
            --surface-highlight: #334155;      /* Hover states */

            /* === BACKGROUNDS === */
            --bg-page: linear-gradient(180deg, #030712 0%, #0C1222 50%, #030712 100%);
            --bg-primary: #030712;
            --bg-modal: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(3, 7, 18, 0.99) 100%);
            --bg-modal-solid: #0C1222;
            --bg-secondary: #0C1222;
            --bg-card: rgba(15, 23, 42, 0.8);
            --bg-card-hover: rgba(30, 41, 59, 0.9);
            --bg-hover: #1E293B;
            --bg-elevated: #1E293B;
            --bg-tertiary: #111827;
            --bg-section: #0C1222;
            --bg-subtle: rgba(15, 23, 42, 0.6);

            /* === GLASSMORPHISM SURFACES === */
            --glass-surface: rgba(15, 23, 42, 0.7);
            --glass-border: rgba(59, 130, 246, 0.15);
            --glass-highlight: rgba(255, 255, 255, 0.05);
            --glass-blur: 20px;

            /* === TEXT HIERARCHY === */
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-tertiary: #94A3B8;
            --text-muted: #64748B;
            --text-faint: #475569;
            --text-light: #94A3B8;
            --text-disabled: #475569;
            --text-inverse: #0F172A;

            /* === BORDERS === */
            --border-default: rgba(59, 130, 246, 0.2);
            --border-muted: rgba(148, 163, 184, 0.1);
            --border-emphasis: rgba(59, 130, 246, 0.3);
            --border-color: rgba(59, 130, 246, 0.15);
            --border-subtle: rgba(148, 163, 184, 0.08);
            --border-light: rgba(148, 163, 184, 0.05);
            --border-strong: rgba(59, 130, 246, 0.4);
            --border-focus: #3B82F6;

            /* === SEMANTIC COLORS === */
            --positive: #10B981;
            --positive-emphasis: #059669;
            --positive-muted: rgba(16, 185, 129, 0.15);
            --positive-glow: rgba(16, 185, 129, 0.5);

            --negative: #EF4444;
            --negative-emphasis: #DC2626;
            --negative-muted: rgba(239, 68, 68, 0.15);
            --negative-glow: rgba(239, 68, 68, 0.5);

            --accent: #3B82F6;
            --accent-emphasis: #2563EB;
            --accent-muted: rgba(59, 130, 246, 0.15);
            --accent-glow: rgba(59, 130, 246, 0.5);

            --warning: #F59E0B;
            --warning-muted: rgba(245, 158, 11, 0.15);

            --neutral: #64748B;

            /* === WIN/LOSS COLORS === */
            --win: #10B981;
            --win-soft: rgba(16, 185, 129, 0.15);
            --win-border: rgba(16, 185, 129, 0.4);
            --loss: #EF4444;
            --loss-soft: rgba(239, 68, 68, 0.15);
            --loss-border: rgba(239, 68, 68, 0.4);

            /* === PICK DIRECTION COLORS === */
            --over: #10B981;
            --over-primary: #10B981;
            --over-light: #34D399;
            --over-glow: rgba(16, 185, 129, 0.5);
            --over-soft: rgba(16, 185, 129, 0.12);
            --over-bg: rgba(16, 185, 129, 0.15);
            --over-border: rgba(16, 185, 129, 0.4);

            --under: #3B82F6;
            --under-primary: #3B82F6;
            --under-light: #60A5FA;
            --under-glow: rgba(59, 130, 246, 0.5);
            --under-soft: rgba(59, 130, 246, 0.12);
            --under-bg: rgba(59, 130, 246, 0.15);
            --under-border: rgba(59, 130, 246, 0.4);

            /* === BRAND COLORS (Electric Blue Primary) === */
            --accent-primary: #3B82F6;
            --accent-primary-dark: #2563EB;
            --accent-primary-light: rgba(59, 130, 246, 0.15);
            --brand-primary: #3B82F6;
            --brand-secondary: #60A5FA;
            --accent-green: #10B981;
            --accent-red: #EF4444;
            --accent-amber: #F59E0B;
            --accent-cyan: #06B6D4;

            /* === CONFIDENCE TIERS (Blue-based) === */
            --conf-elite: #3B82F6;
            --conf-high: #06B6D4;
            --conf-med: #F59E0B;
            --conf-low: #64748B;

            /* === TIER BADGES (Premium Blue gradients) === */
            --tier-elite-bg: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(37, 99, 235, 0.2) 100%);
            --tier-elite-text: #60A5FA;
            --tier-elite-border: rgba(59, 130, 246, 0.5);
            --tier-elite-glow: 0 0 20px rgba(59, 130, 246, 0.4);

            --tier-strong-bg: linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(8, 145, 178, 0.2) 100%);
            --tier-strong-text: #22D3EE;
            --tier-strong-border: rgba(6, 182, 212, 0.5);
            --tier-strong-glow: 0 0 20px rgba(6, 182, 212, 0.4);

            --tier-moderate-bg: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.15) 100%);
            --tier-moderate-text: #FBBF24;
            --tier-moderate-border: rgba(245, 158, 11, 0.4);

            --tier-caution-bg: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.15) 100%);
            --tier-caution-text: #F87171;
            --tier-caution-border: rgba(239, 68, 68, 0.4);

            /* === EDGE (Cyan) === */
            --edge: #06B6D4;
            --edge-glow: rgba(6, 182, 212, 0.5);

            /* === PREMIUM ACCENTS === */
            --gold: #FBBF24;
            --gold-glow: rgba(251, 191, 36, 0.5);
            --platinum: #E2E8F0;

            /* === STAR RATING === */
            --star-filled: #FBBF24;
            --star-empty: #475569;
            --star-glow: 0 0 12px rgba(251, 191, 36, 0.6);

            /* === SHADOW SYSTEM (Deep premium depth) === */
            --shadow-sm:
                0 1px 3px rgba(0, 0, 0, 0.4),
                0 0 0 1px rgba(59, 130, 246, 0.05) inset;
            --shadow-md:
                0 4px 16px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(59, 130, 246, 0.08) inset;
            --shadow-lg:
                0 8px 32px rgba(0, 0, 0, 0.6),
                0 0 0 1px rgba(59, 130, 246, 0.1) inset;
            --shadow-xl:
                0 16px 48px rgba(0, 0, 0, 0.7),
                0 0 0 1px rgba(59, 130, 246, 0.12) inset;
            --shadow-card:
                0 8px 32px rgba(0, 0, 0, 0.4),
                0 0 0 1px rgba(59, 130, 246, 0.1) inset,
                0 0 80px -20px rgba(59, 130, 246, 0.15);

            /* Glow shadows */
            --shadow-glow-positive: 0 0 30px var(--positive-glow);
            --shadow-glow-negative: 0 0 30px var(--negative-glow);
            --shadow-glow-accent: 0 0 30px var(--accent-glow);
            --shadow-glow-over: 0 0 25px var(--over-glow);
            --shadow-glow-under: 0 0 25px var(--under-glow);

            /* === TYPOGRAPHY SCALE (1.25 ratio - Major Third) === */
            --text-5xl: 3rem;        /* 48px - Hero */
            --text-4xl: 2.441rem;    /* 39px - Big stats */
            --text-3xl: 1.953rem;    /* 31px - Page titles */
            --text-2xl: 1.563rem;    /* 25px - Section headers */
            --text-xl: 1.25rem;      /* 20px - Card titles */
            --text-lg: 1.125rem;     /* 18px - Subheadings */
            --text-base: 1rem;       /* 16px - Body */
            --text-sm: 0.875rem;     /* 14px - Small text */
            --text-xs: 0.75rem;      /* 12px - Captions */

            /* === LINE HEIGHTS === */
            --leading-none: 1;
            --leading-tight: 1.2;
            --leading-snug: 1.375;
            --leading-normal: 1.5;
            --leading-relaxed: 1.625;
            --leading-loose: 1.8;

            /* === LETTER SPACING === */
            --tracking-tighter: -0.04em;
            --tracking-tight: -0.02em;
            --tracking-normal: 0;
            --tracking-wide: 0.025em;
            --tracking-wider: 0.05em;
            --tracking-widest: 0.1em;

            /* === FONT WEIGHTS === */
            --weight-normal: 400;
            --weight-medium: 500;
            --weight-semibold: 600;
            --weight-bold: 700;
            --weight-extrabold: 800;

            /* === FONT FAMILIES === */
            --font-display: 'Inter var', 'SF Pro Display', -apple-system, system-ui, sans-serif;
            --font-body: 'Inter', 'SF Pro Text', -apple-system, system-ui, sans-serif;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;

            /* === SPACING SCALE (More generous) === */
            --space-px: 1px;
            --space-0: 0;
            --space-1: 0.25rem;   /* 4px */
            --space-2: 0.5rem;    /* 8px */
            --space-3: 0.75rem;   /* 12px */
            --space-4: 1rem;      /* 16px */
            --space-5: 1.25rem;   /* 20px */
            --space-6: 1.5rem;    /* 24px */
            --space-8: 2rem;      /* 32px */
            --space-10: 2.5rem;   /* 40px */
            --space-12: 3rem;     /* 48px */
            --space-16: 4rem;     /* 64px */
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;
            --space-2xl: 48px;

            /* === RADIUS SYSTEM === */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --radius-2xl: 24px;
            --radius-full: 9999px;
        }

        /* === SMOOTH BASE TRANSITIONS === */
        *, *::before, *::after {
            transition-property: background-color, border-color, color, fill, stroke, opacity, box-shadow, transform;
            transition-duration: 150ms;
            transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* === PREMIUM DARK MODE LAYOUT === */
        body {
            font-family: var(--font-sans);
            background: var(--surface-base);
            color: var(--text-primary);
            padding: var(--space-lg);
            min-height: 100vh;
            font-size: var(--text-base);
            line-height: var(--leading-normal);
            font-weight: var(--weight-medium);
            letter-spacing: var(--tracking-normal);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            /* Prevent scroll chaining on mobile */
            overscroll-behavior: contain;
        }

        /* === PREMIUM CONTAINER (Glass Card) === */
        .container {
            position: relative;
            max-width: 1800px;
            margin: 0 auto;
            background: linear-gradient(
                135deg,
                rgba(22, 27, 34, 0.85) 0%,
                rgba(13, 17, 23, 0.95) 100%
            );
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: var(--radius-2xl);
            border: 1px solid var(--glass-border);
            box-shadow:
                0 0 0 1px rgba(255, 255, 255, 0.02) inset,
                0 8px 32px rgba(0, 0, 0, 0.45);
            /* Removed overflow:hidden - breaks sticky positioning */
        }

        /* Subtle top highlight for depth */
        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.08) 50%,
                transparent
            );
            pointer-events: none;
        }

        /* === STICKY HEADER WRAPPER === */
        .header-wrapper {
            position: sticky;
            top: 0;
            z-index: 100;
            background: linear-gradient(180deg, rgba(3, 7, 18, 0.98) 0%, rgba(12, 18, 34, 0.95) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(59, 130, 246, 0.1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3), 0 0 60px rgba(59, 130, 246, 0.05);
        }

        /* === PREMIUM HEADER === */
        .header {
            background: transparent;
            color: var(--text-primary);
            padding: 18px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 72px;
            position: relative;
        }

        /* Subtle gradient underline */
        .header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 28px;
            right: 28px;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2) 20%, rgba(59, 130, 246, 0.2) 80%, transparent);
        }

        .header h1 {
            font-family: var(--font-display);
            font-size: var(--text-2xl);
            font-weight: var(--weight-bold);
            letter-spacing: var(--tracking-tight);
            color: var(--text-primary);
            text-shadow: 0 0 50px rgba(59, 130, 246, 0.3);
        }

        /* QUANT text highlight */
        .header h1 .brand-accent {
            color: #3B82F6;
            text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
        }

        .header p {
            color: var(--text-tertiary);
            font-size: var(--text-xs);
            font-family: var(--font-mono);
            letter-spacing: var(--tracking-wide);
        }

        /* === PREMIUM KPI SUMMARY BAR === */
        .stats-summary {
            display: flex;
            gap: 1px;
            background: var(--border-muted);
            min-height: 70px;
        }

        .stat-card {
            position: relative;
            background: var(--surface-raised);
            padding: 12px 16px;
            flex: 1;
            text-align: center;
            min-width: 90px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .stat-card:hover {
            background: var(--surface-elevated);
            transform: translateY(-1px);
        }

        .stat-card.primary {
            background: linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.08) 0%,
                var(--surface-raised) 100%
            );
            border-bottom: 2px solid var(--positive);
            box-shadow: inset 0 -20px 40px -30px rgba(59, 130, 246, 0.15);
        }

        /* === HERO STAT VALUES === */
        .stat-value {
            font-family: var(--font-mono);
            font-size: var(--text-xl);
            font-weight: var(--weight-bold);
            color: var(--text-primary);
            line-height: var(--leading-none);
            letter-spacing: var(--tracking-tight);
        }

        .stat-value.highlight {
            color: var(--positive);
            text-shadow: 0 0 24px var(--positive-glow);
        }

        .stat-value.warning {
            color: var(--warning);
            text-shadow: 0 0 20px rgba(210, 153, 34, 0.3);
        }

        .stat-label {
            font-size: var(--text-xs);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-widest);
            margin-top: 6px;
            font-weight: var(--weight-semibold);
        }

        .stat-sublabel {
            font-size: var(--text-xs);
            color: var(--text-faint);
        }

        /* TABS - Dark Mode Style */
        /* === PREMIUM TABS === */
        .tabs {
            display: flex;
            gap: 8px;
            background: var(--surface-raised);
            padding: 10px 20px;
            overflow-x: auto;
            border-bottom: 1px solid var(--border-muted);
        }

        .tab {
            position: relative;
            padding: 10px 18px;
            cursor: pointer;
            border: 1px solid transparent;
            background: transparent;
            font-size: var(--text-sm);
            font-weight: var(--weight-medium);
            color: var(--text-tertiary);
            border-radius: var(--radius-lg);
            transition: all 0.2s ease;
            white-space: nowrap;
            font-family: var(--font-sans);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.04);
            border-color: var(--border-muted);
        }

        .tab.active {
            background: linear-gradient(135deg,
                rgba(59, 130, 246, 0.25) 0%,
                rgba(37, 99, 235, 0.2) 100%
            );
            color: #60A5FA;
            font-weight: var(--weight-semibold);
            border: 1px solid rgba(59, 130, 246, 0.4);
            box-shadow:
                0 0 0 1px rgba(59, 130, 246, 0.1) inset,
                0 0 25px rgba(59, 130, 246, 0.25);
        }

        .tab.active:hover {
            box-shadow:
                0 0 0 1px rgba(59, 130, 246, 0.15) inset,
                0 0 35px rgba(59, 130, 246, 0.35);
            transform: translateY(-1px);
        }

        .tab-count {
            background: rgba(0, 0, 0, 0.2);
            color: inherit;
            font-size: var(--text-xs);
            padding: 2px 8px;
            border-radius: var(--radius-full);
            font-family: var(--font-mono);
        }

        .tab:not(.active) .tab-count {
            background: var(--bg-elevated);
            color: var(--text-muted);
        }

        .tab.active .tab-count {
            background: rgba(255,255,255,0.25);
            color: var(--text-inverse);
        }

        /* === TIER LEGEND === */
        .tier-legend {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 20px;
            background: var(--surface-raised);
            border-bottom: 1px solid var(--border-muted);
            font-size: 11px;
        }

        .tier-legend-label {
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .tier-legend-item {
            padding: 3px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 10px;
        }

        .tier-elite-legend {
            background: var(--tier-elite-bg);
            color: var(--tier-elite-text);
            border: 1px solid var(--tier-elite-border);
        }

        .tier-strong-legend {
            background: var(--tier-strong-bg);
            color: var(--tier-strong-text);
            border: 1px solid var(--tier-strong-border);
        }

        .tier-moderate-legend {
            background: var(--tier-moderate-bg);
            color: var(--tier-moderate-text);
            border: 1px solid var(--tier-moderate-border);
        }

        /* === SEARCH BAR === */
        .search-bar-container {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 20px;
            background: var(--surface-raised);
            border-bottom: 1px solid var(--border-muted);
        }

        .search-input {
            flex: 0 0 250px;
            padding: 10px 16px;
            background: var(--surface-overlay);
            border: 1px solid var(--border-default);
            border-radius: var(--radius-lg);
            color: var(--text-primary);
            font-size: 14px;
            font-family: var(--font-sans);
            outline: none;
            transition: all 0.2s ease;
        }

        .search-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-muted);
        }

        .search-input::placeholder {
            color: var(--text-muted);
        }

        .filter-chips {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .filter-chip {
            padding: 6px 14px;
            background: var(--surface-overlay);
            border: 1px solid var(--border-muted);
            border-radius: var(--radius-full);
            color: var(--text-tertiary);
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .filter-chip:hover {
            background: var(--surface-elevated);
            color: var(--text-primary);
            border-color: var(--border-default);
        }

        .filter-chip.active {
            background: var(--accent-muted);
            color: var(--accent);
            border-color: var(--accent);
        }

        /* === ALL PICKS VIEW === */
        .all-picks-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .all-picks-header h2 {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
        }

        .picks-count {
            font-size: 13px;
            color: var(--text-muted);
        }

        .all-picks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
        }

        /* === MOBILE RESPONSIVE === */
        @media (max-width: 768px) {
            body {
                padding: 8px;
            }

            .header {
                padding: 12px 16px;
                height: auto;
                flex-wrap: wrap;
                gap: 8px;
            }

            .header h1 {
                font-size: 18px;
            }

            .stats-summary {
                flex-wrap: wrap;
            }

            .stat-card {
                min-width: 80px;
                padding: 10px 12px;
            }

            .stat-value {
                font-size: 16px;
            }

            .stat-label {
                font-size: 9px;
            }

            .search-bar-container {
                flex-direction: column;
                gap: 10px;
                padding: 10px 16px;
            }

            .search-input {
                flex: 1 1 100%;
                width: 100%;
            }

            .filter-chips {
                width: 100%;
                justify-content: flex-start;
                overflow-x: auto;
                padding-bottom: 4px;
                /* Scroll indicator gradient */
                mask-image: linear-gradient(to right, black 90%, transparent 100%);
                -webkit-mask-image: linear-gradient(to right, black 90%, transparent 100%);
            }

            .filter-chip {
                flex-shrink: 0;
                padding: 5px 12px;
                font-size: 11px;
            }

            .tabs {
                padding: 8px 12px;
                gap: 6px;
                overflow-x: auto;
                /* Scroll indicator gradient */
                mask-image: linear-gradient(to right, black 85%, transparent 100%);
                -webkit-mask-image: linear-gradient(to right, black 85%, transparent 100%);
            }

            .tab {
                padding: 8px 12px;
                font-size: 12px;
                flex-shrink: 0;
            }

            /* Hide tier legend on mobile - too much space */
            .tier-legend {
                display: none;
            }

            .view-section {
                padding: 16px;
                padding-top: 12px;
                /* Add bottom padding so FAB doesn't cover last card */
                padding-bottom: 80px;
            }

            .all-picks-grid {
                grid-template-columns: 1fr;
                gap: 12px;
            }

            .featured-elite-grid {
                grid-template-columns: 1fr;
                gap: 12px;
            }

            /* Smaller FAB on mobile */
            .bet-slip-toggle {
                bottom: 20px;
                right: 16px;
                padding: 10px 16px;
                font-size: 12px;
                border-radius: 30px;
            }

            .bet-slip-text {
                display: none;
            }

            .bet-slip-icon {
                display: block;
                font-size: 18px;
            }

            .bet-slip-toggle .count {
                width: 22px;
                height: 22px;
                font-size: 11px;
            }

            .bet-slip-panel {
                width: 100%;
                right: -100%;
            }

            /* Compact cards on mobile */
            .game-pick-card, .featured-pick-card {
                padding: 12px;
            }

            .card-hero {
                padding: 10px 12px;
            }

            .card-hero-pick {
                font-size: 18px;
            }

            .card-player-avatar img {
                width: 44px !important;
                height: 44px !important;
            }
        }

        @media (min-width: 769px) {
            .bet-slip-icon {
                display: none;
            }
        }

        /* Custom Scrollbar - Dark Mode */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-elevated);
            border-radius: var(--radius-sm);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: var(--radius-sm);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--border-strong);
        }

        /* Firefox scrollbar */
        * {
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) var(--bg-elevated);
        }

        /* VIEW SECTIONS - Dark Mode */
        .view-section {
            display: none;
            padding: var(--space-xl);
            /* Remove nested scroll - use natural page scroll */
        }

        .view-section.active {
            display: block;
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-xl);
        }

        h2 {
            font-size: var(--text-lg);
            font-weight: var(--weight-semibold);
            color: var(--text-primary);
            font-family: var(--font-sans);
        }

        h3 {
            font-size: var(--text-base);
            font-weight: var(--weight-semibold);
            color: var(--text-primary);
        }

        .filter-bar {
            display: flex;
            gap: var(--space-sm);
            margin-bottom: var(--space-xl);
            flex-wrap: wrap;
            align-items: center;
        }

        .filter-group {
            display: flex;
            gap: var(--space-xs);
        }

        .filter-btn {
            padding: 8px 14px;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
            color: var(--text-secondary);
            border-radius: var(--radius-md);
            font-size: var(--text-sm);
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: var(--font-sans);
        }

        .filter-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--border-strong);
        }

        .filter-btn.active {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
            color: var(--text-inverse);
        }

        .search-box {
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
            color: var(--text-primary);
            border-radius: var(--radius-md);
            font-size: var(--text-sm);
            width: 220px;
            font-family: var(--font-mono);
        }

        .search-box::placeholder {
            color: var(--text-muted);
        }

        .search-box:focus {
            outline: none;
            border-color: var(--accent-primary);
        }

        .expand-collapse-controls {
            display: flex;
            gap: 4px;
        }

        .expand-collapse-controls button {
            background: var(--bg-card);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            font-weight: 500;
            transition: all 0.1s ease;
        }

        .expand-collapse-controls button:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .collapsible-container {
            margin-bottom: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .collapsible-header {
            padding: 10px 16px;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.15s ease;
        }

        .collapsible-header:hover {
            background: var(--bg-hover);
        }

        .toggle-icon {
            font-size: 12px;
            transition: transform 0.2s ease;
            color: var(--text-muted);
        }

        .toggle-icon.collapsed {
            transform: rotate(-90deg);
        }

        .header-stats {
            color: var(--text-muted);
            font-size: 11px;
            font-weight: normal;
            margin-left: auto;
            font-family: 'SF Mono', monospace;
        }

        .game-time-badge {
            background: var(--bg-hover) !important;
            color: var(--text-secondary) !important;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin-left: 8px;
            font-family: 'SF Mono', monospace;
        }

        .collapsible-content {
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
        }

        .collapsible-content.collapsed {
            max-height: 0;
        }

        /* TABLE STYLING - Dark Mode */
        .table-wrapper {
            overflow-x: auto;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            font-size: var(--text-sm);
        }

        thead {
            position: sticky;
            top: 0;
            z-index: 10;
            background: var(--bg-elevated);
        }

        th {
            padding: 12px 10px;
            text-align: left;
            font-size: var(--text-xs);
            font-weight: var(--weight-semibold);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            user-select: none;
            transition: background 0.1s ease;
            white-space: nowrap;
        }

        th:hover {
            background: var(--bg-hover);
            color: var(--text-secondary);
        }

        /* Sort indicators */
        th.sorted-asc::after {
            content: '↑';
            color: var(--accent-primary);
            font-size: var(--text-xs);
            margin-left: 4px;
        }

        th.sorted-desc::after {
            content: '↓';
            color: var(--accent-primary);
            font-size: var(--text-xs);
            margin-left: 4px;
        }

        th:not(.sorted-asc):not(.sorted-desc):hover::after {
            content: '↕';
            color: var(--text-muted);
            font-size: var(--text-xs);
            margin-left: 4px;
        }

        /* === PREMIUM TABLE CELLS === */
        td {
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-muted);
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: var(--text-sm);
        }

        /* === PREMIUM TABLE ROWS === */
        tbody tr {
            transition: all 0.2s ease;
            height: 48px;
        }

        tbody tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.015);
        }

        tbody tr:hover {
            background: linear-gradient(
                90deg,
                rgba(88, 166, 255, 0.06) 0%,
                transparent 100%
            ) !important;
        }

        /* === PREMIUM EXPANDABLE ROWS === */
        .expandable-row {
            cursor: pointer;
            position: relative;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }

        .expandable-row:hover {
            border-left-color: var(--accent);
            background: linear-gradient(
                90deg,
                rgba(88, 166, 255, 0.08) 0%,
                rgba(88, 166, 255, 0.02) 50%,
                transparent 100%
            ) !important;
        }

        .expandable-row.expanded {
            background: var(--surface-raised);
            border-left-color: var(--positive);
            box-shadow: inset 3px 0 0 var(--positive);
        }

        .expandable-row td:first-child {
            position: relative;
            padding-left: 36px !important;
        }

        .expandable-row td:first-child::before {
            content: '›';
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 16px;
            font-weight: 300;
            color: var(--text-faint);
            transition: all 0.2s ease;
        }

        .expandable-row.expanded td:first-child::before {
            content: '‹';
            transform: translateY(-50%) rotate(-90deg);
            color: var(--positive);
        }

        .expandable-row:hover td:first-child::before {
            color: var(--accent);
            transform: translateY(-50%) translateX(2px);
        }

        .expandable-row.expanded:hover td:first-child::before {
            transform: translateY(-50%) rotate(-90deg);
        }

        /* Click hint on hover */
        .expandable-row::after {
            content: 'click to expand';
            position: absolute;
            right: 16px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            font-weight: var(--weight-medium);
            color: var(--text-faint);
            opacity: 0;
            transition: all 0.2s ease;
            pointer-events: none;
            text-transform: uppercase;
            letter-spacing: var(--tracking-wide);
        }

        .expandable-row:hover::after {
            opacity: 0.6;
        }

        .expandable-row.expanded::after {
            content: 'click to collapse';
        }

        /* Subtle hint text (legacy support) */
        .expand-hint {
            font-size: 9px;
            color: var(--text-faint);
            opacity: 0;
            transition: opacity 0.2s ease;
            margin-left: 8px;
        }

        .expandable-row:hover .expand-hint {
            opacity: 1;
        }

        /* === PREMIUM LOGIC ROW (Expandable Details) === */
        .logic-row {
            border-top: none !important;
        }

        .logic-row td {
            padding: 0 !important;
            border-bottom: 1px solid var(--border-muted) !important;
        }

        .logic-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border-muted);
            margin: 0;
        }

        .logic-section {
            background: linear-gradient(
                180deg,
                var(--surface-raised) 0%,
                rgba(15, 20, 25, 0.95) 100%
            );
            padding: 16px 18px;
            min-height: auto;
        }

        .logic-section-title {
            font-size: 10px;
            font-weight: var(--weight-bold);
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: var(--tracking-widest);
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-muted);
            text-shadow: 0 0 20px var(--accent-glow);
        }

        .logic-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            margin-bottom: 10px;
            line-height: 1.4;
            padding: 4px 0;
            border-radius: var(--radius-sm);
            transition: background 0.15s ease;
        }

        .logic-item:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .logic-item:last-child {
            margin-bottom: 0;
        }

        .logic-item-label {
            color: var(--text-tertiary);
            font-size: 10px;
            font-weight: var(--weight-medium);
            flex-shrink: 0;
            margin-right: 12px;
        }

        .logic-item-value {
            color: var(--text-primary);
            font-weight: var(--weight-semibold);
            font-family: var(--font-mono);
            font-size: 11px;
            text-align: right;
            white-space: nowrap;
        }

        .info-indicator {
            color: var(--text-faint);
            font-size: 10px;
            margin-left: 4px;
        }

        /* === PREMIUM BADGE SYSTEM === */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: var(--space-1);
            padding: 5px 12px;
            border-radius: var(--radius-full);
            font-size: var(--text-xs);
            font-weight: var(--weight-semibold);
            font-family: var(--font-mono);
            letter-spacing: var(--tracking-wide);
            text-transform: uppercase;
            transition: all 0.15s ease;
        }

        /* ============================================================
           TIER BADGES - Premium gradients with glow effects
           ============================================================ */

        /* ELITE - Solid purple with white text */
        .badge-elite {
            background: linear-gradient(135deg,
                rgba(59, 130, 246, 0.95) 0%,
                rgba(37, 99, 235, 0.9) 100%
            );
            color: #FFFFFF;
            font-weight: var(--weight-bold);
            border: none;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 2px 8px rgba(59, 130, 246, 0.4),
                0 0 20px rgba(59, 130, 246, 0.25);
            animation: elitePulse 3s ease-in-out infinite;
        }

        @keyframes elitePulse {
            0%, 100% { box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 2px 8px rgba(59, 130, 246, 0.4), 0 0 20px rgba(59, 130, 246, 0.25); }
            50% { box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 4px 12px rgba(59, 130, 246, 0.5), 0 0 30px rgba(59, 130, 246, 0.4); }
        }

        .badge-elite:hover {
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.25),
                0 4px 12px rgba(59, 130, 246, 0.5),
                0 0 30px rgba(59, 130, 246, 0.35);
            transform: translateY(-1px);
            animation: none;
        }

        /* HIGH/STRONG - Solid blue with dark text */
        .badge-high {
            background: linear-gradient(135deg,
                rgba(88, 166, 255, 0.95) 0%,
                rgba(56, 139, 253, 0.9) 100%
            );
            color: #0D1117;
            font-weight: var(--weight-semibold);
            border: none;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 2px 8px rgba(88, 166, 255, 0.35);
        }

        .badge-high:hover {
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.25),
                0 4px 12px rgba(88, 166, 255, 0.45);
            transform: translateY(-1px);
        }

        /* STANDARD/MODERATE - Gradient bg with colored text */
        .badge-standard {
            background: linear-gradient(135deg,
                rgba(210, 153, 34, 0.2) 0%,
                rgba(187, 128, 9, 0.15) 100%
            );
            color: var(--warning);
            border: 1px solid rgba(210, 153, 34, 0.3);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        /* LOW - Muted/subtle */
        .badge-low {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            border: 1px dashed rgba(240, 246, 252, 0.1);
        }

        /* CAUTION - Red gradient bg with red text */
        .badge-caution {
            background: linear-gradient(135deg,
                rgba(248, 81, 73, 0.2) 0%,
                rgba(218, 54, 51, 0.15) 100%
            );
            color: var(--negative);
            border: 1px solid rgba(248, 81, 73, 0.3);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        /* === CONFIDENCE BADGES (Purple) === */
        .badge-confidence-high {
            background: linear-gradient(135deg,
                rgba(59, 130, 246, 0.2) 0%,
                rgba(37, 99, 235, 0.15) 100%
            );
            color: var(--positive);
            font-weight: var(--weight-bold);
            border: 1px solid rgba(59, 130, 246, 0.3);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.05),
                0 0 16px rgba(59, 130, 246, 0.2);
        }

        .badge-confidence-med {
            background: linear-gradient(135deg,
                rgba(88, 166, 255, 0.15) 0%,
                rgba(56, 139, 253, 0.1) 100%
            );
            color: #58A6FF;
            border: 1px solid rgba(88, 166, 255, 0.25);
        }

        .badge-confidence-low {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            border: 1px solid var(--border-muted);
        }

        /* === EDGE BADGES - Purple glowing border === */
        .badge-edge-high {
            background: linear-gradient(135deg,
                rgba(59, 130, 246, 0.15) 0%,
                rgba(37, 99, 235, 0.08) 100%
            );
            border: 2px solid var(--positive);
            color: var(--positive);
            font-weight: var(--weight-bold);
            box-shadow:
                0 0 16px var(--positive-glow),
                inset 0 0 8px rgba(59, 130, 246, 0.1);
        }

        .badge-edge-high:hover {
            box-shadow:
                0 0 24px var(--positive-glow),
                inset 0 0 12px rgba(59, 130, 246, 0.15);
        }

        .badge-edge-med {
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.4);
            color: var(--positive);
        }

        .badge-edge-low {
            background: transparent;
            color: var(--text-muted);
            border: 1px solid var(--border-default);
        }

        /* BET DISPLAY - Visual bar indicator for bet sizing */
        .bet-display {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 50px;
        }

        .bet-display.bet-none {
            color: var(--text-muted);
            text-align: center;
        }

        .bet-amount {
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }

        .bet-bar {
            height: 4px;
            background: var(--bg-hover);
            border-radius: 2px;
            overflow: hidden;
        }

        .bet-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }

        .bet-display.bet-high .bet-amount { color: var(--accent-green); }
        .bet-display.bet-high .bet-bar-fill { background: var(--accent-green); }

        .bet-display.bet-med .bet-amount { color: var(--accent-cyan); }
        .bet-display.bet-med .bet-bar-fill { background: var(--accent-cyan); }

        .bet-display.bet-low .bet-amount { color: var(--text-secondary); }
        .bet-display.bet-low .bet-bar-fill { background: var(--text-muted); }

        /* === DIRECTION COLORS WITH GLOW === */
        .direction-over {
            color: var(--positive);
            text-shadow: 0 0 12px var(--positive-glow);
        }

        .direction-under {
            color: var(--negative);
            text-shadow: 0 0 12px var(--negative-glow);
        }

        /* Team Logo Styles */
        .team-logo {
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 6px;
            border-radius: 2px;
        }

        .team-logo-small {
            width: 16px;
            height: 16px;
        }

        .player-cell {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .player-name {
            font-weight: 500;
        }

        .team-abbr {
            font-size: 10px;
            color: var(--text-muted);
            background: var(--bg-hover);
            padding: 1px 4px;
            border-radius: 2px;
        }

        /* SHAP Factor Display in Expanded Rows */
        .shap-factors {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 8px;
        }

        .shap-group {
            padding: 10px 12px;
            border-radius: 4px;
        }

        .shap-group-over {
            background: rgba(16, 185, 129, 0.08);
            border-left: 3px solid var(--accent-green);
        }

        .shap-group-under {
            background: rgba(239, 68, 68, 0.08);
            border-left: 3px solid var(--accent-red);
        }

        .shap-group-title {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .shap-group-over .shap-group-title {
            color: var(--accent-green);
        }

        .shap-group-under .shap-group-title {
            color: var(--accent-red);
        }

        .shap-factor {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            margin-bottom: 4px;
            color: var(--text-primary);
        }

        .shap-factor-name {
            color: var(--text-secondary);
        }

        .shap-factor-value {
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }

        .shap-factor-dominant {
            color: var(--accent-amber);
        }

        .shap-factor-dominant::after {
            content: ' *';
            color: var(--accent-amber);
        }

        /* ================================================
           GAME CARD STYLES - ESPN/Action Network Aesthetic
           ================================================ */

        /* ============================================
           GAME CARDS GRID - Fixed 2 columns for consistent layout
           FIX (Dec 14, 2025): Changed from minmax(450px, 1fr) to fixed 2 columns
           ============================================ */
        .game-cards-grid {
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 16px 0;
        }

        /* Each game takes full width, picks inside use grid */
        .game-card {
            width: 100%;
        }

        /* Prop Type Card Grid */
        .prop-cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 14px;
            padding: 12px 0;
        }

        /* Player Summary Cards Grid */
        .player-cards-grid {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .player-summary-card {
            background: var(--glass-surface);
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            overflow: hidden;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-md);
        }

        .player-summary-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            box-shadow: var(--shadow-lg), 0 0 30px rgba(59, 130, 246, 0.15);
        }

        .player-summary-header {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 16px 20px;
            cursor: pointer;
            transition: background 0.1s ease;
        }

        .player-summary-header:hover {
            background: var(--bg-hover);
        }

        .player-summary-info {
            flex: 1;
            min-width: 0;
        }

        .player-summary-name {
            display: block;
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
        }

        .player-summary-meta {
            display: block;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .player-summary-stats {
            display: flex;
            gap: 16px;
        }

        .player-summary-stat {
            text-align: center;
        }

        .player-summary-value {
            display: block;
            font-size: 14px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }

        .player-summary-label {
            display: block;
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 2px;
        }

        .player-expand-icon {
            font-size: 12px;
            color: var(--text-muted);
            transition: transform 0.2s ease;
        }

        .player-summary-card.expanded .player-expand-icon {
            transform: rotate(180deg);
        }

        .player-picks-container {
            background: var(--bg-secondary);
            padding: 8px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 8px;
            border-top: 1px solid var(--border-color);
        }

        /* Header badges for elite/strong counts */
        .header-elite-badge {
            background: rgba(59, 130, 246, 0.15);
            color: #3B82F6;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }

        .header-strong-badge {
            background: rgba(59, 130, 246, 0.15);
            color: #3B82F6;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }

        .game-card {
            background: var(--glass-surface);
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-md);
        }

        .game-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            box-shadow: var(--shadow-lg), 0 0 30px rgba(59, 130, 246, 0.15);
        }

        /* Game header - clickable to expand */
        .game-card-header {
            padding: 14px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            background: var(--bg-card);
            transition: background 0.1s ease;
            border-radius: 8px 8px 0 0;
            min-height: 56px;
        }

        .game-card-header:hover {
            background: var(--bg-hover);
        }

        .game-header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .game-header-teams {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            font-size: 15px;
            color: var(--text-primary);
        }

        .game-header-teams .team-logo {
            width: 32px !important;
            height: 32px !important;
        }

        .game-header-at {
            color: var(--text-muted);
            font-size: 11px;
            margin: 0 2px;
        }

        .game-header-right {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .game-time {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .game-picks-count {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 8px;
        }

        .game-toggle {
            font-size: 10px;
            color: var(--text-muted);
            transition: transform 0.2s ease;
        }

        .game-card.expanded .game-toggle {
            transform: rotate(180deg);
        }

        .primetime-badge {
            background: linear-gradient(135deg, #9333ea, #7c3aed);
            color: white;
            font-size: 8px;
            font-weight: 700;
            padding: 2px 5px;
            border-radius: 3px;
            text-transform: uppercase;
        }

        .weather-badge {
            font-size: 9px;
            font-weight: 500;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(100, 149, 237, 0.2);
            color: var(--text-muted);
            margin-left: 6px;
        }
        .weather-badge.dome {
            background: rgba(46, 160, 67, 0.2);
            color: #2ea043;
        }

        /* Game content - hidden by default, shown when expanded */
        .game-card-body {
            display: none;
            padding: 8px 12px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            /* No max-height constraint - show all picks without internal scrolling */
        }

        .game-card.expanded .game-card-body {
            display: block;
        }

        /* Legacy matchup-strip styles - now compact inline */
        .matchup-strip {
            display: none;
        }

        .team-side, .team-info, .team-name, .team-abbr-label, .matchup-center, .vs-label, .game-context, .context-item {
            display: none;
        }

        /* Legacy - removed display:none that was hiding featured cards */

        .picks-count {
            font-size: 10px;
            color: var(--text-muted);
        }

        .featured-picks-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        /* Mini Prop Card Styles */
        .mini-prop-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            border-left: 3px solid var(--text-muted);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: var(--shadow-sm);
        }

        .mini-prop-card:hover {
            transform: scale(1.02);
        }

        .mini-prop-card.pick-over {
            border-left-color: var(--over);
        }

        .mini-prop-card.pick-under {
            border-left-color: var(--under);
        }

        .mini-prop-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 6px;
        }

        .mini-prop-player {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .mini-prop-player .team-logo {
            width: 24px !important;
            height: 24px !important;
        }

        .mini-prop-name {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .mini-prop-pos {
            font-size: 10px;
            color: var(--text-muted);
            background: var(--bg-hover);
            padding: 1px 4px;
            border-radius: 2px;
        }

        .mini-prop-pick {
            font-size: 11px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .mini-prop-pick.pick-over {
            background: var(--over-soft);
            color: var(--over);
        }

        .mini-prop-pick.pick-under {
            background: var(--under-soft);
            color: var(--under);
        }

        .mini-prop-details {
            display: flex;
            gap: 8px;
            margin-bottom: 6px;
        }

        .mini-prop-market {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .mini-prop-line {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-primary);
            font-family: 'SF Mono', monospace;
        }

        .mini-prop-stats {
            display: flex;
            gap: 10px;
            font-size: 10px;
        }

        .mini-prop-conf {
            font-weight: 600;
        }

        .mini-prop-conf.conf-elite {
            color: var(--accent-green);
        }

        .mini-prop-conf.conf-high {
            color: var(--accent-cyan);
        }

        .mini-prop-conf.conf-standard {
            color: var(--text-secondary);
        }

        .mini-prop-edge {
            color: var(--accent-primary);
        }

        .mini-prop-proj {
            color: var(--text-muted);
        }

        .game-card-footer {
            padding: 12px 20px;
            border-top: 1px solid var(--border-color);
            text-align: center;
        }

        .view-all-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .view-all-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--accent-primary);
        }

        .game-card-details {
            border-top: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }

        /* ============================================
           Game Picks Grid - Stacked Card Layout
           FIX (Dec 14, 2025): Player names no longer truncated
           ============================================ */
        .game-picks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 12px;
            padding: 12px;
        }

        @media (min-width: 1400px) {
            .game-picks-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        @media (max-width: 768px) {
            .game-picks-grid {
                grid-template-columns: 1fr;
                gap: 10px;
                padding: 10px;
            }
        }

        /* ============================================
           GAME PICK CARD - Decision-First Design
           Hero: Pick direction + line (the actionable decision)
           ============================================ */
        .game-pick-card {
            background: var(--glass-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-xl);
            border: 1px solid var(--glass-border);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
            box-shadow: var(--shadow-md);
            position: relative;
        }

        .game-pick-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: var(--shadow-xl), 0 0 50px rgba(59, 130, 246, 0.2);
        }

        /* === ELITE TIER - Maximum Visual Impact === */
        .game-pick-card.tier-elite {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, var(--glass-surface) 60%);
            border: 2px solid rgba(59, 130, 246, 0.5);
            box-shadow: var(--shadow-lg), 0 0 60px rgba(59, 130, 246, 0.25);
        }

        .game-pick-card.tier-elite::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3B82F6, #60A5FA, #3B82F6);
        }

        .game-pick-card.tier-elite:hover {
            box-shadow: var(--shadow-xl), 0 0 80px rgba(59, 130, 246, 0.35);
            border-color: rgba(59, 130, 246, 0.7);
        }

        /* === STRONG TIER - Prominent but Secondary === */
        .game-pick-card.tier-strong {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, var(--glass-surface) 60%);
            border: 1px solid rgba(6, 182, 212, 0.4);
            box-shadow: var(--shadow-md), 0 0 40px rgba(6, 182, 212, 0.15);
        }

        .game-pick-card.tier-strong::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #06B6D4, transparent);
        }

        .game-pick-card.tier-strong:hover {
            box-shadow: var(--shadow-lg), 0 0 60px rgba(6, 182, 212, 0.25);
        }

        /* === MODERATE/CAUTION - Muted === */
        .game-pick-card.tier-moderate {
            opacity: 0.85;
            border-color: var(--border-muted);
        }

        .game-pick-card.tier-caution {
            opacity: 0.7;
            border-color: rgba(239, 68, 68, 0.3);
        }

        /* === TOP BAR: Checkbox + Tier Badge === */
        .card-top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px 8px 16px;
        }

        .card-tier-badge {
            font-size: 10px;
            font-weight: 800;
            padding: 4px 12px;
            border-radius: var(--radius-full);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .card-tier-badge.tier-badge-elite {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.25) 100%);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.5);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
            animation: elite-pulse 2s ease-in-out infinite;
        }

        @keyframes elite-pulse {
            0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.4); }
            50% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.6); }
        }

        .card-tier-badge.tier-badge-strong {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(8, 145, 178, 0.2) 100%);
            color: #22D3EE;
            border: 1px solid rgba(6, 182, 212, 0.45);
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.3);
        }

        .card-tier-badge.tier-badge-moderate {
            background: rgba(100, 116, 139, 0.2);
            color: var(--text-muted);
            border: 1px solid rgba(100, 116, 139, 0.3);
        }

        .card-tier-badge.tier-badge-caution {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* === HERO SECTION: The Pick Decision === */
        .card-hero {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 16px 20px;
            margin: 0 16px;
            border-radius: var(--radius-lg);
            transition: all 0.2s ease;
        }

        .card-hero.pick-over {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.08) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .card-hero.pick-under {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.08) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .card-hero-arrow {
            font-size: 28px;
            font-weight: 700;
            line-height: 1;
        }

        .card-hero.pick-over .card-hero-arrow {
            color: #34D399;
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.6);
        }

        .card-hero.pick-under .card-hero-arrow {
            color: #60A5FA;
            text-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
        }

        .card-hero-content {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }

        .card-hero-pick {
            font-size: 24px;
            font-weight: 800;
            font-family: var(--font-mono);
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .card-hero.pick-over .card-hero-pick {
            color: #10B981;
        }

        .card-hero.pick-under .card-hero-pick {
            color: #3B82F6;
        }

        .card-hero-market {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 2px;
        }

        /* === PLAYER ROW === */
        .card-player-row {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 14px 16px;
            border-top: 1px solid var(--border-muted);
            margin-top: 12px;
        }

        .card-player-avatar {
            flex-shrink: 0;
        }

        .card-player-avatar img {
            width: 56px !important;
            height: 56px !important;
            border-radius: 50%;
            border: 2px solid var(--glass-border);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .card-player-details {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 0;
            flex: 1;
        }

        .card-player-name {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-player-meta {
            font-size: 13px;
            color: var(--text-muted);
        }

        /* === STATS ROW - Readable Layout === */
        .card-stats-row {
            display: flex;
            align-items: center;
            justify-content: space-around;
            padding: 14px 16px 16px 16px;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid var(--border-muted);
        }

        .card-stat-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }

        .card-stat-value {
            font-size: 16px;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }

        .card-stat-value.conf-elite {
            color: #60A5FA;
            text-shadow: 0 0 12px rgba(59, 130, 246, 0.5);
        }

        .card-stat-value.conf-high {
            color: #22D3EE;
        }

        .card-stat-value.edge-pos {
            color: #34D399;
        }

        .card-stat-value.edge-neg {
            color: #F87171;
        }

        .card-stat-value.bet-high {
            color: #34D399;
        }

        .card-stat-value.bet-medium {
            color: #22D3EE;
        }

        .card-stat-value.bet-low {
            color: var(--text-muted);
        }

        .card-stat-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .card-stat-divider {
            width: 1px;
            height: 32px;
            background: var(--border-muted);
        }

        /* Legacy pick row styles (kept for other views) */
        .game-picks-list {
            max-height: 600px;
            overflow-y: auto;
            overflow-x: hidden;
        }

        .pick-row-wrapper {
            border-bottom: 1px solid var(--border-color);
        }

        .pick-row-wrapper:last-child {
            border-bottom: none;
        }

        .pick-row {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            padding: 10px 12px;
            cursor: pointer;
            transition: background 0.1s ease;
            gap: 8px;
        }

        .pick-row:hover {
            background: var(--bg-hover);
        }

        .pick-row.pick-over {
            border-left: 3px solid var(--over);
        }

        .pick-row.pick-under {
            border-left: 3px solid var(--under);
        }

        .pick-row-left {
            display: flex;
            align-items: center;
            gap: 6px;
            min-width: 0;
            overflow: hidden;
        }

        .pick-row-player {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .pick-row-pos {
            font-size: 9px;
            color: var(--text-muted);
            background: var(--bg-tertiary);
            padding: 1px 3px;
            border-radius: 2px;
            flex-shrink: 0;
        }

        .pick-row-market {
            font-size: 10px;
            color: var(--text-muted);
            white-space: nowrap;
        }

        /* ROI tier indicators */
        .roi-dot {
            font-size: 8px;
            margin-left: 4px;
            vertical-align: middle;
        }

        .roi-dot.roi-profitable {
            color: var(--accent-green);
        }

        .roi-dot.roi-breakeven {
            color: var(--accent-amber);
        }

        .roi-dot.roi-negative {
            color: var(--accent-red);
            opacity: 0.6;
        }

        .pick-row-divider {
            color: var(--border-color);
            font-size: 10px;
            opacity: 0.5;
        }

        .pick-row-right {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }

        .pick-row-direction {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .pick-row-direction.pick-over {
            background: var(--over-soft);
            color: var(--over);
        }

        .pick-row-direction.pick-under {
            background: var(--under-soft);
            color: var(--under);
        }

        .pick-row-line {
            font-size: 12px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
            color: var(--text-primary);
        }

        .pick-row-conf {
            font-size: 11px;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }

        .pick-row-conf.conf-elite {
            color: var(--accent-green);
        }

        .pick-row-conf.conf-high {
            color: var(--accent-cyan);
        }

        .pick-row-edge {
            font-size: 10px;
            font-family: 'SF Mono', monospace;
            color: var(--text-muted);
        }

        .pick-row-edge.edge-pos {
            color: var(--accent-green);
        }

        .row-tier {
            font-size: 8px;
            font-weight: 600;
            padding: 2px 4px;
            border-radius: 3px;
        }

        .row-tier.tier-elite {
            background: rgba(5, 150, 105, 0.2);
            color: var(--accent-green);
        }

        .row-tier.tier-strong {
            background: rgba(6, 182, 212, 0.2);
            color: var(--accent-cyan);
        }

        .row-tier.tier-caution {
            background: rgba(245, 158, 11, 0.2);
            color: var(--accent-amber);
        }

        .row-tier.tier-moderate {
            background: rgba(156, 163, 175, 0.2);
            color: var(--text-secondary);
        }

        .row-tier.tier-spec {
            background: rgba(156, 163, 175, 0.15);
            color: var(--text-muted);
        }

        .pick-row-chevron {
            font-size: 10px;
            color: var(--text-muted);
            transition: transform 0.2s ease;
        }

        .pick-row-chevron.expanded {
            transform: rotate(180deg);
        }

        .pick-row-expanded {
            display: none;
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-light);
        }

        .pick-row-expanded.show {
            display: block;
        }

        .breakdown-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .breakdown-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 8px;
            background: var(--bg-card);
            border-radius: 4px;
        }

        .breakdown-label {
            font-size: 10px;
            color: var(--text-muted);
        }

        .breakdown-value {
            font-size: 11px;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
            color: var(--text-primary);
        }

        .breakdown-value .positive {
            color: var(--accent-green);
        }

        .breakdown-value .negative {
            color: var(--accent-red);
        }

        .breakdown-value.positive {
            color: var(--accent-green);
        }

        .breakdown-value.negative {
            color: var(--accent-red);
        }

        /* === GENERAL POSITIVE/NEGATIVE COLOR CLASSES === */
        .game-pick-stat-value.positive,
        .card-stat-value.positive {
            color: var(--positive) !important;
        }

        .game-pick-stat-value.negative,
        .card-stat-value.negative {
            color: var(--negative) !important;
        }

        /* === MATCHUP CONTEXT SECTION === */
        .matchup-context {
            margin-bottom: 12px;
            border: 1px solid var(--border-light);
            border-radius: var(--radius-md);
            background: var(--bg-secondary);
            overflow: hidden;
        }

        .matchup-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            cursor: pointer;
            transition: background 0.15s ease;
        }

        .matchup-header:hover {
            background: var(--bg-hover);
        }

        .matchup-vs {
            font-size: 10px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .matchup-team {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .matchup-rank-badge {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: auto;
        }

        .matchup-rank-badge.matchup-favorable {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }

        .matchup-rank-badge.matchup-neutral {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }

        .matchup-rank-badge.matchup-tough {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }

        .matchup-chevron {
            font-size: 10px;
            color: var(--text-muted);
            transition: transform 0.2s ease;
        }

        .matchup-header.expanded .matchup-chevron {
            transform: rotate(180deg);
        }

        .matchup-details {
            display: none;
            padding: 10px 12px;
            border-top: 1px solid var(--border-light);
            background: var(--bg-tertiary);
        }

        .matchup-details.show {
            display: block;
        }

        .matchup-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }

        .matchup-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
        }

        .matchup-label {
            font-size: 10px;
            color: var(--text-muted);
        }

        .matchup-value {
            font-size: 11px;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
            color: var(--text-primary);
        }

        .matchup-value.trend-improving {
            color: var(--accent-green);
        }

        .matchup-value.trend-declining {
            color: var(--accent-red);
        }

        .matchup-value.trend-stable {
            color: var(--text-secondary);
        }

        /* === DEFENSE TREND CHART === */
        .def-chart-section {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-light);
        }

        .def-chart-title {
            display: block;
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-align: center;
        }

        .def-chart-container {
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            min-height: 70px;
            gap: 6px;
            padding: 4px 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
        }

        .def-chart-bar-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 40px;
        }

        .def-chart-bar {
            width: 24px;
            background: linear-gradient(180deg, #3B82F6 0%, rgba(59, 130, 246, 0.6) 100%);
            border-radius: 3px 3px 0 0;
            min-height: 4px;
        }

        .def-chart-bar-container:hover .def-chart-bar {
            background: #3B82F6;
            box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
        }

        .def-chart-label {
            font-size: 10px;
            color: var(--text-secondary);
            font-family: 'SF Mono', monospace;
            margin-bottom: 2px;
            font-weight: 500;
        }

        .def-chart-week {
            font-size: 9px;
            color: var(--text-muted);
            font-family: 'SF Mono', monospace;
            font-weight: 600;
            margin-top: 4px;
        }

        /* === GAME LINE TEAM CONTEXT === */
        .gl-team-context {
            border-top: 1px solid var(--border-light);
        }

        .gl-context-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px;
            cursor: pointer;
            font-size: 12px;
            color: var(--text-secondary);
            transition: background 0.15s ease;
        }

        .gl-context-toggle:hover {
            background: var(--bg-hover);
        }

        .gl-context-header {
            transition: all 0.15s ease;
        }

        .gl-context-header .ctx-chevron {
            transition: transform 0.2s ease;
        }

        .gl-context-header.expanded .ctx-chevron {
            transform: rotate(180deg);
        }

        .gl-context-content {
            display: none;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-light);
        }

        .gl-context-content.show {
            display: block;
        }

        .gl-ats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .gl-team-ats {
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: var(--radius-md);
        }

        .gl-team-header {
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 10px;
            color: var(--text-primary);
            text-align: center;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-light);
        }

        .gl-ats-row, .gl-def-row, .gl-ou-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 11px;
        }

        .gl-ats-label, .gl-def-label, .gl-ou-label {
            color: var(--text-muted);
        }

        .gl-ats-value, .gl-ou-value {
            font-family: 'SF Mono', monospace;
            color: var(--text-primary);
            font-weight: 500;
        }

        .gl-def-rank {
            font-family: 'SF Mono', monospace;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 10px;
        }

        .gl-def-rank.def-elite {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }

        .gl-def-rank.def-average {
            background: rgba(250, 204, 21, 0.2);
            color: #facc15;
        }

        .gl-def-rank.def-poor {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }

        /* Responsive adjustments for game cards */
        @media (max-width: 900px) {
            .game-cards-grid {
                grid-template-columns: 1fr;
            }

            .featured-picks-grid {
                grid-template-columns: 1fr;
            }

            .matchup-strip {
                flex-direction: column;
                gap: 16px;
            }

            .team-side.away,
            .team-side.home {
                justify-content: center;
                flex-direction: row;
            }

            .matchup-center {
                padding: 8px 0;
            }
        }

        .footer {
            padding: var(--space-md) var(--space-lg);
            text-align: center;
            background: var(--bg-card);
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 11px;
        }

        /* Pick Breakdown Modal Styles - Dark Mode */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            z-index: 1000;
            backdrop-filter: blur(8px);
            animation: fadeIn 0.2s ease;
        }

        .modal-overlay.active {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .modal-content {
            position: relative;
            background: linear-gradient(
                180deg,
                rgba(15, 23, 42, 0.98) 0%,
                rgba(3, 7, 18, 0.99) 100%
            );
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-radius: var(--radius-2xl);
            max-width: 720px;
            width: 95%;
            max-height: 95vh;
            overflow-y: auto;
            box-shadow:
                0 0 0 1px rgba(59, 130, 246, 0.1) inset,
                0 32px 100px rgba(0, 0, 0, 0.7),
                0 0 80px rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.15);
            animation: slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Top edge highlight with blue accent */
        .modal-content::before {
            content: '';
            position: absolute;
            top: 0;
            left: 10%;
            right: 10%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
            pointer-events: none;
        }

        /* Modal body content */
        .modal-body {
            padding: 20px 24px;
        }

        .modal-section {
            margin-bottom: 16px;
        }

        .modal-factors-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 24px;
            border-bottom: 1px solid var(--border-subtle);
            background: var(--bg-card);
            border-radius: var(--radius-xl) var(--radius-xl) 0 0;
        }

        .modal-player-info {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .modal-player-details {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .modal-player-name {
            font-size: var(--text-xl);
            font-weight: var(--weight-bold);
            color: var(--text-primary);
            letter-spacing: var(--tracking-tight);
        }

        .modal-player-meta {
            font-size: var(--text-sm);
            color: var(--text-secondary);
        }

        .modal-close {
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            font-size: 20px;
            color: var(--text-muted);
            cursor: pointer;
            padding: 8px 12px;
            border-radius: var(--radius-md);
            transition: all 0.15s ease;
        }

        .modal-close:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--border-color);
        }

        /* === MODAL PICK SUMMARY (Fixed Overlap) === */
        .modal-pick-summary {
            padding: 24px;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-subtle);
        }

        .modal-pick-main {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            gap: 20px;
            flex-wrap: wrap;
        }

        .modal-pick-direction {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .modal-pick-badge {
            font-size: 13px;
            font-weight: var(--weight-bold);
            padding: 8px 18px;
            border-radius: var(--radius-full);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wide);
            white-space: nowrap;
            flex-shrink: 0;
        }

        .modal-pick-badge.pick-over {
            background: linear-gradient(135deg,
                rgba(59, 130, 246, 0.2) 0%,
                rgba(37, 99, 235, 0.15) 100%
            );
            color: var(--over);
            border: 1px solid var(--over-border);
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.15);
        }

        .modal-pick-badge.pick-under {
            background: linear-gradient(135deg,
                rgba(248, 81, 73, 0.2) 0%,
                rgba(218, 54, 51, 0.15) 100%
            );
            color: var(--under);
            border: 1px solid var(--under-border);
        }

        .modal-pick-line {
            font-size: 24px;
            font-weight: var(--weight-bold);
            color: var(--text-primary);
            font-family: var(--font-mono);
            white-space: nowrap;
        }

        .modal-pick-market {
            font-size: var(--text-sm);
            color: var(--text-secondary);
            font-weight: var(--weight-medium);
        }

        .modal-confidence {
            text-align: right;
            flex-shrink: 0;
            min-width: 100px;
        }

        .modal-conf-value {
            font-size: 28px;
            font-weight: var(--weight-bold);
            font-family: var(--font-mono);
            line-height: 1;
        }

        .modal-conf-value.conf-elite {
            color: var(--positive);
            text-shadow: 0 0 20px var(--positive-glow);
        }

        .modal-conf-value.conf-high {
            color: var(--brand-primary);
        }

        .modal-conf-value.conf-standard {
            color: var(--text-secondary);
        }

        .modal-conf-label {
            font-size: var(--text-xs);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-widest);
            margin-top: 4px;
        }

        /* === Hero Stats Row (Fixed Overflow) === */
        .modal-stats-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 8px;
        }

        .modal-stat-item {
            text-align: center;
            padding: 16px 12px;
            background: var(--surface-raised);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-muted);
            min-width: 0; /* Prevent overflow */
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .modal-stat-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .modal-stat-value {
            font-size: 20px;
            font-weight: var(--weight-bold);
            color: var(--text-primary);
            font-family: var(--font-mono);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-shadow: 0 0 30px rgba(248, 250, 252, 0.1);
        }

        .modal-stat-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-widest);
            margin-top: 6px;
        }

        .modal-body {
            padding: 24px;
        }

        .modal-section {
            margin-bottom: 24px;
        }

        .modal-section:last-child {
            margin-bottom: 0;
        }

        .modal-section-title {
            font-size: var(--text-sm);
            font-weight: var(--weight-semibold);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .modal-summary-section {
            background: var(--bg-card);
            padding: 20px;
            border-radius: var(--radius-lg);
            margin-bottom: 20px;
            border: 1px solid var(--border-subtle);
        }

        .modal-summary {
            font-size: var(--text-base);
            line-height: var(--leading-relaxed);
            color: var(--text-secondary);
        }

        /* Projection Callout Box - Dark Mode */
        .narrative-callout {
            background: var(--bg-card);
            border-left: 4px solid var(--over);
            border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            padding: 16px 20px;
            margin-bottom: 16px;
        }

        .narrative-callout.positive {
            border-left-color: var(--over);
            background: var(--over-bg);
        }

        .narrative-callout.caution {
            border-left-color: var(--accent-amber);
            background: var(--tier-moderate-bg);
        }

        .callout-label {
            font-size: var(--text-xs);
            font-weight: var(--weight-semibold);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .callout-value {
            font-size: var(--text-2xl);
            font-weight: var(--weight-bold);
            font-family: var(--font-mono);
            color: var(--text-primary);
        }

        .callout-vs {
            font-size: var(--text-sm);
            font-weight: var(--weight-medium);
            color: var(--text-muted);
        }

        .callout-diff {
            font-size: 13px;
            font-weight: 600;
            color: #3B82F6;
            margin-top: 2px;
        }

        .narrative-callout.caution .callout-diff {
            color: #f59e0b;
        }

        /* ============================================
           BETTINGPROS STYLE COMPONENT LIBRARY
           Clean, analytical components
           ============================================ */

        /* === ANALYSIS BLOCK - Single Accent Border === */
        .analysis-block {
            background: var(--bg-section);
            border-radius: var(--radius-lg);
            padding: 16px 20px;
            border-left: 4px solid var(--neutral);
            margin-bottom: 16px;
        }

        .analysis-block.pick-over {
            border-left-color: var(--over);
        }

        .analysis-block.pick-under {
            border-left-color: var(--under);
        }

        .analysis-text {
            font-size: var(--text-base);
            line-height: var(--leading-relaxed);
            color: var(--text-secondary);
        }

        .analysis-text p {
            margin: 0 0 12px 0;
        }

        .analysis-text p:last-child {
            margin-bottom: 0;
        }

        /* Inline stat highlighting */
        .analysis-text .stat {
            font-family: var(--font-mono);
            font-weight: 600;
            color: var(--text-primary);
            background: rgba(0, 0, 0, 0.04);
            padding: 2px 6px;
            border-radius: var(--radius-sm);
        }

        .analysis-text .stat.positive {
            color: var(--positive);
        }

        .analysis-text .stat.negative {
            color: var(--negative);
        }

        /* Strong emphasis */
        .analysis-text strong {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* === PROBABILITY METER - Dark Mode === */
        .prob-meter-container {
            margin: 20px 0;
        }

        .prob-meter-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .prob-meter-label {
            font-size: var(--text-sm);
            font-weight: var(--weight-semibold);
            color: var(--text-secondary);
        }

        .prob-meter-value {
            font-size: var(--text-xl);
            font-weight: var(--weight-bold);
            font-family: var(--font-mono);
        }

        .prob-meter-value.over { color: var(--over); }
        .prob-meter-value.under { color: var(--under); }

        .prob-meter {
            height: 8px;
            background: var(--bg-elevated);
            border-radius: var(--radius-full);
            overflow: hidden;
            position: relative;
        }

        .prob-fill {
            height: 100%;
            border-radius: var(--radius-full);
            transition: width 0.4s ease;
        }

        .prob-fill.over {
            background: linear-gradient(90deg, #1D4ED8, var(--over), var(--over-light));
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
        }

        .prob-fill.under {
            background: linear-gradient(90deg, #DC2626, var(--under), var(--under-light));
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
        }

        /* === STAR RATING COMPONENT - Gold Stars === */
        .star-rating-container {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 16px 0;
        }

        .star-rating {
            display: flex;
            gap: 2px;
        }

        .star {
            font-size: 18px;
            line-height: 1;
        }

        .star.filled {
            color: var(--star-filled);
            filter: drop-shadow(0 0 3px rgba(251, 191, 36, 0.5));
        }  /* Gold */
        .star.empty { color: var(--star-empty); }    /* Gray */

        .rating-badge {
            background: var(--over-bg);
            color: var(--over);
            padding: 6px 12px;
            border-radius: var(--radius-md);
            font-size: var(--text-xs);
            font-weight: var(--weight-bold);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
        }

        .rating-badge.caution {
            background: var(--tier-moderate-bg);
            color: var(--tier-moderate-text);
        }

        /* === EDGE SOURCE BADGE === */
        .edge-source-container {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 12px 0 16px 0;
            padding: 10px 14px;
            background: var(--bg-elevated);
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
        }

        .edge-source-label {
            font-size: var(--text-xs);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
        }

        .edge-source-badge {
            padding: 4px 10px;
            border-radius: var(--radius-sm);
            font-size: var(--text-xs);
            font-weight: var(--weight-bold);
            text-transform: uppercase;
        }

        .edge-source-badge.source-bias {
            background: rgba(139, 92, 246, 0.2);
            color: #a78bfa;
        }

        .edge-source-badge.source-lvt {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
        }

        .edge-source-badge.source-poisson {
            background: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
        }

        .edge-source-badge.source-gameline {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
        }

        .edge-source-badge.source-default {
            background: var(--bg-card);
            color: var(--text-secondary);
        }

        .edge-sample {
            font-size: var(--text-xs);
            color: var(--text-muted);
            margin-left: auto;
        }

        /* === GAME CONTEXT ROW (Weather + Defense) === */
        .game-context-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }

        .context-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: var(--radius-sm);
            font-size: var(--text-xs);
            font-weight: var(--weight-medium);
        }

        .context-chip.weather-dome {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
        }

        .context-chip.weather-outdoor {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
        }

        .context-chip.weather-high {
            background: rgba(251, 146, 60, 0.2);
            color: #fb923c;
        }

        .context-chip.weather-extreme {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }

        .context-chip.def-rank {
            background: var(--bg-elevated);
            color: var(--text-secondary);
        }

        .context-chip.def-rank.positive {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
        }

        .context-chip.def-rank.negative {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
        }

        /* === BIAS COMPARISON === */
        .bias-comparison {
            display: flex;
            gap: 16px;
            margin: 12px 0;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: var(--radius-sm);
            border: 1px solid var(--surface-3);
        }

        .bias-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
        }

        .bias-label {
            font-size: var(--text-xs);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .bias-value {
            font-size: var(--text-lg);
            font-weight: var(--weight-semibold);
            color: var(--text-primary);
        }

        .bias-value.positive {
            color: var(--color-under);
        }

        .bias-value.negative {
            color: var(--color-over);
        }

        .bias-trend {
            font-size: var(--text-xs);
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .bias-trend.improving {
            color: #4ade80;
        }

        .bias-trend.declining {
            color: #f87171;
        }

        /* === KEY FACTORS LIST - Dark Mode BettingPros Style === */
        .factors-list {
            margin-top: 20px;
            padding: 20px;
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
        }

        .factors-list-title {
            font-size: var(--text-sm);
            font-weight: var(--weight-semibold);
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .factor-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px;
            font-size: var(--text-base);
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            margin-bottom: 6px;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }

        .factor-item:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.06);
        }

        .factor-item:last-child {
            margin-bottom: 0;
        }

        .factor-icon {
            width: 24px;
            text-align: center;
            font-size: 16px;
            flex-shrink: 0;
        }

        .factor-icon.supporting { color: var(--positive); }
        .factor-icon.against { color: var(--negative); }
        .factor-icon.neutral { color: var(--neutral); }

        .factor-label {
            flex: 1;
            color: var(--text-secondary);
            font-size: var(--text-base);
        }

        .factor-value {
            font-family: var(--font-mono);
            font-weight: var(--weight-semibold);
            font-size: var(--text-base);
            text-align: right;
            min-width: 70px;
            flex-shrink: 0;
        }

        /* Color the VALUES only, not borders */
        .factor-value.positive { color: var(--positive); }
        .factor-value.negative { color: var(--negative); }
        .factor-value.neutral { color: var(--text-primary); }

        /* === MODEL DETAILS EXPANDABLE SECTION === */
        .model-details-section {
            margin-top: 20px;
            border-top: 1px solid var(--border-subtle);
            padding-top: 16px;
        }

        .model-details-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            padding: 12px 16px;
            background: var(--surface-raised);
            border: 1px solid var(--border-muted);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: var(--weight-semibold);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .model-details-toggle:hover {
            background: var(--surface-elevated);
            color: var(--text-primary);
            border-color: var(--border-default);
        }

        .model-details-toggle .chevron {
            transition: transform 0.2s ease;
            font-size: 12px;
        }

        .model-details-toggle.expanded .chevron {
            transform: rotate(180deg);
        }

        .model-details-content {
            margin-top: 12px;
            padding: 16px;
            background: var(--surface-raised);
            border-radius: var(--radius-md);
            border: 1px solid var(--border-muted);
            display: none;
        }

        .model-details-content.show {
            display: block;
        }

        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
        }

        .detail-group h5 {
            font-size: 10px;
            font-weight: var(--weight-bold);
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: var(--tracking-widest);
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-muted);
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 12px;
        }

        .detail-row span:first-child {
            color: var(--text-tertiary);
        }

        .detail-row .mono {
            font-family: var(--font-mono);
            font-weight: var(--weight-medium);
            color: var(--text-primary);
        }

        .detail-row .mono.positive {
            color: var(--positive);
        }

        .detail-row .mono.negative {
            color: var(--negative);
        }

        /* === PICK BADGE (Card Header) === */
        .pick-badge {
            padding: 6px 14px;
            border-radius: var(--radius-md);
            font-weight: 700;
            font-size: var(--text-sm);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .pick-badge.over {
            background: var(--over-soft);
            color: var(--over);
        }

        .pick-badge.under {
            background: var(--under-soft);
            color: var(--under);
        }

        /* === UNIFIED NARRATIVE (Legacy + Updated) === */
        .unified-narrative {
            font-size: var(--text-base);
            line-height: var(--leading-relaxed);
            color: var(--text-secondary);
        }

        .unified-narrative.pick-over {
            border-left: 4px solid var(--over);
            padding-left: 16px;
            background: var(--bg-section);
            border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            padding: 16px 20px;
            padding-left: 20px;
            margin-left: 0;
        }

        .unified-narrative.pick-under {
            border-left: 4px solid var(--under);
            padding-left: 16px;
            background: var(--bg-section);
            border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
            padding: 16px 20px;
            padding-left: 20px;
            margin-left: 0;
        }

        .unified-narrative p {
            margin: 0 0 12px 0;
        }

        .unified-narrative p:last-child {
            margin-bottom: 0;
        }

        .unified-narrative .narrative-lead {
            font-size: 15px;
            color: var(--text-primary);
            font-weight: 500;
            margin-bottom: 14px;
        }

        /* Standardized stat highlighting - consistent classes */
        .unified-narrative .stat {
            font-family: var(--font-mono);
            font-weight: 600;
            color: var(--text-primary);
            /* No background - cleaner inline appearance */
        }

        /* Positive stats (favorable to pick) */
        .unified-narrative .stat.positive,
        .unified-narrative .stat-positive {
            color: var(--positive);
        }

        /* Negative stats (against pick) */
        .unified-narrative .stat.negative,
        .unified-narrative .stat-negative {
            color: var(--negative);
        }

        /* Neutral stats (informational) */
        .unified-narrative .stat.neutral,
        .unified-narrative .stat-neutral {
            color: var(--text-primary);
        }

        /* Percentage values - use semantic coloring */
        .unified-narrative .stat-pct {
            font-family: var(--font-mono);
            font-weight: 700;
            color: var(--positive);
        }

        .unified-narrative .stat-pct.caution {
            color: var(--accent-amber);
        }

        .unified-narrative .stat-pct.negative {
            color: var(--negative);
        }

        /* Edge values */
        .unified-narrative .stat-edge {
            font-family: var(--font-mono);
            font-weight: 700;
        }

        .unified-narrative .stat-edge.positive {
            color: var(--positive);
        }

        .unified-narrative .stat-edge.negative {
            color: var(--negative);
        }

        /* Key terms emphasis */
        .unified-narrative .term {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* Pick direction callout - REMOVED redundancy at end */
        .unified-narrative .pick-direction {
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .unified-narrative .pick-direction.over {
            color: var(--over);
        }

        .unified-narrative .pick-direction.under {
            color: var(--under);
        }

        /* Conviction tiers */
        .unified-narrative .conviction-high {
            color: var(--positive);
            font-weight: 600;
        }

        .unified-narrative .conviction-medium {
            color: var(--accent-amber);
            font-weight: 600;
        }

        .unified-narrative .conviction-low {
            color: var(--text-muted);
            font-weight: 600;
        }

        /* Conflict note */
        .unified-narrative .conflict-note {
            display: block;
            margin-top: 12px;
            padding: 12px 14px;
            background: rgba(237, 137, 54, 0.08);
            border-left: 3px solid var(--accent-amber);
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            font-size: var(--text-sm);
            color: var(--text-secondary);
        }

        /* MUTED formula calculations - subtle, not distracting */
        .unified-narrative .math-inline,
        .unified-narrative .calculation {
            font-family: var(--font-mono);
            font-size: var(--text-sm);
            color: var(--text-light);
            font-weight: 400;
        }

        .unified-narrative .narrative-section {
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid var(--border-light);
        }

        /* Legacy support */
        .narrative-text {
            font-size: var(--text-base);
            line-height: var(--leading-relaxed);
            color: var(--text-secondary);
        }

        .narrative-text strong.hl-pct {
            font-family: var(--font-mono);
            color: var(--positive);
            font-weight: 700;
        }

        .narrative-text strong.hl-stat {
            color: var(--text-primary);
            font-weight: 700;
            font-family: var(--font-mono);
        }

        .narrative-text em {
            color: var(--text-primary);
            font-style: normal;
            font-weight: 500;
        }

        .modal-summary p {
            margin: 0 0 8px 0;
        }

        .modal-summary p:last-child {
            margin-bottom: 0;
        }

        .modal-summary strong {
            color: var(--accent-primary);
        }

        /* CONFLICT WARNING */
        .conflict-warning {
            background: linear-gradient(135deg, rgba(229, 62, 62, 0.06), rgba(237, 137, 54, 0.06));
            border: 1px solid var(--negative);
            border-radius: var(--radius-lg);
            padding: 14px 16px;
            margin-bottom: 16px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .conflict-warning-icon {
            font-size: 18px;
            line-height: 1;
            color: var(--negative);
        }

        .conflict-warning-text {
            font-size: var(--text-sm);
            color: var(--text-primary);
            line-height: var(--leading-normal);
        }

        .conflict-warning-title {
            font-weight: 700;
            color: var(--negative);
            margin-bottom: 4px;
        }

        .conflict-warning-detail {
            color: var(--text-secondary);
        }

        /* ====== GAME HISTORY CHART - FIXED LAYOUT ====== */
        .game-history-chart {
            margin-top: 20px;
            padding: 16px;
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
        }

        .game-history-chart-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 16px;
        }

        .game-history-bars {
            position: relative;
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            height: 140px;
            padding: 0 70px; /* INCREASED for combined labels */
            margin-bottom: 8px;
        }

        .game-history-bar-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
            max-width: 60px;
        }

        .game-history-bar-container {
            width: 36px;
            height: 100px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
        }

        .game-history-bar {
            width: 100%;
            border-radius: 6px 6px 0 0;
            position: relative;
            min-height: 8px;
            transition: all 0.2s ease;
        }

        .game-history-bar.hit-over {
            background: linear-gradient(180deg, #1D4ED8 0%, var(--over) 50%, var(--over-light) 100%);
            box-shadow: 0 -2px 8px rgba(59, 130, 246, 0.3);
        }

        .game-history-bar.hit-under {
            background: linear-gradient(180deg, #DC2626 0%, var(--under) 50%, var(--under-light) 100%);
            box-shadow: 0 -2px 8px rgba(248, 113, 113, 0.3);
        }

        .game-history-bar.miss {
            background: linear-gradient(180deg, var(--neutral) 0%, #4A5568 100%);
            opacity: 0.6;
        }

        .game-history-bar:hover {
            filter: brightness(1.15);
            transform: scaleY(1.02);
        }

        .game-history-bar-value {
            position: absolute;
            top: -22px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
            white-space: nowrap;
        }

        .game-history-bar-label {
            margin-top: 8px;
            font-size: 10px;
            color: var(--text-muted);
            text-align: center;
            white-space: nowrap;
        }

        /* === LINE MARKERS - DUAL LABEL POSITIONING === */
        .game-history-line {
            position: absolute;
            left: 60px;
            right: 60px;
            height: 2px;
            background: #F59E0B;
            z-index: 10;
        }

        .game-history-line-label-left {
            position: absolute;
            left: -58px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 11px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: #F59E0B;
            white-space: nowrap;
            background: var(--bg-card);
            padding: 2px 4px;
            border-radius: 3px;
        }

        .game-history-line-label-right {
            position: absolute;
            right: -50px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            font-weight: 600;
            color: #F59E0B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
            background: var(--bg-card);
            padding: 2px 4px;
            border-radius: 3px;
        }

        .game-history-projection {
            position: absolute;
            left: 60px;
            right: 60px;
            height: 2px;
            background: repeating-linear-gradient(
                90deg,
                #3B82F6,
                #3B82F6 4px,
                transparent 4px,
                transparent 8px
            );
            z-index: 9;
        }

        .game-history-projection.no-labels {
            /* When using combined labels, projection line only */
        }

        .game-history-projection-label-left {
            position: absolute;
            left: -58px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 11px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: #3B82F6;
            white-space: nowrap;
            background: var(--bg-card);
            padding: 2px 4px;
            border-radius: 3px;
        }

        .game-history-projection-label-right {
            position: absolute;
            right: -50px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            font-weight: 600;
            color: #3B82F6;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
            background: var(--bg-card);
            padding: 2px 4px;
            border-radius: 3px;
        }

        /* Combined label stack when values are close */
        .game-history-combined-labels {
            position: absolute;
            left: -70px;
            transform: translateY(50%);
            z-index: 15;
        }

        .game-history-combined-labels .label-stack {
            display: flex;
            flex-direction: column;
            gap: 4px;
            background: var(--bg-card);
            padding: 6px 8px;
            border-radius: 6px;
            border: 1px solid var(--border-subtle);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        .game-history-combined-labels .label-line {
            font-size: 11px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: #F59E0B;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .game-history-combined-labels .label-proj {
            font-size: 11px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: #3B82F6;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .game-history-combined-labels small {
            font-size: 8px;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* === LEGEND - BETTER SPACING === */
        .game-history-legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-subtle);
            font-size: 11px;
            color: var(--text-muted);
        }

        .game-history-legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .game-history-legend-swatch {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }

        .game-history-legend-swatch.hit-over {
            background: var(--over);
        }

        .game-history-legend-swatch.hit-under {
            background: var(--under);
        }

        .game-history-legend-swatch.line {
            background: var(--accent-amber);
            height: 3px;
            width: 16px;
        }

        .game-history-legend-swatch.projection {
            background: var(--positive);
            height: 2px;
            width: 16px;
            border-style: dashed;
        }

        .game-history-no-data {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 12px;
        }

        .modal-factors-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .modal-factor-group {
            padding: 12px;
            border-radius: 8px;
            background: var(--bg-tertiary);
        }

        .modal-factor-group.supporting {
            border-left: 3px solid var(--accent-green);
        }

        .modal-factor-group.opposing {
            border-left: 3px solid var(--accent-red);
        }

        .modal-factor-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .modal-factor-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid var(--border-light);
        }

        .modal-factor-item:last-child {
            border-bottom: none;
        }

        .modal-factor-name {
            font-size: 12px;
            color: var(--text-primary);
        }

        .modal-factor-value {
            font-size: 12px;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }

        .modal-factor-value.positive {
            color: var(--accent-green);
        }

        .modal-factor-value.negative {
            color: var(--accent-red);
        }

        .modal-matchup-info {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .modal-matchup-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }

        .modal-matchup-label {
            font-size: 11px;
            color: var(--text-muted);
        }

        .modal-matchup-value {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            font-family: 'SF Mono', monospace;
        }

        .mini-prop-card {
            cursor: pointer;
        }

        /* Featured Elite Picks Section */
        .featured-elite-section {
            margin-bottom: 24px;
        }

        .featured-elite-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .featured-elite-header h3 {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 0;
        }

        .featured-elite-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .featured-elite-subtitle {
            font-size: 10px;
            color: var(--text-muted);
            font-weight: 400;
            margin-left: 8px;
        }

        .featured-elite-badge {
            background: var(--tier-elite-bg);
            color: white;
            font-size: 9px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 10px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        /* ============================================
           COMPACT FEATURED PICKS TABLE - Replaces Cards
           Target: 8 picks in ~200px height
           ============================================ */
        .featured-picks-table-wrapper {
            border: 1px solid var(--border-color);
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: var(--space-lg);
        }

        .featured-picks-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }

        .featured-picks-table thead {
            background: var(--bg-tertiary);
            position: sticky;
            top: 0;
        }

        .featured-picks-table th {
            padding: 6px 8px;
            font-size: 9px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .featured-picks-table td {
            padding: 5px 8px;
            border-bottom: 1px solid var(--border-light);
            white-space: nowrap;
        }

        .featured-picks-table tbody tr {
            height: 28px;
            cursor: pointer;
            transition: background 0.1s ease;
        }

        .featured-picks-table tbody tr:hover {
            background: var(--bg-hover);
        }

        .featured-picks-table tbody tr:nth-child(even) {
            background: var(--bg-tertiary);
        }

        .featured-picks-table tbody tr:nth-child(even):hover {
            background: var(--bg-hover);
        }

        /* Compact player cell with logo */
        .fp-player {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .fp-player .team-logo {
            width: 16px;
            height: 16px;
        }

        .fp-player-name {
            font-weight: 600;
            color: var(--text-primary);
        }

        .fp-pos {
            font-size: 9px;
            color: var(--text-muted);
            margin-left: 4px;
        }

        /* Pick direction badges - compact */
        .fp-pick {
            font-weight: 700;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }

        .fp-pick.pick-over {
            background: var(--over-soft);
            color: var(--over);
        }

        .fp-pick.pick-under {
            background: var(--under-soft);
            color: var(--under);
        }

        .fp-pick .arrow {
            font-size: 11px;
        }

        /* Confidence and edge styling */
        .fp-conf {
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .fp-conf.conf-elite { color: var(--conf-elite); }
        .fp-conf.conf-high { color: var(--conf-high); }
        .fp-conf.conf-standard { color: var(--text-secondary); }

        .fp-edge {
            font-family: 'JetBrains Mono', monospace;
        }

        .fp-edge.edge-pos { color: var(--conf-elite); }
        .fp-edge.edge-neg { color: var(--under); }

        .fp-line {
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }

        /* Tier badges - premium glow style */
        .fp-tier {
            font-size: 9px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: var(--radius-full);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
            transition: all 0.2s ease;
        }

        .fp-tier.tier-elite {
            background: var(--tier-elite-bg);
            color: var(--tier-elite-text);
            border: 1px solid var(--tier-elite-border);
            box-shadow: var(--tier-elite-glow);
        }

        .fp-tier.tier-strong {
            background: var(--tier-strong-bg);
            color: var(--tier-strong-text);
            border: 1px solid var(--tier-strong-border);
            box-shadow: var(--tier-strong-glow);
        }

        .fp-tier.tier-caution {
            background: var(--tier-caution-bg);
            color: var(--tier-caution-text);
            border: 1px solid var(--tier-caution-border);
        }

        .fp-tier.tier-moderate {
            background: var(--tier-moderate-bg);
            color: var(--tier-moderate-text);
            border: 1px solid var(--tier-moderate-border);
        }

        /* ============================================
           FEATURED PICK CARDS - Compact Design
           Target: ~100px height, 4 columns on wide screens
           ============================================ */
        .featured-elite-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }

        @media (min-width: 1400px) {
            .featured-elite-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (min-width: 1800px) {
            .featured-elite-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        /* Featured pick cards use same structure as game-pick-card */
        .featured-pick-card {
            background: var(--glass-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-xl);
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow-md);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .featured-pick-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: var(--shadow-xl), 0 0 50px rgba(59, 130, 246, 0.2);
        }

        .featured-pick-card.tier-elite {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, var(--glass-surface) 60%);
            border: 2px solid rgba(59, 130, 246, 0.5);
            box-shadow: var(--shadow-lg), 0 0 60px rgba(59, 130, 246, 0.25);
        }

        .featured-pick-card.tier-elite::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3B82F6, #60A5FA, #3B82F6);
        }

        .featured-pick-card.tier-elite:hover {
            box-shadow: var(--shadow-xl), 0 0 80px rgba(59, 130, 246, 0.35);
            border-color: rgba(59, 130, 246, 0.7);
        }

        .featured-pick-card.tier-strong {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, var(--glass-surface) 60%);
            border: 1px solid rgba(6, 182, 212, 0.4);
            box-shadow: var(--shadow-md), 0 0 40px rgba(6, 182, 212, 0.15);
        }

        .featured-pick-card.tier-strong::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #06B6D4, transparent);
        }

        .featured-pick-card.tier-strong:hover {
            box-shadow: var(--shadow-lg), 0 0 60px rgba(6, 182, 212, 0.25);
        }

        /* Keyboard focus state */
        .featured-pick-card.keyboard-focus {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
        }

        /* Legacy card styles - kept for compatibility but not used in new layout */
        .card-top-row-legacy {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }

        .card-player-section-legacy {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
            min-width: 0;
        }

        .card-player-section-legacy .team-logo {
            width: 36px !important;
            height: 36px !important;
            flex-shrink: 0;
        }

        .card-player-info-legacy {
            display: flex;
            flex-direction: column;
            min-width: 0;
            flex: 1;
        }

        .card-player-name-legacy {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 160px;
        }

        .card-player-meta {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Pick section - line + badge inline */
        .card-pick-section {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }

        .card-pick-line {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .card-pick-market {
            font-size: 11px;
            color: var(--text-muted);
        }

        .card-pick-badge {
            font-size: 11px;
            font-weight: 700;
            padding: 5px 10px;
            border-radius: 6px;
        }

        .card-pick-badge.pick-over {
            background: var(--over-soft);
            color: var(--over);
        }

        .card-pick-badge.pick-under {
            background: var(--under-soft);
            color: var(--under);
        }

        .card-pick-line {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .card-pick-market {
            font-size: 9px;
            color: var(--text-muted);
            max-width: 70px;
            text-align: right;
            line-height: 1.2;
        }

        /* Stats row - tighter */
        .card-stats-section {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            padding-top: 12px;
            border-top: 1px solid var(--border-subtle);
        }

        .card-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 4px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            transition: background 0.15s ease;
        }

        .card-stat:hover {
            background: var(--bg-hover);
        }

        .card-stat-value {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }

        .card-stat-value.conf-elite {
            color: var(--conf-elite);
            text-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
        }
        .card-stat-value.conf-high { color: var(--conf-high); }
        .card-stat-value.edge-positive {
            color: var(--conf-elite);
            text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        }
        .card-stat-value.edge-negative { color: var(--under); }

        .card-stat-label {
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
            font-weight: 500;
        }

        /* Remove click hint to save space */
        .card-click-hint {
            display: none;
        }

        /* ===== GAME LINE CARDS - REDESIGNED ===== */
        .game-lines-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 16px;
            padding: 12px 0;
        }

        /* Main Card Container - Premium Glass */
        .gl-card {
            background: var(--glass-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-xl);
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow-md);
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .gl-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl), 0 0 50px rgba(59, 130, 246, 0.15);
            border-color: var(--accent);
        }

        /* Tier-based card styling */
        .gl-tier-elite {
            border-color: rgba(59, 130, 246, 0.4);
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, var(--glass-surface) 50%);
            box-shadow: var(--shadow-lg), 0 0 40px rgba(59, 130, 246, 0.15);
        }

        .gl-tier-elite:hover {
            box-shadow: var(--shadow-xl), 0 0 60px rgba(59, 130, 246, 0.25);
        }

        .gl-tier-high {
            border-color: rgba(6, 182, 212, 0.35);
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, var(--glass-surface) 50%);
        }

        .gl-tier-standard {
            border-color: var(--glass-border);
        }

        .gl-tier-low {
            border-color: var(--border-muted);
            opacity: 0.75;
        }

        /* Card Header with Team Logos */
        .gl-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            background: linear-gradient(180deg, var(--bg-tertiary) 0%, var(--bg-card) 100%);
            border-bottom: 1px solid var(--border-color);
        }

        .gl-matchup {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .gl-team {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .gl-team-logo {
            width: 36px;
            height: 36px;
            object-fit: contain;
            border-radius: 4px;
        }

        .gl-team-abbr {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.5px;
        }

        .gl-vs {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .gl-header-meta {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 4px;
        }

        /* Tier Badges - Premium Glow */
        .gl-tier-badge {
            font-size: 10px;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: var(--radius-full);
            text-transform: uppercase;
            letter-spacing: var(--tracking-wider);
            transition: all 0.2s ease;
        }

        .gl-tier-badge-elite {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.25) 100%);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.5);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.35);
        }

        .gl-tier-badge-high {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(8, 145, 178, 0.2) 100%);
            color: #22D3EE;
            border: 1px solid rgba(6, 182, 212, 0.4);
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.3);
        }

        .gl-tier-badge-standard {
            background: rgba(100, 116, 139, 0.2);
            color: var(--text-secondary);
            border: 1px solid rgba(100, 116, 139, 0.3);
        }

        .gl-tier-badge-low {
            background: rgba(71, 85, 105, 0.2);
            color: var(--text-muted);
            border: 1px solid rgba(71, 85, 105, 0.25);
        }

        /* Bets Container */
        .gl-bets-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            padding: 16px;
        }

        /* Individual Bet Block */
        .gl-bet-block {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 14px;
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--pick-accent, var(--accent-primary));
            transition: border-color 0.2s ease;
        }

        .gl-bet-block.gl-spread {
            border-left-color: var(--pick-accent, #3b82f6);
        }

        .gl-bet-block.gl-over {
            border-left-color: #3B82F6;
        }

        .gl-bet-block.gl-under {
            border-left-color: #f97316;
        }

        .gl-bet-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .gl-bet-type {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .gl-bet-type svg {
            opacity: 0.7;
        }

        .gl-total-icon {
            font-size: 14px;
            font-weight: 700;
        }

        .gl-over .gl-total-icon { color: #3B82F6; }
        .gl-under .gl-total-icon { color: #f97316; }

        .gl-bet-tier {
            font-size: 9px;
            padding: 2px 6px;
            border-radius: 8px;
        }

        /* Pick Display - Hero Element */
        .gl-bet-pick {
            font-size: 20px;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 10px;
            font-family: 'JetBrains Mono', monospace;
        }

        .gl-over .gl-bet-pick { color: #3B82F6; }
        .gl-under .gl-bet-pick { color: #f97316; }

        /* Confidence Meter */
        .gl-bet-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        .gl-conf-meter {
            flex: 1;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }

        .gl-conf-bar {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .gl-conf-bar.conf-elite {
            background: linear-gradient(90deg, #3B82F6, #60A5FA);
        }

        .gl-conf-bar.conf-high {
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
        }

        .gl-conf-bar.conf-standard {
            background: linear-gradient(90deg, #8b5cf6, #a78bfa);
        }

        .gl-conf-label {
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 9px;
            font-weight: 700;
            color: var(--text-primary);
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }

        .gl-edge-badge {
            font-size: 10px;
            font-weight: 600;
            padding: 3px 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }

        .gl-bet-fair {
            font-size: 10px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Analysis Toggle */
        .gl-analysis-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px;
            background: var(--bg-tertiary);
            cursor: pointer;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-top: 1px solid var(--border-color);
            transition: background 0.2s ease;
        }

        .gl-analysis-toggle:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        .gl-toggle-icon {
            transition: transform 0.2s ease;
            font-size: 10px;
        }

        .gl-toggle-icon.expanded {
            transform: rotate(180deg);
        }

        /* Analysis Section */
        .gl-analysis {
            padding: 16px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-color);
        }

        .gl-edge-summary {
            margin-bottom: 16px;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border-left: 3px solid var(--accent-primary);
        }

        .gl-edge-title {
            font-size: 10px;
            font-weight: 700;
            color: var(--accent-primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .gl-edge-text {
            font-size: 13px;
            color: var(--text-primary);
            line-height: 1.5;
        }

        /* Projection Visual */
        .gl-projection {
            margin-bottom: 16px;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
        }

        .gl-proj-label {
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .gl-proj-bar {
            position: relative;
            height: 24px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            margin-bottom: 6px;
        }

        .gl-proj-range {
            position: absolute;
            top: 4px;
            height: 16px;
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.3), rgba(59, 130, 246, 0.5));
            border-radius: 8px;
        }

        .gl-proj-line {
            position: absolute;
            top: 0;
            width: 3px;
            height: 100%;
            background: #f97316;
            border-radius: 2px;
        }

        .gl-proj-line::after {
            content: '';
            position: absolute;
            top: -4px;
            left: -3px;
            width: 9px;
            height: 9px;
            background: #f97316;
            border-radius: 50%;
        }

        .gl-proj-median {
            position: absolute;
            top: 0;
            width: 3px;
            height: 100%;
            background: #3B82F6;
            border-radius: 2px;
        }

        .gl-proj-median::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: -3px;
            width: 9px;
            height: 9px;
            background: #3B82F6;
            border-radius: 50%;
        }

        .gl-proj-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .gl-proj-line-label {
            color: #f97316;
            font-weight: 600;
        }

        /* Key Factors */
        .gl-factors {
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
        }

        .gl-factors-title {
            font-size: 10px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        .gl-factor {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            line-height: 1.4;
        }

        .gl-factor-icon {
            font-size: 12px;
            font-weight: 700;
            flex-shrink: 0;
        }

        .gl-factor-pro .gl-factor-icon {
            color: #3B82F6;
        }

        .gl-factor-con .gl-factor-icon {
            color: #ef4444;
        }

        /* Narrative Analysis Styles */
        .gl-narrative {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .gl-narrative-para {
            font-size: 13px;
            line-height: 1.6;
            color: var(--text-secondary);
            margin: 0;
            padding: 10px 12px;
            background: rgba(59, 130, 246, 0.04);
            border-radius: 8px;
            border-left: 3px solid rgba(59, 130, 246, 0.3);
        }

        .gl-narrative-para strong {
            color: var(--text-primary);
            font-weight: 600;
        }

        .gl-narrative-para:hover {
            background: rgba(59, 130, 246, 0.08);
            border-left-color: rgba(59, 130, 246, 0.6);
        }

        /* Filter Bar - Pill Style */
        #game-lines .filter-bar {
            margin-bottom: 16px;
        }

        #game-lines .filter-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        #game-lines .filter-btn {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid var(--border-color);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        #game-lines .filter-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        #game-lines .filter-btn.active {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
            color: white;
        }

        #game-lines .filter-btn .count {
            margin-left: 6px;
            padding: 2px 6px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            font-size: 10px;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .game-lines-grid {
                grid-template-columns: 1fr;
            }

            .gl-bets-container {
                grid-template-columns: 1fr;
            }

            .gl-team-logo {
                width: 28px;
                height: 28px;
            }

            .gl-bet-pick {
                font-size: 18px;
            }

            .gl-analysis {
                display: none;
            }

            .gl-analysis-toggle {
                display: none;
            }
        }

        /* Game Lines View Header */
        .gl-view-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 4px;
        }

        .gl-header-actions {
            display: flex;
            gap: 6px;
        }

        .gl-action-btn {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .gl-action-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-color: var(--accent-primary);
        }

        /* ===== BET TRACKER STYLES ===== */
        .tracker-summary {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .tracker-stat-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid var(--border-color);
        }

        .tracker-stat-card.positive {
            border-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.1);
        }

        .tracker-stat-card.negative {
            border-color: var(--under);
            background: rgba(239, 68, 68, 0.1);
        }

        .tracker-stat-value {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .tracker-stat-card.positive .tracker-stat-value {
            color: var(--accent-green);
        }

        .tracker-stat-card.negative .tracker-stat-value {
            color: var(--under);
        }

        .tracker-stat-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }

        .tracker-chart-container {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
        }

        .tracker-chart-container h3 {
            font-size: 14px;
            margin: 0 0 12px 0;
            color: var(--text-primary);
        }

        .pnl-chart {
            height: 200px;
        }

        .tracker-section {
            margin-bottom: 20px;
        }

        .tracker-section h3 {
            font-size: 14px;
            margin: 0 0 12px 0;
            color: var(--text-primary);
        }

        #tracker table td.positive {
            color: var(--accent-green);
            font-weight: 600;
        }

        #tracker table td.negative {
            color: var(--under);
            font-weight: 600;
        }

        @media (max-width: 768px) {
            .tracker-summary {
                grid-template-columns: repeat(2, 1fr);
            }

            .tracker-stat-card:last-child {
                grid-column: span 2;
            }
        }

        /* ===== BET SELECTION STYLES ===== */
        .bet-checkbox-container {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            position: relative;
            cursor: pointer;
            width: 20px;
            height: 20px;
            margin-right: 8px;
            flex-shrink: 0;
        }

        .bet-checkbox-container input {
            position: absolute;
            opacity: 0;
            cursor: pointer;
            height: 0;
            width: 0;
        }

        .bet-checkmark {
            position: absolute;
            top: 0;
            left: 0;
            height: 18px;
            width: 18px;
            background-color: var(--bg-tertiary);
            border: 2px solid var(--border-color);
            border-radius: 4px;
            transition: all 0.15s ease;
        }

        .bet-checkbox-container:hover .bet-checkmark {
            border-color: var(--accent-primary);
        }

        .bet-checkbox-container input:checked ~ .bet-checkmark {
            background-color: var(--accent-green);
            border-color: var(--accent-green);
        }

        .bet-checkmark:after {
            content: "";
            position: absolute;
            display: none;
        }

        .bet-checkbox-container input:checked ~ .bet-checkmark:after {
            display: block;
            left: 5px;
            top: 2px;
            width: 4px;
            height: 8px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
        }

        /* ===== BET SLIP PANEL - PERSISTENT FAB ===== */
        .bet-slip-toggle {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 14px 24px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4), 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s ease;
            animation: fab-pulse 2s ease-in-out infinite;
        }

        @keyframes fab-pulse {
            0%, 100% { box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4), 0 4px 12px rgba(0,0,0,0.3); }
            50% { box-shadow: 0 8px 32px rgba(16, 185, 129, 0.6), 0 4px 16px rgba(0,0,0,0.4); }
        }

        .bet-slip-toggle:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 32px rgba(16, 185, 129, 0.5), 0 6px 16px rgba(0,0,0,0.4);
        }

        .bet-slip-toggle .count {
            background: white;
            color: #059669;
            border-radius: 50%;
            width: 26px;
            height: 26px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 800;
        }

        /* Empty state - still visible but less prominent */
        .bet-slip-toggle.empty {
            background: linear-gradient(135deg, var(--surface-elevated) 0%, var(--surface-overlay) 100%);
            color: var(--text-secondary);
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            animation: none;
            border: 1px solid var(--border-default);
        }

        .bet-slip-toggle.empty .count {
            background: var(--surface-highlight);
            color: var(--text-muted);
        }

        .bet-slip-toggle.empty:hover {
            background: linear-gradient(135deg, var(--surface-highlight) 0%, var(--surface-elevated) 100%);
            color: var(--text-primary);
            border-color: var(--accent);
        }

        /* Has picks - extra emphasis */
        .bet-slip-toggle.has-picks {
            animation: fab-pulse-active 1.5s ease-in-out infinite;
        }

        @keyframes fab-pulse-active {
            0%, 100% { transform: scale(1); box-shadow: 0 8px 24px rgba(16, 185, 129, 0.5); }
            50% { transform: scale(1.05); box-shadow: 0 12px 36px rgba(16, 185, 129, 0.7); }
        }

        .bet-slip-panel {
            position: fixed;
            top: 0;
            right: -400px;
            width: 380px;
            height: 100vh;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            z-index: 1001;
            transition: right 0.3s ease;
            display: flex;
            flex-direction: column;
            box-shadow: -4px 0 20px rgba(0,0,0,0.3);
        }

        .bet-slip-panel.open {
            right: 0;
        }

        .bet-slip-header {
            padding: 16px 20px;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .bet-slip-header h3 {
            margin: 0;
            font-size: 16px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .bet-slip-close {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }

        .bet-slip-close:hover {
            color: var(--text-primary);
        }

        .bet-slip-content {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }

        .bet-slip-empty {
            text-align: center;
            color: var(--text-muted);
            padding: 40px 20px;
        }

        .bet-slip-empty-icon {
            font-size: 48px;
            margin-bottom: 12px;
            opacity: 0.5;
        }

        .bet-slip-item {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            position: relative;
        }

        .bet-slip-item-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }

        .bet-slip-player {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 13px;
        }

        .bet-slip-remove {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 16px;
            padding: 0;
            opacity: 0.6;
        }

        .bet-slip-remove:hover {
            color: var(--under);
            opacity: 1;
        }

        .bet-slip-details {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            font-size: 12px;
        }

        .bet-slip-pick {
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
        }

        .bet-slip-pick.pick-over {
            background: rgba(8, 145, 178, 0.15);
            color: var(--accent-cyan);
        }

        .bet-slip-pick.pick-under {
            background: rgba(239, 68, 68, 0.15);
            color: var(--under);
        }

        .bet-slip-line {
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .bet-slip-market {
            color: var(--text-muted);
        }

        .bet-slip-conf {
            color: var(--accent-green);
            font-weight: 600;
        }

        .bet-slip-footer {
            padding: 16px 20px;
            background: var(--bg-card);
            border-top: 1px solid var(--border-color);
        }

        .bet-slip-summary {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-size: 13px;
        }

        .bet-slip-summary-label {
            color: var(--text-muted);
        }

        .bet-slip-summary-value {
            color: var(--text-primary);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .bet-slip-actions {
            display: flex;
            gap: 10px;
        }

        .bet-slip-btn {
            flex: 1;
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
        }

        .bet-slip-btn.primary {
            background: var(--accent-green);
            color: white;
        }

        .bet-slip-btn.primary:hover {
            background: #2563EB;
        }

        .bet-slip-btn.secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .bet-slip-btn.secondary:hover {
            background: var(--bg-card);
            border-color: var(--accent-primary);
        }

        .bet-slip-btn.danger {
            background: rgba(239, 68, 68, 0.1);
            color: var(--under);
            border: 1px solid var(--under);
        }

        .bet-slip-btn.danger:hover {
            background: var(--under);
            color: white;
        }

        .bet-slip-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            display: none;
        }

        .bet-slip-overlay.open {
            display: block;
        }

        /* Toast notification for save confirmation */
        .bet-toast {
            position: fixed;
            bottom: 80px;
            right: 20px;
            background: var(--accent-green);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1002;
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s ease;
        }

        .bet-toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .featured-pick-tier {
            font-size: 9px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 3px;
            text-transform: uppercase;
        }

        .featured-pick-tier.tier-elite {
            background: rgba(5, 150, 105, 0.15);
            color: var(--accent-green);
        }

        .featured-pick-tier.tier-strong {
            background: rgba(8, 145, 178, 0.15);
            color: var(--accent-cyan);
        }

        .featured-pick-main {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .featured-pick-direction {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .featured-pick-badge {
            font-size: 14px;
            font-weight: 700;
            padding: 6px 12px;
            border-radius: 6px;
            text-transform: uppercase;
        }

        .featured-pick-badge.pick-over {
            background: var(--over-soft);
            color: var(--over);
        }

        .featured-pick-badge.pick-under {
            background: var(--under-soft);
            color: var(--under);
        }

        .featured-pick-line {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'SF Mono', monospace;
        }

        .featured-pick-market {
            font-size: 11px;
            color: var(--text-secondary);
            text-align: right;
        }

        .featured-pick-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .featured-stat-item {
            text-align: center;
            padding: 8px 4px;
            background: var(--bg-secondary);
            border-radius: 6px;
        }

        .featured-stat-value {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'SF Mono', monospace;
        }

        .featured-stat-value.conf-elite {
            color: var(--accent-green);
        }

        .featured-stat-value.edge-positive {
            color: var(--accent-green);
        }

        .featured-stat-label {
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 2px;
        }

        /* Model Factors Section */
        .model-factors-section {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-color);
        }

        .model-factors-title {
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }

        .model-factors-list {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }

        .factor-tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 500;
        }

        .factor-positive {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .factor-negative {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .factor-neutral {
            background: rgba(156, 163, 175, 0.15);
            color: var(--text-secondary);
            border: 1px solid rgba(156, 163, 175, 0.3);
        }

        /* Model Narrative Section - Why This Pick */
        .model-narrative {
            margin-top: 12px;
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            border-left: 3px solid var(--accent-primary);
        }

        .narrative-title {
            font-size: 10px;
            font-weight: 700;
            color: var(--accent-primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .narrative-content {
            font-size: 11px;
            line-height: 1.5;
            color: var(--text-secondary);
        }

        .narrative-positive {
            color: var(--conf-elite);
            font-weight: 600;
        }

        .narrative-negative {
            color: var(--under);
            font-weight: 600;
        }

        .narrative-neutral {
            color: var(--text-muted);
            font-weight: 500;
        }

        .narrative-section {
            margin-bottom: 8px;
            color: var(--text-primary);
            font-size: 11px;
        }

        .section-icon {
            margin-right: 6px;
            font-size: 12px;
        }

        /* Styled Narrative Paragraphs - Clean prose without boxes */
        .narrative-para {
            margin-bottom: 8px;
            padding: 0;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 13px;
            line-height: 1.7;
        }

        .narrative-para.setup {
            /* No special styling - plain text */
        }

        .narrative-para.volume {
            /* No special styling - plain text */
        }

        .narrative-para.efficiency {
            /* No special styling - plain text */
        }

        .narrative-para.matchup {
            /* No special styling - plain text */
        }

        .narrative-para.verdict-para {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-subtle);
            font-weight: 500;
            color: var(--text-primary);
        }

        /* Narrative Highlight Spans - Subtle inline styling */
        .hl-pct {
            color: var(--text-primary);
            font-weight: 600;
        }

        .hl-stat {
            color: var(--text-primary);
            font-weight: 600;
        }

        .hl-stat.hl-result {
            color: var(--over);
            font-weight: 700;
        }

        .hl-edge {
            font-weight: 600;
        }

        .hl-edge.positive {
            color: var(--over);
        }

        .hl-edge.negative {
            color: var(--under);
        }

        .hl-confidence.high {
            color: var(--over);
            font-weight: 600;
        }

        .hl-confidence.medium {
            color: var(--star-filled);
            font-weight: 600;
        }

        .factors-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin: 16px 0;
        }

        .factors-column {
            padding: 14px;
            border-radius: 8px;
        }

        .factors-supporting {
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .factors-opposing {
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .factors-header {
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .factors-supporting .factors-header { color: #3B82F6; }
        .factors-opposing .factors-header { color: #dc2626; }

        .factor-item {
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin-bottom: 6px;
            font-size: 12px;
            line-height: 1.4;
        }

        .factor-name {
            color: var(--text-secondary);
            min-width: 100px;
            flex-shrink: 0;
        }

        .factor-value {
            font-weight: 600;
            color: var(--text-primary);
            min-width: 50px;
        }

        .factor-note {
            color: var(--text-muted);
            font-size: 11px;
        }

        .factors-empty {
            color: var(--text-muted);
            font-style: italic;
            margin: 12px 0;
            padding-left: 24px;
        }

        .conflict-badge {
            display: inline-block;
            background: rgba(245, 158, 11, 0.15);
            color: #b45309;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            margin-left: 8px;
            font-weight: 500;
        }

        .narrative-conclusion {
            display: none; /* Hidden - stats are now in card header */
        }

        .pick-badge {
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 13px;
        }

        .pick-badge.pick-over {
            background: var(--over-soft);
            color: var(--over);
        }

        .pick-badge.pick-under {
            background: var(--under-soft);
            color: var(--under);
        }

        .conclusion-stats {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .prob {
            font-weight: 600;
            font-size: 13px;
        }

        .prob.high { color: #2563EB; }
        .prob.medium { color: #d97706; }
        .prob.low { color: var(--text-muted); }

        .edge {
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
        }

        .edge.positive {
            background: rgba(59, 130, 246, 0.12);
            color: #3B82F6;
        }

        .edge.negative {
            background: rgba(239, 68, 68, 0.12);
            color: #dc2626;
        }

        /* ============================================================
           PREMIUM ANIMATIONS & MICRO-INTERACTIONS
           ============================================================ */

        /* Fade in animation */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        /* Pulse glow for important elements (purple) */
        @keyframes pulseGlow {
            0%, 100% {
                box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
            }
            50% {
                box-shadow: 0 0 35px rgba(59, 130, 246, 0.5);
            }
        }

        @keyframes shimmer {
            0% {
                background-position: -200% 0;
            }
            100% {
                background-position: 200% 0;
            }
        }

        /* Animation utility classes */
        .animate-in {
            animation: fadeIn 0.3s ease forwards;
        }

        .animate-in-scale {
            animation: fadeInScale 0.25s ease forwards;
        }

        .animate-slide-left {
            animation: slideInLeft 0.3s ease forwards;
        }

        .pulse-glow {
            animation: pulseGlow 2s ease-in-out infinite;
        }

        /* Stagger animation delays for lists */
        .stagger-1 { animation-delay: 0.05s; }
        .stagger-2 { animation-delay: 0.1s; }
        .stagger-3 { animation-delay: 0.15s; }
        .stagger-4 { animation-delay: 0.2s; }
        .stagger-5 { animation-delay: 0.25s; }

        /* Button press effect */
        .btn:active, button:active {
            transform: scale(0.98);
        }

        /* Card hover lift effect */
        .card-lift {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .card-lift:hover {
            transform: translateY(-4px);
            box-shadow:
                0 12px 32px rgba(0, 0, 0, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        }

        /* Shimmer loading effect */
        .shimmer {
            background: linear-gradient(
                90deg,
                rgba(255, 255, 255, 0) 0%,
                rgba(255, 255, 255, 0.05) 50%,
                rgba(255, 255, 255, 0) 100%
            );
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }

        /* ============================================================
           INTERACTIVE FEATURES - TOOLTIPS, SPARKLINES, ENHANCED HOVER
           ============================================================ */

        /* === TOOLTIP SYSTEM === */
        .tooltip-trigger {
            position: relative;
            cursor: help;
        }

        .tooltip-trigger::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%) translateY(4px);
            padding: 8px 12px;
            background: var(--surface-elevated);
            border: 1px solid var(--border-default);
            border-radius: 8px;
            font-size: 11px;
            color: var(--text-primary);
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.15s ease;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            pointer-events: none;
        }

        .tooltip-trigger::before {
            content: '';
            position: absolute;
            bottom: calc(100% + 2px);
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: var(--surface-elevated);
            opacity: 0;
            visibility: hidden;
            transition: all 0.15s ease;
            z-index: 1001;
            pointer-events: none;
        }

        .tooltip-trigger:hover::after,
        .tooltip-trigger:hover::before {
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(0);
        }

        /* === INLINE SPARKLINE CHARTS === */
        .sparkline-cell {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .sparkline-mini {
            display: flex;
            align-items: flex-end;
            gap: 2px;
            height: 20px;
            min-width: 36px;
            padding: 2px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }

        .sparkline-bar {
            width: 6px;
            border-radius: 2px 2px 0 0;
            transition: all 0.15s ease;
            min-height: 3px;
        }

        .sparkline-bar.hit { background: var(--over); }
        .sparkline-bar.miss { background: var(--text-muted); opacity: 0.5; }

        .sparkline-mini:hover .sparkline-bar {
            opacity: 0.7;
        }
        .sparkline-mini:hover .sparkline-bar:hover {
            opacity: 1;
            transform: scaleY(1.15);
        }

        /* === ENHANCED ROW HOVER === */
        .expandable-row {
            transition: all 0.15s ease;
            cursor: pointer;
            border-left: 3px solid transparent;
        }

        .expandable-row:hover {
            background: var(--bg-hover);
        }

        .expandable-row:hover td:first-child {
            border-left-color: var(--accent);
        }

        .expandable-row:hover .badge {
            transform: scale(1.03);
        }

        .expandable-row:hover .player-name,
        .expandable-row:hover .player-name-cell {
            color: var(--accent);
        }

        .expandable-row.expanded {
            background: var(--surface-highlight);
            border-left-color: var(--accent);
        }

        .expandable-row.expanded + .logic-row {
            animation: slideDown 0.2s ease-out;
        }

        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* === KEYBOARD NAVIGATION === */
        .expandable-row.focused {
            outline: 2px solid var(--accent);
            outline-offset: -2px;
        }

        /* === ANIMATED CONFIDENCE METER === */
        .confidence-meter {
            width: 100%;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .confidence-fill.elite {
            background: linear-gradient(90deg, #1D4ED8, #3B82F6);
            box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
        }

        .confidence-fill.elite::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            );
            animation: shimmer 2s infinite;
        }

        .confidence-fill.high {
            background: linear-gradient(90deg, #2563EB, #3B82F6);
        }

        .confidence-fill.standard {
            background: var(--text-muted);
        }

        /* === COPY BUTTON === */
        .copy-btn {
            padding: 4px;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            opacity: 0;
            transition: all 0.15s ease;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .expandable-row:hover .copy-btn {
            opacity: 0.6;
        }

        .copy-btn:hover {
            opacity: 1 !important;
            color: var(--accent);
            background: var(--accent-muted);
        }

        /* === QUICK FILTER PILLS === */
        .quick-filters {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            padding: 0 4px;
        }

        .filter-pill {
            padding: 6px 14px;
            background: var(--bg-card);
            border: 1px solid var(--border-default);
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .filter-pill:hover {
            background: var(--bg-hover);
            border-color: var(--border-emphasis);
            color: var(--text-primary);
        }

        .filter-pill.active {
            background: var(--accent-muted);
            border-color: var(--accent);
            color: var(--accent);
        }

        /* === SCROLL TO TOP BUTTON === */
        .scroll-top-btn {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 44px;
            height: 44px;
            background: var(--accent);
            border: none;
            border-radius: 50%;
            color: white;
            font-size: 20px;
            cursor: pointer;
            opacity: 0;
            visibility: hidden;
            transition: all 0.2s ease;
            z-index: 900;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .scroll-top-btn.visible {
            opacity: 1;
            visibility: visible;
        }

        .scroll-top-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
        }

        /* === TOAST NOTIFICATIONS === */
        .toast-notification {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            padding: 12px 20px;
            background: var(--surface-elevated);
            border: 1px solid var(--border-default);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
            font-weight: 500;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            z-index: 10000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }

        .toast-notification.show {
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(0);
        }

        .toast-notification.success {
            border-left: 4px solid var(--accent);
        }

        /* Tablet breakpoint - 2 column grid */
        @media (max-width: 1024px) {
            .logic-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .logic-section {
                min-height: 100px;
            }
        }

        @media (max-width: 768px) {
            .stats-summary {
                flex-wrap: wrap;
            }

            .stat-card {
                flex: 1 1 calc(25% - 1px);
                min-width: 80px;
            }

            .filter-bar {
                flex-direction: column;
                align-items: stretch;
            }

            .search-box {
                width: 100%;
            }

            .header-stats {
                display: none;
            }

            .logic-grid {
                grid-template-columns: 1fr;
            }

            .modal-content {
                max-height: 90vh;
                margin: 10px;
            }

            .modal-stats-row {
                grid-template-columns: repeat(2, 1fr);
            }

            .modal-factors-grid {
                grid-template-columns: 1fr;
            }

            .modal-matchup-info {
                grid-template-columns: 1fr;
            }

            /* MOBILE TABLE TRANSFORMATION - Convert rows to cards */
            .picks-table thead {
                display: none;
            }

            .picks-table tbody tr.expandable-row {
                display: block;
                padding: 16px;
                margin-bottom: 12px;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                background: var(--bg-card);
                border-left: 3px solid transparent;
            }

            .picks-table tbody tr.expandable-row:hover {
                border-left-color: var(--accent-primary);
            }

            .picks-table tbody tr.expandable-row td {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid var(--border-light);
                border-left: none !important;
            }

            .picks-table tbody tr.expandable-row td:last-child {
                border-bottom: none;
            }

            .picks-table tbody tr.expandable-row td::before {
                content: attr(data-label);
                font-weight: 600;
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                flex-shrink: 0;
                margin-right: 12px;
            }

            .picks-table tbody tr.expandable-row td:first-child::before {
                content: none;  /* Player cell doesn't need label */
            }

            .picks-table tbody tr.expandable-row::after {
                display: none;  /* Hide "click to expand" on mobile */
            }

            .logic-row {
                display: block !important;
            }

            .logic-row td {
                display: block !important;
                padding: 12px !important;
            }

            .logic-grid {
                grid-template-columns: 1fr !important;
            }

            /* Tab scrolling on mobile */
            .tabs {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
            }

            .tabs::-webkit-scrollbar {
                display: none;
            }

            .tab {
                white-space: nowrap;
                flex-shrink: 0;
            }
        }

        /* ========================================
           CHEAT SHEET TABLE STYLES
           BettingPros-inspired dense data table
           ======================================== */

        .table-section {
            background: var(--surface-raised);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-default);
        }

        .table-section-header {
            margin-bottom: 16px;
        }

        .table-section-header h2 {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .table-section-header .subtitle {
            font-size: 13px;
            color: var(--text-muted);
        }

        /* Dense data table */
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
        }

        .data-table th {
            position: sticky;
            top: 0;
            background: var(--surface-elevated);
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-default);
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .data-table th:hover {
            background: var(--surface-highlight);
        }

        .data-table th.no-sort {
            cursor: default;
        }

        .data-table th.no-sort:hover {
            background: var(--surface-elevated);
        }

        .data-table th.sorted-asc::after { content: ' ↑'; color: var(--accent); }
        .data-table th.sorted-desc::after { content: ' ↓'; color: var(--accent); }

        .data-table td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-subtle);
            vertical-align: middle;
        }

        .data-table tr:hover {
            background: var(--bg-card-hover);
        }

        /* Tier-based row highlighting - Visual hierarchy */
        .data-table tr.tier-elite {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.18) 0%, rgba(245, 158, 11, 0.05) 100%) !important;
            box-shadow: inset 5px 0 0 #F59E0B;
        }
        .data-table tr.tier-elite:hover {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.28) 0%, rgba(245, 158, 11, 0.08) 100%) !important;
        }
        /* Gold border on avatar for ELITE (works with headshots) */
        .data-table tr.tier-elite .player-avatar {
            border: 2px solid #F59E0B;
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
        }
        .data-table tr.tier-elite .player-avatar:not(.has-image) {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            color: #fff;
            font-weight: 800;
        }
        .data-table tr.tier-strong {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.03) 100%) !important;
            box-shadow: inset 5px 0 0 #22C55E;
        }
        .data-table tr.tier-strong:hover {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.25) 0%, rgba(34, 197, 94, 0.06) 100%) !important;
        }
        /* Green border on avatar for STRONG (works with headshots) */
        .data-table tr.tier-strong .player-avatar {
            border: 2px solid #22C55E;
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.4);
        }
        .data-table tr.tier-strong .player-avatar:not(.has-image) {
            background: linear-gradient(135deg, #22C55E, #16A34A);
            color: #fff;
            font-weight: 800;
        }

        /* Tier badge in player meta line */
        .tier-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-left: 4px;
            vertical-align: middle;
            text-transform: uppercase;
        }
        .tier-badge.elite {
            background: rgba(245, 158, 11, 0.2);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }
        .tier-badge.strong {
            background: rgba(34, 197, 94, 0.2);
            color: #22C55E;
            border: 1px solid rgba(34, 197, 94, 0.4);
        }

        /* Player cell component */
        .player-cell {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 160px;
        }

        .player-avatar-wrapper {
            position: relative;
            flex-shrink: 0;
        }

        .player-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--surface-elevated);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 11px;
            color: var(--text-secondary);
            flex-shrink: 0;
            border: 1px solid var(--border-muted);
            overflow: hidden;
            position: relative;
        }

        .player-avatar.has-image {
            background: var(--bg-card);
        }

        .player-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center top;
        }

        .player-avatar .avatar-fallback {
            position: absolute;
            width: 100%;
            height: 100%;
            display: none;
            align-items: center;
            justify-content: center;
            background: var(--surface-elevated);
            font-weight: 700;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .player-team-logo {
            position: absolute;
            bottom: -3px;
            right: -3px;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--bg-card);
            border: 1px solid var(--border-default);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .player-team-logo img {
            width: 14px;
            height: 14px;
            object-fit: contain;
        }

        .player-info {
            display: flex;
            flex-direction: column;
            gap: 1px;
        }

        .player-name {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 13px;
        }

        .player-meta {
            font-size: 11px;
            color: var(--text-muted);
        }

        /* Prop cell */
        .prop-cell {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .prop-line {
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
        }

        .prop-market {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        /* Rating stars */
        .rating-stars {
            display: flex;
            gap: 2px;
        }

        .star {
            font-size: 14px;
            color: rgba(100, 116, 139, 0.4);
        }

        .star.filled {
            color: #FBBF24;
            text-shadow: 0 0 6px rgba(251, 191, 36, 0.6);
        }

        /* EV/Diff indicators */
        .ev-positive {
            color: var(--positive);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .ev-negative {
            color: var(--negative);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .diff-cell {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
        }

        .diff-cell.negative { color: var(--negative); }
        .diff-cell.positive { color: var(--positive); }

        .mono {
            font-family: 'JetBrains Mono', monospace;
        }

        /* Pick badge */
        .pick-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
        }

        .pick-badge.over {
            background: var(--over-soft);
            color: var(--over);
            border: 1px solid var(--over-border);
        }

        .pick-badge.under {
            background: var(--under-soft);
            color: var(--under);
            border: 1px solid var(--under-border);
        }

        /* Hit rate bars */
        .hit-rate {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .hit-rate-bar {
            width: 36px;
            height: 5px;
            background: var(--surface-elevated);
            border-radius: 3px;
            overflow: hidden;
        }

        .hit-rate-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .hit-rate-fill.high { background: var(--positive); }
        .hit-rate-fill.medium { background: var(--warning); }
        .hit-rate-fill.low { background: var(--negative); }

        .hit-rate-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
            min-width: 28px;
        }

        /* Filter bar with pills */
        .filter-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-subtle);
            align-items: center;
        }

        .filter-bar.games-row {
            padding: 10px 0;
            gap: 10px;
            border-bottom: none;
        }

        .filter-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            flex-shrink: 0;
        }

        .filter-pills-scroll {
            display: flex;
            gap: 6px;
            overflow-x: auto;
            flex: 1;
            padding-bottom: 4px;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
            scrollbar-color: var(--border-muted) transparent;
        }

        .filter-pills-scroll::-webkit-scrollbar {
            height: 4px;
        }

        .filter-pills-scroll::-webkit-scrollbar-track {
            background: transparent;
        }

        .filter-pills-scroll::-webkit-scrollbar-thumb {
            background: var(--border-muted);
            border-radius: 2px;
        }

        .filter-divider {
            color: var(--border-muted);
            font-size: 12px;
            margin: 0 2px;
        }

        .cs-results-count {
            font-size: 11px;
            color: var(--text-muted);
            flex-shrink: 0;
            margin-left: auto;
        }

        .filter-pill {
            padding: 6px 12px;
            border-radius: 16px;
            background: var(--surface-raised);
            border: 1px solid var(--border-muted);
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .filter-pill.small {
            padding: 4px 10px;
            font-size: 11px;
            border-radius: 12px;
        }

        /* Game pills with team logos */
        .filter-pill.game-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            border-radius: 16px;
        }

        .pill-logo {
            width: 18px;
            height: 18px;
            object-fit: contain;
        }

        .pill-at {
            font-size: 10px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .filter-pill.game-pill.active {
            background: var(--accent-muted);
            border-color: var(--accent);
        }

        .filter-pill.game-pill.active .pill-at {
            color: var(--accent);
        }

        .filter-pill:hover {
            background: var(--surface-overlay);
            border-color: var(--border-default);
        }

        .filter-pill.active {
            background: var(--accent-muted);
            border-color: var(--accent);
            color: var(--accent);
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.5;
        }
        .empty-state-text {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        .empty-state-hint {
            font-size: 13px;
            color: var(--text-muted);
        }

        /* L5 hit rate indicator for all props */
        .l5-hit-indicator {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }
        .l5-hit-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 700;
        }
        .l5-hit-value.high {
            color: #22C55E;
        }
        .l5-hit-value.medium {
            color: #EAB308;
        }
        .l5-hit-value.low {
            color: #EF4444;
        }
        .l5-hit-label {
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.3px;
        }

        /* Analyze button */
        .analyze-btn {
            background: var(--surface-elevated);
            border: 1px solid var(--border-muted);
            border-radius: 6px;
            padding: 6px 10px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .analyze-btn:hover {
            background: var(--accent-muted);
            border-color: var(--accent);
        }

        /* Responsive table scroll */
        .table-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
        }

        /* Mobile table adjustments */
        @media (max-width: 768px) {
            .data-table th:first-child,
            .data-table td:first-child {
                position: sticky;
                left: 0;
                background: var(--surface-raised);
                z-index: 1;
            }

            .data-table td:first-child {
                background: var(--bg-primary);
            }

            .data-table tr:hover td:first-child {
                background: var(--bg-card-hover);
            }

            .player-cell {
                min-width: 140px;
            }

            .player-avatar {
                display: none;
            }

            .filter-bar {
                overflow-x: auto;
                flex-wrap: nowrap;
            }

            .table-section {
                padding: 12px;
                border-radius: 8px;
            }
        }

        /* ========================================
           GAME LINES TABLE STYLES
           ======================================== */

        /* Game cell with team avatars */
        .gl-game-cell {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 180px;
        }

        .gl-team-avatars {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .gl-team-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--surface-elevated);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 10px;
            color: var(--text-secondary);
            border: 1px solid var(--border-muted);
        }

        .gl-team-avatar.away {
            background: var(--surface-raised);
        }

        .gl-team-avatar.home {
            background: var(--accent-muted);
            color: var(--accent);
        }

        .gl-vs {
            font-size: 10px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .gl-game-info {
            display: flex;
            flex-direction: column;
        }

        .gl-game-name {
            font-weight: 600;
            font-size: 13px;
            color: var(--text-primary);
        }

        /* Bet type badge */
        .gl-bet-type {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .gl-bet-type.spread {
            background: rgba(139, 92, 246, 0.15);
            color: #A78BFA;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        /* Game time cell */
        .gl-time-cell {
            font-size: 12px;
            color: var(--text-muted);
            white-space: nowrap;
        }

        .gl-bet-type.total {
            background: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        /* Pick badge for spread */
        .pick-badge.spread {
            background: rgba(139, 92, 246, 0.15);
            color: #A78BFA;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        /* Tier badge */
        .gl-tier-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .gl-tier-badge.elite {
            background: rgba(245, 158, 11, 0.2);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }

        .gl-tier-badge.high {
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .gl-tier-badge.standard {
            background: rgba(148, 163, 184, 0.15);
            color: #94A3B8;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }

        .gl-tier-badge.low {
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Filter count badge */
        .filter-count {
            margin-left: 4px;
            font-size: 10px;
            opacity: 0.7;
        }

        /* Mobile adjustments for game lines */
        @media (max-width: 768px) {
            .gl-game-cell {
                min-width: 140px;
            }

            .gl-team-avatars {
                display: none;
            }

            .gl-game-name {
                font-size: 12px;
            }
        }

        /* ========================================
           BettingPros-Style Game Cards
           ======================================== */

        .bp-games-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            padding: 20px 0;
        }

        /* ===== GAME CARD ===== */
        .bp-game-card {
            background: linear-gradient(180deg, var(--bg-card) 0%, rgba(17, 24, 39, 0.95) 100%);
            border: 1px solid var(--border-default);
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .bp-game-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border-color: var(--border-muted);
        }

        .bp-game-card.elite {
            border-color: rgba(245, 158, 11, 0.6);
            box-shadow: 0 4px 24px rgba(245, 158, 11, 0.15), inset 0 1px 0 rgba(245, 158, 11, 0.3);
            background: linear-gradient(180deg, rgba(245, 158, 11, 0.08) 0%, var(--bg-card) 100%);
        }

        .bp-game-card.elite:hover {
            box-shadow: 0 8px 40px rgba(245, 158, 11, 0.25);
        }

        .bp-game-card.high {
            border-color: rgba(34, 197, 94, 0.5);
            box-shadow: 0 4px 24px rgba(34, 197, 94, 0.1), inset 0 1px 0 rgba(34, 197, 94, 0.2);
            background: linear-gradient(180deg, rgba(34, 197, 94, 0.05) 0%, var(--bg-card) 100%);
        }

        .bp-game-card.high:hover {
            box-shadow: 0 8px 40px rgba(34, 197, 94, 0.2);
        }

        /* ===== CARD HEADER - Clean & Scannable ===== */
        .bp-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            background: linear-gradient(135deg, rgba(255,255,255,0.02) 0%, transparent 100%);
            border-bottom: 1px solid var(--border-subtle);
            gap: 16px;
        }

        .bp-matchup-row {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        /* Team Logo with CSS fallback (no flash) */
        .bp-team-logo-wrapper {
            position: relative;
            width: 48px;
            height: 48px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--team-primary, #374151), var(--team-secondary, #1f2937));
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .bp-logo {
            width: 100%;
            height: 100%;
            object-fit: contain;
            position: relative;
            z-index: 2;
        }

        .bp-logo-text {
            position: absolute;
            z-index: 1;
            font-weight: 800;
            font-size: 14px;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }

        .bp-matchup-center {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }

        .bp-team-names {
            font-weight: 700;
            font-size: 15px;
            color: var(--text-primary);
            letter-spacing: 0.5px;
        }

        .bp-game-time {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .bp-card-badges {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }

        .bp-weather-pill {
            display: contents;
        }

        .bp-weather {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            font-size: 12px;
        }

        .bp-weather-icon {
            font-size: 14px;
        }

        .bp-weather-text {
            font-weight: 600;
            color: var(--text-secondary);
        }

        /* ===== PICKS CONTAINER ===== */
        .bp-picks-container {
            display: flex;
            flex-direction: column;
        }

        /* ===== PICK ROW - Hero Pick Layout ===== */
        .bp-pick-row {
            display: grid;
            grid-template-columns: 100px 1fr auto;
            align-items: center;
            padding: 14px 20px;
            gap: 16px;
            border-bottom: 1px solid var(--border-subtle);
            transition: background 0.2s ease;
        }

        .bp-pick-row:last-child {
            border-bottom: none;
        }

        .bp-pick-row:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        /* Tier row styling */
        .bp-pick-row.tier-elite-row {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.03) 100%);
            border-left: 4px solid #F59E0B;
        }

        .bp-pick-row.tier-high-row {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.02) 100%);
            border-left: 4px solid #22C55E;
        }

        /* Pick Type */
        .bp-pick-type {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .bp-type-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Tier Badge */
        .bp-tier-badge {
            font-size: 9px;
            font-weight: 800;
            padding: 3px 6px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .bp-tier-badge.elite {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            color: #1F2937;
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
        }

        .bp-tier-badge.high {
            background: linear-gradient(135deg, #22C55E, #16A34A);
            color: #1F2937;
        }

        /* Pick Hero - THE MAIN ATTRACTION */
        .bp-pick-hero {
            display: flex;
            justify-content: flex-start;
        }

        .bp-pick-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 16px;
            letter-spacing: 0.5px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .bp-pick-chip:hover {
            transform: scale(1.02);
        }

        .bp-pick-arrow {
            font-size: 14px;
            opacity: 0.8;
        }

        /* Spread picks - Purple */
        .bp-pick-chip.spread {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(99, 102, 241, 0.2) 100%);
            color: #C4B5FD;
            border: 2px solid rgba(139, 92, 246, 0.5);
            box-shadow: 0 4px 16px rgba(139, 92, 246, 0.2);
        }

        /* Over picks - Green with UP arrow */
        .bp-pick-chip.over {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(22, 163, 74, 0.2) 100%);
            color: #86EFAC;
            border: 2px solid rgba(34, 197, 94, 0.5);
            box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2);
        }

        /* Under picks - Red with DOWN arrow */
        .bp-pick-chip.under {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.2) 100%);
            color: #FCA5A5;
            border: 2px solid rgba(239, 68, 68, 0.5);
            box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
        }

        /* Pick Stats */
        .bp-pick-stats {
            display: flex;
            gap: 16px;
        }

        .bp-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 50px;
        }

        .bp-stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 14px;
            color: var(--text-primary);
        }

        .bp-stat-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .bp-stat.positive .bp-stat-value {
            color: #4ADE80;
        }

        .bp-stat.negative .bp-stat-value {
            color: #F87171;
        }

        /* Card Tier Label */
        .bp-card-tier-label {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            padding: 6px 14px;
            border-radius: 20px;
            text-transform: uppercase;
        }

        .bp-card-tier-label.elite {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.3) 0%, rgba(217, 119, 6, 0.3) 100%);
            color: #FCD34D;
            border: 1px solid rgba(245, 158, 11, 0.5);
            box-shadow: 0 0 16px rgba(245, 158, 11, 0.3);
            animation: eliteGlow 2s ease-in-out infinite;
        }

        .bp-card-tier-label.high {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(22, 163, 74, 0.25) 100%);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.2);
        }

        @keyframes eliteGlow {
            0%, 100% { box-shadow: 0 0 16px rgba(245, 158, 11, 0.3); }
            50% { box-shadow: 0 0 24px rgba(245, 158, 11, 0.5); }
        }

        /* ============================================
           GAME LINE CARDS - Premium Grid Layout
           ============================================ */
        .bp-games-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 16px;
            padding: 20px 0;
        }

        .compact-card {
            background: linear-gradient(180deg, var(--bg-card) 0%, rgba(17, 24, 39, 0.98) 100%);
            border: 1px solid var(--border-default);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.25s ease;
            min-height: 140px;
            display: flex;
            flex-direction: column;
        }

        .compact-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            border-color: var(--border-muted);
        }

        /* ELITE cards - prominent gold styling */
        .compact-card.elite {
            border-color: rgba(245, 158, 11, 0.6);
            border-left: 5px solid #F59E0B;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, var(--bg-card) 50%);
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
        }

        .compact-card.elite:hover {
            box-shadow: 0 12px 40px rgba(245, 158, 11, 0.25);
        }

        /* HIGH cards - green accent */
        .compact-card.high {
            border-color: rgba(34, 197, 94, 0.5);
            border-left: 5px solid #22C55E;
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.05) 0%, var(--bg-card) 50%);
        }

        .compact-card.high:hover {
            box-shadow: 0 12px 32px rgba(34, 197, 94, 0.2);
        }

        /* Card Header - Clean with larger logos */
        .compact-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-subtle);
        }

        .compact-team-logo {
            position: relative;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: var(--team-primary, #374151);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .compact-team-logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            position: relative;
            z-index: 2;
        }

        .compact-team-logo .logo-fallback {
            position: absolute;
            z-index: 1;
            font-weight: 700;
            font-size: 11px;
            color: white;
        }

        .compact-matchup {
            flex: 1;
            min-width: 0;
        }

        .compact-teams {
            font-weight: 700;
            font-size: 15px;
            color: var(--text-primary);
            letter-spacing: 0.3px;
        }

        .compact-meta {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .compact-weather {
            font-size: 11px;
            margin-left: 6px;
            opacity: 0.9;
        }

        /* Tier badge - smaller, doesn't compete */
        .compact-tier {
            font-size: 10px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .compact-tier.elite {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            color: #1a1a1a;
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4);
        }

        .compact-tier.high {
            background: linear-gradient(135deg, #22C55E, #16A34A);
            color: #1a1a1a;
        }

        /* Picks Container */
        .compact-picks {
            padding: 12px 16px;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 8px;
        }

        .compact-bet {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        /* Pick chips - larger, more prominent */
        .compact-pick {
            font-weight: 700;
            font-size: 13px;
            padding: 6px 12px;
            border-radius: 6px;
            white-space: nowrap;
            min-width: 85px;
            text-align: center;
        }

        .compact-pick.spread {
            background: rgba(139, 92, 246, 0.2);
            color: #C4B5FD;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        .compact-pick.over {
            background: rgba(34, 197, 94, 0.2);
            color: #86EFAC;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .compact-pick.under {
            background: rgba(239, 68, 68, 0.2);
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Stats - larger, closer to pick */
        .compact-stats {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .compact-conf {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .compact-edge {
            font-weight: 700;
        }

        .compact-edge.positive {
            color: #4ADE80;
        }

        .compact-edge.negative {
            color: #F87171;
        }

        /* ============================================
           FEATURED PROP CARDS - Hybrid Layout
           ============================================ */

        .featured-picks-section {
            margin-bottom: 32px;
        }

        .featured-picks-header {
            margin-bottom: 20px;
        }

        .featured-picks-header h2 {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
        }

        .featured-picks-header .subtitle {
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .featured-picks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 16px;
        }

        /* Featured Prop Card */
        .fp-card {
            background: linear-gradient(180deg, var(--bg-card) 0%, rgba(17, 24, 39, 0.98) 100%);
            border: 1px solid var(--border-default);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
        }

        .fp-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            border-color: var(--border-muted);
        }

        /* ELITE tier */
        .fp-card.elite {
            border-color: rgba(245, 158, 11, 0.6);
            border-left: 5px solid #F59E0B;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, var(--bg-card) 50%);
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
        }

        .fp-card.elite:hover {
            box-shadow: 0 12px 40px rgba(245, 158, 11, 0.25);
        }

        /* HIGH tier */
        .fp-card.high {
            border-color: rgba(34, 197, 94, 0.5);
            border-left: 5px solid #22C55E;
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.05) 0%, var(--bg-card) 50%);
        }

        .fp-card.high:hover {
            box-shadow: 0 12px 32px rgba(34, 197, 94, 0.2);
        }

        /* Card Header */
        .fp-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-subtle);
        }

        .fp-card-player {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .fp-card-avatar-wrapper {
            position: relative;
            flex-shrink: 0;
        }

        .fp-card-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            color: white;
            overflow: hidden;
            position: relative;
            border: 2px solid var(--border-default);
        }

        .fp-card-avatar.has-image {
            background: var(--bg-card);
        }

        .fp-card-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center top;
        }

        .fp-card-avatar .avatar-fallback {
            position: absolute;
            width: 100%;
            height: 100%;
            display: none;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
            font-weight: 700;
            font-size: 14px;
            color: white;
        }

        .fp-card-team-logo {
            position: absolute;
            bottom: -4px;
            right: -4px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--bg-card);
            border: 2px solid var(--border-default);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }

        .fp-card-team-logo img {
            width: 18px;
            height: 18px;
            object-fit: contain;
        }

        .fp-card-info {
            display: flex;
            flex-direction: column;
        }

        .fp-card-name {
            font-weight: 700;
            font-size: 15px;
            color: var(--text-primary);
        }

        .fp-card-meta {
            font-size: 12px;
            color: var(--text-muted);
        }

        .fp-card-tier {
            font-size: 10px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .fp-card-tier.elite {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            color: #000;
        }

        .fp-card-tier.high {
            background: rgba(34, 197, 94, 0.2);
            color: #22C55E;
            border: 1px solid rgba(34, 197, 94, 0.4);
        }

        /* Card Body */
        .fp-card-body {
            padding: 14px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .fp-card-prop {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .fp-card-market {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .fp-card-game {
            font-size: 11px;
            color: var(--text-muted);
        }

        .fp-card-line {
            text-align: right;
        }

        .fp-card-line-val {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .fp-card-proj {
            font-size: 12px;
            color: var(--text-muted);
        }

        .fp-card-proj .positive {
            color: #4ADE80;
        }

        .fp-card-proj .negative {
            color: #F87171;
        }

        /* Card Footer */
        .fp-card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid var(--border-subtle);
        }

        .fp-card-pick {
            font-weight: 700;
            font-size: 14px;
            padding: 6px 14px;
            border-radius: 6px;
            min-width: 80px;
            text-align: center;
        }

        .fp-card-pick.over {
            background: rgba(34, 197, 94, 0.15);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .fp-card-pick.under {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .fp-card-stats {
            display: flex;
            gap: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .fp-card-conf {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .fp-card-edge {
            font-weight: 700;
        }

        .fp-card-edge.positive {
            color: #4ADE80;
        }

        .fp-card-edge.negative {
            color: #F87171;
        }

        /* Section Divider */
        .section-divider {
            display: flex;
            align-items: center;
            margin: 32px 0 24px 0;
            gap: 16px;
        }

        .section-divider::before,
        .section-divider::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-default), transparent);
        }

        .divider-text {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            white-space: nowrap;
        }

        /* Tablet: 2 per row */
        @media (max-width: 1200px) {
            .bp-games-container {
                grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            }

            .featured-picks-grid {
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            }
        }

        /* Mobile: single column */
        @media (max-width: 720px) {
            .bp-games-container {
                grid-template-columns: 1fr;
                gap: 12px;
            }

            .featured-picks-grid {
                grid-template-columns: 1fr;
                gap: 12px;
            }

            .featured-picks-header h2 {
                font-size: 20px;
            }

            .fp-card-header {
                padding: 12px;
            }

            .fp-card-avatar {
                width: 40px;
                height: 40px;
                font-size: 12px;
            }

            .fp-card-avatar .avatar-fallback {
                font-size: 12px;
            }

            .fp-card-team-logo {
                width: 20px;
                height: 20px;
                bottom: -2px;
                right: -2px;
            }

            .fp-card-team-logo img {
                width: 14px;
                height: 14px;
            }

            .fp-card-body {
                padding: 12px;
            }

            .fp-card-line-val {
                font-size: 18px;
            }

            .fp-card-footer {
                padding: 10px 12px;
            }

            .section-divider {
                margin: 24px 0 16px 0;
            }

            .compact-header {
                padding: 12px;
            }

            .compact-team-logo {
                width: 32px;
                height: 32px;
            }

            .compact-picks {
                padding: 10px 12px;
            }
        }

        /* ===== GAME BETS ===== */
        .bp-game-bets {
            padding: 0;
        }

        /* ===== GAME ROW ===== */
        .bp-game-row {
            display: grid;
            grid-template-columns: 100px 1fr 1fr 160px 140px 80px;
            padding: 16px 24px;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            transition: background 0.2s ease;
        }

        .bp-game-row:last-child {
            border-bottom: none;
        }

        .bp-game-row:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        /* Tier-based row styling */
        .bp-game-row.tier-elite-row {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.03) 100%);
            border-left: 3px solid #F59E0B;
        }

        .bp-game-row.tier-high-row {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.02) 100%);
            border-left: 3px solid #22C55E;
        }

        /* Bet Label with tier badge */
        .bp-bet-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .bp-tier-badge {
            font-size: 9px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .bp-tier-badge.elite {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            color: #1F2937;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
        }

        .bp-tier-badge.high {
            background: linear-gradient(135deg, #22C55E, #16A34A);
            color: #1F2937;
            box-shadow: 0 0 8px rgba(34, 197, 94, 0.3);
        }

        /* ===== TEAM ROWS ===== */
        .bp-team-row {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 16px;
            border-radius: 8px;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }

        .bp-team-row.picked {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(99, 102, 241, 0.15) 100%);
            border: 2px solid rgba(59, 130, 246, 0.5);
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255,255,255,0.1);
        }

        .bp-team-row.picked::before {
            content: '✓';
            font-size: 12px;
            font-weight: 700;
            color: #60A5FA;
            margin-right: 4px;
        }

        .bp-team-abbrev {
            font-weight: 800;
            font-size: 15px;
            color: var(--text-primary);
            min-width: 40px;
        }

        .bp-spread-line, .bp-total-line {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 16px;
            color: var(--text-primary);
        }

        .bp-total-label {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            min-width: 50px;
        }

        /* ===== PICK CHIPS - LARGER & PROMINENT ===== */
        .bp-model-pick {
            display: flex;
            justify-content: center;
        }

        .bp-pick-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 15px;
            letter-spacing: 0.5px;
            min-width: 120px;
            text-transform: uppercase;
            transition: all 0.2s ease;
        }

        .bp-pick-chip.spread {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(99, 102, 241, 0.2) 100%);
            color: #A78BFA;
            border: 2px solid rgba(139, 92, 246, 0.5);
            box-shadow: 0 4px 16px rgba(139, 92, 246, 0.2);
        }

        .bp-pick-chip.over {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.25) 0%, rgba(22, 163, 74, 0.2) 100%);
            color: #4ADE80;
            border: 2px solid rgba(34, 197, 94, 0.5);
            box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2);
        }

        .bp-pick-chip.under {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.2) 100%);
            color: #F87171;
            border: 2px solid rgba(239, 68, 68, 0.5);
            box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
        }

        /* ===== CONFIDENCE METER ===== */
        .bp-confidence-meter {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .bp-conf-bar {
            width: 80px;
            height: 8px;
            background: var(--surface-elevated);
            border-radius: 4px;
            overflow: hidden;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
        }

        .bp-conf-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .bp-conf-fill.high {
            background: linear-gradient(90deg, #22C55E, #4ADE80);
            box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        }

        .bp-conf-fill.medium {
            background: linear-gradient(90deg, #F59E0B, #FBBF24);
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
        }

        .bp-conf-fill.low {
            background: linear-gradient(90deg, #EF4444, #F87171);
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
        }

        .bp-conf-label {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 14px;
            color: var(--text-primary);
            min-width: 40px;
        }

        /* ===== EDGE ===== */
        .bp-edge {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 15px;
            padding: 6px 12px;
            border-radius: 6px;
        }

        .bp-edge.ev-positive {
            background: rgba(34, 197, 94, 0.15);
            color: #4ADE80;
        }

        .bp-edge.ev-negative {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
        }

        /* ===== GAMES FOOTER ===== */
        .bp-games-footer {
            padding: 20px 0;
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 1024px) {
            .bp-game-row {
                grid-template-columns: 80px 1fr 1fr 140px 120px 70px;
                padding: 14px 20px;
            }

            .bp-pick-chip {
                min-width: 100px;
                padding: 8px 16px;
                font-size: 13px;
            }
        }

        @media (max-width: 768px) {
            .bp-game-header {
                flex-direction: column;
                gap: 16px;
                align-items: flex-start;
                padding: 16px 20px;
            }

            .bp-game-meta {
                flex-direction: row;
                width: 100%;
                justify-content: space-between;
            }

            .bp-game-row {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                padding: 16px 20px;
            }

            .bp-bet-label {
                width: 100%;
                margin-bottom: 8px;
            }

            .bp-model-pick {
                width: 100%;
                margin-top: 8px;
            }

            .bp-pick-chip {
                width: 100%;
                justify-content: center;
            }

            .bp-bet-label {
                width: 100%;
            }

            .bp-team-row {
                flex: 1;
                min-width: 120px;
            }

            .bp-model-pick,
            .bp-cover-prob,
            .bp-edge,
            .bp-rating {
                flex: 1;
            }

            /* ===== NEW CARD HEADER - Mobile ===== */
            .bp-card-header {
                flex-direction: column;
                padding: 12px 16px;
                gap: 12px;
            }

            .bp-matchup-row {
                width: 100%;
                justify-content: center;
                gap: 12px;
            }

            .bp-team-logo-wrapper {
                width: 40px;
                height: 40px;
            }

            .bp-logo-text {
                font-size: 12px;
            }

            .bp-team-names {
                font-size: 14px;
            }

            .bp-card-badges {
                justify-content: center;
            }

            .bp-weather {
                padding: 3px 8px;
                font-size: 11px;
            }

            /* ===== PICK ROW - Mobile Stack ===== */
            .bp-pick-row {
                grid-template-columns: 1fr;
                gap: 10px;
                padding: 12px 16px;
            }

            .bp-pick-type {
                justify-content: center;
            }

            .bp-pick-hero {
                justify-content: center;
            }

            .bp-pick-chip {
                padding: 12px 24px;
                font-size: 15px;
                justify-content: center;
                width: 100%;
            }

            .bp-pick-stats {
                justify-content: center;
                gap: 24px;
            }

            .bp-stat {
                min-width: 60px;
            }

            .bp-stat-value {
                font-size: 16px;
            }

            .bp-stat-label {
                font-size: 9px;
            }

            /* ===== TIER BADGE - Centered ===== */
            .bp-tier-badge {
                font-size: 8px;
                padding: 2px 5px;
            }

            .bp-card-tier-label {
                font-size: 10px;
                padding: 4px 10px;
            }
        }

        /* Print styles */
        @media print {
            body::before {
                display: none;
            }

            .container {
                box-shadow: none;
                border: 1px solid #ccc;
            }

            .featured-pick-card {
                page-break-inside: avoid;
            }
        }
    '''


def generate_javascript() -> str:
    """Generate JavaScript for interactivity."""
    return '''
        function showView(viewId, clickedTab) {
            document.querySelectorAll('.view-section').forEach(section => {
                section.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(viewId).classList.add('active');
            if (clickedTab) clickedTab.classList.add('active');
        }

        // Search and Filter state
        let currentFilter = 'all';

        function filterPicks() {
            const searchTerm = document.getElementById('player-search').value.toLowerCase();
            const cards = document.querySelectorAll('.game-pick-card, .featured-pick-card');

            cards.forEach(card => {
                const playerName = card.querySelector('.card-player-name, .featured-player-name')?.textContent?.toLowerCase() || '';
                const team = card.querySelector('.card-player-meta, .featured-player-meta')?.textContent?.toLowerCase() || '';
                const tier = card.classList.contains('tier-elite') ? 'elite' :
                             card.classList.contains('tier-strong') ? 'strong' : 'other';
                const isOver = card.querySelector('.card-hero-pick, .featured-hero-pick')?.textContent?.includes('OVER');
                const isUnder = card.querySelector('.card-hero-pick, .featured-hero-pick')?.textContent?.includes('UNDER');

                // Search match
                const matchesSearch = playerName.includes(searchTerm) || team.includes(searchTerm);

                // Filter match
                let matchesFilter = true;
                if (currentFilter === 'elite') matchesFilter = tier === 'elite';
                else if (currentFilter === 'strong') matchesFilter = tier === 'strong' || tier === 'elite';
                else if (currentFilter === 'over') matchesFilter = isOver;
                else if (currentFilter === 'under') matchesFilter = isUnder;

                card.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
            });

            // Update count display
            const visibleCount = document.querySelectorAll('.game-pick-card:not([style*="display: none"]), .featured-pick-card:not([style*="display: none"])').length;
            const countEl = document.querySelector('.picks-count');
            if (countEl) countEl.textContent = `${visibleCount} picks shown`;
        }

        function setFilter(filter, btn) {
            currentFilter = filter;

            // Update active state on chips
            document.querySelectorAll('.filter-chip').forEach(chip => {
                chip.classList.remove('active');
            });
            if (btn) btn.classList.add('active');

            filterPicks();
        }

        function toggleSection(sectionId) {
            const content = document.getElementById(sectionId);
            const icon = document.getElementById(sectionId + '-icon');
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                icon.classList.remove('collapsed');
            } else {
                content.classList.add('collapsed');
                icon.classList.add('collapsed');
            }
        }

        function toggleAllSections(viewId, expand) {
            const view = document.getElementById(viewId);
            const contents = view.querySelectorAll('.collapsible-content');
            const icons = view.querySelectorAll('.toggle-icon');
            contents.forEach(content => {
                if (expand) {
                    content.classList.remove('collapsed');
                } else {
                    content.classList.add('collapsed');
                }
            });
            icons.forEach(icon => {
                if (expand) {
                    icon.classList.remove('collapsed');
                } else {
                    icon.classList.add('collapsed');
                }
            });
        }

        function togglePlayerPicks(playerId) {
            const picksContainer = document.getElementById(playerId + '-picks');
            const icon = document.getElementById(playerId + '-icon');
            const card = picksContainer.closest('.player-summary-card');

            if (picksContainer.style.display === 'none') {
                picksContainer.style.display = 'grid';
                card.classList.add('expanded');
            } else {
                picksContainer.style.display = 'none';
                card.classList.remove('expanded');
            }
        }

        function searchPlayerCards(input) {
            const searchTerm = input.value.toLowerCase();
            const cards = document.querySelectorAll('.player-summary-card');

            cards.forEach(card => {
                const playerName = card.dataset.player || '';
                if (playerName.includes(searchTerm)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function toggleLogic(rowId) {
            const logicRow = document.getElementById('logic-' + rowId);
            if (logicRow) {
                if (logicRow.style.display === 'none') {
                    logicRow.style.display = 'table-row';
                } else {
                    logicRow.style.display = 'none';
                }
            }
        }

        // Toggle model details expandable section
        function toggleModelDetails(button) {
            button.classList.toggle('expanded');
            const content = button.nextElementSibling;
            if (content) {
                content.classList.toggle('show');
            }
        }

        // NEW: Toggle compact game card expanded/collapsed
        function toggleGameCard(gameId) {
            const gameCard = document.getElementById(gameId);
            if (gameCard) {
                gameCard.classList.toggle('expanded');
            }
        }

        // NEW: Expand or collapse all game cards
        function toggleAllGames(expand) {
            const gameCards = document.querySelectorAll('.game-card');
            gameCards.forEach(card => {
                if (expand) {
                    card.classList.add('expanded');
                } else {
                    card.classList.remove('expanded');
                }
            });
        }

        // Legacy function for backward compatibility
        function toggleGameDetails(gameId) {
            toggleGameCard(gameId);
        }

        function togglePickRow(rowId) {
            const expanded = document.getElementById('expanded-' + rowId);
            const chevron = document.getElementById('chevron-' + rowId);
            if (expanded) {
                expanded.classList.toggle('show');
                if (chevron) {
                    chevron.classList.toggle('expanded');
                }
            }
        }

        function toggleMatchupContext(rowId) {
            const details = document.getElementById('matchup-' + rowId);
            const chevron = document.getElementById('matchup-chevron-' + rowId);
            const header = chevron ? chevron.closest('.matchup-header') : null;
            if (details) {
                details.classList.toggle('show');
                if (header) {
                    header.classList.toggle('expanded');
                }
            }
        }

        function toggleTeamContext(gameId) {
            const content = document.getElementById(gameId + '-context');
            const icon = document.getElementById(gameId + '-ctx-icon');
            const toggle = icon ? icon.closest('.gl-context-toggle') : null;
            if (content) {
                content.classList.toggle('show');
                if (icon) {
                    icon.classList.toggle('expanded');
                }
                if (toggle) {
                    toggle.classList.toggle('expanded');
                }
            }
        }

        function sortTable(headerElement, columnIndex, dataType) {
            const table = headerElement.closest('table');
            const tbody = table.querySelector('tbody');
            const allRows = Array.from(tbody.querySelectorAll('tr'));
            const dataRows = allRows.filter(row => !row.classList.contains('logic-row'));

            const rowPairs = [];
            dataRows.forEach(row => {
                const nextRow = row.nextElementSibling;
                const logicRow = (nextRow && nextRow.classList.contains('logic-row')) ? nextRow : null;
                rowPairs.push({ dataRow: row, logicRow: logicRow });
            });

            const currentDirection = headerElement.dataset.sortDirection || 'desc';
            const newDirection = currentDirection === 'asc' ? 'desc' : 'asc';

            rowPairs.sort((a, b) => {
                const cellA = a.dataRow.children[columnIndex];
                const cellB = b.dataRow.children[columnIndex];
                if (!cellA || !cellB) return 0;

                let valueA = cellA.dataset.sort || cellA.textContent.trim();
                let valueB = cellB.dataset.sort || cellB.textContent.trim();

                if (dataType === 'number') {
                    valueA = parseFloat(valueA.replace(/[^0-9.-]/g, '')) || 0;
                    valueB = parseFloat(valueB.replace(/[^0-9.-]/g, '')) || 0;
                } else {
                    valueA = valueA.toLowerCase();
                    valueB = valueB.toLowerCase();
                }

                let comparison = 0;
                if (valueA < valueB) comparison = -1;
                if (valueA > valueB) comparison = 1;
                return newDirection === 'asc' ? comparison : -comparison;
            });

            // Clear existing classes
            table.querySelectorAll('th').forEach(th => {
                th.classList.remove('sorted-asc', 'sorted-desc');
            });

            headerElement.classList.add(newDirection === 'asc' ? 'sorted-asc' : 'sorted-desc');
            headerElement.dataset.sortDirection = newDirection;

            // Rebuild tbody
            tbody.innerHTML = '';
            rowPairs.forEach(pair => {
                tbody.appendChild(pair.dataRow);
                if (pair.logicRow) {
                    tbody.appendChild(pair.logicRow);
                }
            });
        }

        function filterByTier(tier, btn, viewId) {
            document.querySelectorAll('#' + viewId + ' .filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const activeView = document.getElementById(viewId);
            const rows = activeView.querySelectorAll('tbody tr.expandable-row');

            rows.forEach(row => {
                // Find tier badge by class pattern (badge-elite, badge-high, etc.)
                const tierBadge = row.querySelector('.badge[class*="badge-"]');
                if (!tierBadge) return;

                const rowTier = tierBadge.textContent.trim().toUpperCase();
                const logicRow = row.nextElementSibling;

                if (tier === 'all' || rowTier.includes(tier)) {
                    row.style.display = '';
                    if (logicRow && logicRow.classList.contains('logic-row')) {
                        // Keep logic row in sync - show if row was expanded
                        if (row.classList.contains('expanded')) {
                            logicRow.style.display = 'table-row';
                        }
                    }
                } else {
                    row.style.display = 'none';
                    if (logicRow && logicRow.classList.contains('logic-row')) {
                        logicRow.style.display = 'none';
                    }
                }
            });
        }

        function filterGameLineCards(tier, btn) {
            // Update active button state
            document.querySelectorAll('#game-lines .filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Filter cards by tier (updated selector for new gl-card class)
            const cards = document.querySelectorAll('.gl-card');
            let visibleCount = 0;
            cards.forEach(card => {
                const cardTier = card.getAttribute('data-tier');
                if (tier === 'all' || cardTier === tier) {
                    card.style.display = '';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function toggleGameAnalysis(cardId) {
            const analysis = document.getElementById(cardId + '-analysis');
            const icon = document.getElementById(cardId + '-icon');
            if (analysis && icon) {
                if (analysis.style.display === 'none') {
                    analysis.style.display = 'block';
                    icon.classList.add('expanded');
                } else {
                    analysis.style.display = 'none';
                    icon.classList.remove('expanded');
                }
            }
        }

        function expandAllGameAnalysis() {
            document.querySelectorAll('.gl-analysis').forEach(el => {
                el.style.display = 'block';
            });
            document.querySelectorAll('.gl-toggle-icon').forEach(el => {
                el.classList.add('expanded');
            });
        }

        function collapseAllGameAnalysis() {
            document.querySelectorAll('.gl-analysis').forEach(el => {
                el.style.display = 'none';
            });
            document.querySelectorAll('.gl-toggle-icon').forEach(el => {
                el.classList.remove('expanded');
            });
        }

        function searchPlayer(input, viewId) {
            const searchTerm = input.value.toLowerCase();
            const activeView = document.getElementById(viewId);
            const rows = activeView.querySelectorAll('tbody tr.expandable-row');

            rows.forEach(row => {
                const playerCell = row.querySelector('td:first-child');
                if (!playerCell) return;

                const playerName = playerCell.textContent.toLowerCase();
                const logicRow = row.nextElementSibling;

                if (playerName.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                    if (logicRow && logicRow.classList.contains('logic-row')) {
                        logicRow.style.display = 'none';
                    }
                }
            });
        }

        // Pick Breakdown Modal Functions
        function openPickModal(pickData) {
            const modal = document.getElementById('pick-modal');
            if (!modal) return;

            // Parse pick data
            const data = JSON.parse(decodeURIComponent(pickData));

            // Check for conflicts and show warning if needed
            checkAndShowConflicts(data);

            // Populate modal content
            document.getElementById('modal-player-name').textContent = data.player || 'Unknown';
            document.getElementById('modal-player-meta').textContent =
                `${data.position || '?'} | ${data.team || '?'}`;

            // Pick direction
            const pickBadge = document.getElementById('modal-pick-badge');
            pickBadge.textContent = data.pick || 'OVER';
            pickBadge.className = 'modal-pick-badge ' +
                (String(data.pick).toUpperCase().includes('OVER') ? 'pick-over' : 'pick-under');

            document.getElementById('modal-pick-line').textContent = data.line || '-';
            document.getElementById('modal-pick-market').textContent = data.market_display || data.market || '-';

            // Confidence
            const confValue = document.getElementById('modal-conf-value');
            const confPct = data.confidence ? (data.confidence * 100).toFixed(0) : '-';
            confValue.textContent = confPct + '%';
            confValue.className = 'modal-conf-value ' + getConfClass(data.confidence);

            // Stats row - dynamic labels for edge picks
            const projEl = document.getElementById('modal-projection');
            const projLabel = document.getElementById('modal-projection-label');
            const trailEl = document.getElementById('modal-trailing');
            const trailLabel = document.getElementById('modal-trailing-label');
            const pick = String(data.pick || '').toUpperCase();
            const isUnder = pick === 'UNDER';
            const playerBetCount = data.player_bet_count || 0;
            const playerUnderRate = data.player_under_rate || 0;

            // Model confidence (removed projection - backtest shows it's misleading)
            const modelConf = data.p_under || data.confidence || 0;
            if (modelConf > 0) {
                const confPctVal = modelConf <= 1 ? (modelConf * 100).toFixed(0) : modelConf.toFixed(0);
                projEl.textContent = confPctVal + '%';
                if (projLabel) projLabel.textContent = 'Model Conf';
            } else if (playerBetCount >= 5) {
                const hitRate = isUnder ? playerUnderRate : (1 - playerUnderRate);
                projEl.textContent = (hitRate * 100).toFixed(0) + '%';
                if (projLabel) projLabel.textContent = 'Hit Rate';
            } else {
                projEl.textContent = '-';
                if (projLabel) projLabel.textContent = 'Model Conf';
            }

            // Edge (always same)
            document.getElementById('modal-edge').textContent =
                data.edge ? (data.edge > 0 ? '+' : '') + data.edge.toFixed(1) + '%' : '-';

            // Trailing/Sample Size
            if (data.trailing_avg && data.trailing_avg > 0) {
                trailEl.textContent = data.trailing_avg.toFixed(1);
                if (trailLabel) trailLabel.textContent = 'Trailing Avg';
            } else if (playerBetCount >= 5) {
                trailEl.textContent = playerBetCount + ' bets';
                if (trailLabel) trailLabel.textContent = 'Sample Size';
            } else {
                trailEl.textContent = '-';
                if (trailLabel) trailLabel.textContent = 'Trailing Avg';
            }

            // Summary section
            populateSummary(data);

            // Factors section
            populateFactors(data);

            // Matchup info
            populateMatchupInfo(data);

            // Show modal
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function populateSummary(data) {
            const container = document.getElementById('modal-summary');
            if (!container) return;

            const pick = String(data.pick || '').toUpperCase();
            const isUnder = pick === 'UNDER';
            const line = data.line || 0;
            const proj = data.projection || 0;
            const trailing = data.trailing_avg || 0;
            const edge = data.edge || 0;
            const conf = data.confidence ? (data.confidence * 100) : 0;
            const histRate = data.hist_over_rate || 0.5;
            const histCount = data.hist_count || 0;
            const snapShare = data.snap_share ? (data.snap_share * 100) : 0;
            const defEpa = data.def_epa || 0;
            const lvt = data.lvt || 0;
            const gameScript = data.game_script || 0;
            const player = data.player || 'Player';
            const position = data.position || '';
            const team = data.team || '';
            const opponent = data.opponent || 'opponent';
            const market = data.market_display || data.market || '';

            // Edge model inputs
            const source = data.source || 'EDGE';
            const lvtConfidence = data.lvt_confidence || 0;
            const playerBiasConfidence = data.player_bias_confidence || 0;
            const playerUnderRate = data.player_under_rate || 0;
            const playerBetCount = data.player_bet_count || 0;

            // Volume stats (if available)
            const targets = data.targets_mean || 0;
            const receptions = data.receptions_mean || 0;
            const recYards = data.receiving_yards_mean || 0;
            const rushAtt = data.rushing_attempts_mean || 0;
            const rushYards = data.rushing_yards_mean || 0;
            const passAtt = data.passing_attempts_mean || 0;
            const passYards = data.passing_yards_mean || 0;
            const rzTargetShare = data.redzone_target_share || 0;
            const glCarryShare = data.goalline_carry_share || 0;

            // Calculate derived metrics
            const projDiff = proj - line;
            const trailDiff = trailing - line;
            const hitRate = isUnder ? (1 - histRate) : histRate;
            const pickClass = pick === 'OVER' ? 'pick-over' : 'pick-under';
            const pickDirClass = pick === 'OVER' ? 'over' : 'under';
            const pickIcon = pick === 'OVER' ? '▲' : '▼';

            // Determine market type
            const isReceiving = market.toLowerCase().includes('recep') || market.toLowerCase().includes('receiving');
            const isRushing = market.toLowerCase().includes('rush');
            const isPassing = market.toLowerCase().includes('pass');

            // Determine conviction tier
            let convictionClass, convictionLabel;
            if (conf >= 70) {
                convictionClass = 'conviction-high';
                convictionLabel = 'high-conviction';
            } else if (conf >= 60) {
                convictionClass = 'conviction-medium';
                convictionLabel = 'solid';
            } else {
                convictionClass = 'conviction-low';
                convictionLabel = 'moderate';
            }

            // Edge tier
            let edgeDesc;
            const absEdge = Math.abs(edge);
            if (absEdge >= 15) edgeDesc = 'substantial';
            else if (absEdge >= 8) edgeDesc = 'meaningful';
            else if (absEdge >= 5) edgeDesc = 'playable';
            else edgeDesc = 'thin';

            // Defense matchup context
            let defContext;
            if (defEpa > 0.05) defContext = { tier: 'bottom-tier', sentiment: 'favorable' };
            else if (defEpa > 0.02) defContext = { tier: 'below-average', sentiment: 'soft' };
            else if (defEpa < -0.08) defContext = { tier: 'elite', sentiment: 'tough' };
            else if (defEpa < -0.04) defContext = { tier: 'above-average', sentiment: 'challenging' };
            else defContext = { tier: 'league-average', sentiment: 'neutral' };

            // Build formatted HTML
            let html = '';

            // === PROBABILITY METER ===
            html += `
            <div class="prob-meter-container">
                <div class="prob-meter-header">
                    <span class="prob-meter-label">Cover Probability</span>
                    <span class="prob-meter-value ${pickDirClass}">${conf.toFixed(0)}%</span>
                </div>
                <div class="prob-meter">
                    <div class="prob-fill ${pickDirClass}" style="width: ${Math.min(conf, 100)}%"></div>
                </div>
            </div>`;

            // === STAR RATING ===
            let starCount = 0;
            let ratingLabel = '';
            let ratingClass = '';
            if (conf >= 75) { starCount = 5; ratingLabel = 'ELITE'; ratingClass = ''; }
            else if (conf >= 68) { starCount = 4; ratingLabel = 'STRONG'; ratingClass = ''; }
            else if (conf >= 60) { starCount = 3; ratingLabel = 'SOLID'; ratingClass = ''; }
            else if (conf >= 55) { starCount = 2; ratingLabel = 'MODERATE'; ratingClass = 'caution'; }
            else { starCount = 1; ratingLabel = 'SPECULATIVE'; ratingClass = 'caution'; }

            let starsHtml = '';
            for (let i = 0; i < 5; i++) {
                starsHtml += `<span class="star ${i < starCount ? 'filled' : 'empty'}">★</span>`;
            }

            html += `
            <div class="star-rating-container">
                <div class="star-rating">${starsHtml}</div>
                <span class="rating-badge ${ratingClass}">${ratingLabel}</span>
            </div>`;

            // === EDGE SOURCE BADGE ===
            let sourceLabel = 'Edge Model';
            let sourceClass = 'source-default';
            if (source.includes('PLAYER_BIAS')) {
                sourceLabel = 'Player Bias Edge';
                sourceClass = 'source-bias';
            } else if (source.includes('LVT')) {
                sourceLabel = 'LVT Edge';
                sourceClass = 'source-lvt';
            } else if (source.includes('TD_POISSON') || source.includes('POISSON')) {
                sourceLabel = 'TD Poisson';
                sourceClass = 'source-poisson';
            } else if (source.includes('GAME_LINE')) {
                sourceLabel = 'Game Line';
                sourceClass = 'source-gameline';
            }

            html += `
            <div class="edge-source-container">
                <span class="edge-source-label">Edge Source:</span>
                <span class="edge-source-badge ${sourceClass}">${sourceLabel}</span>
                ${playerBetCount >= 10 ? `<span class="edge-sample">(${playerBetCount} historical bets)</span>` : ''}
            </div>`;

            // === GAME CONTEXT: WEATHER + DEFENSE ===
            const weatherTemp = data.weather_temp || 0;
            const weatherWind = data.weather_wind || 0;
            const isDome = data.weather_is_dome || false;
            const weatherConditions = data.weather_conditions || '';
            const weatherSeverity = data.weather_severity || 'None';
            const weatherPassingAdj = data.weather_passing_adj || 0;
            const defRank = data.def_rank;
            const defRankType = data.def_rank_type || 'pass';
            const hasDefRank = defRank !== null && defRank !== undefined && defRank > 0;

            // Handle both old format ('pass'/'rush') and new format ('vs WR1s', 'vs RBs')
            let defTypeLabel;
            if (defRankType.startsWith('vs ')) {
                // New format - extract the position part (e.g., "vs WR1s" -> "WR1s")
                defTypeLabel = defRankType.substring(3);
            } else {
                // Old format fallback
                defTypeLabel = defRankType === 'rush' ? 'RBs' : 'WRs';
            }
            const defClass = hasDefRank ? (defRank >= 24 ? 'positive' : defRank <= 8 ? 'negative' : '') : '';

            html += `<div class="game-context-row">`;

            // Weather widget - use severity for styling
            if (isDome) {
                html += `<span class="context-chip weather-dome">🏟️ Dome</span>`;
            } else if (weatherTemp > 0) {
                const severityClass = weatherSeverity === 'Extreme' ? 'weather-extreme' :
                                      weatherSeverity === 'High' ? 'weather-high' : 'weather-outdoor';
                let weatherText = weatherTemp + '°F, ' + weatherWind + 'mph';
                if (weatherConditions) weatherText += ' - ' + weatherConditions;
                if (weatherPassingAdj < -0.05) {
                    weatherText += ' (' + Math.round(weatherPassingAdj * 100) + '% passing)';
                }
                html += `<span class="context-chip ${severityClass}">${weatherText}</span>`;
            }

            // Defensive ranking
            if (opponent) {
                const rankDisplay = hasDefRank ? `#${defRank}` : 'N/A';
                html += `<span class="context-chip def-rank ${defClass}">${opponent} ${rankDisplay} vs ${defTypeLabel}</span>`;
            }

            html += `</div>`;

            // === SEASON BIAS COMPARISON ===
            const currentSeasonUnderRate = data.current_season_under_rate || 0.5;
            const seasonGamesPlayed = data.season_games_played || 0;

            // Only show if player has at least 3 games this season and meaningful player bias data
            if (seasonGamesPlayed >= 3 && playerBetCount >= 5) {
                // Calculate hit rate in the direction of the bet
                const careerHitRate = isUnder ? playerUnderRate : (1 - playerUnderRate);
                const seasonHitRate = isUnder ? currentSeasonUnderRate : (1 - currentSeasonUnderRate);
                const trendDiff = seasonHitRate - careerHitRate;
                const trendClass = trendDiff > 0.05 ? 'improving' : trendDiff < -0.05 ? 'declining' : '';
                const trendText = trendDiff > 0.05 ? '↑ trending up' : trendDiff < -0.05 ? '↓ trending down' : '';

                html += `
                <div class="bias-comparison">
                    <div class="bias-item">
                        <span class="bias-label">Career ${pick}</span>
                        <span class="bias-value ${careerHitRate >= 0.55 ? 'positive' : careerHitRate < 0.45 ? 'negative' : ''}">${(careerHitRate * 100).toFixed(0)}%</span>
                        <span class="bias-trend">${playerBetCount} bets</span>
                    </div>
                    <div class="bias-item">
                        <span class="bias-label">2025 Season</span>
                        <span class="bias-value ${seasonHitRate >= 0.55 ? 'positive' : seasonHitRate < 0.45 ? 'negative' : ''}">${(seasonHitRate * 100).toFixed(0)}%</span>
                        <span class="bias-trend ${trendClass}">${seasonGamesPlayed} games ${trendText}</span>
                    </div>
                </div>`;
            }

            // === CONCISE NARRATIVE (2-3 paragraphs, no repetition) ===
            html += `<div class="unified-narrative ${pickClass}">`;

            // --- PARAGRAPH 1: The Edge (WHY we're betting) ---
            let para1 = '<p class="narrative-lead">';
            if (source.includes('PLAYER_BIAS') && playerBetCount >= 5) {
                const biasRate = isUnder ? playerUnderRate : (1 - playerUnderRate);
                para1 += `<strong>${player}</strong> has hit ${pick} on this market in <span class="stat-pct">${(biasRate * 100).toFixed(0)}%</span> of <span class="stat">${playerBetCount}</span> tracked bets.`;
                if (lvt !== 0) {
                    para1 += ` Today's ${line} line is <span class="stat">${Math.abs(lvt).toFixed(1)}</span> points ${lvt > 0 ? 'above' : 'below'} his trailing average.`;
                }
            } else if (source.includes('LVT') && lvt !== 0) {
                para1 += `The ${line} line sits <span class="stat-edge">${Math.abs(lvt).toFixed(1)}</span> points ${lvt > 0 ? 'above' : 'below'} <strong>${player}</strong>'s recent production.`;
            } else if (source.includes('TD_POISSON')) {
                const expectedTds = data.expected_tds || 0;
                const tdProb = isUnder ? data.p_under : data.p_over;
                para1 += `<strong>${player}</strong> projects for <span class="stat">${expectedTds.toFixed(2)}</span> TDs with <span class="stat-pct">${(tdProb * 100).toFixed(0)}%</span> probability on ${pick}.`;
            } else if (trailing > 0) {
                para1 += `<strong>${player}</strong>'s trailing average of <span class="stat">${trailing.toFixed(1)}</span> is <span class="stat-edge">${Math.abs(trailDiff).toFixed(1)}</span> points ${trailDiff > 0 ? 'above' : 'below'} the ${line} line.`;
            } else if (proj > 0) {
                para1 += `Model projects <strong>${player}</strong> at <span class="stat">${proj.toFixed(1)}</span> — <span class="stat-edge">${projDiff > 0 ? '+' : ''}${projDiff.toFixed(1)}</span> vs the ${line} line.`;
            } else {
                para1 += `<strong>${player}</strong> shows value on ${pick} ${line} ${market}.`;
            }
            para1 += '</p>';
            html += para1;

            // --- PARAGRAPH 2: Matchup Context (ONLY if meaningful) ---
            let para2 = '<p>';
            let hasMatchupContent = false;

            // Defense context (only if top/bottom 8)
            if (opponent && (defRank <= 8 || defRank >= 24)) {
                // Use position-specific label if available, else generic
                let defTypeLabel2;
                if (defRankType.startsWith('vs ')) {
                    // Position-specific: "vs WR1s" -> "WR1s"
                    defTypeLabel2 = defRankType.substring(3);
                } else {
                    defTypeLabel2 = defRankType === 'rush' ? 'rushing' : 'passing';
                }
                if (defRank >= 24) {
                    para2 += `${opponent} ranks <span class="stat-positive">#${defRank}</span> vs ${defTypeLabel2} — favorable matchup. `;
                } else {
                    para2 += `${opponent} ranks <span class="stat-negative">#${defRank}</span> vs ${defTypeLabel2} — tough draw. `;
                }
                hasMatchupContent = true;
            }

            // Game script (only if meaningful)
            if (gameScript !== 0 && Math.abs(gameScript) > 0.1) {
                if (gameScript < -0.15 && (isReceiving || isPassing)) {
                    para2 += `Projected trailing script boosts passing volume. `;
                    hasMatchupContent = true;
                } else if (gameScript > 0.15 && isRushing) {
                    para2 += `Expected lead favors clock-killing carries. `;
                    hasMatchupContent = true;
                }
            }

            // Season trend (only if different from career by >8%)
            if (seasonGamesPlayed >= 3 && playerBetCount >= 5) {
                const careerRate = isUnder ? playerUnderRate : (1 - playerUnderRate);
                const seasonRate = isUnder ? currentSeasonUnderRate : (1 - currentSeasonUnderRate);
                if (Math.abs(seasonRate - careerRate) > 0.08) {
                    para2 += `<span class="${seasonRate > careerRate ? 'stat-positive' : 'stat-negative'}">${seasonRate > careerRate ? 'Trending up' : 'Trending down'}</span> in 2025 (${(seasonRate * 100).toFixed(0)}% vs ${(careerRate * 100).toFixed(0)}% career). `;
                    hasMatchupContent = true;
                }
            }

            para2 += '</p>';
            if (hasMatchupContent) html += para2;

            // --- PARAGRAPH 3: Verdict (confidence + edge only) ---
            let para3 = `<p><span class="${convictionClass}">${convictionLabel.charAt(0).toUpperCase() + convictionLabel.slice(1)}</span> play at <span class="stat-pct">${conf.toFixed(0)}%</span> confidence`;
            if (Math.abs(edge) >= 5) {
                para3 += ` with <span class="stat-edge ${edge > 0 ? 'positive' : 'negative'}">${edge > 0 ? '+' : ''}${edge.toFixed(1)}%</span> edge`;
            }
            para3 += '.</p>';
            html += para3;

            html += '</div>'; // Close unified-narrative

            // === KEY FACTORS LIST (BettingPros Style) ===
            html += `<div class="factors-list">
                <div class="factors-list-title">Key Factors</div>`;

            // Player Bias factor (show first when that's the source)
            if (playerBetCount >= 5) {
                const biasRate = isUnder ? playerUnderRate : (1 - playerUnderRate);
                const biasSupports = biasRate >= 0.55;
                html += `
                <div class="factor-item">
                    <span class="factor-icon ${biasSupports ? 'supporting' : biasRate >= 0.45 ? 'neutral' : 'against'}">${biasSupports ? '✓' : biasRate >= 0.45 ? '~' : '✗'}</span>
                    <span class="factor-label">Player Bias (${playerBetCount} bets)</span>
                    <span class="factor-value ${biasSupports ? 'positive' : 'negative'}">${(biasRate * 100).toFixed(0)}% ${pick}</span>
                </div>`;
            }

            // Model Projection factor (only show if projection exists)
            if (proj > 0) {
                const projSupports = (pick === 'OVER' && projDiff > 0) || (pick === 'UNDER' && projDiff < 0);
                html += `
                <div class="factor-item">
                    <span class="factor-icon ${projSupports ? 'supporting' : 'against'}">${projSupports ? '✓' : '✗'}</span>
                    <span class="factor-label">Model Projection vs Line</span>
                    <span class="factor-value ${projSupports ? 'positive' : 'negative'}">${projDiff > 0 ? '+' : ''}${projDiff.toFixed(1)}</span>
                </div>`;
            }

            // Trailing average factor
            if (trailing > 0) {
                const trailSupports = (pick === 'OVER' && trailing > line) || (pick === 'UNDER' && trailing < line);
                html += `
                <div class="factor-item">
                    <span class="factor-icon ${trailSupports ? 'supporting' : 'against'}">${trailSupports ? '✓' : '✗'}</span>
                    <span class="factor-label">4-Game Average</span>
                    <span class="factor-value">${trailing.toFixed(1)}</span>
                </div>`;
            }

            // Historical hit rate factor - show pick-direction hit rate with sample size (different from player bias)
            if (histCount >= 3 && playerBetCount < 5) {
                const histSupports = hitRate >= 0.50;
                html += `
                <div class="factor-item">
                    <span class="factor-icon ${histSupports ? 'supporting' : 'against'}">${histSupports ? '✓' : '✗'}</span>
                    <span class="factor-label">${pick} Hit Rate (${histCount} games)</span>
                    <span class="factor-value ${histSupports ? 'positive' : 'negative'}">${(hitRate * 100).toFixed(0)}%</span>
                </div>`;
            }

            // Edge factor
            const edgeSupports = edge > 5;
            html += `
                <div class="factor-item">
                    <span class="factor-icon ${edgeSupports ? 'supporting' : edge > 0 ? 'neutral' : 'against'}">${edgeSupports ? '✓' : edge > 0 ? '~' : '✗'}</span>
                    <span class="factor-label">Model Edge</span>
                    <span class="factor-value ${edge > 0 ? 'positive' : 'negative'}">${edge > 0 ? '+' : ''}${edge.toFixed(1)}%</span>
                </div>`;

            // Defense matchup factor
            if (opponent && defEpa !== 0) {
                const defFavorable = (isReceiving && defEpa > 0) || (isRushing && defEpa > 0);
                html += `
                <div class="factor-item">
                    <span class="factor-icon ${defFavorable ? 'supporting' : defEpa === 0 ? 'neutral' : 'against'}">${defFavorable ? '✓' : defEpa === 0 ? '~' : '✗'}</span>
                    <span class="factor-label">Matchup (${opponent} DEF)</span>
                    <span class="factor-value">${defContext.tier}</span>
                </div>`;
            }

            // Snap share factor (if available)
            if (snapShare > 0) {
                const snapSupports = snapShare >= 65;
                html += `
                <div class="factor-item">
                    <span class="factor-icon ${snapSupports ? 'supporting' : snapShare >= 50 ? 'neutral' : 'against'}">${snapSupports ? '✓' : snapShare >= 50 ? '~' : '✗'}</span>
                    <span class="factor-label">Snap Share</span>
                    <span class="factor-value">${snapShare.toFixed(0)}%</span>
                </div>`;
            }

            html += `</div>`; // Close factors-list

            // === EDGE MODEL DETAILS EXPANDABLE SECTION ===
            const game_script_val = data.game_script || 0;

            html += `
            <div class="model-details-section">
                <button class="model-details-toggle" onclick="toggleModelDetails(this)">
                    <span>Edge Model Details</span>
                    <span class="chevron">▼</span>
                </button>
                <div class="model-details-content">
                    <div class="detail-grid">`;

            // Edge Source section
            html += `
                        <div class="detail-group">
                            <h5>Edge Source</h5>
                            <div class="detail-row"><span>Model</span><span class="mono">${sourceLabel}</span></div>
                            <div class="detail-row"><span>Combined Conf</span><span class="mono ${conf >= 70 ? 'positive' : conf >= 60 ? '' : 'negative'}">${conf.toFixed(1)}%</span></div>`;
            if (lvtConfidence > 0) html += `<div class="detail-row"><span>LVT Conf</span><span class="mono">${(lvtConfidence * 100).toFixed(1)}%</span></div>`;
            if (playerBiasConfidence > 0) html += `<div class="detail-row"><span>Bias Conf</span><span class="mono">${(playerBiasConfidence * 100).toFixed(1)}%</span></div>`;
            html += `</div>`;

            // Player Bias section (if applicable)
            if (playerBetCount >= 5) {
                const biasDirection = playerUnderRate > 0.5 ? 'UNDER' : 'OVER';
                const biasStrength = Math.abs(playerUnderRate - 0.5) * 200;  // 0-100 scale
                html += `
                        <div class="detail-group">
                            <h5>Player Bias</h5>
                            <div class="detail-row"><span>Historical Bets</span><span class="mono">${playerBetCount}</span></div>
                            <div class="detail-row"><span>Under Rate</span><span class="mono">${(playerUnderRate * 100).toFixed(1)}%</span></div>
                            <div class="detail-row"><span>Over Rate</span><span class="mono">${((1 - playerUnderRate) * 100).toFixed(1)}%</span></div>
                            <div class="detail-row"><span>Bias Direction</span><span class="mono ${biasDirection === pick ? 'positive' : 'negative'}">${biasDirection}</span></div>
                        </div>`;
            }

            // Line vs Trailing section
            html += `
                        <div class="detail-group">
                            <h5>Line vs Performance</h5>
                            <div class="detail-row"><span>Market Line</span><span class="mono">${line}</span></div>`;
            if (trailing > 0) html += `<div class="detail-row"><span>4-Game Avg</span><span class="mono">${trailing.toFixed(1)}</span></div>`;
            if (lvt !== 0) html += `<div class="detail-row"><span>LVT Gap</span><span class="mono ${lvt > 0 ? 'positive' : 'negative'}">${lvt > 0 ? '+' : ''}${lvt.toFixed(1)}</span></div>`;
            if (game_script_val !== 0) html += `<div class="detail-row"><span>Game Script</span><span class="mono ${game_script_val > 0 ? 'positive' : 'negative'}">${game_script_val > 0 ? '+' : ''}${game_script_val.toFixed(2)}</span></div>`;
            html += `</div>`;

            html += `
                    </div>
                </div>
            </div>`;

            // === GAME HISTORY CHART ===
            html += renderGameHistoryChart(data);

            container.innerHTML = html;
        }

        function renderGameHistoryChart(data) {
            const gameHistory = data.game_history || {};
            const weeks = gameHistory.weeks || [];
            const opponents = gameHistory.opponents || [];
            const line = data.line || 0;
            const projection = data.projection || 0;
            const pick = String(data.pick || '').toUpperCase();
            const market = (data.market || '').toLowerCase();

            // Determine which stat to show based on market
            // NOTE: TD checks must come before generic pass/rush checks
            let values = [];
            let statLabel = '';
            if (market.includes('anytime_td') || market.includes('1st_td') || market.includes('last_td')) {
                // Anytime TD = sum of rushing + receiving TDs
                const rushTds = gameHistory.rushing_tds || [];
                const recTds = gameHistory.receiving_tds || [];
                values = [];
                for (let i = 0; i < Math.max(rushTds.length, recTds.length); i++) {
                    values.push((rushTds[i] || 0) + (recTds[i] || 0));
                }
                statLabel = 'Total TDs';
            } else if (market.includes('pass_tds')) {
                values = gameHistory.passing_tds || [];
                statLabel = 'Pass TDs';
            } else if (market.includes('rush_tds')) {
                values = gameHistory.rushing_tds || [];
                statLabel = 'Rush TDs';
            } else if (market.includes('rec_tds') || market.includes('receiving_tds')) {
                values = gameHistory.receiving_tds || [];
                statLabel = 'Rec TDs';
            } else if (market.includes('reception_yds') || market.includes('receiving_yards')) {
                values = gameHistory.receiving_yards || [];
                statLabel = 'Receiving Yards';
            } else if (market.includes('receptions')) {
                values = gameHistory.receptions || [];
                statLabel = 'Receptions';
            } else if (market.includes('rush_yds') || market.includes('rushing_yards')) {
                values = gameHistory.rushing_yards || [];
                statLabel = 'Rushing Yards';
            } else if (market.includes('rush_att') || market.includes('rushing_attempts')) {
                values = gameHistory.rushing_attempts || [];
                statLabel = 'Rush Attempts';
            } else if (market.includes('pass_yds') || market.includes('passing_yards')) {
                values = gameHistory.passing_yards || [];
                statLabel = 'Passing Yards';
            } else if (market.includes('completions')) {
                values = gameHistory.completions || [];
                statLabel = 'Completions';
            } else if (market.includes('pass_att') || market.includes('pass_attempts')) {
                values = gameHistory.passing_attempts || [];
                statLabel = 'Pass Attempts';
            } else {
                // Default to receptions for generic
                values = gameHistory.receptions || [];
                statLabel = 'Stat';
            }

            // If no data, show placeholder
            if (values.length === 0) {
                return `
                <div class="game-history-chart">
                    <div class="game-history-chart-title">Last 6 Games</div>
                    <div class="game-history-no-data">No recent game data available</div>
                </div>`;
            }

            // Calculate max for scaling (include line and projection)
            const allVals = [...values, line, projection].filter(v => v > 0);
            const maxVal = Math.max(...allVals) * 1.15; // 15% padding

            // Calculate line and projection positions (as percentage from bottom)
            const linePct = line > 0 ? ((line / maxVal) * 100) : 0;
            const projPct = projection > 0 ? ((projection / maxVal) * 100) : 0;

            // Build bars HTML
            let barsHtml = '';
            for (let i = 0; i < values.length && i < 6; i++) {
                const val = values[i] || 0;
                const wk = weeks[i] || '';
                const opp = opponents[i] || '';
                const heightPct = maxVal > 0 ? ((val / maxVal) * 100) : 0;

                // Determine if this bar hit the current line
                const hitOver = val > line;
                const hitUnder = val < line;
                let barClass = 'miss';
                if (pick === 'OVER' && hitOver) barClass = 'hit-over';
                else if (pick === 'UNDER' && hitUnder) barClass = 'hit-under';
                else if (pick === 'OVER' && !hitOver) barClass = 'miss';
                else if (pick === 'UNDER' && !hitUnder) barClass = 'miss';

                barsHtml += `
                <div class="game-history-bar-group">
                    <div class="game-history-bar-container">
                        <div class="game-history-bar ${barClass}" style="height: ${heightPct}%;">
                            <span class="game-history-bar-value">${val}</span>
                        </div>
                    </div>
                    <span class="game-history-bar-label">Wk${wk}${opp ? ' @' + opp.substring(0, 3).toUpperCase() : ''}</span>
                </div>`;
            }

            // Determine if labels would overlap (within 15% of chart height)
            const labelOverlap = Math.abs(linePct - projPct) < 15 && Math.abs(projection - line) > 1;

            // Combined label block when values are close
            const combinedLabelsHtml = labelOverlap ? `
                <div class="game-history-combined-labels" style="bottom: ${Math.min(linePct, projPct)}%;">
                    <div class="label-stack">
                        <span class="label-line">${line} <small>LINE</small></span>
                    </div>
                </div>` : '';

            // Projection line removed - backtest shows projection is misleading
            // XGBoost classifier is the real signal, not MC projection
            const projLineHtml = '';

            // Line marker (hide left label if using combined labels)
            const lineLabelsHtml = labelOverlap ? '' : `
                        <span class="game-history-line-label-left">${line}</span>
                        <span class="game-history-line-label-right">LINE</span>`;

            return `
            <div class="game-history-chart">
                <div class="game-history-chart-title">Last 6 Games: ${statLabel}</div>
                <div class="game-history-bars">
                    ${barsHtml}
                    <div class="game-history-line" style="bottom: ${linePct}%;">
                        ${lineLabelsHtml}
                    </div>
                    ${projLineHtml}
                    ${combinedLabelsHtml}
                </div>
                <div class="game-history-legend">
                    <div class="game-history-legend-item">
                        <div class="game-history-legend-swatch hit-over"></div>
                        <span>Hit OVER</span>
                    </div>
                    <div class="game-history-legend-item">
                        <div class="game-history-legend-swatch hit-under"></div>
                        <span>Hit UNDER</span>
                    </div>
                    <div class="game-history-legend-item">
                        <div class="game-history-legend-swatch line"></div>
                        <span>Current Line</span>
                    </div>
                    <div class="game-history-legend-item">
                        <div class="game-history-legend-swatch projection"></div>
                        <span>Projection</span>
                    </div>
                </div>
            </div>`;
        }

        function checkAndShowConflicts(data) {
            const warningContainer = document.getElementById('modal-conflict-warning');
            if (!warningContainer) return;

            const pick = String(data.pick || '').toUpperCase();
            const isUnder = pick === 'UNDER';
            const conf = data.confidence ? (data.confidence * 100) : 0;
            const histRate = data.hist_over_rate || 0.5;
            const histCount = data.hist_count || 0;
            const hitRate = isUnder ? (1 - histRate) : histRate;

            let conflicts = [];

            // Check model confidence vs historical hit rate conflict
            if (histCount >= 3 && conf >= 60) {
                // High confidence pick but low historical hit rate
                if (hitRate <= 0.35) {
                    conflicts.push({
                        title: 'Model vs History Conflict',
                        detail: `Model shows ${conf.toFixed(0)}% confidence, but this prop only hits ${pick} ${(hitRate * 100).toFixed(0)}% of the time historically (${histCount} games). The model may be overvaluing current form.`
                    });
                }
            }

            // Check trailing vs projection conflict
            if (data.trailing_avg && data.projection && data.line) {
                const trailSupportsUnder = data.trailing_avg < data.line;
                const projSupportsUnder = data.projection < data.line;
                if (trailSupportsUnder !== projSupportsUnder) {
                    const trailSide = trailSupportsUnder ? 'UNDER' : 'OVER';
                    const projSide = projSupportsUnder ? 'UNDER' : 'OVER';
                    conflicts.push({
                        title: 'Projection vs Trailing Conflict',
                        detail: `Recent average (${data.trailing_avg.toFixed(1)}) suggests ${trailSide}, but model projects ${data.projection.toFixed(1)} suggesting ${projSide}.`
                    });
                }
            }

            // Render conflicts
            if (conflicts.length > 0) {
                warningContainer.innerHTML = conflicts.map(c => `
                    <div class="conflict-warning">
                        <span class="conflict-warning-icon">⚠️</span>
                        <div class="conflict-warning-text">
                            <div class="conflict-warning-title">${c.title}</div>
                            <div class="conflict-warning-detail">${c.detail}</div>
                        </div>
                    </div>
                `).join('');
                warningContainer.style.display = 'block';
            } else {
                warningContainer.innerHTML = '';
                warningContainer.style.display = 'none';
            }
        }

        function closePickModal() {
            const modal = document.getElementById('pick-modal');
            if (modal) {
                modal.classList.remove('active');
                document.body.style.overflow = '';
            }
        }

        // Centralized thresholds (synced with Python constants)
        const CONF_THRESHOLD_ELITE = ''' + str(CONF_THRESHOLD_ELITE) + ''';
        const CONF_THRESHOLD_HIGH = ''' + str(CONF_THRESHOLD_HIGH) + ''';

        function getConfClass(conf) {
            if (!conf) return 'conf-standard';
            const pct = conf * 100;
            if (pct >= CONF_THRESHOLD_ELITE) return 'conf-elite';
            if (pct >= CONF_THRESHOLD_HIGH) return 'conf-high';
            return 'conf-standard';
        }

        function populateFactors(data) {
            const supportingContainer = document.getElementById('modal-supporting-factors');
            const opposingContainer = document.getElementById('modal-opposing-factors');

            if (!supportingContainer || !opposingContainer) return;

            // Clear existing
            supportingContainer.innerHTML = '';
            opposingContainer.innerHTML = '';

            // Build supporting factors
            const supporting = [];
            const opposing = [];

            const pickIsUnder = String(data.pick).toUpperCase() === 'UNDER';

            // Projection vs Line
            if (data.projection && data.line) {
                const projVsLine = data.projection - data.line;
                const projSupportsUnder = projVsLine < 0;
                const factor = {
                    name: 'Model Projection',
                    value: data.projection.toFixed(1),
                    supports: projSupportsUnder === pickIsUnder
                };
                (factor.supports ? supporting : opposing).push(factor);
            }

            // Trailing avg vs Line
            if (data.trailing_avg && data.line) {
                const trailVsLine = data.trailing_avg - data.line;
                const trailSupportsUnder = trailVsLine < 0;
                const factor = {
                    name: 'Trailing Avg',
                    value: data.trailing_avg.toFixed(1),
                    supports: trailSupportsUnder === pickIsUnder
                };
                (factor.supports ? supporting : opposing).push(factor);
            }

            // Historical rate
            if (data.hist_over_rate !== undefined && data.hist_count >= 3) {
                const hitRate = pickIsUnder ? (1 - data.hist_over_rate) : data.hist_over_rate;
                const factor = {
                    name: `Historical (${data.hist_count} games)`,
                    value: (hitRate * 100).toFixed(0) + '% hit',
                    supports: hitRate > 0.5
                };
                (factor.supports ? supporting : opposing).push(factor);
            }

            // Defense matchup
            if (data.def_epa !== undefined) {
                const weakDef = data.def_epa > 0.02;
                const strongDef = data.def_epa < -0.02;
                if (weakDef || strongDef) {
                    const defSupportsUnder = strongDef;
                    const factor = {
                        name: 'Defense Matchup',
                        value: weakDef ? 'Weak DEF' : 'Strong DEF',
                        supports: defSupportsUnder === pickIsUnder
                    };
                    (factor.supports ? supporting : opposing).push(factor);
                }
            }

            // Snap share (usage indicator)
            if (data.snap_share && data.snap_share > 0) {
                const snapPct = data.snap_share * 100;
                const highUsage = snapPct >= 70;
                const lowUsage = snapPct < 50;
                if (highUsage || lowUsage) {
                    // High usage supports OVER, low usage supports UNDER
                    const factor = {
                        name: 'Snap Share',
                        value: snapPct.toFixed(0) + '%',
                        supports: highUsage !== pickIsUnder
                    };
                    (factor.supports ? supporting : opposing).push(factor);
                }
            }

            // LVT (Line vs Trailing) signal
            if (data.lvt !== undefined && Math.abs(data.lvt) > 0.5) {
                // Positive LVT = line above trailing = supports UNDER
                const lvtSupportsUnder = data.lvt > 0;
                const factor = {
                    name: 'LVT Signal',
                    value: (data.lvt > 0 ? '+' : '') + data.lvt.toFixed(1),
                    supports: lvtSupportsUnder === pickIsUnder
                };
                (factor.supports ? supporting : opposing).push(factor);
            }

            // Player Bias (historical under/over tendency)
            if (data.player_under_rate && data.player_bet_count >= 10) {
                // Player bias shows tendency to go under
                const underRate = data.player_under_rate;
                const biasSupportsUnder = underRate > 0.55;
                const biasSupportsOver = underRate < 0.45;
                if (biasSupportsUnder || biasSupportsOver) {
                    const factor = {
                        name: `Player Bias (${data.player_bet_count} bets)`,
                        value: (underRate * 100).toFixed(0) + '% under',
                        supports: biasSupportsUnder === pickIsUnder
                    };
                    (factor.supports ? supporting : opposing).push(factor);
                }
            }

            // Game script (if significant)
            if (data.game_script && Math.abs(data.game_script) > 1) {
                // Positive game script = team favored = more volume expected
                const factor = {
                    name: 'Game Script',
                    value: data.game_script > 0 ? 'Favored' : 'Underdog',
                    supports: (data.game_script > 0) !== pickIsUnder
                };
                (factor.supports ? supporting : opposing).push(factor);
            }

            // Render factors
            if (supporting.length === 0) {
                supportingContainer.innerHTML = '<div class="modal-factor-item"><span class="modal-factor-name">No supporting factors found</span></div>';
            } else {
                supporting.forEach(f => {
                    supportingContainer.innerHTML += `
                        <div class="modal-factor-item">
                            <span class="modal-factor-name">${f.name}</span>
                            <span class="modal-factor-value positive">${f.value}</span>
                        </div>
                    `;
                });
            }

            if (opposing.length === 0) {
                opposingContainer.innerHTML = '<div class="modal-factor-item"><span class="modal-factor-name">No opposing factors found</span></div>';
            } else {
                opposing.forEach(f => {
                    opposingContainer.innerHTML += `
                        <div class="modal-factor-item">
                            <span class="modal-factor-name">${f.name}</span>
                            <span class="modal-factor-value negative">${f.value}</span>
                        </div>
                    `;
                });
            }
        }

        function populateMatchupInfo(data) {
            const container = document.getElementById('modal-matchup-info');
            if (!container) return;

            container.innerHTML = '';

            const items = [];

            // Game
            if (data.game) {
                items.push({ label: 'Matchup', value: data.game });
            }

            // Snap share with context
            if (data.snap_share) {
                const snapPct = (data.snap_share * 100).toFixed(0);
                let snapLabel = snapPct + '%';
                if (snapPct >= 70) snapLabel += ' (starter)';
                else if (snapPct >= 50) snapLabel += ' (rotation)';
                else snapLabel += ' (limited)';
                items.push({ label: 'Snap Share', value: snapLabel });
            }

            // Defense matchup
            if (data.def_epa !== undefined && data.def_epa !== 0) {
                let defValue = '';
                if (data.def_epa > 0.05) defValue = 'Very Weak (plus matchup)';
                else if (data.def_epa > 0.02) defValue = 'Below Average';
                else if (data.def_epa < -0.05) defValue = 'Elite (tough matchup)';
                else if (data.def_epa < -0.02) defValue = 'Above Average';
                else defValue = 'League Average';
                items.push({ label: 'vs Defense', value: defValue });
            }

            // Historical sample
            if (data.hist_count && data.hist_count >= 3) {
                items.push({ label: 'Sample Size', value: data.hist_count + ' similar games' });
            }

            // Line vs Trailing gap
            if (data.line && data.trailing_avg) {
                const gap = data.line - data.trailing_avg;
                const gapPct = ((gap / data.trailing_avg) * 100).toFixed(0);
                let gapText = Math.abs(gap).toFixed(1);
                if (gap > 0) gapText = 'Line ' + gapText + ' above avg (' + Math.abs(gapPct) + '%)';
                else gapText = 'Line ' + Math.abs(gap).toFixed(1) + ' below avg (' + Math.abs(gapPct) + '%)';
                items.push({ label: 'Line Gap', value: gapText });
            }

            items.forEach(item => {
                container.innerHTML += `
                    <div class="modal-matchup-item">
                        <span class="modal-matchup-label">${item.label}</span>
                        <span class="modal-matchup-value">${item.value}</span>
                    </div>
                `;
            });
        }

        // Close modal on overlay click
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('modal-overlay')) {
                closePickModal();
            }
        });

        // Close modal on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closePickModal();
                closeBetSlip();
            }
        });

        // ===== BET SLIP FUNCTIONALITY =====
        const BET_STORAGE_KEY = 'nfl_quant_selected_bets';

        // Load selected bets from localStorage
        function loadSelectedBets() {
            try {
                const stored = localStorage.getItem(BET_STORAGE_KEY);
                return stored ? JSON.parse(stored) : {};
            } catch (e) {
                console.error('Error loading bets:', e);
                return {};
            }
        }

        // Save selected bets to localStorage
        function saveSelectedBets(bets) {
            try {
                localStorage.setItem(BET_STORAGE_KEY, JSON.stringify(bets));
            } catch (e) {
                console.error('Error saving bets:', e);
            }
        }

        // Toggle bet selection when checkbox clicked
        function toggleBetSelection(checkbox) {
            const betData = JSON.parse(checkbox.dataset.bet);
            const bets = loadSelectedBets();

            if (checkbox.checked) {
                bets[betData.id] = {
                    ...betData,
                    selectedAt: new Date().toISOString()
                };
            } else {
                delete bets[betData.id];
            }

            saveSelectedBets(bets);
            updateBetSlipUI();
            updateBetSlipToggle();
        }

        // Remove bet from slip
        function removeBet(betId) {
            const bets = loadSelectedBets();
            delete bets[betId];
            saveSelectedBets(bets);

            // Uncheck the corresponding checkbox
            const checkbox = document.querySelector(`[data-bet-id="${betId}"] .bet-checkbox`);
            if (checkbox) checkbox.checked = false;

            updateBetSlipUI();
            updateBetSlipToggle();
        }

        // Clear all selected bets
        function clearAllBets() {
            if (!confirm('Clear all selected bets?')) return;
            saveSelectedBets({});

            // Uncheck all checkboxes
            document.querySelectorAll('.bet-checkbox').forEach(cb => cb.checked = false);

            updateBetSlipUI();
            updateBetSlipToggle();
        }

        // Update the floating toggle button
        function updateBetSlipToggle() {
            const bets = loadSelectedBets();
            const count = Object.keys(bets).length;
            const toggle = document.getElementById('bet-slip-toggle');
            const countEl = toggle.querySelector('.count');
            const textEl = toggle.querySelector('.bet-slip-text');

            countEl.textContent = count;

            if (count === 0) {
                toggle.classList.add('empty');
                toggle.classList.remove('has-picks');
                if (textEl) textEl.textContent = 'Build Bet Slip';
            } else {
                toggle.classList.remove('empty');
                toggle.classList.add('has-picks');
                if (textEl) textEl.textContent = `View Slip (${count})`;
            }
        }

        // Update bet slip panel content
        function updateBetSlipUI() {
            const bets = loadSelectedBets();
            const betList = Object.values(bets);
            const content = document.getElementById('bet-slip-content');
            const countEl = document.getElementById('bet-slip-count');
            const avgConfEl = document.getElementById('bet-slip-avg-conf');

            if (countEl) countEl.textContent = betList.length;

            if (betList.length === 0) {
                content.innerHTML = `
                    <div class="bet-slip-empty">
                        <div class="bet-slip-empty-icon">📋</div>
                        <p>No bets selected</p>
                        <p style="font-size: 11px; margin-top: 8px;">Click the checkbox next to any pick to add it to your slip</p>
                    </div>
                `;
                if (avgConfEl) avgConfEl.textContent = '-';
                return;
            }

            // Calculate average confidence
            const avgConf = betList.reduce((sum, b) => sum + (b.confidence || 0), 0) / betList.length;
            if (avgConfEl) avgConfEl.textContent = avgConf.toFixed(1) + '%';

            // Build bet items HTML
            let html = '';
            betList.forEach(bet => {
                const pickClass = bet.pick === 'OVER' ? 'pick-over' : 'pick-under';
                html += `
                    <div class="bet-slip-item">
                        <div class="bet-slip-item-header">
                            <span class="bet-slip-player">${bet.player} (${bet.team})</span>
                            <button class="bet-slip-remove" onclick="removeBet('${bet.id}')" title="Remove">×</button>
                        </div>
                        <div class="bet-slip-details">
                            <span class="bet-slip-pick ${pickClass}">${bet.pick}</span>
                            <span class="bet-slip-line">${bet.line}</span>
                            <span class="bet-slip-market">${bet.market_display || bet.market}</span>
                            <span class="bet-slip-conf">${bet.confidence}%</span>
                        </div>
                    </div>
                `;
            });

            content.innerHTML = html;
        }

        // Toggle bet slip panel
        function toggleBetSlip() {
            const panel = document.getElementById('bet-slip-panel');
            const overlay = document.getElementById('bet-slip-overlay');
            panel.classList.toggle('open');
            overlay.classList.toggle('open');
            updateBetSlipUI();
        }

        function closeBetSlip() {
            const panel = document.getElementById('bet-slip-panel');
            const overlay = document.getElementById('bet-slip-overlay');
            panel.classList.remove('open');
            overlay.classList.remove('open');
        }

        // Export bets to JSON file
        function exportBetsJSON() {
            const bets = loadSelectedBets();
            const betList = Object.values(bets);

            if (betList.length === 0) {
                showToast('No bets to export');
                return;
            }

            const exportData = {
                exported_at: new Date().toISOString(),
                week: betList[0]?.week || 'Unknown',
                bets: betList.map(b => ({
                    player: b.player,
                    team: b.team,
                    position: b.position,
                    pick: b.pick,
                    market: b.market,
                    line: b.line,
                    odds: b.odds,
                    confidence: b.confidence,
                    edge: b.edge,
                    projection: b.projection,
                    game: b.game,
                    tier: b.tier
                }))
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `my_bets_week${exportData.week}_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('Bets exported successfully!');
        }

        // Save bets to tracking file (for later result tracking)
        function saveBetsForTracking() {
            const bets = loadSelectedBets();
            const betList = Object.values(bets);

            if (betList.length === 0) {
                showToast('No bets to save');
                return;
            }

            // Create tracking data
            const trackingData = {
                saved_at: new Date().toISOString(),
                week: betList[0]?.week || 0,
                bets: betList.map(b => ({
                    player: b.player,
                    team: b.team,
                    pick: b.pick,
                    market: b.market,
                    line: b.line,
                    odds: b.odds || -110,
                    confidence: b.confidence,
                    projection: b.projection
                }))
            };

            // Copy to clipboard for manual save
            const text = JSON.stringify(trackingData, null, 2);
            navigator.clipboard.writeText(text).then(() => {
                showToast('Bets copied to clipboard! Paste into data/tracking/pending_bets.json');
            }).catch(() => {
                // Fallback: download file
                const blob = new Blob([text], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'pending_bets.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast('Bets saved to pending_bets.json');
            });
        }

        // Restore checkbox states on page load
        function restoreCheckboxStates() {
            const bets = loadSelectedBets();
            document.querySelectorAll('.bet-checkbox').forEach(checkbox => {
                try {
                    const betData = JSON.parse(checkbox.dataset.bet);
                    if (bets[betData.id]) {
                        checkbox.checked = true;
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            });
            updateBetSlipToggle();
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            restoreCheckboxStates();
            updateBetSlipToggle();
            initInteractiveFeatures();
        });

        // ============================================================
        // INTERACTIVE FEATURES - KEYBOARD NAV, FILTERS, SCROLL, COPY
        // ============================================================

        function initInteractiveFeatures() {
            // Add scroll-to-top button
            const scrollBtn = document.createElement('button');
            scrollBtn.className = 'scroll-top-btn';
            scrollBtn.innerHTML = '↑';
            scrollBtn.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
            document.body.appendChild(scrollBtn);

            window.addEventListener('scroll', () => {
                scrollBtn.classList.toggle('visible', window.scrollY > 500);
            });

            // Add toast container
            const toast = document.createElement('div');
            toast.id = 'toast-notification';
            toast.className = 'toast-notification';
            document.body.appendChild(toast);

            // Initialize quick filters
            initQuickFilters();

            // Initialize keyboard navigation
            initKeyboardNav();
        }

        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast-notification');
            if (!toast) return;
            toast.textContent = message;
            toast.className = 'toast-notification show ' + type;
            setTimeout(() => toast.classList.remove('show'), 2500);
        }

        function copyPickToClipboard(player, pick, line, market, conf) {
            const text = player + ' ' + pick + ' ' + line + ' ' + market + ' (' + conf + '% conf)';
            navigator.clipboard.writeText(text).then(() => {
                showToast('Pick copied to clipboard!', 'success');
            }).catch(() => {
                showToast('Failed to copy', 'error');
            });
        }

        function initQuickFilters() {
            document.querySelectorAll('.filter-pill').forEach(pill => {
                pill.addEventListener('click', () => {
                    document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
                    pill.classList.add('active');

                    const filter = pill.dataset.filter;
                    // Target both game-pick-card (By Game view) and expandable-row (legacy/other views)
                    const cards = document.querySelectorAll('.game-pick-card');

                    cards.forEach(card => {
                        let show = true;

                        if (filter === 'elite') {
                            // Elite cards have tier-elite or tier-strong class
                            show = card.classList.contains('tier-elite') || card.classList.contains('tier-strong');
                        } else if (filter === 'over') {
                            // Check for pick-over badge
                            show = card.querySelector('.pick-over') !== null;
                        } else if (filter === 'under') {
                            // Check for pick-under badge
                            show = card.querySelector('.pick-under') !== null;
                        } else if (filter === 'high-edge') {
                            // Find edge value in the edge stat
                            const edgeStat = card.querySelector('.game-pick-stat:last-child .game-pick-stat-value');
                            const edgeText = edgeStat?.textContent || '0';
                            const edge = parseFloat(edgeText.replace(/[^0-9.-]/g, ''));
                            show = edge >= 10;
                        }

                        card.style.display = show ? '' : 'none';
                    });

                    // Update visible count
                    const visibleCount = document.querySelectorAll('.game-pick-card:not([style*="display: none"])').length;
                    const totalCount = cards.length;
                    showToast(`Showing ${visibleCount} of ${totalCount} picks`);
                });
            });
        }

        function initKeyboardNav() {
            document.addEventListener('keydown', (e) => {
                // Only activate if not in input field
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                // Support both featured cards, game pick cards, and table rows
                const featuredCards = document.querySelectorAll('.featured-pick-card:not([style*="display: none"])');
                const gameCards = document.querySelectorAll('.game-pick-card:not([style*="display: none"])');

                // Determine which card set to navigate based on what's visible
                let cards = featuredCards.length > 0 ? featuredCards : gameCards;
                const focusClass = featuredCards.length > 0 ? 'keyboard-focus' : 'focused';
                const activeCard = document.querySelector(`.featured-pick-card.keyboard-focus, .game-pick-card.focused`);

                if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                    e.preventDefault();
                    const currentIndex = activeCard ? Array.from(cards).indexOf(activeCard) : -1;

                    // Featured cards use grid, so left/right make sense; game cards are vertical
                    let nextIndex;
                    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                        nextIndex = Math.min(currentIndex + 1, cards.length - 1);
                    } else {
                        nextIndex = Math.max(currentIndex - 1, 0);
                    }

                    // Clear all focus states
                    document.querySelectorAll('.featured-pick-card').forEach(c => c.classList.remove('keyboard-focus'));
                    document.querySelectorAll('.game-pick-card').forEach(c => c.classList.remove('focused'));

                    if (cards[nextIndex]) {
                        cards[nextIndex].classList.add(focusClass);
                        cards[nextIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }

                if (e.key === 'Enter' && activeCard) {
                    // Click the card directly (opens modal)
                    activeCard.click();
                }

                if (e.key === 'Escape') {
                    // Close modal if open
                    const modal = document.querySelector('.modal-overlay');
                    if (modal && modal.style.display !== 'none') {
                        modal.style.display = 'none';
                        document.body.style.overflow = 'auto';
                    }
                    // Remove all focus states
                    document.querySelectorAll('.featured-pick-card').forEach(c => c.classList.remove('keyboard-focus'));
                    document.querySelectorAll('.game-pick-card').forEach(c => c.classList.remove('focused'));
                }
            });
        }

        // ============================================
        // CHEAT SHEET TABLE - Filter & Sort Functions
        // ============================================

        let csCurrentMarketFilter = 'all';
        let csCurrentDirectionFilter = 'all';
        let csCurrentPositionFilter = 'all';
        let csCurrentTierFilter = 'all';
        let csCurrentSortKey = 'confidence';
        let csCurrentSortAsc = false;

        function filterCheatSheet(marketFilter, btn) {
            csCurrentMarketFilter = marketFilter;

            // Update active pill state
            const filterBar = document.getElementById('cheat-sheet-filters');
            filterBar.querySelectorAll('.filter-pill[data-filter]:not([data-filter="over"]):not([data-filter="under"])').forEach(p => {
                p.classList.remove('active');
            });
            if (btn) btn.classList.add('active');

            applyCheatSheetFilters();
        }

        function filterCheatSheetDirection(direction, btn) {
            // Toggle direction filter
            if (csCurrentDirectionFilter === direction) {
                csCurrentDirectionFilter = 'all';
                btn.classList.remove('active');
            } else {
                csCurrentDirectionFilter = direction;
                // Deactivate other direction pills
                document.querySelectorAll('#cheat-sheet-filters .filter-pill[data-filter="over"], #cheat-sheet-filters .filter-pill[data-filter="under"]').forEach(p => {
                    p.classList.remove('active');
                });
                btn.classList.add('active');
            }

            applyCheatSheetFilters();
        }

        function filterCheatSheetPosition(position, btn) {
            csCurrentPositionFilter = position;
            // Update pill active states for position pills
            document.querySelectorAll('#cheat-sheet-filters .filter-pill[data-filter="all"], #cheat-sheet-filters .filter-pill[data-filter="QB"], #cheat-sheet-filters .filter-pill[data-filter="RB"], #cheat-sheet-filters .filter-pill[data-filter="WR"], #cheat-sheet-filters .filter-pill[data-filter="TE"]').forEach(p => {
                if (p.textContent.includes('Pos') || ['QB','RB','WR','TE'].includes(p.dataset.filter)) {
                    p.classList.remove('active');
                }
            });
            if (btn) btn.classList.add('active');
            applyCheatSheetFilters();
        }

        let csCurrentGameFilter = 'all';
        function filterCheatSheetGame(game, btn) {
            csCurrentGameFilter = game;
            // Update pill active states for game pills
            document.querySelectorAll('#game-filters .filter-pill').forEach(p => p.classList.remove('active'));
            if (btn) btn.classList.add('active');
            applyCheatSheetFilters();
        }

        function applyCheatSheetFilters() {
            const rows = document.querySelectorAll('#cheat-sheet-tbody tr');
            let visibleCount = 0;

            rows.forEach(row => {
                const market = row.dataset.market || '';
                const direction = row.dataset.direction || '';
                const position = row.dataset.position || '';
                const confidence = parseFloat(row.dataset.confidence) || 0;
                const game = row.dataset.game || '';

                let show = true;

                // Market filter
                if (csCurrentMarketFilter !== 'all' && market !== csCurrentMarketFilter) {
                    show = false;
                }

                // Direction filter
                if (csCurrentDirectionFilter !== 'all' && direction !== csCurrentDirectionFilter) {
                    show = false;
                }

                // Position filter
                if (csCurrentPositionFilter !== 'all' && position !== csCurrentPositionFilter) {
                    show = false;
                }

                // Game filter
                if (csCurrentGameFilter !== 'all' && game !== csCurrentGameFilter) {
                    show = false;
                }

                // Tier filter
                if (csCurrentTierFilter === 'elite' && confidence < 65) {
                    show = false;
                } else if (csCurrentTierFilter === 'strong' && confidence < 60) {
                    show = false;
                }

                row.style.display = show ? '' : 'none';
                if (show) visibleCount++;
            });

            // Update count display
            const countEl = document.getElementById('cs-results-count');
            if (countEl) {
                countEl.textContent = visibleCount + ' picks';
            }

            // Show/hide empty state
            let emptyState = document.getElementById('cs-empty-state');
            if (visibleCount === 0) {
                if (!emptyState) {
                    emptyState = document.createElement('div');
                    emptyState.id = 'cs-empty-state';
                    emptyState.className = 'empty-state';
                    emptyState.innerHTML = '<div class="empty-state-icon">🔍</div><div class="empty-state-text">No picks match your filters</div><div class="empty-state-hint">Try adjusting your filter criteria</div>';
                    document.getElementById('cheat-sheet-tbody').parentElement.appendChild(emptyState);
                }
                emptyState.style.display = 'block';
            } else if (emptyState) {
                emptyState.style.display = 'none';
            }
        }

        function sortCheatSheet(sortKey, th) {
            const table = document.getElementById('cheat-sheet-table');
            const tbody = document.getElementById('cheat-sheet-tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Determine sort direction
            if (csCurrentSortKey === sortKey) {
                csCurrentSortAsc = !csCurrentSortAsc;
            } else {
                csCurrentSortKey = sortKey;
                csCurrentSortAsc = false; // Default to descending for most metrics
            }

            // Update header styling
            table.querySelectorAll('th').forEach(h => {
                h.classList.remove('sorted-asc', 'sorted-desc');
            });
            th.classList.add(csCurrentSortAsc ? 'sorted-asc' : 'sorted-desc');

            // Sort rows
            rows.sort((a, b) => {
                let aVal, bVal;

                switch(sortKey) {
                    case 'player':
                        aVal = a.querySelector('.player-name')?.textContent || '';
                        bVal = b.querySelector('.player-name')?.textContent || '';
                        return csCurrentSortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);

                    case 'line':
                        aVal = parseFloat(a.querySelector('.prop-line')?.textContent) || 0;
                        bVal = parseFloat(b.querySelector('.prop-line')?.textContent) || 0;
                        break;

                    case 'projection':
                        aVal = parseFloat(a.cells[2]?.textContent) || 0;
                        bVal = parseFloat(b.cells[2]?.textContent) || 0;
                        break;

                    case 'diff':
                        aVal = parseFloat(a.cells[3]?.textContent) || 0;
                        bVal = parseFloat(b.cells[3]?.textContent) || 0;
                        break;

                    case 'confidence':
                        aVal = parseFloat(a.dataset.confidence) || 0;
                        bVal = parseFloat(b.dataset.confidence) || 0;
                        break;

                    case 'edge':
                        aVal = parseFloat(a.cells[5]?.textContent?.replace('%', '')) || 0;
                        bVal = parseFloat(b.cells[5]?.textContent?.replace('%', '')) || 0;
                        break;

                    case 'hist':
                        aVal = parseFloat(a.querySelector('.hit-rate-value')?.textContent) || 0;
                        bVal = parseFloat(b.querySelector('.hit-rate-value')?.textContent) || 0;
                        break;

                    default:
                        aVal = 0;
                        bVal = 0;
                }

                if (typeof aVal === 'number' && typeof bVal === 'number') {
                    return csCurrentSortAsc ? aVal - bVal : bVal - aVal;
                }
                return 0;
            });

            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }

        // ============================================
        // GAME LINES TABLE - Filter & Sort Functions
        // ============================================

        let glCurrentTypeFilter = 'all';
        let glCurrentTierFilter = 'all';
        let glCurrentSortKey = 'confidence';
        let glCurrentSortAsc = false;

        function filterGameLines(typeFilter, btn) {
            glCurrentTypeFilter = typeFilter;

            // Update active pill state for type filters
            const filterBar = document.getElementById('game-lines-filters');
            filterBar.querySelectorAll('.filter-pill[data-filter="all"], .filter-pill[data-filter="spread"], .filter-pill[data-filter="total"]').forEach(p => {
                p.classList.remove('active');
            });
            if (btn) btn.classList.add('active');

            applyGameLinesFilters();
        }

        function filterGameLinesTier(tier, btn) {
            // Toggle tier filter
            if (glCurrentTierFilter === tier) {
                glCurrentTierFilter = 'all';
                btn.classList.remove('active');
            } else {
                glCurrentTierFilter = tier;
                // Deactivate other tier pills
                document.querySelectorAll('#game-lines-filters .filter-pill[data-filter="ELITE"], #game-lines-filters .filter-pill[data-filter="HIGH"]').forEach(p => {
                    p.classList.remove('active');
                });
                btn.classList.add('active');
            }

            applyGameLinesFilters();
        }

        function applyGameLinesFilters() {
            const rows = document.querySelectorAll('#game-lines-tbody tr');
            let visibleCount = 0;

            rows.forEach(row => {
                const betType = row.dataset.betType || '';
                const tier = row.dataset.tier || '';

                let show = true;

                // Type filter
                if (glCurrentTypeFilter !== 'all' && betType !== glCurrentTypeFilter) {
                    show = false;
                }

                // Tier filter
                if (glCurrentTierFilter !== 'all' && tier.toUpperCase() !== glCurrentTierFilter) {
                    show = false;
                }

                row.style.display = show ? '' : 'none';
                if (show) visibleCount++;
            });
        }

        function sortGameLines(sortKey, th) {
            const table = document.getElementById('game-lines-table');
            const tbody = document.getElementById('game-lines-tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Determine sort direction
            if (glCurrentSortKey === sortKey) {
                glCurrentSortAsc = !glCurrentSortAsc;
            } else {
                glCurrentSortKey = sortKey;
                glCurrentSortAsc = sortKey === 'game'; // Alpha asc by default, numbers desc
            }

            // Update header styling
            table.querySelectorAll('th').forEach(h => {
                h.classList.remove('sorted-asc', 'sorted-desc');
            });
            th.classList.add(glCurrentSortAsc ? 'sorted-asc' : 'sorted-desc');

            // Sort rows
            rows.sort((a, b) => {
                let aVal, bVal;

                switch(sortKey) {
                    case 'game':
                        aVal = a.querySelector('.gl-game-name')?.textContent || '';
                        bVal = b.querySelector('.gl-game-name')?.textContent || '';
                        return glCurrentSortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);

                    case 'type':
                        aVal = a.dataset.betType || '';
                        bVal = b.dataset.betType || '';
                        return glCurrentSortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);

                    case 'confidence':
                        aVal = parseFloat(a.cells[4]?.textContent) || 0;
                        bVal = parseFloat(b.cells[4]?.textContent) || 0;
                        break;

                    case 'edge':
                        aVal = parseFloat(a.cells[5]?.textContent?.replace('%', '')) || 0;
                        bVal = parseFloat(b.cells[5]?.textContent?.replace('%', '')) || 0;
                        break;

                    case 'tier':
                        const tierOrder = { 'elite': 0, 'high': 1, 'standard': 2, 'low': 3 };
                        aVal = tierOrder[a.dataset.tier] ?? 4;
                        bVal = tierOrder[b.dataset.tier] ?? 4;
                        break;

                    default:
                        aVal = 0;
                        bVal = 0;
                }

                if (typeof aVal === 'number' && typeof bVal === 'number') {
                    return glCurrentSortAsc ? aVal - bVal : bVal - aVal;
                }
                return 0;
            });

            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }

        // ============================================
        // BETTINGPROS-STYLE GAME LINES - Filter Functions
        // ============================================

        let bpCurrentTypeFilter = 'all';
        let bpCurrentTierFilter = 'all';

        function filterBPGameLines(typeFilter, btn) {
            bpCurrentTypeFilter = typeFilter;

            // Update active pill state
            const filterBar = document.getElementById('game-lines-filters');
            filterBar.querySelectorAll('.filter-pill[data-filter="all"], .filter-pill[data-filter="spread"], .filter-pill[data-filter="total"]').forEach(p => {
                p.classList.remove('active');
            });
            if (btn) btn.classList.add('active');

            applyBPGameLinesFilters();
        }

        function filterBPGameLinesTier(tier, btn) {
            // Toggle tier filter
            if (bpCurrentTierFilter === tier) {
                bpCurrentTierFilter = 'all';
                btn.classList.remove('active');
            } else {
                bpCurrentTierFilter = tier;
                // Deactivate other tier pills
                document.querySelectorAll('#game-lines-filters .filter-pill[data-filter="elite"], #game-lines-filters .filter-pill[data-filter="high"]').forEach(p => {
                    p.classList.remove('active');
                });
                btn.classList.add('active');
            }

            applyBPGameLinesFilters();
        }

        function applyBPGameLinesFilters() {
            const cards = document.querySelectorAll('.bp-game-card');

            cards.forEach(card => {
                const rows = card.querySelectorAll('.bp-game-row');
                let cardHasVisibleRow = false;

                rows.forEach(row => {
                    const betType = row.dataset.betType || '';
                    const tier = (row.dataset.tier || '').toLowerCase();

                    let show = true;

                    // Type filter
                    if (bpCurrentTypeFilter !== 'all' && betType !== bpCurrentTypeFilter) {
                        show = false;
                    }

                    // Tier filter
                    if (bpCurrentTierFilter !== 'all' && tier !== bpCurrentTierFilter) {
                        show = false;
                    }

                    row.style.display = show ? '' : 'none';
                    if (show) cardHasVisibleRow = true;
                });

                // Hide card if no rows are visible
                card.style.display = cardHasVisibleRow ? '' : 'none';
            });
        }
    '''


def parse_model_reasoning(reasoning: str) -> dict:
    """Parse the model_reasoning string into structured components."""
    result = {
        'vegas_line': None,
        'trailing_avg': None,
        'lvt': None,
        'mc_projection': None,
        'p_under': None,
        'p_over': None,
        'pick': None,
        'ev': None,
        'kelly': None,
        'why': ''
    }

    if not reasoning or pd.isna(reasoning):
        return result

    try:
        # Extract key values using string parsing
        if 'Vegas Line:' in reasoning:
            result['vegas_line'] = reasoning.split('Vegas Line:')[1].split('•')[0].strip()
        if 'Trailing Avg:' in reasoning:
            result['trailing_avg'] = reasoning.split('Trailing Avg:')[1].split('•')[0].strip()
        if 'LVT' in reasoning:
            lvt_part = reasoning.split('LVT')[1].split('•')[0]
            if ':' in lvt_part:
                result['lvt'] = lvt_part.split(':')[1].strip()
        if 'Monte Carlo Projection:' in reasoning:
            result['mc_projection'] = reasoning.split('Monte Carlo Projection:')[1].split('•')[0].strip()
        if 'P(UNDER):' in reasoning:
            result['p_under'] = reasoning.split('P(UNDER):')[1].split('•')[0].strip()
        if 'P(OVER):' in reasoning:
            result['p_over'] = reasoning.split('P(OVER):')[1].split('•')[0].strip()
        if 'Expected Value:' in reasoning:
            result['ev'] = reasoning.split('Expected Value:')[1].split('•')[0].strip()
        if 'Kelly Sizing:' in reasoning:
            result['kelly'] = reasoning.split('Kelly Sizing:')[1].split('•')[0].strip()
        if 'WHY' in reasoning:
            why_part = reasoning.split('WHY')[1]
            if ':' in why_part:
                result['why'] = why_part.split(':')[1].strip()
    except:
        pass

    return result


def generate_player_row(row: pd.Series, row_id: str, include_game: bool = True) -> str:
    """Generate HTML for a player prop row with expandable logic.

    Args:
        row: DataFrame row with player data
        row_id: Unique ID for expandable logic
        include_game: Whether to include Game column (False for By Game view)
    """
    player = row.get('player', row.get('nflverse_name', 'Unknown'))
    team = safe_str(row.get('team', ''), '')
    position = row.get('position', 'N/A')
    game = row.get('game', 'N/A')
    market = format_prop_display(row.get('market', ''))
    raw_market = row.get('market', '')
    pick = row.get('pick', 'N/A')
    line = row.get('line', 0)
    projection = get_projection(row, raw_market)

    # Get probabilities
    model_prob = row.get('model_prob', 0.5)
    if model_prob > 1:
        model_prob = model_prob / 100

    edge_pct = row.get('edge_pct', 0)
    if isinstance(edge_pct, str):
        edge_pct = float(edge_pct.replace('%', ''))

    tier = row.get('confidence', row.get('priority', 'LOW'))
    kelly_units = row.get('kelly_units', row.get('kelly_fraction', 0))
    if kelly_units > 1:
        kelly_units = kelly_units / 100

    # Get model probabilities (active model)
    model_p_under_raw = row.get('model_p_under', 0.5)
    model_validated = row.get('model_validated', False)
    model_reason = row.get('model_reason', '')

    # Use model_p_under from active model
    model_p_under = model_p_under_raw if model_p_under_raw and not pd.isna(model_p_under_raw) else 0.5
    model_p_over = 1 - model_p_under
    # Detect active model from model_reason - show version whenever model was used (even if not validated)
    if MODEL_VERSION_FULL in str(model_reason) or "V1" in str(model_reason):
        model_used = MODEL_VERSION_FULL if model_validated else f"{MODEL_VERSION_FULL}*"  # * = ran but below threshold
    elif model_validated:
        model_used = MODEL_VERSION_FULL  # Use current version
    else:
        model_used = "Edge Model"

    # Get model reasoning and parse it
    model_reasoning = row.get('model_reasoning', '')
    parsed = parse_model_reasoning(model_reasoning)

    # Get additional context
    opponent = row.get('opponent', '')
    model_std = row.get('model_std', 0)
    if pd.isna(model_std):
        model_std = 0

    # Create sensible MC projection display
    # For high-variance stats (std > 50% of projection), show range instead of ± raw std
    if projection > 0 and model_std > projection * 0.5:
        # Show a likely range (±0.5 std ≈ 40% CI) for high variance
        range_low = max(0, projection - model_std * 0.5)
        range_high = projection + model_std * 0.5
        mc_projection_display = f"{projection:.1f} (range: {range_low:.0f}-{range_high:.0f})"
    elif model_std > 0:
        mc_projection_display = f"{projection:.1f} ± {model_std:.1f}"
    else:
        mc_projection_display = f"{projection:.1f}"

    raw_prob = row.get('raw_prob', model_prob)
    calibrated_prob = row.get('calibrated_prob', model_prob)

    # Get model-guided simulation info (V4)
    model_guided = row.get('model_guided', False)
    model_adjustment_pct = row.get('model_adjustment_pct', 0)
    if pd.isna(model_adjustment_pct):
        model_adjustment_pct = 0
    baseline_projection = row.get('baseline_projection', 0)
    if pd.isna(baseline_projection):
        baseline_projection = trailing if trailing else projection
    adjustment_breakdown = row.get('adjustment_breakdown', '')
    if pd.isna(adjustment_breakdown):
        adjustment_breakdown = ''

    # Get defense and context data
    opp_def_epa = row.get('opponent_def_epa', 0)
    if pd.isna(opp_def_epa):
        opp_def_epa = 0
    snap_share = row.get('snap_share', 0)
    if pd.isna(snap_share):
        snap_share = 0
    calibration_tier = row.get('calibration_tier', '')
    injury_qb = row.get('injury_qb_status', 'active')
    injury_wr1 = row.get('injury_wr1_status', 'active')
    injury_rb1 = row.get('injury_rb1_status', 'active')
    weather_adj = row.get('weather_total_adjustment', 0)
    if pd.isna(weather_adj):
        weather_adj = 0

    # V25 Team Synergy data
    synergy_mult = row.get('synergy_multiplier', 1.0)
    if pd.isna(synergy_mult):
        synergy_mult = 1.0
    has_synergy = row.get('has_synergy_bonus', 0)
    if pd.isna(has_synergy):
        has_synergy = 0
    snap_ramp_factor = row.get('snap_ramp_factor', 1.0)
    if pd.isna(snap_ramp_factor):
        snap_ramp_factor = 1.0
    snap_ramp_adjusted = row.get('snap_ramp_adjusted', False)

    # Line vs Trailing calculation
    trailing = projection
    lvt_delta = line - trailing if line and trailing else 0
    lvt_pct = (lvt_delta / trailing * 100) if trailing and trailing != 0 else 0

    # Direction color
    direction_class = 'direction-over' if str(pick).upper() == 'OVER' else 'direction-under'

    # Format bet sizing with visual bar
    bet_display = format_bet_display(kelly_units)

    # Build row cells conditionally based on include_game (use abbreviated game name)
    game_abbrev = abbreviate_game(game)
    game_cell = f'<td style="font-size: 12px; white-space: nowrap;">{game_abbrev}</td>' if include_game else ''
    colspan = "10" if include_game else "9"

    # model_used already set above based on model availability
    # Model validated = badge-high; Edge Model = fallback, badge-standard
    model_badge_class = "badge-high" if model_used.startswith("V") else "badge-standard"

    # Defense matchup interpretation
    def_matchup = "neutral"
    def_matchup_class = ""
    if opp_def_epa > 0.02:
        def_matchup = "favorable (weak DEF)"
        def_matchup_class = "direction-over"
    elif opp_def_epa < -0.02:
        def_matchup = "tough (strong DEF)"
        def_matchup_class = "direction-under"

    # Player tier interpretation
    tier_display = calibration_tier.replace('_', ' ').title() if calibration_tier else "Unknown"

    # Build NARRATIVE explanation that explains the decision-making process
    # Get all factor data
    hist_over_rate = row.get('hist_over_rate', 0)
    if pd.isna(hist_over_rate):
        hist_over_rate = 0
    hist_avg = row.get('hist_avg', 0)
    if pd.isna(hist_avg):
        hist_avg = 0
    hist_count = row.get('hist_count', 0)
    if pd.isna(hist_count):
        hist_count = 0
    edge_pct_val = row.get('edge_pct', 0)
    if pd.isna(edge_pct_val):
        edge_pct_val = 0
    is_divisional = row.get('is_divisional_game', False)
    is_primetime = row.get('is_primetime_game', False)
    primetime_type = row.get('primetime_type', '')
    sharp_confirmed = row.get('sharp_confirmed', '')
    model_reason = row.get('model_reason', '')
    game_script = row.get('game_script_dynamic', 0)
    if pd.isna(game_script):
        game_script = 0
    home_field = row.get('home_field_advantage_points', 0)
    if pd.isna(home_field):
        home_field = 0

    # MC Simulation Volume Projections
    targets_mean = row.get('targets_mean', 0)
    if pd.isna(targets_mean):
        targets_mean = 0
    receptions_mean = row.get('receptions_mean', 0)
    if pd.isna(receptions_mean):
        receptions_mean = 0
    receiving_yards_mean = row.get('receiving_yards_mean', 0)
    if pd.isna(receiving_yards_mean):
        receiving_yards_mean = 0
    rushing_attempts_mean = row.get('rushing_attempts_mean', 0)
    if pd.isna(rushing_attempts_mean):
        rushing_attempts_mean = 0
    rushing_yards_mean = row.get('rushing_yards_mean', 0)
    if pd.isna(rushing_yards_mean):
        rushing_yards_mean = 0
    passing_attempts_mean = row.get('passing_attempts_mean', 0)
    if pd.isna(passing_attempts_mean):
        passing_attempts_mean = 0
    passing_completions_mean = row.get('passing_completions_mean', 0)
    if pd.isna(passing_completions_mean):
        passing_completions_mean = 0
    passing_yards_mean = row.get('passing_yards_mean', 0)
    if pd.isna(passing_yards_mean):
        passing_yards_mean = 0

    # Opportunity Share
    rz_target_share = row.get('redzone_target_share', 0)
    if pd.isna(rz_target_share):
        rz_target_share = 0
    rz_carry_share = row.get('redzone_carry_share', 0)
    if pd.isna(rz_carry_share):
        rz_carry_share = 0
    gl_carry_share = row.get('goalline_carry_share', 0)
    if pd.isna(gl_carry_share):
        gl_carry_share = 0
    team_pass_att = row.get('team_pass_attempts', 0)
    if pd.isna(team_pass_att):
        team_pass_att = 0
    team_rush_att = row.get('team_rush_attempts', 0)
    if pd.isna(team_rush_att):
        team_rush_att = 0

    # Situational Adjustments
    rest_adj = row.get('rest_epa_adjustment', 0)
    if pd.isna(rest_adj):
        rest_adj = 0
    travel_adj = row.get('travel_epa_adjustment', 0)
    if pd.isna(travel_adj):
        travel_adj = 0
    altitude_adj = row.get('altitude_epa_adjustment', 0)
    if pd.isna(altitude_adj):
        altitude_adj = 0
    is_high_altitude = row.get('is_high_altitude', False)

    # Get vs opponent history
    vs_opp_avg = None
    vs_opp_games = 0
    if 'receptions' in str(market).lower():
        vs_opp_avg = row.get('vs_opp_avg_receptions', None)
        vs_opp_games = row.get('vs_opp_games_receptions', 0)
    elif 'reception_yds' in str(market).lower() or 'receiving' in str(market).lower():
        vs_opp_avg = row.get('vs_opp_avg_receiving_yards', None)
        vs_opp_games = row.get('vs_opp_games_receiving_yards', 0)
    elif 'rush' in str(market).lower():
        vs_opp_avg = row.get('vs_opp_avg_rushing_yards', None)
        vs_opp_games = row.get('vs_opp_games_rushing_yards', 0)
    elif 'pass' in str(market).lower():
        vs_opp_avg = row.get('vs_opp_avg_passing_yards', None)
        vs_opp_games = row.get('vs_opp_games_passing_yards', 0)
    if pd.isna(vs_opp_avg):
        vs_opp_avg = None
    if pd.isna(vs_opp_games):
        vs_opp_games = 0

    # Categorize factors as SUPPORTING or CONTRADICTING the pick
    pick_is_under = str(pick).upper() == 'UNDER'
    supporting_factors = []
    contradicting_factors = []

    # 1. Model projection - primary driver
    mc_supports = None
    if projection and line:
        proj_vs_line = projection - line
        mc_supports_under = proj_vs_line < 0
        mc_supports = mc_supports_under == pick_is_under
        mc_strength = abs(proj_vs_line)

    # 2. Line vs trailing (recent form)
    trailing_supports = None
    if trailing and trailing != 0 and lvt_delta != 0:
        # If line > trailing avg, that supports UNDER (player underperforming the line)
        trailing_supports_under = lvt_delta > 0
        trailing_supports = trailing_supports_under == pick_is_under

    # 3. Historical hit rate
    hist_supports = None
    if hist_count >= 3 and hist_over_rate > 0:
        hit_rate = hist_over_rate if not pick_is_under else (1 - hist_over_rate)
        hist_supports = hit_rate > 0.5

    # 4. Vs opponent history
    vs_opp_supports = None
    if vs_opp_avg is not None and vs_opp_games >= 2:
        vs_opp_supports_under = vs_opp_avg < line
        vs_opp_supports = vs_opp_supports_under == pick_is_under

    # 5. Defense matchup (weak DEF = supports OVER, strong DEF = supports UNDER)
    def_supports = None
    def_strength = ""
    if opp_def_epa != 0:
        if opp_def_epa > 0.02:  # Weak defense
            def_supports = not pick_is_under  # Supports OVER
            def_strength = "weak"
        elif opp_def_epa < -0.02:  # Strong defense
            def_supports = pick_is_under  # Supports UNDER
            def_strength = "strong"

    # 6. Usage/snap share (high usage = more opportunities = supports OVER typically)
    usage_supports = None
    usage_level = ""
    if snap_share:
        if snap_share > 0.85:
            usage_supports = not pick_is_under  # High usage supports OVER
            usage_level = "elite"
        elif snap_share > 0.7:
            usage_supports = not pick_is_under
            usage_level = "high"
        elif snap_share < 0.5:
            usage_supports = pick_is_under  # Low usage supports UNDER
            usage_level = "limited"

    # BUILD THE THESIS with QUANTITATIVE BREAKDOWN
    # Get calibration tier and model std
    cal_tier = str(row.get('calibration_tier', '')).lower()
    model_std = row.get('model_std', 0) or 0

    # Calculate quantitative impacts
    # Defense EPA impact: rough estimate ~20-30 attempts/yards per 0.1 EPA for rush, ~10-15 for receptions
    def_impact = 0
    if opp_def_epa and opp_def_epa != 0:
        if 'rush' in str(market).lower():
            def_impact = opp_def_epa * 25  # EPA * multiplier = estimated impact
        elif 'reception' in str(market).lower() or 'receptions' in str(market).lower():
            def_impact = opp_def_epa * 15
        elif 'yds' in str(market).lower() or 'yards' in str(market).lower():
            def_impact = opp_def_epa * 50

    # Build the narrative with numbers
    parts = []

    # Opening: Show the gap
    if line and trailing and trailing != 0 and projection:
        gap = projection - line
        gap_dir = "above" if gap > 0 else "below"
        parts.append(f"Vegas {line} vs Model {projection:.1f} ({gap:+.1f} gap).")

    # Breakdown section
    breakdown = []

    # 1. Trailing average baseline
    if trailing and trailing != 0:
        breakdown.append(f"Baseline: {trailing:.1f}/game (4-week avg)")

    # 2. Snap share context
    if snap_share and snap_share > 0:
        snaps_per_game = snap_share * 65  # ~65 offensive snaps per game
        breakdown.append(f"Usage: {snap_share*100:.0f}% snaps (~{snaps_per_game:.0f}/game)")

    # 3. Defense EPA with estimated impact
    if opp_def_epa and opp_def_epa != 0:
        if opp_def_epa > 0.02:
            breakdown.append(f"Matchup: soft DEF (EPA {opp_def_epa:+.3f}) → ~{abs(def_impact):.1f} boost")
        elif opp_def_epa < -0.02:
            breakdown.append(f"Matchup: tough DEF (EPA {opp_def_epa:+.3f}) → ~{abs(def_impact):.1f} reduction")

    # 4. Vs opponent history with actual numbers
    if vs_opp_avg is not None and vs_opp_games >= 2:
        vs_opp_diff = vs_opp_avg - trailing if trailing else 0
        breakdown.append(f"vs {opponent}: {vs_opp_avg:.1f} avg in {int(vs_opp_games)} games ({vs_opp_diff:+.1f} vs baseline)")

    # 5. Model range
    if model_std and model_std > 0:
        low = projection - model_std
        high = projection + model_std
        breakdown.append(f"Range: {low:.1f}-{high:.1f} (68% confidence)")

    if breakdown:
        parts.append("BREAKDOWN: " + " | ".join(breakdown) + ".")

    # Why the pick
    why_parts = []
    if line and projection:
        if pick_is_under:
            why_parts.append(f"Line {line} is {abs(line - projection):.1f} above projection")
        else:
            why_parts.append(f"Line {line} is {abs(line - projection):.1f} below projection")

    # Risk acknowledgment
    if (def_supports is False and def_strength) or (vs_opp_supports is False and vs_opp_avg is not None):
        risk_text = []
        if def_supports is False and def_strength:
            risk_text.append(f"{def_strength} DEF")
        if vs_opp_supports is False and vs_opp_avg is not None and vs_opp_games >= 2:
            risk_text.append(f"weak vs {opponent}")
        if risk_text:
            why_parts.append(f"Risks: {', '.join(risk_text)}")

    if why_parts:
        parts.append("WHY: " + "; ".join(why_parts) + ".")

    # Closing
    if projection:
        parts.append(f"→ Take {pick}.")

    # Use SHAP-based model_reasoning from CSV if available, otherwise fall back to generated
    csv_reasoning = row.get('model_reasoning', '')
    if csv_reasoning and 'CALCULATION' in str(csv_reasoning):
        # Parse the raw calculation into a clean summary
        # Extract key info: P(UNDER/OVER) and primary signal
        csv_str = str(csv_reasoning)

        # Extract probability
        p_match = re.search(r'P\((UNDER|OVER)\):\s*([\d.]+)%', csv_str)
        pick_match = re.search(r'PICK:\s*(UNDER|OVER)', csv_str)
        lvt_match = re.search(r'line\s+([-+]?[\d.]+)\s+vs\s+proj', csv_str)
        edge_match = re.search(r'Edge:\s*([-+]?[\d.]+)%', csv_str)

        # Build clean explanation
        why_parts_clean = []
        if p_match:
            why_parts_clean.append(f"Model: {p_match.group(2)}% {p_match.group(1)}")
        if lvt_match:
            delta = float(lvt_match.group(1))
            if delta > 0:
                why_parts_clean.append(f"Line set {abs(delta):.1f} above projection")
            else:
                why_parts_clean.append(f"Line set {abs(delta):.1f} below projection")
        if edge_match:
            why_parts_clean.append(f"Edge: {edge_match.group(1)}%")

        if why_parts_clean:
            why_explanation = " • ".join(why_parts_clean)
        else:
            # Fallback to generated narrative
            why_explanation = " ".join(parts)
    else:
        why_explanation = " ".join(parts)

    # Injury context display
    injury_context = []
    if injury_qb and injury_qb != 'active':
        injury_context.append(f"QB: {injury_qb}")
    if injury_wr1 and injury_wr1 != 'active':
        injury_context.append(f"WR1: {injury_wr1}")
    if injury_rb1 and injury_rb1 != 'active':
        injury_context.append(f"RB1: {injury_rb1}")
    injury_display = ", ".join(injury_context) if injury_context else "None"

    # Parse SHAP factors from model_reasoning for grouped display
    over_factors, under_factors, _ = parse_shap_factors(csv_reasoning)

    # Generate SHAP factors HTML
    shap_html = ''
    if over_factors or under_factors:
        shap_html = '<div class="shap-factors">'
        if over_factors:
            shap_html += '<div class="shap-group shap-group-over"><div class="shap-group-title">Favors OVER</div>'
            for name, strength, is_dom in sorted(over_factors, key=lambda x: -x[1])[:4]:
                dom_class = ' shap-factor-dominant' if is_dom else ''
                shap_html += f'<div class="shap-factor"><span class="shap-factor-name">{name}</span><span class="shap-factor-value{dom_class}">{strength:.2f}</span></div>'
            shap_html += '</div>'
        if under_factors:
            shap_html += '<div class="shap-group shap-group-under"><div class="shap-group-title">Favors UNDER</div>'
            for name, strength, is_dom in sorted(under_factors, key=lambda x: -x[1])[:4]:
                dom_class = ' shap-factor-dominant' if is_dom else ''
                shap_html += f'<div class="shap-factor"><span class="shap-factor-name">{name}</span><span class="shap-factor-value{dom_class}">{strength:.2f}</span></div>'
            shap_html += '</div>'
        shap_html += '</div>'

    # Team logo HTML
    team_logo = get_team_logo_html(team, size=18) if team else ''

    # Build game cell with data-label for mobile
    game_cell_with_label = f'<td data-label="Game" style="font-size: 12px; white-space: nowrap;">{game_abbrev}</td>' if include_game else ''

    row_html = f'''
        <tr class="expandable-row" onclick="toggleLogic('{row_id}')">
            <td data-label=""><div class="player-cell">{team_logo}<span class="player-name">{player}</span></div></td>
            <td data-label="Pos">{position}</td>
            {game_cell_with_label}
            <td data-label="Prop">{market}</td>
            <td data-label="Pick"><strong class="{direction_class}">{pick}</strong> {line}</td>
            <td data-label="Proj" data-sort="{projection:.1f}">{projection:.1f}</td>
            <td data-label="Conf" data-sort="{model_prob}">{get_confidence_badge(model_prob)}</td>
            <td data-label="Edge" data-sort="{edge_pct}">{get_edge_badge(edge_pct)}</td>
            <td data-label="Tier"><span class="badge {get_tier_badge_class(tier)}">{tier}</span></td>
            <td data-label="Bet" data-sort="{kelly_units}">{bet_display}</td>
        </tr>
        <tr class="logic-row" id="logic-{row_id}" style="display: none;">
            <td colspan="{colspan}">
                <div class="logic-grid" style="border-left: 3px solid var(--accent-{'green' if str(pick).upper() == 'OVER' else 'red'});">
                    <div class="logic-section">
                        <div class="logic-section-title">Line Analysis</div>
                        <div class="logic-item"><span class="logic-item-label">Vegas Line</span><span class="logic-item-value">{line}</span></div>
                        <div class="logic-item"><span class="logic-item-label">4W Trailing Avg</span><span class="logic-item-value">{trailing:.1f}</span></div>
                        <div class="logic-item"><span class="logic-item-label">LVT Delta</span><span class="logic-item-value {direction_class}">{lvt_delta:+.1f} ({lvt_pct:+.1f}%)</span></div>
                        <div class="logic-item"><span class="logic-item-label">Matchup</span><span class="logic-item-value {def_matchup_class}">{def_matchup}</span></div>
                    </div>
                    <div class="logic-section">
                        <div class="logic-section-title">Model: <span class="badge {model_badge_class}" style="font-size: 9px;">{model_used}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Baseline</span><span class="logic-item-value">{baseline_projection:.1f}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Adjusted</span><span class="logic-item-value">{projection:.1f} <span class="{'direction-over' if model_adjustment_pct > 0 else 'direction-under' if model_adjustment_pct < 0 else ''}">({model_adjustment_pct:+.1f}%)</span></span></div>
                        <div class="logic-item"><span class="logic-item-label">P(Under)</span><span class="logic-item-value">{model_p_under*100:.1f}%</span></div>
                        <div class="logic-item"><span class="logic-item-label">P(Over)</span><span class="logic-item-value">{model_p_over*100:.1f}%</span></div>
                    </div>
                    <div class="logic-section">
                        <div class="logic-section-title">Context</div>
                        <div class="logic-item"><span class="logic-item-label">Snap Share</span><span class="logic-item-value">{snap_share*100:.0f}%</span></div>
                        <div class="logic-item"><span class="logic-item-label">DEF EPA</span><span class="logic-item-value {def_matchup_class}">{opp_def_epa:+.3f}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Injuries</span><span class="logic-item-value">{injury_display}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Weather Adj</span><span class="logic-item-value">{weather_adj:+.1f}%</span></div>
                    </div>
                    <div class="logic-section">
                        <div class="logic-section-title">Team Synergy (V25)</div>
                        <div class="logic-item"><span class="logic-item-label">Synergy Mult</span><span class="logic-item-value {'direction-over' if synergy_mult > 1.02 else 'direction-under' if synergy_mult < 0.98 else ''}">{synergy_mult:.3f}x</span></div>
                        <div class="logic-item"><span class="logic-item-label">Synergy Bonus</span><span class="logic-item-value">{'✓ Active' if has_synergy else '—'}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Snap Ramp</span><span class="logic-item-value {'direction-under' if snap_ramp_factor < 0.9 else ''}">{snap_ramp_factor:.0%}</span></div>
                        <div class="logic-item"><span class="logic-item-label">IR Return</span><span class="logic-item-value">{'Yes' if snap_ramp_adjusted else 'No'}</span></div>
                    </div>
                </div>
                <div class="logic-grid" style="margin-top: 1px;">
                    <div class="logic-section">
                        <div class="logic-section-title">MC Simulation (50K runs)</div>
                        {'<div class="logic-item"><span class="logic-item-label">Proj Targets</span><span class="logic-item-value">' + f"{targets_mean:.1f}" + '</span></div>' if targets_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Receptions</span><span class="logic-item-value">' + f"{receptions_mean:.1f}" + '</span></div>' if receptions_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Rec Yards</span><span class="logic-item-value">' + f"{receiving_yards_mean:.1f}" + '</span></div>' if receiving_yards_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Carries</span><span class="logic-item-value">' + f"{rushing_attempts_mean:.1f}" + '</span></div>' if rushing_attempts_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Rush Yds</span><span class="logic-item-value">' + f"{rushing_yards_mean:.1f}" + '</span></div>' if rushing_yards_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Pass Att</span><span class="logic-item-value">' + f"{passing_attempts_mean:.1f}" + '</span></div>' if passing_attempts_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Comp</span><span class="logic-item-value">' + f"{passing_completions_mean:.1f}" + '</span></div>' if passing_completions_mean > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Proj Pass Yds</span><span class="logic-item-value">' + f"{passing_yards_mean:.1f}" + '</span></div>' if passing_yards_mean > 0 else ''}
                        <div class="logic-item"><span class="logic-item-label">Model Std Dev</span><span class="logic-item-value">{model_std:.1f}</span></div>
                    </div>
                    <div class="logic-section">
                        <div class="logic-section-title">Opportunity Share</div>
                        {'<div class="logic-item"><span class="logic-item-label">RZ Tgt Share</span><span class="logic-item-value">' + f"{rz_target_share*100:.0f}%" + '</span></div>' if rz_target_share > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">RZ Carry Share</span><span class="logic-item-value">' + f"{rz_carry_share*100:.0f}%" + '</span></div>' if rz_carry_share > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">GL Carry Share</span><span class="logic-item-value">' + f"{gl_carry_share*100:.0f}%" + '</span></div>' if gl_carry_share > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Team Pass Att</span><span class="logic-item-value">' + f"{team_pass_att:.0f}" + '</span></div>' if team_pass_att > 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Team Rush Att</span><span class="logic-item-value">' + f"{team_rush_att:.0f}" + '</span></div>' if team_rush_att > 0 else ''}
                        <div class="logic-item"><span class="logic-item-label">Game Script</span><span class="logic-item-value {'direction-over' if game_script > 0.5 else 'direction-under' if game_script < -0.5 else ''}">{game_script:+.2f}</span></div>
                    </div>
                    <div class="logic-section">
                        <div class="logic-section-title">Situational</div>
                        {'<div class="logic-item"><span class="logic-item-label">Rest Adj</span><span class="logic-item-value">' + f"{rest_adj:+.3f}" + '</span></div>' if rest_adj != 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Travel Adj</span><span class="logic-item-value">' + f"{travel_adj:+.3f}" + '</span></div>' if travel_adj != 0 else ''}
                        {'<div class="logic-item"><span class="logic-item-label">Altitude Adj</span><span class="logic-item-value direction-under">' + f"{altitude_adj:+.3f}" + '</span></div>' if altitude_adj != 0 else ''}
                        <div class="logic-item"><span class="logic-item-label">Home Field</span><span class="logic-item-value">{home_field:+.1f} pts</span></div>
                        <div class="logic-item"><span class="logic-item-label">Primetime</span><span class="logic-item-value">{'✓ ' + str(primetime_type) if is_primetime else '—'}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Divisional</span><span class="logic-item-value">{'✓ Yes' if is_divisional else '—'}</span></div>
                    </div>
                    <div class="logic-section">
                        <div class="logic-section-title">vs {opponent} History</div>
                        <div class="logic-item"><span class="logic-item-label">Avg vs Opp</span><span class="logic-item-value">{f'{vs_opp_avg:.1f}' if vs_opp_avg is not None else '—'}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Games vs Opp</span><span class="logic-item-value">{int(vs_opp_games) if vs_opp_games else '—'}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Hit Rate</span><span class="logic-item-value">{f'{hist_over_rate*100:.0f}% OVER' if hist_count > 0 else '—'}</span></div>
                        <div class="logic-item"><span class="logic-item-label">Sample Size</span><span class="logic-item-value">{int(hist_count)} games</span></div>
                    </div>
                </div>
                {shap_html}
                {'<div style="padding: 8px 16px; background: var(--bg-tertiary); border-top: 1px solid var(--border-color);"><div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">Model Adjustment Factors</div><div style="font-size: 11px; color: var(--text-secondary); font-family: var(--font-mono);">' + adjustment_breakdown.replace(';', ' | ') + '</div></div>' if adjustment_breakdown and adjustment_breakdown != 'No significant adjustments' else ''}
                <div style="padding: 12px 16px; background: var(--bg-secondary); border-top: 1px solid var(--border-color);">
                    <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;">Why {str(pick).upper()}</div>
                    <div style="font-size: 12px; color: var(--text-primary);">
                        <strong class="{direction_class}">{str(pick).upper()} {line}</strong> — {why_explanation}
                    </div>
                </div>
                <div class="narrative-story" style="margin: 0; border-radius: 0;">
                    <div class="narrative-content" style="padding: 16px; font-size: 12px; line-height: 1.6;">
                        {generate_prose_only(row)}
                    </div>
                </div>
            </td>
        </tr>
    '''
    return row_html


def generate_game_line_row(row: pd.Series) -> str:
    """Generate HTML for a game line recommendation row (legacy table format)."""
    game = row.get('game', 'N/A')
    bet_type = row.get('bet_type', 'N/A')
    pick = row.get('pick', 'N/A')
    market_line = row.get('market_line', '')
    model_prob = row.get('model_prob', 0.5)
    edge_pct = row.get('edge_pct', 0)
    tier = row.get('confidence_tier', 'LOW')
    units = row.get('recommended_units', 0)

    direction_class = 'direction-over' if 'OVER' in str(pick).upper() else 'direction-under' if 'UNDER' in str(pick).upper() else ''
    bet_display = f"${units * 10:.0f}" if units else "-"

    return f'''
        <tr>
            <td>{game}</td>
            <td>{bet_type.title()}</td>
            <td><strong class="{direction_class}">{pick}</strong></td>
            <td>{market_line if market_line else '-'}</td>
            <td data-sort="{model_prob}">{get_confidence_badge(model_prob)}</td>
            <td data-sort="{edge_pct}">{get_edge_badge(edge_pct)}</td>
            <td><span class="badge {get_tier_badge_class(tier)}">{tier}</span></td>
            <td data-sort="{units}">{bet_display}</td>
        </tr>
    '''


def generate_game_line_card(game_df: pd.DataFrame) -> str:
    """Generate a professional game line card with team logos, structured bets, and collapsible analysis.

    Design principles:
    - Team logos and branding for visual scannability
    - Clear separation between spread and total picks
    - Confidence meters for quick assessment
    - Collapsible model analysis with structured sections
    - Color-coded indicators for over/under and team picks

    Args:
        game_df: DataFrame containing all bets for a single game
    """
    # Helper to get first non-null value from any row for a column
    def get_best_value(col, default=None):
        for _, row in game_df.iterrows():
            val = row.get(col)
            if val is not None and pd.notna(val) and val != '':
                return val
        return default

    # Convert EPA values to floats safely
    def safe_float_local(val, default=0):
        try:
            return float(val) if val is not None and pd.notna(val) else default
        except:
            return default

    # Get game info
    game = get_best_value('game', 'N/A')
    game_id = get_best_value('game_id', game.replace(' ', '_').replace('@', 'at'))

    # Parse teams from game string (format: "AWAY @ HOME")
    teams = game.split(' @ ')
    away_team = teams[0].strip() if len(teams) > 1 else ''
    home_team = teams[1].strip() if len(teams) > 1 else ''

    # Get team colors for accents
    away_color = TEAM_COLORS.get(away_team, ('#666', '#999'))[0]
    home_color = TEAM_COLORS.get(home_team, ('#666', '#999'))[0]

    # Get team logo URLs
    away_logo = TEAM_LOGOS.get(away_team, '')
    home_logo = TEAM_LOGOS.get(home_team, '')

    # Get model data - pull from any row that has it
    home_win_prob = get_best_value('home_win_prob', 0.5)
    home_epa = get_best_value('home_epa')
    away_epa = get_best_value('away_epa')
    home_def_epa = get_best_value('home_def_epa')
    away_def_epa = get_best_value('away_def_epa')

    # Get total projection data if available
    total_p50 = get_best_value('total_p50')
    total_p25 = get_best_value('total_p25')
    total_p75 = get_best_value('total_p75')

    # Best tier for the game card styling
    tier_order = {'ELITE': 0, 'HIGH': 1, 'STANDARD': 2, 'LOW': 3}
    best_tier = min(game_df['confidence_tier'].unique(), key=lambda x: tier_order.get(x, 4))
    tier_class = f'gl-tier-{best_tier.lower()}'

    # Get additional EPA data
    home_pace = get_best_value('home_pace')
    away_pace = get_best_value('away_pace')
    home_pass_epa = get_best_value('home_pass_epa')
    away_pass_epa = get_best_value('away_pass_epa')
    home_rush_epa = get_best_value('home_rush_epa')
    away_rush_epa = get_best_value('away_rush_epa')

    # Get ATS and defensive ranking data (added via enrichment)
    home_ats_record = get_best_value('home_ats_record', '')
    home_last6_ats = get_best_value('home_last6_ats', '')
    away_ats_record = get_best_value('away_ats_record', '')
    away_last6_ats = get_best_value('away_last6_ats', '')
    home_ou_record = get_best_value('home_ou_record', '')
    away_ou_record = get_best_value('away_ou_record', '')
    home_pass_def_rank = int(get_best_value('home_pass_def_rank', 16) or 16)
    home_rush_def_rank = int(get_best_value('home_rush_def_rank', 16) or 16)
    home_total_def_rank = int(get_best_value('home_total_def_rank', 16) or 16)
    away_pass_def_rank = int(get_best_value('away_pass_def_rank', 16) or 16)
    away_rush_def_rank = int(get_best_value('away_rush_def_rank', 16) or 16)
    away_total_def_rank = int(get_best_value('away_total_def_rank', 16) or 16)

    # Build team context HTML (expandable section)
    def get_def_rank_class(rank):
        if rank <= 10:
            return 'def-elite'
        elif rank <= 22:
            return 'def-average'
        return 'def-poor'

    has_ats_data = bool(home_ats_record or away_ats_record)
    team_context_html = ''
    if has_ats_data:
        team_context_html = f'''
        <div class="gl-team-context">
            <div class="gl-context-toggle" onclick="toggleTeamContext('{game_id}')">
                <span class="gl-toggle-icon" id="{game_id}-ctx-icon">▼</span>
                <span>Team Context</span>
            </div>
            <div class="gl-context-content" id="{game_id}-context">
                <div class="gl-ats-grid">
                    <div class="gl-team-ats">
                        <div class="gl-team-header">{away_team}</div>
                        <div class="gl-ats-row">
                            <span class="gl-ats-label">Season ATS:</span>
                            <span class="gl-ats-value">{away_ats_record}</span>
                        </div>
                        <div class="gl-ats-row">
                            <span class="gl-ats-label">Last 6 ATS:</span>
                            <span class="gl-ats-value">{away_last6_ats}</span>
                        </div>
                        <div class="gl-def-row">
                            <span class="gl-def-label">Pass DEF:</span>
                            <span class="gl-def-rank {get_def_rank_class(away_pass_def_rank)}">#{away_pass_def_rank}</span>
                        </div>
                        <div class="gl-def-row">
                            <span class="gl-def-label">Rush DEF:</span>
                            <span class="gl-def-rank {get_def_rank_class(away_rush_def_rank)}">#{away_rush_def_rank}</span>
                        </div>
                        <div class="gl-ou-row">
                            <span class="gl-ou-label">O/U:</span>
                            <span class="gl-ou-value">{away_ou_record}</span>
                        </div>
                    </div>
                    <div class="gl-team-ats">
                        <div class="gl-team-header">{home_team}</div>
                        <div class="gl-ats-row">
                            <span class="gl-ats-label">Season ATS:</span>
                            <span class="gl-ats-value">{home_ats_record}</span>
                        </div>
                        <div class="gl-ats-row">
                            <span class="gl-ats-label">Last 6 ATS:</span>
                            <span class="gl-ats-value">{home_last6_ats}</span>
                        </div>
                        <div class="gl-def-row">
                            <span class="gl-def-label">Pass DEF:</span>
                            <span class="gl-def-rank {get_def_rank_class(home_pass_def_rank)}">#{home_pass_def_rank}</span>
                        </div>
                        <div class="gl-def-row">
                            <span class="gl-def-label">Rush DEF:</span>
                            <span class="gl-def-rank {get_def_rank_class(home_rush_def_rank)}">#{home_rush_def_rank}</span>
                        </div>
                        <div class="gl-ou-row">
                            <span class="gl-ou-label">O/U:</span>
                            <span class="gl-ou-value">{home_ou_record}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''

    # Convert values
    home_epa_val = safe_float_local(home_epa)
    away_epa_val = safe_float_local(away_epa)
    home_def_val = safe_float_local(home_def_epa)
    away_def_val = safe_float_local(away_def_epa)
    home_pass_val = safe_float_local(home_pass_epa)
    away_pass_val = safe_float_local(away_pass_epa)
    home_rush_val = safe_float_local(home_rush_epa)
    away_rush_val = safe_float_local(away_rush_epa)
    home_pace_val = safe_float_local(home_pace)
    away_pace_val = safe_float_local(away_pace)
    home_win_prob_val = safe_float_local(home_win_prob, 0.5)
    away_win_prob = 1 - home_win_prob_val

    # Build SPREAD bet block
    spread_html = ''
    spread_bet = game_df[game_df['bet_type'] == 'spread']
    spread_edge_summary = ''
    if len(spread_bet) > 0:
        spread_row = spread_bet.iloc[0]
        spread_pick = spread_row.get('pick', '')
        market_line = safe_float_local(spread_row.get('market_line', 0))
        model_fair = safe_float_local(spread_row.get('model_fair_line', 0))
        spread_prob = safe_float_local(spread_row.get('model_prob', 0.5))
        spread_edge = safe_float_local(spread_row.get('edge_pct', 0))
        spread_tier = spread_row.get('confidence_tier', 'LOW')

        # Determine which team is being picked
        pick_team = spread_pick.split()[0] if spread_pick else ''
        pick_color = TEAM_COLORS.get(pick_team, ('#3b82f6', '#666'))[0]

        # Confidence meter (0-100%)
        conf_pct = int(spread_prob * 100)
        conf_class = 'conf-elite' if conf_pct >= 75 else 'conf-high' if conf_pct >= 65 else 'conf-standard'

        # Edge indicator
        line_diff = abs(model_fair - market_line)
        if '+' in str(spread_pick):
            spread_edge_summary = f"{line_diff:.1f} pts value" if line_diff >= 2 else "Slight value"
        else:
            spread_edge_summary = f"Line {line_diff:.1f} pts short" if line_diff >= 2 else "Fair value"

        spread_html = f'''
        <div class="gl-bet-block gl-spread" style="--pick-accent: {pick_color}">
            <div class="gl-bet-header">
                <span class="gl-bet-type">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 6L6 18M6 6l12 12"/>
                    </svg>
                    SPREAD
                </span>
                <span class="gl-bet-tier gl-tier-badge-{spread_tier.lower()}">{spread_tier}</span>
            </div>
            <div class="gl-bet-pick">{spread_pick}</div>
            <div class="gl-bet-meta">
                <div class="gl-conf-meter">
                    <div class="gl-conf-bar {conf_class}" style="width: {conf_pct}%"></div>
                    <span class="gl-conf-label">{conf_pct}%</span>
                </div>
                <div class="gl-edge-badge">{spread_edge:+.0f}% edge</div>
            </div>
            <div class="gl-bet-fair">Fair line: {model_fair:+.1f}</div>
        </div>
        '''

    # Build TOTAL bet block
    total_html = ''
    total_bet = game_df[game_df['bet_type'] == 'total']
    total_edge_summary = ''
    if len(total_bet) > 0:
        total_row = total_bet.iloc[0]
        total_pick = total_row.get('pick', '')
        total_line = safe_float_local(total_row.get('market_line', 0))
        total_prob = safe_float_local(total_row.get('model_prob', 0.5))
        total_edge = safe_float_local(total_row.get('edge_pct', 0))
        total_tier = total_row.get('confidence_tier', 'LOW')
        total_median = safe_float_local(total_p50)

        # Determine over/under styling
        is_over = 'OVER' in str(total_pick).upper()
        total_class = 'gl-over' if is_over else 'gl-under'
        total_icon = '↑' if is_over else '↓'

        # Confidence meter
        conf_pct = int(total_prob * 100)
        conf_class = 'conf-elite' if conf_pct >= 75 else 'conf-high' if conf_pct >= 65 else 'conf-standard'

        # Edge summary
        if total_median > 0:
            diff = total_median - total_line
            total_edge_summary = f"Proj: {total_median:.0f}" if total_median > 0 else ""

        total_html = f'''
        <div class="gl-bet-block gl-total {total_class}">
            <div class="gl-bet-header">
                <span class="gl-bet-type">
                    <span class="gl-total-icon">{total_icon}</span>
                    TOTAL
                </span>
                <span class="gl-bet-tier gl-tier-badge-{total_tier.lower()}">{total_tier}</span>
            </div>
            <div class="gl-bet-pick">{total_pick}</div>
            <div class="gl-bet-meta">
                <div class="gl-conf-meter">
                    <div class="gl-conf-bar {conf_class}" style="width: {conf_pct}%"></div>
                    <span class="gl-conf-label">{conf_pct}%</span>
                </div>
                <div class="gl-edge-badge">{total_edge:+.0f}% edge</div>
            </div>
            <div class="gl-bet-fair">{total_edge_summary}</div>
        </div>
        '''

    # Build INTUITIVE NARRATIVE showing how factors relate to create the edge

    # Helper functions for EPA interpretation
    def epa_desc(val, context="offense"):
        """Convert EPA to intuitive description."""
        if context == "offense":
            if val > 0.08: return "explosive"
            if val > 0.04: return "efficient"
            if val > 0.01: return "competent"
            if val > -0.02: return "inconsistent"
            if val > -0.05: return "struggling"
            return "anemic"
        else:  # defense - negative is good
            if val < -0.08: return "elite"
            if val < -0.04: return "stingy"
            if val < -0.01: return "solid"
            if val < 0.02: return "average"
            if val < 0.05: return "leaky"
            return "porous"

    def pace_desc(val):
        if val > 64: return "up-tempo"
        if val > 60: return "fast"
        if val > 56: return "moderate"
        return "slow"

    # Determine the picks we're explaining
    spread_pick_team = ''
    spread_pick_text = ''
    total_is_over = False
    total_pick_text = ''

    if len(spread_bet) > 0:
        spread_pick_text = spread_bet.iloc[0].get('pick', '')
        spread_pick_team = spread_pick_text.split()[0] if spread_pick_text else ''

    if len(total_bet) > 0:
        total_pick_text = total_bet.iloc[0].get('pick', '')
        total_is_over = 'OVER' in str(total_pick_text).upper()

    # Get total projection data
    total_median = safe_float_local(total_p50)
    total_low = safe_float_local(total_p25)
    total_high = safe_float_local(total_p75)
    total_variance = safe_float_local(get_best_value('total_std', 0))
    total_line_val = safe_float_local(total_bet.iloc[0].get('market_line', 0)) if len(total_bet) > 0 else 0

    # Build the narrative as a cohesive story
    narrative_parts = []

    # SPREAD NARRATIVE - Connect offense vs defense matchups
    if spread_pick_team:
        picked_is_home = spread_pick_team == home_team
        picked_team = spread_pick_team
        opponent = away_team if picked_is_home else home_team

        # Get relevant EPA values
        picked_off_epa = home_epa_val if picked_is_home else away_epa_val
        picked_pass_epa = home_pass_val if picked_is_home else away_pass_val
        picked_rush_epa = home_rush_val if picked_is_home else away_rush_val
        opp_def_epa = away_def_val if picked_is_home else home_def_val
        opp_off_epa = away_epa_val if picked_is_home else home_epa_val
        picked_def_epa = home_def_val if picked_is_home else away_def_val

        # Build spread story
        spread_story = f"<strong>Why {spread_pick_text}:</strong> "

        # Offensive advantage narrative
        off_advantage = picked_off_epa - opp_def_epa  # Positive = good matchup
        if off_advantage > 0.06:
            spread_story += f"{picked_team}'s {epa_desc(picked_off_epa)} offense ({picked_off_epa:+.3f} EPA) faces {opponent}'s {epa_desc(opp_def_epa, 'defense')} defense ({opp_def_epa:+.3f} EPA) — a favorable mismatch that should generate points. "
        elif off_advantage > 0.02:
            spread_story += f"{picked_team}'s {epa_desc(picked_off_epa)} attack has a slight edge against {opponent}'s {epa_desc(opp_def_epa, 'defense')} defense. "
        elif off_advantage < -0.04:
            spread_story += f"While {picked_team}'s offense ({epa_desc(picked_off_epa)}) faces a tough test against {opponent}'s {epa_desc(opp_def_epa, 'defense')} D, "

        # Defensive advantage narrative
        def_advantage = opp_off_epa - picked_def_epa  # Negative = picked team's D is better
        picked_def_is_good = picked_def_epa < 0.02  # At least "average" or better (for defense, lower EPA = better)
        opp_off_is_bad = opp_off_epa < -0.02  # Struggling or worse offense

        if def_advantage < -0.04 and picked_def_is_good:
            # Picked team has a GOOD defense that should contain opponent
            spread_story += f"Meanwhile, {picked_team}'s {epa_desc(picked_def_epa, 'defense')} defense should contain {opponent}'s {epa_desc(opp_off_epa)} offense. "
        elif def_advantage < -0.04 and opp_off_is_bad and not picked_def_is_good:
            # Picked team has weak defense, but opponent's offense is even worse
            spread_story += f"Despite {picked_team}'s {epa_desc(picked_def_epa, 'defense')} defense, {opponent}'s {epa_desc(opp_off_epa)} offense limits the damage. "
        elif def_advantage > 0.04 and off_advantage > 0:
            spread_story += f"The concern is {opponent}'s {epa_desc(opp_off_epa)} offense against {picked_team}'s {epa_desc(picked_def_epa, 'defense')} defense, but the offensive edge compensates. "

        # Win probability context
        picked_win_prob = home_win_prob_val if picked_is_home else away_win_prob

        # Get spread info for context
        spread_line = abs(float(spread_bet.iloc[0].get('market_line', 0))) if len(spread_bet) > 0 else 0
        fair_spread = float(spread_bet.iloc[0].get('model_fair_line', 0)) if len(spread_bet) > 0 else 0

        if picked_win_prob > 0.65:
            spread_story += f"The model gives {picked_team} a strong {picked_win_prob:.0%} win probability, suggesting the spread may not fully capture their advantage."
        elif picked_win_prob > 0.55:
            spread_story += f"At {picked_win_prob:.0%} win probability, {picked_team} is the rightful favorite — the value is in the number."
        elif picked_win_prob < 0.45:
            spread_story += f"Despite only a {picked_win_prob:.0%} win probability, the spread offers value as the market overestimates {opponent}."
        else:
            # Coin-flip game (45-55% win prob) - emphasize spread value
            spread_diff = spread_line - abs(fair_spread)
            if spread_diff > 3:
                spread_story += f"This is a coin-flip game ({picked_win_prob:.0%} win prob), but the market is giving {picked_team} {spread_line:.1f} points when the fair line is closer to {fair_spread:+.1f}. That's {spread_diff:.1f} points of value."
            else:
                spread_story += f"At {picked_win_prob:.0%} win probability, the value is in the spread cushion."

        narrative_parts.append(spread_story)

    # TOTAL NARRATIVE - Connect all scoring factors
    if total_pick_text and total_median > 0:
        total_story = f"<strong>Why {total_pick_text}:</strong> "

        # Calculate combined offensive and defensive context
        combined_off_epa = home_epa_val + away_epa_val
        combined_def_epa = home_def_val + away_def_val
        avg_pace = (home_pace_val + away_pace_val) / 2 if home_pace_val > 0 and away_pace_val > 0 else 60

        if total_is_over:
            # OVER narrative - emphasize scoring factors
            scoring_factors = []

            if combined_off_epa > 0.04:
                scoring_factors.append(f"both offenses are clicking ({home_team}: {home_epa_val:+.3f}, {away_team}: {away_epa_val:+.3f} EPA)")
            elif home_epa_val > 0.03 or away_epa_val > 0.03:
                hot_team = home_team if home_epa_val > away_epa_val else away_team
                hot_val = max(home_epa_val, away_epa_val)
                scoring_factors.append(f"{hot_team}'s {epa_desc(hot_val)} offense ({hot_val:+.3f} EPA) can carry scoring")

            if combined_def_epa > 0.02:
                scoring_factors.append(f"both defenses are vulnerable (combined {combined_def_epa:+.3f} EPA allowed)")
            elif home_def_val > 0.02 or away_def_val > 0.02:
                weak_team = home_team if home_def_val > away_def_val else away_team
                scoring_factors.append(f"{weak_team}'s defense is exploitable")

            if avg_pace > 62:
                scoring_factors.append(f"the {pace_desc(avg_pace)} tempo ({avg_pace:.0f} combined plays/game) increases possessions")

            if scoring_factors:
                total_story += " + ".join(scoring_factors[:3]).capitalize() + ". "

            # Projection context
            diff = total_median - total_line_val
            total_story += f"Model projects {total_median:.0f} points (range: {total_low:.0f}-{total_high:.0f}), {abs(diff):.1f} points above the {total_line_val:.0f} line."

        else:
            # UNDER narrative - emphasize defensive/pace factors
            limiting_factors = []

            if combined_def_epa < -0.04:
                limiting_factors.append(f"both defenses are strong ({home_team}: {home_def_val:+.3f}, {away_team}: {away_def_val:+.3f} EPA allowed)")
            elif home_def_val < -0.02 or away_def_val < -0.02:
                strong_team = home_team if home_def_val < away_def_val else away_team
                strong_val = min(home_def_val, away_def_val)
                limiting_factors.append(f"{strong_team}'s {epa_desc(strong_val, 'defense')} defense ({strong_val:+.3f} EPA) limits scoring")

            if combined_off_epa < 0:
                limiting_factors.append(f"neither offense is explosive ({home_team}: {home_epa_val:+.3f}, {away_team}: {away_epa_val:+.3f} EPA)")
            elif home_epa_val < -0.02 or away_epa_val < -0.02:
                weak_team = home_team if home_epa_val < away_epa_val else away_team
                limiting_factors.append(f"{weak_team}'s offense is struggling")

            if avg_pace < 58:
                limiting_factors.append(f"the {pace_desc(avg_pace)} pace ({avg_pace:.0f} plays/game) limits possessions")

            if limiting_factors:
                total_story += " + ".join(limiting_factors[:3]).capitalize() + ". "

            # Projection context
            diff = total_line_val - total_median
            total_story += f"Model projects {total_median:.0f} points (range: {total_low:.0f}-{total_high:.0f}), {abs(diff):.1f} points below the {total_line_val:.0f} line."

        # Add confidence context
        if total_variance < 10:
            total_story += " High model confidence in this projection."
        elif total_variance > 15:
            total_story += " Note: higher variance game — outcome could swing either way."

        narrative_parts.append(total_story)

    # WEATHER IMPACT - Explain how weather affects the pick
    wind_bucket = get_best_value('wind_bucket', 'calm')
    temp_bucket = get_best_value('temp_bucket', 'comfortable')
    weather_pass_mult = safe_float_local(get_best_value('weather_pass_mult', 1.0), 1.0)
    weather_rush_boost = safe_float_local(get_best_value('weather_rush_boost', 0.0), 0.0)
    is_dome = get_best_value('is_dome', False)
    precip_chance = safe_float_local(get_best_value('precip_chance', 0), 0)
    temperature = get_best_value('temperature')
    wind_speed = get_best_value('wind_speed')

    weather_story = ""
    if is_dome:
        weather_story = f"<strong>🏟️ Dome Game:</strong> Controlled environment gives a +3% passing efficiency boost — favorable for high-octane offenses."
    elif wind_bucket in ['high', 'extreme'] or temp_bucket in ['cold', 'extreme_cold'] or precip_chance > 0.4:
        weather_story = "<strong>🌧️ Weather Factor:</strong> "
        weather_factors = []

        if wind_bucket == 'extreme':
            weather_factors.append(f"extreme wind ({wind_speed:.0f} mph) reduces passing EPA to {weather_pass_mult:.0%}")
        elif wind_bucket == 'high':
            weather_factors.append(f"high wind ({wind_speed:.0f} mph) reduces passing EPA to {weather_pass_mult:.0%}")

        if temp_bucket == 'extreme_cold':
            weather_factors.append(f"extreme cold ({temperature:.0f}°F) further suppresses passing")
        elif temp_bucket == 'cold':
            weather_factors.append(f"cold conditions ({temperature:.0f}°F) impact ball handling")

        if precip_chance > 0.5:
            weather_factors.append(f"high precipitation chance ({precip_chance:.0%}) adds uncertainty")
        elif precip_chance > 0.3:
            weather_factors.append(f"precipitation risk ({precip_chance:.0%})")

        if weather_rush_boost > 0.03:
            weather_factors.append(f"rushing gets +{weather_rush_boost:.0%} volume boost")

        if weather_factors:
            weather_story += " + ".join(weather_factors[:3]).capitalize() + ". "
            # Add strategic implication
            if not total_is_over and total_pick_text:
                weather_story += "This supports the UNDER — bad weather historically suppresses scoring."
            elif total_is_over and total_pick_text:
                weather_story += "The OVER pick fights against these conditions — higher risk play."

    if weather_story:
        narrative_parts.append(weather_story)

    # DEPTH CHART / PERSONNEL - Explain starter/backup situations
    home_qb_backup = get_best_value('home_qb_backup_active', False)
    away_qb_backup = get_best_value('away_qb_backup_active', False)
    home_rb_backup = get_best_value('home_rb_backup_active', False)
    away_rb_backup = get_best_value('away_rb_backup_active', False)
    home_qb = get_best_value('home_qb', '')
    away_qb = get_best_value('away_qb', '')
    home_qb_epa = safe_float_local(get_best_value('home_qb_epa', 0))
    away_qb_epa = safe_float_local(get_best_value('away_qb_epa', 0))

    personnel_story = ""
    personnel_factors = []

    if home_qb_backup:
        personnel_factors.append(f"{home_team} starting backup QB {home_qb} (EPA: {home_qb_epa:+.3f})")
    if away_qb_backup:
        personnel_factors.append(f"{away_team} starting backup QB {away_qb} (EPA: {away_qb_epa:+.3f})")
    if home_rb_backup:
        personnel_factors.append(f"{home_team} RB1 out — backup takes over")
    if away_rb_backup:
        personnel_factors.append(f"{away_team} RB1 out — backup takes over")

    if personnel_factors:
        personnel_story = "<strong>📋 Personnel Alert:</strong> " + " • ".join(personnel_factors) + ". "
        # Add strategic implication for spread pick
        if spread_pick_team:
            picked_has_backup = (spread_pick_team == home_team and home_qb_backup) or (spread_pick_team == away_team and away_qb_backup)
            opp_has_backup = (spread_pick_team == home_team and away_qb_backup) or (spread_pick_team == away_team and home_qb_backup)
            if opp_has_backup and not picked_has_backup:
                personnel_story += f"This supports {spread_pick_team} — they face a weakened opponent."
            elif picked_has_backup and not opp_has_backup:
                personnel_story += "Despite the backup situation, the spread still offers value."

    if personnel_story:
        narrative_parts.append(personnel_story)

    # Fallback if no picks to explain
    if not narrative_parts:
        win_leader = home_team if home_win_prob_val > 0.5 else away_team
        narrative_parts.append(f"<strong>Game Context:</strong> {win_leader} is favored at {max(home_win_prob_val, away_win_prob):.0%} win probability. Limited edge identified in current lines.")

    # Combine into factors_html as narrative paragraphs
    factors_html = '<div class="gl-narrative">'
    for part in narrative_parts:
        factors_html += f'<p class="gl-narrative-para">{part}</p>'
    factors_html += '</div>'

    # Build THE EDGE summary (shorter header line)
    edge_summary = ''
    if home_win_prob_val > 0.70:
        edge_summary = f"Significant edge: {home_team} dominates model projections"
    elif away_win_prob > 0.70:
        edge_summary = f"Road upset alert: {away_team} heavily favored by model"
    elif home_win_prob_val > 0.60:
        edge_summary = f"{home_team} is solid favorite — look for spread value"
    elif away_win_prob > 0.60:
        edge_summary = f"{away_team} road favorite — contrarian opportunity"
    elif abs(home_win_prob_val - 0.5) < 0.08:
        edge_summary = f"Coin-flip game — value lies in total and situational spots"
    else:
        leader = home_team if home_win_prob_val > 0.5 else away_team
        edge_summary = f"Lean {leader} — tight matchup with potential market overreaction"

    # Build projection visual if we have total data
    projection_html = ''
    total_median = safe_float_local(total_p50)
    total_low = safe_float_local(total_p25)
    total_high = safe_float_local(total_p75)
    if total_median > 0 and len(total_bet) > 0:
        total_line = safe_float_local(total_bet.iloc[0].get('market_line', 0))
        if total_line > 0:
            # Calculate positions for visual
            range_min = min(total_low, total_line) - 5
            range_max = max(total_high, total_line) + 5
            range_span = range_max - range_min
            line_pos = ((total_line - range_min) / range_span) * 100
            median_pos = ((total_median - range_min) / range_span) * 100

            projection_html = f'''
            <div class="gl-projection">
                <div class="gl-proj-label">Total Projection</div>
                <div class="gl-proj-bar">
                    <div class="gl-proj-range" style="left: {((total_low - range_min) / range_span) * 100}%; width: {((total_high - total_low) / range_span) * 100}%"></div>
                    <div class="gl-proj-line" style="left: {line_pos}%" title="Line: {total_line}"></div>
                    <div class="gl-proj-median" style="left: {median_pos}%" title="Model: {total_median:.0f}"></div>
                </div>
                <div class="gl-proj-labels">
                    <span>{total_low:.0f}</span>
                    <span class="gl-proj-line-label">Line: {total_line:.0f}</span>
                    <span>{total_high:.0f}</span>
                </div>
            </div>
            '''

    # Unique ID for collapsible
    card_id = f"gl-{game_id.replace(' ', '-').replace('@', 'at')}"

    return f'''
    <div class="gl-card {tier_class}" data-tier="{best_tier}" data-game="{game}">
        <div class="gl-card-header">
            <div class="gl-matchup">
                <div class="gl-team gl-away">
                    <img src="{away_logo}" alt="{away_team}" class="gl-team-logo" onerror="this.style.display='none'">
                    <span class="gl-team-abbr">{away_team}</span>
                </div>
                <div class="gl-vs">@</div>
                <div class="gl-team gl-home">
                    <img src="{home_logo}" alt="{home_team}" class="gl-team-logo" onerror="this.style.display='none'">
                    <span class="gl-team-abbr">{home_team}</span>
                </div>
            </div>
            <div class="gl-header-meta">
                <span class="gl-tier-badge gl-tier-badge-{best_tier.lower()}">{best_tier}</span>
            </div>
        </div>

        <div class="gl-bets-container">
            {spread_html}
            {total_html}
        </div>

        {team_context_html}

        <div class="gl-analysis-toggle" onclick="toggleGameAnalysis('{card_id}')">
            <span class="gl-toggle-icon" id="{card_id}-icon">▼</span>
            <span>Model Analysis</span>
        </div>

        <div class="gl-analysis" id="{card_id}-analysis" style="display: none;">
            <div class="gl-edge-summary">
                <div class="gl-edge-title">The Edge</div>
                <div class="gl-edge-text">{edge_summary}</div>
            </div>

            {projection_html}

            <div class="gl-factors">
                <div class="gl-factors-title">Full Model Analysis</div>
                {factors_html if factors_html else '<div class="gl-narrative-para">Limited model context available for this matchup.</div>'}
            </div>
        </div>
    </div>
    '''


def load_tracking_data() -> dict:
    """Load bet tracking history for dashboard display."""
    tracking_file = PROJECT_ROOT / "data" / "tracking" / "bet_results_history.json"
    if not tracking_file.exists():
        return None

    import json
    with open(tracking_file, 'r') as f:
        data = json.load(f)

    if not data.get('bets'):
        return None

    # Helper to handle string booleans from JSON
    def is_true(val):
        """Check if value is truthy (handles string 'True' from JSON)."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == 'true'
        return False

    # Process into summary stats
    bets = data['bets']
    weekly_summaries = data.get('weekly_summaries', {})

    # Calculate overall stats
    total_bets = len(bets)
    wins = sum(1 for b in bets if is_true(b.get('bet_won')))
    losses = sum(1 for b in bets if not is_true(b.get('bet_won')) and not is_true(b.get('is_push')))
    pushes = sum(1 for b in bets if is_true(b.get('is_push')))
    total_profit = sum(b.get('profit', 0) for b in bets if isinstance(b.get('profit'), (int, float)))

    # By market breakdown
    markets = {}
    for bet in bets:
        market = bet.get('market', 'unknown')
        if market not in markets:
            markets[market] = {'bets': 0, 'wins': 0, 'losses': 0, 'profit': 0}
        markets[market]['bets'] += 1
        if is_true(bet.get('bet_won')):
            markets[market]['wins'] += 1
        elif not is_true(bet.get('is_push')):
            markets[market]['losses'] += 1
        if isinstance(bet.get('profit'), (int, float)):
            markets[market]['profit'] += bet['profit']

    # Running P&L by bet
    running_pnl = []
    cumulative = 0
    for bet in sorted(bets, key=lambda x: (x.get('week', 0), x.get('tracked_at', ''))):
        if isinstance(bet.get('profit'), (int, float)):
            cumulative += bet['profit']
            running_pnl.append({
                'week': bet.get('week'),
                'player': bet.get('player'),
                'profit': bet.get('profit'),
                'cumulative': cumulative
            })

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'total_profit': total_profit,
        'roi': (total_profit / total_bets * 100) if total_bets > 0 else 0,
        'markets': markets,
        'weekly_summaries': weekly_summaries,
        'running_pnl': running_pnl,
        'bets': bets
    }


def export_picks_json(recs_df: pd.DataFrame, week: int, season: int = 2025,
                      game_lines_df: pd.DataFrame = None, parlays_df: pd.DataFrame = None) -> Path:
    """Export picks data to JSON for Next.js dashboard.

    Outputs a JSON file in the format expected by the Next.js dashboard at
    deploy/src/data/picks.json
    """
    if len(recs_df) == 0:
        print("No picks to export to JSON")
        return None

    # Filter to player props only (exclude game lines)
    if 'bet_category' in recs_df.columns:
        player_props_df = recs_df[recs_df['bet_category'] != 'GAME_LINE'].copy()
    else:
        player_props_df = recs_df.copy()

    # Calculate stats
    if 'effective_tier' in player_props_df.columns:
        elite_count = len(player_props_df[player_props_df['effective_tier'] == 'ELITE'])
        strong_count = len(player_props_df[player_props_df['effective_tier'] == 'STRONG'])
    else:
        elite_count = 0
        strong_count = 0
    avg_edge = player_props_df['edge_pct'].mean() if 'edge_pct' in player_props_df.columns else 0

    # Get unique games count
    if 'game' in player_props_df.columns:
        games_count = player_props_df['game'].nunique()
    elif 'opponent' in player_props_df.columns:
        games_count = player_props_df['opponent'].nunique()
    else:
        games_count = 13  # Default for full week

    # Load game history data for picks
    game_history_cache = {}
    name_to_player_id = {}  # Map player names to IDs
    try:
        stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if stats_path.exists():
            weekly_stats = pd.read_parquet(stats_path)
            # Filter to current season and recent weeks
            weekly_stats = weekly_stats[
                (weekly_stats['season'] == season) &
                (weekly_stats['week'] < week)
            ].sort_values('week', ascending=False)

            # Create name -> player_id lookup from weekly stats
            for _, row in weekly_stats.drop_duplicates('player_id').iterrows():
                if pd.notna(row.get('player_id')) and row['player_id']:
                    name = row.get('player_display_name', row.get('player_name', ''))
                    if name:
                        name_to_player_id[name] = row['player_id']

            # Try to get player_ids from dataframe or from name lookup
            player_ids = set()
            if 'player_id' in player_props_df.columns:
                player_ids = set(player_props_df['player_id'].dropna().unique())
            else:
                # Lookup from player names
                for player_name in player_props_df['player'].dropna().unique():
                    pid = name_to_player_id.get(player_name)
                    if pid:
                        player_ids.add(pid)

            # Group by player_id for lookup
            for player_id in player_ids:
                if player_id and player_id != '':
                    player_stats = weekly_stats[weekly_stats['player_id'] == player_id].head(6)
                    if len(player_stats) > 0:
                        game_history_cache[player_id] = {
                            'weeks': player_stats['week'].tolist(),
                            'opponents': player_stats.get('opponent_team', player_stats.get('opponent', pd.Series(['UNK']*len(player_stats)))).tolist(),
                            'receiving_yards': player_stats['receiving_yards'].fillna(0).astype(int).tolist() if 'receiving_yards' in player_stats.columns else [],
                            'receptions': player_stats['receptions'].fillna(0).astype(int).tolist() if 'receptions' in player_stats.columns else [],
                            'rushing_yards': player_stats['rushing_yards'].fillna(0).astype(int).tolist() if 'rushing_yards' in player_stats.columns else [],
                            'rushing_attempts': player_stats['carries'].fillna(0).astype(int).tolist() if 'carries' in player_stats.columns else [],
                            'passing_yards': player_stats['passing_yards'].fillna(0).astype(int).tolist() if 'passing_yards' in player_stats.columns else [],
                            'passing_attempts': player_stats['attempts'].fillna(0).astype(int).tolist() if 'attempts' in player_stats.columns else [],
                            'completions': player_stats['completions'].fillna(0).astype(int).tolist() if 'completions' in player_stats.columns else [],
                            'passing_tds': player_stats['passing_tds'].fillna(0).astype(int).tolist() if 'passing_tds' in player_stats.columns else [],
                            'rushing_tds': player_stats['rushing_tds'].fillna(0).astype(int).tolist() if 'rushing_tds' in player_stats.columns else [],
                            'receiving_tds': player_stats['receiving_tds'].fillna(0).astype(int).tolist() if 'receiving_tds' in player_stats.columns else [],
                        }
    except Exception as e:
        print(f"  Warning: Could not load game history: {e}")

    # Load player headshot URLs from rosters
    headshot_lookup = {}
    try:
        rosters_path = PROJECT_ROOT / 'data' / 'nflverse' / 'rosters.parquet'
        if rosters_path.exists():
            rosters_df = pd.read_parquet(rosters_path)
            # Filter to current season for most accurate data
            if 'season' in rosters_df.columns:
                rosters_df = rosters_df[rosters_df['season'] == season]
            # Build lookup by full_name -> headshot_url
            for _, row in rosters_df.iterrows():
                name = str(row.get('full_name', '')).strip()
                url = row.get('headshot_url', '')
                if name and pd.notna(url) and url:
                    headshot_lookup[name] = url
            print(f"  Loaded {len(headshot_lookup)} player headshots from rosters")
    except Exception as e:
        print(f"  Warning: Could not load headshots from rosters: {e}")

    # Load depth chart for player depth positions (WR1, RB2, etc.)
    depth_chart_lookup = {}  # (player_name, team) -> depth_position
    try:
        # Use canonical depth chart loader
        from nfl_quant.data.depth_chart_loader import get_depth_charts
        depth_df = get_depth_charts()

        if not depth_df.empty:
            # Only get offensive skill positions
            skill_positions = ['Wide Receiver', 'Running Back', 'Tight End', 'Quarterback']
            if 'pos_name' in depth_df.columns:
                depth_df = depth_df[depth_df['pos_name'].isin(skill_positions)]

            # Keep only the most recent entry for each (player, team, position)
            if 'player_name' in depth_df.columns and 'team' in depth_df.columns and 'pos_name' in depth_df.columns:
                depth_df = depth_df.drop_duplicates(subset=['player_name', 'team', 'pos_name'], keep='first')

            # Build lookup: (player_name, team) -> "WR1", "RB2", etc.
            pos_abbrev_map = {
                'Wide Receiver': 'WR',
                'Running Back': 'RB',
                'Tight End': 'TE',
                'Quarterback': 'QB',
            }
            for _, row in depth_df.iterrows():
                name = str(row.get('player_name', '')).strip()
                team = str(row.get('team', '')).upper()
                pos_name = row.get('pos_name', '')
                pos_rank = int(row.get('pos_rank', 1)) if pd.notna(row.get('pos_rank')) else 1
                if name and team and pos_name:
                    abbrev = pos_abbrev_map.get(pos_name, 'FLEX')
                    depth_chart_lookup[(name, team)] = f"{abbrev}{pos_rank}"
            print(f"  Loaded {len(depth_chart_lookup)} depth chart positions")
    except Exception as e:
        print(f"  Warning: Could not load depth chart: {e}")

    # Calculate defense allowed by MARKET (how much each team allows for each stat type)
    # Key: team -> market -> {'allowed': float, 'rank': int}
    defense_vs_market = {}
    # Weekly defense: team -> week -> market -> total_allowed (for chart history)
    defense_weekly = {}
    # Team -> week -> opponent lookup (for showing who defense played each week)
    team_week_opponent = {}
    try:
        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
        if schedule_path.exists():
            sched_df = pd.read_parquet(schedule_path)
            sched_season = sched_df[sched_df['season'] == season]
            for _, game in sched_season.iterrows():
                wk = int(game['week'])
                away = str(game['away_team'])
                home = str(game['home_team'])
                # Home team played away team, away team played home team
                if home not in team_week_opponent:
                    team_week_opponent[home] = {}
                if away not in team_week_opponent:
                    team_week_opponent[away] = {}
                team_week_opponent[home][wk] = away
                team_week_opponent[away][wk] = home
    except Exception as e:
        print(f"  Warning: Could not load schedule for defense opponents: {e}")

    # Load game context for weather and vegas data
    # Key: (team, opponent) -> {'vegas_total', 'vegas_spread', 'roof', 'temp', 'wind'}
    game_context_lookup = {}
    try:
        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
        if schedule_path.exists():
            sched_df = pd.read_parquet(schedule_path)
            week_games = sched_df[(sched_df['week'] == week) & (sched_df['season'] == season)]
            for _, game in week_games.iterrows():
                away = str(game.get('away_team', ''))
                home = str(game.get('home_team', ''))
                spread_line = float(game.get('spread_line', 0)) if pd.notna(game.get('spread_line')) else None
                total_line = float(game.get('total_line', 0)) if pd.notna(game.get('total_line')) else None
                roof = str(game.get('roof', '')) if pd.notna(game.get('roof')) else None
                temp = int(game.get('temp', 0)) if pd.notna(game.get('temp')) else None
                wind = int(game.get('wind', 0)) if pd.notna(game.get('wind')) else None

                # Away team context (spread is from home perspective, so flip for away)
                game_context_lookup[(away, home)] = {
                    'vegas_total': total_line,
                    'vegas_spread': -spread_line if spread_line else None,  # Flip for away team
                    'roof': roof,
                    'temp': temp,
                    'wind': wind,
                }
                # Home team context
                game_context_lookup[(home, away)] = {
                    'vegas_total': total_line,
                    'vegas_spread': spread_line,
                    'roof': roof,
                    'temp': temp,
                    'wind': wind,
                }
            print(f"  Loaded game context for {len(week_games)} games")
    except Exception as e:
        print(f"  Warning: Could not load game context: {e}")

    # Map markets to their stat columns and which positions they apply to
    # Column names from NFLverse weekly_stats.parquet
    MARKET_STAT_MAP = {
        'player_reception_yds': {'col': 'receiving_yards', 'positions': ['WR', 'TE', 'RB']},
        'player_receptions': {'col': 'receptions', 'positions': ['WR', 'TE', 'RB']},
        'player_rush_yds': {'col': 'rushing_yards', 'positions': ['RB', 'QB', 'WR']},
        'player_rush_attempts': {'col': 'carries', 'positions': ['RB', 'QB']},  # NFLverse uses 'carries'
        'player_pass_yds': {'col': 'passing_yards', 'positions': ['QB']},
        'player_pass_attempts': {'col': 'attempts', 'positions': ['QB']},  # NFLverse uses 'attempts'
        'player_pass_completions': {'col': 'completions', 'positions': ['QB']},
        # TD markets - count TDs allowed by defense
        'player_pass_tds': {'col': 'passing_tds', 'positions': ['QB']},
        # Anytime TD - position-specific (RBs can score rushing OR receiving TDs)
        'player_anytime_td_RB': {'col': None, 'positions': ['RB'], 'aggregate': ['rushing_tds', 'receiving_tds']},
        'player_anytime_td_WR': {'col': 'receiving_tds', 'positions': ['WR']},
        'player_anytime_td_TE': {'col': 'receiving_tds', 'positions': ['TE']},
        'player_anytime_td_QB': {'col': None, 'positions': ['QB'], 'aggregate': ['rushing_tds', 'passing_tds']},  # QB rushing + passing TDs
    }

    # Load depth charts to get player depth positions (WR1, RB2, etc.)
    # IMPORTANT: Use WEEK-SPECIFIC depth charts for historical defense calculation
    player_depth_lookup = {}  # player_id -> pos_rank (for current week picks)
    player_name_depth = {}    # (player_name, team) -> pos_rank (for current week picks)
    week_team_depth = {}      # {week: {team: {position: [(player_name, pos_rank), ...]}}}

    try:
        # Load raw 2025 depth charts with timestamps for week-specific lookups
        depth_path = NFLVERSE_DIR / 'depth_charts_2025.parquet'
        if depth_path.exists():
            raw_depth = pd.read_parquet(depth_path)
            raw_depth['dt_parsed'] = pd.to_datetime(raw_depth['dt'], utc=True)

            # Get schedule to map dates -> weeks
            sched_path = NFLVERSE_DIR / 'schedules.parquet'
            if sched_path.exists():
                sched = pd.read_parquet(sched_path)
                sched_2025 = sched[sched['season'] == 2025].copy()

                # Build week boundaries from schedule
                week_dates = {}
                for week in sorted(sched_2025['week'].unique()):
                    week_games = sched_2025[sched_2025['week'] == week]
                    if 'gameday' in week_games.columns:
                        min_date = pd.to_datetime(week_games['gameday'].min())
                        week_dates[week] = min_date

                # For each week, get depth chart snapshot from that week
                for week, game_date in week_dates.items():
                    game_dt = pd.Timestamp(game_date, tz='UTC')
                    # Get snapshots before this week's games
                    valid_snapshots = raw_depth[raw_depth['dt_parsed'] <= game_dt]

                    if len(valid_snapshots) == 0:
                        continue

                    week_team_depth[week] = {}

                    for team in valid_snapshots['team'].unique():
                        team_data = valid_snapshots[valid_snapshots['team'] == team]
                        # Get most recent snapshot for this team before the game
                        latest_dt = team_data['dt_parsed'].max()
                        team_latest = team_data[team_data['dt_parsed'] == latest_dt]

                        week_team_depth[week][team] = {}

                        for pos in ['TE', 'WR', 'RB', 'QB']:
                            pos_players = team_latest[team_latest['pos_abb'] == pos]
                            if len(pos_players) > 0:
                                pos_players = pos_players.sort_values('pos_rank')
                                week_team_depth[week][team][pos] = [
                                    (row['player_name'], int(row['pos_rank']))
                                    for _, row in pos_players.iterrows()
                                ]

                print(f"  Built week-specific depth charts for {len(week_team_depth)} weeks")

        # Also load current depth chart for pick display
        from nfl_quant.data.depth_chart_loader import get_depth_charts
        depth_df = get_depth_charts()

        if not depth_df.empty:
            pos_map = {'Wide Receiver': 'WR', 'Running Back': 'RB', 'Tight End': 'TE', 'Quarterback': 'QB'}
            if 'pos_name' in depth_df.columns:
                depth_df = depth_df[depth_df['pos_name'].isin(pos_map.keys())]
            if 'player_name' in depth_df.columns and 'pos_rank' in depth_df.columns:
                depth_df = depth_df[depth_df['player_name'].notna() & depth_df['pos_rank'].notna()]

            for _, row in depth_df.iterrows():
                pid = row.get('gsis_id', '')
                pname = row.get('player_name', '')
                team = row.get('team', '')
                pos_rank = int(row['pos_rank']) if pd.notna(row.get('pos_rank')) else 1
                pos = pos_map.get(row.get('pos_name', ''), '')

                if pid and pid not in player_depth_lookup:
                    player_depth_lookup[pid] = {'pos_rank': pos_rank, 'position': pos}
                key = (pname, team)
                if key not in player_name_depth:
                    player_name_depth[key] = {'pos_rank': pos_rank, 'position': pos}

            print(f"  Loaded depth positions for {len(player_depth_lookup)} players (by ID), {len(player_name_depth)} (by name)")
    except Exception as e:
        import traceback
        print(f"  Warning: Could not load depth charts: {e}")
        traceback.print_exc()

    def get_player_depth_for_week(player_name, team, position, week):
        """Get player's depth position for a specific week based on THAT week's depth chart."""
        if week not in week_team_depth:
            return None
        if team not in week_team_depth[week]:
            return None
        if position not in week_team_depth[week][team]:
            return None

        # Find this player in that week's depth chart
        for pname, rank in week_team_depth[week][team][position]:
            if player_name and pname:
                # Match on last name
                p_last = player_name.split()[-1].lower()
                d_last = pname.split()[-1].lower()
                if p_last == d_last:
                    return rank
        return None

    # Add pos_rank to weekly_stats using WEEK-SPECIFIC depth charts
    def get_player_depth(row):
        """Get player's depth position for their specific week."""
        pname = row.get('player_display_name', row.get('player_name', ''))
        team = row.get('team', '')
        position = row.get('position', '')
        week = row.get('week', 0)

        # Use week-specific depth chart
        week_rank = get_player_depth_for_week(pname, team, position, week)
        if week_rank is not None:
            return week_rank

        # Fallback to current depth chart only for current week
        pid = row.get('player_id', '')
        if pid and pid in player_depth_lookup:
            return player_depth_lookup[pid]['pos_rank']
        key = (pname, team)
        if key in player_name_depth:
            return player_name_depth[key]['pos_rank']
        return None

    try:
        if stats_path.exists():
            # Add depth position to weekly stats
            weekly_stats = weekly_stats.copy()
            weekly_stats['pos_rank'] = weekly_stats.apply(get_player_depth, axis=1)
            depth_matched = weekly_stats['pos_rank'].notna().sum()
            print(f"  Matched depth position for {depth_matched}/{len(weekly_stats)} stat lines")

            for opp_team in weekly_stats['opponent_team'].dropna().unique():
                team_games = weekly_stats[weekly_stats['opponent_team'] == opp_team]
                defense_vs_market[opp_team] = {}
                defense_weekly[opp_team] = {}

                for market, config in MARKET_STAT_MAP.items():
                    col = config['col']
                    positions = config['positions']
                    aggregate_cols = config.get('aggregate', None)

                    # Filter to relevant positions
                    pos_stats = team_games[team_games['position'].isin(positions)]
                    if len(pos_stats) == 0:
                        continue

                    # Get all weeks this opponent played
                    opp_weeks = set(pos_stats['week'].unique())

                    # For each depth position (1, 2, 3), calculate defense allowed
                    for depth_rank in [1, 2, 3]:
                        # Filter to this depth position only
                        depth_stats = pos_stats[pos_stats['pos_rank'] == depth_rank]

                        # Get weeks where this depth rank had stats
                        weeks_with_stats = set(depth_stats['week'].unique()) if len(depth_stats) > 0 else set()

                        # Market key includes depth: "player_receptions_WR_1" for WR1s
                        # For position-specific markets, include position in key
                        if len(positions) == 1:
                            depth_market_key = f"{market}_{depth_rank}"
                        else:
                            # For multi-position markets (receptions: WR/TE/RB), need position in key
                            for pos in positions:
                                pos_depth_stats = depth_stats[depth_stats['position'] == pos]
                                depth_market_key = f"{market}_{pos}_{depth_rank}"

                                # If depth1 had no stats but the position had players, record 0
                                if len(pos_depth_stats) == 0:
                                    if depth_rank == 1:
                                        pos_played_weeks = set(pos_stats[pos_stats['position'] == pos]['week'].unique())
                                        for wk in pos_played_weeks:
                                            if wk not in defense_weekly[opp_team]:
                                                defense_weekly[opp_team][wk] = {}
                                            if depth_market_key not in defense_weekly[opp_team][wk]:
                                                defense_weekly[opp_team][wk][depth_market_key] = 0.0
                                    continue

                                if aggregate_cols:
                                    valid_cols = [c for c in aggregate_cols if c in pos_depth_stats.columns]
                                    if not valid_cols:
                                        continue
                                    pos_depth_stats = pos_depth_stats.copy()
                                    pos_depth_stats['_agg_total'] = pos_depth_stats[valid_cols].fillna(0).sum(axis=1)
                                    weekly_totals = pos_depth_stats.groupby('week')['_agg_total'].sum()
                                elif col and col in pos_depth_stats.columns:
                                    weekly_totals = pos_depth_stats.groupby('week')[col].sum()
                                else:
                                    continue

                                if len(weekly_totals) > 0:
                                    for wk, total in weekly_totals.items():
                                        if wk not in defense_weekly[opp_team]:
                                            defense_weekly[opp_team][wk] = {}
                                        defense_weekly[opp_team][wk][depth_market_key] = round(float(total), 1)

                                # For weeks where this team played but depth1 had no stats, record 0
                                # (Only for depth_rank=1, as starter should have played)
                                if depth_rank == 1:
                                    pos_weeks_with_stats = set(weekly_totals.keys()) if len(weekly_totals) > 0 else set()
                                    # Get weeks where this position had ANY stats (someone played)
                                    pos_played_weeks = set(pos_stats[pos_stats['position'] == pos]['week'].unique())
                                    for wk in pos_played_weeks:
                                        if wk not in pos_weeks_with_stats:
                                            # Week where position had players but TE1 had no stats = 0
                                            if wk not in defense_weekly[opp_team]:
                                                defense_weekly[opp_team][wk] = {}
                                            if depth_market_key not in defense_weekly[opp_team][wk]:
                                                defense_weekly[opp_team][wk][depth_market_key] = 0.0
                            continue  # Already handled multi-position case

                        # Single position case
                        if len(depth_stats) == 0:
                            # If depth1 had no stats but the position had players, record 0
                            if depth_rank == 1:
                                pos_played_weeks = set(pos_stats['week'].unique())
                                for wk in pos_played_weeks:
                                    if wk not in defense_weekly[opp_team]:
                                        defense_weekly[opp_team][wk] = {}
                                    if depth_market_key not in defense_weekly[opp_team][wk]:
                                        defense_weekly[opp_team][wk][depth_market_key] = 0.0
                            continue

                        if aggregate_cols:
                            valid_cols = [c for c in aggregate_cols if c in depth_stats.columns]
                            if not valid_cols:
                                continue
                            depth_stats = depth_stats.copy()
                            depth_stats['_agg_total'] = depth_stats[valid_cols].fillna(0).sum(axis=1)
                            weekly_totals = depth_stats.groupby('week')['_agg_total'].sum()
                        elif col and col in depth_stats.columns:
                            weekly_totals = depth_stats.groupby('week')[col].sum()
                        else:
                            continue

                        if len(weekly_totals) > 0:
                            for wk, total in weekly_totals.items():
                                if wk not in defense_weekly[opp_team]:
                                    defense_weekly[opp_team][wk] = {}
                                defense_weekly[opp_team][wk][depth_market_key] = round(float(total), 1)

                            # For weeks where position played but depth1 had no stats, record 0
                            if depth_rank == 1:
                                weeks_with_stats = set(weekly_totals.keys())
                                pos_played_weeks = set(pos_stats['week'].unique())
                                for wk in pos_played_weeks:
                                    if wk not in weeks_with_stats:
                                        if wk not in defense_weekly[opp_team]:
                                            defense_weekly[opp_team][wk] = {}
                                        if depth_market_key not in defense_weekly[opp_team][wk]:
                                            defense_weekly[opp_team][wk][depth_market_key] = 0.0

                    # Note: We only store depth-specific keys (e.g., player_receptions_TE_1)
                    # No fallback keys are stored - if exact data isn't available, show None

                    # Also store non-depth-filtered totals for aggregate stats (original market key)
                    if aggregate_cols:
                        valid_cols = [c for c in aggregate_cols if c in pos_stats.columns]
                        if not valid_cols:
                            continue
                        pos_stats = pos_stats.copy()
                        pos_stats['_agg_total'] = pos_stats[valid_cols].fillna(0).sum(axis=1)
                        weekly_totals = pos_stats.groupby('week')['_agg_total'].sum()
                    elif col:
                        if col not in pos_stats.columns:
                            continue
                        weekly_totals = pos_stats.groupby('week')[col].sum()
                    else:
                        continue

                    if len(weekly_totals) > 0:
                        avg_allowed = weekly_totals.mean()
                        defense_vs_market[opp_team][market] = {'allowed': round(avg_allowed, 1)}
                        for wk, total in weekly_totals.items():
                            if wk not in defense_weekly[opp_team]:
                                defense_weekly[opp_team][wk] = {}
                            defense_weekly[opp_team][wk][market] = round(float(total), 1)

            print(f"  Calculated defense vs market stats for {len(defense_vs_market)} teams (with depth filtering)")
    except Exception as e:
        import traceback
        print(f"  Warning: Could not calculate defense vs position: {e}")
        traceback.print_exc()

    # Team logo URL mapping - ESPN uses different abbreviations for some teams
    ESPN_TEAM_MAP = {
        'LA': 'lar',    # Rams
        'WSH': 'wsh',   # Commanders
        'JAC': 'jax',   # Jaguars (alternate abbr)
    }
    def get_team_logo_url(team: str) -> str:
        team_abbr = str(team).upper() if team else 'NFL'
        espn_abbr = ESPN_TEAM_MAP.get(team_abbr, team_abbr.lower())
        return f'https://a.espncdn.com/i/teamlogos/nfl/500/{espn_abbr}.png'

    # Normalize player name for matching (strip Jr., Sr., III, etc.)
    def normalize_name(name: str) -> str:
        if not name:
            return ''
        # Remove common suffixes
        normalized = name.strip()
        for suffix in [' Jr.', ' Sr.', ' III', ' II', ' IV', ' V']:
            normalized = normalized.replace(suffix, '')
        return normalized.strip()

    # Build normalized name lookup
    normalized_headshot_lookup = {}
    for name, url in headshot_lookup.items():
        normalized_headshot_lookup[normalize_name(name)] = url

    # Player headshot URL - lookup from rosters data with name normalization
    def get_headshot_url(player_name: str) -> str | None:
        # Try exact match first
        url = headshot_lookup.get(player_name)
        if url:
            return url
        # Try normalized name match
        return normalized_headshot_lookup.get(normalize_name(player_name))

    # Convert tier to lowercase for frontend
    def normalize_tier(tier: str) -> str:
        tier_str = str(tier).lower() if tier else 'moderate'
        if tier_str in ['elite', 'strong', 'moderate', 'caution']:
            return tier_str
        return 'moderate'

    # Calculate star rating from confidence
    def get_stars(confidence: float) -> int:
        conf = float(confidence) if pd.notna(confidence) else 0.5
        if conf >= 0.75:
            return 5
        elif conf >= 0.68:
            return 4
        elif conf >= 0.60:
            return 3
        elif conf >= 0.55:
            return 2
        return 1

    # Build picks list
    picks_list = []
    filtered_count = 0

    for idx, row in player_props_df.iterrows():
        # CRITICAL FIX: Filter out picks where projection conflicts with direction
        # This happens when XGBoost classifier disagrees with Monte Carlo projection
        market = str(row.get('market', ''))

        # Use get_projection() to properly handle TD markets with p_attd
        projection = get_projection(row, market)
        if projection == 0:
            # Fallback to explicit columns
            projection = float(row.get('model_projection', row.get('projection', row.get('line', 0)))) if pd.notna(row.get('model_projection', row.get('projection'))) else float(row.get('line', 0))
        line = float(row.get('line', 0)) if pd.notna(row.get('line')) else 0
        pick_direction = str(row.get('pick', row.get('direction', 'OVER'))).upper()

        # Skip conflicting picks:
        # - OVER picks where projection < line (model says under-perform)
        # - UNDER picks where projection > line (model says over-perform)
        if pick_direction in ['OVER', 'YES'] and projection < line:
            filtered_count += 1
            continue
        if pick_direction in ['UNDER', 'NO'] and projection > line:
            filtered_count += 1
            continue

        # Get player_id from dataframe or lookup by name
        player_id = str(row.get('player_id', '')) if pd.notna(row.get('player_id')) else ''
        if not player_id:
            player_name = str(row.get('player', ''))
            player_id = name_to_player_id.get(player_name, '')

        # Get game history for this player (make a COPY to avoid shared mutations)
        # Each pick needs its own copy since defense_allowed is market-specific
        base_history = game_history_cache.get(player_id, {
            'weeks': [],
            'opponents': [],
            'receiving_yards': [],
            'receptions': [],
            'rushing_yards': [],
            'rushing_attempts': [],
            'passing_yards': [],
            'passing_attempts': [],
            'completions': [],
            'passing_tds': [],
            'rushing_tds': [],
            'receiving_tds': [],
        })
        # Deep copy to prevent shared mutations across picks for same player
        history = copy.deepcopy(base_history)

        # Add CURRENT OPPONENT's defense trend (what they've allowed over recent weeks)
        # This shows the defense the player is facing THIS week, not past opponents
        market_key = str(row.get('market', ''))
        current_opponent = str(row.get('opponent', row.get('opponent_abbr', '')))

        # Normalize team abbreviations (picks may use LAR, stats use LA)
        TEAM_ABBR_MAP = {'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS'}
        current_opponent = TEAM_ABBR_MAP.get(current_opponent, current_opponent)

        position = str(row.get('position', ''))

        # Get player's depth position for depth-filtered defense lookup
        player_depth_rank = None
        depth_pos = row.get('depth_position', '')  # e.g., "WR1", "RB2"
        if depth_pos and len(depth_pos) >= 2:
            try:
                player_depth_rank = int(depth_pos[-1])  # Extract number from "WR1" -> 1
            except ValueError:
                pass
        # Fallback: lookup from depth chart
        if player_depth_rank is None and player_id:
            depth_info = player_depth_lookup.get(player_id, {})
            player_depth_rank = depth_info.get('pos_rank')
        if player_depth_rank is None:
            pname = str(row.get('player', ''))
            team = str(row.get('team', ''))
            depth_info = player_name_depth.get((pname, team), {})
            player_depth_rank = depth_info.get('pos_rank')

        # Get the current opponent's recent defensive performance for this market
        # Always show 6 recent weeks of opponent defense regardless of player's game count
        defense_trend = []
        defense_weeks = []
        if current_opponent and current_opponent in defense_weekly:
            # Get sorted weeks for this opponent (most recent first)
            # Ensure weeks are integers (groupby may return numpy types)
            opp_weeks = sorted([int(w) for w in defense_weekly[current_opponent].keys()], reverse=True)
            # Always take 6 most recent weeks for defense trend
            recent_weeks = opp_weeks[:6]
            # Reverse to chronological order (oldest to newest)
            recent_weeks = list(reversed(recent_weeks))
            defense_weeks = recent_weeks

            # Build depth-specific market key: "player_receptions_WR_1" for WR1s
            # For anytime TD, use position-specific lookup key
            defense_market_key = market_key
            if market_key == 'player_anytime_td' and position in ['RB', 'WR', 'TE', 'QB']:
                defense_market_key = f"{market_key}_{position}"

            # Only use exact depth-specific key - no fallbacks
            depth_market_key = None
            if player_depth_rank and player_depth_rank <= 3:
                # Multi-position markets need position in key: player_receptions_WR_1
                if market_key in ['player_receptions', 'player_reception_yds', 'player_rush_yds']:
                    depth_market_key = f"{market_key}_{position}_{player_depth_rank}"
                else:
                    # Single-position markets: player_pass_yds_1
                    depth_market_key = f"{defense_market_key}_{player_depth_rank}"

            for def_week in recent_weeks:
                week_data = defense_weekly[current_opponent].get(def_week, {})
                # Only use exact depth-specific key - show None if not available
                if depth_market_key and depth_market_key in week_data:
                    defense_trend.append(week_data[depth_market_key])
                else:
                    defense_trend.append(None)

        history['defense_allowed'] = defense_trend
        history['defense_weeks'] = defense_weeks  # Store the actual weeks for defense data
        history['defense_opponent'] = current_opponent  # Track which team's defense this is
        # Add who the defense played each week
        history['defense_opponents'] = [
            team_week_opponent.get(current_opponent, {}).get(wk, '')
            for wk in defense_weeks
        ]

        # Calculate edge using the projection from get_projection() (already calculated above)
        # DO NOT overwrite projection - it was calculated correctly using p_attd for ATTD markets
        line = float(row.get('line', 0)) if pd.notna(row.get('line')) else 0
        # Normalize pick direction BEFORE edge calculation (handles 'Yes', 'OVER', 'Over', etc.)
        pick_direction = str(row.get('pick', row.get('direction', ''))).upper()
        is_over = pick_direction in ['OVER', 'YES']
        edge = projection - line if is_over else line - projection

        # Confidence - use calibrated_prob (XGBoost), combined_confidence (TD Enhanced), or model_prob
        conf_raw = row.get('calibrated_prob') if pd.notna(row.get('calibrated_prob')) else (
            row.get('combined_confidence') if pd.notna(row.get('combined_confidence')) else
            row.get('model_prob') if pd.notna(row.get('model_prob')) else 0.55
        )
        confidence = float(conf_raw) if pd.notna(conf_raw) else 0.55

        # L5 rate calculation
        l5_rate = None
        l5_hits = None
        if 'l5_hit_rate' in row and pd.notna(row['l5_hit_rate']):
            l5_rate = int(float(row['l5_hit_rate']) * 100) if float(row['l5_hit_rate']) <= 1 else int(row['l5_hit_rate'])

        player_name = str(row.get('player', 'Unknown'))
        team = str(row.get('team', ''))
        opponent = str(row.get('opponent', row.get('opponent_abbr', '')))
        position = str(row.get('position', row.get('actual_position', 'FLEX')))

        # Get depth chart position (e.g., "WR1", "RB2")
        depth_position = depth_chart_lookup.get((player_name, team.upper()), None)
        # Also try without suffix normalization
        if not depth_position:
            normalized_name = normalize_name(player_name)
            for (name, t), dp in depth_chart_lookup.items():
                if normalize_name(name) == normalized_name and t == team.upper():
                    depth_position = dp
                    break

        # Get defense vs market stats for opponent
        # For anytime TD, use position-specific lookup (e.g., player_anytime_td_RB for RBs)
        opp_def_allowed = None
        opp_def_rank = None
        market_key = str(row.get('market', ''))
        defense_lookup_key = market_key

        # For anytime TD, append position to get position-specific defense stats
        if market_key == 'player_anytime_td' and position in ['RB', 'WR', 'TE', 'QB']:
            defense_lookup_key = f"{market_key}_{position}"

        if opponent and defense_lookup_key in defense_vs_market.get(opponent, {}):
            opp_def_allowed = defense_vs_market[opponent][defense_lookup_key].get('allowed')
            opp_def_rank = defense_vs_market[opponent][defense_lookup_key].get('rank')

        # Determine display tier based on confidence
        if confidence >= 0.60:
            display_tier = 'TOP_PICK'
        elif confidence >= 0.55:
            display_tier = 'STANDARD'
        else:
            display_tier = 'OTHER'

        pick_data = {
            'id': f"pick-{idx}",
            'player': player_name,
            'position': position,
            'depth_position': depth_position,
            'team': team,
            'opponent': opponent,
            'headshot_url': get_headshot_url(player_name),
            'team_logo_url': get_team_logo_url(row.get('team', '')),
            'market': str(row.get('market', '')),
            'market_display': str(row.get('market_display', row.get('market', ''))),
            'line': line,
            'pick': 'OVER' if str(row.get('pick', row.get('direction', 'OVER'))).upper() in ['OVER', 'YES'] else 'UNDER',
            'projection': round(projection, 1),
            'edge': round(edge, 1),
            'confidence': round(confidence, 2),
            'tier': normalize_tier(row.get('effective_tier', 'moderate')),
            'display_tier': display_tier,  # TOP_PICK, STANDARD, or OTHER
            'stars': get_stars(confidence),
            'ev': round(float(row.get('roi_pct', row.get('edge_pct', 0))) if pd.notna(row.get('roi_pct', row.get('edge_pct'))) else 0, 1),
            'opp_rank': int(row.get('def_rank', 16)) if pd.notna(row.get('def_rank')) else None,
            'opp_def_allowed': opp_def_allowed,
            'opp_def_rank': opp_def_rank,
            'hist_over_rate': round(float(row.get('hist_over_rate', 0.5)) if pd.notna(row.get('hist_over_rate')) else 0.5, 2),
            'hist_count': int(row.get('hist_count', 0)) if pd.notna(row.get('hist_count')) else 0,
            'game': f"{team} vs {opponent}",
            'game_history': history,
        }

        # Add game context (weather & vegas) from lookup
        game_ctx = game_context_lookup.get((team, opponent), {})
        if game_ctx:
            pick_data['vegas_total'] = game_ctx.get('vegas_total')
            pick_data['vegas_spread'] = game_ctx.get('vegas_spread')
            pick_data['roof'] = game_ctx.get('roof')
            pick_data['temp'] = game_ctx.get('temp')
            pick_data['wind'] = game_ctx.get('wind')

        if l5_rate is not None:
            pick_data['l5_rate'] = l5_rate
        if l5_hits:
            pick_data['l5_hits'] = l5_hits

        picks_list.append(pick_data)

    # Sort by confidence descending
    picks_list.sort(key=lambda x: x['confidence'], reverse=True)

    # Load schedule data for game metadata
    games_metadata = []
    try:
        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
        if schedule_path.exists():
            schedule_df = pd.read_parquet(schedule_path)
            week_games = schedule_df[(schedule_df['week'] == week) & (schedule_df['season'] == season)]

            for _, game in week_games.iterrows():
                away = str(game.get('away_team', ''))
                home = str(game.get('home_team', ''))
                game_key = f"{away} vs {home}"
                # Also create normalized key (sorted teams)
                normalized_key = ' vs '.join(sorted([away, home]))

                games_metadata.append({
                    'game': game_key,
                    'normalized': normalized_key,
                    'away_team': away,
                    'home_team': home,
                    'gameday': str(game.get('gameday', '')),
                    'gametime': str(game.get('gametime', '')),
                    'stadium': str(game.get('stadium', '')) if pd.notna(game.get('stadium')) else '',
                    'roof': str(game.get('roof', '')) if pd.notna(game.get('roof')) else '',
                    'temp': int(game.get('temp', 0)) if pd.notna(game.get('temp')) else None,
                    'wind': int(game.get('wind', 0)) if pd.notna(game.get('wind')) else None,
                })

            # Sort by gameday and gametime
            games_metadata.sort(key=lambda x: (x['gameday'], x['gametime']))
            print(f"  Loaded {len(games_metadata)} games with metadata")
    except Exception as e:
        print(f"  Warning: Could not load schedule data: {e}")

    # Build game lines list from game_lines_df
    game_lines_list = []
    if game_lines_df is not None and len(game_lines_df) > 0:
        for idx, row in game_lines_df.iterrows():
            game_str = str(row.get('game', ''))
            home_team = game_str.split(' @ ')[-1] if ' @ ' in game_str else ''
            away_team = game_str.split(' @ ')[0] if ' @ ' in game_str else ''

            game_line = {
                'id': f"gameline-{idx}",
                'game': game_str,
                'bet_type': str(row.get('bet_type', '')),
                'pick': str(row.get('pick', '')),
                'line': safe_float(row.get('market_line', 0)),
                'fair_line': round(safe_float(row.get('model_fair_line', 0)), 1),
                'confidence': round(safe_float(row.get('model_prob', 0.5)), 2),
                'edge': round(safe_float(row.get('edge_pct', 0)), 1),
                'ev': round(safe_float(row.get('expected_roi', 0)), 1),
                'tier': str(row.get('confidence_tier', 'MODERATE')).lower(),
                'kelly_units': round(safe_float(row.get('recommended_units', 0)), 2),
                'home_team': home_team,
                'away_team': away_team,
                # Additional model data for modal
                'home_win_prob': round(safe_float(row.get('home_win_prob', 0.5)), 3),
                'home_epa': round(safe_float(row.get('home_epa', 0)), 3),
                'away_epa': round(safe_float(row.get('away_epa', 0)), 3),
                'home_elo': round(safe_float(row.get('home_elo', 1500)), 0),
                'away_elo': round(safe_float(row.get('away_elo', 1500)), 0),
                'elo_diff': round(safe_float(row.get('elo_diff', 0)), 1),
                'home_record': str(row.get('home_record', '')),
                'away_record': str(row.get('away_record', '')),
                'home_rest_days': int(safe_float(row.get('home_rest_days', 7))),
                'away_rest_days': int(safe_float(row.get('away_rest_days', 7))),
                # ATS records
                'home_ats_record': str(row.get('home_ats_record', '')),
                'away_ats_record': str(row.get('away_ats_record', '')),
                'home_last6_ats': str(row.get('home_last6_ats', '')),
                'away_last6_ats': str(row.get('away_last6_ats', '')),
                # Defense ranks
                'home_total_def_rank': int(safe_float(row.get('home_total_def_rank', 16))),
                'away_total_def_rank': int(safe_float(row.get('away_total_def_rank', 16))),
                # EPA breakdown
                'home_pass_epa': round(safe_float(row.get('home_pass_epa', 0)), 3),
                'away_pass_epa': round(safe_float(row.get('away_pass_epa', 0)), 3),
                'home_rush_epa': round(safe_float(row.get('home_rush_epa', 0)), 3),
                'away_rush_epa': round(safe_float(row.get('away_rush_epa', 0)), 3),
                'home_def_epa': round(safe_float(row.get('home_def_epa', 0)), 3),
                'away_def_epa': round(safe_float(row.get('away_def_epa', 0)), 3),
            }
            game_lines_list.append(game_line)
        # Sort by confidence descending
        game_lines_list.sort(key=lambda x: x['confidence'], reverse=True)

    # Build parlays list from parlays_df
    parlays_list = []
    if parlays_df is not None and len(parlays_df) > 0:
        for idx, row in parlays_df.iterrows():
            # Parse recommended_units - may have 'u' suffix like "0.50u"
            units_val = str(row.get('recommended_units', '0')).replace('u', '').replace('U', '').strip()
            try:
                units_float = float(units_val) if units_val else 0
            except ValueError:
                units_float = 0

            parlay = {
                'id': f"parlay-{idx}",
                'rank': safe_int(row.get('rank', idx + 1)),
                'featured': str(row.get('featured', '')).upper() == 'YES',
                'legs': str(row.get('legs', '')),
                'num_legs': safe_int(row.get('num_legs', 0)),
                'true_odds': safe_int(row.get('true_odds', 0)),
                'model_odds': safe_int(row.get('model_odds', 0)),
                'true_prob': str(row.get('true_prob', '')),
                'model_prob': str(row.get('model_prob', '')),
                'edge': str(row.get('edge', '')),
                'stake': str(row.get('recommended_stake', '')),
                'potential_win': str(row.get('potential_win', '')),
                'ev': str(row.get('expected_value', '')),
                'games': str(row.get('games', '')),
                'sources': str(row.get('sources', '')),
                'units': round(units_float, 2),
            }
            parlays_list.append(parlay)

    # Load team colors and logos from NFLverse
    teams_data = {}
    try:
        teams_path = PROJECT_ROOT / 'data' / 'nflverse' / 'teams.parquet'
        if teams_path.exists():
            teams_df = pd.read_parquet(teams_path)
            for _, row in teams_df.iterrows():
                abbr = str(row.get('team_abbr', '')).upper()
                if abbr:
                    teams_data[abbr] = {
                        'name': str(row.get('team_name', '')),
                        'nick': str(row.get('team_nick', '')),
                        'color': str(row.get('team_color', '#333333')),
                        'color2': str(row.get('team_color2', '#666666')),
                        'color3': str(row.get('team_color3', '')) if pd.notna(row.get('team_color3')) else None,
                        'color4': str(row.get('team_color4', '')) if pd.notna(row.get('team_color4')) else None,
                        'logo': str(row.get('team_logo_espn', '')),
                        'logoSquared': str(row.get('team_logo_squared', '')),
                        'wordmark': str(row.get('team_wordmark', '')),
                        'conf': str(row.get('team_conf', '')),
                        'division': str(row.get('team_division', '')),
                    }
            print(f"  Loaded {len(teams_data)} team colors and logos from NFLverse")
    except Exception as e:
        print(f"  Warning: Could not load team data: {e}")

    # Build final JSON structure
    dashboard_data = {
        'week': week,
        'generated_at': datetime.now().isoformat(),
        'stats': {
            'total_picks': len(picks_list),
            'avg_edge': round(avg_edge, 1) if pd.notna(avg_edge) else 0,
            'games': games_count,
            'elite_count': elite_count,
            'strong_count': strong_count,
        },
        'teams': teams_data,
        'games': games_metadata,
        'picks': picks_list,
        'gameLines': game_lines_list,
        'parlays': parlays_list,
    }

    # Output path for Next.js app (at deploy root, not web subdirectory)
    output_dir = PROJECT_ROOT / 'deploy' / 'src' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'picks.json'

    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print(f"JSON export: {output_path} ({len(picks_list)} picks, {len(game_lines_list)} game lines, {len(parlays_list)} parlays)")
    return output_path


def generate_dashboard(week: int = None, season: int = 2025):
    """Generate the professional dashboard HTML."""
    if week is None:
        week = settings.CURRENT_WEEK

    # Load data - prefer edge recommendations (new pipeline)
    recs_df = load_edge_recommendations(week)
    edge_count = len(recs_df)

    # If edge recommendations are few, MERGE with XGBoost predictions (not replace)
    # This preserves TD Poisson picks while adding more continuous stat picks
    if edge_count < 20:
        recs_path = PROJECT_ROOT / "reports" / "CURRENT_WEEK_RECOMMENDATIONS.csv"
        if recs_path.exists():
            print(f"Edge recommendations ({edge_count}), supplementing with XGBoost predictions")
            xgb_df = pd.read_csv(recs_path)
            # Apply multi-factor scoring for tier only (keeps CAUTION flags)
            xgb_df['effective_tier'] = xgb_df.apply(get_adjusted_tier, axis=1)
            xgb_df['confidence'] = xgb_df['effective_tier']
            # Add defensive rankings (same logic as edge recommendations)
            xgb_df = add_defensive_rankings_to_df(xgb_df, week)

            if edge_count > 0:
                # Merge: keep edge recommendations, add XGBoost picks that don't overlap
                # Identify by (player, market) combination
                edge_keys = set(zip(recs_df['player'].str.lower(), recs_df['market']))
                xgb_df['_key'] = list(zip(xgb_df['player'].str.lower(), xgb_df['market']))
                xgb_new = xgb_df[~xgb_df['_key'].isin(edge_keys)].drop(columns=['_key'])
                recs_df = pd.concat([recs_df, xgb_new], ignore_index=True)
                print(f"  Merged: {edge_count} edge + {len(xgb_new)} XGBoost = {len(recs_df)} total")
            else:
                recs_df = xgb_df
        elif edge_count == 0:
            print(f"Error: No recommendations found for week {week}")
            return None

    # Try multiple possible game lines file names
    game_lines_paths = [
        PROJECT_ROOT / "reports" / f"WEEK{week}_GAME_LINE_RECOMMENDATIONS.csv",
        PROJECT_ROOT / "reports" / f"game_lines_predictions_week{week}.csv",
        PROJECT_ROOT / "data" / f"game_line_predictions_week{week}.csv",
    ]
    game_lines_path = None
    for path in game_lines_paths:
        if path.exists():
            game_lines_path = path
            print(f"Found game lines: {path}")
            break

    # Log tier distribution
    elite_count = len(recs_df[recs_df['effective_tier'] == 'ELITE'])
    strong_count = len(recs_df[recs_df['effective_tier'] == 'STRONG'])
    moderate_count = len(recs_df[recs_df['effective_tier'] == 'MODERATE'])
    caution_count = len(recs_df[recs_df.get('effective_tier', pd.Series()) == 'CAUTION']) if 'effective_tier' in recs_df.columns else 0

    print(f"Tier distribution: {elite_count} ELITE, {strong_count} STRONG, "
          f"{moderate_count} MODERATE, {caution_count} CAUTION")

    game_lines_df = None
    if game_lines_path and game_lines_path.exists():
        game_lines_df = pd.read_csv(game_lines_path)
        print(f"  Loaded {len(game_lines_df)} game line recommendations")

    # Load schedule for game times and venue info
    schedule_df = load_schedule(week)

    # Merge venue info from schedule into recs_df
    if len(schedule_df) > 0 and 'roof' in schedule_df.columns:
        # Create venue lookup by home team
        venue_lookup = {}
        for _, row in schedule_df.iterrows():
            home = row.get('home_team', '')
            away = row.get('away_team', '')
            venue_lookup[home] = {
                'roof': row.get('roof', ''),
                'surface': row.get('surface', '')
            }
            venue_lookup[away] = {
                'roof': row.get('roof', ''),
                'surface': row.get('surface', '')
            }

        # Add venue info to recs_df based on team
        if 'team' in recs_df.columns:
            recs_df['roof'] = recs_df['team'].map(lambda t: venue_lookup.get(t, {}).get('roof', ''))
            recs_df['surface'] = recs_df['team'].map(lambda t: venue_lookup.get(t, {}).get('surface', ''))

    # Load live edges if available
    live_edges_df = load_live_edges()
    if len(live_edges_df) > 0:
        print(f"Loaded {len(live_edges_df)} live edges")

    # Load tracking data for bet tracker tab
    tracking_data = load_tracking_data()
    tracking_count = tracking_data['total_bets'] if tracking_data else 0

    # Load parlay recommendations
    parlays_df = None
    parlay_path = PROJECT_ROOT / "reports" / f"parlay_recommendations_week{week}_{season}.csv"
    if parlay_path.exists():
        parlays_df = pd.read_csv(parlay_path)
        print(f"Loaded {len(parlays_df)} parlay recommendations")
    parlay_count = len(parlays_df) if parlays_df is not None else 0

    # THREE-TIER DISPLAY SYSTEM (updated 2025-12-28):
    # - TOP picks: 60%+ confidence (validated profitable threshold)
    # - STANDARD picks: 55-60% confidence (moderate edge)
    # - OTHER picks: 50-55% confidence (shown but lower priority)

    total_before_filter = len(recs_df)

    # Get confidence column - use model_prob for XGBoost, combined_confidence for edge recs
    confidence_col = recs_df['model_prob'].fillna(recs_df.get('combined_confidence', 0))

    # Keep picks with 50%+ confidence (include all useful picks)
    all_picks_filter = confidence_col >= 0.50

    # Exclude backup players if column exists (they have unreliable projections)
    if 'calibration_tier' in recs_df.columns:
        all_picks_filter &= ~recs_df['calibration_tier'].str.contains('backup', case=False, na=False)

    # Apply filter to main dataframe
    recs_df = recs_df[all_picks_filter].copy()

    # Add display tier based on confidence
    def get_display_tier(conf):
        if conf >= 0.60:
            return 'TOP_PICK'
        elif conf >= 0.55:
            return 'STANDARD'
        else:
            return 'OTHER'

    recs_df['display_tier'] = confidence_col[all_picks_filter].apply(get_display_tier)

    # TIER 2: Mark picks that meet FEATURED thresholds (higher confidence for top picks)
    def meets_featured_threshold(row):
        """Check if pick meets its market-specific FEATURED threshold."""
        market = safe_str(row.get('market', ''), '')
        model_prob = row.get('model_prob', 0)
        threshold = FEATURED_THRESHOLDS.get(market, DEFAULT_FEATURED_THRESHOLD)
        return model_prob >= threshold

    if 'market' in recs_df.columns and 'model_prob' in recs_df.columns:
        recs_df['is_featured'] = recs_df.apply(meets_featured_threshold, axis=1)
    else:
        recs_df['is_featured'] = True

    # Add market profitability flag for visual indicators
    if 'market' in recs_df.columns:
        recs_df['market_roi_tier'] = recs_df['market'].apply(
            lambda m: 'profitable' if m in PROFITABLE_MARKETS
            else 'breakeven' if m in BREAKEVEN_MARKETS
            else 'negative'
        )

    # Log filtering stats
    featured_count = recs_df['is_featured'].sum() if 'is_featured' in recs_df.columns else 0
    top_count = len(recs_df[recs_df['display_tier'] == 'TOP_PICK']) if 'display_tier' in recs_df.columns else 0
    standard_count_tier = len(recs_df[recs_df['display_tier'] == 'STANDARD']) if 'display_tier' in recs_df.columns else 0
    other_count = len(recs_df[recs_df['display_tier'] == 'OTHER']) if 'display_tier' in recs_df.columns else 0

    print(f"\nAll picks (50%+ confidence): {len(recs_df)} of {total_before_filter}")
    print(f"  TOP_PICK (60%+): {top_count}")
    print(f"  STANDARD (55-60%): {standard_count_tier}")
    print(f"  OTHER (50-55%): {other_count}")
    print(f"Featured picks (market-specific thresholds): {featured_count}")
    print("Featured thresholds:")
    for market, thresh in FEATURED_THRESHOLDS.items():
        if market in recs_df['market'].values:
            market_featured = len(recs_df[(recs_df['market'] == market) & (recs_df['is_featured'] == True)])
            market_total = len(recs_df[recs_df['market'] == market])
            print(f"  {market}: {thresh:.0%} threshold → {market_featured}/{market_total} featured")

    # Calculate summary stats (now on filtered data)
    total_picks = len(recs_df)
    avg_edge = recs_df['edge_pct'].mean() if 'edge_pct' in recs_df.columns else 0

    # Count tiers using effective tier
    elite_count = len(recs_df[recs_df['effective_tier'] == 'ELITE'])
    high_count = len(recs_df[recs_df['effective_tier'] == 'STRONG'])  # Was 'HIGH', should be 'STRONG'
    standard_count = len(recs_df[recs_df['effective_tier'] == 'MODERATE'])  # Was 'STANDARD', should be 'MODERATE'

    # Unique games, prop types, players
    games = recs_df['game'].nunique() if 'game' in recs_df.columns else 0
    prop_types = recs_df['market'].nunique() if 'market' in recs_df.columns else 0
    players = recs_df['player'].nunique() if 'player' in recs_df.columns else 0

    # Calculate total bet
    kelly_col = 'kelly_units' if 'kelly_units' in recs_df.columns else 'kelly_fraction'
    total_bet = recs_df[kelly_col].sum() * 100 if kelly_col in recs_df.columns else 0

    game_line_count = len(game_lines_df) if game_lines_df is not None else 0
    live_edges_count = len(live_edges_df) if live_edges_df is not None else 0

    # FEATURED PICKS = Top 10 picks across all markets, sorted by confidence
    featured_df = recs_df[recs_df['is_featured'] == True].copy() if 'is_featured' in recs_df.columns else recs_df.copy()
    top_picks = featured_df.nlargest(10, 'model_prob')

    # Count by market ROI tier
    profitable_count = len(top_picks[top_picks['market'].isin(PROFITABLE_MARKETS)])
    breakeven_count = len(top_picks[top_picks['market'].isin(BREAKEVEN_MARKETS)])
    other_count = len(top_picks) - profitable_count - breakeven_count

    # Build picks note
    picks_note = (
        f"Featured: Top {len(top_picks)} picks by confidence. "
        f"All picks: {len(recs_df)} total across all markets."
    )

    top_picks_count = len(top_picks)
    print(f"\nFeatured Picks: {top_picks_count} (from {featured_count} meeting thresholds)")

    # Generate HTML - COMPACT SINGLE-SCREEN OPTIMIZED
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL QUANT | Week {week} Dashboard</title>
    <style>{generate_css()}</style>
</head>
<body>
    <div class="container">
        <!-- STICKY HEADER WRAPPER - Always visible -->
        <div class="header-wrapper">
            <div class="header">
                <h1>NFL <span class="brand-accent">QUANT</span> | Week {week}</h1>
                <p>{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            </div>

            <div class="stats-summary">
                <div class="stat-card primary">
                    <div class="stat-value highlight">{elite_count + high_count}</div>
                    <div class="stat-label">Top Picks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_edge:.1f}%</div>
                    <div class="stat-label">Avg Edge</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{games}</div>
                    <div class="stat-label">Games</div>
                </div>
            </div>

            <!-- Consolidated 3-Tab Navigation -->
            <div class="tabs">
                <button class="tab active" onclick="showView('cheat-sheet', this)">Picks<span class="tab-count">{total_picks}</span></button>
                <button class="tab" onclick="showView('game-lines', this)">Lines<span class="tab-count">{game_line_count}</span></button>
                <button class="tab" onclick="showView('parlays', this)">Parlays<span class="tab-count">{parlay_count}</span></button>
            </div>
        </div>
'''

    # ============================================
    # CONSOLIDATED 3-TAB LAYOUT
    # ============================================

    # 1. PICKS VIEW - BettingPros-style dense data table (DEFAULT ACTIVE)
    html += generate_cheat_sheet_section(recs_df, format_prop_display, week)

    # 2. GAME LINES VIEW - BettingPros-style table
    html += generate_game_lines_table_section(game_lines_df, week)

    # Parlays View
    html += '''
        <div class="view-section" id="parlays">
            <div class="gl-view-header">
                <div>
                    <h2 style="margin: 0; font-size: 16px;">Parlay Recommendations</h2>
                    <p style="color: var(--text-muted); margin: 4px 0 0 0; font-size: 11px;">
                        Cross-game parlays with correlation-adjusted probabilities and EV-based ranking
                    </p>
                </div>
            </div>
    '''

    if parlays_df is not None and len(parlays_df) > 0:
        # Check for featured column
        has_featured = 'featured' in parlays_df.columns

        # Split into featured and additional
        if has_featured:
            featured_parlays = parlays_df[parlays_df['featured'] == 'YES']
            additional_parlays = parlays_df[parlays_df['featured'] == 'NO']
        else:
            # Fallback: top 3 are featured
            featured_parlays = parlays_df.head(3)
            additional_parlays = parlays_df.iloc[3:]

        # Featured Parlays Section
        if len(featured_parlays) > 0:
            html += '''
                <div class="parlay-featured-section" style="margin-top: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                        <span style="font-size: 18px;">★</span>
                        <h3 style="font-size: 15px; margin: 0; color: #f0b429;">Featured Parlays</h3>
                    </div>
                    <div class="parlay-cards-grid" style="display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));">
            '''

            for idx, prow in featured_parlays.iterrows():
                legs = prow.get('legs', '').split(' | ')
                odds = prow.get('true_odds', 0)
                model_prob = str(prow.get('model_prob', '0%')).replace('%', '')
                edge = str(prow.get('edge', '0%')).replace('%', '')
                ev = str(prow.get('expected_value', '$0')).replace('$', '').replace('N/A', '0')
                games = prow.get('games', '')
                rank = prow.get('rank', 0)
                units = str(prow.get('recommended_units', '0.50u')).replace('u', '')

                try:
                    model_prob_val = float(model_prob)
                    edge_val = float(edge)
                    ev_val = float(ev)
                    units_val = float(units)
                except:
                    model_prob_val = 0
                    edge_val = 0
                    ev_val = 0
                    units_val = 0.5

                # Format legs with bullet points
                legs_html = ''.join([f'<div style="padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">• {leg.strip()}</div>' for leg in legs])

                html += f'''
                    <div class="parlay-card featured" style="background: linear-gradient(135deg, rgba(240, 180, 41, 0.08) 0%, var(--card-bg) 100%); border: 2px solid rgba(240, 180, 41, 0.4); border-radius: 10px; padding: 16px; position: relative;">
                        <div style="position: absolute; top: -10px; left: 16px; background: #f0b429; color: #000; font-size: 11px; font-weight: 700; padding: 2px 10px; border-radius: 10px;">
                            #{rank} FEATURED
                        </div>
                        <div class="parlay-header" style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0 12px 0;">
                            <span style="font-weight: 700; font-size: 22px; color: #f0b429;">{odds:+d}</span>
                            <span style="font-size: 13px; padding: 4px 12px; background: rgba(46, 160, 67, 0.2); border-radius: 6px; color: #2ea043; font-weight: 600;">+${ev_val:.2f} EV</span>
                        </div>
                        <div class="parlay-legs" style="font-size: 13px; color: var(--text-primary); line-height: 1.5; margin-bottom: 12px;">
                            {legs_html}
                        </div>
                        <div class="parlay-footer" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 12px; border-top: 1px solid var(--border-color); padding-top: 12px;">
                            <div style="text-align: center;">
                                <div style="color: var(--text-muted); font-size: 10px;">WIN PROB</div>
                                <div style="font-weight: 600; color: var(--text-primary);">{model_prob_val:.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: var(--text-muted); font-size: 10px;">EDGE</div>
                                <div style="font-weight: 600; color: #2ea043;">{edge_val:.1f}%</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: var(--text-muted); font-size: 10px;">BET SIZE</div>
                                <div style="font-weight: 600; color: var(--text-primary);">{units_val:.2f}u</div>
                            </div>
                        </div>
                    </div>
                '''

            html += '''
                    </div>
                </div>
            '''

        # Additional Parlays Section
        if len(additional_parlays) > 0:
            html += '''
                <div class="parlay-additional-section" style="margin-top: 24px;">
                    <h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-secondary);">Additional Parlays</h3>
                    <div class="parlay-table" style="background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color);">
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                            <thead>
                                <tr style="background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border-color);">
                                    <th style="padding: 10px 12px; text-align: left; font-weight: 600; color: var(--text-muted);">#</th>
                                    <th style="padding: 10px 12px; text-align: left; font-weight: 600; color: var(--text-muted);">LEGS</th>
                                    <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: var(--text-muted);">ODDS</th>
                                    <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: var(--text-muted);">WIN %</th>
                                    <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: var(--text-muted);">EDGE</th>
                                    <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: var(--text-muted);">EV</th>
                                </tr>
                            </thead>
                            <tbody>
            '''

            for idx, prow in additional_parlays.iterrows():
                legs = prow.get('legs', '').split(' | ')
                odds = prow.get('true_odds', 0)
                model_prob = str(prow.get('model_prob', '0%')).replace('%', '')
                edge = str(prow.get('edge', '0%')).replace('%', '')
                ev = str(prow.get('expected_value', '$0')).replace('$', '').replace('N/A', '0')
                rank = prow.get('rank', 0)
                num_legs = prow.get('num_legs', len(legs))

                try:
                    model_prob_val = float(model_prob)
                    edge_val = float(edge)
                    ev_val = float(ev)
                except:
                    model_prob_val = 0
                    edge_val = 0
                    ev_val = 0

                # Compact leg display
                legs_compact = ' + '.join([leg.split()[0] + ' ' + leg.split()[-2] + ' ' + leg.split()[-1] if len(leg.split()) >= 3 else leg for leg in legs])

                html += f'''
                    <tr style="border-bottom: 1px solid var(--border-color);">
                        <td style="padding: 10px 12px; color: var(--text-muted);">#{rank}</td>
                        <td style="padding: 10px 12px; color: var(--text-primary); max-width: 400px;">
                            <span style="font-size: 11px;">{legs_compact}</span>
                        </td>
                        <td style="padding: 10px 12px; text-align: right; font-weight: 600; color: var(--accent-primary);">{odds:+d}</td>
                        <td style="padding: 10px 12px; text-align: right;">{model_prob_val:.1f}%</td>
                        <td style="padding: 10px 12px; text-align: right; color: #2ea043;">{edge_val:.1f}%</td>
                        <td style="padding: 10px 12px; text-align: right; color: #2ea043;">+${ev_val:.2f}</td>
                    </tr>
                '''

            html += '''
                            </tbody>
                        </table>
                    </div>
                </div>
            '''
    else:
        html += '<div style="text-align: center; color: var(--text-muted); padding: 40px;">No parlay recommendations available. Run the parlay generator first.</div>'

    html += '''
        </div>
    '''

    # Bet Tracker View
    if tracking_data:
        # Build weekly results rows
        weekly_rows = ""
        for week_key in sorted(tracking_data['weekly_summaries'].keys(), reverse=True):
            summary = tracking_data['weekly_summaries'][week_key]
            wins = summary.get('wins', 0)
            losses = summary.get('losses', 0)
            roi = summary.get('roi', 0)
            profit = summary.get('total_profit', 0)
            win_rate = summary.get('win_rate', 0)

            roi_class = 'positive' if roi > 0 else 'negative' if roi < 0 else ''
            week_display = week_key.replace('2025_week', 'Week ')

            weekly_rows += f'''
                <tr>
                    <td><strong>{week_display}</strong></td>
                    <td>{summary.get('total_bets', 0)}</td>
                    <td>{wins}W - {losses}L</td>
                    <td>{win_rate:.1%}</td>
                    <td class="{roi_class}">{profit:+.2f}u</td>
                    <td class="{roi_class}">{roi:+.1f}%</td>
                </tr>
            '''

        # Build market breakdown rows
        market_rows = ""
        for market, stats in sorted(tracking_data['markets'].items()):
            wins = stats['wins']
            losses = stats['losses']
            total = stats['bets']
            profit = stats['profit']
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            roi = (profit / total * 100) if total > 0 else 0
            roi_class = 'positive' if roi > 0 else 'negative' if roi < 0 else ''

            market_display = market.replace('player_', '').replace('_', ' ').title()
            market_rows += f'''
                <tr>
                    <td>{market_display}</td>
                    <td>{total}</td>
                    <td>{wins}W - {losses}L</td>
                    <td>{win_rate:.1%}</td>
                    <td class="{roi_class}">{profit:+.2f}u</td>
                    <td class="{roi_class}">{roi:+.1f}%</td>
                </tr>
            '''

        # Build running P&L data points for chart
        pnl_data = tracking_data['running_pnl']
        pnl_points = [p['cumulative'] for p in pnl_data]
        pnl_labels = [f"#{i+1}" for i in range(len(pnl_data))]

        # Overall stats
        total_bets = tracking_data['total_bets']
        overall_wins = tracking_data['wins']
        overall_losses = tracking_data['losses']
        overall_win_rate = tracking_data['win_rate']
        overall_profit = tracking_data['total_profit']
        overall_roi = tracking_data['roi']
        roi_class = 'positive' if overall_roi > 0 else 'negative' if overall_roi < 0 else ''

        html += f'''
        <div class="view-section" id="tracker">
            <h2>Bet Tracker</h2>
            <p style="color: var(--text-muted); margin-bottom: 15px; font-size: 12px;">
                Historical bet results tracking. Run <code>python scripts/tracking/track_bet_results.py --week N</code> to update.
            </p>

            <!-- Overall Summary Cards -->
            <div class="tracker-summary">
                <div class="tracker-stat-card">
                    <div class="tracker-stat-value">{total_bets}</div>
                    <div class="tracker-stat-label">Total Bets</div>
                </div>
                <div class="tracker-stat-card">
                    <div class="tracker-stat-value">{overall_wins}W - {overall_losses}L</div>
                    <div class="tracker-stat-label">Record</div>
                </div>
                <div class="tracker-stat-card">
                    <div class="tracker-stat-value">{overall_win_rate:.1%}</div>
                    <div class="tracker-stat-label">Win Rate</div>
                </div>
                <div class="tracker-stat-card {roi_class}">
                    <div class="tracker-stat-value">{overall_profit:+.2f}u</div>
                    <div class="tracker-stat-label">Profit</div>
                </div>
                <div class="tracker-stat-card {roi_class}">
                    <div class="tracker-stat-value">{overall_roi:+.1f}%</div>
                    <div class="tracker-stat-label">ROI</div>
                </div>
            </div>

            <!-- P&L Chart -->
            <div class="tracker-chart-container">
                <h3>Running P&L</h3>
                <div class="pnl-chart">
                    <canvas id="pnlChart"></canvas>
                </div>
            </div>

            <!-- Weekly Results -->
            <div class="tracker-section">
                <h3>Results by Week</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Week</th>
                                <th>Bets</th>
                                <th>Record</th>
                                <th>Win Rate</th>
                                <th>Profit</th>
                                <th>ROI</th>
                            </tr>
                        </thead>
                        <tbody>
                            {weekly_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- By Market -->
            <div class="tracker-section">
                <h3>Results by Market</h3>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Market</th>
                                <th>Bets</th>
                                <th>Record</th>
                                <th>Win Rate</th>
                                <th>Profit</th>
                                <th>ROI</th>
                            </tr>
                        </thead>
                        <tbody>
                            {market_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            // P&L Chart
            const pnlData = {pnl_points};
            const pnlLabels = {pnl_labels};

            if (document.getElementById('pnlChart')) {{
                new Chart(document.getElementById('pnlChart'), {{
                    type: 'line',
                    data: {{
                        labels: pnlLabels,
                        datasets: [{{
                            label: 'Cumulative P&L (units)',
                            data: pnlData,
                            borderColor: pnlData[pnlData.length - 1] >= 0 ? '#3B82F6' : '#ef4444',
                            backgroundColor: 'transparent',
                            tension: 0.1,
                            pointRadius: 0,
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                display: false
                            }},
                            y: {{
                                grid: {{ color: 'rgba(255,255,255,0.1)' }},
                                ticks: {{ color: '#9ca3af' }}
                            }}
                        }}
                    }}
                }});
            }}
        </script>
        '''
    else:
        html += '''
        <div class="view-section" id="tracker">
            <h2>Bet Tracker</h2>
            <div style="text-align: center; padding: 60px 20px; color: var(--text-muted);">
                <p style="font-size: 16px; margin-bottom: 10px;">No tracking data available</p>
                <p style="font-size: 12px;">Run <code>python scripts/tracking/track_bet_results.py --week N</code> after games complete to track results.</p>
            </div>
        </div>
        '''

    # Footer and close - COMPACT
    html += f'''
        <div class="footer" style="padding: 6px 12px; font-size: 10px;">
            NFL QUANT | Wk{week} | {MODEL_VERSION_FULL} | {datetime.now().strftime("%m/%d %H:%M")}
        </div>
    </div>

    <!-- Pick Breakdown Modal -->
    <div class="modal-overlay" id="pick-modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-player-info">
                    <div class="modal-player-details">
                        <span class="modal-player-name" id="modal-player-name">Player Name</span>
                        <span class="modal-player-meta" id="modal-player-meta">POS | TEAM</span>
                    </div>
                </div>
                <button class="modal-close" onclick="closePickModal()">&times;</button>
            </div>

            <div class="modal-pick-summary">
                <div class="modal-pick-main">
                    <div class="modal-pick-direction">
                        <span class="modal-pick-badge" id="modal-pick-badge">OVER</span>
                        <span class="modal-pick-line" id="modal-pick-line">5.5</span>
                    </div>
                    <div class="modal-confidence">
                        <span class="modal-conf-value" id="modal-conf-value">72%</span>
                        <div class="modal-conf-label">Confidence</div>
                    </div>
                </div>
                <div class="modal-pick-market" id="modal-pick-market">Receptions</div>

                <div class="modal-stats-row">
                    <div class="modal-stat-item">
                        <div class="modal-stat-value" id="modal-projection">72%</div>
                        <div class="modal-stat-label" id="modal-projection-label">Model Conf</div>
                    </div>
                    <div class="modal-stat-item">
                        <div class="modal-stat-value" id="modal-edge">+5.2%</div>
                        <div class="modal-stat-label">Edge</div>
                    </div>
                    <div class="modal-stat-item">
                        <div class="modal-stat-value" id="modal-trailing">5.8</div>
                        <div class="modal-stat-label" id="modal-trailing-label">Trailing Avg</div>
                    </div>
                </div>
            </div>

            <div class="modal-body">
                <!-- Conflict warnings appear here when detected -->
                <div id="modal-conflict-warning" style="display: none;">
                    <!-- Populated by JS when conflicts detected -->
                </div>

                <div class="modal-section modal-summary-section">
                    <div class="modal-section-title">Analysis</div>
                    <div class="modal-summary" id="modal-summary">
                        <!-- Populated by JS - includes narrative + unified key factors list -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bet Slip Toggle Button -->
    <button class="bet-slip-toggle empty" id="bet-slip-toggle" onclick="toggleBetSlip()">
        <span class="bet-slip-icon">+</span>
        <span class="bet-slip-text">Build Bet Slip</span>
        <span class="count">0</span>
    </button>

    <!-- Bet Slip Overlay -->
    <div class="bet-slip-overlay" id="bet-slip-overlay" onclick="closeBetSlip()"></div>

    <!-- Bet Slip Panel -->
    <div class="bet-slip-panel" id="bet-slip-panel">
        <div class="bet-slip-header">
            <h3>📋 My Bet Slip <span id="bet-slip-count">0</span></h3>
            <button class="bet-slip-close" onclick="closeBetSlip()">×</button>
        </div>
        <div class="bet-slip-content" id="bet-slip-content">
            <div class="bet-slip-empty">
                <div class="bet-slip-empty-icon">📋</div>
                <p>No bets selected</p>
                <p style="font-size: 11px; margin-top: 8px;">Click the checkbox next to any pick to add it to your slip</p>
            </div>
        </div>
        <div class="bet-slip-footer">
            <div class="bet-slip-summary">
                <span class="bet-slip-summary-label">Avg Confidence</span>
                <span class="bet-slip-summary-value" id="bet-slip-avg-conf">-</span>
            </div>
            <div class="bet-slip-actions">
                <button class="bet-slip-btn secondary" onclick="exportBetsJSON()">Export JSON</button>
                <button class="bet-slip-btn primary" onclick="saveBetsForTracking()">Save for Tracking</button>
            </div>
            <div class="bet-slip-actions" style="margin-top: 8px;">
                <button class="bet-slip-btn danger" onclick="clearAllBets()">Clear All</button>
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div class="bet-toast" id="bet-toast"></div>

    <script>
{generate_javascript()}
    </script>
</body>
</html>
'''

    # Save dashboard
    output_path = PROJECT_ROOT / "reports" / "pro_dashboard.html"
    output_path.write_text(html)
    print(f"Dashboard generated: {output_path}")

    # Also export JSON for Next.js dashboard
    export_picks_json(recs_df, week, season, game_lines_df=game_lines_df, parlays_df=parlays_df)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NFL QUANT Professional Dashboard")
    parser.add_argument("--week", type=int, default=None, help="Week number")
    parser.add_argument("--json-only", action="store_true", help="Only export JSON for Next.js (skip HTML)")
    args = parser.parse_args()

    if args.json_only:
        # Load data and export JSON only
        from nfl_quant.config import settings
        week = args.week or settings.CURRENT_WEEK
        recs_df = load_edge_recommendations(week)
        edge_count = len(recs_df)
        print(f"Loaded {edge_count} edge recommendations (with p_attd={('p_attd' in recs_df.columns) if len(recs_df) > 0 else False})")

        # MERGE with XGBoost predictions if edge recs are few (don't replace - preserves p_attd)
        if edge_count < 20:
            recs_path = PROJECT_ROOT / "reports" / "CURRENT_WEEK_RECOMMENDATIONS.csv"
            if recs_path.exists():
                print(f"Edge recommendations few ({edge_count}), supplementing with XGBoost predictions")
                xgb_df = pd.read_csv(recs_path)

                # Map confidence column to tier for proper categorization
                def map_confidence_to_tier(conf_str):
                    if pd.isna(conf_str):
                        return 'MODERATE'
                    conf_str = str(conf_str).upper()
                    if 'ELITE' in conf_str or conf_str == 'VERY HIGH' or '90' in conf_str:
                        return 'ELITE'
                    elif 'HIGH' in conf_str or '80' in conf_str or '75' in conf_str:
                        return 'STRONG'
                    elif 'MED' in conf_str or '60' in conf_str or '70' in conf_str:
                        return 'MODERATE'
                    else:
                        return 'MODERATE'

                if 'confidence' in xgb_df.columns:
                    xgb_df['effective_tier'] = xgb_df['confidence'].apply(map_confidence_to_tier)
                elif 'model_confidence' in xgb_df.columns:
                    xgb_df['effective_tier'] = xgb_df['model_confidence'].apply(
                        lambda x: 'ELITE' if x >= 85 else ('STRONG' if x >= 70 else 'MODERATE')
                    )
                else:
                    xgb_df['effective_tier'] = 'MODERATE'

                # Add defensive rankings
                xgb_df = add_defensive_rankings_to_df(xgb_df, week)

                # Filter XGBoost to only picks where projection aligns with pick direction
                def is_projection_aligned(row):
                    proj = row.get('model_projection', row.get('projection', 0))
                    line = row.get('line', 0)
                    pick = str(row.get('pick', '')).upper()
                    if pd.isna(proj) or pd.isna(line):
                        return True  # Keep if we can't check
                    if pick in ['OVER', 'YES']:
                        return proj > line
                    elif pick in ['UNDER', 'NO']:
                        return proj < line
                    return True

                original_count = len(xgb_df)
                xgb_df = xgb_df[xgb_df.apply(is_projection_aligned, axis=1)]
                print(f"  XGBoost: {len(xgb_df)} aligned picks (filtered {original_count - len(xgb_df)} misaligned)")

                # MERGE: keep edge recommendations, add XGBoost picks that don't overlap
                if edge_count > 0:
                    edge_keys = set(zip(recs_df['player'].str.lower(), recs_df['market']))
                    xgb_df['_key'] = list(zip(xgb_df['player'].str.lower(), xgb_df['market']))
                    xgb_new = xgb_df[~xgb_df['_key'].isin(edge_keys)].drop(columns=['_key'])
                    recs_df = pd.concat([recs_df, xgb_new], ignore_index=True)
                    print(f"  Merged: {edge_count} edge + {len(xgb_new)} XGBoost = {len(recs_df)} total")
                else:
                    recs_df = xgb_df

        # Load game lines for JSON export
        game_lines_df = None
        game_lines_paths = [
            PROJECT_ROOT / "reports" / f"WEEK{week}_GAME_LINE_RECOMMENDATIONS.csv",
            PROJECT_ROOT / "reports" / f"game_lines_predictions_week{week}.csv",
        ]
        for path in game_lines_paths:
            if path.exists():
                game_lines_df = pd.read_csv(path)
                print(f"Loaded {len(game_lines_df)} game line recommendations")
                break

        # Load parlays for JSON export
        parlays_df = None
        parlay_path = PROJECT_ROOT / "reports" / f"parlay_recommendations_week{week}_2025.csv"
        if parlay_path.exists():
            parlays_df = pd.read_csv(parlay_path)
            print(f"Loaded {len(parlays_df)} parlay recommendations")

        if len(recs_df) > 0:
            export_picks_json(recs_df, week, game_lines_df=game_lines_df, parlays_df=parlays_df)
        else:
            print(f"No recommendations found for week {week}")
    else:
        generate_dashboard(week=args.week)
