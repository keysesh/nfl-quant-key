#!/usr/bin/env python3
"""
Generate Edge-Based Recommendations

Uses the edge ensemble (LVT + Player Bias) for continuous markets,
TD Poisson edge for touchdown markets, and Game Line edge for spreads/totals.

Usage:
    python scripts/predict/generate_edge_recommendations.py --week 15
    python scripts/predict/generate_edge_recommendations.py --week 15 --market player_receptions
    python scripts/predict/generate_edge_recommendations.py --week 15 --include-td
    python scripts/predict/generate_edge_recommendations.py --week 15 --include-td --include-game-lines
    python scripts/predict/generate_edge_recommendations.py --week 15 --include-attd
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.edges.ensemble import EdgeEnsemble
from nfl_quant.config_paths import MODELS_DIR, DATA_DIR, REPORTS_DIR
from nfl_quant.features.trailing_stats import (
    load_player_stats_for_edge,
    compute_edge_trailing_stats,
    merge_edge_trailing_stats,
    compute_line_vs_trailing,
    EDGE_TRAILING_COL_MAP,
)
from nfl_quant.features.batch_extractor import (
    _add_v28_elo_situational_features,
    _add_player_injury_features,
    _load_v28_elo_ratings,
    _load_unified_injury_data,
)
from nfl_quant.features.rz_opportunity import compute_rz_opportunity_features
from nfl_quant.features.rz_td_conversion import load_and_compute_rz_td_rates
from configs.edge_config import EDGE_MARKETS, TD_POISSON_MARKETS, get_td_poisson_threshold, should_bet, get_injury_mode, InjuryPolicyMode
from configs.ensemble_config import GLOBAL_SETTINGS
from configs.model_config import smooth_sweet_spot, get_market_filter, MarketFilter
from nfl_quant.data.injury_loader import get_injuries, InjuryDataError
from nfl_quant.policy.injury_policy import apply_injury_policy, InjuryMode
from nfl_quant.edges.game_line_edge import GameLineEdge, generate_game_line_edge_recommendations
from nfl_quant.models.td_enhanced_model import TDEnhancedModel, TD_CONFIDENCE_THRESHOLDS

# Team name to abbreviation mapping
TEAM_NAME_TO_ABBREV = {
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


def team_name_to_abbrev(name) -> str:
    """Convert full team name to abbreviation."""
    if pd.isna(name) or not name:
        return ''
    name = str(name)
    # Already an abbreviation?
    if len(name) <= 3 and name.isupper():
        return name
    return TEAM_NAME_TO_ABBREV.get(name, name)


def _add_v23_opponent_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V23 opponent defense context features for prediction.

    Features added:
    - opp_def_epa: Opponent defense EPA (overall quality)
    - has_opponent_context: Binary flag for data availability
    - opp_pass_yds_def_vs_avg: Pass defense z-score (for Player Bias edge)
    - opp_rush_yds_def_vs_avg: Rush defense z-score (for Player Bias edge)
    """
    df = df.copy()

    # Initialize with defaults
    df['opp_def_epa'] = 0.0
    df['has_opponent_context'] = 0
    df['opp_pass_yds_def_vs_avg'] = 0.0
    df['opp_rush_yds_def_vs_avg'] = 0.0

    # Try to load team defensive EPA
    def_epa_path = DATA_DIR / 'nflverse' / 'team_defensive_epa.parquet'
    if def_epa_path.exists() and 'opponent' in df.columns:
        try:
            def_epa = pd.read_parquet(def_epa_path)

            # Get latest season/week EPA for each team
            if 'team' in def_epa.columns and 'def_epa' in def_epa.columns:
                # Get most recent EPA per team
                latest_epa = (
                    def_epa.sort_values(['season', 'week'])
                    .groupby('team')
                    .last()
                    .reset_index()[['team', 'def_epa']]
                )
                latest_epa = latest_epa.rename(columns={'team': 'opponent', 'def_epa': 'opp_def_epa'})

                # Merge by opponent
                df = df.merge(latest_epa, on='opponent', how='left', suffixes=('', '_new'))
                if 'opp_def_epa_new' in df.columns:
                    df['opp_def_epa'] = df['opp_def_epa_new'].fillna(0.0)
                    df = df.drop(columns=['opp_def_epa_new'])
                else:
                    df['opp_def_epa'] = df['opp_def_epa'].fillna(0.0)
        except Exception as e:
            print(f"  Could not load defensive EPA: {e}")

    # Try to load opponent features for defense z-scores (used by Player Bias edge)
    opp_features_path = DATA_DIR / 'nflverse' / 'opponent_features.parquet'
    if opp_features_path.exists() and 'opponent' in df.columns:
        try:
            opp_feat = pd.read_parquet(opp_features_path)

            # Get latest defense z-scores per team
            z_cols = ['opp_pass_yds_def_vs_avg', 'opp_rush_yds_def_vs_avg']
            available = [c for c in z_cols if c in opp_feat.columns]

            if available and 'team' in opp_feat.columns:
                latest_def = (
                    opp_feat.sort_values(['season', 'week'])
                    .groupby('team')
                    .last()
                    .reset_index()[['team'] + available]
                )
                latest_def = latest_def.rename(columns={'team': 'opponent'})

                # Merge
                df = df.merge(latest_def, on='opponent', how='left', suffixes=('', '_new'))
                for col in available:
                    new_col = f'{col}_new'
                    if new_col in df.columns:
                        df[col] = df[new_col].fillna(0.0)
                        df = df.drop(columns=[new_col])
                    else:
                        df[col] = df[col].fillna(0.0)
        except Exception as e:
            print(f"  Could not load opponent features: {e}")

    # Set has_opponent_context flag
    df['has_opponent_context'] = ((df['opp_def_epa'] != 0) |
                                   (df['opp_pass_yds_def_vs_avg'] != 0) |
                                   (df['opp_rush_yds_def_vs_avg'] != 0)).astype(int)

    return df


def validate_odds_freshness(odds_path: Path, max_age_hours: float = 4.0) -> bool:
    """
    Validate that odds file is not stale.

    Args:
        odds_path: Path to odds file
        max_age_hours: Maximum age in hours (default 4h for live betting)

    Returns:
        True if fresh, raises ValueError if stale
    """
    import os
    from datetime import timedelta

    if not odds_path.exists():
        raise FileNotFoundError(f"Odds file not found: {odds_path}")

    file_mtime = datetime.fromtimestamp(os.path.getmtime(odds_path))
    age = datetime.now() - file_mtime
    max_age = timedelta(hours=max_age_hours)

    if age > max_age:
        hours_old = age.total_seconds() / 3600
        raise ValueError(
            f"ODDS STALE: {odds_path.name} is {hours_old:.1f}h old (max: {max_age_hours}h). "
            f"Run: python scripts/fetch/fetch_odds.py --week <WEEK> to refresh."
        )

    print(f"  Odds freshness: {age.total_seconds()/60:.0f} min old (max: {max_age_hours*60:.0f} min)")
    return True


def load_current_odds(week: int, season: int = 2025, validate_freshness: bool = True) -> pd.DataFrame:
    """Load current odds for the week."""
    # Try multiple possible paths
    possible_paths = [
        DATA_DIR / 'odds' / f'odds_player_props_week{week}_{season}.csv',
        DATA_DIR / 'odds' / f'odds_player_props_{season}_week{week}.csv',
        DATA_DIR / f'odds_player_props_week{week}.csv',
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Loading odds from: {path}")

            # V27: Validate odds freshness for production safety
            if validate_freshness:
                try:
                    validate_odds_freshness(path)
                except ValueError as e:
                    print(f"  WARNING: {e}")
                    # Don't fail - just warn. User may want to backtest with old data.

            df = pd.read_csv(path, low_memory=False)
            # Normalize column names (player_name -> player)
            if 'player_name' in df.columns and 'player' not in df.columns:
                df = df.rename(columns={'player_name': 'player'})

            # Filter out non-player props (D/ST, special bets)
            if 'player' in df.columns:
                non_players = df['player'].str.contains('D/ST|No Touchdown|Field Goal|Safety', case=False, na=False)
                if non_players.sum() > 0:
                    df = df[~non_players]
                    print(f"  Filtered {non_players.sum()} non-player props")

            # Derive player team from weekly stats if not present
            if 'team' not in df.columns and 'player' in df.columns:
                try:
                    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
                    if stats_path.exists():
                        stats = pd.read_parquet(stats_path)
                        # Get most recent team for each player
                        latest = stats.sort_values(['season', 'week']).groupby('player_display_name').last()
                        team_lookup = latest['team'].to_dict()

                        # Create normalized lookup for better matching
                        def normalize_name(name):
                            if pd.isna(name):
                                return name
                            name = str(name)
                            # Remove Jr./Sr./III/II suffixes
                            for suffix in [' Jr.', ' Sr.', ' III', ' II', ' IV']:
                                name = name.replace(suffix, '')
                            # Remove periods from initials (A.J. -> AJ)
                            name = name.replace('.', '')
                            return name.strip()

                        # Build normalized lookup
                        normalized_lookup = {}
                        for player_name, team in team_lookup.items():
                            norm_name = normalize_name(player_name)
                            normalized_lookup[norm_name] = team
                            # Also keep original
                            normalized_lookup[player_name] = team

                        # Match with normalization
                        df['_norm_name'] = df['player'].apply(normalize_name)
                        df['team'] = df['_norm_name'].map(normalized_lookup)

                        # Fallback to direct match
                        still_missing = df['team'].isna()
                        df.loc[still_missing, 'team'] = df.loc[still_missing, 'player'].map(team_lookup)

                        df = df.drop(columns=['_norm_name'])

                        matched = df['team'].notna().sum()
                        print(f"  Matched {matched}/{len(df)} players to teams")
                except Exception as e:
                    print(f"  Warning: Could not derive player teams: {e}")

            # Deduplicate - keep first occurrence for each player+market
            if 'player' in df.columns and 'market' in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=['player', 'market'], keep='first')
                if len(df) < before:
                    print(f"  Deduplicated: {before} -> {len(df)} odds")

            # Filter to only unplayed games (commence_time in the future)
            if 'commence_time' in df.columns:
                try:
                    now_utc = pd.Timestamp.now(tz='UTC')
                    df['commence_time_parsed'] = pd.to_datetime(df['commence_time'], utc=True)
                    before_filter = len(df)
                    df = df[df['commence_time_parsed'] > now_utc]
                    df = df.drop(columns=['commence_time_parsed'])
                    filtered_out = before_filter - len(df)
                    if filtered_out > 0:
                        print(f"  Filtered out {filtered_out} props for games already started")
                    print(f"  Remaining props for unplayed games: {len(df)}")
                except Exception as e:
                    print(f"  Warning: Could not filter by game time: {e}")

            return df

    raise FileNotFoundError(
        f"No odds file found for week {week}. Tried:\n" +
        "\n".join(str(p) for p in possible_paths)
    )


def load_player_history() -> pd.DataFrame:
    """Load player historical betting data for features."""
    # Load enriched data for history
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'

    if enriched_path.exists():
        df = pd.read_csv(enriched_path, low_memory=False)
    else:
        combined_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
        df = pd.read_csv(combined_path, low_memory=False)

    return df


def prepare_features(
    odds_df: pd.DataFrame,
    history_df: pd.DataFrame,
    market: str,
    trailing_stats_df: pd.DataFrame = None,
    week: int = None,
    season: int = 2025,
) -> pd.DataFrame:
    """Prepare features for edge evaluation.

    Args:
        odds_df: Current week's odds
        history_df: Historical betting data for player stats
        market: Market to prepare features for
        trailing_stats_df: NFLverse stats with trailing columns (optional, computed if not provided)
        week: Current week (for filtering historical data)
        season: Current season
    """
    from nfl_quant.utils.player_names import normalize_player_name

    df = odds_df.copy()

    # Add season and week columns for trailing stats merge
    df['season'] = season
    df['week'] = week

    # Normalize player names
    df['player_norm'] = df['player'].apply(normalize_player_name)

    # Filter to market
    df = df[df['market'] == market].copy()

    if len(df) == 0:
        return df

    # Prepare history for player stats
    history = history_df.copy()
    history['player_norm'] = history['player'].apply(normalize_player_name)
    history = history.sort_values(['player_norm', 'season', 'week'])

    # Compute player history stats
    if 'under_hit' not in history.columns and 'actual' in history.columns and 'line' in history.columns:
        history['under_hit'] = (history['actual'] < history['line']).astype(int)

    player_stats = history[history['market'] == market].groupby('player_norm').agg({
        'under_hit': ['count', 'mean'],
    }).reset_index()
    player_stats.columns = ['player_norm', 'player_bet_count', 'player_under_rate']

    # Calculate player bias
    if 'actual' in history.columns and 'line' in history.columns:
        market_history = history[history['market'] == market].copy()
        market_history['diff'] = market_history['actual'] - market_history['line']
        player_bias = market_history.groupby('player_norm')['diff'].mean().reset_index()
        player_bias.columns = ['player_norm', 'player_bias']
        player_stats = player_stats.merge(player_bias, on='player_norm', how='left')

    # Calculate current season bias (2025 only)
    current_season = season
    season_history = history[(history['market'] == market) & (history['season'] == current_season)].copy()
    if len(season_history) > 0:
        season_stats = season_history.groupby('player_norm').agg({
            'under_hit': ['count', 'mean'],
        }).reset_index()
        season_stats.columns = ['player_norm', 'season_games_played', 'current_season_under_rate']
        player_stats = player_stats.merge(season_stats, on='player_norm', how='left')
    else:
        player_stats['season_games_played'] = 0
        player_stats['current_season_under_rate'] = 0.5

    # Merge player stats into odds
    df = df.merge(player_stats, on='player_norm', how='left')

    # NO DATA = NO BET: Do NOT fill critical fields with defaults
    # The ensemble will skip bets where required data is missing
    # Critical fields (leave NaN if missing):
    #   - player_under_rate (Player Bias requirement)
    #   - player_bet_count (min games requirement)
    #   - trailing_* stats (LVT requirement) - handled below
    # Non-critical fields (OK to fill with neutral defaults):
    df['player_bias'] = df['player_bias'].fillna(0.0)
    df['current_season_under_rate'] = df['current_season_under_rate'].fillna(0.5)
    df['season_games_played'] = df['season_games_played'].fillna(0)
    # Ensure player_bet_count is initialized (but NOT filled if missing)
    if 'player_bet_count' not in df.columns:
        df['player_bet_count'] = np.nan

    # =========================================================================
    # Compute trailing stats and LVT from NFLverse data
    # =========================================================================
    if trailing_stats_df is None:
        try:
            trailing_stats_df = load_player_stats_for_edge()
            # Filter to historical data only (prevent leakage)
            if week is not None:
                historical_mask = (
                    (trailing_stats_df['season'] < season) |
                    ((trailing_stats_df['season'] == season) & (trailing_stats_df['week'] < week))
                )
                trailing_stats_df = trailing_stats_df[historical_mask]
            trailing_stats_df = compute_edge_trailing_stats(trailing_stats_df)
        except Exception as e:
            print(f"  Warning: Could not load trailing stats: {e}")
            trailing_stats_df = None

    # Merge trailing stats if available
    if trailing_stats_df is not None:
        df = merge_edge_trailing_stats(df, trailing_stats_df, markets=[market])

    # Compute line_vs_trailing for this market
    df['line_vs_trailing'] = compute_line_vs_trailing(df, market)

    # Line features
    df['line_level'] = df['line']
    df['line_in_sweet_spot'] = df['line'].apply(lambda x: smooth_sweet_spot(x, market))
    df['LVT_in_sweet_spot'] = df['line_vs_trailing'] * df['line_in_sweet_spot']

    # Market features
    market_under_rate = history[history['market'] == market]['under_hit'].mean()
    df['market_under_rate'] = market_under_rate if not pd.isna(market_under_rate) else 0.5
    df['market_bias_strength'] = abs(df['market_under_rate'] - 0.5) * 2

    # Game context (defaults if not available)
    df['vegas_spread'] = df.get('vegas_spread', pd.Series(0, index=df.index))
    df['implied_team_total'] = df.get('implied_team_total', pd.Series(24.0, index=df.index))

    # Player features
    # Position-specific averages from data:
    # - target_share: WR=13.1%, TE=9.9%, RB=6.1% (old default 15% was too high)
    # - snap_share: QB=80%, WR=52%, TE=44%, RB=38% (old default 70% was too high)
    # - catch_rate: WR=63%, TE=72%, RB=79% (old default 65% was low for RB/TE)
    # Using overall averages as fallback when position not available
    df['target_share'] = df.get('target_share', pd.Series(0.10, index=df.index))  # 10% overall
    df['snap_share'] = df.get('snap_share', pd.Series(0.50, index=df.index))  # 50% overall (was 70%)
    df['trailing_catch_rate'] = df.get('trailing_catch_rate', pd.Series(0.68, index=df.index))  # 68% overall
    df['pos_rank'] = df.get('pos_rank', pd.Series(2, index=df.index))
    df['is_starter'] = (df['pos_rank'] == 1).astype(int)

    # Interaction features
    player_tendency = df['player_under_rate'] - 0.5
    df['LVT_x_player_tendency'] = df['line_vs_trailing'] * player_tendency
    df['LVT_x_player_bias'] = df['line_vs_trailing'] * df['player_bias']
    df['player_market_aligned'] = np.where(
        (df['player_under_rate'] > 0.5) == (df['market_under_rate'] > 0.5),
        1.0, -1.0
    )

    # =========================================================================
    # V28.1 Features - Elo, Injury, and Situational
    # =========================================================================
    # Ensure team/opponent columns exist for feature extraction
    if 'team' not in df.columns and 'player_team' in df.columns:
        df['team'] = df['player_team']

    # Derive opponent from home_team/away_team if available
    if 'opponent' not in df.columns and 'home_team' in df.columns and 'away_team' in df.columns:
        # Normalize team names to abbreviations
        def normalize_team(name):
            if pd.isna(name):
                return name
            # If already an abbreviation
            if len(str(name)) <= 3:
                return str(name).upper()
            return TEAM_NAME_TO_ABBREV.get(name, name)

        df['home_team_abbrev'] = df['home_team'].apply(normalize_team)
        df['away_team_abbrev'] = df['away_team'].apply(normalize_team)

        # Set opponent: if player's team is home, opponent is away (and vice versa)
        df['opponent'] = np.where(
            df['team'] == df['home_team_abbrev'],
            df['away_team_abbrev'],
            df['home_team_abbrev']
        )
        # Clean up temp columns
        df = df.drop(columns=['home_team_abbrev', 'away_team_abbrev'], errors='ignore')

    # Add Elo and situational features (elo_rating_home, elo_rating_away, elo_diff,
    # rest_days, hfa_adjustment, ybc_proxy)
    try:
        df = _add_v28_elo_situational_features(df)
    except Exception as e:
        print(f"  Warning: Could not add Elo/situational features: {e}")
        # Set defaults for V28.1 features
        df['elo_rating_home'] = 1500.0
        df['elo_rating_away'] = 1500.0
        df['elo_diff'] = 0.0
        df['rest_days'] = 7.0
        df['hfa_adjustment'] = 1.0
        df['ybc_proxy'] = 5.0

    # =========================================================================
    # V23 Opponent Defense Context (Dec 2025 Enhancement)
    # =========================================================================
    try:
        df = _add_v23_opponent_defense_features(df)
    except Exception as e:
        print(f"  Warning: Could not add opponent defense features: {e}")
        df['opp_def_epa'] = 0.0
        df['has_opponent_context'] = 0
        df['opp_pass_yds_def_vs_avg'] = 0.0
        df['opp_rush_yds_def_vs_avg'] = 0.0

    # Compute lvt_x_defense interaction
    df['lvt_x_defense'] = df['line_vs_trailing'] * df['opp_def_epa']

    # Add injury features (injury_status_encoded, practice_status_encoded,
    # has_injury_designation)
    try:
        df = _add_player_injury_features(df)
    except Exception as e:
        print(f"  Warning: Could not add injury features: {e}")
        df['injury_status_encoded'] = 0
        df['practice_status_encoded'] = 0
        df['has_injury_designation'] = 0

    return df


def load_td_poisson_edge():
    """Load the TD Poisson edge model."""
    try:
        from nfl_quant.models.td_poisson_edge import TDPoissonEdge
        edge = TDPoissonEdge.load()
        return edge
    except Exception as e:
        print(f"  Could not load TD Poisson edge: {e}")
        return None


def prepare_td_features(
    odds_df: pd.DataFrame,
    history_df: pd.DataFrame,
    market: str,
    week: int,
    season: int = 2025,
) -> pd.DataFrame:
    """Prepare features for TD Poisson edge evaluation."""
    from nfl_quant.utils.player_names import normalize_player_name

    df = odds_df.copy()

    # Normalize player names
    df['player_norm'] = df['player'].apply(normalize_player_name)

    # Filter to market
    df = df[df['market'] == market].copy()

    if len(df) == 0:
        return df

    # Derive opponent column from team/home_team/away_team (needed for RZ defense lookup)
    if 'opponent' not in df.columns and 'home_team' in df.columns and 'away_team' in df.columns:
        df['home_team_abbrev'] = df['home_team'].apply(
            lambda x: team_name_to_abbrev(x) if pd.notna(x) else ''
        )
        df['away_team_abbrev'] = df['away_team'].apply(
            lambda x: team_name_to_abbrev(x) if pd.notna(x) else ''
        )
        # Player's team
        if 'team' in df.columns:
            df['team_abbrev'] = df['team'].apply(
                lambda x: team_name_to_abbrev(x) if pd.notna(x) else ''
            )
            # LA/LAR normalization
            df['team_abbrev'] = df['team_abbrev'].replace('LA', 'LAR')
            # Set opponent: if player is on home team, opponent is away (and vice versa)
            df['opponent'] = np.where(
                df['team_abbrev'] == df['home_team_abbrev'],
                df['away_team_abbrev'],
                np.where(
                    df['team_abbrev'] == df['away_team_abbrev'],
                    df['home_team_abbrev'],
                    ''
                )
            )

    # Load player stats for trailing TD features
    try:
        stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
        if stats_path.exists():
            stats = pd.read_parquet(stats_path)
        else:
            stats = pd.read_csv(DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)

        stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

        # Filter to historical data only
        stats = stats[
            (stats['season'] < season) |
            ((stats['season'] == season) & (stats['week'] < week))
        ]

        stats = stats.sort_values(['player_norm', 'season', 'week'])

        # Compute trailing TD stats (EWMA with shift)
        ewma_span = 4

        if 'passing_tds' in stats.columns:
            stats['trailing_passing_tds'] = (
                stats.groupby('player_norm')['passing_tds']
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

        if 'passing_yards' in stats.columns:
            stats['trailing_passing_yards'] = (
                stats.groupby('player_norm')['passing_yards']
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

        if 'rushing_tds' in stats.columns:
            stats['trailing_rushing_tds'] = (
                stats.groupby('player_norm')['rushing_tds']
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

        if 'carries' in stats.columns:
            stats['trailing_carries'] = (
                stats.groupby('player_norm')['carries']
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

        if 'receiving_tds' in stats.columns:
            stats['trailing_receiving_tds'] = (
                stats.groupby('player_norm')['receiving_tds']
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

        if 'targets' in stats.columns:
            stats['trailing_targets'] = (
                stats.groupby('player_norm')['targets']
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

        # Get most recent stats per player
        latest_stats = stats.groupby('player_norm').last().reset_index()

        # Merge trailing stats
        trailing_cols = [
            'trailing_passing_tds', 'trailing_passing_yards',
            'trailing_rushing_tds', 'trailing_carries',
            'trailing_receiving_tds', 'trailing_targets',
        ]
        available_cols = ['player_norm'] + [c for c in trailing_cols if c in latest_stats.columns]

        df = df.merge(latest_stats[available_cols], on='player_norm', how='left')

    except Exception as e:
        print(f"  Warning: Could not compute trailing TD stats: {e}")

    # Add Vegas features
    df['vegas_total'] = df.get('vegas_total', pd.Series(45.0, index=df.index)).fillna(45.0)
    df['vegas_spread'] = df.get('vegas_spread', pd.Series(0.0, index=df.index)).fillna(0.0)

    # Compute opponent RZ TD defense from PBP data
    # Actual rates from PBP data (2023-2024): Rush=17.2%, Pass=21.4%
    df['opponent_pass_td_allowed'] = 0.214  # Actual RZ pass TD rate (was 0.08)
    df['opponent_rush_td_allowed'] = 0.172  # Actual RZ rush TD rate (was 0.12)
    df['opp_rz_td_allowed'] = 0.20  # Default total RZ TD rate

    try:
        from nfl_quant.features.defensive_metrics import DefensiveMetricsExtractor

        # Load extractor for current season
        extractor = DefensiveMetricsExtractor(season=season)

        # Compute opponent RZ TD defense for each row
        for idx, row in df.iterrows():
            opponent = row.get('opponent')
            if pd.isna(opponent):
                continue

            try:
                rz_defense = extractor.get_rz_td_defense(
                    defense_team=str(opponent),
                    current_week=int(week),
                    trailing_weeks=4
                )
                df.loc[idx, 'opponent_pass_td_allowed'] = rz_defense['rz_pass_td_rate']
                df.loc[idx, 'opponent_rush_td_allowed'] = rz_defense['rz_rush_td_rate']
                df.loc[idx, 'opp_rz_td_allowed'] = rz_defense['rz_total_td_rate']
            except Exception:
                pass  # Keep defaults

        computed = (df['opponent_pass_td_allowed'] != 0.214).sum()
        print(f"  Computed opponent RZ defense for {computed}/{len(df)} props")

    except Exception as e:
        print(f"  Warning: Could not compute opponent RZ defense: {e}")

    # Fill missing trailing stats with defaults
    for col in ['trailing_passing_tds', 'trailing_rushing_tds', 'trailing_receiving_tds']:
        if col not in df.columns:
            df[col] = 1.5 if 'passing' in col else 0.5
        else:
            default = 1.5 if 'passing' in col else 0.5
            df[col] = df[col].fillna(default)

    for col in ['trailing_passing_yards', 'trailing_carries', 'trailing_targets']:
        if col not in df.columns:
            df[col] = 200.0 if 'yards' in col else 15.0
        else:
            default = 200.0 if 'yards' in col else 15.0
            df[col] = df[col].fillna(default)

    # Red zone features (defaults - would need PBP data for real values)
    df['rz_pass_attempts_share'] = 0.15
    df['trailing_rz_rush_share'] = 0.10
    df['trailing_rz_target_share'] = 0.10

    # =========================================================================
    # V28.1 Features - Elo, Injury, and Situational
    # =========================================================================
    # Ensure team column exists for feature extraction
    if 'team' not in df.columns and 'player_team' in df.columns:
        df['team'] = df['player_team']

    # Derive opponent from home_team/away_team if available
    if 'opponent' not in df.columns and 'home_team' in df.columns and 'away_team' in df.columns:
        def normalize_team(name):
            if pd.isna(name):
                return name
            if len(str(name)) <= 3:
                return str(name).upper()
            return TEAM_NAME_TO_ABBREV.get(name, name)

        df['home_team_abbrev'] = df['home_team'].apply(normalize_team)
        df['away_team_abbrev'] = df['away_team'].apply(normalize_team)
        df['opponent'] = np.where(
            df['team'] == df['home_team_abbrev'],
            df['away_team_abbrev'],
            df['home_team_abbrev']
        )
        df = df.drop(columns=['home_team_abbrev', 'away_team_abbrev'], errors='ignore')

    # Add Elo and situational features
    try:
        df = _add_v28_elo_situational_features(df)
    except Exception as e:
        print(f"  Warning: Could not add Elo/situational features for TD: {e}")
        df['elo_rating_home'] = 1500.0
        df['elo_rating_away'] = 1500.0
        df['elo_diff'] = 0.0
        df['rest_days'] = 7.0
        df['hfa_adjustment'] = 1.0
        df['ybc_proxy'] = 5.0

    # Add injury features
    try:
        df = _add_player_injury_features(df)
    except Exception as e:
        print(f"  Warning: Could not add injury features for TD: {e}")
        df['injury_status_encoded'] = 0
        df['practice_status_encoded'] = 0
        df['has_injury_designation'] = 0

    return df


def generate_td_recommendations(
    td_edge,
    odds_df: pd.DataFrame,
    history_df: pd.DataFrame,
    week: int,
    season: int = 2025,
    min_games: int = 4,  # Minimum games required (skip low-data players)
) -> list:
    """Generate TD prop recommendations using Poisson edge.

    Args:
        min_games: Minimum games played in season to make prediction (default: 5)
    """
    recommendations = []

    # Load player game counts from NFLverse
    try:
        stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
        if stats_path.exists():
            stats = pd.read_parquet(stats_path)
            season_stats = stats[stats['season'] == season]
            # Count games per player this season
            player_games = season_stats.groupby('player_display_name').size().to_dict()
        else:
            player_games = {}
    except Exception:
        player_games = {}

    for market in TD_POISSON_MARKETS:
        # Check if market is in odds
        if market not in odds_df['market'].unique():
            continue

        print(f"\nProcessing TD market: {market}...")

        # Check if model is trained for this market
        if market not in td_edge.models:
            print(f"  No model trained for {market}")
            continue

        # Prepare features
        market_df = prepare_td_features(odds_df, history_df, market, week, season)

        if len(market_df) == 0:
            print(f"  No data for {market}")
            continue

        print(f"  {len(market_df)} props")

        # Get threshold
        try:
            threshold = get_td_poisson_threshold(market)
            min_confidence = threshold.min_confidence
        except ValueError:
            min_confidence = 0.58

        # Evaluate each prop
        skipped_low_sample = 0
        for _, row in market_df.iterrows():
            player_name = row.get('player', 'Unknown')
            games_played = player_games.get(player_name, 0)

            # Skip players with insufficient sample size
            if games_played < min_games:
                skipped_low_sample += 1
                continue

            # Build game context (same pattern as ensemble loop)
            home_team_raw = row.get('home_team', '')
            away_team_raw = row.get('away_team', '')
            home_team = team_name_to_abbrev(home_team_raw) if home_team_raw else ''
            away_team = team_name_to_abbrev(away_team_raw) if away_team_raw else ''
            player_team_raw = row.get('team', '')
            player_team = team_name_to_abbrev(player_team_raw) if player_team_raw else ''
            if player_team == 'LA':
                player_team = 'LAR'
            game_str = f"{away_team} @ {home_team}" if home_team and away_team else ''
            # Determine opponent based on player's team
            if player_team == home_team:
                opponent = away_team
            elif player_team == away_team:
                opponent = home_team
            else:
                opponent = ''

            features = pd.DataFrame([row])
            line = row.get('line', 1.5)

            try:
                result = td_edge.evaluate_bet(row, market, line, min_confidence)

                if result['should_bet']:
                    recommendations.append({
                        'player': player_name,
                        'team': player_team,
                        'market': market,
                        'line': line,
                        'direction': result['direction'],
                        'pick': result['direction'],  # Copy direction to pick for dashboard compatibility
                        'units': 1.0,  # Base unit for TD props
                        'source': 'TD_POISSON',
                        'lvt_confidence': 0.0,
                        'player_bias_confidence': 0.0,
                        'combined_confidence': result['confidence'],
                        'reasoning': f"Expected TDs: {result['expected_tds']:.2f}, P({result['direction']}): {result['confidence']:.1%} ({games_played} games)",
                        'player_under_rate': 0.5,
                        'player_bet_count': games_played,
                        'current_season_under_rate': 0.5,
                        'season_games_played': games_played,
                        # Game context fields
                        'game': game_str,
                        'opponent': opponent,
                        'home_team': home_team,
                        'away_team': away_team,
                        'commence_time': row.get('commence_time', ''),
                        'p_over': result.get('p_over', 0.5),
                        'p_under': result.get('p_under', 0.5),
                        'expected_tds': result.get('expected_tds', 0),
                    })
            except Exception as e:
                continue

        if skipped_low_sample > 0:
            print(f"  Skipped {skipped_low_sample} players with <{min_games} games")

    return recommendations


def generate_attd_recommendations(
    attd_ensemble,
    odds_df: pd.DataFrame,
    history_df: pd.DataFrame,
    week: int,
    season: int = 2025,
    min_games: int = 4,  # Skip players with insufficient data
    min_confidence: float = 0.55,
) -> list:
    """Generate Anytime TD recommendations using ATTD ensemble.

    Args:
        attd_ensemble: Trained ATTDEnsemble model
        odds_df: Current odds DataFrame
        history_df: Historical player data
        week: Current week
        season: Season year
        min_games: Minimum games required
        min_confidence: Minimum probability threshold
    """
    recommendations = []

    # Load player stats for position, game count, and trailing TD stats
    try:
        stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
        if stats_path.exists():
            stats = pd.read_parquet(stats_path)
            # Filter to historical data only (before current week)
            historical_stats = stats[
                (stats['season'] < season) |
                ((stats['season'] == season) & (stats['week'] < week))
            ]
            season_stats = stats[stats['season'] == season]
            player_games = season_stats.groupby('player_display_name').size().to_dict()
            player_positions = season_stats.groupby('player_display_name')['position'].first().to_dict()

            # Create player_id to name mapping for RZ features
            player_id_to_name = stats.groupby('player_id')['player_display_name'].first().to_dict()
            player_name_to_id = {v: k for k, v in player_id_to_name.items()}

            # Compute trailing TD stats per player (EWMA over last 6 games)
            historical_stats = historical_stats.sort_values(['player_display_name', 'season', 'week'])
            ewma_span = 6

            # Trailing rushing TDs
            if 'rushing_tds' in historical_stats.columns:
                historical_stats['trailing_rushing_tds'] = (
                    historical_stats.groupby('player_display_name')['rushing_tds']
                    .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
                )
            # Trailing receiving TDs
            if 'receiving_tds' in historical_stats.columns:
                historical_stats['trailing_receiving_tds'] = (
                    historical_stats.groupby('player_display_name')['receiving_tds']
                    .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
                )
            # Trailing carries (for RZ proxy fallback)
            if 'carries' in historical_stats.columns:
                historical_stats['trailing_carries'] = (
                    historical_stats.groupby('player_display_name')['carries']
                    .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
                )
            # Trailing targets (for RZ proxy fallback)
            if 'targets' in historical_stats.columns:
                historical_stats['trailing_targets'] = (
                    historical_stats.groupby('player_display_name')['targets']
                    .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
                )

            # Get latest stats per player
            player_trailing_stats = historical_stats.groupby('player_display_name').last().reset_index()
            player_trailing_stats = player_trailing_stats.set_index('player_display_name')
        else:
            player_games = {}
            player_positions = {}
            player_trailing_stats = pd.DataFrame()
            player_name_to_id = {}
    except Exception as e:
        print(f"  Warning: Could not load player stats: {e}")
        player_games = {}
        player_positions = {}
        player_trailing_stats = pd.DataFrame()
        player_name_to_id = {}

    # Load ACTUAL RZ features from PBP data (not proxies)
    rz_features_by_player = {}
    rz_td_rates_by_player = {}
    try:
        # Compute RZ opportunity features from PBP
        rz_opps = compute_rz_opportunity_features(season=season)
        if not rz_opps.empty:
            # Get latest week's features per player
            latest_rz = rz_opps.sort_values(['player_id', 'week']).groupby('player_id').last()
            # Map player_id to name
            for player_id, row in latest_rz.iterrows():
                player_name = player_id_to_name.get(player_id)
                if player_name:
                    rz_features_by_player[player_name] = {
                        'trailing_rz_carries': row.get('trailing_rz_carries', 0),
                        'trailing_rz_targets': row.get('trailing_rz_targets', 0),
                        'trailing_rz_touches': row.get('trailing_rz_touches', 0),
                    }
            print(f"  Loaded actual RZ features for {len(rz_features_by_player)} players from PBP")

        # Compute RZ TD rates from PBP
        rz_rates = load_and_compute_rz_td_rates(season=season)
        if not rz_rates.empty:
            latest_rates = rz_rates.sort_values(['player_id', 'week']).groupby('player_id').last()
            for player_id, row in latest_rates.iterrows():
                player_name = player_id_to_name.get(player_id)
                if player_name:
                    rz_td_rates_by_player[player_name] = {
                        'rz_td_per_carry': row.get('rz_td_per_carry', 0.172),  # Actual: 17.2%
                        'rz_td_per_target': row.get('rz_td_per_target', 0.214),  # Actual: 21.4%
                    }
            print(f"  Loaded RZ TD rates for {len(rz_td_rates_by_player)} players from PBP")
    except Exception as e:
        print(f"  Warning: Could not compute RZ features from PBP: {e}")
        print(f"  Falling back to proxy-based RZ features")

    # Load snap counts for snap_share
    snap_share_lookup = {}
    try:
        snap_path = DATA_DIR / 'nflverse' / 'snap_counts.parquet'
        if snap_path.exists():
            snaps = pd.read_parquet(snap_path)
            season_snaps = snaps[snaps['season'] == season]
            # Get average snap share per player
            if 'offense_pct' in season_snaps.columns:
                snap_share_lookup = season_snaps.groupby('player')['offense_pct'].mean().to_dict()
    except Exception:
        pass

    # Load ACTUAL goal-line features from PBP
    gl_features_by_player = {}
    try:
        from nfl_quant.features.goal_line_detector import load_and_compute_goal_line_roles
        gl_roles = load_and_compute_goal_line_roles(season=season)
        if not gl_roles.empty:
            # Get latest GL features per player (use trailing values)
            latest_gl = gl_roles.sort_values(['player_id', 'week']).groupby('player_id').last()
            for player_id, row in latest_gl.iterrows():
                player_name = player_id_to_name.get(player_id)
                if player_name:
                    gl_features_by_player[player_name] = {
                        'gl_carry_share': row.get('gl_carry_share', 0.0),
                        'gl_target_share': row.get('gl_target_share', 0.0),
                        'is_primary_gl_back': row.get('is_primary_gl_back', 0),
                        'is_primary_gl_receiver': row.get('is_primary_gl_receiver', 0),
                    }
            print(f"  Loaded actual GL features for {len(gl_features_by_player)} players from PBP")
    except Exception as e:
        print(f"  Warning: Could not compute GL features from PBP: {e}")
        print(f"  Falling back to proxy-based GL features")

    # Load opponent RZ TD defense
    opp_rz_defense = {}
    try:
        from nfl_quant.features.defensive_metrics import DefensiveMetricsExtractor
        extractor = DefensiveMetricsExtractor(season=season)
        # We'll compute per-opponent below
    except Exception:
        extractor = None

    # Check for anytime_td market in odds, or use player_anytime_td
    attd_markets = ['player_anytime_td', 'anytime_td']
    market_found = None
    for market in attd_markets:
        if market in odds_df['market'].unique():
            market_found = market
            break

    if market_found is None:
        print("  No explicit ATTD market found, using player positions from stats...")
        all_players = odds_df['player'].unique() if 'player' in odds_df.columns else []
        if len(all_players) == 0:
            print("  No players found for ATTD prediction")
            return recommendations
    else:
        print(f"  Found ATTD market: {market_found}")
        all_players = odds_df[odds_df['market'] == market_found]['player'].unique()

    print(f"  Processing {len(all_players)} players...")

    skipped_no_position = 0
    skipped_low_sample = 0
    features_computed = 0

    for player_name in all_players:
        # Get position
        position = player_positions.get(player_name)
        if position not in ['QB', 'RB', 'WR', 'TE']:
            skipped_no_position += 1
            continue

        # Check if model trained for this position
        if position not in attd_ensemble.logistic_models:
            continue

        # Check game count
        games_played = player_games.get(player_name, 0)
        if games_played < min_games:
            skipped_low_sample += 1
            continue

        # Get player row from odds (for game context)
        player_rows = odds_df[odds_df['player'] == player_name]
        if len(player_rows) == 0:
            continue

        row = player_rows.iloc[0]

        # Build game context
        home_team_raw = row.get('home_team', '')
        away_team_raw = row.get('away_team', '')
        home_team = team_name_to_abbrev(home_team_raw) if home_team_raw else ''
        away_team = team_name_to_abbrev(away_team_raw) if away_team_raw else ''
        player_team_raw = row.get('team', '')
        player_team = team_name_to_abbrev(player_team_raw) if player_team_raw else ''
        if player_team == 'LA':
            player_team = 'LAR'
        game_str = f"{away_team} @ {home_team}" if home_team and away_team else ''
        opponent = away_team if player_team == home_team else home_team if player_team == away_team else ''

        # Get player's trailing stats from computed data
        trailing_rushing_tds = 0.0
        trailing_receiving_tds = 0.0
        trailing_carries = 0.0
        trailing_targets = 0.0

        if player_name in player_trailing_stats.index:
            player_stats = player_trailing_stats.loc[player_name]
            trailing_rushing_tds = player_stats.get('trailing_rushing_tds', 0.0)
            trailing_receiving_tds = player_stats.get('trailing_receiving_tds', 0.0)
            trailing_carries = player_stats.get('trailing_carries', 0.0)
            trailing_targets = player_stats.get('trailing_targets', 0.0)
            if pd.isna(trailing_rushing_tds):
                trailing_rushing_tds = 0.0
            if pd.isna(trailing_receiving_tds):
                trailing_receiving_tds = 0.0
            if pd.isna(trailing_carries):
                trailing_carries = 0.0
            if pd.isna(trailing_targets):
                trailing_targets = 0.0
            features_computed += 1

        # Use ACTUAL RZ features from PBP if available, otherwise fall back to proxies
        if player_name in rz_features_by_player:
            # Use actual PBP-computed RZ features
            rz_feats = rz_features_by_player[player_name]
            trailing_rz_carries = rz_feats.get('trailing_rz_carries', 0)
            trailing_rz_targets = rz_feats.get('trailing_rz_targets', 0)
            trailing_rz_touches = rz_feats.get('trailing_rz_touches', 0)
        else:
            # Fall back to proxy-based estimates
            if position == 'RB':
                trailing_rz_carries = trailing_carries * 0.15
                trailing_rz_targets = trailing_targets * 0.12
            elif position == 'QB':
                trailing_rz_carries = trailing_carries * 0.10
                trailing_rz_targets = 0.0
            elif position in ['WR', 'TE']:
                trailing_rz_carries = 0.0
                trailing_rz_targets = trailing_targets * 0.12
            else:
                trailing_rz_carries = 0.0
                trailing_rz_targets = 0.0
            trailing_rz_touches = trailing_rz_carries + trailing_rz_targets

        # Use ACTUAL RZ TD rates from PBP if available
        # Actual rates from 2023-2024 PBP data:
        # - RZ Rush TD rate: 17.2% (was 12%)
        # - RZ Pass/Rec TD rate: 21.4% (was 8%)
        ACTUAL_RZ_RUSH_TD_RATE = 0.172
        ACTUAL_RZ_REC_TD_RATE = 0.214

        if player_name in rz_td_rates_by_player:
            rz_rates = rz_td_rates_by_player[player_name]
            rz_td_per_carry = rz_rates.get('rz_td_per_carry', ACTUAL_RZ_RUSH_TD_RATE)
            rz_td_per_target = rz_rates.get('rz_td_per_target', ACTUAL_RZ_REC_TD_RATE)
        else:
            # Fall back to position-based defaults using actual league rates
            if position == 'RB':
                # RBs with high TD history get slight bump, others get actual average
                rz_td_per_carry = 0.20 if trailing_rushing_tds > 0.3 else ACTUAL_RZ_RUSH_TD_RATE
            elif position == 'QB':
                rz_td_per_carry = 0.10  # QBs have lower RZ rush TD rate
            else:
                rz_td_per_carry = 0.0
            # Estimate rz_td_per_target - use actual average as baseline
            if trailing_rz_targets > 0 and trailing_receiving_tds > 0:
                rz_td_per_target = min(trailing_receiving_tds / max(trailing_rz_targets, 1), 0.35)
            else:
                rz_td_per_target = ACTUAL_RZ_REC_TD_RATE

        # Use ACTUAL goal-line features from PBP if available
        if player_name in gl_features_by_player:
            gl_feats = gl_features_by_player[player_name]
            gl_carry_share = gl_feats.get('gl_carry_share', 0.0)
            gl_target_share = gl_feats.get('gl_target_share', 0.0)
        else:
            # Fall back to proxy-based estimates
            if position == 'RB':
                gl_carry_share = 0.25 if trailing_rz_carries > 2.0 else 0.15 if trailing_rz_carries > 1.0 else 0.10
            elif position == 'QB':
                gl_carry_share = 0.10
            else:
                gl_carry_share = 0.0

            gl_target_share = 0.15 if trailing_rz_targets > 1.5 and position == 'TE' else 0.10 if trailing_rz_targets > 1.0 else 0.05

        # Get snap share
        snap_share = snap_share_lookup.get(player_name, 0.50)
        if snap_share > 1.0:
            snap_share = snap_share / 100.0  # Convert from percentage

        # Get opponent RZ TD defense
        opp_rz_td_allowed = 0.20  # Default
        if extractor and opponent:
            try:
                rz_defense = extractor.get_rz_td_defense(
                    defense_team=opponent,
                    current_week=int(week),
                    trailing_weeks=4
                )
                opp_rz_td_allowed = rz_defense.get('rz_total_td_rate', 0.20)
            except Exception:
                pass

        # Build feature row with computed values
        feature_row = {
            'trailing_rz_carries': trailing_rz_carries,
            'trailing_rz_targets': trailing_rz_targets,
            'trailing_rz_touches': trailing_rz_touches,
            'trailing_rushing_tds': trailing_rushing_tds,
            'trailing_receiving_tds': trailing_receiving_tds,
            'rz_td_per_carry': rz_td_per_carry,
            'rz_td_per_target': rz_td_per_target,
            'gl_carry_share': gl_carry_share,
            'gl_target_share': gl_target_share,
            'snap_share': snap_share,
            'opp_rz_td_allowed': opp_rz_td_allowed,
            'vegas_total': row.get('vegas_total', 45.0) if pd.notna(row.get('vegas_total')) else 45.0,
            'vegas_spread': row.get('vegas_spread', 0.0) if pd.notna(row.get('vegas_spread')) else 0.0,
        }

        try:
            result = attd_ensemble.evaluate_bet(
                pd.Series(feature_row),
                position=position,
                min_confidence=min_confidence,
                implied_odds=0.5,  # Assume even odds
            )

            if result['should_bet']:
                recommendations.append({
                    'player': player_name,
                    'team': player_team,
                    'market': 'player_anytime_td',
                    'line': 0.5,  # ATTD has no line (binary)
                    'direction': 'YES',
                    'units': 1.0,
                    'source': 'ATTD_ENSEMBLE',
                    'lvt_confidence': 0.0,
                    'player_bias_confidence': 0.0,
                    'combined_confidence': result['p_attd'],
                    'reasoning': f"P(ATTD): {result['p_attd']:.1%}, Edge: {result['edge']:.1%} ({position}, {games_played} games)",
                    'player_under_rate': 0.5,
                    'player_bet_count': games_played,
                    'current_season_under_rate': 0.5,
                    'season_games_played': games_played,
                    'game': game_str,
                    'opponent': opponent,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': row.get('commence_time', ''),
                    'p_attd': result['p_attd'],
                    'position': position,
                })
        except Exception:
            continue

    print(f"  Computed trailing stats for {features_computed} players")
    if skipped_no_position > 0:
        print(f"  Skipped {skipped_no_position} players without valid position")
    if skipped_low_sample > 0:
        print(f"  Skipped {skipped_low_sample} players with <{min_games} games")

    return recommendations


def generate_td_enhanced_recommendations(
    td_model: TDEnhancedModel,
    odds_df: pd.DataFrame,
    week: int,
    season: int = 2025,
    min_games: int = 4,
) -> list:
    """
    Generate Anytime TD recommendations using the enhanced XGBoost TD model.

    Walk-forward validated: RB @ 60% = 58.2% WR, +6.6% ROI

    Args:
        td_model: Trained TDEnhancedModel
        odds_df: Current odds DataFrame
        week: Current week
        season: Season year
        min_games: Minimum games required

    Returns:
        List of recommendation dicts
    """
    from nfl_quant.utils.player_names import normalize_player_name

    recommendations = []

    # Load weekly stats for features
    try:
        stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
        if not stats_path.exists():
            print("  Warning: weekly_stats.parquet not found")
            return recommendations

        stats = pd.read_parquet(stats_path)
        stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

        # Filter to historical data only (before current week)
        historical_stats = stats[
            (stats['season'] < season) |
            ((stats['season'] == season) & (stats['week'] < week))
        ]

        # Get season game counts
        season_stats = stats[stats['season'] == season]
        player_games = season_stats.groupby('player_display_name').size().to_dict()
        player_positions = season_stats.groupby('player_display_name')['position'].first().to_dict()

        # Compute trailing features
        historical_stats = historical_stats.sort_values(['player_norm', 'season', 'week'])
        ewma_span = 6

        # Trailing stats needed for TD model
        trailing_cols = {
            'rushing_tds': 'trailing_rushing_tds',
            'receiving_tds': 'trailing_receiving_tds',
            'targets': 'trailing_targets',
            'carries': 'trailing_carries',
            'target_share': 'trailing_target_share',
            'receiving_yards': 'trailing_rec_yds',
            'rushing_yards': 'trailing_rush_yds',
        }

        for src, dst in trailing_cols.items():
            if src in historical_stats.columns:
                historical_stats[dst] = (
                    historical_stats.groupby('player_norm')[src]
                    .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=2).mean())
                )

        # Compute total trailing TDs
        historical_stats['trailing_tds'] = (
            historical_stats.get('trailing_rushing_tds', 0).fillna(0) +
            historical_stats.get('trailing_receiving_tds', 0).fillna(0)
        )

        # Get latest stats per player
        player_trailing = historical_stats.groupby('player_norm').last().reset_index()
        player_trailing = player_trailing.set_index('player_norm')

    except Exception as e:
        print(f"  Error loading stats: {e}")
        return recommendations

    # Find anytime TD market in odds
    attd_markets = ['player_anytime_td', 'anytime_td']
    market_found = None
    for market in attd_markets:
        if market in odds_df['market'].unique():
            market_found = market
            break

    if market_found:
        all_players = odds_df[odds_df['market'] == market_found]['player'].unique()
    else:
        # Use players from any market
        all_players = odds_df['player'].unique() if 'player' in odds_df.columns else []

    print(f"  Processing {len(all_players)} players...")

    skipped_no_position = 0
    skipped_low_sample = 0
    skipped_no_data = 0

    for player_name in all_players:
        # Get position - only RB, WR, TE
        position = player_positions.get(player_name)
        if position not in ['RB', 'WR', 'TE']:
            skipped_no_position += 1
            continue

        # Check game count
        games_played = player_games.get(player_name, 0)
        if games_played < min_games:
            skipped_low_sample += 1
            continue

        # Normalize name for lookup
        player_norm = normalize_player_name(player_name)

        # Get trailing stats
        if player_norm not in player_trailing.index:
            skipped_no_data += 1
            continue

        player_stats = player_trailing.loc[player_norm]

        # Build feature row
        feature_row = {
            'trailing_tds': player_stats.get('trailing_tds', 0),
            'trailing_targets': player_stats.get('trailing_targets', 0),
            'trailing_carries': player_stats.get('trailing_carries', 0),
            'trailing_target_share': player_stats.get('trailing_target_share', 0),
            'trailing_rec_yds': player_stats.get('trailing_rec_yds', 0),
            'trailing_rush_yds': player_stats.get('trailing_rush_yds', 0),
            'is_rb': 1 if position == 'RB' else 0,
            'is_wr': 1 if position == 'WR' else 0,
            'is_te': 1 if position == 'TE' else 0,
            # RZ and opponent features - defaults (would need PBP for real values)
            'rz_targets_per_game': 0.0,
            'rz_carries_per_game': 0.0,
            'gl_carries_per_game': 0.0,
            'rz_td_rate': 0.0,
            'opp_tds_allowed_per_game': 2.5,
            'opp_pass_tds_allowed': 1.5,
            'opp_rush_tds_allowed': 1.0,
            'opp_rz_td_rate': 0.50,
            'team_rz_td_rate': 0.50,
        }

        # Fill NaN values
        for k, v in feature_row.items():
            if pd.isna(v):
                feature_row[k] = 0.0

        # Predict
        try:
            features_df = pd.DataFrame([feature_row])
            p_td = td_model.predict_proba(features_df)[0]
        except Exception as e:
            continue

        # Get position-specific threshold
        threshold = TD_CONFIDENCE_THRESHOLDS.get(position, TD_CONFIDENCE_THRESHOLDS['default'])

        # Check if above threshold
        if p_td < threshold:
            continue

        # Get game context from odds
        player_rows = odds_df[odds_df['player'] == player_name]
        if len(player_rows) == 0:
            continue

        row = player_rows.iloc[0]

        # Build game context
        home_team_raw = row.get('home_team', '')
        away_team_raw = row.get('away_team', '')
        home_team = team_name_to_abbrev(home_team_raw) if home_team_raw else ''
        away_team = team_name_to_abbrev(away_team_raw) if away_team_raw else ''
        player_team_raw = row.get('team', '')
        player_team = team_name_to_abbrev(player_team_raw) if player_team_raw else ''
        if player_team == 'LA':
            player_team = 'LAR'
        game_str = f"{away_team} @ {home_team}" if home_team and away_team else ''
        opponent = away_team if player_team == home_team else home_team if player_team == away_team else ''

        # Add recommendation
        recommendations.append({
            'player': player_name,
            'team': player_team,
            'market': 'player_anytime_td',
            'line': 0.5,  # ATTD has no line (binary)
            'pick': 'YES',  # For dashboard display
            'direction': 'YES',
            'units': 1.0,
            'source': 'TD_ENHANCED',
            'lvt_confidence': 0.0,
            'player_bias_confidence': 0.0,
            'combined_confidence': p_td,
            'reasoning': f"P(TD): {p_td:.1%} >= {threshold:.0%} threshold ({position}, {games_played} games)",
            'player_under_rate': 0.5,
            'player_bet_count': games_played,
            'current_season_under_rate': 0.5,
            'season_games_played': games_played,
            'game': game_str,
            'opponent': opponent,
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': row.get('commence_time', ''),
            'p_td': p_td,
            'position': position,
        })

    print(f"  Generated {len(recommendations)} TD Enhanced recommendations")
    if skipped_no_position > 0:
        print(f"  Skipped {skipped_no_position} players without RB/WR/TE position")
    if skipped_low_sample > 0:
        print(f"  Skipped {skipped_low_sample} players with <{min_games} games")
    if skipped_no_data > 0:
        print(f"  Skipped {skipped_no_data} players without trailing stats")

    return recommendations


def get_active_teams(week: int, season: int = 2025) -> set:
    """
    Get set of teams playing this week (not on bye).

    Used to filter out bye-week players from recommendations.
    Returns both abbreviations AND full team names for matching.
    """
    # Team name to abbreviation mapping
    TEAM_ABBREV = {
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
    ABBREV_TO_TEAM = {v: k for k, v in TEAM_ABBREV.items()}

    schedules_path = DATA_DIR / 'nflverse' / 'schedules.parquet'

    if not schedules_path.exists():
        print("  Warning: schedules.parquet not found, skipping bye week validation")
        return set()

    schedules = pd.read_parquet(schedules_path)

    # Filter to this week's games
    week_games = schedules[
        (schedules['season'] == season) &
        (schedules['week'] == week)
    ]

    if len(week_games) == 0:
        print(f"  Warning: No games found for week {week}")
        return set()

    # Get all teams playing (abbreviations)
    abbrev_teams = set(week_games['home_team'].unique()) | set(week_games['away_team'].unique())

    # Also add full team names for matching
    full_teams = {ABBREV_TO_TEAM.get(t, t) for t in abbrev_teams}

    return abbrev_teams | full_teams


def apply_market_filters(
    recommendations: list,
    odds_df: pd.DataFrame,
    week: int,
    season: int = 2025,
    min_minutes_to_kickoff: int = 10,
) -> list:
    """
    Apply market-specific filters to recommendations.

    V27: Filters based on:
    - max_spread: Skip games with |spread| > threshold
    - min_snap_share: Only established players
    - exclude_positions: Skip positions with no edge (e.g., TEs for receptions)
    - bye_week: Skip players on bye
    - game_started: Skip games that have already started or start within X minutes

    Args:
        recommendations: List of recommendation dicts
        odds_df: Original odds DataFrame (for position lookup)
        week: NFL week
        season: NFL season
        min_minutes_to_kickoff: Minimum minutes before game starts to allow bet (default 10)

    Returns:
        Filtered list of recommendations
    """
    if not recommendations:
        return recommendations

    # Get active teams (not on bye)
    active_teams = get_active_teams(week, season)

    # Build position lookup from odds (if available)
    position_lookup = {}
    if 'position' in odds_df.columns and 'player' in odds_df.columns:
        position_lookup = dict(zip(odds_df['player'], odds_df['position']))

    # Build spread lookup from odds (if available)
    spread_lookup = {}
    if 'vegas_spread' in odds_df.columns and 'player' in odds_df.columns:
        spread_lookup = dict(zip(odds_df['player'], odds_df['vegas_spread']))

    # Build snap_share lookup if available
    snap_lookup = {}
    if 'snap_share' in odds_df.columns and 'player' in odds_df.columns:
        snap_lookup = dict(zip(odds_df['player'], odds_df['snap_share']))

    # V27: Build game start time lookup from odds
    kickoff_lookup = {}
    if 'commence_time' in odds_df.columns and 'player' in odds_df.columns:
        for _, row in odds_df.iterrows():
            player = row.get('player', '')
            commence = row.get('commence_time', '')
            if player and commence:
                try:
                    # Parse ISO format datetime
                    if isinstance(commence, str):
                        kickoff = pd.to_datetime(commence)
                        kickoff_lookup[player] = kickoff
                except Exception:
                    pass

    filtered = []
    filtered_counts = {
        'bye_week': 0,
        'spread_filter': 0,
        'position_filter': 0,
        'snap_share_filter': 0,
        'game_started': 0,
    }

    # Current time for game start check
    now = datetime.now()
    # Handle timezone-aware datetimes
    try:
        now_utc = pd.Timestamp.now(tz='UTC')
    except Exception:
        now_utc = pd.Timestamp.now()

    for rec in recommendations:
        player = rec.get('player', '')
        team = rec.get('team', '')
        market = rec.get('market', '')

        # Skip game line recommendations (they have different filtering)
        if rec.get('bet_category') == 'GAME_LINE':
            filtered.append(rec)
            continue

        # V27: Game start time filter - skip games that have started or start soon
        if player in kickoff_lookup:
            kickoff = kickoff_lookup[player]
            try:
                # Make timezone-aware comparison
                if kickoff.tzinfo is not None:
                    minutes_to_kickoff = (kickoff - now_utc).total_seconds() / 60
                else:
                    minutes_to_kickoff = (kickoff - pd.Timestamp.now()).total_seconds() / 60

                if minutes_to_kickoff < min_minutes_to_kickoff:
                    filtered_counts['game_started'] += 1
                    continue
            except Exception:
                pass  # If we can't parse, don't filter

        # V27: Bye week filter
        if active_teams and team and team not in active_teams:
            filtered_counts['bye_week'] += 1
            continue

        # Get market filter
        market_filter = get_market_filter(market)
        if market_filter is None:
            # No filter configured - keep recommendation
            filtered.append(rec)
            continue

        # V27: Spread filter
        if market_filter.max_spread is not None:
            spread = spread_lookup.get(player, 0.0)
            if spread is not None and abs(spread) > market_filter.max_spread:
                filtered_counts['spread_filter'] += 1
                continue

        # V27: Position filter
        if market_filter.exclude_positions:
            position = position_lookup.get(player, '')
            if position in market_filter.exclude_positions:
                filtered_counts['position_filter'] += 1
                continue

        # V27: Snap share filter
        if market_filter.min_snap_share is not None:
            snap_share = snap_lookup.get(player, 1.0)  # Default to pass filter
            if snap_share is not None and snap_share < market_filter.min_snap_share:
                filtered_counts['snap_share_filter'] += 1
                continue

        # Passed all filters
        filtered.append(rec)

    # Report filtering
    total_filtered = sum(filtered_counts.values())
    if total_filtered > 0:
        print(f"\n  V27 Market Filters Applied:")
        for reason, count in filtered_counts.items():
            if count > 0:
                print(f"    {reason}: {count} removed")
        print(f"    Total: {len(recommendations)} -> {len(filtered)} ({total_filtered} filtered)")

    return filtered


def generate_recommendations(
    week: int,
    markets: list = None,
    season: int = 2025,
    include_td: bool = False,
    include_attd: bool = False,
    include_td_enhanced: bool = False,
    include_game_lines: bool = False,
) -> pd.DataFrame:
    """Generate betting recommendations using edge ensemble, TD Poisson, ATTD, and game lines."""
    print(f"Generating edge recommendations for Week {week}, {season}")

    # Load ensemble
    try:
        ensemble = EdgeEnsemble.load()
        print("Loaded edge ensemble")
    except Exception as e:
        print(f"Failed to load ensemble: {e}")
        print("Run train_ensemble.py first")
        return pd.DataFrame()

    # Pre-load direction edge (must be loaded before parallel processing)
    from nfl_quant.edges.direction_edge import get_direction_edge
    direction_edge = get_direction_edge()
    if direction_edge.loaded:
        print(f"Loaded direction models (v{direction_edge.version})")

    # Load TD Poisson edge if requested
    td_edge = None
    if include_td:
        print("Loading TD Poisson edge...")
        td_edge = load_td_poisson_edge()
        if td_edge:
            print(f"  Loaded TD edge with markets: {list(td_edge.models.keys())}")

    # Load ATTD ensemble if requested
    attd_ensemble = None
    if include_attd:
        print("Loading ATTD ensemble...")
        try:
            from nfl_quant.models.attd_ensemble import ATTDEnsemble
            attd_ensemble = ATTDEnsemble.load()
            print(f"  Loaded ATTD ensemble with positions: {list(attd_ensemble.logistic_models.keys())}")
        except Exception as e:
            print(f"  Failed to load ATTD ensemble: {e}")
            print("  Run train_attd_ensemble.py first")

    # Load TD Enhanced model if requested
    td_enhanced_model = None
    if include_td_enhanced:
        print("Loading TD Enhanced model...")
        try:
            td_enhanced_model = TDEnhancedModel.load()
            print(f"  Loaded TD Enhanced model (trained: {td_enhanced_model.trained_date})")
        except Exception as e:
            print(f"  Failed to load TD Enhanced model: {e}")
            print("  Run train_td_enhanced.py first")

    # Load data
    try:
        odds_df = load_current_odds(week, season)
        print(f"Loaded {len(odds_df)} odds")
    except FileNotFoundError as e:
        print(str(e))
        return pd.DataFrame()

    history_df = load_player_history()
    print(f"Loaded {len(history_df)} historical rows")

    # Pre-load and compute trailing stats (once for all markets)
    print("\nLoading trailing stats for LVT computation...")
    try:
        trailing_stats_df = load_player_stats_for_edge()
        # Filter to historical data only (prevent leakage)
        historical_mask = (
            (trailing_stats_df['season'] < season) |
            ((trailing_stats_df['season'] == season) & (trailing_stats_df['week'] < week))
        )
        trailing_stats_df = trailing_stats_df[historical_mask]
        trailing_stats_df = compute_edge_trailing_stats(trailing_stats_df)
        print(f"  Loaded {len(trailing_stats_df)} player-week stats for trailing calculation")
    except Exception as e:
        print(f"  Warning: Could not load trailing stats: {e}")
        print("  LVT edge will not trigger without trailing stats")
        trailing_stats_df = None

    # Determine markets
    if markets is None:
        markets = EDGE_MARKETS

    # Helper function to process a single market (for parallel execution)
    def process_single_market(market):
        """Process a single market and return recommendations."""
        recommendations = []

        # Prepare features (with trailing stats for LVT)
        market_df = prepare_features(
            odds_df, history_df, market,
            trailing_stats_df=trailing_stats_df,
            week=week, season=season
        )

        if len(market_df) == 0:
            return market, 0, recommendations, 0  # 4th return is no_data_count

        # Track NO_DATA skips
        no_data_count = 0

        # Evaluate each prop
        for _, row in market_df.iterrows():
            decision = ensemble.evaluate_bet(row, market)

            # Track NO_DATA cases (missing required data)
            if decision.source.value == 'NO_DATA':
                no_data_count += 1
                continue

            if decision.should_bet:
                # V31: Apply market-specific filters (direction, prob, line)
                line_val = row.get('line', 0)
                if not should_bet(market, decision.direction, decision.combined_confidence, line_val):
                    continue  # Skip bet that doesn't pass V31 filters

                # Construct game string from home/away teams (convert full names to abbreviations)
                home_team_raw = row.get('home_team', '')
                away_team_raw = row.get('away_team', '')
                home_team = team_name_to_abbrev(home_team_raw)
                away_team = team_name_to_abbrev(away_team_raw)
                player_team_raw = row.get('team', '')
                # Normalize player team to abbreviation for comparison
                player_team = team_name_to_abbrev(player_team_raw) if player_team_raw else ''
                # Handle LA/LAR equivalence
                if player_team == 'LA':
                    player_team = 'LAR'
                game_str = f"{away_team} @ {home_team}" if home_team and away_team else ''
                # Determine opponent based on player's team (compare normalized abbreviations)
                if player_team == home_team:
                    opponent = away_team
                elif player_team == away_team:
                    opponent = home_team
                else:
                    # Fallback if team doesn't match either (shouldn't happen with proper data)
                    opponent = home_team if player_team else ''

                # Get trailing stat for this market (used as projection in dashboard)
                trailing_col = EDGE_TRAILING_COL_MAP.get(market)
                trailing_stat = row.get(trailing_col, None) if trailing_col else None

                recommendations.append({
                    'player': row.get('player', 'Unknown'),
                    'team': player_team,
                    'market': market,
                    'line': row.get('line', 0),
                    'direction': decision.direction,
                    'pick': decision.direction,  # Copy direction to pick for dashboard compatibility
                    'units': decision.units,
                    'source': decision.source.value,
                    'lvt_confidence': decision.lvt_confidence,
                    'player_bias_confidence': decision.player_bias_confidence,
                    'combined_confidence': decision.combined_confidence,
                    'reasoning': decision.reasoning,
                    'player_under_rate': row.get('player_under_rate', 0.5),
                    'player_bet_count': row.get('player_bet_count', 0),
                    'current_season_under_rate': row.get('current_season_under_rate', 0.5),
                    'season_games_played': row.get('season_games_played', 0),
                    # Game context for dashboard grouping
                    'game': game_str,
                    'opponent': opponent,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': row.get('commence_time', ''),
                    # Trailing stat for dashboard projection (prevents UNDER picks from being filtered)
                    'trailing_stat': trailing_stat,
                })

        return market, len(market_df), recommendations, no_data_count

    # Generate recommendations per market (PARALLEL)
    all_recommendations = []
    num_workers = min(len(markets), 4)  # Use up to 4 workers

    print(f"\nProcessing {len(markets)} markets in parallel ({num_workers} workers)...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_market, m): m for m in markets}

        total_no_data = 0
        for future in as_completed(futures):
            market = futures[future]
            try:
                market_name, prop_count, recs, no_data_count = future.result()
                total_no_data += no_data_count
                if prop_count == 0:
                    print(f"  {market_name}: No data")
                else:
                    no_data_str = f" ({no_data_count} skipped - no data)" if no_data_count > 0 else ""
                    print(f"  {market_name}: {prop_count} props -> {len(recs)} recommendations{no_data_str}")
                all_recommendations.extend(recs)
            except Exception as e:
                print(f"  {market}: ERROR - {e}")

        if total_no_data > 0:
            print(f"\n  [INFO] Skipped {total_no_data} props due to insufficient data (no data = no bet)")

    # Generate TD Poisson recommendations if enabled
    if td_edge is not None:
        print("\n" + "="*60)
        print("TD POISSON EDGE RECOMMENDATIONS")
        print("="*60)
        td_recs = generate_td_recommendations(td_edge, odds_df, history_df, week, season)
        if td_recs:
            print(f"\nGenerated {len(td_recs)} TD recommendations")
            all_recommendations.extend(td_recs)
        else:
            print("\nNo TD recommendations generated")

    # Generate ATTD ensemble recommendations if enabled
    if attd_ensemble is not None:
        print("\n" + "="*60)
        print("ANYTIME TD ENSEMBLE RECOMMENDATIONS")
        print("="*60)
        attd_recs = generate_attd_recommendations(attd_ensemble, odds_df, history_df, week, season)
        if attd_recs:
            print(f"\nGenerated {len(attd_recs)} ATTD recommendations")
            all_recommendations.extend(attd_recs)
        else:
            print("\nNo ATTD recommendations generated")

    # Generate TD Enhanced recommendations if enabled
    if td_enhanced_model is not None:
        print("\n" + "="*60)
        print("TD ENHANCED MODEL RECOMMENDATIONS")
        print("(Walk-forward validated: RB @ 60% = 58.2% WR, +6.6% ROI)")
        print("="*60)
        td_enhanced_recs = generate_td_enhanced_recommendations(td_enhanced_model, odds_df, week, season)
        if td_enhanced_recs:
            print(f"\nGenerated {len(td_enhanced_recs)} TD Enhanced recommendations")
            all_recommendations.extend(td_enhanced_recs)
        else:
            print("\nNo TD Enhanced recommendations generated")

    # Generate Game Line recommendations if enabled
    if include_game_lines:
        print("\n" + "="*60)
        print("GAME LINE EDGE RECOMMENDATIONS")
        print("="*60)
        try:
            game_line_df = generate_game_line_edge_recommendations(week, season)
            if len(game_line_df) > 0:
                print(f"\nGenerated {len(game_line_df)} game line recommendations")
                # Convert DataFrame to list of dicts and add bet_category
                for _, row in game_line_df.iterrows():
                    rec = row.to_dict()
                    rec['bet_category'] = 'GAME_LINE'
                    # Add missing columns for unified format
                    rec['player'] = rec.get('game', '')
                    rec['market'] = rec.get('bet_type', 'spread')
                    rec['line'] = rec.get('market_line', 0)
                    all_recommendations.append(rec)
            else:
                print("\nNo game line recommendations generated")
        except Exception as e:
            print(f"\nGame line generation failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_recommendations:
        print("\nNo recommendations generated")
        return pd.DataFrame()

    # V27: Apply market-specific filters (spread, position, bye week)
    print("\nApplying V27 market filters...")
    all_recommendations = apply_market_filters(
        all_recommendations,
        odds_df,
        week=week,
        season=season,
    )

    if not all_recommendations:
        print("\nNo recommendations after filtering")
        return pd.DataFrame()

    # Create DataFrame
    recs_df = pd.DataFrame(all_recommendations)

    # V32: Apply injury policy (restrict-only, never boost)
    injury_mode = get_injury_mode()
    if injury_mode != InjuryPolicyMode.OFF:
        print("\nApplying injury policy...")
        try:
            injuries_df = get_injuries(season=season, refresh=False)
            print(f"  Loaded {len(injuries_df)} injury records")
        except InjuryDataError as e:
            print(f"  WARNING: Could not load injuries: {e}")
            injuries_df = None

        # Map config InjuryPolicyMode to policy InjuryMode
        policy_mode = InjuryMode.CONSERVATIVE
        if injury_mode == InjuryPolicyMode.STRICT:
            policy_mode = InjuryMode.STRICT

        before_count = len(recs_df)
        recs_df = apply_injury_policy(recs_df, injuries_df, mode=policy_mode)
        after_count = len(recs_df)

        if before_count != after_count:
            print(f"  Injury policy: {before_count} -> {after_count} ({before_count - after_count} blocked)")

    # Add market display names (consistent with dashboard format_prop_display)
    MARKET_DISPLAY_MAP = {
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Rec Yards',
        'player_rush_yds': 'Rush Yards',
        'player_rush_attempts': 'Rush Att',
        'player_pass_attempts': 'Pass Att',
        'player_pass_completions': 'Completions',
        'player_pass_yds': 'Pass Yards',
        'player_pass_tds': 'Pass TDs',
        'player_rush_tds': 'Rush TDs',
        'player_rec_tds': 'Rec TDs',
        'player_anytime_td': 'Anytime TD',
    }
    if 'market' in recs_df.columns:
        recs_df['market_display'] = recs_df['market'].apply(
            lambda m: MARKET_DISPLAY_MAP.get(m, m.replace('player_', '').replace('_', ' ').title())
        )

    # Sort by confidence
    recs_df = recs_df.sort_values('combined_confidence', ascending=False)

    # NOTE: No longer limiting picks here - dashboard handles featured vs all picks
    # All valid picks are saved; dashboard shows top 10 as "featured"

    return recs_df


def save_recommendations(recs_df: pd.DataFrame, week: int, season: int = 2025, merge: bool = True):
    """Save recommendations to file, merging with existing if present.

    Args:
        recs_df: New recommendations to save
        week: NFL week
        season: NFL season
        merge: If True, merge with existing file instead of overwriting
    """
    # Ensure reports directory exists
    REPORTS_DIR.mkdir(exist_ok=True)

    # Save CSV
    csv_path = REPORTS_DIR / f'edge_recommendations_week{week}_{season}.csv'

    # Merge with existing recommendations if file exists
    if merge and csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        print(f"\n  Merging with existing {len(existing_df)} recommendations...")

        # Create composite key for deduplication
        recs_df['_key'] = recs_df['player'].str.lower() + '|' + recs_df['market']
        existing_df['_key'] = existing_df['player'].str.lower() + '|' + existing_df['market']

        # Keep new recommendations that don't overlap with existing
        new_only = recs_df[~recs_df['_key'].isin(existing_df['_key'])]

        # Combine: existing + new non-overlapping
        combined_df = pd.concat([existing_df, new_only], ignore_index=True)
        combined_df = combined_df.drop(columns=['_key'])

        print(f"  Merged: {len(existing_df)} existing + {len(new_only)} new = {len(combined_df)} total")
        recs_df = combined_df

    recs_df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("EDGE RECOMMENDATIONS SUMMARY")
    print("="*60)

    print(f"\nTotal recommendations: {len(recs_df)}")
    print(f"Total units: {recs_df['units'].sum():.1f}")

    # By source
    print("\nBy source:")
    for source in recs_df['source'].unique():
        source_df = recs_df[recs_df['source'] == source]
        print(f"  {source}: {len(source_df)} bets, {source_df['units'].sum():.1f} units")

    # By market
    print("\nBy market:")
    for market in recs_df['market'].unique():
        market_df = recs_df[recs_df['market'] == market]
        print(f"  {market}: {len(market_df)} bets")

    # Top picks
    print("\nTop 10 picks:")
    print("-"*80)
    for _, row in recs_df.head(10).iterrows():
        print(
            f"  {row['player']:<20} {row['market']:<20} "
            f"{row['direction']:<6} {row['line']:<6} "
            f"{row['combined_confidence']:.1%} ({row['source']})"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate Edge Recommendations")
    parser.add_argument('--week', type=int, required=True, help='NFL week')
    parser.add_argument('--season', type=int, default=2025, help='NFL season')
    parser.add_argument('--market', type=str, help='Single market to process')
    parser.add_argument('--include-td', action='store_true', help='Include TD Poisson edge predictions')
    parser.add_argument('--include-attd', action='store_true', help='Include Anytime TD ensemble predictions')
    parser.add_argument('--include-td-enhanced', action='store_true', help='Include TD Enhanced model predictions (walk-forward validated)')
    parser.add_argument('--include-game-lines', action='store_true', help='Include game line edge predictions (spreads/totals)')
    args = parser.parse_args()

    print("="*60)
    print("EDGE RECOMMENDATIONS")
    print(f"Week {args.week}, {args.season}")
    if args.include_td:
        print("(Including TD Poisson edge)")
    if args.include_attd:
        print("(Including Anytime TD ensemble)")
    if args.include_td_enhanced:
        print("(Including TD Enhanced model - walk-forward validated)")
    if args.include_game_lines:
        print("(Including Game Line edge)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    markets = [args.market] if args.market else None

    recs_df = generate_recommendations(
        args.week, markets, args.season,
        include_td=args.include_td,
        include_attd=args.include_attd,
        include_td_enhanced=args.include_td_enhanced,
        include_game_lines=args.include_game_lines,
    )

    if len(recs_df) > 0:
        save_recommendations(recs_df, args.week, args.season)


if __name__ == '__main__':
    main()
