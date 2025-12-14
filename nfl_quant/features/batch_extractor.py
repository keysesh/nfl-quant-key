"""
Vectorized Batch Feature Extractor for Model Training.

This module provides high-performance feature extraction using
vectorized pandas operations instead of iterrows().

Features are defined in configs/model_config.py - update there for new versions.

Performance Target: <60 seconds for full training set (vs 10+ minutes with iterrows)

Current features include:
- Core interaction features (LVT, sweet spot, player/market rates)
- Skill features (separation, cushion, snap share, target share)
- Game context features (vegas lines, pace, pressure rates)
- Rush/receiving specific (oline health, box count, slot snap %)
- Opponent context features (V23: actual defense stats, not hardcoded!)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Import opponent features module for ACTUAL opponent data
from nfl_quant.features.opponent_features import (
    calculate_opponent_trailing_defense,
    add_opponent_features_to_data,
    add_defense_epa_feature,
    V23_OPPONENT_FEATURES,
)

# Import broken feature fixes (V25 - calculates features that were just using defaults)
try:
    from nfl_quant.features.broken_feature_fixes import (
        add_broken_features_to_dataframe,
        clear_broken_feature_caches,
    )
    BROKEN_FEATURES_FIX_AVAILABLE = True
except ImportError as e:
    BROKEN_FEATURES_FIX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Broken feature fixes not available: {e}")

# Import V24 position matchup features
try:
    from nfl_quant.features.position_role import (
        load_depth_charts,
        load_ngs_receiving,
        get_all_position_roles_vectorized,
        add_slot_detection_vectorized,
        encode_position_role,
    )
    from nfl_quant.features.defense_vs_position import (
        calculate_defense_vs_position_stats,
        pivot_defense_vs_position,
        load_defense_vs_position_cache,
    )
    from nfl_quant.features.coverage_tendencies import (
        load_participation,
        calculate_team_coverage_tendencies,
        calculate_slot_funnel_score,
        calculate_man_coverage_adjustment,
        load_coverage_cache,
    )
    V24_AVAILABLE = True
except ImportError as e:
    V24_AVAILABLE = False
    logger.warning(f"V24 position matchup features not available: {e}")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Import config for sweet spot calculation
try:
    from configs.model_config import smooth_sweet_spot, FEATURE_FLAGS
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.warning("Config not available, using legacy calculations")


def extract_features_batch(
    odds_with_trailing: pd.DataFrame,
    all_historical_odds: pd.DataFrame,
    market: str,
    target_global_week: int = None
) -> pd.DataFrame:
    """
    Vectorized feature extraction for training.

    Replaces the slow iterrows() approach with merge/groupby operations.
    Features are defined in configs/model_config.py.

    CRITICAL FOR DATA LEAKAGE PREVENTION:
    The `all_historical_odds` parameter MUST be pre-filtered to exclude any
    weeks >= the target prediction week. The caller is responsible for this:

        # CORRECT (for walk-forward validation):
        extract_features_batch(test_data, historical[historical['global_week'] < test_week], market)

        # WRONG (causes data leakage):
        extract_features_batch(test_data, all_data, market)

    Player rates (player_under_rate, player_bias) are computed from all_historical_odds,
    so including future data here will leak information about future outcomes.

    Args:
        odds_with_trailing: Odds data with trailing stats already merged
        all_historical_odds: Historical odds for player/market rates (MUST be temporally filtered!)
        market: Market name (player_receptions, player_rush_yds, etc.)
        target_global_week: If specified, only extract for this week

    Returns:
        DataFrame with all feature columns added
    """
    logger.info(f"  Vectorized extraction for {market}...")

    # Map market to stat column
    stat_col_map = {
        # Core markets (original 4)
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        # Additional markets (expanded)
        'player_rush_attempts': 'carries',
        'player_pass_completions': 'completions',
        'player_pass_attempts': 'attempts',
        'player_pass_tds': 'passing_tds',
        'player_rush_tds': 'rushing_tds',
        'player_receiving_tds': 'receiving_tds',
    }
    stat_col = stat_col_map.get(market)
    trailing_col = f'trailing_{stat_col}'

    if trailing_col not in odds_with_trailing.columns:
        logger.warning(f"Trailing column {trailing_col} not found")
        return pd.DataFrame()

    # Filter to target week if specified
    if target_global_week is not None:
        df = odds_with_trailing[
            (odds_with_trailing['global_week'] == target_global_week) &
            (odds_with_trailing['market'] == market)
        ].copy()
    else:
        df = odds_with_trailing[odds_with_trailing['market'] == market].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 0: Preserve enriched columns from input data
    # =========================================================================
    # Copy enriched columns early before they get overwritten by defaults
    if 'target_share_stats' in df.columns:
        df['target_share'] = df['target_share_stats'].copy()
        logger.debug(f"Using enriched target_share: {df['target_share'].notna().mean():.1%} coverage")

    if 'opponent' in df.columns and 'opponent_team' not in df.columns:
        df['opponent_team'] = df['opponent'].copy()
        logger.debug(f"Using enriched opponent: {df['opponent_team'].notna().mean():.1%} coverage")

    # =========================================================================
    # STEP 1: Pre-compute player betting history (vectorized)
    # =========================================================================
    hist_odds = all_historical_odds[all_historical_odds['market'] == market].copy()

    # Calculate player under rates (vectorized)
    player_rates = _compute_player_under_rates_vectorized(hist_odds)
    df = df.merge(player_rates, on='player_norm', how='left')
    df['player_under_rate'] = df['player_under_rate'].fillna(0.5)

    # Calculate player bias (vectorized)
    player_bias = _compute_player_bias_vectorized(hist_odds)
    df = df.merge(player_bias, on='player_norm', how='left')
    df['player_bias'] = df['player_bias'].fillna(0.0)

    # Calculate market under rate (vectorized by week)
    market_rates = _compute_market_under_rates_vectorized(hist_odds)
    df = df.merge(market_rates, on='global_week', how='left')
    df['market_under_rate'] = df['market_under_rate'].fillna(0.5)

    # =========================================================================
    # STEP 2: Calculate core interaction features (fully vectorized)
    # =========================================================================
    df = _calculate_core_features_vectorized(df, trailing_col, market)

    # =========================================================================
    # STEP 3: Calculate ACTUAL values for broken features (V25 FIX)
    # MUST run BEFORE defaults are set so real values take precedence.
    # Features fixed: adot, trailing_catch_rate, game_pace, pressure_rate,
    # opp_pressure_rate, slot_snap_pct, opp_wr1_receptions_allowed,
    # opp_man_coverage_rate_trailing, man_coverage_adjustment, slot_funnel_score
    # =========================================================================
    if BROKEN_FEATURES_FIX_AVAILABLE:
        try:
            df = add_broken_features_to_dataframe(df, market)
            logger.info(f"    V25 broken features calculated for {market}")
        except Exception as e:
            logger.warning(f"    V25 broken feature fix failed: {e}")

    # =========================================================================
    # STEP 3.5: Merge skill features from lookup tables (vectorized)
    # Now runs AFTER V25 fix so it only fills truly missing values
    # =========================================================================
    df = _merge_skill_features_vectorized(df, market)

    # =========================================================================
    # STEP 4: Add game context features (defaults for training)
    # Only fills truly missing values since Step 3 calculated real values
    # =========================================================================
    df = _add_game_context_defaults(df)

    # =========================================================================
    # STEP 5: Add rush/receiving features
    # =========================================================================
    df = _add_rush_receiving_features(df, market)

    # =========================================================================
    # STEP 6: Add opponent context features (V23 - ACTUAL data, not hardcoded!)
    # =========================================================================
    df = _add_opponent_features_vectorized(df, market)

    # =========================================================================
    # STEP 7: Add V24 position-specific matchup features
    # =========================================================================
    if V24_AVAILABLE:
        df = _add_position_matchup_features_vectorized(df, market)

    # =========================================================================
    # STEP 7.5: Calculate interaction terms (V25 FIX)
    # These depend on opponent features being calculated first in Step 6
    # =========================================================================
    df = _calculate_interaction_terms(df)

    # =========================================================================
    # STEP 8: Add row identifiers
    # =========================================================================
    if 'actual_stat' not in df.columns and stat_col in df.columns:
        df['actual_stat'] = df[stat_col]

    logger.info(f"    Extracted {len(df)} rows with {len(df.columns)} features")

    return df


def _compute_player_under_rates_vectorized(hist_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player under rates using groupby instead of per-row calculation.

    For each player, calculates: sum(under_hit) / count(bets)
    """
    if len(hist_odds) == 0:
        return pd.DataFrame({'player_norm': [], 'player_under_rate': []})

    player_rates = hist_odds.groupby('player_norm').agg({
        'under_hit': ['sum', 'count']
    }).reset_index()
    player_rates.columns = ['player_norm', 'under_sum', 'bet_count']
    player_rates['player_under_rate'] = player_rates['under_sum'] / player_rates['bet_count']

    # Only keep players with >= 3 bets for reliability
    player_rates.loc[player_rates['bet_count'] < 3, 'player_under_rate'] = 0.5

    return player_rates[['player_norm', 'player_under_rate']]


def _compute_player_bias_vectorized(hist_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player bias: average of (actual - line) across historical bets.

    Positive bias = player tends to exceed lines
    Negative bias = player tends to fall short
    """
    if len(hist_odds) == 0 or 'actual_stat' not in hist_odds.columns:
        return pd.DataFrame({'player_norm': [], 'player_bias': []})

    # Calculate per-bet bias
    hist_odds = hist_odds.copy()
    hist_odds['bet_bias'] = hist_odds['actual_stat'] - hist_odds['line']

    player_bias = hist_odds.groupby('player_norm').agg({
        'bet_bias': ['mean', 'count']
    }).reset_index()
    player_bias.columns = ['player_norm', 'player_bias', 'bet_count']

    # Only reliable with >= 3 bets
    player_bias.loc[player_bias['bet_count'] < 3, 'player_bias'] = 0.0

    return player_bias[['player_norm', 'player_bias']]


def _compute_market_under_rates_vectorized(hist_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-wide under rate per week.

    This captures overall market regime (are unders hitting more often?)
    """
    if len(hist_odds) == 0:
        return pd.DataFrame({'global_week': [], 'market_under_rate': []})

    # For each week, compute rate using all prior weeks
    weeks = sorted(hist_odds['global_week'].unique())

    market_rates = []
    for week in weeks:
        prior = hist_odds[hist_odds['global_week'] < week]
        if len(prior) >= 10:  # Need enough history
            rate = prior['under_hit'].mean()
        else:
            rate = 0.5
        # Assign this rate to the NEXT week (no leakage)
        market_rates.append({'global_week': week + 1, 'market_under_rate': rate})

    return pd.DataFrame(market_rates)


def _calculate_core_features_vectorized(
    df: pd.DataFrame,
    trailing_col: str,
    market: str
) -> pd.DataFrame:
    """
    Calculate all core interaction features using vectorized operations.
    """
    df = df.copy()

    # Line vs Trailing (LVT) - percentage method
    # Avoid division by zero
    trailing_safe = df[trailing_col].replace(0, np.nan).fillna(0.001)
    df['line_vs_trailing'] = ((df['line'] - df[trailing_col]) / trailing_safe) * 100

    # Clip extreme values
    df['line_vs_trailing'] = df['line_vs_trailing'].clip(-100, 100)

    # Line level (raw line value)
    df['line_level'] = df['line']

    # Sweet spot (Gaussian decay or binary)
    if CONFIG_AVAILABLE and FEATURE_FLAGS.use_smooth_sweet_spot:
        df['line_in_sweet_spot'] = df['line'].apply(
            lambda x: smooth_sweet_spot(x, market=market)
        )
    else:
        # Legacy binary sweet spot
        df['line_in_sweet_spot'] = ((df['line'] >= 3.5) & (df['line'] <= 7.5)).astype(float)

    # Interaction features (all vectorized)
    df['LVT_x_player_tendency'] = df['line_vs_trailing'] * (df['player_under_rate'] - 0.5)
    df['LVT_x_player_bias'] = df['line_vs_trailing'] * df['player_bias']
    df['LVT_x_regime'] = df['line_vs_trailing'] * (df['market_under_rate'] - 0.5)
    df['LVT_in_sweet_spot'] = df['line_vs_trailing'] * df['line_in_sweet_spot']
    df['market_bias_strength'] = (df['market_under_rate'] - 0.5).abs() * 2
    df['player_market_aligned'] = (df['player_under_rate'] - 0.5) * (df['market_under_rate'] - 0.5)

    # LVT interaction features
    if CONFIG_AVAILABLE and FEATURE_FLAGS.use_lvt_x_defense:
        if 'opp_position_def_epa' in df.columns:
            df['lvt_x_defense'] = df['line_vs_trailing'] * df['opp_position_def_epa']
        else:
            df['lvt_x_defense'] = 0.0

    if CONFIG_AVAILABLE and FEATURE_FLAGS.use_lvt_x_rest:
        if 'days_rest' in df.columns:
            rest_normalized = (df['days_rest'] - 7) / 7.0
            df['lvt_x_rest'] = df['line_vs_trailing'] * rest_normalized
        else:
            df['lvt_x_rest'] = 0.0

    return df


def _merge_skill_features_vectorized(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Merge skill features from pre-loaded lookup tables.
    Uses merge operations instead of per-row function calls.
    """
    df = df.copy()

    # Load lookup tables once
    snap_counts = _load_snap_counts_lookup()
    ngs_receiving = _load_ngs_receiving_lookup()

    # =========================================================================
    # Snap share - merge by player name, season, week
    # =========================================================================
    if len(snap_counts) > 0 and 'player' in df.columns:
        # Normalize player names for matching
        df['player_lower'] = df['player'].str.lower().str.strip() if 'player' in df.columns else df['player_norm']
        snap_counts['player_lower'] = snap_counts['player'].str.lower().str.strip()

        # Get trailing snap share (week - 1)
        snap_lookup = snap_counts[['player_lower', 'season', 'week', 'offense_pct']].copy()
        snap_lookup['lookup_week'] = snap_lookup['week'] + 1  # Shift for no-leakage
        snap_lookup = snap_lookup.rename(columns={'offense_pct': 'snap_share'})

        df = df.merge(
            snap_lookup[['player_lower', 'season', 'lookup_week', 'snap_share']],
            left_on=['player_lower', 'season', 'week'],
            right_on=['player_lower', 'season', 'lookup_week'],
            how='left'
        )
        df['snap_share'] = df['snap_share'].fillna(0.0)
        df = df.drop(columns=['lookup_week', 'player_lower'], errors='ignore')
    else:
        df['snap_share'] = 0.0

    # =========================================================================
    # NGS receiving features - merge by player_id, season, week
    # =========================================================================
    if market in ['player_receptions', 'player_reception_yds'] and len(ngs_receiving) > 0:
        if 'player_id' in df.columns:
            # Get trailing NGS features
            ngs_lookup = ngs_receiving.groupby(['player_gsis_id', 'season', 'week']).agg({
                'avg_separation': 'mean',
                'avg_cushion': 'mean'
            }).reset_index()
            ngs_lookup['lookup_week'] = ngs_lookup['week'] + 1

            df = df.merge(
                ngs_lookup.rename(columns={'player_gsis_id': 'player_id'}),
                left_on=['player_id', 'season', 'week'],
                right_on=['player_id', 'season', 'lookup_week'],
                how='left',
                suffixes=('', '_ngs')
            )
            df['avg_separation'] = df['avg_separation'].fillna(2.8)  # League avg
            df['avg_cushion'] = df['avg_cushion'].fillna(6.2)
            df = df.drop(columns=['lookup_week', 'week_ngs'], errors='ignore')
        else:
            df['avg_separation'] = 2.8
            df['avg_cushion'] = 6.2
    else:
        df['avg_separation'] = 2.8
        df['avg_cushion'] = 6.2

    # Fill remaining skill features with defaults - only fill missing values
    if 'trailing_catch_rate' not in df.columns:
        df['trailing_catch_rate'] = 0.68  # League avg
    else:
        df['trailing_catch_rate'] = df['trailing_catch_rate'].fillna(0.68)

    if 'target_share' not in df.columns:
        df['target_share'] = 0.0
    elif 'target_share_stats' in df.columns:
        # Use enriched target_share if available
        df['target_share'] = df['target_share_stats'].fillna(df['target_share']).fillna(0.0)
    else:
        df['target_share'] = df['target_share'].fillna(0.0)

    if 'opp_wr1_receptions_allowed' not in df.columns:
        df['opp_wr1_receptions_allowed'] = 5.5  # League avg
    else:
        df['opp_wr1_receptions_allowed'] = df['opp_wr1_receptions_allowed'].fillna(5.5)

    return df


def _add_game_context_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game context feature columns with default values.
    Only fills missing values - preserves existing data from enriched sources.
    """
    df = df.copy()

    # Game Context features - only fill if missing or NaN
    if 'game_pace' not in df.columns:
        df['game_pace'] = 64.0  # League avg plays per game
    else:
        df['game_pace'] = df['game_pace'].fillna(64.0)

    if 'vegas_total' not in df.columns:
        df['vegas_total'] = 44.0  # League avg O/U
    else:
        df['vegas_total'] = df['vegas_total'].fillna(44.0)

    if 'vegas_spread' not in df.columns:
        df['vegas_spread'] = 0.0  # Neutral
    else:
        df['vegas_spread'] = df['vegas_spread'].fillna(0.0)

    if 'implied_team_total' not in df.columns:
        df['implied_team_total'] = 22.0  # League avg
    else:
        df['implied_team_total'] = df['implied_team_total'].fillna(22.0)

    # Skill features - only fill if missing or NaN
    if 'adot' not in df.columns:
        df['adot'] = 8.5  # League avg depth of target
    else:
        df['adot'] = df['adot'].fillna(8.5)

    if 'pressure_rate' not in df.columns:
        df['pressure_rate'] = 0.25  # League avg
    else:
        df['pressure_rate'] = df['pressure_rate'].fillna(0.25)

    if 'opp_pressure_rate' not in df.columns:
        df['opp_pressure_rate'] = 0.25
    else:
        df['opp_pressure_rate'] = df['opp_pressure_rate'].fillna(0.25)

    return df


def _calculate_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate interaction terms AFTER opponent features are available.

    V25 FIX: These were always 0 because they ran before opponent features existed.

    Features:
    - lvt_x_defense: line_vs_trailing * opponent_def_epa
    - lvt_x_rest: line_vs_trailing * normalized_rest_days
    """
    df = df.copy()

    # Check for line_vs_trailing
    if 'line_vs_trailing' not in df.columns:
        return df

    lvt = df['line_vs_trailing']

    # lvt_x_defense: Find the best available opponent defense EPA column
    def_epa_col = None
    for col in ['opp_def_epa', 'opp_position_def_epa', 'opponent_def_epa',
                'opp_pass_def_vs_avg', 'opp_rush_def_vs_avg']:
        if col in df.columns and df[col].notna().any():
            def_epa_col = col
            break

    if def_epa_col is not None:
        # Scale defense EPA to reasonable range for interaction
        df['lvt_x_defense'] = lvt * df[def_epa_col].fillna(0)
        logger.debug(f"lvt_x_defense calculated from {def_epa_col}, coverage: {df['lvt_x_defense'].notna().mean():.1%}")
    else:
        # Check if has_opponent_context exists - use it as proxy
        if 'has_opponent_context' in df.columns:
            df['lvt_x_defense'] = lvt * (df['has_opponent_context'] - 0.5) * 0.1
            logger.debug("lvt_x_defense using has_opponent_context as proxy")

    # lvt_x_rest: normalized rest days
    rest_col = None
    for col in ['rest_days', 'days_rest']:
        if col in df.columns and df[col].notna().any():
            rest_col = col
            break

    if rest_col is not None:
        # Normalize around 7 days (standard week)
        rest_normalized = (df[rest_col] - 7) / 7.0
        df['lvt_x_rest'] = lvt * rest_normalized
        logger.debug(f"lvt_x_rest calculated from {rest_col}")
    else:
        # Default: assume standard rest (7 days = 0 effect)
        df['lvt_x_rest'] = 0.0

    return df


def _add_rush_receiving_features(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Add rush/receiving specific features.

    Features:
    - oline_health_score: O-line injury weighted availability
    - box_count_expected: Expected box count from spread
    - slot_snap_pct: Proxy from aDOT (lower aDOT = more slot)
    - target_share_trailing: 4-week rolling target share
    """
    df = df.copy()

    # =========================================================================
    # 1. O-LINE HEALTH SCORE (for rushing markets)
    # =========================================================================
    # Compute from injuries data - weighted by position importance
    # LT=1.0, LG=0.8, C=0.7, RG=0.8, RT=1.0
    if market == 'player_rush_yds':
        df = _compute_oline_health_vectorized(df)
    else:
        df['oline_health_score'] = 1.0  # Default for non-rush markets

    # =========================================================================
    # 2. BOX COUNT EXPECTED (for rushing markets)
    # =========================================================================
    # Derived from vegas_spread: negative spread = opponent loading box
    # Formula: 7 + 0.15 * spread (clamped 6-9)
    # e.g., -7 spread (big favorite) → 7 + 0.15*(-7) = 5.95 → 6 (light boxes)
    # e.g., +7 spread (big underdog) → 7 + 0.15*(7) = 8.05 → 8 (stacked boxes)
    if 'vegas_spread' in df.columns:
        df['box_count_expected'] = (7.0 + 0.15 * df['vegas_spread']).clip(6.0, 9.0)
    else:
        df['box_count_expected'] = 7.0  # Neutral default

    # =========================================================================
    # 3. SLOT SNAP PCT (for receiving markets)
    # =========================================================================
    # Proxy from aDOT: lower aDOT = more slot routes
    # Slot receivers typically run shorter routes (aDOT < 7)
    # Formula: 1 - (aDOT / 15).clip(0, 1) → high aDOT = low slot %
    if market in ['player_receptions', 'player_reception_yds']:
        if 'adot' in df.columns:
            # Lower aDOT → higher slot %, higher aDOT → lower slot %
            df['slot_snap_pct'] = (1.0 - (df['adot'] / 15.0)).clip(0.0, 1.0)
        else:
            df['slot_snap_pct'] = 0.3  # Default ~30% slot
    else:
        df['slot_snap_pct'] = 0.3

    # =========================================================================
    # 4. TARGET SHARE TRAILING (for receiving markets)
    # =========================================================================
    # Use existing target_share if available, otherwise compute from stats
    if market in ['player_receptions', 'player_reception_yds']:
        df = _compute_target_share_trailing_vectorized(df)
    else:
        df['target_share_trailing'] = 0.15  # Default

    return df


def _compute_oline_health_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute O-line health score from injuries data.

    Weights: LT=1.0, LG=0.8, C=0.7, RG=0.8, RT=1.0 (total 4.3)
    Score = 1.0 when all healthy, lower when injured
    """
    df = df.copy()

    # Load injuries data
    injuries = _load_injuries_lookup()

    if len(injuries) == 0 or 'team' not in df.columns:
        df['oline_health_score'] = 1.0
        return df

    # Filter to O-line positions
    oline_positions = ['T', 'G', 'C']
    oline_injuries = injuries[injuries['position'].isin(oline_positions)].copy()

    # Only count "Out" or "Doubtful" as missing
    status_weights = {
        'Out': 0.0,
        'Doubtful': 0.25,
        'Questionable': 0.75,
        'Probable': 1.0,
    }

    # Position importance weights (simplified: T and G weighted equally)
    position_weights = {'T': 1.0, 'G': 0.8, 'C': 0.7}

    # Calculate health score per team/week
    if 'season' in df.columns and 'week' in df.columns:
        # Group by team, season, week
        health_scores = []

        for _, group in df.groupby(['team', 'season', 'week']):
            team = group['team'].iloc[0]
            season = group['season'].iloc[0]
            week = group['week'].iloc[0]

            # Get injuries for this team/week
            team_injuries = oline_injuries[
                (oline_injuries['team'] == team) &
                (oline_injuries['season'] == season) &
                (oline_injuries['week'] == week)
            ]

            # Calculate weighted health
            total_weight = 5 * 0.86  # Assuming 5 OL (2T, 2G, 1C) with avg weight
            healthy_weight = total_weight

            for _, injury in team_injuries.iterrows():
                pos = injury.get('position', 'T')
                status = injury.get('report_status', 'Probable')
                pos_weight = position_weights.get(pos, 0.8)
                status_factor = status_weights.get(status, 1.0)
                # Reduce healthy weight by injured amount
                healthy_weight -= pos_weight * (1 - status_factor)

            score = max(0.5, healthy_weight / total_weight)  # Floor at 0.5
            health_scores.append({
                'team': team, 'season': season, 'week': week,
                'oline_health_score': score
            })

        if health_scores:
            health_df = pd.DataFrame(health_scores)
            df = df.merge(health_df, on=['team', 'season', 'week'], how='left')
            df['oline_health_score'] = df['oline_health_score'].fillna(1.0)
        else:
            df['oline_health_score'] = 1.0
    else:
        df['oline_health_score'] = 1.0

    return df


def _compute_target_share_trailing_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4-week trailing target share from weekly stats.
    """
    df = df.copy()

    # Load weekly stats for target share
    weekly_stats = _load_weekly_stats_lookup()

    if len(weekly_stats) == 0:
        df['target_share_trailing'] = 0.15
        return df

    # Check if target_share column exists
    if 'target_share' not in weekly_stats.columns:
        df['target_share_trailing'] = 0.15
        return df

    # Normalize player names for matching
    weekly_stats = weekly_stats.copy()
    weekly_stats['player_norm'] = weekly_stats['player_display_name'].str.lower().str.strip()

    # Calculate trailing target share (4-week EWMA)
    weekly_stats = weekly_stats.sort_values(['player_norm', 'season', 'week'])
    weekly_stats['target_share_trailing'] = weekly_stats.groupby('player_norm')['target_share'].transform(
        lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
    )

    # Create lookup with week+1 for no leakage
    ts_lookup = weekly_stats[['player_norm', 'season', 'week', 'target_share_trailing']].copy()
    ts_lookup['lookup_week'] = ts_lookup['week'] + 1

    # Merge
    if 'player_norm' in df.columns:
        df = df.merge(
            ts_lookup[['player_norm', 'season', 'lookup_week', 'target_share_trailing']],
            left_on=['player_norm', 'season', 'week'],
            right_on=['player_norm', 'season', 'lookup_week'],
            how='left',
            suffixes=('', '_ts')
        )
        df['target_share_trailing'] = df['target_share_trailing'].fillna(0.15)
        df = df.drop(columns=['lookup_week', 'player_norm_ts'], errors='ignore')
    else:
        df['target_share_trailing'] = 0.15

    return df


# =========================================================================
# LOOKUP TABLE LOADERS (cached)
# =========================================================================

_SNAP_COUNTS_CACHE = None
_NGS_RECEIVING_CACHE = None
_INJURIES_CACHE = None
_WEEKLY_STATS_CACHE = None

# V24 Position Matchup Caches
_V24_POSITION_ROLES_CACHE = None
_V24_DEFENSE_VS_POSITION_CACHE = None
_V24_COVERAGE_TENDENCIES_CACHE = None
_V24_SLOT_FUNNEL_CACHE = None


def _load_snap_counts_lookup() -> pd.DataFrame:
    """Load snap counts with caching."""
    global _SNAP_COUNTS_CACHE
    if _SNAP_COUNTS_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'snap_counts.parquet'
        if path.exists():
            _SNAP_COUNTS_CACHE = pd.read_parquet(path)
        else:
            _SNAP_COUNTS_CACHE = pd.DataFrame()
    return _SNAP_COUNTS_CACHE


def _load_ngs_receiving_lookup() -> pd.DataFrame:
    """Load NGS receiving with caching."""
    global _NGS_RECEIVING_CACHE
    if _NGS_RECEIVING_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_receiving.parquet'
        if path.exists():
            _NGS_RECEIVING_CACHE = pd.read_parquet(path)
        else:
            _NGS_RECEIVING_CACHE = pd.DataFrame()
    return _NGS_RECEIVING_CACHE


def _load_injuries_lookup() -> pd.DataFrame:
    """Load injuries data with caching."""
    global _INJURIES_CACHE
    if _INJURIES_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'injuries.parquet'
        if path.exists():
            _INJURIES_CACHE = pd.read_parquet(path)
        else:
            _INJURIES_CACHE = pd.DataFrame()
    return _INJURIES_CACHE


def _load_weekly_stats_lookup() -> pd.DataFrame:
    """Load weekly stats with caching."""
    global _WEEKLY_STATS_CACHE
    if _WEEKLY_STATS_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if path.exists():
            _WEEKLY_STATS_CACHE = pd.read_parquet(path)
        else:
            _WEEKLY_STATS_CACHE = pd.DataFrame()
    return _WEEKLY_STATS_CACHE


def clear_caches():
    """Clear all cached lookup tables."""
    global _SNAP_COUNTS_CACHE, _NGS_RECEIVING_CACHE, _INJURIES_CACHE, _WEEKLY_STATS_CACHE, _OPP_DEFENSE_CACHE
    global _V24_POSITION_ROLES_CACHE, _V24_DEFENSE_VS_POSITION_CACHE, _V24_COVERAGE_TENDENCIES_CACHE, _V24_SLOT_FUNNEL_CACHE
    _SNAP_COUNTS_CACHE = None
    _NGS_RECEIVING_CACHE = None
    _INJURIES_CACHE = None
    _WEEKLY_STATS_CACHE = None
    _OPP_DEFENSE_CACHE = None
    # V24 caches
    _V24_POSITION_ROLES_CACHE = None
    _V24_DEFENSE_VS_POSITION_CACHE = None
    _V24_COVERAGE_TENDENCIES_CACHE = None
    _V24_SLOT_FUNNEL_CACHE = None
    # V25 broken feature fix caches
    if BROKEN_FEATURES_FIX_AVAILABLE:
        try:
            clear_broken_feature_caches()
        except Exception:
            pass


# =========================================================================
# OPPONENT FEATURES (V23 - Actual data, not hardcoded!)
# =========================================================================

_OPP_DEFENSE_CACHE = None


def _add_opponent_features_vectorized(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Add opponent context features using ACTUAL opponent defense data.

    IMPORTANT: Does NOT fill missing values with defaults.
    - XGBoost handles NaN natively and learns optimal split directions
    - Adds 'has_opponent_context' flag so model knows when data is available
    - This is more honest than pretending missing data is "league average"

    Features added:
    - opp_pass_yds_allowed_trailing: Opponent's trailing pass yards allowed
    - opp_rush_yds_allowed_trailing: Opponent's trailing rush yards allowed
    - opp_receptions_allowed_trailing: Opponent's trailing receptions allowed
    - opp_pass_def_vs_avg: Opponent pass defense vs league average (z-score)
    - opp_rush_def_vs_avg: Opponent rush defense vs league average (z-score)
    - opp_def_epa: Opponent defense EPA
    - has_opponent_context: 1 if opponent data available, 0 if missing

    Args:
        df: DataFrame with 'opponent', 'week', 'season' columns
        market: Market type for logging

    Returns:
        DataFrame with opponent features added (NaN where data unavailable)
    """
    global _OPP_DEFENSE_CACHE
    df = df.copy()

    # NOTE: Do NOT pre-initialize opponent feature columns here!
    # Pre-initialization causes merge collisions (_x, _y suffixes) when actual data is merged.
    # Columns will be added by the merge, and missing ones are added at the end.

    # Check if we have the required columns
    opponent_col = None
    for col in ['opponent', 'opponent_team', 'opp']:
        if col in df.columns:
            opponent_col = col
            break

    if opponent_col is None:
        logger.warning("No opponent column found - opponent features will be NaN")
        # Add missing columns as NaN at the end
        for feat in V23_OPPONENT_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan
        df['has_opponent_context'] = 0
        return df

    # Load weekly stats and calculate opponent defense
    weekly_stats = _load_weekly_stats_lookup()

    if len(weekly_stats) == 0:
        logger.warning("Weekly stats empty - opponent features will be NaN")
        for feat in V23_OPPONENT_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan
        df['has_opponent_context'] = 0
        return df

    # Calculate opponent trailing defense if not cached
    if _OPP_DEFENSE_CACHE is None:
        try:
            _OPP_DEFENSE_CACHE = calculate_opponent_trailing_defense(weekly_stats)
            logger.info(f"Calculated opponent defense for {_OPP_DEFENSE_CACHE['team'].nunique()} teams")
        except Exception as e:
            logger.warning(f"Failed to calculate opponent defense: {e}")
            for feat in V23_OPPONENT_FEATURES:
                if feat not in df.columns:
                    df[feat] = np.nan
            df['has_opponent_context'] = 0
            return df

    if len(_OPP_DEFENSE_CACHE) == 0:
        for feat in V23_OPPONENT_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan
        df['has_opponent_context'] = 0
        return df

    # Select columns to join
    feature_cols = [c for c in _OPP_DEFENSE_CACHE.columns if 'trailing' in c or 'vs_avg' in c]
    join_cols = ['team', 'week', 'season']

    opp_features = _OPP_DEFENSE_CACHE[join_cols + feature_cols].copy()

    # Rename 'team' to match opponent column for joining
    opp_features = opp_features.rename(columns={'team': opponent_col})

    # Prefix with 'opp_' if not already
    rename_map = {}
    for col in feature_cols:
        if not col.startswith('opp_'):
            rename_map[col] = f'opp_{col}'
    opp_features = opp_features.rename(columns=rename_map)

    # Drop any existing opponent columns that would cause merge collision
    # (these might have been added by prior processing with stale/incorrect values)
    cols_to_merge = [c for c in opp_features.columns if c not in [opponent_col, 'week', 'season']]
    for col in cols_to_merge:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Merge opponent features
    merge_cols = [opponent_col]
    if 'week' in df.columns and 'week' in opp_features.columns:
        merge_cols.append('week')
    if 'season' in df.columns and 'season' in opp_features.columns:
        merge_cols.append('season')

    df = df.merge(opp_features, on=merge_cols, how='left')

    # Add defense EPA from pre-computed file
    df = add_defense_epa_feature(df, opponent_col=opponent_col)

    # Add any missing V23 features as NaN (for consistency)
    for feat in V23_OPPONENT_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan

    # Add flag indicating whether opponent context is available
    # Check if ANY of the key opponent features have data
    key_features = ['opp_pass_yds_allowed_trailing', 'opp_rush_yds_allowed_trailing']
    available_key_features = [f for f in key_features if f in df.columns]

    if available_key_features:
        df['has_opponent_context'] = df[available_key_features].notna().any(axis=1).astype(int)
    else:
        df['has_opponent_context'] = 0

    # Log coverage statistics
    coverage = df['has_opponent_context'].mean()
    logger.info(f"Opponent context coverage: {coverage:.1%} of rows have data")

    # DO NOT fill NaN values - let XGBoost handle missing data natively
    # This is more honest than pretending missing data is "league average"

    opp_col_count = len([c for c in df.columns if c.startswith('opp_')])
    logger.info(f"Added {opp_col_count} opponent features + has_opponent_context flag")

    return df


def _add_opponent_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: No longer fills defaults.

    Instead, adds opponent feature columns as NaN and sets has_opponent_context=0.
    XGBoost handles NaN natively - no need to fill with misleading "league average" values.
    """
    df = df.copy()

    # Add columns as NaN (not defaults!)
    for feat in V23_OPPONENT_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan

    # Flag that we don't have opponent context
    df['has_opponent_context'] = 0

    logger.warning("No opponent data available - features will be NaN, has_opponent_context=0")

    return df


# =========================================================================
# V24 POSITION MATCHUP FEATURES
# =========================================================================

def _load_v24_position_caches():
    """Load V24 position matchup caches."""
    global _V24_POSITION_ROLES_CACHE, _V24_DEFENSE_VS_POSITION_CACHE
    global _V24_COVERAGE_TENDENCIES_CACHE, _V24_SLOT_FUNNEL_CACHE

    if not V24_AVAILABLE:
        return None, None, None, None

    cache_dir = PROJECT_ROOT / 'data' / 'cache'

    # Load position roles cache
    if _V24_POSITION_ROLES_CACHE is None:
        roles_path = cache_dir / 'position_roles.parquet'
        if roles_path.exists():
            _V24_POSITION_ROLES_CACHE = pd.read_parquet(roles_path)
            logger.info(f"Loaded position roles cache: {len(_V24_POSITION_ROLES_CACHE)} records")

    # Load defense-vs-position cache
    if _V24_DEFENSE_VS_POSITION_CACHE is None:
        dvp_path = cache_dir / 'defense_vs_position.parquet'
        if dvp_path.exists():
            _V24_DEFENSE_VS_POSITION_CACHE = pd.read_parquet(dvp_path)
            logger.info(f"Loaded defense-vs-position cache: {len(_V24_DEFENSE_VS_POSITION_CACHE)} records")

    # Load coverage tendencies cache
    if _V24_COVERAGE_TENDENCIES_CACHE is None:
        coverage_path = cache_dir / 'coverage_tendencies.parquet'
        if coverage_path.exists():
            _V24_COVERAGE_TENDENCIES_CACHE = pd.read_parquet(coverage_path)
            logger.info(f"Loaded coverage tendencies cache: {len(_V24_COVERAGE_TENDENCIES_CACHE)} records")

    # Load slot funnel cache
    if _V24_SLOT_FUNNEL_CACHE is None:
        funnel_path = cache_dir / 'slot_funnel.parquet'
        if funnel_path.exists():
            _V24_SLOT_FUNNEL_CACHE = pd.read_parquet(funnel_path)
            logger.info(f"Loaded slot funnel cache: {len(_V24_SLOT_FUNNEL_CACHE)} records")

    return (
        _V24_POSITION_ROLES_CACHE,
        _V24_DEFENSE_VS_POSITION_CACHE,
        _V24_COVERAGE_TENDENCIES_CACHE,
        _V24_SLOT_FUNNEL_CACHE
    )


def _add_position_matchup_features_vectorized(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Add V24 position-specific matchup features.

    Features added:
    - pos_rank: Player's depth chart position rank (1, 2, 3)
    - is_starter: Binary flag for pos_rank == 1
    - is_slot_receiver: Binary flag for slot alignment
    - opp_position_yards_allowed_trailing: What defense allows to this position role
    - opp_position_volume_allowed_trailing: Receptions/carries allowed to role
    - opp_man_coverage_rate_trailing: Defense's man coverage rate
    - slot_funnel_score: Defense's slot vulnerability
    - man_coverage_adjustment: Adjustment multiplier based on coverage type
    - position_role_x_opp_yards: Interaction feature
    - has_position_context: Flag indicating data availability

    Args:
        df: DataFrame with player_id, opponent, week, season columns
        market: Market type for logging

    Returns:
        DataFrame with V24 position matchup features added
    """
    if not V24_AVAILABLE:
        logger.debug("V24 features not available, skipping position matchup features")
        return df

    df = df.copy()

    # Initialize V24 feature columns
    V24_FEATURES = [
        'pos_rank',
        'is_starter',
        'is_slot_receiver',
        'slot_alignment_pct',
        'opp_position_yards_allowed_trailing',
        'opp_position_volume_allowed_trailing',
        'opp_man_coverage_rate_trailing',
        'slot_funnel_score',
        'man_coverage_adjustment',
        'position_role_x_opp_yards',
        'has_position_context'
    ]

    for col in V24_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Load caches
    position_roles, defense_vs_position, coverage_tendencies, slot_funnel = _load_v24_position_caches()

    if position_roles is None or len(position_roles) == 0:
        logger.debug("Position roles cache not available")
        df['has_position_context'] = 0
        return df

    # Determine opponent column
    opponent_col = None
    for col in ['opponent', 'opponent_team', 'opp']:
        if col in df.columns:
            opponent_col = col
            break

    if opponent_col is None:
        logger.debug("No opponent column found for V24 features")
        df['has_position_context'] = 0
        return df

    # Check for player_id column
    if 'player_id' not in df.columns:
        logger.debug("No player_id column found for V24 features")
        df['has_position_context'] = 0
        return df

    # Determine team column name in df
    team_col = None
    for col in ['team', 'recent_team', 'player_team']:
        if col in df.columns:
            team_col = col
            break

    # Merge position roles
    role_cols = ['player_id', 'team', 'week', 'season', 'pos_rank', 'is_starter']

    # Add slot columns if available
    if 'is_slot_receiver' in position_roles.columns:
        role_cols.append('is_slot_receiver')
    if 'slot_alignment_pct' in position_roles.columns:
        role_cols.append('slot_alignment_pct')

    # Filter role_cols to those that exist in position_roles
    role_cols = [c for c in role_cols if c in position_roles.columns]

    if len(role_cols) >= 4 and team_col is not None:
        # Rename df team column to match position_roles 'team' column for merge
        df_temp = df.copy()
        if team_col != 'team':
            df_temp['team'] = df_temp[team_col]

        df_temp = df_temp.merge(
            position_roles[role_cols],
            on=['player_id', 'team', 'week', 'season'],
            how='left',
            suffixes=('', '_v24')
        )

        # Copy merged columns back to df
        for col in ['pos_rank', 'is_starter', 'is_slot_receiver', 'slot_alignment_pct']:
            if col in df_temp.columns:
                df[col] = df_temp[col]
            v24_col = f'{col}_v24'
            if v24_col in df_temp.columns:
                df[col] = df[col].fillna(df_temp[v24_col])
    elif len(role_cols) >= 4:
        # No team column, try merging just on player_id, week, season
        role_cols_no_team = [c for c in role_cols if c != 'team']
        if 'player_id' in role_cols_no_team and 'week' in role_cols_no_team and 'season' in role_cols_no_team:
            df = df.merge(
                position_roles[role_cols_no_team].drop_duplicates(subset=['player_id', 'week', 'season']),
                on=['player_id', 'week', 'season'],
                how='left',
                suffixes=('', '_v24')
            )

        # Consolidate columns (prefer V24 versions)
        for col in ['pos_rank', 'is_starter', 'is_slot_receiver', 'slot_alignment_pct']:
            v24_col = f'{col}_v24'
            if v24_col in df.columns:
                df[col] = df[col].fillna(df[v24_col])
                df = df.drop(columns=[v24_col], errors='ignore')

    # Fill defaults for missing position data
    df['pos_rank'] = df['pos_rank'].fillna(2)  # Default to non-starter
    df['is_starter'] = df['is_starter'].fillna(False).astype(int)
    df['is_slot_receiver'] = df['is_slot_receiver'].fillna(False).astype(int)
    df['slot_alignment_pct'] = df['slot_alignment_pct'].fillna(0.0)

    # Merge defense-vs-position stats
    if defense_vs_position is not None and len(defense_vs_position) > 0:
        # Determine position from market
        if market in ['player_receptions', 'player_reception_yds']:
            pos_type = 'wr'
        elif market in ['player_rush_yds', 'player_rush_attempts']:
            pos_type = 'rb'
        elif market in ['player_pass_yds', 'player_pass_completions', 'player_pass_attempts']:
            pos_type = 'qb'  # QBs don't have position roles in same way
        else:
            pos_type = 'wr'  # Default to WR stats

        # Build column names based on position and rank
        yards_cols = []
        volume_cols = []

        for rank in [1, 2, 3]:
            yards_col = f'opp_{pos_type}{rank}_receiving_yards_allowed_trailing'
            if pos_type == 'rb':
                yards_col = f'opp_rb{rank}_rushing_yards_allowed_trailing'

            if yards_col in defense_vs_position.columns:
                yards_cols.append(yards_col)

            volume_col = f'opp_{pos_type}{rank}_receptions_allowed_trailing'
            if pos_type == 'rb':
                volume_col = f'opp_rb{rank}_carries_allowed_trailing'

            if volume_col in defense_vs_position.columns:
                volume_cols.append(volume_col)

        # Merge defense stats by opponent
        dvp_merge = defense_vs_position.rename(columns={'defense_team': opponent_col})
        merge_on = [c for c in [opponent_col, 'week', 'season'] if c in dvp_merge.columns and c in df.columns]

        if len(merge_on) >= 1:
            dvp_cols = [opponent_col, 'week', 'season'] + yards_cols + volume_cols
            dvp_cols = [c for c in dvp_cols if c in dvp_merge.columns]

            df = df.merge(dvp_merge[dvp_cols], on=merge_on, how='left', suffixes=('', '_dvp'))

        # Set position-specific yards/volume based on player's pos_rank
        if len(yards_cols) > 0:
            def get_pos_yards(row):
                rank = int(row['pos_rank']) if pd.notna(row['pos_rank']) else 2
                rank = min(rank, 3)  # Cap at 3
                col = f'opp_{pos_type}{rank}_receiving_yards_allowed_trailing'
                if pos_type == 'rb':
                    col = f'opp_rb{rank}_rushing_yards_allowed_trailing'
                return row.get(col, np.nan)

            df['opp_position_yards_allowed_trailing'] = df.apply(get_pos_yards, axis=1)

        if len(volume_cols) > 0:
            def get_pos_volume(row):
                rank = int(row['pos_rank']) if pd.notna(row['pos_rank']) else 2
                rank = min(rank, 3)
                col = f'opp_{pos_type}{rank}_receptions_allowed_trailing'
                if pos_type == 'rb':
                    col = f'opp_rb{rank}_carries_allowed_trailing'
                return row.get(col, np.nan)

            df['opp_position_volume_allowed_trailing'] = df.apply(get_pos_volume, axis=1)

    # Merge coverage tendencies
    if coverage_tendencies is not None and len(coverage_tendencies) > 0:
        coverage_merge = coverage_tendencies.rename(columns={'defense_team': opponent_col})
        merge_on = [c for c in [opponent_col, 'week', 'season'] if c in coverage_merge.columns and c in df.columns]

        if len(merge_on) >= 1:
            cov_cols = [opponent_col, 'week', 'season', 'man_coverage_rate_trailing']
            cov_cols = [c for c in cov_cols if c in coverage_merge.columns]

            df = df.merge(
                coverage_merge[cov_cols].rename(columns={'man_coverage_rate_trailing': 'opp_man_coverage_rate_trailing'}),
                on=merge_on,
                how='left',
                suffixes=('', '_cov')
            )

    # Merge slot funnel
    if slot_funnel is not None and len(slot_funnel) > 0:
        funnel_merge = slot_funnel.rename(columns={'defense_team': opponent_col})
        merge_on = [c for c in [opponent_col, 'week', 'season'] if c in funnel_merge.columns and c in df.columns]

        if len(merge_on) >= 1:
            funnel_cols = [opponent_col, 'week', 'season', 'slot_funnel_score_trailing']
            funnel_cols = [c for c in funnel_cols if c in funnel_merge.columns]

            df = df.merge(
                funnel_merge[funnel_cols].rename(columns={'slot_funnel_score_trailing': 'slot_funnel_score'}),
                on=merge_on,
                how='left',
                suffixes=('', '_funnel')
            )

    # Calculate man coverage adjustment
    if 'opp_man_coverage_rate_trailing' in df.columns:
        df['man_coverage_adjustment'] = df.apply(
            lambda row: calculate_man_coverage_adjustment(
                is_wr1=bool(row.get('is_starter', False)),
                is_slot=bool(row.get('is_slot_receiver', False)),
                opp_man_coverage_rate=row.get('opp_man_coverage_rate_trailing', np.nan)
            ) if V24_AVAILABLE else 1.0,
            axis=1
        )
    else:
        df['man_coverage_adjustment'] = 1.0

    # Calculate interaction feature
    df['position_role_x_opp_yards'] = (
        df['pos_rank'].fillna(2) *
        df['opp_position_yards_allowed_trailing'].fillna(0)
    )

    # Set has_position_context flag
    df['has_position_context'] = (
        df['pos_rank'].notna() &
        (df['pos_rank'] < 99) &
        (df['opp_position_yards_allowed_trailing'].notna() | df['opp_man_coverage_rate_trailing'].notna())
    ).astype(int)

    # Log coverage
    coverage = df['has_position_context'].mean()
    logger.info(f"V24 position matchup coverage: {coverage:.1%} of rows have context")

    return df
