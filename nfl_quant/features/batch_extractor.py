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

# Import V25 team synergy features
try:
    from nfl_quant.features.team_synergy_extractor import (
        extract_team_synergy_features,
        clear_synergy_cache,
        V25_SYNERGY_FEATURES,
    )
    V25_SYNERGY_AVAILABLE = True
except ImportError as e:
    V25_SYNERGY_AVAILABLE = False
    logger.warning(f"V25 synergy features not available: {e}")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Import config for sweet spot calculation
try:
    from configs.model_config import (
        smooth_sweet_spot,
        FEATURE_FLAGS,
        TRAILING_DEFLATION_FACTORS,
        DEFAULT_TRAILING_DEFLATION,
        EWMA_SPAN,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.warning("Config not available, using legacy calculations")


# =========================================================================
# POSITION-SPECIFIC DEFAULTS (computed from data, not hardcoded)
# =========================================================================
_POSITION_TARGET_SHARE_CACHE = None
_POSITION_CATCH_RATE_CACHE = None
_POSITION_SNAP_SHARE_CACHE = None


def _compute_position_target_share_averages() -> dict:
    """
    Compute position-specific target_share averages from weekly_stats.

    Returns dict like {'WR': 0.131, 'TE': 0.099, 'RB': 0.061, ...}
    Cached after first computation.
    """
    global _POSITION_TARGET_SHARE_CACHE

    if _POSITION_TARGET_SHARE_CACHE is not None:
        return _POSITION_TARGET_SHARE_CACHE

    try:
        ws_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if not ws_path.exists():
            logger.warning("weekly_stats not found for position averages")
            # Fallback to empirically computed averages (from actual data analysis)
            _POSITION_TARGET_SHARE_CACHE = {
                'WR': 0.131, 'TE': 0.099, 'RB': 0.061, 'FB': 0.026, 'QB': 0.0
            }
            return _POSITION_TARGET_SHARE_CACHE

        ws = pd.read_parquet(ws_path)
        if 'target_share' not in ws.columns or 'position' not in ws.columns:
            logger.warning("Missing columns for position averages")
            _POSITION_TARGET_SHARE_CACHE = {
                'WR': 0.131, 'TE': 0.099, 'RB': 0.061, 'FB': 0.026, 'QB': 0.0
            }
            return _POSITION_TARGET_SHARE_CACHE

        # Compute actual averages from data
        pos_avg = ws.groupby('position')['target_share'].mean().to_dict()

        # Ensure we have all key positions
        defaults = {'WR': 0.131, 'TE': 0.099, 'RB': 0.061, 'FB': 0.026, 'QB': 0.0}
        for pos, val in defaults.items():
            if pos not in pos_avg:
                pos_avg[pos] = val

        _POSITION_TARGET_SHARE_CACHE = pos_avg
        logger.info(f"Computed position target_share averages: WR={pos_avg.get('WR', 0):.3f}, "
                   f"TE={pos_avg.get('TE', 0):.3f}, RB={pos_avg.get('RB', 0):.3f}")
        return _POSITION_TARGET_SHARE_CACHE

    except Exception as e:
        logger.warning(f"Failed to compute position averages: {e}")
        _POSITION_TARGET_SHARE_CACHE = {
            'WR': 0.131, 'TE': 0.099, 'RB': 0.061, 'FB': 0.026, 'QB': 0.0
        }
        return _POSITION_TARGET_SHARE_CACHE


def _get_position_target_share_default(position: str) -> float:
    """Get position-specific target_share default (for prediction only)."""
    pos_avg = _compute_position_target_share_averages()
    return pos_avg.get(position, 0.10)  # 0.10 as absolute fallback (lower than old 0.15)


def _fill_target_share_by_position(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Fill missing target_share values.

    For training: Leave as NaN (XGBoost handles missing values natively)
    For prediction: Use position-specific averages computed from data
    """
    if 'target_share' not in df.columns:
        if for_training:
            df['target_share'] = np.nan
        else:
            pos_avg = _compute_position_target_share_averages()
            if 'position' in df.columns:
                df['target_share'] = df['position'].map(lambda p: pos_avg.get(p, 0.10))
            else:
                df['target_share'] = 0.10
        return df

    if for_training:
        # For training, leave NaN as-is - XGBoost handles it
        return df

    # For prediction, fill missing with position-specific averages
    if 'position' in df.columns:
        pos_avg = _compute_position_target_share_averages()
        mask = df['target_share'].isna()
        df.loc[mask, 'target_share'] = df.loc[mask, 'position'].map(
            lambda p: pos_avg.get(p, 0.10)
        )
    else:
        # No position column - use overall average (lower than old 0.15)
        df['target_share'] = df['target_share'].fillna(0.10)

    return df


# =========================================================================
# POSITION-SPECIFIC CATCH RATE (computed from data, not hardcoded 0.65-0.68)
# =========================================================================

def _compute_position_catch_rate_averages() -> dict:
    """
    Compute position-specific catch rates from weekly_stats.

    Actual rates from 2023-2024 data:
    - WR: 63.1% (close to old 65% default)
    - TE: 72.5% (old default underestimated by 11%)
    - RB: 78.9% (old default underestimated by 21%)
    - FB: 77.5%

    Returns dict like {'WR': 0.631, 'TE': 0.725, 'RB': 0.789, ...}
    Cached after first computation.
    """
    global _POSITION_CATCH_RATE_CACHE

    if _POSITION_CATCH_RATE_CACHE is not None:
        return _POSITION_CATCH_RATE_CACHE

    try:
        ws_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if not ws_path.exists():
            logger.warning("weekly_stats not found for catch rate averages")
            # Fallback to empirically computed averages (from actual data analysis)
            _POSITION_CATCH_RATE_CACHE = {
                'WR': 0.631, 'TE': 0.725, 'RB': 0.789, 'FB': 0.775, 'QB': 0.0
            }
            return _POSITION_CATCH_RATE_CACHE

        ws = pd.read_parquet(ws_path)
        if 'receptions' not in ws.columns or 'targets' not in ws.columns:
            logger.warning("Missing receptions/targets for catch rate averages")
            _POSITION_CATCH_RATE_CACHE = {
                'WR': 0.631, 'TE': 0.725, 'RB': 0.789, 'FB': 0.775, 'QB': 0.0
            }
            return _POSITION_CATCH_RATE_CACHE

        # Compute actual catch rates from data (receptions / targets)
        rec_positions = ['WR', 'TE', 'RB', 'FB']
        rec_data = ws[(ws['position'].isin(rec_positions)) & (ws['targets'] > 0)]

        pos_rates = {}
        for pos in rec_positions:
            pos_data = rec_data[rec_data['position'] == pos]
            total_rec = pos_data['receptions'].sum()
            total_targets = pos_data['targets'].sum()
            if total_targets > 0:
                pos_rates[pos] = total_rec / total_targets
            else:
                # Fallback to precomputed averages
                defaults = {'WR': 0.631, 'TE': 0.725, 'RB': 0.789, 'FB': 0.775}
                pos_rates[pos] = defaults.get(pos, 0.68)

        pos_rates['QB'] = 0.0  # QBs don't have catch rates

        _POSITION_CATCH_RATE_CACHE = pos_rates
        logger.info(f"Computed position catch_rate averages: WR={pos_rates.get('WR', 0):.3f}, "
                   f"TE={pos_rates.get('TE', 0):.3f}, RB={pos_rates.get('RB', 0):.3f}")
        return _POSITION_CATCH_RATE_CACHE

    except Exception as e:
        logger.warning(f"Failed to compute catch rate averages: {e}")
        _POSITION_CATCH_RATE_CACHE = {
            'WR': 0.631, 'TE': 0.725, 'RB': 0.789, 'FB': 0.775, 'QB': 0.0
        }
        return _POSITION_CATCH_RATE_CACHE


def _fill_catch_rate_by_position(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Fill missing trailing_catch_rate values.

    For training: Leave as NaN (XGBoost handles missing values natively)
    For prediction: Use position-specific averages computed from data
    """
    if 'trailing_catch_rate' not in df.columns:
        if for_training:
            df['trailing_catch_rate'] = np.nan
        else:
            pos_rates = _compute_position_catch_rate_averages()
            if 'position' in df.columns:
                df['trailing_catch_rate'] = df['position'].map(lambda p: pos_rates.get(p, 0.68))
            else:
                df['trailing_catch_rate'] = 0.68  # Overall league average
        return df

    if for_training:
        # For training, leave NaN as-is - XGBoost handles it
        return df

    # For prediction, fill missing with position-specific averages
    if 'position' in df.columns:
        pos_rates = _compute_position_catch_rate_averages()
        mask = df['trailing_catch_rate'].isna()
        df.loc[mask, 'trailing_catch_rate'] = df.loc[mask, 'position'].map(
            lambda p: pos_rates.get(p, 0.68)
        )
    else:
        # No position column - use overall average
        df['trailing_catch_rate'] = df['trailing_catch_rate'].fillna(0.68)

    return df


# =========================================================================
# POSITION-SPECIFIC SNAP SHARE (computed from data, not hardcoded 0.70)
# =========================================================================

def _compute_position_snap_share_averages() -> dict:
    """
    Compute position-specific snap shares from snap_counts.

    Actual rates from data:
    - QB: 79.8% (higher than 70% default)
    - WR: 51.6% (70% default overestimates by 26%)
    - TE: 44.2% (70% default overestimates by 37%)
    - RB: 37.7% (70% default overestimates by 46%)
    - FB: 26.3%

    Returns dict like {'QB': 0.798, 'WR': 0.516, 'TE': 0.442, 'RB': 0.377, ...}
    Cached after first computation.
    """
    global _POSITION_SNAP_SHARE_CACHE

    if _POSITION_SNAP_SHARE_CACHE is not None:
        return _POSITION_SNAP_SHARE_CACHE

    try:
        snap_path = PROJECT_ROOT / 'data' / 'nflverse' / 'snap_counts.parquet'
        if not snap_path.exists():
            logger.warning("snap_counts not found for snap share averages")
            # Fallback to empirically computed averages
            _POSITION_SNAP_SHARE_CACHE = {
                'QB': 0.798, 'WR': 0.516, 'TE': 0.442, 'RB': 0.377, 'FB': 0.263
            }
            return _POSITION_SNAP_SHARE_CACHE

        snaps = pd.read_parquet(snap_path)
        if 'offense_pct' not in snaps.columns or 'position' not in snaps.columns:
            logger.warning("Missing columns for snap share averages")
            _POSITION_SNAP_SHARE_CACHE = {
                'QB': 0.798, 'WR': 0.516, 'TE': 0.442, 'RB': 0.377, 'FB': 0.263
            }
            return _POSITION_SNAP_SHARE_CACHE

        # Compute actual snap shares from data
        skill_positions = ['QB', 'WR', 'TE', 'RB', 'FB']
        skill_data = snaps[(snaps['position'].isin(skill_positions)) & (snaps['offense_snaps'] > 0)]

        pos_shares = {}
        for pos in skill_positions:
            pos_data = skill_data[skill_data['position'] == pos]
            if len(pos_data) > 0:
                pos_shares[pos] = pos_data['offense_pct'].mean()
            else:
                # Fallback to precomputed averages
                defaults = {'QB': 0.798, 'WR': 0.516, 'TE': 0.442, 'RB': 0.377, 'FB': 0.263}
                pos_shares[pos] = defaults.get(pos, 0.50)

        _POSITION_SNAP_SHARE_CACHE = pos_shares
        logger.info(f"Computed position snap_share averages: QB={pos_shares.get('QB', 0):.3f}, "
                   f"WR={pos_shares.get('WR', 0):.3f}, RB={pos_shares.get('RB', 0):.3f}")
        return _POSITION_SNAP_SHARE_CACHE

    except Exception as e:
        logger.warning(f"Failed to compute snap share averages: {e}")
        _POSITION_SNAP_SHARE_CACHE = {
            'QB': 0.798, 'WR': 0.516, 'TE': 0.442, 'RB': 0.377, 'FB': 0.263
        }
        return _POSITION_SNAP_SHARE_CACHE


def _fill_snap_share_by_position(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Fill missing snap_share values.

    For training: Leave as NaN (XGBoost handles missing values natively)
    For prediction: Use position-specific averages computed from data
    """
    if 'snap_share' not in df.columns:
        if for_training:
            df['snap_share'] = np.nan
        else:
            pos_shares = _compute_position_snap_share_averages()
            if 'position' in df.columns:
                df['snap_share'] = df['position'].map(lambda p: pos_shares.get(p, 0.50))
            else:
                df['snap_share'] = 0.50  # Overall fallback
        return df

    if for_training:
        # For training, leave NaN as-is - XGBoost handles it
        return df

    # For prediction, fill missing with position-specific averages
    if 'position' in df.columns:
        pos_shares = _compute_position_snap_share_averages()
        mask = df['snap_share'].isna()
        df.loc[mask, 'snap_share'] = df.loc[mask, 'position'].map(
            lambda p: pos_shares.get(p, 0.50)
        )
    else:
        # No position column - use overall average
        df['snap_share'] = df['snap_share'].fillna(0.50)

    return df


def extract_features_batch(
    odds_with_trailing: pd.DataFrame,
    all_historical_odds: pd.DataFrame,
    market: str,
    target_global_week: int = None,
    for_training: bool = True,
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
        for_training: If True (default), leave missing values as NaN for XGBoost.
                      If False, fill missing values with position-specific averages.

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
    # FIX: Load from ENRICHED.csv as fallback if historical odds are empty
    # =========================================================================
    # Handle empty or missing historical odds
    if len(all_historical_odds) == 0 or 'market' not in all_historical_odds.columns:
        hist_odds = pd.DataFrame()
    else:
        hist_odds = all_historical_odds[all_historical_odds['market'] == market].copy()

    # If historical odds are empty or very small, load from enriched training data
    if len(hist_odds) < 100:
        hist_odds = _load_enriched_historical_odds_fallback(market)
        if len(hist_odds) > 0:
            logger.info(f"    Using enriched historical odds fallback: {len(hist_odds)} rows")

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
    df = _merge_skill_features_vectorized(df, market, for_training=for_training)

    # =========================================================================
    # STEP 4: Add game context features (defaults for training)
    # Only fills truly missing values since Step 3 calculated real values
    # =========================================================================
    df = _add_game_context_defaults(df, for_training=for_training)

    # =========================================================================
    # STEP 5: Add rush/receiving features
    # =========================================================================
    df = _add_rush_receiving_features(df, market, for_training=for_training)

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
    # STEP 8: Add V25 team synergy features
    # Models compound effects of multiple players returning simultaneously
    # Features: team_synergy_multiplier, oline_health_score_v25, wr_corps_health,
    #           has_synergy_bonus, cascade_efficiency_boost, wr_coverage_reduction,
    #           returning_player_count, has_synergy_context
    # =========================================================================
    if V25_SYNERGY_AVAILABLE:
        try:
            # Determine week and season from data
            current_week = int(df['week'].max()) if 'week' in df.columns else None
            current_season = int(df['season'].max()) if 'season' in df.columns else 2025

            df = extract_team_synergy_features(
                df,
                week=current_week,
                season=current_season,
                team_col='recent_team'
            )
            logger.info(f"    V25 synergy features added for {market}")
        except Exception as e:
            logger.warning(f"    V25 synergy feature extraction failed: {e}")
            # Set defaults for synergy features
            from nfl_quant.features.feature_defaults import FEATURE_DEFAULTS
            for feature in V25_SYNERGY_FEATURES:
                df[feature] = FEATURE_DEFAULTS.get(feature, 0.0)

    # =========================================================================
    # STEP 9: V28 Elo and Situational Features
    # =========================================================================
    df = _add_v28_elo_situational_features(df)

    # =========================================================================
    # STEP 9.5: V28.1 Player Injury Features (for backtesting)
    # =========================================================================
    df = _add_player_injury_features(df)

    # =========================================================================
    # STEP 10: Add row identifiers
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

    # Apply trailing stat deflation (regression to mean adjustment)
    # This corrects for the systematic overshoot of trailing stats vs actuals
    if CONFIG_AVAILABLE:
        deflation_factor = TRAILING_DEFLATION_FACTORS.get(market, DEFAULT_TRAILING_DEFLATION)
    else:
        deflation_factor = 0.90  # Fallback: overall median deflation

    # Deflate trailing stat to more realistic expectation
    deflated_trailing = df[trailing_col] * deflation_factor

    # Line vs Trailing (LVT) - percentage method
    # Avoid division by zero
    trailing_safe = deflated_trailing.replace(0, np.nan).fillna(0.001)
    df['line_vs_trailing'] = ((df['line'] - deflated_trailing) / trailing_safe) * 100

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

    # V29 Features: Vegas Agreement Signal
    # Week 16 analysis showed: WITH Vegas = 67% win rate, AGAINST = 27%
    # lvt_direction: +1 if line > trailing (Vegas expects UNDER), -1 if opposite
    df['lvt_direction'] = np.sign(df['line_vs_trailing'])
    # vegas_agreement: 1 if betting UNDER and Vegas expects UNDER (LVT > 0)
    # This is for UNDER bets - the model predicts P(UNDER), so when LVT > 0,
    # the bet agrees with Vegas direction
    df['vegas_agreement'] = (df['line_vs_trailing'] > 0).astype(float)

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


def _merge_skill_features_vectorized(df: pd.DataFrame, market: str, for_training: bool = True) -> pd.DataFrame:
    """
    Merge skill features from pre-loaded lookup tables.
    Uses merge operations instead of per-row function calls.

    Args:
        for_training: If True, leave missing values as NaN. If False, use position-specific defaults.
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
        # Fill missing snap_share with position-specific averages (not 0.0!)
        df = _fill_snap_share_by_position(df, for_training=for_training)
        df = df.drop(columns=['lookup_week', 'player_lower'], errors='ignore')
    else:
        # No snap data - use position-specific defaults
        df = _fill_snap_share_by_position(df, for_training=for_training)

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
            # Actual values from NGS data: separation=3.07, cushion=6.09
            if for_training:
                pass  # Leave NaN for XGBoost
            else:
                df['avg_separation'] = df['avg_separation'].fillna(3.07)  # Actual from NGS
                df['avg_cushion'] = df['avg_cushion'].fillna(6.09)  # Actual from NGS
            df = df.drop(columns=['lookup_week', 'week_ngs'], errors='ignore')
        else:
            if for_training:
                df['avg_separation'] = np.nan
                df['avg_cushion'] = np.nan
            else:
                df['avg_separation'] = 3.07  # Actual from NGS (was 2.8)
                df['avg_cushion'] = 6.09  # Actual from NGS (was 6.2)
    else:
        if for_training:
            df['avg_separation'] = np.nan
            df['avg_cushion'] = np.nan
        else:
            df['avg_separation'] = 3.07  # Actual from NGS (was 2.8)
            df['avg_cushion'] = 6.09  # Actual from NGS (was 6.2)

    # Fill remaining skill features with defaults
    # For training: leave NaN (XGBoost handles it natively)
    # For prediction: use position-specific averages computed from data
    df = _fill_catch_rate_by_position(df, for_training=for_training)

    # =========================================================================
    # FIX: Compute target_share from weekly_stats when not in enriched data
    # This fixes the issue where live odds don't have target_share_stats
    # For training: leave NaN (XGBoost handles it natively)
    # For prediction: use position-specific averages
    # =========================================================================
    if 'target_share_stats' in df.columns:
        # Use enriched target_share if available
        if for_training:
            df['target_share'] = df['target_share_stats']  # Keep NaN for training
        else:
            df['target_share'] = df['target_share_stats'].fillna(0.0)
    elif 'target_share' not in df.columns or (df['target_share'] == 0).all():
        # Compute from weekly_stats when not available
        df = _compute_target_share_from_weekly_stats(df, for_training=for_training)
    else:
        # Fill missing values appropriately
        df = _fill_target_share_by_position(df, for_training=for_training)

    # Actual WR1 receptions allowed: 5.2 (was 5.5)
    if 'opp_wr1_receptions_allowed' not in df.columns:
        if for_training:
            df['opp_wr1_receptions_allowed'] = np.nan
        else:
            df['opp_wr1_receptions_allowed'] = 5.2  # Actual from data (was 5.5)
    elif not for_training:
        df['opp_wr1_receptions_allowed'] = df['opp_wr1_receptions_allowed'].fillna(5.2)

    return df


def _add_game_context_defaults(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Add game context feature columns with default values.
    Only fills missing values - preserves existing data from enriched sources.

    Args:
        for_training: If True, leave missing as NaN. If False, use computed averages.
    """
    df = df.copy()

    # Game Context features - only fill if missing or NaN
    # For training: use NaN for XGBoost. For prediction: use league averages.
    # Actual game_pace from PBP: ~65 offensive plays per team per game
    if 'game_pace' not in df.columns:
        df['game_pace'] = np.nan if for_training else 65.0  # Actual from PBP
    elif not for_training:
        df['game_pace'] = df['game_pace'].fillna(65.0)

    if 'vegas_total' not in df.columns:
        df['vegas_total'] = np.nan if for_training else 44.0  # League avg O/U
    elif not for_training:
        df['vegas_total'] = df['vegas_total'].fillna(44.0)

    if 'vegas_spread' not in df.columns:
        df['vegas_spread'] = np.nan if for_training else 0.0  # Neutral
    elif not for_training:
        df['vegas_spread'] = df['vegas_spread'].fillna(0.0)

    if 'implied_team_total' not in df.columns:
        df['implied_team_total'] = np.nan if for_training else 22.0  # League avg
    elif not for_training:
        df['implied_team_total'] = df['implied_team_total'].fillna(22.0)

    # Skill features - only fill if missing or NaN
    # Actual ADOT from PBP: 7.75 (was 8.5)
    if 'adot' not in df.columns:
        df['adot'] = np.nan if for_training else 7.75  # Actual from PBP (was 8.5)
    elif not for_training:
        df['adot'] = df['adot'].fillna(7.75)

    # Pressure rate - actual league avg is 15.4% (not 25%!)
    # Computed from 2024 participation data: 45,905 plays, 15.4% pressure rate
    ACTUAL_PRESSURE_RATE = 0.154  # Computed from data, was incorrectly 0.25
    if 'pressure_rate' not in df.columns:
        df['pressure_rate'] = np.nan if for_training else ACTUAL_PRESSURE_RATE
    elif not for_training:
        df['pressure_rate'] = df['pressure_rate'].fillna(ACTUAL_PRESSURE_RATE)

    if 'opp_pressure_rate' not in df.columns:
        df['opp_pressure_rate'] = np.nan if for_training else ACTUAL_PRESSURE_RATE
    elif not for_training:
        df['opp_pressure_rate'] = df['opp_pressure_rate'].fillna(ACTUAL_PRESSURE_RATE)

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


def _add_rush_receiving_features(df: pd.DataFrame, market: str, for_training: bool = True) -> pd.DataFrame:
    """
    Add rush/receiving specific features.

    Features:
    - oline_health_score: O-line injury weighted availability
    - box_count_expected: Expected box count from spread
    - slot_snap_pct: Proxy from aDOT (lower aDOT = more slot)
    - target_share_trailing: 4-week rolling target share

    Args:
        for_training: If True, leave missing as NaN. If False, use position-specific defaults.
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
        df = _compute_target_share_trailing_vectorized(df, for_training=for_training)
    else:
        # For non-receiving markets, target_share_trailing not as relevant
        # For training: leave as NaN. For prediction: use position-specific default
        if for_training:
            df['target_share_trailing'] = np.nan
        else:
            df = _fill_target_share_by_position(df, for_training=False)
            if 'target_share' in df.columns:
                df['target_share_trailing'] = df['target_share']
            else:
                df['target_share_trailing'] = 0.10

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


def _compute_target_share_from_weekly_stats(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Compute target_share from weekly_stats when not present in input data.

    This fixes the issue where target_share_stats is only available in enriched
    training data but not in live odds data. We compute it from weekly_stats.parquet.

    Uses the PRIOR week's target_share to avoid data leakage.

    Args:
        for_training: If True, leave missing as NaN. If False, use position-specific defaults.
    """
    df = df.copy()

    # Skip if target_share already populated with real values
    if 'target_share' in df.columns and (df['target_share'] > 0).any():
        logger.debug(f"target_share already populated: {(df['target_share'] > 0).mean():.1%} non-zero")
        return df

    # Load weekly stats
    weekly_stats = _load_weekly_stats_lookup()

    if len(weekly_stats) == 0:
        logger.warning("weekly_stats not available for target_share computation")
        # For training: leave as NaN. For prediction: use position-specific defaults
        return _fill_target_share_by_position(df, for_training=for_training)

    if 'target_share' not in weekly_stats.columns:
        logger.warning("target_share column not in weekly_stats")
        return _fill_target_share_by_position(df, for_training=for_training)

    # Normalize player names for matching
    weekly_stats = weekly_stats.copy()
    weekly_stats['player_norm'] = weekly_stats['player_display_name'].str.lower().str.strip()

    # Use prior week's target_share (week - 1) to avoid leakage
    # First, compute 4-week trailing average for smoothing
    weekly_stats = weekly_stats.sort_values(['player_norm', 'season', 'week'])
    weekly_stats['target_share_smooth'] = weekly_stats.groupby('player_norm')['target_share'].transform(
        lambda x: x.shift(1).ewm(span=EWMA_SPAN if CONFIG_AVAILABLE else 4, min_periods=1).mean()
    )

    # Create lookup with week+1 for no leakage (prior week's value for current week)
    ts_lookup = weekly_stats[['player_norm', 'season', 'week', 'target_share_smooth']].copy()
    ts_lookup['lookup_week'] = ts_lookup['week'] + 1
    ts_lookup = ts_lookup.rename(columns={'target_share_smooth': 'target_share_computed'})

    # Merge
    if 'player_norm' in df.columns:
        df = df.merge(
            ts_lookup[['player_norm', 'season', 'lookup_week', 'target_share_computed']],
            left_on=['player_norm', 'season', 'week'],
            right_on=['player_norm', 'season', 'lookup_week'],
            how='left',
            suffixes=('', '_ws')
        )

        # Fill target_share with computed values where missing
        if 'target_share' not in df.columns:
            df['target_share'] = df['target_share_computed']
        else:
            df['target_share'] = df['target_share'].fillna(df['target_share_computed'])

        df = df.drop(columns=['lookup_week', 'target_share_computed', 'player_norm_ws'], errors='ignore')

        # For prediction only: fill remaining NaN with position-specific defaults
        if not for_training:
            df = _fill_target_share_by_position(df, for_training=False)

        coverage = (df['target_share'].notna() & (df['target_share'] > 0)).mean()
        logger.info(f"    target_share computed from weekly_stats: {coverage:.1%} coverage")
    else:
        # No player_norm column - use position-specific fallbacks
        df = _fill_target_share_by_position(df, for_training=for_training)

    return df


def _compute_target_share_trailing_vectorized(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Compute 4-week trailing target share from weekly stats.

    Args:
        for_training: If True, leave missing as NaN. If False, use position-specific defaults.
    """
    df = df.copy()

    # Load weekly stats for target share
    weekly_stats = _load_weekly_stats_lookup()

    if len(weekly_stats) == 0:
        # For training: leave as NaN. For prediction: use position-specific defaults
        if for_training:
            df['target_share_trailing'] = np.nan
        else:
            df = _fill_target_share_by_position(df, for_training=False)
            df['target_share_trailing'] = df.get('target_share', 0.10)
        return df

    # Check if target_share column exists
    if 'target_share' not in weekly_stats.columns:
        if for_training:
            df['target_share_trailing'] = np.nan
        else:
            df = _fill_target_share_by_position(df, for_training=False)
            df['target_share_trailing'] = df.get('target_share', 0.10)
        return df

    # Normalize player names for matching
    weekly_stats = weekly_stats.copy()
    weekly_stats['player_norm'] = weekly_stats['player_display_name'].str.lower().str.strip()

    # Calculate trailing target share (4-week EWMA)
    weekly_stats = weekly_stats.sort_values(['player_norm', 'season', 'week'])
    weekly_stats['target_share_trailing'] = weekly_stats.groupby('player_norm')['target_share'].transform(
        lambda x: x.shift(1).ewm(span=EWMA_SPAN if CONFIG_AVAILABLE else 4, min_periods=1).mean()
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
        # For prediction only: fill missing with position-specific defaults
        if not for_training:
            df = _fill_target_share_by_position(df, for_training=False)
            mask = df['target_share_trailing'].isna()
            if 'target_share' in df.columns:
                df.loc[mask, 'target_share_trailing'] = df.loc[mask, 'target_share']
        df = df.drop(columns=['lookup_week', 'player_norm_ts'], errors='ignore')
    else:
        if for_training:
            df['target_share_trailing'] = np.nan
        else:
            df = _fill_target_share_by_position(df, for_training=False)
            df['target_share_trailing'] = df.get('target_share', 0.10)

    return df


# =========================================================================
# LOOKUP TABLE LOADERS (cached)
# =========================================================================

_SNAP_COUNTS_CACHE = None
_NGS_RECEIVING_CACHE = None
_INJURIES_CACHE = None
_WEEKLY_STATS_CACHE = None
_ENRICHED_ODDS_CACHE = None

# V24 Position Matchup Caches
_V24_POSITION_ROLES_CACHE = None
_V24_DEFENSE_VS_POSITION_CACHE = None
_V24_COVERAGE_TENDENCIES_CACHE = None
_V24_SLOT_FUNNEL_CACHE = None


def _load_enriched_historical_odds_fallback(market: str) -> pd.DataFrame:
    """
    Load historical odds from ENRICHED.csv as fallback.

    This is used when the all_historical_odds parameter is empty or too small,
    which happens during live predictions. We load from the enriched training
    data to compute player_under_rate, player_bias, and market_under_rate.
    """
    global _ENRICHED_ODDS_CACHE

    if _ENRICHED_ODDS_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
                # Normalize player names
                if 'player' in df.columns:
                    df['player_norm'] = df['player'].str.lower().str.strip()
                # Add global_week if not present
                if 'global_week' not in df.columns and 'season' in df.columns and 'week' in df.columns:
                    df['global_week'] = (df['season'] - 2023) * 18 + df['week']
                _ENRICHED_ODDS_CACHE = df
                logger.debug(f"Loaded enriched odds fallback: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load enriched odds: {e}")
                _ENRICHED_ODDS_CACHE = pd.DataFrame()
        else:
            logger.warning(f"Enriched odds not found at {path}")
            _ENRICHED_ODDS_CACHE = pd.DataFrame()

    if len(_ENRICHED_ODDS_CACHE) == 0:
        return pd.DataFrame()

    # Filter to the requested market
    return _ENRICHED_ODDS_CACHE[_ENRICHED_ODDS_CACHE['market'] == market].copy()


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
    global _ENRICHED_ODDS_CACHE
    _SNAP_COUNTS_CACHE = None
    _NGS_RECEIVING_CACHE = None
    _INJURIES_CACHE = None
    _WEEKLY_STATS_CACHE = None
    _OPP_DEFENSE_CACHE = None
    _ENRICHED_ODDS_CACHE = None
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
    # V25 synergy caches
    if V25_SYNERGY_AVAILABLE:
        try:
            clear_synergy_cache()
        except Exception:
            pass
    # V28 Elo/situational caches
    try:
        clear_v28_caches()
    except Exception:
        pass
    # V28.1 unified injury cache
    try:
        clear_injury_cache()
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
        coverage_merge = coverage_tendencies.rename(columns={'defense_team': opponent_col}).copy()
        # FIX: Ensure consistent types for merge keys (coverage cache has float weeks)
        if 'week' in coverage_merge.columns:
            coverage_merge['week'] = coverage_merge['week'].astype(int)
        if 'season' in coverage_merge.columns:
            coverage_merge['season'] = coverage_merge['season'].astype(int)

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
            logger.debug(f"Coverage merge result: {df['opp_man_coverage_rate_trailing'].notna().mean():.1%} coverage")

    # Merge slot funnel
    if slot_funnel is not None and len(slot_funnel) > 0:
        funnel_merge = slot_funnel.rename(columns={'defense_team': opponent_col}).copy()
        # FIX: Ensure consistent types for merge keys
        if 'week' in funnel_merge.columns:
            funnel_merge['week'] = funnel_merge['week'].astype(int)
        if 'season' in funnel_merge.columns:
            funnel_merge['season'] = funnel_merge['season'].astype(int)

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
            logger.debug(f"Slot funnel merge result: {df['slot_funnel_score'].notna().mean():.1%} coverage")

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


# =========================================================================
# V28 ELO AND SITUATIONAL FEATURES
# =========================================================================

_V28_ELO_CACHE = None
_V28_SCHEDULE_CACHE = None


def _load_v28_elo_ratings():
    """Load V28 Elo ratings from saved file or initialize from schedule."""
    global _V28_ELO_CACHE

    if _V28_ELO_CACHE is not None:
        return _V28_ELO_CACHE

    try:
        from nfl_quant.models.elo_ratings import load_elo_ratings, initialize_elo_from_nflverse

        elo_path = PROJECT_ROOT / 'data' / 'models' / 'elo_ratings.json'
        if elo_path.exists():
            _V28_ELO_CACHE = load_elo_ratings(str(elo_path))
            logger.info(f"Loaded V28 Elo ratings: {len(_V28_ELO_CACHE.ratings)} teams")
        else:
            # Initialize from nflverse schedule
            logger.info("Initializing V28 Elo ratings from schedule...")
            _V28_ELO_CACHE = initialize_elo_from_nflverse(seasons=[2023, 2024, 2025])

    except Exception as e:
        logger.warning(f"V28 Elo ratings not available: {e}")
        _V28_ELO_CACHE = None

    return _V28_ELO_CACHE


def _load_v28_schedule():
    """Load schedule for rest days and HFA."""
    global _V28_SCHEDULE_CACHE

    if _V28_SCHEDULE_CACHE is not None:
        return _V28_SCHEDULE_CACHE

    schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if not schedule_path.exists():
        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.csv'

    if not schedule_path.exists():
        _V28_SCHEDULE_CACHE = pd.DataFrame()
        return _V28_SCHEDULE_CACHE

    try:
        if schedule_path.suffix == '.parquet':
            _V28_SCHEDULE_CACHE = pd.read_parquet(schedule_path)
        else:
            _V28_SCHEDULE_CACHE = pd.read_csv(schedule_path)
        logger.debug(f"Loaded V28 schedule: {len(_V28_SCHEDULE_CACHE)} games")
    except Exception as e:
        logger.warning(f"Failed to load schedule: {e}")
        _V28_SCHEDULE_CACHE = pd.DataFrame()

    return _V28_SCHEDULE_CACHE


def _add_v28_elo_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add V28 Elo and situational features.

    Features:
    - elo_rating_home: Home team Elo rating
    - elo_rating_away: Away team Elo rating
    - elo_diff: Home - Away Elo (with HFA)
    - ybc_proxy: Yards Before Contact proxy (filled elsewhere)
    - rest_days: Days since last game
    - hfa_adjustment: Team-specific HFA factor
    """
    df = df.copy()

    # Initialize V28 feature columns with defaults
    V28_FEATURES = [
        'elo_rating_home',
        'elo_rating_away',
        'elo_diff',
        'ybc_proxy',
        'rest_days',
        'hfa_adjustment',
    ]

    for feat in V28_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan

    # Determine team/opponent columns
    team_col = None
    for col in ['team', 'recent_team', 'player_team']:
        if col in df.columns:
            team_col = col
            break

    opponent_col = None
    for col in ['opponent', 'opponent_team', 'opp']:
        if col in df.columns:
            opponent_col = col
            break

    if team_col is None:
        logger.debug("V28: No team column found, using defaults")
        df['elo_rating_home'] = 1500.0
        df['elo_rating_away'] = 1500.0
        df['elo_diff'] = 0.0
        df['rest_days'] = 7.0
        df['hfa_adjustment'] = 1.0
        df['ybc_proxy'] = 5.0
        return df

    # Load Elo ratings
    elo = _load_v28_elo_ratings()
    schedule = _load_v28_schedule()

    if elo is not None:
        # Add Elo ratings for each team
        df['elo_rating_home'] = df[team_col].map(lambda t: elo.get_rating(t))
        if opponent_col:
            df['elo_rating_away'] = df[opponent_col].map(lambda t: elo.get_rating(t))
            # Elo diff includes HFA
            df['elo_diff'] = df.apply(
                lambda row: elo.get_elo_diff(row[team_col], row[opponent_col])
                if pd.notna(row[team_col]) and pd.notna(row[opponent_col]) else 0.0,
                axis=1
            )
        else:
            df['elo_rating_away'] = 1500.0
            df['elo_diff'] = df['elo_rating_home'] - 1500.0
    else:
        df['elo_rating_home'] = 1500.0
        df['elo_rating_away'] = 1500.0
        df['elo_diff'] = 0.0

    # Add rest days from schedule
    if len(schedule) > 0 and 'home_rest' in schedule.columns:
        # Create rest lookup
        away_rest = schedule[['season', 'week', 'away_team', 'away_rest']].copy()
        away_rest.columns = ['season', 'week', 'team', 'rest_days']
        home_rest = schedule[['season', 'week', 'home_team', 'home_rest']].copy()
        home_rest.columns = ['season', 'week', 'team', 'rest_days']
        rest_lookup = pd.concat([away_rest, home_rest]).drop_duplicates(['season', 'week', 'team'])

        # Merge rest days
        if 'season' in df.columns and 'week' in df.columns:
            df = df.merge(
                rest_lookup.rename(columns={'team': team_col}),
                on=['season', 'week', team_col],
                how='left',
                suffixes=('', '_lookup')
            )
            if 'rest_days_lookup' in df.columns:
                df['rest_days'] = df['rest_days_lookup'].fillna(7.0)
                df = df.drop(columns=['rest_days_lookup'])
            else:
                df['rest_days'] = df['rest_days'].fillna(7.0)
    else:
        df['rest_days'] = 7.0

    # Add HFA adjustment
    try:
        from nfl_quant.features.situational_features import get_hfa_adjustment
        df['hfa_adjustment'] = df[team_col].apply(lambda t: get_hfa_adjustment(t, is_home=True))
    except Exception:
        df['hfa_adjustment'] = 1.0

    # YBC proxy defaults (actual calculation happens in core.py per-player)
    if 'position' in df.columns:
        position_ybc = {'WR': 8.0, 'TE': 6.5, 'RB': 1.5, 'QB': 2.0}
        df['ybc_proxy'] = df['position'].map(lambda p: position_ybc.get(p, 5.0))
    else:
        df['ybc_proxy'] = 5.0

    # Fill any remaining NaN
    df['elo_rating_home'] = df['elo_rating_home'].fillna(1500.0)
    df['elo_rating_away'] = df['elo_rating_away'].fillna(1500.0)
    df['elo_diff'] = df['elo_diff'].fillna(0.0)
    df['rest_days'] = df['rest_days'].fillna(7.0)
    df['hfa_adjustment'] = df['hfa_adjustment'].fillna(1.0)
    df['ybc_proxy'] = df['ybc_proxy'].fillna(5.0)

    logger.info(f"V28 Elo/situational features added: elo_home={df['elo_rating_home'].mean():.0f}, elo_diff={df['elo_diff'].mean():.1f}")

    return df


def clear_v28_caches():
    """Clear V28 Elo and situational feature caches."""
    global _V28_ELO_CACHE, _V28_SCHEDULE_CACHE
    _V28_ELO_CACHE = None
    _V28_SCHEDULE_CACHE = None


# =============================================================================
# V28.1 PLAYER INJURY FEATURES (for backtesting)
# =============================================================================

# Cache for unified injury data
_UNIFIED_INJURY_CACHE = None


def _load_unified_injury_data() -> pd.DataFrame:
    """
    Load unified injury data (NFLverse + Sleeper combined).

    Priority:
    1. Unified injury history (data/processed/unified_injury_history.parquet)
    2. Fall back to NFLverse only (data/nflverse/injuries.parquet)
    """
    global _UNIFIED_INJURY_CACHE

    if _UNIFIED_INJURY_CACHE is not None:
        return _UNIFIED_INJURY_CACHE

    # Try unified first
    unified_path = PROJECT_ROOT / 'data' / 'processed' / 'unified_injury_history.parquet'
    if unified_path.exists():
        try:
            _UNIFIED_INJURY_CACHE = pd.read_parquet(unified_path)
            logger.info(f"Loaded unified injury data: {len(_UNIFIED_INJURY_CACHE)} records (NFLverse + Sleeper)")
            return _UNIFIED_INJURY_CACHE
        except Exception as e:
            logger.warning(f"Failed to load unified injuries: {e}")

    # Fall back to NFLverse only
    nflverse_path = PROJECT_ROOT / 'data' / 'nflverse' / 'injuries.parquet'
    if nflverse_path.exists():
        try:
            _UNIFIED_INJURY_CACHE = pd.read_parquet(nflverse_path)
            # Add required columns for compatibility
            _UNIFIED_INJURY_CACHE['injury_status_encoded'] = _UNIFIED_INJURY_CACHE['report_status'].map({
                'Out': 3, 'Doubtful': 2, 'Questionable': 1, 'Probable': 0
            }).fillna(0).astype(int)
            _UNIFIED_INJURY_CACHE['practice_status_encoded'] = _UNIFIED_INJURY_CACHE['practice_status'].map({
                'Did Not Participate In Practice': 2, 'DNP': 2,
                'Limited Participation in Practice': 1, 'Limited': 1,
                'Full Participation in Practice': 0, 'Full': 0
            }).fillna(0).astype(int)
            _UNIFIED_INJURY_CACHE['player_name'] = _UNIFIED_INJURY_CACHE['full_name']
            _UNIFIED_INJURY_CACHE['player_id'] = _UNIFIED_INJURY_CACHE['gsis_id']
            logger.info(f"Loaded NFLverse injuries: {len(_UNIFIED_INJURY_CACHE)} records")
            return _UNIFIED_INJURY_CACHE
        except Exception as e:
            logger.warning(f"Failed to load NFLverse injuries: {e}")

    _UNIFIED_INJURY_CACHE = pd.DataFrame()
    return _UNIFIED_INJURY_CACHE


def _add_player_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add player-level injury features from unified injury data (NFLverse + Sleeper).

    Features:
    - injury_status_encoded: 0=None/Probable, 1=Questionable, 2=Doubtful, 3=Out
    - practice_status_encoded: 0=Full, 1=Limited, 2=DNP
    - has_injury_designation: Binary flag if player has any injury status

    Uses unified injury history for full backtesting coverage:
    - NFLverse: 2024 season (week-by-week official reports)
    - Sleeper: 2025 season (consolidated snapshots)
    """
    df = df.copy()

    # Initialize defaults
    df['injury_status_encoded'] = 0
    df['practice_status_encoded'] = 0
    df['has_injury_designation'] = 0

    # Load unified injuries data
    injuries = _load_unified_injury_data()

    if len(injuries) == 0:
        logger.debug("No injury data available - using defaults")
        return df

    if 'season' not in df.columns or 'week' not in df.columns:
        logger.debug("Missing season/week columns for injury merge")
        return df

    # Prepare injury lookup
    required_cols = ['season', 'week', 'injury_status_encoded', 'practice_status_encoded']
    available_cols = [c for c in required_cols if c in injuries.columns]

    if len(available_cols) < len(required_cols):
        logger.warning(f"Missing injury columns: {set(required_cols) - set(available_cols)}")
        return df

    # Try multiple merge strategies
    n_matched = 0

    # Strategy 1: Merge by player_id (gsis_id)
    player_id_col = None
    for col in ['gsis_id', 'player_id', 'id']:
        if col in df.columns:
            player_id_col = col
            break

    if player_id_col and 'player_id' in injuries.columns:
        inj_by_id = injuries[['player_id', 'season', 'week', 'injury_status_encoded', 'practice_status_encoded']].copy()
        inj_by_id = inj_by_id.rename(columns={'player_id': player_id_col})
        inj_by_id = inj_by_id.drop_duplicates([player_id_col, 'season', 'week'])

        df = df.merge(
            inj_by_id,
            on=[player_id_col, 'season', 'week'],
            how='left',
            suffixes=('', '_inj_id')
        )

        # Apply merged values
        for col in ['injury_status_encoded', 'practice_status_encoded']:
            merge_col = f'{col}_inj_id'
            if merge_col in df.columns:
                df[col] = df[merge_col].fillna(df[col]).astype(int)
                df = df.drop(columns=[merge_col])

        n_matched = (df['injury_status_encoded'] > 0).sum()

    # Strategy 2: Merge by player_name + team (for rows not matched by ID)
    if 'player_name' in injuries.columns:
        player_name_col = None
        for col in ['player', 'player_name', 'player_norm']:
            if col in df.columns:
                player_name_col = col
                break

        team_col = None
        for col in ['team', 'recent_team', 'player_team']:
            if col in df.columns:
                team_col = col
                break

        if player_name_col and team_col:
            # Normalize names for matching
            df['_player_norm'] = df[player_name_col].str.lower().str.strip()
            injuries['_player_norm'] = injuries['player_name'].str.lower().str.strip()

            inj_by_name = injuries[['_player_norm', 'team', 'season', 'week', 'injury_status_encoded', 'practice_status_encoded']].copy()
            inj_by_name = inj_by_name.rename(columns={'team': team_col})
            inj_by_name = inj_by_name.drop_duplicates(['_player_norm', team_col, 'season', 'week'])

            df = df.merge(
                inj_by_name,
                on=['_player_norm', team_col, 'season', 'week'],
                how='left',
                suffixes=('', '_inj_name')
            )

            # Apply merged values (only if not already matched)
            for col in ['injury_status_encoded', 'practice_status_encoded']:
                merge_col = f'{col}_inj_name'
                if merge_col in df.columns:
                    # Only update if current value is 0 (not already matched)
                    mask = df[col] == 0
                    df.loc[mask, col] = df.loc[mask, merge_col].fillna(0).astype(int)
                    df = df.drop(columns=[merge_col])

            df = df.drop(columns=['_player_norm'], errors='ignore')

    # Set has_injury_designation flag
    df['has_injury_designation'] = (df['injury_status_encoded'] > 0).astype(int)

    # Log results
    n_injured = df['has_injury_designation'].sum()
    logger.info(f"Player injury features added: {n_injured}/{len(df)} players have injury designations")

    return df


def clear_injury_cache():
    """Clear unified injury data cache."""
    global _UNIFIED_INJURY_CACHE
    _UNIFIED_INJURY_CACHE = None
