"""
Semantic defaults for all features to replace blanket fillna(0).

This module provides domain-appropriate default values for each feature type,
replacing the dangerous practice of using fillna(0) everywhere.

Usage:
    from nfl_quant.features.feature_defaults import safe_fillna, FEATURE_DEFAULTS

    # Replace: df.fillna(0)
    # With:
    df = safe_fillna(df, FEATURE_DEFAULTS)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeagueAverages2024:
    """League-wide statistical averages for 2024 NFL season."""
    # Passing
    comp_pct: float = 0.648
    yards_per_attempt: float = 7.1
    yards_per_completion: float = 11.0
    td_rate_pass: float = 0.045
    int_rate: float = 0.025
    sack_rate: float = 0.065

    # Rushing
    yards_per_carry: float = 4.3
    td_rate_rush: float = 0.032
    fumble_rate: float = 0.008

    # Receiving
    yards_per_target: float = 7.8
    yards_per_reception: float = 11.2
    catch_rate: float = 0.68
    td_rate_receiving: float = 0.055

    # NGS
    avg_separation: float = 2.8
    avg_cushion: float = 6.2
    avg_yac_above_expectation: float = 0.0
    eight_box_rate: float = 0.25
    rush_efficiency: float = 0.0
    snap_share: float = 0.0


LEAGUE_AVG = LeagueAverages2024()


# Complete feature defaults mapping
FEATURE_DEFAULTS: Dict[str, float] = {
    # =========================================================================
    # V17 Classifier Features (Core V12 + V17 Extensions)
    # =========================================================================
    'line_vs_trailing': 0.0,
    'line_level': 0.0,
    'line_in_sweet_spot': 0.0,  # V17: Now Gaussian decay instead of binary
    'player_under_rate': 0.5,
    'player_bias': 0.0,
    'market_under_rate': 0.5,
    'LVT_x_player_tendency': 0.0,
    'LVT_x_player_bias': 0.0,
    'LVT_x_regime': 0.0,
    'LVT_in_sweet_spot': 0.0,
    'market_bias_strength': 0.0,
    'player_market_aligned': 0.0,

    # V17 NEW: LVT × Defense EPA interaction
    # Hypothesis: Strong defenses amplify the LVT signal for unders
    'lvt_x_defense': 0.0,

    # V17 NEW: LVT × Rest Days interaction
    # Hypothesis: Rest deviations may modulate LVT predictive power
    'lvt_x_rest': 0.0,

    # =========================================================================
    # Trailing Stats - Passing
    # =========================================================================
    'trailing_comp_pct': LEAGUE_AVG.comp_pct,
    'trailing_yards_per_attempt': LEAGUE_AVG.yards_per_attempt,
    'trailing_yards_per_completion': LEAGUE_AVG.yards_per_completion,
    'trailing_td_rate_pass': LEAGUE_AVG.td_rate_pass,
    'trailing_int_rate': LEAGUE_AVG.int_rate,
    'trailing_pass_attempts': 0.0,
    'trailing_completions': 0.0,
    'trailing_pass_yards': 0.0,
    'trailing_passing_yards': 0.0,
    'trailing_pass_td': 0.0,
    'trailing_attempts': 0.0,

    # =========================================================================
    # Trailing Stats - Rushing
    # =========================================================================
    'trailing_yards_per_carry': LEAGUE_AVG.yards_per_carry,
    'trailing_td_rate_rush': LEAGUE_AVG.td_rate_rush,
    'trailing_carries': 0.0,
    'trailing_rush_yards': 0.0,
    'trailing_rushing_yards': 0.0,
    'trailing_rush_td': 0.0,

    # =========================================================================
    # Trailing Stats - Receiving
    # =========================================================================
    'trailing_yards_per_target': LEAGUE_AVG.yards_per_target,
    'trailing_yards_per_reception': LEAGUE_AVG.yards_per_reception,
    'trailing_catch_rate': LEAGUE_AVG.catch_rate,
    'trailing_td_rate_receiving': LEAGUE_AVG.td_rate_receiving,
    'trailing_targets': 0.0,
    'trailing_receptions': 0.0,
    'trailing_rec_yards': 0.0,
    'trailing_receiving_yards': 0.0,
    'trailing_rec_td': 0.0,

    # =========================================================================
    # Usage Features
    # =========================================================================
    'trailing_snaps': 0.0,
    'snap_share': 0.0,
    'snap_trend': 0.0,
    'trailing_wopr': 0.0,
    'trailing_racr': 1.0,  # Neutral RACR (yards = air yards)
    'target_share': 0.0,
    'air_yards_share': 0.0,

    # =========================================================================
    # Defense/Matchup Features (Team-wide)
    # =========================================================================
    'opp_pass_def_epa': 0.0,
    'opp_rush_def_epa': 0.0,
    'trailing_def_epa': 0.0,
    'opp_pass_def_rank': 16.0,  # Middle of the league
    'opp_rush_def_rank': 16.0,
    'trailing_opp_pass_def_epa': 0.0,
    'trailing_opp_rush_def_epa': 0.0,
    'completion_pct_allowed': LEAGUE_AVG.comp_pct,

    # =========================================================================
    # Position-Specific Defense (V4 - 2025-12-04)
    # How defense performs against specific position groups
    # =========================================================================
    # WR-specific defense
    'def_vs_wr_epa': 0.0,
    'def_vs_wr_yds_per_play': 8.0,  # League avg WR yards per target
    'def_vs_wr_rank': 16.0,

    # RB-specific defense (rushing + receiving)
    'def_vs_rb_epa': 0.0,
    'def_vs_rb_yds_per_play': 4.5,  # League avg RB yards per touch
    'def_vs_rb_rank': 16.0,

    # TE-specific defense
    'def_vs_te_epa': 0.0,
    'def_vs_te_yds_per_play': 7.5,  # League avg TE yards per target
    'def_vs_te_rank': 16.0,

    # QB-specific defense (pass defense overall)
    'def_vs_qb_epa': 0.0,
    'def_vs_qb_yds_per_play': 7.0,  # League avg passing yards per attempt
    'def_vs_qb_rank': 16.0,

    # Unified position-agnostic feature (uses player's actual position)
    'opp_position_def_epa': 0.0,
    'opp_position_def_rank': 16.0,

    # V17: WR1-specific defense (addresses depth chart gap)
    # Average receptions allowed to opponent WR1s (league avg ~5.5)
    'opp_wr1_receptions_allowed': 5.5,

    # =========================================================================
    # NGS Features
    # =========================================================================
    'avg_separation': LEAGUE_AVG.avg_separation,
    'avg_cushion': LEAGUE_AVG.avg_cushion,
    'avg_yac_above_expectation': 0.0,
    'yac_above_expectation': 0.0,
    'eight_box_rate': LEAGUE_AVG.eight_box_rate,
    'opp_eight_box_rate': LEAGUE_AVG.eight_box_rate,
    'rush_efficiency': 0.0,

    # =========================================================================
    # EPA Features
    # =========================================================================
    'trailing_receiving_epa': 0.0,
    'trailing_rushing_epa': 0.0,
    'trailing_passing_epa': 0.0,

    # =========================================================================
    # Context Features
    # =========================================================================
    'team_pace': 64.0,  # League average plays per game
    'week': 1,
    'team_implied_total': 22.0,  # Average team score
    'spread_line': 0.0,
    'game_total': 44.0,

    # =========================================================================
    # Injury/Status Features
    # =========================================================================
    'games_since_return': 0,
    'first_game_back': 0,
    'teammate_out_boost': 1.0,
    'teammate_injury_boost': 1.0,

    # V24 Classifier Injury Features
    'player_injury_status': 0,  # 0=healthy, 1=questionable, 2=doubtful
    'qb_injury_status': 0,  # 0=healthy, 1=questionable, 2=doubtful, 3=out/backup
    'team_wr1_out': 0,  # Binary: 1 if WR1 is out (opportunity boost for WR2/3)
    'team_rb1_out': 0,  # Binary: 1 if RB1 is out (opportunity boost for RB2)

    # V23 Opponent Context Features
    'opp_pass_yds_def_vs_avg': 0.0,  # Fixed: was opp_pass_def_vs_avg
    'opp_rush_yds_def_vs_avg': 0.0,  # Fixed: was opp_rush_def_vs_avg
    'opp_def_epa': 0.0,
    'has_opponent_context': 0,  # Flag: 1 if opponent data available
    'backup_qb_flag': 0,
    'days_rest': 7,

    # =========================================================================
    # Game Context Features (V3 - 2025-12-04)
    # =========================================================================
    'team_implied_total': 22.0,  # League avg is ~22 points per team
    'opponent_implied_total': 22.0,
    'game_total': 44.0,  # League avg over/under
    'spread': 0.0,  # Neutral spread default
    'short_rest_flag': 0,

    # =========================================================================
    # V18 Phase 1 Features (2025-12-05)
    # =========================================================================
    # Game Pace & Context
    'game_pace': 64.0,            # Expected plays = (team + opp) / 2, league avg ~64
    'vegas_total': 44.0,          # Over/under line, league avg ~44
    'vegas_spread': 0.0,          # Point spread (negative = favored)
    'implied_team_total': 22.0,   # (total + spread) / 2

    # Skill Features (NGS + Pressure)
    'adot': 8.5,                  # Average depth of target, league avg ~8.5
    'pressure_rate': 0.25,        # O-line pressure rate allowed, ~25% league avg
    'opp_pressure_rate': 0.25,    # Opponent's pass rush pressure rate generated

    # =========================================================================
    # V19 Rush/Receiving Improvement Features (2025-12-05)
    # =========================================================================
    # Rushing Context (improves player_rush_yds market)
    'oline_health_score': 1.0,    # 1.0 = full O-line healthy, weighted by position importance
    'box_count_expected': 7.0,    # Expected defenders in box (7 = neutral, 8+ = stacked)

    # Receiving Efficiency (improves player_reception_yds market)
    'slot_snap_pct': 0.3,         # % of snaps from slot, affects route types/YAC
    'target_share_trailing': 0.15, # 4-week rolling target share (WR1 ~25%, WR2 ~15%, WR3 ~8%)

    # =========================================================================
    # V24 Position Matchup Features (2025-12-08)
    # =========================================================================
    'pos_rank': 1,                # Depth chart position (1 = starter)
    'is_starter': 1,              # Binary: 1 if pos_rank == 1
    'is_slot_receiver': 0,        # Binary: 1 if slot receiver
    'slot_alignment_pct': 0.3,    # Percentage of snaps in slot alignment
    'opp_position_yards_allowed_trailing': 0.0,  # EWMA yards opponent allows
    'opp_position_volume_allowed_trailing': 0.0, # EWMA receptions/carries
    'opp_man_coverage_rate_trailing': 0.3,       # Man coverage % (~30% league avg)
    'slot_funnel_score': 0.0,     # Slot vulnerability score
    'man_coverage_adjustment': 1.0,              # Coverage type multiplier (0.93-1.07)
    'position_role_x_opp_yards': 0.0,            # Interaction feature
    'has_position_context': 0,    # Flag: 1 if position data available

    # =========================================================================
    # V25 Team Health Synergy Features (2025-12-11)
    # Models compound effects of multiple players returning simultaneously
    # Key insight: Evans + Godwin together ≠ Evans alone + Godwin alone
    # =========================================================================
    'team_synergy_multiplier': 1.0,   # Overall team health synergy (1.0 = neutral)
    'oline_health_score_v25': 1.0,    # O-line cohesion score (weighted by position)
    'wr_corps_health': 1.0,           # WR corps depth health (WR1+WR2+WR3)
    'has_synergy_bonus': 0,           # Flag: 1 if positive synergy active
    'cascade_efficiency_boost': 1.0,  # Efficiency boost from teammate returning
    'wr_coverage_reduction': 0.0,     # Coverage reduction from healthy WR corps
    'returning_player_count': 0,      # Number of key players returning
    'has_synergy_context': 0,         # Flag: 1 if synergy data available
}


def safe_fillna(
    df: pd.DataFrame,
    feature_defaults: Dict[str, float] = None,
    log_unexpected: bool = True
) -> pd.DataFrame:
    """
    Fill missing values with semantic defaults per feature.

    This replaces the dangerous pattern of .fillna(0) which can introduce
    systematic bias (e.g., filling player_under_rate with 0 instead of 0.5).

    Args:
        df: DataFrame with features
        feature_defaults: Dict mapping column names to default values.
                         If None, uses FEATURE_DEFAULTS.
        log_unexpected: If True, log warning for columns not in defaults

    Returns:
        DataFrame with NaN values filled appropriately

    Example:
        # Old (dangerous):
        X_train = df[features].fillna(0)

        # New (safe):
        X_train = safe_fillna(df[features])
    """
    if feature_defaults is None:
        feature_defaults = FEATURE_DEFAULTS

    df = df.copy()
    unexpected_nans = []
    filled_counts = {}

    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            if col in feature_defaults:
                df[col] = df[col].fillna(feature_defaults[col])
                filled_counts[col] = (nan_count, feature_defaults[col])
            else:
                unexpected_nans.append(col)
                df[col] = df[col].fillna(0)
                filled_counts[col] = (nan_count, 0)

    if unexpected_nans and log_unexpected:
        logger.warning(
            f"Unexpected NaN in columns not in FEATURE_DEFAULTS, filled with 0: {unexpected_nans}"
        )

    if filled_counts:
        logger.debug(f"Filled NaN values: {filled_counts}")

    return df


def get_position_defaults(position: str) -> Dict[str, float]:
    """
    Get position-specific default overrides.

    Args:
        position: Player position ('QB', 'RB', 'WR', 'TE')

    Returns:
        Dict with FEATURE_DEFAULTS updated with position-specific values
    """
    overrides = {
        'QB': {
            'trailing_attempts': 30.0,
            'trailing_pass_yards': 220.0,
            'trailing_passing_yards': 220.0,
            'snap_share': 1.0,
            'trailing_comp_pct': LEAGUE_AVG.comp_pct,
        },
        'RB': {
            'trailing_carries': 12.0,
            'trailing_targets': 3.0,
            'snap_share': 0.4,
            'trailing_yards_per_carry': LEAGUE_AVG.yards_per_carry,
        },
        'WR': {
            'trailing_targets': 6.0,
            'trailing_receptions': 4.0,
            'snap_share': 0.7,
            'trailing_yards_per_target': LEAGUE_AVG.yards_per_target,
        },
        'TE': {
            'trailing_targets': 4.0,
            'trailing_receptions': 3.0,
            'snap_share': 0.6,
            'trailing_yards_per_target': LEAGUE_AVG.yards_per_target,
        },
    }

    defaults = FEATURE_DEFAULTS.copy()
    if position in overrides:
        defaults.update(overrides[position])
    return defaults


def safe_fillna_by_position(
    df: pd.DataFrame,
    position_col: str = 'position'
) -> pd.DataFrame:
    """
    Fill missing values using position-specific defaults.

    Args:
        df: DataFrame with features and a position column
        position_col: Name of the column containing position info

    Returns:
        DataFrame with NaN values filled using position-appropriate defaults
    """
    if position_col not in df.columns:
        logger.warning(f"Position column '{position_col}' not found, using generic defaults")
        return safe_fillna(df)

    df = df.copy()

    for position in df[position_col].unique():
        if pd.isna(position):
            continue

        mask = df[position_col] == position
        pos_defaults = get_position_defaults(str(position).upper())

        for col in df.columns:
            if col == position_col:
                continue

            nan_mask = mask & df[col].isna()
            if nan_mask.any():
                default_val = pos_defaults.get(col, FEATURE_DEFAULTS.get(col, 0))
                df.loc[nan_mask, col] = default_val

    return df


def get_default(feature_name: str) -> float:
    """
    Get the default value for a single feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Default value for the feature (0 if not in FEATURE_DEFAULTS)
    """
    return FEATURE_DEFAULTS.get(feature_name, 0.0)


def validate_defaults_coverage(feature_list: List[str]) -> Dict[str, List[str]]:
    """
    Check which features have defaults and which don't.

    Args:
        feature_list: List of feature names to check

    Returns:
        Dict with 'covered' and 'missing' lists
    """
    covered = [f for f in feature_list if f in FEATURE_DEFAULTS]
    missing = [f for f in feature_list if f not in FEATURE_DEFAULTS]

    return {
        'covered': covered,
        'missing': missing,
        'coverage_pct': len(covered) / len(feature_list) * 100 if feature_list else 0
    }
