"""
Pre-prediction input validation to catch bad data before inference.

This module validates feature data before it's passed to models, catching
issues like NaN values, Inf values, and out-of-bounds data that could
cause silent failures or poor predictions.

Usage:
    from nfl_quant.validation.input_validation import validate_and_log

    if not validate_and_log(X, context="week 14 predictions"):
        logger.warning("Proceeding with prediction despite validation issues")
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Feature bounds for sanity checking
# Format: {feature_name: (min_valid, max_valid)}
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    # =========================================================================
    # V17 Classifier Features (Core V12 + V17 Extensions)
    # =========================================================================
    'line_vs_trailing': (-100.0, 100.0),
    'line_level': (0.0, 500.0),
    'player_under_rate': (0.0, 1.0),
    'player_bias': (-50.0, 50.0),
    'market_under_rate': (0.0, 1.0),
    'line_in_sweet_spot': (0.0, 1.0),  # V17: Now continuous (Gaussian decay)
    'LVT_x_player_tendency': (-50.0, 50.0),
    'LVT_x_player_bias': (-100.0, 100.0),
    'LVT_x_regime': (-50.0, 50.0),
    'LVT_in_sweet_spot': (-100.0, 100.0),
    'market_bias_strength': (0.0, 1.0),
    'player_market_aligned': (-0.25, 0.25),

    # V17 NEW: LVT × Defense EPA interaction
    'lvt_x_defense': (-50.0, 50.0),  # LVT * EPA, EPA in [-0.5, 0.5]

    # V17 NEW: LVT × Rest Days interaction
    'lvt_x_rest': (-100.0, 100.0),  # LVT * normalized rest

    # =========================================================================
    # Rates (must be 0-1)
    # =========================================================================
    'trailing_comp_pct': (0.0, 1.0),
    'trailing_catch_rate': (0.0, 1.0),
    'trailing_td_rate_pass': (0.0, 0.3),
    'trailing_td_rate_rush': (0.0, 0.3),
    'trailing_td_rate_receiving': (0.0, 0.3),
    'snap_share': (0.0, 1.0),
    'target_share': (0.0, 1.0),
    'air_yards_share': (0.0, 1.0),

    # =========================================================================
    # Per-play efficiency
    # =========================================================================
    'trailing_yards_per_carry': (0.0, 15.0),
    'trailing_yards_per_target': (0.0, 25.0),
    'trailing_yards_per_reception': (0.0, 30.0),
    'trailing_yards_per_attempt': (0.0, 20.0),
    'trailing_yards_per_completion': (0.0, 30.0),

    # =========================================================================
    # Ranks
    # =========================================================================
    'opp_pass_def_rank': (1.0, 32.0),
    'opp_rush_def_rank': (1.0, 32.0),

    # =========================================================================
    # EPA (centered at 0, range approx -0.5 to 0.5 for team-level)
    # =========================================================================
    'opp_pass_def_epa': (-0.5, 0.5),
    'opp_rush_def_epa': (-0.5, 0.5),
    'trailing_def_epa': (-0.5, 0.5),
    'trailing_opp_pass_def_epa': (-0.5, 0.5),
    'trailing_opp_rush_def_epa': (-0.5, 0.5),
    'trailing_receiving_epa': (-50.0, 50.0),
    'trailing_rushing_epa': (-50.0, 50.0),
    'trailing_passing_epa': (-50.0, 50.0),

    # =========================================================================
    # Position-Specific Defense (V4 - 2025-12-04)
    # =========================================================================
    'def_vs_wr_epa': (-0.5, 0.5),
    'def_vs_wr_yds_per_play': (0.0, 20.0),
    'def_vs_wr_rank': (1.0, 32.0),
    'def_vs_rb_epa': (-0.5, 0.5),
    'def_vs_rb_yds_per_play': (0.0, 15.0),
    'def_vs_rb_rank': (1.0, 32.0),
    'def_vs_te_epa': (-0.5, 0.5),
    'def_vs_te_yds_per_play': (0.0, 20.0),
    'def_vs_te_rank': (1.0, 32.0),
    'def_vs_qb_epa': (-0.5, 0.5),
    'def_vs_qb_yds_per_play': (0.0, 15.0),
    'def_vs_qb_rank': (1.0, 32.0),
    'opp_position_def_epa': (-0.5, 0.5),
    'opp_position_def_rank': (1.0, 32.0),

    # =========================================================================
    # NGS
    # =========================================================================
    'avg_separation': (0.0, 10.0),
    'avg_cushion': (0.0, 15.0),
    'avg_yac_above_expectation': (-10.0, 10.0),
    'yac_above_expectation': (-10.0, 10.0),
    'eight_box_rate': (0.0, 1.0),
    'opp_eight_box_rate': (0.0, 1.0),
    'rush_efficiency': (-10.0, 10.0),

    # =========================================================================
    # Context
    # =========================================================================
    'team_pace': (50.0, 85.0),
    'week': (1, 22),  # Including playoffs
    'team_implied_total': (10.0, 40.0),
    'game_total': (30.0, 70.0),
    'spread_line': (-30.0, 30.0),
    'days_rest': (3, 14),

    # =========================================================================
    # WOPR/RACR
    # =========================================================================
    'trailing_wopr': (0.0, 1.5),
    'trailing_racr': (0.0, 3.0),

    # =========================================================================
    # V18 Game Context Features (2025-12-05)
    # =========================================================================
    'game_pace': (50.0, 85.0),
    'vegas_total': (30.0, 70.0),
    'vegas_spread': (-30.0, 30.0),
    'implied_team_total': (10.0, 40.0),
    'adot': (0.0, 25.0),
    'pressure_rate': (0.0, 0.6),
    'opp_pressure_rate': (0.0, 0.6),

    # =========================================================================
    # V19 Rush/Receiving Features (2025-12-05)
    # =========================================================================
    'oline_health_score': (0.0, 1.0),
    'box_count_expected': (5.0, 10.0),
    'slot_snap_pct': (0.0, 1.0),
    'target_share_trailing': (0.0, 1.0),

    # =========================================================================
    # V23 Opponent Context Features (2025-12-07)
    # =========================================================================
    'opp_pass_def_vs_avg': (-100.0, 100.0),
    'opp_rush_def_vs_avg': (-100.0, 100.0),
    'opp_def_epa': (-0.5, 0.5),
    'has_opponent_context': (0, 1),

    # =========================================================================
    # V24 Position Matchup Features (2025-12-08)
    # =========================================================================
    'pos_rank': (1, 10),
    'is_starter': (0, 1),
    'is_slot_receiver': (0, 1),
    'slot_alignment_pct': (0.0, 1.0),
    'opp_position_yards_allowed_trailing': (0.0, 200.0),
    'opp_position_volume_allowed_trailing': (0.0, 20.0),
    'opp_man_coverage_rate_trailing': (0.0, 1.0),
    'slot_funnel_score': (-2.0, 2.0),
    'man_coverage_adjustment': (0.8, 1.2),
    'position_role_x_opp_yards': (0.0, 2000.0),
    'has_position_context': (0, 1),

    # =========================================================================
    # V25 Team Health Synergy Features (2025-12-11)
    # =========================================================================
    'team_synergy_multiplier': (0.7, 1.3),
    'oline_health_score_v25': (0.0, 1.0),
    'wr_corps_health': (0.0, 1.0),
    'has_synergy_bonus': (0, 1),
    'cascade_efficiency_boost': (0.8, 1.3),
    'wr_coverage_reduction': (0.0, 0.3),
    'returning_player_count': (0, 10),
    'has_synergy_context': (0, 1),
}


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.is_valid = True
        self.nan_issues: List[str] = []
        self.inf_issues: List[str] = []
        self.bound_issues: List[str] = []
        self.warnings: List[str] = []

    def add_nan_issue(self, col: str, count: int):
        self.nan_issues.append(f"{col}: {count} NaN values")
        self.is_valid = False

    def add_inf_issue(self, col: str, count: int):
        self.inf_issues.append(f"{col}: {count} Inf values")
        self.is_valid = False

    def add_bound_issue(
        self,
        col: str,
        low: float,
        high: float,
        actual_min: float,
        actual_max: float,
        count: int
    ):
        self.bound_issues.append(
            f"{col}: {count} values outside [{low}, {high}] "
            f"(actual range: [{actual_min:.2f}, {actual_max:.2f}])"
        )
        self.warnings.append(col)

    def __str__(self) -> str:
        if self.is_valid and not self.warnings:
            return "Validation PASSED"

        lines = ["Validation Issues:"]
        if self.nan_issues:
            lines.append("  NaN: " + "; ".join(self.nan_issues))
        if self.inf_issues:
            lines.append("  Inf: " + "; ".join(self.inf_issues))
        if self.bound_issues:
            lines.append("  Bounds: " + "; ".join(self.bound_issues))
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/debugging."""
        return {
            'is_valid': self.is_valid,
            'nan_issues': self.nan_issues,
            'inf_issues': self.inf_issues,
            'bound_issues': self.bound_issues,
            'warnings': self.warnings,
        }


def validate_features(
    X: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]] = None,
    fail_on_nan: bool = True,
    fail_on_bounds: bool = False
) -> ValidationResult:
    """
    Validate features before prediction.

    Args:
        X: Feature DataFrame
        bounds: Dict mapping column names to (min, max) bounds.
               If None, uses FEATURE_BOUNDS.
        fail_on_nan: If True, NaN values cause validation failure
        fail_on_bounds: If True, out-of-bounds values cause validation failure

    Returns:
        ValidationResult with details of any issues found
    """
    if bounds is None:
        bounds = FEATURE_BOUNDS

    result = ValidationResult()

    # Check for NaN
    for col in X.columns:
        nan_count = X[col].isna().sum()
        if nan_count > 0:
            if fail_on_nan:
                result.add_nan_issue(col, nan_count)
            else:
                logger.warning(f"Column {col} has {nan_count} NaN values")

    # Check for infinity
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(X[col]).sum()
        if inf_count > 0:
            result.add_inf_issue(col, inf_count)

    # Check bounds
    for col, (low, high) in bounds.items():
        if col in X.columns:
            col_data = X[col].dropna()
            if len(col_data) == 0:
                continue

            mask = (col_data < low) | (col_data > high)
            outlier_count = mask.sum()
            if outlier_count > 0:
                result.add_bound_issue(
                    col, low, high,
                    col_data.min(), col_data.max(),
                    outlier_count
                )
                if fail_on_bounds:
                    result.is_valid = False

    return result


def validate_and_log(X: pd.DataFrame, context: str = "") -> bool:
    """
    Validate features and log results.

    Args:
        X: Feature DataFrame
        context: String describing the prediction context (for logging)

    Returns:
        True if validation passed, False otherwise
    """
    result = validate_features(X)

    if not result.is_valid:
        logger.error(f"Feature validation FAILED {context}: {result}")
        return False
    elif result.warnings:
        logger.warning(f"Feature validation warnings {context}: {result}")

    return True


def create_validation_report(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create detailed validation report for debugging.

    Returns DataFrame with stats for each feature.
    """
    report_data = []

    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            row = {
                'feature': col,
                'dtype': str(X[col].dtype),
                'nan_count': X[col].isna().sum(),
                'nan_pct': X[col].isna().mean() * 100,
                'inf_count': np.isinf(X[col]).sum() if np.issubdtype(X[col].dtype, np.floating) else 0,
                'min': X[col].min(),
                'max': X[col].max(),
                'mean': X[col].mean(),
                'std': X[col].std(),
            }

            # Check bounds
            if col in FEATURE_BOUNDS:
                low, high = FEATURE_BOUNDS[col]
                row['expected_min'] = low
                row['expected_max'] = high
                col_data = X[col].dropna()
                if len(col_data) > 0:
                    row['out_of_bounds'] = ((col_data < low) | (col_data > high)).sum()
                else:
                    row['out_of_bounds'] = 0
            else:
                row['expected_min'] = None
                row['expected_max'] = None
                row['out_of_bounds'] = None

            report_data.append(row)

    return pd.DataFrame(report_data)


def clip_to_bounds(
    X: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Clip feature values to their expected bounds.

    Use with caution - this changes the data! Only use when you've
    validated that out-of-bounds values are data errors, not real extremes.

    Args:
        X: Feature DataFrame
        bounds: Dict mapping column names to (min, max) bounds

    Returns:
        DataFrame with values clipped to bounds
    """
    if bounds is None:
        bounds = FEATURE_BOUNDS

    X = X.copy()

    for col, (low, high) in bounds.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=low, upper=high)

    return X


def check_feature_drift(
    current_X: pd.DataFrame,
    baseline_stats: Dict[str, Dict[str, float]],
    threshold_std: float = 3.0
) -> Dict[str, List[str]]:
    """
    Check for feature drift between current data and baseline statistics.

    Args:
        current_X: Current feature DataFrame
        baseline_stats: Dict of {feature: {mean, std}} from training data
        threshold_std: Number of standard deviations to flag as drift

    Returns:
        Dict with 'drifted' and 'stable' feature lists
    """
    drifted = []
    stable = []

    for col in current_X.columns:
        if col not in baseline_stats:
            continue

        baseline = baseline_stats[col]
        current_mean = current_X[col].mean()

        if baseline['std'] > 0:
            z_score = abs(current_mean - baseline['mean']) / baseline['std']
            if z_score > threshold_std:
                drifted.append(f"{col} (z={z_score:.2f})")
            else:
                stable.append(col)

    return {
        'drifted': drifted,
        'stable': stable,
    }
