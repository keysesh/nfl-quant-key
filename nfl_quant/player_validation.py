"""
Player Prop Input Validation for Betting Safety

Defensive programming to catch data errors before they corrupt predictions.
"""

import logging
from typing import Optional

from nfl_quant.schemas import PlayerPropInput
from nfl_quant.constants import (
    RECEIVING_POSITIONS, 
    RUSHING_POSITIONS, 
    PASSING_POSITIONS,
    VALIDATION_BOUNDS,
    MIN_THRESHOLDS,
)

logger = logging.getLogger(__name__)


def validate_player_prop_input(player_input: PlayerPropInput) -> tuple[bool, Optional[str]]:
    """
    Validate player prop input for betting safety.
    
    Args:
        player_input: Player prop input to validate
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if input is valid
        - error_message: None if valid, error description if invalid
    """
    
    # RULE 1: Receiving positions must have target_share if they have targets
    if player_input.position in RECEIVING_POSITIONS:
        if player_input.trailing_target_share is None:
            # This is actually OK for new players with no history
            # Just warn if they have significant snap share
            if player_input.trailing_snap_share > 0.3:
                logger.warning(
                    f"{player_input.player_name} ({player_input.position}) has "
                    f"no target_share but {player_input.trailing_snap_share:.0%} snap share"
                )
        
        # Validate target share bounds
        if player_input.trailing_target_share is not None:
            min_bound, max_bound = VALIDATION_BOUNDS['target_share']
            if not (min_bound <= player_input.trailing_target_share <= max_bound):
                return False, (
                    f"{player_input.player_name}: target_share {player_input.trailing_target_share:.3f} "
                    f"out of bounds [{min_bound}, {max_bound}]"
                )
    
    # RULE 2: Rushing positions must have carry_share if they carry
    if player_input.position in RUSHING_POSITIONS:
        if player_input.trailing_carry_share is None:
            if player_input.trailing_snap_share > 0.3:
                logger.warning(
                    f"{player_input.player_name} ({player_input.position}) has "
                    f"no carry_share but {player_input.trailing_snap_share:.0%} snap share"
                )
    
    # RULE 3: Yards per opportunity sanity check
    min_bound, max_bound = VALIDATION_BOUNDS['yards_per_carry']
    if player_input.position in RUSHING_POSITIONS:
        # For rushing, check YPC bounds
        ypc = player_input.trailing_yards_per_opportunity
        if not (min_bound <= ypc <= max_bound):
            return False, (
                f"{player_input.player_name}: yards_per_carry {ypc:.2f} "
                f"out of bounds [{min_bound}, {max_bound}]"
            )
    
    # RULE 4: Snap share bounds
    min_bound, max_bound = VALIDATION_BOUNDS['snap_share']
    if not (min_bound <= player_input.trailing_snap_share <= max_bound):
        return False, (
            f"{player_input.player_name}: snap_share {player_input.trailing_snap_share:.3f} "
            f"out of bounds [{min_bound}, {max_bound}]"
        )
    
    # RULE 5: TD rate bounds
    min_bound, max_bound = (0, 0.15)  # Max 15% TD rate (very high)
    if not (min_bound <= player_input.trailing_td_rate <= max_bound):
        return False, (
            f"{player_input.player_name}: estimated_rate {player_input.trailing_td_rate:.3f} "
            f"out of bounds [{min_bound}, {max_bound}]"
        )
    
    return True, None


def validate_historical_stats(player_data: dict, position: str, player_name: str) -> tuple[bool, Optional[str]]:
    """
    Validate historical player stats data.
    
    Args:
        player_data: Dictionary of player stats
        position: Player position
        player_name: Player name for error messages
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    # RULE 1: Receptions can't exceed targets
    if 'receptions' in player_data and 'targets' in player_data:
        rec = player_data.get('receptions', 0)
        tgt = player_data.get('targets', 0)
        if tgt > 0 and rec > tgt:
            return False, (
                f"{player_name}: {rec} receptions > {tgt} targets (data error)"
            )
    
    # RULE 2: Target share consistency
    if 'targets' in player_data and 'target_share' in player_data and position in RECEIVING_POSITIONS:
        targets = player_data.get('targets', 0)
        target_share = player_data.get('target_share', 0)
        
        # Rough check: target_share should be proportional to targets
        # Team typically throws ~35-45 times per game
        estimated_team_targets = 40
        expected_share = targets / (estimated_team_targets * player_data.get('weeks_played', 4))
        
        # Allow some variance
        if target_share > 0 and abs(target_share - expected_share) > 0.3:
            logger.warning(
                f"{player_name}: target_share {target_share:.3f} doesn't match "
                f"targets {targets} (expected ~{expected_share:.3f})"
            )
    
    # RULE 3: Active players should have some production
    if position in ['RB', 'WR', 'TE']:
        snap_share = player_data.get('trailing_snap_share', 0)
        targets = player_data.get('targets', 0)
        carries = player_data.get('carries', 0)
        
        # If player has high snap share, they should have some touches
        if snap_share > 0.5 and targets == 0 and carries == 0:
            logger.warning(
                f"{player_name} ({position}) has {snap_share:.0%} snap share "
                "but 0 targets and 0 carries (check data)"
            )
    
    return True, None


def validate_prediction_results(predicted: float, actual: float, prop_type: str, player_name: str) -> Optional[str]:
    """
    Validate that predictions are in reasonable range compared to actual.
    
    This catches cases where model is wildly off and might indicate data issues.
    
    Args:
        predicted: Predicted value
        actual: Actual value
        prop_type: Type of prop (e.g., 'receiving_yards')
        player_name: Player name for error messages
    
    Returns:
        Error message if validation fails, None otherwise
    """
    
    # RULE 1: Prediction error bounds
    error = abs(predicted - actual)
    relative_error = error / max(actual, 1.0)  # Avoid division by zero
    
    # For yard props, allow up to 200% error (e.g., predicted 10, actual 30)
    # This is generous to allow for game-to-game variance
    if relative_error > 2.0 and actual > 20:
        return (
            f"{player_name}: {prop_type} prediction error too large: "
            f"predicted {predicted:.1f}, actual {actual:.1f} "
            f"(error: {error:.1f}, {relative_error:.0%})"
        )
    
    return None




