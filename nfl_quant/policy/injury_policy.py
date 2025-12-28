"""
Injury Policy Module

Applies injury-based restrictions to betting recommendations.

SAFETY POLICY (CRITICAL):
- Injuries can ONLY RESTRICT recommendations (block/penalize)
- Injuries must NEVER BOOST a player
- No automatic role promotion / usage redistribution from Sleeper alone
- If injury data is missing, default conservative (block OVERs)

Usage:
    from nfl_quant.policy.injury_policy import apply_injury_policy, InjuryMode

    # Apply injury restrictions to recommendations
    filtered_recs = apply_injury_policy(recs_df, injuries_df, mode=InjuryMode.CONSERVATIVE)
"""

import pandas as pd
import logging
from typing import Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InjuryMode(str, Enum):
    """Injury policy mode."""
    STRICT = "STRICT"          # If injury loader fails -> abort pipeline
    CONSERVATIVE = "CONSERVATIVE"  # If fails -> block all OVERs, tag as NO_INJURY_DATA
    OFF = "OFF"                # Ignore injuries (dev only)


@dataclass
class InjuryAction:
    """Action to take for an injury status."""
    block_over: bool = False
    block_under: bool = False
    confidence_penalty: float = 0.0
    reason: str = ""


# Injury policy rules (restrict-only, never boost)
INJURY_POLICIES = {
    "OUT": InjuryAction(
        block_over=True,
        block_under=True,
        confidence_penalty=1.0,
        reason="Player is OUT"
    ),
    "DOUBTFUL": InjuryAction(
        block_over=True,
        block_under=False,
        confidence_penalty=0.2,  # Penalty on UNDER bets
        reason="Player is DOUBTFUL"
    ),
    "QUESTIONABLE": InjuryAction(
        block_over=True,      # Block OVER for volume-based markets
        block_under=False,
        confidence_penalty=0.1,  # Small penalty on UNDER
        reason="Player is QUESTIONABLE"
    ),
    "ACTIVE": InjuryAction(
        block_over=False,
        block_under=False,
        confidence_penalty=0.0,
        reason=""
    ),
    "UNKNOWN": InjuryAction(
        block_over=False,
        block_under=False,
        confidence_penalty=0.05,  # Small conservative penalty
        reason="Injury status unknown"
    ),
}

# Volume-based markets where QUESTIONABLE should block OVER
VOLUME_MARKETS = {
    'player_receptions',
    'player_rush_attempts',
    'player_pass_attempts',
    'player_targets',
}


def apply_injury_policy(
    recs_df: pd.DataFrame,
    injuries_df: Optional[pd.DataFrame],
    mode: InjuryMode = InjuryMode.CONSERVATIVE
) -> pd.DataFrame:
    """
    Apply injury policy to recommendations.

    This function ONLY restricts recommendations - it never boosts.

    Args:
        recs_df: DataFrame with recommendations (must have 'player', 'direction', 'market')
        injuries_df: DataFrame from injury_loader.get_injuries()
        mode: InjuryMode determining fail behavior

    Returns:
        Filtered/penalized recommendations DataFrame with added columns:
        - injury_status: Player's injury status
        - injury_risk: Risk score [0,1]
        - injury_action: Action taken (BLOCK/PENALTY/NONE)
        - injury_reason: Human-readable reason

    Raises:
        ValueError: In STRICT mode, if injuries_df is None
    """
    if recs_df.empty:
        return recs_df.copy()

    # Handle missing injury data
    if injuries_df is None or injuries_df.empty:
        return _handle_missing_injury_data(recs_df, mode)

    result = recs_df.copy()

    # Initialize injury columns
    result['injury_status'] = 'ACTIVE'
    result['injury_risk'] = 0.0
    result['injury_action'] = 'NONE'
    result['injury_reason'] = ''

    # Get player column name
    player_col = 'player' if 'player' in result.columns else 'player_name'
    if player_col not in result.columns:
        logger.warning("No player column found in recommendations")
        return result

    # Match injuries to recommendations
    for idx, row in result.iterrows():
        player_name = row[player_col]
        team = row.get('team', None)
        market = row.get('market', '')
        direction = row.get('direction', '')

        # Find matching injury
        injury_match = _find_player_injury(player_name, team, injuries_df)

        if injury_match is not None:
            status = injury_match['status']
            risk = injury_match['risk_score']

            result.at[idx, 'injury_status'] = status
            result.at[idx, 'injury_risk'] = risk

            # Get policy for this status
            policy = INJURY_POLICIES.get(status, INJURY_POLICIES['UNKNOWN'])

            # Check if OVER should be blocked for QUESTIONABLE in volume markets
            block_over = policy.block_over
            if status == 'QUESTIONABLE' and market not in VOLUME_MARKETS:
                block_over = False  # Only block OVER for volume markets when QUESTIONABLE

            # Determine action
            is_over = direction.upper() == 'OVER'
            is_under = direction.upper() == 'UNDER'

            if is_over and block_over:
                result.at[idx, 'injury_action'] = 'BLOCKED'
                result.at[idx, 'injury_reason'] = f"{policy.reason} - OVER blocked"
            elif is_under and policy.block_under:
                result.at[idx, 'injury_action'] = 'BLOCKED'
                result.at[idx, 'injury_reason'] = f"{policy.reason} - UNDER blocked"
            elif policy.confidence_penalty > 0:
                result.at[idx, 'injury_action'] = 'PENALTY'
                result.at[idx, 'injury_reason'] = f"{policy.reason} (penalty: -{policy.confidence_penalty:.1%})"
            else:
                result.at[idx, 'injury_action'] = 'NONE'

    # Apply penalties to confidence (never boost, only penalize)
    if 'combined_confidence' in result.columns:
        for idx, row in result.iterrows():
            if row['injury_action'] == 'PENALTY':
                status = row['injury_status']
                policy = INJURY_POLICIES.get(status, INJURY_POLICIES['UNKNOWN'])
                # Apply penalty (reduce confidence)
                old_conf = result.at[idx, 'combined_confidence']
                new_conf = max(0, old_conf - policy.confidence_penalty)
                result.at[idx, 'combined_confidence'] = new_conf
                # Ensure we never boost
                assert new_conf <= old_conf, "SAFETY VIOLATION: Injury boosted confidence!"

    # Filter out blocked recommendations
    blocked_mask = result['injury_action'] == 'BLOCKED'
    blocked_count = blocked_mask.sum()

    if blocked_count > 0:
        blocked_players = result.loc[blocked_mask, [player_col, 'injury_status', 'direction']].values.tolist()
        logger.info(f"[INJURY POLICY] Blocked {blocked_count} recommendations:")
        for player, status, direction in blocked_players[:5]:  # Show first 5
            logger.info(f"  - {player}: {status} ({direction})")
        if blocked_count > 5:
            logger.info(f"  ... and {blocked_count - 5} more")

    # Return unblocked recommendations
    result = result[~blocked_mask].copy()

    penalty_count = (result['injury_action'] == 'PENALTY').sum()
    if penalty_count > 0:
        logger.info(f"[INJURY POLICY] Applied penalty to {penalty_count} recommendations")

    return result


def _handle_missing_injury_data(recs_df: pd.DataFrame, mode: InjuryMode) -> pd.DataFrame:
    """Handle case when injury data is not available."""

    if mode == InjuryMode.STRICT:
        raise ValueError(
            "STRICT mode: Cannot proceed without injury data. "
            "Run 'python scripts/fetch/fetch_injuries_sleeper.py' to fetch injuries."
        )

    if mode == InjuryMode.OFF:
        logger.warning("[INJURY POLICY] Mode=OFF, skipping injury checks")
        result = recs_df.copy()
        result['injury_status'] = 'UNKNOWN'
        result['injury_risk'] = 0.0
        result['injury_action'] = 'SKIPPED'
        result['injury_reason'] = 'Injury policy disabled'
        return result

    # CONSERVATIVE mode: block all OVERs, tag as NO_INJURY_DATA
    logger.warning("[INJURY POLICY] No injury data - CONSERVATIVE mode: blocking all OVERs")

    result = recs_df.copy()
    result['injury_status'] = 'UNKNOWN'
    result['injury_risk'] = 0.0
    result['injury_action'] = 'NONE'
    result['injury_reason'] = 'NO_INJURY_DATA'

    # Block OVERs
    direction_col = 'direction'
    if direction_col in result.columns:
        over_mask = result[direction_col].str.upper() == 'OVER'
        result.loc[over_mask, 'injury_action'] = 'BLOCKED'
        result.loc[over_mask, 'injury_reason'] = 'NO_INJURY_DATA - OVERs blocked by conservative policy'

        blocked_count = over_mask.sum()
        if blocked_count > 0:
            logger.warning(f"[INJURY POLICY] Blocked {blocked_count} OVER bets due to missing injury data")

        # Filter out blocked
        result = result[~over_mask].copy()

    return result


def _find_player_injury(
    player_name: str,
    team: Optional[str],
    injuries_df: pd.DataFrame
) -> Optional[pd.Series]:
    """Find injury record for a player."""
    if injuries_df.empty:
        return None

    # Normalize search name
    search_name = player_name.lower().strip()

    # Try exact match first
    matches = injuries_df[
        injuries_df['player_name'].str.lower().str.strip() == search_name
    ]

    # Try contains match if no exact match
    if len(matches) == 0:
        matches = injuries_df[
            injuries_df['player_name'].str.lower().str.contains(search_name, na=False)
        ]

    # Filter by team if provided and multiple matches
    if team and len(matches) > 1:
        team_upper = team.upper()
        team_matches = matches[matches['team'] == team_upper]
        if len(team_matches) > 0:
            matches = team_matches

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def get_injury_summary(recs_df: pd.DataFrame) -> dict:
    """Get summary of injury actions taken on recommendations."""
    if 'injury_action' not in recs_df.columns:
        return {'status': 'NOT_APPLIED'}

    action_counts = recs_df['injury_action'].value_counts().to_dict()

    return {
        'status': 'APPLIED',
        'total_recs': len(recs_df),
        'blocked': action_counts.get('BLOCKED', 0),
        'penalty': action_counts.get('PENALTY', 0),
        'none': action_counts.get('NONE', 0),
        'skipped': action_counts.get('SKIPPED', 0),
    }
