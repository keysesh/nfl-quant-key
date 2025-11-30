"""
Matchup-aware feature engineering for contextual player prop predictions.
"""

import logging
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from nfl_quant.schemas import PlayerPropInput, MatchupHistory
from nfl_quant.data.matchup_extractor import ContextBuilder

logger = logging.getLogger(__name__)


class MatchupAwareFeatures:
    """Add matchup-specific and contextual features to player inputs."""

    def __init__(self, context_builder: Optional[ContextBuilder] = None):
        """
        Initialize with context builder.

        Args:
            context_builder: ContextBuilder instance for historical data
        """
        self.context_builder = context_builder

    def enhance_player_input(
        self,
        player_input: PlayerPropInput,
        matchup_history: Optional[MatchupHistory] = None,
        qb_connections: Optional[Dict[str, Dict[str, Any]]] = None,
        situational_multipliers: Optional[Dict[str, float]] = None,
        qb_status: Optional[Dict[str, str]] = None
    ) -> PlayerPropInput:
        """
        Enhance player input with contextual factors.

        Args:
            player_input: Base player input
            matchup_history: Historical matchup data
            qb_connections: QB connection history
            situational_multipliers: Situational adjustment multipliers
            qb_status: QB status dictionary {'starting_qb': name, 'status': 'healthy'/'injured'}

        Returns:
            Enhanced PlayerPropInput with contextual factors
        """
        # Start with base input fields
        enhanced_dict = player_input.model_dump()

        # Add matchup context
        if matchup_history:
            enhanced_dict.update(self._add_matchup_features(
                player_input, matchup_history
            ))

        # Add QB connection context
        if qb_connections and qb_status:
            enhanced_dict.update(self._add_qb_connection_features(
                player_input, qb_connections, qb_status
            ))

        # Add situational context
        if situational_multipliers:
            enhanced_dict.update(self._add_situational_features(
                player_input, situational_multipliers
            ))

        # Create enhanced input
        return PlayerPropInput(**enhanced_dict)

    def _add_matchup_features(
        self,
        player_input: PlayerPropInput,
        matchup_history: MatchupHistory
    ) -> Dict[str, Any]:
        """Add matchup-specific features."""
        features = {}

        # Historical performance vs opponent
        opponent_history = matchup_history.vs_opponent_team.get(player_input.opponent)

        if opponent_history:
            features['historical_yards_vs_opponent'] = opponent_history.get('avg_yards', None)
            features['historical_target_share_vs_opponent'] = opponent_history.get('avg_target_share', None)

            # Calculate matchup strength score (inverse of historical performance)
            # Higher yards = easier matchup = lower strength score
            avg_yards = opponent_history.get('avg_yards', 0)
            if avg_yards > 0:
                # Normalize to 0-1 scale (higher = tougher matchup)
                # Assuming league average is ~60 yards for WR
                league_avg = 60.0
                matchup_strength = 1.0 - min(avg_yards / (league_avg * 1.5), 1.0)
                features['matchup_strength_score'] = matchup_strength
            else:
                features['matchup_strength_score'] = 0.5  # Neutral

        # Opponent CB/LB rank (simplified - would need actual CB/LB data)
        # For now, use opponent defensive EPA as proxy
        if player_input.position == 'WR' or player_input.position == 'TE':
            # Use opponent pass defense EPA (inverted: negative EPA = good defense = higher rank)
            opponent_def_epa = player_input.opponent_def_epa_vs_position
            # Convert EPA to rank (0-1 scale, higher = better CB)
            # Typical EPA range: -0.2 to +0.2
            features['opponent_cb_rank'] = max(0.0, min(1.0, (opponent_def_epa + 0.2) / 0.4))

        elif player_input.position == 'RB':
            # Use opponent rush defense EPA
            opponent_def_epa = player_input.opponent_def_epa_vs_position
            features['opponent_lb_rank'] = max(0.0, min(1.0, (opponent_def_epa + 0.2) / 0.4))

        return features

    def _add_qb_connection_features(
        self,
        player_input: PlayerPropInput,
        qb_connections: Dict[str, Dict[str, Any]],
        qb_status: Dict[str, str]
    ) -> Dict[str, Any]:
        """Add QB-dependent features."""
        features = {}

        if player_input.position not in ['WR', 'TE', 'RB']:
            return features

        starting_qb = qb_status.get('starting_qb')
        qb_status_str = qb_status.get('status', 'healthy')

        features['starting_qb_name'] = starting_qb
        features['qb_injury_status'] = qb_status_str

        # Get connection history
        if starting_qb and starting_qb in qb_connections:
            connection = qb_connections[starting_qb]
            features['target_share_with_starting_qb'] = connection.get('target_share', None)
            features['yards_per_target_with_qb'] = connection.get('yards_per_target', None)

            # Calculate QB style match score (simplified)
            # Based on historical efficiency together
            ypt = connection.get('yards_per_target', 0)
            if ypt > 0:
                # Normalize to 0-1 (higher = better match)
                features['qb_style_match_score'] = min(1.0, ypt / 12.0)  # Assuming 12.0 is elite
            else:
                features['qb_style_match_score'] = 0.5

        # Check for backup QB connection
        backup_qb = qb_status.get('backup_qb')
        if backup_qb and backup_qb in qb_connections:
            connection = qb_connections[backup_qb]
            features['target_share_with_backup_qb'] = connection.get('target_share', None)

        return features

    def _add_situational_features(
        self,
        player_input: PlayerPropInput,
        multipliers: Dict[str, float]
    ) -> Dict[str, Any]:
        """Add situational context features."""
        features = {}

        features['teammate_injury_multiplier'] = multipliers.get('target_share_multiplier', 1.0)
        features['lineup_change_multiplier'] = multipliers.get('snap_share_multiplier', 1.0)
        features['opportunity_multiplier'] = multipliers.get('opportunity_multiplier', 1.0)

        return features


def build_contextual_features(
    player_input: PlayerPropInput,
    context_builder: Optional[ContextBuilder] = None,
    matchup_history: Optional[MatchupHistory] = None,
    injury_data: Optional[Dict[str, Any]] = None
) -> PlayerPropInput:
    """
    Build contextual features for a player input.

    Args:
        player_input: Base player input
        context_builder: ContextBuilder instance
        matchup_history: Optional pre-built matchup history
        injury_data: Optional injury/lineup data

    Returns:
        Enhanced PlayerPropInput with contextual factors
    """
    # Build matchup history if not provided
    if matchup_history is None and context_builder:
        matchup_history = context_builder.build_matchup_history(
            player_name=player_input.player_name,
            position=player_input.position,
            current_week=player_input.week,
            seasons=[2025]  # Current season
        )

    # Build QB connections if WR/TE/RB
    qb_connections = {}
    if context_builder and player_input.position in ['WR', 'TE', 'RB']:
        qb_connections = context_builder.build_qb_connection_history(
            wr_name=player_input.player_name
        )

    # Build situational multipliers
    situational_multipliers = {}
    if context_builder and injury_data:
        situational_multipliers = context_builder.build_situational_adjustments(
            team=player_input.team,
            week=player_input.week,
            player_name=player_input.player_name,
            position=player_input.position,
            injury_data=injury_data
        )

    # Build QB status
    qb_status = {}
    if injury_data:
        qb_status = {
            'starting_qb': injury_data.get('starting_qb', ''),
            'backup_qb': injury_data.get('backup_qb'),
            'status': injury_data.get('qb_status', 'healthy')
        }

    # Enhance input
    feature_enhancer = MatchupAwareFeatures(context_builder)
    enhanced_input = feature_enhancer.enhance_player_input(
        player_input=player_input,
        matchup_history=matchup_history,
        qb_connections=qb_connections,
        situational_multipliers=situational_multipliers,
        qb_status=qb_status if qb_status else None
    )

    return enhanced_input
