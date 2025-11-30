"""
Matchup extraction and QB connection data collection from play-by-play data.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from nfl_quant.schemas import (
    MatchupRecord,
    QBConnectionRecord,
    SituationalContext,
    MatchupHistory
)
from nfl_quant.config_enhanced import config

logger = logging.getLogger(__name__)

# Load feature engineering config for injury adjustments
_injury_adj = config.feature_engineering.injury_adjustments


class MatchupExtractor:
    """Extract individual matchups and QB connections from PBP data."""

    def __init__(self, pbp_df: pd.DataFrame):
        """
        Initialize with play-by-play data.

        Args:
            pbp_df: nflfastR play-by-play DataFrame
        """
        self.pbp = pbp_df

    def extract_qb_wr_connections(self, season: int) -> pd.DataFrame:
        """
        Extract QB-WR connection data from PBP.

        Args:
            season: Season year

        Returns:
            DataFrame with QB-WR connection records
        """
        logger.info(f"Extracting QB-WR connections for {season}...")

        # Filter to passing plays
        pass_plays = self.pbp[
            (self.pbp['play_type'] == 'pass') &
            (self.pbp['season'] == season) &
            (self.pbp['passer_player_id'].notna()) &
            (self.pbp['receiver_player_id'].notna())
        ].copy()

        if len(pass_plays) == 0:
            logger.warning(f"No passing plays found for season {season}")
            return pd.DataFrame()

        connections = []

        # Group by game, QB, and receiver
        for (game_id, qb_id, receiver_id), group in pass_plays.groupby(
            ['game_id', 'passer_player_id', 'receiver_player_id']
        ):
            if pd.isna(qb_id) or pd.isna(receiver_id):
                continue

            week = group['week'].iloc[0]
            qb_name = group['passer_player_name'].iloc[0] if 'passer_player_name' in group.columns else qb_id
            receiver_name = group['receiver_player_name'].iloc[0] if 'receiver_player_name' in group.columns else receiver_id

            qb_team = group['posteam'].iloc[0]
            receiver_pos = self._get_position(receiver_id) or 'WR'

            # Get QB attempts in game
            qb_game_attempts = len(
                pass_plays[
                    (pass_plays['game_id'] == game_id) &
                    (pass_plays['passer_player_id'] == qb_id)
                ]
            )

            # Get receiver stats
            wr_targets = len(group)
            wr_receptions = group['complete_pass'].sum()
            wr_yards = group['receiving_yards'].sum()
            wr_tds = group['pass_touchdown'].sum()
            wr_air_yards = group['air_yards'].sum() if 'air_yards' in group.columns else 0.0

            if wr_targets == 0:
                continue

            target_share = wr_targets / qb_game_attempts if qb_game_attempts > 0 else 0.0
            yards_per_target = wr_yards / wr_targets if wr_targets > 0 else 0.0
            catch_rate = wr_receptions / wr_targets if wr_targets > 0 else 0.0
            adot = group['air_yards'].mean() if 'air_yards' in group.columns else 0.0
            air_yards_per_target = wr_air_yards / wr_targets if wr_targets > 0 else 0.0

            # Get game script (simplified - score differential)
            game_script = self._get_game_script(group)

            connections.append({
                'qb_player_id': qb_id,
                'qb_name': qb_name,
                'qb_team': qb_team,
                'wr_player_id': receiver_id,
                'wr_name': receiver_name,
                'wr_position': receiver_pos,
                'week': week,
                'season': season,
                'game_id': game_id,
                'qb_attempts': qb_game_attempts,
                'wr_targets': wr_targets,
                'wr_target_share': target_share,
                'wr_receptions': wr_receptions,
                'wr_receiving_yards': wr_yards,
                'wr_receiving_tds': wr_tds,
                'yards_per_target': yards_per_target,
                'catch_rate': catch_rate,
                'adot': adot,
                'air_yards_per_target': air_yards_per_target,
                'qb_injury_status': 'healthy',  # Would need injury data
                'wr_injury_status': 'healthy',
                'game_script': game_script,
            })

        df = pd.DataFrame(connections)
        logger.info(f"Extracted {len(df)} QB-WR connection records")

        return df

    def extract_team_matchups(self, season: int) -> pd.DataFrame:
        """
        Extract team-level matchup data (WR vs opponent team).

        Args:
            season: Season year

        Returns:
            DataFrame with team matchup records
        """
        logger.info(f"Extracting team matchups for {season}...")

        pass_plays = self.pbp[
            (self.pbp['play_type'] == 'pass') &
            (self.pbp['season'] == season) &
            (self.pbp['receiver_player_id'].notna())
        ].copy()

        if len(pass_plays) == 0:
            return pd.DataFrame()

        matchups = []

        # Group by game, receiver, and defense team
        for (game_id, receiver_id, defteam), group in pass_plays.groupby(
            ['game_id', 'receiver_player_id', 'defteam']
        ):
            if pd.isna(receiver_id) or pd.isna(defteam):
                continue

            week = group['week'].iloc[0]
            receiver_name = group['receiver_player_name'].iloc[0] if 'receiver_player_name' in group.columns else receiver_id
            receiver_pos = self._get_position(receiver_id) or 'WR'
            offense_team = group['posteam'].iloc[0]

            qb_id = group['passer_player_id'].iloc[0] if len(group) > 0 else None
            qb_name = group['passer_player_name'].iloc[0] if 'passer_player_name' in group.columns else None

            # Get QB attempts
            qb_attempts = len(
                pass_plays[
                    (pass_plays['game_id'] == game_id) &
                    (pass_plays['posteam'] == offense_team) &
                    (pass_plays['play_type'] == 'pass')
                ]
            )

            # Receiver stats vs this defense
            targets = len(group)
            receptions = group['complete_pass'].sum()
            yards = group['receiving_yards'].sum()
            tds = group['pass_touchdown'].sum()
            air_yards = group['air_yards'].sum() if 'air_yards' in group.columns else 0.0

            if targets == 0:
                continue

            target_share = targets / qb_attempts if qb_attempts > 0 else 0.0
            yards_per_target = yards / targets if targets > 0 else 0.0
            catch_rate = receptions / targets if targets > 0 else 0.0
            air_yards_per_target = air_yards / targets if targets > 0 else 0.0

            matchups.append({
                'offense_player_id': receiver_id,
                'offense_player_name': receiver_name,
                'offense_position': receiver_pos,
                'offense_team': offense_team,
                'defense_player_id': None,  # Team-level
                'defense_player_name': None,
                'defense_position': None,
                'defense_team': defteam,
                'week': week,
                'season': season,
                'game_id': game_id,
                'targets_against': targets,
                'receptions_against': receptions,
                'yards_against': yards,
                'tds_against': tds,
                'air_yards_per_target': air_yards_per_target,
                'catch_rate': catch_rate,
                'qb_player_id': qb_id,
                'qb_name': qb_name,
                'qb_attempts': qb_attempts,
                'target_share_vs_matchup': target_share,
                'yards_per_target_vs_matchup': yards_per_target,
                'catch_rate_vs_matchup': catch_rate,
            })

        df = pd.DataFrame(matchups)
        logger.info(f"Extracted {len(df)} team matchup records")

        return df

    def _get_position(self, player_id: str) -> Optional[str]:
        """Get player position (simplified - would need roster data)."""
        # This is a placeholder - would need actual roster data
        # For now, infer from play types
        return None

    def _get_game_script(self, group: pd.DataFrame) -> float:
        """Calculate game script (score differential)."""
        # Simplified - would need actual score data
        return 0.0


class ContextBuilder:
    """Build contextual factors for predictions."""

    def __init__(self, matchup_df: Optional[pd.DataFrame] = None,
                 qb_connections_df: Optional[pd.DataFrame] = None):
        """
        Initialize with matchup and connection data.

        Args:
            matchup_df: DataFrame with matchup records
            qb_connections_df: DataFrame with QB-WR connection records
        """
        self.matchup_df = matchup_df
        self.qb_connections_df = qb_connections_df

    def build_matchup_history(
        self,
        player_name: str,
        position: str,
        current_week: int,
        seasons: List[int]
    ) -> MatchupHistory:
        """
        Build historical matchup context for a player.

        Args:
            player_name: Player name
            position: Player position
            current_week: Current week
            seasons: List of seasons to include

        Returns:
            MatchupHistory object
        """
        history = MatchupHistory(
            player_name=player_name,
            position=position,
        )

        if self.matchup_df is None or len(self.matchup_df) == 0:
            return history

        # Filter to player and historical weeks
        player_matchups = self.matchup_df[
            (self.matchup_df['offense_player_name'] == player_name) &
            (self.matchup_df['season'].isin(seasons)) &
            (self.matchup_df['week'] < current_week)
        ].copy()

        if len(player_matchups) == 0:
            return history

        # Team-level opponent history
        for opponent, group in player_matchups.groupby('defense_team'):
            history.vs_opponent_team[opponent] = {
                'avg_targets': group['targets_against'].mean(),
                'avg_yards': group['yards_against'].mean(),
                'avg_tds': group['tds_against'].mean(),
                'avg_yards_per_target': group['yards_per_target_vs_matchup'].mean(),
                'avg_target_share': group['target_share_vs_matchup'].mean(),
                'sample_size': len(group),
            }

        history.total_games = player_matchups['game_id'].nunique()

        return history

    def build_qb_connection_history(
        self,
        wr_name: str,
        qb_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build QB connection history for a WR.

        Args:
            wr_name: WR name
            qb_name: Optional specific QB name

        Returns:
            Dictionary mapping QB name to connection stats
        """
        if self.qb_connections_df is None or len(self.qb_connections_df) == 0:
            return {}

        wr_connections = self.qb_connections_df[
            self.qb_connections_df['wr_name'] == wr_name
        ].copy()

        if qb_name:
            wr_connections = wr_connections[wr_connections['qb_name'] == qb_name]

        if len(wr_connections) == 0:
            return {}

        connections = {}

        for qb, group in wr_connections.groupby('qb_name'):
            connections[qb] = {
                'target_share': group['wr_target_share'].mean(),
                'yards_per_target': group['yards_per_target'].mean(),
                'catch_rate': group['catch_rate'].mean(),
                'adot': group['adot'].mean(),
                'games_together': len(group),
                'total_targets': group['wr_targets'].sum(),
            }

        return connections

    def build_situational_adjustments(
        self,
        team: str,
        week: int,
        player_name: str,
        position: str,
        injury_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate situational multipliers based on teammate injuries.

        When key players are injured, targets/carries redistribute to remaining players.
        This affects ALL position groups comprehensively.

        Args:
            team: Team abbreviation
            week: Week number
            player_name: Player name
            position: Player position
            injury_data: Optional injury data dictionary

        Returns:
            Dictionary with multipliers
        """
        multipliers = {
            'target_share_multiplier': 1.0,
            'carry_share_multiplier': 1.0,
            'snap_share_multiplier': 1.0,
            'efficiency_multiplier': 1.0,
            'opportunity_multiplier': 1.0,
        }

        if injury_data is None:
            return multipliers

        # Helper to check if status indicates player is OUT
        def is_out(status: str) -> bool:
            """Check if status indicates player is out/unavailable."""
            if not status:
                return False
            status_lower = status.lower()
            return status_lower in ['out', 'injured', 'pup', 'ir', 'doubtful', 'sus', 'nfi']

        # QB injury adjustment (for WRs/TEs/RBs) - reduced efficiency with backup
        if position in ['WR', 'TE', 'RB']:
            qb_status = injury_data.get('qb_status', 'healthy')
            if is_out(qb_status):
                # Typically see reduction in targets with backup QB
                qb_adj = _injury_adj.get('qb_out', {})
                multipliers['target_share_multiplier'] = qb_adj.get('target_share_multiplier', 0.85)
                multipliers['efficiency_multiplier'] = qb_adj.get('efficiency_multiplier', 0.90)

        # === COMPREHENSIVE TEAMMATE INJURY REDISTRIBUTION ===

        if position == 'WR':
            # Count how many WRs are out to calculate total target boost
            wr1_status = injury_data.get('top_wr_1_status', 'active')
            wr2_status = injury_data.get('top_wr_2_status', 'active')
            wr3_status = injury_data.get('top_wr_3_status', 'active')

            wr1_name = injury_data.get('top_wr_1', '')
            wr2_name = injury_data.get('top_wr_2', '')
            wr3_name = injury_data.get('top_wr_3', '')

            wr1_out = is_out(wr1_status)
            wr2_out = is_out(wr2_status)
            wr3_out = is_out(wr3_status)

            # Don't boost if THIS player is the one who is out
            if player_name == wr1_name and wr1_out:
                multipliers['target_share_multiplier'] = 0.0
                multipliers['opportunity_multiplier'] = 0.0
            elif player_name == wr2_name and wr2_out:
                multipliers['target_share_multiplier'] = 0.0
                multipliers['opportunity_multiplier'] = 0.0
            elif player_name == wr3_name and wr3_out:
                multipliers['target_share_multiplier'] = 0.0
                multipliers['opportunity_multiplier'] = 0.0
            else:
                # Boost targets for remaining WRs when teammates are out
                boost = 1.0

                if wr1_out and player_name != wr1_name:
                    # WR1 out: Major boost (15% more targets from config)
                    wr_adj = _injury_adj['wr1_out']
                    boost *= wr_adj.get('other_wr_target_share_boost', 1.15)  # Default 1.15x from config

                if wr2_out and player_name != wr2_name:
                    # WR2 out: Moderate boost (10-15% more targets)
                    boost *= 1.12

                if wr3_out and player_name != wr3_name:
                    # WR3 out: Small boost (5-8% more targets)
                    boost *= 1.06

                # Also check if TE1 is out (WRs may get TE's targets)
                te1_status = injury_data.get('top_te_status', 'active')
                if is_out(te1_status):
                    boost *= 1.08  # 8% boost from TE targets

                multipliers['target_share_multiplier'] *= boost
                multipliers['opportunity_multiplier'] *= boost

        elif position == 'TE':
            # TE benefits when WRs are out (more targets)
            wr1_status = injury_data.get('top_wr_1_status', 'active')
            wr2_status = injury_data.get('top_wr_2_status', 'active')

            te1_status = injury_data.get('top_te_status', 'active')
            te1_name = injury_data.get('top_te', '')

            # If THIS TE is out, zero their projection
            if player_name == te1_name and is_out(te1_status):
                multipliers['target_share_multiplier'] = 0.0
                multipliers['opportunity_multiplier'] = 0.0
            else:
                boost = 1.0

                if is_out(wr1_status):
                    boost *= 1.15  # 15% boost when WR1 out
                if is_out(wr2_status):
                    boost *= 1.08  # 8% boost when WR2 out

                # If TE1 is out and this is TE2, big boost
                if is_out(te1_status) and player_name != te1_name:
                    boost *= 1.40  # 40% boost for TE2 when TE1 out

                multipliers['target_share_multiplier'] *= boost
                multipliers['opportunity_multiplier'] *= boost

        elif position == 'RB':
            # RB benefits from other RBs being out
            rb1_status = injury_data.get('top_rb_status', 'active')
            rb2_status = injury_data.get('top_rb_2_status', 'active')

            rb1_name = injury_data.get('top_rb', '')
            rb2_name = injury_data.get('top_rb_2', '')

            # If THIS RB is out, zero their projection
            if player_name == rb1_name and is_out(rb1_status):
                multipliers['carry_share_multiplier'] = 0.0
                multipliers['target_share_multiplier'] = 0.0
                multipliers['opportunity_multiplier'] = 0.0
            elif player_name == rb2_name and is_out(rb2_status):
                multipliers['carry_share_multiplier'] = 0.0
                multipliers['target_share_multiplier'] = 0.0
                multipliers['opportunity_multiplier'] = 0.0
            else:
                carry_boost = 1.0
                target_boost = 1.0

                if is_out(rb1_status) and player_name != rb1_name:
                    # RB1 out: Major boost for RB2 (20% more carries)
                    rb_adj = _injury_adj['rb1_out']
                    carry_boost *= rb_adj.get('rb2_carry_share_multiplier', 1.20)  # Default 1.20x
                    target_boost *= 1.30
                    multipliers['snap_share_multiplier'] *= rb_adj.get('rb2_snap_share_multiplier', 1.20)

                if is_out(rb2_status) and player_name != rb2_name:
                    # RB2 out: Moderate boost for RB1/RB3
                    carry_boost *= 1.20  # 20% more carries
                    target_boost *= 1.15  # 15% more targets

                # WR injuries can boost RB receiving work
                wr1_status = injury_data.get('top_wr_1_status', 'active')
                if is_out(wr1_status):
                    target_boost *= 1.10  # 10% more targets for RB

                multipliers['carry_share_multiplier'] *= carry_boost
                multipliers['target_share_multiplier'] *= target_boost
                multipliers['opportunity_multiplier'] *= (carry_boost * target_boost) ** 0.5  # Geometric mean

        # Combined opportunity multiplier (for other positions)
        if multipliers['opportunity_multiplier'] == 1.0:
            multipliers['opportunity_multiplier'] = (
                multipliers['target_share_multiplier'] *
                multipliers['carry_share_multiplier'] *
                multipliers['snap_share_multiplier']
            )

        return multipliers
