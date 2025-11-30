"""
Role Change Detection and Opportunity Inheritance

Detects when players change roles (backup to starter) and adjusts predictions accordingly.
Critical for scenarios like:
- RB1 goes to IR, RB2 becomes lead back
- WR1 injured, WR2/WR3 see increased targets
- QB change affects entire offensive output
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
CONFIGS_DIR = Path(__file__).parent.parent.parent / 'configs'


class RoleChangeDetector:
    """
    Detects role changes and calculates opportunity inheritance for players.
    """

    def __init__(self, history_df: pd.DataFrame, injuries: Dict[str, Any]):
        self.history_df = history_df
        self.injuries = injuries
        self.role_overrides = load_role_overrides()
        self.detected_changes = []

    def detect_all_role_changes(self) -> List[Dict]:
        """
        Detect all role changes based on injury data and snap share patterns.

        Returns:
            List of role change dictionaries
        """
        changes = []

        # Group by team
        teams = self.history_df['team'].unique()

        for team in teams:
            team_changes = self._detect_team_role_changes(team)
            changes.extend(team_changes)

        # Add manual overrides
        for override in self.role_overrides:
            if override not in changes:
                changes.append(override)

        self.detected_changes = changes
        return changes

    def _detect_team_role_changes(self, team: str) -> List[Dict]:
        """Detect role changes for a specific team."""
        changes = []

        # Get team roster from history
        team_players = self.history_df[self.history_df['team'] == team]

        # Check for RB role changes
        rb_changes = self._detect_rb_role_change(team, team_players)
        changes.extend(rb_changes)

        # Check for WR role changes
        wr_changes = self._detect_wr_role_change(team, team_players)
        changes.extend(wr_changes)

        return changes

    def _detect_rb_role_change(self, team: str, team_players: pd.DataFrame) -> List[Dict]:
        """Detect RB role changes (backup becoming starter)."""
        changes = []

        # Find RBs on this team
        rbs = team_players[team_players['position'] == 'RB']['player_display_name'].unique()

        if len(rbs) < 2:
            return changes

        # Calculate average carries for each RB
        rb_stats = {}
        for rb in rbs:
            rb_data = team_players[team_players['player_display_name'] == rb]
            if 'carries' in rb_data.columns and len(rb_data) > 0:
                rb_stats[rb] = {
                    'avg_carries': rb_data['carries'].mean(),
                    'avg_rushing_yards': rb_data['rushing_yards'].mean() if 'rushing_yards' in rb_data.columns else 0,
                    'games': len(rb_data),
                    'recent_carries': rb_data.sort_values('week', ascending=False).head(2)['carries'].mean() if len(rb_data) >= 2 else rb_data['carries'].mean()
                }

        if len(rb_stats) < 2:
            return changes

        # Sort by average carries (RB1 = most carries)
        sorted_rbs = sorted(rb_stats.items(), key=lambda x: x[1]['avg_carries'], reverse=True)

        # Check if RB1 is injured
        rb1_name = sorted_rbs[0][0]
        rb1_injured = False

        for injury_name, injury_info in self.injuries.items():
            if rb1_name.lower() in injury_name.lower() or injury_name.lower() in rb1_name.lower():
                if injury_info.get('status') in ['Out', 'IR', 'Doubtful']:
                    rb1_injured = True
                    break

        if rb1_injured and len(sorted_rbs) >= 2:
            rb2_name = sorted_rbs[1][0]
            rb2_stats = rb_stats[rb2_name]
            rb1_stats = rb_stats[rb1_name]

            # RB2 will inherit RB1's workload
            inherited_carries = rb1_stats['avg_carries']
            efficiency = rb2_stats['avg_rushing_yards'] / rb2_stats['avg_carries'] if rb2_stats['avg_carries'] > 0 else 4.0

            change = {
                'player': rb2_name,
                'team': team,
                'position': 'RB',
                'change_type': 'BACKUP_TO_STARTER',
                'injured_player': rb1_name,
                'stat_type': 'rushing_yards',
                'inherited_volume': inherited_carries,
                'efficiency_per_touch': efficiency,
                'expected_yards': inherited_carries * efficiency,
                'confidence': 'HIGH' if rb1_stats['avg_carries'] > 12 else 'MEDIUM',
                'reason': f'{rb1_name} IR/Out - {rb2_name} becomes RB1',
                'detected_date': datetime.now().isoformat()
            }
            changes.append(change)

        return changes

    def _detect_wr_role_change(self, team: str, team_players: pd.DataFrame) -> List[Dict]:
        """Detect WR role changes (increased targets due to injury)."""
        changes = []

        # Find WRs on this team
        wrs = team_players[team_players['position'] == 'WR']['player_display_name'].unique()

        if len(wrs) < 2:
            return changes

        # Calculate average targets for each WR
        wr_stats = {}
        for wr in wrs:
            wr_data = team_players[team_players['player_display_name'] == wr]
            if 'targets' in wr_data.columns and len(wr_data) > 0:
                wr_stats[wr] = {
                    'avg_targets': wr_data['targets'].mean(),
                    'avg_rec_yards': wr_data['receiving_yards'].mean() if 'receiving_yards' in wr_data.columns else 0,
                    'avg_receptions': wr_data['receptions'].mean() if 'receptions' in wr_data.columns else 0,
                    'games': len(wr_data)
                }

        if len(wr_stats) < 2:
            return changes

        # Sort by average targets (WR1 = most targets)
        sorted_wrs = sorted(wr_stats.items(), key=lambda x: x[1]['avg_targets'], reverse=True)

        # Check if WR1 is injured
        wr1_name = sorted_wrs[0][0]
        wr1_injured = False

        for injury_name, injury_info in self.injuries.items():
            if wr1_name.lower() in injury_name.lower() or injury_name.lower() in wr1_name.lower():
                if injury_info.get('status') in ['Out', 'IR', 'Doubtful']:
                    wr1_injured = True
                    break

        if wr1_injured and len(sorted_wrs) >= 2:
            # WR2 and WR3 will absorb targets
            wr1_stats = wr_stats[wr1_name]

            for i in range(1, min(3, len(sorted_wrs))):
                beneficiary = sorted_wrs[i][0]
                beneficiary_stats = wr_stats[beneficiary]

                # Estimate target increase (50% of WR1's targets redistributed)
                inherited_targets = wr1_stats['avg_targets'] * 0.5 / (min(2, len(sorted_wrs) - 1))

                change = {
                    'player': beneficiary,
                    'team': team,
                    'position': 'WR',
                    'change_type': 'TARGET_INCREASE',
                    'injured_player': wr1_name,
                    'stat_type': 'receiving_yards',
                    'inherited_volume': inherited_targets,
                    'expected_target_increase': inherited_targets,
                    'current_targets': beneficiary_stats['avg_targets'],
                    'new_expected_targets': beneficiary_stats['avg_targets'] + inherited_targets,
                    'confidence': 'MEDIUM',
                    'reason': f'{wr1_name} Out - {beneficiary} sees increased targets',
                    'detected_date': datetime.now().isoformat()
                }
                changes.append(change)

        return changes

    def get_player_role_change(self, player_name: str) -> Optional[Dict]:
        """Get role change info for a specific player."""
        # Check manual overrides first (highest priority)
        for override in self.role_overrides:
            if override['player'].lower() == player_name.lower():
                return override

        # Check detected changes
        for change in self.detected_changes:
            if change['player'].lower() == player_name.lower():
                return change

        return None


def detect_role_changes(history_df: pd.DataFrame, injuries: Dict) -> List[Dict]:
    """
    Main function to detect all role changes.

    Args:
        history_df: Player historical stats
        injuries: Current injury dictionary

    Returns:
        List of role change dictionaries
    """
    detector = RoleChangeDetector(history_df, injuries)
    return detector.detect_all_role_changes()


def calculate_inherited_workload(
    new_starter: str,
    injured_player: str,
    stat_type: str,
    history_df: pd.DataFrame,
    blend_factor: float = 0.7
) -> Dict[str, float]:
    """
    Calculate expected workload for player inheriting a role.

    Uses a blend of:
    - Inherited volume from injured player (weighted higher)
    - New player's efficiency metrics (yards per touch, etc.)

    Args:
        new_starter: Player name who is becoming starter
        injured_player: Player who was the previous starter
        stat_type: Type of stat (rushing_yards, receiving_yards, etc.)
        history_df: Historical stats DataFrame
        blend_factor: How much weight to give inherited volume (0.7 = 70%)

    Returns:
        Dictionary with adjusted prediction metrics
    """
    # Get injured player's workload
    injured_data = history_df[
        history_df['player_display_name'].str.lower() == injured_player.lower()
    ]

    # Get new starter's efficiency
    new_starter_data = history_df[
        history_df['player_display_name'].str.lower() == new_starter.lower()
    ]

    if len(injured_data) == 0 or len(new_starter_data) == 0:
        return {'adjusted_mean': None, 'confidence': 'LOW', 'reason': 'Insufficient data'}

    if stat_type == 'rushing_yards':
        # Inherit carries, apply new player's efficiency
        inherited_carries = injured_data['carries'].mean()
        new_player_ypc = (
            new_starter_data['rushing_yards'].sum() / new_starter_data['carries'].sum()
            if new_starter_data['carries'].sum() > 0 else 4.0
        )

        # Blend: heavy weight on inherited volume, light weight on current production
        inherited_production = inherited_carries * new_player_ypc
        current_production = new_starter_data['rushing_yards'].mean()

        adjusted_mean = blend_factor * inherited_production + (1 - blend_factor) * current_production

        return {
            'adjusted_mean': adjusted_mean,
            'inherited_carries': inherited_carries,
            'yards_per_carry': new_player_ypc,
            'original_mean': current_production,
            'adjustment_factor': adjusted_mean / current_production if current_production > 0 else 1.0,
            'confidence': 'HIGH' if inherited_carries > 10 else 'MEDIUM',
            'reason': f'Inheriting {inherited_carries:.1f} carries at {new_player_ypc:.2f} YPC'
        }

    elif stat_type == 'receiving_yards':
        # Inherit targets, apply efficiency
        inherited_targets = injured_data['targets'].mean() if 'targets' in injured_data.columns else 0
        new_player_ypt = (
            new_starter_data['receiving_yards'].sum() / new_starter_data['targets'].sum()
            if 'targets' in new_starter_data.columns and new_starter_data['targets'].sum() > 0 else 8.0
        )

        inherited_production = inherited_targets * new_player_ypt
        current_production = new_starter_data['receiving_yards'].mean()

        adjusted_mean = blend_factor * inherited_production + (1 - blend_factor) * current_production

        return {
            'adjusted_mean': adjusted_mean,
            'inherited_targets': inherited_targets,
            'yards_per_target': new_player_ypt,
            'original_mean': current_production,
            'adjustment_factor': adjusted_mean / current_production if current_production > 0 else 1.0,
            'confidence': 'MEDIUM',
            'reason': f'Inheriting {inherited_targets:.1f} targets at {new_player_ypt:.2f} YPT'
        }

    elif stat_type == 'receptions':
        inherited_targets = injured_data['targets'].mean() if 'targets' in injured_data.columns else 0
        new_player_catch_rate = (
            new_starter_data['receptions'].sum() / new_starter_data['targets'].sum()
            if 'targets' in new_starter_data.columns and new_starter_data['targets'].sum() > 0 else 0.65
        )

        inherited_production = inherited_targets * new_player_catch_rate
        current_production = new_starter_data['receptions'].mean()

        adjusted_mean = blend_factor * inherited_production + (1 - blend_factor) * current_production

        return {
            'adjusted_mean': adjusted_mean,
            'inherited_targets': inherited_targets,
            'catch_rate': new_player_catch_rate,
            'original_mean': current_production,
            'adjustment_factor': adjusted_mean / current_production if current_production > 0 else 1.0,
            'confidence': 'MEDIUM',
            'reason': f'Inheriting {inherited_targets:.1f} targets at {new_player_catch_rate:.1%} catch rate'
        }

    return {'adjusted_mean': None, 'confidence': 'LOW', 'reason': 'Unknown stat type'}


def load_role_overrides() -> List[Dict]:
    """
    Load manual role change overrides from JSON file.

    This allows users to manually specify role changes that aren't
    automatically detected (e.g., from news sources).
    """
    override_file = CONFIGS_DIR / 'role_overrides.json'

    if not override_file.exists():
        # Create empty file
        with open(override_file, 'w') as f:
            json.dump([], f, indent=2)
        return []

    with open(override_file, 'r') as f:
        return json.load(f)


def save_role_override(override: Dict) -> None:
    """
    Save a manual role change override.

    Args:
        override: Dictionary with role change info
            Required keys: player, team, position, change_type, stat_type
            Optional: injured_player, inherited_volume, expected_mean, reason
    """
    overrides = load_role_overrides()

    # Add timestamp
    override['created_date'] = datetime.now().isoformat()

    # Remove duplicates (same player)
    overrides = [o for o in overrides if o['player'].lower() != override['player'].lower()]

    overrides.append(override)

    override_file = CONFIGS_DIR / 'role_overrides.json'
    with open(override_file, 'w') as f:
        json.dump(overrides, f, indent=2)

    print(f"Saved role override for {override['player']}")


def create_rj_harvey_override():
    """
    Create the specific override for RJ Harvey (example).

    This is a helper function to demonstrate manual override creation.
    """
    override = {
        'player': 'RJ Harvey',
        'team': 'DEN',
        'position': 'RB',
        'change_type': 'BACKUP_TO_STARTER',
        'injured_player': 'Javonte Williams',  # Note: This was actually JK Dobbins
        'stat_type': 'rushing_yards',
        'inherited_volume': 15.3,  # Dobbins' avg carries
        'expected_mean': 70.0,  # Based on Dobbins' production
        'confidence': 'HIGH',
        'reason': 'JK Dobbins to IR - RJ Harvey becomes RB1 for Denver',
        'apply_to_all_stats': True  # Also affects receiving
    }

    save_role_override(override)
    return override
