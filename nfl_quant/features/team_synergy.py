"""
Team Health Synergy Module - Models compound/synergistic effects of multiple players.

This module calculates the non-linear impact when multiple players return from injury
simultaneously. Key insight: Evans + Godwin together ≠ Evans alone + Godwin alone.

Synergy effects modeled:
1. Position group health scores (O-line cohesion, WR corps depth)
2. Interaction multipliers (specific player combinations)
3. Player cascade effects (how returning players affect teammates)
4. Negative synergy (new combinations without chemistry)

Usage:
    from nfl_quant.features.team_synergy import (
        calculate_team_synergy_adjustment,
        apply_synergy_adjustments,
        generate_synergy_report,
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# POSITION IMPORTANCE WEIGHTS
# =============================================================================
# Relative importance of each position to team offensive/defensive output

OFFENSIVE_POSITION_WEIGHTS = {
    'QB': 1.00,    # Foundation of all offensive production
    'LT': 0.85,    # Protects blind side, enables deep shots
    'C': 0.70,     # Line calls, run game anchor
    'LG': 0.50,    # Interior protection/run lanes
    'RG': 0.50,    # Interior protection/run lanes
    'RT': 0.60,    # Pass protection, TE side runs
    'WR': 0.70,    # Base WR weight (adjusted by depth)
    'WR1': 0.80,   # Primary target, coverage attention
    'WR2': 0.60,   # Secondary option, coverage balance
    'WR3': 0.45,   # Slot/underneath mismatch creator
    'TE': 0.55,    # Blocking + receiving versatility
    'TE1': 0.55,   # Primary TE
    'TE2': 0.30,   # Blocking TE
    'RB': 0.55,    # Base RB weight
    'RB1': 0.65,   # Primary back - run game, check downs, protection
    'RB2': 0.35,   # Change of pace back
    'FB': 0.25,    # Fullback - limited but situationally important
}

DEFENSIVE_POSITION_WEIGHTS = {
    'EDGE': 0.75,  # Pass rush
    'DT': 0.55,    # Interior pressure
    'LB': 0.50,    # Run defense + coverage
    'CB': 0.70,    # Coverage
    'CB1': 0.80,   # Shutdown corner
    'CB2': 0.60,   # Secondary corner
    'S': 0.55,     # Safety
    'FS': 0.60,    # Free safety - coverage
    'SS': 0.50,    # Strong safety - run support
}


# =============================================================================
# SYNERGY MULTIPLIER DEFINITIONS
# =============================================================================
# When specific combinations are healthy, apply these multipliers

@dataclass
class SynergyCondition:
    """Defines a synergy condition and its multiplier."""
    name: str
    multiplier: float
    description: str
    required_positions: List[str]  # All must be healthy
    min_health_pct: float = 0.80   # Minimum health % for each position
    applies_to: str = 'offense'    # 'offense', 'defense', or 'both'


SYNERGY_CONDITIONS = [
    # Offensive synergies
    SynergyCondition(
        name='full_wr_corps',
        multiplier=1.08,
        description='WR1 + WR2 both healthy - coverage cannot bracket both',
        required_positions=['WR1', 'WR2'],
        applies_to='offense'
    ),
    SynergyCondition(
        name='oline_cohesion',
        multiplier=1.12,
        description='Full O-line 5/5 starters - communication and chemistry',
        required_positions=['LT', 'LG', 'C', 'RG', 'RT'],
        min_health_pct=0.95,
        applies_to='offense'
    ),
    SynergyCondition(
        name='oline_majority',
        multiplier=1.06,
        description='O-line 4/5 starters - partial cohesion',
        required_positions=['LT', 'LG', 'C', 'RG', 'RT'],
        min_health_pct=0.75,
        applies_to='offense'
    ),
    SynergyCondition(
        name='rb_oline_combo',
        multiplier=1.10,
        description='RB1 + full O-line - run game efficiency compounds',
        required_positions=['RB1', 'LT', 'LG', 'C', 'RG', 'RT'],
        min_health_pct=0.90,
        applies_to='offense'
    ),
    SynergyCondition(
        name='te_lt_combo',
        multiplier=1.05,
        description='TE + LT healthy - play-action protection schemes',
        required_positions=['TE1', 'LT'],
        applies_to='offense'
    ),
    SynergyCondition(
        name='triple_threat_wr',
        multiplier=1.12,
        description='WR1 + WR2 + WR3 all healthy - full route tree stress',
        required_positions=['WR1', 'WR2', 'WR3'],
        applies_to='offense'
    ),
    SynergyCondition(
        name='qb_wr1_connection',
        multiplier=1.06,
        description='QB + WR1 both fully healthy - timing routes at full speed',
        required_positions=['QB', 'WR1'],
        min_health_pct=0.95,
        applies_to='offense'
    ),

    # Defensive synergies
    SynergyCondition(
        name='secondary_healthy',
        multiplier=1.08,
        description='Both starting corners healthy - can play man coverage',
        required_positions=['CB1', 'CB2'],
        applies_to='defense'
    ),
    SynergyCondition(
        name='pass_rush_duo',
        multiplier=1.10,
        description='Both edge rushers healthy - consistent pressure',
        required_positions=['EDGE', 'EDGE'],  # Need both edges
        applies_to='defense'
    ),
]


# Negative synergy (degradation factors)
DEGRADATION_CONDITIONS = [
    SynergyCondition(
        name='oline_breakdown',
        multiplier=0.85,
        description='2+ O-line starters out - communication breakdown',
        required_positions=['LT', 'LG', 'C', 'RG', 'RT'],
        min_health_pct=0.60,  # If below this, apply penalty
        applies_to='offense'
    ),
    SynergyCondition(
        name='wr_corps_depleted',
        multiplier=0.75,
        description='WR1 AND WR2 both out - coverage focuses on remaining',
        required_positions=['WR1', 'WR2'],
        min_health_pct=0.20,  # If below this (both out), apply penalty
        applies_to='offense'
    ),
    SynergyCondition(
        name='new_oline_combo',
        multiplier=0.92,
        description='New O-line combination (< 50 snaps together) - timing issues',
        required_positions=['LT', 'LG', 'C', 'RG', 'RT'],
        min_health_pct=0.0,  # Special case - checked by snap count
        applies_to='offense'
    ),
]


# =============================================================================
# PLAYER CASCADE EFFECTS
# =============================================================================
# How specific player returns affect their teammates

@dataclass
class CascadeEffect:
    """Defines how a returning player affects a specific teammate."""
    returning_player_position: str  # Position of returning player
    affected_positions: List[str]   # Positions that are affected
    effects: Dict[str, float]       # Effect name -> multiplier
    description: str
    conditions: Optional[Dict[str, Any]] = None  # Additional conditions


CASCADE_DEFINITIONS = [
    # WR1 returning helps other WRs
    CascadeEffect(
        returning_player_position='WR1',
        affected_positions=['WR2', 'WR3', 'SLOT'],
        effects={
            'coverage_reduction': 0.15,      # 15% less coverage attention
            'efficiency_boost': 1.08,        # 8% catch rate boost
            'target_share_adjust': -0.05,    # 5% target reduction (goes to WR1)
        },
        description='WR1 draws coverage, creates easier looks for WR2/WR3'
    ),
    CascadeEffect(
        returning_player_position='WR1',
        affected_positions=['QB'],
        effects={
            'deep_ball_boost': 1.20,         # 20% more deep attempts viable
            'sack_rate_reduction': 0.95,     # 5% fewer sacks (quicker release)
            'epa_boost': 1.10,               # 10% EPA improvement
        },
        description='WR1 gives QB a reliable target, faster decisions'
    ),

    # WR2 returning balances coverage
    CascadeEffect(
        returning_player_position='WR2',
        affected_positions=['WR1'],
        effects={
            'coverage_reduction': 0.10,      # 10% less bracket coverage
            'efficiency_boost': 1.05,        # 5% efficiency boost
        },
        description='WR2 prevents bracket coverage on WR1'
    ),

    # Full O-line helps everyone
    CascadeEffect(
        returning_player_position='OLINE',  # Special: any O-line return
        affected_positions=['QB', 'RB1', 'WR1', 'WR2', 'TE1'],
        effects={
            'protection_boost': 1.15,        # 15% more time in pocket
            'run_lane_boost': 1.12,          # 12% better run blocking
        },
        description='O-line cohesion improves all offensive production'
    ),

    # RB1 returning changes game script
    CascadeEffect(
        returning_player_position='RB1',
        affected_positions=['WR1', 'WR2', 'TE1'],
        effects={
            'play_action_boost': 1.08,       # 8% more effective play action
            'coverage_reduction': 0.05,      # 5% - defense respects run more
        },
        description='RB1 makes defense respect run, opens passing game'
    ),

    # TE1 returning adds dimension
    CascadeEffect(
        returning_player_position='TE1',
        affected_positions=['RB1'],
        effects={
            'run_blocking_boost': 1.10,      # 10% better run blocking support
        },
        description='TE1 adds run blocking dimension'
    ),

    # Defensive cascades
    CascadeEffect(
        returning_player_position='CB1',
        affected_positions=['CB2', 'S'],
        effects={
            'coverage_flexibility': 1.15,    # 15% more coverage scheme options
            'blitz_rate_boost': 1.10,        # 10% more blitz opportunities
        },
        description='CB1 allows defense to play more aggressive schemes'
    ),
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class UnitHealthScore:
    """Health score for a position unit."""
    unit_name: str
    health_pct: float
    starters_available: int
    starters_total: int
    key_players_healthy: List[str]
    key_players_injured: List[str]
    snaps_together: int = 0  # For O-line cohesion tracking


@dataclass
class PlayerStatus:
    """Status of a player for synergy calculations."""
    player_name: str
    player_id: str
    position: str
    position_rank: int  # 1 = starter, 2 = backup, etc.
    team: str
    game_status: str  # 'Active', 'Questionable', 'Out', 'IR'
    snap_expectation: float  # 0.0 - 1.0
    weeks_missed: int = 0
    games_since_return: int = 0
    is_returning: bool = False


@dataclass
class SynergyResult:
    """Complete synergy calculation result."""
    team: str
    team_multiplier: float
    offense_multiplier: float
    defense_multiplier: float
    unit_health_scores: Dict[str, UnitHealthScore]
    active_synergies: List[Dict[str, Any]]
    active_degradations: List[Dict[str, Any]]
    player_cascades: Dict[str, Dict[str, float]]
    analysis_timestamp: str
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    notes: List[str]


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================

def calculate_unit_health(
    player_statuses: List[PlayerStatus],
    unit_type: str,
    team: str
) -> UnitHealthScore:
    """
    Calculate health score for a position unit.

    Args:
        player_statuses: List of all players with their status
        unit_type: 'oline', 'wr', 'rb', 'te', 'secondary', 'dline'
        team: Team abbreviation

    Returns:
        UnitHealthScore with health percentage and details
    """
    # Map unit types to positions
    unit_positions = {
        'oline': ['LT', 'LG', 'C', 'RG', 'RT', 'OL', 'T', 'G'],
        'wr': ['WR'],
        'rb': ['RB', 'FB'],
        'te': ['TE'],
        'secondary': ['CB', 'S', 'FS', 'SS', 'DB'],
        'dline': ['DE', 'DT', 'NT', 'EDGE', 'DL'],
        'lb': ['LB', 'ILB', 'OLB', 'MLB'],
    }

    target_positions = unit_positions.get(unit_type, [])
    if not target_positions:
        return UnitHealthScore(
            unit_name=unit_type,
            health_pct=1.0,
            starters_available=0,
            starters_total=0,
            key_players_healthy=[],
            key_players_injured=[]
        )

    # Filter to team and unit
    unit_players = [
        p for p in player_statuses
        if p.team == team and any(pos in p.position.upper() for pos in target_positions)
    ]

    if not unit_players:
        return UnitHealthScore(
            unit_name=unit_type,
            health_pct=1.0,
            starters_available=0,
            starters_total=0,
            key_players_healthy=[],
            key_players_injured=[]
        )

    # Calculate weighted health
    total_weight = 0.0
    weighted_health = 0.0
    starters = 0
    starters_healthy = 0
    healthy_names = []
    injured_names = []

    for player in unit_players:
        # Get position weight
        pos_key = f"{player.position.upper()}{player.position_rank}" if player.position_rank <= 3 else player.position.upper()
        weight = OFFENSIVE_POSITION_WEIGHTS.get(
            pos_key,
            OFFENSIVE_POSITION_WEIGHTS.get(player.position.upper(), 0.5)
        )

        # Starters = position_rank 1
        if player.position_rank == 1:
            starters += 1
            if player.game_status not in ['Out', 'IR', 'Doubtful']:
                starters_healthy += 1

        # Calculate health contribution
        total_weight += weight
        if player.game_status in ['Active', 'Probable']:
            health_contrib = weight * player.snap_expectation
            weighted_health += health_contrib
            healthy_names.append(player.player_name)
        elif player.game_status == 'Questionable':
            health_contrib = weight * player.snap_expectation * 0.7  # 70% discount for questionable
            weighted_health += health_contrib
            healthy_names.append(f"{player.player_name} (Q)")
        else:
            injured_names.append(player.player_name)

    health_pct = weighted_health / total_weight if total_weight > 0 else 0.0

    return UnitHealthScore(
        unit_name=unit_type,
        health_pct=health_pct,
        starters_available=starters_healthy,
        starters_total=starters,
        key_players_healthy=healthy_names[:5],  # Top 5
        key_players_injured=injured_names[:5]
    )


def check_synergy_condition(
    condition: SynergyCondition,
    player_statuses: List[PlayerStatus],
    team: str
) -> Tuple[bool, float]:
    """
    Check if a synergy condition is met.

    Returns:
        Tuple of (is_met: bool, effective_multiplier: float)
    """
    # Get players at required positions
    required = set(condition.required_positions)
    found_health = {}

    for player in player_statuses:
        if player.team != team:
            continue

        # Check various position formats
        pos_variants = [
            player.position.upper(),
            f"{player.position.upper()}{player.position_rank}",
        ]

        for pos in pos_variants:
            if pos in required:
                current_health = found_health.get(pos, 0.0)
                if player.game_status in ['Active', 'Probable']:
                    found_health[pos] = max(current_health, player.snap_expectation)
                elif player.game_status == 'Questionable':
                    found_health[pos] = max(current_health, player.snap_expectation * 0.7)

    # Check if all required positions meet minimum health
    all_met = True
    avg_health = 0.0

    for pos in required:
        health = found_health.get(pos, 0.0)
        avg_health += health
        if health < condition.min_health_pct:
            all_met = False

    avg_health = avg_health / len(required) if required else 0.0

    if all_met:
        # Scale multiplier by average health (partial benefit)
        scaled_mult = 1.0 + (condition.multiplier - 1.0) * avg_health
        return True, scaled_mult

    return False, 1.0


def calculate_cascade_effects(
    returning_players: List[PlayerStatus],
    all_players: List[PlayerStatus],
    team: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculate cascade effects from returning players on their teammates.

    Returns:
        Dict of player_name -> {effect_name: multiplier}
    """
    cascades = {}

    for returning in returning_players:
        if returning.team != team or not returning.is_returning:
            continue

        # Find applicable cascade definitions
        for cascade in CASCADE_DEFINITIONS:
            # Check if returning player matches
            pos_match = (
                cascade.returning_player_position == returning.position.upper() or
                cascade.returning_player_position == f"{returning.position.upper()}{returning.position_rank}"
            )

            # Special case for OLINE
            if cascade.returning_player_position == 'OLINE':
                pos_match = returning.position.upper() in ['LT', 'LG', 'C', 'RG', 'RT', 'OL', 'T', 'G']

            if not pos_match:
                continue

            # Apply effects to affected positions
            for affected_pos in cascade.affected_positions:
                for player in all_players:
                    if player.team != team:
                        continue

                    player_pos_match = (
                        player.position.upper() == affected_pos or
                        f"{player.position.upper()}{player.position_rank}" == affected_pos
                    )

                    if player_pos_match and player.player_name != returning.player_name:
                        if player.player_name not in cascades:
                            cascades[player.player_name] = {}

                        # Scale effects by returning player's snap expectation
                        for effect_name, effect_value in cascade.effects.items():
                            scaled_effect = 1.0 + (effect_value - 1.0) * returning.snap_expectation
                            cascades[player.player_name][effect_name] = scaled_effect

    return cascades


def calculate_team_synergy_adjustment(
    player_statuses: List[PlayerStatus],
    team: str,
    returning_players: Optional[List[PlayerStatus]] = None,
    include_cascades: bool = True
) -> SynergyResult:
    """
    Calculate compound adjustment for team health status.

    This is the main entry point for synergy calculations.

    Args:
        player_statuses: List of all players with current status
        team: Team abbreviation (e.g., 'TB', 'KC')
        returning_players: Optional list of players returning from injury
        include_cascades: Whether to calculate cascade effects

    Returns:
        SynergyResult with all adjustments and analysis
    """
    if returning_players is None:
        returning_players = [p for p in player_statuses if p.is_returning]

    # Calculate unit health scores
    unit_scores = {}
    for unit in ['oline', 'wr', 'rb', 'te', 'secondary', 'dline', 'lb']:
        unit_scores[unit] = calculate_unit_health(player_statuses, unit, team)

    # Check synergy conditions
    offense_mult = 1.0
    defense_mult = 1.0
    active_synergies = []

    for condition in SYNERGY_CONDITIONS:
        is_met, effective_mult = check_synergy_condition(condition, player_statuses, team)

        if is_met:
            if condition.applies_to == 'offense':
                offense_mult *= effective_mult
            elif condition.applies_to == 'defense':
                defense_mult *= effective_mult
            else:
                offense_mult *= effective_mult
                defense_mult *= effective_mult

            active_synergies.append({
                'name': condition.name,
                'multiplier': effective_mult,
                'description': condition.description
            })

    # Check degradation conditions
    active_degradations = []

    for condition in DEGRADATION_CONDITIONS:
        is_met, _ = check_synergy_condition(condition, player_statuses, team)

        # For degradations, we apply penalty when condition is NOT met
        if not is_met and condition.name != 'new_oline_combo':
            if condition.applies_to == 'offense':
                offense_mult *= condition.multiplier
            elif condition.applies_to == 'defense':
                defense_mult *= condition.multiplier

            active_degradations.append({
                'name': condition.name,
                'multiplier': condition.multiplier,
                'description': condition.description
            })

    # Calculate cascade effects
    cascades = {}
    if include_cascades:
        cascades = calculate_cascade_effects(returning_players, player_statuses, team)

    # Calculate overall team multiplier
    team_mult = (offense_mult + defense_mult) / 2.0

    # Determine confidence based on data quality
    healthy_count = sum(1 for p in player_statuses if p.team == team and p.game_status in ['Active', 'Probable'])
    total_count = sum(1 for p in player_statuses if p.team == team)

    if total_count >= 45 and healthy_count >= 35:
        confidence = 'HIGH'
    elif total_count >= 30:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    # Generate notes
    notes = []
    if unit_scores.get('oline') and unit_scores['oline'].health_pct >= 0.95:
        notes.append(f"Full O-line healthy ({unit_scores['oline'].starters_available}/{unit_scores['oline'].starters_total} starters)")
    if unit_scores.get('wr') and unit_scores['wr'].health_pct >= 0.90:
        notes.append(f"WR corps at full strength")
    if returning_players:
        for rp in returning_players[:3]:  # Top 3
            if rp.team == team:
                notes.append(f"{rp.player_name} returning at {rp.snap_expectation:.0%} snaps")

    return SynergyResult(
        team=team,
        team_multiplier=team_mult,
        offense_multiplier=offense_mult,
        defense_multiplier=defense_mult,
        unit_health_scores=unit_scores,
        active_synergies=active_synergies,
        active_degradations=active_degradations,
        player_cascades=cascades,
        analysis_timestamp=datetime.now().isoformat(),
        confidence=confidence,
        notes=notes
    )


# =============================================================================
# INTEGRATION WITH PREDICTIONS
# =============================================================================

def apply_synergy_adjustments(
    predictions_df: pd.DataFrame,
    home_synergy: SynergyResult,
    away_synergy: SynergyResult,
    home_team: str,
    away_team: str
) -> pd.DataFrame:
    """
    Apply synergy adjustments to predictions DataFrame.

    Args:
        predictions_df: DataFrame with player predictions
        home_synergy: SynergyResult for home team
        away_synergy: SynergyResult for away team
        home_team: Home team abbreviation
        away_team: Away team abbreviation

    Returns:
        DataFrame with synergy-adjusted predictions
    """
    df = predictions_df.copy()

    # Add tracking columns
    if 'synergy_adjusted' not in df.columns:
        df['synergy_adjusted'] = False
    if 'synergy_multiplier' not in df.columns:
        df['synergy_multiplier'] = 1.0
    if 'cascade_effects' not in df.columns:
        df['cascade_effects'] = None

    # Identify team column
    team_col = 'team' if 'team' in df.columns else 'recent_team'
    if team_col not in df.columns:
        logger.warning("No team column found in predictions DataFrame")
        return df

    # Apply team-level multipliers
    for synergy, team in [(home_synergy, home_team), (away_synergy, away_team)]:
        team_mask = df[team_col] == team

        if not team_mask.any():
            continue

        # Apply to volume predictions
        volume_cols = [
            'receptions_mean', 'receiving_yards_mean', 'targets_mean',
            'rushing_yards_mean', 'rushing_attempts_mean',
            'passing_yards_mean', 'passing_attempts_mean', 'completions_mean'
        ]

        for col in volume_cols:
            if col in df.columns:
                # Apply offense multiplier to all volume stats
                df.loc[team_mask, col] *= synergy.offense_multiplier

        df.loc[team_mask, 'synergy_adjusted'] = True
        df.loc[team_mask, 'synergy_multiplier'] = synergy.offense_multiplier

        # Apply player cascade effects
        for player_name, effects in synergy.player_cascades.items():
            player_mask = team_mask & (df['player_name'].str.lower() == player_name.lower())

            if not player_mask.any():
                continue

            idx = df[player_mask].index[0]

            # Apply specific effects
            if 'efficiency_boost' in effects:
                # Boost catch rate expectation
                if 'receptions_mean' in df.columns and 'targets_mean' in df.columns:
                    targets = df.loc[idx, 'targets_mean']
                    if targets > 0:
                        df.loc[idx, 'receptions_mean'] *= effects['efficiency_boost']

            if 'coverage_reduction' in effects:
                # Coverage reduction can boost yards per reception
                if 'receiving_yards_mean' in df.columns:
                    df.loc[idx, 'receiving_yards_mean'] *= (1 + effects['coverage_reduction'] * 0.5)

            if 'deep_ball_boost' in effects:
                # QB-specific: more deep attempts
                if 'passing_yards_mean' in df.columns:
                    df.loc[idx, 'passing_yards_mean'] *= effects['deep_ball_boost']

            if 'run_lane_boost' in effects:
                if 'rushing_yards_mean' in df.columns:
                    df.loc[idx, 'rushing_yards_mean'] *= effects['run_lane_boost']

            # Store cascade info
            df.loc[idx, 'cascade_effects'] = json.dumps(effects)

    return df


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_synergy_report(synergy: SynergyResult) -> str:
    """
    Generate a formatted synergy report for display.

    Args:
        synergy: SynergyResult from calculate_team_synergy_adjustment

    Returns:
        Formatted string report
    """
    lines = []

    lines.append(f"\n{'='*60}")
    lines.append(f"TEAM HEALTH SYNERGY ANALYSIS: {synergy.team}")
    lines.append(f"{'='*60}")

    # Unit Health Scores
    lines.append("\nUnit Health Scores:")
    lines.append("-" * 40)
    for unit_name, score in synergy.unit_health_scores.items():
        if score.starters_total > 0:
            lines.append(
                f"  {unit_name.upper():12} | {score.health_pct:5.1%} | "
                f"{score.starters_available}/{score.starters_total} starters"
            )

    # Synergy Multipliers
    lines.append(f"\nSynergy Multipliers:")
    lines.append("-" * 40)
    lines.append(f"  Offense: {synergy.offense_multiplier:.2f}x")
    lines.append(f"  Defense: {synergy.defense_multiplier:.2f}x")
    lines.append(f"  Overall: {synergy.team_multiplier:.2f}x")

    # Active Synergies
    if synergy.active_synergies:
        lines.append(f"\nActive Synergy Bonuses:")
        lines.append("-" * 40)
        for syn in synergy.active_synergies:
            lines.append(f"  ✓ {syn['name']}: +{(syn['multiplier']-1)*100:.1f}%")
            lines.append(f"    {syn['description']}")

    # Active Degradations
    if synergy.active_degradations:
        lines.append(f"\nActive Degradation Penalties:")
        lines.append("-" * 40)
        for deg in synergy.active_degradations:
            lines.append(f"  ✗ {deg['name']}: {(deg['multiplier']-1)*100:.1f}%")
            lines.append(f"    {deg['description']}")

    # Player Cascades
    if synergy.player_cascades:
        lines.append(f"\nPlayer Cascade Effects:")
        lines.append("-" * 40)
        for player, effects in list(synergy.player_cascades.items())[:5]:
            effect_strs = [f"{k}: {v:.2f}x" for k, v in effects.items()]
            lines.append(f"  {player}: {', '.join(effect_strs)}")

    # Notes
    if synergy.notes:
        lines.append(f"\nNotes:")
        lines.append("-" * 40)
        for note in synergy.notes:
            lines.append(f"  • {note}")

    lines.append(f"\nConfidence: {synergy.confidence}")
    lines.append(f"Analyzed: {synergy.analysis_timestamp}")
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_player_statuses_from_injuries(
    injuries_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    depth_charts_df: Optional[pd.DataFrame] = None,
    snap_counts_df: Optional[pd.DataFrame] = None,
    week: int = None,
    season: int = 2025
) -> List[PlayerStatus]:
    """
    Load player statuses from injury reports and roster data.

    Args:
        injuries_df: DataFrame with injury reports
        rosters_df: DataFrame with current rosters
        depth_charts_df: Optional depth chart data
        snap_counts_df: Optional snap count data for ramp calculation
        week: Current week
        season: Current season

    Returns:
        List of PlayerStatus objects
    """
    statuses = []

    # Get unique players from roster
    for _, row in rosters_df.iterrows():
        player_id = row.get('player_id', row.get('gsis_id', ''))
        player_name = row.get('player_name', row.get('full_name', ''))
        team = row.get('team', row.get('recent_team', ''))
        position = row.get('position', '')

        if not player_name or not team:
            continue

        # Default status
        game_status = 'Active'
        snap_expectation = 1.0
        weeks_missed = 0
        games_since_return = 0
        is_returning = False

        # Check injury status
        if injuries_df is not None and len(injuries_df) > 0:
            injury_mask = (
                injuries_df['player_name'].str.lower() == player_name.lower()
            ) if 'player_name' in injuries_df.columns else pd.Series([False] * len(injuries_df))

            if injury_mask.any():
                injury_row = injuries_df[injury_mask].iloc[0]
                status = injury_row.get('status', injury_row.get('injury_status', 'Active'))

                if status in ['Out', 'IR', 'Reserve/Injured']:
                    game_status = 'Out'
                    snap_expectation = 0.0
                elif status in ['Doubtful']:
                    game_status = 'Doubtful'
                    snap_expectation = 0.1
                elif status in ['Questionable']:
                    game_status = 'Questionable'
                    snap_expectation = 0.7
                elif status in ['Probable', 'Day-to-day']:
                    game_status = 'Probable'
                    snap_expectation = 0.9

        # Get depth chart position
        position_rank = 1  # Default to starter
        if depth_charts_df is not None and len(depth_charts_df) > 0:
            depth_mask = depth_charts_df['player_name'].str.lower() == player_name.lower()
            if depth_mask.any():
                depth_row = depth_charts_df[depth_mask].iloc[0]
                position_rank = depth_row.get('depth_chart_position', depth_row.get('pos_rank', 1))
                if isinstance(position_rank, str):
                    try:
                        position_rank = int(position_rank)
                    except (ValueError, TypeError):
                        position_rank = 1

        statuses.append(PlayerStatus(
            player_name=player_name,
            player_id=str(player_id),
            position=position,
            position_rank=position_rank,
            team=team,
            game_status=game_status,
            snap_expectation=snap_expectation,
            weeks_missed=weeks_missed,
            games_since_return=games_since_return,
            is_returning=is_returning
        ))

    return statuses


def detect_returning_players_from_stats(
    weekly_stats: pd.DataFrame,
    player_statuses: List[PlayerStatus],
    current_week: int,
    season: int,
    min_weeks_missed: int = 2
) -> List[PlayerStatus]:
    """
    Detect which players are returning from injury based on weekly stats gaps.

    Updates the is_returning flag and snap_expectation on PlayerStatus objects.
    """
    from nfl_quant.features.ir_return_detector import detect_returning_players_snap_status

    # Use existing IR detection
    returning_info = detect_returning_players_snap_status(
        weekly_stats, current_week, season, min_weeks_missed
    )

    # Update player statuses
    for info in returning_info:
        for status in player_statuses:
            if (status.player_name.lower() == info['player_name'].lower() and
                status.team == info['team']):
                status.is_returning = True
                status.snap_expectation = info['expected_snap_pct']
                status.games_since_return = info['games_since_return']
                status.weeks_missed = info['weeks_missed']
                break

    return [s for s in player_statuses if s.is_returning]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_game_synergy(
    home_team: str,
    away_team: str,
    injuries_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    weekly_stats_df: pd.DataFrame,
    depth_charts_df: Optional[pd.DataFrame] = None,
    week: int = None,
    season: int = 2025
) -> Tuple[SynergyResult, SynergyResult]:
    """
    Calculate synergy for both teams in a game.

    Convenience function that handles all data loading and calculation.

    Returns:
        Tuple of (home_synergy, away_synergy)
    """
    # Load player statuses
    player_statuses = load_player_statuses_from_injuries(
        injuries_df, rosters_df, depth_charts_df, None, week, season
    )

    # Detect returning players
    returning_players = detect_returning_players_from_stats(
        weekly_stats_df, player_statuses, week, season
    )

    # Calculate synergy for each team
    home_synergy = calculate_team_synergy_adjustment(
        player_statuses, home_team, returning_players
    )

    away_synergy = calculate_team_synergy_adjustment(
        player_statuses, away_team, returning_players
    )

    return home_synergy, away_synergy


# =============================================================================
# BETTING IMPLICATIONS
# =============================================================================

def calculate_team_total_adjustment(
    synergy: SynergyResult,
    base_team_total: float
) -> Dict[str, float]:
    """
    Calculate adjusted team total based on synergy.

    Args:
        synergy: SynergyResult for the team
        base_team_total: Vegas implied team total

    Returns:
        Dict with raw total, adjusted total, and delta
    """
    # Offense multiplier directly affects scoring
    adjusted_total = base_team_total * synergy.offense_multiplier

    return {
        'raw_total': base_team_total,
        'adjusted_total': adjusted_total,
        'delta': adjusted_total - base_team_total,
        'multiplier': synergy.offense_multiplier
    }


def get_synergy_betting_implications(
    synergy: SynergyResult,
    base_team_total: float = None
) -> List[str]:
    """
    Get betting implications from synergy analysis.

    Returns list of actionable insights.
    """
    implications = []

    # Team total adjustment
    if base_team_total and synergy.offense_multiplier > 1.05:
        adj = calculate_team_total_adjustment(synergy, base_team_total)
        implications.append(
            f"Team total adjustment: {adj['raw_total']:.1f} → {adj['adjusted_total']:.1f} "
            f"(+{adj['delta']:.1f} points from synergy)"
        )

    # Active synergies
    for syn in synergy.active_synergies:
        if syn['multiplier'] >= 1.08:
            implications.append(
                f"Strong synergy: {syn['name']} (+{(syn['multiplier']-1)*100:.0f}% boost)"
            )

    # Player-specific implications from cascades
    for player, effects in list(synergy.player_cascades.items())[:3]:
        if 'efficiency_boost' in effects and effects['efficiency_boost'] >= 1.05:
            implications.append(
                f"{player}: Efficiency boost from returning teammate "
                f"(+{(effects['efficiency_boost']-1)*100:.0f}% catch rate)"
            )
        if 'coverage_reduction' in effects and effects['coverage_reduction'] >= 0.10:
            implications.append(
                f"{player}: Coverage reduction, easier targets "
                f"({effects['coverage_reduction']*100:.0f}% less attention)"
            )

    # Degradation warnings
    for deg in synergy.active_degradations:
        implications.append(
            f"⚠️ {deg['name']}: {(deg['multiplier']-1)*100:.0f}% penalty - {deg['description']}"
        )

    return implications


if __name__ == '__main__':
    # Example usage
    print("Team Health Synergy Module")
    print("=" * 40)
    print(f"Synergy conditions defined: {len(SYNERGY_CONDITIONS)}")
    print(f"Degradation conditions defined: {len(DEGRADATION_CONDITIONS)}")
    print(f"Cascade effects defined: {len(CASCADE_DEFINITIONS)}")
    print(f"Position weights (offense): {len(OFFENSIVE_POSITION_WEIGHTS)}")
    print(f"Position weights (defense): {len(DEFENSIVE_POSITION_WEIGHTS)}")
