"""
IR Return Detector - Automatic detection of players returning from IR to reduced roles.

This module automatically:
1. Detects players who missed consecutive weeks (IR/injury)
2. Identifies if a teammate absorbed their role during absence
3. Calculates appropriate volume reduction for the returning player

Works across all positions: WR, RB, TE
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_ir_returns(
    weekly_stats: pd.DataFrame,
    current_week: int,
    season: int,
    min_games_before_ir: int = 3,
    min_weeks_missed: int = 3,
    volume_takeover_threshold: float = 0.5
) -> List[Dict]:
    """
    Automatically detect players returning from IR who should have reduced volume.

    Args:
        weekly_stats: DataFrame with player weekly stats
        current_week: Current NFL week
        season: Current season
        min_games_before_ir: Minimum games played before IR to be considered
        min_weeks_missed: Minimum consecutive weeks missed to be considered IR
        volume_takeover_threshold: If teammate absorbed this % of volume, flag for reduction

    Returns:
        List of dicts with IR return info and suggested adjustments
    """
    ir_returns = []

    # Filter to current season
    season_stats = weekly_stats[weekly_stats['season'] == season].copy()

    if len(season_stats) == 0:
        logger.warning("No stats found for current season")
        return ir_returns

    # Get unique players with their teams and positions
    players = season_stats.groupby(['player_id', 'player_name', 'team', 'position']).agg({
        'week': ['min', 'max', 'count', lambda x: sorted(x.unique().tolist())]
    }).reset_index()
    players.columns = ['player_id', 'player_name', 'team', 'position',
                       'first_week', 'last_week', 'games_played', 'weeks_played']

    # Find players with gaps in their schedule (potential IR)
    for _, player in players.iterrows():
        weeks_list = player['weeks_played']

        # Skip players who played recently (not returning from IR)
        if current_week - 1 in weeks_list or current_week - 2 in weeks_list:
            continue

        # Find gap in weeks (IR period)
        gap_start, gap_end = find_week_gap(weeks_list, current_week)

        if gap_start is None or gap_end is None:
            continue

        weeks_missed = gap_end - gap_start
        games_before_ir = len([w for w in weeks_list if w < gap_start])

        # Check if meets IR criteria
        if weeks_missed < min_weeks_missed or games_before_ir < min_games_before_ir:
            continue

        # Get player's stats BEFORE IR
        pre_ir_stats = season_stats[
            (season_stats['player_id'] == player['player_id']) &
            (season_stats['week'] < gap_start)
        ]

        if len(pre_ir_stats) == 0:
            continue

        # Calculate pre-IR volume based on position
        position = player['position']
        pre_ir_volume = calculate_player_volume(pre_ir_stats, position)

        if pre_ir_volume < 3:  # Skip low-volume players
            continue

        # Find teammates at same position who absorbed volume during IR
        takeover_info = detect_volume_takeover(
            season_stats,
            player['team'],
            position,
            player['player_id'],
            gap_start,
            gap_end,
            pre_ir_volume
        )

        if takeover_info['takeover_detected']:
            # Calculate suggested reduction
            reduction_rate = calculate_reduction_rate(
                pre_ir_volume,
                takeover_info['teammate_volume_increase'],
                takeover_info['teammate_current_volume'],
                weeks_missed
            )

            ir_returns.append({
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'position': position,
                'weeks_missed': weeks_missed,
                'gap_start': gap_start,
                'gap_end': gap_end,
                'pre_ir_volume': pre_ir_volume,
                'takeover_player': takeover_info['takeover_player'],
                'takeover_player_id': takeover_info['takeover_player_id'],
                'teammate_volume_before': takeover_info['teammate_volume_before'],
                'teammate_volume_after': takeover_info['teammate_volume_after'],
                'teammate_current_volume': takeover_info['teammate_current_volume'],
                'volume_reduction_rate': reduction_rate,
                'suggested_new_volume': pre_ir_volume * reduction_rate,
                'confidence': takeover_info['confidence']
            })

            logger.info(f"   Detected IR return: {player['player_name']} ({player['team']} {position})")
            logger.info(f"      Missed weeks {gap_start}-{gap_end} ({weeks_missed} weeks)")
            logger.info(f"      Pre-IR volume: {pre_ir_volume:.1f}")
            logger.info(f"      Takeover by: {takeover_info['takeover_player']} "
                       f"({takeover_info['teammate_volume_before']:.1f} → {takeover_info['teammate_current_volume']:.1f})")
            logger.info(f"      Suggested reduction: {reduction_rate:.0%}")

    return ir_returns


def find_week_gap(weeks_played: List[int], current_week: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the most recent gap in weeks played (IR period).

    Returns:
        Tuple of (gap_start_week, gap_end_week) or (None, None) if no gap
    """
    if not weeks_played:
        return None, None

    weeks_played = sorted(weeks_played)
    last_played = max(weeks_played)

    # If last played recently, no significant gap
    if current_week - last_played <= 2:
        return None, None

    # Find where the gap starts (last week played before current gap)
    gap_start = last_played + 1
    gap_end = current_week - 1  # Week before current

    return gap_start, gap_end


def calculate_player_volume(stats: pd.DataFrame, position: str) -> float:
    """
    Calculate average volume per game based on position.

    RB: carries + targets
    WR/TE: targets
    """
    if position == 'RB':
        carries = stats['carries'].mean() if 'carries' in stats.columns else 0
        targets = stats['targets'].mean() if 'targets' in stats.columns else 0
        return carries + targets
    elif position in ['WR', 'TE']:
        return stats['targets'].mean() if 'targets' in stats.columns else 0
    else:
        return 0


def detect_volume_takeover(
    season_stats: pd.DataFrame,
    team: str,
    position: str,
    excluded_player_id: str,
    gap_start: int,
    gap_end: int,
    original_volume: float
) -> Dict:
    """
    Detect if a teammate absorbed the injured player's volume.
    """
    # Get teammates at same position
    teammates = season_stats[
        (season_stats['team'] == team) &
        (season_stats['position'] == position) &
        (season_stats['player_id'] != excluded_player_id)
    ]

    if len(teammates) == 0:
        return {'takeover_detected': False}

    # Get teammate stats BEFORE and DURING the IR period
    results = []

    for teammate_id in teammates['player_id'].unique():
        teammate_stats = teammates[teammates['player_id'] == teammate_id]
        teammate_name = teammate_stats['player_name'].iloc[0]

        # Before IR
        before_ir = teammate_stats[teammate_stats['week'] < gap_start]
        # During IR
        during_ir = teammate_stats[
            (teammate_stats['week'] >= gap_start) &
            (teammate_stats['week'] <= gap_end)
        ]
        # Most recent (current role)
        recent = teammate_stats[teammate_stats['week'] >= gap_end - 2]

        if len(before_ir) < 2 or len(during_ir) < 2:
            continue

        volume_before = calculate_player_volume(before_ir, position)
        volume_during = calculate_player_volume(during_ir, position)
        volume_recent = calculate_player_volume(recent, position) if len(recent) > 0 else volume_during

        volume_increase = volume_during - volume_before
        volume_increase_pct = volume_increase / max(original_volume, 1)

        results.append({
            'player_id': teammate_id,
            'player_name': teammate_name,
            'volume_before': volume_before,
            'volume_during': volume_during,
            'volume_recent': volume_recent,
            'volume_increase': volume_increase,
            'volume_increase_pct': volume_increase_pct
        })

    if not results:
        return {'takeover_detected': False}

    # Find teammate with biggest volume increase
    results_df = pd.DataFrame(results)
    best_match = results_df.loc[results_df['volume_increase'].idxmax()]

    # Determine confidence based on volume increase
    if best_match['volume_increase_pct'] >= 0.7:
        confidence = 'HIGH'
    elif best_match['volume_increase_pct'] >= 0.4:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    # Only flag as takeover if significant increase
    takeover_detected = best_match['volume_increase_pct'] >= 0.3

    return {
        'takeover_detected': takeover_detected,
        'takeover_player': best_match['player_name'],
        'takeover_player_id': best_match['player_id'],
        'teammate_volume_before': best_match['volume_before'],
        'teammate_volume_after': best_match['volume_during'],
        'teammate_current_volume': best_match['volume_recent'],
        'teammate_volume_increase': best_match['volume_increase'],
        'confidence': confidence
    }


def calculate_reduction_rate(
    pre_ir_volume: float,
    teammate_volume_increase: float,
    teammate_current_volume: float,
    weeks_missed: int
) -> float:
    """
    Calculate suggested volume reduction rate for returning player.

    Logic:
    - If teammate fully absorbed role and maintained it, larger reduction
    - If teammate volume increase was small, smaller reduction
    - Longer IR = more established takeover = larger reduction
    """
    # Base reduction based on how much teammate absorbed
    absorption_rate = min(teammate_volume_increase / max(pre_ir_volume, 1), 1.0)

    # Weeks factor - longer IR means more established replacement
    weeks_factor = min(weeks_missed / 8, 1.0)  # Max at 8 weeks

    # Calculate reduction (1.0 = no reduction, 0.5 = 50% reduction)
    # If teammate absorbed 80% of volume for 6+ weeks, returning player gets ~40% of old volume
    base_reduction = 1.0 - (absorption_rate * 0.6)  # Max 60% reduction from absorption
    weeks_adjustment = weeks_factor * 0.2  # Up to additional 20% reduction for long IR

    reduction_rate = max(0.3, base_reduction - weeks_adjustment)  # Floor at 30% of original

    return reduction_rate


def apply_ir_return_adjustments(
    df: pd.DataFrame,
    ir_returns: List[Dict],
    position_col: str = 'position'
) -> pd.DataFrame:
    """
    Apply automatic IR return adjustments to predictions DataFrame.

    Args:
        df: Predictions DataFrame
        ir_returns: List of IR return dicts from detect_ir_returns()

    Returns:
        DataFrame with adjusted predictions
    """
    if not ir_returns:
        logger.info("   No IR returns detected requiring adjustment")
        return df

    # Add column to track adjustments
    if 'ir_return_adjusted' not in df.columns:
        df['ir_return_adjusted'] = False

    for ir_info in ir_returns:
        player_name = ir_info['player_name'].lower()

        # Find player in DataFrame
        mask = df['player_name'].str.lower() == player_name
        if not mask.any():
            logger.warning(f"   IR return player {ir_info['player_name']} not found in predictions")
            continue

        player_idx = df[mask].index[0]
        position = ir_info['position']
        reduction_rate = ir_info['volume_reduction_rate']

        # Apply reductions based on position
        if position == 'RB':
            # Reduce rushing predictions
            if 'rushing_attempts_mean' in df.columns:
                old_attempts = df.loc[player_idx, 'rushing_attempts_mean']
                df.loc[player_idx, 'rushing_attempts_mean'] *= reduction_rate
                new_attempts = df.loc[player_idx, 'rushing_attempts_mean']

            if 'rushing_yards_mean' in df.columns:
                old_yards = df.loc[player_idx, 'rushing_yards_mean']
                df.loc[player_idx, 'rushing_yards_mean'] *= reduction_rate
                new_yards = df.loc[player_idx, 'rushing_yards_mean']

            # Also reduce receiving for RBs
            if 'targets_mean' in df.columns:
                df.loc[player_idx, 'targets_mean'] *= reduction_rate
            if 'receptions_mean' in df.columns:
                df.loc[player_idx, 'receptions_mean'] *= reduction_rate
            if 'receiving_yards_mean' in df.columns:
                df.loc[player_idx, 'receiving_yards_mean'] *= reduction_rate

        elif position in ['WR', 'TE']:
            # Reduce receiving predictions
            if 'targets_mean' in df.columns:
                df.loc[player_idx, 'targets_mean'] *= reduction_rate
            if 'receptions_mean' in df.columns:
                old_rec = df.loc[player_idx, 'receptions_mean']
                df.loc[player_idx, 'receptions_mean'] *= reduction_rate
                new_rec = df.loc[player_idx, 'receptions_mean']
            if 'receiving_yards_mean' in df.columns:
                old_yards = df.loc[player_idx, 'receiving_yards_mean']
                df.loc[player_idx, 'receiving_yards_mean'] *= reduction_rate
                new_yards = df.loc[player_idx, 'receiving_yards_mean']

        df.loc[player_idx, 'ir_return_adjusted'] = True

        logger.info(f"   ✅ AUTO IR RETURN: {ir_info['player_name']} ({ir_info['team']} {position})")
        logger.info(f"      Missed {ir_info['weeks_missed']} weeks, {ir_info['takeover_player']} took over")
        logger.info(f"      Volume reduction: {reduction_rate:.0%}")
        if position == 'RB' and 'rushing_attempts_mean' in df.columns:
            logger.info(f"      Rush attempts: {old_attempts:.1f} → {new_attempts:.1f}")
        elif position in ['WR', 'TE'] and 'receptions_mean' in df.columns:
            logger.info(f"      Receptions: {old_rec:.1f} → {new_rec:.1f}")

    adjusted_count = df['ir_return_adjusted'].sum()
    logger.info(f"   Applied {adjusted_count} automatic IR return adjustment(s)")

    return df


def get_ir_return_context(
    weekly_stats: pd.DataFrame,
    player_name: str,
    team: str,
    position: str,
    current_week: int,
    season: int
) -> Optional[Dict]:
    """
    Get IR return context for a specific player.

    Useful for generating explanations in recommendations.
    """
    ir_returns = detect_ir_returns(
        weekly_stats,
        current_week,
        season,
        min_games_before_ir=2,
        min_weeks_missed=2
    )

    for ir_info in ir_returns:
        if (ir_info['player_name'].lower() == player_name.lower() and
            ir_info['team'] == team):
            return ir_info

    return None


# ============================================================================
# SNAP COUNT RAMP-UP SYSTEM
# ============================================================================
# Players returning from injury often have limited snap counts ("pitch count")
# This system tracks games since return and applies graduated adjustments

# Snap share expectations by games since return
SNAP_RAMP_SCHEDULE = {
    # games_since_return: expected_snap_pct
    0: 0.40,   # First game back: ~40% snaps (pitch count)
    1: 0.60,   # Second game: ~60% snaps
    2: 0.75,   # Third game: ~75% snaps
    3: 0.85,   # Fourth game: ~85% snaps
    4: 0.95,   # Fifth game+: near full workload
}


def get_snap_ramp_factor(games_since_return: int, injury_severity: str = 'standard') -> float:
    """
    Get expected snap share multiplier based on games since IR return.

    Args:
        games_since_return: Number of games played since returning (0 = first game back)
        injury_severity: 'standard', 'severe' (slower ramp), or 'minor' (faster ramp)

    Returns:
        Float multiplier (0.0-1.0) for expected snap share
    """
    if injury_severity == 'severe':
        # Slower ramp for major injuries (ACL, broken bones, etc.)
        severe_ramp = {0: 0.30, 1: 0.45, 2: 0.60, 3: 0.75, 4: 0.85, 5: 0.95}
        return severe_ramp.get(games_since_return, 1.0)
    elif injury_severity == 'minor':
        # Faster ramp for minor injuries
        minor_ramp = {0: 0.60, 1: 0.80, 2: 0.95}
        return minor_ramp.get(games_since_return, 1.0)
    else:
        # Standard ramp
        return SNAP_RAMP_SCHEDULE.get(games_since_return, 1.0)


def detect_returning_players_snap_status(
    weekly_stats: pd.DataFrame,
    current_week: int,
    season: int,
    min_weeks_missed: int = 2
) -> List[Dict]:
    """
    Detect players returning from injury and calculate their snap ramp status.

    Returns list of dicts with:
    - player info
    - games_since_return
    - expected_snap_pct
    - volume_adjustment_factor
    """
    returning_players = []

    season_stats = weekly_stats[weekly_stats['season'] == season].copy()
    if len(season_stats) == 0:
        return returning_players

    # Group by player
    for player_id in season_stats['player_id'].unique():
        player_stats = season_stats[season_stats['player_id'] == player_id]

        if len(player_stats) == 0:
            continue

        player_name = player_stats['player_name'].iloc[0]
        team = player_stats['recent_team'].iloc[0] if 'recent_team' in player_stats.columns else player_stats['team'].iloc[0]
        position = player_stats['position'].iloc[0]

        weeks_played = sorted(player_stats['week'].unique())

        if len(weeks_played) < 2:
            continue

        # Find gaps (missed weeks)
        all_weeks = list(range(min(weeks_played), current_week))
        missed_weeks = [w for w in all_weeks if w not in weeks_played]

        if not missed_weeks:
            continue

        # Find most recent gap
        consecutive_misses = []
        current_streak = []
        for w in sorted(missed_weeks):
            if not current_streak or w == current_streak[-1] + 1:
                current_streak.append(w)
            else:
                if len(current_streak) >= min_weeks_missed:
                    consecutive_misses.append(current_streak)
                current_streak = [w]
        if len(current_streak) >= min_weeks_missed:
            consecutive_misses.append(current_streak)

        if not consecutive_misses:
            continue

        # Get most recent IR stint
        last_ir = consecutive_misses[-1]
        ir_end_week = max(last_ir)

        # How many games since return?
        games_since_return = len([w for w in weeks_played if w > ir_end_week])

        # If they haven't returned yet (current_week is right after IR)
        if games_since_return == 0 and current_week > ir_end_week:
            # First game back this week
            games_since_return = 0
        elif games_since_return > 4:
            # Fully ramped up, no adjustment needed
            continue

        snap_factor = get_snap_ramp_factor(games_since_return)

        returning_players.append({
            'player_id': player_id,
            'player_name': player_name,
            'team': team,
            'position': position,
            'weeks_missed': len(last_ir),
            'ir_end_week': ir_end_week,
            'games_since_return': games_since_return,
            'expected_snap_pct': snap_factor,
            'volume_adjustment': snap_factor,  # Use snap factor directly for volume
            'status': 'pitch_count' if games_since_return <= 1 else 'ramping_up'
        })

        logger.info(f"   Snap ramp detected: {player_name} ({team} {position})")
        logger.info(f"      Missed {len(last_ir)} weeks (ended week {ir_end_week})")
        logger.info(f"      Games since return: {games_since_return}")
        logger.info(f"      Expected snap share: {snap_factor:.0%}")

    return returning_players


def calculate_teammate_volume_redistribution(
    weekly_stats: pd.DataFrame,
    team: str,
    position: str,
    returning_players: List[Dict],
    current_week: int,
    season: int
) -> Dict[str, float]:
    """
    Calculate how volume should be redistributed when players return.

    When a WR1 returns on a pitch count, WR2/WR3 who filled in will see reduced volume.

    Returns:
        Dict of player_name -> volume_adjustment_factor
    """
    adjustments = {}

    season_stats = weekly_stats[
        (weekly_stats['season'] == season) &
        (weekly_stats['recent_team'] == team if 'recent_team' in weekly_stats.columns else weekly_stats['team'] == team) &
        (weekly_stats['position'] == position)
    ]

    if len(season_stats) == 0:
        return adjustments

    # Get all players at this position on this team
    all_players = season_stats.groupby(['player_id', 'player_name']).agg({
        'targets': 'mean' if 'targets' in season_stats.columns else lambda x: 0,
        'week': 'count'
    }).reset_index()
    all_players.columns = ['player_id', 'player_name', 'avg_targets', 'games']

    returning_names = [p['player_name'].lower() for p in returning_players if p['team'] == team]

    # Calculate total returning volume
    total_returning_volume = 0
    for rp in returning_players:
        if rp['team'] == team and rp['position'] == position:
            # Estimate their historical volume
            player_hist = season_stats[season_stats['player_id'] == rp['player_id']]
            if len(player_hist) > 0 and 'targets' in player_hist.columns:
                hist_targets = player_hist['targets'].mean()
                # They're returning at reduced snap count
                expected_return_volume = hist_targets * rp['expected_snap_pct']
                total_returning_volume += expected_return_volume

    # Players who filled in during absence will see reduced targets
    for _, player in all_players.iterrows():
        if player['player_name'].lower() in returning_names:
            continue

        # This player may have absorbed volume - reduce proportionally
        # Simple heuristic: reduce by half of what returning players are taking back
        if total_returning_volume > 0 and player['avg_targets'] > 3:
            reduction = min(0.3, total_returning_volume / (player['avg_targets'] * 3))
            adjustments[player['player_name']] = 1.0 - reduction
            logger.info(f"   Volume redistribution: {player['player_name']} -> {1.0-reduction:.0%} of recent volume")

    return adjustments


def apply_snap_ramp_adjustments(
    df: pd.DataFrame,
    returning_players: List[Dict],
    teammate_adjustments: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Apply snap count ramp-up adjustments to predictions DataFrame.

    Args:
        df: Predictions DataFrame with columns like 'receptions_mean', 'receiving_yards_mean', etc.
        returning_players: List from detect_returning_players_snap_status()
        teammate_adjustments: Optional dict from calculate_teammate_volume_redistribution()

    Returns:
        DataFrame with adjusted predictions
    """
    if 'snap_ramp_adjusted' not in df.columns:
        df['snap_ramp_adjusted'] = False
    if 'snap_ramp_factor' not in df.columns:
        df['snap_ramp_factor'] = 1.0

    # Apply adjustments to returning players
    for rp in returning_players:
        player_name = rp['player_name'].lower()
        mask = df['player_name'].str.lower() == player_name

        if not mask.any():
            continue

        idx = df[mask].index[0]
        factor = rp['volume_adjustment']

        # Adjust volume-based predictions
        for col in ['receptions_mean', 'receiving_yards_mean', 'targets_mean',
                    'rushing_yards_mean', 'rushing_attempts_mean']:
            if col in df.columns:
                df.loc[idx, col] *= factor

        df.loc[idx, 'snap_ramp_adjusted'] = True
        df.loc[idx, 'snap_ramp_factor'] = factor

        logger.info(f"   Applied snap ramp: {rp['player_name']} ({rp['team']}) -> {factor:.0%} volume")
        logger.info(f"      Status: {rp['status']}, Games back: {rp['games_since_return']}")

    # Apply teammate adjustments (players who filled in during absence)
    if teammate_adjustments:
        for player_name, factor in teammate_adjustments.items():
            mask = df['player_name'].str.lower() == player_name.lower()
            if not mask.any():
                continue

            idx = df[mask].index[0]

            for col in ['receptions_mean', 'receiving_yards_mean', 'targets_mean']:
                if col in df.columns:
                    df.loc[idx, col] *= factor

            df.loc[idx, 'snap_ramp_adjusted'] = True
            df.loc[idx, 'snap_ramp_factor'] = factor

            logger.info(f"   Applied teammate adjustment: {player_name} -> {factor:.0%} volume (WR returning)")

    return df


# ============================================================================
# MANUAL OVERRIDE SYSTEM
# ============================================================================
# For cases where automatic detection fails or we have specific intel

MANUAL_SNAP_OVERRIDES = {
    # Format: (player_name_lower, team, week, season): snap_factor
    # Add specific overrides here when we have intel from news/reports
}


def load_manual_overrides(filepath: str = None) -> Dict:
    """Load manual snap overrides from a JSON file if it exists."""
    import json
    import os

    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'data', 'manual_snap_overrides.json'
        )

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    return {}


def save_manual_override(
    player_name: str,
    team: str,
    week: int,
    season: int,
    snap_factor: float,
    reason: str = None,
    filepath: str = None
):
    """Save a manual snap override."""
    import json
    import os

    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'data', 'manual_snap_overrides.json'
        )

    overrides = load_manual_overrides(filepath)

    key = f"{player_name.lower()}_{team}_{week}_{season}"
    overrides[key] = {
        'player_name': player_name,
        'team': team,
        'week': week,
        'season': season,
        'snap_factor': snap_factor,
        'reason': reason
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(overrides, f, indent=2)

    logger.info(f"Saved manual override: {player_name} ({team}) Week {week} -> {snap_factor:.0%}")
