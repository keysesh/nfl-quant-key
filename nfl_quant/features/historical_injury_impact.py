#!/usr/bin/env python3
"""
Historical Injury Impact Analysis

Analyzes actual performance changes when key players are injured,
using real historical data instead of generic multipliers.

This provides more accurate projections based on how teams actually
redistribute targets/carries when specific players are out.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_pbp_data(season: int) -> pd.DataFrame:
    """Load play-by-play data for a season."""
    pbp_path = PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{season}.parquet'
    if pbp_path.exists():
        return pd.read_parquet(pbp_path)
    return pd.DataFrame()


def load_player_stats(season: int) -> pd.DataFrame:
    """Load player weekly stats for a season."""
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    if stats_path.exists():
        df = pd.read_parquet(stats_path)
        if 'season' in df.columns:
            return df[df['season'] == season]
    return pd.DataFrame()


def get_games_when_player_out(
    player_name: str,
    team: str,
    season: int,
    injury_weeks: list = None
) -> pd.DataFrame:
    """
    Identify games when a specific player was OUT.

    Args:
        player_name: Name of injured player
        team: Team abbreviation
        season: Season year
        injury_weeks: Specific weeks when player was out (if known)

    Returns:
        DataFrame of games when player was inactive
    """
    stats_df = load_player_stats(season)
    if stats_df.empty:
        return pd.DataFrame()

    # Get all weeks the team played
    team_games = stats_df[stats_df['team'] == team]['week'].unique()

    # Get weeks when player was active (had stats)
    player_active_weeks = stats_df[
        (stats_df['player_display_name'] == player_name) &
        (stats_df['team'] == team)
    ]['week'].unique()

    # Weeks when player was out = team games - player active weeks
    out_weeks = [w for w in team_games if w not in player_active_weeks]

    if injury_weeks:
        out_weeks = [w for w in out_weeks if w in injury_weeks]

    return pd.DataFrame({'week': out_weeks, 'team': team, 'season': season})


def calculate_historical_redistribution(
    team: str,
    injured_player: str,
    injured_position: str,
    beneficiary_player: str,
    beneficiary_position: str,
    seasons: list = [2025]  # DEFAULT TO CURRENT SEASON ONLY - roster changes between seasons
) -> Dict[str, float]:
    """
    Calculate actual historical performance boost when injured_player was out.

    Returns multipliers based on real data:
    - target_share_multiplier: How much more targets beneficiary got
    - yards_per_game_multiplier: How much more yards
    - td_rate_multiplier: How TD rate changed

    Args:
        team: Team abbreviation
        injured_player: Player who was out
        injured_position: Position of injured player (WR, RB, QB)
        beneficiary_player: Player who benefits from injury
        beneficiary_position: Position of beneficiary
        seasons: Seasons to analyze

    Returns:
        Dictionary with multipliers based on historical data
    """
    result = {
        'target_share_multiplier': 1.0,
        'carry_share_multiplier': 1.0,
        'yards_per_game_multiplier': 1.0,
        'td_rate_multiplier': 1.0,
        'sample_size': 0,
        'confidence': 'low'
    }

    all_with_out = []
    all_baseline = []

    for season in seasons:
        stats_df = load_player_stats(season)
        if stats_df.empty:
            continue

        # Find weeks when injured player was out
        out_games = get_games_when_player_out(injured_player, team, season)
        if out_games.empty:
            continue

        out_weeks = out_games['week'].tolist()

        # Get beneficiary's stats when injured player was OUT
        beneficiary_with_out = stats_df[
            (stats_df['player_display_name'] == beneficiary_player) &
            (stats_df['team'] == team) &
            (stats_df['week'].isin(out_weeks))
        ]

        # Get beneficiary's baseline stats when injured player was ACTIVE
        active_weeks = stats_df[
            (stats_df['player_display_name'] == injured_player) &
            (stats_df['team'] == team)
        ]['week'].unique()

        beneficiary_baseline = stats_df[
            (stats_df['player_display_name'] == beneficiary_player) &
            (stats_df['team'] == team) &
            (stats_df['week'].isin(active_weeks))
        ]

        if len(beneficiary_with_out) > 0:
            all_with_out.append(beneficiary_with_out)
        if len(beneficiary_baseline) > 0:
            all_baseline.append(beneficiary_baseline)

    if not all_with_out or not all_baseline:
        logger.debug(f"No historical data for {beneficiary_player} when {injured_player} out")
        return result

    # Combine data
    with_out_df = pd.concat(all_with_out, ignore_index=True)
    baseline_df = pd.concat(all_baseline, ignore_index=True)

    result['sample_size'] = len(with_out_df)

    # CRITICAL: Require minimum sample sizes for reliable multipliers
    # Single-game samples are highly unreliable and should not be used
    min_baseline_games = 3  # Need at least 3 baseline games
    min_out_games = 3  # Need at least 3 "out" games

    if len(baseline_df) < min_baseline_games or len(with_out_df) < min_out_games:
        logger.debug(
            f"Insufficient sample size for {beneficiary_player} when {injured_player} out: "
            f"baseline={len(baseline_df)}, out={len(with_out_df)} "
            f"(need {min_baseline_games}/{min_out_games})"
        )
        # Return default multipliers (no adjustment) for insufficient data
        return result

    # Set confidence based on sample size
    if len(with_out_df) >= 5:
        result['confidence'] = 'high'
    elif len(with_out_df) >= 3:
        result['confidence'] = 'medium'
    else:
        result['confidence'] = 'low'

    # Calculate target share multiplier (for pass catchers)
    if beneficiary_position in ['WR', 'TE', 'RB']:
        if 'targets' in with_out_df.columns and 'targets' in baseline_df.columns:
            # Calculate actual TARGET SHARE change, not just target count
            # This is more accurate as it accounts for changes in team passing volume
            try:
                # Get team total targets for each week
                stats_combined = pd.concat([load_player_stats(s) for s in seasons if not load_player_stats(s).empty], ignore_index=True)

                # Weeks when injured player was out
                out_weeks_all = []
                for season in seasons:
                    out_games = get_games_when_player_out(injured_player, team, season)
                    if not out_games.empty:
                        out_weeks_all.extend([(season, w) for w in out_games['week'].tolist()])

                # Weeks when injured player was active
                active_weeks_all = []
                for season in seasons:
                    season_stats = stats_combined[stats_combined['season'] == season]
                    active = season_stats[
                        (season_stats['player_display_name'] == injured_player) &
                        (season_stats['team'] == team)
                    ]['week'].unique()
                    active_weeks_all.extend([(season, w) for w in active])

                # Calculate team total targets per week
                team_stats = stats_combined[stats_combined['team'] == team]

                # Beneficiary's share when injured out
                shares_when_out = []
                for season, week in out_weeks_all:
                    # Get player's targets for this specific week
                    player_week_data = with_out_df[
                        (with_out_df.get('season', pd.Series([season]*len(with_out_df))) == season) &
                        (with_out_df['week'] == week)
                    ]
                    if len(player_week_data) == 0:
                        continue  # Player didn't play this week

                    week_player_targets = player_week_data['targets'].iloc[0] if len(player_week_data) == 1 else player_week_data['targets'].sum()
                    week_team_targets = team_stats[(team_stats['season'] == season) & (team_stats['week'] == week)]['targets'].sum()

                    if week_team_targets > 0:
                        shares_when_out.append(week_player_targets / week_team_targets)

                # Beneficiary's share when injured active
                shares_baseline = []
                for season, week in active_weeks_all:
                    # Get player's targets for this specific week
                    player_week_data = baseline_df[
                        (baseline_df.get('season', pd.Series([season]*len(baseline_df))) == season) &
                        (baseline_df['week'] == week)
                    ]
                    if len(player_week_data) == 0:
                        continue  # Player didn't play this week

                    week_player_targets = player_week_data['targets'].iloc[0] if len(player_week_data) == 1 else player_week_data['targets'].sum()
                    week_team_targets = team_stats[(team_stats['season'] == season) & (team_stats['week'] == week)]['targets'].sum()

                    if week_team_targets > 0:
                        shares_baseline.append(week_player_targets / week_team_targets)

                if shares_when_out and shares_baseline:
                    avg_share_out = sum(shares_when_out) / len(shares_when_out)
                    avg_share_baseline = sum(shares_baseline) / len(shares_baseline)
                    if avg_share_baseline > 0:
                        raw_multiplier = avg_share_out / avg_share_baseline

                        # Cap extreme multipliers to prevent unrealistic projections
                        # Max reasonable boost: 3x targets (e.g., WR2 becomes WR1)
                        MAX_MULTIPLIER = 3.0
                        if raw_multiplier > MAX_MULTIPLIER:
                            logger.warning(
                                f"Capping extreme multiplier for {beneficiary_player}: "
                                f"{raw_multiplier:.2f}x → {MAX_MULTIPLIER}x "
                                f"(baseline={len(baseline_df)}, out={len(with_out_df)})"
                            )
                            result['target_share_multiplier'] = MAX_MULTIPLIER
                        else:
                            result['target_share_multiplier'] = raw_multiplier

                        logger.debug(f"Target share: {avg_share_baseline:.3f} -> {avg_share_out:.3f} = {result['target_share_multiplier']:.2f}x")
                else:
                    # Fallback to raw target count if share calculation fails
                    avg_targets_with_out = with_out_df['targets'].mean()
                    avg_targets_baseline = baseline_df['targets'].mean()
                    if avg_targets_baseline > 0:
                        result['target_share_multiplier'] = avg_targets_with_out / avg_targets_baseline
            except Exception as e:
                # Fallback to simple target count comparison
                logger.debug(f"Could not calculate target share, using count: {e}")
                avg_targets_with_out = with_out_df['targets'].mean()
                avg_targets_baseline = baseline_df['targets'].mean()

                if avg_targets_baseline > 0:
                    result['target_share_multiplier'] = avg_targets_with_out / avg_targets_baseline

    # Calculate carry share multiplier (for RBs)
    if beneficiary_position == 'RB':
        if 'carries' in with_out_df.columns and 'carries' in baseline_df.columns:
            avg_carries_with_out = with_out_df['carries'].mean()
            avg_carries_baseline = baseline_df['carries'].mean()

            if avg_carries_baseline > 0:
                result['carry_share_multiplier'] = avg_carries_with_out / avg_carries_baseline

    # Calculate yards per game multiplier
    if beneficiary_position in ['WR', 'TE']:
        if 'receiving_yards' in with_out_df.columns:
            avg_yards_with_out = with_out_df['receiving_yards'].mean()
            avg_yards_baseline = baseline_df['receiving_yards'].mean()

            if avg_yards_baseline > 0:
                result['yards_per_game_multiplier'] = avg_yards_with_out / avg_yards_baseline
    elif beneficiary_position == 'RB':
        # Include both rushing and receiving
        if 'rushing_yards' in with_out_df.columns:
            total_with_out = (with_out_df['rushing_yards'].fillna(0) +
                             with_out_df.get('receiving_yards', pd.Series([0]*len(with_out_df))).fillna(0)).mean()
            total_baseline = (baseline_df['rushing_yards'].fillna(0) +
                             baseline_df.get('receiving_yards', pd.Series([0]*len(baseline_df))).fillna(0)).mean()

            if total_baseline > 0:
                result['yards_per_game_multiplier'] = total_with_out / total_baseline

    logger.info(f"Historical impact: {beneficiary_player} when {injured_player} out: "
               f"targets={result['target_share_multiplier']:.2f}x, "
               f"carries={result['carry_share_multiplier']:.2f}x "
               f"(n={result['sample_size']}, conf={result['confidence']})")

    return result


def build_team_injury_history(team: str, seasons: list = [2024, 2025]) -> Dict:
    """
    Build comprehensive injury impact history for a team.

    Analyzes all historical instances where key players were out
    and measures actual performance changes for remaining players.

    Returns:
        Dictionary mapping (injured_player -> beneficiary) to historical multipliers
    """
    history = {}

    # Load depth charts to identify key positions
    from nfl_quant.utils.contextual_integration import _load_team_depth_charts
    depth_charts = _load_team_depth_charts()

    if team not in depth_charts:
        return history

    dc = depth_charts[team]

    # Analyze WR1 injury impact
    if 'wr1' in dc:
        wr1 = dc['wr1']

        # Impact on WR2
        if 'wr2' in dc:
            wr2 = dc['wr2']
            impact = calculate_historical_redistribution(
                team, wr1, 'WR', wr2, 'WR', seasons
            )
            history[f"{wr1}_out_impact_on_{wr2}"] = impact

        # Impact on TE1
        if 'te1' in dc:
            te1 = dc['te1']
            impact = calculate_historical_redistribution(
                team, wr1, 'WR', te1, 'TE', seasons
            )
            history[f"{wr1}_out_impact_on_{te1}"] = impact

        # Impact on RB1
        if 'rb1' in dc:
            rb1 = dc['rb1']
            impact = calculate_historical_redistribution(
                team, wr1, 'WR', rb1, 'RB', seasons
            )
            history[f"{wr1}_out_impact_on_{rb1}"] = impact

    # Analyze RB1 injury impact
    if 'rb1' in dc and 'rb2' in dc:
        rb1 = dc['rb1']
        rb2 = dc['rb2']
        impact = calculate_historical_redistribution(
            team, rb1, 'RB', rb2, 'RB', seasons
        )
        history[f"{rb1}_out_impact_on_{rb2}"] = impact

    return history


def get_injury_adjusted_projection(
    player_name: str,
    team: str,
    position: str,
    baseline_projection: float,
    injury_data: Dict,
    seasons: list = [2025],  # DEFAULT TO CURRENT SEASON ONLY - roster changes between seasons
    stat_type: str = 'yards'
) -> Tuple[float, str]:
    """
    Get injury-adjusted projection using historical data.

    Args:
        player_name: Player to project
        team: Team abbreviation
        position: Player position
        baseline_projection: Original projection without injury adjustment
        injury_data: Current injury data for the team
        seasons: Historical seasons to analyze
        stat_type: Type of stat to adjust ('yards', 'targets', 'carries')

    Returns:
        Tuple of (adjusted_projection, confidence_level)
    """
    from nfl_quant.utils.contextual_integration import _load_team_depth_charts
    depth_charts = _load_team_depth_charts()

    if team not in depth_charts:
        return baseline_projection, 'none'

    dc = depth_charts[team]

    # Check for injured teammates and get historical impact
    def is_out(status):
        if not status:
            return False
        return status.lower() in ['out', 'pup', 'ir', 'doubtful']

    total_multiplier = 1.0
    confidence = 'none'

    # Check WR1 injury
    wr1_status = injury_data.get('top_wr_1_status', 'active')
    if is_out(wr1_status) and position in ['WR', 'TE', 'RB']:
        wr1_name = injury_data.get('top_wr_1', dc.get('wr1', ''))

        # Get historical impact
        impact = calculate_historical_redistribution(
            team, wr1_name, 'WR', player_name, position, seasons
        )

        if impact['sample_size'] > 0:
            if stat_type == 'yards':
                total_multiplier *= impact['yards_per_game_multiplier']
            elif stat_type == 'targets':
                total_multiplier *= impact['target_share_multiplier']
            elif stat_type == 'carries':
                total_multiplier *= impact['carry_share_multiplier']

            confidence = impact['confidence']

    # Check RB1 injury (for RBs)
    if position == 'RB':
        rb1_status = injury_data.get('top_rb_status', 'active')
        if is_out(rb1_status):
            rb1_name = injury_data.get('top_rb', dc.get('rb1', ''))

            if player_name != rb1_name:
                impact = calculate_historical_redistribution(
                    team, rb1_name, 'RB', player_name, 'RB', seasons
                )

                if impact['sample_size'] > 0:
                    if stat_type == 'carries':
                        total_multiplier *= impact['carry_share_multiplier']
                    elif stat_type == 'targets':
                        total_multiplier *= impact['target_share_multiplier']
                    elif stat_type == 'yards':
                        total_multiplier *= impact['yards_per_game_multiplier']

                    confidence = impact['confidence']

    adjusted = baseline_projection * total_multiplier

    logger.debug(f"Historical injury adjustment for {player_name}: "
                f"{baseline_projection:.1f} -> {adjusted:.1f} "
                f"(multiplier={total_multiplier:.2f}, conf={confidence})")

    return adjusted, confidence


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=== Historical Injury Impact Analysis ===")
    print()

    # Test for ARI (Marvin Harrison Jr. out)
    print("Testing ARI - Marvin Harrison Jr. OUT:")
    history = build_team_injury_history('ARI', [2024, 2025])

    for key, value in history.items():
        print(f"  {key}:")
        print(f"    Targets: {value['target_share_multiplier']:.2f}x")
        print(f"    Carries: {value['carry_share_multiplier']:.2f}x")
        print(f"    Yards: {value['yards_per_game_multiplier']:.2f}x")
        print(f"    Sample size: {value['sample_size']}")
        print(f"    Confidence: {value['confidence']}")
        print()
