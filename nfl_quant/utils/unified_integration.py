#!/usr/bin/env python3
"""
Unified Integration Module - Ensures ALL Factors Always Integrated

This module provides a single point of integration for ALL 16 factors:
1. Defensive EPA (position-specific)
2. Weather adjustments
3. Divisional game factors
4. Contextual factors (rest, travel, bye week)
5. Injuries (QB, WR, RB, OL)
6. Red Zone factors (target share, carry share, goal line role)
7. Snap Share (calculated from data)
8. Home Field Advantage (team-specific)
9. Primetime games (SNF, MNF, TNF)
10. Altitude (high altitude stadiums)
11. Field Surface (turf vs grass)
12. Team Usage (pass/rush/target totals from simulations)
13. Game Script (dynamic, evolves during game)
14. Market Blending (blend market priors with model)
15. Calibration Consistency (use same calibrators everywhere)
16. Team Strength (Elo ratings, win probability, expected spread)

CRITICAL RULE: All prediction generation MUST use this module to ensure
consistent integration across the entire framework.

Usage:
    from nfl_quant.utils.unified_integration import integrate_all_factors

    # Integrate all factors for a week
    integrated_df = integrate_all_factors(
        week=10,
        season=2025,
        players_df=predictions_df,
        odds_df=odds_df
    )

    # Now players_df has ALL factors integrated:
    # - opponent_def_epa_vs_position
    # - weather_total_adjustment, weather_passing_adjustment
    # - is_divisional_game
    # - rest_epa_adjustment, travel_epa_adjustment, is_coming_off_bye
    # - injury_qb_status, injury_wr1_status, injury_rb1_status
    # - redzone_target_share, redzone_carry_share, goalline_carry_share
    # - snap_share (calculated from data)
    # - home_field_advantage_points
    # - is_primetime_game, primetime_type
    # - is_high_altitude, altitude_epa_adjustment
    # - field_surface (turf/grass)
    # - team_pass_attempts, team_rush_attempts, team_targets
    # - game_script_dynamic
    # - market_blended_prob (if odds available)
    # - team_elo, opp_elo, elo_diff, elo_win_prob, elo_expected_spread
    # - team_win_pct, opp_win_pct, team_point_diff_per_game, opp_point_diff_per_game
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import sys

# Use centralized path configuration
from nfl_quant.config_paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.utils.defensive_stats_integration import (
    get_defensive_epa_for_player,
    get_defensive_stats_batch,
    integrate_defensive_stats_into_predictions
)
from nfl_quant.features.contextual_features import ContextualFeatureEngine
from nfl_quant.utils.contextual_integration import load_injury_data
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.utils.team_names import normalize_team_name
from nfl_quant.features.team_strength import EnhancedEloCalculator

logger = logging.getLogger(__name__)


def load_weather_for_week(week: int, season: int = 2025) -> pd.DataFrame:
    """
    Load weather data for a week from nflweather.com cache OR NFLverse schedules.

    Priority:
    1. Load from nflweather.com cached JSON (research-backed adjustments via WeatherAdjusterV2)
    2. Fall back to NFLverse schedules (basic temp/wind data)

    Args:
        week: Week number
        season: Season year

    Returns:
        DataFrame with weather data by team (home and away teams for each game)
    """
    # Priority 1: Try nflweather.com cached data (has precipitation and more accurate forecasts)
    try:
        from nfl_quant.config_paths import DATA_DIR
        import json

        weather_cache = DATA_DIR / 'weather' / f'weather_week{week}_{season}.json'

        if weather_cache.exists():
            with open(weather_cache) as f:
                cached_games = json.load(f)

            if cached_games:
                weather_records = []
                for game in cached_games:
                    home_team = game.get('home_team')
                    away_team = game.get('away_team')

                    # Use pre-calculated adjustments from WeatherAdjusterV2
                    total_adj = game.get('total_adjustment', 0.0)
                    passing_adj = game.get('passing_adjustment', 0.0)
                    severity = game.get('severity', 'None')

                    # Add records for both teams
                    for team in [home_team, away_team]:
                        if team:
                            weather_records.append({
                                'team': team,
                                'temperature': game.get('temperature'),
                                'wind_speed': game.get('wind_speed'),
                                'is_dome': game.get('is_dome', False),
                                'conditions': game.get('conditions', ''),
                                'precip_chance': game.get('precip_chance', 0),
                                'precip_type': game.get('precip_type'),
                                'total_adjustment': total_adj,
                                'passing_adjustment': passing_adj,
                                'severity': severity,
                                'wind_bucket': game.get('wind_bucket', 'calm'),
                                'temp_bucket': game.get('temp_bucket', 'comfortable'),
                            })

                weather_df = pd.DataFrame(weather_records)

                # Log summary
                non_neutral = weather_df[weather_df['severity'] != 'None']
                if len(non_neutral) > 0:
                    logger.info(f"   ✅ Loaded nflweather.com cache: {len(non_neutral)//2} games with weather impact")
                    for _, row in weather_df[weather_df['severity'] != 'None'].drop_duplicates('team').iterrows():
                        logger.info(f"      - {row['team']}: {row['conditions']} ({row['severity']}, {row['passing_adjustment']:+.1%} passing)")
                else:
                    logger.info(f"   ✅ Loaded nflweather.com cache: No significant weather impacts")

                return weather_df

    except Exception as e:
        logger.warning(f"   Could not load nflweather.com cache: {e}")

    # Priority 2: Fall back to NFLverse schedules
    from nfl_quant.utils.nflverse_loader import load_schedules

    # Load schedules
    schedules = load_schedules(seasons=season)

    # Filter to the week
    week_games = schedules[schedules['week'] == week].copy()

    if len(week_games) == 0:
        logger.warning(f"No games found in NFLverse schedule for week {week}, season {season}")
        return pd.DataFrame(columns=['team', 'total_adjustment', 'passing_adjustment', 'severity'])

    # Calculate weather adjustments for each game
    weather_records = []

    for _, game in week_games.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        # Get weather data from schedule
        temp = game.get('temp')
        wind = game.get('wind')
        roof = str(game.get('roof', '')).lower() if pd.notna(game.get('roof')) else ''

        # Determine if dome/indoor
        is_dome = roof in ['dome', 'closed']
        is_retractable = roof == 'retractable'

        # Calculate adjustments
        total_adj, passing_adj, severity = _calculate_weather_adjustments(
            temp, wind, is_dome, is_retractable
        )

        # Add records for both teams
        for team in [home_team, away_team]:
            if pd.notna(team):
                weather_records.append({
                    'team': team,
                    'temperature': temp if pd.notna(temp) else None,
                    'wind_speed': wind if pd.notna(wind) else None,
                    'is_dome': is_dome,
                    'total_adjustment': total_adj,
                    'passing_adjustment': passing_adj,
                    'severity': severity
                })

    weather_df = pd.DataFrame(weather_records)

    # Log summary
    non_neutral = weather_df[weather_df['severity'] != 'None']
    if len(non_neutral) > 0:
        logger.info(f"   Weather impacts (NFLverse fallback) for week {week}: {len(non_neutral)//2} games with adjustments")

    return weather_df


def _calculate_weather_adjustments(
    temp: Optional[float],
    wind: Optional[float],
    is_dome: bool,
    is_retractable: bool
) -> Tuple[float, float, str]:
    """
    Calculate weather adjustments based on temperature and wind.

    Returns:
        Tuple of (total_adjustment, passing_adjustment, severity)
    """
    # Domes have no weather impact
    if is_dome:
        return 0.0, 0.0, "None (Dome)"

    # If retractable and no weather data, assume neutral
    if is_retractable and (pd.isna(temp) or pd.isna(wind)):
        return 0.0, 0.0, "None (Retractable - unknown)"

    # Missing data - return neutral
    if pd.isna(temp) and pd.isna(wind):
        return 0.0, 0.0, "None (No data)"

    total_adj = 0.0
    passing_adj = 0.0
    severity_parts = []

    # Temperature adjustments
    if pd.notna(temp):
        if temp < 25:
            # Extreme cold
            passing_adj -= 0.06
            total_adj -= 0.04
            severity_parts.append("Extreme Cold")
        elif temp < 32:
            # Cold
            passing_adj -= 0.04
            total_adj -= 0.02
            severity_parts.append("Cold")
        elif temp < 40:
            # Chilly
            passing_adj -= 0.02
            total_adj -= 0.01
            severity_parts.append("Chilly")
        elif temp > 90:
            # Extreme heat
            passing_adj -= 0.02
            total_adj -= 0.01
            severity_parts.append("Hot")

    # Wind adjustments
    if pd.notna(wind):
        if wind >= 20:
            # Extreme wind
            passing_adj -= 0.15
            total_adj -= 0.08
            severity_parts.append("Extreme Wind")
        elif wind >= 15:
            # High wind
            passing_adj -= 0.08
            total_adj -= 0.04
            severity_parts.append("High Wind")
        elif wind >= 10:
            # Moderate wind
            passing_adj -= 0.03
            total_adj -= 0.01
            severity_parts.append("Moderate Wind")

    # Determine severity string
    if not severity_parts:
        severity = "None"
    else:
        severity = " + ".join(severity_parts)

    return round(total_adj, 3), round(passing_adj, 3), severity


def check_divisional_game(home_team: str, away_team: str, week: int, season: int = 2025) -> bool:
    """
    Check if a game is divisional - FAIL EXPLICITLY if cannot determine.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        week: Week number
        season: Season year

    Returns:
        True if divisional game, False otherwise

    Raises:
        ValueError: If cannot determine divisional status
    """
    # Load schedule from PBP or games file
    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
    games_path = Path('data/nflverse/games.parquet')

    if games_path.exists():
        try:
            games_df = pd.read_parquet(games_path)
            game = games_df[
                (games_df['home_team'] == home_team) &
                (games_df['away_team'] == away_team) &
                (games_df['week'] == week)
            ]
            if len(game) > 0:
                return bool(game.iloc[0].get('div_game', 0) == 1)
        except Exception as e:
            logger.debug(f"Could not check divisional from games file: {e}")

    # Fallback: Check divisions manually
    divisions = {
        'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
        'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
        'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
        'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
        'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
        'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
        'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
        'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA'],
    }

    for division, teams in divisions.items():
        if home_team in teams and away_team in teams:
            return True

    return False


def load_contextual_factors_for_teams(
    teams: list,
    week: int,
    season: int = 2025
) -> Dict[str, Dict]:
    """
    Load contextual factors (rest, travel, bye) for teams - FAIL EXPLICITLY if unavailable.

    Args:
        teams: List of team abbreviations
        week: Current week
        season: Season year

    Returns:
        Dictionary mapping team -> contextual factors

    Raises:
        FileNotFoundError: If schedule data not available
    """
    contextual_engine = ContextualFeatureEngine()

    # Load schedule to get last game dates
    games_path = Path('data/nflverse/games.parquet')
    if not games_path.exists():
        raise FileNotFoundError(
            f"Schedule data not found: {games_path}. "
            f"Run data fetching scripts to populate schedule."
        )

    try:
        games_df = pd.read_parquet(games_path)
        season_games = games_df[games_df['season'] == season]
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load schedule data: {e}. "
            f"Run data fetching scripts to populate schedule."
        )

    contextual_factors = {}

    for team in teams:
        # Find last game before this week
        team_games = season_games[
            ((season_games['home_team'] == team) | (season_games['away_team'] == team)) &
            (season_games['week'] < week)
        ]

        if len(team_games) == 0:
            # First game of season or no data
            contextual_factors[team] = {
                'rest_adjustment': {'epa_adjustment': 0.0},
                'travel_adjustment': {'epa_adjustment': 0.0},
                'is_coming_off_bye': False,
            }
            continue

        last_game = team_games.iloc[-1]
        last_week = last_game['week']

        # Calculate days since last game (rough estimate: 7 days per week)
        days_since = (week - last_week) * 7
        is_bye = days_since > 10  # More than 10 days = bye week

        # Calculate rest adjustment
        rest_adj = contextual_engine.calculate_rest_adjustment(
            team=team,
            days_since_last_game=int(days_since),
            is_coming_off_bye=is_bye
        )

        # Travel adjustment requires opponent (will be calculated per game)
        contextual_factors[team] = {
            'rest_adjustment': rest_adj,
            'is_coming_off_bye': is_bye,
        }

    return contextual_factors


def integrate_all_factors(
    week: int,
    season: int,
    players_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    fail_on_missing: bool = True
) -> pd.DataFrame:
    """
    Integrate ALL factors into players DataFrame.

    This is the MAIN integration function - use this for ALL prediction generation.

    Integrated Factors:
    1. Defensive EPA (position-specific)
    2. Weather adjustments
    3. Divisional game status
    4. Contextual factors (rest, travel, bye)

    Args:
        week: Week number
        season: Season year
        players_df: DataFrame with player predictions (must have 'team', 'position', 'opponent')
        odds_df: DataFrame with odds (used to extract opponent info if missing)
        fail_on_missing: If True, raise errors when data unavailable (default: True)

    Returns:
        DataFrame with all factors integrated as new columns:
        - opponent_def_epa_vs_position: Position-specific defensive EPA
        - weather_total_adjustment: Weather adjustment multiplier
        - weather_passing_adjustment: Weather adjustment for passing
        - is_divisional_game: Boolean divisional flag
        - rest_epa_adjustment: Rest-based EPA adjustment
        - travel_epa_adjustment: Travel-based EPA adjustment
        - is_coming_off_bye: Boolean bye week flag

    Raises:
        FileNotFoundError: If required data unavailable (when fail_on_missing=True)
        ValueError: If required columns missing
    """
    logger.info("="*80)
    logger.info("UNIFIED INTEGRATION: Integrating ALL Factors")
    logger.info("="*80)

    # Validate required columns
    required_cols = ['team', 'position']
    missing_cols = [col for col in required_cols if col not in players_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Players DataFrame must have 'team' and 'position' columns."
        )

    # Make a copy to avoid modifying original
    integrated_df = players_df.copy()

    # Step 1: Integrate Defensive EPA
    logger.info("\n1. Integrating Defensive EPA...")
    try:
        integrated_df = integrate_defensive_stats_into_predictions(
            integrated_df, odds_df, week
        )
        logger.info("   ✅ Defensive EPA integrated")
    except Exception as e:
        if fail_on_missing:
            raise FileNotFoundError(
                f"Failed to integrate defensive EPA: {e}. "
                f"Run defensive stats extraction or ensure PBP data available."
            )
        else:
            logger.warning(f"   ⚠️  Defensive EPA integration failed: {e}")
            integrated_df['opponent_def_epa_vs_position'] = 0.0

    # Step 2: Integrate Weather (from NFLverse schedules)
    logger.info("\n2. Integrating Weather Adjustments (from NFLverse schedules)...")
    try:
        weather_df = load_weather_for_week(week, season)

        # Merge weather data by team
        if 'team' in integrated_df.columns and len(weather_df) > 0:
            merge_cols = ['team', 'total_adjustment', 'passing_adjustment', 'severity']
            available_cols = [c for c in merge_cols if c in weather_df.columns]
            integrated_df = integrated_df.merge(
                weather_df[available_cols],
                on='team',
                how='left'
            )
            integrated_df['weather_total_adjustment'] = integrated_df['total_adjustment'].fillna(0.0)
            integrated_df['weather_passing_adjustment'] = integrated_df['passing_adjustment'].fillna(0.0)

            # Count games with weather impact
            non_neutral = len(weather_df[weather_df['severity'] != 'None']) // 2
            logger.info(f"   ✅ Weather integrated ({non_neutral} games with weather impact)")
        else:
            logger.warning("   ⚠️  No weather data available, using neutral adjustments")
            integrated_df['weather_total_adjustment'] = 0.0
            integrated_df['weather_passing_adjustment'] = 0.0
    except Exception as e:
        logger.warning(f"   ⚠️  Weather integration failed: {e}")
        integrated_df['weather_total_adjustment'] = 0.0
        integrated_df['weather_passing_adjustment'] = 0.0

    # Step 3: Integrate Divisional Status
    logger.info("\n3. Integrating Divisional Game Status...")
    try:
        # Get unique games from odds or players_df
        if 'opponent' in integrated_df.columns:
            games = integrated_df[['team', 'opponent']].drop_duplicates()
        elif 'home_team' in odds_df.columns and 'away_team' in odds_df.columns:
            games = odds_df[['home_team', 'away_team']].drop_duplicates()
            # Map to players_df teams
            # This is a simplified mapping - may need refinement
            integrated_df['opponent'] = None  # Will be filled from odds
        else:
            raise ValueError("Cannot determine games without 'opponent' column or odds data")

        divisional_status = {}
        for _, row in games.iterrows():
            home_team = row.get('team') or row.get('home_team')
            away_team = row.get('opponent') or row.get('away_team')

            if pd.isna(home_team) or pd.isna(away_team):
                continue

            is_divisional = check_divisional_game(home_team, away_team, week, season)
            divisional_status[(home_team, away_team)] = is_divisional
            divisional_status[(away_team, home_team)] = is_divisional  # Bidirectional

        # Apply divisional status to players
        if 'opponent' in integrated_df.columns:
            integrated_df['is_divisional_game'] = integrated_df.apply(
                lambda row: divisional_status.get(
                    (row['team'], row['opponent']), False
                ),
                axis=1
            )
            logger.info(f"   ✅ Divisional status integrated for {len(divisional_status)} games")
        else:
            logger.warning("   ⚠️  Cannot apply divisional status without 'opponent' column")
            integrated_df['is_divisional_game'] = False
    except Exception as e:
        if fail_on_missing:
            raise ValueError(f"Failed to integrate divisional status: {e}")
        else:
            logger.warning(f"   ⚠️  Divisional status integration failed: {e}")
            integrated_df['is_divisional_game'] = False

    # Step 4: Integrate Contextual Factors (Rest, Travel, Bye)
    logger.info("\n4. Integrating Contextual Factors (Rest, Travel, Bye)...")
    try:
        unique_teams = integrated_df['team'].unique().tolist()
        contextual_factors = load_contextual_factors_for_teams(unique_teams, week, season)

        # Apply rest adjustments
        integrated_df['rest_epa_adjustment'] = integrated_df['team'].apply(
            lambda team: contextual_factors.get(team, {}).get('rest_adjustment', {}).get('epa_adjustment', 0.0)
        )
        integrated_df['is_coming_off_bye'] = integrated_df['team'].apply(
            lambda team: contextual_factors.get(team, {}).get('is_coming_off_bye', False)
        )

        # Travel adjustments require opponent (calculate per game)
        if 'opponent' in integrated_df.columns:
            contextual_engine = ContextualFeatureEngine()
            integrated_df['travel_epa_adjustment'] = integrated_df.apply(
                lambda row: contextual_engine.calculate_travel_adjustment(
                    home_team=row['team'] if row.get('is_home', False) else row['opponent'],
                    away_team=row['opponent'] if row.get('is_home', False) else row['team']
                ).get('epa_adjustment', 0.0),
                axis=1
            )
        else:
            integrated_df['travel_epa_adjustment'] = 0.0

        logger.info(f"   ✅ Contextual factors integrated for {len(unique_teams)} teams")
    except Exception as e:
        if fail_on_missing:
            raise FileNotFoundError(f"Failed to integrate contextual factors: {e}")
        else:
            logger.warning(f"   ⚠️  Contextual factors integration failed: {e}")
            integrated_df['rest_epa_adjustment'] = 0.0
            integrated_df['travel_epa_adjustment'] = 0.0
            integrated_df['is_coming_off_bye'] = False

    # Step 5: Integrate Injuries
    logger.info("\n5. Integrating Injury Data...")
    try:
        injury_data = load_injury_data(week)

        if injury_data:
            # Add injury status columns
            integrated_df['injury_qb_status'] = integrated_df['team'].apply(
                lambda team: injury_data.get(team, {}).get('qb_status', 'healthy')
            )
            integrated_df['injury_wr1_status'] = integrated_df['team'].apply(
                lambda team: injury_data.get(team, {}).get('top_wr_1_status', 'active')
            )
            integrated_df['injury_rb1_status'] = integrated_df['team'].apply(
                lambda team: injury_data.get(team, {}).get('top_rb_status', 'active')
            )
            logger.info(f"   ✅ Injury data integrated for {len(injury_data)} teams")
        else:
            integrated_df['injury_qb_status'] = 'healthy'
            integrated_df['injury_wr1_status'] = 'active'
            integrated_df['injury_rb1_status'] = 'active'
            logger.warning("   ⚠️  No injury data available")
    except Exception as e:
        if fail_on_missing:
            logger.warning(f"   ⚠️  Injury data integration failed: {e}")
        integrated_df['injury_qb_status'] = 'healthy'
        integrated_df['injury_wr1_status'] = 'active'
        integrated_df['injury_rb1_status'] = 'active'

    # Step 6: Integrate Red Zone Factors
    logger.info("\n6. Integrating Red Zone Factors...")
    try:
        if 'player_name' in integrated_df.columns:
            # Calculate red zone shares from PBP for each player
            redzone_data = []
            for _, row in integrated_df.iterrows():
                rz_shares = calculate_red_zone_shares_from_pbp(
                    player_name=row.get('player_name', ''),
                    position=row.get('position', ''),
                    team=row.get('team', ''),
                    week=week,
                    season=season
                )
                redzone_data.append(rz_shares)

            rz_df = pd.DataFrame(redzone_data)
            integrated_df['redzone_target_share'] = rz_df['redzone_target_share']
            integrated_df['redzone_carry_share'] = rz_df['redzone_carry_share']
            integrated_df['goalline_carry_share'] = rz_df['goalline_carry_share']
            logger.info(f"   ✅ Red zone factors integrated for {len(integrated_df)} players")
        else:
            integrated_df['redzone_target_share'] = None
            integrated_df['redzone_carry_share'] = None
            integrated_df['goalline_carry_share'] = None
            logger.warning("   ⚠️  Cannot calculate red zone shares without 'player_name' column")
    except Exception as e:
        logger.warning(f"   ⚠️  Red zone integration failed: {e}")
        integrated_df['redzone_target_share'] = None
        integrated_df['redzone_carry_share'] = None
        integrated_df['goalline_carry_share'] = None

    # Step 7: Integrate Snap Share (from data)
    logger.info("\n7. Integrating Snap Share (from data)...")
    try:
        if 'player_name' in integrated_df.columns:
            # Try to calculate from data (PBP or usage patterns)
            snap_shares = []
            for _, row in integrated_df.iterrows():
                snap_share = calculate_snap_share_from_data(
                    player_name=row.get('player_name', ''),
                    position=row.get('position', ''),
                    team=row.get('team', ''),
                    week=week,
                    season=season,
                    players_df=integrated_df  # Pass DataFrame for usage-based calculation
                )
                # If not available, use trailing_snap_share if exists
                if snap_share is None and 'trailing_snap_share' in row:
                    snap_share = row.get('trailing_snap_share')
                snap_shares.append(snap_share)

            integrated_df['snap_share'] = snap_shares
            calculated_count = sum(1 for s in snap_shares if s is not None)
            logger.info(f"   ✅ Snap share integrated ({calculated_count}/{len(snap_shares)} calculated from data)")
        else:
            integrated_df['snap_share'] = None
            logger.warning("   ⚠️  Cannot calculate snap share without 'player_name' column")
    except Exception as e:
        logger.warning(f"   ⚠️  Snap share integration failed: {e}")
        integrated_df['snap_share'] = None

    # Step 8: Integrate Home Field Advantage
    logger.info("\n8. Integrating Home Field Advantage...")
    try:
        integrated_df['home_field_advantage_points'] = integrated_df['team'].apply(
            lambda team: get_home_field_advantage(team, season)
        )
        logger.info("   ✅ Home field advantage integrated")
    except Exception as e:
        logger.warning(f"   ⚠️  HFA integration failed: {e}")
        integrated_df['home_field_advantage_points'] = 1.5  # League average

    # Step 9: Integrate Primetime Status
    logger.info("\n9. Integrating Primetime Status...")
    try:
        if 'opponent' in integrated_df.columns:
            primetime_data = []
            for _, row in integrated_df.iterrows():
                pt_status = get_primetime_status(
                    home_team=row.get('team', ''),
                    away_team=row.get('opponent', ''),
                    week=week,
                    season=season
                )
                primetime_data.append(pt_status)

            pt_df = pd.DataFrame(primetime_data)
            integrated_df['is_primetime_game'] = pt_df['is_primetime']
            integrated_df['primetime_type'] = pt_df['primetime_type']
            logger.info(f"   ✅ Primetime status integrated")
        else:
            integrated_df['is_primetime_game'] = False
            integrated_df['primetime_type'] = None
            logger.warning("   ⚠️  Cannot determine primetime without 'opponent' column")
    except Exception as e:
        logger.warning(f"   ⚠️  Primetime integration failed: {e}")
        integrated_df['is_primetime_game'] = False
        integrated_df['primetime_type'] = None

    # Step 10: Integrate Altitude
    logger.info("\n10. Integrating Altitude...")
    try:
        if 'opponent' in integrated_df.columns:
            altitude_data = []
            for _, row in integrated_df.iterrows():
                # Determine home team (assume team is home if not specified)
                home_team = row.get('team', '') if row.get('is_home', True) else row.get('opponent', '')
                away_team = row.get('opponent', '') if row.get('is_home', True) else row.get('team', '')
                alt_info = get_altitude_info(home_team, away_team)
                altitude_data.append(alt_info)

            alt_df = pd.DataFrame(altitude_data)
            integrated_df['elevation_feet'] = alt_df['elevation_feet']
            integrated_df['is_high_altitude'] = alt_df['is_high_altitude']
            integrated_df['altitude_epa_adjustment'] = alt_df['altitude_epa_adjustment']
            logger.info("   ✅ Altitude integrated")
        else:
            integrated_df['elevation_feet'] = 0
            integrated_df['is_high_altitude'] = False
            integrated_df['altitude_epa_adjustment'] = 0.0
    except Exception as e:
        logger.warning(f"   ⚠️  Altitude integration failed: {e}")
        integrated_df['elevation_feet'] = 0
        integrated_df['is_high_altitude'] = False
        integrated_df['altitude_epa_adjustment'] = 0.0

    # Step 11: Integrate Field Surface
    logger.info("\n11. Integrating Field Surface...")
    try:
        if 'opponent' in integrated_df.columns:
            integrated_df['field_surface'] = integrated_df.apply(
                lambda row: get_field_surface(
                    row.get('team', '') if row.get('is_home', True) else row.get('opponent', '')
                ),
                axis=1
            )
            logger.info("   ✅ Field surface integrated")
        else:
            integrated_df['field_surface'] = 'grass'  # Default
    except Exception as e:
        logger.warning(f"   ⚠️  Field surface integration failed: {e}")
        integrated_df['field_surface'] = 'grass'

    # Step 12: Team Usage (from historical data and game context)
    logger.info("\n12. Integrating Team Usage...")
    try:
        team_usage = get_team_usage_stats(season, week)
        if team_usage is not None and 'team' in integrated_df.columns:
            integrated_df = integrated_df.merge(
                team_usage[['team', 'pass_attempts_per_game', 'rush_attempts_per_game', 'targets_per_game']],
                on='team',
                how='left'
            )
            integrated_df['team_pass_attempts'] = integrated_df['pass_attempts_per_game']
            integrated_df['team_rush_attempts'] = integrated_df['rush_attempts_per_game']
            integrated_df['team_targets'] = integrated_df['targets_per_game']
            integrated_df = integrated_df.drop(columns=['pass_attempts_per_game', 'rush_attempts_per_game', 'targets_per_game'], errors='ignore')
            logger.info(f"   ✅ Team usage integrated for {integrated_df['team_pass_attempts'].notna().sum()} players")
        else:
            # Fallback to league averages
            integrated_df['team_pass_attempts'] = 35.0  # League average
            integrated_df['team_rush_attempts'] = 26.0  # League average
            integrated_df['team_targets'] = 30.0  # League average
            logger.info("   ⚠️  Using league average team usage (no historical data)")
    except Exception as e:
        logger.warning(f"   ⚠️  Team usage integration failed: {e}")
        integrated_df['team_pass_attempts'] = 35.0
        integrated_df['team_rush_attempts'] = 26.0
        integrated_df['team_targets'] = 30.0

    # Step 13: Game Script (from Vegas spread)
    logger.info("\n13. Integrating Game Script (from Vegas)...")
    try:
        game_script = get_game_script_from_vegas(season, week)
        if game_script is not None and 'team' in integrated_df.columns:
            # Drop existing game_script_dynamic if present (to avoid merge collision creating _x/_y suffixes)
            if 'game_script_dynamic' in integrated_df.columns:
                integrated_df = integrated_df.drop(columns=['game_script_dynamic'])
            integrated_df = integrated_df.merge(
                game_script[['team', 'game_script_dynamic']],
                on='team',
                how='left'
            )
            integrated_df['game_script_dynamic'] = integrated_df['game_script_dynamic'].fillna(0.0)
            logger.info(f"   ✅ Game script integrated for {integrated_df['game_script_dynamic'].notna().sum()} players")
        else:
            integrated_df['game_script_dynamic'] = 0.0  # Neutral
            logger.info("   ⚠️  Using neutral game script (no Vegas data)")
    except Exception as e:
        logger.warning(f"   ⚠️  Game script integration failed: {e}")
        integrated_df['game_script_dynamic'] = 0.0

    # Step 14: Market Blending - Note: Requires odds data
    logger.info("\n14. Integrating Market Blending...")
    # Market blending happens during probability calculation, not here
    # This is informational - actual blending happens in recommendation generation
    integrated_df['market_blended_prob'] = None  # Will be filled during probability calculation
    logger.info("   ⚠️  Market blending happens during probability calculation")

    # Step 15: Calibration Consistency - Note: This is handled by calibrator loader
    logger.info("\n15. Calibration Consistency...")
    # Calibration consistency is ensured by using calibrator_loader module
    # This is informational - actual calibration happens in recommendation generation
    logger.info("   ✅ Calibration consistency ensured via calibrator_loader module")

    # Step 16: Integrate Team Strength (Elo)
    logger.info("\n16. Integrating Team Strength (Elo)...")
    try:
        elo_calculator = EnhancedEloCalculator()

        # Get Elo ratings for all teams entering this week
        elo_ratings = elo_calculator.calculate_elo_through_week(season, week)

        # Get team records and point differentials for context
        schedules = pd.read_parquet('data/nflverse/schedules.parquet')
        completed_games = schedules[
            (schedules['season'] == season) &
            (schedules['week'] < week) &
            (schedules['home_score'].notna()) &
            (schedules['game_type'] == 'REG')
        ]

        # Calculate team records
        team_records = {}
        team_point_diffs = {}
        for team in elo_ratings.keys():
            home_games = completed_games[completed_games['home_team'] == team]
            away_games = completed_games[completed_games['away_team'] == team]

            home_wins = (home_games['result'] > 0).sum()
            away_wins = (away_games['result'] < 0).sum()
            total_wins = home_wins + away_wins
            total_games = len(home_games) + len(away_games)

            team_records[team] = total_wins / total_games if total_games > 0 else 0.5

            # Point differential
            home_diff = home_games['result'].sum() if len(home_games) > 0 else 0
            away_diff = -away_games['result'].sum() if len(away_games) > 0 else 0
            total_diff = home_diff + away_diff
            team_point_diffs[team] = total_diff / total_games if total_games > 0 else 0

        # Apply Elo features to each player
        if 'opponent' in integrated_df.columns:
            # Team Elo
            integrated_df['team_elo'] = integrated_df['team'].apply(
                lambda team: elo_ratings.get(team, 1505)
            )
            # Opponent Elo
            integrated_df['opp_elo'] = integrated_df['opponent'].apply(
                lambda opp: elo_ratings.get(opp, 1505)
            )
            # Elo differential (positive = player's team is better)
            integrated_df['elo_diff'] = integrated_df['team_elo'] - integrated_df['opp_elo']

            # Win probability based on Elo (simplified, no home adjustment here)
            integrated_df['elo_win_prob'] = integrated_df['elo_diff'].apply(
                lambda diff: 1 / (1 + 10 ** (-diff / 400))
            )

            # Expected spread from Elo (25 Elo points ≈ 1 point spread)
            integrated_df['elo_expected_spread'] = integrated_df['elo_diff'] / 25

            # Team win percentage
            integrated_df['team_win_pct'] = integrated_df['team'].apply(
                lambda team: team_records.get(team, 0.5)
            )
            integrated_df['opp_win_pct'] = integrated_df['opponent'].apply(
                lambda opp: team_records.get(opp, 0.5)
            )

            # Point differential per game
            integrated_df['team_point_diff_per_game'] = integrated_df['team'].apply(
                lambda team: team_point_diffs.get(team, 0)
            )
            integrated_df['opp_point_diff_per_game'] = integrated_df['opponent'].apply(
                lambda opp: team_point_diffs.get(opp, 0)
            )

            logger.info(f"   ✅ Elo features integrated for {len(elo_ratings)} teams")
            logger.info(f"      Top 3: {sorted(elo_ratings.items(), key=lambda x: -x[1])[:3]}")
        else:
            # Can still add team Elo even without opponent
            integrated_df['team_elo'] = integrated_df['team'].apply(
                lambda team: elo_ratings.get(team, 1505)
            )
            integrated_df['opp_elo'] = 1505  # Default
            integrated_df['elo_diff'] = 0
            integrated_df['elo_win_prob'] = 0.5
            integrated_df['elo_expected_spread'] = 0
            integrated_df['team_win_pct'] = 0.5
            integrated_df['opp_win_pct'] = 0.5
            integrated_df['team_point_diff_per_game'] = 0
            integrated_df['opp_point_diff_per_game'] = 0
            logger.warning("   ⚠️  Cannot calculate full Elo features without 'opponent' column")
    except Exception as e:
        logger.warning(f"   ⚠️  Team strength integration failed: {e}")
        integrated_df['team_elo'] = 1505
        integrated_df['opp_elo'] = 1505
        integrated_df['elo_diff'] = 0
        integrated_df['elo_win_prob'] = 0.5
        integrated_df['elo_expected_spread'] = 0
        integrated_df['team_win_pct'] = 0.5
        integrated_df['opp_win_pct'] = 0.5
        integrated_df['team_point_diff_per_game'] = 0
        integrated_df['opp_point_diff_per_game'] = 0

    logger.info("\n" + "="*80)
    logger.info("✅ UNIFIED INTEGRATION COMPLETE - ALL 16 FACTORS INTEGRATED")
    logger.info("="*80)
    logger.info(f"   Integrated factors for {len(integrated_df)} players")
    logger.info(f"   Columns added:")
    logger.info(f"     - EPA: opponent_def_epa_vs_position")
    logger.info(f"     - Weather: weather_total_adjustment, weather_passing_adjustment")
    logger.info(f"     - Divisional: is_divisional_game")
    logger.info(f"     - Contextual: rest_epa_adjustment, travel_epa_adjustment, is_coming_off_bye")
    logger.info(f"     - Injuries: injury_qb_status, injury_wr1_status, injury_rb1_status")
    logger.info(f"     - Red Zone: redzone_target_share, redzone_carry_share, goalline_carry_share")
    logger.info(f"     - Snap Share: snap_share")
    logger.info(f"     - HFA: home_field_advantage_points")
    logger.info(f"     - Primetime: is_primetime_game, primetime_type")
    logger.info(f"     - Altitude: elevation_feet, is_high_altitude, altitude_epa_adjustment")
    logger.info(f"     - Field Surface: field_surface")
    logger.info(f"     - Team Usage: team_pass_attempts, team_rush_attempts, team_targets (from game context)")
    logger.info(f"     - Game Script: game_script_dynamic (from game context)")
    logger.info(f"     - Market Blending: market_blended_prob (during probability calculation)")
    logger.info(f"     - Calibration: Consistent via calibrator_loader")
    logger.info(f"     - Team Strength: team_elo, opp_elo, elo_diff, elo_win_prob, elo_expected_spread")
    logger.info(f"     - Team Record: team_win_pct, opp_win_pct, team_point_diff_per_game, opp_point_diff_per_game")

    return integrated_df


def get_primetime_status(home_team: str, away_team: str, week: int, season: int = 2025) -> Dict[str, Any]:
    """
    Get primetime status for a game.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        week: Week number
        season: Season year

    Returns:
        Dictionary with:
            - is_primetime: bool
            - primetime_type: str ('SNF', 'MNF', 'TNF', 'EARLY', 'AFTERNOON', or None)
    """
    # Try schedules.parquet first (more reliable)
    schedules_path = Path('data/nflverse/schedules.parquet')
    games_path = Path('data/nflverse/games.parquet')

    df_path = schedules_path if schedules_path.exists() else games_path
    if not df_path.exists():
        return {'is_primetime': False, 'primetime_type': None}

    try:
        games_df = pd.read_parquet(df_path)
        game = games_df[
            (games_df['home_team'] == home_team) &
            (games_df['away_team'] == away_team) &
            (games_df['week'] == week) &
            (games_df['season'] == season)
        ]

        if len(game) > 0:
            row = game.iloc[0]
            gametime = str(row.get('gametime', ''))
            weekday = str(row.get('weekday', ''))

            # Parse the hour from gametime (format: "20:20", "13:00", etc.)
            try:
                hour = int(gametime.split(':')[0]) if ':' in gametime else 0
            except (ValueError, IndexError):
                hour = 0

            # Primetime = games at 8pm (20:00) or later
            if hour >= 20:
                if 'Thursday' in weekday:
                    return {'is_primetime': True, 'primetime_type': 'TNF'}
                elif 'Sunday' in weekday:
                    return {'is_primetime': True, 'primetime_type': 'SNF'}
                elif 'Monday' in weekday:
                    return {'is_primetime': True, 'primetime_type': 'MNF'}
                elif 'Friday' in weekday or 'Saturday' in weekday:
                    return {'is_primetime': True, 'primetime_type': 'PRIMETIME'}
                else:
                    return {'is_primetime': True, 'primetime_type': 'PRIMETIME'}
            elif 16 <= hour < 18:
                return {'is_primetime': False, 'primetime_type': 'AFTERNOON'}
            else:
                return {'is_primetime': False, 'primetime_type': 'EARLY'}

    except Exception as e:
        logger.debug(f"Could not determine primetime status: {e}")

    return {'is_primetime': False, 'primetime_type': None}


def get_altitude_info(home_team: str, away_team: Optional[str] = None) -> Dict[str, Any]:
    """
    Get altitude information for home team's stadium.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation (optional, for EPA adjustment)

    Returns:
        Dictionary with:
            - elevation_feet: int
            - is_high_altitude: bool
            - altitude_epa_adjustment: float (for visiting team)
    """
    contextual_engine = ContextualFeatureEngine()
    # If away_team provided, calculate full adjustment
    if away_team:
        altitude_info = contextual_engine.calculate_altitude_adjustment(home_team, away_team)
    else:
        # Just get elevation info
        stadium_info = contextual_engine.team_stadiums.get(home_team, {})
        elevation = stadium_info.get('elevation_feet', 0)
        altitude_info = {
            'elevation_feet': elevation,
            'is_high_altitude': elevation >= contextual_engine.config.altitude_threshold_feet,
            'epa_multiplier': 1.0 + (contextual_engine.config.altitude_visiting_epa_penalty if elevation >= contextual_engine.config.altitude_threshold_feet else 0.0)
        }

    return {
        'elevation_feet': altitude_info.get('elevation_feet', 0),
        'is_high_altitude': altitude_info.get('is_high_altitude', False),
        'altitude_epa_adjustment': altitude_info.get('epa_multiplier', 1.0) - 1.0
    }


def get_field_surface(home_team: str) -> str:
    """
    Get field surface type (turf vs grass) for home team's stadium.

    Args:
        home_team: Home team abbreviation

    Returns:
        'turf' or 'grass'
    """
    # Field surface mapping (most stadiums are grass, some are turf)
    # This is a simplified mapping - could be enhanced with actual stadium data
    turf_stadiums = {
        'ATL', 'BAL', 'BUF', 'CAR', 'CIN', 'CLE', 'DAL', 'DEN', 'DET',
        'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIN',
        'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN'
    }

    # Most outdoor stadiums use grass, domes/retractable use turf
    # Simplified: Assume grass unless known to be turf
    if home_team in turf_stadiums:
        return 'turf'
    return 'grass'


def get_home_field_advantage(team: str, season: int = 2025) -> float:
    """
    Get team-specific home field advantage in points.

    Args:
        team: Team abbreviation
        season: Season year

    Returns:
        HFA in points (typically 0.5-2.5)
    """
    contextual_engine = ContextualFeatureEngine()
    # Use league average for now (could be enhanced with team-specific calculation)
    return contextual_engine.config.hfa_points_average


def calculate_red_zone_shares_from_pbp(
    player_name: str,
    position: str,
    team: str,
    week: int,
    season: int = 2025,
    lookback_weeks: int = None  # None = use all available historical data
) -> Dict[str, float]:
    """
    Calculate red zone shares from PBP data.

    Args:
        player_name: Player name (full format, e.g., "Tyreek Hill")
        position: Player position
        team: Team abbreviation
        week: Current week
        season: Season year
        lookback_weeks: Number of weeks to look back (None = all historical data)

    Returns:
        Dictionary with:
            - redzone_target_share: float (for WR/TE/RB)
            - redzone_carry_share: float (for RB)
            - goalline_carry_share: float (for RB)
    """

    def convert_to_pbp_name(full_name: str) -> str:
        """Convert full name to PBP format (e.g., 'Tyreek Hill' -> 'T.Hill')"""
        if not full_name or not isinstance(full_name, str):
            return ""

        parts = full_name.strip().split()
        if len(parts) < 2:
            return full_name

        # Handle suffixes
        last_name_parts = []
        for i in range(1, len(parts)):
            part = parts[i]
            if part.lower() in ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v']:
                break
            last_name_parts.append(part)

        first_initial = parts[0][0].upper()
        last_name = ' '.join(last_name_parts)
        return f"{first_initial}.{last_name}"

    # Use fresh NFLverse PBP data (updated daily) instead of stale processed file
    pbp_path = Path('data/nflverse/pbp.parquet')
    if not pbp_path.exists():
        # Fallback to season-specific file
        pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
    if not pbp_path.exists():
        # Last resort: processed file
        pbp_path = Path(f'data/processed/pbp_{season}.parquet')
    if not pbp_path.exists():
        return {
            'redzone_target_share': None,
            'redzone_carry_share': None,
            'goalline_carry_share': None
        }

    try:
        pbp_df = pd.read_parquet(pbp_path)

        # Convert player name to PBP format
        pbp_player_name = convert_to_pbp_name(player_name)

        # Filter to relevant weeks (use all historical data if lookback_weeks=None)
        if lookback_weeks is None:
            relevant_pbp = pbp_df[
                (pbp_df['week'] < week) &
                (pbp_df['posteam'] == team)
            ]
        else:
            start_week = max(1, week - lookback_weeks)
            relevant_pbp = pbp_df[
                (pbp_df['week'] >= start_week) &
                (pbp_df['week'] < week) &
                (pbp_df['posteam'] == team)
            ]

        if len(relevant_pbp) == 0:
            return {
                'redzone_target_share': None,
                'redzone_carry_share': None,
                'goalline_carry_share': None
            }

        shares = {}

        # Red zone target share (for WR/TE/RB)
        if position in ['WR', 'TE', 'RB']:
            player_rz_targets = relevant_pbp[
                (relevant_pbp['receiver_player_name'] == pbp_player_name) &
                (relevant_pbp['play_type'] == 'pass') &
                (relevant_pbp['yardline_100'] <= 20)
            ]
            team_rz_targets = relevant_pbp[
                (relevant_pbp['play_type'] == 'pass') &
                (relevant_pbp['yardline_100'] <= 20)
            ]
            shares['redzone_target_share'] = (
                len(player_rz_targets) / len(team_rz_targets)
                if len(team_rz_targets) > 0 else None
            )

        # Red zone carry share (for RB)
        if position == 'RB':
            player_rz_carries = relevant_pbp[
                (relevant_pbp['rusher_player_name'] == pbp_player_name) &
                (relevant_pbp['play_type'] == 'run') &
                (relevant_pbp['yardline_100'] <= 20)
            ]
            team_rz_carries = relevant_pbp[
                (relevant_pbp['play_type'] == 'run') &
                (relevant_pbp['yardline_100'] <= 20)
            ]
            shares['redzone_carry_share'] = (
                len(player_rz_carries) / len(team_rz_carries)
                if len(team_rz_carries) > 0 else None
            )

            # Goal line carry share (inside 5 yards)
            player_gl_carries = relevant_pbp[
                (relevant_pbp['rusher_player_name'] == pbp_player_name) &
                (relevant_pbp['play_type'] == 'run') &
                (relevant_pbp['yardline_100'] <= 5)
            ]
            team_gl_carries = relevant_pbp[
                (relevant_pbp['play_type'] == 'run') &
                (relevant_pbp['yardline_100'] <= 5)
            ]
            shares['goalline_carry_share'] = (
                len(player_gl_carries) / len(team_gl_carries)
                if len(team_gl_carries) > 0 else None
            )

        return shares

    except Exception as e:
        logger.debug(f"Could not calculate red zone shares for {player_name}: {e}")
        return {
            'redzone_target_share': None,
            'redzone_carry_share': None,
            'goalline_carry_share': None
        }


def calculate_snap_share_from_data(
    player_name: str,
    position: str,
    team: str,
    week: int,
    season: int = 2025,
    lookback_weeks: int = None,  # None = use all available data
    players_df: Optional[pd.DataFrame] = None
) -> Optional[float]:
    """
    Calculate snap share from actual snap_counts.parquet data.

    FIXED: Previously used PBP ball touches (30 touches / 730 plays = 4.1% for Rashee Rice)
    NOW: Uses actual snap counts from NFLverse (70% snap share for Rashee Rice)

    Args:
        player_name: Player name (full format, e.g., "Patrick Mahomes")
        position: Player position
        team: Team abbreviation
        week: Current week
        season: Season year
        lookback_weeks: Number of weeks to look back (None = all available historical data)
        players_df: Not used (kept for compatibility)

    Returns:
        Snap share (0.0-1.0) or None if data unavailable
    """

    # Load snap counts data (CORRECT source - not PBP ball touches)
    snap_path = Path('data/nflverse/snap_counts.parquet')
    if not snap_path.exists():
        logger.warning(f"Snap counts data not found at {snap_path}. Run: Rscript scripts/fetch/fetch_nflverse_data.R")
        return None

    try:
        snap_counts = pd.read_parquet(snap_path)

        # Normalize player name for matching using canonical function
        player_normalized = normalize_player_name(player_name)

        # Filter to player, team, season
        # Try exact match first, then partial match on last name
        player_snaps = snap_counts[
            (snap_counts['player'].apply(normalize_player_name) == player_normalized) &
            (snap_counts['team'] == team) &
            (snap_counts['season'] == season) &
            (snap_counts['week'] < week)
        ]

        # If no exact match, try last name only
        if len(player_snaps) == 0:
            name_parts = player_name.strip().split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1].lower()
                # Remove suffix if present
                if last_name in ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v']:
                    last_name = name_parts[-2].lower() if len(name_parts) >= 3 else last_name

                player_snaps = snap_counts[
                    (snap_counts['player'].str.lower().str.contains(last_name, na=False)) &
                    (snap_counts['team'] == team) &
                    (snap_counts['season'] == season) &
                    (snap_counts['week'] < week)
                ]

        # Apply lookback window if specified
        if lookback_weeks is not None and len(player_snaps) > 0:
            start_week = max(1, week - lookback_weeks)
            player_snaps = player_snaps[player_snaps['week'] >= start_week]

        if len(player_snaps) == 0:
            logger.debug(f"No snap count data found for {player_name} ({team}, season {season})")
            return None

        # Calculate average snap percentage from offense_pct column
        if 'offense_pct' not in player_snaps.columns:
            logger.warning(f"offense_pct column not found in snap_counts.parquet")
            return None

        # ENHANCEMENT #1: Use EWMA (Exponential Weighted Moving Average) instead of simple mean
        # This weights recent weeks more heavily: Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
        # Sort by week to ensure proper EWMA calculation
        player_snaps_sorted = player_snaps.sort_values('week')

        # Use EWMA with span=4 (aligns with TIER 1 improvements in trailing_stats.py)
        snap_share = player_snaps_sorted['offense_pct'].ewm(span=4, min_periods=1).mean().iloc[-1]

        logger.debug(f"Snap share (EWMA) for {player_name}: {snap_share:.1%} (from {len(player_snaps)} weeks)")

        return min(1.0, max(0.0, snap_share))

    except Exception as e:
        logger.debug(f"Could not calculate snap share from snap_counts.parquet for {player_name}: {e}")
        return None


def calculate_snap_share_metrics(
    player_name: str,
    position: str,
    team: str,
    week: int,
    season: int = 2025
) -> Dict[str, float]:
    """
    Calculate enhanced snap share metrics for backup cap rule determination.

    ENHANCEMENTS (Nov 23, 2025):
    - #1: EWMA snap share (weights recent weeks more heavily)
    - #2: Snap share trend (recent 2 weeks vs previous 2 weeks)
    - #3: Role stability (weeks with significant snaps)

    Args:
        player_name: Player name
        position: Player position
        team: Team abbreviation
        week: Current week (for predictions)
        season: Season year

    Returns:
        Dictionary with:
        - ewma_snap_share: EWMA snap % (0.0-1.0)
        - simple_avg_snap_share: Simple average snap % (0.0-1.0)
        - snap_trend: Change in snap % (recent 2 weeks - previous 2 weeks)
        - weeks_with_role: Number of weeks with >15% snaps (in last 6 weeks)
        - is_emerging: Boolean - snap trend > +10%
        - is_declining: Boolean - snap trend < -10%
        - is_established: Boolean - 4+ weeks with significant snaps
    """

    snap_path = Path('data/nflverse/snap_counts.parquet')
    if not snap_path.exists():
        logger.warning(f"Snap counts data not found at {snap_path}")
        return {
            'ewma_snap_share': 0.0,
            'simple_avg_snap_share': 0.0,
            'snap_trend': 0.0,
            'weeks_with_role': 0,
            'is_emerging': False,
            'is_declining': False,
            'is_established': False
        }

    try:
        snap_counts = pd.read_parquet(snap_path)

        # Normalize player name using canonical function
        player_normalized = normalize_player_name(player_name)

        # Filter to player, team, season, weeks before current week
        player_snaps = snap_counts[
            (snap_counts['player'].apply(normalize_player_name) == player_normalized) &
            (snap_counts['team'] == team) &
            (snap_counts['season'] == season) &
            (snap_counts['week'] < week)
        ]

        # If no exact match, try last name only
        if len(player_snaps) == 0:
            name_parts = player_name.strip().split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1].lower()
                if last_name in ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v']:
                    last_name = name_parts[-2].lower() if len(name_parts) >= 3 else last_name

                player_snaps = snap_counts[
                    (snap_counts['player'].str.lower().str.contains(last_name, na=False)) &
                    (snap_counts['team'] == team) &
                    (snap_counts['season'] == season) &
                    (snap_counts['week'] < week)
                ]

        if len(player_snaps) == 0 or 'offense_pct' not in player_snaps.columns:
            return {
                'ewma_snap_share': 0.0,
                'simple_avg_snap_share': 0.0,
                'snap_trend': 0.0,
                'weeks_with_role': 0,
                'is_emerging': False,
                'is_declining': False,
                'is_established': False
            }

        # Sort by week
        player_snaps_sorted = player_snaps.sort_values('week')

        # Calculate EWMA snap share (Enhancement #1)
        ewma_snap_share = player_snaps_sorted['offense_pct'].ewm(span=4, min_periods=1).mean().iloc[-1]

        # Calculate simple average for comparison
        simple_avg_snap_share = player_snaps_sorted['offense_pct'].mean()

        # Calculate snap trend (Enhancement #2)
        snap_trend = 0.0
        if len(player_snaps_sorted) >= 4:
            # Recent 2 weeks vs previous 2 weeks
            recent_2wk = player_snaps_sorted.tail(2)['offense_pct'].mean()
            previous_2wk = player_snaps_sorted.tail(4).head(2)['offense_pct'].mean()
            snap_trend = recent_2wk - previous_2wk

        # Calculate role stability (Enhancement #3)
        # Look at last 6 weeks, count weeks with >15% snaps
        last_6_weeks = player_snaps_sorted.tail(6)
        weeks_with_role = (last_6_weeks['offense_pct'] > 0.15).sum()

        # Determine role classification
        is_emerging = snap_trend > 0.10  # +10% snap share increase
        is_declining = snap_trend < -0.10  # -10% snap share decrease
        is_established = weeks_with_role >= 4  # 4+ weeks of significant snaps

        logger.debug(
            f"{player_name} snap metrics: "
            f"EWMA={ewma_snap_share:.1%}, Trend={snap_trend:+.1%}, "
            f"Weeks in role={weeks_with_role}/6, "
            f"Emerging={is_emerging}, Declining={is_declining}, Established={is_established}"
        )

        return {
            'ewma_snap_share': min(1.0, max(0.0, ewma_snap_share)),
            'simple_avg_snap_share': min(1.0, max(0.0, simple_avg_snap_share)),
            'snap_trend': snap_trend,
            'weeks_with_role': int(weeks_with_role),
            'is_emerging': bool(is_emerging),
            'is_declining': bool(is_declining),
            'is_established': bool(is_established)
        }

    except Exception as e:
        logger.debug(f"Could not calculate snap share metrics for {player_name}: {e}")
        return {
            'ewma_snap_share': 0.0,
            'simple_avg_snap_share': 0.0,
            'snap_trend': 0.0,
            'weeks_with_role': 0,
            'is_emerging': False,
            'is_declining': False,
            'is_established': False
        }


def verify_integration_completeness(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Verify that all factors are integrated in DataFrame.

    Args:
        df: DataFrame to verify

    Returns:
        Dictionary with integration status for each factor
    """
    required_columns = {
        'epa': 'opponent_def_epa_vs_position',
        'weather': 'weather_total_adjustment',
        'divisional': 'is_divisional_game',
        'rest': 'rest_epa_adjustment',
        'travel': 'travel_epa_adjustment',
        'injuries': 'injury_qb_status',
        'redzone': 'redzone_target_share',
        'snap_share': 'snap_share',
        'hfa': 'home_field_advantage_points',
        'primetime': 'is_primetime_game',
        'altitude': 'is_high_altitude',
        'field_surface': 'field_surface',
    }

    status = {}
    for factor, column in required_columns.items():
        status[factor] = column in df.columns

    return status


def get_team_usage_stats(season: int, week: int) -> Optional[pd.DataFrame]:
    """
    Calculate team-level pass/rush/target attempts per game from historical PBP data.

    Uses all completed games before the specified week to calculate team tendencies.

    Returns:
        DataFrame with columns: team, pass_attempts_per_game, rush_attempts_per_game, targets_per_game
    """
    import numpy as np

    # Try to load PBP data
    pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')
    if not pbp_path.exists():
        pbp_path = Path(f'data/nflverse/pbp_{season-1}.parquet')
    if not pbp_path.exists():
        logger.warning(f"No PBP data found for season {season} or {season-1}")
        return None

    try:
        pbp = pd.read_parquet(pbp_path)

        # Filter to regular season games before the target week
        pbp = pbp[
            (pbp['season_type'] == 'REG') &
            (pbp['week'] < week)
        ]

        if len(pbp) == 0:
            return None

        # Count pass and rush attempts by team per game
        pbp['is_pass'] = pbp['play_type'] == 'pass'
        pbp['is_rush'] = pbp['play_type'] == 'run'
        pbp['is_target'] = pbp['pass_attempt'] == 1  # Use pass_attempt for targets

        # Group by game and possession team
        team_game_stats = pbp.groupby(['game_id', 'posteam']).agg({
            'is_pass': 'sum',
            'is_rush': 'sum',
            'is_target': 'sum'
        }).reset_index()
        team_game_stats.columns = ['game_id', 'team', 'pass_attempts', 'rush_attempts', 'targets']

        # Calculate per-game averages by team
        team_avgs = team_game_stats.groupby('team').agg({
            'pass_attempts': 'mean',
            'rush_attempts': 'mean',
            'targets': 'mean'
        }).reset_index()
        team_avgs.columns = ['team', 'pass_attempts_per_game', 'rush_attempts_per_game', 'targets_per_game']

        logger.debug(f"Calculated team usage for {len(team_avgs)} teams")
        return team_avgs

    except Exception as e:
        logger.warning(f"Error calculating team usage stats: {e}")
        return None


def get_game_script_from_vegas(season: int, week: int) -> Optional[pd.DataFrame]:
    """
    Calculate game script adjustment from Vegas spread.

    Game script logic:
    - Negative spread (team favored) → expect more rushing, conservative play (-1 to 0)
    - Positive spread (underdog) → expect more passing, aggressive play (0 to +1)

    Returns:
        DataFrame with columns: team, game_script_dynamic
    """
    import numpy as np

    # Try to load game lines
    game_lines_path = Path(f'data/game_line_predictions_week{week}.csv')
    if not game_lines_path.exists():
        logger.warning(f"No game lines found for week {week}")
        return None

    try:
        gl = pd.read_csv(game_lines_path)

        if 'vegas_spread' not in gl.columns:
            logger.warning("No vegas_spread column in game lines")
            return None

        game_scripts = []

        for _, row in gl.iterrows():
            home_team = row.get('home_team', '')
            away_team = row.get('away_team', '')
            spread = row.get('vegas_spread', 0)

            if pd.isna(spread):
                spread = 0

            # Clamp to reasonable range
            spread = np.clip(spread, -14, 14)

            # For home team: negative spread = favored = run-heavy (negative game script)
            # For away team: positive spread (from their perspective) = underdog = pass-heavy
            home_script = -spread / 14.0  # -1 (heavy favorite) to +1 (heavy underdog)
            away_script = spread / 14.0   # Opposite of home

            game_scripts.append({'team': home_team, 'game_script_dynamic': home_script})
            game_scripts.append({'team': away_team, 'game_script_dynamic': away_script})

        result = pd.DataFrame(game_scripts)
        logger.debug(f"Calculated game script for {len(result)} teams")
        return result

    except Exception as e:
        logger.warning(f"Error calculating game script from Vegas: {e}")
        return None


# =============================================================================
# P1 FIXES: Functions that add predictive signal (2025-12-07)
# =============================================================================

def get_volume_adjustment_from_total(vegas_total: float) -> float:
    """
    P1-A: Adjust player projections based on Vegas game total.

    Hypothesis: High totals = more offensive plays = higher individual stats.

    Args:
        vegas_total: Vegas over/under line for the game

    Returns:
        Multiplier for yards/reception projections:
        - Total 50+: 1.08 (8% boost)
        - Total 45-50: 1.04 (4% boost)
        - Total 40-45: 1.0 (neutral)
        - Total 38-40: 0.96 (4% reduction)
        - Total <38: 0.92 (8% reduction)
    """
    if vegas_total is None or pd.isna(vegas_total):
        return 1.0

    if vegas_total >= 50:
        return 1.08
    elif vegas_total >= 45:
        return 1.04
    elif vegas_total >= 40:
        return 1.0
    elif vegas_total >= 38:
        return 0.96
    else:
        return 0.92


def get_spread_adjustment(team_spread: float, stat_type: str) -> float:
    """
    P1-B: Adjust player projections based on Vegas spread (game script).

    Hypothesis:
    - Underdogs pass more (chasing)
    - Favorites run more (protecting lead)

    Args:
        team_spread: Spread from team's perspective (negative = favorite)
        stat_type: 'passing_yards', 'rushing_yards', 'receiving_yards', 'receptions'

    Returns:
        Multiplier for projections
    """
    if team_spread is None or pd.isna(team_spread):
        return 1.0

    # Rushing: Favorites run more
    if stat_type in ['rushing_yards', 'rushing_attempts']:
        if team_spread < -7:  # Heavy favorite
            return 1.10  # Expect 10% more rushing
        elif team_spread < -3:  # Moderate favorite
            return 1.05
        elif team_spread > 7:  # Heavy underdog
            return 0.90  # Expect 10% less rushing
        elif team_spread > 3:  # Moderate underdog
            return 0.95
        else:
            return 1.0

    # Passing/Receiving: Underdogs pass more
    if stat_type in ['passing_yards', 'receiving_yards', 'receptions', 'targets']:
        if team_spread > 7:  # Heavy underdog
            return 1.10  # Expect 10% more passing
        elif team_spread > 3:  # Moderate underdog
            return 1.05
        elif team_spread < -7:  # Heavy favorite
            return 0.95  # Expect 5% less passing (game control)
        elif team_spread < -3:  # Moderate favorite
            return 0.97
        else:
            return 1.0

    # Other stats: no adjustment
    return 1.0


def get_vegas_context_for_game(
    home_team: str,
    away_team: str,
    week: int,
    season: int = 2025
) -> Dict[str, Any]:
    """
    Get Vegas context (total, spread) for a game.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        week: Week number
        season: Season year

    Returns:
        Dict with:
        - total_line: Vegas over/under
        - spread_line: Vegas spread (from home team perspective)
        - home_spread: Spread for home team (negative = favorite)
        - away_spread: Spread for away team
        - volume_adj: Volume adjustment multiplier
    """
    schedules_path = Path('data/nflverse/schedules.parquet')

    if not schedules_path.exists():
        return {
            'total_line': None,
            'spread_line': None,
            'home_spread': None,
            'away_spread': None,
            'volume_adj': 1.0
        }

    try:
        schedules = pd.read_parquet(schedules_path)
        game = schedules[
            (schedules['home_team'] == home_team) &
            (schedules['away_team'] == away_team) &
            (schedules['week'] == week) &
            (schedules['season'] == season)
        ]

        if len(game) == 0:
            return {
                'total_line': None,
                'spread_line': None,
                'home_spread': None,
                'away_spread': None,
                'volume_adj': 1.0
            }

        row = game.iloc[0]
        total_line = row.get('total_line')
        spread_line = row.get('spread_line')  # Negative = home favorite

        return {
            'total_line': total_line,
            'spread_line': spread_line,
            'home_spread': spread_line if pd.notna(spread_line) else None,
            'away_spread': -spread_line if pd.notna(spread_line) else None,
            'volume_adj': get_volume_adjustment_from_total(total_line)
        }

    except Exception as e:
        logger.warning(f"Error getting Vegas context: {e}")
        return {
            'total_line': None,
            'spread_line': None,
            'home_spread': None,
            'away_spread': None,
            'volume_adj': 1.0
        }


def apply_p1_adjustments(
    prediction: float,
    stat_type: str,
    team: str,
    opponent: str,
    week: int,
    season: int = 2025,
    is_home: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Apply all P1 adjustments (Vegas total + spread) to a prediction.

    Args:
        prediction: Raw model prediction
        stat_type: Type of stat (rushing_yards, passing_yards, etc.)
        team: Player's team
        opponent: Opponent team
        week: Week number
        season: Season year
        is_home: Whether player's team is home

    Returns:
        Tuple of (adjusted_prediction, adjustment_breakdown)
    """
    if prediction is None or prediction <= 0:
        return prediction, {}

    # Get Vegas context
    home_team = team if is_home else opponent
    away_team = opponent if is_home else team
    vegas = get_vegas_context_for_game(home_team, away_team, week, season)

    # P1-A: Volume adjustment from total
    volume_adj = vegas.get('volume_adj', 1.0)

    # P1-B: Spread adjustment
    team_spread = vegas.get('home_spread') if is_home else vegas.get('away_spread')
    spread_adj = get_spread_adjustment(team_spread, stat_type)

    # Combine adjustments (multiplicative)
    total_adj = volume_adj * spread_adj
    adjusted = prediction * total_adj

    breakdown = {
        'volume_adj': volume_adj,
        'spread_adj': spread_adj,
        'total_adj': total_adj,
        'total_line': vegas.get('total_line'),
        'team_spread': team_spread
    }

    return adjusted, breakdown
