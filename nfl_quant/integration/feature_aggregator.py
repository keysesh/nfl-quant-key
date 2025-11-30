#!/usr/bin/env python3
"""
Feature Aggregator - Collects All Contextual Features

This module aggregates features from all data sources:
1. Defensive EPA matchups
2. Weather/environment
3. Rest/travel context
4. Snap count trends
5. Injury redistribution
6. Target share velocity
7. Next Gen Stats
8. Team pace

All features are combined into adjustment multipliers that modify
the base statistical projection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import logging
from datetime import datetime

from nfl_quant.integration.enhanced_prediction import (
    AllFeatures,
    DefensiveMatchupFeatures,
    WeatherFeatures,
    RestTravelFeatures,
    SnapCountFeatures,
    InjuryImpactFeatures,
    TargetShareFeatures,
    NGSFeatures,
    TeamPaceFeatures,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NFLVERSE_DIR = DATA_DIR / "nflverse"
INJURIES_DIR = DATA_DIR / "injuries"


class FeatureAggregator:
    """
    Aggregates contextual features from all data sources.

    This is the central hub that pulls together:
    - Defensive matchup quality
    - Weather and environment
    - Rest and travel factors
    - Snap participation trends
    - Injury impact
    - Target share velocity
    - Next Gen Stats
    - Team pace
    """

    def __init__(self):
        """Initialize feature aggregator and load all data sources."""
        self._defensive_epa = None
        self._team_pace = None
        self._schedules = None
        self._snap_counts = None
        self._injuries = None
        self._target_shares = None
        self._ngs_receiving = None
        self._ngs_rushing = None
        self._ngs_passing = None

        # Team location data for travel calculation
        self.team_locations = self._load_team_locations()

        # Weather fetcher for real-time weather data
        self._weather_fetcher = None
        try:
            from nfl_quant.data.weather_fetcher import create_weather_fetcher
            import os
            self._weather_fetcher = create_weather_fetcher()
            # Set API key from environment variable (secure)
            api_key = os.getenv('METEOSTAT_API_KEY', '')
            if api_key:
                self._weather_fetcher.api_key = api_key
                logger.info("Weather fetcher initialized with API key from environment")
            else:
                logger.info("Weather fetcher initialized without API key (using seasonal forecasts)")
        except Exception as e:
            logger.warning(f"Could not initialize weather fetcher: {e}")

        # Load all data sources
        self._load_all_data()

    def _load_all_data(self):
        """Load all data sources with graceful fallbacks."""
        # Defensive EPA
        try:
            path = NFLVERSE_DIR / "team_defensive_epa.parquet"
            if path.exists():
                self._defensive_epa = pd.read_parquet(path)
                logger.info(f"Loaded defensive EPA: {len(self._defensive_epa)} teams")
        except Exception as e:
            logger.warning(f"Could not load defensive EPA: {e}")

        # Team Pace
        try:
            path = NFLVERSE_DIR / "team_pace.parquet"
            if path.exists():
                self._team_pace = pd.read_parquet(path)
                logger.info(f"Loaded team pace: {len(self._team_pace)} teams")
        except Exception as e:
            logger.warning(f"Could not load team pace: {e}")

        # Schedules (for weather, rest, travel)
        try:
            path = NFLVERSE_DIR / "schedules_2024_2025.csv"
            if path.exists():
                self._schedules = pd.read_csv(path)
                logger.info(f"Loaded schedules: {len(self._schedules)} games")
        except Exception as e:
            logger.warning(f"Could not load schedules: {e}")

        # Snap Counts
        try:
            path = NFLVERSE_DIR / "snap_counts_2025.csv"
            if path.exists():
                self._snap_counts = pd.read_csv(path)
                logger.info(f"Loaded snap counts: {len(self._snap_counts)} records")
        except Exception as e:
            logger.warning(f"Could not load snap counts: {e}")

        # Current Injuries (prefer Sleeper latest, fall back to legacy file)
        try:
            injury_paths = [
                INJURIES_DIR / "injuries_latest.csv",
                INJURIES_DIR / "current_injuries.csv",
            ]
            for path in injury_paths:
                if path.exists():
                    self._injuries = pd.read_csv(path)
                    logger.info(f"Loaded injuries from {path.name}: {len(self._injuries)} entries")
                    break
            if self._injuries is None:
                logger.info("No injury file available in data/injuries")
        except Exception as e:
            logger.warning(f"Could not load injuries: {e}")

        # Target Shares
        try:
            path = NFLVERSE_DIR / "player_target_shares.parquet"
            if path.exists():
                self._target_shares = pd.read_parquet(path)
                logger.info(f"Loaded target shares: {len(self._target_shares)} records")
        except Exception as e:
            logger.warning(f"Could not load target shares: {e}")

        # NGS Data
        try:
            for stat_type in ['receiving', 'rushing', 'passing']:
                path = NFLVERSE_DIR / f"ngs_{stat_type}_historical.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    setattr(self, f"_ngs_{stat_type}", df)
                    logger.info(f"Loaded NGS {stat_type}: {len(df)} records")
        except Exception as e:
            logger.warning(f"Could not load NGS data: {e}")

        # Play-by-play for QB connections
        self._pbp = None
        try:
            path = NFLVERSE_DIR / "pbp_2025.parquet"
            if path.exists():
                self._pbp = pd.read_parquet(path)
                logger.info(f"Loaded PBP 2025: {len(self._pbp)} plays")
        except Exception as e:
            logger.warning(f"Could not load PBP data: {e}")

    def _load_team_locations(self) -> Dict[str, Dict[str, float]]:
        """Load team location data for travel calculations."""
        return {
            'ARI': {'lat': 33.5276, 'lon': -112.2626, 'tz_offset': -7},
            'ATL': {'lat': 33.7555, 'lon': -84.4009, 'tz_offset': -5},
            'BAL': {'lat': 39.2780, 'lon': -76.6227, 'tz_offset': -5},
            'BUF': {'lat': 42.7738, 'lon': -78.7870, 'tz_offset': -5},
            'CAR': {'lat': 35.2258, 'lon': -80.8528, 'tz_offset': -5},
            'CHI': {'lat': 41.8623, 'lon': -87.6167, 'tz_offset': -6},
            'CIN': {'lat': 39.0954, 'lon': -84.5161, 'tz_offset': -5},
            'CLE': {'lat': 41.5061, 'lon': -81.6995, 'tz_offset': -5},
            'DAL': {'lat': 32.7473, 'lon': -97.0945, 'tz_offset': -6},
            'DEN': {'lat': 39.7439, 'lon': -105.0201, 'tz_offset': -7},
            'DET': {'lat': 42.3400, 'lon': -83.0456, 'tz_offset': -5},
            'GB': {'lat': 44.5013, 'lon': -88.0622, 'tz_offset': -6},
            'HOU': {'lat': 29.6847, 'lon': -95.4107, 'tz_offset': -6},
            'IND': {'lat': 39.7601, 'lon': -86.1639, 'tz_offset': -5},
            'JAX': {'lat': 30.3239, 'lon': -81.6373, 'tz_offset': -5},
            'KC': {'lat': 39.0489, 'lon': -94.4839, 'tz_offset': -6},
            'LV': {'lat': 36.0909, 'lon': -115.1833, 'tz_offset': -8},
            'LAC': {'lat': 33.9535, 'lon': -118.3390, 'tz_offset': -8},  # Chargers - SoFi Stadium
            'LAR': {'lat': 33.9535, 'lon': -118.3390, 'tz_offset': -8},  # Rams - SoFi Stadium (STANDARD)
            'LA': {'lat': 33.9535, 'lon': -118.3390, 'tz_offset': -8},   # Backward compatibility alias
            'MIA': {'lat': 25.9580, 'lon': -80.2389, 'tz_offset': -5},
            'MIN': {'lat': 44.9738, 'lon': -93.2575, 'tz_offset': -6},
            'NE': {'lat': 42.0909, 'lon': -71.2643, 'tz_offset': -5},
            'NO': {'lat': 29.9511, 'lon': -90.0812, 'tz_offset': -6},
            'NYG': {'lat': 40.8128, 'lon': -74.0742, 'tz_offset': -5},
            'NYJ': {'lat': 40.8128, 'lon': -74.0742, 'tz_offset': -5},
            'PHI': {'lat': 39.9008, 'lon': -75.1675, 'tz_offset': -5},
            'PIT': {'lat': 40.4468, 'lon': -80.0158, 'tz_offset': -5},
            'SF': {'lat': 37.4032, 'lon': -121.9698, 'tz_offset': -8},
            'SEA': {'lat': 47.5952, 'lon': -122.3316, 'tz_offset': -8},
            'TB': {'lat': 27.9759, 'lon': -82.5033, 'tz_offset': -5},
            'TEN': {'lat': 36.1665, 'lon': -86.7713, 'tz_offset': -6},
            'WAS': {'lat': 38.9076, 'lon': -76.8645, 'tz_offset': -5},
        }

    def get_all_features(
        self,
        player_name: str,
        team: str,
        position: str,
        opponent: str,
        week: int,
        season: int = 2025,
        market: str = "receiving_yards",
    ) -> AllFeatures:
        """
        Aggregate all features for a player prediction.

        Args:
            player_name: Player name (e.g., "T.Kelce")
            team: Team abbreviation (e.g., "KC")
            position: Position (QB, RB, WR, TE)
            opponent: Opponent team abbreviation
            week: Week number
            season: Season year
            market: Market type for context

        Returns:
            AllFeatures containing all contextual adjustments
        """
        features = AllFeatures()

        # 1. Defensive Matchup
        features.defensive_matchup = self._get_defensive_matchup_features(
            opponent, position, market
        )

        # 2. Weather/Environment
        features.weather = self._get_weather_features(team, opponent, week, season)

        # 3. Rest/Travel
        features.rest_travel = self._get_rest_travel_features(
            team, opponent, week, season
        )

        # 4. Snap Counts
        features.snap_counts = self._get_snap_count_features(
            player_name, team, week, season
        )

        # 5. Injury Impact
        features.injury_impact = self._get_injury_impact_features(
            player_name, team, position, week, season
        )

        # 6. Target Share
        if position in ['WR', 'TE', 'RB']:
            features.target_share = self._get_target_share_features(
                player_name, team, week, season
            )

        # 7. Next Gen Stats
        features.ngs = self._get_ngs_features(player_name, position, season, week)

        # 8. Team Pace
        features.team_pace = self._get_team_pace_features(team, opponent)

        # 9. QB Connection (for WR/TE)
        qb_conn = self._get_qb_connection_features(player_name, team, position, week, season)
        from nfl_quant.integration.enhanced_prediction import QBConnectionFeatures
        features.qb_connection = QBConnectionFeatures(
            qb_connection_targets=qb_conn['qb_connection_targets'],
            qb_connection_completions=qb_conn['qb_connection_completions'],
            qb_connection_rate=qb_conn['qb_connection_rate'],
            qb_connection_multiplier=qb_conn['qb_connection_multiplier'],
        )

        # 10. Historical Matchup Performance
        hist_matchup = self._get_historical_matchup_features(
            player_name, team, opponent, market, week, season
        )
        from nfl_quant.integration.enhanced_prediction import HistoricalMatchupFeatures
        features.historical_matchup = HistoricalMatchupFeatures(
            vs_opponent_avg=hist_matchup['vs_opponent_avg'],
            vs_opponent_games=hist_matchup['vs_opponent_games'],
            vs_opponent_multiplier=hist_matchup['vs_opponent_multiplier'],
        )

        return features

    def _get_defensive_matchup_features(
        self, opponent: str, position: str, market: str
    ) -> DefensiveMatchupFeatures:
        """Get defensive matchup quality features."""
        features = DefensiveMatchupFeatures(opponent_team=opponent)

        if self._defensive_epa is None:
            return features

        # Get opponent defensive EPA
        opp_data = self._defensive_epa[self._defensive_epa['team'] == opponent]

        if len(opp_data) == 0:
            return features

        def_epa = opp_data['def_epa_allowed'].iloc[0]
        features.opponent_def_epa = float(def_epa)

        # Rank defenses (higher EPA allowed = worse defense = higher rank number)
        all_teams_sorted = self._defensive_epa.sort_values('def_epa_allowed', ascending=True)
        rank = all_teams_sorted.index.tolist().index(opp_data.index[0]) + 1
        features.def_epa_rank = rank

        # Calculate matchup multiplier
        # League average EPA is ~0.0, range is roughly -0.1 to +0.1
        # Positive EPA = bad defense = good matchup
        # Scale: 0.05 EPA difference = 5% adjustment
        league_avg_epa = self._defensive_epa['def_epa_allowed'].mean()
        epa_diff = def_epa - league_avg_epa

        # Adjust based on market type
        if 'pass' in market or 'receiving' in market or 'receptions' in market:
            # Passing/receiving markets more sensitive to pass defense
            multiplier = 1.0 + (epa_diff * 2.0)  # 0.05 EPA diff = 10% adjustment
        elif 'rush' in market or 'carries' in market:
            # Rushing markets
            multiplier = 1.0 + (epa_diff * 1.5)  # 0.05 EPA diff = 7.5% adjustment
        else:
            multiplier = 1.0 + (epa_diff * 1.0)

        # Cap multiplier to reasonable range
        features.matchup_multiplier = float(max(0.85, min(1.20, multiplier)))

        return features

    def _get_weather_features(
        self, team: str, opponent: str, week: int, season: int
    ) -> WeatherFeatures:
        """Get weather and environment features from schedule data or weather fetcher."""
        features = WeatherFeatures()

        # Determine home team
        home_team = None
        away_team = None

        if self._schedules is not None:
            game = self._schedules[
                (self._schedules['season'] == season) &
                (self._schedules['week'] == week) &
                (
                    ((self._schedules['home_team'] == team) & (self._schedules['away_team'] == opponent)) |
                    ((self._schedules['home_team'] == opponent) & (self._schedules['away_team'] == team))
                )
            ]

            if len(game) > 0:
                game = game.iloc[0]
                home_team = game['home_team']
                away_team = game['away_team']
            else:
                # No game found in schedule
                return features
        else:
            # No schedule data
            return features

        # Try weather fetcher first (uses API or cached data)
        weather_fetched = False
        if self._weather_fetcher is not None:
            try:
                weather_data = self._weather_fetcher.get_game_weather(
                    home_team=home_team,
                    away_team=away_team,
                    week=week,
                    season=season
                )

                # CRITICAL: NO HARDCODED DEFAULTS - require actual data from weather fetcher
                if 'temperature' in weather_data:
                    features.temperature = weather_data['temperature']
                    weather_fetched = True
                if 'wind_speed' in weather_data:
                    features.wind_speed = weather_data['wind_speed']
                if 'is_dome' in weather_data:
                    features.is_dome = weather_data['is_dome']

                # Log the source
                source = weather_data.get('source', 'unknown')
                logger.debug(f"Weather for {away_team}@{home_team}: {features.temperature}F, {features.wind_speed}mph wind (source: {source})")

            except Exception as e:
                logger.warning(f"Weather fetcher error: {e}")

        # Fall back to schedule data if weather fetcher didn't provide data
        if not weather_fetched:
            # CRITICAL: NO HARDCODED DEFAULTS - use actual schedule data
            if pd.notna(game.get('temp')):
                features.temperature = float(game['temp'])
            else:
                # Check if dome - domes have controlled environment
                if pd.notna(game.get('roof')):
                    roof = str(game['roof']).lower()
                    if roof in ['dome', 'closed']:
                        features.temperature = 72.0  # Standard dome temperature (actual setting)
                        features.is_dome = True
                    elif roof == 'retractable':
                        # For retractable roofs, we need actual weather data
                        logger.warning(
                            f"No temperature data for {away_team}@{home_team} week {week}. "
                            f"Retractable roof - cannot determine if open/closed. Using neutral 72F."
                        )
                        features.temperature = 72.0
                        features.is_dome = True
                    else:
                        logger.error(
                            f"Missing temperature data for {away_team}@{home_team} week {week}. "
                            f"NO HARDCODED DEFAULTS - update weather data or NFLverse schedules."
                        )
                        # Leave as default 70.0 but log error - this will impact predictions
                else:
                    logger.error(
                        f"Missing temperature and roof data for {away_team}@{home_team} week {week}. "
                        f"NO HARDCODED DEFAULTS - update weather data or NFLverse schedules."
                    )

            if pd.notna(game.get('wind')):
                features.wind_speed = float(game['wind'])
            elif features.is_dome:
                # Domes have no wind - this is actual (not a hardcoded default)
                features.wind_speed = 0.0
            else:
                logger.warning(
                    f"Missing wind data for {away_team}@{home_team} week {week}. "
                    f"NO HARDCODED DEFAULTS - using 0.0 but this may affect predictions."
                )
                # Note: 0.0 wind is a reasonable conservative assumption for outdoor games
                # but should be logged as potentially inaccurate

            if pd.notna(game.get('roof')):
                roof = str(game['roof']).lower()
                features.is_dome = roof in ['dome', 'closed', 'retractable']

        # Calculate wind bucket and multipliers based on final values
        wind = features.wind_speed
        if wind < 10:
            features.wind_bucket = "calm"
            features.passing_epa_multiplier = 1.00
            features.deep_target_multiplier = 1.00
            features.rush_boost = 0.0
        elif wind < 15:
            features.wind_bucket = "moderate"
            features.passing_epa_multiplier = 0.97
            features.deep_target_multiplier = 0.95
            features.rush_boost = 0.02
        elif wind < 20:
            features.wind_bucket = "high"
            features.passing_epa_multiplier = 0.92
            features.deep_target_multiplier = 0.88
            features.rush_boost = 0.03
        else:
            features.wind_bucket = "extreme"
            features.passing_epa_multiplier = 0.85
            features.deep_target_multiplier = 0.75
            features.rush_boost = 0.08

        # Dome advantage
        if features.is_dome:
            features.passing_epa_multiplier = min(features.passing_epa_multiplier * 1.02, 1.05)

        # Temperature adjustments (stacking with wind)
        if features.temperature < 25:
            # Extreme cold
            features.passing_epa_multiplier *= 0.94
        elif features.temperature < 32:
            # Cold
            features.passing_epa_multiplier *= 0.96
        elif features.temperature > 90:
            # Extreme heat (rare)
            features.passing_epa_multiplier *= 0.98

        return features

    def _get_rest_travel_features(
        self, team: str, opponent: str, week: int, season: int
    ) -> RestTravelFeatures:
        """Get rest days and travel context."""
        features = RestTravelFeatures()

        if self._schedules is None:
            return features

        # Find the game
        game = self._schedules[
            (self._schedules['season'] == season) &
            (self._schedules['week'] == week) &
            (
                ((self._schedules['home_team'] == team) & (self._schedules['away_team'] == opponent)) |
                ((self._schedules['home_team'] == opponent) & (self._schedules['away_team'] == team))
            )
        ]

        if len(game) == 0:
            return features

        game = game.iloc[0]

        # Determine if home or away
        is_home = game['home_team'] == team
        features.is_home_game = is_home

        # Days of rest
        if is_home and pd.notna(game.get('home_rest')):
            features.days_rest = int(game['home_rest'])
        elif not is_home and pd.notna(game.get('away_rest')):
            features.days_rest = int(game['away_rest'])

        # Short rest (Thursday night, etc.)
        features.is_short_rest = features.days_rest < 6

        # Bye week return (10+ days rest)
        features.is_bye_week_return = features.days_rest >= 10

        # Rest EPA multiplier
        if features.is_short_rest:
            features.rest_epa_multiplier = 0.975  # -2.5% penalty
        elif features.is_bye_week_return:
            features.rest_epa_multiplier = 1.015  # +1.5% boost
        elif features.days_rest >= 8:
            features.rest_epa_multiplier = 1.01  # +1% for extra rest
        else:
            features.rest_epa_multiplier = 1.0

        # Travel distance (if away game)
        if not is_home:
            if team in self.team_locations and opponent in self.team_locations:
                home_loc = self.team_locations[opponent]
                away_loc = self.team_locations[team]

                # Haversine distance
                lat1, lon1 = np.radians(away_loc['lat']), np.radians(away_loc['lon'])
                lat2, lon2 = np.radians(home_loc['lat']), np.radians(home_loc['lon'])

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance_miles = 3956 * c  # Earth radius in miles

                features.travel_distance_miles = float(distance_miles)

                # Timezone change
                features.timezone_change = abs(away_loc['tz_offset'] - home_loc['tz_offset'])

                # Travel EPA penalty
                if distance_miles > 2000:
                    features.travel_epa_multiplier = 0.985  # -1.5% for coast-to-coast
                elif distance_miles > 1500:
                    features.travel_epa_multiplier = 0.99  # -1%
                elif features.timezone_change >= 2:
                    features.travel_epa_multiplier = 0.995  # -0.5% for timezone shift
                else:
                    features.travel_epa_multiplier = 1.0
            else:
                features.travel_epa_multiplier = 1.0
        else:
            features.travel_epa_multiplier = 1.0

        # Divisional game
        if pd.notna(game.get('div_game')):
            features.is_divisional_game = bool(game['div_game'])

        # Primetime (check weekday)
        if pd.notna(game.get('weekday')):
            weekday = str(game['weekday']).lower()
            features.is_primetime = weekday in ['thursday', 'monday'] or (
                weekday == 'sunday' and pd.notna(game.get('gametime')) and '20:' in str(game['gametime'])
            )

        return features

    def _get_snap_count_features(
        self, player_name: str, team: str, week: int, season: int
    ) -> SnapCountFeatures:
        """Get snap count trends for a player."""
        features = SnapCountFeatures()

        if self._snap_counts is None:
            return features

        # Filter to player and recent weeks
        # Handle different name formats
        player_data = self._snap_counts[
            (self._snap_counts['team'] == team) &
            (self._snap_counts['season'] == season) &
            (self._snap_counts['week'] < week) &
            (self._snap_counts['week'] >= max(1, week - 4))
        ]

        # Try to match player name
        if '.' in player_name:
            # Short format like T.Kelce - extract last name
            last_name = player_name.split('.')[-1]
            player_matches = player_data[
                player_data['player'].str.contains(last_name, case=False, na=False)
            ]
        else:
            player_matches = player_data[
                player_data['player'].str.contains(player_name, case=False, na=False)
            ]

        if len(player_matches) == 0:
            return features

        player_matches = player_matches.sort_values('week')

        # Average snap percentage
        features.avg_offense_pct = float(player_matches['offense_pct'].mean())
        features.recent_offense_pct = float(player_matches['offense_pct'].iloc[-1])

        # Snap trend (regression slope)
        if len(player_matches) >= 2:
            weeks = player_matches['week'].values
            snaps = player_matches['offense_pct'].values
            if len(weeks) > 1:
                slope = np.polyfit(weeks, snaps, 1)[0]
                features.snap_trend = float(slope)

        # Volatility
        features.snap_volatility = float(player_matches['offense_pct'].std())

        # Primary option
        features.is_primary_option = features.avg_offense_pct > 0.5

        # Role change detection (>10% change in snap share)
        if len(player_matches) >= 2:
            recent_avg = player_matches['offense_pct'].tail(2).mean()
            older_avg = player_matches['offense_pct'].head(2).mean()
            if abs(recent_avg - older_avg) > 0.10:
                features.role_change_detected = True

        # Snap share multiplier
        # Increasing snap share = higher projection
        if features.snap_trend > 0.05:  # >5% increase per week
            features.snap_share_multiplier = 1.05
        elif features.snap_trend > 0.02:
            features.snap_share_multiplier = 1.02
        elif features.snap_trend < -0.05:  # Decreasing
            features.snap_share_multiplier = 0.95
        elif features.snap_trend < -0.02:
            features.snap_share_multiplier = 0.98
        else:
            features.snap_share_multiplier = 1.0

        return features

    def _get_injury_impact_features(
        self, player_name: str, team: str, position: str, week: int, season: int
    ) -> InjuryImpactFeatures:
        """Get injury redistribution impact."""
        features = InjuryImpactFeatures()

        if self._injuries is None:
            return features

        # Check player's own injury status
        if '.' in player_name:
            last_name = player_name.split('.')[-1]
            player_injury = self._injuries[
                self._injuries['player_name'].str.contains(last_name, case=False, na=False) &
                (self._injuries['team'] == team)
            ]
        else:
            player_injury = self._injuries[
                self._injuries['player_name'].str.contains(player_name, case=False, na=False) &
                (self._injuries['team'] == team)
            ]

        if len(player_injury) > 0:
            player_injury = player_injury.iloc[0]
            features.player_injury_status = str(player_injury['injury_status'])
            features.player_game_probability = float(player_injury.get('game_probability', 1.0))

        # Check teammates who are OUT
        team_injuries = self._injuries[
            (self._injuries['team'] == team) &
            (self._injuries['injury_status'].isin(['Out', 'Doubtful']))
        ]

        out_teammates = []
        for _, teammate in team_injuries.iterrows():
            out_teammates.append({
                'name': teammate['player_name'],
                'position': teammate['position'],
            })

        features.teammates_out = out_teammates

        # Calculate redistribution boost
        # If a WR is out, remaining WRs get more targets
        if position in ['WR', 'TE']:
            wr_te_out = len([t for t in out_teammates if t['position'] in ['WR', 'TE']])
            if wr_te_out >= 2:
                features.target_share_boost = 0.15  # +15% target share
                features.injury_redistribution_multiplier = 1.10  # +10% projection
            elif wr_te_out == 1:
                features.target_share_boost = 0.08
                features.injury_redistribution_multiplier = 1.05
            else:
                features.injury_redistribution_multiplier = 1.0

        elif position == 'RB':
            rb_out = len([t for t in out_teammates if t['position'] == 'RB'])
            if rb_out >= 1:
                features.carry_share_boost = 0.20
                features.injury_redistribution_multiplier = 1.12
            else:
                features.injury_redistribution_multiplier = 1.0
        else:
            features.injury_redistribution_multiplier = 1.0

        return features

    def _get_target_share_features(
        self, player_name: str, team: str, week: int, season: int
    ) -> TargetShareFeatures:
        """Get target share trends."""
        features = TargetShareFeatures()

        if self._target_shares is None:
            return features

        # Get player's target share history
        player_data = self._target_shares[
            (self._target_shares['posteam'] == team) &
            (self._target_shares['week'] < week) &
            (self._target_shares['week'] >= max(1, week - 4))
        ]

        if len(player_data) == 0:
            return features

        # Need to match player_id - this is tricky without direct mapping
        # For now, use team-level data
        features.current_target_share = float(player_data['target_share'].mean())

        # Trend
        if len(player_data) >= 2:
            weeks = player_data['week'].values
            shares = player_data['target_share'].values
            if len(weeks) > 1 and len(set(weeks)) > 1:
                slope = np.polyfit(weeks, shares, 1)[0]
                features.target_share_trend = float(slope)

        return features

    def _get_ngs_features(
        self, player_name: str, position: str, season: int, week: int
    ) -> NGSFeatures:
        """Get Next Gen Stats features using player_short_name matching."""
        features = NGSFeatures()

        # Match using player_short_name (e.g., "T.Kelce")
        if position in ['WR', 'TE'] and self._ngs_receiving is not None:
            # Get receiving NGS data
            player_ngs = self._ngs_receiving[
                (self._ngs_receiving['player_short_name'] == player_name) &
                (self._ngs_receiving['week'] < week)
            ]

            if len(player_ngs) >= 2:
                # Use trailing 4 weeks
                recent_ngs = player_ngs.nlargest(4, 'week')

                # Separation (higher = better)
                features.avg_separation = float(recent_ngs['avg_separation'].mean())

                # Cushion (yards of cushion at snap)
                features.avg_cushion = float(recent_ngs['avg_cushion'].mean())

                # YAC over expected
                if 'avg_yac_above_expectation' in recent_ngs.columns:
                    features.yac_over_expected = float(recent_ngs['avg_yac_above_expectation'].mean())

                # Calculate skill multiplier
                # Above average separation (league avg ~2.5 yards) = boost
                league_avg_sep = 2.5
                sep_diff = features.avg_separation - league_avg_sep
                sep_mult = 1.0 + (sep_diff * 0.03)  # 0.5 yards extra separation = +1.5%

                # YAC over expected boost
                yac_mult = 1.0 + (features.yac_over_expected * 0.02)  # 1 yard YAC above expected = +2%

                features.ngs_skill_multiplier = float(max(0.90, min(1.15, sep_mult * yac_mult)))

        elif position == 'RB' and self._ngs_rushing is not None:
            # Get rushing NGS data
            player_ngs = self._ngs_rushing[
                (self._ngs_rushing['player_short_name'] == player_name) &
                (self._ngs_rushing['week'] < week)
            ]

            if len(player_ngs) >= 2:
                recent_ngs = player_ngs.nlargest(4, 'week')

                # Rush yards over expected
                if 'rush_yards_over_expected_per_att' in recent_ngs.columns:
                    features.rush_yards_over_expected = float(recent_ngs['rush_yards_over_expected_per_att'].mean())

                # Efficiency
                if 'efficiency' in recent_ngs.columns:
                    efficiency = float(recent_ngs['efficiency'].mean())
                    # Efficiency above 100 = good, below = bad
                    eff_diff = (efficiency - 100) / 100
                    features.ngs_skill_multiplier = float(max(0.90, min(1.15, 1.0 + eff_diff * 0.05)))

        elif position == 'QB' and self._ngs_passing is not None:
            # Get passing NGS data
            player_ngs = self._ngs_passing[
                (self._ngs_passing['player_short_name'] == player_name) &
                (self._ngs_passing['week'] < week)
            ]

            if len(player_ngs) >= 2:
                recent_ngs = player_ngs.nlargest(4, 'week')

                # Time to throw
                if 'avg_time_to_throw' in recent_ngs.columns:
                    features.qb_time_to_throw = float(recent_ngs['avg_time_to_throw'].mean())

                # CPOE (Completion % over expected)
                if 'completion_percentage_above_expectation' in recent_ngs.columns:
                    features.qb_cpoe = float(recent_ngs['completion_percentage_above_expectation'].mean())

                # Aggressiveness
                if 'aggressiveness' in recent_ngs.columns:
                    features.qb_aggressiveness = float(recent_ngs['aggressiveness'].mean())

                # QB skill multiplier based on CPOE
                # League average CPOE is 0, elite QBs have +3 to +5
                cpoe_mult = 1.0 + (features.qb_cpoe / 100 * 0.5)  # +3 CPOE = +1.5% boost
                features.ngs_skill_multiplier = float(max(0.90, min(1.15, cpoe_mult)))

        return features

    def _get_historical_matchup_features(
        self,
        player_name: str,
        team: str,
        opponent: str,
        market: str,
        week: int,
        season: int
    ) -> Dict[str, float]:
        """Get historical performance vs specific opponent."""
        features = {
            'vs_opponent_avg': 0.0,
            'vs_opponent_games': 0,
            'vs_opponent_multiplier': 1.0,
        }

        # Get player stats vs this opponent from historical data
        weekly_data = self.param_provider.weekly_data if hasattr(self, 'param_provider') else None

        if weekly_data is None:
            # Use DynamicParameterProvider
            from nfl_quant.data.dynamic_parameters import get_parameter_provider
            weekly_data = get_parameter_provider().weekly_data

        # Map market to column
        market_col_map = {
            'receptions': 'receptions',
            'receiving_yards': 'receiving_yards',
            'rushing_yards': 'rushing_yards',
            'carries': 'carries',
            'targets': 'targets',
        }
        col = market_col_map.get(market, market)

        if col not in weekly_data.columns:
            return features

        # Get games vs opponent
        vs_opponent = weekly_data[
            (weekly_data['player_name'] == player_name) &
            (weekly_data['opponent_team'] == opponent if 'opponent_team' in weekly_data.columns else True)
        ]

        if len(vs_opponent) >= 1 and col in vs_opponent.columns:
            avg_vs_opp = vs_opponent[col].mean()
            features['vs_opponent_avg'] = float(avg_vs_opp)
            features['vs_opponent_games'] = len(vs_opponent)

            # Compare to player's overall average
            player_overall = weekly_data[weekly_data['player_name'] == player_name]
            if len(player_overall) > 0:
                overall_avg = player_overall[col].mean()
                if overall_avg > 0:
                    # Performance ratio vs this opponent
                    ratio = avg_vs_opp / overall_avg
                    # Cap to reasonable range
                    features['vs_opponent_multiplier'] = float(max(0.85, min(1.20, ratio)))

        return features

    def _get_qb_connection_features(
        self,
        player_name: str,
        team: str,
        position: str,
        week: int,
        season: int
    ) -> Dict[str, float]:
        """Get QB-WR/TE connection chemistry from PBP data."""
        features = {
            'qb_connection_targets': 0,
            'qb_connection_completions': 0,
            'qb_connection_rate': 0.0,
            'qb_connection_multiplier': 1.0,
        }

        if self._pbp is None or position not in ['WR', 'TE']:
            return features

        # Get targets to this player from PBP
        # Filter to pass plays where this player was the target
        player_targets = self._pbp[
            (self._pbp['receiver'] == player_name) &
            (self._pbp['posteam'] == team) &
            (self._pbp['week'] < week)
        ]

        if len(player_targets) == 0:
            # Try partial match
            if '.' in player_name:
                last_name = player_name.split('.')[-1]
                player_targets = self._pbp[
                    (self._pbp['receiver'].str.contains(last_name, case=False, na=False)) &
                    (self._pbp['posteam'] == team) &
                    (self._pbp['week'] < week)
                ]

        if len(player_targets) >= 5:
            features['qb_connection_targets'] = len(player_targets)

            # Calculate completion rate with this specific QB
            completions = player_targets[player_targets['complete_pass'] == 1] if 'complete_pass' in player_targets.columns else player_targets
            features['qb_connection_completions'] = len(completions)

            comp_rate = len(completions) / len(player_targets) if len(player_targets) > 0 else 0
            features['qb_connection_rate'] = float(comp_rate)

            # Calculate air yards per target (proxy for connection strength)
            if 'air_yards' in player_targets.columns:
                avg_air_yards = player_targets['air_yards'].mean()
                # Higher air yards = more trusted downfield = boost
                # League avg ~8-10 yards
                if avg_air_yards > 12:
                    features['qb_connection_multiplier'] = 1.05  # Trusted deep target
                elif avg_air_yards > 10:
                    features['qb_connection_multiplier'] = 1.02
                elif avg_air_yards < 6:
                    features['qb_connection_multiplier'] = 0.98  # Short routes only
                else:
                    features['qb_connection_multiplier'] = 1.0

            # Boost if high target volume (favorite target)
            team_targets = self._pbp[
                (self._pbp['posteam'] == team) &
                (self._pbp['week'] < week) &
                (self._pbp['pass_attempt'] == 1 if 'pass_attempt' in self._pbp.columns else True)
            ]
            if len(team_targets) > 0:
                target_share = len(player_targets) / len(team_targets)
                if target_share > 0.25:  # >25% of team targets
                    features['qb_connection_multiplier'] *= 1.03  # Primary target boost

        return features

    def _get_team_pace_features(self, team: str, opponent: str) -> TeamPaceFeatures:
        """Get team pace (plays per game) features."""
        features = TeamPaceFeatures()

        if self._team_pace is None:
            return features

        # Team pace
        team_data = self._team_pace[self._team_pace['team'] == team]
        if len(team_data) > 0:
            features.team_plays_per_game = float(team_data['plays_per_game'].iloc[0])

        # Opponent pace
        opp_data = self._team_pace[self._team_pace['team'] == opponent]
        if len(opp_data) > 0:
            features.opponent_plays_per_game = float(opp_data['plays_per_game'].iloc[0])

        # Expected game pace (average of both)
        features.expected_game_pace = (features.team_plays_per_game + features.opponent_plays_per_game) / 2

        # Pace multiplier (league average is ~65 plays per team)
        league_avg_pace = 65.0
        if self._team_pace is not None and len(self._team_pace) > 0:
            league_avg_pace = self._team_pace['plays_per_game'].mean()

        pace_diff_pct = (features.team_plays_per_game - league_avg_pace) / league_avg_pace

        # Higher pace = more opportunities = higher projection
        # 10% more plays = 5% higher projection (conservative)
        features.pace_multiplier = float(1.0 + (pace_diff_pct * 0.5))

        # Cap to reasonable range
        features.pace_multiplier = max(0.90, min(1.15, features.pace_multiplier))

        return features

    def apply_all_adjustments(
        self,
        base_mean: float,
        base_std: float,
        features: AllFeatures,
        market: str = "receiving_yards"
    ) -> Tuple[float, float]:
        """
        Apply all feature adjustments to base projection.

        Args:
            base_mean: Base statistical mean
            base_std: Base statistical standard deviation
            features: All contextual features
            market: Market type

        Returns:
            Tuple of (adjusted_mean, adjusted_std)
        """
        # Collect all multipliers
        multipliers = []

        # 1. Defensive matchup
        multipliers.append(features.defensive_matchup.matchup_multiplier)

        # 2. Weather (apply to passing/receiving markets)
        if 'pass' in market or 'receiving' in market or 'receptions' in market:
            multipliers.append(features.weather.passing_epa_multiplier)

        # 3. Rest/Travel
        multipliers.append(features.rest_travel.rest_epa_multiplier)
        multipliers.append(features.rest_travel.travel_epa_multiplier)

        # 4. Snap count trends
        multipliers.append(features.snap_counts.snap_share_multiplier)

        # 5. Injury redistribution
        multipliers.append(features.injury_impact.injury_redistribution_multiplier)

        # 6. Team pace
        multipliers.append(features.team_pace.pace_multiplier)

        # 7. NGS skill (if available)
        multipliers.append(features.ngs.ngs_skill_multiplier)

        # 8. QB Connection (for receiving markets)
        if 'receiving' in market or 'receptions' in market:
            multipliers.append(features.qb_connection.qb_connection_multiplier)

        # 9. Historical matchup performance
        multipliers.append(features.historical_matchup.vs_opponent_multiplier)

        # Apply multiplicatively
        total_multiplier = 1.0
        for mult in multipliers:
            total_multiplier *= mult

        adjusted_mean = base_mean * total_multiplier

        # Adjust std based on uncertainty from contextual factors
        # More extreme adjustments = more uncertainty
        adjustment_magnitude = abs(total_multiplier - 1.0)
        uncertainty_boost = 1.0 + (adjustment_magnitude * 0.5)  # 10% adjustment = 5% more variance

        adjusted_std = base_std * uncertainty_boost

        return adjusted_mean, adjusted_std


def create_feature_aggregator() -> FeatureAggregator:
    """Factory function to create feature aggregator."""
    return FeatureAggregator()
