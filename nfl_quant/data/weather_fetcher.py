#!/usr/bin/env python3
"""
NFL Game Weather Fetcher

Fetches weather data for NFL games using the Meteostat Point API.
This provides accurate weather conditions at stadium coordinates.

Based on Tom Bliss's NFL Weather Data methodology.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import logging
import json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEATHER_CACHE_DIR = DATA_DIR / "weather"
NFLVERSE_DIR = DATA_DIR / "nflverse"


# Stadium coordinates for all NFL teams
# Source: Tom Bliss NFL Weather Data
STADIUM_COORDINATES = {
    'ARI': {'lat': 33.5276, 'lon': -112.2626, 'name': 'State Farm Stadium', 'roof': 'retractable'},
    'ATL': {'lat': 33.7555, 'lon': -84.4009, 'name': 'Mercedes-Benz Stadium', 'roof': 'retractable'},
    'BAL': {'lat': 39.2780, 'lon': -76.6227, 'name': 'M&T Bank Stadium', 'roof': 'outdoor'},
    'BUF': {'lat': 42.7738, 'lon': -78.7870, 'name': 'Highmark Stadium', 'roof': 'outdoor'},
    'CAR': {'lat': 35.2258, 'lon': -80.8528, 'name': 'Bank of America Stadium', 'roof': 'outdoor'},
    'CHI': {'lat': 41.8623, 'lon': -87.6167, 'name': 'Soldier Field', 'roof': 'outdoor'},
    'CIN': {'lat': 39.0954, 'lon': -84.5161, 'name': 'Paycor Stadium', 'roof': 'outdoor'},
    'CLE': {'lat': 41.5061, 'lon': -81.6995, 'name': 'Cleveland Browns Stadium', 'roof': 'outdoor'},
    'DAL': {'lat': 32.7473, 'lon': -97.0945, 'name': 'AT&T Stadium', 'roof': 'retractable'},
    'DEN': {'lat': 39.7439, 'lon': -105.0201, 'name': 'Empower Field', 'roof': 'outdoor'},
    'DET': {'lat': 42.3400, 'lon': -83.0456, 'name': 'Ford Field', 'roof': 'dome'},
    'GB': {'lat': 44.5013, 'lon': -88.0622, 'name': 'Lambeau Field', 'roof': 'outdoor'},
    'HOU': {'lat': 29.6847, 'lon': -95.4107, 'name': 'NRG Stadium', 'roof': 'retractable'},
    'IND': {'lat': 39.7601, 'lon': -86.1639, 'name': 'Lucas Oil Stadium', 'roof': 'retractable'},
    'JAX': {'lat': 30.3239, 'lon': -81.6373, 'name': 'TIAA Bank Field', 'roof': 'outdoor'},
    'KC': {'lat': 39.0489, 'lon': -94.4839, 'name': 'Arrowhead Stadium', 'roof': 'outdoor'},
    'LV': {'lat': 36.0909, 'lon': -115.1833, 'name': 'Allegiant Stadium', 'roof': 'dome'},
    'LAC': {'lat': 33.9535, 'lon': -118.3390, 'name': 'SoFi Stadium', 'roof': 'dome'},
    'LAR': {'lat': 33.9535, 'lon': -118.3390, 'name': 'SoFi Stadium', 'roof': 'dome'},  # Standard abbreviation
    'LA': {'lat': 33.9535, 'lon': -118.3390, 'name': 'SoFi Stadium', 'roof': 'dome'},  # Backward compatibility
    'MIA': {'lat': 25.9580, 'lon': -80.2389, 'name': 'Hard Rock Stadium', 'roof': 'outdoor'},
    'MIN': {'lat': 44.9738, 'lon': -93.2575, 'name': 'U.S. Bank Stadium', 'roof': 'dome'},
    'NE': {'lat': 42.0909, 'lon': -71.2643, 'name': 'Gillette Stadium', 'roof': 'outdoor'},
    'NO': {'lat': 29.9511, 'lon': -90.0812, 'name': 'Caesars Superdome', 'roof': 'dome'},
    'NYG': {'lat': 40.8128, 'lon': -74.0742, 'name': 'MetLife Stadium', 'roof': 'outdoor'},
    'NYJ': {'lat': 40.8128, 'lon': -74.0742, 'name': 'MetLife Stadium', 'roof': 'outdoor'},
    'PHI': {'lat': 39.9008, 'lon': -75.1675, 'name': 'Lincoln Financial Field', 'roof': 'outdoor'},
    'PIT': {'lat': 40.4468, 'lon': -80.0158, 'name': 'Acrisure Stadium', 'roof': 'outdoor'},
    'SF': {'lat': 37.4032, 'lon': -121.9698, 'name': "Levi's Stadium", 'roof': 'outdoor'},
    'SEA': {'lat': 47.5952, 'lon': -122.3316, 'name': 'Lumen Field', 'roof': 'outdoor'},
    'TB': {'lat': 27.9759, 'lon': -82.5033, 'name': 'Raymond James Stadium', 'roof': 'outdoor'},
    'TEN': {'lat': 36.1665, 'lon': -86.7713, 'name': 'Nissan Stadium', 'roof': 'outdoor'},
    'WAS': {'lat': 38.9076, 'lon': -76.8645, 'name': 'FedExField', 'roof': 'outdoor'},
}


class WeatherFetcher:
    """
    Fetches and caches weather data for NFL games.

    Uses Meteostat API for historical weather data and provides
    forecasts for upcoming games when available.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the weather fetcher.

        Args:
            api_key: Optional Meteostat API key for higher rate limits
        """
        self.api_key = api_key
        self._weather_cache: Dict[str, Dict[str, Any]] = {}
        self._load_cached_weather()

    def _load_cached_weather(self):
        """Load any cached weather data."""
        cache_file = WEATHER_CACHE_DIR / "game_weather_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._weather_cache = json.load(f)
                logger.info(f"Loaded {len(self._weather_cache)} cached weather entries")
            except Exception as e:
                logger.warning(f"Could not load weather cache: {e}")

    def _save_weather_cache(self):
        """Save weather cache to disk."""
        WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = WEATHER_CACHE_DIR / "game_weather_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._weather_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save weather cache: {e}")

    def get_game_weather(
        self,
        home_team: str,
        away_team: str,
        game_datetime: Optional[datetime] = None,
        week: int = 0,
        season: int = 2025
    ) -> Dict[str, Any]:
        """
        Get weather conditions for a specific game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_datetime: Optional specific game time
            week: Week number
            season: Season year

        Returns:
            Dictionary with weather conditions
        """
        # Check if dome stadium (no weather impact)
        stadium_info = STADIUM_COORDINATES.get(home_team, {})
        roof_type = stadium_info.get('roof', 'outdoor')

        if roof_type in ['dome', 'retractable']:
            # Dome games have controlled climate
            return {
                'temperature': 72.0,  # Controlled temperature
                'wind_speed': 0.0,
                'wind_direction': 0,
                'humidity': 50.0,
                'precipitation': 0.0,
                'is_dome': True,
                'roof_type': roof_type,
                'conditions': 'Dome',
                'source': 'stadium_config',
            }

        # Check cache first
        cache_key = f"{season}_{week}_{home_team}_{away_team}"
        if cache_key in self._weather_cache:
            cached = self._weather_cache[cache_key]
            logger.info(f"Using cached weather for {away_team} @ {home_team}")
            return cached

        # Try to fetch from schedule data first
        weather_data = self._get_weather_from_schedule(home_team, away_team, week, season)

        if weather_data['temperature'] != 70.0 or weather_data['wind_speed'] != 0.0:
            # Found real data in schedule
            self._weather_cache[cache_key] = weather_data
            self._save_weather_cache()
            return weather_data

        # If game is in the past, try to fetch historical data
        if game_datetime and game_datetime < datetime.now():
            weather_data = self._fetch_historical_weather(
                home_team, game_datetime
            )
            if weather_data:
                self._weather_cache[cache_key] = weather_data
                self._save_weather_cache()
                return weather_data

        # For future games, use seasonal averages based on location and month
        weather_data = self._get_seasonal_forecast(home_team, week, season)

        # Cache the forecast
        self._weather_cache[cache_key] = weather_data
        self._save_weather_cache()

        return weather_data

    def _get_weather_from_schedule(
        self,
        home_team: str,
        away_team: str,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """Get weather data from schedule file if available."""
        default_weather = {
            'temperature': 70.0,
            'wind_speed': 0.0,
            'wind_direction': 0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'is_dome': False,
            'roof_type': 'outdoor',
            'conditions': 'Unknown',
            'source': 'default',
        }

        schedule_path = NFLVERSE_DIR / "schedules_2024_2025.csv"
        if not schedule_path.exists():
            return default_weather

        try:
            schedules = pd.read_csv(schedule_path)
            game = schedules[
                (schedules['season'] == season) &
                (schedules['week'] == week) &
                (schedules['home_team'] == home_team) &
                (schedules['away_team'] == away_team)
            ]

            if len(game) == 0:
                return default_weather

            game = game.iloc[0]

            # Get temperature
            if pd.notna(game.get('temp')):
                default_weather['temperature'] = float(game['temp'])
                default_weather['source'] = 'schedule'

            # Get wind
            if pd.notna(game.get('wind')):
                default_weather['wind_speed'] = float(game['wind'])

            # Get roof type
            if pd.notna(game.get('roof')):
                roof = str(game['roof']).lower()
                default_weather['roof_type'] = roof
                default_weather['is_dome'] = roof in ['dome', 'closed']

            # Classify conditions based on available data
            if default_weather['is_dome']:
                default_weather['conditions'] = 'Dome'
            elif default_weather['wind_speed'] >= 20:
                default_weather['conditions'] = 'High Wind'
            elif default_weather['temperature'] < 32:
                default_weather['conditions'] = 'Cold'
            else:
                default_weather['conditions'] = 'Clear'

            return default_weather

        except Exception as e:
            logger.warning(f"Error reading schedule: {e}")
            return default_weather

    def _fetch_historical_weather(
        self,
        home_team: str,
        game_datetime: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical weather from Meteostat API.

        Uses the RapidAPI Meteostat endpoint for Point Hourly data.
        """
        if not self.api_key:
            return None

        try:
            import requests
        except ImportError:
            logger.warning("requests not installed, cannot fetch weather")
            return None

        stadium = STADIUM_COORDINATES.get(home_team, {})
        if not stadium:
            return None

        lat = stadium['lat']
        lon = stadium['lon']

        # Format date for API (YYYY-MM-DD)
        start_date = (game_datetime - timedelta(hours=1)).strftime('%Y-%m-%d')
        end_date = (game_datetime + timedelta(hours=4)).strftime('%Y-%m-%d')

        # RapidAPI endpoint for Meteostat
        url = "https://meteostat.p.rapidapi.com/point/hourly"

        querystring = {
            "lat": str(lat),
            "lon": str(lon),
            "start": start_date,
            "end": end_date,
            "tz": "America/New_York"  # Adjust based on game location
        }

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }

        try:
            response = requests.get(url, headers=headers, params=querystring, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Weather API returned {response.status_code}")
                return None

            data = response.json()
            hourly_data = data.get('data', [])

            if not hourly_data:
                return None

            # Find the hour closest to game time
            game_hour = game_datetime.hour
            best_match = None
            min_diff = 24

            for entry in hourly_data:
                entry_hour = int(entry.get('time', '00:00')[:2])
                diff = abs(entry_hour - game_hour)
                if diff < min_diff:
                    min_diff = diff
                    best_match = entry

            if not best_match:
                return None

            # Convert from metric to imperial
            temp_c = best_match.get('temp', 20)
            temp_f = temp_c * 9/5 + 32

            wind_kmh = best_match.get('wspd', 0)
            wind_mph = wind_kmh * 0.621371

            return {
                'temperature': round(temp_f, 1),
                'wind_speed': round(wind_mph, 1),
                'wind_direction': best_match.get('wdir', 0),
                'humidity': best_match.get('rhum', 50),
                'precipitation': best_match.get('prcp', 0) * 0.0394,  # mm to inches
                'is_dome': False,
                'roof_type': 'outdoor',
                'conditions': self._classify_conditions(temp_f, wind_mph, best_match.get('prcp', 0)),
                'source': 'meteostat_api',
            }

        except Exception as e:
            logger.warning(f"Error fetching weather from API: {e}")
            return None

    def _classify_conditions(self, temp_f: float, wind_mph: float, precip_mm: float) -> str:
        """Classify weather conditions based on data."""
        if precip_mm > 0.3:
            if temp_f <= 34:
                return 'Heavy Snow'
            else:
                return 'Heavy Rain'
        elif precip_mm > 0:
            if temp_f <= 34:
                return 'Light Snow'
            else:
                return 'Light Rain'
        elif wind_mph >= 20:
            return 'High Wind'
        elif temp_f < 32:
            return 'Cold'
        elif temp_f > 85:
            return 'Hot'
        else:
            return 'Clear'

    def _get_seasonal_forecast(
        self,
        home_team: str,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """
        Generate weather forecast based on historical seasonal averages.

        Uses location and time of year to estimate conditions.
        """
        stadium = STADIUM_COORDINATES.get(home_team, {})
        lat = stadium.get('lat', 40.0)

        # Estimate month from week number
        # Week 1 is typically early September
        # Weeks 1-4: September
        # Weeks 5-8: October
        # Weeks 9-13: November
        # Weeks 14-18: December/January

        if week <= 4:
            month = 9  # September
        elif week <= 8:
            month = 10  # October
        elif week <= 13:
            month = 11  # November
        else:
            month = 12  # December

        # Base temperature on latitude and month
        # Northern stadiums (lat > 40) are colder
        # Southern stadiums (lat < 35) are warmer

        # September averages
        if month == 9:
            if lat > 42:  # Northern (GB, BUF, NE)
                base_temp = 65
                wind_avg = 8
            elif lat > 38:  # Mid-Atlantic (PIT, CLE, BAL)
                base_temp = 70
                wind_avg = 7
            else:  # Southern (MIA, TB, JAX)
                base_temp = 82
                wind_avg = 6

        # October
        elif month == 10:
            if lat > 42:
                base_temp = 52
                wind_avg = 10
            elif lat > 38:
                base_temp = 58
                wind_avg = 9
            else:
                base_temp = 75
                wind_avg = 7

        # November
        elif month == 11:
            if lat > 42:
                base_temp = 40
                wind_avg = 12
            elif lat > 38:
                base_temp = 48
                wind_avg = 10
            else:
                base_temp = 68
                wind_avg = 8

        # December/January
        else:
            if lat > 42:
                base_temp = 32
                wind_avg = 14
            elif lat > 38:
                base_temp = 38
                wind_avg = 12
            else:
                base_temp = 62
                wind_avg = 9

        # Add some variance
        temp_variance = np.random.normal(0, 5)
        wind_variance = np.random.normal(0, 3)

        temperature = round(base_temp + temp_variance, 1)
        wind_speed = round(max(0, wind_avg + wind_variance), 1)

        # Determine conditions
        if wind_speed >= 20:
            conditions = 'High Wind'
        elif temperature < 32:
            conditions = 'Cold'
        elif temperature > 85:
            conditions = 'Hot'
        else:
            conditions = 'Clear'

        return {
            'temperature': temperature,
            'wind_speed': wind_speed,
            'wind_direction': 0,  # Would need more data for accurate direction
            'humidity': 50.0,
            'precipitation': 0.0,
            'is_dome': False,
            'roof_type': 'outdoor',
            'conditions': conditions,
            'source': 'seasonal_forecast',
        }

    def get_weather_impact_multipliers(
        self,
        weather_data: Dict[str, Any],
        market: str
    ) -> Dict[str, float]:
        """
        Calculate impact multipliers based on weather conditions.

        Args:
            weather_data: Weather conditions dictionary
            market: Market type (passing, receiving, rushing, etc.)

        Returns:
            Dictionary of impact multipliers
        """
        multipliers = {
            'passing_epa': 1.0,
            'deep_target': 1.0,
            'rush_boost': 0.0,
            'overall': 1.0,
        }

        # Dome = neutral conditions
        if weather_data.get('is_dome', False):
            multipliers['passing_epa'] = 1.02  # Slight dome advantage
            return multipliers

        wind_speed = weather_data.get('wind_speed', 0)
        temperature = weather_data.get('temperature', 70)

        # Wind impact
        if wind_speed < 10:
            # Calm
            pass
        elif wind_speed < 15:
            # Moderate wind
            multipliers['passing_epa'] = 0.97
            multipliers['deep_target'] = 0.95
            multipliers['rush_boost'] = 0.02
        elif wind_speed < 20:
            # High wind
            multipliers['passing_epa'] = 0.92
            multipliers['deep_target'] = 0.88
            multipliers['rush_boost'] = 0.03
        else:
            # Extreme wind
            multipliers['passing_epa'] = 0.85
            multipliers['deep_target'] = 0.75
            multipliers['rush_boost'] = 0.08

        # Temperature impact
        if temperature < 25:
            # Extreme cold
            multipliers['passing_epa'] *= 0.94
        elif temperature < 32:
            # Cold
            multipliers['passing_epa'] *= 0.96
        elif temperature > 90:
            # Extreme heat
            multipliers['passing_epa'] *= 0.98

        # Calculate overall impact for market
        if 'pass' in market or 'receiving' in market or 'receptions' in market:
            multipliers['overall'] = multipliers['passing_epa']
        elif 'rush' in market or 'carries' in market:
            multipliers['overall'] = 1.0 + multipliers['rush_boost']
        else:
            multipliers['overall'] = multipliers['passing_epa']

        return multipliers


def create_weather_fetcher() -> WeatherFetcher:
    """Factory function to create weather fetcher."""
    return WeatherFetcher()


def test_weather_fetcher():
    """Test the weather fetcher."""
    print("=" * 70)
    print("WEATHER FETCHER TEST")
    print("=" * 70)

    fetcher = create_weather_fetcher()

    # Test dome stadium
    print("\n1. Dome Stadium (NO @ Saints):")
    weather = fetcher.get_game_weather('NO', 'ATL', week=11, season=2025)
    print(f"   Temperature: {weather['temperature']}°F")
    print(f"   Wind: {weather['wind_speed']} mph")
    print(f"   Is Dome: {weather['is_dome']}")
    print(f"   Source: {weather['source']}")

    # Test outdoor stadium
    print("\n2. Outdoor Stadium (BUF @ Bills):")
    weather = fetcher.get_game_weather('BUF', 'KC', week=11, season=2025)
    print(f"   Temperature: {weather['temperature']}°F")
    print(f"   Wind: {weather['wind_speed']} mph")
    print(f"   Is Dome: {weather['is_dome']}")
    print(f"   Conditions: {weather['conditions']}")
    print(f"   Source: {weather['source']}")

    # Test Green Bay (cold weather)
    print("\n3. Cold Weather Stadium (GB):")
    weather = fetcher.get_game_weather('GB', 'MIN', week=13, season=2025)
    print(f"   Temperature: {weather['temperature']}°F")
    print(f"   Wind: {weather['wind_speed']} mph")
    print(f"   Conditions: {weather['conditions']}")

    # Test impact multipliers
    print("\n4. Weather Impact Multipliers:")
    test_weather = {
        'temperature': 28,
        'wind_speed': 18,
        'is_dome': False,
    }
    multipliers = fetcher.get_weather_impact_multipliers(test_weather, 'receiving_yards')
    print(f"   Test conditions: 28°F, 18 mph wind")
    print(f"   Passing EPA: {multipliers['passing_epa']:.3f}")
    print(f"   Deep Target: {multipliers['deep_target']:.3f}")
    print(f"   Rush Boost: {multipliers['rush_boost']:.3f}")
    print(f"   Overall Impact: {multipliers['overall']:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_weather_fetcher()
