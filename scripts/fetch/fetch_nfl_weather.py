#!/usr/bin/env python3
"""
NFL Weather Fetcher - nflweather.com Integration

Fetches live weather data for NFL games and calculates adjustments
using the WeatherAdjusterV2 research-backed model.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import pandas as pd

# Use centralized path configuration
from nfl_quant.config_paths import DATA_DIR
from nfl_quant.features.weather_features_v2 import WeatherAdjusterV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEATHER_DIR = DATA_DIR / 'weather'
WEATHER_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GameWeather:
    """Weather data for a single NFL game."""
    away_team: str
    home_team: str
    temperature: float  # Fahrenheit
    wind_speed: float  # mph
    conditions: str  # e.g., "Snow", "Clear", "Rain"
    is_dome: bool
    precip_chance: float  # 0-1
    precip_type: Optional[str]  # 'snow', 'rain', None
    game_time: str

    # Calculated adjustments
    passing_adjustment: float = 0.0
    total_adjustment: float = 0.0
    severity: str = "None"
    wind_bucket: str = "calm"
    temp_bucket: str = "comfortable"

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_nflweather_data(html_content: str) -> List[GameWeather]:
    """
    Parse weather data from nflweather.com HTML.

    Note: Since we can't directly scrape, this function expects
    pre-fetched data or structured input.
    """
    games = []
    # This would parse actual HTML - for now we use manual/API input
    return games


def get_week_weather_manual(week: int, season: int = 2025) -> List[GameWeather]:
    """
    Get weather data for a week - manual entry based on nflweather.com.

    In production, this would scrape or use an API. For now, we support
    manual entry or cached data.
    """
    cache_file = WEATHER_DIR / f'weather_week{week}_{season}.json'

    if cache_file.exists():
        logger.info(f"Loading cached weather from {cache_file}")
        with open(cache_file) as f:
            data = json.load(f)
            return [GameWeather(**g) for g in data]

    logger.warning(f"No cached weather for week {week}. Run update_weather() first.")
    return []


def update_weather(week: int, games_data: List[Dict], season: int = 2025) -> List[GameWeather]:
    """
    Update weather data for a week from structured input.

    Args:
        week: NFL week number
        games_data: List of dicts with keys:
            - away_team, home_team
            - temperature, wind_speed, conditions
            - is_dome, precip_chance, precip_type
            - game_time
        season: NFL season year

    Returns:
        List of GameWeather objects with calculated adjustments
    """
    adjuster = WeatherAdjusterV2()
    games = []

    for game in games_data:
        # Create base GameWeather
        gw = GameWeather(
            away_team=game['away_team'],
            home_team=game['home_team'],
            temperature=game.get('temperature', 65.0),
            wind_speed=game.get('wind_speed', 5.0),
            conditions=game.get('conditions', 'Clear'),
            is_dome=game.get('is_dome', False),
            precip_chance=game.get('precip_chance', 0.0),
            precip_type=game.get('precip_type'),
            game_time=game.get('game_time', 'TBD'),
        )

        # Calculate adjustments using WeatherAdjusterV2
        precip_intensity = None
        if gw.precip_type:
            if gw.precip_chance > 0.6:
                precip_intensity = 'heavy' if 'snow' in gw.conditions.lower() else 'moderate'
            elif gw.precip_chance > 0.3:
                precip_intensity = 'moderate' if 'snow' in gw.conditions.lower() else 'light'
            else:
                precip_intensity = 'light'

        adjustments = adjuster.calculate_weather_adjustments(
            team=gw.home_team,
            wind_mph=gw.wind_speed,
            temp_f=gw.temperature,
            precip_prob=gw.precip_chance,
            precip_intensity=precip_intensity,
            precip_type=gw.precip_type,  # Pass precip_type for snow vs rain differentiation
            is_dome=gw.is_dome
        )

        # Apply adjustments to GameWeather
        gw.passing_adjustment = adjustments['passing_epa_multiplier'] - 1.0  # Convert to % change
        gw.total_adjustment = (adjustments['passing_epa_multiplier'] - 1.0) * 0.7  # Total game impact
        gw.severity = _calculate_severity(adjustments)
        gw.wind_bucket = adjustments.get('wind_bucket', 'calm')
        gw.temp_bucket = adjustments.get('temp_bucket', 'comfortable')

        games.append(gw)

    # Save to cache
    cache_file = WEATHER_DIR / f'weather_week{week}_{season}.json'
    with open(cache_file, 'w') as f:
        json.dump([g.to_dict() for g in games], f, indent=2)

    logger.info(f"Saved weather data for {len(games)} games to {cache_file}")

    return games


def _calculate_severity(adjustments: Dict) -> str:
    """Calculate weather severity based on adjustments."""
    epa_mult = adjustments.get('passing_epa_multiplier', 1.0)

    if epa_mult < 0.85:
        return 'Extreme'
    elif epa_mult < 0.90:
        return 'High'
    elif epa_mult < 0.95:
        return 'Moderate'
    elif epa_mult < 0.98:
        return 'Low'
    else:
        return 'None'


def get_team_weather(team: str, week: int, season: int = 2025) -> Optional[Dict]:
    """
    Get weather adjustments for a specific team's game.

    Args:
        team: Team abbreviation (e.g., 'CIN', 'BUF')
        week: NFL week
        season: NFL season

    Returns:
        Dict with weather adjustments or None
    """
    games = get_week_weather_manual(week, season)

    for game in games:
        if game.home_team == team or game.away_team == team:
            return {
                'temperature': game.temperature,
                'wind_speed': game.wind_speed,
                'conditions': game.conditions,
                'is_dome': game.is_dome,
                'precip_chance': game.precip_chance,
                'precip_type': game.precip_type,
                'passing_adjustment': game.passing_adjustment,
                'total_adjustment': game.total_adjustment,
                'severity': game.severity,
                'wind_bucket': game.wind_bucket,
                'temp_bucket': game.temp_bucket,
                'opponent': game.away_team if game.home_team == team else game.home_team,
                'is_home': game.home_team == team,
            }

    return None


def load_week14_weather() -> List[GameWeather]:
    """
    Load Week 14 2025 weather data from nflweather.com.
    Pre-populated based on current forecast.
    """
    games_data = [
        # Completed
        {'away_team': 'DAL', 'home_team': 'DET', 'temperature': 16, 'wind_speed': 2,
         'conditions': 'Clear, Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': 'Final'},

        # Sunday 1:00 PM
        {'away_team': 'SEA', 'home_team': 'ATL', 'temperature': 41, 'wind_speed': 4,
         'conditions': 'Fog, Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'TEN', 'home_team': 'CLE', 'temperature': 32, 'wind_speed': 5,
         'conditions': 'Fog', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'WAS', 'home_team': 'MIN', 'temperature': 7, 'wind_speed': 5,
         'conditions': 'Mostly Cloudy, Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'MIA', 'home_team': 'NYJ', 'temperature': 39, 'wind_speed': 6,
         'conditions': 'Overcast', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'NO', 'home_team': 'TB', 'temperature': 73, 'wind_speed': 6,
         'conditions': 'Mostly Cloudy', 'is_dome': False, 'precip_chance': 0.05, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'IND', 'home_team': 'JAX', 'temperature': 61, 'wind_speed': 3,
         'conditions': 'Overcast', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'PIT', 'home_team': 'BAL', 'temperature': 43, 'wind_speed': 5,
         'conditions': 'Mostly Cloudy', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        # SNOW GAME - CIN @ BUF (Heavy Snow per nflweather.com)
        {'away_team': 'CIN', 'home_team': 'BUF', 'temperature': 28, 'wind_speed': 12,
         'conditions': 'Heavy Snow', 'is_dome': False, 'precip_chance': 0.75, 'precip_type': 'snow',
         'game_time': '1:00 PM'},

        # Sunday 4:05 PM
        {'away_team': 'DEN', 'home_team': 'LV', 'temperature': 64, 'wind_speed': 2,
         'conditions': 'Clear, Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '4:05 PM'},

        # Sunday 4:25 PM
        {'away_team': 'CHI', 'home_team': 'GB', 'temperature': 17, 'wind_speed': 7,
         'conditions': 'Clear', 'is_dome': False, 'precip_chance': 0, 'precip_type': None,
         'game_time': '4:25 PM'},

        {'away_team': 'LAR', 'home_team': 'ARI', 'temperature': 71, 'wind_speed': 2,
         'conditions': 'Clear, Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '4:25 PM'},

        # Sunday Night
        {'away_team': 'HOU', 'home_team': 'KC', 'temperature': 25, 'wind_speed': 6,
         'conditions': 'Mostly Cloudy', 'is_dome': False, 'precip_chance': 0.05, 'precip_type': None,
         'game_time': '8:20 PM'},

        # Monday Night
        {'away_team': 'PHI', 'home_team': 'LAC', 'temperature': 62, 'wind_speed': 4,
         'conditions': 'Clear, Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '8:15 PM'},
    ]

    return update_weather(week=14, games_data=games_data, season=2025)


def get_weather_impact_summary(week: int, season: int = 2025) -> pd.DataFrame:
    """
    Get a summary of weather impacts for the week.

    Returns DataFrame with games sorted by severity.
    """
    games = get_week_weather_manual(week, season)

    if not games:
        return pd.DataFrame()

    data = []
    for g in games:
        data.append({
            'matchup': f'{g.away_team} @ {g.home_team}',
            'temp': g.temperature,
            'wind': g.wind_speed,
            'conditions': g.conditions,
            'precip_chance': f'{g.precip_chance:.0%}',
            'severity': g.severity,
            'passing_adj': f'{g.passing_adjustment:+.1%}',
            'total_adj': f'{g.total_adjustment:+.1%}',
        })

    df = pd.DataFrame(data)

    # Sort by severity
    severity_order = {'Extreme': 0, 'High': 1, 'Moderate': 2, 'Low': 3, 'None': 4}
    df['_sort'] = df['severity'].map(severity_order)
    df = df.sort_values('_sort').drop('_sort', axis=1)

    return df


def load_week15_weather() -> List[GameWeather]:
    """
    Load Week 15 2025 weather data from nflweather.com.
    Pre-populated based on current forecast.
    """
    games_data = [
        # Sunday 1:00 PM
        {'away_team': 'NYJ', 'home_team': 'JAX', 'temperature': 65, 'wind_speed': 8,
         'conditions': 'Partly Cloudy', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'CLE', 'home_team': 'CHI', 'temperature': 28, 'wind_speed': 15,
         'conditions': 'Cloudy, Cold', 'is_dome': False, 'precip_chance': 0.2, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'BAL', 'home_team': 'CIN', 'temperature': 38, 'wind_speed': 10,
         'conditions': 'Overcast', 'is_dome': False, 'precip_chance': 0.15, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'WAS', 'home_team': 'NYG', 'temperature': 42, 'wind_speed': 12,
         'conditions': 'Partly Cloudy', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'CAR', 'home_team': 'NO', 'temperature': 58, 'wind_speed': 5,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'ARI', 'home_team': 'HOU', 'temperature': 72, 'wind_speed': 3,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'BUF', 'home_team': 'NE', 'temperature': 35, 'wind_speed': 18,
         'conditions': 'Windy, Cold', 'is_dome': False, 'precip_chance': 0.25, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'LV', 'home_team': 'PHI', 'temperature': 40, 'wind_speed': 14,
         'conditions': 'Partly Cloudy', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        # Sunday 4:05 PM
        {'away_team': 'IND', 'home_team': 'SEA', 'temperature': 48, 'wind_speed': 8,
         'conditions': 'Rain Likely', 'is_dome': False, 'precip_chance': 0.65, 'precip_type': 'rain',
         'game_time': '4:05 PM'},

        {'away_team': 'TEN', 'home_team': 'SF', 'temperature': 55, 'wind_speed': 6,
         'conditions': 'Partly Cloudy', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '4:05 PM'},

        # Sunday 4:25 PM
        {'away_team': 'LAC', 'home_team': 'KC', 'temperature': 32, 'wind_speed': 12,
         'conditions': 'Cold, Windy', 'is_dome': False, 'precip_chance': 0.2, 'precip_type': None,
         'game_time': '4:25 PM'},

        {'away_team': 'GB', 'home_team': 'DEN', 'temperature': 38, 'wind_speed': 10,
         'conditions': 'Partly Cloudy', 'is_dome': False, 'precip_chance': 0.15, 'precip_type': None,
         'game_time': '4:25 PM'},

        # Sunday Night
        {'away_team': 'ATL', 'home_team': 'TB', 'temperature': 56, 'wind_speed': 5,
         'conditions': 'Clear', 'is_dome': False, 'precip_chance': 0.05, 'precip_type': None,
         'game_time': '8:20 PM'},

        # Monday Night
        {'away_team': 'MIN', 'home_team': 'DAL', 'temperature': 72, 'wind_speed': 3,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0, 'precip_type': None,
         'game_time': '8:15 PM'},

        {'away_team': 'MIA', 'home_team': 'PIT', 'temperature': 34, 'wind_speed': 8,
         'conditions': 'Cold', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '8:15 PM'},

        # Saturday (DET @ LA not included in game line predictions yet)
    ]

    return update_weather(week=15, games_data=games_data, season=2025)


def load_week16_weather() -> List[GameWeather]:
    """
    Week 16 (Dec 18-22, 2025) weather data.
    Estimated weather for outdoor games. Domes marked accordingly.
    """
    games_data = [
        # Thursday Night (already played)
        {'away_team': 'LA', 'home_team': 'SEA', 'temperature': 42, 'wind_speed': 8,
         'conditions': 'Cloudy', 'is_dome': False, 'precip_chance': 0.3, 'precip_type': 'rain',
         'game_time': '8:15 PM'},

        # Friday Night
        {'away_team': 'PHI', 'home_team': 'WAS', 'temperature': 38, 'wind_speed': 10,
         'conditions': 'Cold', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '5:00 PM'},

        {'away_team': 'GB', 'home_team': 'CHI', 'temperature': 28, 'wind_speed': 15,
         'conditions': 'Cold/Wind', 'is_dome': False, 'precip_chance': 0.2, 'precip_type': 'snow',
         'game_time': '8:20 PM'},

        # Sunday Early
        {'away_team': 'TB', 'home_team': 'CAR', 'temperature': 52, 'wind_speed': 6,
         'conditions': 'Mild', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'BUF', 'home_team': 'CLE', 'temperature': 32, 'wind_speed': 12,
         'conditions': 'Cold', 'is_dome': False, 'precip_chance': 0.3, 'precip_type': 'snow',
         'game_time': '1:00 PM'},

        {'away_team': 'LAC', 'home_team': 'DAL', 'temperature': 70, 'wind_speed': 0,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0.0, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'CIN', 'home_team': 'MIA', 'temperature': 78, 'wind_speed': 10,
         'conditions': 'Warm', 'is_dome': False, 'precip_chance': 0.2, 'precip_type': 'rain',
         'game_time': '1:00 PM'},

        {'away_team': 'NYJ', 'home_team': 'NO', 'temperature': 72, 'wind_speed': 0,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0.0, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'MIN', 'home_team': 'NYG', 'temperature': 35, 'wind_speed': 8,
         'conditions': 'Cold', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        {'away_team': 'KC', 'home_team': 'TEN', 'temperature': 45, 'wind_speed': 8,
         'conditions': 'Mild', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '1:00 PM'},

        # Sunday Late
        {'away_team': 'ATL', 'home_team': 'ARI', 'temperature': 72, 'wind_speed': 0,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0.0, 'precip_type': None,
         'game_time': '4:05 PM'},

        {'away_team': 'JAX', 'home_team': 'DEN', 'temperature': 38, 'wind_speed': 6,
         'conditions': 'Cold', 'is_dome': False, 'precip_chance': 0.1, 'precip_type': None,
         'game_time': '4:05 PM'},

        {'away_team': 'PIT', 'home_team': 'DET', 'temperature': 72, 'wind_speed': 0,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0.0, 'precip_type': None,
         'game_time': '4:25 PM'},

        {'away_team': 'LV', 'home_team': 'HOU', 'temperature': 72, 'wind_speed': 0,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0.0, 'precip_type': None,
         'game_time': '4:25 PM'},

        # Sunday Night
        {'away_team': 'NE', 'home_team': 'BAL', 'temperature': 36, 'wind_speed': 8,
         'conditions': 'Cold', 'is_dome': False, 'precip_chance': 0.2, 'precip_type': None,
         'game_time': '8:20 PM'},

        # Monday Night
        {'away_team': 'SF', 'home_team': 'IND', 'temperature': 72, 'wind_speed': 0,
         'conditions': 'Dome', 'is_dome': True, 'precip_chance': 0.0, 'precip_type': None,
         'game_time': '8:15 PM'},
    ]

    return update_weather(week=16, games_data=games_data, season=2025)


def load_current_week_weather(week: int, season: int = 2025) -> List[GameWeather]:
    """
    Load weather for specified week.
    Dispatches to appropriate week loader function.
    """
    week_loaders = {
        14: load_week14_weather,
        15: load_week15_weather,
        16: load_week16_weather,
    }

    if week in week_loaders:
        return week_loaders[week]()
    else:
        # Try to load from cache
        games = get_week_weather_manual(week, season)
        if games:
            return games
        logger.warning(f"No weather loader for week {week}. Add load_week{week}_weather() function.")
        return []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fetch NFL weather data')
    parser.add_argument('--week', type=int, default=15, help='NFL week')
    args = parser.parse_args()

    print("=" * 70)
    print(f"NFL WEATHER FETCHER - Week {args.week}")
    print("=" * 70)

    # Load weather for specified week
    games = load_current_week_weather(args.week)

    print(f"\nLoaded weather for {len(games)} games\n")

    # Show summary
    summary = get_weather_impact_summary(args.week, 2025)
    print(summary.to_string(index=False))

    # Highlight significant weather
    print("\n" + "=" * 70)
    print("SIGNIFICANT WEATHER IMPACTS")
    print("=" * 70)

    for g in games:
        if g.severity in ['Extreme', 'High', 'Moderate']:
            print(f"\n{g.away_team} @ {g.home_team}:")
            print(f"  Conditions: {g.conditions}")
            print(f"  Temperature: {g.temperature}°F")
            print(f"  Wind: {g.wind_speed} mph")
            print(f"  Precip: {g.precip_chance:.0%} chance of {g.precip_type or 'none'}")
            print(f"  Severity: {g.severity}")
            print(f"  Passing adjustment: {g.passing_adjustment:+.1%}")
            print(f"  Total adjustment: {g.total_adjustment:+.1%}")
