#!/usr/bin/env python3
"""
Fetch weather forecasts for NFL games
Uses Open-Meteo API (free, no API key needed) and stadium locations
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# NFL Stadium Coordinates  # noqa: E501
STADIUM_LOCATIONS = {
    'ARI': {'lat': 33.5276, 'lon': -112.2626,
            'name': 'State Farm Stadium', 'dome': True},
    'ATL': {'lat': 33.7553, 'lon': -84.4006,
            'name': 'Mercedes-Benz Stadium', 'dome': True},
    'BAL': {'lat': 39.2780, 'lon': -76.6227,
            'name': 'M&T Bank Stadium', 'dome': False},
    'BUF': {'lat': 42.7738, 'lon': -78.7870,
            'name': 'Highmark Stadium', 'dome': False},
    'CAR': {'lat': 35.2258, 'lon': -80.8528,
            'name': 'Bank of America Stadium', 'dome': False},
    'CHI': {'lat': 41.8623, 'lon': -87.6167,
            'name': 'Soldier Field', 'dome': False},
    'CIN': {'lat': 39.0954, 'lon': -84.5160,
            'name': 'Paycor Stadium', 'dome': False},
    'CLE': {'lat': 41.5061, 'lon': -81.6995,
            'name': 'Cleveland Browns Stadium', 'dome': False},
    'DAL': {'lat': 32.7473, 'lon': -97.0945,
            'name': 'AT&T Stadium', 'dome': True},
    'DEN': {'lat': 39.7439, 'lon': -105.0201,
            'name': 'Empower Field', 'dome': False, 'altitude': 5280},
    'DET': {'lat': 42.3400, 'lon': -83.0456,
            'name': 'Ford Field', 'dome': True},
    'GB': {'lat': 44.5013, 'lon': -88.0622,
           'name': 'Lambeau Field', 'dome': False},
    'HOU': {'lat': 29.6847, 'lon': -95.4107,
            'name': 'NRG Stadium', 'dome': True},
    'IND': {'lat': 39.7601, 'lon': -86.1639,
            'name': 'Lucas Oil Stadium', 'dome': True},
    'JAX': {'lat': 30.3240, 'lon': -81.6373,
            'name': 'EverBank Stadium', 'dome': False},
    'KC': {'lat': 39.0489, 'lon': -94.4839,
           'name': 'Arrowhead Stadium', 'dome': False},
    'LV': {'lat': 36.0909, 'lon': -115.1833,
           'name': 'Allegiant Stadium', 'dome': True},
    'LAC': {'lat': 33.9534, 'lon': -118.3390,
            'name': 'SoFi Stadium', 'dome': True},
    'LAR': {'lat': 33.9534, 'lon': -118.3390,
            'name': 'SoFi Stadium', 'dome': True},
    'LA': {'lat': 33.9534, 'lon': -118.3390,
           'name': 'SoFi Stadium', 'dome': True},
    'MIA': {'lat': 25.9580, 'lon': -80.2389,
            'name': 'Hard Rock Stadium', 'dome': False},
    'MIN': {'lat': 44.9738, 'lon': -93.2577,
            'name': 'U.S. Bank Stadium', 'dome': True},
    'NE': {'lat': 42.0909, 'lon': -71.2643,
           'name': 'Gillette Stadium', 'dome': False},
    'NO': {'lat': 29.9511, 'lon': -90.0812,
           'name': 'Caesars Superdome', 'dome': True},
    'NYG': {'lat': 40.8128, 'lon': -74.0742,
            'name': 'MetLife Stadium', 'dome': False},
    'NYJ': {'lat': 40.8128, 'lon': -74.0742,
            'name': 'MetLife Stadium', 'dome': False},
    'PHI': {'lat': 39.9008, 'lon': -75.1675,
            'name': 'Lincoln Financial Field', 'dome': False},
    'PIT': {'lat': 40.4468, 'lon': -80.0158,
            'name': 'Acrisure Stadium', 'dome': False},
    'SF': {'lat': 37.4032, 'lon': -121.9698,
           'name': "Levi's Stadium", 'dome': False},
    'SEA': {'lat': 47.5952, 'lon': -122.3316,
            'name': 'Lumen Field', 'dome': False},
    'TB': {'lat': 27.9759, 'lon': -82.5033,
           'name': 'Raymond James Stadium', 'dome': False},
    'TEN': {'lat': 36.1665, 'lon': -86.7713,
            'name': 'Nissan Stadium', 'dome': False},
    'WAS': {'lat': 38.9076, 'lon': -76.8645,
            'name': 'FedExField', 'dome': False},
}


def fetch_weather_forecast(lat, lon, game_time):
    """
    Fetch weather forecast from Open-Meteo API (free, no key needed)

    Args:
        lat: Latitude
        lon: Longitude
        game_time: datetime of game

    Returns:
        dict with weather data
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ('temperature_2m,precipitation,windspeed_10m,'
                   'windgusts_10m,snowfall'),
        'temperature_unit': 'fahrenheit',
        'windspeed_unit': 'mph',
        'precipitation_unit': 'inch',
        'timezone': 'America/New_York'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Find closest hour to game time
        hourly = data['hourly']
        times = pd.to_datetime(hourly['time'])

        # Find index closest to game time
        time_diff = abs(times - game_time)
        closest_idx = time_diff.argmin()

        weather = {
            'temperature': hourly['temperature_2m'][closest_idx],
            'precipitation': hourly['precipitation'][closest_idx],
            'wind_speed': hourly['windspeed_10m'][closest_idx],
            'wind_gusts': hourly['windgusts_10m'][closest_idx],
            'snowfall': hourly['snowfall'][closest_idx],
            'forecast_time': times[closest_idx].isoformat()
        }

        return weather

    except Exception as e:
        print(f"⚠️  Weather API error: {e}")
        return None


def calculate_weather_impact(weather, is_dome):
    """
    Calculate weather impact on game

    Returns:
        dict with impact scores and adjustments
    """
    if is_dome:
        return {
            'total_adjustment': 0.0,
            'passing_adjustment': 0.0,
            'severity': 'None (Dome)',
            'notes': 'Indoor stadium - no weather impact'
        }

    if weather is None:
        return {
            'total_adjustment': 0.0,
            'passing_adjustment': 0.0,
            'severity': 'Unknown',
            'notes': 'Weather data unavailable'
        }

    temp = weather['temperature']
    wind = weather['wind_speed']
    precip = weather['precipitation']
    snow = weather['snowfall']

    # Initialize impacts
    total_adj = 0.0
    passing_adj = 0.0
    notes = []

    # Temperature impact
    if temp < 20:
        total_adj -= 0.08  # Very cold = lower scoring
        passing_adj -= 0.10
        notes.append(f"Extreme cold ({temp}°F)")
    elif temp < 32:
        total_adj -= 0.04
        passing_adj -= 0.05
        notes.append(f"Freezing ({temp}°F)")
    elif temp > 95:
        total_adj -= 0.03  # Extreme heat = fatigue
        notes.append(f"Extreme heat ({temp}°F)")

    # Wind impact (biggest factor for passing)
    if wind > 20:
        total_adj -= 0.10
        passing_adj -= 0.20  # Major passing impact
        notes.append(f"High winds ({wind} mph)")
    elif wind > 15:
        total_adj -= 0.05
        passing_adj -= 0.10
        notes.append(f"Windy ({wind} mph)")
    elif wind > 12:
        passing_adj -= 0.05
        notes.append(f"Breezy ({wind} mph)")

    # Precipitation impact
    if snow > 0.5:
        total_adj -= 0.15  # Snow = major impact
        passing_adj -= 0.20
        notes.append(f"Heavy snow ({snow} in)")
    elif snow > 0.1:
        total_adj -= 0.08
        passing_adj -= 0.12
        notes.append(f"Snow ({snow} in)")
    elif precip > 0.3:
        total_adj -= 0.08  # Heavy rain
        passing_adj -= 0.10
        notes.append(f"Heavy rain ({precip} in/hr)")
    elif precip > 0.1:
        total_adj -= 0.04
        passing_adj -= 0.05
        notes.append(f"Rain ({precip} in/hr)")

    # Determine severity
    if abs(total_adj) > 0.12:
        severity = "Extreme"
    elif abs(total_adj) > 0.06:
        severity = "High"
    elif abs(total_adj) > 0.03:
        severity = "Moderate"
    elif abs(total_adj) > 0:
        severity = "Low"
    else:
        severity = "None"

    return {
        'total_adjustment': round(total_adj, 3),
        'passing_adjustment': round(passing_adj, 3),
        'severity': severity,
        'notes': '; '.join(notes) if notes else 'Good conditions'
    }


def fetch_week_weather(week, year=2025):
    """Fetch weather for all games in a week"""

    print("=" * 80)
    print("NFL WEATHER FORECAST FETCHER")
    print("=" * 80)
    print(f"Week: {week}, Year: {year}")
    print()

    # Load game schedule (you'll need to fetch this or have it)
    # For now, using a simple approach
    print("📅 Note: You need to provide game schedule with times")
    print("   For now, assuming Sunday 1PM ET games")

    # Default game time (Sunday 1PM ET)
    # In production, fetch actual game times from schedule
    sunday_1pm = datetime(year, 10, 27, 13, 0, 0)  # Adjust for actual week

    weather_data = []

    print("\n🌤️  Fetching weather forecasts...")

    # Example: Get weather for a few stadiums
    # First 5 for demo
    for team, location in list(STADIUM_LOCATIONS.items())[:5]:
        print(f"\n{team} - {location['name']}")

        if location['dome']:
            print("   🏟️  Dome stadium - no weather impact")
            weather = None
        else:
            weather = fetch_weather_forecast(
                location['lat'],
                location['lon'],
                sunday_1pm
            )

            if weather:
                print(f"   Temperature: {weather['temperature']}°F")
                print(f"   Wind: {weather['wind_speed']} mph")
                print(f"   Precipitation: {weather['precipitation']} in/hr")

        impact = calculate_weather_impact(weather, location['dome'])

        weather_data.append({
            'team': team,
            'stadium': location['name'],
            'is_dome': location['dome'],
            'temperature': weather['temperature'] if weather else None,
            'wind_speed': weather['wind_speed'] if weather else None,
            'precipitation': weather['precipitation'] if weather else None,
            'snowfall': weather['snowfall'] if weather else None,
            'total_adjustment': impact['total_adjustment'],
            'passing_adjustment': impact['passing_adjustment'],
            'severity': impact['severity'],
            'notes': impact['notes']
        })

    # Create DataFrame
    weather_df = pd.DataFrame(weather_data)

    # Save
    output_dir = Path('data/weather')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'weather_week{week}_{timestamp}.csv'

    weather_df.to_csv(output_file, index=False)

    print(f"\n💾 Saved to: {output_file}")

    # Show significant weather impacts
    print("\n🌧️  SIGNIFICANT WEATHER IMPACTS:")
    significant = weather_df[abs(weather_df['total_adjustment']) > 0.05]

    if len(significant) > 0:
        cols = ['team', 'severity', 'total_adjustment', 'notes']
        print(significant[cols].to_string(index=False))
    else:
        print("   None - all games have good conditions")

    return weather_df


def main():
    import sys

    week = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    fetch_week_weather(week)

    print("\n" + "=" * 80)
    print("⚠️  IMPORTANT NOTES:")
    print("=" * 80)
    print("1. Weather forecasts are most accurate within 3-5 days")
    print("2. Check forecasts again on Friday/Saturday before games")
    print("3. Dome stadiums have no weather impact (marked)")
    print("4. Wind >15mph significantly impacts passing")
    print("5. Snow/rain heavily favors UNDERS")
    print("=" * 80)


if __name__ == "__main__":
    main()
