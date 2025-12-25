"""
Player Lookup Utility - Robust player-to-team mapping using rosters.

Uses rosters.parquet as the source of truth for player teams, which has full
season data including current week (unlike weekly_stats which lags).
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import re

# Cache for player-team mapping
_player_team_cache: Optional[Dict[str, str]] = None
_cache_season: Optional[int] = None


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching.

    Handles variations like:
    - "Travis Etienne Jr." -> "travis etienne jr"
    - "Kenneth Walker III" -> "kenneth walker iii"
    - Extra whitespace
    """
    if not name or pd.isna(name):
        return ''
    # Lowercase and strip
    name = str(name).lower().strip()
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name)
    return name


def get_player_team_map(season: int = 2025, data_dir: Path = None) -> Dict[str, str]:
    """Get player name -> team abbreviation mapping from rosters.

    Args:
        season: NFL season year
        data_dir: Path to data directory (defaults to project data dir)

    Returns:
        Dict mapping player full_name to team abbreviation (e.g., 'JAX', 'LAR')
    """
    global _player_team_cache, _cache_season

    # Return cached if available for same season
    if _player_team_cache is not None and _cache_season == season:
        return _player_team_cache

    # Find data directory
    if data_dir is None:
        # Try to find project root
        current = Path(__file__).resolve()
        for _ in range(5):  # Walk up to 5 levels
            if (current / 'data' / 'nflverse').exists():
                data_dir = current / 'data'
                break
            current = current.parent
        else:
            raise FileNotFoundError("Could not find data/nflverse directory")

    rosters_path = data_dir / 'nflverse' / 'rosters.parquet'
    if not rosters_path.exists():
        raise FileNotFoundError(f"Rosters file not found: {rosters_path}")

    # Load rosters
    rosters = pd.read_parquet(rosters_path)

    # Filter to current season
    season_rosters = rosters[rosters['season'] == season].copy()

    if len(season_rosters) == 0:
        raise ValueError(f"No roster data found for season {season}")

    # Build mapping from full_name to team
    # Use most recent entry per player (in case of trades)
    season_rosters = season_rosters.drop_duplicates(subset=['full_name'], keep='last')

    raw_map = dict(zip(season_rosters['full_name'], season_rosters['team']))

    # Normalize LA -> LAR for Rams consistency
    player_team_map = {}
    for name, team in raw_map.items():
        normalized_team = 'LAR' if team == 'LA' else team
        player_team_map[name] = normalized_team
        # Also add normalized name version for fuzzy matching
        player_team_map[normalize_player_name(name)] = normalized_team

    # Cache it
    _player_team_cache = player_team_map
    _cache_season = season

    return player_team_map


def lookup_player_team(player_name: str, home_team: str = None, away_team: str = None,
                       season: int = 2025, data_dir: Path = None) -> Optional[str]:
    """Look up a player's team from rosters.

    Args:
        player_name: Player name to look up
        home_team: Home team abbreviation (for context, not used for assignment)
        away_team: Away team abbreviation (for context, not used for assignment)
        season: NFL season year
        data_dir: Path to data directory

    Returns:
        Team abbreviation (e.g., 'JAX', 'LAR') or None if not found
    """
    if not player_name or pd.isna(player_name):
        return None

    try:
        player_map = get_player_team_map(season, data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load player map: {e}")
        return None

    # Try exact match first
    if player_name in player_map:
        return player_map[player_name]

    # Try normalized match
    normalized = normalize_player_name(player_name)
    if normalized in player_map:
        return player_map[normalized]

    # Try partial matching for common name variations
    # E.g., "Travis Etienne Jr." vs "Travis Etienne"
    for roster_name, team in player_map.items():
        roster_normalized = normalize_player_name(roster_name)
        # Check if one is substring of other (for Jr., III, etc.)
        if normalized in roster_normalized or roster_normalized in normalized:
            # Verify it's a substantial match (not just "Jr" matching everywhere)
            if len(normalized) > 5 and len(roster_normalized) > 5:
                return team

    # Not found - return None (don't fallback to home_team!)
    return None


def enrich_odds_with_teams(odds_df: pd.DataFrame, season: int = 2025) -> pd.DataFrame:
    """Add correct team column to odds DataFrame.

    Args:
        odds_df: DataFrame with 'player', 'home_team', 'away_team' columns
        season: NFL season year

    Returns:
        DataFrame with 'team' column added (player's actual team)
    """
    df = odds_df.copy()

    # Get player map
    try:
        player_map = get_player_team_map(season)
    except Exception as e:
        print(f"Error loading player map: {e}")
        df['team'] = None
        return df

    # Team name to abbreviation mapping
    team_name_to_abbrev = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
    }

    def get_team_for_player(row):
        player = row.get('player') or row.get('player_name', '')
        if not player:
            return None

        # Look up from rosters
        team = lookup_player_team(player, season=season)
        if team:
            return team_name_to_abbrev.get(team, team) if len(team) > 4 else team

        # Not found - log warning but don't assign home_team
        home = row.get('home_team', '')
        away = row.get('away_team', '')
        print(f"  Warning: Could not find team for player '{player}' (game: {away} @ {home})")
        return None

    df['team'] = df.apply(get_team_for_player, axis=1)

    # Log stats
    found = df['team'].notna().sum()
    total = len(df)
    print(f"  Enriched teams: {found}/{total} players matched ({100*found/total:.1f}%)")

    return df


def clear_cache():
    """Clear the player-team cache."""
    global _player_team_cache, _cache_season
    _player_team_cache = None
    _cache_season = None
