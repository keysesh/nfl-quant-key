"""Team name normalization utilities"""

# Comprehensive NFL team name mapping
TEAM_ABBREVIATIONS = {
    # Full names to abbreviations
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",  # Fixed: was "LA", should be "LAR"
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    
    # Abbreviations (return as-is)
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "KC",
    "LV": "LV",
    "LAC": "LAC",
    "LA": "LAR",  # Map LA to LAR for consistency
    "LAR": "LAR",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SF": "SF",
    "SEA": "SEA",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    
    # Historical/alternate names
    "Oakland Raiders": "LV",
    "St. Louis Rams": "LAR",
    "San Diego Chargers": "LAC",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
}


def normalize_team_name(team_name):
    """
    Normalize team name to standard 3-letter abbreviation.
    
    Handles:
    - Full team names ("Kansas City Chiefs" → "KC")
    - Abbreviations ("KC" → "KC")
    - Historical names
    - Case variations
    
    Args:
        team_name: Team name string (full or abbreviated)
        
    Returns:
        Standard 3-letter abbreviation (uppercase)
        
    Examples:
        >>> normalize_team_name("Kansas City Chiefs")
        'KC'
        >>> normalize_team_name("kc")
        'KC'
        >>> normalize_team_name("Washington Commanders")
        'WAS'
    """
    if not isinstance(team_name, str):
        return ""
    
    team_name = team_name.strip()
    
    # Try exact match first (case-insensitive)
    for full_name, abbr in TEAM_ABBREVIATIONS.items():
        if team_name.lower() == full_name.lower():
            return abbr
    
    # Try as abbreviation (uppercase)
    team_upper = team_name.upper()
    if team_upper in TEAM_ABBREVIATIONS.values():
        return team_upper
    
    # If not found, try first 2-3 letters (fallback)
    if len(team_name) <= 3:
        return team_name.upper()
    
    # Unknown team - return first 3 letters uppercase
    return team_name[:3].upper()


def get_all_team_abbreviations():
    """
    Get list of all valid NFL team abbreviations.
    
    Returns:
        Set of 32 NFL team abbreviations
    """
    return set(TEAM_ABBREVIATIONS.values())

