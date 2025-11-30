"""Player name normalization utilities"""

import re

def normalize_player_name(name):
    """
    Normalize player name for matching across data sources.

    Handles:
    - Jr., Sr., II, III, IV suffixes
    - A.J. vs AJ format
    - Case insensitivity
    - Extra whitespace

    Args:
        name: Player name string

    Returns:
        Normalized lowercase name suitable for matching

    Examples:
        >>> normalize_player_name("Deebo Samuel Sr.")
        'deebo samuel'
        >>> normalize_player_name("A.J. Brown")
        'aj brown'
        >>> normalize_player_name("Josh Allen")
        'josh allen'
    """
    if not isinstance(name, str):
        return ""

    name = name.strip()

    # Skip special cases (defense, team names, etc.)
    if any(x in name.lower() for x in ['d/st', 'defense', 'no touchdown', 'team']):
        return ""

    # Remove suffixes (Jr., Sr., II, III, IV, V)
    suffixes = [
        r'\s+Jr\.?$',
        r'\s+Sr\.?$',
        r'\s+II$',
        r'\s+III$',
        r'\s+IV$',
        r'\s+V$',
    ]
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)

    # Normalize A.J. to AJ (remove periods from initials) - BEFORE lowercasing
    # Handle A.J., D.K., T.J., etc.
    name = re.sub(r'([A-Z])\.(\s*)', r'\1\2', name)

    # Convert to lowercase for matching
    name = name.lower()

    # Remove extra whitespace
    name = ' '.join(name.split())

    # Return the normalized full name (first + last)
    # NFLverse uses full names like "Josh Allen", not abbreviated "J.Allen"
    return name
