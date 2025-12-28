"""
NFL Position Constants for Betting Analytics

These constants prevent position-based bugs by making position logic explicit and centralized.
Follow the principle: "If it involves position logic, it should reference these constants."

NOTE: Prop type names (e.g., 'rushing_attempts') are BETTING MARKET names, not NFLverse column names.
For actual NFLverse data columns, use:
- `carries` (not `rushing_attempts`) for rush attempts
- `attempts` (not `passing_attempts`) for pass attempts
- `completions` (not `passing_completions`) for completions
"""

# All positions that can catch passes (receiving positions)
# Critical for reception props, receiving yards props, target share calculations
RECEIVING_POSITIONS = ['RB', 'WR', 'TE']

# All positions that can rush (running positions)
# Yes, WR/TE can rush on jet sweeps, end-arounds, trick plays
# QB rushing is increasingly common (Lamar, Hurts, etc.)
# FB (fullback) included for completeness
RUSHING_POSITIONS = ['QB', 'RB', 'FB', 'WR', 'TE']

# All positions that can pass
PASSING_POSITIONS = ['QB']

# Positions that can score anytime TDs
ANYTIME_TD_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K']

# Default fallback values for betting safety
FALLBACK_VALUES = {
    'target_share': 0.10,      # Conservative default for receiving positions
    'carry_share': 0.15,       # Conservative default for RB positions (was 0.65, way too high)
    'snap_share': 0.60,        # Default snap share
    'yards_per_opportunity': 5.0,  # Default efficiency
    'td_rate': 0.05,           # Default TD rate
}

# Minimum thresholds for statistical significance
MIN_THRESHOLDS = {
    'targets': 3,              # Minimum targets to be meaningful
    'carries': 3,              # Minimum carries to be meaningful
    'attempts': 3,             # Minimum pass attempts
}

# Bounds for data validation (catch data errors)
VALIDATION_BOUNDS = {
    'yards_per_carry': (-5, 15),      # Reasonable YPC range
    'yards_per_target': (0, 25),      # Reasonable YPT range
    'completion_rate': (0, 1),        # 0-100%
    'target_share': (0, 1),           # 0-100% of team targets
    'carry_share': (0, 1),            # 0-100% of team carries
    'snap_share': (0, 1),             # 0-100% of team snaps
}

# Position-specific prop types (for validation)
PROP_TYPES_BY_POSITION = {
    'QB': ['passing_yards', 'passing_tds', 'completions', 'attempts', 'interceptions', 
           'rushing_yards', 'rushing_attempts', 'passing_attempts'],
    'RB': ['rushing_yards', 'rushing_attempts', 'rushing_tds',
           'receptions', 'receiving_yards', 'targets', 'receiving_tds'],
    'WR': ['receptions', 'receiving_yards', 'targets', 'receiving_tds'],
    'TE': ['receptions', 'receiving_yards', 'targets', 'receiving_tds'],
    'K': ['field_goals_made', 'kicking_points'],
}

# All betting props that need receiving stats
RECEIVING_PROPS = ['receptions', 'receiving_yards', 'targets', 'receiving_tds']

# All betting props that need rushing stats
RUSHING_PROPS = ['rushing_yards', 'rushing_attempts', 'rushing_tds']

# All betting props that need passing stats
PASSING_PROPS = ['passing_yards', 'passing_tds', 'completions', 'attempts', 'interceptions']

# All betting props that need kicking stats
KICKING_PROPS = ['field_goals_made', 'kicking_points']

def get_required_features_for_prop(prop_type: str) -> set:
    """
    Get the minimum features required for a betting prop.
    
    Args:
        prop_type: Type of prop (e.g., 'receiving_yards', 'rushing_tds')
    
    Returns:
        Set of required feature names
    """
    feature_map = {
        # Receiving props
        'receptions': {'targets', 'target_share', 'snap_share'},
        'receiving_yards': {'targets', 'target_share', 'yards_per_target', 'snap_share'},
        'targets': {'target_share', 'snap_share'},
        'receiving_tds': {'targets', 'target_share', 'td_rate', 'snap_share'},
        
        # Rushing props
        'rushing_yards': {'carry_share', 'yards_per_carry', 'snap_share'},
        'rushing_attempts': {'carry_share', 'snap_share'},
        'rushing_tds': {'carry_share', 'td_rate', 'snap_share'},
        
        # Passing props
        'passing_yards': {'attempts', 'completion_pct', 'yards_per_completion'},
        'passing_tds': {'attempts', 'td_rate'},
        'completions': {'attempts', 'completion_pct'},
        'attempts': {'snap_share'},
    }
    
    return feature_map.get(prop_type, set())


def get_positions_for_prop(prop_type: str) -> list:
    """
    Get all positions that can be bet on for a given prop type.
    
    Args:
        prop_type: Type of prop (e.g., 'receiving_yards')
    
    Returns:
        List of valid positions for this prop
    """
    prop_to_positions = {
        # Receiving
        'receptions': RECEIVING_POSITIONS,
        'receiving_yards': RECEIVING_POSITIONS,
        'targets': RECEIVING_POSITIONS,
        'receiving_tds': RECEIVING_POSITIONS,
        
        # Rushing
        'rushing_yards': RUSHING_POSITIONS,
        'rushing_attempts': RUSHING_POSITIONS,
        'rushing_tds': RUSHING_POSITIONS,
        
        # Passing
        'passing_yards': PASSING_POSITIONS,
        'passing_tds': PASSING_POSITIONS,
        'completions': PASSING_POSITIONS,
        'attempts': PASSING_POSITIONS,
        'interceptions': PASSING_POSITIONS,
        
        # Kicking
        'field_goals_made': ['K'],
        'kicking_points': ['K'],
    }
    
    return prop_to_positions.get(prop_type, [])

