"""
Data Matching Utilities for NFL QUANT Framework

This module provides robust team name normalization and data source matching
to handle variations across different data sources (NFLverse, odds APIs, schedules).

Framework Rule Compliance:
- Single source of truth: NFLverse schedule file
- Fail fast on unrecognized teams
- Comprehensive logging of all mappings
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# TEAM NAME NORMALIZATION
# ==============================================================================

# Canonical team abbreviations (NFLverse standard)
CANONICAL_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
]

# Team name mapping: various formats → canonical abbreviation
TEAM_NAME_MAPPINGS = {
    # Full names (DraftKings/Odds API format)
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Los Angeles Rams': 'LA',
    'Los Angeles Chargers': 'LAC',
    'Las Vegas Raiders': 'LV',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'Seattle Seahawks': 'SEA',
    'San Francisco 49ers': 'SF',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',

    # Alternative formats
    'LA Rams': 'LA',
    'LA Chargers': 'LAC',
    'Las Vegas': 'LV',
    'L.A. Rams': 'LA',
    'L.A. Chargers': 'LAC',
    'San Francisco': 'SF',
    'New England': 'NE',
    'New Orleans': 'NO',
    'New York': None,  # Ambiguous - need Giants or Jets clarification
    'Tampa Bay': 'TB',
    'Kansas City': 'KC',
    'Green Bay': 'GB',

    # Historical names
    'Oakland Raiders': 'LV',
    'San Diego Chargers': 'LAC',
    'St. Louis Rams': 'LA',
    'Washington Redskins': 'WAS',
    'Washington Football Team': 'WAS',

    # Canonical abbreviations (identity mapping)
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
    'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
    'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
    'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC',
    'LA': 'LA', 'LAC': 'LAC', 'LV': 'LV', 'MIA': 'MIA',
    'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO', 'NYG': 'NYG',
    'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT', 'SEA': 'SEA',
    'SF': 'SF', 'TB': 'TB', 'TEN': 'TEN', 'WAS': 'WAS',
}


def normalize_team_name(team_name: str, fail_on_unknown: bool = True) -> Optional[str]:
    """
    Normalize team name to canonical NFLverse abbreviation.

    Args:
        team_name: Team name in any format (full name, abbreviation, etc.)
        fail_on_unknown: If True, raise ValueError on unrecognized team name

    Returns:
        Canonical team abbreviation (e.g., 'BUF', 'KC') or None if unknown

    Raises:
        ValueError: If team_name is unrecognized and fail_on_unknown=True

    Examples:
        >>> normalize_team_name('Buffalo Bills')
        'BUF'
        >>> normalize_team_name('LA Rams')
        'LA'
        >>> normalize_team_name('Kansas City Chiefs')
        'KC'
    """
    if not team_name:
        if fail_on_unknown:
            raise ValueError("Team name cannot be empty")
        return None

    # Strip whitespace
    team_name = team_name.strip()

    # Check direct mapping
    if team_name in TEAM_NAME_MAPPINGS:
        canonical = TEAM_NAME_MAPPINGS[team_name]
        if canonical is None and fail_on_unknown:
            raise ValueError(f"Ambiguous team name: '{team_name}' (need Giants or Jets clarification)")
        return canonical

    # Try case-insensitive match
    for key, value in TEAM_NAME_MAPPINGS.items():
        if key.lower() == team_name.lower():
            if value is None and fail_on_unknown:
                raise ValueError(f"Ambiguous team name: '{team_name}' (need Giants or Jets clarification)")
            return value

    # Unrecognized team
    if fail_on_unknown:
        raise ValueError(
            f"Unrecognized team name: '{team_name}'. "
            f"Valid teams: {', '.join(CANONICAL_TEAMS)}"
        )

    logger.warning(f"Unrecognized team name: '{team_name}'")
    return None


# ==============================================================================
# GAME MATCHING
# ==============================================================================

def create_game_key(home_team: str, away_team: str) -> str:
    """
    Create canonical game key from team abbreviations.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation

    Returns:
        Game key in format "AWAY@HOME" (e.g., "BUF@HOU")

    Examples:
        >>> create_game_key('HOU', 'BUF')
        'BUF@HOU'
    """
    return f"{away_team}@{home_team}"


def match_odds_to_schedule(
    odds_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    week: int,
    season: int = 2025
) -> pd.DataFrame:
    """
    Match odds data to schedule games using team name normalization.

    This function:
    1. Loads the canonical schedule for the specified week/season
    2. Normalizes team names in odds data to NFLverse abbreviations
    3. Creates game keys for both odds and schedule
    4. Joins odds to schedule, flagging mismatches

    Args:
        odds_df: DataFrame with odds data (must have 'home_team', 'away_team' columns)
        schedule_df: DataFrame with schedule data (must have 'home_team', 'away_team', 'week', 'season')
        week: Week number (1-18)
        season: Season year (default: 2025)

    Returns:
        DataFrame with odds matched to schedule, including:
        - All original odds columns
        - schedule_game_id: Official game ID from schedule
        - schedule_home_team: Canonical home team abbreviation
        - schedule_away_team: Canonical away team abbreviation
        - match_status: 'MATCHED', 'WRONG_WEEK', or 'NO_MATCH'

    Raises:
        ValueError: If odds_df is missing required columns
    """
    # Validate odds dataframe
    required_cols = ['home_team', 'away_team']
    missing = [col for col in required_cols if col not in odds_df.columns]
    if missing:
        raise ValueError(f"Odds data missing required columns: {missing}")

    # Filter schedule to specified week/season
    schedule_week = schedule_df[
        (schedule_df['week'] == week) &
        (schedule_df['season'] == season)
    ].copy()

    if len(schedule_week) == 0:
        raise ValueError(f"No games found in schedule for Week {week}, {season}")

    logger.info(f"Matching odds to {len(schedule_week)} scheduled games for Week {week}")

    # Normalize team names in odds data
    odds_df = odds_df.copy()
    odds_df['odds_home_canonical'] = odds_df['home_team'].apply(
        lambda x: normalize_team_name(x, fail_on_unknown=False)
    )
    odds_df['odds_away_canonical'] = odds_df['away_team'].apply(
        lambda x: normalize_team_name(x, fail_on_unknown=False)
    )

    # Create game keys
    odds_df['odds_game_key'] = odds_df.apply(
        lambda row: create_game_key(row['odds_home_canonical'], row['odds_away_canonical'])
        if row['odds_home_canonical'] and row['odds_away_canonical'] else None,
        axis=1
    )

    schedule_week['schedule_game_key'] = schedule_week.apply(
        lambda row: create_game_key(row['home_team'], row['away_team']),
        axis=1
    )

    # Log unrecognized team names
    unrecognized_home = odds_df[odds_df['odds_home_canonical'].isna()]['home_team'].unique()
    unrecognized_away = odds_df[odds_df['odds_away_canonical'].isna()]['away_team'].unique()
    if len(unrecognized_home) > 0:
        logger.warning(f"Unrecognized home teams in odds: {list(unrecognized_home)}")
    if len(unrecognized_away) > 0:
        logger.warning(f"Unrecognized away teams in odds: {list(unrecognized_away)}")

    # Match odds to schedule
    matched = odds_df.merge(
        schedule_week[['game_id', 'schedule_game_key', 'home_team', 'away_team']],
        left_on='odds_game_key',
        right_on='schedule_game_key',
        how='left',
        suffixes=('', '_schedule')
    )

    # Add match status
    matched['match_status'] = matched.apply(
        lambda row: 'MATCHED' if pd.notna(row.get('game_id')) else 'NO_MATCH',
        axis=1
    )

    # Rename schedule columns for clarity
    matched.rename(columns={
        'game_id': 'schedule_game_id',
        'home_team_schedule': 'schedule_home_team',
        'away_team_schedule': 'schedule_away_team'
    }, inplace=True)

    # Log matching statistics
    total_odds = len(odds_df)
    matched_count = (matched['match_status'] == 'MATCHED').sum()
    unmatched_count = total_odds - matched_count

    logger.info(f"Odds matching results:")
    logger.info(f"  Total odds records: {total_odds}")
    logger.info(f"  Matched to schedule: {matched_count} ({matched_count/total_odds*100:.1f}%)")
    logger.info(f"  Unmatched: {unmatched_count} ({unmatched_count/total_odds*100:.1f}%)")

    if unmatched_count > 0:
        logger.warning(f"\nUnmatched games in odds data:")
        unmatched_games = matched[matched['match_status'] == 'NO_MATCH'][
            ['home_team', 'away_team', 'odds_game_key']
        ].drop_duplicates()
        for _, row in unmatched_games.iterrows():
            logger.warning(f"  {row['away_team']} @ {row['home_team']} (key: {row['odds_game_key']})")

        logger.info(f"\nScheduled games for Week {week}:")
        for _, row in schedule_week.iterrows():
            logger.info(f"  {row['schedule_game_key']} (game_id: {row['game_id']})")

    return matched


# ==============================================================================
# COLUMN MATCHING (FLEXIBLE)
# ==============================================================================

# Common column name variations across data sources
COLUMN_ALIASES = {
    # Player identification
    'player': ['player_name', 'player_display_name', 'player', 'name', 'full_name'],
    'player_id': ['player_id', 'gsis_id', 'pfr_id', 'player_gsis_id'],

    # Team identification
    'team': ['team', 'posteam', 'team_abbr', 'team_code', 'recent_team'],
    'opponent': ['opponent', 'defteam', 'opp', 'opponent_abbr'],

    # Game identification
    'game_id': ['game_id', 'old_game_id', 'nflverse_game_id', 'game_key'],
    'week': ['week', 'game_week', 'week_num'],
    'season': ['season', 'year', 'season_year'],

    # Stats
    'targets': ['targets', 'target', 'receiving_targets', 'tgt'],
    'receptions': ['receptions', 'rec', 'catches', 'receiving_rec'],
    'receiving_yards': ['receiving_yards', 'rec_yds', 'rec_yards', 'receiving_yds'],
    'rushing_yards': ['rushing_yards', 'rush_yds', 'rush_yards', 'rushing_yds'],
    'passing_yards': ['passing_yards', 'pass_yds', 'pass_yards', 'passing_yds'],

    # Odds
    'market': ['market', 'bet_type', 'prop_type', 'market_type'],
    'line': ['line', 'prop_line', 'handicap', 'total'],
    'odds': ['odds', 'price', 'odds_american', 'american_odds'],
}


def find_column(df: pd.DataFrame, canonical_name: str, fail_on_missing: bool = True) -> Optional[str]:
    """
    Find column in dataframe using flexible matching.

    Args:
        df: DataFrame to search
        canonical_name: Canonical column name (e.g., 'player', 'team')
        fail_on_missing: If True, raise ValueError if column not found

    Returns:
        Actual column name in dataframe, or None if not found

    Raises:
        ValueError: If column not found and fail_on_missing=True

    Examples:
        >>> df = pd.DataFrame({'player_name': ['Josh Allen'], 'team_abbr': ['BUF']})
        >>> find_column(df, 'player')
        'player_name'
        >>> find_column(df, 'team')
        'team_abbr'
    """
    if canonical_name not in COLUMN_ALIASES:
        if fail_on_missing:
            raise ValueError(
                f"Unknown canonical column name: '{canonical_name}'. "
                f"Valid names: {', '.join(COLUMN_ALIASES.keys())}"
            )
        return None

    # Try each alias
    for alias in COLUMN_ALIASES[canonical_name]:
        if alias in df.columns:
            return alias

    # Not found
    if fail_on_missing:
        raise ValueError(
            f"Column '{canonical_name}' not found in dataframe. "
            f"Tried: {', '.join(COLUMN_ALIASES[canonical_name])}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    return None


def get_canonical_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Get mapping from canonical column names to actual column names in dataframe.

    Args:
        df: DataFrame to search
        column_mapping: Dict of {alias: canonical_name} (e.g., {'player': 'player_name'})

    Returns:
        Dict of {canonical_name: actual_column_name} for columns found in df

    Examples:
        >>> df = pd.DataFrame({'player_name': ['Josh Allen'], 'team': ['BUF']})
        >>> get_canonical_columns(df, {'player': 'player', 'team': 'team'})
        {'player': 'player_name', 'team': 'team'}
    """
    result = {}
    for alias, canonical in column_mapping.items():
        actual = find_column(df, canonical, fail_on_missing=False)
        if actual:
            result[canonical] = actual
    return result


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_week_schedule(week: int, season: int = 2025) -> pd.DataFrame:
    """
    Load schedule for specified week and season.

    Args:
        week: Week number (1-18)
        season: Season year (default: 2025)

    Returns:
        DataFrame with schedule data for the specified week

    Raises:
        FileNotFoundError: If schedule file not found
        ValueError: If no games found for specified week/season
    """
    schedule_path = Path('data/nflverse/schedules.parquet')

    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Schedule file not found: {schedule_path}. "
            "Run scripts/fetch/pull_2024_season_data.py to fetch schedule data."
        )

    schedule = pd.read_parquet(schedule_path)
    week_schedule = schedule[
        (schedule['week'] == week) &
        (schedule['season'] == season)
    ].copy()

    if len(week_schedule) == 0:
        raise ValueError(f"No games found in schedule for Week {week}, {season}")

    logger.info(f"Loaded {len(week_schedule)} games for Week {week}, {season}")
    return week_schedule


def validate_odds_against_schedule(odds_file: Path, week: int, season: int = 2025) -> Tuple[bool, str]:
    """
    Validate that odds file matches the official schedule.

    Args:
        odds_file: Path to odds CSV file
        week: Week number
        season: Season year

    Returns:
        Tuple of (is_valid, message)
        - is_valid: True if all odds match schedule, False otherwise
        - message: Detailed validation message
    """
    try:
        # Load odds and schedule
        odds_df = pd.read_csv(odds_file)
        schedule_df = load_week_schedule(week, season)

        # Match odds to schedule
        matched = match_odds_to_schedule(odds_df, schedule_df, week, season)

        # Calculate match rate
        total = len(matched)
        matched_count = (matched['match_status'] == 'MATCHED').sum()
        match_rate = matched_count / total if total > 0 else 0

        # Determine validity
        is_valid = match_rate >= 0.90  # At least 90% must match

        # Build message
        if is_valid:
            msg = f"✅ Odds file VALID: {matched_count}/{total} records ({match_rate*100:.1f}%) match schedule"
        else:
            msg = f"❌ Odds file INVALID: Only {matched_count}/{total} records ({match_rate*100:.1f}%) match schedule"

            # Add details about mismatches
            unmatched = matched[matched['match_status'] != 'MATCHED']
            unique_games = unmatched[['home_team', 'away_team']].drop_duplicates()
            if len(unique_games) > 0:
                msg += f"\n\nUnmatched games ({len(unique_games)}):"
                for _, row in unique_games.head(10).iterrows():
                    msg += f"\n  {row['away_team']} @ {row['home_team']}"
                if len(unique_games) > 10:
                    msg += f"\n  ... and {len(unique_games) - 10} more"

        return is_valid, msg

    except Exception as e:
        return False, f"❌ Validation failed: {str(e)}"


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=== Team Name Normalization Examples ===")
    examples = [
        'Buffalo Bills',
        'Kansas City Chiefs',
        'LA Rams',
        'Los Angeles Chargers',
        'Green Bay Packers',
        'BUF',
        'KC'
    ]

    for example in examples:
        try:
            canonical = normalize_team_name(example)
            print(f"{example:30} → {canonical}")
        except ValueError as e:
            print(f"{example:30} → ERROR: {e}")

    print("\n=== Game Key Examples ===")
    games = [
        ('HOU', 'BUF'),
        ('BAL', 'NYJ'),
        ('KC', 'IND')
    ]

    for home, away in games:
        key = create_game_key(home, away)
        print(f"{away} @ {home:3} → {key}")
