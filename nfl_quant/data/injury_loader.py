"""
Canonical Injury Data Loader

Single entry point for all injury data access.
Uses Sleeper API as the free data source with strict schema validation.

SAFETY POLICY:
- Injuries can only RESTRICT recommendations (block/penalize)
- Injuries must NEVER BOOST a player
- No automatic role promotion / usage redistribution
- If injury data is missing, default conservative (block OVERs)

Usage:
    from nfl_quant.data.injury_loader import get_injuries, InjuryDataError

    # Get current injuries
    df = get_injuries(season=2025, week=18)

    # Force refresh from Sleeper API
    df = get_injuries(season=2025, week=18, refresh=True)
"""

import pandas as pd
import requests
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# Custom exception for injury data errors
class InjuryDataError(Exception):
    """Raised when injury data cannot be loaded or is invalid."""
    pass


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
INJURY_CACHE_DIR = PROJECT_ROOT / 'data' / 'injuries'
SLEEPER_CACHE_FILE = INJURY_CACHE_DIR / 'sleeper_injuries_latest.parquet'
SLEEPER_METADATA_FILE = INJURY_CACHE_DIR / 'sleeper_metadata.json'
MAX_CACHE_AGE_HOURS = 6  # Injuries can change frequently

# Sleeper API endpoint
SLEEPER_API_URL = "https://api.sleeper.app/v1/players/nfl"

# Required columns after normalization
REQUIRED_COLUMNS = [
    'source', 'updated_at', 'player_key', 'player_name',
    'team', 'pos', 'status', 'risk_score'
]

# Valid injury status values
class InjuryStatus(str, Enum):
    OUT = "OUT"
    DOUBTFUL = "DOUBTFUL"
    QUESTIONABLE = "QUESTIONABLE"
    ACTIVE = "ACTIVE"
    UNKNOWN = "UNKNOWN"


# Match type provenance (in order of reliability)
class InjuryMatchType(str, Enum):
    """How the injury match was made."""
    GSIS_ID = "gsis_id"          # Direct gsis_id match (most reliable)
    EXACT = "exact"              # Exact name + team match
    NORMALIZED = "normalized"    # Normalized name match (handles Jr., III)
    PARTIAL = "partial"          # Last name + first initial
    NONE = "none"                # No match found


class InjuryMatchConfidence(str, Enum):
    """Confidence level in the match."""
    HIGH = "high"    # gsis_id or exact name match
    MEDIUM = "med"   # Normalized name match
    LOW = "low"      # Partial match
    NONE = "none"    # No match


from dataclasses import dataclass


@dataclass
class InjuryMatchResult:
    """Result of matching a player to injury data with provenance."""
    matched: bool
    injury_record: Optional[pd.Series]

    # Provenance fields
    match_type: InjuryMatchType
    match_confidence: InjuryMatchConfidence
    match_source: str  # Description of how match was made

    # For convenience
    @property
    def status(self) -> str:
        """Get injury status or ACTIVE if no match."""
        if self.injury_record is not None:
            return self.injury_record.get('status', InjuryStatus.UNKNOWN.value)
        return InjuryStatus.ACTIVE.value

    @property
    def risk_score(self) -> float:
        """Get risk score or 0.0 if no match."""
        if self.injury_record is not None:
            return self.injury_record.get('risk_score', 0.5)
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for embedding in DataFrames."""
        return {
            'injury_match_type': self.match_type.value,
            'injury_match_confidence': self.match_confidence.value,
            'injury_match_source': self.match_source,
            'injury_matched': self.matched,
            'injury_status': self.status,
            'injury_risk_score': self.risk_score,
        }


def get_injuries(
    season: int = 2025,
    week: Optional[int] = None,
    refresh: bool = False,
    source: str = "sleeper"
) -> pd.DataFrame:
    """
    Get injury data with schema validation.

    Args:
        season: NFL season (for context, Sleeper returns current injuries)
        week: NFL week (for context, Sleeper returns current injuries)
        refresh: Force refetch from API

    Returns:
        Normalized DataFrame with injury data

    Raises:
        InjuryDataError: If data cannot be loaded or is invalid
    """
    if source != "sleeper":
        raise InjuryDataError(f"Unsupported injury source: {source}")

    # Check cache validity
    if not refresh and _is_cache_valid():
        logger.info("Using cached injury data")
        try:
            df = pd.read_parquet(SLEEPER_CACHE_FILE)
            _validate_schema(df)
            return df
        except Exception as e:
            logger.warning(f"Cache read failed: {e}, refetching...")

    # Fetch from Sleeper API
    try:
        df = _fetch_sleeper_injuries()
    except Exception as e:
        # If fetch fails, try to use stale cache
        if SLEEPER_CACHE_FILE.exists():
            logger.warning(f"Fetch failed ({e}), attempting stale cache...")
            try:
                df = pd.read_parquet(SLEEPER_CACHE_FILE)
                # Check if cache is not too old (max 24h for stale)
                metadata = _load_metadata()
                if metadata:
                    cache_age = datetime.now() - datetime.fromisoformat(metadata['retrieved_at'])
                    if cache_age > timedelta(hours=24):
                        raise InjuryDataError(
                            f"Stale cache too old ({cache_age.total_seconds()/3600:.1f}h). "
                            "Cannot use injury data older than 24 hours."
                        )
                logger.warning(f"Using stale cache (fetched: {metadata.get('retrieved_at', 'unknown')})")
                _validate_schema(df)
                return df
            except InjuryDataError:
                raise
            except Exception as cache_error:
                raise InjuryDataError(f"Both fetch and cache failed: {e}, {cache_error}")
        else:
            raise InjuryDataError(f"Fetch failed and no cache available: {e}")

    # Normalize the data
    df = _normalize_sleeper_data(df)

    # Validate schema
    _validate_schema(df)

    # Save to cache
    _save_to_cache(df)

    logger.info(f"Loaded {len(df)} injury records")
    return df


def _is_cache_valid() -> bool:
    """Check if cache file exists and is fresh enough."""
    if not SLEEPER_CACHE_FILE.exists():
        return False

    metadata = _load_metadata()
    if not metadata:
        return False

    try:
        retrieved_at = datetime.fromisoformat(metadata['retrieved_at'])
        cache_age = datetime.now() - retrieved_at
        return cache_age < timedelta(hours=MAX_CACHE_AGE_HOURS)
    except (KeyError, ValueError) as e:
        logger.warning(f"Cache metadata invalid: {e}")
        return False


def _load_metadata() -> Optional[dict]:
    """Load cache metadata."""
    if not SLEEPER_METADATA_FILE.exists():
        return None
    try:
        with open(SLEEPER_METADATA_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Cache metadata JSON invalid: {e}")
        return None
    except OSError as e:
        logger.warning(f"Failed to read cache metadata: {e}")
        return None


def _save_metadata(count: int) -> None:
    """Save cache metadata."""
    metadata = {
        'source': 'sleeper',
        'retrieved_at': datetime.now().isoformat(),
        'record_count': count
    }
    INJURY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SLEEPER_METADATA_FILE, 'w') as f:
        json.dump(metadata, f)


def _fetch_sleeper_injuries() -> pd.DataFrame:
    """Fetch injury data from Sleeper API."""
    logger.info("Fetching injuries from Sleeper API...")

    try:
        response = requests.get(SLEEPER_API_URL, timeout=30)
        response.raise_for_status()
        players = response.json()
    except requests.RequestException as e:
        raise InjuryDataError(f"Sleeper API request failed: {e}")

    logger.info(f"Retrieved {len(players):,} players from Sleeper")

    # Extract injury records
    injury_records = []
    for player_id, player_data in players.items():
        # Only include players with injury status
        injury_status = player_data.get('injury_status')
        if not injury_status:
            continue

        # Skip inactive players
        if player_data.get('status') != 'Active':
            continue

        injury_records.append({
            'sleeper_id': player_id,
            'first_name': player_data.get('first_name', ''),
            'last_name': player_data.get('last_name', ''),
            'team': player_data.get('team', ''),
            'position': player_data.get('position', ''),
            'injury_status': injury_status,
            'injury_body_part': player_data.get('injury_body_part', ''),
            'injury_notes': player_data.get('injury_notes', ''),
            'raw_payload': json.dumps({
                k: v for k, v in player_data.items()
                if k in ['injury_status', 'injury_body_part', 'injury_notes',
                        'injury_start_date', 'status', 'active']
            })
        })

    if not injury_records:
        # Empty but valid - no injuries reported
        logger.info("No injuries found in Sleeper data")
        return pd.DataFrame(columns=[
            'sleeper_id', 'first_name', 'last_name', 'team', 'position',
            'injury_status', 'injury_body_part', 'injury_notes', 'raw_payload'
        ])

    return pd.DataFrame(injury_records)


def _normalize_sleeper_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Sleeper data to canonical schema."""
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ['gsis_id', 'raw_payload'])

    df = df.copy()

    # Build player_name
    df['player_name'] = (
        df['first_name'].fillna('').str.strip() + ' ' +
        df['last_name'].fillna('').str.strip()
    ).str.strip()

    # Build player_key (use sleeper_id as primary key for now)
    df['player_key'] = df['sleeper_id']

    # Normalize status to enum values
    df['status'] = df['injury_status'].apply(_normalize_status)

    # Calculate risk score based on status
    df['risk_score'] = df['status'].apply(_get_risk_score)

    # Normalize team (uppercase)
    df['team'] = df['team'].fillna('').str.strip().str.upper()

    # Normalize position
    df['pos'] = df['position'].fillna('').str.strip().str.upper()

    # Add source and timestamp
    df['source'] = 'sleeper'
    df['updated_at'] = datetime.now().isoformat()

    # Add gsis_id lookup using player resolver
    df['gsis_id'] = df['sleeper_id'].apply(_lookup_gsis_id)

    # Select and reorder columns
    result_columns = REQUIRED_COLUMNS + ['gsis_id', 'raw_payload']
    for col in result_columns:
        if col not in df.columns:
            df[col] = ''

    return df[result_columns]


def _lookup_gsis_id(sleeper_id: str) -> Optional[str]:
    """Look up gsis_id from Sleeper ID using player resolver."""
    try:
        from nfl_quant.data.player_resolver import sleeper_to_gsis, ResolverNotAvailableError
        return sleeper_to_gsis(sleeper_id)
    except ImportError as e:
        logger.warning(f"Could not import player_resolver: {e}")
        return None
    except ResolverNotAvailableError as e:
        logger.debug(f"Resolver unavailable for sleeper_id {sleeper_id}: {e}")
        return None
    except RuntimeError as e:
        logger.warning(f"Resolver runtime error for sleeper_id {sleeper_id}: {e}")
        return None


def _normalize_status(status: Optional[str]) -> str:
    """Normalize injury status to enum value."""
    if not status:
        return InjuryStatus.UNKNOWN.value

    status_lower = str(status).lower().strip()

    if 'out' in status_lower:
        return InjuryStatus.OUT.value
    if 'ir' in status_lower or 'injured reserve' in status_lower:
        return InjuryStatus.OUT.value  # IR = OUT
    if 'pup' in status_lower:
        return InjuryStatus.OUT.value  # PUP = OUT
    if 'doubtful' in status_lower:
        return InjuryStatus.DOUBTFUL.value
    if 'question' in status_lower:
        return InjuryStatus.QUESTIONABLE.value
    if status_lower in ('active', 'healthy', ''):
        return InjuryStatus.ACTIVE.value

    return InjuryStatus.UNKNOWN.value


def _get_risk_score(status: str) -> float:
    """Get risk score [0,1] based on status."""
    scores = {
        InjuryStatus.OUT.value: 1.0,
        InjuryStatus.DOUBTFUL.value: 0.75,
        InjuryStatus.QUESTIONABLE.value: 0.5,
        InjuryStatus.UNKNOWN.value: 0.5,  # Conservative
        InjuryStatus.ACTIVE.value: 0.0,
    }
    return scores.get(status, 0.5)


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required columns and valid data."""
    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise InjuryDataError(f"Missing required columns: {missing}")

    # Check status values are valid (allow empty for no injuries case)
    if len(df) > 0:
        valid_statuses = {s.value for s in InjuryStatus}
        invalid_statuses = set(df['status'].unique()) - valid_statuses
        if invalid_statuses:
            raise InjuryDataError(f"Invalid status values: {invalid_statuses}")

        # Check risk_score is in [0, 1]
        if df['risk_score'].min() < 0 or df['risk_score'].max() > 1:
            raise InjuryDataError(
                f"risk_score out of range [0,1]: "
                f"[{df['risk_score'].min()}, {df['risk_score'].max()}]"
            )

        # Check team is uppercase
        teams_with_lower = df[df['team'].str.contains('[a-z]', na=False, regex=True)]
        if len(teams_with_lower) > 0:
            raise InjuryDataError(f"Team names not uppercase: {teams_with_lower['team'].unique()}")

    logger.info(f"Schema validation passed: {len(df)} records")


def _save_to_cache(df: pd.DataFrame) -> None:
    """Save DataFrame to cache."""
    INJURY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SLEEPER_CACHE_FILE, index=False)
    _save_metadata(len(df))
    logger.info(f"Saved {len(df)} injury records to cache")


def get_injury_freshness() -> dict:
    """Get freshness status of injury data."""
    if not SLEEPER_CACHE_FILE.exists():
        return {
            'exists': False,
            'path': str(SLEEPER_CACHE_FILE),
            'status': 'MISSING'
        }

    metadata = _load_metadata()
    if not metadata:
        return {
            'exists': True,
            'path': str(SLEEPER_CACHE_FILE),
            'status': 'UNKNOWN',
            'note': 'No metadata file'
        }

    try:
        retrieved_at = datetime.fromisoformat(metadata['retrieved_at'])
        cache_age = datetime.now() - retrieved_at
        hours_old = cache_age.total_seconds() / 3600

        return {
            'exists': True,
            'path': str(SLEEPER_CACHE_FILE),
            'retrieved_at': metadata['retrieved_at'],
            'record_count': metadata.get('record_count', 0),
            'hours_old': round(hours_old, 1),
            'status': 'FRESH' if hours_old < MAX_CACHE_AGE_HOURS else 'STALE',
            'threshold_hours': MAX_CACHE_AGE_HOURS
        }
    except Exception as e:
        return {
            'exists': True,
            'path': str(SLEEPER_CACHE_FILE),
            'status': 'ERROR',
            'error': str(e)
        }


def clear_injury_cache() -> None:
    """Clear the injury cache."""
    if SLEEPER_CACHE_FILE.exists():
        SLEEPER_CACHE_FILE.unlink()
    if SLEEPER_METADATA_FILE.exists():
        SLEEPER_METADATA_FILE.unlink()
    logger.info("Cleared injury cache")


def match_player_to_injury_with_provenance(
    player_name: str,
    team: str,
    gsis_id: Optional[str] = None,
    injuries_df: Optional[pd.DataFrame] = None
) -> InjuryMatchResult:
    """
    Match a player to injury data with full provenance tracking.

    Uses multiple matching strategies in order of reliability:
    1. gsis_id exact match (HIGH confidence)
    2. Name + team exact match (HIGH confidence)
    3. Normalized name + team match (MEDIUM confidence)
    4. Partial name match - last name + first initial (LOW confidence)

    Args:
        player_name: Player name from recommendation
        team: Team abbreviation
        gsis_id: NFLverse gsis_id if available
        injuries_df: Pre-loaded injuries DataFrame

    Returns:
        InjuryMatchResult with provenance fields
    """
    # Handle missing injuries DataFrame
    if injuries_df is None:
        try:
            injuries_df = get_injuries()
        except InjuryDataError as e:
            return InjuryMatchResult(
                matched=False,
                injury_record=None,
                match_type=InjuryMatchType.NONE,
                match_confidence=InjuryMatchConfidence.NONE,
                match_source=f"injury_data_unavailable: {e}"
            )

    if injuries_df.empty:
        return InjuryMatchResult(
            matched=False,
            injury_record=None,
            match_type=InjuryMatchType.NONE,
            match_confidence=InjuryMatchConfidence.NONE,
            match_source="no_injuries_reported"
        )

    team_upper = team.upper() if team else ''

    # Strategy 1: gsis_id match (most reliable)
    if gsis_id and 'gsis_id' in injuries_df.columns:
        gsis_matches = injuries_df[injuries_df['gsis_id'] == gsis_id]
        if len(gsis_matches) > 0:
            return InjuryMatchResult(
                matched=True,
                injury_record=gsis_matches.iloc[0],
                match_type=InjuryMatchType.GSIS_ID,
                match_confidence=InjuryMatchConfidence.HIGH,
                match_source=f"gsis_id={gsis_id}"
            )

    # Strategy 2: Exact name + team match
    name_lower = player_name.lower().strip()
    exact_matches = injuries_df[
        (injuries_df['player_name'].str.lower() == name_lower) &
        (injuries_df['team'] == team_upper)
    ]
    if len(exact_matches) > 0:
        return InjuryMatchResult(
            matched=True,
            injury_record=exact_matches.iloc[0],
            match_type=InjuryMatchType.EXACT,
            match_confidence=InjuryMatchConfidence.HIGH,
            match_source=f"exact_name={player_name}, team={team_upper}"
        )

    # Strategy 3: Normalized name match (handles Jr., III, etc.)
    from nfl_quant.data.player_resolver import PlayerResolver
    resolver = PlayerResolver()
    norm_name = resolver._normalize_name(player_name)

    for _, row in injuries_df.iterrows():
        if row['team'] != team_upper:
            continue
        injury_norm = resolver._normalize_name(row['player_name'])
        if norm_name == injury_norm:
            return InjuryMatchResult(
                matched=True,
                injury_record=row,
                match_type=InjuryMatchType.NORMALIZED,
                match_confidence=InjuryMatchConfidence.MEDIUM,
                match_source=f"normalized={norm_name}, injury_name={row['player_name']}"
            )

    # Strategy 4: Partial name match (last name + first initial)
    name_parts = player_name.lower().split()
    if len(name_parts) >= 2:
        last_name = name_parts[-1]
        first_initial = name_parts[0][0] if name_parts[0] else ''

        for _, row in injuries_df.iterrows():
            if row['team'] != team_upper:
                continue
            inj_parts = row['player_name'].lower().split()
            if len(inj_parts) >= 2:
                if (inj_parts[-1] == last_name and
                    inj_parts[0] and inj_parts[0][0] == first_initial):
                    return InjuryMatchResult(
                        matched=True,
                        injury_record=row,
                        match_type=InjuryMatchType.PARTIAL,
                        match_confidence=InjuryMatchConfidence.LOW,
                        match_source=f"partial={first_initial}. {last_name}, injury_name={row['player_name']}"
                    )

    # No match found
    return InjuryMatchResult(
        matched=False,
        injury_record=None,
        match_type=InjuryMatchType.NONE,
        match_confidence=InjuryMatchConfidence.NONE,
        match_source=f"no_match_found: player={player_name}, team={team_upper}"
    )


def match_player_to_injury(
    player_name: str,
    team: str,
    gsis_id: Optional[str] = None,
    injuries_df: Optional[pd.DataFrame] = None
) -> Optional[pd.Series]:
    """
    Match a player from recommendations to injury data.

    BACKWARDS COMPATIBILITY: Returns only the injury record.
    For full provenance, use match_player_to_injury_with_provenance().

    Args:
        player_name: Player name from recommendation
        team: Team abbreviation
        gsis_id: NFLverse gsis_id if available
        injuries_df: Pre-loaded injuries DataFrame

    Returns:
        Injury record (pd.Series) if matched, None otherwise
    """
    result = match_player_to_injury_with_provenance(
        player_name=player_name,
        team=team,
        gsis_id=gsis_id,
        injuries_df=injuries_df
    )
    return result.injury_record if result.matched else None


def get_player_injury_status(
    player_name: str,
    team: Optional[str] = None,
    injuries_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Get injury status for a specific player.

    Args:
        player_name: Player name to look up
        team: Team abbreviation (optional, helps disambiguate)
        injuries_df: Pre-loaded injuries DataFrame (optional)

    Returns:
        Dict with status, risk_score, and other injury info
    """
    if injuries_df is None:
        try:
            injuries_df = get_injuries()
        except InjuryDataError:
            return {
                'status': InjuryStatus.UNKNOWN.value,
                'risk_score': 0.5,
                'found': False,
                'note': 'Could not load injury data'
            }

    if injuries_df.empty:
        return {
            'status': InjuryStatus.ACTIVE.value,
            'risk_score': 0.0,
            'found': False,
            'note': 'No injuries reported'
        }

    # Normalize search name
    search_name = player_name.lower().strip()

    # Search by name
    matches = injuries_df[
        injuries_df['player_name'].str.lower().str.contains(search_name, na=False)
    ]

    # Filter by team if provided
    if team and len(matches) > 1:
        team_upper = team.upper()
        team_matches = matches[matches['team'] == team_upper]
        if len(team_matches) > 0:
            matches = team_matches

    if len(matches) == 0:
        return {
            'status': InjuryStatus.ACTIVE.value,
            'risk_score': 0.0,
            'found': False,
            'note': 'Player not on injury report'
        }

    # Return first match
    row = matches.iloc[0]
    return {
        'status': row['status'],
        'risk_score': row['risk_score'],
        'found': True,
        'player_name': row['player_name'],
        'team': row['team'],
        'pos': row['pos']
    }
