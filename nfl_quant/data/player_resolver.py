"""
Player Identity Resolver

Maps between different player ID systems:
- Sleeper IDs → NFLverse gsis_ids
- Player names → IDs (with normalization)

Handles name variations:
- Jr., II, III suffixes
- Nicknames (Mike/Michael, etc.)
- Hyphenated names
- Special characters

Network Determinism:
- ONLINE mode: Can fetch from Sleeper API (Phase 0 only)
- OFFLINE mode: Must use snapshot, fails if unavailable (production runs)
- CONSERVATIVE mode: Use snapshot if available, mark unavailable if not

Usage:
    # Standard usage (online, auto-caches)
    resolver = PlayerResolver()
    gsis_id = resolver.sleeper_to_gsis("1408")

    # Production run (offline, from snapshot)
    resolver = PlayerResolver.from_snapshot(Path("runs/123/inputs/players.parquet"))

    # Phase 0 - fetch and save snapshot
    resolver = PlayerResolver(mode=ResolverMode.ONLINE)
    resolver.save_snapshot(Path("runs/123/inputs/players.parquet"))
"""

import pandas as pd
import requests
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'player_cache'
SLEEPER_PLAYER_CACHE = CACHE_DIR / 'sleeper_players.parquet'
SLEEPER_METADATA_FILE = CACHE_DIR / 'sleeper_players_metadata.json'
MAPPING_CACHE_FILE = CACHE_DIR / 'id_mappings.json'
MAX_CACHE_AGE_HOURS = 24  # Player data changes slowly

# Sleeper API
SLEEPER_API_URL = "https://api.sleeper.app/v1/players/nfl"


class ResolverMode(Enum):
    """Resolver network behavior mode.

    ONLINE: Can fetch from Sleeper API (use in Phase 0 only)
    OFFLINE: Must use snapshot, fails if unavailable (production runs)
    CONSERVATIVE: Use snapshot if available, mark unavailable if not
    """
    ONLINE = "online"
    OFFLINE = "offline"
    CONSERVATIVE = "conservative"


class ResolverNotAvailableError(Exception):
    """Raised when resolver data is not available in OFFLINE mode."""
    pass


@dataclass
class PlayerRecord:
    """Normalized player record."""
    sleeper_id: str
    gsis_id: Optional[str]
    espn_id: Optional[str]
    full_name: str
    first_name: str
    last_name: str
    team: str
    position: str
    status: str
    normalized_name: str  # For matching


class PlayerResolver:
    """
    Resolves player identities across different ID systems.

    Supports network determinism modes:
    - ONLINE: Can fetch from Sleeper API (Phase 0)
    - OFFLINE: Must use snapshot, fails if unavailable
    - CONSERVATIVE: Use snapshot if available, mark unavailable if not

    For production runs, use from_snapshot() to ensure no network calls.
    """

    _instance = None
    _players_df: Optional[pd.DataFrame] = None
    _sleeper_to_gsis_map: Optional[Dict[str, str]] = None
    _name_to_player_map: Optional[Dict[str, List[PlayerRecord]]] = None

    def __new__(cls, mode: ResolverMode = ResolverMode.ONLINE, snapshot_path: Optional[Path] = None):
        # Allow multiple instances for different modes/snapshots
        if snapshot_path is not None:
            # Don't use singleton for snapshot-specific instances
            return super().__new__(cls)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, mode: ResolverMode = ResolverMode.ONLINE, snapshot_path: Optional[Path] = None):
        # Track initialization per instance
        instance_key = f'{mode.value}_{snapshot_path}'
        if hasattr(self, '_init_key') and self._init_key == instance_key:
            return  # Already initialized with same config

        self._init_key = instance_key
        self._mode = mode
        self._snapshot_path = snapshot_path
        self._available = True  # Track if resolver is usable

        self._load_or_fetch_players()

    @classmethod
    def from_snapshot(cls, snapshot_path: Path, strict: bool = True) -> 'PlayerResolver':
        """
        Load resolver from a pinned snapshot file.

        Use this in production runs to ensure no network calls.

        Args:
            snapshot_path: Path to snapshot parquet file
            strict: If True (default), raise error if snapshot missing.
                    If False, return unavailable resolver.

        Returns:
            PlayerResolver instance

        Raises:
            ResolverNotAvailableError: If strict=True and snapshot missing
        """
        mode = ResolverMode.OFFLINE if strict else ResolverMode.CONSERVATIVE
        return cls(mode=mode, snapshot_path=snapshot_path)

    def _load_or_fetch_players(self, force_refresh: bool = False) -> None:
        """Load player data from snapshot, cache, or API based on mode."""

        # Mode 1: Load from specific snapshot file
        if self._snapshot_path is not None:
            if self._snapshot_path.exists():
                try:
                    self._load_from_snapshot(self._snapshot_path)
                    logger.info(f"Loaded resolver from snapshot: {self._snapshot_path}")
                    return
                except Exception as e:
                    error_msg = f"Failed to load snapshot {self._snapshot_path}: {e}"
                    if self._mode == ResolverMode.OFFLINE:
                        raise ResolverNotAvailableError(error_msg)
                    else:  # CONSERVATIVE
                        logger.warning(error_msg)
                        self._mark_unavailable("snapshot_load_failed")
                        return
            else:
                error_msg = f"Snapshot file not found: {self._snapshot_path}"
                if self._mode == ResolverMode.OFFLINE:
                    raise ResolverNotAvailableError(error_msg)
                else:  # CONSERVATIVE
                    logger.warning(error_msg)
                    self._mark_unavailable("snapshot_missing")
                    return

        # Mode 2: OFFLINE without snapshot - must use cache
        if self._mode == ResolverMode.OFFLINE:
            if self._is_cache_valid():
                try:
                    self._load_from_cache()
                    return
                except Exception as e:
                    raise ResolverNotAvailableError(f"Cache load failed in OFFLINE mode: {e}")
            else:
                raise ResolverNotAvailableError(
                    "No valid cache in OFFLINE mode. Run Phase 0 to fetch player data."
                )

        # Mode 3: ONLINE or CONSERVATIVE - try cache first, fetch if needed
        if not force_refresh and self._is_cache_valid():
            try:
                self._load_from_cache()
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
                if self._mode == ResolverMode.CONSERVATIVE:
                    self._mark_unavailable("cache_load_failed")
                    return

        # Fetch from API (ONLINE mode only)
        if self._mode == ResolverMode.ONLINE:
            self._fetch_from_sleeper()
            self._save_to_cache()
        else:
            # CONSERVATIVE mode without cache
            self._mark_unavailable("no_cache_available")

    def _mark_unavailable(self, reason: str) -> None:
        """Mark resolver as unavailable (CONSERVATIVE mode)."""
        self._available = False
        self._unavailable_reason = reason
        self._players_df = pd.DataFrame()
        self._sleeper_to_gsis_map = {}
        self._name_to_player_map = {}
        logger.warning(f"PlayerResolver unavailable: {reason}")

    @property
    def is_available(self) -> bool:
        """Check if resolver has valid player data."""
        return self._available

    @property
    def unavailable_reason(self) -> Optional[str]:
        """Get reason why resolver is unavailable (if applicable)."""
        return getattr(self, '_unavailable_reason', None)

    def _load_from_snapshot(self, path: Path) -> None:
        """Load player data from a snapshot file."""
        logger.info(f"Loading player data from snapshot: {path}")
        self._players_df = pd.read_parquet(path)
        self._build_lookup_maps()
        logger.info(f"Loaded {len(self._players_df):,} players from snapshot")

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid and fresh."""
        if not SLEEPER_PLAYER_CACHE.exists() or not SLEEPER_METADATA_FILE.exists():
            return False

        try:
            with open(SLEEPER_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            retrieved_at = datetime.fromisoformat(metadata['retrieved_at'])
            cache_age = datetime.now() - retrieved_at
            return cache_age < timedelta(hours=MAX_CACHE_AGE_HOURS)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Cache metadata invalid: {e}")
            return False
        except OSError as e:
            logger.warning(f"Failed to read cache metadata: {e}")
            return False

    def _load_from_cache(self) -> None:
        """Load player data from cache."""
        logger.info("Loading player data from cache...")
        self._players_df = pd.read_parquet(SLEEPER_PLAYER_CACHE)
        self._build_lookup_maps()
        logger.info(f"Loaded {len(self._players_df):,} players from cache")

    def _fetch_from_sleeper(self) -> None:
        """Fetch full player database from Sleeper API."""
        logger.info("Fetching player data from Sleeper API...")

        try:
            response = requests.get(SLEEPER_API_URL, timeout=60)
            response.raise_for_status()
            players = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Sleeper API request failed: {e}")

        logger.info(f"Retrieved {len(players):,} players from Sleeper")

        # Convert to DataFrame
        records = []
        for sleeper_id, player_data in players.items():
            # Skip non-NFL players
            if player_data.get('sport') != 'nfl':
                continue

            # Skip players without a team (free agents)
            team = player_data.get('team', '')
            if not team:
                continue

            # Handle fantasy_positions safely (can be None)
            fantasy_pos = player_data.get('fantasy_positions')
            fantasy_pos_str = ','.join(fantasy_pos) if fantasy_pos else ''

            records.append({
                'sleeper_id': sleeper_id,
                'gsis_id': player_data.get('gsis_id'),
                'espn_id': str(player_data.get('espn_id', '')) if player_data.get('espn_id') else None,
                'sportradar_id': player_data.get('sportradar_id'),
                'first_name': player_data.get('first_name', ''),
                'last_name': player_data.get('last_name', ''),
                'full_name': player_data.get('full_name', ''),
                'team': team.upper(),
                'position': player_data.get('position', ''),
                'status': player_data.get('status', ''),
                'fantasy_positions': fantasy_pos_str,
                'depth_chart_order': player_data.get('depth_chart_order'),
                'depth_chart_position': player_data.get('depth_chart_position'),
            })

        self._players_df = pd.DataFrame(records)

        # Add normalized name for matching
        self._players_df['normalized_name'] = self._players_df['full_name'].apply(
            self._normalize_name
        )

        self._build_lookup_maps()
        logger.info(f"Processed {len(self._players_df):,} NFL players with teams")

    def _build_lookup_maps(self) -> None:
        """Build lookup maps from player data."""
        if self._players_df is None or self._players_df.empty:
            self._sleeper_to_gsis_map = {}
            self._name_to_player_map = {}
            return

        # Sleeper ID → gsis_id mapping
        valid_gsis = self._players_df[self._players_df['gsis_id'].notna()]
        self._sleeper_to_gsis_map = dict(zip(
            valid_gsis['sleeper_id'],
            valid_gsis['gsis_id']
        ))

        # Normalized name → list of players (for name matching)
        self._name_to_player_map = {}
        for _, row in self._players_df.iterrows():
            record = PlayerRecord(
                sleeper_id=row['sleeper_id'],
                gsis_id=row['gsis_id'] if pd.notna(row['gsis_id']) else None,
                espn_id=row['espn_id'] if pd.notna(row['espn_id']) else None,
                full_name=row['full_name'],
                first_name=row['first_name'],
                last_name=row['last_name'],
                team=row['team'],
                position=row['position'],
                status=row['status'],
                normalized_name=row['normalized_name']
            )

            norm_name = row['normalized_name']
            if norm_name not in self._name_to_player_map:
                self._name_to_player_map[norm_name] = []
            self._name_to_player_map[norm_name].append(record)

        logger.info(f"Built maps: {len(self._sleeper_to_gsis_map):,} gsis mappings, "
                   f"{len(self._name_to_player_map):,} unique names")

    def _save_to_cache(self) -> None:
        """Save player data to cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self._players_df is not None and not self._players_df.empty:
            self._players_df.to_parquet(SLEEPER_PLAYER_CACHE, index=False)

            metadata = {
                'retrieved_at': datetime.now().isoformat(),
                'record_count': len(self._players_df),
                'gsis_coverage': len(self._sleeper_to_gsis_map),
            }
            with open(SLEEPER_METADATA_FILE, 'w') as f:
                json.dump(metadata, f)

            logger.info(f"Saved player cache: {len(self._players_df):,} records")

    def save_snapshot(self, path: Path) -> Dict:
        """
        Save current player data to a snapshot file for run isolation.

        Use in Phase 0 to pin player data for the run.

        Args:
            path: Path to save snapshot parquet file

        Returns:
            Metadata dict with snapshot info

        Raises:
            RuntimeError: If no player data available
        """
        if self._players_df is None or self._players_df.empty:
            raise RuntimeError("Cannot save snapshot: no player data loaded")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save parquet
        self._players_df.to_parquet(path, index=False)

        # Build metadata
        metadata = {
            'snapshot_path': str(path),
            'saved_at': datetime.now().isoformat(),
            'record_count': len(self._players_df),
            'gsis_coverage': len(self._sleeper_to_gsis_map) if self._sleeper_to_gsis_map else 0,
            'unique_teams': self._players_df['team'].nunique(),
            'mode': self._mode.value,
        }

        # Save metadata alongside
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved player snapshot: {len(self._players_df):,} records to {path}")
        return metadata

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize a player name for matching.

        Handles:
        - Case insensitivity
        - Suffix removal (Jr., II, III, IV, Sr.)
        - Hyphen handling
        - Special characters
        - Common nicknames
        """
        if not name:
            return ""

        # Lowercase
        name = name.lower().strip()

        # Remove common suffixes
        suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # Replace hyphens and apostrophes with space
        name = re.sub(r"['-]", ' ', name)

        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)

        return name

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def sleeper_to_gsis(self, sleeper_id: str) -> Optional[str]:
        """
        Get NFLverse gsis_id from Sleeper ID.

        Args:
            sleeper_id: Sleeper player ID

        Returns:
            gsis_id if found, None otherwise
        """
        if not self._sleeper_to_gsis_map:
            return None
        return self._sleeper_to_gsis_map.get(str(sleeper_id))

    def gsis_to_sleeper(self, gsis_id: str) -> Optional[str]:
        """
        Get Sleeper ID from NFLverse gsis_id.

        Args:
            gsis_id: NFLverse gsis_id

        Returns:
            Sleeper ID if found, None otherwise
        """
        if self._players_df is None or self._players_df.empty:
            return None

        matches = self._players_df[self._players_df['gsis_id'] == gsis_id]
        if matches.empty:
            return None
        return matches.iloc[0]['sleeper_id']

    def find_player(
        self,
        name: str,
        team: Optional[str] = None,
        position: Optional[str] = None
    ) -> Optional[PlayerRecord]:
        """
        Find a player by name, optionally filtering by team/position.

        Args:
            name: Player name (flexible matching)
            team: Team abbreviation (optional, improves accuracy)
            position: Position (optional, improves accuracy)

        Returns:
            PlayerRecord if found, None otherwise
        """
        if not self._name_to_player_map:
            return None

        norm_name = self._normalize_name(name)

        # Exact match first
        candidates = self._name_to_player_map.get(norm_name, [])

        # If no exact match, try partial matching
        if not candidates:
            candidates = self._find_similar_names(norm_name)

        if not candidates:
            return None

        # Filter by team if provided
        if team:
            team_upper = team.upper()
            team_matches = [c for c in candidates if c.team == team_upper]
            if team_matches:
                candidates = team_matches

        # Filter by position if provided
        if position:
            pos_upper = position.upper()
            pos_matches = [c for c in candidates if c.position == pos_upper]
            if pos_matches:
                candidates = pos_matches

        # Return first match (prefer active players)
        active = [c for c in candidates if c.status == 'Active']
        if active:
            return active[0]
        return candidates[0] if candidates else None

    def _find_similar_names(self, norm_name: str) -> List[PlayerRecord]:
        """Find players with similar names (handles partial matches)."""
        candidates = []

        name_tokens = norm_name.split()
        if len(name_tokens) < 2:
            return candidates

        search_last = name_tokens[-1]
        search_first = name_tokens[0]

        for stored_name, players in self._name_to_player_map.items():
            stored_tokens = stored_name.split()
            if len(stored_tokens) < 2:
                continue

            stored_last = stored_tokens[-1]
            stored_first = stored_tokens[0]

            # Last name must match exactly
            if search_last != stored_last:
                continue

            # First name matching strategies (in order of strictness)
            # 1. Exact first name match
            if search_first == stored_first:
                candidates.extend(players)
                continue

            # 2. First name starts with same 3+ chars (Mike/Michael, Rob/Robert)
            min_len = min(len(search_first), len(stored_first))
            if min_len >= 3 and search_first[:3] == stored_first[:3]:
                candidates.extend(players)
                continue

            # 3. One is initial of the other (J. Allen -> Josh Allen)
            if len(search_first) == 1 and stored_first.startswith(search_first):
                candidates.extend(players)
                continue
            if len(stored_first) == 1 and search_first.startswith(stored_first):
                candidates.extend(players)
                continue

        return candidates

    def find_player_by_gsis(self, gsis_id: str) -> Optional[PlayerRecord]:
        """
        Find a player by NFLverse gsis_id.

        Args:
            gsis_id: NFLverse gsis_id

        Returns:
            PlayerRecord if found, None otherwise
        """
        if self._players_df is None or self._players_df.empty:
            return None

        matches = self._players_df[self._players_df['gsis_id'] == gsis_id]
        if matches.empty:
            return None

        row = matches.iloc[0]
        return PlayerRecord(
            sleeper_id=row['sleeper_id'],
            gsis_id=row['gsis_id'] if pd.notna(row['gsis_id']) else None,
            espn_id=row['espn_id'] if pd.notna(row['espn_id']) else None,
            full_name=row['full_name'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            team=row['team'],
            position=row['position'],
            status=row['status'],
            normalized_name=row['normalized_name']
        )

    def get_team_players(
        self,
        team: str,
        position: Optional[str] = None,
        active_only: bool = True
    ) -> List[PlayerRecord]:
        """
        Get all players for a team.

        Args:
            team: Team abbreviation (e.g., 'BUF')
            position: Filter by position (optional)
            active_only: Only return active players

        Returns:
            List of PlayerRecords
        """
        if self._players_df is None or self._players_df.empty:
            return []

        team_upper = team.upper()
        mask = self._players_df['team'] == team_upper

        if position:
            mask &= self._players_df['position'] == position.upper()

        if active_only:
            mask &= self._players_df['status'] == 'Active'

        players = []
        for _, row in self._players_df[mask].iterrows():
            players.append(PlayerRecord(
                sleeper_id=row['sleeper_id'],
                gsis_id=row['gsis_id'] if pd.notna(row['gsis_id']) else None,
                espn_id=row['espn_id'] if pd.notna(row['espn_id']) else None,
                full_name=row['full_name'],
                first_name=row['first_name'],
                last_name=row['last_name'],
                team=row['team'],
                position=row['position'],
                status=row['status'],
                normalized_name=row['normalized_name']
            ))

        return players

    def refresh(self) -> None:
        """
        Force refresh player data from Sleeper API.

        Only works in ONLINE mode. Raises error in OFFLINE mode.
        """
        if self._mode == ResolverMode.OFFLINE:
            raise RuntimeError("Cannot refresh in OFFLINE mode - use ONLINE mode for Phase 0")
        if self._mode == ResolverMode.CONSERVATIVE:
            logger.warning("Refresh called in CONSERVATIVE mode - attempting network call")

        self._load_or_fetch_players(force_refresh=True)

    def get_coverage_stats(self) -> Dict:
        """Get statistics about ID coverage."""
        if self._players_df is None or self._players_df.empty:
            return {'total': 0, 'with_gsis': 0, 'coverage_pct': 0}

        total = len(self._players_df)
        with_gsis = len(self._players_df[self._players_df['gsis_id'].notna()])

        return {
            'total': total,
            'with_gsis': with_gsis,
            'coverage_pct': round(with_gsis / total * 100, 1) if total > 0 else 0,
            'unique_teams': self._players_df['team'].nunique(),
            'positions': self._players_df['position'].value_counts().to_dict()
        }


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

_resolver_instance: Optional[PlayerResolver] = None
_run_resolver: Optional[PlayerResolver] = None  # Run-specific resolver


def get_resolver(mode: ResolverMode = ResolverMode.ONLINE) -> PlayerResolver:
    """Get or create the singleton PlayerResolver instance."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = PlayerResolver(mode=mode)
    return _resolver_instance


def set_run_resolver(resolver: PlayerResolver) -> None:
    """
    Set a run-specific resolver for production pipelines.

    Call this in Phase 0 after loading from snapshot.
    Subsequent calls to get_run_resolver() will use this instance.
    """
    global _run_resolver
    _run_resolver = resolver
    logger.info(f"Set run resolver (available: {resolver.is_available})")


def get_run_resolver() -> Optional[PlayerResolver]:
    """
    Get the run-specific resolver.

    Returns None if not set. Use in production pipelines.
    """
    return _run_resolver


def clear_run_resolver() -> None:
    """Clear the run-specific resolver."""
    global _run_resolver
    _run_resolver = None


def sleeper_to_gsis(sleeper_id: str) -> Optional[str]:
    """Convenience function: Get gsis_id from Sleeper ID."""
    # Prefer run resolver if set
    resolver = _run_resolver or get_resolver()
    return resolver.sleeper_to_gsis(sleeper_id)


def find_player(
    name: str,
    team: Optional[str] = None,
    position: Optional[str] = None
) -> Optional[PlayerRecord]:
    """Convenience function: Find player by name."""
    resolver = _run_resolver or get_resolver()
    return resolver.find_player(name, team, position)


def get_gsis_id(
    name: str,
    team: Optional[str] = None,
    position: Optional[str] = None
) -> Optional[str]:
    """Convenience function: Get gsis_id by player name."""
    resolver = _run_resolver or get_resolver()
    player = resolver.find_player(name, team, position)
    return player.gsis_id if player else None


def is_resolver_available() -> bool:
    """Check if the current resolver is available."""
    resolver = _run_resolver or get_resolver()
    return resolver.is_available


def clear_player_cache() -> None:
    """Clear the player cache files."""
    if SLEEPER_PLAYER_CACHE.exists():
        SLEEPER_PLAYER_CACHE.unlink()
    if SLEEPER_METADATA_FILE.exists():
        SLEEPER_METADATA_FILE.unlink()
    if MAPPING_CACHE_FILE.exists():
        MAPPING_CACHE_FILE.unlink()
    logger.info("Cleared player cache")
