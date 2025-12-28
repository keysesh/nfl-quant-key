"""
QB Starter Detection Module

Identifies whether a QB is expected to be the starter for the upcoming game.
Critical for passing market predictions - backup QBs should not get starter-level projections.

Data Sources:
1. Depth charts (pos_rank = 1 means starter)
2. Snap counts (offense_pct > 0.6 in recent games = likely starter)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class QBRole(Enum):
    """QB role classification."""
    STARTER = "starter"
    BACKUP = "backup"
    UNKNOWN = "unknown"


@dataclass
class QBStarterInfo:
    """Information about a QB's expected role."""
    player_name: str
    team: str
    role: QBRole
    confidence: float  # 0-1, how confident we are in this classification
    depth_chart_rank: Optional[int] = None
    recent_snap_pct: Optional[float] = None
    reason: str = ""


class QBStarterDetector:
    """
    Detects whether a QB is expected to be a starter.

    Uses:
    1. Depth chart position (pos_rank = 1 = starter)
    2. Recent snap counts (high snap % = starter)
    """

    def __init__(self, data_dir: str = "data/nflverse"):
        self.data_dir = Path(data_dir)
        self._depth_charts: Optional[pd.DataFrame] = None
        self._snap_counts: Optional[pd.DataFrame] = None
        self._cache: Dict[str, QBStarterInfo] = {}

    def _load_depth_charts(self) -> pd.DataFrame:
        """Load depth chart data using canonical loader."""
        if self._depth_charts is not None:
            return self._depth_charts

        from nfl_quant.data.depth_chart_loader import get_depth_charts
        df = get_depth_charts()

        # Filter to QBs
        if 'pos_abb' in df.columns:
            df = df[df['pos_abb'] == 'QB'].copy()
        elif 'position' in df.columns:
            df = df[df['position'] == 'QB'].copy()
        self._depth_charts = df

        return self._depth_charts

    def _load_snap_counts(self) -> pd.DataFrame:
        """Load snap count data."""
        if self._snap_counts is not None:
            return self._snap_counts

        path = self.data_dir / "snap_counts.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Filter to QBs
            if 'position' in df.columns:
                df = df[df['position'] == 'QB'].copy()
            self._snap_counts = df
        else:
            self._snap_counts = pd.DataFrame()

        return self._snap_counts

    def get_depth_chart_rank(self, player_name: str, team: str) -> Optional[int]:
        """
        Get player's depth chart rank (1 = starter, 2 = backup, etc.)

        Returns None if not found.
        """
        dc = self._load_depth_charts()
        if len(dc) == 0:
            return None

        # Try exact match first
        player_data = dc[
            (dc['player_name'].str.lower() == player_name.lower()) &
            (dc['team'] == team)
        ]

        if len(player_data) == 0:
            # Try fuzzy match on last name
            last_name = player_name.split()[-1].lower()
            player_data = dc[
                (dc['player_name'].str.lower().str.contains(last_name)) &
                (dc['team'] == team)
            ]

        if len(player_data) > 0:
            # Get most recent entry
            if 'dt' in player_data.columns:
                player_data = player_data.sort_values('dt', ascending=False)

            if 'pos_rank' in player_data.columns:
                return int(player_data['pos_rank'].iloc[0])
            elif 'depth_team' in player_data.columns:
                return int(player_data['depth_team'].iloc[0])

        return None

    def get_recent_snap_pct(
        self,
        player_name: str,
        team: str,
        season: int = 2025,
        num_weeks: int = 2
    ) -> Optional[float]:
        """
        Get player's average snap percentage over recent weeks.

        Returns None if not found or insufficient data.
        """
        snap = self._load_snap_counts()
        if len(snap) == 0:
            return None

        # Filter to player
        player_data = snap[
            (snap['player'].str.lower() == player_name.lower()) &
            (snap['season'] == season)
        ]

        if len(player_data) == 0:
            # Try fuzzy match
            last_name = player_name.split()[-1].lower()
            player_data = snap[
                (snap['player'].str.lower().str.contains(last_name)) &
                (snap['team'] == team) &
                (snap['season'] == season)
            ]

        if len(player_data) == 0:
            return None

        # Get most recent weeks
        recent = player_data.sort_values('week', ascending=False).head(num_weeks)

        if 'offense_pct' in recent.columns:
            return float(recent['offense_pct'].mean())

        return None

    def classify_qb(
        self,
        player_name: str,
        team: str,
        season: int = 2025,
    ) -> QBStarterInfo:
        """
        Classify a QB as starter, backup, or unknown.

        Logic:
        1. If depth chart says rank 1 -> STARTER
        2. If depth chart says rank 2+ -> BACKUP
        3. If no depth chart, use snap counts:
           - >60% snap share in last 2 weeks -> STARTER
           - <40% snap share -> BACKUP
           - 40-60% -> UNKNOWN (QB controversy)
        4. If no data at all -> UNKNOWN
        """
        cache_key = f"{player_name}_{team}_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get depth chart rank
        dc_rank = self.get_depth_chart_rank(player_name, team)

        # Get recent snap percentage
        snap_pct = self.get_recent_snap_pct(player_name, team, season)

        # Classification logic
        role = QBRole.UNKNOWN
        confidence = 0.5
        reason = ""

        # Primary: Use depth chart if available
        if dc_rank is not None:
            if dc_rank == 1:
                role = QBRole.STARTER
                confidence = 0.9
                reason = f"Depth chart rank = {dc_rank} (starter)"
            else:
                role = QBRole.BACKUP
                confidence = 0.85
                reason = f"Depth chart rank = {dc_rank} (backup)"

        # Secondary: Adjust based on snap counts
        if snap_pct is not None:
            if role == QBRole.STARTER and snap_pct < 0.5:
                # Depth chart says starter but low snap share - might be benched
                role = QBRole.UNKNOWN
                confidence = 0.5
                reason = f"Depth chart starter but only {snap_pct:.0%} snaps (possible benching)"
            elif role == QBRole.BACKUP and snap_pct > 0.8:
                # Depth chart says backup but high snap share - probably promoted
                role = QBRole.STARTER
                confidence = 0.75
                reason = f"Depth chart backup but {snap_pct:.0%} snaps (promoted to starter)"
            elif role == QBRole.UNKNOWN:
                # No depth chart, use snap counts only
                if snap_pct > 0.6:
                    role = QBRole.STARTER
                    confidence = 0.7
                    reason = f"No depth chart data, but {snap_pct:.0%} snaps suggests starter"
                elif snap_pct < 0.4:
                    role = QBRole.BACKUP
                    confidence = 0.7
                    reason = f"No depth chart data, but only {snap_pct:.0%} snaps suggests backup"
                else:
                    reason = f"No depth chart data, {snap_pct:.0%} snaps is ambiguous"

        # If still unknown with no data, remain conservative
        if role == QBRole.UNKNOWN and not reason:
            reason = "No depth chart or snap data available"

        result = QBStarterInfo(
            player_name=player_name,
            team=team,
            role=role,
            confidence=confidence,
            depth_chart_rank=dc_rank,
            recent_snap_pct=snap_pct,
            reason=reason
        )

        self._cache[cache_key] = result
        return result

    def is_expected_starter(
        self,
        player_name: str,
        team: str,
        season: int = 2025,
        min_confidence: float = 0.6
    ) -> bool:
        """
        Quick check: Is this QB expected to be a starter?

        Returns True only if we're confident they're the starter.
        """
        info = self.classify_qb(player_name, team, season)
        return info.role == QBRole.STARTER and info.confidence >= min_confidence

    def get_team_starter(
        self,
        team: str,
        season: int = 2025
    ) -> Optional[str]:
        """
        Get the expected starting QB for a team.

        Returns player name or None if unclear.
        """
        dc = self._load_depth_charts()
        if len(dc) == 0:
            return None

        # Get rank 1 QB for team from depth chart
        team_qbs = dc[dc['team'] == team]

        if len(team_qbs) == 0:
            return None

        # Get most recent depth chart entry for rank 1
        if 'dt' in team_qbs.columns:
            team_qbs = team_qbs.sort_values('dt', ascending=False)

        if 'pos_rank' in team_qbs.columns:
            starter = team_qbs[team_qbs['pos_rank'] == 1]
        elif 'depth_team' in team_qbs.columns:
            starter = team_qbs[team_qbs['depth_team'] == 1]
        else:
            return None

        if len(starter) > 0:
            return starter['player_name'].iloc[0]

        return None

    def should_generate_passing_projection(
        self,
        player_name: str,
        team: str,
        season: int = 2025
    ) -> Tuple[bool, str]:
        """
        Determine if we should generate a passing projection for this player.

        Returns (should_project, reason)

        Only generates projections for expected starters with high confidence.
        """
        info = self.classify_qb(player_name, team, season)

        if info.role == QBRole.STARTER and info.confidence >= 0.7:
            return True, f"Expected starter: {info.reason}"

        if info.role == QBRole.BACKUP:
            return False, f"Backup QB: {info.reason}"

        if info.role == QBRole.UNKNOWN:
            # Be conservative - don't project uncertain situations
            return False, f"Uncertain role: {info.reason}"

        # Low confidence starter
        return False, f"Low confidence ({info.confidence:.0%}): {info.reason}"


# Global instance for easy access
_detector: Optional[QBStarterDetector] = None


def get_qb_starter_detector() -> QBStarterDetector:
    """Get or create the global QBStarterDetector instance."""
    global _detector
    if _detector is None:
        _detector = QBStarterDetector()
    return _detector


def is_qb_starter(player_name: str, team: str, season: int = 2025) -> bool:
    """Quick check if a QB is expected to be a starter."""
    return get_qb_starter_detector().is_expected_starter(player_name, team, season)


def should_project_passing(player_name: str, team: str, season: int = 2025) -> Tuple[bool, str]:
    """Check if we should generate passing projections for this player."""
    return get_qb_starter_detector().should_generate_passing_projection(player_name, team, season)
