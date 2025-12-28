"""Injury impact model for team EPA adjustments."""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from nfl_quant.config import settings
from nfl_quant.schemas import InjuryImpact

logger = logging.getLogger(__name__)

# DEPRECATED: Hardcoded multipliers replaced with historical analysis (CLAUDE.md framework principle)
# Use nfl_quant.features.historical_injury_impact.calculate_historical_redistribution()
# for data-driven player-specific injury adjustments based on actual PBP data
#
# These defaults remain ONLY for team-level EPA adjustments when historical data unavailable
DEFAULT_INJURY_MULTIPLIERS = {
    "qb": {
        "out": -0.8,
        "ir": -0.8,
        "questionable": -0.3,
        "doubtful": -0.5,
        "day_to_day": -0.2,
    },
    "ol": {
        "out": -0.15,
        "ir": -0.15,
        "questionable": -0.05,
        "doubtful": -0.08,
        "day_to_day": -0.03,
    },
    "wr": {
        "out": -0.3,
        "ir": -0.3,
        "questionable": -0.1,
        "doubtful": -0.15,
        "day_to_day": -0.05,
    },
    "te": {
        "out": -0.25,
        "ir": -0.25,
        "questionable": -0.08,
        "doubtful": -0.12,
        "day_to_day": -0.04,
    },
    "rb": {
        "out": -0.2,
        "ir": -0.2,
        "questionable": -0.07,
        "doubtful": -0.1,
        "day_to_day": -0.03,
    },
    "dl": {
        "out": -0.1,
        "ir": -0.1,
        "questionable": -0.03,
        "doubtful": -0.05,
        "day_to_day": -0.02,
    },
    "lb": {
        "out": -0.12,
        "ir": -0.12,
        "questionable": -0.04,
        "doubtful": -0.06,
        "day_to_day": -0.02,
    },
    "db": {
        "out": -0.08,
        "ir": -0.08,
        "questionable": -0.03,
        "doubtful": -0.04,
        "day_to_day": -0.01,
    },
}


class InjuryImpactModel:
    """Models injury impacts on team EPAs."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize injury model with YAML config.

        Args:
            config_path: Path to injury multipliers YAML, or None for defaults
        """
        self.multipliers = self._load_config(config_path)
        self.offensive_positions = ["qb", "ol", "wr", "te", "rb"]
        self.defensive_positions = ["dl", "lb", "db"]

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load injury multipliers from YAML or use defaults.

        Args:
            config_path: Path to YAML config

        Returns:
            Multipliers dictionary
        """
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded injury multipliers from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config {config_path}: {e}. Using defaults.")
        return DEFAULT_INJURY_MULTIPLIERS

    def _normalize_position(self, pos: str) -> str:
        """Normalize position string.

        Args:
            pos: Position string from NFLverse (e.g., QB, WR, RB, T, G, C, DE, etc.)

        Returns:
            Normalized position
        """
        if not pos:
            return "other"
        pos_lower = pos.lower()
        if pos_lower in ["qb"]:
            return "qb"
        if pos_lower in ["ol", "ot", "og", "c", "t", "g"]:
            return "ol"
        if pos_lower in ["wr", "wr/te", "wr/slot"]:
            return "wr"
        if pos_lower in ["te", "te/h"]:
            return "te"
        if pos_lower in ["rb", "rb/wr", "hb", "fb"]:
            return "rb"
        if pos_lower in ["de", "dt", "dl", "edge", "nt"]:
            return "dl"
        if pos_lower in ["lb", "ilb", "olb", "mlb"]:
            return "lb"
        if pos_lower in ["cb", "s", "db", "ss", "fs"]:
            return "db"
        return "other"

    def _normalize_status(self, status: Optional[str]) -> Optional[str]:
        """Normalize injury status string.

        Args:
            status: Status string from NFLverse (e.g., Out, Questionable, Doubtful, IR)

        Returns:
            Normalized status
        """
        if not status or pd.isna(status):
            return None
        status_lower = str(status).lower()
        if "out" in status_lower:
            return "out"
        if "ir" in status_lower or "injured reserve" in status_lower:
            return "ir"
        if "question" in status_lower:
            return "questionable"
        if "doubtful" in status_lower:
            return "doubtful"
        if "day" in status_lower or "probable" in status_lower:
            return "day_to_day"
        return None

    def _get_multiplier(self, position: str, status: Optional[str]) -> float:
        """Get injury multiplier for position/status.

        Args:
            position: Normalized position
            status: Normalized status

        Returns:
            Multiplier (negative value)
        """
        if not status or status == "active":
            return 0.0

        pos_config = self.multipliers.get(position, {})
        return pos_config.get(status, 0.0)

    def _is_offensive(self, position: str) -> bool:
        """Check if position is on offense.

        Args:
            position: Normalized position

        Returns:
            True if offensive position
        """
        return position in self.offensive_positions

    def compute_injury_impact(
        self, season: int, week: int, team_abbr: str
    ) -> InjuryImpact:
        """Compute injury impact for a team in a given week using NFLverse data.

        Args:
            season: Season year (e.g., 2023, 2024, 2025)
            week: Week number (1-18)
            team_abbr: Team abbreviation (e.g., "KC")

        Returns:
            InjuryImpact object
        """
        offensive_impact = 0.0
        defensive_impact = 0.0
        injury_count = 0
        missing_qb = False
        missing_ol_count = 0
        player_impacts = []

        try:
            # Load NFLverse rosters and injuries from R-fetched files
            nflverse_dir = Path("data/nflverse")

            # Load rosters
            rosters_file = nflverse_dir / "rosters.parquet"
            if not rosters_file.exists():
                rosters_file = nflverse_dir / "rosters.csv"

            if rosters_file.exists():
                if rosters_file.suffix == ".parquet":
                    rosters = pd.read_parquet(rosters_file)
                else:
                    rosters = pd.read_csv(rosters_file, low_memory=False)
                if "season" in rosters.columns:
                    rosters = rosters[rosters["season"] == season]
            else:
                logger.warning(f"No rosters file found. Run: Rscript scripts/fetch/fetch_nflverse_data.R")
                rosters = pd.DataFrame()

            # Load injuries from local cache (fetched via scripts/fetch/fetch_injuries_api.py)
            injuries_file = Path("data/injuries/current_injuries.csv")
            if injuries_file.exists():
                injuries = pd.read_csv(injuries_file)
            else:
                logger.warning(f"No injuries file found at {injuries_file}")
                injuries = pd.DataFrame()

            # Filter rosters to the specific team
            team_rosters = rosters[rosters['team'] == team_abbr].copy() if not rosters.empty else pd.DataFrame()

            if len(team_rosters) == 0:
                logger.warning(f"No roster data found for team {team_abbr} in season {season}")
                return InjuryImpact(
                    team=team_abbr,
                    week=week,
                    total_impact_offensive_epa=0.0,
                    total_impact_defensive_epa=0.0,
                    injury_count=0,
                    missing_qb=False,
                    missing_ol_count=0,
                    player_impacts=[],
                )

            # Filter injuries to the specific team and week
            # Support both NFLverse format (has 'week' column) and Sleeper format (no 'week' column)
            if 'week' in injuries.columns:
                team_injuries = injuries[
                    (injuries['team'] == team_abbr) & (injuries['week'] == week)
                ].copy()
            else:
                # Sleeper format - just filter by team (assumes current injuries are always relevant)
                team_injuries = injuries[injuries['team'] == team_abbr].copy()

            # Merge rosters with injuries to get injured players
            # Use gsis_id as the common key (NFLverse format) or skip merge for Sleeper format
            if 'gsis_id' in injuries.columns and 'gsis_id' in team_rosters.columns:
                injured_players = team_rosters.merge(
                    team_injuries,
                    on=['gsis_id', 'team'],
                    how='inner',
                    suffixes=('_roster', '_injury')
                )
            else:
                # Sleeper format - use direct injury data without roster merge
                # This is simpler and works directly with Sleeper's player data
                injured_players = team_injuries

            # Iterate over injured players
            for _, player in injured_players.iterrows():
                # Get injury status from report_status or practice_status (NFLverse) or injury_status (Sleeper)
                status = player.get('report_status') or player.get('practice_status') or player.get('injury_status')

                if not status or pd.isna(status):
                    continue

                injury_count += 1

                # Normalize position (use position from injury report, fallback to roster)
                position = self._normalize_position(
                    player.get('position_injury') or player.get('position_roster', '') or player.get('position', '')
                )
                if position == "other":
                    continue

                status_norm = self._normalize_status(status)
                multiplier = self._get_multiplier(position, status_norm)

                if multiplier == 0.0:
                    continue

                # Track impacts
                is_offensive = self._is_offensive(position)
                if is_offensive:
                    offensive_impact += multiplier
                    if position == "qb":
                        missing_qb = True
                    elif position == "ol":
                        missing_ol_count += 1
                else:
                    defensive_impact += multiplier

                player_impacts.append(
                    {
                        "player_id": player.get('gsis_id') or player.get('player_id', 'unknown'),
                        "name": player.get('full_name_injury') or player.get('full_name_roster') or player.get('player_name', 'Unknown'),
                        "position": position,
                        "status": status_norm,
                        "multiplier": multiplier,
                        "is_offensive": is_offensive,
                    }
                )

            # Cap OL stacking (max 3 OL on field at once)
            if missing_ol_count > 0:
                max_ol_impact = DEFAULT_INJURY_MULTIPLIERS["ol"]["out"] * 3
                offensive_impact = max(offensive_impact, max_ol_impact)

            # CRITICAL: Cap total impact to prevent unrealistic adjustments
            # Not all injured players are starters. Apply reasonable caps.
            # The original multipliers are too large for simulation EPA scales.
            # Typical team EPA ranges from -0.15 to +0.15, so injury impacts
            # should be much smaller (0.01-0.10 range, not 0.3-0.8 range).
            #
            # SCALE DOWN: Apply 10% of calculated impact to be realistic
            # This makes a QB out = -0.08 EPA (about 1-2 points difference)
            INJURY_SCALE_FACTOR = 0.10

            offensive_impact = offensive_impact * INJURY_SCALE_FACTOR
            defensive_impact = defensive_impact * INJURY_SCALE_FACTOR

            # Apply reasonable caps after scaling
            MAX_OFFENSIVE_IMPACT = -0.15  # Cap at -0.15 EPA (significant but not game-breaking)
            MAX_DEFENSIVE_IMPACT = -0.10  # Cap at -0.10 EPA

            # Apply caps (impacts are negative, so we use max to cap)
            offensive_impact = max(offensive_impact, MAX_OFFENSIVE_IMPACT)
            defensive_impact = max(defensive_impact, MAX_DEFENSIVE_IMPACT)

            logger.debug(f"Team {team_abbr} injury impacts after scaling/caps: Off={offensive_impact:.3f}, Def={defensive_impact:.3f}")

        except Exception as e:
            logger.error(f"Error computing injury impact for {team_abbr} (season={season}, week={week}): {e}")
            # Return zero impact on error
            return InjuryImpact(
                team=team_abbr,
                week=week,
                total_impact_offensive_epa=0.0,
                total_impact_defensive_epa=0.0,
                injury_count=0,
                missing_qb=False,
                missing_ol_count=0,
                player_impacts=[],
            )

        return InjuryImpact(
            team=team_abbr,
            week=week,
            total_impact_offensive_epa=offensive_impact,
            total_impact_defensive_epa=defensive_impact,
            injury_count=injury_count,
            missing_qb=missing_qb,
            missing_ol_count=missing_ol_count,
            player_impacts=player_impacts,
        )

    def apply_injury_adjustments(
        self, offensive_epa: float, defensive_epa: float, injury_impact: InjuryImpact
    ) -> tuple[float, float]:
        """Apply injury adjustments to team EPAs.

        Args:
            offensive_epa: Baseline offensive EPA
            defensive_epa: Baseline defensive EPA
            injury_impact: Computed injury impact

        Returns:
            Tuple of (adjusted_off_epa, adjusted_def_epa)
        """
        adjusted_off_epa = offensive_epa + injury_impact.total_impact_offensive_epa
        adjusted_def_epa = defensive_epa + injury_impact.total_impact_defensive_epa

        return adjusted_off_epa, adjusted_def_epa


# Import at module level
from pathlib import Path


# =============================================================================
# V24 INJURY FEATURE EXTRACTION
# =============================================================================

def get_v24_injury_features(
    player_name: str,
    team: str,
    position: str,
    week: int,
    season: int,
) -> dict:
    """
    Extract V24 classifier injury features for a player.

    Features:
    - player_injury_status: 0=healthy, 1=questionable, 2=doubtful
    - qb_injury_status: 0=healthy, 1=questionable, 2=doubtful, 3=out/backup
    - team_wr1_out: Binary, 1 if WR1 is out (opportunity boost for WR2/3)
    - team_rb1_out: Binary, 1 if RB1 is out (opportunity boost for RB2)

    Args:
        player_name: Player's name
        team: Team abbreviation
        position: Player position
        week: Current week
        season: Current season

    Returns:
        Dict with V24 injury features
    """
    features = {
        'player_injury_status': 0,
        'qb_injury_status': 0,
        'team_wr1_out': 0,
        'team_rb1_out': 0,
    }

    try:
        # Load injury data
        injuries_file = Path("data/injuries/current_injuries.csv")
        if not injuries_file.exists():
            # Try alternate paths
            alt_paths = [
                Path("data/nflverse/injuries.parquet"),
                Path("data/injuries.csv"),
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    injuries_file = alt_path
                    break
            else:
                logger.debug("No injury file found - using defaults")
                return features

        # Load injuries
        if injuries_file.suffix == '.parquet':
            injuries = pd.read_parquet(injuries_file)
        else:
            injuries = pd.read_csv(injuries_file)

        if injuries.empty:
            return features

        # Filter to team
        team_injuries = injuries[injuries['team'] == team].copy()

        # ========================================
        # 1. Player's own injury status
        # ========================================
        player_injury = _get_player_injury_status(team_injuries, player_name)
        features['player_injury_status'] = player_injury

        # ========================================
        # 2. QB injury status for this team
        # ========================================
        qb_status = _get_qb_injury_status(team_injuries)
        features['qb_injury_status'] = qb_status

        # ========================================
        # 3. Check if WR1 is out (for WR2/WR3/TE opportunity)
        # ========================================
        if position in ['WR', 'TE']:
            features['team_wr1_out'] = _check_wr1_out(team_injuries, team, season)

        # ========================================
        # 4. Check if RB1 is out (for RB2 opportunity)
        # ========================================
        if position == 'RB':
            features['team_rb1_out'] = _check_rb1_out(team_injuries, team, season)

    except Exception as e:
        logger.warning(f"Error extracting V24 injury features for {player_name}: {e}")

    return features


def _get_player_injury_status(injuries: pd.DataFrame, player_name: str) -> int:
    """
    Get player's injury status as numeric code.

    Returns:
        0=healthy, 1=questionable, 2=doubtful
    """
    if injuries.empty:
        return 0

    # Find player in injury report
    # Try multiple name columns
    name_cols = ['player_name', 'full_name', 'name']
    player_injury = pd.DataFrame()

    for col in name_cols:
        if col in injuries.columns:
            mask = injuries[col].str.contains(player_name, case=False, na=False)
            if mask.any():
                player_injury = injuries[mask]
                break

    if player_injury.empty:
        return 0  # Not on injury report = healthy

    # Get status
    status_cols = ['injury_status', 'report_status', 'status']
    status = None

    for col in status_cols:
        if col in player_injury.columns:
            status = str(player_injury.iloc[0][col]).lower()
            break

    if not status or pd.isna(status):
        return 0

    # Map status to numeric
    if 'doubtful' in status:
        return 2
    elif 'question' in status:
        return 1
    else:
        return 0  # Out players shouldn't have props, treat as healthy default


def _get_qb_injury_status(injuries: pd.DataFrame) -> int:
    """
    Get QB injury status for the team.

    Returns:
        0=healthy, 1=questionable, 2=doubtful, 3=out/backup playing
    """
    if injuries.empty:
        return 0

    # Find QB injuries
    pos_cols = ['position', 'pos']
    qb_injuries = pd.DataFrame()

    for col in pos_cols:
        if col in injuries.columns:
            mask = injuries[col].str.upper() == 'QB'
            if mask.any():
                qb_injuries = injuries[mask]
                break

    if qb_injuries.empty:
        return 0  # No QB on injury report

    # Get worst status (backup playing = worst case)
    status_cols = ['injury_status', 'report_status', 'status']
    statuses = []

    for _, row in qb_injuries.iterrows():
        for col in status_cols:
            if col in row.index and not pd.isna(row[col]):
                statuses.append(str(row[col]).lower())
                break

    if not statuses:
        return 0

    # Check for out status (backup playing)
    for status in statuses:
        if 'out' in status or 'ir' in status:
            return 3  # Backup QB playing

    # Check for doubtful
    for status in statuses:
        if 'doubtful' in status:
            return 2

    # Check for questionable
    for status in statuses:
        if 'question' in status:
            return 1

    return 0


def _check_wr1_out(injuries: pd.DataFrame, team: str, season: int) -> int:
    """
    Check if team's WR1 is out.

    Uses depth charts to identify WR1, then checks injury status.

    Returns:
        1 if WR1 is out, 0 otherwise
    """
    if injuries.empty:
        return 0

    try:
        # Load depth charts to identify WR1 using canonical loader
        from nfl_quant.data.depth_chart_loader import get_depth_charts
        depth_charts = get_depth_charts(season=season)

        if depth_charts.empty:
            return 0

        # Handle team column naming (club_code or team)
        team_col = 'club_code' if 'club_code' in depth_charts.columns else 'team'
        pos_col = 'position' if 'position' in depth_charts.columns else 'pos_name'

        # Filter to current season and team
        team_dc = depth_charts[
            (depth_charts[team_col] == team)
        ]
        if pos_col in team_dc.columns:
            team_dc = team_dc[team_dc[pos_col].str.contains('WR|Wide Receiver', case=False, na=False)]

        if team_dc.empty:
            return 0

        # Get most recent depth chart
        team_dc = team_dc.sort_values('week', ascending=False)

        # WR1 is typically depth_team 1
        wr1_rows = team_dc[team_dc['depth_team'] == 1]
        if wr1_rows.empty:
            return 0

        wr1_name = wr1_rows.iloc[0]['full_name']

        # Check if WR1 is out
        name_cols = ['player_name', 'full_name', 'name']
        for col in name_cols:
            if col in injuries.columns:
                mask = injuries[col].str.contains(wr1_name, case=False, na=False)
                if mask.any():
                    wr1_injury = injuries[mask].iloc[0]
                    status_cols = ['injury_status', 'report_status', 'status']
                    for scol in status_cols:
                        if scol in wr1_injury.index:
                            status = str(wr1_injury[scol]).lower()
                            if 'out' in status or 'ir' in status:
                                return 1
                    break

    except Exception as e:
        logger.debug(f"Error checking WR1 status: {e}")

    return 0


def _check_rb1_out(injuries: pd.DataFrame, team: str, season: int) -> int:
    """
    Check if team's RB1 is out.

    Uses depth charts to identify RB1, then checks injury status.

    Returns:
        1 if RB1 is out, 0 otherwise
    """
    if injuries.empty:
        return 0

    try:
        # Load depth charts to identify RB1 using canonical loader
        from nfl_quant.data.depth_chart_loader import get_depth_charts
        depth_charts = get_depth_charts(season=season)

        if depth_charts.empty:
            return 0

        # Handle team column naming (club_code or team)
        team_col = 'club_code' if 'club_code' in depth_charts.columns else 'team'
        pos_col = 'position' if 'position' in depth_charts.columns else 'pos_name'

        # Filter to team
        team_dc = depth_charts[
            (depth_charts[team_col] == team)
        ]
        if pos_col in team_dc.columns:
            team_dc = team_dc[team_dc[pos_col].str.contains('RB|Running Back', case=False, na=False)]

        if team_dc.empty:
            return 0

        # Get most recent depth chart
        team_dc = team_dc.sort_values('week', ascending=False)

        # RB1 is typically depth_team 1
        rb1_rows = team_dc[team_dc['depth_team'] == 1]
        if rb1_rows.empty:
            return 0

        rb1_name = rb1_rows.iloc[0]['full_name']

        # Check if RB1 is out
        name_cols = ['player_name', 'full_name', 'name']
        for col in name_cols:
            if col in injuries.columns:
                mask = injuries[col].str.contains(rb1_name, case=False, na=False)
                if mask.any():
                    rb1_injury = injuries[mask].iloc[0]
                    status_cols = ['injury_status', 'report_status', 'status']
                    for scol in status_cols:
                        if scol in rb1_injury.index:
                            status = str(rb1_injury[scol]).lower()
                            if 'out' in status or 'ir' in status:
                                return 1
                    break

    except Exception as e:
        logger.debug(f"Error checking RB1 status: {e}")

    return 0


