"""
Contextual factors integration helper.
Provides easy-to-use functions for integrating contextual factors into predictions.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd

from nfl_quant.schemas import PlayerPropInput
from nfl_quant.features.matchup_features import build_contextual_features
from nfl_quant.data.matchup_extractor import MatchupExtractor, ContextBuilder
from nfl_quant.utils.player_names import normalize_player_name

logger = logging.getLogger(__name__)


def enhance_player_input_with_context(
    player_input: PlayerPropInput,
    pbp_df: Optional[pd.DataFrame] = None,
    injury_data: Optional[Dict[str, Any]] = None,
    season: Optional[int] = None
) -> PlayerPropInput:
    """
    Enhance player input with contextual factors (matchups, QB connections, situational).

    This is the main entry point for adding contextual factors to a PlayerPropInput.

    Args:
        player_input: Base player input
        pbp_df: Optional play-by-play DataFrame (if None, will try to load)
        injury_data: Optional injury/lineup data dictionary
        season: Season year (defaults to current season)

    Returns:
        Enhanced PlayerPropInput with contextual factors
    """
    # Auto-detect season if not provided
    if season is None:
        from nfl_quant.utils.season_utils import get_current_season
        season = get_current_season()

    # Try to load PBP data if not provided
    if pbp_df is None:
        pbp_path = Path(f'data/processed/pbp_{season}.parquet')
        if pbp_path.exists():
            try:
                pbp_df = pd.read_parquet(pbp_path)
                logger.debug(f"Loaded PBP data from {pbp_path}")
            except Exception as e:
                logger.warning(f"Could not load PBP data: {e}")
                return player_input  # Return unenhanced input
        else:
            logger.debug("No PBP data available - skipping contextual enhancements")
            return player_input  # Return unenhanced input

    # Extract matchup and connection data
    extractor = MatchupExtractor(pbp_df)

    # Try to load existing matchup data (if cached)
    matchup_path = Path(f'data/matchups/team_matchups_{season}.parquet')
    qb_connections_path = Path(f'data/connections/qb_wr_connections_{season}.parquet')

    matchup_df = None
    qb_connections_df = None

    if matchup_path.exists():
        try:
            matchup_df = pd.read_parquet(matchup_path)
            logger.debug(f"Loaded matchup data from {matchup_path}")
        except Exception:
            pass

    if qb_connections_path.exists():
        try:
            qb_connections_df = pd.read_parquet(qb_connections_path)
            logger.debug(f"Loaded QB connections from {qb_connections_path}")
        except Exception:
            pass

    # If not cached, extract on-the-fly (slower but works)
    if matchup_df is None or len(matchup_df) == 0:
        try:
            matchup_df = extractor.extract_team_matchups(season)
            # Cache for future use
            matchup_path.parent.mkdir(parents=True, exist_ok=True)
            if len(matchup_df) > 0:
                matchup_df.to_parquet(matchup_path, index=False)
                logger.debug(f"Cached matchup data to {matchup_path}")
        except Exception as e:
            logger.warning(f"Could not extract matchup data: {e}")
            matchup_df = pd.DataFrame()

    if qb_connections_df is None or len(qb_connections_df) == 0:
        try:
            qb_connections_df = extractor.extract_qb_wr_connections(season)
            # Cache for future use
            qb_connections_path.parent.mkdir(parents=True, exist_ok=True)
            if len(qb_connections_df) > 0:
                qb_connections_df.to_parquet(qb_connections_path, index=False)
                logger.debug(f"Cached QB connections to {qb_connections_path}")
        except Exception as e:
            logger.warning(f"Could not extract QB connections: {e}")
            qb_connections_df = pd.DataFrame()

    # Build context
    context_builder = ContextBuilder(
        matchup_df=matchup_df if len(matchup_df) > 0 else None,
        qb_connections_df=qb_connections_df if len(qb_connections_df) > 0 else None
    )

    # Enhance input
    enhanced_input = build_contextual_features(
        player_input=player_input,
        context_builder=context_builder,
        matchup_history=None,  # Will be built by build_contextual_features
        injury_data=injury_data
    )

    return enhanced_input


def load_injury_data(week: int) -> Dict[str, Dict[str, Any]]:
    """
    Load injury/lineup data for a week.

    Tracks injury status for ALL key positions (WR1/2/3, RB1/2, TE1, QB)
    to enable proper target/carry redistribution.

    Args:
        week: Week number

    Returns:
        Dictionary mapping team to comprehensive injury data
    """
    injury_data = {}

    # Try to load from injury files (priority order)
    injury_paths = [
        Path('data/injuries/injuries_latest.csv'),  # Sleeper latest (preferred)
        Path('data/injuries/current_injuries.csv'),  # Symlink to latest
        Path(f'data/injuries/injuries_week{week}.csv'),  # Week-specific
    ]

    injury_df = None
    for injury_path in injury_paths:
        if injury_path.exists():
            try:
                injury_df = pd.read_csv(injury_path)
                logger.info(f"   Loaded injuries from {injury_path.name}")
                break
            except Exception as e:
                logger.warning(f"Could not load {injury_path}: {e}")

    if injury_df is not None and len(injury_df) > 0:
        try:
            # Determine if this is Sleeper format (has injury_status) or NFLverse format (has status)
            is_sleeper_format = 'injury_status' in injury_df.columns
            status_col = 'injury_status' if is_sleeper_format else 'status'

            # Build lookup of injured players by team
            # BUG FIX (Nov 23, 2025): Use NORMALIZED names for matching to handle Jr./Sr./II/III variations
            # BUG FIX (Nov 24, 2025): Handle players with missing team (use position + name matching)
            injured_by_team = {}
            injured_by_name = {}  # Fallback for players without team

            for team in injury_df['team'].dropna().unique():
                if pd.isna(team) or team == '':
                    continue
                team_injuries = injury_df[injury_df['team'] == team]
                injured_by_team[team] = {
                    normalize_player_name(row.get('player_name', '')): row.get(status_col, 'active')
                    for _, row in team_injuries.iterrows()
                }

            # Also build global name lookup for players without team
            for _, row in injury_df.iterrows():
                player_name = normalize_player_name(row.get('player_name', ''))
                if player_name:
                    injured_by_name[player_name] = row.get(status_col, 'active')

            # Try to load depth chart / roster info to identify WR1/2/3, RB1/2, TE1
            depth_chart = _load_team_depth_charts()

            # Group by team
            for team in injury_df['team'].dropna().unique():
                if pd.isna(team) or team == '':
                    continue

                team_injuries = injury_df[injury_df['team'] == team]

                # Initialize comprehensive injury tracking
                team_injury_data = {
                    'qb_status': 'healthy',
                    'starting_qb': '',
                    'top_wr_1': '',
                    'top_wr_1_status': 'active',
                    'top_wr_2': '',
                    'top_wr_2_status': 'active',
                    'top_wr_3': '',
                    'top_wr_3_status': 'active',
                    'top_rb': '',
                    'top_rb_status': 'active',
                    'top_rb_2': '',
                    'top_rb_2_status': 'active',
                    'top_te': '',
                    'top_te_status': 'active',
                }

                # Get depth chart for this team (if available)
                if team in depth_chart:
                    dc = depth_chart[team]
                    # Set WR1/2/3 from depth chart
                    if 'wr1' in dc:
                        team_injury_data['top_wr_1'] = dc['wr1']
                    if 'wr2' in dc:
                        team_injury_data['top_wr_2'] = dc['wr2']
                    if 'wr3' in dc:
                        team_injury_data['top_wr_3'] = dc['wr3']
                    # Set RB1/2 from depth chart
                    if 'rb1' in dc:
                        team_injury_data['top_rb'] = dc['rb1']
                    if 'rb2' in dc:
                        team_injury_data['top_rb_2'] = dc['rb2']
                    # Set TE1 from depth chart
                    if 'te1' in dc:
                        team_injury_data['top_te'] = dc['te1']
                    # Set QB from depth chart
                    if 'qb1' in dc:
                        team_injury_data['starting_qb'] = dc['qb1']

                # Now check injury status for each position
                injured_players = injured_by_team.get(team, {})

                # Helper to get injury status with NORMALIZED name matching
                # BUG FIX (Nov 23, 2025): Normalize depth chart names before lookup
                # BUG FIX (Nov 24, 2025): Use fallback to global name lookup if team missing
                def get_injury_status(player_name: str) -> str:
                    """
                    Get injury status using normalized name matching.

                    Handles Jr./Sr./II/III suffixes, A.J. vs AJ, case variations.
                    Example: "Marvin Harrison Jr." matches "Marvin Harrison"
                    """
                    if not player_name:
                        return 'active'

                    normalized = normalize_player_name(player_name)

                    # Try team-specific lookup first
                    status = injured_players.get(normalized, None)

                    # Fallback to global name lookup if not found (handles missing team)
                    if status is None:
                        status = injured_by_name.get(normalized, 'active')

                    if isinstance(status, str):
                        return status.lower()
                    return 'active'

                # Check each tracked player's injury status
                if team_injury_data['top_wr_1']:
                    team_injury_data['top_wr_1_status'] = get_injury_status(team_injury_data['top_wr_1'])
                if team_injury_data['top_wr_2']:
                    team_injury_data['top_wr_2_status'] = get_injury_status(team_injury_data['top_wr_2'])
                if team_injury_data['top_wr_3']:
                    team_injury_data['top_wr_3_status'] = get_injury_status(team_injury_data['top_wr_3'])
                if team_injury_data['top_rb']:
                    team_injury_data['top_rb_status'] = get_injury_status(team_injury_data['top_rb'])
                if team_injury_data['top_rb_2']:
                    team_injury_data['top_rb_2_status'] = get_injury_status(team_injury_data['top_rb_2'])
                if team_injury_data['top_te']:
                    team_injury_data['top_te_status'] = get_injury_status(team_injury_data['top_te'])
                if team_injury_data['starting_qb']:
                    team_injury_data['qb_status'] = get_injury_status(team_injury_data['starting_qb'])

                # VALIDATION (Nov 23, 2025): Log discrepancies between status and game_probability
                # This catches cases where injury file has "Out" but our lookup returns "active"
                if 'game_probability' in injury_df.columns:
                    for pos_key, status_key in [
                        ('top_wr_1', 'top_wr_1_status'),
                        ('top_wr_2', 'top_wr_2_status'),
                        ('top_wr_3', 'top_wr_3_status'),
                        ('top_rb', 'top_rb_status'),
                        ('top_rb_2', 'top_rb_2_status'),
                        ('top_te', 'top_te_status'),
                        ('starting_qb', 'qb_status'),
                    ]:
                        player_name = team_injury_data.get(pos_key, '')
                        status = team_injury_data.get(status_key, 'active')

                        if player_name:
                            # Check injury file for game_probability
                            player_injury = team_injuries[
                                team_injuries['player_name'].apply(normalize_player_name) ==
                                normalize_player_name(player_name)
                            ]

                            if len(player_injury) > 0:
                                game_prob = player_injury.iloc[0].get('game_probability', 1.0)

                                # Flag if status is 'active' but game_probability is 0
                                if status == 'active' and game_prob == 0.0:
                                    logger.warning(
                                        f"   🚨 VALIDATION ERROR: {team} {player_name} marked 'active' "
                                        f"but injury file shows 0% game probability. Setting to OUT."
                                    )
                                    team_injury_data[status_key] = 'out'

                # Fallback: If no depth chart, use injured WRs/RBs from injury report
                # (This assumes first injured WR is a key WR, etc.)
                if not team_injury_data['top_wr_1']:
                    wr_injuries = team_injuries[team_injuries['position'] == 'WR']
                    for i, (_, row) in enumerate(wr_injuries.iterrows()):
                        player_name = row.get('player_name', '')
                        status = row.get(status_col, 'active')
                        if isinstance(status, str):
                            status = status.lower()
                        if i == 0 and not team_injury_data['top_wr_1']:
                            team_injury_data['top_wr_1'] = player_name
                            team_injury_data['top_wr_1_status'] = status
                        elif i == 1 and not team_injury_data['top_wr_2']:
                            team_injury_data['top_wr_2'] = player_name
                            team_injury_data['top_wr_2_status'] = status
                        elif i == 2 and not team_injury_data['top_wr_3']:
                            team_injury_data['top_wr_3'] = player_name
                            team_injury_data['top_wr_3_status'] = status

                if not team_injury_data['top_rb']:
                    rb_injuries = team_injuries[team_injuries['position'] == 'RB']
                    for i, (_, row) in enumerate(rb_injuries.iterrows()):
                        player_name = row.get('player_name', '')
                        status = row.get(status_col, 'active')
                        if isinstance(status, str):
                            status = status.lower()
                        if i == 0 and not team_injury_data['top_rb']:
                            team_injury_data['top_rb'] = player_name
                            team_injury_data['top_rb_status'] = status
                        elif i == 1 and not team_injury_data['top_rb_2']:
                            team_injury_data['top_rb_2'] = player_name
                            team_injury_data['top_rb_2_status'] = status

                if not team_injury_data['top_te']:
                    te_injuries = team_injuries[team_injuries['position'] == 'TE']
                    if len(te_injuries) > 0:
                        row = te_injuries.iloc[0]
                        team_injury_data['top_te'] = row.get('player_name', '')
                        status = row.get(status_col, 'active')
                        if isinstance(status, str):
                            status = status.lower()
                        team_injury_data['top_te_status'] = status

                if not team_injury_data['starting_qb']:
                    qb_injuries = team_injuries[team_injuries['position'] == 'QB']
                    if len(qb_injuries) > 0:
                        row = qb_injuries.iloc[0]
                        team_injury_data['starting_qb'] = row.get('player_name', '')
                        status = row.get(status_col, 'active')
                        if isinstance(status, str):
                            status = status.lower()
                        team_injury_data['qb_status'] = status

                injury_data[team] = team_injury_data
        except Exception as e:
            logger.warning(f"Could not process injury data: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.warning("   ⚠️  No injury data files found")

    return injury_data


def _load_team_depth_charts() -> Dict[str, Dict[str, str]]:
    """
    Load team depth charts to identify WR1/2/3, RB1/2, TE1, QB1.

    PRIORITY: Use official NFLverse depth chart data (pos_rank = actual depth chart position).
    This is the most accurate source, updated weekly.

    Returns:
        Dictionary mapping team to position depth chart
    """
    depth_charts = {}

    # FIRST: Try to load official NFLverse depth chart data (MOST ACCURATE)
    depth_chart_path = Path('data/nflverse/depth_charts_2025.parquet')
    if depth_chart_path.exists():
        try:
            dc_df = pd.read_parquet(depth_chart_path)

            # Use the latest depth chart (most recent dt)
            if 'dt' in dc_df.columns:
                latest_dt = dc_df['dt'].max()
                dc_df = dc_df[dc_df['dt'] == latest_dt]
                logger.info(f"Using official NFLverse depth charts (as of {latest_dt})")

            # Build depth chart for each team
            for team in dc_df['team'].dropna().unique():
                team_dc_data = dc_df[dc_df['team'] == team]
                team_dc = {}

                # WRs - use pos_rank (actual depth chart position)
                wrs = team_dc_data[team_dc_data['pos_abb'] == 'WR'].sort_values('pos_rank')
                wr_names = wrs['player_name'].tolist()
                if len(wr_names) >= 1:
                    team_dc['wr1'] = wr_names[0]
                if len(wr_names) >= 2:
                    team_dc['wr2'] = wr_names[1]
                if len(wr_names) >= 3:
                    team_dc['wr3'] = wr_names[2]

                # RBs
                rbs = team_dc_data[team_dc_data['pos_abb'] == 'RB'].sort_values('pos_rank')
                rb_names = rbs['player_name'].tolist()
                if len(rb_names) >= 1:
                    team_dc['rb1'] = rb_names[0]
                if len(rb_names) >= 2:
                    team_dc['rb2'] = rb_names[1]

                # TEs
                tes = team_dc_data[team_dc_data['pos_abb'] == 'TE'].sort_values('pos_rank')
                te_names = tes['player_name'].tolist()
                if len(te_names) >= 1:
                    team_dc['te1'] = te_names[0]

                # QBs
                qbs = team_dc_data[team_dc_data['pos_abb'] == 'QB'].sort_values('pos_rank')
                qb_names = qbs['player_name'].tolist()
                if len(qb_names) >= 1:
                    team_dc['qb1'] = qb_names[0]

                depth_charts[team] = team_dc

            logger.info(f"Loaded official depth charts for {len(depth_charts)} teams")
            return depth_charts
        except Exception as e:
            logger.warning(f"Could not load official depth charts: {e}")

    # FALLBACK: Try to load from rosters file (has depth info based on usage)
    rosters_path = Path('data/nflverse/rosters.parquet')
    if rosters_path.exists():
        try:
            rosters_df = pd.read_parquet(rosters_path)

            # Filter to current season
            if 'season' in rosters_df.columns:
                current_season = rosters_df['season'].max()
                rosters_df = rosters_df[rosters_df['season'] == current_season]

            # Get depth chart by team
            for team in rosters_df['team'].dropna().unique():
                team_roster = rosters_df[rosters_df['team'] == team]

                team_dc = {}

                # WRs - sort by some usage metric if available, otherwise alphabetically
                wrs = team_roster[team_roster['position'] == 'WR']
                if 'depth_chart_position' in wrs.columns:
                    wrs = wrs.sort_values('depth_chart_position')
                wr_names = wrs['full_name'].tolist() if 'full_name' in wrs.columns else wrs['player_name'].tolist() if 'player_name' in wrs.columns else []
                if len(wr_names) >= 1:
                    team_dc['wr1'] = wr_names[0]
                if len(wr_names) >= 2:
                    team_dc['wr2'] = wr_names[1]
                if len(wr_names) >= 3:
                    team_dc['wr3'] = wr_names[2]

                # RBs
                rbs = team_roster[team_roster['position'] == 'RB']
                if 'depth_chart_position' in rbs.columns:
                    rbs = rbs.sort_values('depth_chart_position')
                rb_names = rbs['full_name'].tolist() if 'full_name' in rbs.columns else rbs['player_name'].tolist() if 'player_name' in rbs.columns else []
                if len(rb_names) >= 1:
                    team_dc['rb1'] = rb_names[0]
                if len(rb_names) >= 2:
                    team_dc['rb2'] = rb_names[1]

                # TEs
                tes = team_roster[team_roster['position'] == 'TE']
                if 'depth_chart_position' in tes.columns:
                    tes = tes.sort_values('depth_chart_position')
                te_names = tes['full_name'].tolist() if 'full_name' in tes.columns else tes['player_name'].tolist() if 'player_name' in tes.columns else []
                if len(te_names) >= 1:
                    team_dc['te1'] = te_names[0]

                # QBs
                qbs = team_roster[team_roster['position'] == 'QB']
                if 'depth_chart_position' in qbs.columns:
                    qbs = qbs.sort_values('depth_chart_position')
                qb_names = qbs['full_name'].tolist() if 'full_name' in qbs.columns else qbs['player_name'].tolist() if 'player_name' in qbs.columns else []
                if len(qb_names) >= 1:
                    team_dc['qb1'] = qb_names[0]

                depth_charts[team] = team_dc

        except Exception as e:
            logger.warning(f"Could not load depth charts from rosters: {e}")

    return depth_charts































