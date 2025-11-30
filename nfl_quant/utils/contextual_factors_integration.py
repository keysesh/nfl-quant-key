#!/usr/bin/env python3
"""
Integrate contextual factors (matchups, QB connections, situational) into predictions.

This module applies contextual adjustments to model predictions before matching to lines.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.matchup_extractor import MatchupExtractor, ContextBuilder
from nfl_quant.utils.contextual_integration import load_injury_data
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.data.dynamic_parameters import get_parameter_provider

logger = logging.getLogger(__name__)


def apply_contextual_adjustments(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    week: int,
    season: int = 2025
) -> pd.DataFrame:
    """
    Apply contextual factors to predictions DataFrame.

    This function:
    1. Loads matchup and QB connection data
    2. Loads injury/lineup data
    3. Applies adjustments to predictions based on:
       - Historical matchup performance vs opponent
       - QB-dependent target shares
       - Situational context (injuries, lineup changes)

    Args:
        predictions_df: DataFrame with model predictions
        odds_df: DataFrame with odds (contains opponent info)
        week: Current week
        season: Season year

    Returns:
        DataFrame with contextual adjustments applied
    """
    logger.info("Applying contextual factors to predictions...")

    # Try to load PBP data
    pbp_path = Path(f'data/processed/pbp_{season}.parquet')
    if not pbp_path.exists():
        logger.warning(f"PBP data not found at {pbp_path} - skipping contextual adjustments")
        return predictions_df

    try:
        pbp_df = pd.read_parquet(pbp_path)
        logger.info(f"Loaded PBP data: {len(pbp_df):,} plays")
    except Exception as e:
        logger.warning(f"Could not load PBP data: {e}")
        return predictions_df

    # Load matchup and connection data
    matchup_df, qb_connections_df = _load_contextual_data(season, pbp_df)

    # Load injury data
    injury_data = load_injury_data(week)

    # Create context builder
    context_builder = ContextBuilder(
        matchup_df=matchup_df if len(matchup_df) > 0 else None,
        qb_connections_df=qb_connections_df if len(qb_connections_df) > 0 else None
    )

    # Apply adjustments to each prediction
    adjusted_predictions = []

    for idx, row in predictions_df.iterrows():
        adjusted_row = row.copy()

        # Get player info
        player_name = str(row.get('player_dk', '') or row.get('player_pbp', '')).strip()
        position = str(row.get('position', '')).strip()
        team = str(row.get('team', '')).strip()
        opponent = str(row.get('opponent', '')).strip()

        if not player_name or not position:
            adjusted_predictions.append(adjusted_row)
            continue

        # Build matchup history
        matchup_history = context_builder.build_matchup_history(
            player_name=player_name,
            position=position,
            current_week=week,
            seasons=[season]
        )

        # Build QB connections (for WR/TE/RB)
        qb_connections = {}
        if position in ['WR', 'TE', 'RB']:
            qb_connections = context_builder.build_qb_connection_history(
                wr_name=player_name
            )

        # Build situational multipliers
        situational_multipliers = {}
        if team and team in injury_data:
            situational_multipliers = context_builder.build_situational_adjustments(
                team=team,
                week=week,
                player_name=player_name,
                position=position,
                injury_data=injury_data[team]
            )

        # Apply matchup adjustments
        if matchup_history and opponent and opponent in matchup_history.vs_opponent_team:
            opponent_stats = matchup_history.vs_opponent_team[opponent]
            historical_yards_per_target = opponent_stats.get('avg_yards_per_target', None)

            if historical_yards_per_target and historical_yards_per_target > 0:
                if 'receiving_yards_mean' in adjusted_row.index:
                    # Blend historical (20%) with model (80%)
                    current_projection = adjusted_row.get('receiving_yards_mean', 0)
                    if current_projection > 0:
                        # Estimate total yards from yards per target
                        # Assume ~6 targets per game average
                        historical_yards_estimate = historical_yards_per_target * 6
                        blended = 0.8 * current_projection + 0.2 * historical_yards_estimate
                        adjusted_row['receiving_yards_mean'] = blended

        # Apply QB-dependent target share adjustments (for WR/TE/RB)
        if position in ['WR', 'TE', 'RB'] and qb_connections:
            # Get current QB status
            qb_status = 'healthy'
            starting_qb = ''
            if team and team in injury_data:
                qb_status = injury_data[team].get('qb_status', 'healthy')
                starting_qb = injury_data[team].get('starting_qb', '')

            # Adjust target share based on QB
            if starting_qb and starting_qb in qb_connections:
                connection = qb_connections[starting_qb]
                historical_target_share = connection.get('target_share', None)

                if historical_target_share and historical_target_share > 0:
                    # Adjust receptions projection based on QB connection
                    if 'receptions_mean' in adjusted_row.index:
                        current_receptions = adjusted_row.get('receptions_mean', 0)
                        if current_receptions > 0:
                            # Estimate QB attempts from team context using NFLverse data
                            param_provider = get_parameter_provider()
                            avg_attempts = param_provider.get_team_pass_attempts(team)
                            historical_targets = historical_target_share * avg_attempts
                            current_targets = current_receptions * 1.2

                            if current_targets > 0:
                                ts_multiplier = historical_targets / current_targets
                                ts_multiplier = max(0.7, min(1.3, ts_multiplier))
                                adjusted_row['receptions_mean'] = current_receptions * ts_multiplier

            # Apply backup QB adjustment if QB is injured
            if qb_status == 'injured':
                backup_qb = injury_data.get(team, {}).get('backup_qb', '')
                if backup_qb and backup_qb in qb_connections:
                    # Reduce target share with backup QB (typically 15% reduction)
                    if 'receptions_mean' in adjusted_row.index:
                        current_receptions = adjusted_row.get('receptions_mean', 0)
                        backup_multiplier = 0.85
                        adjusted_row['receptions_mean'] = current_receptions * backup_multiplier

        # Apply situational multipliers
        if situational_multipliers:
            opportunity_multiplier = situational_multipliers.get('opportunity_multiplier', 1.0)

            # Apply to usage-based projections
            if 'receptions_mean' in adjusted_row.index:
                current = adjusted_row.get('receptions_mean', 0)
                adjusted_row['receptions_mean'] = current * opportunity_multiplier

            if 'rushing_yards_mean' in adjusted_row.index:
                current = adjusted_row.get('rushing_yards_mean', 0)
                adjusted_row['rushing_yards_mean'] = current * opportunity_multiplier

            if 'receiving_yards_mean' in adjusted_row.index:
                efficiency_mult = situational_multipliers.get('efficiency_multiplier', 1.0)
                current = adjusted_row.get('receiving_yards_mean', 0)
                adjusted_row['receiving_yards_mean'] = current * opportunity_multiplier * efficiency_mult

        adjusted_predictions.append(adjusted_row)

    adjusted_df = pd.DataFrame(adjusted_predictions)

    logger.info(f"Applied contextual adjustments to {len(adjusted_df)} predictions")

    # Log summary of adjustments
    if len(matchup_df) > 0:
        logger.info(f"  ✅ Used {len(matchup_df):,} matchup records")
    if len(qb_connections_df) > 0:
        logger.info(f"  ✅ Used {len(qb_connections_df):,} QB connection records")
    if injury_data:
        logger.info(f"  ✅ Loaded injury data for {len(injury_data)} teams")

    return adjusted_df


def _load_contextual_data(season: int, pbp_df: Optional[pd.DataFrame] = None) -> tuple:
    """
    Load or extract matchup and QB connection data.

    Returns:
        Tuple of (matchup_df, qb_connections_df)
    """
    matchup_path = Path(f'data/matchups/team_matchups_{season}.parquet')
    qb_connections_path = Path(f'data/connections/qb_wr_connections_{season}.parquet')

    matchup_df = pd.DataFrame()
    qb_connections_df = pd.DataFrame()

    # Try to load cached data
    if matchup_path.exists():
        try:
            matchup_df = pd.read_parquet(matchup_path)
            logger.debug(f"Loaded matchup data from cache: {len(matchup_df):,} records")
        except Exception as e:
            logger.warning(f"Could not load matchup cache: {e}")

    if qb_connections_path.exists():
        try:
            qb_connections_df = pd.read_parquet(qb_connections_path)
            logger.debug(f"Loaded QB connections from cache: {len(qb_connections_df):,} records")
        except Exception as e:
            logger.warning(f"Could not load QB connections cache: {e}")

    # Extract on-the-fly if not cached and PBP available
    if (len(matchup_df) == 0 or len(qb_connections_df) == 0) and pbp_df is not None:
        extractor = MatchupExtractor(pbp_df)

        if len(matchup_df) == 0:
            try:
                matchup_df = extractor.extract_team_matchups(season)
                matchup_path.parent.mkdir(parents=True, exist_ok=True)
                if len(matchup_df) > 0:
                    matchup_df.to_parquet(matchup_path, index=False)
                    logger.info(f"Extracted and cached {len(matchup_df):,} matchup records")
            except Exception as e:
                logger.warning(f"Could not extract matchups: {e}")

        if len(qb_connections_df) == 0:
            try:
                qb_connections_df = extractor.extract_qb_wr_connections(season)
                qb_connections_path.parent.mkdir(parents=True, exist_ok=True)
                if len(qb_connections_df) > 0:
                    qb_connections_df.to_parquet(qb_connections_path, index=False)
                    logger.info(f"Extracted and cached {len(qb_connections_df):,} QB connection records")
            except Exception as e:
                logger.warning(f"Could not extract QB connections: {e}")

    return matchup_df, qb_connections_df


if __name__ == '__main__':
    # Test script
    import sys

    if len(sys.argv) < 2:
        print("Usage: python integrate_contextual_factors.py <week>")
        sys.exit(1)

    week = int(sys.argv[1])

    # Load test data
    predictions_path = Path(f'data/nfl_player_props_predictions.csv')
    odds_path = Path('data/nfl_player_props_draftkings.csv')

    if not predictions_path.exists() or not odds_path.exists():
        print("❌ Required data files not found")
        sys.exit(1)

    predictions = pd.read_csv(predictions_path)
    odds = pd.read_csv(odds_path)

    print(f"📊 Applying contextual factors to {len(predictions)} predictions...")
    adjusted = apply_contextual_adjustments(predictions, odds, week)

    print(f"✅ Adjusted {len(adjusted)} predictions")
    print(f"\n📋 Sample adjustments:")
    print(adjusted.head(10)[['player_dk', 'position', 'receiving_yards_mean', 'receptions_mean']].to_string())
