"""
TIER 1 & TIER 2 Feature Integration Module

Provides unified interface to extract all enhanced features:
- EWMA-weighted trailing stats
- Regime detection features
- Game script features
- NGS advanced metrics
- Situational EPA

Usage:
    from nfl_quant.features.tier1_2_integration import extract_all_tier1_2_features

    features = extract_all_tier1_2_features(
        player_name="T.Hill",
        position="WR",
        team="MIA",
        opponent="KC",
        current_week=11,
        season=2025,
        pbp_df=pbp_df
    )
"""

import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_all_tier1_2_features(
    player_name: str,
    position: str,
    team: str,
    opponent: str,
    current_week: int,
    season: int,
    pbp_df: pd.DataFrame,
    use_ewma: bool = True,
    use_regime: bool = True,
    use_game_script: bool = True,
    use_ngs: bool = True,
    use_situational_epa: bool = True
) -> Dict[str, float]:
    """
    Extract all TIER 1 & TIER 2 features for a player.

    Args:
        player_name: Player name (e.g., "T.Hill")
        position: Position (QB, RB, WR, TE)
        team: Player's team abbreviation
        opponent: Opponent team abbreviation
        current_week: Current NFL week
        season: Season year
        pbp_df: Play-by-play DataFrame
        use_ewma: Use exponential weighted moving average
        use_regime: Extract regime detection features
        use_game_script: Extract game script features
        use_ngs: Extract NGS advanced metrics
        use_situational_epa: Extract situational EPA features

    Returns:
        Dictionary with all extracted features
    """
    all_features = {}

    # =========================================================================
    # TIER 1 FEATURE #1: EWMA-Weighted Trailing Stats
    # =========================================================================
    logger.debug(f"Extracting trailing stats for {player_name} (EWMA={use_ewma})")

    try:
        from .trailing_stats import get_trailing_stats_extractor

        extractor = get_trailing_stats_extractor()
        trailing_stats = extractor.get_trailing_stats(
            player_name=player_name,
            position=position,
            current_week=current_week,
            trailing_weeks=4,
            use_ewma=use_ewma
        )
        all_features.update(trailing_stats)
        logger.debug(f"  ✓ Trailing stats: {len(trailing_stats)} features")
    except Exception as e:
        logger.warning(f"  ✗ Error extracting trailing stats: {e}")
        all_features.update({
            'trailing_snaps': 0.0,
            'trailing_attempts': 0.0,
            'trailing_carries': 0.0,
        })

    # =========================================================================
    # TIER 1 FEATURE #2: Regime Detection Features
    # =========================================================================
    if use_regime:
        logger.debug(f"Extracting regime features for {player_name}")
        try:
            regime_features = extractor.get_regime_features(
                player_name=player_name,
                team=team,
                current_week=current_week,
                position=position
            )
            all_features.update(regime_features)
            logger.debug(f"  ✓ Regime features: {len(regime_features)} features")

            # Log if regime detected
            if regime_features.get('is_in_regime', 0.0) == 1.0:
                logger.info(
                    f"  📍 REGIME DETECTED: {player_name} - "
                    f"{regime_features.get('weeks_since_regime_change', 0):.0f} weeks since change, "
                    f"confidence={regime_features.get('regime_confidence', 0.0):.2f}"
                )
        except Exception as e:
            logger.warning(f"  ✗ Error extracting regime features: {e}")
            all_features.update({
                'weeks_since_regime_change': 999.0,
                'is_in_regime': 0.0,
                'regime_confidence': 0.0,
            })

    # =========================================================================
    # TIER 1 FEATURE #3: Game Script Features
    # =========================================================================
    if use_game_script:
        logger.debug(f"Extracting game script features for {player_name}")
        try:
            game_script_features = extractor.get_game_script_features(
                player_name=player_name,
                position=position,
                current_week=current_week,
                trailing_weeks=4
            )
            all_features.update(game_script_features)
            logger.debug(f"  ✓ Game script features: {len(game_script_features)} features")

            # Log if high sensitivity (game script dependent player)
            sensitivity = game_script_features.get('game_script_sensitivity', 0.0)
            if sensitivity > 5.0:
                logger.info(
                    f"  📊 HIGH GAME SCRIPT SENSITIVITY: {player_name} - "
                    f"sensitivity={sensitivity:.1f}, "
                    f"leading={game_script_features.get('usage_when_leading', 0):.1f}, "
                    f"trailing={game_script_features.get('usage_when_trailing', 0):.1f}"
                )
        except Exception as e:
            logger.warning(f"  ✗ Error extracting game script features: {e}")
            all_features.update({
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            })

    # =========================================================================
    # TIER 2 FEATURE #1: NGS Advanced Metrics
    # =========================================================================
    if use_ngs:
        logger.debug(f"Extracting NGS features for {player_name}")
        try:
            from .ngs_features import NGSFeatureExtractor

            ngs_extractor = NGSFeatureExtractor()

            if position == 'QB':
                ngs_features = ngs_extractor.get_qb_features(
                    player_name=player_name,
                    season=season,
                    week=current_week,
                    trailing_weeks=4
                )
            elif position in ['WR', 'TE']:
                ngs_features = ngs_extractor.get_receiver_features(
                    player_name=player_name,
                    season=season,
                    week=current_week,
                    trailing_weeks=4
                )
            elif position == 'RB':
                ngs_features = ngs_extractor.get_rusher_features(
                    player_name=player_name,
                    season=season,
                    week=current_week,
                    trailing_weeks=4
                )
            else:
                ngs_features = {}

            all_features.update(ngs_features)
            logger.debug(f"  ✓ NGS features: {len(ngs_features)} features")

            # Log notable NGS metrics
            if position in ['WR', 'TE']:
                sep = ngs_features.get('avg_separation', 0.0)
                if sep > 3.0:
                    logger.info(f"  🎯 HIGH SEPARATION: {player_name} - {sep:.2f} yards")
            elif position == 'RB':
                ryoe = ngs_features.get('rush_yards_over_expected_per_att', 0.0)
                if abs(ryoe) > 0.5:
                    logger.info(f"  ⚡ YARDS OVER EXPECTED: {player_name} - {ryoe:+.2f} per carry")

        except Exception as e:
            logger.warning(f"  ✗ Error extracting NGS features: {e}")
            # NGS features have defaults in the extractor

    # =========================================================================
    # TIER 2 FEATURE #2: Situational EPA
    # =========================================================================
    if use_situational_epa:
        logger.debug(f"Extracting situational EPA for opponent {opponent}")
        try:
            from ..utils.epa_utils import get_all_situational_epa_features

            epa_features = get_all_situational_epa_features(
                pbp_df=pbp_df,
                team=opponent,
                position=position,
                weeks=10,
                season=season,
                is_defense=True  # Get opponent's defensive EPA
            )
            all_features.update(epa_features)
            logger.debug(f"  ✓ Situational EPA: {len(epa_features)} features")

            # Log favorable matchups (positive EPA = bad defense)
            redzone_epa = epa_features.get('redzone_epa', 0.0)
            if redzone_epa > 0.10:
                logger.info(f"  🎯 FAVORABLE REDZONE MATCHUP: {opponent} allows {redzone_epa:+.3f} EPA")

        except Exception as e:
            logger.warning(f"  ✗ Error extracting situational EPA: {e}")
            all_features.update({
                'redzone_epa': 0.0,
                'third_down_epa': 0.0,
                'two_minute_epa': 0.0,
            })
            if position == 'RB':
                all_features['goalline_epa'] = 0.0

    return all_features


def get_feature_columns_for_position(position: str, include_ngs: bool = True) -> list:
    """
    Get feature column names for a specific position.

    Args:
        position: Position (QB, RB, WR, TE)
        include_ngs: Include NGS features

    Returns:
        List of feature column names
    """
    # Base features (all positions)
    features = [
        # Trailing stats (EWMA-weighted)
        'trailing_snaps',
        'trailing_attempts',
        'trailing_carries',

        # Regime detection
        'weeks_since_regime_change',
        'is_in_regime',
        'regime_confidence',

        # Game script
        'usage_when_leading',
        'usage_when_trailing',
        'usage_when_close',
        'game_script_sensitivity',

        # Situational EPA
        'redzone_epa',
        'third_down_epa',
        'two_minute_epa',
    ]

    # Position-specific NGS features
    if include_ngs:
        if position == 'QB':
            features.extend([
                'avg_time_to_throw',
                'completion_pct_above_exp',
                'aggressiveness',
                'avg_air_yards_to_sticks',
                'passer_rating_ngs',
                'expected_completion_pct',
            ])
        elif position in ['WR', 'TE']:
            features.extend([
                'avg_separation',
                'avg_cushion',
                'avg_yac_above_expectation',
                'catch_percentage',
                'avg_intended_air_yards',
                'percent_share_of_intended_air_yards',
            ])
        elif position == 'RB':
            features.extend([
                'rush_yards_over_expected_per_att',
                'efficiency',
                'percent_attempts_gte_eight_defenders',
                'avg_time_to_los',
                'expected_rush_yards',
                'rush_pct_over_expected',
                'goalline_epa',  # RBs get goalline EPA
            ])

    return features


def validate_features(features: Dict[str, float], position: str) -> Dict[str, float]:
    """
    Validate and clean extracted features.

    Args:
        features: Extracted feature dictionary
        position: Player position

    Returns:
        Cleaned feature dictionary
    """
    expected_cols = get_feature_columns_for_position(position, include_ngs=True)

    # Fill missing features with 0.0
    for col in expected_cols:
        if col not in features:
            logger.warning(f"Missing feature '{col}', filling with 0.0")
            features[col] = 0.0

    # Ensure all numeric
    for key, value in features.items():
        if not isinstance(value, (int, float)):
            logger.warning(f"Non-numeric value for '{key}': {value}, converting to 0.0")
            features[key] = 0.0

    # Clip extreme values
    features['weeks_since_regime_change'] = min(features.get('weeks_since_regime_change', 999), 999)
    features['is_in_regime'] = max(0.0, min(1.0, features.get('is_in_regime', 0.0)))
    features['regime_confidence'] = max(0.0, min(1.0, features.get('regime_confidence', 0.0)))

    return features


def summarize_feature_extraction(features: Dict[str, float]) -> None:
    """
    Print summary of extracted features.

    Args:
        features: Extracted feature dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("TIER 1 & TIER 2 FEATURE EXTRACTION SUMMARY")
    logger.info("="*60)

    # Count features by category
    trailing = len([k for k in features if k.startswith('trailing_')])
    regime = len([k for k in features if 'regime' in k or 'is_in_regime' in k])
    game_script = len([k for k in features if 'usage_when' in k or 'sensitivity' in k])
    epa = len([k for k in features if 'epa' in k])
    ngs = len(features) - trailing - regime - game_script - epa

    logger.info(f"Trailing Stats (EWMA): {trailing} features")
    logger.info(f"Regime Detection:      {regime} features")
    logger.info(f"Game Script:           {game_script} features")
    logger.info(f"Situational EPA:       {epa} features")
    logger.info(f"NGS Metrics:           {ngs} features")
    logger.info(f"TOTAL:                 {len(features)} features")
    logger.info("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load sample PBP data
    pbp_path = Path('data/nflverse/pbp_2025.parquet')
    if not pbp_path.exists():
        logger.error(f"PBP file not found: {pbp_path}")
        sys.exit(1)

    pbp_df = pd.read_parquet(pbp_path)
    logger.info(f"Loaded {len(pbp_df):,} plays from {pbp_path}")

    # Extract features for example players
    test_players = [
        {"name": "T.Hill", "position": "WR", "team": "MIA", "opponent": "KC"},
        {"name": "C.McCaffrey", "position": "RB", "team": "SF", "opponent": "GB"},
        {"name": "P.Mahomes", "position": "QB", "team": "KC", "opponent": "MIA"},
    ]

    for player in test_players:
        logger.info(f"\n{'='*60}")
        logger.info(f"EXTRACTING FEATURES: {player['name']} ({player['position']})")
        logger.info(f"{'='*60}")

        features = extract_all_tier1_2_features(
            player_name=player['name'],
            position=player['position'],
            team=player['team'],
            opponent=player['opponent'],
            current_week=11,
            season=2025,
            pbp_df=pbp_df
        )

        # Validate
        features = validate_features(features, player['position'])

        # Summarize
        logger.info(f"\n{player['name']} Features:")
        for key, value in sorted(features.items()):
            logger.info(f"  {key:40s} = {value:8.3f}")
