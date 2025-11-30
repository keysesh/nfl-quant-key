#!/usr/bin/env python3
"""
Example: Integrating TIER 1 & TIER 2 Features into Prediction Pipeline

This script demonstrates how to:
1. Extract all enhanced features for players
2. Use them in predictions
3. Compare old vs new feature sets

Usage:
    python scripts/examples/integrate_tier1_2_features_example.py --week 11
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.features.tier1_2_integration import (
    extract_all_tier1_2_features,
    get_feature_columns_for_position,
    validate_features,
    summarize_feature_extraction
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pbp_data(season: int = 2025) -> pd.DataFrame:
    """Load play-by-play data."""
    pbp_path = PROJECT_ROOT / f'data/nflverse/pbp_{season}.parquet'

    if not pbp_path.exists():
        logger.error(f"PBP file not found: {pbp_path}")
        logger.info("Run: python scripts/fetch/pull_2024_season_data.py")
        sys.exit(1)

    logger.info(f"Loading PBP data from {pbp_path}")
    pbp_df = pd.read_parquet(pbp_path)
    logger.info(f"  ✓ Loaded {len(pbp_df):,} plays")

    return pbp_df


def get_sample_players(week: int) -> list:
    """
    Get sample players for testing.

    In production, this would load from your active player roster.
    """
    return [
        {
            "name": "T.Hill",
            "position": "WR",
            "team": "MIA",
            "opponent": "KC",
            "expected_targets": 8.5  # For comparison
        },
        {
            "name": "C.McCaffrey",
            "position": "RB",
            "team": "SF",
            "opponent": "GB",
            "expected_carries": 18.2
        },
        {
            "name": "P.Mahomes",
            "position": "QB",
            "team": "KC",
            "opponent": "MIA",
            "expected_attempts": 34.5
        },
        {
            "name": "T.Kelce",
            "position": "TE",
            "team": "KC",
            "opponent": "MIA",
            "expected_targets": 7.2
        },
    ]


def compare_feature_sets(player: dict, old_features: dict, new_features: dict) -> None:
    """Compare old (basic) vs new (TIER 1 & 2) feature sets."""
    logger.info(f"\n{'='*80}")
    logger.info(f"FEATURE COMPARISON: {player['name']} ({player['position']})")
    logger.info(f"{'='*80}")

    # Old features (baseline)
    logger.info("\nOLD FEATURES (Baseline):")
    logger.info(f"  trailing_snaps:    {old_features.get('trailing_snaps', 0):.2f}")
    logger.info(f"  trailing_attempts: {old_features.get('trailing_attempts', 0):.2f}")
    logger.info(f"  trailing_carries:  {old_features.get('trailing_carries', 0):.2f}")

    # New features (enhanced)
    logger.info("\nNEW FEATURES (TIER 1 & 2):")

    # Trailing stats (with EWMA)
    logger.info("\n  [TIER 1] Trailing Stats (EWMA-weighted):")
    logger.info(f"    trailing_snaps:    {new_features.get('trailing_snaps', 0):.2f}")
    logger.info(f"    trailing_attempts: {new_features.get('trailing_attempts', 0):.2f}")
    logger.info(f"    trailing_carries:  {new_features.get('trailing_carries', 0):.2f}")

    # Regime features
    logger.info("\n  [TIER 1] Regime Detection:")
    logger.info(f"    weeks_since_regime_change: {new_features.get('weeks_since_regime_change', 0):.0f}")
    logger.info(f"    is_in_regime:              {new_features.get('is_in_regime', 0):.0f}")
    logger.info(f"    regime_confidence:         {new_features.get('regime_confidence', 0):.2f}")

    # Game script features
    logger.info("\n  [TIER 1] Game Script:")
    logger.info(f"    usage_when_leading:    {new_features.get('usage_when_leading', 0):.1f}")
    logger.info(f"    usage_when_trailing:   {new_features.get('usage_when_trailing', 0):.1f}")
    logger.info(f"    usage_when_close:      {new_features.get('usage_when_close', 0):.1f}")
    logger.info(f"    game_script_sensitivity: {new_features.get('game_script_sensitivity', 0):.1f}")

    # NGS features
    logger.info("\n  [TIER 2] NGS Advanced Metrics:")
    if player['position'] in ['WR', 'TE']:
        logger.info(f"    avg_separation:         {new_features.get('avg_separation', 0):.2f} yards")
        logger.info(f"    avg_intended_air_yards: {new_features.get('avg_intended_air_yards', 0):.2f} yards")
    elif player['position'] == 'RB':
        logger.info(f"    rush_yards_over_expected: {new_features.get('rush_yards_over_expected_per_att', 0):+.2f} per carry")
        logger.info(f"    efficiency:               {new_features.get('efficiency', 0):.2f}")
    elif player['position'] == 'QB':
        logger.info(f"    avg_time_to_throw:       {new_features.get('avg_time_to_throw', 0):.2f} sec")
        logger.info(f"    completion_pct_above_exp: {new_features.get('completion_pct_above_exp', 0):+.1f}%")

    # Situational EPA
    logger.info("\n  [TIER 2] Situational EPA (Opponent Defense):")
    logger.info(f"    redzone_epa:     {new_features.get('redzone_epa', 0):+.3f}")
    logger.info(f"    third_down_epa:  {new_features.get('third_down_epa', 0):+.3f}")
    logger.info(f"    two_minute_epa:  {new_features.get('two_minute_epa', 0):+.3f}")
    if player['position'] == 'RB':
        logger.info(f"    goalline_epa:    {new_features.get('goalline_epa', 0):+.3f}")

    # Feature count comparison
    logger.info(f"\n  FEATURE COUNT:")
    logger.info(f"    Old features:  {len(old_features)}")
    logger.info(f"    New features:  {len(new_features)}")
    logger.info(f"    Added:         {len(new_features) - len(old_features)}")


def extract_features_for_player(
    player: dict,
    week: int,
    season: int,
    pbp_df: pd.DataFrame,
    use_tier1_2: bool = True
) -> dict:
    """
    Extract features for a player.

    Args:
        player: Player info dict
        week: Current week
        season: Season year
        pbp_df: Play-by-play data
        use_tier1_2: If True, extract TIER 1 & 2 features. If False, baseline only.

    Returns:
        Feature dictionary
    """
    if use_tier1_2:
        # Extract all TIER 1 & TIER 2 features
        features = extract_all_tier1_2_features(
            player_name=player['name'],
            position=player['position'],
            team=player['team'],
            opponent=player['opponent'],
            current_week=week,
            season=season,
            pbp_df=pbp_df,
            use_ewma=True,
            use_regime=True,
            use_game_script=True,
            use_ngs=True,
            use_situational_epa=True
        )

        # Validate
        features = validate_features(features, player['position'])

    else:
        # Extract only baseline features (old approach)
        from nfl_quant.features.trailing_stats import get_trailing_stats_extractor

        extractor = get_trailing_stats_extractor()
        features = extractor.get_trailing_stats(
            player_name=player['name'],
            position=player['position'],
            current_week=week,
            trailing_weeks=4,
            use_ewma=False  # Old approach: simple mean
        )

    return features


def main():
    parser = argparse.ArgumentParser(description="Test TIER 1 & 2 feature integration")
    parser.add_argument('--week', type=int, default=11, help="NFL week number")
    parser.add_argument('--season', type=int, default=2025, help="Season year")
    parser.add_argument('--player', type=str, help="Specific player name to test (optional)")

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info(f"TIER 1 & TIER 2 FEATURE INTEGRATION TEST")
    logger.info(f"Week {args.week}, Season {args.season}")
    logger.info(f"{'='*80}\n")

    # Load data
    pbp_df = load_pbp_data(season=args.season)

    # Get players
    players = get_sample_players(args.week)

    # Filter to specific player if requested
    if args.player:
        players = [p for p in players if args.player.lower() in p['name'].lower()]
        if not players:
            logger.error(f"Player '{args.player}' not found in sample players")
            sys.exit(1)

    # Extract features for each player
    all_results = []

    for player in players:
        logger.info(f"\n{'*'*80}")
        logger.info(f"Processing: {player['name']} ({player['position']})")
        logger.info(f"{'*'*80}")

        # Extract OLD features (baseline)
        logger.info("\nExtracting BASELINE features (old approach)...")
        old_features = extract_features_for_player(
            player, args.week, args.season, pbp_df, use_tier1_2=False
        )

        # Extract NEW features (TIER 1 & 2)
        logger.info("Extracting TIER 1 & TIER 2 features (new approach)...")
        new_features = extract_features_for_player(
            player, args.week, args.season, pbp_df, use_tier1_2=True
        )

        # Compare
        compare_feature_sets(player, old_features, new_features)

        # Store results
        all_results.append({
            'player': player['name'],
            'position': player['position'],
            'old_feature_count': len(old_features),
            'new_feature_count': len(new_features),
            'features_added': len(new_features) - len(old_features),
            'old_trailing_attempts': old_features.get('trailing_attempts', 0),
            'new_trailing_attempts': new_features.get('trailing_attempts', 0),
            'ewma_diff': new_features.get('trailing_attempts', 0) - old_features.get('trailing_attempts', 0),
            'regime_detected': new_features.get('is_in_regime', 0) == 1.0,
            'game_script_sensitivity': new_features.get('game_script_sensitivity', 0),
            'redzone_epa': new_features.get('redzone_epa', 0),
        })

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Feature Extraction Results")
    logger.info(f"{'='*80}\n")

    results_df = pd.DataFrame(all_results)
    logger.info(results_df.to_string(index=False))

    # Save to CSV
    output_path = PROJECT_ROOT / f'reports/tier1_2_feature_test_week{args.week}.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Results saved to: {output_path}")

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Players tested:        {len(players)}")
    logger.info(f"Avg features added:    {results_df['features_added'].mean():.1f}")
    logger.info(f"Regimes detected:      {results_df['regime_detected'].sum()}")
    logger.info(f"Avg EWMA diff:         {results_df['ewma_diff'].mean():+.2f}")
    logger.info(f"{'='*80}\n")

    logger.info("✓ TIER 1 & TIER 2 feature integration test complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review feature extraction logs above")
    logger.info("2. Integrate into generate_model_predictions.py")
    logger.info("3. Retrain models with new features")
    logger.info("4. Validate with temporal cross-validation")


if __name__ == "__main__":
    main()
